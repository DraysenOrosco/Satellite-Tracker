"""
Close-approach screening: finds pairs of objects that pass within a few kilometres of each
other during the next few hours of simulated time.

Method
    1. Propagate every object every STEP_S seconds (vectorised SGP4).
    2. At each step, find pairs closer than a coarse radius with a uniform spatial hash
       grid. The radius is wide enough that nothing moving at orbital speeds can slip
       through between steps.
    3. Assume straight-line relative motion within +/- STEP_S/2 of the step to estimate
       each pair's closest approach; keep pairs that get near the threshold.
    4. Refine the survivors with SGP4 (1 s samples, a parabola fit, then an exact evaluation).

Pairs with almost no relative motion (docked vehicles, formation flyers) are skipped.

General-perturbations elements are only good to about a kilometre (worse for debris), so
this is a screening estimate, not a collision prediction.

The work runs in a separate worker process. A thread would compete with the renderer
for Python's GIL; in testing that slowed the app down and stretched a 30 s screening
to several minutes.
"""

import multiprocessing as mp
import queue
from dataclasses import dataclass

import numpy as np
from sgp4.api import SatrecArray

from data import make_satrec

THRESHOLD_KM = 5.0
WINDOW_HOURS = 6.0
STEP_S = 20.0
MAX_RELATIVE_SPEED_KM_S = 16.0
MIN_RELATIVE_SPEED_KM_S = 0.05
COARSE_RADIUS_KM = THRESHOLD_KM + MAX_RELATIVE_SPEED_KM_S * STEP_S / 2
CHUNK_STEPS = 15

# Neighbouring grid cells to compare with: the cell itself plus half of the 26 neighbours,
# so every pair of adjacent cells is visited exactly once.
_NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
HALF_NEIGHBOURS = np.array([o for o in _NEIGHBOURS if o > (0, 0, 0)] + [(0, 0, 0)], dtype=np.int64)


@dataclass
class Approach:
    i: int              # catalogue index of the first object
    j: int              # catalogue index of the second object
    tca_tt: float       # time of closest approach (TT, Julian date)
    miss_km: float      # distance at closest approach
    speed_km_s: float   # relative speed at closest approach


def _cell_keys(cells):
    offset = 1 << 20
    return ((cells[:, 0] + offset) << 42) | ((cells[:, 1] + offset) << 21) | (cells[:, 2] + offset)


def close_pairs(points, radius):
    """Index pairs (i, j) of points closer than `radius`, via a uniform grid hash."""
    cells = np.floor(points / radius).astype(np.int64)
    keys = _cell_keys(cells)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    everyone = np.arange(len(points))

    found_i, found_j = [], []
    for offset in HALF_NEIGHBOURS:
        neighbour_keys = _cell_keys(cells + offset)
        start = np.searchsorted(sorted_keys, neighbour_keys, "left")
        counts = np.searchsorted(sorted_keys, neighbour_keys, "right") - start
        total = int(counts.sum())
        if total == 0:
            continue
        src = np.repeat(everyone, counts)
        # Expand each [start, start + count) range into the indices it covers
        dst = order[np.arange(total) + np.repeat(start - (np.cumsum(counts) - counts), counts)]
        if not offset.any():
            keep = src < dst  # same cell: each pair once, never a point with itself
            src, dst = src[keep], dst[keep]
        found_i.append(src)
        found_j.append(dst)

    if not found_i:
        return np.array([], dtype=int), np.array([], dtype=int)
    i, j = np.concatenate(found_i), np.concatenate(found_j)
    near = np.linalg.norm(points[i] - points[j], axis=1) < radius
    return i[near], j[near]


# -----------------------------------------------------------------------------
# Screening (runs inside the worker process)
# -----------------------------------------------------------------------------
class Screening:
    def __init__(self, satrecs, ts):
        self.satrecs = satrecs
        self.array = SatrecArray(satrecs)
        self.ts = ts

    def _sgp4_time(self, tt):
        t = self.ts.tt_jd(tt)
        return t.whole, t.tai_fraction - t._leap_seconds() / 86400.0

    def run(self, start_tt, report, cancelled):
        """Approaches in [start_tt, start_tt + WINDOW_HOURS], or None if cancelled."""
        n_steps = int(WINDOW_HOURS * 3600 / STEP_S) + 1
        step_tt = start_tt + np.arange(n_steps) * STEP_S / 86400.0
        jd, fraction = self._sgp4_time(step_tt)

        candidates = {}  # (i, j, 10-minute bucket) -> (estimated miss, estimated TCA)
        for c0 in range(0, n_steps, CHUNK_STEPS):
            if cancelled():
                return None
            errors, r, v = self.array.sgp4(jd[c0:c0 + CHUNK_STEPS], fraction[c0:c0 + CHUNK_STEPS])
            for k in range(r.shape[1]):
                ok = np.flatnonzero((errors[:, k] == 0) & np.isfinite(r[:, k, 0]))
                pos, vel = r[ok, k], v[ok, k]
                a, b = close_pairs(pos, COARSE_RADIUS_KM)
                if len(a) == 0:
                    continue

                # Straight-line relative motion around this step
                dr, dv = pos[b] - pos[a], vel[b] - vel[a]
                speed2 = np.einsum("ij,ij->i", dv, dv)
                tau = np.clip(-np.einsum("ij,ij->i", dr, dv) / np.maximum(speed2, 1e-12),
                              -STEP_S / 2, STEP_S / 2)
                miss = np.linalg.norm(dr + dv * tau[:, None], axis=1)
                hit = (speed2 > MIN_RELATIVE_SPEED_KM_S ** 2) & (miss < THRESHOLD_KM + 2.0)

                tca = step_tt[c0 + k] + tau[hit] / 86400.0
                for i, j, d, t in zip(ok[a[hit]], ok[b[hit]], miss[hit], tca):
                    key = (int(min(i, j)), int(max(i, j)), int(t * 144))
                    if key not in candidates or d < candidates[key][0]:
                        candidates[key] = (d, t)

            report(0.9 * min(1.0, (c0 + CHUNK_STEPS) / n_steps))

        return self._refine(candidates, report, cancelled)

    def _refine(self, candidates, report, cancelled):
        step_s = 1.0
        offsets_s = np.arange(-STEP_S, STEP_S + step_s, step_s)
        events = []
        for n, ((i, j, _), (_, t_est)) in enumerate(candidates.items()):
            if n % 200 == 0:
                if cancelled():
                    return None
                report(0.9 + 0.1 * n / max(len(candidates), 1))
            jd, fraction = self._sgp4_time(t_est + offsets_s / 86400.0)
            ei, ri, _ = self.satrecs[i].sgp4_array(jd, fraction)
            ej, rj, _ = self.satrecs[j].sgp4_array(jd, fraction)
            if ei.any() or ej.any():
                continue

            # Squared distance is close to a parabola near its minimum: fit through the
            # best 1 s sample and its neighbours, then evaluate SGP4 exactly there.
            d2 = np.einsum("ij,ij->i", ri - rj, ri - rj)
            m = int(np.argmin(d2))
            shift = 0.0
            if 0 < m < len(d2) - 1:
                curvature = d2[m - 1] - 2 * d2[m] + d2[m + 1]
                if curvature > 0:
                    shift = 0.5 * (d2[m - 1] - d2[m + 1]) / curvature * step_s
            tca = t_est + (offsets_s[m] + shift) / 86400.0

            jd, fraction = self._sgp4_time(np.array([tca]))
            ei, ri, vi = self.satrecs[i].sgp4_array(jd, fraction)
            ej, rj, vj = self.satrecs[j].sgp4_array(jd, fraction)
            miss = float(np.linalg.norm(ri[0] - rj[0]))
            if not ei.any() and not ej.any() and miss < THRESHOLD_KM:
                speed = float(np.linalg.norm(vi[0] - vj[0]))
                events.append(Approach(i, j, float(tca), miss, speed))

        # The same encounter can be found from two neighbouring buckets; keep one
        events.sort(key=lambda e: (e.i, e.j, e.tca_tt))
        unique = []
        for event in events:
            last = unique[-1] if unique else None
            if last and (last.i, last.j) == (event.i, event.j) and abs(last.tca_tt - event.tca_tt) < 120 / 86400:
                if event.miss_km < last.miss_km:
                    unique[-1] = event
            else:
                unique.append(event)
        unique.sort(key=lambda e: e.tca_tt)
        return unique


def _worker_main(records, requests, results, progress):
    """Worker process: builds the SGP4 models once, then screens each requested window.

    A newer request cancels the one in progress.
    """
    from skyfield.api import load

    ts = load.timescale()
    screening = Screening([make_satrec(record) for record in records], ts)
    pending = None
    while True:
        job = pending if pending is not None else requests.get()
        pending = None
        if job is None:
            return
        job_id, start_tt = job

        def cancelled():
            nonlocal pending
            try:
                pending = requests.get_nowait()
            except queue.Empty:
                return False
            return True

        def report(fraction):
            progress.value = fraction

        try:
            events = screening.run(start_tt, report, cancelled)
            if events is not None:
                results.put((job_id, start_tt, events, None))
        except Exception as exc:  # report and keep serving
            results.put((job_id, start_tt, [], f"{type(exc).__name__}: {exc}"))


# -----------------------------------------------------------------------------
# Front end (used by the app)
# -----------------------------------------------------------------------------
class ApproachScreener:
    """Screens time windows in a background process. Call poll() regularly, then read
    `events`, `window`, `progress`, `running` and `error`."""

    def __init__(self, records):
        self.records = records
        self.events = []
        self.window = None          # (start_tt, end_tt) of the last finished screening
        self.running = False
        self.error = None
        self._job_id = 0
        self._process = None

    def _ensure_process(self):
        if self._process is not None and self._process.is_alive():
            return
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._results = context.Queue()
        self._progress = context.Value("d", 0.0, lock=False)
        self._process = context.Process(target=_worker_main, daemon=True,
                                        args=(self.records, self._requests, self._results, self._progress))
        self._process.start()

    @property
    def progress(self):
        return self._progress.value if self._process is not None else 0.0

    def start(self, start_tt):
        self._ensure_process()
        self._job_id += 1
        self._progress.value = 0.0
        self.running = True
        self._requests.put((self._job_id, start_tt))

    def poll(self):
        """Collect finished results. Returns True when new events arrived."""
        if self._process is None:
            return False
        updated = False
        while True:
            try:
                job_id, start_tt, events, error = self._results.get_nowait()
            except queue.Empty:
                break
            if job_id != self._job_id:
                continue  # superseded
            self.running = False
            self.error = error
            if error is None:
                self.events = events
                self.window = (start_tt, start_tt + WINDOW_HOURS / 24)
                updated = True
        if self.running and not self._process.is_alive():
            self.running = False
            self.error = "screening process stopped"
        return updated

    def close(self):
        if self._process is not None and self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
