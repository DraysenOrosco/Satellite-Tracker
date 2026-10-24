"""
Satellite catalogue: downloads live orbital elements from CelesTrak, caches them on disk,
and sorts every object into a display category.

Elements are fetched as OMM (CSV). The older TLE format can't hold catalogue numbers
above 99999, which new launches now use. Cached .tle files from older versions of the
app are still read while they are fresh.
"""

import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec
from skyfield.api import EarthSatellite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "tle_cache")
BUNDLED_TLE = os.path.join(BASE_DIR, "stations.tle")

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=csv"
USER_AGENT = "SatelliteTracker/1.0 (educational PyVista viewer)"

# CelesTrak only refreshes element sets every couple of hours and asks clients not to
# download the same group more often than that (repeat offenders get blocked).
CACHE_MAX_AGE_HOURS = 2

DEBRIS_GROUPS = ["cosmos-1408-debris", "fengyun-1c-debris", "iridium-33-debris", "cosmos-2251-debris"]
RECENT_GROUP = "last-30-days"

# "active" is every operational satellite (~16k, mostly Starlink). The other groups add
# objects "active" leaves out (station modules, debris) and drive the categories below.
DEFAULT_GROUPS = ["stations", RECENT_GROUP, "active", "gnss", "weather", "science"] + DEBRIS_GROUPS

# key: (legend label, colour, marker size in px, label priority - lower wins)
CATEGORIES = {
    "station": ("Space stations & crew", "#f87171", 10, 0),
    "recent": ("Launched last 30 days", "#f472b6", 6, 1),
    "gnss": ("Navigation (GNSS)", "#34d399", 7, 2),
    "weather": ("Weather & Earth science", "#facc15", 7, 2),
    "geo": ("Geostationary", "#fb923c", 6, 3),
    "other": ("Other active", "#e2e8f0", 4, 4),
    "constellation": ("Broadband constellations", "#a78bfa", 4, 5),
    "starlink": ("Starlink", "#7d8fb0", 3, 6),
    "debris": ("Debris", "#b08968", 3, 7),
}

STATION_PATTERN = re.compile(
    r"\b(ISS|CSS|TIANHE|WENTIAN|MENGTIAN|SHENZHOU|TIANZHOU|CREW DRAGON|DRAGON|CYGNUS|PROGRESS|SOYUZ)\b"
)
DEBRIS_PATTERN = re.compile(r"\b(DEB|R/B)\b")
CONSTELLATIONS = ("ONEWEB", "KUIPER", "QIANFAN", "HULIANWANG", "GUOWANG", "IRIDIUM", "GLOBALSTAR", "ORBCOMM")


@dataclass
class Catalog:
    satellites: list
    categories: list
    records: list           # source elements per satellite, see make_satrec()
    updated: float | None   # unix time the data was downloaded, None for bundled data
    source: str


def orbital_period_minutes(sat):
    # no_kozai is the mean motion in radians per minute
    return 2 * np.pi / sat.model.no_kozai


def make_satrec(record):
    """A fresh, independent SGP4 model (for use on another thread)."""
    if record[0] == "tle":
        return Satrec.twoline2rv(record[1], record[2])
    satrec = Satrec()
    omm.initialize(satrec, record[1])
    return satrec


def classify(sat, groups):
    name = sat.name.upper()

    if "stations" in groups or STATION_PATTERN.search(name):
        return "station"
    if RECENT_GROUP in groups:
        return "recent"
    if groups & set(DEBRIS_GROUPS) or DEBRIS_PATTERN.search(name):
        return "debris"
    if "STARLINK" in name:
        return "starlink"
    if any(c in name for c in CONSTELLATIONS):
        return "constellation"
    if "gnss" in groups:
        return "gnss"
    if "weather" in groups or "science" in groups:
        return "weather"
    if 1300 < orbital_period_minutes(sat) < 1600 and sat.model.ecco < 0.05:
        return "geo"
    return "other"


def _cached(group, max_age_hours):
    """Freshest cached file for a group (.csv preferred over a legacy .tle), or None."""
    best = None
    for ext in (".csv", ".tle"):
        path = os.path.join(CACHE_DIR, group + ext)
        if os.path.exists(path):
            age = (time.time() - os.path.getmtime(path)) / 3600
            if age < max_age_hours and (best is None or age < best[1]):
                best = (path, age)
    return best


def fetch_group(group, offline=False):
    """Path to up-to-date elements for a CelesTrak group, downloading them if the cache is stale.

    Falls back to a stale cached copy if the download fails. Returns None if nothing is available.
    """
    fresh = _cached(group, CACHE_MAX_AGE_HOURS)
    if fresh:
        return fresh[0]
    stale = _cached(group, float("inf"))
    if offline:
        return stale[0] if stale else None

    url = CELESTRAK_URL.format(group=group)
    path = os.path.join(CACHE_DIR, f"{group}.csv")
    try:
        print(f"Downloading '{group}' from CelesTrak...")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if not data.startswith(b"OBJECT_NAME"):
            raise ValueError(f"unexpected response: {data[:80]!r}")

        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  could not download '{group}': {exc}")
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
            print("  (CelesTrak temporarily blocks clients that re-download the same data within ~2 hours)")
        if stale:
            print(f"  using cached copy from {stale[1]:.1f} h ago")
            return stale[0]
        return None


def read_elements(path, ts):
    """Yield (EarthSatellite, record) from an OMM .csv or a 3-line .tle file."""
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            for fields in omm.parse_csv(f):
                yield EarthSatellite.from_omm(ts, fields), ("omm", fields)
        return

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip() for line in f if line.strip()]
    for i in range(len(lines) - 2):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 ") and not name.startswith(("1 ", "2 ")):
            yield EarthSatellite(line1, line2, name.strip(), ts), ("tle", line1, line2)


def load_catalog(ts, groups=DEFAULT_GROUPS, offline=False):
    satellites = {}
    groups_of = defaultdict(set)
    newest = None

    for group in groups:
        path = fetch_group(group, offline)
        if path is None:
            continue
        newest = max(newest or 0, os.path.getmtime(path))
        for sat, record in read_elements(path, ts):
            satnum = sat.model.satnum
            satellites.setdefault(satnum, (sat, record))
            groups_of[satnum].add(group)

    source = "CelesTrak"
    if not satellites:
        print("No live data available - falling back to the bundled stations.tle")
        if not os.path.exists(BUNDLED_TLE):
            raise FileNotFoundError(f"Could not find TLE file:\n{BUNDLED_TLE}")
        for sat, record in read_elements(BUNDLED_TLE, ts):
            satellites.setdefault(sat.model.satnum, (sat, record))
            groups_of[sat.model.satnum].add("stations")
        newest, source = None, "bundled sample"

    sats = [sat for sat, _ in satellites.values()]
    records = [record for _, record in satellites.values()]
    categories = [classify(sat, groups_of[sat.model.satnum]) for sat in sats]
    print(f"Loaded {len(sats):,} objects ({source})")
    return Catalog(sats, categories, records, newest, source)
