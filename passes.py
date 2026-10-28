"""
Passes of a satellite over an observer: when it rises, how high it climbs, and whether
it can be seen with the naked eye.

A pass is "visible" when, while the satellite is above MIN_ELEVATION_DEG, it is in
sunlight and the observer's sky is dark (Sun more than 6 degrees below the horizon).
That is when satellites shine against a dark sky. Sunlight uses the same low-precision
Sun direction and cylindrical Earth shadow as the 3D view, so no planetary ephemeris
has to be downloaded.
"""

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0
MIN_ELEVATION_DEG = 10.0
DARK_SUN_ELEVATION_DEG = -6.0     # end of civil twilight
SEARCH_DAYS = 2.0
SAMPLES_PER_PASS = 60


@dataclass
class Pass:
    rise_tt: float           # TT Julian dates
    peak_tt: float
    set_tt: float
    max_elevation: float     # degrees
    rise_azimuth: float      # degrees from north, clockwise
    set_azimuth: float
    visible: bool


def sun_direction(t):
    """Unit vector to the Sun in the equator-of-date frame (close enough to TEME).

    Low-precision formula from the Astronomical Almanac, good to about 0.01 degrees,
    which avoids downloading a planetary ephemeris. For an array of times the result
    is 3 x N.
    """
    n = t.tt - 2451545.0
    mean_longitude = np.radians((280.460 + 0.9856474 * n) % 360)
    mean_anomaly = np.radians((357.528 + 0.9856003 * n) % 360)
    ecliptic_longitude = (mean_longitude + np.radians(1.915) * np.sin(mean_anomaly)
                          + np.radians(0.020) * np.sin(2 * mean_anomaly))
    obliquity = np.radians(23.439 - 0.0000004 * n)
    return np.array([np.cos(ecliptic_longitude),
                     np.cos(obliquity) * np.sin(ecliptic_longitude),
                     np.sin(obliquity) * np.sin(ecliptic_longitude)])


def sunlit(positions_km, sun):
    """Which positions (N x 3) are outside Earth's cylindrical shadow. `sun` is one
    direction or one per position (N x 3)."""
    along = np.einsum("ij,ij->i", positions_km, np.broadcast_to(sun, positions_km.shape))
    across2 = np.einsum("ij,ij->i", positions_km, positions_km) - along ** 2
    return ~((along < 0) & (across2 < EARTH_RADIUS_KM ** 2))


def find_passes(sat, observer, ts, start_tt, days=SEARCH_DAYS):
    """Passes of `sat` over `observer` (a skyfield wgs84 position) in [start, start + days].

    Returns (passes, always_up). always_up is True for a satellite that never rises or
    sets because it stays above the horizon, e.g. a geostationary one in view.
    """
    t0, t1 = ts.tt_jd(start_tt), ts.tt_jd(start_tt + days)
    times, events = sat.find_events(observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG)
    if not np.isin(events, (0, 2)).any():   # never rises or sets (culminations alone are possible)
        elevation = (sat - observer).at(t0).altaz()[0].degrees
        return [], bool(elevation >= MIN_ELEVATION_DEG)

    passes, rise = [], None
    if events[0] != 0:
        rise = start_tt                      # already up when the search starts
    for t, event in zip(times.tt, events):
        if event == 0:
            rise = t
        elif event == 2 and rise is not None:
            passes.append(_describe(sat, observer, ts, rise, t))
            rise = None
    return passes, False                     # a pass cut off at the end is dropped


def _describe(sat, observer, ts, rise_tt, set_tt):
    t = ts.tt_jd(np.linspace(rise_tt, set_tt, SAMPLES_PER_PASS))
    elevation, azimuth, _ = (sat - observer).at(t).altaz()
    peak = int(np.argmax(elevation.degrees))

    # GCRS vs the equator-of-date Sun differ by ~0.3 degrees, and geocentric vs geodetic
    # zenith by ~0.2: both negligible next to the 6 degree twilight margin
    sun = sun_direction(t)
    up = observer.at(t).position.km
    sun_elevation = np.degrees(np.arcsin(np.einsum("ij,ij->j", up, sun) / np.linalg.norm(up, axis=0)))
    lit = sunlit(sat.at(t).position.km.T, sun.T)
    visible = (elevation.degrees >= MIN_ELEVATION_DEG) & lit & (sun_elevation < DARK_SUN_ELEVATION_DEG)

    return Pass(rise_tt, float(t.tt[peak]), set_tt, float(elevation.degrees[peak]),
                float(azimuth.degrees[0]), float(azimuth.degrees[-1]), bool(visible.any()))
