"""
The observer's location: saved settings, typed coordinates and city lookup.

City names are looked up with OpenStreetMap's Nominatim service (free, no key). Its
usage policy asks for an identifying User-Agent and at most one request per second;
the app only sends one when Enter is pressed. Lookups run on a background thread so
the 3D view never freezes.
"""

import json
import queue
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

import settings

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q={query}"
USER_AGENT = "SatelliteTracker/1.0 (educational PyVista viewer)"
COORDINATES = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class Location:
    lat: float
    lon: float
    name: str


def format_coordinates(lat, lon):
    return f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"


def parse_coordinates(text):
    """'51.51, -0.13' (or '51.51 -0.13') -> Location, or None if not a valid pair."""
    match = COORDINATES.match(text)
    if not match:
        return None
    lat, lon = float(match[1]), float(match[2])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Location(lat, lon, format_coordinates(lat, lon))


def load_location():
    """The saved location, or None if there is none (or it is unreadable)."""
    try:
        data = settings.load_setting("location")
        return Location(float(data["lat"]), float(data["lon"]), str(data["name"]))
    except (KeyError, TypeError, ValueError):
        return None


def save_location(location):
    settings.save_setting("location", asdict(location))


def place_name(display_name):
    """'Honolulu, Honolulu County, Hawaii, United States' -> 'Honolulu, United States'."""
    parts = [part.strip() for part in display_name.split(",")]
    return parts[0] if len(parts) == 1 else f"{parts[0]}, {parts[-1]}"


class Geocoder:
    """Looks up a place name on a background thread. Call lookup(), then poll() each frame."""

    def __init__(self):
        self.results = queue.Queue()
        self.busy = False

    def lookup(self, query):
        self.busy = True
        threading.Thread(target=self._worker, args=(query,), daemon=True).start()

    def poll(self):
        """(Location, None) or (None, reason) once a lookup has finished, otherwise None."""
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _worker(self, query):
        try:
            places = json.loads(self._fetch(NOMINATIM_URL.format(query=urllib.parse.quote(query))))
            if not places:
                self.results.put((None, "not found"))
                return
            place = places[0]
            self.results.put((Location(float(place["lat"]), float(place["lon"]),
                                       place_name(place["display_name"])), None))
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            self.results.put((None, type(exc).__name__))

    @staticmethod
    def _fetch(url):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()
