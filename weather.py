"""
Live weather overlays draped just above the globe.

Layers are fetched as global equirectangular images (lon -180..180, lat 90..-90), which
match the globe's texture coordinates, so they line up with the map underneath.

    clouds  EUMETSAT "world cloud map" - IR 10.8 um mosaic of all geostationary
            weather satellites, every 3 hours. Converted to white clouds with alpha.
    precip  NASA GPM IMERG precipitation rate (30 min, near real time) via NASA GIBS.

Both are free public WMS services; no API key needed. Downloads run on a background
thread so the 3D view never freezes, and results are cached on disk.
"""

import io
import os
import queue
import re
import threading
import time
import urllib.request

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "weather_cache")
USER_AGENT = "SatelliteTracker/1.0 (educational PyVista viewer)"

# How long a downloaded layer is considered fresh before we look for a newer one
REFRESH_SECONDS = 30 * 60


def _wms_map(base, layer, width, height):
    return (f"{base}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer}&STYLES="
            f"&FORMAT=image/png&TRANSPARENT=TRUE&CRS=EPSG:4326&BBOX=-90,-180,90,180"
            f"&WIDTH={width}&HEIGHT={height}")


def _wms_caps(base):
    return f"{base}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"


EUMETSAT_WMS = "https://view.eumetsat.int/geoserver/ows"
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

LAYERS = {
    "clouds": {
        "label": "Clouds",
        "source": "EUMETSAT IR mosaic",
        "base": EUMETSAT_WMS,
        "layer": "mumi:worldcloudmap_ir108",
        "size": (4096, 2048),
    },
    "precip": {
        "label": "Precipitation",
        "source": "NASA IMERG",
        "base": GIBS_WMS,
        "layer": "IMERG_Precipitation_Rate_30min",
        "size": (2048, 1024),
    },
}

# Overlay modes the UI cycles through, and the layers each one shows
MODES = {
    "off": ("Off", []),
    "clouds": ("Clouds", ["clouds"]),
    "precip": ("Rain", ["precip"]),
    "both": ("Both", ["clouds", "precip"]),
}


def clouds_to_rgba(image):
    """IR brightness -> white cloud layer. Cold (bright) cloud tops opaque, warm ground clear."""
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    brightness = rgba[..., :3].mean(axis=2)
    alpha = np.clip((brightness - 100.0) / 120.0, 0.0, 1.0) ** 1.3
    alpha *= rgba[..., 3] / 255.0

    out = np.empty(rgba.shape, dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = 242, 246, 255
    out[..., 3] = (alpha * 235).astype(np.uint8)
    return out


def precip_to_rgba(image):
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[..., 3] = (rgba[..., 3] * 0.9).astype(np.uint8)
    return rgba


PROCESSORS = {"clouds": clouds_to_rgba, "precip": precip_to_rgba}


def _get(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _latest_time(spec):
    """Observation time of the newest image (the layer's default TIME in GetCapabilities)."""
    caps = _get(_wms_caps(spec["base"])).decode("utf-8", errors="replace")
    start = caps.find(f"<Name>{spec['layer']}</Name>")
    if start < 0:
        return None
    block = caps[start:caps.find("</Layer>", start)]
    match = re.search(r'<Dimension[^>]*name="time"[^>]*default="([^"]+)"', block)
    return match.group(1) if match else None


class WeatherService:
    """Fetches layers on demand in the background. Call request() freely, poll() each frame."""

    def __init__(self, offline=False):
        self.offline = offline
        self.results = queue.Queue()
        self.loading = set()
        self.status = {key: "Not loaded" for key in LAYERS}
        self.observed = {key: None for key in LAYERS}
        self.last_attempt = {key: 0.0 for key in LAYERS}

    def request(self, key):
        if key in self.loading:
            return
        # Retry failures after a minute, refresh good layers after REFRESH_SECONDS
        wait = REFRESH_SECONDS if self.observed[key] else 60
        if time.time() - self.last_attempt[key] < wait:
            return
        self.last_attempt[key] = time.time()
        self.loading.add(key)
        if self.observed[key] is None:
            self.status[key] = "Loading..."
        threading.Thread(target=self._worker, args=(key,), daemon=True).start()

    def poll(self):
        """Yield (key, rgba array) for layers that finished loading since the last poll."""
        while True:
            try:
                key, rgba, observed, error = self.results.get_nowait()
            except queue.Empty:
                return
            self.loading.discard(key)
            if rgba is None:
                if self.observed[key] is None:
                    self.status[key] = f"Unavailable ({error})"
                continue
            self.observed[key] = observed
            self.status[key] = f"{LAYERS[key]['source']}  ·  {observed}"
            yield key, rgba

    # ---- background thread ---------------------------------------------------
    def _worker(self, key):
        try:
            image, observed = self._load(key)
            self.results.put((key, PROCESSORS[key](image), observed, None))
        except Exception as exc:  # network errors, bad images - report and keep running
            self.results.put((key, None, None, type(exc).__name__))

    def _load(self, key):
        spec = LAYERS[key]
        image_path = os.path.join(CACHE_DIR, f"{key}.png")
        time_path = os.path.join(CACHE_DIR, f"{key}.time")

        cached = os.path.exists(image_path) and os.path.exists(time_path)
        fresh = cached and time.time() - os.path.getmtime(image_path) < REFRESH_SECONDS
        if cached and (fresh or self.offline):
            with open(time_path, encoding="utf-8") as f:
                return Image.open(image_path), f.read().strip()
        if self.offline:
            raise FileNotFoundError("offline")

        try:
            raw_time = _latest_time(spec)
            data = _get(_wms_map(spec["base"], spec["layer"], *spec["size"]))
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception:
            if cached:  # stale beats nothing
                with open(time_path, encoding="utf-8") as f:
                    return Image.open(image_path), f.read().strip()
            raise

        observed = _format_time(raw_time)
        os.makedirs(CACHE_DIR, exist_ok=True)
        image.save(image_path)
        with open(time_path, "w", encoding="utf-8") as f:
            f.write(observed)
        return image, observed


def _format_time(iso):
    if not iso:
        return "latest"
    match = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", iso)
    return f"{match.group(1)} {match.group(2)} UTC" if match else iso
