"""
Satellite Tracker

Every tracked object from CelesTrak (active satellites, recent launches and major debris
clouds), propagated with SGP4 on an accelerated simulation clock, over a rotating Earth
with day/night shading, live weather, coverage footprints, ground tracks and
close-approach screening.

Usage
    python main.py                       # everything (see data.DEFAULT_GROUPS)
    python main.py --groups stations visual
    python main.py --offline             # only use cached / bundled data

Controls
    Click        track a satellite (chase camera); click panel rows and buttons
    / or Ctrl+F  search by name or NORAD ID
    Tab          next satellite of the same type (Shift+Tab: previous)
    Esc / H      release target and return to the home view
    Space        pause / resume
    Left/Right   slower / faster simulation
    0 / Home     back to real time (live)
    L            labels
    N F G C      day/night, coverage footprint, ground track, close approaches
    W            cycle weather overlay
    Scroll       zoom (also adjusts the chase distance while tracking)
"""

import argparse
import math
import time

import numpy as np
import pyvista as pv
from pyvista import examples
from sgp4.api import SatrecArray, accelerated
from skyfield.api import load, wgs84

from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

import conjunctions
import weather
from data import CATEGORIES, DEFAULT_GROUPS, load_catalog, orbital_period_minutes
from ui import (ACCENT, BACKGROUND, DISABLED, FONT_MONO, FONT_REGULAR, FONT_SEMIBOLD, MUTED, TEXT,
                WARNING, Panel, make_text, set_text_color)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
MU_KM3_S2 = 398600.4418

SPEEDS = [1, 10, 60, 300, 1000, 3000]
DEFAULT_SPEED_INDEX = 2
TRACK_SAMPLES = 240
MAX_EXTRAPOLATION_S = 60          # re-run SGP4 for everything after this much simulated time
TARGET_FPS = 60
HOVER_RADIUS_PX = 10
CAMERA_TRANSITION_S = 1.2
MAX_LABELS = 60
LABEL_REFRESH_S = 0.25
TOAST_SECONDS = 2.5
MAX_SEARCH_RESULTS = 8

# Shell radii (multiples of the Earth radius). Drawn in this order, so the night shade
# darkens the ground and the weather, while footprint and ground track stay bright.
WEATHER_SHELL_RADIUS = {"clouds": 1.006, "precip": 1.009}
NIGHT_SHELL_RADIUS = 1.010
FOOTPRINT_RADIUS = 1.012
GROUND_TRACK_RADIUS = 1.013

NIGHT_MAX_OPACITY = 0.72
SHADOW_DIM = 0.3                   # brightness of satellites in Earth's shadow
FOOTPRINT_MIN_ELEVATION_DEG = 10   # typical minimum elevation for a usable radio link
FOOTPRINT_RINGS = 10
FOOTPRINT_SEGMENTS = 96
GROUND_TRACK_ORBITS = 3
GROUND_TRACK_SAMPLES = 720

MAX_APPROACH_ROWS = 6
ALERT_MISS_KM = 1.0                # approaches drawn as 3D alerts while they happen
ALERT_WINDOW_S = 120
TRACKED_ALERT_LEAD_S = 600         # warn about the tracked satellite's approaches this early

VTK_CURSOR_DEFAULT = 0
VTK_CURSOR_HAND = 9


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sgp4_time(t):
    """Julian date split (whole, fraction) the same way Skyfield feeds SGP4 (UTC-based)."""
    return t.whole, t.tai_fraction - t._leap_seconds() / 86400.0


def sun_direction(t):
    """Unit vector to the Sun in the equator-of-date frame (close enough to TEME).

    Low-precision formula from the Astronomical Almanac, good to about 0.01 degrees,
    which avoids downloading a planetary ephemeris.
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


def static_info(sat):
    """Orbit facts that don't change as the satellite moves."""
    model = sat.model
    mean_motion_rad_s = model.no_kozai / 60.0
    semi_major_axis = (MU_KM3_S2 / mean_motion_rad_s ** 2) ** (1 / 3)

    return {
        "norad": str(model.satnum),
        "cospar": model.intldesg.strip() or "—",
        "inclination": f"{np.degrees(model.inclo):.2f}°",
        "period": f"{orbital_period_minutes(sat):,.1f} min",
        "perigee": f"{semi_major_axis * (1 - model.ecco) - EARTH_RADIUS_KM:,.0f} km",
        "apogee": f"{semi_major_axis * (1 + model.ecco) - EARTH_RADIUS_KM:,.0f} km",
        "epoch": sat.epoch.utc_strftime("%Y-%m-%d %H:%M"),
    }


def format_lat(degrees):
    return f"{abs(degrees):.2f}° {'N' if degrees >= 0 else 'S'}"


def format_lon(degrees):
    return f"{abs(degrees):.2f}° {'E' if degrees >= 0 else 'W'}"


def live_info(sat, t):
    geocentric = sat.at(t)
    lat, lon = wgs84.latlon_of(geocentric)

    return {
        "latitude": format_lat(lat.degrees),
        "longitude": format_lon(lon.degrees),
        "altitude": f"{wgs84.height_of(geocentric).km:,.1f} km",
        "velocity": f"{np.linalg.norm(geocentric.velocity.km_per_s):.2f} km/s",
    }


def format_age(seconds):
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.0f} days ago"


def format_duration(seconds):
    minutes = int(abs(seconds) // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def format_eta(seconds):
    if abs(seconds) < 60:
        return "now"
    text = format_duration(seconds)
    return f"in {text}" if seconds > 0 else f"{text} ago"


def truncate(text, length):
    return text if len(text) <= length else text[:length - 1] + "…"


def random_sphere_points(n, radius):
    phi = np.random.uniform(0, 2 * np.pi, n)
    costheta = np.random.uniform(-1, 1, n)
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.column_stack((x, y, z))


def make_polyline(points):
    """
    Convert Nx3 points into a connected PolyData line for PyVista.
    """
    n = len(points)
    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.hstack(([n], np.arange(n)))
    return poly


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def normalize(v):
    return v / np.linalg.norm(v)


def rgba_line(points, color):
    """Polyline with per-point RGBA colours (alpha is updated in place later)."""
    mesh = make_polyline(points)
    rgba = np.zeros((len(points), 4), dtype=np.uint8)
    rgba[:, :3] = pv.Color(color).int_rgb
    mesh["rgba"] = rgba
    return mesh


def add_colored(plotter, mesh, array, point_size=None, spheres=False, line_width=None, lighting=False):
    """Add a mesh coloured directly by an RGB/RGBA point array.

    Uses a plain VTK mapper. pyvista's add_mesh(scalars=...) inserts a Python pipeline
    stage that runs on every render, which costs several ms per frame across dozens of
    meshes that change every frame.
    """
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(array)
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    prop = actor.GetProperty()
    prop.SetLighting(lighting)
    if point_size:
        prop.SetRepresentationToPoints()
        prop.SetPointSize(point_size)
        prop.SetRenderPointsAsSpheres(spheres)
    if line_width:
        prop.SetLineWidth(line_width)
    plotter.renderer.AddActor(actor)
    return actor


class Track:
    """One sampled orbit for a satellite; only the fade (alpha) slides along it each frame."""

    def __init__(self, mesh, actor):
        self.mesh = mesh
        self.actor = actor
        self.rgba = mesh["rgba"]
        self.vtk_rgba = mesh.GetPointData().GetArray("rgba")
        self.epoch = None


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
class SatelliteTracker:
    def __init__(self, catalog, ts, off_screen=False, offline=False):
        self.catalog = catalog
        self.satellites = catalog.satellites
        self.ts = ts
        self.n = len(self.satellites)

        self.names = [sat.name.strip() for sat in self.satellites]
        self.names_lower = [name.lower() for name in self.names]
        self.norad = [str(sat.model.satnum) for sat in self.satellites]
        self.categories = np.array(catalog.categories)
        self.category_members = {key: np.flatnonzero(self.categories == key) for key in CATEGORIES}
        self.category_visible = {key: True for key in CATEGORIES}
        self.priority = np.array([CATEGORIES[c][3] for c in self.categories])
        self.periods_days = np.array([orbital_period_minutes(sat) for sat in self.satellites]) / 1440.0
        self.sat_array = SatrecArray([sat.model for sat in self.satellites])
        self._static_cache = {}
        self.base_tt = None   # time of the last full SGP4 run, see _update_positions()

        # Simulation clock (Terrestrial Time, Julian days)
        self.sim_tt = ts.now().tt
        self.speed_index = DEFAULT_SPEED_INDEX
        self.paused = False
        self.live = False      # clock follows the wall clock exactly (see go_live)
        self.last_wall = time.monotonic()

        # Interaction state
        self.mouse = None
        self.press_pos = None
        self.hover_index = None
        self.hover_group = []
        self.ui_hover = None
        self.selected = None
        self.follow_zoom = 1.0
        self.labels_visible = False
        self.label_slots = []
        self.last_label_refresh = 0.0
        self.running = True
        self.camera_anim = None
        self.last_text_update = 0.0
        self.toast_until = 0.0
        self.cursor = VTK_CURSOR_DEFAULT

        # Map layers
        self.layers = {"night": True, "footprint": True, "groundtrack": True, "approaches": True}
        self.sun = np.array([1.0, 0.0, 0.0])
        self.sunlit = np.ones(self.n, dtype=bool)
        self.last_night_update = 0.0
        self.ground_track_key = None
        self.coverage_km = None

        # Search
        self.search_active = False
        self.search_query = ""
        self.search_results = []
        self.search_matches = 0
        self.search_sel = 0

        # Close-approach screening
        self.screener = conjunctions.ApproachScreener(catalog.records)
        self.screen_started_tt = None
        self.events = None   # numpy columns of screener.events, see _ingest_events()
        self.hide_starlink_pairs = True
        self.approach_rows = []
        self.watch = None
        self.alerted = set()

        self.weather = weather.WeatherService(offline=offline)
        self.weather_mode = "off"

        self.pl = pv.Plotter(window_size=(1440, 900), off_screen=off_screen)
        self.pl.set_background(BACKGROUND)
        self.ren = self.pl.renderer

        self._build_scene()
        self._build_ui()
        self._bind_events()

        self.home_camera = [
            (self.R * 4.0, -self.R * 2.0, self.R * 1.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self.pl.camera_position = self.home_camera

    def static(self, index):
        if index not in self._static_cache:
            self._static_cache[index] = static_info(self.satellites[index])
        return self._static_cache[index]

    # ---- scene -----------------------------------------------------------------
    def _build_scene(self):
        pl = self.pl

        globe = examples.load_globe()
        texture = examples.load_globe_texture()
        self.R = max(abs(b) for b in globe.bounds)
        self.scale = self.R / EARTH_RADIUS_KM

        # The bundled globe is low-poly; its flat facets poke above low orbits when the
        # camera is close. Subdivide it and push the new points back out onto a sphere.
        globe = globe.triangulate().subdivide(3, subfilter="linear")
        globe.points = globe.points / np.linalg.norm(globe.points, axis=1, keepdims=True) * self.R
        # Texture u = 0.5 (longitude 0) faces +x and north is +z, so rotating the Earth about z
        # by the Greenwich sidereal angle puts it in the same frame (TEME) SGP4 positions use.
        self.globe_actor = pl.add_mesh(globe, texture=texture, smooth_shading=True,
                                       specular=0.15, specular_power=12)
        self.earth_actors = [self.globe_actor]

        # Weather shells: copies of the globe (same texture coordinates) slightly larger
        self.weather_actors = {}
        for key, radius in WEATHER_SHELL_RADIUS.items():
            shell = globe.copy()
            shell.points = shell.points * radius
            actor = pl.add_mesh(shell, color="white", smooth_shading=True, lighting=(key == "clouds"),
                                pickable=False)
            actor.SetVisibility(False)
            self.weather_actors[key] = actor
            self.earth_actors.append(actor)
        self.weather_loaded = set()

        # Night side: a dark shell whose per-vertex opacity follows the Sun. It doesn't spin
        # with the Earth; the terminator stays fixed relative to the Sun.
        night = pv.Sphere(radius=self.R * NIGHT_SHELL_RADIUS, theta_resolution=180, phi_resolution=90)
        self.night_normals = night.points / np.linalg.norm(night.points, axis=1, keepdims=True)
        rgba = np.zeros((night.n_points, 4), dtype=np.uint8)
        rgba[:, :3] = pv.Color("#000814").int_rgb
        night["rgba"] = rgba
        self.night_actor = add_colored(pl, night, "rgba")
        self.night_rgba = night["rgba"]
        self.night_vtk_rgba = night.GetPointData().GetArray("rgba")

        # Thin atmosphere shell
        atmosphere = pv.Sphere(radius=self.R * 1.015, theta_resolution=120, phi_resolution=60)
        pl.add_mesh(atmosphere, color="#4f9dff", opacity=0.08, smooth_shading=True, lighting=False, pickable=False)

        # Coverage footprint of the selected satellite: a filled cap (minimum elevation)
        # inside a ring at the true horizon.
        rings, segments = FOOTPRINT_RINGS, FOOTPRINT_SEGMENTS
        faces = []
        for k in range(rings):
            for m in range(segments):
                a, b = k * segments + m, k * segments + (m + 1) % segments
                faces.append([4, a, b, b + segments, a + segments])
        self.footprint_mesh = pv.PolyData(np.zeros(((rings + 1) * segments, 3)), np.array(faces).ravel())
        self.footprint_actor = pl.add_mesh(self.footprint_mesh, color=ACCENT, opacity=0.16, lighting=False,
                                           pickable=False)
        self.horizon_mesh = make_polyline(np.zeros((segments + 1, 3)))
        self.horizon_actor = pl.add_mesh(self.horizon_mesh, color=ACCENT, line_width=1.5, opacity=0.85,
                                         lighting=False, pickable=False)
        for actor in (self.footprint_actor, self.horizon_actor):
            actor.SetVisibility(False)

        # Ground track of the selected satellite, stored in Earth-fixed coordinates and
        # spun with the Earth
        self.ground_mesh = rgba_line(np.zeros((GROUND_TRACK_SAMPLES + 1, 3)), ACCENT)
        self.ground_actor = add_colored(pl, self.ground_mesh, "rgba", line_width=2)
        self.ground_actor.SetVisibility(False)
        self.earth_actors.append(self.ground_actor)

        # Star field, far outside the orbits
        star_radius = self.R * 60
        pl.add_points(random_sphere_points(5000, star_radius), color="#8090b0", point_size=1.5,
                      opacity=0.6, lighting=False, pickable=False)
        pl.add_points(random_sphere_points(500, star_radius), color="white", point_size=2.5,
                      opacity=0.9, lighting=False, pickable=False)

        # Satellite markers: one point cloud per category (own size and visibility). Colours
        # are per point so satellites in Earth's shadow can be dimmed.
        self.positions, self.velocities = self._update_positions(self.sim_time())
        self.category_meshes = {}
        self.category_actors = {}
        for key, (_, color, size, _) in CATEGORIES.items():
            members = self.category_members[key]
            if len(members) == 0:
                continue
            mesh = pv.PolyData(self.positions[members])
            mesh["rgb"] = np.tile(np.array(pv.Color(color).int_rgb, dtype=np.uint8), (len(members), 1))
            actor = add_colored(pl, mesh, "rgb", point_size=size, spheres=True, lighting=True)
            self.category_meshes[key] = (mesh, mesh["rgb"], mesh.GetPointData().GetArray("rgb"),
                                         np.array(pv.Color(color).int_rgb, dtype=np.float32))
            self.category_actors[key] = actor

        # Orbit tracks are created on demand (stations, plus whatever is hovered/selected)
        self.tracks = {}

        # Hover and selection markers
        self.hover_mesh = pv.PolyData(np.zeros((1, 3)))
        self.hover_actor = pl.add_mesh(self.hover_mesh, color="white", style="points", point_size=20,
                                       render_points_as_spheres=True, opacity=0.35, pickable=False)
        self.hover_actor.SetVisibility(False)

        self.select_mesh = pv.PolyData(np.zeros((1, 3)))
        self.select_actor = pl.add_mesh(self.select_mesh, color=ACCENT, style="points", point_size=26,
                                        render_points_as_spheres=True, opacity=0.45, pickable=False)
        self.select_actor.SetVisibility(False)

        # Line from the selected satellite straight down to the ground
        self.nadir_mesh = make_polyline(np.zeros((2, 3)))
        self.nadir_actor = pl.add_mesh(self.nadir_mesh, color=ACCENT, line_width=1.5, opacity=0.6, pickable=False)
        self.nadir_actor.SetVisibility(False)

        # Close-approach alerts: rings on objects in an approach right now, and a line
        # joining the pair being watched
        self.alert_mesh = pv.PolyData(np.zeros((1, 3)))
        self.alert_actor = pl.add_mesh(self.alert_mesh, color=WARNING, style="points", point_size=18,
                                       render_points_as_spheres=True, opacity=0.55, pickable=False)
        self.alert_actor.SetVisibility(False)
        self.pair_mesh = make_polyline(np.zeros((2, 3)))
        self.pair_actor = pl.add_mesh(self.pair_mesh, color=WARNING, line_width=2, opacity=0.9, pickable=False)
        self.pair_actor.SetVisibility(False)

        pl.enable_anti_aliasing("msaa")

    # ---- UI --------------------------------------------------------------------
    def _build_ui(self):
        ren = self.ren

        # Title (top-left)
        self.title_panel = Panel(ren, accent=ACCENT)
        self.title_panel.add_text(17, TEXT, FONT_SEMIBOLD, "SATELLITE TRACKER")
        self.title_subtitle = self.title_panel.add_text(11, MUTED, FONT_REGULAR)

        # Search (left, under the title)
        self.search_panel = Panel(ren, min_width=360)
        self.search_prompt = self.search_panel.add_field("Search by name or NORAD ID", 12, value_color=ACCENT,
                                                         key="search:open")
        self.search_prompt.SetInput("/")
        self.search_input = self.search_panel.add_text(14, TEXT, FONT_MONO)
        self.search_results_rows = [
            self.search_panel.add_field("", 11, glyph_color=TEXT, value_color=MUTED, value_font=FONT_REGULAR,
                                        key=f"result:{k}")
            for k in range(MAX_SEARCH_RESULTS)]
        self.search_hint = self.search_panel.add_text(10, MUTED, FONT_REGULAR)

        # Close approaches (left, under the search)
        self.approach_panel = Panel(ren, min_width=400)
        self.approach_panel.add_text(10, MUTED, FONT_SEMIBOLD,
                                     f"CLOSE APPROACHES  ·  UNDER {conjunctions.THRESHOLD_KM:g} KM  ·  "
                                     f"NEXT {conjunctions.WINDOW_HOURS:g} H")
        self.approach_status = self.approach_panel.add_text(11, TEXT, FONT_REGULAR)
        self.approach_panel.add_gap(2)
        self.approach_values = [
            self.approach_panel.add_field("", 11, glyph_color=WARNING, value_color=TEXT, key=f"approach:{k}")
            for k in range(MAX_APPROACH_ROWS)]
        self.approach_panel.add_gap(4)
        self.approach_filter = self.approach_panel.add_field("Hide Starlink–Starlink pairs", 10, value_color=ACCENT,
                                                             key="approach-filter")
        self.approach_panel.add_text(10, MUTED, FONT_REGULAR,
                                     "Click an event to watch it  ·  estimates (orbit data ~1 km accurate)")

        # Simulation clock and speed controls (top-centre)
        self.clock_panel = Panel(ren)
        self.clock_status = self.clock_panel.add_text(10, MUTED, FONT_SEMIBOLD)
        self.clock_text = self.clock_panel.add_text(20, TEXT, FONT_MONO)
        self.clock_panel.add_gap(4)
        self.speed_chips = self.clock_panel.add_chips(
            ["PAUSE"] + [f"{s:,}×" for s in SPEEDS] + ["LIVE"],
            keys=["pause"] + [f"speed:{i}" for i in range(len(SPEEDS))] + ["live"])

        # Legend (top-right) - rows toggle categories
        self.legend_panel = Panel(ren)
        self.legend_panel.add_text(10, MUTED, FONT_SEMIBOLD, "OBJECT TYPES  ·  CLICK TO SHOW / HIDE")
        self.legend_panel.add_gap(2)
        for key, (label, color, _, _) in CATEGORIES.items():
            value = self.legend_panel.add_field(label, 12, glyph_color=color, value_color=MUTED,
                                                value_font=FONT_REGULAR, key=f"category:{key}")
            value.SetInput(f"{len(self.category_members[key]):,}")

        # Map layers and weather (right, under the legend)
        self.layers_panel = Panel(ren)
        self.layers_panel.add_text(10, MUTED, FONT_SEMIBOLD, "MAP LAYERS")
        self.layers_panel.add_gap(2)
        self.layer_values = {}
        for key, label in [("night", "Day / night"), ("footprint", "Coverage footprint"),
                           ("groundtrack", "Ground track"), ("approaches", "Close approaches")]:
            self.layer_values[key] = self.layers_panel.add_field(label, 11, value_color=ACCENT, key=f"layer:{key}")
        self.night_note = self.layers_panel.add_text(10, MUTED, FONT_REGULAR,
                                                     "Dimmed satellites are in Earth's shadow")
        self.layers_panel.add_gap(6)
        self.layers_panel.add_text(10, MUTED, FONT_SEMIBOLD, "WEATHER  ·  W")
        self.layers_panel.add_gap(2)
        self.weather_chips = self.layers_panel.add_chips(
            [label for label, _ in weather.MODES.values()], keys=[f"weather:{m}" for m in weather.MODES])
        self.layers_panel.add_gap(2)
        self.weather_status = {key: self.layers_panel.add_text(10, MUTED, FONT_REGULAR) for key in weather.LAYERS}

        # Tracking card (bottom-left)
        self.track_panel = Panel(ren, accent=ACCENT, min_width=330)
        self.track_panel.add_text(10, ACCENT, FONT_SEMIBOLD, "TRACKING")
        self.track_name = self.track_panel.add_text(17, TEXT, FONT_SEMIBOLD)
        self.track_category = self.track_panel.add_text(11, MUTED)
        self.track_panel.add_gap(6)
        self.track_fields = {}
        for key, label in [("latitude", "Latitude"), ("longitude", "Longitude"), ("altitude", "Altitude"),
                           ("velocity", "Velocity"), ("light", "Illumination"), ("coverage", "Coverage radius"),
                           (None, None), ("norad", "NORAD ID"), ("cospar", "COSPAR ID"),
                           ("inclination", "Inclination"), ("period", "Period"), ("perigee", "Perigee"),
                           ("apogee", "Apogee"), ("epoch", "Elements epoch (UTC)"), (None, None),
                           ("approach", "Next close approach"), ("approach_with", "With"),
                           ("separation", "Separation now")]:
            if key is None:
                self.track_panel.add_gap(6)
            else:
                self.track_fields[key] = self.track_panel.add_field(label, 12)
        self.track_panel.add_gap(8)
        self.track_panel.add_text(10, MUTED, FONT_REGULAR, "Esc  release    ·    Tab  next    ·    Scroll  zoom")
        self.track_panel.set_visible(False)

        self.idle_panel = Panel(ren)
        self.idle_panel.add_text(10, MUTED, FONT_SEMIBOLD, "NO TARGET")
        self.idle_panel.add_text(12, TEXT, FONT_REGULAR, "Click a satellite, search with /, or press Tab")

        # Controls (bottom-right) - rows with an action are clickable too
        self.help_panel = Panel(ren)
        self.help_panel.add_text(10, MUTED, FONT_SEMIBOLD, "CONTROLS")
        self.help_panel.add_gap(2)
        self.help_values = {}
        for key, action, text in [("track", "Track satellite", "Click"), ("search", "Search", "/  or  Ctrl+F"),
                                  ("next", "Next satellite", "Tab"), ("home", "Release, home view", "Esc"),
                                  ("pause", "Pause / resume", ""), ("speed", "Simulation speed", "Left / Right"),
                                  ("live", "Back to real time", "0  or  Home"),
                                  ("labels", "Labels", ""), ("view", "Rotate / zoom", "Drag / Scroll")]:
            clickable = key in ("search", "next", "home", "pause", "live", "labels")
            self.help_values[key] = self.help_panel.add_field(action, 11, value_color=ACCENT,
                                                              key=f"action:{key}" if clickable else None)
            self.help_values[key].SetInput(text)

        # Satellite name labels: a pool assigned to the most important visible satellites
        self.labels = []
        for _ in range(MAX_LABELS):
            label = make_text(ren, 11, TEXT, FONT_REGULAR)
            label.GetTextProperty().SetOpacity(0.92)
            label.GetTextProperty().SetShadow(True)
            label.SetVisibility(False)
            self.labels.append(label)

        # Toast for quick feedback after an action (bottom-centre, drawn over labels)
        self.toast = Panel(ren, opacity=1.0)
        self.toast_text = self.toast.add_text(12, TEXT, FONT_SEMIBOLD)
        self.toast.set_visible(False)

        # Hover tooltip (created last and fully opaque so it draws over the other panels)
        self.tooltip = Panel(ren, opacity=1.0)
        self.tip_name = self.tooltip.add_text(13, TEXT, FONT_SEMIBOLD)
        self.tip_category = self.tooltip.add_text(10, MUTED, FONT_SEMIBOLD)
        self.tooltip.add_gap(4)
        self.tip_fields = {key: self.tooltip.add_field(label, 11) for key, label in
                           [("norad", "NORAD ID"), ("altitude", "Altitude"), ("velocity", "Velocity"),
                            ("latitude", "Latitude"), ("longitude", "Longitude"), ("light", "Illumination")]}
        self.tooltip.add_gap(4)
        self.tip_hint = self.tooltip.add_text(10, ACCENT, FONT_REGULAR)
        self.tooltip.set_visible(False)

        self.clickable_panels = [self.search_panel, self.approach_panel, self.clock_panel, self.legend_panel,
                                 self.layers_panel, self.help_panel]
        self.ui_panels = self.clickable_panels + [self.title_panel, self.track_panel, self.idle_panel]

        self._refresh_search()
        self._refresh_controls()
        self._refresh_legend()

    def _refresh_controls(self):
        self.speed_chips[0].text.SetInput("PLAY" if self.paused else "PAUSE")
        self.speed_chips[0].set_active(self.paused, WARNING)
        for i, chip in enumerate(self.speed_chips[1:-1]):
            chip.set_active(i == self.speed_index and not self.live)
        self.speed_chips[-1].set_active(self.live)

        for chip in self.weather_chips:
            chip.set_active(chip.key == f"weather:{self.weather_mode}")

        def on_off(actor, prefix, on, on_text="ON", off_text="OFF"):
            actor.SetInput(f"{prefix}  ·  {on_text if on else off_text}" if prefix else (on_text if on else off_text))
            set_text_color(actor, ACCENT if on else MUTED)

        for key, hotkey in (("night", "N"), ("footprint", "F"), ("groundtrack", "G"), ("approaches", "C")):
            on_off(self.layer_values[key], hotkey, self.layers[key])
        self.layers_panel.set_row_visible(self.night_note, self.layers["night"])
        on_off(self.help_values["labels"], "L", self.labels_visible)
        on_off(self.help_values["pause"], "Space", not self.paused, "RUNNING", "PAUSED")
        if self.paused:
            set_text_color(self.help_values["pause"], WARNING)
        on_off(self.approach_filter, "", self.hide_starlink_pairs)

    def _refresh_legend(self):
        for key, (_, color, _, _) in CATEGORIES.items():
            row = self.legend_panel.field(f"category:{key}")
            shown = self.category_visible[key]
            row["glyph"].set_color(color if shown else DISABLED)
            set_text_color(row["label"], TEXT if shown else DISABLED)
            set_text_color(row["value"], MUTED if shown else DISABLED)
            if key in self.category_actors:
                self.category_actors[key].SetVisibility(shown)

    def show_toast(self, message, color=TEXT):
        self.toast_text.SetInput(message)
        set_text_color(self.toast_text, color)
        self.toast.set_visible(True)
        self.toast_until = time.monotonic() + TOAST_SECONDS

    # ---- events ------------------------------------------------------------------
    def _bind_events(self):
        iren = self.pl.iren
        if iren is None:
            return

        # Drop pyvista's default key bindings (e.g. +/- change point sizes), and VTK's own
        # single-letter shortcuts (q/e quit, w wireframe, r reset, f fly-to...): an observer on
        # the style's CharEvent replaces its built-in handler. They'd fire while typing a search.
        iren.clear_key_event_callbacks()
        iren.style.AddObserver("CharEvent", lambda *_: None)

        def key(names, callback):
            for name in names:
                # Hotkeys are ignored while typing in the search box
                self.pl.add_key_event(name, lambda cb=callback: None if self.search_active else cb())

        key(["Tab"], self._on_tab)
        key(["Escape", "h", "H"], self.release)
        key(["space"], self.toggle_pause)
        key(["Right", "Up"], lambda: self.change_speed(+1))
        key(["Left", "Down"], lambda: self.change_speed(-1))
        key(["l", "L"], self.toggle_labels)
        key(["w", "W"], self.cycle_weather)
        key(["n", "N"], lambda: self.toggle_layer("night"))
        key(["f", "F"], self._on_f)
        key(["g", "G"], lambda: self.toggle_layer("groundtrack"))
        key(["c", "C"], lambda: self.toggle_layer("approaches"))
        key(["slash"], self.open_search)
        key(["Home", "0"], self.go_live)

        iren.add_observer("KeyPressEvent", self._on_search_key)
        iren.add_observer("MouseMoveEvent", self._on_mouse_move)
        iren.add_observer("LeaveEvent", self._on_leave)
        iren.add_observer("LeftButtonPressEvent", self._on_left_press)
        iren.add_observer("LeftButtonReleaseEvent", self._on_left_release)
        iren.add_observer("MouseWheelForwardEvent", lambda *_: self._on_wheel(0.85))
        iren.add_observer("MouseWheelBackwardEvent", lambda *_: self._on_wheel(1 / 0.85))
        iren.add_observer("ExitEvent", lambda *_: self.stop())

    def _event_position(self):
        # Read from the interactor itself: pyvista routes some mouse observers through
        # the interaction style, which has no event position of its own.
        return self.pl.iren.interactor.GetEventPosition()

    def _on_mouse_move(self, *_):
        self.mouse = self._event_position()

    def _on_leave(self, *_):
        self.mouse = None

    def _on_left_press(self, *_):
        self.press_pos = self._event_position()

    def _on_left_release(self, *_):
        if self.press_pos is None:
            return
        x, y = self._event_position()
        px, py = self.press_pos
        self.press_pos = None
        # Only a click if the mouse barely moved; otherwise it was a rotate-drag
        if abs(x - px) > 4 or abs(y - py) > 4:
            return
        self.click(x, y)

    def _on_wheel(self, factor):
        if self.selected is not None:
            self.follow_zoom = min(max(self.follow_zoom * factor, 0.25), 8.0)

    def _on_tab(self):
        shift = self.pl.iren.interactor.GetShiftKey()
        self.select_next(-1 if shift else 1)

    def _on_f(self):
        if self.pl.iren.interactor.GetControlKey():
            self.open_search()
        else:
            self.toggle_layer("footprint")

    # ---- search --------------------------------------------------------------------
    def open_search(self):
        if self.search_active:
            return
        self.search_active = True
        self.search_query = ""
        self._run_search()

    def close_search(self):
        self.search_active = False
        self.search_query = ""
        self._run_search()

    def _on_search_key(self, *_):
        if not self.search_active:
            return
        iren = self.pl.iren.interactor
        sym, char = iren.GetKeySym(), iren.GetKeyCode()

        if sym == "Escape":
            self.close_search()
        elif sym in ("Return", "KP_Enter"):
            self.choose_search_result(self.search_sel)
        elif sym == "BackSpace":
            self.search_query = self.search_query[:-1]
            self._run_search()
        elif sym in ("Up", "Down"):
            if self.search_results:
                step = -1 if sym == "Up" else 1
                self.search_sel = (self.search_sel + step) % len(self.search_results)
                self._refresh_search()
        elif sym == "slash" and not self.search_query:
            pass  # the key that opened the search
        elif char and len(char) == 1 and char.isprintable() and len(self.search_query) < 40:
            self.search_query += char
            self._run_search()

    def _run_search(self):
        query = self.search_query.strip().lower()
        results = []
        if query:
            matches = {i for i, name in enumerate(self.names_lower) if query in name}
            if query.isdigit():
                matches.update(i for i, number in enumerate(self.norad) if number.startswith(query))
            self.search_matches = len(matches)
            # Exact NORAD ID, then names starting with the query, then the most important types
            results = sorted(matches, key=lambda i: (self.norad[i] != query,
                                                     not self.names_lower[i].startswith(query),
                                                     self.priority[i], len(self.names[i])))
            results = results[:MAX_SEARCH_RESULTS]
        self.search_results = results
        self.search_sel = 0
        self._refresh_search()

    def _refresh_search(self):
        panel = self.search_panel
        active = self.search_active
        panel.set_row_visible("search:open", not active)
        panel.set_row_visible(self.search_input, active)
        panel.set_row_visible(self.search_hint, active)

        for k, value in enumerate(self.search_results_rows):
            key = f"result:{k}"
            if active and k < len(self.search_results):
                i = self.search_results[k]
                row = panel.field(key)
                row["label"].SetInput(truncate(self.names[i], 30))
                value.SetInput(f"{self.norad[i]}  ·  {truncate(CATEGORIES[self.categories[i]][0], 18)}")
                row["glyph"].set_color(CATEGORIES[self.categories[i]][1])
                panel.set_row_visible(key, True)
            else:
                panel.set_row_visible(key, False)

        if active:
            if not self.search_query:
                hint = "Type a name (e.g. ISS, Hubble, GOES) or a NORAD ID"
            elif not self.search_results:
                hint = "No matches  ·  Esc to close"
            else:
                more = f"{self.search_matches:,} matches  ·  " if self.search_matches > len(self.search_results) else ""
                hint = f"{more}Up/Down choose  ·  Enter track  ·  Esc close"
            self.search_hint.SetInput(hint)
            panel.set_hover(f"result:{self.search_sel}" if self.search_results else None)

    def choose_search_result(self, k):
        if not (0 <= k < len(self.search_results)):
            return
        index = self.search_results[k]
        self.close_search()
        category = self.categories[index]
        if not self.category_visible[category]:
            self.toggle_category(category)
        if not self.valid[index]:
            self.show_toast(f"{self.names[index]}: no valid position (re-entered or bad elements)", WARNING)
            return
        self.select(index)
        self.show_toast(f"Tracking {self.names[index]}")

    # ---- actions -------------------------------------------------------------------
    def click(self, x, y):
        if self.search_active and not self.search_panel.contains(x, y):
            self.close_search()

        for panel in self.clickable_panels:
            if panel.contains(x, y):
                key = panel.hit(x, y)
                if key:
                    self.run_ui_action(key)
                return
        if any(panel.contains(x, y) for panel in self.ui_panels):
            return

        index, _ = self.satellite_at(x, y)
        if index is not None:
            self.select(index)

    def run_ui_action(self, key):
        kind, _, arg = key.partition(":")
        if kind == "pause":
            self.toggle_pause()
        elif kind == "speed":
            self.live = False
            self.speed_index = int(arg)
            self._refresh_controls()
            self.show_toast(f"Simulation speed  {SPEEDS[self.speed_index]:,}×")
        elif kind == "category":
            self.toggle_category(arg)
        elif kind == "weather":
            self.set_weather(arg)
        elif kind == "layer":
            self.toggle_layer(arg)
        elif kind == "search":
            self.open_search()
        elif kind == "result":
            self.choose_search_result(int(arg))
        elif kind == "approach":
            self.watch_approach(int(arg))
        elif kind == "live":
            self.go_live()
        elif kind == "approach-filter":
            self.hide_starlink_pairs = not self.hide_starlink_pairs
            self._refresh_controls()
            self.last_text_update = 0.0
        elif kind == "action":
            {"search": self.open_search, "next": lambda: self.select_next(1), "home": self.release,
             "pause": self.toggle_pause, "live": self.go_live, "labels": self.toggle_labels}[arg]()

    def select(self, index, keep_watch=False):
        if not keep_watch:
            self.watch = None
        self.selected = index
        self.follow_zoom = 1.0
        self._start_camera_transition()
        self.select_actor.SetVisibility(True)
        self.nadir_actor.SetVisibility(True)
        self.last_text_update = 0.0
        self.last_label_refresh = 0.0

    def select_next(self, step):
        """Cycle through the selected satellite's category (or space stations when nothing is selected)."""
        if self.selected is not None:
            category = self.categories[self.selected]
        else:
            category = next((k for k in CATEGORIES if self.category_visible[k] and len(self.category_members[k])),
                            None)
        if category is None:
            return
        members = self.category_members[category]
        members = members[self.valid[members]]
        if len(members) == 0:
            return
        if self.selected is None or self.selected not in members:
            index = members[0] if step > 0 else members[-1]
        else:
            position = int(np.flatnonzero(members == self.selected)[0])
            index = members[(position + step) % len(members)]
        self.select(int(index))

    def release(self):
        self.selected = None
        self.watch = None
        self._start_camera_transition()
        self.select_actor.SetVisibility(False)
        self.nadir_actor.SetVisibility(False)
        self.last_label_refresh = 0.0

    def toggle_pause(self):
        self.live = False
        self.paused = not self.paused
        self._refresh_controls()
        self.last_text_update = 0.0
        self.show_toast("Paused" if self.paused else "Resumed")

    def change_speed(self, step):
        self.live = False
        self.speed_index = min(max(self.speed_index + step, 0), len(SPEEDS) - 1)
        self._refresh_controls()
        self.last_text_update = 0.0
        self.show_toast(f"Simulation speed  {SPEEDS[self.speed_index]:,}×")

    def toggle_labels(self):
        self.labels_visible = not self.labels_visible
        self.last_label_refresh = 0.0
        self._refresh_controls()
        self.show_toast("Labels on" if self.labels_visible else "Labels off")

    def toggle_layer(self, key):
        self.layers[key] = not self.layers[key]
        self.last_night_update = 0.0
        self.ground_track_key = None
        self._refresh_controls()
        names = {"night": "Day / night shading", "footprint": "Coverage footprint",
                 "groundtrack": "Ground track", "approaches": "Close approaches"}
        self.show_toast(f"{names[key]} {'on' if self.layers[key] else 'off'}")

    def toggle_category(self, key):
        self.category_visible[key] = not self.category_visible[key]
        if self.selected is not None and not self.category_visible[self.categories[self.selected]]:
            self.release()
        self._refresh_legend()
        self.last_label_refresh = 0.0
        label = CATEGORIES[key][0]
        self.show_toast(f"{label} shown" if self.category_visible[key] else f"{label} hidden")

    def set_weather(self, mode):
        self.weather_mode = mode
        layers = weather.MODES[mode][1]
        self._refresh_controls()
        message = f"Weather overlay: {weather.MODES[mode][0]}"
        if any(key not in self.weather_loaded for key in layers):
            message += "  (downloading...)"
        self.show_toast(message)
        self.last_text_update = 0.0

    def cycle_weather(self):
        modes = list(weather.MODES)
        self.set_weather(modes[(modes.index(self.weather_mode) + 1) % len(modes)])

    def watch_approach(self, row):
        """Jump the clock to just before a close approach, slow down and track the first object."""
        if row >= len(self.approach_rows):
            return
        k = self.approach_rows[row]
        i, j = int(self.events["i"][k]), int(self.events["j"][k])
        tca, miss = float(self.events["tca"][k]), float(self.events["miss"][k])
        for index in (i, j):
            if not self.category_visible[self.categories[index]]:
                self.toggle_category(self.categories[index])
        # Arrive 30 s early at real-time speed: at ~10 km/s relative speed the encounter
        # itself only lasts a few seconds
        self.sim_tt = tca - 30 / 86400.0
        self.live = False
        self.speed_index = 0
        self.paused = False
        self._refresh_controls()
        self.watch = {"i": i, "j": j, "tca": tca, "miss": miss}
        self.select(i, keep_watch=True)
        clock = self.ts.tt_jd(tca).utc_strftime("%H:%M:%S")
        self.show_toast(f"Watching {self.names[i]} × {self.names[j]}  ·  {miss:.2f} km at {clock} UTC", WARNING)

    def go_live(self):
        """Back to the real current time at 1x, e.g. after watching an approach hours ahead."""
        self.live = True
        self.paused = False
        self.speed_index = 0
        self.watch = None
        self._refresh_controls()
        self.last_text_update = 0.0
        self.show_toast("Live  ·  real time")

    def stop(self):
        self.running = False

    # ---- simulation ----------------------------------------------------------------
    def sim_time(self):
        return self.ts.tt_jd(self.sim_tt)

    def _propagate(self, t):
        """All satellites at once with vectorised SGP4. Positions in km, TEME frame."""
        jd, fraction = sgp4_time(t)
        errors, r, v = self.sat_array.sgp4(np.array([jd]), np.array([fraction]))
        positions = r[:, 0, :]
        valid = (errors[:, 0] == 0) & np.isfinite(positions).all(axis=1)
        positions[~valid] = 0.0  # decayed / failed: parked at the centre, hidden inside the Earth
        return positions, v[:, 0, :], valid

    def _update_positions(self, t):
        """Positions (scene units) and velocities (km/s) of every satellite at t.

        Full SGP4 for ~20k objects costs ~10 ms, so it only runs once the clock has moved
        MAX_EXTRAPOLATION_S since the last run. In between, orbits are extrapolated with
        velocity and two-body gravity (under 0.5 km of error after 60 s). Objects whose
        numbers are on screen (tracked satellite, watched pair) are always exact.
        """
        dt = None if self.base_tt is None else (t.tt - self.base_tt) * 86400.0
        if dt is None or abs(dt) > MAX_EXTRAPOLATION_S:
            self.base_r, self.base_v, self.valid = self._propagate(t)
            self.base_tt, dt = t.tt, 0.0

        r, v = self.base_r, self.base_v.copy()
        if dt:
            # Invalid objects are parked at the origin; the floor keeps them there
            r3 = np.maximum(np.einsum("ij,ij->i", r, r), 1.0) ** 1.5
            accel = -MU_KM3_S2 * r / r3[:, None]
            r = r + v * dt + 0.5 * accel * dt * dt
            v += accel * dt
        positions = r * self.scale

        exact = {self.selected} | ({self.watch["i"], self.watch["j"]} if self.watch else set())
        jd, fraction = sgp4_time(t)
        for index in exact - {None}:
            error, r_exact, v_exact = self.satellites[index].model.sgp4(jd, fraction)
            if error == 0:
                positions[index] = np.array(r_exact) * self.scale
                v[index] = v_exact
        return positions, v

    def _track(self, index):
        if index not in self.tracks:
            mesh = rgba_line(np.zeros((TRACK_SAMPLES + 1, 3)), CATEGORIES[self.categories[index]][1])
            actor = add_colored(self.pl, mesh, "rgba", line_width=1.5)
            self.tracks[index] = Track(mesh, actor)
        return self.tracks[index]

    def _update_tracks(self, t):
        wanted = set()
        if self.category_visible["station"]:
            wanted.update(int(i) for i in self.category_members["station"] if self.valid[i])
        for index in (self.selected, self.hover_index):
            if index is not None:
                wanted.add(index)
        if self.watch:
            wanted.add(self.watch["j"])

        for index, track in self.tracks.items():
            if index not in wanted:
                track.actor.SetVisibility(False)

        dim = self.selected is not None
        for index in wanted:
            track = self._track(index)
            track.actor.SetVisibility(True)
            period = self.periods_days[index]

            # Resample the orbit once per revolution (it slowly precesses)
            if track.epoch is None or abs(t.tt - track.epoch) > period:
                track.epoch = t.tt
                times = self.ts.tt_jd(track.epoch + np.linspace(0.0, period, TRACK_SAMPLES + 1))
                errors, r, _ = self.satellites[index].model.sgp4_array(*sgp4_time(times))
                if np.any(errors):
                    track.actor.SetVisibility(False)
                    continue
                track.mesh.points = r * self.scale

            # u in [-1, 1): negative behind the satellite, positive ahead
            phase = ((t.tt - track.epoch) / period) % 1.0
            k = np.arange(TRACK_SAMPLES + 1) / TRACK_SAMPLES
            u = ((k - phase + 0.5) % 1.0) * 2.0 - 1.0
            alpha = np.where(u < 0, 0.9 * (1.0 + u) ** 1.5, 0.9 - 0.75 * u)

            emphasised = index == self.selected or (self.watch and index == self.watch["j"])
            if emphasised or (index == self.hover_index and not dim):
                alpha = np.clip(alpha * 1.15, 0, 1)
            elif dim:
                alpha *= 0.18
            track.rgba[:, 3] = (alpha * 255).astype(np.uint8)
            track.vtk_rgba.Modified()
            track.actor.GetProperty().SetLineWidth(3.0 if emphasised else 1.5)

    def _update_sunlight(self, now):
        """Night-side shading and which satellites are in Earth's shadow."""
        # Cylindrical shadow model: behind the Earth (away from the Sun) and within one Earth radius of the axis
        along = self.positions @ self.sun
        across2 = np.einsum("ij,ij->i", self.positions, self.positions) - along ** 2
        self.sunlit = ~((along < 0) & (across2 < self.R ** 2))

        night_on = self.layers["night"]
        self.night_actor.SetVisibility(night_on)
        for key, (mesh, rgb, vtk_rgb, base) in self.category_meshes.items():
            if night_on:
                factor = np.where(self.sunlit[self.category_members[key]], 1.0, SHADOW_DIM)
                rgb[:] = (base[None, :] * factor[:, None]).astype(np.uint8)
            else:
                rgb[:] = base.astype(np.uint8)
            vtk_rgb.Modified()

        if night_on and now - self.last_night_update > 0.2:
            self.last_night_update = now
            cos_zenith = self.night_normals @ self.sun
            # Fully lit above ~5 deg solar elevation, full night ~10 deg below the horizon
            darkness = smoothstep((0.08 - cos_zenith) / 0.26)
            self.night_rgba[:, 3] = (darkness * NIGHT_MAX_OPACITY * 255).astype(np.uint8)
            self.night_vtk_rgba.Modified()

    def _update_footprint(self):
        show = bool(self.layers["footprint"] and self.selected is not None and self.valid[self.selected])
        self.coverage_km = None
        if show:
            p = self.positions[self.selected]
            r = np.linalg.norm(p)
            elevation = np.radians(FOOTPRINT_MIN_ELEVATION_DEG)
            ratio = self.R * np.cos(elevation) / r
            show = bool(ratio < 1.0)
        self.footprint_actor.SetVisibility(show)
        self.horizon_actor.SetVisibility(show)
        if not show:
            return

        # Earth-central angle of the coverage edge: where the satellite sits at the minimum
        # elevation (fill), and at 0 degrees (horizon ring)
        cap = np.arccos(ratio) - elevation
        horizon = np.arccos(self.R / r)
        self.coverage_km = cap * EARTH_RADIUS_KM

        up = p / r
        e1 = np.cross(up, [0.0, 0.0, 1.0])
        e1 = normalize(e1 if np.linalg.norm(e1) > 1e-6 else np.cross(up, [1.0, 0.0, 0.0]))
        e2 = np.cross(up, e1)
        theta = np.linspace(0, 2 * np.pi, FOOTPRINT_SEGMENTS, endpoint=False)
        around = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2

        angles = np.linspace(0, cap, FOOTPRINT_RINGS + 1)
        dirs = (np.cos(angles)[:, None, None] * up + np.sin(angles)[:, None, None] * around[None]).reshape(-1, 3)
        self.footprint_mesh.points = dirs * self.R * FOOTPRINT_RADIUS

        ring = np.cos(horizon) * up + np.sin(horizon) * around
        self.horizon_mesh.points = np.vstack([ring, ring[:1]]) * self.R * GROUND_TRACK_RADIUS

    def _update_ground_track(self, t):
        index = self.selected
        show = bool(self.layers["groundtrack"] and index is not None and self.valid[index])
        if not show:
            self.ground_actor.SetVisibility(False)
            self.ground_track_key = None
            return

        period = self.periods_days[index]
        if self.ground_track_key is None or self.ground_track_key[0] != index \
                or abs(t.tt - self.ground_track_key[1]) > period / 90:
            span = GROUND_TRACK_ORBITS * period
            times = self.ts.tt_jd(t.tt + np.linspace(0.0, span, GROUND_TRACK_SAMPLES + 1))
            errors, r, _ = self.satellites[index].model.sgp4_array(*sgp4_time(times))
            if np.any(errors):
                self.ground_actor.SetVisibility(False)
                return
            # Into Earth-fixed coordinates: undo the Earth's rotation at each sample time
            theta = np.radians(times.gmst * 15.0)
            x = r[:, 0] * np.cos(theta) + r[:, 1] * np.sin(theta)
            y = -r[:, 0] * np.sin(theta) + r[:, 1] * np.cos(theta)
            fixed = np.column_stack([x, y, r[:, 2]])
            self.ground_mesh.points = fixed / np.linalg.norm(fixed, axis=1, keepdims=True) * self.R * GROUND_TRACK_RADIUS

            rgba = self.ground_mesh["rgba"]
            rgba[:, 3] = (np.linspace(1.0, 0.15, GROUND_TRACK_SAMPLES + 1) * 230).astype(np.uint8)
            self.ground_mesh.GetPointData().GetArray("rgba").Modified()
            self.ground_track_key = (index, t.tt)
        self.ground_actor.SetVisibility(True)

    # ---- close approaches ------------------------------------------------------------
    def _update_screening(self):
        """Keep a screening window ahead of the simulation clock."""
        screener = self.screener
        if screener.poll():
            self._ingest_events(screener.events)
        if screener.running:
            return
        started = self.screen_started_tt
        stale = started is None or self.sim_tt > started + (conjunctions.WINDOW_HOURS - 1) / 24 \
            or self.sim_tt < started - 1 / 1440
        if stale:
            self.screen_started_tt = self.sim_tt
            screener.start(self.sim_tt)

    def _ingest_events(self, events):
        starlink = self.categories == "starlink"
        i = np.array([e.i for e in events], dtype=int)
        j = np.array([e.j for e in events], dtype=int)
        self.events = {
            "i": i, "j": j,
            "tca": np.array([e.tca_tt for e in events]),
            "miss": np.array([e.miss_km for e in events]),
            "speed": np.array([e.speed_km_s for e in events]),
            "starlink_pair": starlink[i] & starlink[j] if len(events) else np.array([], dtype=bool),
        }
        self.last_text_update = 0.0

    def _upcoming_events(self):
        """Indices of events not yet passed (respecting the Starlink filter)."""
        if self.events is None:
            return np.array([], dtype=int)
        keep = self.events["tca"] > self.sim_tt - 60 / 86400.0
        if self.hide_starlink_pairs:
            keep &= ~self.events["starlink_pair"]
        return np.flatnonzero(keep)

    def _update_alerts(self):
        """3D markers on objects in a close approach right now, plus the watched pair's line."""
        show = bool(self.layers["approaches"] and self.events is not None and len(self.events["tca"]))
        points = []
        if show:
            near_now = np.abs(self.events["tca"] - self.sim_tt) * 86400 < ALERT_WINDOW_S
            near_now &= self.events["miss"] < ALERT_MISS_KM
            if self.hide_starlink_pairs:
                near_now &= ~self.events["starlink_pair"]
            for k in np.flatnonzero(near_now)[:50]:
                for index in (self.events["i"][k], self.events["j"][k]):
                    if self.valid[index] and self.category_visible[self.categories[index]]:
                        points.append(self.positions[index])
        if points:
            self.alert_mesh.points = np.array(points)
        self.alert_actor.SetVisibility(bool(points))

        watch = self.watch
        if watch and self.valid[watch["i"]] and self.valid[watch["j"]]:
            self.pair_mesh.points = np.array([self.positions[watch["i"]], self.positions[watch["j"]]])
            self.pair_actor.SetVisibility(True)
        else:
            self.pair_actor.SetVisibility(False)

        # Warn ahead of the tracked satellite's own approaches
        if self.selected is not None and self.events is not None and self.layers["approaches"]:
            ev = self.events
            mine = (ev["i"] == self.selected) | (ev["j"] == self.selected)
            soon = mine & (ev["tca"] > self.sim_tt) & ((ev["tca"] - self.sim_tt) * 86400 < TRACKED_ALERT_LEAD_S)
            for k in np.flatnonzero(soon):
                key = (int(ev["i"][k]), int(ev["j"][k]), round(float(ev["tca"][k]) * 1440))
                if key in self.alerted:
                    continue
                self.alerted.add(key)
                other = int(ev["j"][k] if ev["i"][k] == self.selected else ev["i"][k])
                eta = format_eta((ev["tca"][k] - self.sim_tt) * 86400)
                self.show_toast(f"Close approach {eta}: {self.names[other]} passes {ev['miss'][k]:.2f} km away",
                                WARNING)

    def _next_approach(self, index):
        if self.events is None:
            return None
        ev = self.events
        mine = ((ev["i"] == index) | (ev["j"] == index)) & (ev["tca"] > self.sim_tt - 60 / 86400.0)
        ks = np.flatnonzero(mine)
        if len(ks) == 0:
            return None
        k = ks[np.argmin(ev["tca"][ks])]
        other = int(ev["j"][k] if ev["i"][k] == index else ev["i"][k])
        return other, float(ev["miss"][k]), float(ev["tca"][k])

    # ---- screen-space picking --------------------------------------------------------
    def _project(self, points):
        """World points -> display pixels, plus a mask of points that are actually visible."""
        w, h = self.ren.GetSize()
        camera = self.ren.GetActiveCamera()
        m = camera.GetCompositeProjectionTransformMatrix(w / max(h, 1), -1, 1)
        matrix = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])

        clip = points @ matrix[:, :3].T + matrix[:, 3]
        wcoord = clip[:, 3]
        safe_w = np.where(np.abs(wcoord) < 1e-12, 1e-12, wcoord)
        screen = np.c_[(clip[:, 0] / safe_w + 1) * 0.5 * w, (clip[:, 1] / safe_w + 1) * 0.5 * h]

        # Hidden behind the Earth? (segment camera -> point hits the sphere)
        cam = np.array(camera.GetPosition())
        d = points - cam
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * d @ cam
        c = cam @ cam - (self.R * 0.995) ** 2
        disc = b * b - 4 * a * c
        t_hit = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
        occluded = (disc > 0) & (t_hit > 0) & (t_hit < 1)

        on_screen = (screen[:, 0] >= 0) & (screen[:, 0] <= w) & (screen[:, 1] >= 0) & (screen[:, 1] <= h)
        return screen, (wcoord > 0) & ~occluded & on_screen

    def satellite_at(self, x, y):
        """Nearest visible satellite within HOVER_RADIUS_PX of (x, y), and everything co-located with it."""
        screen, visible = self.screen_positions
        dist = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
        dist[~visible] = np.inf
        dist[dist > HOVER_RADIUS_PX] = np.inf
        # Among everything under the cursor, favour important types (a station over a Starlink)
        index = int(np.argmin(dist + 2.0 * self.priority))
        if not np.isfinite(dist[index]):
            return None, []
        near = np.hypot(screen[:, 0] - screen[index, 0], screen[:, 1] - screen[index, 1]) < 3
        return index, np.flatnonzero(near & visible).tolist()

    # ---- camera ----------------------------------------------------------------------
    def _follow_target(self):
        """Chase camera: behind, above and slightly to the side of the satellite."""
        position = self.positions[self.selected]
        radial = normalize(position)
        forward = normalize(self.velocities[self.selected])
        side = np.cross(forward, radial)

        offset = self.R * self.follow_zoom * (-0.5 * forward + 0.22 * radial + 0.16 * side)
        return position + offset, position, radial

    def _home_target(self):
        return tuple(np.array(v, dtype=float) for v in self.home_camera)

    def _start_camera_transition(self):
        camera = self.ren.GetActiveCamera()
        self.camera_anim = {
            "start": time.monotonic(),
            "from": (np.array(camera.GetPosition()), np.array(camera.GetFocalPoint()),
                     np.array(camera.GetViewUp())),
        }

    def _update_camera(self):
        if self.selected is not None and not self.valid[self.selected]:
            self.release()
        if self.selected is None and self.camera_anim is None:
            return  # free orbit with the mouse

        target = self._follow_target() if self.selected is not None else self._home_target()

        if self.camera_anim is not None:
            s = float(smoothstep((time.monotonic() - self.camera_anim["start"]) / CAMERA_TRANSITION_S))
            pos0, focal0, up0 = self.camera_anim["from"]
            pos1, focal1, up1 = target

            # Swing around the Earth rather than cutting through it
            r0, r1 = np.linalg.norm(pos0), np.linalg.norm(pos1)
            direction = normalize((1 - s) * pos0 / r0 + s * pos1 / r1)
            position = direction * ((1 - s) * r0 + s * r1)
            focal = (1 - s) * focal0 + s * focal1
            up = normalize((1 - s) * up0 + s * up1)
            if s >= 1.0:
                self.camera_anim = None
        else:
            position, focal, up = target

        camera = self.ren.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(*up)
        # Fixed clipping range: recomputing it from scene bounds every frame is expensive
        camera.SetClippingRange(self.R * 0.01, self.R * 80)

    # ---- per-frame update ------------------------------------------------------------
    def frame(self):
        now = time.monotonic()
        dt = min(now - self.last_wall, 0.25)  # don't jump after the window was blocked
        self.last_wall = now
        if self.live:
            self.sim_tt = self.ts.now().tt
        elif not self.paused:
            self.sim_tt += dt * SPEEDS[self.speed_index] / 86400.0

        t = self.sim_time()
        self.positions, self.velocities = self._update_positions(t)
        for key, (mesh, _, _, _) in self.category_meshes.items():
            mesh.points = self.positions[self.category_members[key]]

        # Spin the Earth (and everything draped on it) to the current sidereal angle
        gmst_degrees = t.gmst * 15.0
        for actor in self.earth_actors:
            actor.SetOrientation(0.0, 0.0, gmst_degrees)

        self.sun = sun_direction(t)
        self._update_sunlight(now)
        self._update_weather()
        self._update_screening()
        self._update_camera()

        screen, visible = self._project(self.positions)
        shown = np.zeros(self.n, dtype=bool)
        for key, members in self.category_members.items():
            if self.category_visible[key]:
                shown[members] = True
        self.screen_positions = (screen, visible & self.valid & shown)

        self._update_hover()
        self._update_tracks(t)
        self._update_footprint()
        self._update_ground_track(t)
        self._update_alerts()
        self._update_markers(now)
        self._update_labels(now)

        if now - self.last_text_update > 0.1:
            self.last_text_update = now
            self._update_text(t, now)

        if self.toast.visible and now > self.toast_until:
            self.toast.set_visible(False)

        self._layout()

    def _update_weather(self):
        for key, rgba in self.weather.poll():
            texture = pv.Texture(rgba)
            texture.interpolate = True
            self.weather_actors[key].SetTexture(texture)
            self.weather_loaded.add(key)
            self.last_text_update = 0.0

        layers = weather.MODES[self.weather_mode][1]
        for key in layers:
            self.weather.request(key)  # throttled: only downloads when missing or stale
        for key, actor in self.weather_actors.items():
            actor.SetVisibility(key in layers and key in self.weather_loaded)

    def _update_markers(self, now):
        if self.selected is not None:
            p = self.positions[self.selected]
            self.select_mesh.points = p.reshape(1, 3)
            self.nadir_mesh.points = np.array([p, normalize(p) * self.R])
            self.select_actor.GetProperty().SetPointSize(24 + 6 * math.sin(now * 4.0))

        if self.hover_index is not None and self.hover_index != self.selected:
            self.hover_mesh.points = self.positions[self.hover_index].reshape(1, 3)
            self.hover_actor.SetVisibility(True)
        else:
            self.hover_actor.SetVisibility(False)

    def _update_labels(self, now):
        """Label the most important visible satellites without letting labels overlap.

        Which satellites get a label is re-decided a few times a second; the labels
        themselves follow their satellites every frame.
        """
        screen, visible = self.screen_positions

        if now - self.last_label_refresh > LABEL_REFRESH_S:
            self.last_label_refresh = now
            if self.labels_visible:
                candidates = np.flatnonzero(visible)
                if len(candidates):
                    # Most important type first, then closest to the camera
                    cam = np.array(self.ren.GetActiveCamera().GetPosition())
                    depth = np.linalg.norm(self.positions[candidates] - cam, axis=1)
                    candidates = candidates[np.lexsort((depth, self.priority[candidates]))][:500]
            else:
                candidates = np.array([], dtype=int)
            # Always label the tracked object and the other half of a watched approach
            for index in ([self.watch["j"]] if self.watch else []) + [self.selected]:
                if index is not None and visible[index]:
                    candidates = np.r_[index, candidates[candidates != index]]

            w, h = self.ren.GetSize()
            placed = []
            # Start with the panels as occupied space so labels never run under the UI
            boxes = [(x0, y0, x0 + bw, y0 + bh) for x0, y0, bw, bh in
                     (p.box for p in self.ui_panels + [self.toast] if p.visible)]
            for i in candidates:
                x, y = screen[i]
                box = (x + 10, y + 4, x + 14 + 6.3 * len(self.names[i]), y + 20)
                if box[2] > w or box[3] > h:
                    continue
                if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                    continue
                placed.append(int(i))
                boxes.append(box)
                if len(placed) == MAX_LABELS:
                    break
            self.label_slots = placed

        for slot, label in enumerate(self.labels):
            if slot >= len(self.label_slots) or not visible[self.label_slots[slot]]:
                label.SetVisibility(False)
                continue
            i = self.label_slots[slot]
            if any(p.contains(*screen[i]) for p in self.ui_panels):
                label.SetVisibility(False)
                continue
            label.SetInput(self.names[i])
            watched = self.watch is not None and i == self.watch["j"]
            set_text_color(label, ACCENT if i == self.selected else WARNING if watched else TEXT)
            label.SetDisplayPosition(round(screen[i][0] + 12), round(screen[i][1] + 6))
            label.SetVisibility(True)

    def _update_hover(self):
        index, group, ui_key = None, [], None
        if self.mouse is not None and self.press_pos is None:
            x, y = self.mouse
            panel = next((p for p in self.ui_panels if p.contains(x, y)), None)
            if panel is not None:
                ui_key = panel.hit(x, y) if panel in self.clickable_panels else None
            else:
                index, group = self.satellite_at(x, y)

        if ui_key != self.ui_hover:
            self.ui_hover = ui_key
            for panel in self.clickable_panels:
                panel.set_hover(ui_key if ui_key and panel.hit(*self.mouse) == ui_key else None)
            if ui_key and ui_key.startswith("result:"):
                self.search_sel = int(ui_key.partition(":")[2])
            if self.search_active:
                self._refresh_search()

        if index != self.hover_index:
            self.last_text_update = 0.0
        self.hover_index, self.hover_group = index, group
        self.tooltip.set_visible(index is not None)

        cursor = VTK_CURSOR_HAND if (ui_key or index is not None) else VTK_CURSOR_DEFAULT
        if cursor != self.cursor and self.pl.render_window is not None:
            self.cursor = cursor
            self.pl.render_window.SetCurrentCursor(cursor)

    def _light_text(self, index):
        return "Sunlit" if self.sunlit[index] else "In Earth's shadow"

    def _update_text(self, t, now):
        speed = SPEEDS[self.speed_index]
        if self.paused:
            self.clock_status.SetInput("SIMULATION  ·  PAUSED")
            set_text_color(self.clock_status, WARNING)
        elif self.live:
            self.clock_status.SetInput("LIVE  ·  REAL TIME")
            set_text_color(self.clock_status, ACCENT)
        else:
            status = f"SIMULATION  ·  {speed:,}× REAL TIME"
            offset = (self.sim_tt - self.ts.now().tt) * 86400.0
            if abs(offset) >= 60:
                status += f"  ·  {format_duration(offset).upper()} {'AHEAD' if offset > 0 else 'BEHIND'}"
            self.clock_status.SetInput(status)
            set_text_color(self.clock_status, MUTED)
        self.clock_text.SetInput(t.utc_strftime("%Y-%m-%d  %H:%M:%S") + " UTC")

        if self.catalog.updated:
            source = f"{self.catalog.source} data fetched {format_age(time.time() - self.catalog.updated)}"
        else:
            source = f"{self.catalog.source} data"
        self.title_subtitle.SetInput(f"{self.n:,} objects  ·  {source}")

        if self.search_active:
            caret = "_" if int(now * 2) % 2 == 0 else " "
            self.search_input.SetInput(f"> {self.search_query}{caret}")

        layers = weather.MODES[self.weather_mode][1]
        for key, actor in self.weather_status.items():
            if key in layers:
                actor.SetInput(f"{weather.LAYERS[key]['label']}:  {self.weather.status[key]}")
            else:
                actor.SetInput(" ")
        if not layers:
            self.weather_status["clouds"].SetInput("Live clouds & rain from weather satellites")

        self._update_approach_panel()

        tracking = self.selected is not None
        self.track_panel.set_visible(tracking)
        self.idle_panel.set_visible(not tracking)
        if tracking:
            index = self.selected
            category = self.categories[index]
            info = {**self.static(index), **live_info(self.satellites[index], t)}
            info["light"] = self._light_text(index)
            info["coverage"] = f"{self.coverage_km:,.0f} km" if self.coverage_km else "—"
            self.track_name.SetInput(self.names[index])
            self.track_category.SetInput(CATEGORIES[category][0])
            set_text_color(self.track_category, CATEGORIES[category][1])
            for key in ("latitude", "longitude", "altitude", "velocity", "light", "coverage", "norad", "cospar",
                        "inclination", "period", "perigee", "apogee", "epoch"):
                self.track_fields[key].SetInput(info[key])
            set_text_color(self.track_fields["light"], TEXT if self.sunlit[index] else MUTED)

            next_approach = self._next_approach(index) if self.events is not None else None
            if self.watch and self.watch["i"] == index:
                other, miss, tca = self.watch["j"], self.watch["miss"], self.watch["tca"]
                next_approach = (other, miss, tca)
            self.track_panel.set_row_visible(self.track_fields["approach"], next_approach is not None)
            self.track_panel.set_row_visible(self.track_fields["approach_with"], next_approach is not None)
            if next_approach:
                other, miss, tca = next_approach
                self.track_fields["approach"].SetInput(f"{miss:.2f} km {format_eta((tca - self.sim_tt) * 86400)}")
                self.track_fields["approach_with"].SetInput(truncate(self.names[other], 26))
                set_text_color(self.track_fields["approach"], WARNING if miss < ALERT_MISS_KM else TEXT)
            watching = self.watch is not None and self.watch["i"] == index
            self.track_panel.set_row_visible(self.track_fields["separation"], watching)
            if watching:
                separation = np.linalg.norm(self.positions[index] - self.positions[self.watch["j"]]) / self.scale
                self.track_fields["separation"].SetInput(f"{separation:,.2f} km")
                set_text_color(self.track_fields["separation"], WARNING)

        if self.hover_index is not None:
            index = self.hover_index
            category = self.categories[index]
            info = {**self.static(index), **live_info(self.satellites[index], t)}
            info["light"] = self._light_text(index)
            self.tip_name.SetInput(self.names[index])
            self.tip_category.SetInput(CATEGORIES[category][0].upper())
            set_text_color(self.tip_category, CATEGORIES[category][1])
            for key, actor in self.tip_fields.items():
                actor.SetInput(info[key])
            others = len(self.hover_group) - 1
            hint = "Click to track" if index != self.selected else "Currently tracking"
            if others > 0:
                hint += f"   ·   +{others} more within a few pixels"
            self.tip_hint.SetInput(hint)

    def _update_approach_panel(self):
        screener = self.screener
        upcoming = self._upcoming_events()

        if screener.error:
            status = f"Screening failed: {screener.error}"
        elif self.events is None:
            status = f"Screening the next {conjunctions.WINDOW_HOURS:g} h of orbits...  {screener.progress:.0%}"
        else:
            under_alert = int((self.events["miss"][upcoming] < ALERT_MISS_KM).sum()) if len(upcoming) else 0
            status = f"{len(upcoming):,} passes ahead  ·  {under_alert:,} under {ALERT_MISS_KM:g} km  ·  closest first"
            if screener.running:
                status += f"  ·  updating {screener.progress:.0%}"
        self.approach_status.SetInput(status)

        # Show the closest upcoming encounters
        rows = []
        if len(upcoming):
            rows = upcoming[np.argsort(self.events["miss"][upcoming])][:MAX_APPROACH_ROWS].tolist()
        self.approach_rows = rows
        for k, value in enumerate(self.approach_values):
            key = f"approach:{k}"
            if k < len(rows):
                e = rows[k]
                i, j = self.events["i"][e], self.events["j"][e]
                miss = self.events["miss"][e]
                row = self.approach_panel.field(key)
                row["label"].SetInput(truncate(f"{self.names[i]} × {self.names[j]}", 40))
                value.SetInput(f"{miss:5.2f} km  {format_eta((self.events['tca'][e] - self.sim_tt) * 86400):>10}")
                row["glyph"].set_color("#f87171" if miss < ALERT_MISS_KM else WARNING)
                self.approach_panel.set_row_visible(key, True)
            else:
                self.approach_panel.set_row_visible(key, False)

    def _layout(self):
        w, h = self.ren.GetSize()
        margin, gap = 20, 10

        # Left column: title, search, close approaches; tracking card at the bottom
        self.title_panel.layout(margin, h - margin, 0, 1)
        self.search_panel.layout(margin, self.title_panel.box[1] - gap, 0, 1)
        self.track_panel.layout(margin, margin, 0, 0)
        self.idle_panel.layout(margin, margin, 0, 0)

        self.approach_panel.set_visible(self.layers["approaches"] and not self.search_active)
        if self.approach_panel.visible:
            self.approach_panel.layout(margin, self.search_panel.box[1] - gap, 0, 1)
            bottom = self.track_panel if self.track_panel.visible else self.idle_panel
            if self.approach_panel.box[1] < bottom.box[1] + bottom.box[3] + gap:
                self.approach_panel.set_visible(False)  # not enough room this frame

        # Centre and right column
        self.clock_panel.layout(w / 2, h - margin, 0.5, 1)
        self.legend_panel.layout(w - margin, h - margin, 1, 1)
        self.layers_panel.layout(w - margin, self.legend_panel.box[1] - gap, 1, 1)
        self.help_panel.layout(w - margin, margin, 1, 0)
        self.toast.layout(w / 2, margin + 40, 0.5, 0)
        if self.mouse is not None and self.tooltip.visible:
            self.tooltip.layout(self.mouse[0] + 18, self.mouse[1] - 18, 0, 1, clamp_to=(w, h))

    # ---- main loop -------------------------------------------------------------------
    def run(self):
        self.frame()
        self.pl.show(title="Satellite Tracker", interactive_update=True, auto_close=False)
        iren = self.pl.iren.interactor
        frame_time = 1.0 / TARGET_FPS

        while self.running:
            start = time.monotonic()
            self.pl.iren.process_events()
            if iren.GetDone() or self.pl.render_window is None or not self.running:
                break
            self.frame()
            self.pl.render()
            time.sleep(max(0.0, frame_time - (time.monotonic() - start)))

        self.screener.close()
        self.pl.close()


def parse_args():
    parser = argparse.ArgumentParser(description="3D satellite tracker")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, metavar="GROUP",
                        help="CelesTrak groups to load, e.g. stations visual active starlink gnss weather "
                             f"(default: {' '.join(DEFAULT_GROUPS)})")
    parser.add_argument("--offline", action="store_true",
                        help="don't download anything; use cached or bundled data only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not accelerated:
        print("Warning: sgp4 is running without its C extension, so the app will be very slow "
              "(about 2 FPS). Use Python 3.13; see the README.")
    timescale = load.timescale()
    catalog = load_catalog(timescale, args.groups, offline=args.offline)
    SatelliteTracker(catalog, timescale, offline=args.offline).run()
