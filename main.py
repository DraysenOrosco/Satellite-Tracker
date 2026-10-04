import os 
import numpy as np
import pyvista as pv 
from pyvista import examples
from skyfield.api import load
from skyfield.iokit import parse_tle_file


def random_sphere_points(n, radius):
    phi = np.random.uniform(0, 2 * np.pi, n)
    costheta = np.random.uniform(-1, 1, n)
    theta  = np.arccos(costheta)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.column_stack((x, y, z))

def satellite_color(name):
    upper_name = name.upper()
    
    if "ISS" in upper_name:
        return "red"
    elif "CSS" in upper_name or "TIANHE" in upper_name or "SHENZHOU" in upper_name or "TIANZHOU" in upper_name:
        return "cyan"
    elif "CREW" in upper_name or "CYGNUS" in upper_name or "PROGRESS" in upper_name:
        return "yellow"
    elif "DEB" in upper_name or "CAMERA" in upper_name or "MODULE" in upper_name:
        return "orange"
    else:
        return "white"


def make_polyline(points):
    """
    Convert Nx3 points into a connected PolyData line for PyVista.
    """
    n = len(points)
    poly = pv.PolyData()
    poly.points = points
    
    poly.lines = np.hstack(([n], np.arange(n)))
    return poly 

def compute_satellite_point(sat, t, scale):
    geocentric = sat.at(t)
    x_km, y_km, z_km = geocentric.position.km
    return np.array([x_km * scale, y_km * scale, z_km * scale])

def compute_orbit_points(sat, ts, scale, minutes_ahead=90, steps=120):
    times = ts.utc(
        ts.now().utc_datetime().year,
        ts.now().utc_datetime().month,
        ts.now().utc_datetime().day,
        ts.now().utc_datetime().hour,
        ts.now().utc_datetime().minute,
        ts.now().utc_datetime().second + np.linspace(0, minutes_ahead * 60, steps)
    )

    geocentric = sat.at(times)
    xyz_km = geocentric.position.km.T
    return xyz_km * scale


# Scene setup
dataset = examples.load_globe()
texture = examples.load_globe_texture()

pl = pv.Plotter()
pl.set_background("black")
pl.add_mesh(dataset, texture=texture)

#stars
dim_points = random_sphere_points(4000, 120)
dim_stars = pv.PolyData(dim_points)
pl.add_points(
    dim_stars,
    point_size=10,
    render_points_as_spheres=True,
    opacity=0.35
)

bright_points = random_sphere_points(400, 120)
bright_stars = pv.PolyData(bright_points)
pl.add_points(
    bright_stars,
    color="white",
    point_size=10,
    render_points_as_spheres=True,
    opacity=0.9
)

# Nebula shells
nebula1 = pv.Sphere(radius=110, theta_resolution=60, phi_resolution=60)
nebula2 = pv.Sphere(radius=130, theta_resolution=60, phi_resolution=60)

pl.add_mesh(nebula1, color="#3344aa", opacity=0.03, lighting=False)
pl.add_mesh(nebula2, color="#662266", opacity=0.02, lighting=False)


# Load local TLE file only
# Load local TLE file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tle_path = os.path.join(BASE_DIR, "stations.tle")

if not os.path.exists(tle_path):
    raise FileNotFoundError(
        f"Could not find TLE file:\n{tle_path}\n\n"
        "Download/create stations.tle and put it next to main.py."
    )

ts = load.timescale()

with open(tle_path, "rb") as f:
    satellites = list(parse_tle_file(f, ts))

print("Loaded", len(satellites), "satellites")


if not satellites:
    raise ValueError("No satellites were parsed from stations.tle")


# Scene scale
print("Earth mesh bounds", dataset.bounds)
print("First satellite:", satellites[0].name)

sat = satellites[0]
t = ts.now()
geocentric = sat.at(t)

x_km, y_km, z_km = geocentric.position.km
print("Satellite position (km):", x_km, y_km, z_km)

bounds = dataset.bounds
x_min, x_max, y_min, y_max, z_min, z_max = bounds

earth_radius_scene = max(
    abs(x_min), abs(x_max),
    abs(y_min), abs(y_max),
    abs(z_min), abs(z_max)
)

earth_radius_km = 6371.0
scale = earth_radius_scene / earth_radius_km

#create actors for satellites 
satellite_meshes = {}
satellite_line_meshes ={}
satellite_point_actors = {}
satellite_line_actors = {}

label_points = []
label_text = []

t0 = ts.now()

for sat in satellites:
    color = satellite_color(sat.name)
    
    point = compute_satellite_point(sat, t0, scale).reshape(1, 3)
    point_mesh = pv.PolyData(point)
    point_actor = pl.add_mesh(
        point_mesh,
        color=color,
        style="points",
        point_size=12,
        render_points_as_spheres=False
    )
    
    
#orbit line
    orbit_points = compute_orbit_points(sat, ts, scale, minutes_ahead=90, steps=120)
    orbit_mesh = make_polyline(orbit_points)
    line_actor = pl.add_mesh(
        orbit_mesh,
        color=color,
        line_width=2,
        opacity=0.7
    )

    satellite_meshes[sat.name] = point_mesh
    satellite_line_meshes[sat.name] = orbit_mesh
    satellite_point_actors[sat.name] = point_actor
    satellite_line_actors[sat.name] = line_actor

    label_points.append(point[0])
    label_text.append(sat.name)

    #on and off labels by default
    labels_visible = False
    label_actor = None

def build_labels():
    global label_actor
    if label_actor is not None:
        pl.remove_actor(label_actor)
        
    if len(label_points) > 0:
        label_actor = pl.add_point_labels(
            np.array(label_points),
            label_text,
            font_size=10,
            text_color="white",
            shape_opacity=0.15,
            always_visible=True
        )
        
        if not labels_visible:
            label_actor.SetVisibility(False)
            
build_labels()

positions_by_color = {}
labels_points = []
labels_text = []

for sat in satellites:
    geocentric = sat.at(t)
    x_km, y_km, z_km = geocentric.position.km
    
    point = [x_km * scale, y_km * scale, z_km * scale]
    color = satellite_color(sat.name)
    
    if color not in positions_by_color:
        positions_by_color[color] = []

    positions_by_color[color].append(point)

            
    labels_points.append(point)
    labels_text.append(sat.name)
        
# Plot grouped points by color
for color, points in positions_by_color.items():
    poly = pv.PolyData(np.array(points))
    pl.add_points(
        poly,
        color=color,
        point_size=12,
        render_points_as_spheres=True
    )

# Optional labels for all satellites
pl.add_point_labels(
    np.array(labels_points),
    labels_text,
    font_size=10,
    text_color="white",
    point_size=8,
    shape_opacity=0.15,
    always_visible=True
)

# -----------------------------
# Simple legend text
# -----------------------------
legend_lines = [
    "Satellite Colors:",
    "Red = ISS-related",
    "Cyan = CSS / China station-related",
    "Yellow = Crew / Cargo craft",
    "Orange = Debris / camera / module",
    "White = Other"
]

pl.add_text(
    "\n".join(legend_lines),
    position="upper_right",
    font_size=10,
    color="white"
)

sat_point = np.array([[x_km * scale, y_km * scale, z_km * scale]])
print("satellite position(scene units):", sat_point)

sat_poly = pv.PolyData(sat_point)
pl.add_points(
    sat_poly,
    color="red",
    point_size=14,
    render_points_as_spheres=True
)

pl.add_point_labels(
    sat_point,
    [sat.name],
    font_size=12,
    text_color="white",
    point_color="red",
    point_size=12,
    shape_opacity=0.2,
    always_visible=True
)

#camera reset    
original_camera_position = pl.camera_position

def reset_camera_callback():
    pl.camera_position = original_camera_position
    pl.render()
    
pl.add_key_event("r", reset_camera_callback)
pl.add_text("press R to reset camera", position="upper_left", font_size=10, color='white')

def update_scene():
    global label_points
    
    current_time=ts.now()
    new_label_points = []
    
    for sat in satellites:
        point = compute_satellite_point(sat, current_time, scale).reshape(1, 3)
        satellite_meshes[sat.name].points = point
        new_label_points.append(point[0])
        
        orbit_points = compute_orbit_points(sat, ts, scale, minutes_ahead=90, steps=120)
        satellite_line_meshes[sat.name].points = orbit_points
        satellite_line_meshes[sat.name].lines = np.hstack(([len(orbit_points)], np.arange(len(orbit_points))))
        
    label_points = new_label_points
        
    if label_actor is not None:
        build_labels()
        
    pl.render()
    
def timer_callback(*args):
    update_scene()
    
pl.add_timer_event(max_steps=1000000, duration=2000, callback=timer_callback)

pl.show()