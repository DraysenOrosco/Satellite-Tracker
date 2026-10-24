# Satellite-Tracker

A 3D satellite tracker built with PyVista. It shows about 19,000 objects from CelesTrak: active satellites, launches from the last 30 days, and the major debris clouds. Each object is propagated with SGP4 on an accelerated simulation clock over a rotating Earth.

## Features

- **Search:** find any object by name or NORAD ID and fly to it.
- **Day/night shading:** shades the night side of the Earth. Satellites in Earth's shadow are dimmed, and the tracking card shows whether a satellite is sunlit.
- **Coverage footprint:** for the tracked satellite, the area it covers at a minimum elevation of 10° (filled) and its horizon (ring).
- **Ground track:** where the tracked satellite will pass over the ground during its next 3 orbits.
- **Close-approach screening:** finds every pass under 5 km in the next 6 hours, in a background process. The closest upcoming passes are listed first, and clicking one jumps the clock there so you can watch it. Passes under 1 km get 3D alerts, and the satellite you're tracking warns you 10 minutes ahead.
- **Weather overlay:** live global clouds (EUMETSAT) and precipitation (NASA IMERG).
- **Chase camera, hover tooltips and labels.** Each object type can be shown or hidden.

## Run

```
pip install pyvista skyfield sgp4 numpy pillow
python main.py                          # everything
python main.py --groups stations visual # a lighter set (any CelesTrak group names)
python main.py --offline                # no downloads: use cached data, or the bundled stations.tle
```

Orbital elements are downloaded in OMM (CSV) format, because new objects now have catalogue numbers too large for TLEs. They're cached in `tle_cache/` for 2 hours, since CelesTrak temporarily blocks clients that download the same data more often than that. Weather images are cached in `weather_cache/` for 30 minutes.

## Controls

| Action | Input |
| --- | --- |
| Track a satellite (chase camera) | Click it |
| Search by name or NORAD ID | `/` or Ctrl+F, then Up/Down and Enter |
| Next / previous satellite of the same type | Tab / Shift+Tab |
| Release target, home view | Esc or H |
| Pause / resume | Space |
| Simulation speed | Left / Right, or the speed buttons |
| Labels | L |
| Day/night, footprint, ground track, close approaches | N, F, G, C (or click in Map Layers) |
| Weather overlay (Off / Clouds / Rain / Both) | W, or the weather buttons |
| Show / hide an object type | Click its legend row |
| Watch a close approach | Click it in the Close Approaches panel |
| Rotate / zoom | Drag / scroll |

## Accuracy notes

- Orbital elements of this kind (TLE/OMM, propagated with SGP4) are good to roughly 1 km near their epoch, and worse for debris and as they age. Close approaches are screening estimates, not collision predictions. Starlink satellites also manoeuvre on their own to avoid each other, which is why Starlink–Starlink pairs are hidden by default.
- The Sun's position uses a low-precision almanac formula, accurate to about 0.01°.
- Screening covers 6 hours of simulated time and takes about 30 s, so at the fastest speeds it can't keep up.

## Data sources

- **Orbits:** [CelesTrak](https://celestrak.org) GP element sets
- **Clouds:** EUMETSAT world cloud map, an infrared 10.8 µm mosaic of geostationary weather satellites, updated every 3 hours
- **Precipitation:** NASA GPM IMERG, 30-minute near-real-time data, served by NASA GIBS

## Files

- `main.py`: the app (scene, simulation, interaction)
- `data.py`: CelesTrak download, caching and object categories
- `conjunctions.py`: close-approach screening (runs in a worker process)
- `weather.py`: weather layer download and processing (runs in the background)
- `ui.py`: on-screen panels, buttons and text
