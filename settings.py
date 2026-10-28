"""
User settings kept between runs, in settings.json next to the app (git-ignored).
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")


def _read():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            settings = json.load(f)
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def load_setting(key, default=None):
    return _read().get(key, default)


def save_setting(key, value):
    """Store one setting, keeping the others. Writes a temp file, then renames it."""
    settings = _read()
    settings[key] = value
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)
