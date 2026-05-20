"""Project paths shared by scripts and runtime code."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets" / "robotiq_2f85"
BASE_XML = ASSETS_DIR / "2f85.xml"
TACTILE_XML = ASSETS_DIR / "2f85_tactile.xml"
TACTILE_SCENE_XML = ASSETS_DIR / "scene_tactile.xml"
PTS_SPHERES_XML = ASSETS_DIR / "2f85_pts_spheres.xml"
PTS_SPHERES_SCENE_XML = ASSETS_DIR / "scene_pts_spheres.xml"
PROPS_DIR = PROJECT_ROOT / "assets" / "props"
