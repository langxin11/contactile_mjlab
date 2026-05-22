"""Asset paths shipped with the contactile_mjlab package.

All paths are derived from ``__file__`` so they remain correct whether the
package is run from a source checkout or installed in site-packages.
"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
ROBOTIQ_DIR = ASSETS_DIR / "robotiq_2f85"
PROPS_DIR = ASSETS_DIR / "props"

PTS_SPHERES_XML = ROBOTIQ_DIR / "2f85_pts_spheres.xml"
PTS_SPHERES_SCENE_XML = ROBOTIQ_DIR / "scene_pts_spheres.xml"
HANGING_BOX_XML = PROPS_DIR / "hanging_box.xml"

BASE_XML = ROBOTIQ_DIR / "2f85.xml"
TACTILE_XML = ROBOTIQ_DIR / "2f85_tactile.xml"
TACTILE_SCENE_XML = ROBOTIQ_DIR / "scene_tactile.xml"
