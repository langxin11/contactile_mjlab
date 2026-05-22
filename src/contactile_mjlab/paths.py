"""Asset paths shipped with the contactile_mjlab package.

All paths are derived from ``__file__`` so they remain correct whether the
package is run from a source checkout or installed in site-packages.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = _PACKAGE_DIR / "assets" / "robotiq_2f85"
BASE_XML = ASSETS_DIR / "2f85.xml"
TACTILE_XML = ASSETS_DIR / "2f85_tactile.xml"
TACTILE_SCENE_XML = ASSETS_DIR / "scene_tactile.xml"
PTS_SPHERES_XML = ASSETS_DIR / "2f85_pts_spheres.xml"
PTS_SPHERES_SCENE_XML = ASSETS_DIR / "scene_pts_spheres.xml"
PROPS_DIR = _PACKAGE_DIR / "assets" / "props"
