"""Verify packaged asset paths exist, ship inside the package, and load in MuJoCo."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

import tactile_grasp
from tactile_grasp import paths


@pytest.mark.parametrize(
    "xml_path",
    [
        paths.BASE_XML,
        paths.TACTILE_XML,
        paths.TACTILE_SCENE_XML,
        paths.PTS_SPHERES_XML,
        paths.PTS_SPHERES_SCENE_XML,
        paths.HANGING_BOX_XML,
    ],
)
def test_xml_loadable(xml_path: Path) -> None:
    """Each shipped XML must exist, live inside the package, and parse via MuJoCo."""
    assert xml_path.is_file(), f"missing: {xml_path}"
    package_dir = Path(tactile_grasp.__file__).resolve().parent
    resolved = xml_path.resolve()
    assert package_dir in resolved.parents, (
        f"{xml_path} is not under packaged dir {package_dir}; assets must ship with the wheel."
    )
    mujoco.MjSpec.from_file(str(xml_path))
