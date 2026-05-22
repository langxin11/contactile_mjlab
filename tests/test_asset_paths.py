"""Verify packaged asset paths exist and resolve inside the installed package."""

from __future__ import annotations

from pathlib import Path

import pytest

import contactile_mjlab
from contactile_mjlab import paths


@pytest.mark.parametrize(
    "attr",
    [
        "ASSETS_DIR",
        "BASE_XML",
        "TACTILE_XML",
        "TACTILE_SCENE_XML",
        "PTS_SPHERES_XML",
        "PTS_SPHERES_SCENE_XML",
    ],
)
def test_asset_path_exists_inside_package(attr: str) -> None:
    """Each exported asset path must exist AND live inside the installed package dir."""
    p: Path = getattr(paths, attr)
    assert p.exists(), f"{attr} -> {p} does not exist"
    package_dir = Path(contactile_mjlab.__file__).resolve().parent
    resolved = p.resolve()
    assert package_dir in resolved.parents, (
        f"{attr} -> {resolved} is not under packaged dir {package_dir}; "
        "assets must ship with the wheel."
    )
