"""Compile one or more MJCF files and print model statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_grasp.paths import (
    BASE_XML,
    PTS_SPHERES_SCENE_XML,
    PTS_SPHERES_XML,
    TACTILE_SCENE_XML,
    TACTILE_XML,
)


def _check(path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    print(
        f"{path}: nq={model.nq} nv={model.nv} nsensor={model.nsensor} "
        f"nsite={model.nsite} nu={model.nu}"
    )


def main() -> None:
    """Compile the configured MJCF files and print basic model stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or [
        BASE_XML,
        TACTILE_XML,
        TACTILE_SCENE_XML,
        PTS_SPHERES_XML,
        PTS_SPHERES_SCENE_XML,
    ]
    for path in paths:
        _check(path.resolve())


if __name__ == "__main__":
    main()
