"""Launch the MuJoCo viewer for the tactile scene."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_grasp.paths import PTS_SPHERES_SCENE_XML, TACTILE_SCENE_XML


def main() -> None:
    """Open a tactile scene in the interactive MuJoCo viewer."""
    scene_xml = PTS_SPHERES_SCENE_XML if "--pts" in sys.argv else TACTILE_SCENE_XML
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


if __name__ == "__main__":
    main()
