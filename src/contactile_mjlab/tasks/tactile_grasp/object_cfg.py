"""Object configuration for the tactile grasp task."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

from ...paths import PROPS_DIR


def object_spec() -> mujoco.MjSpec:
    """Load the hanging-box object spec."""
    return mujoco.MjSpec.from_file(str(PROPS_DIR / "hanging_box.xml"))


def build_object_cfg() -> EntityCfg:
    """Build the object entity config."""
    return EntityCfg(
        spec_fn=object_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.154),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
