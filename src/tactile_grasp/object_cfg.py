"""Object configuration for the tactile grasp task."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

from .paths import BOX_TALL_XML, CUBE_24MM_XML, CYLINDER_24MM_XML

OBJECT_XML_BY_NAME = {
    "cube_24mm": CUBE_24MM_XML,
    "box_tall": BOX_TALL_XML,
    "cylinder_24mm": CYLINDER_24MM_XML,
}


def object_spec(object_name: str) -> mujoco.MjSpec:
    """Load a tabletop primitive object spec."""
    return mujoco.MjSpec.from_file(str(OBJECT_XML_BY_NAME[object_name]))


def build_object_cfg(object_name: str = "cube_24mm") -> EntityCfg:
    """Build one tabletop primitive object entity config."""
    return EntityCfg(
        spec_fn=lambda object_name=object_name: object_spec(object_name),
        init_state=EntityCfg.InitialStateCfg(
            pos=(2.0, 2.0, 0.012),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )


def build_object_cfgs() -> dict[str, EntityCfg]:
    """Build all tabletop primitive object configs keyed by entity name."""
    return {object_name: build_object_cfg(object_name) for object_name in OBJECT_XML_BY_NAME}
