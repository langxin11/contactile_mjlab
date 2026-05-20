"""Robot configuration for tactile grasp task variants."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from ...mjlab.action_terms import RobotiqCommandActionCfg
from ...mjlab.actuators import RobotiqGeneralActuatorCfg
from ...paths import PTS_SPHERES_XML, TACTILE_XML
from .constants import TACTILE_MODEL_PTS_SPHERES, TACTILE_MODEL_TOUCH_SITE


def robot_spec(tactile_model: str) -> mujoco.MjSpec:
    """Load the tactile robot spec for the selected model."""
    xml_path = {
        TACTILE_MODEL_TOUCH_SITE: TACTILE_XML,
        TACTILE_MODEL_PTS_SPHERES: PTS_SPHERES_XML,
    }[tactile_model]
    return mujoco.MjSpec.from_file(str(xml_path))


def build_robot_cfg(tactile_model: str) -> EntityCfg:
    """Build the Robotiq entity config."""
    return EntityCfg(
        spec_fn=lambda: robot_spec(tactile_model),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                RobotiqGeneralActuatorCfg(
                    target_names_expr=("split",),
                    transmission_type="tendon",
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "left_driver_joint": 0.0,
                "left_spring_link_joint": 0.0,
                "left_follower": 0.0,
                "right_driver_joint": 0.0,
                "right_spring_link_joint": 0.0,
                "right_follower_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


def build_action_cfg(delta_u_max: float) -> RobotiqCommandActionCfg:
    """Build the shared Robotiq delta-command action config."""
    return RobotiqCommandActionCfg(
        entity_name="robot",
        actuator_name="fingers_actuator",
        tendon_name="split",
        delta_u_max=delta_u_max,
    )
