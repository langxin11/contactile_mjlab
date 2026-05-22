"""Robotiq + PTS spheres 模型构造."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from ._mdp_legacy.actions import RobotiqCommandActionCfg
from ._mdp_legacy.actuators import RobotiqGeneralActuatorCfg
from .paths import PTS_SPHERES_XML


def robot_spec() -> mujoco.MjSpec:
    """加载 PTS spheres 触觉模型 spec."""
    return mujoco.MjSpec.from_file(str(PTS_SPHERES_XML))


def build_robot_cfg() -> EntityCfg:
    """构建 Robotiq entity config."""
    return EntityCfg(
        spec_fn=robot_spec,
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
    """Robotiq Δu 动作 config."""
    return RobotiqCommandActionCfg(
        entity_name="robot",
        actuator_name="fingers_actuator",
        tendon_name="split",
        delta_u_max=delta_u_max,
    )
