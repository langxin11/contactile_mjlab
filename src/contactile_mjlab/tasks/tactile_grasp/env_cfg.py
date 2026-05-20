"""Environment config builder for tactile grasp tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from . import reward_terms, tactile_terms
from .constants import (
    LEFT_TAXEL_FORCE_SENSOR_NAMES,
    LEFT_TOUCH_SENSOR_NAMES,
    OBJECT_CFG,
    PTS_SPHERES_TASK_ID,
    RIGHT_TAXEL_FORCE_SENSOR_NAMES,
    RIGHT_TOUCH_SENSOR_NAMES,
    ROBOT_JOINT_CFG,
    TACTILE_ACTIVITY_THRESHOLD_BY_TASK,
    TACTILE_MODEL_PTS_SPHERES,
    TACTILE_MODEL_TOUCH_SITE,
    TOUCH_SITE_TASK_ID,
)
from .object_cfg import build_object_cfg
from .robot_cfg import build_action_cfg, build_robot_cfg


@dataclass(kw_only=True)
class TactileGraspTaskConfig:
    """Thin wrapper over the mjlab config knobs used by this project."""

    tactile_model: str = TACTILE_MODEL_PTS_SPHERES
    num_envs: int = 1
    env_spacing: float = 0.5
    decimation: int = 10
    timestep: float = 0.002
    episode_length_s: float = 3.0
    delta_u_max: float = 3.0
    touch_scale: float = 10.0
    force_scale: float = 20.0
    torque_scale: float = 2.0
    drop_height_threshold: float = 0.08
    success_height_threshold: float = 0.14
    success_hold_steps: int = 25
    success_tactile_threshold: float | None = None
    enable_corruption: bool = False
    auto_reset: bool = True

    def build(self) -> ManagerBasedRlEnvCfg:
        """Build the underlying manager-based environment config."""
        if self.tactile_model == TACTILE_MODEL_TOUCH_SITE:
            left_sensor_names = LEFT_TOUCH_SENSOR_NAMES
            right_sensor_names = RIGHT_TOUCH_SENSOR_NAMES
            tactile_func = tactile_terms.touch_map
            tactile_scale = 1.0 / self.touch_scale
            tactile_threshold = TACTILE_ACTIVITY_THRESHOLD_BY_TASK[TOUCH_SITE_TASK_ID]
        elif self.tactile_model == TACTILE_MODEL_PTS_SPHERES:
            left_sensor_names = LEFT_TAXEL_FORCE_SENSOR_NAMES
            right_sensor_names = RIGHT_TAXEL_FORCE_SENSOR_NAMES
            tactile_func = tactile_terms.taxel_force_map
            tactile_scale = 1.0 / self.force_scale
            tactile_threshold = TACTILE_ACTIVITY_THRESHOLD_BY_TASK[PTS_SPHERES_TASK_ID]
        else:
            raise ValueError(f"Unsupported tactile model: {self.tactile_model}")

        if self.success_tactile_threshold is not None:
            tactile_threshold = self.success_tactile_threshold

        actor_terms = {
            "left_tactile": ObservationTermCfg(
                func=tactile_func,
                params={"sensor_names": left_sensor_names},
                scale=tactile_scale,
            ),
            "right_tactile": ObservationTermCfg(
                func=tactile_func,
                params={"sensor_names": right_sensor_names},
                scale=tactile_scale,
            ),
            "left_wrench": ObservationTermCfg(
                func=tactile_terms.pad_wrench,
                params={"force_sensor": "left_pad_force", "torque_sensor": "left_pad_torque"},
                scale=(
                    1.0 / self.force_scale,
                    1.0 / self.force_scale,
                    1.0 / self.force_scale,
                    1.0 / self.torque_scale,
                    1.0 / self.torque_scale,
                    1.0 / self.torque_scale,
                ),
            ),
            "right_wrench": ObservationTermCfg(
                func=tactile_terms.pad_wrench,
                params={"force_sensor": "right_pad_force", "torque_sensor": "right_pad_torque"},
                scale=(
                    1.0 / self.force_scale,
                    1.0 / self.force_scale,
                    1.0 / self.force_scale,
                    1.0 / self.torque_scale,
                    1.0 / self.torque_scale,
                    1.0 / self.torque_scale,
                ),
            ),
            "gripper_command": ObservationTermCfg(func=tactile_terms.gripper_command),
            "joint_pos": ObservationTermCfg(
                func=joint_pos_rel,
                params={"asset_cfg": ROBOT_JOINT_CFG},
            ),
            "joint_vel": ObservationTermCfg(
                func=joint_vel_rel,
                params={"asset_cfg": ROBOT_JOINT_CFG},
            ),
            "last_action": ObservationTermCfg(func=last_action),
        }

        return ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                entities={
                    "robot": build_robot_cfg(self.tactile_model),
                    "object": build_object_cfg(),
                },
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
            ),
            observations={
                "actor": ObservationGroupCfg(
                    actor_terms,
                    enable_corruption=self.enable_corruption,
                ),
                "critic": ObservationGroupCfg({**actor_terms}),
            },
            actions={"gripper_command": build_action_cfg(self.delta_u_max)},
            events={
                "reset_scene_to_default": EventTermCfg(
                    func=reset_scene_to_default,
                    mode="reset",
                )
            },
            rewards={
                "alive": RewardTermCfg(func=reward_terms.alive, weight=1.0),
                "tactile_force": RewardTermCfg(
                    func=reward_terms.tactile_force_l2,
                    weight=-0.01,
                    params={
                        "left_sensor_names": left_sensor_names,
                        "right_sensor_names": right_sensor_names,
                    },
                ),
                "action_rate": RewardTermCfg(
                    func=reward_terms.action_l2,
                    weight=-0.001,
                ),
                "close_command": RewardTermCfg(
                    func=reward_terms.close_command_l2,
                    weight=-0.001,
                ),
                "drop_penalty": RewardTermCfg(
                    func=lambda env: env.termination_manager.get_term("object_drop").float(),
                    weight=-5.0,
                ),
            },
            terminations={
                "time_out": TerminationTermCfg(func=time_out, time_out=True),
                "object_drop": TerminationTermCfg(
                    func=reward_terms.object_height_below,
                    params={
                        "minimum_height": self.drop_height_threshold,
                        "asset_cfg": OBJECT_CFG,
                    },
                ),
                "stable_grasp": TerminationTermCfg(
                    func=reward_terms.stable_grasp_hold,
                    params={
                        "hold_steps": self.success_hold_steps,
                        "minimum_height": self.success_height_threshold,
                        "minimum_tactile_signal": tactile_threshold,
                        "left_sensor_names": left_sensor_names,
                        "right_sensor_names": right_sensor_names,
                        "asset_cfg": OBJECT_CFG,
                    },
                ),
            },
            sim=SimulationCfg(
                mujoco=MujocoCfg(
                    timestep=self.timestep,
                    cone="elliptic",
                    impratio=10.0,
                )
            ),
            viewer=ViewerConfig(),
            decimation=self.decimation,
            episode_length_s=self.episode_length_s,
            auto_reset=self.auto_reset,
        )
