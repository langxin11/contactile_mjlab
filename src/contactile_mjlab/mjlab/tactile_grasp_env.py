"""mjlab-native tactile grasp environment."""
# ruff: noqa: E402

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _PROJECT_ROOT / ".cache"
(_CACHE_DIR / "warp").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WARP_CACHE_PATH", str(_CACHE_DIR / "warp"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "mpl"))

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from ..paths import PROJECT_ROOT, TACTILE_XML
from . import mdp
from .action_terms import RobotiqCommandActionCfg
from .actuators import RobotiqGeneralActuatorCfg

LEFT_TOUCH_NAMES = tuple(f"left_touch_{row}{col}" for row in range(3) for col in range(3))
RIGHT_TOUCH_NAMES = tuple(f"right_touch_{row}{col}" for row in range(3) for col in range(3))

_ROBOT_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "left_driver_joint",
        "left_spring_link_joint",
        "left_follower",
        "right_driver_joint",
        "right_spring_link_joint",
        "right_follower_joint",
    ),
)
_OBJECT_CFG = SceneEntityCfg("object")


def _robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(TACTILE_XML))


def _object_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(PROJECT_ROOT / "assets" / "props" / "hanging_box.xml"))


def _robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_robot_spec,
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


def _object_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_object_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.154),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )


@dataclass(kw_only=True)
class TactileGraspEnvConfig:
    """Small wrapper over the mjlab config knobs used by this project."""

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
    enable_corruption: bool = False
    auto_reset: bool = True

    def build(self) -> ManagerBasedRlEnvCfg:
        """Build the underlying mjlab environment config."""
        actor_terms = {
            "left_touch": ObservationTermCfg(
                func=mdp.touch_map,
                params={"sensor_names": LEFT_TOUCH_NAMES},
                scale=1.0 / self.touch_scale,
            ),
            "right_touch": ObservationTermCfg(
                func=mdp.touch_map,
                params={"sensor_names": RIGHT_TOUCH_NAMES},
                scale=1.0 / self.touch_scale,
            ),
            "left_wrench": ObservationTermCfg(
                func=mdp.pad_wrench,
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
                func=mdp.pad_wrench,
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
            "gripper_command": ObservationTermCfg(func=mdp.gripper_command),
            "joint_pos": ObservationTermCfg(
                func=joint_pos_rel,
                params={"asset_cfg": _ROBOT_JOINT_CFG},
            ),
            "joint_vel": ObservationTermCfg(
                func=joint_vel_rel,
                params={"asset_cfg": _ROBOT_JOINT_CFG},
            ),
            "last_action": ObservationTermCfg(func=last_action),
        }

        observations = {
            "actor": ObservationGroupCfg(
                actor_terms,
                enable_corruption=self.enable_corruption,
            ),
            "critic": ObservationGroupCfg({**actor_terms}),
        }

        actions: dict[str, ActionTermCfg] = {
            "gripper_command": RobotiqCommandActionCfg(
                entity_name="robot",
                actuator_name="fingers_actuator",
                tendon_name="split",
                delta_u_max=self.delta_u_max,
            ),
        }

        rewards = {
            "alive": RewardTermCfg(func=mdp.alive, weight=1.0),
            "touch_force": RewardTermCfg(
                func=mdp.touch_force_l2,
                weight=-0.01,
                params={
                    "left_sensor_names": LEFT_TOUCH_NAMES,
                    "right_sensor_names": RIGHT_TOUCH_NAMES,
                },
            ),
            "action_rate": RewardTermCfg(
                func=mdp.action_l2,
                weight=-0.001,
            ),
            "close_command": RewardTermCfg(
                func=mdp.close_command_l2,
                weight=-0.001,
            ),
            "drop_penalty": RewardTermCfg(
                func=lambda env: env.termination_manager.get_term("object_drop").float(),
                weight=-5.0,
            ),
        }

        terminations = {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "object_drop": TerminationTermCfg(
                func=mdp.object_height_below,
                params={
                    "minimum_height": self.drop_height_threshold,
                    "asset_cfg": _OBJECT_CFG,
                },
            ),
            "stable_grasp": TerminationTermCfg(
                func=mdp.stable_grasp_hold,
                params={
                    "hold_steps": self.success_hold_steps,
                    "minimum_height": self.success_height_threshold,
                    "left_sensor_names": LEFT_TOUCH_NAMES,
                    "right_sensor_names": RIGHT_TOUCH_NAMES,
                    "asset_cfg": _OBJECT_CFG,
                },
            ),
        }

        return ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                entities={
                    "robot": _robot_cfg(),
                    "object": _object_cfg(),
                },
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
            ),
            observations=observations,
            actions=actions,
            events={
                "reset_scene_to_default": EventTermCfg(
                    func=reset_scene_to_default,
                    mode="reset",
                )
            },
            rewards=rewards,
            terminations=terminations,
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


class TactileGraspEnv(ManagerBasedRlEnv):
    """mjlab environment for tactile Robotiq grasping."""

    def __init__(
        self,
        config: TactileGraspEnvConfig | None = None,
        *,
        device: str = "cpu",
        render_mode: str | None = None,
    ) -> None:
        """Construct the environment from the lightweight project config."""
        self.project_config = config or TactileGraspEnvConfig()
        super().__init__(self.project_config.build(), device=device, render_mode=render_mode)


def tactile_grasp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return a PPO runner config compatible with `mjlab.rl`."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="contactile_mjlab",
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=200,
    )
