"""tactile_grasp 任务的环境 cfg 构造（mjlab idiom）."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CameraCfg, LightCfg
from mjlab.viewer import ViewerConfig

from .constants import (
    LEFT_TAXEL_FORCE_SENSOR_NAMES,
    OBJECT_CFG,
    RIGHT_TAXEL_FORCE_SENSOR_NAMES,
    ROBOT_JOINT_CFG,
    TACTILE_ACTIVITY_THRESHOLD,
)
from .mdp import events, rewards, terminations
from .mdp import observations as obs
from .object_cfg import build_object_cfgs
from .robot_cfg import build_action_cfg, build_robot_cfg

DECIMATION = 10
TIMESTEP = 0.002
EPISODE_LENGTH_S = 3.0
PLAY_EPISODE_LENGTH_S = 6.0

DELTA_U_MAX = 3.0

NORMAL_FORCE_SCALE = 15.0
TANGENTIAL_FORCE_SCALE = 4.0
FORCE_SCALE = 20.0
TORQUE_SCALE = 2.0

TACTILE_HISTORY_LENGTH = 5
WRENCH_HISTORY_LENGTH = 3

NUM_ENVS = 64
PLAY_NUM_ENVS = 1
ENV_SPACING = 0.5
SIM_NCONMAX = 128
SIM_NJMAX = 256

DROP_HEIGHT = 0.002
SUCCESS_HEIGHT = 0.08
SUCCESS_HOLD_STEPS = 25

TACTILE_CONTACT_THRESHOLD = 0.005
REACH_K_POS = 10.0
ALIGN_K_XY = 20.0
HOLD_LIFT_THRESHOLD = 0.03

W_REACH = 0.6
W_ALIGN = 0.8
W_CONTACT = 0.2
W_COVERAGE = 1.2
W_LIFT_DELTA = 8.0
W_HOLD = 2.0
W_FLOOR = -12.0
W_ACTION_SMOOTHNESS = -0.01
W_CLOSE_NEAR = 2.5
CLOSE_NEAR_K_D = 30.0
W_DROP_PENALTY = -5.0


def _actor_observation_terms() -> dict[str, ObservationTermCfg]:
    """Build the actor observation term dict (kept private to env_cfgs)."""
    return {
        "left_taxel_normal": ObservationTermCfg(
            func=obs.taxel_normal_force,
            params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / NORMAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "left_taxel_tangential": ObservationTermCfg(
            func=obs.taxel_tangential_force,
            params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / TANGENTIAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "right_taxel_normal": ObservationTermCfg(
            func=obs.taxel_normal_force,
            params={"sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / NORMAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "right_taxel_tangential": ObservationTermCfg(
            func=obs.taxel_tangential_force,
            params={"sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / TANGENTIAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "left_pad_force": ObservationTermCfg(
            func=obs.pad_force,
            params={"sensor_name": "left_pad_force"},
            scale=1.0 / FORCE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "left_pad_torque": ObservationTermCfg(
            func=obs.pad_torque,
            params={"sensor_name": "left_pad_torque"},
            scale=1.0 / TORQUE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "right_pad_force": ObservationTermCfg(
            func=obs.pad_force,
            params={"sensor_name": "right_pad_force"},
            scale=1.0 / FORCE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "right_pad_torque": ObservationTermCfg(
            func=obs.pad_torque,
            params={"sensor_name": "right_pad_torque"},
            scale=1.0 / TORQUE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "gripper_command": ObservationTermCfg(func=obs.gripper_command),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "vision_proxy": ObservationTermCfg(func=obs.vision_proxy),
        "last_action": ObservationTermCfg(func=last_action),
    }


def _add_debug_camera_and_light(spec) -> None:
    """Add debug camera/light for viewer and future image work."""
    CameraCfg(
        name="overhead_debug",
        body="world",
        mode="fixed",
        fovy=45.0,
        pos=(0.0, -0.32, 0.42),
        quat=(0.9238795, 0.3826834, 0.0, 0.0),
    ).edit_spec(spec)
    LightCfg(
        name="overhead_key",
        body="world",
        mode="fixed",
        type="spot",
        pos=(0.0, -0.25, 0.45),
        dir=(0.0, 0.4, -1.0),
        cutoff=60.0,
    ).edit_spec(spec)


def make_tactile_grasp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the tactile_grasp env cfg; if play=True, apply play overrides to the fresh cfg."""
    actor_terms = _actor_observation_terms()
    force_stage = 2 if play else None

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": build_robot_cfg(), **build_object_cfgs()},
            num_envs=NUM_ENVS,
            env_spacing=ENV_SPACING,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            spec_fn=_add_debug_camera_and_light,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
            "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
        },
        actions={"cartesian_gripper": build_action_cfg(DELTA_U_MAX)},
        events={
            "reset_pick_lift_scene": EventTermCfg(
                func=events.reset_pick_lift_scene,
                mode="reset",
                params={"force_stage": force_stage},
            )
        },
        rewards={
            "reach3d": RewardTermCfg(
                func=rewards.reach3d,
                weight=W_REACH,
                params={"k_pos": REACH_K_POS},
            ),
            "align": RewardTermCfg(
                func=rewards.align_xy,
                weight=W_ALIGN,
                params={"k_xy": ALIGN_K_XY},
            ),
            "contact": RewardTermCfg(
                func=rewards.tactile_contact_binary,
                weight=W_CONTACT,
                params={
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "threshold": TACTILE_CONTACT_THRESHOLD,
                },
            ),
            "coverage": RewardTermCfg(
                func=rewards.taxel_coverage,
                weight=W_COVERAGE,
                params={
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "threshold": TACTILE_CONTACT_THRESHOLD,
                },
            ),
            "lift_delta": RewardTermCfg(func=rewards.lift_delta, weight=W_LIFT_DELTA),
            "hold": RewardTermCfg(
                func=rewards.hold_bonus,
                weight=W_HOLD,
                params={
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "threshold": TACTILE_CONTACT_THRESHOLD,
                    "lift_threshold": HOLD_LIFT_THRESHOLD,
                },
            ),
            "floor_collision": RewardTermCfg(
                func=rewards.robot_floor_collision,
                weight=W_FLOOR,
            ),
            "action_smoothness": RewardTermCfg(
                func=rewards.action_smoothness_l1,
                weight=W_ACTION_SMOOTHNESS,
            ),
            "close_near_object": RewardTermCfg(
                func=rewards.close_near_object,
                weight=W_CLOSE_NEAR,
                params={"k_d": CLOSE_NEAR_K_D, "action_name": "cartesian_gripper"},
            ),
            "drop_penalty": RewardTermCfg(func=rewards.drop_penalty, weight=W_DROP_PENALTY),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "object_drop": TerminationTermCfg(
                func=terminations.object_height_below,
                params={"minimum_height": DROP_HEIGHT, "asset_cfg": OBJECT_CFG},
            ),
            "robot_out_of_workspace": TerminationTermCfg(func=terminations.robot_out_of_workspace),
            "stable_grasp": TerminationTermCfg(
                func=terminations.stable_grasp_hold,
                params={
                    "hold_steps": SUCCESS_HOLD_STEPS,
                    "minimum_height": SUCCESS_HEIGHT,
                    "minimum_tactile_signal": TACTILE_ACTIVITY_THRESHOLD,
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "asset_cfg": OBJECT_CFG,
                },
            ),
        },
        sim=SimulationCfg(
            nconmax=SIM_NCONMAX,
            njmax=SIM_NJMAX,
            mujoco=MujocoCfg(timestep=TIMESTEP, cone="elliptic", impratio=10.0),
        ),
        viewer=ViewerConfig(),
        curriculum={
            "pick_lift_stage": CurriculumTermCfg(
                func=events.pick_lift_curriculum,
                params={"force_stage": force_stage},
            )
        },
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        auto_reset=True,
    )

    if play:
        cfg.scene.num_envs = PLAY_NUM_ENVS
        cfg.episode_length_s = PLAY_EPISODE_LENGTH_S
        cfg.observations["actor"].enable_corruption = False

    return cfg
