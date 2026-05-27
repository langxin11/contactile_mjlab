"""tactile_grasp event and curriculum terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import reset_scene_to_default

from ..constants import OBJECT_ENTITY_NAMES, OBJECT_HALF_HEIGHTS
from .actions import CartesianMocapAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pick_lift_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    force_stage: int | None = None,
) -> dict[str, int]:
    """Set and report the pick-lift curriculum stage."""
    del env_ids
    if force_stage is not None:
        stage = int(force_stage)
        if stage not in (0, 1, 2):
            raise ValueError(f"force_stage must be one of {{0, 1, 2}}, got {force_stage!r}.")
    elif env.common_step_counter < 20_000:
        stage = 0
    elif env.common_step_counter < 80_000:
        stage = 1
    else:
        stage = 2
    env._tactile_curriculum_stage = stage
    return {"stage": stage}


class reset_pick_lift_scene:
    """Reset tabletop objects and top-down robot mocap pose.

    Class-based event term: __init__ runs during EventManager construction,
    which is before ObservationManager dry-runs the observation terms. We
    pre-allocate the per-env caches there so observation helpers' strict
    "must be initialized" guards see valid (zero) state during shape
    detection, while the actual sampling still happens on reset.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        """Pre-allocate per-env caches so observation dry-run sees valid state."""
        del cfg
        _ensure_buffers(env)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        force_stage: int | None = None,
    ) -> None:
        """Sample stage cfg, place active object on table, and command top-down mocap."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        reset_scene_to_default(env, env_ids)
        stage = pick_lift_curriculum(env, env_ids, force_stage=force_stage)["stage"]
        cfg = _stage_cfg(stage)

        num = len(env_ids)
        active_ids = _sample_active_ids(num, cfg["object_ids"], env.device)
        xy_range = float(cfg["xy_range"])
        yaw_range = float(cfg["yaw_range"])
        object_xy = (torch.rand((num, 2), device=env.device) * 2.0 - 1.0) * xy_range
        object_yaw = (torch.rand(num, device=env.device) * 2.0 - 1.0) * yaw_range
        robot_xy = object_xy + (torch.rand((num, 2), device=env.device) * 2.0 - 1.0) * float(
            cfg["robot_xy_offset"]
        )
        robot_yaw = object_yaw + (torch.rand(num, device=env.device) * 2.0 - 1.0) * float(
            cfg["robot_yaw_offset"]
        )

        _ensure_buffers(env)
        env._tactile_active_object_ids[env_ids] = active_ids
        env._tactile_active_object_local_pos[env_ids, :2] = object_xy
        env._tactile_active_object_local_yaw[env_ids] = object_yaw

        half_heights = torch.tensor(OBJECT_HALF_HEIGHTS, device=env.device, dtype=torch.float32)
        env._tactile_active_object_local_pos[env_ids, 2] = half_heights[active_ids]
        env._tactile_active_object_init_z[env_ids] = (
            half_heights[active_ids] + env.scene.env_origins[env_ids, 2]
        )

        for object_id, object_name in enumerate(OBJECT_ENTITY_NAMES):
            entity = env.scene[object_name]
            root_state = torch.zeros((num, 13), device=env.device, dtype=torch.float32)
            root_state[:, 3] = 1.0
            local_pos = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
            inactive_x = 1.5 + float(object_id) * 0.25
            local_pos[:, 0] = inactive_x
            local_pos[:, 1] = 1.5
            local_pos[:, 2] = half_heights[object_id]

            active_mask = active_ids == object_id
            if torch.any(active_mask):
                local_pos[active_mask, :2] = object_xy[active_mask]
                local_pos[active_mask, 2] = half_heights[object_id]
                root_state[active_mask, 3:7] = _yaw_quat(object_yaw[active_mask])

            root_state[:, 0:3] = local_pos + env.scene.env_origins[env_ids]
            entity.write_root_state_to_sim(root_state, env_ids=env_ids)

        robot_pos = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
        robot_pos[:, :2] = robot_xy
        robot_pos[:, 2] = 0.24
        env._tactile_robot_init_pos_local[env_ids] = robot_pos
        env._tactile_robot_init_yaw[env_ids] = robot_yaw

        action = env.action_manager.get_term("cartesian_gripper")
        if isinstance(action, CartesianMocapAction):
            action.set_pose_command(robot_pos, robot_yaw, env_ids=env_ids)


def _stage_cfg(stage: int) -> dict[str, float | tuple[int, ...]]:
    if stage == 0:
        return {
            "object_ids": (0,),
            "xy_range": 0.0,
            "yaw_range": 0.0,
            "robot_xy_offset": 0.0,
            "robot_yaw_offset": 0.0,
        }
    if stage == 1:
        return {
            "object_ids": (0, 1),
            "xy_range": 0.03,
            "yaw_range": math.pi / 6.0,
            "robot_xy_offset": 0.015,
            "robot_yaw_offset": math.pi / 12.0,
        }
    return {
        "object_ids": (0, 1, 2),
        "xy_range": 0.08,
        "yaw_range": math.pi,
        "robot_xy_offset": 0.03,
        "robot_yaw_offset": math.pi / 4.0,
    }


def _sample_active_ids(num: int, object_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    ids = torch.tensor(object_ids, device=device, dtype=torch.long)
    sampled = torch.randint(0, len(object_ids), (num,), device=device)
    return ids[sampled]


def _ensure_buffers(env: "ManagerBasedRlEnv") -> None:
    if not hasattr(env, "_tactile_active_object_ids"):
        env._tactile_active_object_ids = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
        env._tactile_active_object_local_pos = torch.zeros(
            (env.num_envs, 3), device=env.device, dtype=torch.float32
        )
        env._tactile_active_object_local_yaw = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        env._tactile_robot_init_pos_local = torch.zeros(
            (env.num_envs, 3), device=env.device, dtype=torch.float32
        )
        env._tactile_robot_init_yaw = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        env._tactile_active_object_init_z = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    half = yaw * 0.5
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat


__all__ = ["pick_lift_curriculum", "reset_pick_lift_scene", "reset_scene_to_default"]
