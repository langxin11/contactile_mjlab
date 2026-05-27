"""Tactile observation helpers for contactile grasp tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..constants import OBJECT_ENTITY_NAMES
from .actions import CartesianMocapAction, RobotiqCommandAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _sensor_tensor(env: "ManagerBasedRlEnv", name: str) -> torch.Tensor:
    """Read one XML-defined builtin sensor by prefixed name."""
    return env.scene[name].data


def sensor_values(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Stack arbitrary builtin sensors into a flat observation tensor."""
    values = [_sensor_tensor(env, f"{entity_name}/{name}") for name in sensor_names]
    return torch.cat(values, dim=-1)


def gripper_command(
    env: "ManagerBasedRlEnv",
    action_name: str = "cartesian_gripper",
) -> torch.Tensor:
    """Expose the normalized Robotiq command buffer."""
    action_term = env.action_manager.get_term(action_name)
    assert isinstance(action_term, RobotiqCommandAction | CartesianMocapAction)
    return action_term.command / 255.0


def active_object_position(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return active object root position for each env."""
    active_ids = getattr(env, "_tactile_active_object_ids", None)
    if active_ids is None:
        active_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    for object_id, object_name in enumerate(OBJECT_ENTITY_NAMES):
        mask = active_ids == object_id
        if torch.any(mask):
            out[mask] = env.scene[object_name].data.root_link_pos_w[mask]
    return out


def active_object_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return active object yaw for each env."""
    active_ids = getattr(env, "_tactile_active_object_ids", None)
    if active_ids is None:
        active_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    for object_id, object_name in enumerate(OBJECT_ENTITY_NAMES):
        mask = active_ids == object_id
        if torch.any(mask):
            quat = env.scene[object_name].data.root_link_quat_w[mask]
            out[mask] = _yaw_from_quat(quat)
    return out


def robot_position(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return robot root position in world frame."""
    return env.scene["robot"].data.root_link_pos_w


def robot_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return robot commanded yaw when available."""
    action_term = env.action_manager.get_term("cartesian_gripper")
    if isinstance(action_term, CartesianMocapAction):
        return action_term.yaw_command
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def vision_proxy(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Low-dimensional visual proxy: relative object pose and object-type one-hot."""
    obj_pos = active_object_position(env)
    rel_pos = obj_pos - robot_position(env)
    dyaw = active_object_yaw(env) - robot_yaw(env)
    one_hot = torch.zeros(
        (env.num_envs, len(OBJECT_ENTITY_NAMES)), device=env.device, dtype=torch.float32
    )
    active_ids = getattr(env, "_tactile_active_object_ids", None)
    if active_ids is None:
        active_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    one_hot[torch.arange(env.num_envs, device=env.device), active_ids] = 1.0
    return torch.cat(
        [
            rel_pos,
            torch.sin(dyaw).unsqueeze(-1),
            torch.cos(dyaw).unsqueeze(-1),
            one_hot,
        ],
        dim=-1,
    )


def _stack_taxel_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str,
) -> torch.Tensor:
    """Stack each taxel's 3D force into shape (N, n_taxels, 3) — internal helper."""
    per_taxel = [_sensor_tensor(env, f"{entity_name}/{name}") for name in sensor_names]
    return torch.stack(per_taxel, dim=1)


def taxel_normal_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Per-taxel normal force (z component), shape (N, n_taxels).

    PTS sphere taxel site-local frame: z = normal (out of finger), xy = tangential.
    """
    stacked = _stack_taxel_force(env, sensor_names, entity_name)
    return stacked[..., 2]


def taxel_tangential_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Per-taxel tangential force (xy), flattened to (N, 2 * n_taxels)."""
    stacked = _stack_taxel_force(env, sensor_names, entity_name)
    tangential = stacked[..., :2]
    return tangential.reshape(tangential.shape[0], -1)


def pad_force(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Pad-aggregated 3D force."""
    return _sensor_tensor(env, f"{entity_name}/{sensor_name}")


def pad_torque(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Pad-aggregated 3D torque."""
    return _sensor_tensor(env, f"{entity_name}/{sensor_name}")


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert world quaternion to yaw around z."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
