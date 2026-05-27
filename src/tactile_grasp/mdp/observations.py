"""Tactile observation helpers for contactile grasp tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..constants import OBJECT_ENTITY_NAMES
from .actions import CartesianMocapAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


TAXEL_TANGENTIAL_FORCE_LIMIT = 4.0
TAXEL_NORMAL_FORCE_LIMIT = 15.0


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
    command = getattr(action_term, "command", None)
    if command is None:
        raise TypeError(
            f"Action term '{action_name}' must expose a 'command' tensor for gripper_command()."
        )
    return command / 255.0


def _require_active_object_ids(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return initialized active object ids or raise a clear error."""
    active_ids = getattr(env, "_tactile_active_object_ids", None)
    if active_ids is None:
        raise RuntimeError(
            "Observation helper requires '_tactile_active_object_ids' to be initialized. "
            "Call env.reset() before requesting active-object observations."
        )
    return active_ids


def active_object_position(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return active object root position for each env."""
    active_ids = _require_active_object_ids(env)
    out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    for object_id, object_name in enumerate(OBJECT_ENTITY_NAMES):
        mask = active_ids == object_id
        if torch.any(mask):
            out[mask] = env.scene[object_name].data.root_link_pos_w[mask]
    return out


def active_object_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return active object yaw for each env."""
    active_ids = _require_active_object_ids(env)
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
    active_ids = _require_active_object_ids(env)
    obj_pos = active_object_position(env)
    rel_pos = obj_pos - robot_position(env)
    dyaw = active_object_yaw(env) - robot_yaw(env)
    one_hot = torch.zeros(
        (env.num_envs, len(OBJECT_ENTITY_NAMES)), device=env.device, dtype=torch.float32
    )
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


def _clip_taxel_force(force: torch.Tensor) -> torch.Tensor:
    """Clip taxel force to the configured sensor range."""
    clipped = force.clone()
    clipped[..., :2] = torch.clamp(
        clipped[..., :2],
        min=-TAXEL_TANGENTIAL_FORCE_LIMIT,
        max=TAXEL_TANGENTIAL_FORCE_LIMIT,
    )
    clipped[..., 2] = torch.clamp(
        clipped[..., 2],
        min=-TAXEL_NORMAL_FORCE_LIMIT,
        max=TAXEL_NORMAL_FORCE_LIMIT,
    )
    return clipped


def tool_position(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return the tool reference point in world frame."""
    robot = env.scene["robot"]
    find_sites = getattr(robot, "find_sites", None)
    if callable(find_sites):
        local_site_ids, site_names = find_sites(("left_pad_ft_site", "right_pad_ft_site"))
        if len(local_site_ids) == 2 and set(site_names) == {
            "left_pad_ft_site",
            "right_pad_ft_site",
        }:
            site_ids = robot.indexing.site_ids[local_site_ids]
            site_positions = env.sim.data.site_xpos[:, site_ids, :]
            return site_positions.mean(dim=1)

    action_term = env.action_manager.get_term("cartesian_gripper")
    pose_command_local = getattr(action_term, "pose_command_local", None)
    env_origins = getattr(env.scene, "env_origins", None)
    if pose_command_local is not None and env_origins is not None:
        return pose_command_local + env_origins

    return robot_position(env)


def taxel_normal_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Per-taxel normal force (z component), shape (N, n_taxels).

    PTS sphere taxel site-local frame: z = normal (out of finger), xy = tangential.
    """
    stacked = _clip_taxel_force(_stack_taxel_force(env, sensor_names, entity_name))
    return stacked[..., 2]


def taxel_tangential_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Per-taxel tangential force (xy), flattened to (N, 2 * n_taxels)."""
    stacked = _clip_taxel_force(_stack_taxel_force(env, sensor_names, entity_name))
    tangential = stacked[..., :2]
    return tangential.reshape(tangential.shape[0], -1)


def taxel_contact_mask(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
    threshold: float = 0.05,
) -> torch.Tensor:
    """Return whether each taxel is active based on its 3D force norm."""
    stacked = _stack_taxel_force(env, sensor_names, entity_name)
    return torch.linalg.vector_norm(stacked, dim=-1) > threshold


def taxel_contact_count(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
    threshold: float = 0.05,
) -> torch.Tensor:
    """Count active taxels for each env."""
    return taxel_contact_mask(
        env,
        sensor_names=sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    ).sum(dim=1, dtype=torch.int64)


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
