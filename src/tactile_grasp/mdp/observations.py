"""Tactile observation helpers for contactile grasp tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .actions import RobotiqCommandAction

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


def taxel_force_map(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return the flattened per-taxel 3-axis force map."""
    return sensor_values(env, sensor_names=sensor_names, entity_name=entity_name)


def pad_wrench(
    env: "ManagerBasedRlEnv",
    force_sensor: str,
    torque_sensor: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Concatenate fingertip force and torque observations."""
    force = _sensor_tensor(env, f"{entity_name}/{force_sensor}")
    torque = _sensor_tensor(env, f"{entity_name}/{torque_sensor}")
    return torch.cat([force, torque], dim=-1)


def gripper_command(
    env: "ManagerBasedRlEnv",
    action_name: str = "gripper_command",
) -> torch.Tensor:
    """Expose the normalized Robotiq command buffer."""
    action_term = env.action_manager.get_term(action_name)
    assert isinstance(action_term, RobotiqCommandAction)
    return action_term.command / 255.0


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
