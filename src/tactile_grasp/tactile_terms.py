"""Tactile observation helpers for contactile grasp tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._mdp_legacy.actions import RobotiqCommandAction

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
