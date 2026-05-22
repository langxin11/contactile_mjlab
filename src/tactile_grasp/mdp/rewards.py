"""tactile_grasp 奖励函数."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import observations as obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tactile_force_l2(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """惩罚双指尖触觉幅值平方和."""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.square(torch.cat([left, right], dim=-1)), dim=1)


def total_tactile_signal(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """双指尖触觉绝对值之和（被 terminations 复用）."""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.abs(torch.cat([left, right], dim=-1)), dim=1)


def alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """未终止的环境给 +1."""
    return (~env.termination_manager.terminated).float()


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """惩罚动作幅度平方."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def close_command_l2(
    env: "ManagerBasedRlEnv",
    action_name: str = "gripper_command",
) -> torch.Tensor:
    """惩罚多余的夹爪闭合命令."""
    command = obs.gripper_command(env, action_name=action_name)
    return torch.sum(torch.square(command), dim=1)


def drop_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """物体掉落 termination 命中时的负奖励通道."""
    return env.termination_manager.get_term("object_drop").float()
