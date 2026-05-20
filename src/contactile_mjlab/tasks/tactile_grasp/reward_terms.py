"""Reward and termination helpers for tactile grasp tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg

from .constants import OBJECT_CFG
from .tactile_terms import gripper_command, sensor_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tactile_force_l2(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize the squared tactile magnitude across both fingertips."""
    left = sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.square(torch.cat([left, right], dim=-1)), dim=1)


def total_tactile_signal(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return the total absolute tactile signal across both fingertips."""
    left = sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.abs(torch.cat([left, right], dim=-1)), dim=1)


def alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Reward each environment that has not terminated."""
    return (~env.termination_manager.terminated).float()


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize the squared raw policy action."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def close_command_l2(
    env: "ManagerBasedRlEnv",
    action_name: str = "gripper_command",
) -> torch.Tensor:
    """Penalize unnecessary gripper closure."""
    command = gripper_command(env, action_name=action_name)
    return torch.sum(torch.square(command), dim=1)


def object_height(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """Return the object root height."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]


def object_height_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """Terminate when the object falls below a threshold."""
    return object_height(env, asset_cfg=asset_cfg) < minimum_height


class stable_grasp_hold:
    """Terminate after a sustained contact-and-hold window."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        """Allocate the per-environment hold counter."""
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset the hold counter for selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._counter[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        hold_steps: int,
        minimum_height: float,
        minimum_tactile_signal: float,
        left_sensor_names: tuple[str, ...],
        right_sensor_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = OBJECT_CFG,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        """Return a done mask for stable, sustained grasps."""
        height_ok = object_height(env, asset_cfg=asset_cfg) > minimum_height
        touch_ok = total_tactile_signal(
            env,
            left_sensor_names=left_sensor_names,
            right_sensor_names=right_sensor_names,
            entity_name=entity_name,
        ) > minimum_tactile_signal
        stable = height_ok & touch_ok
        self._counter = torch.where(stable, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
