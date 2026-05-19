"""Observation, reward, and termination terms for the tactile grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .action_terms import RobotiqCommandAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_ROBOT_CFG = SceneEntityCfg(
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


def _sensor_tensor(env: "ManagerBasedRlEnv", name: str) -> torch.Tensor:
    """Read one XML-defined builtin sensor by prefixed name."""
    return env.scene[name].data


def touch_map(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Stack scalar touch sensors into a flattened tactile map."""
    values = [_sensor_tensor(env, f"{entity_name}/{name}") for name in sensor_names]
    return torch.cat(values, dim=-1)


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


def total_touch_force(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Sum tactile magnitudes across both fingertips."""
    left = touch_map(env, left_sensor_names, entity_name=entity_name)
    right = touch_map(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(left + right, dim=1)


def touch_force_l2(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize the squared tactile force magnitude."""
    left = touch_map(env, left_sensor_names, entity_name=entity_name)
    right = touch_map(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.square(torch.cat([left, right], dim=-1)), dim=1)


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
    asset_cfg: SceneEntityCfg = _OBJECT_CFG,
) -> torch.Tensor:
    """Return the object root height."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]


def object_height_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _OBJECT_CFG,
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
        left_sensor_names: tuple[str, ...],
        right_sensor_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = _OBJECT_CFG,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        """Return a done mask for stable, sustained grasps."""
        height_ok = object_height(env, asset_cfg=asset_cfg) > minimum_height
        touch_ok = total_touch_force(
            env,
            left_sensor_names=left_sensor_names,
            right_sensor_names=right_sensor_names,
            entity_name=entity_name,
        ) > 0.0
        stable = height_ok & touch_ok
        self._counter = torch.where(stable, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
