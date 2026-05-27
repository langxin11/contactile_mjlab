"""tactile_grasp 终止条件."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

from ..constants import OBJECT_CFG
from . import observations as obs
from . import rewards as rew

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def object_height(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """Active object root link 的 z 坐标."""
    del asset_cfg
    return obs.active_object_position(env)[:, 2]


def object_height_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """物体低于阈值时终止."""
    return object_height(env, asset_cfg=asset_cfg) < minimum_height


def robot_out_of_workspace(env: "ManagerBasedRlEnv", margin: float = 1.0e-4) -> torch.Tensor:
    """夹爪 mocap 命令越过 workspace 时终止（正常 clipping 下应为 False）."""
    action = env.action_manager.get_term("cartesian_gripper")
    pos = action.pose_command_local
    cfg = action.cfg
    out = (
        (pos[:, 0] < cfg.x_range[0] - margin)
        | (pos[:, 0] > cfg.x_range[1] + margin)
        | (pos[:, 1] < cfg.y_range[0] - margin)
        | (pos[:, 1] > cfg.y_range[1] + margin)
        | (pos[:, 2] < cfg.z_range[0] - margin)
        | (pos[:, 2] > cfg.z_range[1] + margin)
    )
    return out


class stable_grasp_hold:
    """持续 contact-and-hold 的终止判定（带 per-env 计数器）."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        """Initialize per-env hold counters."""
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset counters for the given env ids (or all if None)."""
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
        """Return termination mask when stable grasp has been held for hold_steps."""
        height_ok = object_height(env, asset_cfg=asset_cfg) > minimum_height
        touch_ok = (
            rew.total_tactile_signal(
                env,
                left_sensor_names=left_sensor_names,
                right_sensor_names=right_sensor_names,
                entity_name=entity_name,
            )
            > minimum_tactile_signal
        )
        stable = height_ok & touch_ok
        self._counter = torch.where(stable, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
