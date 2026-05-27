"""Custom mjlab action terms for the tactile gripper task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class RobotiqCommandActionCfg(ActionTermCfg):
    """Integrate scalar actions into Robotiq-style actuator commands."""

    actuator_name: str
    tendon_name: str
    delta_u_max: float = 3.0
    init_u: float = 0.0
    u_min: float = 0.0
    u_max: float = 255.0

    def build(self, env: ManagerBasedRlEnv) -> "RobotiqCommandAction":
        """Build the action term."""
        return RobotiqCommandAction(self, env)


class RobotiqCommandAction(ActionTerm):
    """Action term that writes XML actuator ctrl values directly."""

    cfg: RobotiqCommandActionCfg

    def __init__(self, cfg: RobotiqCommandActionCfg, env: ManagerBasedRlEnv) -> None:
        """Cache actuator indices and command buffers."""
        super().__init__(cfg=cfg, env=env)
        tendon_ids, tendon_names = self._entity.find_tendons((cfg.tendon_name,))
        self._tendon_ids = torch.tensor(tendon_ids, device=self.device, dtype=torch.long)
        self._tendon_names = tendon_names
        self._raw_action = torch.zeros((self.num_envs, 1), device=self.device)
        self._command = torch.full(
            (self.num_envs, 1), float(cfg.init_u), device=self.device, dtype=torch.float32
        )

    @property
    def action_dim(self) -> int:
        """Return the scalar action dimension."""
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        """Return the latest unclipped policy action."""
        return self._raw_action

    @property
    def command(self) -> torch.Tensor:
        """Return the current Robotiq command buffer."""
        return self._command

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset actions and command state."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_action[env_ids] = 0.0
        self._command[env_ids] = float(self.cfg.init_u)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip incoming actions to the normalized range."""
        self._raw_action[:] = torch.clamp(actions, -1.0, 1.0)
        delta = self._raw_action * float(self.cfg.delta_u_max)
        self._command[:] = torch.clamp(
            self._command + delta,
            min=float(self.cfg.u_min),
            max=float(self.cfg.u_max),
        )

    def apply_actions(self) -> None:
        """Write the latest Robotiq command into the actuator target buffer."""
        self._entity.set_tendon_len_target(self._command, tendon_ids=self._tendon_ids)


@dataclass(kw_only=True)
class CartesianMocapActionCfg(ActionTermCfg):
    """Integrate Cartesian mocap deltas plus Robotiq-style gripper command."""

    actuator_name: str
    tendon_name: str
    pos_step: float = 0.01
    yaw_step: float = 0.05
    delta_u_max: float = 3.0
    init_u: float = 0.0
    u_min: float = 0.0
    u_max: float = 255.0
    x_range: tuple[float, float] = (-0.12, 0.12)
    y_range: tuple[float, float] = (-0.12, 0.12)
    z_range: tuple[float, float] = (0.04, 0.30)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.24)
    default_yaw: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "CartesianMocapAction":
        """Build the action term."""
        return CartesianMocapAction(self, env)


class CartesianMocapAction(ActionTerm):
    """Action term for world-frame mocap pose deltas and Robotiq command."""

    cfg: CartesianMocapActionCfg

    def __init__(self, cfg: CartesianMocapActionCfg, env: ManagerBasedRlEnv) -> None:
        """Cache actuator indices and Cartesian command buffers."""
        super().__init__(cfg=cfg, env=env)
        self._env = env
        tendon_ids, tendon_names = self._entity.find_tendons((cfg.tendon_name,))
        self._tendon_ids = torch.tensor(tendon_ids, device=self.device, dtype=torch.long)
        self._tendon_names = tendon_names
        self._raw_action = torch.zeros((self.num_envs, 5), device=self.device)
        self._command = torch.full(
            (self.num_envs, 1), float(cfg.init_u), device=self.device, dtype=torch.float32
        )
        self._pose_command_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._yaw_command = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        """Return [dx, dy, dz, dyaw, du] action dimension."""
        return 5

    @property
    def raw_action(self) -> torch.Tensor:
        """Return the latest unclipped policy action."""
        return self._raw_action

    @property
    def command(self) -> torch.Tensor:
        """Return the current Robotiq command buffer."""
        return self._command

    @property
    def pose_command_local(self) -> torch.Tensor:
        """Return mocap command position relative to each env origin."""
        return self._pose_command_local

    @property
    def yaw_command(self) -> torch.Tensor:
        """Return yaw command around world z."""
        return self._yaw_command

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset Cartesian command, yaw, and Robotiq command state."""
        env_ids = self._resolve_env_ids(env_ids)
        self._raw_action[env_ids] = 0.0
        self._command[env_ids] = float(self.cfg.init_u)

        default_pos = torch.tensor(self.cfg.default_pos, device=self.device, dtype=torch.float32)
        init_pos = getattr(self._env, "_tactile_robot_init_pos_local", None)
        init_yaw = getattr(self._env, "_tactile_robot_init_yaw", None)
        if init_pos is None:
            self._pose_command_local[env_ids] = default_pos
        else:
            self._pose_command_local[env_ids] = init_pos[env_ids]
        if init_yaw is None:
            self._yaw_command[env_ids] = float(self.cfg.default_yaw)
        else:
            self._yaw_command[env_ids] = init_yaw[env_ids]
        self._write_mocap_pose(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip and integrate Cartesian and gripper action commands."""
        self._raw_action[:] = torch.clamp(actions, -1.0, 1.0)
        pos_delta = self._raw_action[:, 0:3] * float(self.cfg.pos_step)
        self._pose_command_local[:] = self._pose_command_local + pos_delta
        self._pose_command_local[:, 0] = torch.clamp(
            self._pose_command_local[:, 0], self.cfg.x_range[0], self.cfg.x_range[1]
        )
        self._pose_command_local[:, 1] = torch.clamp(
            self._pose_command_local[:, 1], self.cfg.y_range[0], self.cfg.y_range[1]
        )
        self._pose_command_local[:, 2] = torch.clamp(
            self._pose_command_local[:, 2], self.cfg.z_range[0], self.cfg.z_range[1]
        )
        self._yaw_command[:] = torch.clamp(
            self._yaw_command + self._raw_action[:, 3] * float(self.cfg.yaw_step),
            self.cfg.yaw_range[0],
            self.cfg.yaw_range[1],
        )
        delta = self._raw_action[:, 4:5] * float(self.cfg.delta_u_max)
        self._command[:] = torch.clamp(
            self._command + delta,
            min=float(self.cfg.u_min),
            max=float(self.cfg.u_max),
        )

    def apply_actions(self) -> None:
        """Write mocap pose and Robotiq tendon target into the simulation buffer."""
        self._write_mocap_pose()
        self._entity.set_tendon_len_target(self._command, tendon_ids=self._tendon_ids)

    def set_pose_command(
        self,
        pos_local: torch.Tensor,
        yaw: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set the Cartesian command buffer from a reset event."""
        env_ids = self._resolve_env_ids(env_ids)
        self._pose_command_local[env_ids] = pos_local
        self._yaw_command[env_ids] = yaw
        self._write_mocap_pose(env_ids)

    def _write_mocap_pose(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        pos_local = self._pose_command_local[env_ids]
        yaw = self._yaw_command[env_ids]
        origins = self._env.scene.env_origins[env_ids]
        pose = torch.zeros((pos_local.shape[0], 7), device=self.device, dtype=torch.float32)
        pose[:, 0:3] = pos_local + origins
        pose[:, 3:7] = _top_down_yaw_quat(yaw)
        self._entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return env_ids


def _top_down_yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Build quaternion for yaw around world z followed by 180 degrees about x."""
    half = yaw * 0.5
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    quat[:, 1] = torch.cos(half)
    quat[:, 2] = torch.sin(half)
    return quat
