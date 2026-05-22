"""Custom mjlab action terms for the tactile gripper task."""

from __future__ import annotations

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
