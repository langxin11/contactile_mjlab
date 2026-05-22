"""Control helpers for the Robotiq 2F-85 actuator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GripperCommandBuffer:
    """Integrates normalized scalar actions into the Robotiq control range."""

    init_u: float = 0.0
    delta_u_max: float = 3.0
    u_min: float = 0.0
    u_max: float = 255.0

    def __post_init__(self) -> None:
        """Initialize the internal command state."""
        self.u = float(np.clip(self.init_u, self.u_min, self.u_max))

    def reset(self, init_u: float | None = None) -> float:
        """Reset the command buffer to the configured or provided value."""
        if init_u is None:
            init_u = self.init_u
        self.u = float(np.clip(init_u, self.u_min, self.u_max))
        return self.u

    def step(self, action: float) -> float:
        """Advance the command buffer by one normalized action step."""
        clipped = float(np.clip(action, -1.0, 1.0))
        self.u = float(np.clip(self.u + clipped * self.delta_u_max, self.u_min, self.u_max))
        return self.u
