"""极简 smoke：跑一步 step 验证包可用."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile_grasp import make_env


def main() -> None:
    """Reset and step the env once to confirm the package is importable."""
    env = make_env()
    observations, _ = env.reset()
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    observations, reward, terminated, truncated, _ = env.step(action)
    print(
        "tactile-grasp ready: "
        f"actor_obs_shape={tuple(observations['actor'].shape)} "
        f"reward_shape={tuple(reward.shape)} "
        f"terminated={terminated.cpu().tolist()} truncated={truncated.cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
