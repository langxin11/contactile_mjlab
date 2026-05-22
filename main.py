"""跑一段 episode 并打印观测统计."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tactile_grasp import TASK_ID, make_env


def main() -> None:
    """Run one short episode and print observation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    env = make_env(
        TASK_ID,
        episode_length_s=args.steps * 0.02,
        auto_reset=False,
    )
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    print(f"task_id={TASK_ID}")
    print(f"obs.shape={tuple(actor_obs.shape)} dtype={actor_obs.dtype}")
    print(f"obs.min={float(actor_obs.min()):.6f} obs.max={float(actor_obs.max()):.6f}")

    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    step = 0
    while not bool(torch.any(terminated | truncated)) and step < args.steps:
        action = torch.ones((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        observations, reward, terminated, truncated, _ = env.step(action)
        step += 1

    actor_obs = observations["actor"]
    print(
        f"steps={step} terminated={terminated.cpu().tolist()} truncated={truncated.cpu().tolist()}"
    )
    print(f"final_reward={float(reward[0].cpu().item()):.6f}")
    print(f"finite={bool(np.isfinite(actor_obs.cpu().numpy()).all())}")


if __name__ == "__main__":
    main()
