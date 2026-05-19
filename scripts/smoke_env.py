"""Run a short episode and summarize observation statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contactile_mjlab import TactileGraspEnv, TactileGraspEnvConfig


def main() -> None:
    """Run a short mjlab episode and report contact and observation statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    env = TactileGraspEnv(
        TactileGraspEnvConfig(
            episode_length_s=args.steps * 0.02,
            auto_reset=False,
        )
    )
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    print("reset")
    print(f"obs.shape={tuple(actor_obs.shape)} dtype={actor_obs.dtype}")
    print(f"obs.min={float(actor_obs.min()):.6f} obs.max={float(actor_obs.max()):.6f}")

    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    step = 0
    first_touch_step = None
    while not bool(torch.any(terminated | truncated)) and step < args.steps:
        action = torch.ones((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        observations, reward, terminated, truncated, _ = env.step(action)
        actor_obs = observations["actor"]
        touch_sum = float(torch.sum(actor_obs[:, :18]).cpu().item())
        if first_touch_step is None and touch_sum > 0.0:
            first_touch_step = step + 1
        step += 1

    print("done")
    print(
        f"steps={step} terminated={terminated.cpu().tolist()} "
        f"truncated={truncated.cpu().tolist()}"
    )
    print(f"final_reward={float(reward[0].cpu().item()):.6f}")
    print(f"first_touch_step={first_touch_step}")
    print(
        f"final_obs.shape={tuple(actor_obs.shape)} "
        f"finite={bool(np.isfinite(actor_obs.cpu().numpy()).all())}"
    )


if __name__ == "__main__":
    main()
