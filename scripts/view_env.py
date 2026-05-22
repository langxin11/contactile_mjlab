"""带 viewer 跑 tactile grasp env 做可视化检查."""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from tactile_grasp import TASK_ID, load_env_cfg


def main() -> None:
    """Run the tactile grasp env with a viewer for visual inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.auto_reset = True
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="human")

    env.reset()
    step = 0
    while True:
        action = 0.3 * torch.randn(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        if step % 50 == 0 or terminated.any() or truncated.any():
            status = " [TERM]" if terminated.any() else (" [TRUNC]" if truncated.any() else "")
            print(f"step={step:4d}  reward={reward[0].item():+.4f}{status}")


if __name__ == "__main__":
    main()
