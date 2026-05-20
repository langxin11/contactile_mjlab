"""Run the tactile grasp training env with viewer for visual inspection."""

from __future__ import annotations

import argparse

import torch

from contactile_mjlab import DEFAULT_TASK_ID, make_env
from contactile_mjlab.tasks.tactile_grasp.constants import TACTILE_OBS_DIM_BY_TASK


def main() -> None:
    """Run the selected tactile grasp task in the interactive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=str, default=DEFAULT_TASK_ID)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env = make_env(
        args.task_id,
        play=True,
        num_envs=1,
        episode_length_s=6.0,
        auto_reset=True,
        device=args.device,
        render_mode="human",
    )
    tactile_obs_dim = TACTILE_OBS_DIM_BY_TASK[args.task_id]

    obs, _ = env.reset()
    step = 0
    while True:
        action = 0.3 * torch.randn(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1

        touch = obs["actor"][:, :tactile_obs_dim].abs().sum().item()
        cmd = obs["actor"][:, tactile_obs_dim + 12].item()

        if step % 50 == 0 or terminated.any() or truncated.any():
            status = ""
            if terminated.any():
                status += " [TERM]"
            if truncated.any():
                status += " [TRUNC]"
            print(
                f"step={step:4d}  touch_sum={touch:.4f}  cmd={cmd:.3f}"
                f"  reward={reward[0].item():+.4f}{status}"
            )


if __name__ == "__main__":
    main()
