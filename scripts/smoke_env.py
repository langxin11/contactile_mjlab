"""Run a short episode and summarize observation statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contactile_mjlab import DEFAULT_TASK_ID, PTS_SPHERES_TASK_ID, TOUCH_SITE_TASK_ID, make_env
from contactile_mjlab.tasks.tactile_grasp.constants import (
    TACTILE_ACTIVITY_THRESHOLD_BY_TASK,
    TACTILE_OBS_DIM_BY_TASK,
)


def _run_task(task_id: str, steps: int) -> None:
    """Run one short mjlab episode and report contact statistics."""
    env = make_env(
        task_id,
        episode_length_s=steps * 0.02,
        auto_reset=False,
    )
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    tactile_obs_dim = TACTILE_OBS_DIM_BY_TASK[task_id]
    tactile_threshold = TACTILE_ACTIVITY_THRESHOLD_BY_TASK[task_id]
    print(f"task_id={task_id}")
    print("reset")
    print(f"obs.shape={tuple(actor_obs.shape)} dtype={actor_obs.dtype}")
    print(f"obs.min={float(actor_obs.min()):.6f} obs.max={float(actor_obs.max()):.6f}")

    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    step = 0
    first_touch_step = None
    while not bool(torch.any(terminated | truncated)) and step < steps:
        action = torch.ones((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        observations, reward, terminated, truncated, _ = env.step(action)
        actor_obs = observations["actor"]
        touch_sum = float(torch.sum(torch.abs(actor_obs[0, :tactile_obs_dim])).cpu().item())
        if first_touch_step is None and touch_sum > tactile_threshold:
            first_touch_step = step + 1
        step += 1

    print("done")
    print(
        f"steps={step} terminated={terminated.cpu().tolist()} truncated={truncated.cpu().tolist()}"
    )
    print(f"final_reward={float(reward[0].cpu().item()):.6f}")
    print(f"first_touch_step={first_touch_step}")
    print(
        f"final_obs.shape={tuple(actor_obs.shape)} "
        f"finite={bool(np.isfinite(actor_obs.cpu().numpy()).all())}"
    )


def main() -> None:
    """Run short episodes and summarize observation statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        choices=[TOUCH_SITE_TASK_ID, PTS_SPHERES_TASK_ID],
        help="Run only the selected task id. Repeat to run multiple tasks.",
    )
    args = parser.parse_args()

    task_ids = args.task_ids or [TOUCH_SITE_TASK_ID, DEFAULT_TASK_ID]
    for task_id in task_ids:
        _run_task(task_id, args.steps)


if __name__ == "__main__":
    main()
