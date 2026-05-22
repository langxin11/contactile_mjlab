"""Train a PPO baseline using mjlab's native RL stack."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_grasp import TASK_ID as DEFAULT_TASK_ID
from tactile_grasp import load_env_cfg, load_rl_cfg, load_runner_cls

try:
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
except ImportError as exc:  # pragma: no cover - runtime dependency gate
    raise ImportError(
        "Training requires the RL dependencies bundled with the installed mjlab package."
    ) from exc


def main() -> None:
    """Train a PPO baseline and save checkpoints with mjlab's runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=str, default=DEFAULT_TASK_ID)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-length-s", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/rsl_rl/tactile_grasp"))
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(args.task_id)
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_length_s
    env = ManagerBasedRlEnv(cfg, device=args.device)
    vec_env = RslRlVecEnvWrapper(env)
    runner_cfg = load_rl_cfg(args.task_id)
    if args.max_iterations is not None:
        runner_cfg.max_iterations = args.max_iterations
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_cls(
        vec_env,
        train_cfg=asdict(runner_cfg),
        log_dir=str(args.log_dir),
        device=args.device,
    )
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print(f"task_id={args.task_id} log_dir={args.log_dir}")


if __name__ == "__main__":
    main()
