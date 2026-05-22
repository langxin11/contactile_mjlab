"""Exercise the Robotiq command buffer and actuator direction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile_grasp import make_env


def main() -> None:
    """Drive the mjlab action term and print command/joint motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--action", type=float, default=1.0)
    args = parser.parse_args()

    env = make_env()
    env.reset()
    robot = env.scene["robot"]
    gripper = env.action_manager.get_term("gripper_command")

    print("step action u driver_joints")
    for step in range(args.steps):
        action = torch.full(
            (env.num_envs, env.action_manager.total_action_dim),
            float(args.action),
            device=env.device,
        )
        env.step(action)
        joints = robot.data.joint_pos[0, :2].cpu().numpy()
        command = float(gripper.command[0, 0].cpu().item())
        print(f"{step:02d} {args.action:+.2f} {command:7.3f} {joints}")


if __name__ == "__main__":
    main()
