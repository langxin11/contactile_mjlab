"""mdp/ 子包应包含五个标准子模块."""

from __future__ import annotations

import importlib


def test_mdp_submodules_exist():
    """All six mdp submodules must import cleanly."""
    for name in ("actions", "actuators", "observations", "rewards", "terminations", "events"):
        importlib.import_module(f"tactile_grasp.mdp.{name}")


def test_observations_exports():
    """Observations module must expose the canonical taxel/pad/gripper callables."""
    from tactile_grasp.mdp import observations

    for name in (
        "taxel_normal_force",
        "taxel_tangential_force",
        "pad_force",
        "pad_torque",
        "gripper_command",
    ):
        assert callable(getattr(observations, name)), name


def test_rewards_exports():
    """Rewards module must expose the current staged reward callables."""
    from tactile_grasp.mdp import rewards

    for name in (
        "staged_pickup",
        "taxel_coverage",
        "lift_delta",
        "hold_bonus",
        "action_smoothness_l1",
        "robot_floor_collision",
        "drop_penalty",
    ):
        assert callable(getattr(rewards, name)), name


def test_terminations_exports():
    """Terminations module must expose object_height_below and stable_grasp_hold."""
    from tactile_grasp.mdp import terminations

    assert callable(getattr(terminations, "object_height_below"))
    assert getattr(terminations, "stable_grasp_hold") is not None


def test_legacy_dirs_gone():
    """Old reward_terms / tactile_terms / _mdp_legacy modules must not import."""
    for name in (
        "tactile_grasp._mdp_legacy",
        "tactile_grasp.tactile_terms",
        "tactile_grasp.reward_terms",
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{name} 仍可 import — 拆分未完成")
