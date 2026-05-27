"""Top-down pick-lift task behavior tests."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from tactile_grasp import load_env_cfg
from tactile_grasp.constants import OBJECT_ENTITY_NAMES
from tactile_grasp.mdp import events, observations


def _make_env(num_envs: int = 4) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg, device="cpu")


def test_pick_lift_action_dim_is_cartesian_plus_gripper() -> None:
    """Policy action must be [dx, dy, dz, dyaw, du]."""
    env = _make_env(num_envs=2)

    assert env.action_manager.total_action_dim == 5
    assert "cartesian_gripper" in env.action_manager.active_terms


def test_cartesian_action_clips_to_workspace_bounds() -> None:
    """Mocap command integration must stay inside configured workspace bounds."""
    env = _make_env(num_envs=2)
    env.reset()
    action = env.action_manager.get_term("cartesian_gripper")

    big_action = torch.full((env.num_envs, 5), 10.0, device=env.device)
    for _ in range(64):
        action.process_actions(big_action)

    pos = action.pose_command_local
    assert torch.all(pos[:, 0] <= action.cfg.x_range[1])
    assert torch.all(pos[:, 1] <= action.cfg.y_range[1])
    assert torch.all(pos[:, 2] <= action.cfg.z_range[1])
    assert torch.all(action.yaw_command <= action.cfg.yaw_range[1])


def test_cartesian_action_step_writes_all_env_mocaps() -> None:
    """Stepping with all envs must write mocap pose with mjlab-compatible indexing."""
    env = _make_env(num_envs=3)
    env.reset()

    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    env.step(action)


def test_reset_activates_one_tabletop_object_per_env() -> None:
    """Reset must place exactly one object variant on the table in each env."""
    env = _make_env(num_envs=6)
    env.reset()

    active_ids = env._tactile_active_object_ids
    assert active_ids.shape == (env.num_envs,)
    assert torch.all((active_ids >= 0) & (active_ids < len(OBJECT_ENTITY_NAMES)))

    active_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    for object_id, object_name in enumerate(OBJECT_ENTITY_NAMES):
        entity = env.scene[object_name]
        local_pos = entity.data.root_link_pos_w - env.scene.env_origins
        is_active = active_ids == object_id
        active_count += is_active.long()

        if torch.any(is_active):
            assert torch.all(torch.abs(local_pos[is_active, 0]) <= 0.081)
            assert torch.all(torch.abs(local_pos[is_active, 1]) <= 0.081)
            assert torch.all(local_pos[is_active, 2] > 0.0)

        if torch.any(~is_active):
            inactive_xy = local_pos[~is_active, :2]
            assert torch.all(torch.linalg.norm(inactive_xy, dim=1) > 1.0)

    assert torch.all(active_count == 1)


def test_vision_proxy_reports_active_object_relative_pose_and_one_hot() -> None:
    """vision_proxy is the low-dimensional replacement point for future images."""
    env = _make_env(num_envs=4)
    env.reset()

    proxy = observations.vision_proxy(env)

    assert proxy.shape == (env.num_envs, 5 + len(OBJECT_ENTITY_NAMES))
    assert torch.all(torch.isfinite(proxy))
    one_hot = proxy[:, 5:]
    assert torch.allclose(one_hot.sum(dim=1), torch.ones(env.num_envs, device=env.device))
    assert torch.all(one_hot[torch.arange(env.num_envs), env._tactile_active_object_ids] == 1.0)


def test_curriculum_stage_thresholds() -> None:
    """Curriculum stage must follow the configured global step thresholds."""
    env = _make_env(num_envs=1)

    env.common_step_counter = 0
    assert events.pick_lift_curriculum(env, env_ids=None)["stage"] == 0

    env.common_step_counter = 20_000
    assert events.pick_lift_curriculum(env, env_ids=None)["stage"] == 1

    env.common_step_counter = 80_000
    assert events.pick_lift_curriculum(env, env_ids=None)["stage"] == 2
