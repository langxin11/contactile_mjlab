"""触觉观测维度按 normal/tangential 拆分 + history_length 装配后正确."""

from __future__ import annotations

from tactile_grasp import make_env


def test_actor_obs_dim_with_history():
    """Single-frame 80 → 320 after history stacking on tactile + wrench groups."""
    env = make_env()
    obs, _ = env.reset()
    actor = obs["actor"]
    assert actor.ndim == 2
    # taxel_normal 两边 18 → 90 (×5)
    # taxel_tangential 两边 36 → 180 (×5)
    # pad_force 两边 6 → 18 (×3)
    # pad_torque 两边 6 → 18 (×3)
    # joint_pos+vel 12 + gripper_cmd 1 + last_action 1 = 14
    # 总和 90 + 180 + 18 + 18 + 12 + 1 + 1 = 320
    assert actor.shape == (env.num_envs, 320), actor.shape


def test_normal_force_dim():
    """taxel_normal_force returns (N, 9) — one z component per taxel."""
    from tactile_grasp.constants import LEFT_TAXEL_FORCE_SENSOR_NAMES
    from tactile_grasp.mdp import observations

    env = make_env()
    env.reset()
    out = observations.taxel_normal_force(env, sensor_names=LEFT_TAXEL_FORCE_SENSOR_NAMES)
    assert out.shape == (env.num_envs, 9), out.shape


def test_tangential_force_dim():
    """taxel_tangential_force returns (N, 18) — xy × 9 taxels."""
    from tactile_grasp.constants import LEFT_TAXEL_FORCE_SENSOR_NAMES
    from tactile_grasp.mdp import observations

    env = make_env()
    env.reset()
    out = observations.taxel_tangential_force(env, sensor_names=LEFT_TAXEL_FORCE_SENSOR_NAMES)
    assert out.shape == (env.num_envs, 18), out.shape
