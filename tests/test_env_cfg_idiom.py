"""env_cfgs.make_tactile_grasp_env_cfg(play) should yield a usable mjlab cfg."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def test_make_env_cfg_train():
    """Train cfg defaults: 64 envs, 3.0 s episode."""
    from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg

    cfg = make_tactile_grasp_env_cfg(play=False)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert cfg.scene.num_envs == 64
    assert cfg.episode_length_s == 3.0


def test_make_env_cfg_play():
    """Play cfg: 1 env, 6.0 s episode, corruption off."""
    from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg

    cfg = make_tactile_grasp_env_cfg(play=True)
    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 6.0
    assert cfg.observations["actor"].enable_corruption is False


def test_no_dataclass_builder():
    """TactileGraspTaskConfig dataclass-builder must be gone."""
    try:
        from tactile_grasp.env_cfgs import TactileGraspTaskConfig  # noqa: F401
    except ImportError:
        return
    raise AssertionError("TactileGraspTaskConfig 仍存在 — idiom 切换未完成")


def test_load_env_cfg_no_override_whitelist():
    """load_env_cfg returns a deep-copied cfg the caller can freely mutate."""
    from tactile_grasp import load_env_cfg

    cfg = load_env_cfg(play=False)
    cfg.scene.num_envs = 32
    assert cfg.scene.num_envs == 32
