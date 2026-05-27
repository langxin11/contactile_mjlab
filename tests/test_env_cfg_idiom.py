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
    import tactile_grasp.env_cfgs as env_cfgs

    assert not hasattr(env_cfgs, "TactileGraspTaskConfig"), (
        "TactileGraspTaskConfig 仍存在 — idiom 切换未完成"
    )


def test_load_env_cfg_no_override_whitelist():
    """load_env_cfg returns a deep-copied cfg the caller can freely mutate."""
    from tactile_grasp import load_env_cfg

    cfg = load_env_cfg(play=False)
    cfg.scene.num_envs = 32
    assert cfg.scene.num_envs == 32


def test_scene_uses_plane_terrain_for_per_env_origins():
    """SceneCfg 必须带 plane terrain，否则 env_origins 全零、多 env 夹爪在 viewer 中会重合.

    背景：mjlab 自动把 fixed-base 夹爪包成 mocap，reset_scene_to_default 会把
    env_origins 写到 mocap pose；只有 terrain != None 时 env_origins 才是网格.
    """
    from mjlab.terrains import TerrainEntityCfg

    from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg

    cfg = make_tactile_grasp_env_cfg(play=False)
    assert cfg.scene.terrain is not None, "SceneCfg.terrain is None — 多 env 会全部重合"
    assert isinstance(cfg.scene.terrain, TerrainEntityCfg)
    assert cfg.scene.terrain.terrain_type == "plane"
