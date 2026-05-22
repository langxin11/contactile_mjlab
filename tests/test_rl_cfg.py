"""PPO default values pinning (spec §6.1)."""

from __future__ import annotations

from tactile_grasp.rl_cfg import tactile_grasp_ppo_runner_cfg


def test_defaults():
    """Verify upgraded baseline defaults."""
    cfg = tactile_grasp_ppo_runner_cfg()
    assert cfg.actor.hidden_dims == (256, 256)
    assert cfg.critic.hidden_dims == (256, 256)
    assert cfg.actor.obs_normalization is True
    assert cfg.critic.obs_normalization is True
    assert cfg.num_steps_per_env == 48
    assert cfg.max_iterations == 3000
    assert cfg.experiment_name == "tactile_grasp"
    assert cfg.logger == "wandb"
    assert cfg.wandb_project == "tactile_grasp"


def test_preserved_ppo_hparams():
    """Pin only the project-distinct PPO overrides; library defaults are out of scope."""
    cfg = tactile_grasp_ppo_runner_cfg()
    assert cfg.algorithm.learning_rate == 3.0e-4
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.save_interval == 50
