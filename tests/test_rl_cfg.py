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
    """Preserved hyperparameters should remain unchanged."""
    cfg = tactile_grasp_ppo_runner_cfg()
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.desired_kl == 0.01
    assert cfg.algorithm.learning_rate == 3.0e-4
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.value_loss_coef == 1.0
    assert cfg.algorithm.use_clipped_value_loss is True
    assert cfg.algorithm.num_learning_epochs == 5
    assert cfg.algorithm.num_mini_batches == 4
    assert cfg.algorithm.max_grad_norm == 1.0
    assert cfg.save_interval == 50
