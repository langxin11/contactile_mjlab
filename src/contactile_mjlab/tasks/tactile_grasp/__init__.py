"""tactile_grasp 任务注册与便利入口."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import (
    list_tasks,
    load_rl_cfg,
    load_runner_cls,
    register_mjlab_task,
)
from mjlab.tasks.registry import load_env_cfg as _load_env_cfg

from .constants import TASK_ID
from .env_cfg import TactileGraspTaskConfig
from .rl_cfg import tactile_grasp_ppo_runner_cfg


def _build(*, play: bool):
    return TactileGraspTaskConfig(
        num_envs=1 if play else 64,
        episode_length_s=6.0 if play else 3.0,
        auto_reset=True,
    ).build()


if TASK_ID not in list_tasks():
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=_build(play=False),
        play_env_cfg=_build(play=True),
        rl_cfg=tactile_grasp_ppo_runner_cfg(),
        runner_cls=None,
    )


def load_env_cfg(task_id: str = TASK_ID, *, play: bool = False, **overrides):
    """Load and optionally override a registered environment config."""
    cfg = _load_env_cfg(task_id, play=play)
    if "num_envs" in overrides:
        cfg.scene.num_envs = overrides.pop("num_envs")
    if "episode_length_s" in overrides:
        cfg.episode_length_s = overrides.pop("episode_length_s")
    if "auto_reset" in overrides:
        cfg.auto_reset = overrides.pop("auto_reset")
    if "env_spacing" in overrides:
        cfg.scene.env_spacing = overrides.pop("env_spacing")
    if overrides:
        raise ValueError(f"Unsupported overrides: {sorted(overrides)}")
    return cfg


def make_env(
    task_id: str = TASK_ID,
    *,
    play: bool = False,
    device: str = "cpu",
    render_mode: str | None = None,
    **cfg_overrides,
) -> ManagerBasedRlEnv:
    """Instantiate the registered tactile grasp environment."""
    return ManagerBasedRlEnv(
        load_env_cfg(task_id, play=play, **cfg_overrides),
        device=device,
        render_mode=render_mode,
    )


__all__ = ["TASK_ID", "load_env_cfg", "load_rl_cfg", "load_runner_cls", "make_env"]
