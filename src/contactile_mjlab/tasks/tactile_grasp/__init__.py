"""Task registration and helpers for tactile grasp environments."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import (
    list_tasks,
    load_rl_cfg,
    load_runner_cls,
    register_mjlab_task,
)
from mjlab.tasks.registry import (
    load_env_cfg as _load_env_cfg,
)

from .constants import (
    DEFAULT_TASK_ID,
    PTS_SPHERES_TASK_ID,
    TACTILE_MODEL_PTS_SPHERES,
    TACTILE_MODEL_TOUCH_SITE,
    TOUCH_SITE_TASK_ID,
)
from .env_cfg import TactileGraspTaskConfig
from .rl_cfg import tactile_grasp_ppo_runner_cfg


def _registered_env_cfg(tactile_model: str, *, play: bool) -> TactileGraspTaskConfig:
    """Build the registered config for one tactile model."""
    return TactileGraspTaskConfig(
        tactile_model=tactile_model,
        num_envs=1 if play else 64,
        episode_length_s=6.0 if play else 3.0,
        auto_reset=True,
    )


def register_tasks() -> None:
    """Register the tactile grasp task variants exactly once."""
    registrations = (
        (TOUCH_SITE_TASK_ID, TACTILE_MODEL_TOUCH_SITE),
        (PTS_SPHERES_TASK_ID, TACTILE_MODEL_PTS_SPHERES),
    )
    for task_id, tactile_model in registrations:
        if task_id in list_tasks():
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=_registered_env_cfg(tactile_model, play=False).build(),
            play_env_cfg=_registered_env_cfg(tactile_model, play=True).build(),
            rl_cfg=tactile_grasp_ppo_runner_cfg(),
            runner_cls=None,
        )


def load_env_cfg(task_id: str = DEFAULT_TASK_ID, *, play: bool = False, **overrides):
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
        unknown = ", ".join(sorted(overrides))
        raise ValueError(f"Unsupported env config overrides: {unknown}")
    return cfg


def make_env(
    task_id: str = DEFAULT_TASK_ID,
    *,
    play: bool = False,
    device: str = "cpu",
    render_mode: str | None = None,
    **cfg_overrides,
) -> ManagerBasedRlEnv:
    """Instantiate one registered tactile grasp environment."""
    return ManagerBasedRlEnv(
        load_env_cfg(task_id, play=play, **cfg_overrides),
        device=device,
        render_mode=render_mode,
    )


register_tasks()

__all__ = [
    "DEFAULT_TASK_ID",
    "PTS_SPHERES_TASK_ID",
    "TOUCH_SITE_TASK_ID",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "make_env",
    "register_tasks",
]
