"""tactile_grasp: Robotiq 2F-85 + PTS spheres 触觉抓取任务包."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import (
    list_tasks,
    load_rl_cfg,
    load_runner_cls,
    register_mjlab_task,
)
from mjlab.tasks.registry import load_env_cfg as _load_env_cfg

from .constants import TASK_ID
from .env_cfgs import make_tactile_grasp_env_cfg
from .rl_cfg import tactile_grasp_ppo_runner_cfg

if TASK_ID not in list_tasks():
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_tactile_grasp_env_cfg(play=False),
        play_env_cfg=make_tactile_grasp_env_cfg(play=True),
        rl_cfg=tactile_grasp_ppo_runner_cfg(),
        runner_cls=None,
    )


def load_env_cfg(task_id: str = TASK_ID, *, play: bool = False):
    """Return a fresh deep-copied env cfg; caller mutates fields directly."""
    return deepcopy(_load_env_cfg(task_id, play=play))


def make_env(
    *,
    play: bool = False,
    device: str = "cpu",
    render_mode: str | None = None,
) -> ManagerBasedRlEnv:
    """Instantiate the registered env with no field overrides."""
    return ManagerBasedRlEnv(
        load_env_cfg(TASK_ID, play=play),
        device=device,
        render_mode=render_mode,
    )


__all__ = ["TASK_ID", "load_env_cfg", "load_rl_cfg", "load_runner_cls", "make_env"]
