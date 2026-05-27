"""SceneCfg 加 plane terrain 之后，env_origins 必须形成 per-env 网格.

这是"多 env 夹爪不再重合"的必要条件 —— mjlab reset_scene_to_default 把
env_origins 写到 mocap pose；如果 env_origins 全零，所有 env 的夹爪都落在
世界原点.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from tactile_grasp import load_env_cfg


def test_env_origins_form_grid_for_multi_env():
    """多环境 scene 必须有互不相同的 xy env origins."""
    cfg = load_env_cfg(play=False)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cpu")

    origins = env.scene.env_origins
    assert origins.shape == (4, 3), origins.shape

    xy = origins[:, :2].cpu()
    unique_xy = torch.unique(xy, dim=0)
    assert unique_xy.shape[0] == 4, (
        f"expected 4 distinct env xy origins, got {unique_xy.shape[0]}; "
        f"origins=\n{origins.cpu().numpy()}"
    )
