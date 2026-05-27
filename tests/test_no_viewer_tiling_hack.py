"""viewer_tiling monkey-patch 已退役（被 SceneCfg.terrain=plane 替代）.

历史：在没有 terrain 时多 env 的夹爪都会重合在世界原点，曾用
src/tactile_grasp/viewer_tiling.py monkey-patch MjlabViserScene 加视觉偏移
绕过。现在改用 mjlab idiomatic 的 plane terrain，三件套必须删干净.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_play_tiled_script_removed():
    """scripts/play_tiled.py 不应再存在."""
    assert not (REPO_ROOT / "scripts" / "play_tiled.py").exists(), (
        "scripts/play_tiled.py 应已删除 —— idiomatic 路径用 plane terrain"
    )


def test_viewer_tiling_module_removed():
    """viewer_tiling 模块文件不应再存在."""
    assert not (REPO_ROOT / "src" / "tactile_grasp" / "viewer_tiling.py").exists()


def test_viewer_tiling_not_importable():
    """tactile_grasp.viewer_tiling 不应再可 import."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tactile_grasp.viewer_tiling")


def test_old_viewer_tiling_test_removed():
    """旧 viewer_tiling 测试文件不应再存在."""
    assert not (REPO_ROOT / "tests" / "test_viewer_tiling.py").exists()
