"""测试 PTSSpheres 坐标系检查脚本的命令行接口."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_pts_frames.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import inspect_pts_frames  # noqa: E402


def test_help_does_not_expose_label_options() -> None:
    """帮助文本不应再暴露标签显示相关开关."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert "--show-labels" not in result.stdout
    assert "--no-show-labels" not in result.stdout
    assert "--label-prefix" not in result.stdout


def test_help_does_not_expose_static_run_mode() -> None:
    """帮助文本不应暴露静态运行模式入口."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert "--run-mode" not in result.stdout
    assert "static" not in result.stdout


def test_overlay_position_applies_scene_offset() -> None:
    """自定义 frame 位置应跟随 mjviser 的 scene offset."""
    site_position = np.array([1.0, 2.0, 3.0])
    scene_offset = np.array([-0.5, 0.25, 1.0])

    position = inspect_pts_frames._overlay_position(site_position, scene_offset)

    np.testing.assert_allclose(position, np.array([0.5, 2.25, 4.0]))
