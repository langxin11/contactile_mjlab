"""新包 layout 与命名验证."""

from __future__ import annotations


def test_top_level_import():
    """顶层 tactile_grasp 包应可 import 并暴露 TASK_ID/make_env."""
    import tactile_grasp

    assert hasattr(tactile_grasp, "TASK_ID")
    assert hasattr(tactile_grasp, "make_env")


def test_task_id_value():
    """TASK_ID 应与重命名后的常量值一致."""
    import tactile_grasp

    assert tactile_grasp.TASK_ID == "Mjlab-TactileGrasp-Robotiq2F85"


def test_no_old_package():
    """旧包名 contactile_mjlab 应已无法 import."""
    import importlib

    try:
        importlib.import_module("contactile_mjlab")
    except ModuleNotFoundError:
        return
    raise AssertionError("contactile_mjlab 仍可 import — 重命名未完成")


def test_no_subdir_tasks():
    """tactile_grasp.tasks 子包应已不存在."""
    import importlib

    try:
        importlib.import_module("tactile_grasp.tasks")
    except ModuleNotFoundError:
        return
    raise AssertionError("tactile_grasp.tasks 仍存在 — 扁平化未完成")
