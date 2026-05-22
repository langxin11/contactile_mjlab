# mjlab Idiom Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `contactile_mjlab` 包重构为 `tactile_grasp`，对齐 mjlab 上游惯例（`make_xxx_env_cfg` + `mdp/` 五拆分），丢弃 TouchSite 代码路径，加触觉时间窗口，把训练入口切到 `mjlab.scripts.train`。

**Architecture:** 9 个独立 commit，按"先收拢资产 → 删死代码 → 删 TouchSite → 重命名扁平化 → mdp/ 拆分 → 观测重塑 → env_cfgs 换 idiom → rl_cfg 调默认值 → 极简 train/play 入口"顺序推进。每个 commit 之间 import smoke + pytest 都必须绿，并保留可独立回滚的能力。

**Tech Stack:** Python 3.12, mjlab 1.3+, MuJoCo 3.3+, rsl_rl, PyTorch, pytest, uv（包/环境管理）

**Spec:** `docs/superpowers/specs/2026-05-22-mjlab-idiom-refactor-design.md`

---

## 工作流约定

- **所有命令走 uv**：`uv run python ...`、`uv run pytest ...`、`uv sync` / `uv pip install`
- **每个 commit 前**：
  1. `uv run pytest tests/ -q` 通过
  2. `uv run python scripts/smoke_env.py` 通过（重命名 commit 之前）或新等效命令通过（重命名之后）
- **每个 commit 后**：在新分支上 `git status` 应为 clean
- **回滚条件**：任何"验证"步骤失败 → `git restore .` 回到 commit 前状态，重新尝试

---

## Task 1: 资产迁入包内

**目标：** `assets/` 从 repo 根迁到 `src/contactile_mjlab/assets/`，`paths.py` 改用包内相对路径，`pyproject.toml` 加 `package-data`。包名暂不变。

**Files:**
- Move: `assets/` → `src/contactile_mjlab/assets/`
- Modify: `src/contactile_mjlab/paths.py` (整个文件重写)
- Modify: `pyproject.toml` (加 `[tool.setuptools.package-data]`)
- Test: `tests/test_asset_paths.py` (new)

- [ ] **Step 1.1: 创建资产路径冒烟测试**

新建 `tests/test_asset_paths.py`：

```python
"""所有 paths.py 中常量指向的 XML 文件必须存在且可被 MuJoCo 加载。"""
from __future__ import annotations

import mujoco
import pytest

from contactile_mjlab import paths


@pytest.mark.parametrize(
    "xml_path",
    [
        paths.BASE_XML,
        paths.TACTILE_XML,
        paths.TACTILE_SCENE_XML,
        paths.PTS_SPHERES_XML,
        paths.PTS_SPHERES_SCENE_XML,
        paths.HANGING_BOX_XML,
    ],
)
def test_xml_loadable(xml_path):
    assert xml_path.is_file(), f"missing: {xml_path}"
    mujoco.MjSpec.from_file(str(xml_path))
```

注意：`paths.HANGING_BOX_XML` 当前不存在（当前用 `PROPS_DIR / "hanging_box.xml"` 拼），本 task 会补充该常量。

- [ ] **Step 1.2: 跑测试确认会失败**

```bash
uv run pytest tests/test_asset_paths.py -v
```

Expected: FAIL — `paths.HANGING_BOX_XML` AttributeError（常量尚未定义）

- [ ] **Step 1.3: 物理迁移 `assets/` 目录到包内**

```bash
git mv assets src/contactile_mjlab/assets
```

验证目录结构：

```bash
ls src/contactile_mjlab/assets/robotiq_2f85/ && ls src/contactile_mjlab/assets/props/
```

Expected: 列出 `2f85_pts_spheres.xml` 等 XML 和 `assets/` 子目录 + `hanging_box.xml`

- [ ] **Step 1.4: 重写 `paths.py`**

替换 `src/contactile_mjlab/paths.py` 全部内容：

```python
"""项目内资产路径常量。"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
ROBOTIQ_DIR = ASSETS_DIR / "robotiq_2f85"
PROPS_DIR = ASSETS_DIR / "props"

# 主线模型
PTS_SPHERES_XML = ROBOTIQ_DIR / "2f85_pts_spheres.xml"
PTS_SPHERES_SCENE_XML = ROBOTIQ_DIR / "scene_pts_spheres.xml"
HANGING_BOX_XML = PROPS_DIR / "hanging_box.xml"

# 保留以备未来调试 / V0 对照
BASE_XML = ROBOTIQ_DIR / "2f85.xml"
TACTILE_XML = ROBOTIQ_DIR / "2f85_tactile.xml"
TACTILE_SCENE_XML = ROBOTIQ_DIR / "scene_tactile.xml"
```

`PROJECT_ROOT` 常量被删除——任何引用它的代码（仅 `mjlab/tactile_grasp_env.py:41,68`，属死代码）会在 Task 2 一并清理。

- [ ] **Step 1.5: 在 `pyproject.toml` 加 package-data**

在现有 `[tool.setuptools.packages.find]` 段下方追加：

```toml
[tool.setuptools.package-data]
contactile_mjlab = [
    "assets/**/*.xml",
    "assets/**/*.stl",
    "assets/**/*.obj",
    "assets/**/*.png",
    "assets/**/*.mtl",
]
```

- [ ] **Step 1.6: 同步依赖**

```bash
uv sync --extra cu128 --group dev
```

Expected: 无报错；`contactile_mjlab` 重新装为可编辑包，资产 package-data 被打包。

- [ ] **Step 1.7: 跑资产路径测试**

```bash
uv run pytest tests/test_asset_paths.py -v
```

Expected: 6 个测试全部 PASS。

- [ ] **Step 1.8: 跑全套 smoke**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
```

Expected: pytest 全绿；smoke 输出 `contactile-mjlab ready: ...`。

- [ ] **Step 1.9: Commit**

```bash
git add src/contactile_mjlab/assets src/contactile_mjlab/paths.py pyproject.toml tests/test_asset_paths.py
git status  # 确认旧 assets/ 已通过 git mv 标记为重命名
git commit -m "chore: 资产从 repo 根迁入 src/contactile_mjlab/assets"
```

**回滚条件：** smoke_env.py 失败 → 极可能是某个 XML 内部相对路径解析坏；检查 `meshdir="assets"` 是否仍指向 `src/contactile_mjlab/assets/robotiq_2f85/assets/`（确实如此，无需改）。如仍坏，`git restore .`。

---

## Task 2: 删除死代码

**目标：** 删 `src/contactile_mjlab/mjlab/tactile_grasp_env.py`（325 行死代码）与 `mjlab/mdp.py`（165 行死代码）。`action_terms.py` / `actuators.py` 保留（被 robot_cfg.py 引用，alive）。

**Files:**
- Delete: `src/contactile_mjlab/mjlab/tactile_grasp_env.py`
- Delete: `src/contactile_mjlab/mjlab/mdp.py`
- Modify: `src/contactile_mjlab/mjlab/__init__.py`

- [ ] **Step 2.1: 确认确为死代码**

```bash
grep -rn 'TactileGraspEnv\|tactile_grasp_env\|from .mjlab.mdp\|from ..mjlab.mdp' src/ scripts/ tests/ main.py
```

Expected: 命中仅在 `src/contactile_mjlab/mjlab/__init__.py:3` 和 `src/contactile_mjlab/mjlab/tactile_grasp_env.py` 自身。无外部使用者。

- [ ] **Step 2.2: 删两个文件**

```bash
rm src/contactile_mjlab/mjlab/tactile_grasp_env.py
rm src/contactile_mjlab/mjlab/mdp.py
```

- [ ] **Step 2.3: 改 `src/contactile_mjlab/mjlab/__init__.py`**

替换全部内容为：

```python
"""mjlab-native action / actuator wrappers."""

from .action_terms import RobotiqCommandAction, RobotiqCommandActionCfg
from .actuators import RobotiqGeneralActuatorCfg

__all__ = [
    "RobotiqCommandAction",
    "RobotiqCommandActionCfg",
    "RobotiqGeneralActuatorCfg",
]
```

- [ ] **Step 2.4: 验证无破口**

```bash
uv run python -c "import contactile_mjlab; print(contactile_mjlab.DEFAULT_TASK_ID)"
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
```

Expected: 全部成功；smoke 输出 obs shape 不变。

- [ ] **Step 2.5: Commit**

```bash
git add -A src/contactile_mjlab/mjlab
git commit -m "refactor: 删除 mjlab/tactile_grasp_env.py 与 mjlab/mdp.py 死代码"
```

**回滚条件：** import 报 `ModuleNotFoundError` → 检查是否漏了某个再 export；`git restore .`。

---

## Task 3: 删除 TouchSite 代码路径

**目标：** 拆掉 `tactile_model == TOUCH_SITE` 分支，删 `TOUCH_SITE_TASK_ID` / `LEFT_TOUCH_SENSOR_NAMES` / `touch_map` / 查表 dict。XML 资产保留（spec §7.2）。

**Files:**
- Modify: `src/contactile_mjlab/tasks/tactile_grasp/constants.py`
- Modify: `src/contactile_mjlab/tasks/tactile_grasp/__init__.py`
- Modify: `src/contactile_mjlab/tasks/tactile_grasp/env_cfg.py`
- Modify: `src/contactile_mjlab/tasks/tactile_grasp/robot_cfg.py`
- Modify: `src/contactile_mjlab/tasks/tactile_grasp/tactile_terms.py`
- Modify: `src/contactile_mjlab/__init__.py`
- Modify: `main.py`
- Modify: `scripts/smoke_env.py`、`scripts/view_env.py`
- Test: `tests/test_no_touchsite.py` (new)

- [ ] **Step 3.1: 写"无 TouchSite 残留"测试**

新建 `tests/test_no_touchsite.py`：

```python
"""TouchSite 代码路径应已删除（资产保留）。"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_touch_site_symbols_in_code():
    """除 paths.py 中 TACTILE_XML/TACTILE_SCENE_XML 路径常量外，
    src/ scripts/ tests/ main.py 中不应出现 TouchSite 代码符号。"""
    result = subprocess.run(
        [
            "grep", "-rn",
            "--include=*.py",
            r"TOUCH_SITE\|touch_map\|TACTILE_MODEL\|LEFT_TOUCH_SENSOR\|RIGHT_TOUCH_SENSOR\|tactile_model",
            "src/", "scripts/", "tests/", "main.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    hits = [
        line for line in result.stdout.splitlines()
        if line  # exclude本测试自身
        and "test_no_touchsite.py" not in line
        and "TACTILE_XML" not in line          # paths.py 资产常量豁免
        and "TACTILE_SCENE_XML" not in line   # paths.py 资产常量豁免
    ]
    assert not hits, "残留 TouchSite 符号:\n" + "\n".join(hits)
```

- [ ] **Step 3.2: 跑测试确认会失败**

```bash
uv run pytest tests/test_no_touchsite.py -v
```

Expected: FAIL — 列出当前所有 TouchSite 残留点（应有 10+ 行）。

- [ ] **Step 3.3: 清理 `constants.py`**

替换 `src/contactile_mjlab/tasks/tactile_grasp/constants.py` 全部内容：

```python
"""tactile_grasp 任务共享常量。"""

from __future__ import annotations

from mjlab.managers import SceneEntityCfg

TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres"

# 触觉成功判定阈值（单值）
TACTILE_ACTIVITY_THRESHOLD = 1.0e-3

TAXEL_INDEXES = tuple(f"{row}{col}" for row in range(3) for col in range(3))

LEFT_TAXEL_FORCE_SENSOR_NAMES = tuple(f"left_taxel_force_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_FORCE_SENSOR_NAMES = tuple(f"right_taxel_force_{index}" for index in TAXEL_INDEXES)

LEFT_TAXEL_BODY_NAMES = tuple(f"left_taxel_body_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_BODY_NAMES = tuple(f"right_taxel_body_{index}" for index in TAXEL_INDEXES)

LEFT_TAXEL_SITE_NAMES = tuple(f"left_taxel_site_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_SITE_NAMES = tuple(f"right_taxel_site_{index}" for index in TAXEL_INDEXES)

ROBOT_JOINT_NAMES = (
    "left_driver_joint",
    "left_spring_link_joint",
    "left_follower",
    "right_driver_joint",
    "right_spring_link_joint",
    "right_follower_joint",
)

ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES)
OBJECT_CFG = SceneEntityCfg("object")
```

（删除：`TOUCH_SITE_TASK_ID`、`PTS_SPHERES_TASK_ID`、`DEFAULT_TASK_ID`、`TACTILE_MODEL_TOUCH_SITE`、`TACTILE_MODEL_PTS_SPHERES`、`TACTILE_OBS_DIM_BY_TASK`、`TACTILE_ACTIVITY_THRESHOLD_BY_TASK`、`LEFT_TOUCH_SENSOR_NAMES`、`RIGHT_TOUCH_SENSOR_NAMES`。`TASK_ID` 取代旧的 `DEFAULT_TASK_ID`/`PTS_SPHERES_TASK_ID`。）

- [ ] **Step 3.4: 清理 `tactile_terms.py`**

打开 `src/contactile_mjlab/tasks/tactile_grasp/tactile_terms.py`，删除 `touch_map` 函数（第 30-36 行）。其余保留不动。

- [ ] **Step 3.5: 简化 `env_cfg.py`（消 TouchSite 分支）**

打开 `src/contactile_mjlab/tasks/tactile_grasp/env_cfg.py`：

1. 删除 `tactile_model` 字段（第 43 行）、`touch_scale` 字段（第 50 行）。
2. 删除 `build()` 内的 `if/elif/else` 分支（第 62-78 行），统一改为：

```python
        left_sensor_names = LEFT_TAXEL_FORCE_SENSOR_NAMES
        right_sensor_names = RIGHT_TAXEL_FORCE_SENSOR_NAMES
        tactile_func = tactile_terms.taxel_force_map
        tactile_scale = 1.0 / self.force_scale
        tactile_threshold = TACTILE_ACTIVITY_THRESHOLD
        if self.success_tactile_threshold is not None:
            tactile_threshold = self.success_tactile_threshold
```

3. import 段（第 22-34 行）改为：

```python
from .constants import (
    LEFT_TAXEL_FORCE_SENSOR_NAMES,
    OBJECT_CFG,
    RIGHT_TAXEL_FORCE_SENSOR_NAMES,
    ROBOT_JOINT_CFG,
    TACTILE_ACTIVITY_THRESHOLD,
)
```

4. `build_robot_cfg(self.tactile_model)` 改为 `build_robot_cfg()`（第 130 行）。

- [ ] **Step 3.6: 简化 `robot_cfg.py`**

打开 `src/contactile_mjlab/tasks/tactile_grasp/robot_cfg.py`，整文件改为：

```python
"""Robotiq + PTS spheres 模型构造。"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from ...mjlab.action_terms import RobotiqCommandActionCfg
from ...mjlab.actuators import RobotiqGeneralActuatorCfg
from ...paths import PTS_SPHERES_XML


def robot_spec() -> mujoco.MjSpec:
    """加载 PTS spheres 触觉模型 spec。"""
    return mujoco.MjSpec.from_file(str(PTS_SPHERES_XML))


def build_robot_cfg() -> EntityCfg:
    """构建 Robotiq entity config。"""
    return EntityCfg(
        spec_fn=robot_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                RobotiqGeneralActuatorCfg(
                    target_names_expr=("split",),
                    transmission_type="tendon",
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "left_driver_joint": 0.0,
                "left_spring_link_joint": 0.0,
                "left_follower": 0.0,
                "right_driver_joint": 0.0,
                "right_spring_link_joint": 0.0,
                "right_follower_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


def build_action_cfg(delta_u_max: float) -> RobotiqCommandActionCfg:
    """Robotiq Δu 动作 config。"""
    return RobotiqCommandActionCfg(
        entity_name="robot",
        actuator_name="fingers_actuator",
        tendon_name="split",
        delta_u_max=delta_u_max,
    )
```

（删除：`robot_spec` 的 `tactile_model` 参数、`TACTILE_MODEL_*` import、`TACTILE_XML` import。）

- [ ] **Step 3.7: 简化 `tasks/tactile_grasp/__init__.py`**

替换全部内容为：

```python
"""tactile_grasp 任务注册与便利入口。"""

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
    return ManagerBasedRlEnv(
        load_env_cfg(task_id, play=play, **cfg_overrides),
        device=device,
        render_mode=render_mode,
    )


__all__ = ["TASK_ID", "load_env_cfg", "load_rl_cfg", "load_runner_cls", "make_env"]
```

注意：override 白名单暂时保留，将在 Task 7 一并去除——本 task 只做 TouchSite 清理。

- [ ] **Step 3.8: 简化顶层 `src/contactile_mjlab/__init__.py`**

替换全部内容为：

```python
"""contactile_mjlab: 触觉抓取任务包（PTS spheres）。"""

from . import tasks as tasks
from .tasks.tactile_grasp import (
    TASK_ID,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
    make_env,
)

__all__ = [
    "TASK_ID",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "make_env",
    "tasks",
]
```

- [ ] **Step 3.9: 更新 `main.py`、`scripts/smoke_env.py`、`scripts/view_env.py`**

`main.py`（替换全部内容）：

```python
"""跑一段 episode 并打印观测统计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from contactile_mjlab import TASK_ID, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    env = make_env(
        TASK_ID,
        episode_length_s=args.steps * 0.02,
        auto_reset=False,
    )
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    print(f"task_id={TASK_ID}")
    print(f"obs.shape={tuple(actor_obs.shape)} dtype={actor_obs.dtype}")
    print(f"obs.min={float(actor_obs.min()):.6f} obs.max={float(actor_obs.max()):.6f}")

    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    step = 0
    while not bool(torch.any(terminated | truncated)) and step < args.steps:
        action = torch.ones((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        observations, reward, terminated, truncated, _ = env.step(action)
        step += 1

    actor_obs = observations["actor"]
    print(f"steps={step} terminated={terminated.cpu().tolist()} truncated={truncated.cpu().tolist()}")
    print(f"final_reward={float(reward[0].cpu().item()):.6f}")
    print(f"finite={bool(np.isfinite(actor_obs.cpu().numpy()).all())}")


if __name__ == "__main__":
    main()
```

`scripts/smoke_env.py`（替换全部内容）：

```python
"""极简 smoke：跑一步 step 验证包可用。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from contactile_mjlab import make_env


def main() -> None:
    env = make_env()
    observations, _ = env.reset()
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    observations, reward, terminated, truncated, _ = env.step(action)
    print(
        "contactile-mjlab ready: "
        f"actor_obs_shape={tuple(observations['actor'].shape)} "
        f"reward_shape={tuple(reward.shape)} "
        f"terminated={terminated.cpu().tolist()} truncated={truncated.cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
```

`scripts/view_env.py`（替换全部内容）：

```python
"""带 viewer 跑 tactile grasp env 做可视化检查。"""

from __future__ import annotations

import argparse

import torch

from contactile_mjlab import TASK_ID, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env = make_env(
        TASK_ID,
        play=True,
        num_envs=1,
        episode_length_s=6.0,
        auto_reset=True,
        device=args.device,
        render_mode="human",
    )

    env.reset()
    step = 0
    while True:
        action = 0.3 * torch.randn(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        if step % 50 == 0 or terminated.any() or truncated.any():
            status = " [TERM]" if terminated.any() else (" [TRUNC]" if truncated.any() else "")
            print(f"step={step:4d}  reward={reward[0].item():+.4f}{status}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.10: 跑 TouchSite 残留测试**

```bash
uv run pytest tests/test_no_touchsite.py -v
```

Expected: PASS。

- [ ] **Step 3.11: 跑全套验证**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
uv run python main.py --steps 30
```

Expected: 全绿；smoke 输出 obs shape 应为 `(1, 80)`（27+27+6+6+1+6+6+1 = 80）。

- [ ] **Step 3.12: Commit**

```bash
git add -A src/ tests/ scripts/ main.py
git commit -m "refactor: 删除 TouchSite 代码路径（资产保留以备对照）"
```

**回滚条件：** smoke 输出 obs shape 不是 80 → 漏改某处；`git restore .`。

---

## Task 4: 包重命名 + 目录扁平化

**目标：** `contactile_mjlab` → `tactile_grasp`，子目录 `tasks/tactile_grasp/` 扁平化到包根，删除 `envs/`、`mjlab/`（其内容迁到 `tactile_grasp/mdp_legacy.py` 临时位置，Task 5 再正式拆分）。Task ID 改名。

**Files:**
- Move: `src/contactile_mjlab/` → `src/tactile_grasp/`
- Flatten: `src/tactile_grasp/tasks/tactile_grasp/*` → `src/tactile_grasp/*`
- Delete: `src/tactile_grasp/envs/`、`src/tactile_grasp/tasks/`
- Move: `src/tactile_grasp/mjlab/action_terms.py` → `src/tactile_grasp/_mdp_legacy/actions.py` (中转, Task 5 拆分)
- Move: `src/tactile_grasp/mjlab/actuators.py` → `src/tactile_grasp/_mdp_legacy/actuators.py` (中转)
- Delete: `src/tactile_grasp/mjlab/`
- Modify: `pyproject.toml`（`name = "tactile-grasp"`、`package-data` key、entry-point）
- Modify: 所有 `from contactile_mjlab...` 调用方
- Modify: `src/tactile_grasp/constants.py`（`TASK_ID` 改名）

- [ ] **Step 4.1: 写"新包名 importable"测试**

新建 `tests/test_package_layout.py`：

```python
"""新包 layout 与命名验证。"""
from __future__ import annotations

import pkgutil


def test_top_level_import():
    import tactile_grasp
    assert hasattr(tactile_grasp, "TASK_ID")
    assert hasattr(tactile_grasp, "make_env")


def test_task_id_value():
    import tactile_grasp
    assert tactile_grasp.TASK_ID == "Mjlab-TactileGrasp-Robotiq2F85"


def test_no_old_package():
    """旧包名 contactile_mjlab 应已无法 import。"""
    import importlib
    try:
        importlib.import_module("contactile_mjlab")
    except ModuleNotFoundError:
        return
    raise AssertionError("contactile_mjlab 仍可 import — 重命名未完成")


def test_no_subdir_tasks():
    """tactile_grasp.tasks 子包应已不存在。"""
    import importlib
    try:
        importlib.import_module("tactile_grasp.tasks")
    except ModuleNotFoundError:
        return
    raise AssertionError("tactile_grasp.tasks 仍存在 — 扁平化未完成")
```

- [ ] **Step 4.2: 跑测试确认会失败**

```bash
uv run pytest tests/test_package_layout.py -v
```

Expected: 全 FAIL（模块名都还是旧的）。

- [ ] **Step 4.3: 文件搬运 — 第一步：rename 顶层目录**

```bash
git mv src/contactile_mjlab src/tactile_grasp
```

- [ ] **Step 4.4: 文件搬运 — 第二步：扁平化 `tasks/tactile_grasp/*`**

```bash
git mv src/tactile_grasp/tasks/tactile_grasp/constants.py     src/tactile_grasp/constants.py
git mv src/tactile_grasp/tasks/tactile_grasp/env_cfg.py       src/tactile_grasp/env_cfg.py
git mv src/tactile_grasp/tasks/tactile_grasp/object_cfg.py    src/tactile_grasp/object_cfg.py
git mv src/tactile_grasp/tasks/tactile_grasp/reward_terms.py  src/tactile_grasp/reward_terms.py
git mv src/tactile_grasp/tasks/tactile_grasp/rl_cfg.py        src/tactile_grasp/rl_cfg.py
git mv src/tactile_grasp/tasks/tactile_grasp/robot_cfg.py     src/tactile_grasp/robot_cfg.py
git mv src/tactile_grasp/tasks/tactile_grasp/tactile_terms.py src/tactile_grasp/tactile_terms.py
rm src/tactile_grasp/tasks/tactile_grasp/__init__.py
rm src/tactile_grasp/tasks/__init__.py
rmdir src/tactile_grasp/tasks/tactile_grasp src/tactile_grasp/tasks
```

- [ ] **Step 4.5: 文件搬运 — 第三步：中转 `mjlab/` 子包**

```bash
mkdir -p src/tactile_grasp/_mdp_legacy
git mv src/tactile_grasp/mjlab/action_terms.py src/tactile_grasp/_mdp_legacy/actions.py
git mv src/tactile_grasp/mjlab/actuators.py    src/tactile_grasp/_mdp_legacy/actuators.py
rm src/tactile_grasp/mjlab/__init__.py
rmdir src/tactile_grasp/mjlab
```

新建 `src/tactile_grasp/_mdp_legacy/__init__.py`：

```python
"""中转目录：Task 5 时拆入正式 mdp/ 子包。"""

from .actions import RobotiqCommandAction, RobotiqCommandActionCfg
from .actuators import RobotiqGeneralActuatorCfg

__all__ = ["RobotiqCommandAction", "RobotiqCommandActionCfg", "RobotiqGeneralActuatorCfg"]
```

- [ ] **Step 4.6: 文件搬运 — 第四步：删除 `envs/` 子包**

```bash
rm src/tactile_grasp/envs/__init__.py
rmdir src/tactile_grasp/envs
```

- [ ] **Step 4.7: 改 `constants.py` 中 `TASK_ID`**

打开 `src/tactile_grasp/constants.py`：

```python
TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85"
```

（去掉 `-PTSSpheres` 后缀。）

- [ ] **Step 4.8: 修复包内 import 路径**

`src/tactile_grasp/env_cfg.py`：

```python
from . import reward_terms, tactile_terms
from .constants import (...)         # 原 from .constants import ... 即可
from .object_cfg import build_object_cfg
from .robot_cfg import build_action_cfg, build_robot_cfg
```

（已经是相对 import，文件搬位置后路径仍正确。）

`src/tactile_grasp/tactile_terms.py` 第 9 行：

```python
# 原: from ...mjlab.action_terms import RobotiqCommandAction
from ._mdp_legacy.actions import RobotiqCommandAction
```

`src/tactile_grasp/robot_cfg.py` 第 8-10 行：

```python
# 原: from ...mjlab.action_terms import RobotiqCommandActionCfg
# 原: from ...mjlab.actuators import RobotiqGeneralActuatorCfg
# 原: from ...paths import PTS_SPHERES_XML
from ._mdp_legacy.actions import RobotiqCommandActionCfg
from ._mdp_legacy.actuators import RobotiqGeneralActuatorCfg
from .paths import PTS_SPHERES_XML
```

`src/tactile_grasp/object_cfg.py` 第 8 行：

```python
# 原: from ...paths import PROPS_DIR
from .paths import PROPS_DIR
```

`src/tactile_grasp/reward_terms.py` 第 11-12 行：保持（相对 `.`，本来就对）。

- [ ] **Step 4.9: 重写 `src/tactile_grasp/__init__.py`**

替换全部内容（合并旧顶层 + 旧 `tasks/tactile_grasp/__init__.py`）：

```python
"""tactile_grasp: Robotiq 2F-85 + PTS spheres 触觉抓取任务包。"""

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
    return ManagerBasedRlEnv(
        load_env_cfg(task_id, play=play, **cfg_overrides),
        device=device,
        render_mode=render_mode,
    )


__all__ = ["TASK_ID", "load_env_cfg", "load_rl_cfg", "load_runner_cls", "make_env"]
```

- [ ] **Step 4.10: 改 `pyproject.toml`**

字段改动：

```toml
[project]
name = "tactile-grasp"               # was "contactile-mjlab"
```

```toml
[tool.setuptools.package-data]
tactile_grasp = [                    # was "contactile_mjlab"
    "assets/**/*.xml",
    "assets/**/*.stl",
    "assets/**/*.obj",
    "assets/**/*.png",
    "assets/**/*.mtl",
]
```

```toml
[project.entry-points."mjlab.tasks"]
tactile_grasp = "tactile_grasp"      # was contactile_mjlab = "contactile_mjlab"
```

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["tactile_grasp*"]         # 新增
```

- [ ] **Step 4.11: 重装并同步**

```bash
uv sync --extra cu128 --group dev
```

Expected: 成功；`tactile_grasp` 被装为可编辑包。

- [ ] **Step 4.12: 全局替换 `contactile_mjlab` → `tactile_grasp`**

更新外部调用方：

| 文件 | 改动 |
|---|---|
| `main.py:14` | `from contactile_mjlab import` → `from tactile_grasp import` |
| `scripts/smoke_env.py:12` | `from contactile_mjlab import` → `from tactile_grasp import` |
| `scripts/view_env.py:9` | `from contactile_mjlab import` → `from tactile_grasp import` |
| `scripts/check_mjcf.py:13` | `from contactile_mjlab.paths import` → `from tactile_grasp.paths import` |
| `scripts/test_gripper_ctrl.py:13` | `from contactile_mjlab import make_env` → `from tactile_grasp import make_env` |
| `scripts/visualize_taxels.py:13` | `from contactile_mjlab.paths import` → `from tactile_grasp.paths import` |
| `scripts/inspect_pts_frames.py:28-29` | `from contactile_mjlab.paths import` → `from tactile_grasp.paths import`；`from contactile_mjlab.tasks.tactile_grasp.constants import` → `from tactile_grasp.constants import` |
| `scripts/train_ppo.py:12` | `from contactile_mjlab import DEFAULT_TASK_ID, ...` → `from tactile_grasp import TASK_ID as DEFAULT_TASK_ID, ...` (Task 9 一并删此文件，本 task 先保持其能运行) |

批量替换命令：

```bash
sed -i 's/contactile_mjlab/tactile_grasp/g' main.py scripts/*.py tests/test_inspect_pts_frames_cli.py
```

然后逐一检查 `git diff` 中是否出现了不该改的字段（如 `package-data` 已在 4.10 中手动改过，避免重复），逻辑特殊点的手动调整：

- `main.py`：去掉旧的 `TOUCH_SITE_TASK_ID, PTS_SPHERES_TASK_ID` import（Task 3 已完成？再 grep 一次确认）。
- `scripts/train_ppo.py`：把 `DEFAULT_TASK_ID` 改成 `TASK_ID`。
- `scripts/inspect_pts_frames.py:29-31` 已从 `contactile_mjlab.tasks.tactile_grasp.constants` 改为 `tactile_grasp.constants`。

`sys.path.insert` 那行（`sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))`）继续保留，无问题。

- [ ] **Step 4.13: 跑包 layout 测试**

```bash
uv run pytest tests/test_package_layout.py tests/test_no_touchsite.py tests/test_asset_paths.py -v
```

Expected: 全 PASS。

- [ ] **Step 4.14: 跑 smoke + main**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
uv run python main.py --steps 30
```

Expected: 全绿；smoke 输出 obs shape `(1, 80)` 不变。

- [ ] **Step 4.15: Commit**

```bash
git add -A
git commit -m "refactor: 包重命名 contactile_mjlab → tactile_grasp，子目录扁平化"
```

**回滚条件：** ImportError 提示找不到 `tactile_grasp.tasks` 或残留 `contactile_mjlab` → grep 漏改的；`git restore .` 后重试，重点检查 sed 是否漏覆盖某文件。

---

## Task 5: mdp/ 子包拆分（actions / observations / rewards / events / terminations）

**目标：** 把 `_mdp_legacy/` + `tactile_terms.py` + `reward_terms.py` 按 mjlab 上游惯例拆成 `mdp/{actions,actuators,observations,rewards,events,terminations}.py`。

**Files:**
- Create: `src/tactile_grasp/mdp/__init__.py`
- Create: `src/tactile_grasp/mdp/actions.py` (from `_mdp_legacy/actions.py`)
- Create: `src/tactile_grasp/mdp/actuators.py` (from `_mdp_legacy/actuators.py`)
- Create: `src/tactile_grasp/mdp/observations.py` (from `tactile_terms.py`)
- Create: `src/tactile_grasp/mdp/rewards.py` (from `reward_terms.py` reward 部分)
- Create: `src/tactile_grasp/mdp/terminations.py` (from `reward_terms.py` termination 部分)
- Create: `src/tactile_grasp/mdp/events.py` (新建：re-export mjlab 内置)
- Delete: `src/tactile_grasp/_mdp_legacy/`、`tactile_terms.py`、`reward_terms.py`
- Modify: `src/tactile_grasp/env_cfg.py`（import 切到 `.mdp`）
- Modify: `src/tactile_grasp/robot_cfg.py`（import 切到 `.mdp`）

- [ ] **Step 5.1: 写 mdp 子包 layout 测试**

新建 `tests/test_mdp_layout.py`：

```python
"""mdp/ 子包应包含五个标准子模块。"""
from __future__ import annotations

import importlib


def test_mdp_submodules_exist():
    for name in ("actions", "actuators", "observations", "rewards", "terminations", "events"):
        importlib.import_module(f"tactile_grasp.mdp.{name}")


def test_observations_exports():
    from tactile_grasp.mdp import observations
    assert callable(getattr(observations, "taxel_force_map"))
    assert callable(getattr(observations, "pad_wrench"))
    assert callable(getattr(observations, "gripper_command"))


def test_rewards_exports():
    from tactile_grasp.mdp import rewards
    for name in ("alive", "action_l2", "close_command_l2", "tactile_force_l2"):
        assert callable(getattr(rewards, name)), name


def test_terminations_exports():
    from tactile_grasp.mdp import terminations
    assert callable(getattr(terminations, "object_height_below"))
    assert getattr(terminations, "stable_grasp_hold") is not None


def test_legacy_dirs_gone():
    import importlib
    for name in ("tactile_grasp._mdp_legacy", "tactile_grasp.tactile_terms", "tactile_grasp.reward_terms"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{name} 仍可 import — 拆分未完成")
```

- [ ] **Step 5.2: 跑测试确认会失败**

```bash
uv run pytest tests/test_mdp_layout.py -v
```

Expected: 全 FAIL。

- [ ] **Step 5.3: 创建 `mdp/` 目录**

```bash
mkdir -p src/tactile_grasp/mdp
```

新建 `src/tactile_grasp/mdp/__init__.py`：

```python
"""tactile_grasp MDP terms：按 mjlab 上游惯例切分。"""

from . import actions, actuators, events, observations, rewards, terminations

__all__ = ["actions", "actuators", "events", "observations", "rewards", "terminations"]
```

- [ ] **Step 5.4: 迁 actions / actuators**

```bash
git mv src/tactile_grasp/_mdp_legacy/actions.py    src/tactile_grasp/mdp/actions.py
git mv src/tactile_grasp/_mdp_legacy/actuators.py  src/tactile_grasp/mdp/actuators.py
rm src/tactile_grasp/_mdp_legacy/__init__.py
rmdir src/tactile_grasp/_mdp_legacy
```

文件内容不变（绝对导入，无相对 import）。

- [ ] **Step 5.5: 迁 observations**

```bash
git mv src/tactile_grasp/tactile_terms.py src/tactile_grasp/mdp/observations.py
```

打开 `src/tactile_grasp/mdp/observations.py`，把第 9 行：

```python
from ._mdp_legacy.actions import RobotiqCommandAction
```

改为：

```python
from .actions import RobotiqCommandAction
```

其余内容不变（含 `touch_map` 已在 Task 3 删除；`taxel_force_map`、`pad_wrench`、`gripper_command`、`sensor_values`、`_sensor_tensor` 保留）。

- [ ] **Step 5.6: 创建 `mdp/rewards.py`**

新建 `src/tactile_grasp/mdp/rewards.py`，从原 `reward_terms.py` 的第 1-58 行（导入 + `tactile_force_l2`、`total_tactile_signal`、`alive`、`action_l2`、`close_command_l2`）+ 新增 `drop_penalty`：

```python
"""tactile_grasp 奖励函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import observations as obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tactile_force_l2(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """惩罚双指尖触觉幅值平方和。"""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.square(torch.cat([left, right], dim=-1)), dim=1)


def total_tactile_signal(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """双指尖触觉绝对值之和（被 terminations 复用）。"""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.abs(torch.cat([left, right], dim=-1)), dim=1)


def alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """未终止的环境给 +1。"""
    return (~env.termination_manager.terminated).float()


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """惩罚动作幅度平方。"""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def close_command_l2(
    env: "ManagerBasedRlEnv",
    action_name: str = "gripper_command",
) -> torch.Tensor:
    """惩罚多余的夹爪闭合命令。"""
    command = obs.gripper_command(env, action_name=action_name)
    return torch.sum(torch.square(command), dim=1)


def drop_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """物体掉落 termination 命中时的负奖励通道。"""
    return env.termination_manager.get_term("object_drop").float()
```

- [ ] **Step 5.7: 创建 `mdp/terminations.py`**

新建 `src/tactile_grasp/mdp/terminations.py`，从原 `reward_terms.py` 的第 61-117 行（`object_height`、`object_height_below`、`stable_grasp_hold` 类）：

```python
"""tactile_grasp 终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg

from ..constants import OBJECT_CFG
from . import rewards as rew

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def object_height(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """物体根 link 的 z 坐标。"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]


def object_height_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """物体低于阈值时终止。"""
    return object_height(env, asset_cfg=asset_cfg) < minimum_height


class stable_grasp_hold:
    """持续 contact-and-hold 的终止判定（带 per-env 计数器）。"""

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._counter[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        hold_steps: int,
        minimum_height: float,
        minimum_tactile_signal: float,
        left_sensor_names: tuple[str, ...],
        right_sensor_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = OBJECT_CFG,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        height_ok = object_height(env, asset_cfg=asset_cfg) > minimum_height
        touch_ok = (
            rew.total_tactile_signal(
                env,
                left_sensor_names=left_sensor_names,
                right_sensor_names=right_sensor_names,
                entity_name=entity_name,
            )
            > minimum_tactile_signal
        )
        stable = height_ok & touch_ok
        self._counter = torch.where(stable, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
```

- [ ] **Step 5.8: 创建 `mdp/events.py`**

新建 `src/tactile_grasp/mdp/events.py`：

```python
"""tactile_grasp event terms（当前仅复用 mjlab 内置）。"""

from __future__ import annotations

from mjlab.envs.mdp.events import reset_scene_to_default

__all__ = ["reset_scene_to_default"]
```

- [ ] **Step 5.9: 删除 `reward_terms.py`**

```bash
rm src/tactile_grasp/reward_terms.py
```

- [ ] **Step 5.10: 改 `env_cfg.py` 的 import**

打开 `src/tactile_grasp/env_cfg.py`：

把：

```python
from mjlab.envs.mdp.events import reset_scene_to_default
from . import reward_terms, tactile_terms
```

改为：

```python
from .mdp import events
from .mdp import observations as tactile_terms  # 暂留 alias，Task 6 起逐步换名
from .mdp import rewards as reward_terms        # 暂留 alias
from .mdp import terminations
```

再把原 `func=lambda env: env.termination_manager.get_term("object_drop").float()`（约第 169 行）改为：

```python
func=reward_terms.drop_penalty,
```

把原 `func=reward_terms.object_height_below`（约第 176 行）改为 `func=terminations.object_height_below`。
把原 `func=reward_terms.stable_grasp_hold`（约第 183 行）改为 `func=terminations.stable_grasp_hold`。

把：

```python
events={
    "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
    )
},
```

改为：

```python
events={
    "reset_scene_to_default": EventTermCfg(
        func=events.reset_scene_to_default,
        mode="reset",
    )
},
```

- [ ] **Step 5.11: 改 `robot_cfg.py` import**

打开 `src/tactile_grasp/robot_cfg.py` 第 8-9 行：

```python
# 原 from ._mdp_legacy.actions import RobotiqCommandActionCfg
# 原 from ._mdp_legacy.actuators import RobotiqGeneralActuatorCfg
from .mdp.actions import RobotiqCommandActionCfg
from .mdp.actuators import RobotiqGeneralActuatorCfg
```

- [ ] **Step 5.12: 跑 mdp layout 测试**

```bash
uv run pytest tests/test_mdp_layout.py -v
```

Expected: 全 PASS。

- [ ] **Step 5.13: 跑全套验证**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
```

Expected: 全绿；obs shape 仍为 `(1, 80)`。

- [ ] **Step 5.14: Commit**

```bash
git add -A
git commit -m "refactor: mdp/ 按 mjlab 惯例拆五块（actions/observations/rewards/events/terminations）"
```

**回滚条件：** ImportError 或 obs shape 变化 → 函数迁移漏一个；`git restore .`，逐个核对 `reward_terms.py` / `tactile_terms.py` 内容是否完整搬运。

---

## Task 6: 触觉观测拆 normal / tangential + history_length

**目标：** `taxel_force_map`（27-d per pad，含法向+切向混合）→ 拆为 `taxel_normal_force`（9-d 法向 z）+ `taxel_tangential_force`（18-d 切向 xy）。`pad_wrench`（6-d）拆为 `pad_force`（3-d）+ `pad_torque`（3-d）。所有触觉/wrench 观测加 `history_length`。

**Files:**
- Modify: `src/tactile_grasp/mdp/observations.py`
- Modify: `src/tactile_grasp/env_cfg.py`
- Test: `tests/test_observation_shapes.py` (new)

- [ ] **Step 6.1: 写观测维度测试**

新建 `tests/test_observation_shapes.py`：

```python
"""触觉观测维度按 normal/tangential 拆分 + history_length 装配后正确。"""
from __future__ import annotations

import torch

from tactile_grasp import make_env


def test_actor_obs_dim_with_history():
    """单帧 80 → 加 history 后 320。"""
    env = make_env()
    obs, _ = env.reset()
    actor = obs["actor"]
    assert actor.ndim == 2
    # 触觉 + wrench 加 history；joint_pos/vel/gripper_cmd/last_action 不加
    # taxel_normal 两边 18 → 90 (×5)
    # taxel_tangential 两边 36 → 180 (×5)
    # pad_force 两边 6 → 18 (×3)
    # pad_torque 两边 6 → 18 (×3)
    # joint_pos+vel 12 + gripper_cmd 1 + last_action 1 = 14（其中 gripper_cmd 是标量 / 1-d）
    # 总和 90 + 180 + 18 + 18 + 12 + 1 + 1 = 320
    assert actor.shape == (env.num_envs, 320), actor.shape


def test_normal_force_dim():
    """taxel_normal_force 单 pad 返回 9 维。"""
    from tactile_grasp.mdp import observations
    from tactile_grasp.constants import LEFT_TAXEL_FORCE_SENSOR_NAMES
    env = make_env()
    env.reset()
    out = observations.taxel_normal_force(env, sensor_names=LEFT_TAXEL_FORCE_SENSOR_NAMES)
    assert out.shape == (env.num_envs, 9), out.shape


def test_tangential_force_dim():
    """taxel_tangential_force 单 pad 返回 18 维（xy × 9 taxel）。"""
    from tactile_grasp.mdp import observations
    from tactile_grasp.constants import LEFT_TAXEL_FORCE_SENSOR_NAMES
    env = make_env()
    env.reset()
    out = observations.taxel_tangential_force(env, sensor_names=LEFT_TAXEL_FORCE_SENSOR_NAMES)
    assert out.shape == (env.num_envs, 18), out.shape
```

- [ ] **Step 6.2: 跑测试确认会失败**

```bash
uv run pytest tests/test_observation_shapes.py -v
```

Expected: FAIL — `taxel_normal_force` / `taxel_tangential_force` 不存在 AttributeError；或维度 80 ≠ 320。

- [ ] **Step 6.3: 在 `mdp/observations.py` 加新拆分函数**

打开 `src/tactile_grasp/mdp/observations.py`，在文件末尾追加：

```python
def _stack_taxel_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str,
) -> torch.Tensor:
    """把每个 taxel 的 3D 力堆成 (N, n_taxels, 3) — 内部辅助。"""
    per_taxel = [_sensor_tensor(env, f"{entity_name}/{name}") for name in sensor_names]
    return torch.stack(per_taxel, dim=1)  # (N, n_taxels, 3)


def taxel_normal_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """每个 taxel 的法向力（z 分量），输出 (N, n_taxels)。

    PTS sphere taxel 局部坐标系：z = 法向（指向手指外侧），xy = 切向。
    """
    stacked = _stack_taxel_force(env, sensor_names, entity_name)
    return stacked[..., 2]


def taxel_tangential_force(
    env: "ManagerBasedRlEnv",
    sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """每个 taxel 的切向力（xy 分量），展开为 (N, 2 * n_taxels)。"""
    stacked = _stack_taxel_force(env, sensor_names, entity_name)
    tangential = stacked[..., :2]  # (N, n_taxels, 2)
    return tangential.reshape(tangential.shape[0], -1)


def pad_force(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """指尖聚合 3D 力。"""
    return _sensor_tensor(env, f"{entity_name}/{sensor_name}")


def pad_torque(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """指尖聚合 3D 力矩。"""
    return _sensor_tensor(env, f"{entity_name}/{sensor_name}")
```

（旧 `taxel_force_map`、`pad_wrench` 暂时保留，Task 7 重写 env_cfg 时再删。）

- [ ] **Step 6.4: 改 `env_cfg.py` — 在常量块里加 scale + history**

在 `env_cfg.py` 的 `dataclass` 字段 (`force_scale: float = 20.0` 一带) 之后加：

```python
normal_force_scale: float = 5.0          # 5 N → 1.0
tangential_force_scale: float = 2.0      # 切向幅值典型小于法向
tactile_history_length: int = 5          # 100 ms @ 50 Hz
wrench_history_length: int = 3           # 60 ms
```

- [ ] **Step 6.5: 改 `env_cfg.py` — 重写 `actor_terms` 字典**

把 `build()` 内的 `actor_terms = {...}`（约第 80-125 行）整段替换为：

```python
        actor_terms = {
            "left_taxel_normal": ObservationTermCfg(
                func=tactile_terms.taxel_normal_force,
                params={"sensor_names": left_sensor_names},
                scale=1.0 / self.normal_force_scale,
                history_length=self.tactile_history_length,
            ),
            "left_taxel_tangential": ObservationTermCfg(
                func=tactile_terms.taxel_tangential_force,
                params={"sensor_names": left_sensor_names},
                scale=1.0 / self.tangential_force_scale,
                history_length=self.tactile_history_length,
            ),
            "right_taxel_normal": ObservationTermCfg(
                func=tactile_terms.taxel_normal_force,
                params={"sensor_names": right_sensor_names},
                scale=1.0 / self.normal_force_scale,
                history_length=self.tactile_history_length,
            ),
            "right_taxel_tangential": ObservationTermCfg(
                func=tactile_terms.taxel_tangential_force,
                params={"sensor_names": right_sensor_names},
                scale=1.0 / self.tangential_force_scale,
                history_length=self.tactile_history_length,
            ),
            "left_pad_force": ObservationTermCfg(
                func=tactile_terms.pad_force,
                params={"sensor_name": "left_pad_force"},
                scale=1.0 / self.force_scale,
                history_length=self.wrench_history_length,
            ),
            "left_pad_torque": ObservationTermCfg(
                func=tactile_terms.pad_torque,
                params={"sensor_name": "left_pad_torque"},
                scale=1.0 / self.torque_scale,
                history_length=self.wrench_history_length,
            ),
            "right_pad_force": ObservationTermCfg(
                func=tactile_terms.pad_force,
                params={"sensor_name": "right_pad_force"},
                scale=1.0 / self.force_scale,
                history_length=self.wrench_history_length,
            ),
            "right_pad_torque": ObservationTermCfg(
                func=tactile_terms.pad_torque,
                params={"sensor_name": "right_pad_torque"},
                scale=1.0 / self.torque_scale,
                history_length=self.wrench_history_length,
            ),
            "gripper_command": ObservationTermCfg(func=tactile_terms.gripper_command),
            "joint_pos": ObservationTermCfg(
                func=joint_pos_rel,
                params={"asset_cfg": ROBOT_JOINT_CFG},
            ),
            "joint_vel": ObservationTermCfg(
                func=joint_vel_rel,
                params={"asset_cfg": ROBOT_JOINT_CFG},
            ),
            "last_action": ObservationTermCfg(func=last_action),
        }
```

- [ ] **Step 6.6: 跑维度测试**

```bash
uv run pytest tests/test_observation_shapes.py -v
```

Expected: 全 PASS（obs shape (N, 320)、normal (N,9)、tangential (N,18)）。

如果测试期待 320 但实际不同，先 print actor.shape 看实际值。常见原因：

- `gripper_command` 返回标量但 mjlab 默认 reshape 为 (N,1)，应该没问题。若需调整测试期望值，按实际 print 数值改测试常量，记入 Step 6.7 之前。

- [ ] **Step 6.7: 跑完整 smoke**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
uv run python main.py --steps 30
```

Expected: 全绿；smoke 输出 `actor_obs_shape=(1, 320)`。

- [ ] **Step 6.8: Commit**

```bash
git add -A
git commit -m "feat: 触觉观测拆 normal/tangential + 加 history_length (100/60 ms)"
```

**回滚条件：** obs shape 与期望不符 → 检查传感器 reshape 逻辑（PTS 单 taxel 力是否真为 3 维？跑 `python scripts/inspect_pts_frames.py` 看 sensor data shape）。

---

## Task 7: `env_cfg.py` 改为 `make_xxx_env_cfg(play=False)` idiom

**目标：** 弃用 `TactileGraspTaskConfig` dataclass-builder，改为 `env_cfgs.py` 模块顶层常量 + `make_tactile_grasp_env_cfg(play=False)` 函数。`load_env_cfg` override 白名单去掉。文件改名 `env_cfg.py` → `env_cfgs.py`。

**Files:**
- Rename: `src/tactile_grasp/env_cfg.py` → `src/tactile_grasp/env_cfgs.py`
- Rewrite: `src/tactile_grasp/env_cfgs.py`
- Modify: `src/tactile_grasp/__init__.py`
- Modify: `src/tactile_grasp/mdp/observations.py`（删旧 `taxel_force_map` / `pad_wrench`）

- [ ] **Step 7.1: 写"新 idiom" 测试**

新建 `tests/test_env_cfg_idiom.py`：

```python
"""env_cfgs.make_tactile_grasp_env_cfg(play) 应返回可直接用的 mjlab cfg。"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def test_make_env_cfg_train():
    from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg
    cfg = make_tactile_grasp_env_cfg(play=False)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert cfg.scene.num_envs == 64
    assert cfg.episode_length_s == 3.0


def test_make_env_cfg_play():
    from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg
    cfg = make_tactile_grasp_env_cfg(play=True)
    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 6.0
    assert cfg.observations["actor"].enable_corruption is False


def test_no_dataclass_builder():
    """TactileGraspTaskConfig 应已删除。"""
    try:
        from tactile_grasp.env_cfgs import TactileGraspTaskConfig  # noqa: F401
    except ImportError:
        return
    raise AssertionError("TactileGraspTaskConfig 仍存在 — idiom 切换未完成")


def test_load_env_cfg_no_override_whitelist():
    """load_env_cfg 不应再有 override 白名单（任意覆写交给 tyro）。"""
    from tactile_grasp import load_env_cfg
    cfg = load_env_cfg(play=False)
    # 直接通过返回的 cfg 改字段应该 work
    cfg.scene.num_envs = 32
    assert cfg.scene.num_envs == 32
```

- [ ] **Step 7.2: 跑测试确认失败**

```bash
uv run pytest tests/test_env_cfg_idiom.py -v
```

Expected: FAIL — `env_cfgs` 模块不存在。

- [ ] **Step 7.3: 改名 + 重写 `env_cfgs.py`**

```bash
git mv src/tactile_grasp/env_cfg.py src/tactile_grasp/env_cfgs.py
```

整文件替换为：

```python
"""tactile_grasp 任务的环境 cfg 构造（mjlab idiom）。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .constants import (
    LEFT_TAXEL_FORCE_SENSOR_NAMES,
    OBJECT_CFG,
    RIGHT_TAXEL_FORCE_SENSOR_NAMES,
    ROBOT_JOINT_CFG,
    TACTILE_ACTIVITY_THRESHOLD,
)
from .mdp import events, observations as obs, rewards, terminations
from .object_cfg import build_object_cfg
from .robot_cfg import build_action_cfg, build_robot_cfg

# ---------- 控制 / 仿真 ----------
DECIMATION = 10
TIMESTEP = 0.002
EPISODE_LENGTH_S = 3.0
PLAY_EPISODE_LENGTH_S = 6.0

# ---------- 动作 ----------
DELTA_U_MAX = 3.0

# ---------- 观测缩放 ----------
NORMAL_FORCE_SCALE = 5.0
TANGENTIAL_FORCE_SCALE = 2.0
FORCE_SCALE = 20.0
TORQUE_SCALE = 2.0

# ---------- 时间窗口 ----------
TACTILE_HISTORY_LENGTH = 5
WRENCH_HISTORY_LENGTH = 3

# ---------- 环境数 / 间距 ----------
NUM_ENVS = 64
PLAY_NUM_ENVS = 1
ENV_SPACING = 0.5

# ---------- 成功 / 失败阈值 ----------
DROP_HEIGHT = 0.08
SUCCESS_HEIGHT = 0.14
SUCCESS_HOLD_STEPS = 25

# ---------- 奖励权重 ----------
W_ALIVE = 1.0
W_TACTILE_FORCE = -0.01
W_ACTION_RATE = -0.001
W_CLOSE_COMMAND = -0.001
W_DROP_PENALTY = -5.0


def _actor_observation_terms() -> dict[str, ObservationTermCfg]:
    return {
        "left_taxel_normal": ObservationTermCfg(
            func=obs.taxel_normal_force,
            params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / NORMAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "left_taxel_tangential": ObservationTermCfg(
            func=obs.taxel_tangential_force,
            params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / TANGENTIAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "right_taxel_normal": ObservationTermCfg(
            func=obs.taxel_normal_force,
            params={"sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / NORMAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "right_taxel_tangential": ObservationTermCfg(
            func=obs.taxel_tangential_force,
            params={"sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES},
            scale=1.0 / TANGENTIAL_FORCE_SCALE,
            history_length=TACTILE_HISTORY_LENGTH,
        ),
        "left_pad_force": ObservationTermCfg(
            func=obs.pad_force,
            params={"sensor_name": "left_pad_force"},
            scale=1.0 / FORCE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "left_pad_torque": ObservationTermCfg(
            func=obs.pad_torque,
            params={"sensor_name": "left_pad_torque"},
            scale=1.0 / TORQUE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "right_pad_force": ObservationTermCfg(
            func=obs.pad_force,
            params={"sensor_name": "right_pad_force"},
            scale=1.0 / FORCE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "right_pad_torque": ObservationTermCfg(
            func=obs.pad_torque,
            params={"sensor_name": "right_pad_torque"},
            scale=1.0 / TORQUE_SCALE,
            history_length=WRENCH_HISTORY_LENGTH,
        ),
        "gripper_command": ObservationTermCfg(func=obs.gripper_command),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel, params={"asset_cfg": ROBOT_JOINT_CFG}
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": ROBOT_JOINT_CFG}
        ),
        "last_action": ObservationTermCfg(func=last_action),
    }


def make_tactile_grasp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 tactile_grasp 环境 cfg；play=True 在共享基底上 mutate。"""
    actor_terms = _actor_observation_terms()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": build_robot_cfg(), "object": build_object_cfg()},
            num_envs=NUM_ENVS,
            env_spacing=ENV_SPACING,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
            "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
        },
        actions={"gripper_command": build_action_cfg(DELTA_U_MAX)},
        events={
            "reset_scene_to_default": EventTermCfg(
                func=events.reset_scene_to_default, mode="reset"
            )
        },
        rewards={
            "alive": RewardTermCfg(func=rewards.alive, weight=W_ALIVE),
            "tactile_force": RewardTermCfg(
                func=rewards.tactile_force_l2,
                weight=W_TACTILE_FORCE,
                params={
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                },
            ),
            "action_rate": RewardTermCfg(func=rewards.action_l2, weight=W_ACTION_RATE),
            "close_command": RewardTermCfg(func=rewards.close_command_l2, weight=W_CLOSE_COMMAND),
            "drop_penalty": RewardTermCfg(func=rewards.drop_penalty, weight=W_DROP_PENALTY),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "object_drop": TerminationTermCfg(
                func=terminations.object_height_below,
                params={"minimum_height": DROP_HEIGHT, "asset_cfg": OBJECT_CFG},
            ),
            "stable_grasp": TerminationTermCfg(
                func=terminations.stable_grasp_hold,
                params={
                    "hold_steps": SUCCESS_HOLD_STEPS,
                    "minimum_height": SUCCESS_HEIGHT,
                    "minimum_tactile_signal": TACTILE_ACTIVITY_THRESHOLD,
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "asset_cfg": OBJECT_CFG,
                },
            ),
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=TIMESTEP, cone="elliptic", impratio=10.0)
        ),
        viewer=ViewerConfig(),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        auto_reset=True,
    )

    if play:
        cfg.scene.num_envs = PLAY_NUM_ENVS
        cfg.episode_length_s = PLAY_EPISODE_LENGTH_S
        cfg.observations["actor"].enable_corruption = False

    return cfg
```

- [ ] **Step 7.4: 重写 `src/tactile_grasp/__init__.py` — 删 override 白名单 + 切到新 idiom**

替换全部内容：

```python
"""tactile_grasp: Robotiq 2F-85 + PTS spheres 触觉抓取任务包。"""

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
    """加载并深拷贝注册的 env cfg；任意字段覆写交给调用方。"""
    return deepcopy(_load_env_cfg(task_id, play=play))


def make_env(
    *,
    play: bool = False,
    device: str = "cpu",
    render_mode: str | None = None,
) -> ManagerBasedRlEnv:
    """便利函数：直接构造环境实例（用注册值，不接受字段覆写）。"""
    return ManagerBasedRlEnv(
        load_env_cfg(TASK_ID, play=play),
        device=device,
        render_mode=render_mode,
    )


__all__ = ["TASK_ID", "load_env_cfg", "load_rl_cfg", "load_runner_cls", "make_env"]
```

注意：`make_env` 不再接受 `num_envs / episode_length_s / auto_reset` 等参数（全部交给 tyro CLI 覆写）。`main.py`、`view_env.py` 需相应更新。

- [ ] **Step 7.5: 更新 `main.py`、`scripts/view_env.py`**

`main.py` 中调用：

```python
env = make_env(TASK_ID, episode_length_s=args.steps * 0.02, auto_reset=False)
```

改为：

```python
cfg = load_env_cfg(TASK_ID)
cfg.episode_length_s = args.steps * 0.02
cfg.auto_reset = False
from mjlab.envs import ManagerBasedRlEnv
env = ManagerBasedRlEnv(cfg, device="cpu")
```

并在 import 段加 `from tactile_grasp import TASK_ID, load_env_cfg`（删掉 `make_env`）。

`scripts/view_env.py` 同理：

```python
cfg = load_env_cfg(TASK_ID, play=True)
cfg.scene.num_envs = 1
cfg.episode_length_s = 6.0
cfg.auto_reset = True
env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="human")
```

- [ ] **Step 7.6: 删 `env_cfgs.py` 里残留旧符号 + 清理 observations.py**

打开 `src/tactile_grasp/mdp/observations.py`，删掉旧的 `taxel_force_map`、`pad_wrench` 函数（被替换了）。`sensor_values` 保留（被 rewards/terminations 用）。

- [ ] **Step 7.7: 跑全套验证**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
uv run python main.py --steps 30
```

Expected: 全绿；smoke obs shape `(1, 320)`。

- [ ] **Step 7.8: Commit**

```bash
git add -A
git commit -m "feat: env_cfgs.py 改为 make_xxx_env_cfg idiom，去掉 override 白名单"
```

**回滚条件：** 任何调用方报 `TypeError: make_env() got an unexpected keyword argument`，是 4.12 / 5.x 漏的接口对齐；`git restore .` 后检查所有 `make_env(` 调用点。

---

## Task 8: `rl_cfg.py` 默认值升级

**目标：** PPO 默认值改到 baseline 水平：hidden 256×256、obs_normalization、num_steps_per_env=48、max_iterations=3000、experiment_name="tactile_grasp"、wandb logger。

**Files:**
- Modify: `src/tactile_grasp/rl_cfg.py`
- Test: `tests/test_rl_cfg.py` (new)

- [ ] **Step 8.1: 写 rl_cfg 默认值测试**

新建 `tests/test_rl_cfg.py`：

```python
"""PPO 默认值校对（spec §6.1）。"""
from __future__ import annotations

from tactile_grasp.rl_cfg import tactile_grasp_ppo_runner_cfg


def test_defaults():
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
    """保留项不应改动。"""
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
```

- [ ] **Step 8.2: 跑测试确认失败**

```bash
uv run pytest tests/test_rl_cfg.py -v
```

Expected: FAIL（旧值）。

- [ ] **Step 8.3: 改 `rl_cfg.py`**

整文件替换为：

```python
"""tactile_grasp PPO 训练配置。"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def tactile_grasp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """tactile_grasp baseline 的 PPO runner 配置。"""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="tactile_grasp",
        logger="wandb",
        wandb_project="tactile_grasp",
        save_interval=50,
        upload_model=False,
        num_steps_per_env=48,
        max_iterations=3000,
    )
```

注意：`wandb_project` 字段名以 `RslRlOnPolicyRunnerCfg` 的实际字段名为准。如果 `RslRlBaseRunnerCfg` 不接受 `wandb_project` 字段，把它移到 `wandb_kwargs` 或调用方覆写。先用 dataclasses 字段名做：

```bash
uv run python -c "from mjlab.rl import RslRlOnPolicyRunnerCfg; import dataclasses; print([f.name for f in dataclasses.fields(RslRlOnPolicyRunnerCfg)])"
```

如果输出不含 `wandb_project`，把那行删除并改测试期望（在 Step 8.1 同步去掉 `wandb_project` 断言）。

- [ ] **Step 8.4: 跑 rl_cfg 测试**

```bash
uv run pytest tests/test_rl_cfg.py -v
```

Expected: 全 PASS。

- [ ] **Step 8.5: 跑全套 + smoke 训练 10 iter**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
uv run python scripts/train_ppo.py --max-iterations 5
```

Expected: 全绿；训练能跑 5 iter 不报错（如果 wandb logger 在本地没 wandb login，会 prompt 或 fail → 此情况下临时把 `logger="wandb"` 暂调为 `"tensorboard"` 跑通后再切回，**或**用 `WANDB_MODE=offline uv run python ...` 离线跑）。

如果 wandb 无法跑：保留 wandb 默认值，跳过 train smoke，由 Task 9 重新用 `mjlab.scripts.train` 入口验证。

- [ ] **Step 8.6: Commit**

```bash
git add -A
git commit -m "chore: rl_cfg.py 升级默认值（256x256, obs_norm, num_steps=48, wandb）"
```

**回滚条件：** `RslRlBaseRunnerCfg` 不接受某字段 → 查 mjlab.rl.config.py 实际字段；`git restore .`，按实际字段名重写。

---

## Task 9: `scripts/{train,play}.py` 极简 wrapper + 删 `train_ppo.py`

**目标：** 新增极简 `scripts/train.py` / `scripts/play.py`，调 `mjlab.scripts.train.main` / `play.main`。删除自实现的 `scripts/train_ppo.py`。

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/play.py`
- Delete: `scripts/train_ppo.py`
- Test: `tests/test_scripts_entry.py` (new)

- [ ] **Step 9.1: 写 train/play wrapper 测试**

新建 `tests/test_scripts_entry.py`：

```python
"""train.py / play.py 入口应能 import 且其包含 tactile_grasp side-effect import。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_script_help_lists_task():
    """train.py --help 应列出已注册的 TASK_ID。"""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/train.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # tyro 退出码可能为 0 或 1（看 mjlab.scripts.train 的 _cli 流程）；只要出现 TASK_ID
    combined = result.stdout + result.stderr
    assert "Mjlab-TactileGrasp-Robotiq2F85" in combined, combined


def test_old_train_ppo_removed():
    assert not (REPO_ROOT / "scripts" / "train_ppo.py").exists()
```

- [ ] **Step 9.2: 跑测试确认失败**

```bash
uv run pytest tests/test_scripts_entry.py -v
```

Expected: FAIL（`scripts/train.py` 不存在）。

- [ ] **Step 9.3: 创建 `scripts/train.py`**

新建 `scripts/train.py`：

```python
"""触发 tactile_grasp 任务注册，然后调 mjlab 的训练入口。"""

from __future__ import annotations

import tactile_grasp  # noqa: F401 -- import side-effect triggers task registration

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 9.4: 创建 `scripts/play.py`**

新建 `scripts/play.py`：

```python
"""触发 tactile_grasp 任务注册，然后调 mjlab 的 play 入口。"""

from __future__ import annotations

import tactile_grasp  # noqa: F401

from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 9.5: 删除 `scripts/train_ppo.py`**

```bash
rm scripts/train_ppo.py
```

- [ ] **Step 9.6: 跑 wrapper 测试**

```bash
uv run pytest tests/test_scripts_entry.py -v
```

Expected: PASS。

- [ ] **Step 9.7: 跑训练 smoke（10 iter）**

```bash
WANDB_MODE=offline uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 10
```

Expected: 训练跑完 10 iter 不报错；日志目录形如 `logs/rsl_rl/tactile_grasp/<timestamp>/`。

（如果 mjlab.scripts.train.main 在调用前期已 raise，先 `uv run python scripts/train.py --help` 看 tyro 帮助文本是否正常列出。如确需 wandb login，跳过这步、由用户手动验证。）

- [ ] **Step 9.8: 跑 play smoke**

```bash
uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85 --agent zero --num-envs 1
```

Expected: 加载零策略 rollout 一段。如果需要 checkpoint 才能跑，跳过这步、由用户手动验证。

- [ ] **Step 9.9: 跑全套验证**

```bash
uv run pytest tests/ -q
uv run python scripts/smoke_env.py
```

Expected: 全绿。

- [ ] **Step 9.10: 更新 `README.md`**

打开 `README.md`，把"快速开始"段中的 `uv run python main.py` 增补 train/play 命令：

```bash
uv sync --extra cu128 --group dev

# 极简 smoke
uv run python scripts/smoke_env.py

# 训练
uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 100

# 回放
uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85
```

把"详见 plan.md"改为"详见 `docs/superpowers/specs/2026-05-22-mjlab-idiom-refactor-design.md`"。

- [ ] **Step 9.11: Commit**

```bash
git add -A
git commit -m "feat: scripts/{train,play}.py 极简 wrapper（调 mjlab.scripts.train），删 train_ppo.py"
```

**回滚条件：** train.py 调用 `mjlab.scripts.train.main` 报缺字段 → 查最新 mjlab API 是否变化；`git restore .`。

---

## 验收（按 spec §12 全量复核）

完成全部 9 个 commit 后，跑：

```bash
# 1. 目录结构
test ! -d src/contactile_mjlab && test -d src/tactile_grasp && echo "目录结构 OK"

# 2. 测试全绿
uv run pytest tests/ -q
# Expected: 7 test files 全 PASS（test_asset_paths / test_no_touchsite / test_package_layout /
#           test_mdp_layout / test_observation_shapes / test_env_cfg_idiom / test_rl_cfg /
#           test_scripts_entry / test_inspect_pts_frames_cli）

# 3. train smoke
WANDB_MODE=offline uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 10
# Expected: 完整跑通；日志目录 logs/rsl_rl/tactile_grasp/<timestamp>/

# 4. play smoke
uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85 --agent zero
# Expected: 加载并 rollout 不报错

# 5. pip install -e 验证 package-data
uv pip install -e .
uv run python -c "import tactile_grasp; env = tactile_grasp.make_env(); env.reset(); print('package-data OK')"

# 6. 观测维度
uv run python -c "from tactile_grasp import make_env; env = make_env(); obs, _ = env.reset(); print(obs['actor'].shape)"
# Expected: torch.Size([1, 320])

# 7. 无 TouchSite 残留
grep -rn "TouchSite\|TOUCH_SITE\|touch_map\|TACTILE_MODEL\|tactile_model" src/ scripts/ tests/ main.py | grep -v 'TACTILE_XML\|TACTILE_SCENE_XML'
# Expected: 无命中

# 8. commit 数
git log --oneline | head -10
# Expected: 看到 9 个本次重构的 commit + 之前的 spec commit
```

---

## 已知风险与变更点

| 风险 | 缓解 |
|---|---|
| `taxel_normal_force` / `taxel_tangential_force` 假设 sensor 返回 (N, 3) — 实际若是 (N,) 或其他 shape，Task 6 维度测试会 fail | Task 6 Step 6.6 即时 print 排查；若 sensor 输出 1D，需调整 `_stack_taxel_force` 逻辑 |
| `RslRlOnPolicyRunnerCfg` 字段未必有 `wandb_project` | Task 8 Step 8.3 包含字段探测命令 |
| Task 4 `sed` 批量替换可能误伤注释/字符串 | 4.12 强调 `git diff` 人工核对 |
| `mjlab.scripts.train` 的 tyro CLI 解析 task_id 可能与本地手写 wrapper 行为差异 | Task 9 Step 9.7 实际跑一次 |
| Task 7 `make_env` 接口变更后 `main.py` / `view_env.py` 行为可能微妙变化 | 7.5 显式更新；7.7 跑 main 验证 |

---

## 下一步

完成本计划后，可考虑：

1. 跑一次完整 baseline 训练（500-1000 iter）评估当前观测设计是否能学到稳定抓持
2. 按 spec §10 进入 V2 设计（7-DoF 末端动作 + 桌面物体 + mocap floating gripper）
3. 按 spec §11.1 处理 `object_drop` 不可达问题（V2 场景重设计时一并）
