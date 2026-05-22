# mjlab 范式重构设计

- 日期：2026-05-22
- 状态：草案，待 review
- 范围：把 `contactile_mjlab` 包整体重构为 `tactile_grasp`，对齐 mjlab 上游惯例；丢弃 TouchSite 代码路径（保留资产）；引入 mjlab 原生 history/delay；用 `mjlab.scripts.train` 作训练入口

---

## 1. 目标与非目标

### 1.1 目标

1. 把当前 `contactile_mjlab` 包的目录结构和装配方式对齐 mjlab 上游惯例（参考 `mjlab.tasks.velocity` / `mjlab.tasks.tracking`），让 mjlab 用户能无障碍读懂本仓库
2. 删除 task-based 重构前的死代码（`mjlab/tactile_grasp_env.py`、`mjlab/mdp.py`）
3. 丢弃 TouchSite 代码路径（task_id、常量、观测函数、env_cfg 分支），但保留 TouchSite XML 资产以备未来重启对照实验
4. 把训练入口从自实现的 argparse 切到 mjlab 上游 `mjlab.scripts.train`，获得 tyro CLI 全字段覆写、标准日志目录、checkpoint resume、wandb 集成、多 GPU 支持
5. 引入 mjlab 原生 `ObservationTermCfg.history_length`，给触觉信号加 100ms 时间窗口
6. 把单一 27-d taxel 力观测按物理意义拆成 normal（9-d）+ tangential（18-d），为 V2 slip proxy 留接口
7. 资产从 repo 根 `assets/` 迁入 `src/tactile_grasp/assets/`，让包可 pip-install

### 1.2 非目标

- **不**做 6-DoF 末端控制（V2 工作，本次重构只为它留前向兼容性）
- **不**做 sim2real 噪声/延迟实测调优（V3 工作）
- **不**重新设计奖励函数体系（保留当前 5 个 reward term，只做命名/拆分整理）
- **不**修复 `object_drop` termination 实际不可达的问题（V2 场景重设计时一并处理）
- **不**做 RL 算法替换或网络架构升级（保留 PPO + MLP，仅调超参默认值）

---

## 2. 当前状态摘要

### 2.1 现有目录结构

```
src/contactile_mjlab/
├── __init__.py
├── control.py                          # GripperCommandBuffer
├── paths.py                            # 路径常量（指向 repo 根 assets/）
├── envs/
│   └── __init__.py                     # 5 行 shim：re-export make_env
├── mjlab/
│   ├── __init__.py                     # re-export 死代码
│   ├── action_terms.py                 # RobotiqCommandAction(Cfg)  [实际在用]
│   ├── actuators.py                    # RobotiqGeneralActuatorCfg  [实际在用]
│   ├── mdp.py                          # 165 行死代码
│   └── tactile_grasp_env.py            # 325 行死代码（task-based 前的旧实现）
└── tasks/
    ├── __init__.py
    └── tactile_grasp/
        ├── __init__.py                 # register_tasks() + load_env_cfg(override 白名单)
        ├── constants.py                # task id / sensor / 阈值表
        ├── env_cfg.py                  # TactileGraspTaskConfig dataclass-builder
        ├── object_cfg.py
        ├── reward_terms.py             # reward + termination 混在一起
        ├── rl_cfg.py
        ├── robot_cfg.py
        └── tactile_terms.py            # touch_map / taxel_force_map / pad_wrench / gripper_command
```

### 2.2 当前与 mjlab 惯例的偏离

- 任务注册被 `register_tasks()` 包裹 + `if already in list_tasks(): continue` 保护，而非顶层模块直接调
- env cfg 用 `TactileGraspTaskConfig` dataclass-builder 模式，不是上游 `make_xxx_env_cfg(play=False)` 函数模式
- `load_env_cfg` 用 override 白名单限制可改字段（违反"任意 cfg 字段都可 CLI 覆写"的核心思想）
- 自实现 `scripts/train_ppo.py`（argparse），未走 `mjlab.scripts.train`
- mdp terms 没按 mjlab 惯例 `mdp/{actions,observations,rewards,events,terminations}.py` 拆分
- 资产位于 repo 根，不在包内，pip install 后找不到

---

## 3. 目标架构

### 3.1 目录结构

```
src/tactile_grasp/
├── __init__.py            # 顶层 register_mjlab_task(...) + re-export
├── env_cfgs.py            # make_tactile_grasp_env_cfg(play=False) -> ManagerBasedRlEnvCfg
├── rl_cfg.py              # tactile_grasp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg
├── robot_cfg.py           # build_robot_cfg(): Robotiq 2F-85 PTS spheres
├── object_cfg.py          # build_object_cfg(): hanging_box
├── constants.py           # TASK_ID / 关节名 / 传感器名 / 阈值（单值，不再按 task 查表）
├── control.py             # GripperCommandBuffer（保留）
├── paths.py               # ASSETS_DIR = Path(__file__).parent / "assets"
├── mdp/
│   ├── __init__.py        # 薄壳：按子模块导入，不 re-export
│   ├── actions.py         # RobotiqCommandActionCfg / RobotiqCommandAction
│   ├── actuators.py       # RobotiqGeneralActuatorCfg
│   ├── observations.py    # taxel_normal_force / taxel_tangential_force /
│   │                      # pad_force / pad_torque / gripper_command
│   ├── rewards.py         # alive / tactile_force_l2 / action_l2 /
│   │                      # close_command_l2 / drop_penalty
│   ├── events.py          # reset_scene_to_default (re-export from mjlab)
│   └── terminations.py    # object_height_below / stable_grasp_hold
└── assets/
    ├── robotiq_2f85/
    │   ├── 2f85.xml                      # 基础模型（保留）
    │   ├── 2f85_tactile.xml              # TouchSite 资产（保留，代码不再引用）
    │   ├── scene_tactile.xml             # 同上（保留）
    │   ├── 2f85_pts_spheres.xml          # 主线
    │   └── scene_pts_spheres.xml
    └── props/
        └── hanging_box.xml
```

### 3.2 关键架构决策

1. **没有 `tasks/<name>/` 子目录**：本包只服务一个任务族，注册写在顶层 `__init__.py`。未来加第二个任务再升级到 `tasks/<name>/` 结构。
2. **`env_cfgs.py` 用 `make_xxx_env_cfg(play=False)` 函数模式**，弃用 `TactileGraspTaskConfig` dataclass-builder。`play=True` 在共享基底上 mutate（参考 g1 `unitree_g1_rough_env_cfg`）。
3. **`mdp/` 按 mjlab 惯例 5 文件分割**：actions / observations / rewards / events / terminations。`env_cfgs.py` 通过 `from .mdp import observations, rewards, events, terminations, actions` 拼装。
4. **`mdp/__init__.py` 薄壳**：不 re-export，避免命名冲突和循环导入。
5. **任务注册无包裹**：顶层 `__init__.py` 直接调一次 `register_mjlab_task(...)`，不需 `register_tasks()` 函数和重复保护。
6. **`load_env_cfg` 去掉 override 白名单**：直接返回 `deepcopy`，由 tyro CLI 在调用方按字段覆写。
7. **Task ID 改名**：`Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres` → `Mjlab-TactileGrasp-Robotiq2F85`（删后缀，已是唯一变体）。
8. **资产迁入包内**：`src/tactile_grasp/assets/`；`paths.py` 改为 `Path(__file__).parent / "assets"`；`pyproject.toml` 加 `package-data` 包含 XML/STL/OBJ。

---

## 4. `env_cfgs.py` 内部结构

### 4.1 顶层模块常量（取代 dataclass 字段）

```python
# ---------- 控制 / 仿真 ----------
DECIMATION = 10
TIMESTEP = 0.002                  # 500 Hz 物理，50 Hz 控制 + 观测
EPISODE_LENGTH_S = 3.0
PLAY_EPISODE_LENGTH_S = 6.0

# ---------- 动作 ----------
DELTA_U_MAX = 3.0                 # Robotiq 单步命令增量上限

# ---------- 观测缩放 ----------
NORMAL_FORCE_SCALE = 5.0          # 5 N → 归一化 1.0
TANGENTIAL_FORCE_SCALE = 2.0      # 切向幅值典型小于法向
FORCE_SCALE = 20.0                # 指尖聚合 force（pad_force）
TORQUE_SCALE = 2.0                # 指尖聚合 torque（pad_torque）

# ---------- 时间窗口 ----------
TACTILE_HISTORY_LENGTH = 5        # 100 ms @ 50 Hz：覆盖 slip onset
WRENCH_HISTORY_LENGTH = 3         # 60 ms：聚合 wrench 的瞬态

# ---------- 环境数 / 间距 ----------
NUM_ENVS = 64
PLAY_NUM_ENVS = 1
ENV_SPACING = 0.5

# ---------- 成功 / 失败阈值 ----------
DROP_HEIGHT = 0.08
SUCCESS_HEIGHT = 0.14
SUCCESS_HOLD_STEPS = 25
TACTILE_ACTIVITY_THRESHOLD = 1.0e-3

# ---------- 奖励权重 ----------
W_ALIVE = 1.0
W_TACTILE_FORCE = -0.01
W_ACTION_RATE = -0.001
W_CLOSE_COMMAND = -0.001
W_DROP_PENALTY = -5.0
```

集中放置让 tyro CLI 覆写不变（`--env.rewards.alive.weight ...`），但默认值集中可见。

### 4.2 函数签名与装配骨架

```python
def make_tactile_grasp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    actor_terms = _actor_observation_terms()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": build_robot_cfg(), "object": build_object_cfg()},
            num_envs=NUM_ENVS,
            env_spacing=ENV_SPACING,
        ),
        observations={
            "actor":  ObservationGroupCfg(actor_terms,       enable_corruption=False),
            "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
        },
        actions={"gripper_command": build_gripper_action_cfg(DELTA_U_MAX)},
        events={"reset_scene_to_default": EventTermCfg(func=events.reset_scene_to_default, mode="reset")},
        rewards={
            "alive":         RewardTermCfg(func=rewards.alive,            weight=W_ALIVE),
            "tactile_force": RewardTermCfg(func=rewards.tactile_force_l2, weight=W_TACTILE_FORCE,
                                            params={"left_sensor_names":  LEFT_TAXEL_FORCE_SENSOR_NAMES,
                                                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES}),
            "action_rate":   RewardTermCfg(func=rewards.action_l2,        weight=W_ACTION_RATE),
            "close_command": RewardTermCfg(func=rewards.close_command_l2, weight=W_CLOSE_COMMAND),
            "drop_penalty":  RewardTermCfg(func=rewards.drop_penalty,     weight=W_DROP_PENALTY),
        },
        terminations={
            "time_out":     TerminationTermCfg(func=mdp_builtin.time_out, time_out=True),
            "object_drop":  TerminationTermCfg(func=terminations.object_height_below,
                                                params={"minimum_height": DROP_HEIGHT, "asset_cfg": OBJECT_CFG}),
            "stable_grasp": TerminationTermCfg(func=terminations.stable_grasp_hold,
                                                params={"hold_steps":             SUCCESS_HOLD_STEPS,
                                                        "minimum_height":         SUCCESS_HEIGHT,
                                                        "minimum_tactile_signal": TACTILE_ACTIVITY_THRESHOLD,
                                                        "left_sensor_names":      LEFT_TAXEL_FORCE_SENSOR_NAMES,
                                                        "right_sensor_names":     RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                                                        "asset_cfg":              OBJECT_CFG}),
        },
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=TIMESTEP, cone="elliptic", impratio=10.0)),
        viewer=ViewerConfig(),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        auto_reset=True,
    )

    if play:
        cfg.scene.num_envs = PLAY_NUM_ENVS
        cfg.episode_length_s = PLAY_EPISODE_LENGTH_S
        cfg.observations["actor"].enable_corruption = False
        # 未来加 randomization 后这里 pop("push_object") 等

    return cfg
```

### 4.3 顶层任务注册（`__init__.py`）

```python
"""tactile_grasp: Robotiq 2F-85 + PTS spheres 触觉抓取任务包。"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
    register_mjlab_task,
)

from .constants import TASK_ID
from .env_cfgs import make_tactile_grasp_env_cfg
from .rl_cfg import tactile_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_tactile_grasp_env_cfg(play=False),
    play_env_cfg=make_tactile_grasp_env_cfg(play=True),
    rl_cfg=tactile_grasp_ppo_runner_cfg(),
    runner_cls=None,
)


def make_env(*, play: bool = False, device: str = "cpu", render_mode: str | None = None):
    """便利函数：直接构造环境实例。"""
    cfg = load_env_cfg(TASK_ID, play=play)
    return ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)


__all__ = ["TASK_ID", "make_env", "load_env_cfg", "load_rl_cfg", "load_runner_cls"]
```

---

## 5. `mdp/` 模块切分

### 5.1 内容映射

| 文件 | 内容 | 来源 |
|---|---|---|
| `mdp/actions.py` | `RobotiqCommandActionCfg`, `RobotiqCommandAction` | ← `mjlab/action_terms.py`（整体迁移） |
| `mdp/actuators.py` | `RobotiqGeneralActuatorCfg` | ← `mjlab/actuators.py`（整体迁移） |
| `mdp/observations.py` | `taxel_normal_force`, `taxel_tangential_force`, `pad_force`, `pad_torque`, `gripper_command` | ← `tasks/tactile_grasp/tactile_terms.py`（删 `touch_map`；`pad_wrench` 拆 → `pad_force` + `pad_torque`；`taxel_force_map` 拆 → `taxel_normal_force` + `taxel_tangential_force`） |
| `mdp/rewards.py` | `alive`, `tactile_force_l2`, `action_l2`, `close_command_l2`, `drop_penalty`（新命名） | ← `tasks/tactile_grasp/reward_terms.py`（reward 函数部分） |
| `mdp/terminations.py` | `object_height_below`, `stable_grasp_hold` | ← `tasks/tactile_grasp/reward_terms.py`（termination helper 部分） |
| `mdp/events.py` | `reset_scene_to_default` re-export | 新建 |

### 5.2 触觉观测函数签名

```python
# mdp/observations.py

def taxel_normal_force(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """每个 taxel 的法向力（z 分量），共 9 维 / pad。"""
    ...

def taxel_tangential_force(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """每个 taxel 的切向力（xy 分量），共 18 维 / pad。"""
    ...

def pad_force(env, sensor_name: str) -> torch.Tensor:
    """指尖聚合 3D 力。"""
    ...

def pad_torque(env, sensor_name: str) -> torch.Tensor:
    """指尖聚合 3D 力矩。"""
    ...

def gripper_command(env) -> torch.Tensor:
    """当前 Robotiq 命令归一化值（来自 RobotiqCommandAction 内部状态）。"""
    ...
```

坐标系：taxel 局部坐标系（提交 a66d517 已对齐到实物 PTS 约定，z=法向，xy=切向）。

### 5.3 `drop_penalty` 提升为命名函数

当前 `env_cfg.py:169` 是 `lambda env: env.termination_manager.get_term("object_drop").float()`，闭包字符串引用 termination 名。重构后：

```python
# mdp/rewards.py
def drop_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """物体掉落 termination 命中时的负奖励通道。"""
    return env.termination_manager.get_term("object_drop").float()
```

好处：可 pickle、可 tyro CLI 引用、错误信息友好。

### 5.4 ObservationTermCfg history 装配

```python
def _actor_observation_terms() -> dict[str, ObservationTermCfg]:
    return {
        "left_taxel_normal":     ObservationTermCfg(func=observations.taxel_normal_force,
                                                     params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
                                                     scale=1.0 / NORMAL_FORCE_SCALE,
                                                     history_length=TACTILE_HISTORY_LENGTH),
        "left_taxel_tangential": ObservationTermCfg(func=observations.taxel_tangential_force,
                                                     params={"sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES},
                                                     scale=1.0 / TANGENTIAL_FORCE_SCALE,
                                                     history_length=TACTILE_HISTORY_LENGTH),
        "right_taxel_normal":     ObservationTermCfg(... history_length=TACTILE_HISTORY_LENGTH),
        "right_taxel_tangential": ObservationTermCfg(... history_length=TACTILE_HISTORY_LENGTH),
        "left_force":  ObservationTermCfg(func=observations.pad_force,  params={"sensor_name": "left_pad_force"},
                                          scale=1.0 / FORCE_SCALE,  history_length=WRENCH_HISTORY_LENGTH),
        "left_torque": ObservationTermCfg(func=observations.pad_torque, params={"sensor_name": "left_pad_torque"},
                                          scale=1.0 / TORQUE_SCALE, history_length=WRENCH_HISTORY_LENGTH),
        "right_force":  ObservationTermCfg(... history_length=WRENCH_HISTORY_LENGTH),
        "right_torque": ObservationTermCfg(... history_length=WRENCH_HISTORY_LENGTH),
        "gripper_command": ObservationTermCfg(func=observations.gripper_command),                     # history_length=0
        "joint_pos":       ObservationTermCfg(func=mdp_builtin.joint_pos_rel, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "joint_vel":       ObservationTermCfg(func=mdp_builtin.joint_vel_rel, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "last_action":     ObservationTermCfg(func=mdp_builtin.last_action),
    }
```

观测维度对比：

| 类别 | 单帧 | 加 history 后 |
|---|---|---|
| Taxel normal (两边) | 18 | 90 |
| Taxel tangential (两边) | 36 | 180 |
| Pad force (两边) | 6 | 18 |
| Pad torque (两边) | 6 | 18 |
| Joint pos + vel | 12 | 12 |
| Gripper cmd + last action | 2 | 2 |
| **总计** | **80** | **320** |

`flatten_history_dim=True`（默认）→ 输出仍是 flat 向量，MLP 直接消费。

---

## 6. `rl_cfg.py` 与训练入口

### 6.1 PPO 默认值更新

| 字段 | 当前 | 新值 | 理由 |
|---|---|---|---|
| `actor/critic hidden_dims` | `(128, 128)` | `(256, 256)` | 320-d 输入，128 偏小；触觉 baseline 普遍 256+ |
| `obs_normalization` | `False` | `True` | 触觉力幅值跨数量级，running-mean 关键 |
| `num_steps_per_env` | `32` | `48` | 64 envs × 48 steps = 3072/iter，更稳 |
| `max_iterations` | `200` | `3000` | 200 iter 是 smoke 值，不能跑出 baseline |
| `experiment_name` | `"contactile_mjlab"` | `"tactile_grasp"` | 包名对齐 |
| `logger` | `"tensorboard"` | `"wandb"` | 用户偏好 |
| `wandb_project` | (未设) | `"tactile_grasp"` | |

保留：`clip_param=0.2`, `entropy_coef=0.01`, `gamma=0.99`, `lam=0.95`, `desired_kl=0.01`, `learning_rate=3e-4`, `schedule="adaptive"`, `value_loss_coef=1.0`, `use_clipped_value_loss=True`, `num_learning_epochs=5`, `num_mini_batches=4`, `max_grad_norm=1.0`, `save_interval=50`。

### 6.2 训练 / play 入口

`scripts/train.py`（新）：

```python
"""Train a tactile grasp policy via mjlab's training entrypoint."""
from __future__ import annotations

import tactile_grasp  # noqa: F401 -- import side-effect triggers task registration
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
```

`scripts/play.py`（新）：对称的 play 入口，调 `mjlab.scripts.play.main()`。

`scripts/train_ppo.py`：**删除**。

使用方式：

```bash
uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 \
    --env.scene.num-envs 128 \
    --env.rewards.tactile-force.weight -0.02 \
    --agent.max-iterations 5000

uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85
```

### 6.3 `pyproject.toml`

```toml
[project]
name = "tactile-grasp"      # ← was "contactile-mjlab"

[tool.setuptools.packages.find]
where = ["src"]
include = ["tactile_grasp*"]

[tool.setuptools.package-data]
tactile_grasp = [
    "assets/**/*.xml",
    "assets/**/*.stl",
    "assets/**/*.obj",
    "assets/**/*.png",
    "assets/**/*.mtl",
]
```

依赖（`mjlab`、`torch` 等）保留。

---

## 7. 迁移计划

### 7.1 删除清单

```
src/contactile_mjlab/                   ← 整个包目录删除（内容已迁）
├── mjlab/                              ← 子包删除
│   ├── __init__.py                     死代码
│   ├── tactile_grasp_env.py            死代码 (325 行)
│   ├── mdp.py                          死代码 (165 行)
│   ├── action_terms.py                 (迁 → mdp/actions.py)
│   └── actuators.py                    (迁 → mdp/actuators.py)
├── envs/                               5 行 shim
├── tasks/                              扁平化
│   ├── __init__.py
│   └── tactile_grasp/                  (内容迁 → src/tactile_grasp/)
├── __init__.py                         (重写)
├── control.py                          (迁 → tactile_grasp/control.py)
└── paths.py                            (迁 → tactile_grasp/paths.py)

scripts/train_ppo.py                    ← 删除（被 scripts/train.py 取代）
```

### 7.2 TouchSite 清理

**删除（代码层）**：
- `TOUCH_SITE_TASK_ID` / `TACTILE_MODEL_TOUCH_SITE`
- `LEFT_TOUCH_SENSOR_NAMES` / `RIGHT_TOUCH_SENSOR_NAMES`
- `TACTILE_ACTIVITY_THRESHOLD_BY_TASK` / `TACTILE_OBS_DIM_BY_TASK` 表（降级为单值常量）
- `touch_map` 函数
- `env_cfg.py` 的 `if tactile_model == TOUCH_SITE` 分支
- 所有脚本对 `TOUCH_SITE_TASK_ID` 的引用

**保留（资产层）**：
- `assets/robotiq_2f85/2f85_tactile.xml`
- `assets/robotiq_2f85/scene_tactile.xml`
- `assets/robotiq_2f85/2f85.xml`（基础模型）
- `paths.py` 中 `TACTILE_XML` / `TACTILE_SCENE_XML` 路径常量保留（调试脚本可能仍引用）

### 7.3 资产迁移

```
repo_root/assets/robotiq_2f85/*  →  src/tactile_grasp/assets/robotiq_2f85/*
repo_root/assets/props/*         →  src/tactile_grasp/assets/props/*
repo_root/assets/                ←  迁完删除
```

`paths.py` 重写：

```python
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
ROBOTIQ_DIR = ASSETS_DIR / "robotiq_2f85"
PROPS_DIR = ASSETS_DIR / "props"

# 主线
PTS_SPHERES_XML = ROBOTIQ_DIR / "2f85_pts_spheres.xml"
PTS_SPHERES_SCENE_XML = ROBOTIQ_DIR / "scene_pts_spheres.xml"
HANGING_BOX_XML = PROPS_DIR / "hanging_box.xml"

# 保留以备未来调试 / V0 对照
BASE_XML = ROBOTIQ_DIR / "2f85.xml"
TACTILE_XML = ROBOTIQ_DIR / "2f85_tactile.xml"
TACTILE_SCENE_XML = ROBOTIQ_DIR / "scene_tactile.xml"
```

XML 内部相对路径（`<mesh file="..."/>`、`<include file="..."/>`）保持不变（相对 XML 本身）。

### 7.4 调用方 import 更新

| 文件 | 改动 |
|---|---|
| `main.py` | `from contactile_mjlab import make_env` → `from tactile_grasp import make_env` |
| `scripts/smoke_env.py` | 包名替换 + 删 `TOUCH_SITE_TASK_ID` 引用 |
| `scripts/view_env.py` | 包名替换 + 删 `TACTILE_OBS_DIM_BY_TASK` 表用法 |
| `scripts/check_mjcf.py` | 包名替换 |
| `scripts/visualize_taxels.py` | 包名替换 |
| `scripts/inspect_pts_frames.py` | 包名替换 |
| `scripts/test_gripper_ctrl.py` | 包名替换 |
| `tests/test_inspect_pts_frames_cli.py` | 包名替换（逻辑不变） |

### 7.5 提交策略

按职责拆 9 个 commit，每个 commit 后跑对应验证关卡：

1. `chore: 资产从 repo 根迁入 package`
2. `refactor: 删除死代码 (mjlab/tactile_grasp_env.py, mjlab/mdp.py)`
3. `refactor: 删除 TouchSite 代码路径（资产保留）`
4. `refactor: 包重命名 contactile_mjlab → tactile_grasp，扁平化目录`
5. `refactor: mdp/ 拆分（actions/observations/rewards/events/terminations）`
6. `feat: 触觉观测拆 normal/tangential，加 history_length`
7. `feat: env_cfgs.py 改 mjlab idiom（make_xxx_env_cfg）`
8. `chore: rl_cfg.py 更新默认值（256x256, wandb, history-aware）`
9. `feat: scripts/{train,play}.py 极简 wrapper，删 train_ppo.py`

---

## 8. 测试与验证

### 8.1 新增测试

按职责分门别类：

| 文件 | 范围 |
|---|---|
| `tests/test_task_registration.py` | import 后 `list_tasks()` 包含 `TASK_ID`，`load_env_cfg` / `load_rl_cfg` 不报错 |
| `tests/test_env_smoke.py` | `make_env()` 成功、`reset() + step(zero_action) × 5` 不抛异常 |
| `tests/test_observation_shapes.py` | normal 9-d、tangential 18-d、history 后维度 = `K × D`；总观测维度 320 |
| `tests/test_asset_paths.py` | `ASSETS_DIR` 下所有 XML 存在且 MuJoCo 可加载 |
| `tests/test_inspect_pts_frames_cli.py` | 现有，仅包名替换 |

### 8.2 验证关卡（按顺序）

1. **Import smoke**：`uv run python -c "import tactile_grasp; print(tactile_grasp.TASK_ID)"`
2. **Env smoke**：`uv run python scripts/smoke_env.py`
3. **View smoke**：`uv run python scripts/view_env.py`（人眼确认场景渲染）
4. **Train smoke**：`uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 10`
5. **Play smoke**：`uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85`
6. **测试套件**：`uv run pytest tests/`

---

## 9. 频率与时间窗口

### 9.1 仿真侧

| 层 | 频率 |
|---|---|
| 物理推进（`timestep=0.002`） | 500 Hz |
| MuJoCo 传感器内部刷新 | 500 Hz（每物理步） |
| 观测计算 + history push（`decimation=10`） | 50 Hz |
| 控制下发 | 50 Hz |

`TACTILE_HISTORY_LENGTH=5` × 20 ms = **100 ms 触觉时间窗口**（覆盖典型 slip onset 时间尺度）。

### 9.2 真机侧

**真实 Contactile PTS 最低发布频率 = 100 Hz**（用户确认）。

### 9.3 频率匹配策略（V1 选项 i）

V1 部署时把真机 PTS 降采样到 50 Hz（每 2 帧取 1 帧），喂给 50 Hz 训练的策略。100 ms 时间窗口在两端一致，策略可迁移。

**已知信息损失**：真机 100 Hz 高频瞬态丢失。可接受，因为 50 Hz 已经覆盖 slip onset 主要时间尺度。

### 9.4 未来升级路径（V2/V3 sim2real）

**选项 (iii)**：自定义 observation term，控制周期仍 50 Hz、但触觉观测在两个控制步之间累积 2 帧物理状态（在 mjlab 内 hook MuJoCo 物理回调）。最贴真机频率匹配，实现复杂度中等。本次重构**不实现**，但 mdp/observations.py 函数签名保持开放，未来加新 term 时不需要改其它代码。

---

## 10. V2 前瞻（不在本次重构范围内）

### 10.1 V2 任务定义

- **动作**：7-D（dx, dy, dz, dyaw, dpitch, droll, du），SE(3) 末端位姿增量 + 1-D 指尖命令
- **物体**：桌面/地面放置，多形状（方块/圆柱/圆球），尺寸/质量/摩擦/初始位姿随机
- **任务**：approach → grasp → lift stably → 维持目标高度
- **场景**：去掉 tendon-hung，引入桌面 + mocap 驱动的夹爪 floating body

### 10.2 复用清单（本次重构投资的回报）

- 所有 `mdp/` 子模块（actions, observations, rewards, events, terminations）
- Taxel normal/tangential 拆分
- ObservationTermCfg history_length 机制
- PPO 配置框架（仅调超参）
- mjlab.scripts.train 入口

### 10.3 V2 新增清单（不在本次范围）

- `GripperPoseActionCfg`（6-D mocap pose 增量）
- `approach_distance` / `object_pose` observation terms
- `lift_height_reward` / `approach_reward` reward terms
- `lift_height_reached` termination
- 新 scene XML（桌面 + mocap 夹爪）
- Object 随机化 events（`randomize_object_shape`, `randomize_object_pose`）

---

## 11. 已知限制与延后项

### 11.1 `object_drop` termination 当前不可达

`hanging_box.xml` 的 tendon `range="0 0.02"` 限制物体最低高度 ≈ 0.16 m，**永远不会触发 `DROP_HEIGHT=0.08`**。结果是 `drop_penalty` 通道权重 `-5.0` 但实际不激活。

**本次重构不修**，理由：
- V2 场景全面替换，drop termination 会自然重新设计
- 修了会改 V1 学习动态，干扰 baseline 评估
- 当前 `alive + tactile_force_l2 + action_l2 + close_command_l2` 仍构成完整学习信号

### 11.2 触觉观测频率 < 真机 PTS 频率

仿真 50 Hz vs 真机 ≥ 100 Hz。V1 部署降采样真机端，已在 §9.3 描述。V3 sim2real 阶段升级为 (iii)。

### 11.3 网络架构未优化

`hidden_dims=(256, 256)` 单一 MLP，无 CNN 编码 3×3 阵列、无 RNN 跨长时依赖。当前 V1 baseline 评估优先。后续若 baseline 不足，升级路径已开放（`RslRlModelCfg.class_name` 支持 MLPModel / CNNModel / RNNModel）。

---

## 12. 验收标准

本次重构验收满足以下全部：

1. `src/contactile_mjlab/` 目录不存在；`src/tactile_grasp/` 按 §3.1 结构存在
2. `tests/` 下 5 个测试文件全部通过（包括新增 4 个 + 现有 1 个）
3. `uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 10` 完整跑通，日志目录形如 `logs/rsl_rl/tactile_grasp/<timestamp>/`
4. `uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85` 能加载最新 checkpoint 并 rollout
5. `uv pip install -e .` 后能在另一个 Python session 里 `import tactile_grasp` 并 `make_env()` 成功（验证 assets package-data 生效）
6. 观测维度（含 history）= 320，分项符合 §5.4 表格
7. `grep -rn "TouchSite\|TOUCH_SITE\|touch_map\|tactile_model" src/ scripts/ tests/ main.py` 无命中（除 `paths.py` 里 `TACTILE_XML` / `TACTILE_SCENE_XML` 路径常量外）
8. `git log --oneline` 显示 9 个 commit 按 §7.5 顺序

---

## 13. 待执行步骤（重构开始时）

本设计批准后，由 writing-plans skill 转化为按 commit 排序的实施计划，每个 commit 包含：
- 改动的文件清单
- 验证命令
- 回滚条件
