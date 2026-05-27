# AGENTS

## 角色

本文件是仓库的协作契约，收集 agent 和人在动手前都应该知道、且不容易从代码里直接读出来的东西。

不是研究方案，不是实现进度看板，不是设计文档索引：

- 当前实现状态（task ID、观测维度、history 长度等会随调参改动的事实）：看 `docs/source/design/task_architecture.rst` 及 `docs/source/design/` 下各专题页
- 安装与运行命令：看 `docs/source/install.rst`、`docs/source/usage.rst`

写每一条之前的判断标准：**会随某次 PR 漂掉的事实不要写进 AGENTS.md。** 把会漂的事实留给代码、design 文档或 `pyproject.toml`。

---

## 1. 当前阶段目标

当前主线目标：建立一个**可验证、可训练的 MuJoCo tactile gripper baseline**，不是完整 sim2real 系统。

阶段验收顺序（前一项不达标，后一项不算）：

1. MJCF 可加载
2. 传感器读数合理
3. 环境 API 稳定
4. 训练脚本可跑通

阶段切换以 PPO baseline 在当前 task 上**稳定可复现地收敛**为准。

---

## 2. 永久硬约束（未来阶段也不解除）

每条都附 "why"，未来想动这些时先回到"为什么当初这样定"。

### 2.1 观测真值边界

**Policy observation 不能依赖 MuJoCo 内部接触真值。**

- Why：仿真才有的真值在实机没有；让 policy 依赖它等于把 sim2real gap 写死。
- 真值可以用于 reward、termination、debug，不能进 policy 输入。

### 2.2 控制抽象

**保持 Robotiq 2F-85 的真实控制接口（位置命令 `u ∈ [0, 255]`），不引入理想 force actuator 替代。**

- Why：实机上用的就是这个接口；引入理想 actuator 会让 policy 学到无法迁移的力控行为。

### 2.3 资源文件

**不覆盖 `assets/robotiq_2f85/2f85.xml` 等上游原始 XML。**

- Why：保留上游可追溯版本，方便升级和对照。
- 触觉变体派生为独立文件，例如 `assets/robotiq_2f85/2f85_tactile.xml`、`2f85_pts_spheres.xml`。

### 2.4 流程纪律

**design 文档是路线，不是实现规范原样执行。**

- Why：design 描述目标，实现要根据当前阶段和已有代码重新判断。
- **行为变更必须同步对应文档。** 不能只改代码不更新 design 或 usage 文档。

---

## 3. 当前阶段范围边界（不在 scope）

下面这些**当前阶段不做**；要做时另起 task 或新阶段，不要扩当前 task。

- 整臂 / Franka / 笛卡尔 IK / 6DoF 控制
- 同时维护多种触觉传感器变体（当前只 PTS spheres）
- 完整 sim2real 部署（实时环、完整域随机化）

---

## 4. 阶段性约束（带解禁条件）

和永久约束分开：未来满足条件后会解除。条件不满足前，agent 不要主动放开。

| 当前不做 | 解禁条件 |
|---|---|
| 动作扩成 `[Δu, Δforce_limit]` | 一维 `Δu` baseline 稳定收敛后讨论 |
| pad 切多个碰撞 geom | 出现接触抖动且诊断证明是 geom 粒度问题 |
| 当前 task 范围外的物体或抓取姿态 | 当前 task 验证可复现收敛后再扩 |
| 引入新的传感器变体 | 现有 PTS spheres 实现稳定且确有对比需求 |

---

## 5. 工作流契约

这一块是 agent 最容易踩的部分。

### 5.1 工具链

- **Python 依赖与运行一律走 `uv`。** 不直接调用系统 `python` / `pip` / `pytest`。
- 安装：
  - CPU：`uv sync --extra cpu --group dev`
  - CUDA：`uv sync --extra <cuda-tag> --group dev`，`<cuda-tag>` 以 `pyproject.toml [project.optional-dependencies]` 当前提供的为准
- 运行 Python 时若本机 `PYTHONPATH` 被 ROS 等污染，前缀清空：`PYTHONPATH= uv run python ...`。

### 5.2 Git / Commit

- **不绕过 pre-commit hooks。** 不用 `--no-verify`、`--no-gpg-sign` 等。失败就修底层问题。
- 钩子失败时**新建一次 commit** 修复，不要 `git commit --amend`（pre-commit 失败时 commit 没真正发生，amend 会改前一次提交）。
- 每个独立改动一次 commit，便于 review 和回滚。

### 5.3 Docstring 与注释

- 公开模块 / 类 / 函数写 docstring，遵循 **Google 中文风格**：节名（`Args` / `Returns` / `Raises` / `Attributes`）保持英文 Google 写法，内容可中文。
- docstring 第一行结尾用**英文 `.`**，不用中文 `。`（`ruff D415` 会卡）。
- 行内注释默认中文，简洁，只解释"为什么"或"非直观约束"，不重复代码字面含义。

示例：

```python
def step(action: torch.Tensor) -> torch.Tensor:
    """执行一步环境推进并返回奖励.

    Args:
        action: 策略输出的一步动作，范围为 ``[-1, 1]``.

    Returns:
        当前环境步对应的奖励张量.
    """
```

### 5.4 验证矩阵

按"改了什么"分类，至少跑对应这一格再开 PR。具体 task ID 和 smoke 命令以 `docs/source/usage.rst` 为准，不要照抄本文件。

| 你改了 | 至少跑 |
|---|---|
| `assets/**/*.xml` 或 MJCF | `uv run python scripts/check_mjcf.py`；必要时 `uv run python scripts/view_env.py` |
| `src/tactile_grasp/mdp/**` | `uv run python scripts/smoke_env.py --steps 40` + `uv run pytest tests/` |
| `src/tactile_grasp/env_cfgs.py` / `robot_cfg.py` | smoke_env 同上 + `uv run pytest tests/` |
| `src/tactile_grasp/rl_cfg.py` | `uv run pytest tests/test_rl_cfg.py` + 一个 2-iter 训练 smoke |
| `scripts/train.py` / `scripts/play.py` | `WANDB_MODE=offline uv run python scripts/train.py <task-id> --agent.max-iterations 2 --env.scene.num-envs 4 --env.episode-length-s 0.5 --gpu-ids None` |
| 其它 `.py` | `uv run ruff check .` + `uv run pytest tests/` |
| `pyproject.toml` 或依赖 | `uv lock --check` 后跑对应 extra 的 `uv sync` |
| `docs/source/**` | `uv run sphinx-build -b html docs/source docs/_build_docscheck` |
| `AGENTS.md` | 检查范围边界与代码现状一致 |

---

## 6. 信息源指针

只列"去哪里看当前状态"，不在本文件里复制状态本身。

| 想知道什么 | 看哪里 |
|---|---|
| 当前注册了哪些 task | `uv run python -c "from tactile_grasp import list_tasks; print(list_tasks())"` 或 `docs/source/design/task_architecture.rst` |
| 当前观测维度、history 长度、taxel 布局 | `src/tactile_grasp/env_cfgs.py` + `docs/source/design/tactile_pipeline.rst` |
| 当前 PPO 超参 | `src/tactile_grasp/rl_cfg.py` |
| 当前依赖版本、CUDA tag | `pyproject.toml` + `docs/source/install.rst` |
| 历史决策 | git log + PR description |

---

## 7. 代码组织约定

固定的目录责任划分（**大重构时可调，小改不应破坏**）：

- `assets/`：MJCF / XML / 贴图等仿真资源（变体派生，不覆盖原始）
- `src/tactile_grasp/`：包入口、env / robot / rl 配置
- `src/tactile_grasp/mdp/`：actions / observations / rewards / events / terminations 五块
- `scripts/`：调试、可视化、smoke、训练入口
- `tests/`：pytest 测试
- `docs/source/`：用户文档（`usage.rst` / `install.rst`）+ 设计文档（`design/`）+ API（`api/`）

包根（`src/tactile_grasp/__init__.py`）重导出：`make_env` / `load_env_cfg` / `load_rl_cfg` / `load_runner_cls` / `list_tasks` / `register_mjlab_task` / `TASK_ID`。

---

## 8. 元数据

- 本文件 last review：2026-05-23（task-based API 重构完成）
- 与上游耦合点（升级时这里要检查）：
  - mjlab task registry API：`register_mjlab_task` / `load_env_cfg`
  - `TransmissionType` enum 位置（`mjlab.entity.entity`）
  - `scripts/train.py` 使用的 tyro CLI 子命令风格
