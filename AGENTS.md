# AGENTS

## 角色

本文件是仓库协作规范，不是研究方案，不是实现进度看板，也不是设计文档索引。

- 路线与原则：看 `plan.md`
- 当前实现现状：看 `docs/source/design/task_architecture.rst`
- 具体实现细节：看 `docs/source/design/` 下各专题页

## 当前目标

当前主线目标是先建立一个可验证、可训练的 MuJoCo tactile gripper baseline，而不是立即做完整 sim2real 系统。

当前优先级固定为：

1. `MJCF` 可加载
2. 传感器读数合理
3. 环境 API 稳定
4. 训练脚本可跑通

## 主线实现

当前主线是 task-based API，不再以 legacy 单文件环境作为推荐入口。

- 包根入口：`tactile_grasp.make_env()`
- 环境配置入口：`tactile_grasp.load_env_cfg()`
- PPO 配置入口：`tactile_grasp.load_rl_cfg()`
- 主线实现目录：`src/tactile_grasp/`
- legacy 调试目录：`src/tactile_grasp/_mdp_legacy/`

当前注册任务：`Mjlab-TactileGrasp-Robotiq2F85`（PTS spheres 触觉传感器单一实现，TouchSite 路径已删除）。

## 动作与观测边界

当前动作空间固定为一维连续 `Δu`，范围 `[-1, 1]`。

- 语义：Robotiq 位置命令增量
- 内部命令：`u in [0, 255]`
- 当前不包含：`Franka` 机械臂控制、笛卡尔 `IK`、whole-arm `6DoF` 动作

如果未来做整臂抓取，那是后续阶段的新任务定义，不应混入当前 tactile gripper baseline。

观测与奖励边界：

- policy observation 不应依赖 MuJoCo 内部接触真值
- MuJoCo 内部真值可以用于 reward、termination 和 debug
- 当前触觉是 builtin `<force>` sensor 的 world-frame 三轴力直读
- 当前未实现 local-frame tactile、history、slip proxy

## 代码组织约束

- `assets/`：MJCF、XML、贴图等仿真资源
- `src/tactile_grasp/`：主线 task 配置、触觉、奖励、PPO 配置
- `src/tactile_grasp/_mdp_legacy/`：legacy 兼容与底层封装（Task 5 拆入正式 mdp/）
- `scripts/`：调试、可视化、smoke test、训练入口
- `docs/source/`：用户文档、设计文档、API 文档

修改时遵守这些约束：

- 不要覆盖原始 `assets/robotiq_2f85/2f85.xml`
- 触觉变体应派生为独立 XML，如 `2f85_tactile.xml`、`2f85_pts_spheres.xml`
- 优先做可验证的小步改动，不做一次性大重构
- 行为变更要同步更新设计文档，不要只改代码

## 运行方式

推荐使用 `uv`。

安装：

- CPU：`uv sync --extra cpu --group dev`
- CUDA 12.8：`uv sync --extra cu128 --group dev`

常用运行命令：

- 包加载检查：`uv run python main.py`
- MJCF 编译检查：`uv run python scripts/check_mjcf.py`
- 环境 smoke test：`uv run python scripts/smoke_env.py --steps 40`
- Viewer：`uv run python scripts/view_env.py --task-id Mjlab-TactileGrasp-Robotiq2F85 --device cpu`
- 最小训练：`uv run python scripts/train_ppo.py --task-id Mjlab-TactileGrasp-Robotiq2F85 --device cpu --num-envs 8 --episode-length-s 0.5 --max-iterations 1`

## 必做验证

每次重要改动后，至少按影响范围完成下面这些检查：

### 改了 XML / MJCF

- `uv run python scripts/check_mjcf.py`
- 必要时补跑 `uv run python scripts/view_env.py --task-id ...`

### 改了观测 / 奖励 / 环境配置

- `uv run python scripts/smoke_env.py --steps 40`
- 确认 observation shape、dtype、finite 状态合理

### 改了训练相关代码

- `uv run python scripts/train_ppo.py --task-id Mjlab-TactileGrasp-Robotiq2F85 --device cpu --num-envs 8 --episode-length-s 0.5 --max-iterations 1`

### 改了 Python 代码

- `uv run ruff check .`

### 改了文档

- `uv run sphinx-build -b html docs/source docs/_build_docscheck`

## 文档规范

文档分工保持稳定：

- `plan.md`：路线与原则
- `docs/source/usage.rst`：怎么运行、怎么切 task、怎么训练
- `docs/source/design/task_architecture.rst`：当前实现现状总览
- `docs/source/design/*.rst`：各子系统专题说明
- `docs/source/api/*.rst`：API 参考

不要把临时方案文档、聊天结论或实验记录直接当成实现规范。

## 注释与 Docstring 规范

仓库统一采用“Google 中文风格”：

- Python 的模块、类、公共函数优先写 docstring
- docstring 结构遵循 Google 风格
- docstring 内容说明可以用中文
- `Args`、`Returns`、`Raises`、`Attributes` 等节名保持 Google 风格写法
- 普通行内注释和块注释默认使用中文，要求简洁、只解释必要上下文
- 注释应解释“为什么”或“不直观的实现约束”，不要重复代码字面含义

推荐示例：

```python
def step(action: torch.Tensor) -> torch.Tensor:
    """执行一步环境推进并返回奖励。

    Args:
        action: 策略输出的一步动作，范围为 ``[-1, 1]``。

    Returns:
        当前环境步对应的奖励张量。
    """
```

当前工具链约束如下：

- `ruff` + `pydocstyle` 会检查 docstring 是否基本符合 Google 风格
- 当前不会自动检查“注释文本是否为中文”
- 因此“中文”部分靠仓库规范执行，“Google 结构”部分靠工具兜底

对应检查命令：

- `uv run ruff check .`

## 不要做的事

- 不要引入理想 force actuator 替代原始 2F-85 控制抽象
- 不要过早把 pad 切成多个碰撞 geom
- 不要在第一阶段把动作扩成 `[Δu, Δforce_limit]`
- 不要把当前夹爪任务直接改成 whole-arm `6DoF` 控制
- 不要把研究方案文档当成实现规范原样执行
- 不要改了行为却不更新对应文档和验证脚本
