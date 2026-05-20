快速开始
========

本页面向使用者，回答三件事：

1. 怎么创建并运行主线环境
2. 怎么在 ``PTSSpheres`` 和 ``TouchSite`` 之间切换
3. 仓库里有哪些脚本可用于 smoke test、viewer 和训练

环境创建
--------

默认主线任务为 ``PTSSpheres``（per-taxel 三轴力触觉）。通过 ``make_env`` 创建环境：

.. code-block:: python

   import torch

   from contactile_mjlab import DEFAULT_TASK_ID, PTS_SPHERES_TASK_ID, TOUCH_SITE_TASK_ID, make_env

   # 默认 PTSSpheres 任务
   env = make_env(DEFAULT_TASK_ID, play=True)

   # 回归对照 TouchSite 任务
   env = make_env(TOUCH_SITE_TASK_ID, play=True)

   obs, _ = env.reset()
   action = torch.zeros((env.num_envs, env.action_manager.total_action_dim))
   obs, reward, terminated, truncated, _ = env.step(action)

如果你只是想确认环境能起起来，优先用：

.. code-block:: bash

   uv run python main.py
   uv run python scripts/smoke_env.py --steps 40

动作空间
--------

动作为一维连续标量，范围 ``[-1, 1]``，语义是 Robotiq 2F-85 的位置命令增量。
当前主线不包含 ``UR`` 机械臂控制、笛卡尔 ``IK`` 或 whole-arm ``6DoF`` 动作。

观测结构
--------

默认主线任务 ``PTS_SPHERES_TASK_ID`` 的观测 ``obs["actor"]`` 由以下部分拼接：

- 左指 3×3 taxel 三轴力（27 维）
- 右指 3×3 taxel 三轴力（27 维）
- 左指全局 force（3 维）+ torque（3 维）
- 右指全局 force（3 维）+ torque（3 维）
- 归一化夹爪命令（1 维）
- 夹爪关节位置 + 速度
- 上一步动作（1 维）

对照任务 ``TOUCH_SITE_TASK_ID`` 使用 3×3 标量 touch map（18 维触觉），其余相同。

训练
----

通过 task id 加载 PPO 配置：

.. code-block:: python

   from contactile_mjlab import DEFAULT_TASK_ID, load_rl_cfg

   cfg = load_rl_cfg(DEFAULT_TASK_ID)

最小训练 smoke run：

.. code-block:: bash

   uv run python scripts/train_ppo.py \
     --task-id Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres \
     --device cpu \
     --num-envs 8 \
     --episode-length-s 0.5 \
     --max-iterations 1

调试脚本
--------

项目 ``scripts/`` 目录包含以下调试工具：

- ``smoke_env.py`` — 环境随机动作 smoke test
- ``test_gripper_ctrl.py`` — 单独测试夹爪控制回路
- ``visualize_taxels.py`` — 触觉阵列可视化
- ``check_mjcf.py`` — MJCF 模型加载检查
- ``view_env.py`` — 环境可视化
- ``train_ppo.py`` — PPO 训练入口

阅读路径
--------

- 想先跑起来：看 :doc:`install` 和本页
- 想理解实现：从 :doc:`design/task_architecture` 开始
- 想查代码 API：看 :doc:`api/index`
