快速开始
========

本页面向使用者，回答三件事：

1. 怎么创建并运行主线环境
2. 怎么训练 / play 主线 PPO 策略
3. 仓库里有哪些脚本可用于 smoke test、viewer 和坐标系检查

主线只有一个任务：``Mjlab-TactileGrasp-Robotiq2F85`` —— 单条 PTSSpheres
路径，per-taxel 三轴力按 normal / tangential 拆分后带 history。

环境创建
--------

最简单的方式是通过包顶层 ``make_env``：

.. code-block:: python

   import torch

   from tactile_grasp import make_env

   env = make_env(play=True)  # play=True → 1 env, episode_length 6 s

   obs, _ = env.reset()
   action = torch.zeros(
       (env.num_envs, env.action_manager.total_action_dim),
       device=env.device,
   )
   obs, reward, terminated, truncated, _ = env.step(action)

   print(obs["actor"].shape)  # torch.Size([1, 320])

如果只想确认环境能起起来：

.. code-block:: bash

   PYTHONPATH= uv run python scripts/smoke_env.py

直接走 viewer 做可视化检查（默认 ``--device cuda``，CPU 用户加 ``--device cpu``）：

.. code-block:: bash

   PYTHONPATH= uv run python scripts/view_env.py --device cpu

字段覆盖：cfg 后处理 idiom
--------------------------

``make_env`` 本身不接收字段覆盖；调用方拿到 cfg 后直接改字段，再交给
``ManagerBasedRlEnv``。这是 mjlab idiom：

.. code-block:: python

   from mjlab.envs import ManagerBasedRlEnv

   from tactile_grasp import TASK_ID, load_env_cfg

   cfg = load_env_cfg(TASK_ID, play=False)
   cfg.scene.num_envs = 16
   cfg.episode_length_s = 1.5
   cfg.auto_reset = True

   env = ManagerBasedRlEnv(cfg, device="cpu")

动作空间
--------

动作为一维连续标量，范围 ``[-1, 1]``，语义是 Robotiq 2F-85 的位置命令增量 ``Δu``：

.. code-block:: text

   u = clip(u + action * delta_u_max, 0, 255)

``delta_u_max`` 默认 ``3.0``。当前主线不包含 UR 机械臂控制、笛卡尔 IK
或 whole-arm 6DoF 动作。

观测结构
--------

``obs["actor"]`` 在主线任务下固定为 320 维，由以下项按字典顺序拼接（每项尾部都套了
mjlab ``history_length`` buffer）：

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - 观测项
     - 每步形状
     - history_length
     - 总贡献维度
   * - left_taxel_normal
     - ``[B, 9]``
     - 5
     - 45
   * - left_taxel_tangential
     - ``[B, 18]``
     - 5
     - 90
   * - right_taxel_normal
     - ``[B, 9]``
     - 5
     - 45
   * - right_taxel_tangential
     - ``[B, 18]``
     - 5
     - 90
   * - left_pad_force
     - ``[B, 3]``
     - 3
     - 9
   * - left_pad_torque
     - ``[B, 3]``
     - 3
     - 9
   * - right_pad_force
     - ``[B, 3]``
     - 3
     - 9
   * - right_pad_torque
     - ``[B, 3]``
     - 3
     - 9
   * - gripper_command
     - ``[B, 1]``
     - 1
     - 1
   * - joint_pos
     - ``[B, 6]``
     - 1
     - 6
   * - joint_vel
     - ``[B, 6]``
     - 1
     - 6
   * - last_action
     - ``[B, 1]``
     - 1
     - 1
   * - **合计**
     -
     -
     - **320**

Per-taxel 力在 site-local 坐标系中读取（``site.Z`` 为 pad 法向，
``site.X / site.Y`` 为切向），由 ``quat="1 0 -1 0"`` 在 XML 中固化。

训练 / Play
-----------

``scripts/train.py`` 与 ``scripts/play.py`` 只做一件事：``import tactile_grasp``
触发任务注册，然后转发到 ``mjlab.scripts.train.main`` / ``mjlab.scripts.play.main``。
所以所有命令行参数都遵循 mjlab CLI。

最小训练 smoke：

.. code-block:: bash

   WANDB_MODE=offline PYTHONPATH= uv run python scripts/train.py \
       Mjlab-TactileGrasp-Robotiq2F85 \
       --agent.max-iterations 100 \
       --gpu-ids None

GPU 训练（默认 0 号卡）：

.. code-block:: bash

   PYTHONPATH= uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85

Play 已保存的策略：

.. code-block:: bash

   PYTHONPATH= uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85

调试脚本
--------

``scripts/`` 目录还包含：

- ``smoke_env.py`` — 单步 reset + step smoke 测试
- ``view_env.py`` — 带 viewer 的随机动作可视化
- ``test_gripper_ctrl.py`` — 单独测试夹爪控制回路
- ``visualize_taxels.py`` — 触觉阵列可视化
- ``inspect_pts_frames.py`` — 用 ``mjviser`` 动态推进并查看 PTSSpheres 的
  pad / taxel site / sensor 局部坐标系
- ``generate_pts_spheres_xml.py`` — 从 ``2f85.xml`` 生成 ``2f85_pts_spheres.xml``
- ``check_mjcf.py`` — MJCF 模型加载检查

阅读路径
--------

- 想先跑起来：看 :doc:`install` 和本页
- 想理解实现：从 :doc:`design/task_architecture` 开始
- 想查代码 API：看 :doc:`api/index`
