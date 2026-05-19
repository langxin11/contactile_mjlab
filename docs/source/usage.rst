快速开始
========

环境创建
--------

.. code-block:: python

   from contactile_mjlab import TactileGraspEnv, TactileGraspEnvConfig

   config = TactileGraspEnvConfig(num_envs=4)
   env = TactileGraspEnv(config)

   obs, _ = env.reset()
   action = torch.zeros((env.num_envs, env.action_manager.total_action_dim))
   obs, reward, terminated, truncated, _ = env.step(action)

动作空间
--------

动作为一维连续标量，范围 ``[-1, 1]``，语义是 Robotiq 2F-85 的位置命令增量。

观测结构
--------

观测 ``obs["actor"]`` 由以下部分拼接：

- 左指 3×3 touch map（9 维）
- 右指 3×3 touch map（9 维）
- 左指全局 force（3 维）+ torque（3 维）
- 右指全局 force（3 维）+ torque（3 维）
- 归一化夹爪命令（1 维）
- 夹爪关节位置 + 速度
- 上一步动作（1 维）

训练
----

通过 ``tactile_grasp_ppo_runner_cfg`` 获取 PPO 训练配置：

.. code-block:: python

   from contactile_mjlab import tactile_grasp_ppo_runner_cfg

   cfg = tactile_grasp_ppo_runner_cfg()

调试脚本
--------

项目 ``scripts/`` 目录包含以下调试工具：

- ``smoke_env.py`` — 环境随机动作 smoke test
- ``test_gripper_ctrl.py`` — 单独测试夹爪控制回路
- ``visualize_taxels.py`` — 触觉阵列可视化
- ``check_mjcf.py`` — MJCF 模型加载检查
