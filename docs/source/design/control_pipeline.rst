控制与训练管线
==============

本文只讨论动作语义、Robotiq actuator 封装、train/play 配置和 PPO 超参数。

动作语义
--------

策略输出一维连续动作 ``action ∈ [-1, 1]``，语义不是绝对开度，而是位置命令增量：

.. code-block:: text

   u = clip(u + action * delta_u_max, 0, 255)

其中：

- ``delta_u_max = 3.0``（``env_cfgs.DELTA_U_MAX``）
- ``u ∈ [0, 255]``
- ``u`` 由 ``RobotiqCommandAction`` 在环境内部维护

接口目的：让策略学“再闭合一点 / 再松开一点”，而不是直接回归 Robotiq 寄存器
绝对位置。

当前动作边界
------------

当前主线只控制 Robotiq 2F-85 的开合，不控制整条机械臂。

- 当前已实现：一维 ``Δu`` 动作
- 当前未实现：UR 机械臂控制、笛卡尔 IK、whole-arm 6DoF 动作

如果后续要引入整臂位姿控制，应作为新的任务层定义，而不是改写当前 tactile
gripper baseline 的动作语义。

控制写入链路
------------

当前动作项与 actuator 的关系是：

1. ``RobotiqCommandAction.process_actions()`` 对 policy action 做 clip 与累积
2. ``RobotiqCommandAction.apply_actions()`` 通过
   ``entity.set_tendon_len_target`` 将命令写入 ``split`` tendon
3. ``RobotiqGeneralActuator`` 将 XML 中的 ``<general>`` actuator 包装成
   position-like 命令字段

实现位置：

- ``tactile_grasp.mdp.actions`` —— ``RobotiqCommandAction`` 与 cfg
- ``tactile_grasp.mdp.actuators`` —— ``RobotiqGeneralActuator`` 与 cfg
- 在 ``robot_cfg.build_robot_cfg()`` / ``build_action_cfg()`` 中装配

MuJoCo 侧仍然使用原始 2F-85 的 tendon + general actuator 抽象，没有引入理想
力控接口。

Train / Play 配置
-----------------

注册时通过同一个工厂 ``make_tactile_grasp_env_cfg(play)`` 同时构造两套 cfg：

============ ========= ================= =====================
模式          num_envs  episode_length_s  当前用途
============ ========= ================= =====================
train         64        3.0 s             PPO 采样训练
play          1         6.0 s             viewer / 调试 / 单环境检查
============ ========= ================= =====================

两者的共同事实：

- ``auto_reset = True``
- 当前没有实现 domain randomization
- 主要差异只有 ``scene.num_envs``、``episode_length_s`` 以及
  ``observations["actor"].enable_corruption`` 在 play 下显式设为 ``False``

字段覆盖 idiom
--------------

``tactile_grasp.make_env`` 不接受字段覆盖。调用方拿到 cfg 后直接 mutate：

.. code-block:: python

   from mjlab.envs import ManagerBasedRlEnv

   from tactile_grasp import TASK_ID, load_env_cfg

   cfg = load_env_cfg(TASK_ID, play=False)
   cfg.scene.num_envs = 16
   cfg.episode_length_s = 1.0
   env = ManagerBasedRlEnv(cfg, device="cpu")

PPO 配置
--------

``tactile_grasp.rl_cfg.tactile_grasp_ppo_runner_cfg`` 当前返回：

==================== ==================================
参数                   当前值
==================== ==================================
Actor / Critic        MLP ``(256, 256)`` + ELU
Distribution          Gaussian, ``init_std=1.0``, scalar std
obs_normalization     True (actor & critic)
Learning rate         ``3e-4``, adaptive schedule, ``desired_kl=0.01``
PPO clip              ``0.2``
GAE λ / γ             ``0.95 / 0.99``
Entropy coef          ``0.01``
Epochs / Minibatches  ``5 / 4``
Max grad norm         ``1.0``
Steps per env         ``48``
Max iterations        ``3000``
Logger                ``wandb`` (``WANDB_MODE=offline`` 可关闭网络)
Upload model          ``False``
Save interval         ``50`` iterations
==================== ==================================

为什么仍然用 MLP
-----------------

主线 actor observation 现在是 320 维（含 history flatten），仍在普通 MLP +
obs_normalization 能稳定处理的范围内。

因此当前版本有意不引入：

- 专门的 tactile encoder
- privileged critic observation
- recurrent / transformer 时序建模

History 的 buffer 形状已经准备好，未来 encoder 可以直接接 ``[B, T, …]``
切片；这些都属于后续扩展项，而不是当前实现缺失。

CLI 入口
--------

``scripts/train.py`` 和 ``scripts/play.py`` 只做两件事：

1. ``import tactile_grasp`` —— 触发任务注册
2. 转发到 ``mjlab.scripts.train.main`` / ``mjlab.scripts.play.main``

因此所有 CLI 参数（``--agent.max-iterations``、``--gpu-ids``、checkpoint
管理等）由 mjlab 提供，详见 mjlab 文档。常用：

.. code-block:: bash

   # 训练（CPU smoke）
   WANDB_MODE=offline PYTHONPATH= uv run python scripts/train.py \
       Mjlab-TactileGrasp-Robotiq2F85 \
       --agent.max-iterations 100 \
       --gpu-ids None

   # 训练（GPU 默认）
   PYTHONPATH= uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85

   # Play
   PYTHONPATH= uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85

相关页面
--------

- :doc:`task_architecture`
- :doc:`reward_design`
- :doc:`tactile_pipeline`
