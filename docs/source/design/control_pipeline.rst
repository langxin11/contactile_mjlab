控制与训练管线
==============

本文只讨论动作语义、Robotiq actuator 封装、mocap 末端控制、train/play 配置和
PPO 超参数。

动作语义
--------

策略输出五维连续动作 ``[dx, dy, dz, dyaw, du] ∈ [-1, 1]^5``：

.. code-block:: text

   p = clip(p + [dx, dy, dz] * pos_step, workspace)
   yaw = clip(yaw + dyaw * yaw_step, [-pi, pi])
   u = clip(u + du * delta_u_max, 0, 255)

其中：

- ``pos_step = 0.01 m``
- ``yaw_step = 0.05 rad``
- ``delta_u_max = 3.0``（``env_cfgs.DELTA_U_MAX``）
- ``u ∈ [0, 255]``
- ``p/yaw/u`` 由 ``CartesianMocapAction`` 在环境内部维护

接口目的：让策略学 top-down reach / descend / grasp / lift，同时保留 Robotiq
真实位置命令寄存器语义。

当前动作边界
------------

当前主线控制 fixed-base 夹爪的 mocap 末端位姿，不控制真实机械臂关节。

- 当前已实现：世界系 ``dx/dy/dz/dyaw`` + Robotiq ``du``
- 当前未实现：UR / Franka 机械臂控制、笛卡尔 IK、whole-arm 6DoF 动作

不在 XML 里给夹爪加虚拟 slide/hinge base joints；mjlab 会把 fixed-base
entity 自动包装成 mocap body，动作项直接写 mocap pose。

控制写入链路
------------

当前动作项与 actuator 的关系是：

1. ``CartesianMocapAction.process_actions()`` 对 5D policy action 做 clip 与累积
2. ``CartesianMocapAction.apply_actions()`` 写 robot mocap pose，并通过
   ``entity.set_tendon_len_target`` 将 ``u`` 写入 ``split`` tendon
3. ``RobotiqGeneralActuator`` 继续将 XML 中的 ``<general>`` actuator 包装成
   position-like 命令字段

实现位置：

- ``tactile_grasp.mdp.actions`` —— ``CartesianMocapAction`` 与 cfg
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
- train 按 curriculum stage 逐步开放 object type / pose 随机
- play 固定使用 Stage 2 随机范围
- 主要差异只有 ``scene.num_envs``、``episode_length_s`` 以及
  ``observations["actor"].enable_corruption`` 在 play 下显式设为 ``False``

Curriculum stage：

==================== ==================================================
阶段                 随机范围
==================== ==================================================
Stage 0              只用 cube，物体固定中心，夹爪初始对准
Stage 1              cube/box，物体 ``xy ±0.03 m``，yaw ``±pi/6``
Stage 2              cube/box/cylinder，物体 ``xy ±0.08 m``，yaw ``±pi``
==================== ==================================================

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

主线 actor observation 现在是 332 维（含 history flatten 与 ``vision_proxy``），仍在普通 MLP +
obs_normalization 能稳定处理的范围内。

因此当前版本有意不引入：

- 专门的 tactile encoder
- privileged critic observation
- 图像 observation / CNN encoder
- recurrent / transformer 时序建模

当前 scene 中有 ``overhead_debug`` camera 用于 viewer / 调试。Policy 不直接吃图像；
``vision_proxy`` 是未来图像 encoder 或视觉估计器的替换点。

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

top-down pick-lift 的 PTS spheres 和桌面物体会产生比 hanging baseline 更多的
接触约束。环境默认设置 ``SimulationCfg(nconmax=128, njmax=256)``，避免
MJWarp 在 512 env 训练时使用过小的 heuristic 并打印 ``nefc overflow``。
如果后续增大 taxel 数或物体复杂度，应优先调大 ``--env.sim.njmax``，必要时同步
调大 ``--env.sim.nconmax``。

相关页面
--------

- :doc:`task_architecture`
- :doc:`reward_design`
- :doc:`tactile_pipeline`
