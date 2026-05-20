控制与训练管线
==============

本文只讨论动作语义、Robotiq actuator 封装、train/play 配置和 PPO 超参数。

动作语义
--------

策略输出一维连续动作 ``action ∈ [-1, 1]``，语义不是绝对开度，而是位置命令增量：

.. code-block:: text

   u = clip(u + action * delta_u_max, 0, 255)

其中：

- ``delta_u_max = 3.0``
- ``u ∈ [0, 255]``
- ``u`` 由 ``RobotiqCommandAction`` 在环境内部维护

这个接口的目的很明确：让策略学“再闭合一点 / 再松开一点”，而不是直接回归
Robotiq 寄存器绝对位置。

当前动作边界
------------

当前主线任务只控制 Robotiq 2F-85 的开合，不控制整条机械臂。

- 当前已实现：一维 ``Δu`` 动作
- 当前未实现：``UR`` 机械臂控制、笛卡尔 ``IK``、whole-arm ``6DoF`` 动作

如果后续要引入整臂位姿控制，那应作为新的任务层定义，而不是直接改写当前 tactile gripper baseline 的动作语义。

控制写入链路
------------

当前动作项与 actuator 的关系是：

1. ``RobotiqCommandAction.process_actions()`` 对 policy action 做 clip 与累积
2. ``RobotiqCommandAction.apply_actions()`` 将命令写入 split tendon target
3. ``RobotiqGeneralActuatorCfg`` 将 XML 中的 ``<general>`` actuator 包装成
   position-like 控制语义

MuJoCo 侧仍然使用原始 2F-85 的 tendon + general actuator 抽象，没有引入理想力控接口。

Train / Play 配置
-----------------

任务注册时会为每个 task 同时注册 train 与 play 两套配置：

============ ========= ================= =====================
模式          num_envs  episode_length_s  当前用途
============ ========= ================= =====================
train         64        3.0 s             PPO 采样训练
play          1         6.0 s             viewer / 调试 / 单环境检查
============ ========= ================= =====================

当前两者的共同事实：

- ``auto_reset = True``
- 当前没有实现 domain randomization
- 主要差异只有并行环境数和 episode 长度

也就是说，文档里不应再写“train 开启随机化 / play 关闭随机化”。

PPO 配置
--------

当前 ``rl_cfg.py`` 的配置如下：

==================== ==================================
参数                   当前值
==================== ==================================
Actor / Critic        MLP ``(128, 128)`` + ELU
Distribution          Gaussian, ``init_std=1.0``
Learning rate         ``3e-4``, adaptive schedule
PPO clip              ``0.2``
GAE λ / γ             ``0.95 / 0.99``
Entropy coef          ``0.01``
Epochs / Minibatches  ``5 / 4``
Max grad norm         ``1.0``
Steps per env         ``32``
Max iterations        ``200``
Logger                ``tensorboard``
Upload model          ``False``
==================== ==================================

为什么当前仍然用 MLP
---------------------

当前主线 ``PTSSpheres`` 的 actor observation 为 80 维，仍在普通 MLP 能稳定处理的范围内。

因此当前版本有意不引入：

- tactile encoder
- force history
- privileged critic observation

这些都属于后续扩展项，而不是当前实现缺失。

相关页面
--------

- :doc:`task_architecture`
- :doc:`reward_design`
- :doc:`tactile_pipeline`
