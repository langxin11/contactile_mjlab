实现总览
========

本文是当前 task-based 实现的总入口，帮助读者快速理解：

- 主线任务是什么
- 代码入口在哪里
- 环境在运行时如何从 action 走到 observation / reward / termination
- 哪些能力已经实现，哪些还明确留到后续阶段

如果你想看“当前仓库做到了哪一步”，优先看本文。
根目录的 ``plan.md`` 只保留路线与原则，不再同步当前实现细节。

主线任务
--------

仓库当前保留两个任务：

.. list-table::
   :header-rows: 1

   * - Task ID
     - 作用
   * - ``Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres``
     - 主线任务，使用 per-taxel 三轴力
   * - ``Mjlab-TactileGrasp-Robotiq2F85-TouchSite``
     - 回归对照任务，保留旧 3×3 标量 touch map

两者共享同一套控制接口、奖励结构、终止条件和 PPO 配置，只在触觉观测路径上不同。

这里的“任务”含义需要特别说明：当前主线是 tactile gripper baseline，不是 whole-arm manipulation。
因此动作接口仍然固定为 Robotiq 2F-85 的一维 ``Δu``，而不是机械臂末端的 ``6DoF`` 位姿控制。

代码入口
--------

推荐从包根入口开始阅读：

- ``contactile_mjlab.make_env()``：按 task id 创建环境
- ``contactile_mjlab.load_env_cfg()``：读取已注册 task 的环境配置
- ``contactile_mjlab.load_rl_cfg()``：读取 PPO 配置
- ``contactile_mjlab.tasks.tactile_grasp``：任务注册与实现主体

实现组织关系：

::

   contactile_mjlab
   ├── __init__.py
   │   └── re-export make_env / load_env_cfg / load_rl_cfg / task ids
   └── tasks/tactile_grasp/
       ├── __init__.py        # register_tasks(), make_env(), load_env_cfg()
       ├── env_cfg.py         # 组装 Scene / Obs / Action / Reward / Done
       ├── robot_cfg.py       # 选择 XML、配置 actuator、初始关节状态
       ├── object_cfg.py      # hanging_box 物体配置
       ├── tactile_terms.py   # touch / taxel force / wrench 观测读取
       ├── reward_terms.py    # reward 与 termination helper
       ├── rl_cfg.py          # PPO runner config
       └── constants.py       # task id、sensor 名称、阈值、关节名

实现现状矩阵
------------

下表是当前仓库进度的单一事实来源。

.. list-table::
   :header-rows: 1

   * - 子系统
     - 当前状态
     - 备注
   * - 资产层
     - 已完成
     - ``2f85.xml``、``2f85_tactile.xml``、``2f85_pts_spheres.xml`` 与对应 scene 文件都已存在
   * - 环境层
     - 已完成
     - task registry、train/play config、``make_env()`` / ``load_env_cfg()`` / ``load_rl_cfg()`` 已接通
   * - 触觉层
     - 已完成但简化
     - TouchSite 与 PTSSpheres 都能工作；PTSSpheres 当前是 builtin ``<force>`` sensor 直读
   * - 控制层
     - 已完成
     - ``Δu`` 控制、Robotiq tendon/general actuator 封装已接通；未采用理想 force actuator，也未引入 whole-arm ``6DoF`` 动作
   * - 奖励与终止
     - 已完成但简化
     - 最小 reward 集合与 ``stable_grasp`` 判定已实现；无复杂 shaping
   * - 训练层
     - 已完成 smoke
     - PPO 最小训练可跑；正式 benchmark、长期收敛结论尚未整理
   * - sim2real 过渡层
     - 未实现
     - 无 local-frame tactile、无 history、无 randomization、无 slip proxy
   * - sim2real 部署层
     - 未开始
     - 仅保留设计原则，未进入部署与标定实现

当前明确未实现项
----------------

- taxel local-frame force
- taxel force history
- slip proxy
- domain randomization
- sim2real 部署链路

当前明确不在 Phase 1
--------------------

- pad 多碰撞几何拆分
- 动作扩展到 ``[Δu, Δforce_limit]``
- whole-arm ``6DoF`` 动作或笛卡尔 ``IK`` 控制
- 理想 force actuator 抽象

运行时数据流
------------

::

   policy(obs["actor"])
      -> action in [-1, 1]
      -> RobotiqCommandAction.process_actions()
      -> u = clip(u + action * delta_u_max, 0, 255)
      -> write tendon target
      -> MuJoCo step (decimation = 10, dt = 0.002)
      -> builtin sensors update
      -> tactile_terms / mjlab.mdp read observations
      -> reward_terms compute reward and termination helpers
      -> env returns obs, reward, terminated, truncated

当前 actor observation 的共同骨架为：

- 左右触觉观测
- 左右 pad 全局 wrench
- 归一化夹爪命令
- 夹爪相关关节位置 / 速度
- 上一步动作

其中触觉部分按任务分支：

==================== ==================== ===================
任务                  每指触觉形状          总 observation 维度
==================== ==================== ===================
TouchSite             ``[B, 9]``           44
PTSSpheres            ``[B, 27]``          80
==================== ==================== ===================

与 mjlab 的关系
---------------

- ``pyproject.toml`` 中的 ``mjlab.tasks`` entry point 指向 ``contactile_mjlab``
- ``register_tasks()`` 为 train / play 两种配置注册两个 task
- ``ManagerBasedRlEnvCfg`` 是唯一的环境装配中心
- 训练脚本通过 ``load_rl_cfg()`` 取 PPO 配置，再交给 mjlab 的 RSL-RL runner

当前验证状态
------------

当前主线已经完成以下验证类型：

- MJCF 编译检查
- ``reset()`` / ``step()`` smoke test
- TouchSite / PTSSpheres 双任务并存验证
- 最小 PPO smoke run

这意味着仓库已经进入“可运行、可验证、可继续迭代”的阶段，而不是早期纯方案阶段。

文档地图
--------

- :doc:`tactile_pipeline`：触觉信号怎么从 XML 传感器变成 observation tensor
- :doc:`reward_design`：reward / termination 的当前逻辑与后续预留项
- :doc:`control_pipeline`：动作语义、actuator 封装、train/play 与 PPO 配置
- :doc:`pts_taxel_scheme`：PTS sphere-taxel 的 MJCF 建模与命名约束

相关页面
--------

- :doc:`tactile_pipeline`
- :doc:`reward_design`
- :doc:`control_pipeline`
- :doc:`pts_taxel_scheme`
