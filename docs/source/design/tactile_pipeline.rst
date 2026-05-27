触觉管线设计
============

本文只解释一件事：当前实现里，触觉信号如何从 MuJoCo XML 传感器走到
``obs["actor"]``。

单一观测路径
------------

仓库当前只有一条触觉路径：PTSSpheres 模型 + builtin ``<force>`` 3-axis sensor。
旧的 TouchSite 标量路径已经从源码中删除，只在 ``2f85_tactile.xml`` 资产层留作
对照模型。

每侧 3×3 共 9 个 sphere taxel，每个 taxel 一个 ``<force>`` sensor，读到的是
site-local 坐标系下的三维力（``site.Z`` 为 pad 法向，``site.X`` / ``site.Y``
为切向）。

读取链路
--------

::

   env.scene["robot/<sensor_name>"].data
      -> mdp.observations._sensor_tensor()
      -> taxel_normal_force()     # [B, 9]    site.Z
         taxel_tangential_force() # [B, 18]   site.X, site.Y flatten
         pad_force()              # [B, 3]
         pad_torque()             # [B, 3]
         gripper_command()        # [B, 1]
         vision_proxy()           # [B, 8]
      -> ObservationTermCfg(history_length=...)
      -> obs["actor"]

每个 taxel 力都被显式拆成 normal / tangential 两个独立的 ObservationTerm，
方便分别配 scale 与（未来的）encoder。

归一化与形状
------------

在 ``env_cfgs.py`` 中固定的 scale：

==================== =====================================
观测项                当前缩放
==================== =====================================
taxel normal          ``1 / NORMAL_FORCE_SCALE = 1 / 5``
taxel tangential      ``1 / TANGENTIAL_FORCE_SCALE = 1 / 2``
pad force             ``1 / FORCE_SCALE = 1 / 20``
pad torque            ``1 / TORQUE_SCALE = 1 / 2``
==================== =====================================

History
-------

mjlab 的 ``ObservationTermCfg.history_length`` 会自动维护时间窗 buffer 并
flatten 进 obs。当前配置：

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - 观测项
     - history_length
     - 单项总维度
   * - taxel normal
     - 5
     - 9 × 5 = 45
   * - taxel tangential
     - 5
     - 18 × 5 = 90
   * - pad force / torque
     - 3
     - 3 × 3 = 9
   * - 其他 (gripper_command, joint_pos, joint_vel, vision_proxy, last_action)
     - 1
     - 原样

主线 actor observation 总维度为 **332**，详细拆分见
:doc:`task_architecture`。

为什么当前采用 builtin sensor
------------------------------

实现选择 builtin ``<force>`` sensor，而不是手写遍历 ``data.contact``：

1. 每个 taxel geom 与一个 sensor 一一对应，读数路径最短。
2. 第一阶段目标是“先把 taxel 级读数链路打通”，不是高保真 contact filtering。
3. 与 PTS 资产生成器 ``scripts/generate_pts_spheres_xml.py`` 直接对齐。

这个选择的当前代价：

- taxel 读到的是 *所有* 接触叠加，不区分“来自目标物体”与“来自其他几何”
- 没有迁移到 mjlab managed contact sensor 抽象

这些都属于后续增强项，而不是当前实现的 bug。

当前未实现项
------------

- **slip proxy** —— 切向力变化率 / 滑移检测尚未引入
- **target-only contact filtering** —— 不过滤非目标几何接触
- **managed contact sensor migration** —— 未迁移到更高层 mjlab 抽象

normal / tangential 拆分与 history 已经实现，不再列在“未实现”里。

活动阈值
--------

``stable_grasp_hold`` 成功判定依赖一个最小触觉活动阈值（``constants.py``）：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 常量
     - 值
   * - ``TACTILE_ACTIVITY_THRESHOLD``
     - ``1.0e-3``

这是单值阈值，对应单一 PTSSpheres 路径。其目的不是做语义判断，而是过滤 welded
taxel force sensor 在无接触时的静态微小噪声。

相关页面
--------

- :doc:`task_architecture`
- :doc:`reward_design`
- :doc:`pts_taxel_scheme`
