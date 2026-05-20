触觉管线设计
============

本文只解释一件事：当前实现里，触觉信号如何从 MuJoCo XML 传感器走到
``obs["actor"]``。

双任务观测路径
--------------

当前仓库支持两条触觉观测路径：

=========================== =================================== ====================
任务                         传感器类型                           每指观测形状
=========================== =================================== ====================
TouchSite                    ``<touch>`` scalar sensor           ``[B, 9]``
PTSSpheres                   ``<force>`` 3-axis builtin sensor   ``[B, 27]``
=========================== =================================== ====================

两条路径都经过同一个 observation 组装流程，只是在 ``env_cfg.py`` 中根据
``tactile_model`` 选择不同的 sensor 名称和读取函数。

读取链路
--------

当前主线不手写 contact extraction，而是直接读取 XML 中定义好的 builtin sensor。

::

   env.scene["robot/<sensor_name>"].data
      -> tactile_terms._sensor_tensor()
      -> tactile_terms.sensor_values()
      -> touch_map() 或 taxel_force_map()
      -> ObservationTermCfg
      -> obs["actor"]

对应实现：

- ``touch_map()``：将 9 个标量 touch sensor 拼成一维向量
- ``taxel_force_map()``：将 9 个三轴力 sensor 拼成 27 维向量
- ``pad_wrench()``：读取左右 pad 的 ``force + torque``
- ``gripper_command()``：读取动作项内部维护的归一化 ``u / 255``

归一化与形状
------------

当前 observation term 直接在 ``ObservationTermCfg`` 中做固定比例缩放：

==================== ===============================
观测项                当前缩放
==================== ===============================
TouchSite 触觉         ``1 / touch_scale``，默认 ``1 / 10``
PTSSpheres 触觉        ``1 / force_scale``，默认 ``1 / 20``
Pad torque            ``1 / torque_scale``，默认 ``1 / 2``
Pad force             ``1 / force_scale``，默认 ``1 / 20``
==================== ===============================

主线任务 ``PTSSpheres`` 的 actor observation 维度为：

- 左指触觉 27
- 右指触觉 27
- 左右 wrench 12
- gripper command 1
- joint pos 6
- joint vel 6
- last action 1

合计 80 维。

为什么当前采用 builtin sensor
------------------------------

当前实现选择 builtin ``<force>`` / ``<touch>`` sensor，而不是手写遍历
``data.contact`` 的接触聚合逻辑，原因很直接：

1. 每个 taxel geom 与一个 sensor 一一对应，读数路径最短。
2. 在 task package 重构与 PTS 资产改造同时进行时，builtin sensor 更稳。
3. 第一阶段目标是“先把 taxel 级读数链路打通”，不是先追求高保真 contact filtering。

这个选择的代价也明确存在：

- ``PTSSpheres`` 当前读到的是 world-frame 三轴力
- 当前不区分“来自目标物体”的接触和“来自其他几何”的接触
- 当前没有时间历史缓冲

这些都属于后续增强项，而不是当前实现的 bug。

当前未实现项
------------

**Local-frame force**
  当前没有做 ``F_local = R_pad^T @ F_world`` 转换。

**Taxel force history**
  当前每步只提供瞬时 taxel force，没有 ``[9, H, 3]`` 历史 buffer。

**Managed contact sensor migration**
  当前没有迁移到更高层的 mjlab contact sensor 管理接口。

活动阈值
--------

``stable_grasp`` 成功判定依赖一个最小触觉活动阈值：

==================== =======================================
任务                  阈值
==================== =======================================
TouchSite             0.0
PTSSpheres            ``1.0e-3``
==================== =======================================

其中 ``PTSSpheres = 1.0e-3`` 的目的不是做语义判断，而是过滤 welded taxel
force sensor 的静态微小噪声。

相关页面
--------

- :doc:`task_architecture`
- :doc:`reward_design`
- :doc:`pts_taxel_scheme`
