奖励与终止设计
==============

本文只解释当前环境真正启用的 reward 与 termination，不重复环境装配或 taxel 建模细节。

当前奖励项
----------

奖励项都在 ``TactileGraspTaskConfig.build()`` 中配置，核心思路是：

- 用 ``alive`` 保持训练信号稳定
- 用轻量惩罚限制过大触觉力和过激动作
- 用 ``drop_penalty`` 明确区分失败

==================== ======= ===============================================
奖励项                权重    当前作用
==================== ======= ===============================================
``alive``             +1.0   鼓励 episode 持续进行
``tactile_force``     -0.01  惩罚左右指尖总触觉信号平方和
``action_rate``       -0.001 惩罚 raw action 平方
``close_command``     -0.001 惩罚过大的闭合命令 ``(u/255)^2``
``drop_penalty``      -5.0   物体掉落时施加固定负奖励
==================== ======= ===============================================

这里的 ``tactile_force`` 对两条任务路径都成立：

- TouchSite：惩罚 18 个标量 touch 值
- PTSSpheres：惩罚 54 维三轴 taxel force

终止条件
--------

==================== ============ ============================================
终止项                类型          当前条件
==================== ============ ============================================
``time_out``          超时         达到 ``episode_length_s``
``object_drop``       失败         物体高度低于 ``drop_height_threshold``
``stable_grasp``      成功         连续若干步同时满足高度与触觉阈值
==================== ============ ============================================

默认参数来自 ``env_cfg.py``：

- ``drop_height_threshold = 0.08``
- ``success_height_threshold = 0.14``
- ``success_hold_steps = 25``

``stable_grasp`` 判定逻辑
-------------------------

当前成功判定不是基于 object truth contact list，而是基于：

.. code-block:: python

   height_ok = object_height > minimum_height
   touch_ok = total_tactile_signal > minimum_tactile_signal
   stable = height_ok & touch_ok
   counter = counter + 1 if stable else 0
   done = counter >= hold_steps

当前的最小触觉阈值为：

==================== =========================================
任务                  ``minimum_tactile_signal``
==================== =========================================
TouchSite             0.0
PTSSpheres            ``1.0e-3``
==================== =========================================

之所以给 ``PTSSpheres`` 加 ``1.0e-3``，是因为 welded taxel force sensor
在无接触时会有极小静态噪声；如果阈值为 0，会导致成功条件过早触发。

为什么当前 reward 保持极简
---------------------------

当前版本刻意不加入复杂 shaping，原因是实现优先级仍是：

1. 先确认 MJCF / 传感器 / reset-step 链路稳定
2. 再确认策略能在最小奖励集下启动训练
3. 最后再逐步加更细的接触质量指标

因此当前 reward 更像“安全起步版”，而不是最终研究版。

后续预留项
----------

以下项仍属于后续增强，不是当前主线实现的一部分：

- 目标夹持力奖励
- 左右指尖力平衡奖励
- slip proxy / 滑移惩罚
- 依赖触觉历史的时序奖励

相关页面
--------

- :doc:`task_architecture`
- :doc:`tactile_pipeline`
- :doc:`control_pipeline`
