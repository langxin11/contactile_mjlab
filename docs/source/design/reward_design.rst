奖励与终止设计
==============

本文只解释当前 ``tactile_grasp`` 环境真正启用的 reward 与 termination，
不重复环境装配或 taxel 建模细节。

当前奖励项
----------

奖励项都在 ``env_cfgs.make_tactile_grasp_env_cfg`` 中配置，定义在
``tactile_grasp.mdp.rewards``。核心思路：

- 用 ``alive`` 保持训练信号稳定
- 用轻量惩罚限制过大触觉力和过激动作
- 用 ``drop_penalty`` 明确区分失败

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 奖励项
     - 权重
     - 当前作用
   * - ``alive``
     - +1.0
     - 鼓励 episode 持续进行
   * - ``tactile_force``
     - -0.01
     - 惩罚双指尖 taxel force 平方和（54 维原始 sensor）
   * - ``action_rate``
     - -0.001
     - 惩罚 raw action 平方
   * - ``close_command``
     - -0.001
     - 惩罚过大的闭合命令 ``(u/255)^2``
   * - ``drop_penalty``
     - -5.0
     - ``object_drop`` 终止时施加 -5

``tactile_force_l2`` 是对原始 sensor 读数计算的，不依赖 normal / tangential
拆分后的观测项 —— 它直接对左右指各 9 个三维 force sensor 的全部 54 维做平方和。

终止条件
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 终止项
     - 类型
     - 条件
   * - ``time_out``
     - 超时
     - 达到 ``episode_length_s`` (train 3 s / play 6 s)
   * - ``object_drop``
     - 失败
     - ``object_height_below(minimum_height=DROP_HEIGHT)``
   * - ``stable_grasp``
     - 成功
     - ``stable_grasp_hold`` 连续若干步同时满足高度与触觉阈值

默认参数来自 ``env_cfgs.py``：

- ``DROP_HEIGHT = 0.08``
- ``SUCCESS_HEIGHT = 0.14``
- ``SUCCESS_HOLD_STEPS = 25``
- ``TACTILE_ACTIVITY_THRESHOLD = 1.0e-3``（来自 ``constants.py``）

``stable_grasp_hold`` 判定逻辑
------------------------------

``stable_grasp_hold`` 是 class-based termination，每个 env 维护一个计数器：

.. code-block:: python

   height_ok = object_height > minimum_height
   touch_ok = total_tactile_signal > minimum_tactile_signal
   stable = height_ok & touch_ok
   counter = counter + 1 where stable else 0
   done = counter >= hold_steps

``total_tactile_signal`` 在 ``mdp.rewards`` 中实现（被 termination 复用），
等于左右指 18 个三维 sensor 读数全部展平后的 ``sum(abs(·))``。

阈值 ``minimum_tactile_signal = 1.0e-3`` 是为了过滤 welded taxel force sensor
在无接触时的静态噪声 —— 不是语义判断。

为什么当前 reward 保持极简
---------------------------

实现优先级仍然是：

1. 先确认 MJCF / 传感器 / reset-step 链路稳定
2. 再确认策略能在最小奖励集下启动训练
3. 最后再逐步加更细的接触质量指标

因此当前 reward 是“安全起步版”，而不是最终研究版。

后续预留项
----------

以下项仍属于后续增强：

- 目标夹持力 / 力平衡奖励
- slip proxy / 滑移惩罚
- 依赖触觉历史 buffer 的时序奖励（history 已经有了，shape 是 ready 的，
  只是当前 reward 函数没有用 history slice）

相关页面
--------

- :doc:`task_architecture`
- :doc:`tactile_pipeline`
- :doc:`control_pipeline`
