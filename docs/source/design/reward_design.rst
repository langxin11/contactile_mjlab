奖励与终止设计
==============

本文只解释当前 top-down pick-lift ``tactile_grasp`` 环境真正启用的 reward 与 termination，
不重复环境装配或 taxel 建模细节。

当前奖励项
----------

奖励项都在 ``env_cfgs.make_tactile_grasp_env_cfg`` 中配置，定义在
``tactile_grasp.mdp.rewards``。当前是分阶段 shaping 组合，目标是让策略经过
"对准 → 下探 → 轻触 → 多点接触 → 闭合抬升 → 稳定悬停" 的链路，而不是悬停少动作。
每一项都是有界的：正向项用 ``exp(-k·d)`` 或 ``0/1`` 二值，负向项直接用负权重。

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 奖励项
     - 权重
     - 当前作用
   * - ``reach3d``
     - +0.6
     - ``exp(-k_pos·‖p_obj − p_tool‖)``，``k_pos = 10``，引导 3D 接近
   * - ``align``
     - +0.8
     - ``exp(-k_xy·‖Δxy‖)``，``k_xy = 20``，鼓励先对准 XY 再下探
   * - ``contact``
     - +0.2
     - 任一 taxel force 范数超过 ``0.05 N`` 即得 1，弱信号防止悬停
   * - ``coverage``
     - +1.2
     - 双指各 3×3 taxel 中激活数比例（左右平均，单边截断到 9）
   * - ``lift_delta``
     - +8.0
     - ``relu(z_obj − z_obj_init)``；reset 时缓存 ``_tactile_active_object_init_z``
   * - ``hold``
     - +2.0
     - 满足 ``lift_delta > 0.03`` 且双指都有 taxel 接触时给 1
   * - ``floor_collision``
     - -12.0
     - 机器人 geom 与 plane terrain geom 接触时给 1（penalty，不终止）
   * - ``action_smoothness``
     - -0.01
     - ``Σ|a_t − a_{t-1}|``，惩罚抖动
   * - ``close_command``
     - -0.05
     - ``(u / 255)^2``，惩罚多余的闭合命令
   * - ``drop_penalty``
     - -5.0
     - ``object_drop`` 终止时施加 -5

触觉观测在进入 actor/critic 前会被裁剪到 sensor 物理量程：

- 切向力 ``site.X / site.Y`` 裁剪到 ``[-4, 4] N``
- 法向力 ``site.Z`` 裁剪到 ``[-15, 15] N``

观测项 scale 取量程倒数（``1/4`` 和 ``1/15``），裁剪后归一化值大致落在
``[-1, 1]``，避免极少数高力 sample 把 normalization 拉偏。

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
   * - ``robot_out_of_workspace``
     - 失败
     - mocap command 越过 workspace；正常 action clipping 下应为 False
   * - ``stable_grasp``
     - 成功
     - ``stable_grasp_hold`` 连续若干步同时满足 lift height 与触觉阈值

默认参数来自 ``env_cfgs.py``：

- ``DROP_HEIGHT = 0.002``
- ``SUCCESS_HEIGHT = 0.08``
- ``SUCCESS_HOLD_STEPS = 25``
- ``TACTILE_ACTIVITY_THRESHOLD = 1.0e-3``（来自 ``constants.py``）

``stable_grasp_hold`` 判定逻辑
------------------------------

``stable_grasp_hold`` 是 class-based termination，每个 env 维护一个计数器：

.. code-block:: python

   height_ok = active_object_height > minimum_height
   touch_ok = total_tactile_signal > minimum_tactile_signal
   stable = height_ok & touch_ok
   counter = counter + 1 where stable else 0
   done = counter >= hold_steps

``total_tactile_signal`` 在 ``mdp.rewards`` 中实现（被 termination 复用），
等于左右指 18 个三维 sensor 读数全部展平后的 ``sum(abs(·))``。

阈值 ``minimum_tactile_signal = 1.0e-3`` 是为了过滤 welded taxel force sensor
在无接触时的静态噪声 —— 不是语义判断。

为什么这样组合 reward
---------------------

这一版 reward 把 "对准 / 接触 / 抬升 / 稳定" 拆成阶梯式 shaping：

1. ``reach3d`` 与 ``align`` 先把夹爪带到物体上方并 XY 对齐
2. ``contact`` 与 ``coverage`` 鼓励真正轻触并覆盖多 taxel，而不是悬停
3. ``lift_delta`` 与 ``hold`` 鼓励抬起且要双指都还在接触，避免抛物
4. ``floor_collision`` / ``action_smoothness`` / ``close_command`` / ``drop_penalty``
   惩罚明显错误（撞地、抖动、空闭合、掉物），但都不直接终止 episode

仍属于后续增强：

- 目标夹持力 / 力平衡 reward
- place target / release reward
- slip proxy / 滑移惩罚
- 依赖触觉历史 buffer 的时序 reward（history 已经有了，shape 是 ready 的，
  只是当前 reward 函数没有用 history slice）

相关页面
--------

- :doc:`task_architecture`
- :doc:`tactile_pipeline`
- :doc:`control_pipeline`
