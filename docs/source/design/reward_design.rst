奖励与终止设计
==============

本文只解释当前 top-down pick-lift ``tactile_grasp`` 环境真正启用的 reward 与 termination，
不重复环境装配或 taxel 建模细节。

当前奖励项
----------

奖励项都在 ``env_cfgs.make_tactile_grasp_env_cfg`` 中配置，定义在
``tactile_grasp.mdp.rewards``。当前是 *单项乘法门控级联* 加 *若干独立惩罚项* 的
组合：bootstrap chain（reach → close → contact → lift）被吸收进
``staged_pickup``；持有、撞地、掉物、抖动各自作为独立加法项。

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 奖励项
     - 权重
     - 当前作用
   * - ``staged_pickup``
     - +3.0
     - ``reach · (1 + close · (1 + contact · (1 + lift)))``，下方详细展开
   * - ``hold``
     - +2.0
     - 满足 ``lift_delta > 0.03`` 且双指都有 taxel 接触时给 1
   * - ``floor_collision``
     - -12.0
     - 机器人 geom 与 plane terrain geom 接触时给 1（penalty，不终止）
   * - ``action_smoothness``
     - -0.01
     - ``Σ|a_t − a_{t-1}|``，惩罚抖动
   * - ``drop_penalty``
     - -5.0
     - ``object_drop`` 终止时施加 -5

``staged_pickup`` 的四个内部因子（各自落在 ``[0, 1]``，cascade 输出落在
``[0, 4]``）：

- ``reach = exp(-k_pos · d_aniso)``，``k_pos = 10``
- ``close = command · exp(-k_d · d_aniso)``，``k_d = 30``
- ``contact = taxel_coverage``（双指 3×3 taxel 激活比例平均）
- ``lift = clamp(lift_delta / 0.08, 0, 1)``

其中 ``d_aniso = sqrt(2·(Δx² + Δy²) + Δz²)`` 是 *各向异性* 3D 距离：xy 平方项
权重是 z 平方项的 2 倍，所以 xy 不对齐时 reach 衰减更快。这一个项替代了之前的
``reach3d`` / ``align`` / ``close_near_object`` / ``contact`` / ``coverage`` /
``lift_delta`` 六个加法项。

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

旧版（加法 bootstrap）的失败模式：policy 学会悬停在物体上方 2-3 cm，半闭夹爪，
单步只靠 ``reach3d + align + close_near_object`` 就能拿到 ~80% 的可得 shape
奖励，且没有任何结构性激励去真正下探接触。

乘法门控通过 ``reach · (1 + close · (1 + contact · (1 + lift)))`` 强制阶段推进：

1. ``reach`` 是外层门，远离物体时整条 cascade 直接为 0；
2. ``close`` 只在 reach 已经偏大时才贡献信号，避免"远处闭爪"的伪奖励；
3. ``contact``（= ``taxel_coverage``）平滑地把接触强度传入；
4. ``lift`` 在 contact 已发生时才解锁，且在 8 cm（``SUCCESS_HEIGHT``）饱和。

下表给出几个典型阶段的单步奖励（含 ``W_STAGED_PICKUP = 3.0``，``k_pos = 10``，
``k_d = 30``，``lift_cap = 0.08``）：

============================== ====== ====== ======= ===== ======== =======
阶段                            reach  close  contact lift  cascade  reward
============================== ====== ====== ======= ===== ======== =======
Initial (far)                   0.30   0      0       0     0.30     0.90
Aligned + half-closed hover     0.70   0.14   0       0     0.80     2.40
First contact (4/9 taxels)      0.95   0.70   0.44    0     1.91     5.72
Lifted to 4 cm                  0.95   0.90   0.90    0.50  2.96     8.88
Saturated (≥ 8 cm)              1.00   1.00   1.00    1.00  4.00    12.00
============================== ====== ====== ======= ===== ======== =======

"悬停刷分"基线（2.40）严格小于"下探到接触"（5.72），policy 必须穿越接触门
才能拿到更高单步奖励。``hold`` / ``floor_collision`` / ``drop_penalty`` /
``action_smoothness`` 仍作为独立加法项（不参与 cascade）补足成功条件、
安全约束与运动平滑性。

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
