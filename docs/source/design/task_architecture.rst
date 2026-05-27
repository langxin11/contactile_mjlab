实现总览
========

本文是当前 ``tactile_grasp`` 实现的总入口，帮助读者快速理解：

- 主线任务是什么
- 代码入口在哪里
- 环境在运行时如何从 action 走到 observation / reward / termination
- 哪些能力已经实现，哪些还明确留到后续阶段

如果你想看“当前仓库做到了哪一步”，优先看本文。

主线任务
--------

仓库当前只保留一个任务：

==================================================== ================================================
Task ID                                              作用
==================================================== ================================================
``Mjlab-TactileGrasp-Robotiq2F85``                   主线任务，top-down pick-lift，使用 5D mocap+gripper 动作与 PTS 触觉
==================================================== ================================================

旧的 ``-PTSSpheres`` / ``-TouchSite`` 后缀已经退役；TouchSite 路径在源码层
也已经删除。

任务范围说明：当前主线是 *top-down tactile pick-lift baseline*，不是完整
pick-and-place。夹爪具备 ``dx/dy/dz/dyaw`` 末端空间自由度，但这些自由度通过
mjlab 对 fixed-base 夹爪的 mocap 包装写位姿命令实现，不在 XML 中加入虚拟
slide/hinge base joints。

代码入口
--------

推荐从包根入口开始阅读：

- ``tactile_grasp.make_env()`` —— 直接拿一个 ``ManagerBasedRlEnv``
- ``tactile_grasp.load_env_cfg()`` —— 拿到 cfg 后自行 mutate，再交给
  ``ManagerBasedRlEnv``
- ``tactile_grasp.load_rl_cfg()`` —— 读取 PPO runner cfg
- ``tactile_grasp.TASK_ID`` —— 唯一的 task id 常量

包采用扁平 layout（不再有 ``tasks/`` 子层），主要文件：

::

   tactile_grasp/
   ├── __init__.py          # 注册 task，re-export make_env / load_env_cfg / TASK_ID
   ├── constants.py         # TASK_ID、sensor 名称、阈值、关节名
   ├── env_cfgs.py          # make_tactile_grasp_env_cfg(play) —— Scene / Obs / Action / Reward / Done 装配
   ├── robot_cfg.py         # Robotiq + PTS spheres EntityCfg + action cfg
   ├── object_cfg.py        # cube / box / cylinder tabletop EntityCfgs
   ├── paths.py             # 包内资产路径
   ├── rl_cfg.py            # PPO runner cfg
   ├── assets/              # MJCF + props（已迁入包内，pip install 后自带）
   └── mdp/
       ├── actions.py       # CartesianMocapAction(Cfg) + RobotiqCommandAction(Cfg)
       ├── actuators.py     # RobotiqGeneralActuator(Cfg) —— XML actuator 包装
       ├── observations.py  # tactile split、pad wrench、vision_proxy、gripper_command
       ├── rewards.py       # reach / lift / touch / force / action penalties
       ├── terminations.py  # object_height_below / stable_grasp_hold / workspace
       └── events.py        # pick-lift reset randomization + curriculum

.. note::

   ``assets/robotiq_2f85/`` 下的 ``scene_*.xml``（``scene.xml`` /
   ``scene_pts_spheres.xml`` / ``scene_tactile.xml``）**不是训练入口**，
   只是给 ``mujoco.viewer.launch`` / ``scripts/view_env.py`` /
   ``scripts/check_mjcf.py`` 这类一次性预览用的独立 MJCF（里面写死了
   floor / light / 单个 freejoint object / tendon anchor）。

   mjlab 训练实际加载的是裸夹爪 ``2f85_pts_spheres.xml``，再由
   ``mjlab.scene.Scene`` 把它和 tabletop primitive objects（``cube_24mm`` /
   ``box_tall`` / ``cylinder_24mm``）一起 ``attach`` 进 mjlab 自带的
   ``scene.xml`` 骨架。修改 ``scene_*.xml`` 不会影响训练，只影响独立预览。

.. note::

   **多环境如何在物理层分散**：``SceneCfg`` 里挂了
   ``terrain=TerrainEntityCfg(terrain_type="plane")``，这一项做两件事：

   1. 给场景加一块共享地面（mjlab ``scene.xml`` 骨架本身没有 floor）。
   2. 让 ``TerrainEntity`` 用 ``env_spacing`` 生成 ``env_origins`` 网格。

   配合 mjlab 对 fixed-base entity 的 ``auto_wrap_fixed_base_mocap`` 自动
   mocap 包装，``reset_scene_to_default`` 每次会把每个 env 的夹爪 mocap
   pose 写到对应格点，多 env 在 **物理 + 可视化** 两层都自然分开，不需要
   任何 viewer 侧补丁。

   历史上仓库里曾存在 ``scripts/play_tiled.py`` +
   ``src/tactile_grasp/viewer_tiling.py``，那是早期没用 terrain 时为了在
   viser viewer 里看到多 env 而做的纯视觉 monkey-patch；现已退役删除。
   想避免回归请看 ``tests/test_no_viewer_tiling_hack.py``。

任务注册
--------

注册发生在 ``tactile_grasp/__init__.py`` 导入时：

.. code-block:: python

   from tactile_grasp.constants import TASK_ID
   from tactile_grasp.env_cfgs import make_tactile_grasp_env_cfg
   from tactile_grasp.rl_cfg import tactile_grasp_ppo_runner_cfg

   register_mjlab_task(
       task_id=TASK_ID,
       env_cfg=make_tactile_grasp_env_cfg(play=False),
       play_env_cfg=make_tactile_grasp_env_cfg(play=True),
       rl_cfg=tactile_grasp_ppo_runner_cfg(),
       runner_cls=None,
   )

``scripts/train.py`` 与 ``scripts/play.py`` 都只是先 ``import tactile_grasp``
触发上面这段注册，然后转发到 ``mjlab.scripts.train.main`` /
``mjlab.scripts.play.main``。

观测维度
--------

``obs["actor"]`` 在主线任务下是 332 维：

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - 观测项
     - 每步形状
     - history_length
     - 总贡献维度
   * - left_taxel_normal
     - ``[B, 9]``
     - 5
     - 45
   * - left_taxel_tangential
     - ``[B, 18]``
     - 5
     - 90
   * - right_taxel_normal
     - ``[B, 9]``
     - 5
     - 45
   * - right_taxel_tangential
     - ``[B, 18]``
     - 5
     - 90
   * - left_pad_force
     - ``[B, 3]``
     - 3
     - 9
   * - left_pad_torque
     - ``[B, 3]``
     - 3
     - 9
   * - right_pad_force
     - ``[B, 3]``
     - 3
     - 9
   * - right_pad_torque
     - ``[B, 3]``
     - 3
     - 9
   * - gripper_command
     - ``[B, 1]``
     - 1
     - 1
   * - joint_pos
     - ``[B, 6]``
     - 1
     - 6
   * - joint_vel
     - ``[B, 6]``
     - 1
     - 6
   * - vision_proxy
     - ``[B, 8]``
     - 1
     - 8
   * - last_action
     - ``[B, 5]``
     - 1
     - 5
   * - **合计**
     -
     -
     - **332**

每个 taxel 的 3D 力按 ``z`` 法向 / ``xy`` 切向拆成两个独立的 ObservationTerm，
方便后续做不同的 scale / encoder。

``vision_proxy`` 是未来视觉估计器或图像 encoder 的低维替代输入，包含 active
object 相对夹爪的 ``dx/dy/dz``、相对 yaw 的 ``sin/cos``，以及
``cube_24mm`` / ``box_tall`` / ``cylinder_24mm`` 的 one-hot 类型。

奖励 / 终止
-----------

奖励项（来自 ``mdp/rewards.py``）：

==================== ======= =====================================
奖励项                权重    作用
==================== ======= =====================================
``alive``             +1.0   未终止时给 +1
``reach_xy``          -2.0   惩罚夹爪与 active object 的水平距离
``lift_height``       +8.0   鼓励 active object 被抬高
``touch``             +1.0   触觉信号超过阈值时给接触奖励
``tactile_force``     -0.01  双指总 taxel force 平方和
``action_rate``       -0.001 raw action 平方
``close_command``     -0.001 归一化命令 ``(u/255)^2``
``drop_penalty``      -5.0   object_drop 触发时 -5
==================== ======= =====================================

终止项（来自 ``mdp/terminations.py`` 与 ``mjlab.envs.mdp.time_out``）：

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - 终止项
     - 类型
     - 条件
   * - ``time_out``
     - 超时
     - 达到 ``episode_length_s``
   * - ``object_drop``
     - 失败
     - active object 低于地面容差
   * - ``robot_out_of_workspace``
     - 失败
     - mocap 命令越过 workspace（正常 clip 下应为 False）
   * - ``stable_grasp``
     - 成功
     - ``stable_grasp_hold`` 连续 25 步同时满足 lift height & touch

``stable_grasp_hold`` 是一个有状态 termination：每个 env 维护一个计数器，
每步在 ``height_ok & touch_ok`` 时 +1，否则清零，达到 ``hold_steps`` 返回 True。

动作
----

五维连续动作 ``[dx, dy, dz, dyaw, du] ∈ [-1, 1]^5``，由
``CartesianMocapAction`` 内部累积：

.. code-block:: text

   p_new = clip(p + [dx, dy, dz] * 0.01, workspace)
   yaw_new = clip(yaw + dyaw * 0.05, [-pi, pi])
   u_new = clip(u + du * delta_u_max, 0, 255)

``p/yaw`` 写入 robot mocap pose，``u`` 写入 XML 中的 ``split`` tendon target。
MuJoCo 侧仍然使用原始 2F-85 的 tendon + general actuator 抽象，没有引入理想
force actuator。

运行时数据流
------------

::

   policy(obs["actor"])
      -> action ∈ [-1, 1]^5
      -> CartesianMocapAction.process_actions()  # 累积到 mocap pose + u
      -> CartesianMocapAction.apply_actions()    # 写 mocap pose + tendon target
      -> MuJoCo step (decimation=10, dt=0.002)
      -> builtin force/torque sensors update
      -> mdp.observations 拼装 332 维 actor obs（带 history buffer）
      -> mdp.rewards / mdp.terminations 计算 reward 与 termination
      -> env 返回 obs, reward, terminated, truncated

train / play 差异
-----------------

``make_tactile_grasp_env_cfg(play=True)`` 只改三件事：

- ``cfg.scene.num_envs = 1``
- ``cfg.episode_length_s = 6.0``
- ``cfg.observations["actor"].enable_corruption = False``

其余（actor / critic 观测、reward、terminations、actuator）与 train 完全一致。
play 模式的 reset/curriculum 固定为 Stage 2，用于 viewer 中检查完整随机范围。
train 模式按 ``common_step_counter`` 逐步开放随机范围：

- Stage 0：只用 cube，物体固定中心，夹爪对准物体。
- Stage 1：启用 cube/box，物体 ``xy`` 和 yaw 小范围随机。
- Stage 2：启用 cube/box/cylinder，物体 ``xy`` 和 yaw 全范围随机。

与 mjlab 的关系
---------------

- ``pyproject.toml`` 通过 ``mjlab.tasks`` entry point 暴露 ``tactile_grasp``
- 任务注册借助 ``mjlab.tasks.registry.register_mjlab_task``
- 训练 / play 直接复用 ``mjlab.scripts.train.main`` /
  ``mjlab.scripts.play.main``
- ``ManagerBasedRlEnvCfg`` 是唯一的环境装配中心

当前验证状态
------------

- MJCF 编译检查
- ``reset()`` / ``step()`` smoke test（``scripts/smoke_env.py``）
- Viewer 可视化（``scripts/view_env.py``）
- 观测维度回归测试（``tests/test_observation_shapes.py`` ⇒ 332）
- 最小 PPO smoke run（``WANDB_MODE=offline ... --agent.max-iterations 100``）

当前明确未实现 / 后续项
-----------------------

- slip proxy / 滑移检测
- 目标接触过滤（区分目标物体 vs 其他几何）
- 迁移到 mjlab managed contact sensor 接口
- 长期 PPO 收敛 benchmark
- 图像 observation / CNN policy（当前只有 debug camera，policy 使用 ``vision_proxy``）
- sim2real 部署链路

文档地图
--------

- :doc:`tactile_pipeline` —— 触觉信号从 XML 传感器到 obs tensor
- :doc:`reward_design` —— reward / termination 的当前定义
- :doc:`control_pipeline` —— 动作语义、actuator 封装、PPO 配置
- :doc:`pts_taxel_scheme` —— PTS sphere-taxel 的 MJCF 建模
