PTS Sphere-Taxel 建模
=====================

本文只讨论主线 ``Mjlab-TactileGrasp-Robotiq2F85`` 任务对应的 MJCF 建模，
不讨论 reward 或训练逻辑。

资产分层
--------

主线资产已经迁入包内（``tactile_grasp/assets/``）：

- ``robotiq_2f85/2f85.xml`` —— 原始 Robotiq 2F-85 模型
- ``robotiq_2f85/2f85_pts_spheres.xml`` —— PTSSpheres 主线模型
- ``robotiq_2f85/scene_pts_spheres.xml`` —— 主线场景
- ``robotiq_2f85/2f85_tactile.xml`` —— 旧 TouchSite 模型（仅作历史保留，主线不再加载）
- ``robotiq_2f85/scene_tactile.xml`` —— TouchSite 对照场景

``2f85_pts_spheres.xml`` 通过 ``scripts/generate_pts_spheres_xml.py`` 从
``2f85.xml`` 重复生成，不再依赖手工维护。

当前建模原则
------------

主线模型采用：

- 左右指各 9 个 sphere taxel geom
- 每个 taxel geom 对应一个 builtin ``<force>`` sensor
- 每侧保留一个全局 pad ``force`` / ``torque`` 参考 site（``left_pad_force`` /
  ``left_pad_torque`` / 右侧同名）
- taxel site 通过 ``quat="1 0 -1 0"`` 旋转，使 ``site.Z`` 指向 pad 法向，
  ``site.X / site.Y`` 为切向 —— 与实物 PTS 传感器约定对齐

当前没有做的事同样明确：

- 不拆分 pad collision geom
- 不引入额外的 PTS visual-only 网格资产

XML 结构
--------

每个指尖的 taxel 结构是：

::

   <body name="left_pad">
     <geom name="left_pad1"/>
     <geom name="left_pad2"/>
     <body name="left_taxel_body_00">
       <geom name="left_taxel_geom_00" type="sphere"/>
       <site name="left_taxel_site_00" quat="1 0 -1 0"/>
     </body>
     ...
   </body>

   <sensor>
     <force name="left_taxel_force_00" site="left_taxel_site_00"/>
     ...
   </sensor>

per-taxel 力读数是“site 上的 builtin force sensor”，不是后处理生成的伪传感器。

命名约定
--------

统一采用 ``{side}_taxel_{kind}_{row}{col}``：

- ``left_taxel_body_00`` .. ``left_taxel_body_22``
- ``left_taxel_geom_00`` .. ``left_taxel_geom_22``
- ``left_taxel_site_00`` .. ``left_taxel_site_22``
- ``left_taxel_force_00`` .. ``left_taxel_force_22``

右侧用 ``right_`` 前缀镜像。``tactile_grasp.constants`` 暴露这些名称作为
``LEFT_TAXEL_FORCE_SENSOR_NAMES`` / ``RIGHT_TAXEL_FORCE_SENSOR_NAMES`` 元组。

当前 XML 参数
-------------

下表只列当前 XML 中真正写入的关键参数：

====================== ==========================
参数                   当前值
====================== ==========================
taxel geom type        ``sphere``
sphere radius          ``0.0028`` m
taxel site size        ``0.0008`` m
taxel site quat        ``1 0 -1 0``
taxel geom mass        ``1e-6``
taxel geom friction    ``0.7 0.03 0.01``
taxel geom solimp      ``0.95 0.99 0.001``
taxel geom solref      ``0.004 1``
taxel geom priority    ``2``
====================== ==========================

site quat ``1 0 -1 0`` 等价于绕 pad 局部 Y 轴旋转 ``-90°``：``site.Z`` 指向
接触法向（沿 ``-pad.X`` 方向，朝指间），``site.X = +pad.Z`` 沿 pad 长度指向
指尖，``site.Y = +pad.Y`` 保持横向。

观测拆分如何对应
----------------

观测层在 ``mdp/observations.py`` 中按 site-local 轴做拆分：

- ``taxel_normal_force`` 取 ``site.Z`` 分量，形状 ``[B, 9]``
- ``taxel_tangential_force`` 取 ``site.X / site.Y`` 分量并 flatten，形状 ``[B, 18]``

也就是说，是 *XML 中的 quat* 把 site 转到了 normal-aligned 朝向，观测代码不再
做坐标变换。

调试工具
--------

- 坐标系检查：``scripts/inspect_pts_frames.py`` 用 ``mjviser`` 动态推进并
  可视化 pad / taxel site / sensor 的局部坐标系
- MJCF 自检：``scripts/check_mjcf.py``
- 触觉阵列可视化：``scripts/visualize_taxels.py``

相关页面
--------

- :doc:`task_architecture`
- :doc:`tactile_pipeline`
