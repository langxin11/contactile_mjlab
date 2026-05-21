PTS Sphere-Taxel 建模
=====================

本文只讨论 ``PTSSpheres`` 任务对应的 MJCF 建模，不讨论 reward 或训练逻辑。

资产分层
--------

当前资产关系是：

- ``assets/robotiq_2f85/2f85.xml``：原始 Robotiq 2F-85 模型
- ``assets/robotiq_2f85/2f85_tactile.xml``：TouchSite 对照模型
- ``assets/robotiq_2f85/2f85_pts_spheres.xml``：PTSSpheres 主线模型
- ``assets/robotiq_2f85/scene_pts_spheres.xml``：带 object 的主线场景

这三层是“派生”关系，不覆盖原始文件。
其中 ``2f85_pts_spheres.xml`` 现在可以通过
``scripts/generate_pts_spheres_xml.py`` 从 ``2f85.xml`` 重复生成，不再依赖手工维护。

当前建模原则
------------

``PTSSpheres`` 当前采用：

- 左右指各 9 个 sphere taxel geom
- 每个 taxel geom 对应一个 builtin ``<force>`` sensor
- 每侧仍保留一个全局 pad ``force`` / ``torque`` 参考 site
- taxel site 坐标系绕 pad 局部 Y 轴旋转 ``-90°``，用于定义传感器坐标系
- 旧 touch site 不出现在 ``PTSSpheres`` 模型里，只保留在 ``TouchSite`` 对照模型中

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
       <site name="left_taxel_site_00"/>
     </body>
     ...
   </body>

   <sensor>
     <force name="left_taxel_force_00" site="left_taxel_site_00"/>
     ...
   </sensor>

也就是说，当前 per-taxel 力读数是“site 上的 builtin force sensor”，不是后处理生成的伪传感器。

命名约定
--------

命名统一采用 ``{side}_taxel_{kind}_{row}{col}``：

- ``left_taxel_body_00`` .. ``left_taxel_body_22``
- ``left_taxel_geom_00`` .. ``left_taxel_geom_22``
- ``left_taxel_site_00`` .. ``left_taxel_site_22``
- ``left_taxel_force_00`` .. ``left_taxel_force_22``

右侧全部用 ``right_`` 前缀镜像命名。

当前 XML 参数
-------------

下表只列当前实现中真正写进 XML 的关键参数：

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

``taxel site quat`` 会被 MuJoCo 归一化，等价于绕 pad 局部 Y 轴旋转 ``-90°``：
``site.Z = -pad.X`` 指向接触法向内侧，``site.X = +pad.Z`` 沿 pad 长度指向指尖，
``site.Y = +pad.Y`` 保持横向。

文档里不再保留未写入当前 XML 的 ``condim``、替代 ``solimp`` 或其他候选参数说明。

与 TouchSite 的关系
-------------------

两套模型的关系应理解为：

- TouchSite：保留旧的 ``3×3`` 标量接触读数，适合回归和 baseline 对照
- PTSSpheres：切换为 per-taxel 三轴力，是当前主线任务

两者共享原始 2F-85 的主体机构、tendon、actuator 和对象场景，只更换触觉层。

相关页面
--------

- 坐标系检查推荐使用 ``scripts/inspect_pts_frames.py``，可直接查看 pad 与 taxel site/sensor 的局部坐标系。
- :doc:`task_architecture`
- :doc:`tactile_pipeline`
