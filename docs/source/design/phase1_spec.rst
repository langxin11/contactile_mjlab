Phase 1 实现规格
==================

目标
----

第一阶段默认直接做 V1：保留 V0 的调试思路，但正式环境接口以 ``3x3`` 触觉阵列版本为主。全局 wrench 作为辅助观测和调试工具保留，不单独做一套独立环境。

环境接口
--------

- 提供一个单环境类，职责是 Robotiq 2F-85 触觉抓取 baseline
- 动作为一维连续标量，范围 ``[-1, 1]``
- 动作语义是位置命令增量 ``Δu``
- 环境内部维护当前夹爪命令 ``u``
- 每一步更新规则为：``u = clip(u + action * delta_u_max, 0, 255)``

默认参数：

- ``delta_u_max = 3.0``
- 观测类型为 ``float32``

观测定义
--------

第一阶段观测由以下部分拼接：

- 左指 ``3x3`` touch map，共 ``9``
- 右指 ``3x3`` touch map，共 ``9``
- 左指全局 force，共 ``3``
- 左指全局 torque，共 ``3``
- 右指全局 force，共 ``3``
- 右指全局 torque，共 ``3``
- 当前夹爪命令 ``u / 255``，共 ``1``
- 夹爪相关位置状态，共 ``nq``
- 夹爪相关速度状态，共 ``nv``
- 上一步动作，共 ``1``

约束：

- touch map 展平后拼接
- touch、force、torque 均做固定比例归一化
- 无接触时触觉值应接近零
- 不向策略暴露 MuJoCo 内部 contact list、contact normal 或 object truth state

奖励与终止
----------

第一阶段奖励定义为：

- 基础存活奖励 ``+1.0``
- 力惩罚：基于左右 ``3x3`` touch map 总和的平方项
- 动作惩罚：``action^2``
- 闭合惩罚：``(u / 255)^2``
- 掉落惩罚：掉落时施加固定负奖励

默认只保留这些项，不在第一阶段加入滑移奖励、复杂接触平衡项或 curriculum-specific shaping。

终止条件：

- 物体掉落
- 达到最大步数
- 仿真出现非法状态或无法恢复的数值错误

成功条件：

- 在 episode 结束前，物体在连续一段时间内保持稳定抓持且未掉落

模型与文件约束
--------------

- 原始模型保留为 ``assets/robotiq_2f85/2f85.xml``
- 触觉模型派生为 ``assets/robotiq_2f85/2f85_tactile.xml``
- 左右指尖各添加 ``9`` 个 touch site
- 左右指尖各添加一个全局 force / torque 参考 site
- 第一阶段不拆分 pad collision geom

Smoke Test
----------

第一阶段最小验收标准：

1. ``2f85_tactile.xml`` 能成功加载
2. viewer 中可见左右 ``3x3`` taxel site
3. 接触物体时至少部分 touch sensor 输出非零
4. 全局 force / torque 在接触时有响应
5. 环境 ``reset()`` / ``step()`` 可运行一个短 episode
6. 随机动作下观测 shape、dtype、范围稳定
