# AGENTS

## 当前目标

本仓库当前目标是先建立 MuJoCo tactile gripper baseline，而不是立即追求完整 sim2real 系统。

## 实现优先级

按以下顺序推进：

1. MJCF 可加载
2. 传感器读数正确
3. 环境 API 稳定
4. 训练脚本可跑通

## 代码组织约束

- MJCF 资源放在 `assets/`
- 环境逻辑放在 `src/.../envs/`
- 观测、奖励、终止逻辑分层实现
- 调试与可视化脚本放在 `scripts/`

## 修改原则

- 不要覆盖原始 `2f85.xml`
- 触觉版本应派生为 `2f85_tactile.xml`
- 优先做可验证的小步改动，不做大而全重构

## 观测与奖励边界

- MuJoCo 内部真值可用于 reward、termination 和 debug
- 最终 policy observation 不应依赖内部接触真值
- 默认先实现位置增量控制 `Δu`，不实现理想力控接口

## 运行与验证

每次重要改动后至少验证：

- 模型可编译加载
- 触觉或 wrench 传感器有合理输出
- 环境 reset / step 正常
- 最小训练或 smoke test 可运行

## 不要做的事

- 不要引入理想 force actuator 替代原始 2F-85 控制抽象
- 不要过早把 pad 切成多个碰撞 geom
- 不要在第一阶段把动作扩成 `[Δu, Δforce_limit]`
- 不要把研究方案文档当成实现规范直接执行
