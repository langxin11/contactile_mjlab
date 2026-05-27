# Top-Down Pick-Lift Reward Refactor Design

Date: 2026-05-27

## Context

当前 `tactile_grasp` 的 top-down pick-lift baseline 已经具备可训练的任务骨架，但 reward 与触觉观测标定仍然偏向“少动作、少接触、少闭合”的保守行为。当前主线目标不是完整 sim2real 系统，而是建立一个可验证、可训练的 MuJoCo tactile gripper baseline，因此本轮调整只重构 reward 与 tactile observation calibration，使策略更稳定地学到：

1. 对准物体上方；
2. 下探接近；
3. 轻触并形成多点接触；
4. 闭合夹爪；
5. 抬升物体；
6. 稳定悬停。

## Scope

本轮只改以下内容：

- `src/tactile_grasp/mdp/observations.py` 中与触觉力和 tool pose 相关的 helper；
- `src/tactile_grasp/mdp/rewards.py` 中 reward helper 的重构；
- `src/tactile_grasp/mdp/events.py` 中 reset 时缓存 active object 初始高度及 reward 所需状态；
- `src/tactile_grasp/env_cfgs.py` 中 observation scale 与 reward term 装配；
- 与以上行为直接相关的测试和设计文档。

本轮明确不改：

- task id；
- action 接口和动作维度；
- curriculum 阶段划分；
- robot-floor collision 的 termination 行为；
- reset 初始高度采样逻辑；
- `play` 强制 stage 2 的现有行为。

## Constraints

实现必须满足以下约束：

- policy observation 不能依赖 MuJoCo 内部接触真值；
- 继续使用 Robotiq `u in [0, 255]` 控制接口；
- 总 reward 采用加权和，不使用负对数和乘法门控作为主组合方式；
- 正向项采用有界平滑项或归一化计数项；
- 负向项采用二值罚或简单范数罚；
- 地面碰撞只重罚，不终止 episode；
- 触觉力量程限制只在 observation 侧通过裁剪体现，不额外引入 reward 侧超量程惩罚；
- 夹爪闭合惩罚保留，且控制量越大惩罚越高。

## Observation Design

### Tactile force clipping

对每个 taxel 的 3D 力先做硬裁剪，再输出给 actor/critic：

- tangential `x/y`: 裁剪到 `[-4.0, 4.0]`；
- normal `z`: 裁剪到 `[-15.0, 15.0]`。

对应 helper 行为：

- `taxel_tangential_force()` 返回裁剪后的 `xy`，shape 不变；
- `taxel_normal_force()` 返回裁剪后的 `z`，shape 不变。

本轮不改 `pad_torque()`。`pad_force()` 默认不裁剪，除非实现时发现它已直接进入 actor 且量级明显失衡；本轮设计默认保持现状。

### Observation scaling

`env_cfgs` 中同步调整 tactile observation scale：

- `taxel_tangential_force`: `scale = 1 / 4.0`
- `taxel_normal_force`: `scale = 1 / 15.0`

其余 actor/critic observation 项的 shape 与 history length 保持不变。

### Tool position

`reach3d` 和 `align` 不再允许复用 `robot_position()`，因为当前实现返回 robot root position，不能保证代表真实末端/tool 位置。

新增 `tool_position()` helper，定义为左右 `pad_ft_site` 世界坐标的中点。若运行时找不到 site，则实现可以退回到明确的 mocap pose，但默认路径必须优先使用 pad site midpoint。

## Reward Design

总 reward 定义为：

`r = w_reach * r_reach3d + w_align * r_align + w_contact * r_contact + w_cover * r_coverage + w_lift * r_lift_delta + w_hold * r_hold - w_floor * c_floor - w_act * c_act - w_close * c_close - w_drop * c_drop`

### Positive terms

#### `r_reach3d`

定义：

`exp(-k_pos * ||p_obj - p_tool||_2)`

作用：鼓励 tool 中心接近物体，保留平滑梯度。

#### `r_align`

定义：

`exp(-k_xy * ||delta_xy||_2)`

其中 `delta_xy` 是物体与 tool 的平面位置差。

作用：单独强化“从正上方对准”，避免策略只靠 z 接近。

#### `r_contact`

固定为二值弱奖励，不采用连续比例版。

定义：

`1.0` 当任意单个 taxel 满足 `||f_xyz||_2 > 0.05`，否则 `0.0`。

作用：仅表达“已经发生有效触觉接触”，权重低于 `coverage` 和 `lift`。

#### `r_coverage`

定义：

`0.5 * min(n_left, 9) / 9 + 0.5 * min(n_right, 9) / 9`

其中 `n_left` 与 `n_right` 为左右两侧 active taxel 数，active 的判定使用与 `r_contact` 相同的 per-taxel threshold，即 `||f_xyz||_2 > 0.05`。

作用：鼓励双侧、多点、均衡接触，高于简单 contact reward。

#### `r_lift_delta`

定义：

`relu(z_obj - z_obj_init)`

其中 `z_obj_init` 为 reset 后 active object 的初始高度。该项不再直接奖励物体绝对高度，避免桌面静止高度带来的常数偏置。

#### `r_hold`

定义：

当 `lift_delta > h_hold` 且左右两侧都至少存在一个 active taxel 时，给固定小 bonus，否则为 `0`。

作用：在成功 termination 之外，额外偏好“抬起后仍稳定夹持”的状态。

### Penalty terms

#### `c_floor`

定义：

当任意 robot geom 与 floor geom 接触时，返回 `1.0`，否则 `0.0`。

实现按 env 维度聚合，不按接触对数量累加。物体与 floor 的接触不计入该项。

#### `c_act`

定义：

`sum_i |a_i(t) - a_i(t-1)|`

作用：作为弱动作平滑正则，直接复用已有 `last_action` / `prev_action` 语义，不主导策略。

#### `c_close`

定义：

`(u / 255.0)^2`

其中 `u` 为 Robotiq 归一化前控制量。

作用：保留闭合代价，但权重低于 `coverage` 与 `lift`，避免再次把策略压成完全不开夹爪。

#### `c_drop`

保留现有 object drop penalty 通道，语义不变。

### Removed reward terms

以下旧项从环境 reward term 装配中删除：

- `alive`
- `reach_xy`
- `lift_height`
- `tactile_force_l2`

`close_command_l2` 保留实现，但作为正式 `c_close` 使用。

## State and Helper Plumbing

### Reset-time cached state

在 reset 事件中缓存：

- `env._tactile_active_object_init_z`

其值为每个 env 当前 active object 的 reset 初始世界高度。

### New observation/reward helpers

新增或补齐以下 helper：

- `tool_position()`
- `taxel_contact_mask()`
- `taxel_contact_count()`
- `robot_floor_collision()`

设计要求：

- `taxel_contact_mask()` 输出每个 taxel 是否 active 的布尔张量；
- `taxel_contact_count()` 支持按侧统计 active taxel 数；
- `robot_floor_collision()` 通过缓存 robot 全部 geom ids 与 floor geom id，逐步扫描 MuJoCo contact buffer，按 env 返回二值结果。

### Contact buffer scan

`robot_floor_collision()` 的固定实现路径：

1. 缓存 robot 全部 geom ids；
2. 缓存 floor geom id；
3. 每步扫描 contact buffer；
4. 判断是否存在 `robot geom <-> floor geom` 接触；
5. 按 env 聚合为二值输出。

不允许把 object-floor 接触误判为 robot-floor collision。

## Default Parameters

### Reward weights

默认权重固定为：

- `w_reach = 0.6`
- `w_align = 0.8`
- `w_contact = 0.2`
- `w_cover = 1.2`
- `w_lift = 8.0`
- `w_hold = 2.0`
- `w_floor = 12.0`
- `w_act = 0.01`
- `w_close = 0.05`
- `w_drop = 5.0`

这些默认值体现以下优先级：

- floor collision penalty 最高；
- `lift_delta` 高于 `reach` 与 `contact`；
- `coverage` 高于简单 `contact`；
- `close penalty` 弱于 `coverage/lift`；
- `action smoothness` 仅作弱正则。

### Shape parameters

实现中同时固定以下默认参数：

- `contact_threshold = 0.05`
- `k_pos = 10.0`
- `k_xy = 20.0`
- `h_hold = 0.03`

若实现中需要将这些值集中到 `env_cfgs` 常量区，应保持语义与默认值一致。

## Testing

至少覆盖以下测试：

- observation clipping：
  - `taxel_tangential_force()` 输出不超过 `[-4, 4]`
  - `taxel_normal_force()` 输出不超过 `[-15, 15]`
  - actor observation shape 不变
- `lift_delta`：
  - reset 后桌面静止物体时接近 `0`
  - 抬高后严格增大
- `action_smoothness`：
  - `a_t == a_{t-1}` 时为 `0`
  - 动作变化越大 penalty 越大
- `close penalty`：
  - `u` 越大惩罚越大
  - `u = 0` 时最小
- `taxel_coverage`：
  - active taxel 数更多时 reward 更高
  - 双侧均衡高于单侧集中
- `robot_floor_collision`：
  - 任意 robot geom 与 floor 接触时触发
  - object-floor 接触不触发
- regression：
  - `smoke_env`
  - `pytest`
  - reward 名称或文档描述变更时，同步更新设计文档

## Implementation Notes

实现顺序建议如下：

1. 先用单元测试锁定 clipping、lift delta、smoothness、coverage、floor collision、tool position；
2. 再重写 `observations.py` 和 `rewards.py`；
3. 然后更新 `events.py` 的 reset-time cache；
4. 最后调整 `env_cfgs.py` 的 reward 装配、scale 和常量命名；
5. 同步更新 `docs/source/design/reward_design.rst` 等设计文档。

## Out of Scope

本轮不处理以下事项：

- reset 采样策略重设；
- 新 action channel；
- 新 tactile sensor variant；
- robot-floor collision termination；
- sim2real deployment 逻辑；
- force saturation 的 reward 处罚；
- `pad_force` 全量重标定。
