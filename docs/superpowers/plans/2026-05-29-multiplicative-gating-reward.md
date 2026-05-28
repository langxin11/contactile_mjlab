# Multiplicative-Gating Reward Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the additive `reach3d + align + close_near_object + contact + coverage + lift_delta` bootstrap chain in the `tactile_grasp` env with a single multiplicatively-gated `staged_pickup` reward term, so the policy must progress through each phase to unlock the next phase's signal.

**Architecture:** One new reward function (`staged_pickup`) in `tactile_grasp.mdp.rewards`, internally composing the existing `taxel_coverage` and `lift_delta` primitives plus a new anisotropic distance metric. `env_cfgs.make_tactile_grasp_env_cfg` swaps six reward dict entries for one. Four now-unused reward functions (`reach3d`, `align_xy`, `close_near_object`, `tactile_contact_binary`) and their tests are deleted. Three docs files updated. No observation, action, or curriculum changes.

**Tech Stack:** Python 3 · PyTorch · mjlab (`ManagerBasedRlEnvCfg`, `RewardTermCfg`) · pytest · ruff · uv

**Spec:** `docs/superpowers/specs/2026-05-29-multiplicative-gating-reward.md` (locked).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/tactile_grasp/mdp/rewards.py` | Modify | Add `staged_pickup`; delete `reach3d`, `align_xy`, `close_near_object`, `tactile_contact_binary` |
| `src/tactile_grasp/env_cfgs.py` | Modify | Swap 6 reward dict entries for 1; update module constants |
| `tests/test_reward_refactor.py` | Modify | Add 1 helper + 5 new tests; delete 6 obsolete tests; update cfg test |
| `docs/source/api/mdp_rewards.rst` | Modify | Update top-line reward list |
| `docs/source/design/reward_design.rst` | Modify | Replace reward table with new composition + explain multiplicative gating |
| `docs/source/design/task_architecture.rst` | Modify | Update reward table |

No new files are created. The test file keeps its name (`test_reward_refactor.py`) since the scope — verifying reward refactor invariants — still fits.

---

## Task 1 — Add `staged_pickup` reward function (TDD)

**Files:**
- Modify: `src/tactile_grasp/mdp/rewards.py` (add new function only; do NOT delete anything yet)
- Test: `tests/test_reward_refactor.py` (append helper + 5 new tests)

This task adds the new reward function and its unit tests without touching cfg wiring. After this task, the env cfg still uses the old additive chain; the new function exists but is unwired. All previously-passing tests remain green.

- [ ] **Step 1: Add the dependency-patching context manager**

Append to `tests/test_reward_refactor.py` near the top (after the existing imports, before the first test). Add `contextlib` to the existing imports.

Replace the existing imports block (lines 1-11):

```python
"""Reward refactor unit tests for top-down pick-lift."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tactile_grasp import load_env_cfg
from tactile_grasp.mdp import events, observations, rewards
```

Then append the helper at the end of the file (or anywhere after `_FakeEnv` definition):

```python
@contextmanager
def _staged_pickup_deps(
    obj_pos: torch.Tensor,
    tool_pos: torch.Tensor,
    command: torch.Tensor,
    coverage: torch.Tensor,
    lift: torch.Tensor,
):
    """Patch the observation/reward primitives that staged_pickup composes."""
    originals = (
        observations.active_object_position,
        observations.tool_position,
        observations.gripper_command,
        rewards.taxel_coverage,
        rewards.lift_delta,
    )
    observations.active_object_position = lambda _e: obj_pos
    observations.tool_position = lambda _e: tool_pos
    observations.gripper_command = lambda _e, action_name="cartesian_gripper": command
    rewards.taxel_coverage = lambda _e, **_kw: coverage
    rewards.lift_delta = lambda _e: lift
    try:
        yield
    finally:
        (
            observations.active_object_position,
            observations.tool_position,
            observations.gripper_command,
            rewards.taxel_coverage,
            rewards.lift_delta,
        ) = originals
```

- [ ] **Step 2: Append the 5 failing tests for `staged_pickup`**

Append at the end of `tests/test_reward_refactor.py`:

```python
def test_staged_pickup_returns_near_zero_at_full_separation() -> None:
    """staged_pickup should approach zero when tool is far from object and gripper open."""
    env = _FakeEnv(num_envs=1)
    with _staged_pickup_deps(
        obj_pos=torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32),
        tool_pos=torch.tensor([[0.15, 0.15, 0.25]], dtype=torch.float32),
        command=torch.tensor([[0.0]], dtype=torch.float32),
        coverage=torch.tensor([0.0], dtype=torch.float32),
        lift=torch.tensor([0.0], dtype=torch.float32),
    ):
        out = rewards.staged_pickup(
            env,
            k_pos=10.0,
            k_d=30.0,
            lift_cap=0.08,
            left_sensor_names=("left_taxel_force_00",),
            right_sensor_names=("right_taxel_force_00",),
            threshold=0.005,
        )

    assert out.item() < 0.01


def test_staged_pickup_saturates_at_four_when_all_factors_are_one() -> None:
    """staged_pickup output equals 4 when every factor saturates at 1."""
    env = _FakeEnv(num_envs=1)
    with _staged_pickup_deps(
        obj_pos=torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32),
        tool_pos=torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32),
        command=torch.tensor([[1.0]], dtype=torch.float32),
        coverage=torch.tensor([1.0], dtype=torch.float32),
        lift=torch.tensor([0.08], dtype=torch.float32),
    ):
        out = rewards.staged_pickup(
            env,
            k_pos=10.0,
            k_d=30.0,
            lift_cap=0.08,
            left_sensor_names=("left_taxel_force_00",),
            right_sensor_names=("right_taxel_force_00",),
            threshold=0.005,
        )

    assert torch.allclose(out, torch.tensor([4.0], dtype=torch.float32), atol=1e-5)


def test_staged_pickup_is_monotonic_in_each_factor() -> None:
    """Increasing any single factor (reach/close/contact/lift) strictly increases output."""
    env = _FakeEnv(num_envs=1)
    baseline_obj = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32)
    baseline_tool = torch.tensor([[0.0, 0.0, 0.05]], dtype=torch.float32)  # 3cm z gap
    baseline_command = torch.tensor([[0.5]], dtype=torch.float32)
    baseline_coverage = torch.tensor([0.5], dtype=torch.float32)
    baseline_lift = torch.tensor([0.02], dtype=torch.float32)

    kwargs = dict(
        k_pos=10.0,
        k_d=30.0,
        lift_cap=0.08,
        left_sensor_names=("left_taxel_force_00",),
        right_sensor_names=("right_taxel_force_00",),
        threshold=0.005,
    )

    with _staged_pickup_deps(
        baseline_obj, baseline_tool, baseline_command, baseline_coverage, baseline_lift
    ):
        baseline = rewards.staged_pickup(env, **kwargs).item()

    # Closer tool -> larger reach factor
    with _staged_pickup_deps(
        baseline_obj,
        torch.tensor([[0.0, 0.0, 0.03]], dtype=torch.float32),  # 1cm gap
        baseline_command,
        baseline_coverage,
        baseline_lift,
    ):
        closer = rewards.staged_pickup(env, **kwargs).item()
    assert closer > baseline, "reach should increase when distance shrinks"

    # Higher command -> larger close factor
    with _staged_pickup_deps(
        baseline_obj,
        baseline_tool,
        torch.tensor([[0.9]], dtype=torch.float32),
        baseline_coverage,
        baseline_lift,
    ):
        more_closed = rewards.staged_pickup(env, **kwargs).item()
    assert more_closed > baseline, "close should increase with gripper command"

    # More taxel coverage -> larger contact factor
    with _staged_pickup_deps(
        baseline_obj,
        baseline_tool,
        baseline_command,
        torch.tensor([0.9], dtype=torch.float32),
        baseline_lift,
    ):
        more_contact = rewards.staged_pickup(env, **kwargs).item()
    assert more_contact > baseline, "contact should increase with coverage"

    # Higher lift -> larger lift factor
    with _staged_pickup_deps(
        baseline_obj,
        baseline_tool,
        baseline_command,
        baseline_coverage,
        torch.tensor([0.05], dtype=torch.float32),
    ):
        more_lift = rewards.staged_pickup(env, **kwargs).item()
    assert more_lift > baseline, "lift should increase with object height gain"


def test_staged_pickup_anisotropic_distance_penalizes_xy_offset_more_than_z() -> None:
    """A 1cm xy offset should reduce the cascade more than a 1cm z offset."""
    env = _FakeEnv(num_envs=1)
    obj = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32)
    tool_xy_off = torch.tensor([[0.01, 0.0, 0.02]], dtype=torch.float32)
    tool_z_off = torch.tensor([[0.0, 0.0, 0.03]], dtype=torch.float32)
    command = torch.tensor([[0.0]], dtype=torch.float32)  # close=0 to isolate reach
    coverage = torch.tensor([0.0], dtype=torch.float32)
    lift = torch.tensor([0.0], dtype=torch.float32)

    kwargs = dict(
        k_pos=10.0,
        k_d=30.0,
        lift_cap=0.08,
        left_sensor_names=("left_taxel_force_00",),
        right_sensor_names=("right_taxel_force_00",),
        threshold=0.005,
    )

    with _staged_pickup_deps(obj, tool_xy_off, command, coverage, lift):
        xy_off = rewards.staged_pickup(env, **kwargs).item()
    with _staged_pickup_deps(obj, tool_z_off, command, coverage, lift):
        z_off = rewards.staged_pickup(env, **kwargs).item()

    assert xy_off < z_off, (
        f"xy offset should penalize reach more than z offset (xy={xy_off}, z={z_off})"
    )


def test_staged_pickup_lift_saturates_at_cap() -> None:
    """Lifting beyond lift_cap should not increase the cascade further."""
    env = _FakeEnv(num_envs=1)
    obj = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32)
    tool = torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32)
    command = torch.tensor([[1.0]], dtype=torch.float32)
    coverage = torch.tensor([1.0], dtype=torch.float32)

    kwargs = dict(
        k_pos=10.0,
        k_d=30.0,
        lift_cap=0.08,
        left_sensor_names=("left_taxel_force_00",),
        right_sensor_names=("right_taxel_force_00",),
        threshold=0.005,
    )

    with _staged_pickup_deps(obj, tool, command, coverage, torch.tensor([0.08], dtype=torch.float32)):
        at_cap = rewards.staged_pickup(env, **kwargs).item()
    with _staged_pickup_deps(obj, tool, command, coverage, torch.tensor([0.20], dtype=torch.float32)):
        beyond_cap = rewards.staged_pickup(env, **kwargs).item()

    assert torch.allclose(
        torch.tensor(at_cap), torch.tensor(beyond_cap), atol=1e-5
    ), f"cascade should saturate at lift_cap (at_cap={at_cap}, beyond_cap={beyond_cap})"
```

- [ ] **Step 3: Run the new tests and confirm they fail**

```bash
uv run pytest tests/test_reward_refactor.py -k staged_pickup -v
```

Expected: 5 tests fail with `AttributeError: module 'tactile_grasp.mdp.rewards' has no attribute 'staged_pickup'`.

- [ ] **Step 4: Implement `staged_pickup` in `src/tactile_grasp/mdp/rewards.py`**

Append the new function at the end of `src/tactile_grasp/mdp/rewards.py` (after `tactile_contact`):

```python
def staged_pickup(
    env: "ManagerBasedRlEnv",
    k_pos: float,
    k_d: float,
    lift_cap: float,
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    threshold: float,
    action_name: str = "cartesian_gripper",
    entity_name: str = "robot",
) -> torch.Tensor:
    """Multiplicatively-gated bootstrap cascade for the pick-lift chain.

    Returns ``reach * (1 + close * (1 + contact * (1 + lift)))`` with all
    factors in ``[0, 1]``; output is in ``[0, 4]``.

    The reach factor uses an anisotropic distance with xy weighted twice as
    heavily as z, so xy alignment dominates over z proximity at equal raw
    distance.
    """
    delta = obs.active_object_position(env) - obs.tool_position(env)
    d_aniso = torch.sqrt(
        2.0 * (delta[:, 0] ** 2 + delta[:, 1] ** 2) + delta[:, 2] ** 2
    )
    reach = torch.exp(-k_pos * d_aniso)
    command = obs.gripper_command(env, action_name=action_name).squeeze(-1)
    close = command * torch.exp(-k_d * d_aniso)
    contact = taxel_coverage(
        env,
        left_sensor_names=left_sensor_names,
        right_sensor_names=right_sensor_names,
        threshold=threshold,
        entity_name=entity_name,
    )
    lift = torch.clamp(lift_delta(env) / lift_cap, max=1.0)
    return reach * (1.0 + close * (1.0 + contact * (1.0 + lift)))
```

- [ ] **Step 5: Run the new tests and confirm they pass**

```bash
uv run pytest tests/test_reward_refactor.py -k staged_pickup -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Run the full test file and confirm no regressions**

```bash
uv run pytest tests/test_reward_refactor.py -v
```

Expected: all tests pass (existing ones unaffected, 5 new ones green).

- [ ] **Step 7: Commit**

```bash
git add src/tactile_grasp/mdp/rewards.py tests/test_reward_refactor.py
git commit -m "$(cat <<'EOF'
feat(reward): add multiplicatively-gated staged_pickup cascade

reach * (1 + close * (1 + contact * (1 + lift))) replacement for the
additive bootstrap chain. Uses anisotropic 3D distance (xy weighted 2x
relative to z) and reuses taxel_coverage / lift_delta as inner factors.

Function added only; cfg wiring swap and old-function removal come in a
follow-up commit.
EOF
)"
```

---

## Task 2 — Swap cfg wiring and delete now-unused functions

**Files:**
- Modify: `src/tactile_grasp/env_cfgs.py` (replace 6 reward entries + 6 constants with 1 + 4)
- Modify: `src/tactile_grasp/mdp/rewards.py` (delete `reach3d`, `align_xy`, `close_near_object`, `tactile_contact_binary`)
- Modify: `tests/test_reward_refactor.py` (update cfg name test, delete 6 obsolete tests)

After this task, the env builds with `staged_pickup` as the sole shaping term; the old additive primitives no longer exist in the codebase.

- [ ] **Step 1: Update `test_pick_lift_cfg_uses_new_reward_term_names` to expect the new set**

In `tests/test_reward_refactor.py`, replace the existing test:

```python
def test_pick_lift_cfg_uses_new_reward_term_names() -> None:
    """Registered env cfg 应使用新的奖励项命名（含 close_near_object）."""
    cfg = load_env_cfg(play=False)

    reward_terms = cfg.rewards

    for name in (
        "reach3d",
        "align",
        "contact",
        "coverage",
        "lift_delta",
        "hold",
        "floor_collision",
        "action_smoothness",
        "close_near_object",
        "drop_penalty",
    ):
        assert name in reward_terms, name

    for name in ("alive", "reach_xy", "lift_height", "tactile_force", "close_command"):
        assert name not in reward_terms, name
```

with:

```python
def test_pick_lift_cfg_uses_multiplicative_gating_reward_terms() -> None:
    """Registered env cfg 使用单一 staged_pickup 级联项替代旧加法 bootstrap 链."""
    cfg = load_env_cfg(play=False)

    reward_terms = cfg.rewards

    for name in (
        "staged_pickup",
        "hold",
        "floor_collision",
        "action_smoothness",
        "drop_penalty",
    ):
        assert name in reward_terms, name

    for name in (
        "alive",
        "reach_xy",
        "lift_height",
        "tactile_force",
        "close_command",
        "reach3d",
        "align",
        "contact",
        "coverage",
        "lift_delta",
        "close_near_object",
    ):
        assert name not in reward_terms, name
```

- [ ] **Step 2: Delete obsolete unit tests in `tests/test_reward_refactor.py`**

Delete these six test functions (they exercise functions that will no longer exist):

- `test_reach3d_uses_tool_to_object_distance`
- `test_align_xy_ignores_z_offset`
- `test_tactile_contact_binary_returns_one_when_any_taxel_is_active`
- `test_close_near_object_is_command_value_at_zero_distance`
- `test_close_near_object_decays_to_near_zero_when_far`
- `test_close_near_object_is_zero_when_command_is_zero`

Keep all other tests untouched (`taxel_coverage`, `lift_delta`, `hold_bonus`, `floor_collision`, `action_smoothness_l1`, helpers tests, etc.).

- [ ] **Step 3: Run the test file to confirm Step 1 + 2 give an expected failure**

```bash
uv run pytest tests/test_reward_refactor.py::test_pick_lift_cfg_uses_multiplicative_gating_reward_terms -v
```

Expected: FAIL — `staged_pickup` is not yet in `cfg.rewards` (env cfg still uses old chain).

```bash
uv run pytest tests/test_reward_refactor.py -v
```

Expected: only `test_pick_lift_cfg_uses_multiplicative_gating_reward_terms` fails; no NameError or import error from the deleted tests.

- [ ] **Step 4: Update module constants in `src/tactile_grasp/env_cfgs.py`**

In `src/tactile_grasp/env_cfgs.py`, replace the constants block (lines 58-73):

```python
TACTILE_CONTACT_THRESHOLD = 0.005
REACH_K_POS = 10.0
ALIGN_K_XY = 20.0
HOLD_LIFT_THRESHOLD = 0.03

W_REACH = 0.6
W_ALIGN = 0.8
W_CONTACT = 0.2
W_COVERAGE = 1.2
W_LIFT_DELTA = 8.0
W_HOLD = 2.0
W_FLOOR = -12.0
W_ACTION_SMOOTHNESS = -0.01
W_CLOSE_NEAR = 2.5
CLOSE_NEAR_K_D = 30.0
W_DROP_PENALTY = -5.0
```

with:

```python
TACTILE_CONTACT_THRESHOLD = 0.005
HOLD_LIFT_THRESHOLD = 0.03

STAGED_PICKUP_K_POS = 10.0
STAGED_PICKUP_K_D = 30.0
STAGED_PICKUP_LIFT_CAP = 0.08

W_STAGED_PICKUP = 3.0
W_HOLD = 2.0
W_FLOOR = -12.0
W_ACTION_SMOOTHNESS = -0.01
W_DROP_PENALTY = -5.0
```

- [ ] **Step 5: Replace the rewards dict in `src/tactile_grasp/env_cfgs.py`**

In `make_tactile_grasp_env_cfg`, replace the entire `rewards={...}` block (lines 181-235) with:

```python
        rewards={
            "staged_pickup": RewardTermCfg(
                func=rewards.staged_pickup,
                weight=W_STAGED_PICKUP,
                params={
                    "k_pos": STAGED_PICKUP_K_POS,
                    "k_d": STAGED_PICKUP_K_D,
                    "lift_cap": STAGED_PICKUP_LIFT_CAP,
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "threshold": TACTILE_CONTACT_THRESHOLD,
                    "action_name": "cartesian_gripper",
                },
            ),
            "hold": RewardTermCfg(
                func=rewards.hold_bonus,
                weight=W_HOLD,
                params={
                    "left_sensor_names": LEFT_TAXEL_FORCE_SENSOR_NAMES,
                    "right_sensor_names": RIGHT_TAXEL_FORCE_SENSOR_NAMES,
                    "threshold": TACTILE_CONTACT_THRESHOLD,
                    "lift_threshold": HOLD_LIFT_THRESHOLD,
                },
            ),
            "floor_collision": RewardTermCfg(
                func=rewards.robot_floor_collision,
                weight=W_FLOOR,
            ),
            "action_smoothness": RewardTermCfg(
                func=rewards.action_smoothness_l1,
                weight=W_ACTION_SMOOTHNESS,
            ),
            "drop_penalty": RewardTermCfg(func=rewards.drop_penalty, weight=W_DROP_PENALTY),
        },
```

- [ ] **Step 6: Delete obsolete reward functions from `src/tactile_grasp/mdp/rewards.py`**

Delete these four functions entirely:

- `reach3d` (lines 49-52)
- `align_xy` (lines 55-58)
- `tactile_contact_binary` (lines 61-81)
- `close_near_object` (lines 152-171)

Keep `taxel_coverage`, `lift_delta`, `hold_bonus`, `action_smoothness_l1`, `robot_floor_collision`, `drop_penalty`, `staged_pickup`, and all helpers (`tactile_force_l2`, `total_tactile_signal`, `alive`, `action_l2`, `reach_xy`, `lift_height`, `tactile_contact`, `_ensure_collision_cache`).

- [ ] **Step 7: Run the full test file and confirm all tests pass**

```bash
uv run pytest tests/test_reward_refactor.py -v
```

Expected: all tests pass, including `test_pick_lift_cfg_uses_multiplicative_gating_reward_terms`.

- [ ] **Step 8: Verify the env still loads end-to-end**

```bash
uv run python -c "from tactile_grasp import load_env_cfg; cfg = load_env_cfg(play=False); print(sorted(cfg.rewards.keys()))"
```

Expected output: `['action_smoothness', 'drop_penalty', 'floor_collision', 'hold', 'staged_pickup']`

- [ ] **Step 9: Run the broader test suite to catch any cross-file regression**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (no other test file should reference the deleted reward functions).

- [ ] **Step 10: Commit**

```bash
git add src/tactile_grasp/mdp/rewards.py src/tactile_grasp/env_cfgs.py tests/test_reward_refactor.py
git commit -m "$(cat <<'EOF'
feat(reward): swap to multiplicative-gating cfg, drop additive primitives

env_cfgs now wires a single staged_pickup reward (W=3.0) in place of the
six additive bootstrap terms (reach3d, align, contact, coverage,
lift_delta, close_near_object). The now-unused reward functions and their
unit tests are deleted; module constants collapse from 10 weight/k values
to 5.
EOF
)"
```

---

## Task 3 — Update documentation

**Files:**
- Modify: `docs/source/api/mdp_rewards.rst`
- Modify: `docs/source/design/reward_design.rst`
- Modify: `docs/source/design/task_architecture.rst`

This task updates the three docs that reference the old reward composition. No code changes.

- [ ] **Step 1: Update `docs/source/api/mdp_rewards.rst`**

Replace lines 4-5:

```rst
``alive`` / ``tactile_force_l2`` / ``action_l2`` / ``close_near_object`` /
``drop_penalty``，以及被 termination 复用的 ``total_tactile_signal``。
```

with:

```rst
``alive`` / ``tactile_force_l2`` / ``action_l2`` / ``staged_pickup`` /
``drop_penalty``，以及被 termination 复用的 ``total_tactile_signal``。
```

- [ ] **Step 2: Replace the reward table and composition explanation in `docs/source/design/reward_design.rst`**

Replace lines 7-52 (the "当前奖励项" section through to the end of the reward table):

```rst
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

其中 ``d_aniso = sqrt(2·(Δx² + Δy²) + Δz²)`` 是 *各向异性* 3D 距离：xy 方向权重
是 z 方向的 2 倍，所以 xy 不对齐时 reach 衰减更快。这一个项替代了之前的
``reach3d`` / ``align`` / ``close_near_object`` / ``contact`` / ``coverage`` /
``lift_delta`` 六个加法项。
```

Then replace lines 110-122 (the "为什么这样组合 reward" section):

```rst
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

下表给出几个典型阶段的单步奖励（含 ``W_STAGED_PICKUP = 3.0``）：

============================== ====== ====== ======= ===== ======== =======
阶段                            reach  close  contact lift  cascade  reward
============================== ====== ====== ======= ===== ======== =======
Initial (far)                   0.30   0      0       0     0.30     0.90
Aligned + half-closed hover     0.70   0.14   0       0     0.80     2.40
First contact (4/9 taxels)      0.95   0.70   0.44    0     1.94     5.83
Lifted to 4 cm                  0.95   0.90   0.90    0.50  2.80     8.40
Saturated (≥ 8 cm)              1.00   1.00   1.00    1.00  4.00    12.00
============================== ====== ====== ======= ===== ======== =======

"悬停刷分"基线（2.40）严格小于"下探到接触"（5.83），policy 必须穿越接触门
才能拿到更高单步奖励。``hold`` / ``floor_collision`` / ``drop_penalty`` /
``action_smoothness`` 仍作为独立加法项（不参与 cascade）补足成功条件、
安全约束与运动平滑性。
```

- [ ] **Step 3: Update the reward table in `docs/source/design/task_architecture.rst`**

Replace lines 200-213 (the reward table):

```rst
============================== ======= =====================================
奖励项                          权重    作用
============================== ======= =====================================
``reach3d``                     +0.6    ``exp(-k_pos·‖p_obj − p_tool‖)``
``align``                       +0.8    ``exp(-k_xy·‖Δxy‖)``
``contact``                     +0.2    任一 taxel force 范数超 ``0.005 N`` 给 1
``coverage``                    +1.2    双指 3×3 taxel 激活比例平均
``lift_delta``                  +8.0    ``relu(z_obj − z_obj_init)``
``hold``                        +2.0    抬高且双指均接触时给 1
``floor_collision``             -12.0   机器人 geom 撞地 plane 给 1
``action_smoothness``           -0.01   ``Σ|a_t − a_{t-1}|``
``close_near_object``           +0.8    ``exp(-30·d) · (u/255)``，近物时鼓励闭合
``drop_penalty``                -5.0    object_drop 触发时 -5
============================== ======= =====================================
```

with:

```rst
============================== ======= =====================================
奖励项                          权重    作用
============================== ======= =====================================
``staged_pickup``               +3.0    乘法门控级联 ``reach·(1+close·(1+contact·(1+lift)))``
``hold``                        +2.0    抬高且双指均接触时给 1
``floor_collision``             -12.0   机器人 geom 撞地 plane 给 1
``action_smoothness``           -0.01   ``Σ|a_t − a_{t-1}|``
``drop_penalty``                -5.0    object_drop 触发时 -5
============================== ======= =====================================
```

Also update line 60 (the directory comment):

```rst
       ├── rewards.py       # reach3d / align / contact / coverage / lift / hold / penalties
```

to:

```rst
       ├── rewards.py       # staged_pickup cascade + hold / penalties
```

- [ ] **Step 4: Verify docs render without missing-reference warnings (optional but recommended)**

If the project has a Sphinx build target, run it. Otherwise just visually inspect for stale references:

```bash
grep -rE "reach3d|align_xy|close_near_object|tactile_contact_binary" docs/source/ || echo "no stale refs"
```

Expected: `no stale refs`.

- [ ] **Step 5: Commit**

```bash
git add docs/source/api/mdp_rewards.rst docs/source/design/reward_design.rst docs/source/design/task_architecture.rst
git commit -m "$(cat <<'EOF'
docs(reward): describe multiplicative-gating staged_pickup cascade

Updates the API reward summary, the design doc reward table and rationale,
and the task architecture reward table to reflect the new single-shape-term
composition replacing the additive bootstrap chain.
EOF
)"
```

---

## Acceptance verification (after all three tasks)

- [ ] **All tests pass**

```bash
uv run pytest tests/ -v
```

- [ ] **Env builds and shows the new reward set**

```bash
uv run python -c "from tactile_grasp import load_env_cfg; cfg = load_env_cfg(play=False); print(sorted(cfg.rewards.keys()))"
```

Expected: `['action_smoothness', 'drop_penalty', 'floor_collision', 'hold', 'staged_pickup']`

- [ ] **Short training smoke run shows non-zero `staged_pickup` channel from iter 0**

```bash
WANDB_MODE=offline PYTHONPATH= uv run python scripts/train.py Mjlab-TactileGrasp-Robotiq2F85 --agent.max-iterations 50 --agent.run-name multgate_smoke
```

Expected: runs to completion without runtime errors; TensorBoard `Episode_Reward/staged_pickup` channel is non-zero from iteration 0.
