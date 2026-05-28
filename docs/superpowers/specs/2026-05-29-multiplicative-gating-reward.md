# Multiplicative-Gating Reward Cascade — Top-Down Tactile Pick-Lift

**Date:** 2026-05-29
**Owner:** langxin11
**Status:** Spec — pending user review before plan/implementation
**Touches:** `src/tactile_grasp/mdp/rewards.py`, `src/tactile_grasp/env_cfgs.py`, `tests/test_reward_refactor.py`, `docs/source/api/mdp_rewards.rst`, `docs/source/design/reward_design.rst`, `docs/source/design/task_architecture.rst`

## Background — observed failure mode

After the additive-bootstrap redesign (`2026-05-28-reward-bootstrap-redesign.md`) and tuning
to `W_CLOSE_NEAR=2.5`, `entropy_coef=0.025`, training reached this steady state by iter ~150:

- `close_near_object` accumulating reward (final ~0.37 per episode).
- `Train/mean_reward` 3.27 → 4.15.
- `Policy/mean_std` stable at 1.06 (no collapse).
- `contact`, `coverage`, `lift_delta`, `hold` channels still effectively zero.

Visual rollouts confirm the policy learns to **hover ~2.6 cm above the object with the
gripper ~1/3 closed** — it harvests `close_near_object` (additive ∈ [0, 1]) and `reach3d`
without ever descending into contact. Descending costs nothing in the additive scheme but
also yields no extra gradient until contact actually fires, so the policy is stable at this
local optimum.

## Diagnosis — additive composition is gameable

The current reward dict adds bootstrap terms independently:

    total_shape = reach3d + align + close_near + contact + coverage + lift_delta + hold

Each term peaks at ≤1 per step. The policy can collect ~80% of the achievable per-step shape
reward simply by satisfying the easy outer terms (reach, align, close_near) and ignore the
strictly harder inner terms (contact, coverage, lift_delta, hold). There is no architectural
penalty for skipping phases.

Upstream `mjlab.tasks.manipulation.mdp.rewards.staged_position_reward` solves this by
multiplicative gating: `reaching * (1 + bringing)`. Bringing only contributes signal when
reaching is high, forcing phase progression. We generalize this pattern to the four-stage
pick-lift chain.

## Goal

Replace the additive bootstrap composition with a single multiplicatively-gated reward term
`staged_pickup` that progressively unlocks downstream phases (close → contact → lift) as
upstream phases (reach → close → contact) are satisfied. The policy can no longer farm
outer-stage reward without making progress on inner stages.

## Non-goals

- Changing the curriculum stages or randomization ranges.
- Modifying `hold`, `floor_collision`, `drop_penalty`, or `action_smoothness` semantics.
- Restructuring observation channels or actor/critic shapes.
- Re-tuning PPO hyperparameters before the redesign is empirically validated.
- Adding any new low-level primitives (action, sensor, observation).

## Design

### The cascade

Define a new reward function `staged_pickup` that returns a single scalar per env:

```python
def staged_pickup(
    env: ManagerBasedRlEnv,
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

    reach × (1 + close × (1 + contact × (1 + lift)))

    Each factor is in [0, 1]; the cascade output is in [0, 4].
    """
```

### Inner factors

All four factors are bounded to `[0, 1]` so the cascade output is bounded to `[0, 4]`.

**1. Reach factor (anisotropic distance)**

The xy direction is weighted twice as heavily as the z direction, absorbing the previous
`align_xy` strictness into the cascade's outermost gate:

    d_aniso = sqrt(2 · (Δx² + Δy²) + Δz²)
    reach   = exp(-k_pos · d_aniso)             ∈ (0, 1]

Default `k_pos = 10.0` (matches current `REACH_K_POS`).

**2. Close factor (command × proximity)**

Reuses the anisotropic distance so the cascade has a single, consistent distance semantic:

    close = gripper_command · exp(-k_d · d_aniso)   ∈ [0, 1]

`gripper_command` is the normalized `u/u_max ∈ [0, 1]` already exposed by
`mdp.observations.gripper_command`. Default `k_d = 30.0` (matches current
`CLOSE_NEAR_K_D`).

**3. Contact factor (smooth coverage)**

Reuses `taxel_coverage(threshold)`, which returns the average fraction of active taxels
across both fingers:

    contact = taxel_coverage(left, right, threshold)   ∈ [0, 1]

Using smooth coverage instead of the binary `tactile_contact_binary` provides a non-zero
gradient as soon as a single taxel activates, avoiding a step discontinuity that would
re-introduce the same bootstrap problem at a deeper level.

**4. Lift factor (capped progress)**

Reuses `lift_delta()` and saturates at the success-height threshold:

    lift = clamp(lift_delta / lift_cap, 0, 1)          ∈ [0, 1]

Default `lift_cap = 0.08` (= `SUCCESS_HEIGHT`). 0–8 cm of lift maps linearly to 0–1; further
lifting saturates the cascade contribution. The independent `hold` reward provides the
additional signal beyond 3 cm lift.

### The cascade composition

    staged_pickup = reach · (1 + close · (1 + contact · (1 + lift)))

Reward magnitude across phases (final reward = `W_STAGED_PICKUP · staged_pickup`, with
`W_STAGED_PICKUP = 3.0`):

| Phase                              | reach | close | contact | lift | cascade | reward |
|------------------------------------|-------|-------|---------|------|---------|--------|
| Initial (far)                      | 0.30  | 0     | 0       | 0    | 0.30    | 0.90   |
| Aligned + half-closed, hovering    | 0.70  | 0.14  | 0       | 0    | 0.80    | 2.40   |
| First contact (4/9 taxels active)  | 0.95  | 0.70  | 0.44    | 0    | 1.91    | 5.72   |
| Lifted to 4 cm                     | 0.95  | 0.90  | 0.90    | 0.50 | 2.96    | 8.88   |
| Saturated (lifted to ≥ 8 cm)       | 1.00  | 1.00  | 1.00    | 1.00 | 4.00    | 12.00  |

The "hover and farm" baseline (2.40) is now strictly dominated by "descend to contact" (5.72)
— ~2.4× difference per step — so the policy has a strong gradient to push through the
contact phase.

### Reward dict — before vs. after

| Term                       | Current weight | New weight | Action                                           |
|----------------------------|---------------:|-----------:|--------------------------------------------------|
| `reach3d`                  |           0.6  |          — | Remove (absorbed into cascade `reach` factor)    |
| `align`                    |           0.8  |          — | Remove (absorbed into anisotropic distance)      |
| `close_near_object`        |           2.5  |          — | Remove (absorbed into cascade `close` factor)    |
| `contact`                  |           0.2  |          — | Remove (absorbed into cascade `contact` factor)  |
| `coverage`                 |           1.2  |          — | Remove (function reused inside cascade)          |
| `lift_delta`               |           8.0  |          — | Remove (absorbed into cascade `lift` factor)     |
| `staged_pickup` (new)      |             —  |        3.0 | Add as the single shaped bootstrap term          |
| `hold`                     |           2.0  |        2.0 | Keep (additive, independent)                     |
| `floor_collision`          |         −12.0  |      −12.0 | Keep (additive, independent)                     |
| `drop_penalty`             |          −5.0  |       −5.0 | Keep (additive, independent)                     |
| `action_smoothness`        |          −0.01 |      −0.01 | Keep (additive, independent)                     |

### Module-level constants in `env_cfgs.py`

```python
# Removed
REACH_K_POS = 10.0
ALIGN_K_XY = 20.0
CLOSE_NEAR_K_D = 30.0
W_REACH = 0.6
W_ALIGN = 0.8
W_CONTACT = 0.2
W_COVERAGE = 1.2
W_LIFT_DELTA = 8.0
W_CLOSE_NEAR = 2.5

# Added
STAGED_PICKUP_K_POS = 10.0
STAGED_PICKUP_K_D = 30.0
STAGED_PICKUP_LIFT_CAP = 0.08
W_STAGED_PICKUP = 3.0

# Unchanged
TACTILE_CONTACT_THRESHOLD = 0.005
HOLD_LIFT_THRESHOLD = 0.03
W_HOLD = 2.0
W_FLOOR = -12.0
W_DROP_PENALTY = -5.0
W_ACTION_SMOOTHNESS = -0.01
```

### Function signature contract

`staged_pickup` must:

- Return a `torch.Tensor` of shape `(num_envs,)` and dtype float32.
- Take `k_pos`, `k_d`, `lift_cap` as explicit float params (so `RewardTermCfg.params`
  controls them, mirroring existing reward signatures).
- Reuse `mdp.observations.active_object_position`, `mdp.observations.tool_position`,
  `mdp.observations.gripper_command`, `taxel_coverage`, and `lift_delta` rather than
  re-deriving them.
- Be numerically safe at episode reset (when `_tactile_active_object_init_z` may have
  just been populated): rely on `lift_delta`'s existing guard.

## Tests

Add to `tests/test_reward_refactor.py` (and rename the file if its scope no longer matches —
that decision belongs in the implementation plan):

1. `test_staged_pickup_returns_zero_at_full_separation` — when tool is far from object and
   gripper open, cascade should be near zero.
2. `test_staged_pickup_monotonic_in_each_factor` — with the other three factors held at a
   non-zero value, increasing any one factor strictly increases the cascade output.
3. `test_staged_pickup_saturates_at_four` — when all factors = 1, output = 4 (within fp
   tolerance).
4. `test_staged_pickup_anisotropic_distance` — a 1 cm xy offset reduces `reach` more than a
   1 cm z offset (because xy has 2× weight in `d_aniso`).
5. `test_staged_pickup_lift_saturates_at_cap` — lifting beyond `lift_cap` does not increase
   the cascade further.
6. `test_pick_lift_cfg_uses_staged_pickup` — `make_tactile_grasp_env_cfg().rewards`
   contains `staged_pickup` and does NOT contain `reach3d`, `align`, `close_near_object`,
   `contact`, `coverage`, `lift_delta`.

Existing assertions in `test_pick_lift_cfg_uses_new_reward_term_names` will need updating
to match the new reward names — that update is part of this redesign.

## Documentation

- `docs/source/api/mdp_rewards.rst` — add `staged_pickup` entry; remove the six absorbed
  entries.
- `docs/source/design/reward_design.rst` — explain the multiplicative gating principle,
  show the cascade formula, document the anisotropic distance and the per-phase reward
  table from the Design section.
- `docs/source/design/task_architecture.rst` — update the "rewards" subsection to reflect
  the new single-shape-term composition.

## Acceptance criteria

1. `tests/test_reward_refactor.py` (or its renamed successor) passes, including the six
   new tests above.
2. `make_tactile_grasp_env_cfg()` builds without error.
3. `scripts/train.py Mjlab-TactileGrasp-Robotiq2F85` runs without runtime errors for at
   least 50 iterations.
4. TensorBoard channel `Episode_Reward/staged_pickup` is non-zero from iter 0 and grows.
5. Documentation builds without warnings about missing reward terms.

## Risks and mitigations

- **Cascade is too sparse at iter 0.** If `reach=0.3` (initial far position), the policy
  sees `~0.9` reward per step versus `~3.0` from the additive baseline at the same state.
  Mitigation: `W_STAGED_PICKUP=3.0` is a single tunable; if iter-0 signal is too weak we
  bump it. The empirical test is the 50-iter sanity run.

- **Lift cap suppresses the "go higher" gradient between 3-8 cm.** Inside the cascade,
  `lift` saturates only at 8 cm, so gradient is preserved across the full lift range. Beyond
  8 cm the independent `hold` term takes over.

- **Removing `align` weakens the xy strictness compared to the current additive baseline.**
  At a pure-xy offset δ (with z = 0), `d_aniso = √2 · δ`, so `reach = exp(-10 · √2 · δ) ≈
  exp(-14.1 · δ)`. The previous design had `reach3d = exp(-10 · δ)` plus
  `align = exp(-20 · δ)`, with a combined effective slope near 30 for small δ on a
  per-weight basis. The new design's xy gate is qualitatively stricter than z (the 2× xy
  weight is what we want) but quantitatively softer than the previous reach+align combo.
  Mitigation: if empirical xy alignment regresses, the first knob is increasing the xy
  weight in `d_aniso` from 2 to a larger value (e.g. 4 gives a pure-xy slope of 20). This
  is a single-line change in `staged_pickup`.

- **Coverage no longer rewards two-finger balance directly.** Inside the cascade, contact
  is the average across both fingers, so unbalanced contact reduces the gate. The
  per-finger balance gradient comes from `hold` (which requires BOTH fingers active).
