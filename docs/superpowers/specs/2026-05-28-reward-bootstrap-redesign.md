# Reward Bootstrap Redesign — Top-Down Tactile Pick-Lift

**Date:** 2026-05-28
**Owner:** langxin11
**Status:** Spec — pending user review before plan/implementation
**Touches:** `src/tactile_grasp/mdp/rewards.py`, `src/tactile_grasp/env_cfgs.py`, `tests/test_reward_refactor.py`, `docs/source/design/reward_design.rst`, `docs/source/design/task_architecture.rst`

## Background — observed failure mode

After >100 PPO iterations on `Mjlab-TactileGrasp-Robotiq2F85` (`scripts/play.py` rollout, full
stage-2 randomization):

- The gripper learns to descend approximately above the active object.
- The gripper stays fully open (`u ≈ 0`) for the entire episode.
- TensorBoard reward channels show only `reach3d` accumulating meaningful values; `contact`,
  `coverage`, `lift_delta`, `hold` are effectively flat at zero.

## Diagnosis — bootstrap chain break

The current reward composition forms a cascade `reach → close → contact → grip → lift`, but
the middle link is structurally underspecified:

1. **`close_command` is pure penalty.** Its current form is `-0.05·(u/255)²`. Closing without
   touching anything is strictly worse than staying open, by up to `-0.05` per step (`-7.5`
   per 150-step episode at `u = 255`).
2. **Contact threshold is too coarse to bootstrap.** `TACTILE_CONTACT_THRESHOLD = 0.05 N`
   per-taxel: light brushes against the object register zero, so even random exploration that
   briefly touches the object collects no positive feedback.
3. **Downstream rewards are gated on actual grip.** `contact`, `coverage`, `lift_delta`, and
   `hold` only fire after the gripper succeeds in producing measurable force. From the
   policy's view, this is an unreachable cliff.

Net effect: the per-step optimum visible to the policy is "approach with `u = 0`" (+~0.9), and
trying to close is strictly worse short-term (-0.05) with no positive feedback until a remote
event (taxel force > 0.05 N) that the policy never explores far enough to discover.

## Goal

Fill the missing bootstrap link so the policy receives positive gradient for the action
"close gripper *while above the object*", and dense positive gradient as soon as taxels
register any contact. Both changes must preserve the meaning and bounds of existing terms.

## Non-goals

- Changing the curriculum thresholds or the stage-2 randomization range.
- Restructuring `lift_delta` / `hold` (they work correctly once contact is established).
- Adding any new observation channels or modifying actor/critic shapes.
- Tuning PPO hyperparameters.
- Adding a separate "light touch" smoothing term (rejected as Approach C — would change
  three things at once and complicate debugging if this redesign underperforms).

## Design

### Change 1 — replace `close_command` penalty with `close_near_object` reward

**Remove:**

- `tactile_grasp.mdp.rewards.close_command_l2`
- The `close_command` `RewardTermCfg` in `env_cfgs.make_tactile_grasp_env_cfg`
- The `W_CLOSE_COMMAND` constant in `env_cfgs.py`
- The `test_close_command_penalty_grows_with_command` unit test

**Add `close_near_object` to `tactile_grasp.mdp.rewards`:**

```python
def close_near_object(
    env: "ManagerBasedRlEnv",
    k_d: float,
    action_name: str = "cartesian_gripper",
) -> torch.Tensor:
    """Reward gripper command magnitude scaled by tool-object proximity."""
    delta = obs.active_object_position(env) - obs.tool_position(env)
    proximity = torch.exp(-k_d * torch.linalg.norm(delta, dim=1))
    command = obs.gripper_command(env, action_name=action_name).squeeze(-1)
    return proximity * command
```

**Wiring in `env_cfgs.py`:**

```python
CLOSE_NEAR_K_D = 30.0
W_CLOSE_NEAR = 0.8

"close_near_object": RewardTermCfg(
    func=rewards.close_near_object,
    weight=W_CLOSE_NEAR,
    params={"k_d": CLOSE_NEAR_K_D, "action_name": "cartesian_gripper"},
),
```

**Why this shape:**

- `proximity = exp(-k_d·d_3d)` with `k_d = 30.0` decays from `1.0` at the object to `~0.05`
  at 10 cm, so closing only matters once the gripper has actually descended near the object.
- `command` is `u/255 ∈ [0, 1]` already (verified by
  `test_gripper_command_uses_command_attribute_without_concrete_action_type`), so the
  product is in `[0, 1]` and the weighted term is bounded by `W_CLOSE_NEAR = 0.8`.
- Weight `0.8` is set comparable to `W_ALIGN`: large enough to outweigh the historical bias
  toward `u = 0` that the policy already learned, but smaller than `W_LIFT_DELTA` so it does
  not dominate the actual task objective.

**How `k_d = 30.0` was chosen (rationale, not measurement):**

*Geometric anchor — why ~3 cm is the meaningful scale.* Robotiq 2F-85 fully-open width is
85 mm, so each pad sits ~42 mm from the gripper center. The tabletop objects have a maximum
lateral half-extent of ~12 mm (cube 24 mm, cylinder 24 mm). When the tool center (= midpoint
of the two pad sites) is within ~3 cm laterally of the object center, the closing trajectory
still has a chance to sweep a pad across the object. Past ~5 cm, even a full close cannot
reach the object — closing in that region is wasted command. So 3 cm is roughly the boundary
of "closing is geometrically meaningful," and we want the reward to be active inside that
boundary, decaying fast outside it.

*Why `k_d` must be sharper than `reach3d` / `align`.* If `close_near_object` had the same
decay rate as `reach3d` (`k_pos = 10`), it would just duplicate the reach signal and add no
new bootstrap information. Forcing `k_d > k_pos, k_xy` makes the close-incentive fire only
*after* the gripper has actually descended near the object, which is the conjunction we want
the policy to discover.

| Distance | `reach3d` (k=10) | `align_xy` (k=20) | `close_near_object` (k=30) |
|----------|------------------|-------------------|-----------------------------|
| 1 cm     | 0.90             | 0.82              | 0.74                        |
| 3 cm     | 0.74             | 0.55              | **0.41**                    |
| 5 cm     | 0.61             | 0.37              | 0.22                        |
| 10 cm    | 0.37             | 0.14              | 0.05                        |

*Honest uncertainty.* The 3 cm anchor uses nominal 2F-85 dimensions, not measurements from
the MJCF (pad thickness, exact closing stroke). A precise measurement might justify `k_d`
anywhere in `[20, 40]`; `30` is a defensible starting point, not a swept optimum. Also note
that `d = 0` means tool-center is at object-center (i.e., centered above), *not* that pads
are touching — the term is a soft gravitational pull on alignment, not a contact gate.

*Tuning guidance after training.* If `close_near_object` curves stay near zero, the policy
never approaches close enough; lower `k_d` (e.g., 20) to extend the effective radius. If
`close_near_object` saturates quickly but `contact` / `lift_delta` do not follow, the term is
rewarding "loitering above the object with `u = 255`"; raise `k_d` (e.g., 40) to make it
more selective.

### Change 2 — lower `TACTILE_CONTACT_THRESHOLD`

```python
# env_cfgs.py
TACTILE_CONTACT_THRESHOLD = 0.005   # was 0.05
```

This constant propagates through `params` into `contact`, `coverage`, and `hold` (see
`make_tactile_grasp_env_cfg`). The dependent constants (`TACTILE_ACTIVITY_THRESHOLD` in
`constants.py`, which gates the `stable_grasp` termination signal) are unchanged.

**Why 0.005 (5 mN per-taxel 3D force norm):**

- The existing docstring for `TACTILE_ACTIVITY_THRESHOLD` already notes that 1 mN of total
  signal across all 18 taxels is enough to distinguish "no contact" from "static welded
  sensor noise". A 5 mN per-taxel threshold sits two orders of magnitude above that noise
  floor and three orders of magnitude below the original 50 mN gate.
- Lower would risk `coverage` saturating on simulator floating-point noise during
  no-contact phases. Higher leaves the cliff in place.

## Reward shape — per-step bounds after the change

(Approximate, assuming positions/forces near their respective regime midpoints.)

| Situation                                       | Before (per step) | After (per step) |
|-------------------------------------------------|-------------------|------------------|
| `u = 0`, tool aligned above object              | +0.9              | +0.9             |
| `u = 255`, aligned, taxels still below 50 mN    | +0.85             | **+1.7**         |
| `u = 255`, aligned, light brush registers       | +0.85             | **+2.4**         |
| `u = 255`, gripped, lifted ~5 cm and holding    | +3.0 – +3.5       | **+3.5 – +4.0**  |

The "close while above object" action goes from neutral-or-worse to clearly positive, and the
"first taxel reading" event jumps to immediately positive instead of waiting for the policy
to clear the 50 mN gate.

## Files touched

### `src/tactile_grasp/mdp/rewards.py`

- Remove `close_command_l2`.
- Add `close_near_object` (see Design § Change 1).

### `src/tactile_grasp/env_cfgs.py`

- Remove `W_CLOSE_COMMAND`; add `W_CLOSE_NEAR = 0.8`, `CLOSE_NEAR_K_D = 30.0`.
- Change `TACTILE_CONTACT_THRESHOLD = 0.05` → `0.005`.
- In `rewards={...}`: remove the `close_command` entry, add a `close_near_object` entry that
  passes `k_d=CLOSE_NEAR_K_D, action_name="cartesian_gripper"`.

### `tests/test_reward_refactor.py`

- Delete `test_close_command_penalty_grows_with_command`.
- Update `test_pick_lift_cfg_uses_new_reward_term_names`:
  - Replace `"close_command"` with `"close_near_object"` in the must-exist set.
  - Add `"close_command"` to the must-not-exist set.
- Add three new tests for `rewards.close_near_object`:
  1. `test_close_near_object_is_command_value_at_zero_distance` — `d=0`, `u=255` → `1.0`.
  2. `test_close_near_object_decays_with_distance` — `u=255`, `d=0.1 m` → near `0`.
  3. `test_close_near_object_is_zero_when_command_is_zero` — `d=0`, `u=0` → `0.0`.

All three use the same `_FakeEnv` + `observations.active_object_position`/`tool_position`
monkeypatch pattern already used by `test_reach3d_uses_tool_to_object_distance`.

### `docs/source/design/reward_design.rst`

- Update the reward table: replace the `close_command` row with `close_near_object` and
  update its description (positive proximity-gated closing incentive instead of penalty).
- Add a one-paragraph note explaining the bootstrap rationale ("close_near_object is the
  shaping term that lets the policy discover the contact regime").
- Update the threshold note: 50 mN → 5 mN.

### `docs/source/design/task_architecture.rst`

- Update the reward summary table identically (weights, names, description).

### `docs/source/usage.rst`

- No changes required (`grep` finds no references to `close_command` or to the contact
  threshold value as of 2026-05-28).

## Validation

### Unit / integration tests

- `uv run ruff check src tests` clean.
- `uv run pytest` — all existing tests still green, three new tests added.
- `uv run python scripts/smoke_env.py` — 40 step smoke run completes, no NaNs.

### Behavioral sanity (manual)

- `uv run python scripts/play.py Mjlab-TactileGrasp-Robotiq2F85-Play` with a pretrained
  open-gripper checkpoint should still show approach behavior (reach3d unchanged).
- After re-training to ≥50 iterations, `tensorboard` reward curves for `close_near_object`
  and `contact` should both rise from zero — that is the primary success signal.

### Definition of done

- All tests pass.
- Reward composition documented in `reward_design.rst` matches the new term set.
- Commit history shows the rewards refactor and the threshold tightening as one logically
  coherent change (single commit OK; the threshold change is small enough that splitting it
  out adds no debuggability).

## Risks and mitigations

- **`coverage` saturates on noise at threshold = 5 mN.** Mitigation: the existing 1 mN
  noise floor estimate gives 5× headroom; if observed during training, raise to 0.01 N.
- **`close_near_object` rewards "fly close + spam close" without contact.** Bounded at
  `W_CLOSE_NEAR = 0.8`, which is well below `lift_delta + hold` once the gripper actually
  grips (≥+2). Action smoothness still discourages oscillatory closing.
- **Removing `close_command` lets the policy close redundantly when far from object.**
  `close_near_object` gives 0 reward when far (no penalty), so the policy has no incentive
  to stay open either. In practice `action_smoothness` should suppress useless oscillation;
  if not, re-introduce a tiny `-0.005·(u/255)²` term outside the proximity gate.

## Out of scope (deferred)

- Adding a `light_touch = tanh(total_tactile_signal / k)` smoothing term (Approach C).
  Reserve for a follow-up if Approach B alone does not break through.
- Replacing `floor_collision` binary penalty with a soft barrier function.
- Curriculum tuning (e.g., longer stage 0 or 1).
