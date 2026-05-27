"""Reward refactor unit tests for top-down pick-lift."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tactile_grasp.mdp import events, observations, rewards


class _FakeActionManager:
    def __init__(
        self, action: torch.Tensor, prev_action: torch.Tensor, command: torch.Tensor
    ) -> None:
        self.action = action
        self.prev_action = prev_action
        self._term = SimpleNamespace(command=command)

    def get_term(self, name: str) -> SimpleNamespace:
        assert name == "cartesian_gripper"
        return self._term


class _FakeSensor:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class _FakeScene(dict):
    def __init__(self, env_origins: torch.Tensor | None = None) -> None:
        super().__init__()
        if env_origins is None:
            env_origins = torch.zeros((1, 3), dtype=torch.float32)
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.device = "cpu"
        self.num_envs = num_envs
        self.scene = _FakeScene(env_origins=torch.zeros((num_envs, 3), dtype=torch.float32))
        self.action_manager = _FakeActionManager(
            action=torch.zeros((num_envs, 5), dtype=torch.float32),
            prev_action=torch.zeros((num_envs, 5), dtype=torch.float32),
            command=torch.zeros((num_envs, 1), dtype=torch.float32),
        )
        self.termination_manager = SimpleNamespace(
            get_term=lambda name: torch.zeros(num_envs, dtype=torch.bool)
        )


@pytest.mark.parametrize(
    "helper",
    (
        observations.active_object_position,
        observations.active_object_yaw,
        observations.vision_proxy,
    ),
)
def test_active_object_observation_helpers_require_initialized_ids(helper) -> None:
    """Object observation helpers should fail clearly before reset initializes ids."""
    env = _FakeEnv(num_envs=1)
    env.scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.zeros((1, 3), dtype=torch.float32))
    )

    with pytest.raises(RuntimeError, match="_tactile_active_object_ids"):
        helper(env)


def test_gripper_command_uses_command_attribute_without_concrete_action_type() -> None:
    """gripper_command should only require a command tensor on the action term."""
    env = _FakeEnv(num_envs=2)
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(
            command=torch.tensor([[0.0], [255.0]], dtype=torch.float32)
        )
    )

    out = observations.gripper_command(env)

    assert torch.allclose(out, torch.tensor([[0.0], [1.0]], dtype=torch.float32))


def test_gripper_command_raises_when_action_term_has_no_command() -> None:
    """gripper_command should raise clearly when the action term lacks command."""
    env = _FakeEnv(num_envs=1)
    env.action_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace())

    with pytest.raises(TypeError, match="command"):
        observations.gripper_command(env)


def test_taxel_force_observations_clip_to_sensor_limits() -> None:
    """Taxel observation helpers must clamp tangential and normal force components."""
    env = _FakeEnv(num_envs=1)
    env.scene["robot/left_taxel_force_00"] = _FakeSensor(
        torch.tensor([[9.0, -8.5, 22.0]], dtype=torch.float32)
    )

    tangential = observations.taxel_tangential_force(
        env, sensor_names=("left_taxel_force_00",), entity_name="robot"
    )
    normal = observations.taxel_normal_force(
        env, sensor_names=("left_taxel_force_00",), entity_name="robot"
    )

    assert torch.allclose(tangential, torch.tensor([[4.0, -4.0]], dtype=torch.float32))
    assert torch.allclose(normal, torch.tensor([[15.0]], dtype=torch.float32))


def test_tool_position_uses_pad_midpoint_instead_of_root_position() -> None:
    """Tool position should come from pad sites rather than the robot root body."""
    env = _FakeEnv(num_envs=1)
    env.scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32)),
        indexing=SimpleNamespace(site_ids=torch.tensor([5, 6], dtype=torch.long)),
        find_sites=lambda names: ([0, 1], ["left_pad_ft_site", "right_pad_ft_site"]),
    )
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            site_xpos=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.1, -0.02, 0.08],
                        [0.1, 0.02, 0.08],
                    ]
                ],
                dtype=torch.float32,
            )
        )
    )

    out = observations.tool_position(env)

    assert torch.allclose(out, torch.tensor([[0.1, 0.0, 0.08]], dtype=torch.float32))


def test_tool_position_falls_back_to_pose_command_when_pad_sites_unavailable() -> None:
    """Tool position should use pose_command_local plus env origins when sites are unavailable."""
    env = _FakeEnv(num_envs=2)
    env.scene.env_origins = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, 0.5]], dtype=torch.float32)
    env.scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[8.0, 8.0, 8.0], [7.0, 7.0, 7.0]], dtype=torch.float32)
        ),
        find_sites=lambda names: ([], []),
    )
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(
            pose_command_local=torch.tensor(
                [[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]], dtype=torch.float32
            )
        )
    )

    out = observations.tool_position(env)

    assert torch.allclose(
        out,
        torch.tensor([[1.1, 1.8, 3.3], [-0.6, -1.5, 0.4]], dtype=torch.float32),
    )


def test_taxel_contact_mask_uses_per_taxel_force_norm_threshold() -> None:
    """Contact mask should threshold each taxel on its 3D force norm."""
    env = _FakeEnv(num_envs=2)
    env.scene.update(
        {
            "robot/left_taxel_force_00": _FakeSensor(
                torch.tensor([[0.03, 0.04, 0.0], [0.04, 0.0, 0.0]])
            ),
            "robot/left_taxel_force_01": _FakeSensor(
                torch.tensor([[0.0, 0.0, 0.051], [0.02, 0.02, 0.02]])
            ),
        }
    )

    out = observations.taxel_contact_mask(
        env,
        sensor_names=("left_taxel_force_00", "left_taxel_force_01"),
        threshold=0.05,
    )

    assert torch.equal(out, torch.tensor([[False, True], [False, False]]))


def test_taxel_contact_count_sums_active_taxels_per_env() -> None:
    """Contact count should sum active taxels along the taxel axis for each env."""
    env = _FakeEnv(num_envs=2)
    env.scene.update(
        {
            "robot/left_taxel_force_00": _FakeSensor(
                torch.tensor([[0.1, 0.0, 0.0], [0.01, 0.01, 0.01]])
            ),
            "robot/left_taxel_force_01": _FakeSensor(
                torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.06]])
            ),
            "robot/left_taxel_force_02": _FakeSensor(
                torch.tensor([[0.03, 0.04, 0.0], [0.1, 0.0, 0.0]])
            ),
        }
    )

    out = observations.taxel_contact_count(
        env,
        sensor_names=("left_taxel_force_00", "left_taxel_force_01", "left_taxel_force_02"),
        threshold=0.05,
    )

    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([1, 2], dtype=torch.int64))


def test_lift_delta_is_zero_at_table_height_and_positive_when_raised() -> None:
    """lift_delta should measure positive height gain relative to cached reset height."""
    env = _FakeEnv(num_envs=2)
    env._tactile_active_object_ids = torch.zeros(2, dtype=torch.long)
    env._tactile_active_object_init_z = torch.tensor([0.012, 0.012], dtype=torch.float32)
    env.scene["cube_24mm"] = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor(
                [[0.0, 0.0, 0.012], [0.0, 0.0, 0.03]],
                dtype=torch.float32,
            )
        )
    )

    out = rewards.lift_delta(env)

    assert torch.allclose(out, torch.tensor([0.0, 0.018], dtype=torch.float32))


def test_lift_delta_requires_reset_height_cache() -> None:
    """lift_delta should fail fast when reset-time initial height was never cached."""
    env = _FakeEnv(num_envs=1)
    env._tactile_active_object_ids = torch.zeros(1, dtype=torch.long)
    env.scene["cube_24mm"] = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.02]], dtype=torch.float32))
    )

    with pytest.raises(RuntimeError, match="_tactile_active_object_init_z"):
        rewards.lift_delta(env)


def test_pick_lift_curriculum_rejects_invalid_forced_stage() -> None:
    """Forced curriculum stage should be validated instead of silently falling through."""
    env = SimpleNamespace(common_step_counter=0)

    with pytest.raises(ValueError, match="force_stage"):
        events.pick_lift_curriculum(env, env_ids=None, force_stage=3)


def test_action_smoothness_l1_uses_current_minus_previous_action() -> None:
    """Smoothness should sum per-dimension absolute action deltas."""
    env = _FakeEnv(num_envs=2)
    env.action_manager.action = torch.tensor(
        [[0.0, 0.5, -0.5, 0.25, -0.25], [1.0, -1.0, 0.0, 0.25, 0.5]],
        dtype=torch.float32,
    )
    env.action_manager.prev_action = torch.tensor(
        [[0.0, 0.25, -0.25, 0.0, -0.5], [0.5, -0.5, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    out = rewards.action_smoothness_l1(env)

    assert torch.allclose(out, torch.tensor([1.0, 1.75], dtype=torch.float32))


def test_close_command_penalty_grows_with_command() -> None:
    """close_command_l2 should increase with larger normalized close command."""
    env = _FakeEnv(num_envs=2)
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(
            command=torch.tensor([[0.0], [255.0]], dtype=torch.float32)
        )
    )

    out = rewards.close_command_l2(env)

    assert torch.allclose(out, torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_reach3d_uses_tool_to_object_distance() -> None:
    """reach3d should decay with tool-object distance and equal 1 at zero distance."""
    env = _FakeEnv(num_envs=1)
    original_obj = observations.active_object_position
    original_tool = observations.tool_position
    try:
        observations.active_object_position = lambda _env: torch.tensor(
            [[0.0, 0.0, 0.02]], dtype=torch.float32
        )
        observations.tool_position = lambda _env: torch.tensor(
            [[0.0, 0.0, 0.02]], dtype=torch.float32
        )
        near = rewards.reach3d(env, k_pos=10.0)
        observations.tool_position = lambda _env: torch.tensor(
            [[0.1, 0.0, 0.02]], dtype=torch.float32
        )
        far = rewards.reach3d(env, k_pos=10.0)
    finally:
        observations.active_object_position = original_obj
        observations.tool_position = original_tool

    assert near.item() > far.item()
    assert torch.allclose(near, torch.tensor([1.0], dtype=torch.float32))


def test_align_xy_ignores_z_offset() -> None:
    """align_xy should only care about planar mismatch."""
    env = _FakeEnv(num_envs=1)
    original_obj = observations.active_object_position
    original_tool = observations.tool_position
    try:
        observations.active_object_position = lambda _env: torch.tensor(
            [[0.01, -0.01, 0.01]], dtype=torch.float32
        )
        observations.tool_position = lambda _env: torch.tensor(
            [[0.01, -0.01, 0.20]], dtype=torch.float32
        )
        perfect_xy = rewards.align_xy(env, k_xy=20.0)
        observations.tool_position = lambda _env: torch.tensor(
            [[0.04, -0.01, 0.20]], dtype=torch.float32
        )
        shifted_xy = rewards.align_xy(env, k_xy=20.0)
    finally:
        observations.active_object_position = original_obj
        observations.tool_position = original_tool

    assert torch.allclose(perfect_xy, torch.tensor([1.0], dtype=torch.float32))
    assert perfect_xy.item() > shifted_xy.item()


def test_tactile_contact_binary_returns_one_when_any_taxel_is_active() -> None:
    """Binary contact should activate when either finger has any active taxel."""
    env = _FakeEnv(num_envs=2)
    env.scene.update(
        {
            "robot/left_taxel_force_00": _FakeSensor(
                torch.tensor([[0.03, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
            ),
            "robot/right_taxel_force_00": _FakeSensor(
                torch.tensor([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=torch.float32)
            ),
        }
    )

    out = rewards.tactile_contact_binary(
        env,
        left_sensor_names=("left_taxel_force_00",),
        right_sensor_names=("right_taxel_force_00",),
        threshold=0.05,
    )

    assert torch.equal(out, torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_taxel_coverage_rewards_balanced_multi_taxel_contact() -> None:
    """Coverage should average capped left/right active-taxel fractions."""
    env = _FakeEnv(num_envs=2)
    left_counts = torch.tensor([9, 3], dtype=torch.int64)
    right_counts = torch.tensor([4, 12], dtype=torch.int64)
    original_count = observations.taxel_contact_count
    try:
        observations.taxel_contact_count = (
            lambda _env, sensor_names, threshold=0.05, entity_name="robot": left_counts
            if sensor_names[0].startswith("left_")
            else right_counts
        )

        out = rewards.taxel_coverage(
            env,
            left_sensor_names=("left_taxel_force_00",),
            right_sensor_names=("right_taxel_force_00",),
            threshold=0.05,
        )
    finally:
        observations.taxel_contact_count = original_count

    assert torch.allclose(
        out,
        torch.tensor([13.0 / 18.0, 2.0 / 3.0], dtype=torch.float32),
    )


def test_hold_bonus_requires_lift_and_bilateral_contact() -> None:
    """hold_bonus should require lift above threshold plus contact on both fingers."""
    env = _FakeEnv(num_envs=3)
    original_lift = rewards.lift_delta
    original_count = observations.taxel_contact_count
    try:
        rewards.lift_delta = lambda _env: torch.tensor([0.05, 0.05, 0.01], dtype=torch.float32)
        counts = {
            "left": torch.tensor([1, 0, 1], dtype=torch.int64),
            "right": torch.tensor([1, 1, 1], dtype=torch.int64),
        }
        observations.taxel_contact_count = (
            lambda _env, sensor_names, threshold=0.05, entity_name="robot": counts["left"]
            if sensor_names[0].startswith("left_")
            else counts["right"]
        )

        out = rewards.hold_bonus(
            env,
            left_sensor_names=("left_taxel_force_00",),
            right_sensor_names=("right_taxel_force_00",),
            threshold=0.05,
            lift_threshold=0.03,
        )
    finally:
        rewards.lift_delta = original_lift
        observations.taxel_contact_count = original_count

    assert torch.equal(out, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))


def test_robot_floor_collision_detects_robot_geom_pair_only() -> None:
    """robot_floor_collision should only count robot geom contacts against the floor."""
    env = _FakeEnv(num_envs=3)
    env._tactile_robot_geom_ids = torch.tensor([7, 8], dtype=torch.long)
    env._tactile_floor_geom_id = 99
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            nacon=torch.tensor([5], dtype=torch.int32),
            contact=SimpleNamespace(
                geom=torch.tensor(
                    [
                        [50, 99],
                        [7, 99],
                        [99, 8],
                        [7, 123],
                        [50, 51],
                    ],
                    dtype=torch.int32,
                ),
                worldid=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32),
            ),
        )
    )

    out = rewards.robot_floor_collision(env)

    assert torch.equal(out, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32))


def test_robot_floor_collision_initializes_cached_geom_ids_from_scene_when_missing() -> None:
    """robot_floor_collision should lazily cache robot and floor geom ids from the scene."""
    env = _FakeEnv(num_envs=1)
    env.scene["robot"] = SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([7, 8], dtype=torch.long))
    )
    env.scene["terrain"] = SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([99], dtype=torch.long))
    )
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            nacon=torch.tensor([1], dtype=torch.int32),
            contact=SimpleNamespace(
                geom=torch.tensor([[8, 99]], dtype=torch.int32),
                worldid=torch.tensor([0], dtype=torch.int32),
            ),
        )
    )

    out = rewards.robot_floor_collision(env)

    assert torch.equal(out, torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(env._tactile_robot_geom_ids, torch.tensor([7, 8], dtype=torch.long))
    assert int(env._tactile_floor_geom_id) == 99
