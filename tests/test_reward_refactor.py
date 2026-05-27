"""Reward refactor unit tests for top-down pick-lift."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tactile_grasp.mdp import observations, rewards


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
        self._tactile_active_object_init_z = torch.full((num_envs,), 0.012, dtype=torch.float32)


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


def test_lift_delta_is_zero_at_table_height_and_positive_when_raised() -> None:
    """Relative lift reward should ignore the object's resting table height."""
    env = _FakeEnv(num_envs=2)
    original = observations.active_object_position
    try:
        observations.active_object_position = lambda _env: torch.tensor(
            [[0.0, 0.0, 0.012], [0.0, 0.0, 0.040]], dtype=torch.float32
        )
        out = rewards.lift_delta(env)
    finally:
        observations.active_object_position = original

    assert torch.allclose(out, torch.tensor([0.0, 0.028], dtype=torch.float32))


def test_action_smoothness_l1_uses_current_minus_previous_action() -> None:
    """Smoothness penalty should be zero for unchanged actions and grow with deltas."""
    env = _FakeEnv(num_envs=2)
    env.action_manager.action = torch.tensor(
        [[0.2, -0.2, 0.0, 0.1, 0.0], [0.5, 0.0, -0.5, 0.0, 1.0]], dtype=torch.float32
    )
    env.action_manager.prev_action = torch.tensor(
        [[0.2, -0.2, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )

    out = rewards.action_smoothness_l1(env)

    assert torch.allclose(out, torch.tensor([0.0, 2.0], dtype=torch.float32))


def test_close_command_penalty_grows_with_command() -> None:
    """Close penalty should increase monotonically with normalized gripper command."""
    env = _FakeEnv(num_envs=3)
    env.action_manager = _FakeActionManager(
        action=torch.zeros((3, 5), dtype=torch.float32),
        prev_action=torch.zeros((3, 5), dtype=torch.float32),
        command=torch.tensor([[0.0], [127.5], [255.0]], dtype=torch.float32),
    )

    out = rewards.close_command_l2(env, action_name="cartesian_gripper")

    assert torch.allclose(out, torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32))


def test_taxel_coverage_rewards_balanced_multi_taxel_contact() -> None:
    """Coverage reward should prefer bilateral coverage over a single active taxel."""
    env = _FakeEnv(num_envs=2)
    env.scene.update(
        {
            "robot/left_taxel_force_00": _FakeSensor(
                torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
            ),
            "robot/left_taxel_force_01": _FakeSensor(
                torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
            ),
            "robot/right_taxel_force_00": _FakeSensor(
                torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
            ),
            "robot/right_taxel_force_01": _FakeSensor(
                torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
            ),
        }
    )
    left = ("left_taxel_force_00", "left_taxel_force_01")
    right = ("right_taxel_force_00", "right_taxel_force_01")

    out = rewards.taxel_coverage(
        env,
        left_sensor_names=left,
        right_sensor_names=right,
        threshold=0.25,
    )

    assert out[0] > out[1]


def test_robot_floor_collision_detects_robot_geom_pair_only() -> None:
    """Robot-floor contact should trigger regardless of object-floor contacts."""
    env = _FakeEnv(num_envs=2)
    env._tactile_robot_geom_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    env._tactile_floor_geom_id = 99
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            nacon=torch.tensor([3], dtype=torch.int32),
            contact=SimpleNamespace(
                geom=torch.tensor([[10, 99], [40, 99], [11, 98]], dtype=torch.int32),
                worldid=torch.tensor([0, 1, 1], dtype=torch.int32),
            ),
        )
    )

    out = rewards.robot_floor_collision(env)

    assert torch.equal(out, torch.tensor([1.0, 0.0], dtype=torch.float32))


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


def test_pick_lift_cfg_scales_tactile_observations_by_sensor_range() -> None:
    """Actor observation scales should match the configured tactile force limits."""
    from tactile_grasp import load_env_cfg

    cfg = load_env_cfg(play=False)

    assert cfg.observations["actor"].terms["left_taxel_normal"].scale == 1.0 / 15.0
    assert cfg.observations["actor"].terms["left_taxel_tangential"].scale == 1.0 / 4.0
