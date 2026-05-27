"""Reward refactor unit tests for top-down pick-lift."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tactile_grasp.mdp import observations


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
