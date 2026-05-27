"""tactile_grasp 奖励函数."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import observations as obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tactile_force_l2(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """惩罚双指尖触觉幅值平方和."""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.square(torch.cat([left, right], dim=-1)), dim=1)


def total_tactile_signal(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    entity_name: str = "robot",
) -> torch.Tensor:
    """双指尖触觉绝对值之和（被 terminations 复用）."""
    left = obs.sensor_values(env, left_sensor_names, entity_name=entity_name)
    right = obs.sensor_values(env, right_sensor_names, entity_name=entity_name)
    return torch.sum(torch.abs(torch.cat([left, right], dim=-1)), dim=1)


def alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """未终止的环境给 +1."""
    return (~env.termination_manager.terminated).float()


def action_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """惩罚动作幅度平方."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def reach3d(env: "ManagerBasedRlEnv", k_pos: float) -> torch.Tensor:
    """Reward tool-object proximity in 3D."""
    delta = obs.active_object_position(env) - obs.tool_position(env)
    return torch.exp(-k_pos * torch.linalg.norm(delta, dim=1))


def align_xy(env: "ManagerBasedRlEnv", k_xy: float) -> torch.Tensor:
    """Reward planar alignment between tool and active object."""
    delta_xy = obs.active_object_position(env)[:, :2] - obs.tool_position(env)[:, :2]
    return torch.exp(-k_xy * torch.linalg.norm(delta_xy, dim=1))


def tactile_contact_binary(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    threshold: float = 0.05,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return 1 when either fingertip has any active taxel."""
    left = obs.taxel_contact_count(
        env,
        left_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    )
    right = obs.taxel_contact_count(
        env,
        right_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    )
    return ((left + right) > 0).to(torch.float32)


def taxel_coverage(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    threshold: float = 0.05,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward balanced multi-taxel contact on both fingers."""
    left = obs.taxel_contact_count(
        env,
        left_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    ).to(torch.float32)
    right = obs.taxel_contact_count(
        env,
        right_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    ).to(torch.float32)
    return 0.5 * torch.clamp(left, max=9.0) / 9.0 + 0.5 * torch.clamp(right, max=9.0) / 9.0


def lift_delta(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Reward positive object lift relative to cached reset height."""
    init_z = getattr(env, "_tactile_active_object_init_z", None)
    if init_z is None:
        raise RuntimeError(
            "lift_delta() requires '_tactile_active_object_init_z' to be initialized. "
            "Call the reset event before requesting lift reward terms."
        )
    current_z = obs.active_object_position(env)[:, 2]
    return torch.relu(current_z - init_z)


def hold_bonus(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    threshold: float = 0.05,
    lift_threshold: float = 0.0,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return 1 when the object is lifted and both fingers maintain contact."""
    left = obs.taxel_contact_count(
        env,
        left_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    )
    right = obs.taxel_contact_count(
        env,
        right_sensor_names,
        entity_name=entity_name,
        threshold=threshold,
    )
    stable = (lift_delta(env) > lift_threshold) & (left > 0) & (right > 0)
    return stable.to(torch.float32)


def action_smoothness_l1(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize per-step action changes with a zero previous-action fallback."""
    prev_action = getattr(env.action_manager, "prev_action", None)
    if prev_action is None:
        prev_action = torch.zeros_like(env.action_manager.action)
    return torch.sum(torch.abs(env.action_manager.action - prev_action), dim=1)


def close_command_l2(
    env: "ManagerBasedRlEnv",
    action_name: str = "cartesian_gripper",
) -> torch.Tensor:
    """惩罚多余的夹爪闭合命令."""
    command = obs.gripper_command(env, action_name=action_name)
    return torch.sum(torch.square(command), dim=1)


def _ensure_collision_cache(env: "ManagerBasedRlEnv") -> None:
    """Populate cached robot and floor geom ids needed by contact-scan rewards."""
    robot_geom_ids = getattr(env, "_tactile_robot_geom_ids", None)
    if robot_geom_ids is None:
        env._tactile_robot_geom_ids = env.scene["robot"].indexing.geom_ids.to(torch.long)

    floor_geom_id = getattr(env, "_tactile_floor_geom_id", None)
    if floor_geom_id is None:
        try:
            env._tactile_floor_geom_id = int(env.scene["terrain"].indexing.geom_ids[0])
        except KeyError as exc:
            raise RuntimeError(
                "robot_floor_collision() requires terrain geom ids in env.scene."
            ) from exc


def robot_floor_collision(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return 1 for envs where any robot geom contacts the floor geom."""
    _ensure_collision_cache(env)

    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ncon = int(env.sim.data.nacon[0])
    if ncon <= 0:
        return out

    contact_geom = env.sim.data.contact.geom[:ncon]
    contact_world = env.sim.data.contact.worldid[:ncon].to(torch.long)
    robot_geom_ids = env._tactile_robot_geom_ids.to(contact_geom.device)
    floor_geom_id = int(env._tactile_floor_geom_id)
    geom0 = contact_geom[:, 0]
    geom1 = contact_geom[:, 1]
    hit = (torch.isin(geom0, robot_geom_ids) & (geom1 == floor_geom_id)) | (
        torch.isin(geom1, robot_geom_ids) & (geom0 == floor_geom_id)
    )
    if torch.any(hit):
        out[contact_world[hit].to(out.device)] = 1.0
    return out


def drop_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """物体掉落 termination 命中时的负奖励通道."""
    return env.termination_manager.get_term("object_drop").float()


def reach_xy(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Active object 与夹爪水平距离."""
    delta = obs.active_object_position(env)[:, :2] - obs.robot_position(env)[:, :2]
    return torch.linalg.norm(delta, dim=1)


def lift_height(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Active object root height."""
    return obs.active_object_position(env)[:, 2]


def tactile_contact(
    env: "ManagerBasedRlEnv",
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    threshold: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return 1 when tactile signal exceeds threshold."""
    signal = total_tactile_signal(
        env,
        left_sensor_names=left_sensor_names,
        right_sensor_names=right_sensor_names,
        entity_name=entity_name,
    )
    return (signal > threshold).float()
