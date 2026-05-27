"""tactile_grasp 任务共享常量."""

from __future__ import annotations

from mjlab.managers import SceneEntityCfg

TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85"

# 触觉成功判定阈值（单值）
TACTILE_ACTIVITY_THRESHOLD = 1.0e-3

TAXEL_INDEXES = tuple(f"{row}{col}" for row in range(3) for col in range(3))

LEFT_TAXEL_FORCE_SENSOR_NAMES = tuple(f"left_taxel_force_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_FORCE_SENSOR_NAMES = tuple(f"right_taxel_force_{index}" for index in TAXEL_INDEXES)

LEFT_TAXEL_BODY_NAMES = tuple(f"left_taxel_body_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_BODY_NAMES = tuple(f"right_taxel_body_{index}" for index in TAXEL_INDEXES)

LEFT_TAXEL_SITE_NAMES = tuple(f"left_taxel_site_{index}" for index in TAXEL_INDEXES)
RIGHT_TAXEL_SITE_NAMES = tuple(f"right_taxel_site_{index}" for index in TAXEL_INDEXES)

ROBOT_JOINT_NAMES = (
    "left_driver_joint",
    "left_spring_link_joint",
    "left_follower",
    "right_driver_joint",
    "right_spring_link_joint",
    "right_follower_joint",
)

ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES)

OBJECT_ENTITY_NAMES = ("cube_24mm", "box_tall", "cylinder_24mm")
OBJECT_HALF_HEIGHTS = (0.012, 0.024, 0.012)
OBJECT_CFGS = tuple(SceneEntityCfg(name) for name in OBJECT_ENTITY_NAMES)
OBJECT_CFG = OBJECT_CFGS[0]
