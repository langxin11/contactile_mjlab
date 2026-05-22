"""tactile_grasp 任务共享常量."""

from __future__ import annotations

from mjlab.managers import SceneEntityCfg

TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres"

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
OBJECT_CFG = SceneEntityCfg("object")
