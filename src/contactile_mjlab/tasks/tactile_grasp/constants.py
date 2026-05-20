"""Constants shared by tactile grasp task variants."""

from __future__ import annotations

from mjlab.managers import SceneEntityCfg

TOUCH_SITE_TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85-TouchSite"
PTS_SPHERES_TASK_ID = "Mjlab-TactileGrasp-Robotiq2F85-PTSSpheres"
DEFAULT_TASK_ID = PTS_SPHERES_TASK_ID

TACTILE_MODEL_TOUCH_SITE = "touch_site"
TACTILE_MODEL_PTS_SPHERES = "pts_spheres"

TACTILE_OBS_DIM_BY_TASK = {
    TOUCH_SITE_TASK_ID: 18,
    PTS_SPHERES_TASK_ID: 54,
}

TACTILE_ACTIVITY_THRESHOLD_BY_TASK = {
    TOUCH_SITE_TASK_ID: 0.0,
    PTS_SPHERES_TASK_ID: 1.0e-3,
}

TAXEL_INDEXES = tuple(f"{row}{col}" for row in range(3) for col in range(3))

LEFT_TOUCH_SENSOR_NAMES = tuple(f"left_touch_{index}" for index in TAXEL_INDEXES)
RIGHT_TOUCH_SENSOR_NAMES = tuple(f"right_touch_{index}" for index in TAXEL_INDEXES)

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
