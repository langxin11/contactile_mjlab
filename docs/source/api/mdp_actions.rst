``tactile_grasp.mdp.actions``
=============================

``CartesianMocapAction`` —— 五维 ``[dx, dy, dz, dyaw, du]`` 动作累积到 robot
mocap pose 与 Robotiq ``u ∈ [0, 255]``，并写入 ``split`` tendon target。

``RobotiqCommandAction`` 仍保留为纯 gripper 命令动作实现，供回归和后续对照使用。

.. automodule:: tactile_grasp.mdp.actions
   :members:
   :undoc-members:
