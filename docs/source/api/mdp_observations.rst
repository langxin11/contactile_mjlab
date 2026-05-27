``tactile_grasp.mdp.observations``
==================================

触觉观测读取与拆分：

- ``taxel_normal_force`` —— site-local ``z`` 分量，``[B, 9]``
- ``taxel_tangential_force`` —— site-local ``xy`` 分量 flatten，``[B, 18]``
- ``pad_force`` / ``pad_torque`` —— 单 pad 全局 wrench
- ``gripper_command`` —— Robotiq command ``u / 255``
- ``vision_proxy`` —— active object 相对夹爪 pose + object type one-hot
- ``sensor_values`` —— 通用 builtin sensor stacker（被 reward / termination 复用）

.. automodule:: tactile_grasp.mdp.observations
   :members:
   :undoc-members:
