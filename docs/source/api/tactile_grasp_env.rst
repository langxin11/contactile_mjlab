Tactile Grasp Environment (Legacy)
=====================================

.. warning::

   本模块已被 ``contactile_mjlab.tasks.tactile_grasp`` 替代。
   主线任务通过 ``make_env()`` 和 ``TactileGraspTaskConfig`` 创建环境，
   不再使用 ``TactileGraspEnv`` 或 ``TactileGraspEnvConfig``。

   保留这个页面的原因只是：

   - 回溯重构前单文件实现
   - 为仍然引用 legacy 调试接口的代码提供文档
   - 对照新旧实现拆分后的职责变化

   新接口请优先参考 :doc:`tasks_env_cfg`、:doc:`tasks_tactile_terms`
   和 :doc:`tasks_reward_terms`。

.. automodule:: contactile_mjlab.mjlab.tactile_grasp_env
