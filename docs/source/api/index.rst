API 参考
========

主线接口优先查看 ``contactile_mjlab.tasks.tactile_grasp`` 相关模块。
旧的 ``contactile_mjlab.mjlab.*`` 页面仅用于兼容、调试和回溯，不再是推荐入口。

Task 模块（活跃）
-----------------

.. toctree::
   :maxdepth: 2

   tasks_env_cfg
   tasks_tactile_terms
   tasks_reward_terms
   tasks_constants

Legacy 模块
-----------

以下模块为重构前的旧接口，保留以兼容调试脚本，但主线任务不再使用。

.. toctree::
   :maxdepth: 2

   tactile_grasp_env
   mdp
   action_terms
   actuators
   control
   paths
