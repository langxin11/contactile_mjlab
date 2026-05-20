设计文档
========

本节面向维护者，解释当前 task-based 实现如何组织、运行，以及哪些设计点
是刻意保持简化的。

推荐阅读顺序：

1. :doc:`task_architecture` — 先建立整体心智模型
2. :doc:`tactile_pipeline` — 再看触觉信号从 MuJoCo 到策略观测的路径
3. :doc:`reward_design` — 然后看 reward / termination 的当前定义
4. :doc:`control_pipeline` — 最后看动作语义与 PPO 训练配置
5. :doc:`pts_taxel_scheme` — 需要查 MJCF 细节时再看 taxel 建模页

.. toctree::
   :maxdepth: 2

   task_architecture
   tactile_pipeline
   reward_design
   control_pipeline
   pts_taxel_scheme
