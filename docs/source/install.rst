安装与环境配置
==============

依赖
----

- Python >= 3.12, < 3.14
- MuJoCo >= 3.3.7
- mjlab >= 1.3.0（含 cpu / cu128 后端）

使用 uv 安装
------------

.. code-block:: bash

   # CPU 后端
   uv sync --extra cpu --group dev

   # CUDA 12.8 后端
   uv sync --extra cu128 --group dev

验证导入
--------

.. code-block:: bash

   uv run python -c "from contactile_mjlab import TactileGraspEnv; print('OK')"

Smoke test
----------

.. code-block:: bash

   uv run python main.py
