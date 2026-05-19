# contactile-mjlab

基于 MuJoCo / mjlab 的触觉抓取强化学习实验环境。

目标系统：**基于阵列触觉反馈的 Robotiq 2F-85 限力位置闭环抓取控制**。

## 快速开始

```bash
# CPU 后端
uv sync --extra cpu --group dev

# 验证
uv run python main.py
```

## 目录

```text
assets/    — MJCF 模型与道具
src/       — Python 包 (contactile_mjlab)
scripts/   — 调试与训练脚本
docs/      — Sphinx 文档
```

## 文档

```bash
cd docs && uv run sphinx-build -b html source _build
# 浏览器打开 docs/_build/html/index.html
```

## 路线

- **V0** — 全局 wrench baseline
- **V1** — 3×3 法向触觉阵列（当前阶段）
- **V2** — 触觉阵列 + wrench + slip proxy

详见 `plan.md`。
