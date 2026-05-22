"""TouchSite 代码路径应已删除（资产保留）."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_touch_site_symbols_in_code():
    """Ensure no TouchSite code symbols remain in src/scripts/tests/main.py.

    Path constants TACTILE_XML / TACTILE_SCENE_XML in paths.py are exempt.
    """
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            r"TOUCH_SITE\|touch_map\|TACTILE_MODEL\|LEFT_TOUCH_SENSOR\|RIGHT_TOUCH_SENSOR\|tactile_model",
            "src/",
            "scripts/",
            "tests/",
            "main.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    hits = [
        line
        for line in result.stdout.splitlines()
        if line
        and "test_no_touchsite.py" not in line
        and "TACTILE_XML" not in line  # paths.py 资产常量豁免
        and "TACTILE_SCENE_XML" not in line  # paths.py 资产常量豁免
    ]
    assert not hits, "残留 TouchSite 符号:\n" + "\n".join(hits)
