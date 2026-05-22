"""Wrapper: register tactile_grasp tasks then delegate to mjlab.scripts.train."""

from __future__ import annotations

from mjlab.scripts.train import main

import tactile_grasp  # noqa: F401 -- import side-effect registers the task

if __name__ == "__main__":
    main()
