"""Sphinx configuration for contactile-mjlab."""

from __future__ import annotations

import sys
from pathlib import Path

_source_dir = Path(__file__).resolve().parent
_project_root = _source_dir.parents[1]
sys.path.insert(0, str(_project_root / "src"))

project = "contactile-mjlab"
copyright = "2026"  # noqa: A001
author = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
