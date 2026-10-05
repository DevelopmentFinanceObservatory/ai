"""Smoke test proving the `planner` package is wired into the workspace.

`planner` ships as a second top-level module inside `services/harness/src/`
(see `[tool.uv.build-backend] module-name` in that service's pyproject), so an
import failure here means the build-backend config, not a bug in the package.
"""

from __future__ import annotations

import planner


def test_package_importable() -> None:
    assert planner.__version__
