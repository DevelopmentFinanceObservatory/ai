"""Guards the invariant that keeps pytest's rootdir at the workspace root. No
`pyproject.toml` in the tree other than the workspace root's may carry pytest
configuration.

pytest picks its `configfile` by walking up from the test path(s) it is given
and stopping at the first ancestor directory holding recognised pytest
configuration (a `[tool.pytest.ini_options]` or `[tool.pytest]` table in
`pyproject.toml`, or a `pytest.ini`/`tox.ini`/`setup.cfg`). Whichever file it
stops at becomes `rootdir`, and `confcutdir` (the directory outside which
conftest.py files are not collected) defaults to `rootdir` when not given
explicitly on the command line. A member's own `pyproject.toml` carrying such
a table would win that search for any invocation pointed at its tests (for
example `pytest services/harness/tests/test_server.py`, run directly rather
than through a `just` recipe), moving rootdir and confcutdir down into that
member and silently dropping the workspace root's `conftest.py`, including
its autouse fixture that keeps a developer's real `.env` out of the suite.
"""

from __future__ import annotations

import pytest
from _pytest.config.findpaths import locate_config


def test_no_member_pyproject_carries_pytest_config(pytestconfig: pytest.Config) -> None:
    root = pytestconfig.rootpath
    offenders = {}
    for pyproject in sorted(root.glob("*/*/pyproject.toml")):
        rootdir, inifile, _, _ = locate_config(pyproject.parent, [pyproject.parent])
        if rootdir != root:
            offenders[str(pyproject.relative_to(root))] = str(inifile)
    assert not offenders, (
        "these member pyproject.toml files carry pytest configuration that "
        f"shadows the workspace root's: {offenders}"
    )
