"""No `__init__.py` in this directory (deliberately -- see below): pytest's
`--import-mode=importlib` module-naming falls back to a root-relative unique
name only when no `__init__.py` chain resolves a package for a test file.
`services/analysis-mcp`, like every service directory in this workspace,
has a hyphen in its name, so it can never itself join that package chain
(`_pytest.pathlib.resolve_package_path` stops at the first ancestor whose
name isn't a valid identifier). With a `tests/__init__.py` present, every
service's own `tests` package therefore resolves to the identical dotted
name `tests.<module>`, and pytest's importlib import mode silently reuses
whichever service's module it imported first for every later same-named
file (`test_digest.py`, `test_server.py`, ...) instead of raising -- tests
never fail, they just quietly stop being the ones this file wrote. Run from
the workspace root (`uv run pytest`, what CI runs), that would mask every
one of this service's own `test_digest.py`/`test_digest_budget.py` cases
behind `data-mcp`'s identically-named, identically-shaped ones.

Removing `__init__.py` sends every file here through the root-relative
fallback name instead, which is unique by construction. The one cost: a
bare `from digest_corpus import ...` (see `test_digest_budget.py`) needs
this directory on `sys.path`, since nothing implicitly adds it once it's
not a package -- this file supplies exactly that, once, at collection
time, so it does not matter which import mode or which command (`uv run
pytest` at the workspace root, `just test-analysis-mcp` scoped here) is
in use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from common.materialisation import set_store

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
async def _reset_materialisation_store() -> Any:
    """A defensive backstop, not a fix for a known leak: every verb test
    file's own `seeded` fixture already calls `set_store(None)` in its own
    teardown, so this guarantees the process-wide holder is clear after
    every test in this package regardless -- including `test_server.py`'s
    `_build_store`-focused tests (which never touch the holder themselves)
    and anything future that might call `set_store(...)` without its own
    explicit teardown. The whole workspace suite runs in one pytest
    process, so a store left set here is exactly as reachable by a later
    data-mcp or harness test as the reverse.
    """
    yield
    set_store(None)
