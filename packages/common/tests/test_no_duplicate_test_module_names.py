"""Guards the pytest module-naming invariant this suite relies on. No two
test files collected in the same run may resolve to the same dotted module
name under `--import-mode=importlib`. When two do, pytest does not raise.
It silently imports whichever file it reaches first and reuses that cached
module for the other's same-named file, so the second file's own tests never
run. `pyproject.toml`'s `consider_namespace_packages` comment explains the
mechanism this guards against.

Checking the module name pytest actually assigned to each collected item,
rather than recomputing a hypothetical name for every file under `testpaths`,
keeps this accurate for a partial run (e.g. `just test-data-mcp`) as well as
the full-workspace one. A same-name collision only matters between two files
that are actually collected together in one process.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_no_two_collected_test_files_share_a_dotted_module_name(
    request: pytest.FixtureRequest,
) -> None:
    files_by_module_name: dict[str, set[Path]] = {}
    for item in request.session.items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        files_by_module_name.setdefault(module.__name__, set()).add(Path(item.path))

    collisions = {name: paths for name, paths in files_by_module_name.items() if len(paths) > 1}
    assert not collisions, f"test files collide on one pytest module name: {collisions}"
