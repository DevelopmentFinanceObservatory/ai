"""A guard: the studio's frozen planner path stays deleted.

`harness.studio_planner` and `harness.studio_plan` were the studio's own
pinned planner agent and its plan contract. The frozen adapter that drove
them, `harness.planner_seam`, exported `PlannerAgent` and
`PlannerAgentError`. That module path now names its replacement instead
(`harness.planner_seam.PlannerSeam`, over `planner.agent` and
`translator.agent`), so banning the path would also ban legitimate use of
the current seam. Banning `PlannerAgent`/`PlannerAgentError` by name is
strictly stronger than banning the old path was: it catches the frozen
adapter reintroduced under any module, not only the one it used to live at.

This walks every module under `services/harness/src` and asserts none of
the banned names is imported, so a reintroduced import fails loudly here
instead of resurfacing as a stale docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BANNED_MODULES = {"harness.studio_planner", "harness.studio_plan"}
_BANNED_SYMBOLS = {"PlannerAgent", "PlannerAgentError"}

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _frozen_adapter_refs(source: str) -> set[str]:
    tree = ast.parse(source)
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _BANNED_MODULES:
                refs.add(node.module)
            refs.update(alias.name for alias in node.names if alias.name in _BANNED_SYMBOLS)
        elif isinstance(node, ast.Import):
            refs.update(alias.name for alias in node.names if alias.name in _BANNED_MODULES)
    return refs


def _banned_imports_under(root: Path) -> dict[Path, set[str]]:
    offenders: dict[Path, set[str]] = {}
    for path in root.rglob("*.py"):
        hit = _frozen_adapter_refs(path.read_text())
        if hit:
            offenders[path] = hit
    return offenders


def test_no_module_imports_the_frozen_planner_path() -> None:
    assert _banned_imports_under(_SRC_ROOT) == {}


def test_the_guard_catches_a_banned_module_import(tmp_path: Path) -> None:
    """`_banned_imports_under` is provably failable: a scratch module naming
    one of the banned modules is caught, not silently passed."""
    scratch = tmp_path / "offender.py"
    scratch.write_text("from harness.studio_planner import create_planner_agent\n")

    assert _banned_imports_under(tmp_path) == {scratch: {"harness.studio_planner"}}


def test_the_guard_catches_a_banned_symbol_import(tmp_path: Path) -> None:
    """The symbol ban is provably failable too: `PlannerAgent` imported from
    any module, not only the old `harness.planner_seam` path, is caught --
    the coverage a module-path ban on that path alone could not offer, since
    the path is now legitimately in use by its replacement."""
    scratch = tmp_path / "offender.py"
    scratch.write_text("from some.other.module import PlannerAgent\n")

    assert _banned_imports_under(tmp_path) == {scratch: {"PlannerAgent"}}
