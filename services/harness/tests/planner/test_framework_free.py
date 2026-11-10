"""The planner's call record and tool bodies, the submission chain and the translator import no ADK.

Each module's source is parsed, not imported, so the check reads each
module's own imports only. `planner/tools.py` holds the planner's tool bodies
over a `PlannerCall`, `planner/source_reads.py` the reads they make and
`planner/skills.py` the skill reads; `planner/agent.py` wraps them for ADK.
`planner/call.py`, `submission/service.py` and `submission/prices_choice.py`
still load ADK transitively, through `harness.graph` and `harness.turn_events`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[2] / "src"

_MODULES = sorted(
    [
        "planner/call.py",
        "planner/skills.py",
        "planner/source_reads.py",
        "planner/tools.py",
        *(path.relative_to(_SRC).as_posix() for path in (_SRC / "submission").glob("*.py")),
        *(path.relative_to(_SRC).as_posix() for path in (_SRC / "translator").glob("*.py")),
    ]
)


def _adk_imports(source: str) -> list[str]:
    """Every `google.adk` module `source` imports, as written."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            imported += [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    return [name for name in imported if name == "google.adk" or name.startswith("google.adk.")]


@pytest.mark.parametrize("module", _MODULES)
def test_the_module_imports_no_adk(module: str) -> None:
    assert _adk_imports((_SRC / module).read_text(encoding="utf-8")) == []
