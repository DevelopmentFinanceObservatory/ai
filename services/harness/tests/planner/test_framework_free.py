"""Every planner module but its session reader, a planner run's bookkeeping,
the submission chain and the translator import no ADK.

Each module's source is parsed. The check reads each module's own imports, so a
module can load ADK through a module it imports. `planner/tools.py` holds the
planner's tool bodies over a `PlannerCall`, `planner/source_reads.py` the reads
they make and `planner/skills.py` the skill reads. `planner/specs.py` declares
each tool as a `planner/tool_spec.py` `ToolSpec`, whose declaration
`planner/declaration.py` compacts, and `planner/pydantic_agent.py` serves the
specs to Pydantic AI. `planner/session.py` reads an ADK session for a tool
call (`SessionCall`), and is the one planner module left out.
`harness/planner_run.py` holds a planner run's bookkeeping, which
`harness/planner_node.py` drives from the planner's model hooks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[2] / "src"

_MODULES = sorted(
    [
        "harness/planner_run.py",
        *(
            path.relative_to(_SRC).as_posix()
            for path in (_SRC / "planner").glob("*.py")
            if path.name != "session.py"
        ),
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
