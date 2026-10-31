"""Run these tests under the fixtures the harness suite runs under.

The workspace-root `conftest.py` and `services/harness/tests/conftest.py` are
loaded by path and their fixtures re-exported, so every test here gets the same
autouse isolation (no `.env`, no shared secret, no graph or runner calls) as
`tests/test_orchestrator_path.py`.
"""

import importlib.util
from pathlib import Path

from _pytest.fixtures import FixtureFunctionDefinition

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_workspace = _load("workspace_conftest", _ROOT / "conftest.py")
from tests import conftest as _harness  # noqa: E402 -- after the workspace env defaults

globals().update(
    {
        name: value
        for module in (_workspace, _harness)
        for name, value in vars(module).items()
        if isinstance(value, FixtureFunctionDefinition)
    }
)
