"""Run these tests under the fixtures `harness-ab` runs under.

`harness-ab/conftest.py` loads the workspace-root and harness-suite conftests by path
and re-exports their fixtures; this module loads it by path and re-exports the same
fixtures, so every test here gets the same autouse isolation (no `.env`, no shared
secret, no graph or runner calls). It adds `rig`, plan-mcp over each transport.
"""

import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureFunctionDefinition

_AB_CONFTEST = Path(__file__).resolve().parent.parent / "harness-ab" / "conftest.py"

_spec = importlib.util.spec_from_file_location("harness_ab_conftest", _AB_CONFTEST)
assert _spec is not None and _spec.loader is not None
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)

globals().update(
    {
        name: value
        for name, value in vars(_ab).items()
        if isinstance(value, FixtureFunctionDefinition)
    }
)

from entry import quiet_logs  # noqa: E402 -- after the harness suite's env defaults
from rig import TRANSPORTS, Rig, serve  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _quiet() -> None:
    quiet_logs()


@pytest.fixture(params=TRANSPORTS)
async def rig(request: pytest.FixtureRequest) -> AsyncIterator[Rig]:
    async with serve(request.param) as served:
        yield served
