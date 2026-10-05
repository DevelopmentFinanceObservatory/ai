"""Shared fixtures for the data-mcp test suite.

`_reset_client`, `_reset_store`, `_reset_frame_store`, and `_empty_mint_ledger` are
autouse: every test wires its own fake upstream client, definition store, and (where
relevant) frame materialisation store, starts with no mints held, and nothing should leak
into the next test, including a later test in another package entirely, since this
workspace's whole suite runs in one process (`pyproject.toml`'s root `testpaths`).
Living here once, rather than repeated per test module, keeps that reset behaviour singly
defined; a module that needs the same name importable for its own purposes
(`test_search_tool.py`, re-exported for `test_search_tool_definition_store.py`) imports it
from here rather than redefining it.
"""

from __future__ import annotations

from typing import Any

import pytest
from common.materialisation import set_store as set_frame_store
from data_mcp import client_holder, get_data_tool
from data_mcp.definition_store import set_store
from data_mcp.mint_ledger import MintLedger


@pytest.fixture(autouse=True)
async def _reset_client() -> Any:
    yield
    client_holder.set_client(None)


@pytest.fixture(autouse=True)
async def _reset_store() -> Any:
    yield
    set_store(None)


@pytest.fixture(autouse=True)
async def _reset_frame_store() -> Any:
    yield
    set_frame_store(None)


@pytest.fixture(autouse=True)
def _empty_mint_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_data_tool, "_mint_ledger", MintLedger())
