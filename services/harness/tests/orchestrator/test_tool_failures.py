import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from orchestrator.tool_failures import answering_tool_failures


def test_a_cancelled_call_logs_nothing_and_leaves_adk_to_end_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ADK runs `on_tool_error_callback` for cleanup when a tool call is
    cancelled, and a cancellation is no tool failure."""
    answer = answering_tool_failures({"search"})
    tool_context = SimpleNamespace(session=SimpleNamespace(id="s"), invocation_id="i")

    with caplog.at_level(logging.ERROR, logger="orchestrator.tool_failures"):
        result = answer(
            cast(Any, SimpleNamespace(name="search")),
            {},
            cast(Any, tool_context),
            asyncio.CancelledError(),
        )

    assert result is None
    assert not [r for r in caplog.records if r.name == "orchestrator.tool_failures"]
