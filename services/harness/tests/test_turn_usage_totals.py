"""The `studio.turn` line carries the model usage summed over the turn's model
calls (`harness.conversation.note_model_usage`), across the segments of a turn
that paused and resumed."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from common.logging import JSON_FIELDS
from harness.conversation import current_model_usage
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import Reply, call_part, reply, text_part, usage_metadata
from .scripted_tools import CannedSearch, definite_envelope
from .studio_routes import (
    CLIENT_FEATURES,
    CONVERSATION,
    orchestrator_app,
    stream,
    studio_client,
    turn_body,
)

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_TOTALS = ("model_calls", "input_tokens", "cached_tokens", "output_tokens", "thinking_tokens")


def _totals(caplog: pytest.LogCaptureFixture) -> list[tuple[Any, ...]]:
    """The usage totals of every closed `studio.turn` line, in order."""
    return [
        tuple(fields[key] for key in _TOTALS)
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.turn"
        and fields.get("status") == "closed"
    ]


async def test_a_turn_s_line_sums_the_usage_of_its_model_calls(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    app, _ = orchestrator_app(
        Reply(
            chunks=[[call_part("search", {"parts": [{"variable": "life expectancy"}]})]],
            usage=usage_metadata(prompt=1_000, cached=400, candidates=20, thoughts=30),
        ),
        reply(
            text_part("Life expectancy in Kenya was 62 years."),
            usage=usage_metadata(prompt=1_500, cached=1_000, candidates=40, thoughts=5),
        ),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(app) as client:
            frames = await stream(
                client, "/api/studio/turn", turn_body("Life expectancy in Kenya?")
            )
    assert frames[-1]["event"] == "done"

    assert _totals(caplog) == [(2, 2_500, 1_400, 60, 35)]
    (message,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("studio.turn")
    ]
    assert (
        "model_calls=2 input_tokens=2500 cached_tokens=1400 output_tokens=60 thinking_tokens=35"
        in message
    )


async def test_a_resumed_turn_s_line_sums_both_segments(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    app, _ = orchestrator_app(
        Reply(chunks=[[asking]], usage=usage_metadata(prompt=700, candidates=10, thoughts=3)),
        reply(
            text_part("A it is."),
            usage=usage_metadata(prompt=900, cached=300, candidates=20, thoughts=4),
        ),
        engine=pg_engine,
    )

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(app) as client:
            paused = await stream(client, "/api/studio/turn", turn_body("Compare them"))
            assert paused[-1]["event"] == "elicitation.request"
            assert _totals(caplog) == []
            # A request of its own starts with no sink, as a process's does.
            current_model_usage.set(None)
            (elicitation,) = [f for f in paused if f["event"] == "elicitation.request"]
            (question,) = elicitation["questions"]
            answer = {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": elicitation["interrupt_id"],
                "decision": "answer",
                "answers": {
                    question["id"]: {"kind": "option", "option_id": question["options"][0]["id"]}
                },
                "client_features": CLIENT_FEATURES,
            }
            frames = await stream(client, "/api/studio/interrupt", answer)
    assert frames[-1]["event"] == "done"

    assert _totals(caplog) == [(2, 1_600, 300, 30, 7)]
