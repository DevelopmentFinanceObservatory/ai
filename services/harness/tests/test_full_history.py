"""`STUDIO_HISTORY=full`: a follow-up turn's orchestrator opens on the previous
turn's conversation whole.

Turn N's first request must extend turn N-1's last request byte for byte, the
only shape Gemini's implicit cache serves, so each test compares the
`contents` the model client was sent (`harness.gemini_wire` passes them on) as
google-genai JSON. The reload test sends turn 2 through a second app on the
same database, so the carried conversation passes through the conversation
store first.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from harness.settings import studio_history
from harness.studio_state import CARRIED_HISTORY_TOKEN_CAP
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import Reply, Request, call_part, text_part, thought_part, usage_metadata
from .scripted_tools import CannedSearch, definite_envelope
from .studio_routes import orchestrator_app, stream, studio_client, turn_body

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_RESET = "studio history: chain reset"


@pytest.fixture
def full_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", "full")


def _contents(request: Request) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json", exclude_none=True) for c in request.content_list]


def _assert_extends(later: Request, earlier: Request) -> None:
    """`later`'s contents start with every content of `earlier`, the same
    bytes in the same order, and both send one system instruction and tools."""
    before, after = json.dumps(_contents(earlier)), json.dumps(_contents(later))
    # The earlier list's closing bracket is where the later one goes on.
    prefix = before[:-1] + ", "
    if not after.startswith(prefix):
        offset = next(
            (i for i, (a, b) in enumerate(zip(prefix, after, strict=False)) if a != b),
            min(len(prefix), len(after)),
        )
        pytest.fail(
            f"first difference at byte {offset}: earlier {prefix[offset - 80 : offset + 80]!r} "
            f"later {after[offset - 80 : offset + 80]!r}"
        )
    assert earlier.generate_config.system_instruction == later.generate_config.system_instruction
    assert earlier.generate_config.tools == later.generate_config.tools


def _opening_texts(request: Request, count: int) -> list[str]:
    """The text of the last `count` contents of `request`."""
    return [content["parts"][0]["text"] for content in _contents(request)[-count:]]


# Turn 1: a signed thought and a search, then a signed thought and the answer.
_TURN_1 = (
    Reply(
        chunks=[
            [
                thought_part("Search first.", signature=b"sig-1"),
                call_part("search", {"parts": [{"variable": "life expectancy"}]}, signature=b"c-1"),
            ]
        ]
    ),
    Reply(
        chunks=[
            [thought_part("Answer it.", signature=b"sig-2"), text_part("Life expectancy ")],
            [text_part("in Kenya was 62 years.")],
        ],
        usage=usage_metadata(prompt=9_000, candidates=40),
    ),
)


async def _turns(app: Any, *messages: str) -> None:
    async with studio_client(app) as client:
        for message in messages:
            frames = await stream(client, "/api/studio/turn", turn_body(message))
            assert frames[-1]["event"] in {"done", "error"}


@pytest.mark.usefixtures("full_history")
async def test_a_follow_up_s_first_request_extends_the_previous_turn_s_last_whole(
    pg_engine: AsyncEngine,
) -> None:
    app, model = orchestrator_app(
        *_TURN_1,
        text_part("Uganda's was 63."),
        text_part("Both rose."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )

    await _turns(app, "Life expectancy in Kenya?", "And Uganda?", "Compare them")

    first, last, second, third = model.requests
    assert first is not last
    _assert_extends(second, last)
    # Turn 3 extends turn 2's one request in turn.
    _assert_extends(third, second)
    # After the carried messages: turn 2's date, canvas and digest tail, then its message.
    tail, question = _opening_texts(second, 2)
    assert tail.endswith('Earlier turns in this conversation: {"total_turns": 1}')
    assert '"turns"' not in tail
    assert question == "And Uganda?"
    assert len(_contents(second)) == len(_contents(last)) + 3


@pytest.mark.usefixtures("full_history")
async def test_the_follow_up_extends_the_turn_another_process_stored(
    pg_engine: AsyncEngine,
) -> None:
    app, model = orchestrator_app(
        *_TURN_1, resolver=CannedSearch(definite_envelope()), engine=pg_engine
    )
    await _turns(app, "Life expectancy in Kenya?")
    # A second app on the same database loads the conversation from the store.
    reloaded, model_after = orchestrator_app(text_part("Uganda's was 63."), engine=pg_engine)
    await _turns(reloaded, "And Uganda?")

    _assert_extends(model_after.requests[0], model.requests[-1])


@pytest.mark.usefixtures("full_history")
@pytest.mark.parametrize(
    ("turn_1", "reason"),
    [
        pytest.param(
            (Reply(open_error=ValueError("refused")),),
            "previous_turn_carried_none",
            id="failed-turn",
        ),
        pytest.param(
            (
                Reply(
                    chunks=[[text_part("62 years.")]],
                    usage=usage_metadata(prompt=CARRIED_HISTORY_TOKEN_CAP, candidates=1),
                ),
            ),
            "token_cap",
            id="token-cap",
        ),
    ],
)
async def test_a_turn_with_nothing_to_carry_starts_a_fresh_chain_from_the_digest(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    turn_1: tuple[Reply, ...],
    reason: str,
) -> None:
    app, model = orchestrator_app(
        *turn_1, text_part("Uganda's was 63."), text_part("Both rose."), engine=pg_engine
    )

    with caplog.at_level(logging.INFO, logger="harness.studio_state"):
        await _turns(app, "Life expectancy in Kenya?", "And Uganda?", "Compare them")

    *_, second, third = model.requests
    # Turn 2 opens as the digest opens: its two contents alone, the turns listed.
    assert len(_contents(second)) == 2
    digest, question = _opening_texts(second, 2)
    assert '"turns": [' in digest
    assert question == "And Uganda?"
    # Turn 3 extends the fresh chain.
    _assert_extends(third, second)
    resets = [r.getMessage() for r in caplog.records if r.getMessage().startswith(_RESET)]
    assert len(resets) == 1
    assert f"reason={reason}" in resets[0]


async def test_under_the_digest_a_follow_up_opens_from_the_digest_and_carries_nothing(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STUDIO_HISTORY", raising=False)
    app, model = orchestrator_app(
        *_TURN_1,
        text_part("Uganda's was 63."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )

    await _turns(app, "Life expectancy in Kenya?", "And Uganda?")

    second = model.requests[-1]
    assert len(_contents(second)) == 2
    assert '"turns": [' in _opening_texts(second, 2)[0]
    async with pg_engine.connect() as connection:
        stored = (await connection.execute(text("SELECT turn_state FROM conversations"))).scalar()
    assert stored.get("carried_history") is None


@pytest.mark.parametrize(("value", "read"), [("", "digest"), (" Full", "full")])
def test_the_history_setting_reads_blank_as_the_digest_and_ignores_case(
    monkeypatch: pytest.MonkeyPatch, value: str, read: str
) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", value)
    assert studio_history() == read


def test_an_unknown_history_setting_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", "frozen")
    with pytest.raises(ValidationError):
        studio_history()
