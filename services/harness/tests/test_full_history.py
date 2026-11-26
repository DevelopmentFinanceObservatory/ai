"""The `full` history arm: a follow-up turn's orchestrator opens on the previous
turn's conversation whole. And `STUDIO_HISTORY=split`, which gives each
conversation the arm a hash of its id assigns and stores.

Turn N's first request must extend turn N-1's last request byte for byte, the
only shape Gemini's implicit cache serves, so each test compares the
`contents` the model client was sent (`harness.gemini_wire` passes them on) as
google-genai JSON. The reload tests send a later turn through a second app on
the same database, so the carried conversation and the stored arm pass
through the conversation store first.

`_FULL_ID` and `_DIGEST_ID` are conversation ids `split` assigns the `full`
and the `digest` arm at the default share; `test_the_chosen_ids_hash_into_their_arms`
holds them to it. Under `split` a test that reads the process setting at any
site but the turn's begin would run both on `digest`, so the split tests drive
whole turns over the routes and read the requests, the spans and the
`studio.turn` lines.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pytest
from common.logging import JSON_FIELDS
from harness.settings import HistorySettings
from harness.studio_state import (
    CARRIED_HISTORY_TOKEN_CAP,
    ConversationState,
    TurnUsage,
    history_arm,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import Reply, Request, call_part, text_part, thought_part, usage_metadata
from .scripted_tools import CannedSearch, definite_envelope
from .studio_routes import CLIENT_FEATURES, orchestrator_app, stream, studio_client, turn_body

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_RESET = "studio history: chain reset"
# Conversation ids `split` assigns the `full` and the `digest` arm at 0.5.
_FULL_ID = "conv-full"
_DIGEST_ID = "conv-a"


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


async def _turns(app: Any, *messages: str, conversation_id: str | None = None) -> None:
    ids = {"conversation_id": conversation_id} if conversation_id is not None else {}
    async with studio_client(app) as client:
        for message in messages:
            frames = await stream(client, "/api/studio/turn", turn_body(message, **ids))
            assert frames[-1]["event"] in {"done", "error"}


def _turn_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """The JSON fields of every closed `studio.turn` line."""
    return [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.turn"
        and fields.get("status") == "closed"
    ]


def _history_fields(line: dict[str, Any]) -> tuple[Any, ...]:
    return (
        line["turn"],
        line["history"],
        line["history_arm"],
        line["history_mode"],
        line["history_assigned_turn"],
        line["history_chain"],
    )


async def _stored(engine: AsyncEngine, conversation_id: str, key: str) -> Any:
    """`key` of the turn state stored for `conversation_id`."""
    async with engine.connect() as connection:
        state = (
            await connection.execute(
                text("SELECT turn_state FROM conversations WHERE conversation_id = :id"),
                {"id": conversation_id},
            )
        ).scalar_one()
    return state.get(key)


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
    pg_engine: AsyncEngine,
) -> None:
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


# --- split: the arm a conversation is assigned and keeps --------------------------


def _split(share: float = 0.5) -> HistorySettings:
    return HistorySettings(studio_history="split", studio_history_full_share=share)


def test_the_chosen_ids_hash_into_their_arms() -> None:
    assert history_arm(_FULL_ID, full_share=0.5) == "full"
    assert history_arm(_DIGEST_ID, full_share=0.5) == "digest"


@pytest.mark.parametrize("share", [0.0, 0.2, 0.5, 1.0])
def test_split_assigns_the_full_arm_to_about_the_share_and_by_the_salted_hash(
    share: float,
) -> None:
    """The arm is `full` where the first 8 bytes of sha256("studio_history:" +
    id), over 2**64, fall below the share; over 2,000 ids the `full` count is
    within four standard deviations of the share, and the same id always
    gets the same arm."""
    ids = [f"conversation-{index}" for index in range(2_000)]
    arms = [history_arm(conversation_id, full_share=share) for conversation_id in ids]

    def hashed(conversation_id: str) -> float:
        digest = hashlib.sha256(f"studio_history:{conversation_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    assert arms == ["full" if hashed(i) < share else "digest" for i in ids]
    assert arms == [history_arm(conversation_id, full_share=share) for conversation_id in ids]
    spread = 4 * (len(ids) * share * (1 - share)) ** 0.5
    assert abs(arms.count("full") - len(ids) * share) <= spread


def _history(usage: TurnUsage) -> tuple[Any, ...]:
    return (
        usage.turn,
        usage.history,
        usage.history_arm,
        usage.history_mode,
        usage.history_assigned_turn,
    )


def test_the_first_split_turn_stores_the_arm_and_a_moved_share_moves_no_later_turn() -> None:
    state = ConversationState(conversation_id=_FULL_ID)

    state.begin_turn(history=_split())
    assert state.history_assignment == {"arm": "full", "turn": 1}
    assert _history(state.turn_usage) == (1, "full", "full", "split", 1)

    state.close_turn(final_text="done")
    state.begin_turn(history=_split(share=0.0))
    assert state.history_assignment == {"arm": "full", "turn": 1}
    assert _history(state.turn_usage) == (2, "full", "full", "split", 1)


def test_a_forced_arm_overrides_the_stored_one_and_leaves_it_stored() -> None:
    """A forced arm applies to every turn begun under it and stores none, so
    the first turn under `split` afterwards assigns the arm and names itself
    as the assigning turn, and a stored arm returns once `split` does."""
    state = ConversationState(conversation_id=_FULL_ID)

    state.begin_turn(history=HistorySettings(studio_history="digest"))
    assert state.history_assignment is None
    assert _history(state.turn_usage) == (1, "digest", None, "forced", None)
    state.close_turn(final_text="done")

    state.begin_turn(history=_split())
    assert _history(state.turn_usage) == (2, "full", "full", "split", 2)
    state.close_turn(final_text="done")

    state.begin_turn(history=HistorySettings(studio_history="digest"))
    assert state.history_assignment == {"arm": "full", "turn": 2}
    assert _history(state.turn_usage) == (3, "digest", "full", "forced", 2)
    state.close_turn(final_text="done")

    state.begin_turn(history=_split())
    assert _history(state.turn_usage) == (4, "full", "full", "split", 2)


def test_begin_turn_reads_the_setting_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", " Split ")
    monkeypatch.setenv("STUDIO_HISTORY_FULL_SHARE", "1")
    state = ConversationState(conversation_id=_DIGEST_ID)

    state.begin_turn()

    assert _history(state.turn_usage) == (1, "full", "full", "split", 1)


@pytest.mark.parametrize(
    "between",
    [
        pytest.param(None, id="unchanged"),
        pytest.param({"STUDIO_HISTORY_FULL_SHARE": "0"}, id="share-zero-reloaded"),
        pytest.param({"STUDIO_HISTORY": "full"}, id="forced-full-reloaded"),
    ],
)
async def test_a_split_conversation_on_the_full_arm_extends_its_previous_turn(
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    between: dict[str, str] | None,
) -> None:
    """Under `split`, a conversation assigned `full` carries its conversation
    into the follow-up, which extends turn 1's last request. With the process
    setting changed between the turns, turn 2 runs on a second app over the
    same database: the arm stored on turn 1 still decides (a share of 0
    would assign `digest` afresh), and a forced `full` keeps the chain."""
    monkeypatch.setenv("STUDIO_HISTORY", "split")
    app, model = orchestrator_app(
        *_TURN_1,
        text_part("Uganda's was 63."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        last = model.requests[-1]
        if between is not None:
            for name, value in between.items():
                monkeypatch.setenv(name, value)
            app, model = orchestrator_app(text_part("Uganda's was 63."), engine=pg_engine)
        await _turns(app, "And Uganda?", conversation_id=_FULL_ID)

    _assert_extends(model.requests[-1], last)
    mode = "forced" if between == {"STUDIO_HISTORY": "full"} else "split"
    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "full", "full", mode, 1, "extended"),
    ]
    assert await _stored(pg_engine, _FULL_ID, "history_assignment") == {"arm": "full", "turn": 1}


async def test_a_forced_digest_opens_a_full_arm_conversation_from_the_digest_and_keeps_its_arm(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rollback: under `STUDIO_HISTORY=digest` a conversation assigned `full`
    opens from the digest, and its stored arm stays. Back under `split` it
    is `full` again, on a fresh chain, since the digest turn carried none."""
    monkeypatch.setenv("STUDIO_HISTORY", "split")
    app, model = orchestrator_app(
        *_TURN_1,
        text_part("Uganda's was 63."),
        text_part("Both rose."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        monkeypatch.setenv("STUDIO_HISTORY", "digest")
        await _turns(app, "And Uganda?", conversation_id=_FULL_ID)
        second = model.requests[-1]
        assert await _stored(pg_engine, _FULL_ID, "history_assignment") == {
            "arm": "full",
            "turn": 1,
        }
        monkeypatch.setenv("STUDIO_HISTORY", "split")
        await _turns(app, "Compare them", conversation_id=_FULL_ID)

    assert len(_contents(second)) == 2
    assert '"turns": [' in _opening_texts(second, 2)[0]
    assert len(_contents(model.requests[-1])) == 2
    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "digest", "full", "forced", 1, None),
        (3, "full", "full", "split", 1, "reset:previous_turn_carried_none"),
    ]


async def test_each_turn_s_spans_and_line_name_its_own_conversation_s_arm(
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exported_spans: InMemorySpanExporter,
) -> None:
    """Two conversations on different arms in one process, turns interleaved:
    every model span of a turn names the arm and mode its `studio.turn` line
    does, and the line's text says it too."""
    monkeypatch.setenv("STUDIO_HISTORY", "split")
    app, _model = orchestrator_app(
        *_TURN_1,
        text_part("Ghana's was 64."),
        text_part("Uganda's was 63."),
        text_part("Togo's was 61."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )
    seen: list[tuple[str, int, set[tuple[Any, Any]]]] = []
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        for conversation_id, message in [
            (_FULL_ID, "Life expectancy in Kenya?"),
            (_DIGEST_ID, "Life expectancy in Ghana?"),
            (_FULL_ID, "And Uganda?"),
            (_DIGEST_ID, "And Togo?"),
        ]:
            exported_spans.clear()
            await _turns(app, message, conversation_id=conversation_id)
            chats = [s for s in exported_spans.get_finished_spans() if s.name.startswith("chat ")]
            assert chats
            spans = {
                (
                    (s.attributes or {}).get("studio.surface.history"),
                    (s.attributes or {}).get("studio.history.mode"),
                )
                for s in chats
            }
            seen.append((conversation_id, len(chats), spans))

    lines = _turn_lines(caplog)
    assert [(conversation_id, spans) for conversation_id, _, spans in seen] == [
        (_FULL_ID, {("full", "split")}),
        (_DIGEST_ID, {("digest", "split")}),
        (_FULL_ID, {("full", "split")}),
        (_DIGEST_ID, {("digest", "split")}),
    ]
    assert [_history_fields(line) for line in lines] == [
        (1, "full", "full", "split", 1, "fresh"),
        (1, "digest", "digest", "split", 1, None),
        (2, "full", "full", "split", 1, "extended"),
        (2, "digest", "digest", "split", 1, None),
    ]
    assert lines[2]["carried_tokens"] == 9_040
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("studio.turn ")]
    assert "turn=2 status=closed" in messages[2]
    assert (
        "history=full history_arm=full history_mode=split history_assigned_turn=1 "
        "history_chain=extended carried_tokens=9040"
    ) in messages[2]


def _answer(frames: list[dict[str, Any]], conversation_id: str) -> dict[str, Any]:
    """The `/interrupt` answer to the stream's question, picking its first option."""
    (elicitation,) = [f for f in frames if f["event"] == "elicitation.request"]
    (question,) = elicitation["questions"]
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": conversation_id,
        "interrupt_id": elicitation["interrupt_id"],
        "decision": "answer",
        "answers": {question["id"]: {"kind": "option", "option_id": question["options"][0]["id"]}},
        "client_features": CLIENT_FEATURES,
    }


async def test_a_paused_full_turn_resumed_after_a_rollback_keeps_its_arm(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A `full` turn paused on a question and answered on a second app started
    with `STUDIO_HISTORY=digest` keeps the arm it began with: its run stores
    its conversation, and its line says `full` and `split`. The turn after it
    begins under `digest`."""
    monkeypatch.setenv("STUDIO_HISTORY", "split")
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    app, model = orchestrator_app(
        *_TURN_1, asking, resolver=CannedSearch(definite_envelope()), engine=pg_engine
    )
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        async with studio_client(app) as client:
            paused = await stream(
                client, "/api/studio/turn", turn_body("And Uganda?", conversation_id=_FULL_ID)
            )
        assert paused[-1]["event"] == "elicitation.request"
        monkeypatch.setenv("STUDIO_HISTORY", "digest")
        resumed_app, resumed_model = orchestrator_app(
            text_part("Uganda's was 63."), text_part("Both rose."), engine=pg_engine
        )
        async with studio_client(resumed_app) as client:
            frames = await stream(client, "/api/studio/interrupt", _answer(paused, _FULL_ID))
        assert frames[-1]["event"] == "done"
        carried = await _stored(pg_engine, _FULL_ID, "carried_history")
        await _turns(resumed_app, "Compare them", conversation_id=_FULL_ID)

    _assert_extends(model.requests[-1], model.requests[1])
    assert carried is not None
    assert carried["turn"] == 2
    assert len(_contents(resumed_model.requests[-1])) == 2
    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "full", "full", "split", 1, "extended"),
        (3, "digest", "full", "forced", 1, None),
    ]


@pytest.mark.parametrize(
    ("value", "read"), [("", "split"), (" Full", "full"), ("DIGEST", "digest"), ("split", "split")]
)
def test_the_history_setting_reads_blank_as_split_and_ignores_case(
    monkeypatch: pytest.MonkeyPatch, value: str, read: str
) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", value)
    assert HistorySettings().studio_history == read


@pytest.mark.parametrize(("value", "read"), [("", 0.5), ("0", 0.0), ("0.25", 0.25), ("1", 1.0)])
def test_the_full_share_reads_blank_as_half(
    monkeypatch: pytest.MonkeyPatch, value: str, read: float
) -> None:
    monkeypatch.setenv("STUDIO_HISTORY_FULL_SHARE", value)
    assert HistorySettings().studio_history_full_share == read


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("STUDIO_HISTORY", "frozen"),
        ("STUDIO_HISTORY_FULL_SHARE", "1.5"),
        ("STUDIO_HISTORY_FULL_SHARE", "-0.1"),
        ("STUDIO_HISTORY_FULL_SHARE", "nan"),
        ("STUDIO_HISTORY_FULL_SHARE", "half"),
    ],
)
def test_an_unknown_history_setting_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        HistorySettings()
