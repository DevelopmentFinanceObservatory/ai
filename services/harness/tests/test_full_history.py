"""The `full` history arm, where a follow-up turn's orchestrator opens on the
previous turn's conversation whole, and `STUDIO_HISTORY=split`, which gives each
conversation the arm a hash of its id assigns and stores.

Turn N's first request must extend turn N-1's last request byte for byte, the
only shape Gemini's implicit cache serves, so each test compares the
`contents` the model client was sent (`harness.gemini_wire` passes them on) as
google-genai JSON. The reload tests send a later turn through a second app on
the same database, so the carried conversation and the stored arm pass
through the conversation store first.

`_FULL_ID` and `_DIGEST_ID` are conversation ids `split` assigns the `full`
and the `digest` arm at the default share; `test_the_chosen_ids_hash_into_their_arms`
holds them to it. The split tests drive whole turns over the routes and read
the requests, the spans and the `studio.turn` lines, since a code path that
took the arm from the process setting, which reads `split`, would run both
conversations on `digest` and only whole turns show it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from common.logging import JSON_FIELDS
from harness.conversation import current_history
from harness.generative import GeminiEdit
from harness.graph.orchestrate import _carried_opening, carried_record
from harness.graph.state import ORCHESTRATOR_CARRIES_KEY, ORCHESTRATOR_CHAIN_RESET_KEY, CarriedRef
from harness.settings import HistorySettings
from harness.studio_state import (
    CARRIED_HISTORY_TOKEN_CAP,
    SEGMENT_TTL_S,
    ConversationState,
    TurnUsage,
    history_arm,
    make_carried_history,
)
from harness.studio_store import Renewal, SegmentClose, StudioStore, record_closed_turn
from harness.turn.node import TurnKey
from harness.turn.store import Cursor, TurnStatus
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from orchestrator.run import StoredHistory
from pydantic import ValidationError
from pydantic_ai import Conversation
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import Reply, Request, call_part, text_part, thought_part, usage_metadata
from .scripted_tools import CannedSearch, definite_envelope
from .store_clock import StoreClock
from .studio_routes import (
    CLIENT_FEATURES,
    CONVERSATION,
    MODEL,
    orchestrator_app,
    stream,
    studio_client,
    turn_body,
)
from .test_generative import _FakeClient
from .test_studio import (
    COMPONENT,
    WRITE_TOKEN,
    ScriptedExecutor,
    _card,
    _card_results,
    _described,
    _edit_body,
    _search,
)
from .test_studio_apply_unit import (
    _CONSTANT_2023,
    _ENVELOPE,
    _SPEC,
    _YEARS,
    _apply_body,
    _change_body,
    _Changer,
    _first_results,
    _first_turn,
    _Places,
    _Search,
)
from .test_studio_apply_unit import _UGANDA as _UGANDA_CHANGE
from .test_studio_resume import _leave_once, _resume_body

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_RESET = "studio history: chain reset"
# The principal the route tests' bearer token signs in as.
_PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
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


# A follow-up's answer, which reports its usage as every Gemini call does.
_UGANDA = Reply(
    chunks=[[text_part("Uganda's was 63.")]], usage=usage_metadata(prompt=500, candidates=10)
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


# A carried conversation's copies a cursor could hold: the text in its inputs,
# and in its state the conversation it closed on or the one a pause waits with.
_COPY_KEYS = ("carried_history", "orchestrator:carried", "orchestrator:history")


async def _cursors(engine: AsyncEngine) -> list[tuple[int, str, str, str, str]]:
    """Each cursor's turn, status, inputs and state as stored text, and its
    row version stamp (xmin), by turn."""
    async with engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT turn_index, status, CAST(inputs AS text), CAST(state AS text), "
                "xmin::text FROM turn_cursors ORDER BY turn_index"
            )
        )
        return [tuple(row) for row in rows]


def _copies(cursor: tuple[int, str, str, str, str]) -> list[str]:
    """The carried-conversation keys `cursor`'s inputs or state hold."""
    _, _, inputs, state, _ = cursor
    return [key for key in _COPY_KEYS if f'"{key}"' in inputs or f'"{key}"' in state]


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
    # The conversation's row holds the one copy of the chain. A follow-up's
    # inputs name the turn it extends, and no cursor holds a copy.
    assert (await _stored(pg_engine, CONVERSATION, "carried_history"))["turn"] == 3
    cursors = await _cursors(pg_engine)
    assert [
        json.loads(inputs).get("carried_from", {}).get("turn") for _, _, inputs, _, _ in cursors
    ] == [
        None,
        1,
        2,
    ]
    assert [_copies(cursor) for cursor in cursors] == [[], [], []]


@pytest.mark.usefixtures("full_history")
async def test_no_closed_cursor_holds_a_carried_conversation_and_none_is_rewritten(
    pg_engine: AsyncEngine,
) -> None:
    """Three full turns, then a fourth left paused on a question, whose
    cursor holds the conversation its run waits with, and a fifth that
    abandons it. Every cursor closes holding no copy of a carried
    conversation, and the fifth turn's claim writes no earlier turn's row
    but the one it closes."""
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    app, _ = orchestrator_app(
        *_TURN_1,
        _UGANDA,
        text_part("Both rose."),
        asking,
        text_part("Done."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )
    await _turns(app, "Life expectancy in Kenya?", "And Uganda?", "Compare them")
    async with studio_client(app) as client:
        paused = await stream(client, "/api/studio/turn", turn_body("And Rwanda?"))
    assert paused[-1]["event"] == "elicitation.request"
    before = await _cursors(pg_engine)
    assert _copies(before[3]) == ["orchestrator:history"]

    await _turns(app, "Something else")

    after = await _cursors(pg_engine)
    assert [(turn, status) for turn, status, *_ in after] == [
        (1, "done"),
        (2, "done"),
        (3, "done"),
        (4, "abandoned"),
        (5, "done"),
    ]
    assert [_copies(cursor) for cursor in after] == [[]] * 5
    assert after[:3] == before[:3]


_REFUSED = Reply(open_error=ValueError("refused"))


@pytest.mark.usefixtures("full_history")
@pytest.mark.parametrize(
    ("first_call", "reload"),
    [
        pytest.param("answers", False, id="answers"),
        # The node's state is dropped with the failed call, so only the turn
        # the reset was recorded on carries it to the line.
        pytest.param("fails", False, id="first-call-fails"),
        # Another request reloads the conversation before the turn closes.
        pytest.param("fails", True, id="reloaded-then-fails"),
    ],
)
async def test_a_follow_up_whose_carried_conversation_changed_since_its_claim_opens_from_the_digest(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    first_call: str,
    reload: bool,
) -> None:
    """The claim names the stored conversation by its hash. One the
    orchestrator reads under another hash, as one stored after the claim,
    resets the chain `unloadable` and the turn opens from the digest. Its
    line logs the reset, with no carried tokens, whether its first call
    answers or fails."""
    store = StudioStore(pg_engine)

    async def reload_then_refuse(_request: Request) -> Reply:
        await store.load(_PRINCIPAL, CONVERSATION)
        return _REFUSED

    follow_up = {"answers": _UGANDA, "fails": reload_then_refuse if reload else _REFUSED}
    app, model = orchestrator_app(
        *_TURN_1,
        follow_up[first_call],
        resolver=CannedSearch(definite_envelope()),
        store=store,
    )
    await _turns(app, "Life expectancy in Kenya?")

    class _Changed(CarriedRef):
        @classmethod
        def of(cls, *, turn: int, history: str) -> CarriedRef:
            return CarriedRef(turn=turn, sha256=hashlib.sha256(b"another").hexdigest())

    monkeypatch.setattr("harness.studio.CarriedRef", _Changed)
    # Only this turn's lines: an earlier test can leave the logger at INFO.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "And Uganda?")

    request = model.requests[-1]
    assert len(_contents(request)) == 2
    assert '"turns": [' in _opening_texts(request, 2)[0]
    assert [
        (line["history_chain"], line.get("carried_tokens")) for line in _turn_lines(caplog)
    ] == [("reset:unloadable", None)]


@pytest.mark.usefixtures("full_history")
async def test_a_follow_up_that_pauses_as_it_opens_logs_the_chain_reset_its_orchestrator_met(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up whose orchestrator finds the carried conversation changed
    since the claim, and asks the user at once, writes its paused line with
    the reset, `unloadable`, where the claim's check said `extended`."""
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    app, _ = orchestrator_app(
        *_TURN_1, asking, resolver=CannedSearch(definite_envelope()), engine=pg_engine
    )
    await _turns(app, "Life expectancy in Kenya?")

    class _Changed(CarriedRef):
        @classmethod
        def of(cls, *, turn: int, history: str) -> CarriedRef:
            return CarriedRef(turn=turn, sha256=hashlib.sha256(b"another").hexdigest())

    monkeypatch.setattr("harness.studio.CarriedRef", _Changed)
    # Only this turn's lines: an earlier test can leave the logger at INFO.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(app) as client:
            paused = await stream(client, "/api/studio/turn", turn_body("And Uganda?"))

    assert paused[-1]["event"] == "elicitation.request"
    (line,) = [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.turn"
    ]
    assert (line["status"], line["history_chain"], line["carried_tokens"]) == (
        "paused",
        "reset:unloadable",
        None,
    )


@pytest.mark.usefixtures("full_history")
@pytest.mark.parametrize("cut", ["acknowledgement-lost", "lease-lost"])
async def test_a_turn_cut_after_its_done_write_committed_still_carries_its_conversation(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, cut: str
) -> None:
    """The write that ends turn 1 `done` commits, and then its
    acknowledgement is lost (`_after_failed_write` reads the turn back) or
    a lost lease cuts the segment (`_settle` reads it back). Either read back
    hands on the turn's conversation, and turn 2 extends it."""
    app, model = orchestrator_app(
        *_TURN_1,
        _UGANDA,
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
        heartbeat_seconds=0.05,
    )
    turns, store = app.state.studio.turns, app.state.studio.store
    real_finish = turns.finish
    cuts: list[str] = []

    async def finish_then_cut(cursor: Any, **kwargs: Any) -> Any:
        finished = await real_finish(cursor, **kwargs)
        if kwargs.get("status") is TurnStatus.DONE and not cuts:
            cuts.append(cut)
            if cut == "acknowledgement-lost":
                raise ConnectionResetError("the COMMIT's acknowledgement was lost")
            # The next renewal finds the lease lost, which cuts the segment
            # here.
            real_renew = store.renew

            async def lost(*args: Any, **kwargs: Any) -> Renewal:
                store.renew = real_renew
                return Renewal(held=False, stop_id=None)

            store.renew = lost
            await asyncio.sleep(5)
            pytest.fail("the lost lease did not cut the segment")
        return finished

    turns.finish = finish_then_cut
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", "And Uganda?")

    assert cuts == [cut]
    _assert_extends(model.requests[-1], model.requests[1])
    assert [(line["turn"], line["history_chain"]) for line in _turn_lines(caplog)] == [
        (1, "fresh"),
        (2, "extended"),
    ]


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
async def test_a_follow_up_to_a_turn_whose_closing_write_failed_resets_as_close_failed(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Turn 1 answers, and the write that would store its record and its
    carry fails. Once its lease lapses, turn 2's claim records turn 1
    `state_not_saved`, and turn 2 opens from the digest, its chain reset
    `close_failed`."""
    clock = await StoreClock.start(pg_engine)
    store = StudioStore(pg_engine, **clock.store_kwargs)
    app, model = orchestrator_app(
        *_TURN_1, _UGANDA, resolver=CannedSearch(definite_envelope()), store=store
    )
    closing = store.close_segment

    async def close_failed(*args: Any, **kwargs: Any) -> SegmentClose:
        return SegmentClose.FAILED

    store.close_segment = close_failed  # type: ignore[method-assign]
    await _turns(app, "Life expectancy in Kenya?")
    store.close_segment = closing  # type: ignore[method-assign]
    await clock.advance(SEGMENT_TTL_S + 1)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "And Uganda?")

    request = model.requests[-1]
    assert len(_contents(request)) == 2
    assert '"turns": [' in _opening_texts(request, 2)[0]
    assert [
        (line["turn"], line["error"], line["history_chain"]) for line in _turn_lines(caplog)
    ] == [(1, "state_not_saved", "fresh"), (2, None, "reset:close_failed")]


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
        pytest.param(
            # A turn whose calls reported no usage, as a bound's closing
            # sentence the harness wrote after no call completed.
            (Reply(chunks=[[text_part("62 years.")]], usage=None),),
            "no_usage",
            id="no-usage",
        ),
    ],
)
async def test_a_turn_with_nothing_to_carry_starts_a_fresh_chain_from_the_digest(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    turn_1: tuple[Reply, ...],
    reason: str,
) -> None:
    app, model = orchestrator_app(*turn_1, _UGANDA, text_part("Both rose."), engine=pg_engine)

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
    id), over 2**64, fall below the share. Over 2,000 ids the `full` count is
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


def test_begin_turn_reads_no_environment_and_runs_digest_when_named_no_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUDIO_HISTORY", "full")
    state = ConversationState(conversation_id=_FULL_ID)

    state.begin_turn()

    assert _history(state.turn_usage) == (1, "digest", None, "forced", None)


async def test_a_turn_begins_under_the_setting_the_app_read_at_startup(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The app reads `STUDIO_HISTORY` once, as it is built. A later change
    of the environment reaches no turn."""
    monkeypatch.setenv("STUDIO_HISTORY", " Split ")
    monkeypatch.setenv("STUDIO_HISTORY_FULL_SHARE", "1")
    app, _ = orchestrator_app(engine=pg_engine)
    monkeypatch.setenv("STUDIO_HISTORY", "digest")

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", conversation_id=_DIGEST_ID)

    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
    ]


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
    same database. The arm stored on turn 1 decides (a share of 0
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
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Under a rollback to `STUDIO_HISTORY=digest`, a conversation assigned `full`
    opens from the digest, and its stored arm stays. Back under `split` it
    is `full` again, on a fresh chain, since the digest turn carried none.
    Each setting runs on an app of its own, as a restart reads it."""
    app, _ = orchestrator_app(
        *_TURN_1, resolver=CannedSearch(definite_envelope()), engine=pg_engine, history=_split()
    )
    rolled_back, rolled_back_model = orchestrator_app(
        _UGANDA, engine=pg_engine, history=HistorySettings(studio_history="digest")
    )
    restored, restored_model = orchestrator_app(
        text_part("Both rose."), engine=pg_engine, history=_split()
    )
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        await _turns(rolled_back, "And Uganda?", conversation_id=_FULL_ID)
        assert await _stored(pg_engine, _FULL_ID, "history_assignment") == {
            "arm": "full",
            "turn": 1,
        }
        await _turns(restored, "Compare them", conversation_id=_FULL_ID)

    (second,) = rolled_back_model.requests
    assert len(_contents(second)) == 2
    assert '"turns": [' in _opening_texts(second, 2)[0]
    assert len(_contents(restored_model.requests[-1])) == 2
    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "digest", "full", "forced", 1, None),
        (3, "full", "full", "split", 1, "reset:previous_turn_carried_none"),
    ]


def _card_change_app(engine: AsyncEngine, body: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """A first turn that draws card `cmp_01unit0001`, a card change `body` on
    it answered by a `_Changer`, and a third turn that answers in text. The
    card offers its places."""
    *calls, answer = _first_turn()
    app, model = orchestrator_app(
        *calls,
        Reply(chunks=[[answer]], usage=usage_metadata(prompt=9_000, candidates=40)),
        _UGANDA,
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_Search(),
        engine=engine,
        history=_split(),
    )
    app.state.studio.place_options = _Places()
    return app, model, {**body, "conversation_id": _FULL_ID}, _Changer()


def _edit_card_app(engine: AsyncEngine) -> tuple[Any, Any, Any, Any]:
    """A first turn that draws card `cmp_01test0001`, an Edit Card on it whose
    model renames it, and a third turn that answers in text."""
    revised = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
    }
    decision = {
        "text": "Renamed.",
        "patch_json": json.dumps({"title": "Kenya"}),
        "recreate_json": None,
    }
    *calls, answer = _card()
    app, model = orchestrator_app(
        *calls,
        Reply(chunks=[[answer]], usage=usage_metadata(prompt=9_000, candidates=40)),
        _UGANDA,
        resolver=_search(),
        executor=ScriptedExecutor(
            [
                *_card_results(write_token=WRITE_TOKEN),
                _described(),
                {"outcome": "ok", "frame_refs": [], "component": revised},
            ]
        ),
        edit_model=GeminiEdit(
            model_name="gemini-edit-test",
            timeout_s=30,
            client=_FakeClient(json.dumps(decision), usage_metadata(prompt=600, candidates=6)),
        ),
        engine=engine,
        history=_split(),
    )
    return app, model, {**_edit_body("rename it"), "conversation_id": _FULL_ID}, None


@pytest.mark.parametrize(
    ("build", "roles", "reason", "place"),
    [
        pytest.param(
            lambda engine: _card_change_app(engine, _change_body("cmp_01unit0001", _SPEC, _YEARS)),
            [],
            "previous_turn_card_change",
            None,
            id="apply-change-years",
        ),
        pytest.param(
            lambda engine: _card_change_app(
                engine, _change_body("cmp_01unit0001", _SPEC, _UGANDA_CHANGE)
            ),
            [],
            "previous_turn_card_change",
            "country/UGA",
            id="apply-change-places",
        ),
        pytest.param(
            lambda engine: _card_change_app(
                engine, _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
            ),
            [],
            "previous_turn_card_change",
            None,
            id="apply-unit",
        ),
        pytest.param(_edit_card_app, ["edit"], "previous_turn_carried_none", None, id="edit-card"),
    ],
)
async def test_a_question_after_a_card_change_or_an_edit_card_on_the_full_arm_opens_from_the_digest(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    build: Any,
    roles: list[str],
    reason: str,
    place: str | None,
) -> None:
    """A card change in place and an Edit Card run no orchestrator, so they
    carry no conversation: the question after one opens a fresh chain from
    the whole digest, `previous_turn_card_change` after a card change and
    `previous_turn_carried_none` after an Edit Card, and the arm the first
    turn assigned stays stored. Only the Edit Card's model call writes a
    `studio.model_call` line for the middle turn. After a change of places,
    the ledger the next turn's planner pins from offers the retrieval the
    changed card drew, over its new `place`."""
    app, model, middle, changer = build(pg_engine)
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(app, "Aid to Kenya", conversation_id=_FULL_ID)
        assert (await _stored(pg_engine, _FULL_ID, "carried_history"))["tokens"] == 9_040
        if changer is not None:
            app.state.studio.recorder._executor = changer
        async with studio_client(app) as client:
            frames = await stream(client, "/api/studio/turn", middle)
        assert frames[-1]["event"] == "done"
        assert await _stored(pg_engine, _FULL_ID, "carried_history") is None
        await _turns(app, "And Uganda?", conversation_id=_FULL_ID)

    request = model.requests[-1]
    assert len(_contents(request)) == 2
    assert '"turns": [' in _opening_texts(request, 2)[0]
    lines = _turn_lines(caplog)
    assert [_history_fields(line) for line in lines] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "full", "full", "split", 1, None),
        (3, "full", "full", "split", 1, f"reset:{reason}"),
    ]
    assert await _stored(pg_engine, _FULL_ID, "history_assignment") == {"arm": "full", "turn": 1}
    calls = [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.model_call"
    ]
    assert [call["role"] for call in calls if call["trace_id"] == lines[1]["trace_id"]] == roles
    if place is not None:
        conversation = app.state.studio.store.current(_PRINCIPAL, _FULL_ID)
        ledger = conversation.prior_ledger_state()
        assert ledger is not None
        newest = ledger["readings"][-1]["spec_id"]
        assert newest != _SPEC
        recorded = conversation.recorded_identity(newest)
        assert recorded is not None
        assert recorded.identity["entities"]["financingRecipient"] == [place]


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
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exported_spans: InMemorySpanExporter,
) -> None:
    """A `full` turn paused on a question and answered on a second app started
    with `STUDIO_HISTORY=digest` keeps the arm it began with: its run stores
    its conversation, and its line and the resumed segment's model spans say
    `full` and `split`. The turn after it begins under `digest`."""
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
        # The in-process transport runs each request in the test's context,
        # so the arm `/turn` set is cleared, as a request of its own starts.
        current_history.set(None)
        exported_spans.clear()
        async with studio_client(resumed_app) as client:
            frames = await stream(client, "/api/studio/interrupt", _answer(paused, _FULL_ID))
        assert frames[-1]["event"] == "done"
        resumed_spans = {
            (
                (span.attributes or {}).get("studio.surface.history"),
                (span.attributes or {}).get("studio.history.mode"),
            )
            for span in exported_spans.get_finished_spans()
            if span.name.startswith("chat ")
        }
        carried = await _stored(pg_engine, _FULL_ID, "carried_history")
        await _turns(resumed_app, "Compare them", conversation_id=_FULL_ID)

    _assert_extends(model.requests[-1], model.requests[1])
    assert resumed_spans == {("full", "split")}
    assert carried is not None
    assert carried["turn"] == 2
    assert len(_contents(resumed_model.requests[-1])) == 2
    assert [_history_fields(line) for line in _turn_lines(caplog)] == [
        (1, "full", "full", "split", 1, "fresh"),
        (2, "full", "full", "split", 1, "extended"),
        (3, "digest", "full", "forced", 1, None),
    ]


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        # The instance holds no state for the conversation.
        pytest.param(None, "not_in_process", id="not-in-process"),
        # It holds the conversation, which stores no carried conversation.
        pytest.param({}, "unloadable", id="none-stored"),
    ],
)
def test_an_orchestrator_that_cannot_read_the_named_conversation_records_why(
    stored: dict[str, Any] | None, reason: str
) -> None:
    key = TurnKey(principal=_PRINCIPAL, conversation_id=_FULL_ID, turn_index=2)
    ctx = SimpleNamespace(key=key, state={})
    ref = CarriedRef.of(turn=1, history="{}")

    resets: list[tuple[TurnKey, str]] = []
    source = SimpleNamespace(
        history=lambda _key: stored, reset=lambda key, why: resets.append((key, why))
    )

    opened = _carried_opening(ctx, ref, source, model_name=MODEL)  # type: ignore[arg-type]

    assert opened is None
    assert ctx.state == {ORCHESTRATOR_CHAIN_RESET_KEY: reason}
    assert resets == [(key, reason)]


async def test_the_carried_source_tells_a_conversation_out_of_process_from_one_storing_none(
    pg_engine: AsyncEngine,
) -> None:
    store = StudioStore(pg_engine)
    source = make_carried_history(store)
    key = TurnKey(principal=_PRINCIPAL, conversation_id=_FULL_ID, turn_index=1)
    assert source.history(key) is None

    await store.create(_PRINCIPAL, _FULL_ID)

    assert source.history(key) == {}


def _response(input_tokens: int, output_tokens: int) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("x")],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.parametrize(
    ("responses", "tokens"),
    [
        # A final response the harness wrote, a bound's closing sentence,
        # reports nothing. The call before it sizes the conversation.
        pytest.param([(9_000, 40), (12_000, 60), (0, 0)], 12_060, id="harness-final"),
        pytest.param([(0, 0)], None, id="none-reported"),
    ],
)
def test_the_carried_tokens_are_the_largest_call_s_and_none_when_no_call_reported_any(
    responses: list[tuple[int, int]], tokens: int | None
) -> None:
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Kenya?")])]
    messages += [_response(*usage) for usage in responses]
    history = StoredHistory(conversation=Conversation(messages=messages))

    record = carried_record(history, model_name="gemini-test")

    assert (record["tokens"], record["model"]) == (tokens, "gemini-test")


async def _edit_carried(engine: AsyncEngine, conversation_id: str, edit: dict[str, Any]) -> None:
    """Apply `edit` to the carried conversation stored for `conversation_id`."""
    async with engine.begin() as connection:
        state = (
            await connection.execute(
                text("SELECT turn_state FROM conversations WHERE conversation_id = :id"),
                {"id": conversation_id},
            )
        ).scalar_one()
        state["carried_history"] |= edit
        await connection.execute(
            text(
                "UPDATE conversations SET turn_state = CAST(:state AS jsonb) "
                "WHERE conversation_id = :id"
            ),
            {"id": conversation_id, "state": json.dumps(state)},
        )


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        # A deploy that changed the message schema.
        pytest.param({"history": '{"conversation": 5}'}, "unloadable", id="unloadable"),
        # Another orchestrator model, whose thought signatures are its own.
        pytest.param({"model": "gemini-older"}, "model_changed", id="model-changed"),
    ],
)
async def test_a_carried_conversation_the_turn_cannot_extend_opens_from_the_digest(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    edit: dict[str, Any],
    reason: str,
) -> None:
    full = HistorySettings(studio_history="full")
    app, _ = orchestrator_app(
        *_TURN_1, resolver=CannedSearch(definite_envelope()), engine=pg_engine, history=full
    )
    await _turns(app, "Life expectancy in Kenya?")
    assert (await _stored(pg_engine, CONVERSATION, "carried_history"))["model"] == MODEL
    await _edit_carried(pg_engine, CONVERSATION, edit)
    reloaded, model = orchestrator_app(_UGANDA, engine=pg_engine, history=full)

    # Only this turn's lines: an earlier test can leave the logger at INFO.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(reloaded) as client:
            frames = await stream(client, "/api/studio/turn", turn_body("And Uganda?"))

    assert frames[-1]["event"] == "done"
    (request,) = model.requests
    assert len(_contents(request)) == 2
    assert '"turns": [' in _opening_texts(request, 2)[0]
    assert [line["history_chain"] for line in _turn_lines(caplog)] == [f"reset:{reason}"]


async def test_a_turn_resumed_on_an_instance_that_carried_an_earlier_turn_carries_its_own(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Turn 1 runs on instance A, which keeps the conversation it carried in
    process. Turn 2 begins on instance B under a forced `digest`, so its run
    carries nothing, pauses on a question, and is answered on A. Turn 2's
    record carries nothing, so turn 3 on A opens a fresh chain from the
    digest, where A's earlier conversation would extend turn 1 and skip
    turn 2."""
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    first, first_model = orchestrator_app(
        *_TURN_1,
        _UGANDA,
        text_part("Both rose."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
        history=HistorySettings(studio_history="full"),
    )
    second, _ = orchestrator_app(
        asking, engine=pg_engine, history=HistorySettings(studio_history="digest")
    )
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(first, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        async with studio_client(second) as client:
            paused = await stream(
                client, "/api/studio/turn", turn_body("And Uganda?", conversation_id=_FULL_ID)
            )
        assert paused[-1]["event"] == "elicitation.request"
        async with studio_client(first) as client:
            frames = await stream(client, "/api/studio/interrupt", _answer(paused, _FULL_ID))
        assert frames[-1]["event"] == "done"
        assert await _stored(pg_engine, _FULL_ID, "carried_history") is None
        await _turns(first, "Compare them", conversation_id=_FULL_ID)

    assert len(_contents(first_model.requests[-1])) == 2
    assert [
        (line["turn"], line["history"], line["history_chain"]) for line in _turn_lines(caplog)
    ] == [
        (1, "full", "fresh"),
        (2, "digest", None),
        (3, "full", "reset:previous_turn_carried_none"),
    ]


async def test_the_turn_after_a_full_turn_resumed_on_another_instance_extends_its_last_request(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Turn 2 runs on instance A under `full`, pauses on a question, and is
    answered on instance B, whose run closes on the reply. Turn 3 on A
    opens on the conversation B's run carried: its first request extends
    B's last byte for byte."""
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    full = HistorySettings(studio_history="full")
    first, first_model = orchestrator_app(
        *_TURN_1,
        asking,
        text_part("Both rose."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
        history=full,
    )
    second, second_model = orchestrator_app(_UGANDA, engine=pg_engine, history=full)
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _turns(first, "Life expectancy in Kenya?", conversation_id=_FULL_ID)
        async with studio_client(first) as client:
            paused = await stream(
                client, "/api/studio/turn", turn_body("And Uganda?", conversation_id=_FULL_ID)
            )
        assert paused[-1]["event"] == "elicitation.request"
        async with studio_client(second) as client:
            frames = await stream(client, "/api/studio/interrupt", _answer(paused, _FULL_ID))
        assert frames[-1]["event"] == "done"
        await _turns(first, "Compare them", conversation_id=_FULL_ID)

    (resumed,) = second_model.requests
    _assert_extends(first_model.requests[-1], resumed)
    assert [
        (line["turn"], line["history"], line["history_chain"]) for line in _turn_lines(caplog)
    ] == [(1, "full", "fresh"), (2, "full", "extended"), (3, "full", "extended")]


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


_CARD_CHANGE_TURN = {"card_change": {"component_id": "cmp_01unit0001", "kind": "window"}}


@pytest.mark.parametrize(
    ("turns", "reason"),
    [
        pytest.param([{}, _CARD_CHANGE_TURN], "previous_turn_card_change", id="card-change"),
        pytest.param(
            [{}, {**_CARD_CHANGE_TURN, "error": "change_failed"}],
            "previous_turn_card_change",
            id="card-change-that-failed",
        ),
        pytest.param(
            [_CARD_CHANGE_TURN, {}], "previous_turn_carried_none", id="card-change-then-another"
        ),
    ],
)
def test_only_the_turn_right_after_a_card_change_names_it_as_its_reset(
    turns: list[dict[str, Any]], reason: str
) -> None:
    """A turn that follows a card change and carries no conversation resets
    with `previous_turn_card_change`, whether the change succeeded or closed
    with an error. The reason comes from the previous turn's record only. When
    another turn whose record holds no card change mark follows the change
    and carries nothing, the turn after it resets with
    `previous_turn_carried_none`."""
    conversation = ConversationState(conversation_id="conv-card-change")
    conversation.turns = [
        {"turn": number, "user": "", "assistant": "", **record}
        for number, record in enumerate(turns, start=1)
    ]

    assert conversation.carried_opening(model_name="gemini-test") is None
    assert conversation.turn_usage.history_chain == f"reset:{reason}"


@pytest.mark.parametrize(("carried", "marked"), [(True, True), (False, False)])
def test_only_a_turn_that_answered_with_a_carry_is_recorded_as_close_failed(
    carried: bool, marked: bool
) -> None:
    """`record_closed_turn` records a turn whose closing write failed. Only a
    turn whose orchestrator answered with a conversation to carry
    (`ORCHESTRATOR_CARRIES_KEY`) leaves the `close_failed` marker. A `done`
    turn that carried nothing, such as one whose plan card was set aside,
    leaves none, so the next turn reads `previous_turn_carried_none`."""
    conversation = ConversationState(conversation_id="conv-close")
    conversation.begin_turn(history=HistorySettings(studio_history="full"))
    cursor = Cursor(
        id=1,
        conversation=1,
        turn_index=1,
        node="orchestrate",
        node_input=None,
        status=TurnStatus.DONE,
        round=0,
        pause=None,
        owed=None,
        inputs={},
        state={ORCHESTRATOR_CARRIES_KEY: True} if carried else {},
        error=None,
        version=1,
        record_count=0,
        spec_digests={},
    )

    record_closed_turn(conversation, cursor)

    assert (conversation.carried_history == {"turn": 1, "reset": "close_failed"}) is marked


@pytest.mark.parametrize("first_call", ["answers", "fails"])
async def test_a_follow_up_resumed_on_an_instance_with_another_model_opens_from_the_digest(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, first_call: str
) -> None:
    """Turn 2 is claimed on instance A, whose model wrote turn 1's
    conversation, so the claim names it, and its client leaves during the
    first call. `/resume` runs the turn on instance B, whose model differs:
    B's orchestrator opens from the whole digest, since the conversation's
    thought signatures are A's model's, and the turn's line logs
    `model_changed` with no carried tokens, whether B's first call answers
    or fails."""
    entered = asyncio.Event()

    async def hang(_request: Request) -> Reply:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    full = HistorySettings(studio_history="full")
    first, _ = orchestrator_app(
        *_TURN_1, hang, resolver=CannedSearch(definite_envelope()), engine=pg_engine, history=full
    )
    second, second_model = orchestrator_app(
        _UGANDA if first_call == "answers" else _REFUSED,
        engine=pg_engine,
        history=full,
        model_name="gemini-other",
    )
    await _turns(first, "Life expectancy in Kenya?")
    await _leave_once(first, "/api/studio/turn", turn_body("And Uganda?"), entered)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(second) as client:
            frames = await stream(client, "/api/studio/resume", _resume_body(2))

    assert frames[-1]["event"] == ("done" if first_call == "answers" else "error")
    (request,) = second_model.requests
    assert len(_contents(request)) == 2
    assert '"turns": [' in _opening_texts(request, 2)[0]
    assert [
        (line["turn"], line["history_chain"], line.get("carried_tokens"))
        for line in _turn_lines(caplog)
    ] == [(2, "reset:model_changed", None)]
