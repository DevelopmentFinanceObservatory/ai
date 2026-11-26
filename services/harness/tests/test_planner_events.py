"""The records the planner stores equal the conversation's stored expectation.

Each conversation runs on one script of a scripted google-genai client, through
`harness.planner_adapter.PlannerAdapter` as `tests.test_planner_node` runs it. The
adapter serves the real `ask_user_question` and `get_skill`, and a `propose_plan`
whose answers the script sets. Each stored turn is then checked against the
conversation's stored expectation (`tests.planner_expectations`) on what its
readers read: what `session_steps` reads from the turn's transcript, and its
state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from compiler.surface import ToolCallResult, ToolSurface
from google.genai import types
from harness.planner_adapter import PlannerAdapter
from harness.scripted_plan import scripted_plan
from harness.settings import Settings
from harness.turn import transcript
from harness.turn.store import AgentRecord
from harness.turn_events import PLANNER_ASK
from planner.call import PlannerCall, Step
from planner.plan import PLAN_STATE_KEY
from planner.session import ASKED_CALL_ID, PLANNER_ASK_PAUSE, session_steps
from planner.skills import SKILLS_READ_STATE_KEY
from planner.specs import PROPOSE_PLAN, planner_tool_specs, skill_tool_spec
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import ASK_ACCEPTED, EndRun

from .planner_expectations import expect, expectation_name
from .planner_host import PlannerHarness, planner_calls
from .scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)
from .session_events import (
    model_response,
    one_answer,
    tool_answers,
    tool_call,
    turn_pause,
    user_answer,
)
from .test_planner_node import _steps

_QUESTION = "How much aid did Ghana receive?"
_USAGE = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)
_ACCEPTED = {"outcome": "ok", "status": "ready", "summary": "Aid to Ghana."}
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}
_ROUNDS_KEY = "planner:plan_rounds"


# --- The tools the planner serves ----------------------------------------------------


@dataclass
class _Skills:
    """A tool surface serving `get_skill`: each page's text, after its delay. It
    lists no tools, so a translation the adapter runs finds no schema."""

    delays: Mapping[str, float]

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del tool, on_progress
        skill = str(arguments["skill_id"])
        await asyncio.sleep(self.delays.get(skill, 0.0))
        return ToolCallResult(is_error=False, text=f"The {skill} page.")


def _propose(*answers: dict[str, Any]) -> ToolSpec:
    """A `propose_plan` answering `answers` in turn, counting its rounds in state, recording
    the plan and ending the run on an `ok` answer."""
    left = list(answers)

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del arguments
        call.state[_ROUNDS_KEY] = call.state.get(_ROUNDS_KEY, 0) + 1
        answer = left.pop(0)
        if answer["outcome"] == "ok":
            call.state[PLAN_STATE_KEY] = scripted_plan(_QUESTION, spec_id=None).model_dump_json()
            end_run()
        return answer

    return ToolSpec(
        name=PROPOSE_PLAN,
        description="Propose the plan.",
        parameters=(Parameter("question", str | None, None),),
        body=body,
    )


def _stuck(started: asyncio.Event) -> ToolSpec:
    """A tool that writes to state, sets `started` and then waits forever."""

    async def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run, arguments
        call.state["stuck"] = "started"
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return ToolSpec(name="stuck", description="Hangs.", parameters=(), body=body)


def _specs(
    surface: ToolSurface, *proposals: dict[str, Any], stuck: asyncio.Event | None = None
) -> list[ToolSpec]:
    ask = next(spec for spec in planner_tool_specs() if spec.name == PLANNER_ASK)
    return [
        skill_tool_spec(surface),
        ask,
        _propose(*proposals),
        *([_stuck(stuck)] if stuck is not None else []),
    ]


# --- One conversation -------------------------------------------------


@dataclass
class _Stored:
    """The turn an entry left, as the turn loop stores it."""

    records: list[AgentRecord]
    state: dict[str, Any]


async def _conversation(adapter: PlannerAdapter, *, stop: asyncio.Event | None = None) -> _Stored:
    """Run one planner entry of `adapter` in a turn of its own, and read the
    stored turn. With `stop`, the entry is cancelled once it is set, as the
    hanging tool sets it."""
    harness = PlannerHarness()
    entry = harness.run(adapter, message=_QUESTION, question=_QUESTION, readings=[])
    if stop is not None:
        running = asyncio.create_task(entry)
        await asyncio.wait_for(stop.wait(), timeout=5)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    else:
        await entry
    return _Stored(records=await harness.records(), state=dict(await harness.state()))


async def _on(
    replies: Sequence[Reply],
    *proposals: dict[str, Any],
    delays: Mapping[str, float] | None = None,
    stop: bool = False,
) -> _Stored:
    """The entry on `replies` through `PlannerAdapter`, serving `_specs`."""
    scripted = ScriptedGenai(replies)
    surface = cast(ToolSurface, _Skills(delays or {}))
    started = asyncio.Event() if stop else None
    adapter = PlannerAdapter(
        settings=Settings(),
        surface=surface,
        specs=_specs(surface, *proposals, stuck=started),
        client=scripted.client,
    )
    return await _conversation(adapter, stop=started)


# --- What is compared ------------------------------------------------------------------


async def _assert_expected(request: pytest.FixtureRequest, stored: _Stored) -> None:
    """Assert that `stored` equals the stored expectation of the test `request` runs.
    The expectation's `events`, the session's own, and `frames`, Studio's of them,
    are not read."""
    expect(
        expectation_name(request),
        {"steps": _steps(stored.records), "state": stored.state},
        unread=("events", "frames"),
    )


def _planner_records(stored: _Stored) -> list[AgentRecord]:
    return [record for record in stored.records if record.agent == "planner"]


# --- The conversations ----------------------------------------------------------------


def _ask(id: str) -> types.Part:
    return call_part(PLANNER_ASK, {"message": "Which donor do you mean?"}, id=id)


def _proposal(id: str, **args: Any) -> types.Part:
    return call_part(PROPOSE_PLAN, args, id=id)


_CONVERSATIONS: dict[str, tuple[list[Reply], tuple[dict[str, Any], ...]]] = {
    "accepted-proposal": ([reply(_proposal("p1", question="q"), usage=_USAGE)], (_ACCEPTED,)),
    "refused-then-accepted": (
        [reply(_proposal("p1"), usage=_USAGE), reply(_proposal("p2", question="q"))],
        (_REFUSED, _ACCEPTED),
    ),
    "accepted-ask": ([reply(_ask("a1"), usage=_USAGE)], ()),
    "skill-then-proposal": (
        [
            reply(call_part("get_skill", {"skill_id": "oda-dac2a"}, id="s1"), usage=_USAGE),
            reply(_proposal("p1", question="q"), usage=_USAGE),
        ],
        (_ACCEPTED,),
    ),
    "ask-beside-proposal": ([reply(_ask("a1"), _proposal("p1", question="q"))], (_ACCEPTED,)),
    "text-and-no-call": ([reply(text_part("No plan answers this."), usage=_USAGE)], ()),
    "thought-parts": (
        [
            reply(
                thought_part("Unsigned summary.", signature=None),
                text_part("Proposing."),
                thought_part("Signed.", signature=b"sig-1"),
                _proposal("p1", question="q"),
                usage=_USAGE,
            )
        ],
        (_ACCEPTED,),
    ),
}


@pytest.mark.parametrize("conversation", list(_CONVERSATIONS))
async def test_a_conversation_stores_the_session_its_readers_read(
    conversation: str, request: pytest.FixtureRequest
) -> None:
    replies, proposals = _CONVERSATIONS[conversation]

    stored = await _on(replies, *proposals)

    assert _planner_records(stored), "the planner stored no record"
    await _assert_expected(request, stored)


async def test_a_thought_only_response_is_stored_as_one_event(
    request: pytest.FixtureRequest,
) -> None:
    stored = await _on([reply(thought_part("Only thinking.", signature=None))])

    await _assert_expected(request, stored)
    assert len(_planner_records(stored)) == 1


async def test_parallel_writes_to_one_list_store_the_value_the_last_call_left(
    request: pytest.FixtureRequest,
) -> None:
    """Two skill reads in one response each append their page to the list of pages
    read. The first call answers last, so the list it wrote holds both pages,
    which the session stores."""
    stored = await _on(
        [
            reply(
                call_part("get_skill", {"skill_id": "slow"}, id="s1"),
                call_part("get_skill", {"skill_id": "fast"}, id="s2"),
            ),
            reply(text_part("Read both.")),
        ],
        delays={"slow": 0.05},
    )

    await _assert_expected(request, stored)
    assert [page["skill_id"] for page in stored.state[SKILLS_READ_STATE_KEY]] == ["fast", "slow"]


async def test_a_stop_while_calls_run_stores_nothing_of_their_response() -> None:
    """A Stop while one call of a response hangs closes the turn as the last
    boundary stored it: neither that call's writes nor those of the call beside
    it that answered, nor the response before them, nor any record of the run."""
    stored = await _on(
        [
            reply(call_part("get_skill", {"skill_id": "oda-dac2a"}, id="s1")),
            reply(_proposal("p1"), call_part("stuck", id="h1")),
        ],
        _REFUSED,
        stop=True,
    )

    assert _planner_records(stored) == []
    assert stored.state == {}


# The id a call without one is read under, in place of the random part drawn for it.
_DRAWN_CALL_ID = "adk-<drawn>"


async def test_calls_without_ids_get_ids_starting_adk(request: pytest.FixtureRequest) -> None:
    """Gemini may send a call without an id. The planner's model
    (`harness.gemini_wire`) gives it one starting `adk-`, so the stored turn
    differs between runs only in the random part of the id, which is read as
    `_DRAWN_CALL_ID`."""
    stored = await _on([reply(call_part(PROPOSE_PLAN, {"question": "q"}))], _ACCEPTED)

    (drawn,) = _call_ids(stored)
    assert drawn.startswith("adk-")
    renamed = [_renamed(record, drawn, _DRAWN_CALL_ID) for record in stored.records]
    await _assert_expected(request, _Stored(records=renamed, state=stored.state))


def _call_ids(stored: _Stored) -> list[str]:
    return [call.get("id") or "" for call in planner_calls(stored.records)]


def _renamed(record: AgentRecord, old: str, new: str) -> AgentRecord:
    body = json.loads(json.dumps(record.body).replace(old, new))
    return AgentRecord(agent=record.agent, kind=record.kind, body=body)


def test_the_answer_to_a_planner_pause_answers_the_ask_the_pause_names() -> None:
    """Two accepted asks precede the pause, which put the first to the user,
    as the pause on a re-offered stored ask does: the answer is the first
    call's, whichever ask came last."""
    first = {"message": "Which source did you mean?"}
    picked = one_answer({"spec_id": "oda"})
    later = {"message": "Which year did you mean?"}
    records = [
        transcript.user_message(_QUESTION),
        model_response(tool_call("a1", PLANNER_ASK, first), tool_call("a2", PLANNER_ASK, later)),
        tool_answers(("a1", PLANNER_ASK, ASK_ACCEPTED), ("a2", PLANNER_ASK, ASK_ACCEPTED)),
        turn_pause(PLANNER_ASK_PAUSE, "int_1", {**first, ASKED_CALL_ID: "a1"}),
        user_answer("int_1", picked),
    ]

    *_, answered = session_steps(records)

    assert answered == Step("answer", "a1", first, {"result": picked}, record=4)
