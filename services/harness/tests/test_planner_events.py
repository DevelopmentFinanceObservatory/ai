"""The events each planner runner stores equal the stored expectation of the conversation.

Each conversation runs on one script of a scripted google-genai client, through
`harness.planner_seam.PlannerSeam` as `tests.test_planner_node` runs it, once
on each runner: the ADK planner and the Pydantic AI planner, whose seam runs
`harness.planner_node`. Both serve the same tools: the real
`ask_user_question` and `get_skill`, and a `propose_plan` whose answers the
script sets. Each stored session is then checked against the conversation's
stored expectation (`tests.planner_expectations`) on what its readers read:
each event as `tests.test_planner_node._read` reads it, what `session_steps`
and Studio's translator read from them, and its state.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolSurface
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.planner_seam import PlannerSeam
from harness.scripted_plan import scripted_plan
from harness.settings import Settings
from harness.studio_state import ConversationState, StudioStore
from harness.studio_translate import _RecordingExecutor, _Translator
from harness.turn_events import PLANNER_ASK
from planner.call import PlannerCall
from planner.plan import PLAN_STATE_KEY
from planner.settings import AGENT_NAME
from planner.skills import SKILLS_READ_STATE_KEY
from planner.specs import PROPOSE_PLAN, planner_tool_specs, skill_tool_spec
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import EndRun

from .planner_expectations import STORING_RUNNER, expect, expectation_name, on_each_runner
from .planner_seam_runners import PlannerRunner
from .scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)
from .test_planner_node import _frames as _numbered_frames
from .test_planner_node import _read, _steps
from .test_planner_seam import _SeamHarness

_QUESTION = "How much aid did Ghana receive?"
_USAGE = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)
_ACCEPTED = {"outcome": "ok", "status": "ready", "summary": "Aid to Ghana."}
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}
_ROUNDS_KEY = "planner:plan_rounds"
_PLAN_JSON = scripted_plan(_QUESTION, spec_id=None).model_dump_json()


# --- The tools both planners serve ----------------------------------------------------


@dataclass
class _Skills:
    """A tool surface serving `get_skill`: each page's text, after its delay. It
    lists no tools, so a translation the seam runs finds no schema."""

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
            call.state[PLAN_STATE_KEY] = _PLAN_JSON
            end_run()
        return answer

    return ToolSpec(
        name=PROPOSE_PLAN,
        description="Propose the plan.",
        parameters=(Parameter("question", str | None, None),),
        body=body,
    )


def _stuck(started: asyncio.Event) -> ToolSpec:
    """A tool that writes to state, sets `started` and then never answers."""

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


# --- One conversation on each planner -------------------------------------------------


@dataclass
class _Stored:
    """The session an entry left, as the session service stores it."""

    events: list[Event]
    state: dict[str, Any]


async def _conversation(seam: PlannerSeam, *, stop: asyncio.Event | None = None) -> _Stored:
    """Run one planner entry of `seam` in a session of its own, and read the
    stored session. With `stop`, the entry is cancelled once it is set, as the
    hanging tool sets it."""
    harness = _SeamHarness()
    entry = harness.run(
        seam,
        new_message=types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        question=_QUESTION,
        readings=[],
    )
    if stop is not None:
        running = asyncio.create_task(entry)
        await asyncio.wait_for(stop.wait(), timeout=5)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    else:
        await entry
    return _Stored(events=await harness.events(), state=dict(await harness.state()))


async def _on(
    runner: PlannerRunner,
    replies: Sequence[Reply],
    *proposals: dict[str, Any],
    delays: Mapping[str, float] | None = None,
    stop: bool = False,
) -> _Stored:
    """The entry on `replies` through `PlannerSeam` on `runner`, serving `_specs`."""
    scripted = ScriptedGenai(replies)
    surface = cast(ToolSurface, _Skills(delays or {}))
    started = asyncio.Event() if stop else None
    seam = PlannerSeam(
        settings=Settings(planner_runner=runner),
        surface=surface,
        specs=_specs(surface, *proposals, stuck=started),
        client=scripted.client,
    )
    return await _conversation(seam, stop=started)


# --- What is compared ------------------------------------------------------------------


def _frames(events: Sequence[Event]) -> list[dict[str, Any]]:
    """The frames Studio's translator makes of `events`, less each line's
    duration, each id Studio draws numbered (`tests.test_planner_node._frames`)."""
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    return _numbered_frames(
        [frame.model_dump(mode="json") for event in events for frame in translator.translate(event)]
    )


async def _no_executor(
    *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
    del on_progress
    raise AssertionError(f"no executor call expected: {tool}({args})")


def _assert_expected(
    request: pytest.FixtureRequest, stored: _Stored, runner: PlannerRunner
) -> None:
    """Assert that `stored` equals the stored expectation of the test `request` runs."""
    expect(
        expectation_name(request, runner=runner),
        {
            "events": _read(stored.events),
            "steps": _steps(stored.events),
            "frames": _frames(stored.events),
            "state": stored.state,
        },
        stores=runner == STORING_RUNNER,
    )


def _planner_events(stored: _Stored) -> list[Event]:
    return [event for event in stored.events if event.author == AGENT_NAME]


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


@on_each_runner
@pytest.mark.parametrize("conversation", list(_CONVERSATIONS))
async def test_the_stored_session_equals_the_adk_planners(
    conversation: str, runner: PlannerRunner, request: pytest.FixtureRequest
) -> None:
    replies, proposals = _CONVERSATIONS[conversation]

    stored = await _on(runner, replies, *proposals)

    assert _planner_events(stored), "the planner stored no event"
    _assert_expected(request, stored, runner)


@on_each_runner
async def test_a_thought_only_response_is_stored_the_same_on_both(
    runner: PlannerRunner, request: pytest.FixtureRequest
) -> None:
    stored = await _on(runner, [reply(thought_part("Only thinking.", signature=None))])

    _assert_expected(request, stored, runner)
    assert len(_planner_events(stored)) == 1


@on_each_runner
async def test_parallel_writes_to_one_list_store_the_value_the_last_call_left(
    runner: PlannerRunner, request: pytest.FixtureRequest
) -> None:
    """Two skill reads in one response each append their page to the list of pages
    read. The first call answers last, so the list it wrote holds both pages,
    which ADK stores though it merges the calls' writes in call order."""
    stored = await _on(
        runner,
        [
            reply(
                call_part("get_skill", {"skill_id": "slow"}, id="s1"),
                call_part("get_skill", {"skill_id": "fast"}, id="s2"),
            ),
            reply(text_part("Read both.")),
        ],
        delays={"slow": 0.05},
    )

    _assert_expected(request, stored, runner)
    assert [page["skill_id"] for page in stored.state[SKILLS_READ_STATE_KEY]] == ["fast", "slow"]


@on_each_runner
async def test_a_stop_while_calls_run_stores_nothing_of_their_response(
    runner: PlannerRunner, request: pytest.FixtureRequest
) -> None:
    """A Stop while one call of a response hangs stores neither that call's writes
    nor those of the call beside it that answered, as ADK stores neither. The
    response before it keeps its answers and writes."""
    stored = await _on(
        runner,
        [
            reply(call_part("get_skill", {"skill_id": "oda-dac2a"}, id="s1")),
            reply(_proposal("p1"), call_part("stuck", id="h1")),
        ],
        _REFUSED,
        stop=True,
    )

    _assert_expected(request, stored, runner)
    assert SKILLS_READ_STATE_KEY in stored.state
    assert not {"stuck", _ROUNDS_KEY} & set(stored.state)


# The id a call without one is read under, in place of the random part ADK draws.
_DRAWN_CALL_ID = "adk-<drawn>"


@on_each_runner
async def test_calls_without_ids_get_ids_of_adks_form_on_both_planners(
    runner: PlannerRunner, request: pytest.FixtureRequest
) -> None:
    """Gemini may send a call without an id. ADK gives it one starting `adk-`, and
    so does the Pydantic AI planner's model (`harness.planner_model`), so the
    stored sessions differ only in the random part of the id, which is read as
    `_DRAWN_CALL_ID`."""
    stored = await _on(runner, [reply(call_part(PROPOSE_PLAN, {"question": "q"}))], _ACCEPTED)

    (drawn,) = _call_ids(stored)
    assert drawn.startswith("adk-")
    renamed = [_renamed(event, drawn, _DRAWN_CALL_ID) for event in stored.events]
    _assert_expected(request, _Stored(events=renamed, state=stored.state), runner)


def _call_ids(stored: _Stored) -> list[str]:
    return [call.id or "" for event in stored.events for call in event.get_function_calls()]


def _renamed(event: Event, old: str, new: str) -> Event:
    dumped = event.model_dump_json().replace(old, new)
    return Event.model_validate_json(dumped)
