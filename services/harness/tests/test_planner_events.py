"""The events a Pydantic AI planner run stores equal the ADK planner's.

Each conversation runs twice on the same script of a scripted google-genai
client, through `harness.planner_seam.PlannerSeam` as `tests.test_planner_node`
runs it: once on the ADK planner and once on the Pydantic AI planner, whose
seam runs `harness.planner_node`. Both serve the same tools: the real
`ask_user_question` and `get_skill`, and a `propose_plan` whose answers the
script sets. The stored sessions are then compared on what their readers
read: each event as `tests.test_planner_node._read` reads it, what
`session_steps` and Studio's translator read from them, and their state.
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
from planner.agent import AGENT_NAME
from planner.call import PlannerCall
from planner.plan import PLAN_STATE_KEY
from planner.session import session_steps
from planner.skills import SKILLS_READ_STATE_KEY
from planner.specs import PROPOSE_PLAN, planner_tool_specs, skill_tool_spec
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import EndRun

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
from .test_planner_node import _read
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


async def _stuck(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
    """A tool that writes to state and then never answers."""
    del end_run, arguments
    call.state["stuck"] = "started"
    _STUCK.set()
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


_STUCK = asyncio.Event()


def _specs(surface: ToolSurface, *proposals: dict[str, Any], stuck: bool = False) -> list[ToolSpec]:
    ask = next(spec for spec in planner_tool_specs() if spec.name == PLANNER_ASK)
    stuck_specs = [ToolSpec(name="stuck", description="Hangs.", parameters=(), body=_stuck)]
    return [
        skill_tool_spec(surface),
        ask,
        _propose(*proposals),
        *(stuck_specs if stuck else []),
    ]


# --- One conversation on each planner -------------------------------------------------


@dataclass
class _Stored:
    """The session an entry left, as the session service stores it."""

    events: list[Event]
    state: dict[str, Any]


async def _conversation(seam: PlannerSeam, *, stop: bool = False) -> _Stored:
    """Run one planner entry of `seam` in a session of its own, and read the
    stored session. With `stop`, the entry is cancelled once a tool hangs."""
    harness = _SeamHarness()
    entry = harness.run(
        seam,
        new_message=types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        question=_QUESTION,
        readings=[],
    )
    if stop:
        _STUCK.clear()
        running = asyncio.create_task(entry)
        await asyncio.wait_for(_STUCK.wait(), timeout=5)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    else:
        await entry
    return _Stored(events=await harness.events(), state=dict(await harness.state()))


async def _both(
    replies: Sequence[Reply],
    *proposals: dict[str, Any],
    delays: Mapping[str, float] | None = None,
    stop: bool = False,
) -> tuple[_Stored, _Stored]:
    """The entry on `replies` through `PlannerSeam` on the ADK planner and on the
    Pydantic AI planner (`harness.planner_node`), each serving `_specs`."""
    stored = []
    for runner in ("adk", "pydantic_ai"):
        scripted = ScriptedGenai(replies)
        surface = cast(ToolSurface, _Skills(delays or {}))
        seam = PlannerSeam(
            settings=Settings(planner_runner=runner),
            surface=surface,
            specs=_specs(surface, *proposals, stuck=stop),
            client=scripted.client,
        )
        stored.append(await _conversation(seam, stop=stop))
    adk, pydantic = stored
    return adk, pydantic


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


def _assert_same(adk: _Stored, pydantic: _Stored) -> None:
    assert _read(pydantic.events) == _read(adk.events)
    assert list(session_steps(pydantic.events)) == list(session_steps(adk.events))
    assert _frames(pydantic.events) == _frames(adk.events)
    assert pydantic.state == adk.state


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


@pytest.mark.parametrize("conversation", list(_CONVERSATIONS))
async def test_the_stored_session_equals_the_adk_planners(conversation: str) -> None:
    replies, proposals = _CONVERSATIONS[conversation]

    adk, pydantic = await _both(replies, *proposals)

    assert _planner_events(adk), "the ADK planner stored no event"
    _assert_same(adk, pydantic)


async def test_a_thought_only_response_is_stored_the_same_on_both() -> None:
    adk, pydantic = await _both([reply(thought_part("Only thinking.", signature=None))])

    _assert_same(adk, pydantic)
    assert len(_planner_events(pydantic)) == 1


async def test_parallel_writes_to_one_list_store_the_value_the_last_call_left() -> None:
    """Two skill reads in one response each append their page to the list of pages
    read. The first call answers last, so the list it wrote holds both pages,
    which ADK stores though it merges the calls' writes in call order."""
    adk, pydantic = await _both(
        [
            reply(
                call_part("get_skill", {"skill_id": "slow"}, id="s1"),
                call_part("get_skill", {"skill_id": "fast"}, id="s2"),
            ),
            reply(text_part("Read both.")),
        ],
        delays={"slow": 0.05},
    )

    _assert_same(adk, pydantic)
    assert [page["skill_id"] for page in adk.state[SKILLS_READ_STATE_KEY]] == ["fast", "slow"]


async def test_a_stop_while_calls_run_stores_nothing_of_their_response() -> None:
    """A Stop while one call of a response hangs stores neither that call's writes
    nor those of the call beside it that answered, as ADK stores neither. The
    response before it keeps its answers and writes."""
    adk, pydantic = await _both(
        [
            reply(call_part("get_skill", {"skill_id": "oda-dac2a"}, id="s1")),
            reply(_proposal("p1"), call_part("stuck", id="h1")),
        ],
        _REFUSED,
        stop=True,
    )

    _assert_same(adk, pydantic)
    assert SKILLS_READ_STATE_KEY in adk.state
    assert not {"stuck", _ROUNDS_KEY} & set(adk.state)


async def test_calls_without_ids_get_ids_of_adks_form_on_both_planners() -> None:
    """Gemini may send a call without an id. ADK gives it one starting `adk-`, and
    so does the Pydantic AI planner's model (`harness.planner_model`), so the
    stored sessions differ only in the random part of the id."""
    adk, pydantic = await _both([reply(call_part(PROPOSE_PLAN, {"question": "q"}))], _ACCEPTED)

    adk_ids, pydantic_ids = (_call_ids(stored) for stored in (adk, pydantic))
    assert len(adk_ids) == len(pydantic_ids) == 1
    assert adk_ids[0].startswith("adk-")
    assert pydantic_ids[0].startswith("adk-")
    renamed = [_renamed(event, pydantic_ids[0], adk_ids[0]) for event in pydantic.events]
    assert _read(renamed) == _read(adk.events)


def _call_ids(stored: _Stored) -> list[str]:
    return [call.id or "" for event in stored.events for call in event.get_function_calls()]


def _renamed(event: Event, old: str, new: str) -> Event:
    dumped = event.model_dump_json().replace(old, new)
    return Event.model_validate_json(dumped)
