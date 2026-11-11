"""The events a Pydantic AI planner run stores equal the ADK planner's.

Each conversation runs twice on the same script of a scripted google-genai
client. The ADK planner runs as `harness.planner_seam.PlannerSeam` runs it:
`create_planner_agent` on `Gemini`, through a workflow node's `ctx.run_node`
on the planner's branch. The Pydantic AI planner runs `run_planner` on
`planner_model`, inside a node of the same name and branch that yields the
events `harness.planner_events` builds. Both serve the same tools: the real
`ask_user_question` and `get_skill`, and a `propose_plan` whose answers the
script sets. The stored sessions are then compared: their events field by
field apart from ids and timestamps, what `session_steps` and Studio's
translator read from them, and their state.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolSurface
from google.adk import Context
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.workflow import START, BaseNode, Workflow, node
from google.genai import types
from harness.planner_events import (
    NO_CONTENT_ERROR_CODE,
    answered_every_call,
    model_response_event,
    tool_answers_event,
)
from harness.planner_model import ResponseMetadata, planner_model
from harness.scripted_plan import scripted_plan
from harness.sentences import SeamCode
from harness.studio_state import ConversationState, StudioStore
from harness.studio_translate import _RecordingExecutor, _Translator
from harness.turn_events import PLANNER_ASK
from planner.agent import (
    AGENT_NAME,
    BRIEF_STATE_KEY,
    SpecTool,
    create_planner_agent,
    planner_instruction,
    session_steps,
)
from planner.call import PlannerCall
from planner.plan import PLAN_STATE_KEY
from planner.pydantic_agent import (
    PLANNER_PROFILE,
    PlannerDeps,
    PlannerResponse,
    PlannerToolAnswer,
    planner_agent,
    run_planner,
)
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

_MODEL = "gemini-3.8-flash"
_BRANCH = "dfo_planner"
_QUESTION = "How much aid did Ghana receive?"
_BRIEF = "Context for this question: the readings."
_USAGE = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)
_ACCEPTED = {"outcome": "ok", "status": "ready", "summary": "Aid to Ghana."}
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}
_ROUNDS_KEY = "planner:plan_rounds"
_PLAN_JSON = scripted_plan(_QUESTION, spec_id=None).model_dump_json()


# --- The tools both planners serve ----------------------------------------------------


@dataclass
class _Skills:
    """A tool surface serving `get_skill`: each page's text, after its delay."""

    delays: Mapping[str, float]

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
    """The session a planner run left, as the session service stores it."""

    session: Session

    @property
    def events(self) -> list[Event]:
        return self.session.events

    @property
    def state(self) -> dict[str, Any]:
        return self.session.state


async def _conversation(planner: BaseNode, *, stop: bool = False) -> _Stored:
    """Run a turn whose planner node runs `planner` on the planner's branch, and read
    the stored session. With `stop`, the turn is cancelled once a tool hangs."""

    async def _planner_node(ctx: Context, node_input: Any) -> None:
        del node_input
        ctx.state[BRIEF_STATE_KEY] = _BRIEF
        await ctx.run_node(planner, None, override_branch=_BRANCH)

    workflow = Workflow(
        name="turn", edges=[(START, node(_planner_node, name="planner", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="app", user_id="user")
    runner = Runner(node=workflow, app_name="app", session_service=sessions)

    async def turn() -> None:
        async for _ in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            pass

    if stop:
        _STUCK.clear()
        running = asyncio.create_task(turn())
        await asyncio.wait_for(_STUCK.wait(), timeout=5)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    else:
        await turn()
    stored = await sessions.get_session(app_name="app", user_id="user", session_id=session.id)
    assert stored is not None
    return _Stored(stored)


def _adk_planner(specs: Sequence[ToolSpec], scripted: ScriptedGenai) -> BaseNode:
    agent = create_planner_agent(model_name=_MODEL, question_fixed=True, reads_user_words=False)
    agent.model = Gemini(model=_MODEL, client=scripted.client)
    agent.tools = [SpecTool(spec) for spec in specs]
    return agent


def _failed(name: str, error: BaseException) -> dict[str, Any]:
    raise AssertionError(f"{name} raised {error!r}")


def _pydantic_planner(specs: Sequence[ToolSpec], scripted: ScriptedGenai) -> BaseNode:
    """The planner node over `run_planner`: each response's event as it arrives, and
    one event for a response's answers once every call of it has answered."""
    reported: list[ResponseMetadata] = []

    def after_response(closed_by: SeamCode | None, metadata: ResponseMetadata) -> None:
        assert closed_by is None
        reported.append(metadata)

    model = planner_model(
        _MODEL,
        client=scripted.client,
        budget=None,
        after_response=after_response,
        profile=PLANNER_PROFILE,
    )
    agent = planner_agent(
        specs,
        instruction=planner_instruction(has_translator=False, has_skills=True, question_fixed=True),
        model=model,
    )

    async def dfo_planner(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        del node_input
        ctx.event_author = AGENT_NAME
        deps = PlannerDeps(
            state=ctx.session.state,
            earlier=tuple(session_steps(ctx.session.events)),
            tool_answered=lambda name, answer: None,
            tool_failed=_failed,
        )
        response: PlannerResponse | None = None
        answers: list[PlannerToolAnswer] = []
        brief = ctx.session.state[BRIEF_STATE_KEY]
        async for item in run_planner(agent, deps, question=_QUESTION, brief=brief):
            if isinstance(item, PlannerResponse):
                response, answers = item, []
                metadata = reported.pop(0)
                yield model_response_event(
                    item,
                    usage=metadata.usage,
                    finish_reason=metadata.finish_reason,
                    model_version=metadata.model_version,
                )
                continue
            assert response is not None
            answers.append(item)
            if answered_every_call(response, answers):
                yield tool_answers_event(response, answers, ended=deps.ended)

    return node(dfo_planner, name=AGENT_NAME)


async def _both(
    replies: Sequence[Reply],
    *proposals: dict[str, Any],
    delays: Mapping[str, float] | None = None,
    stop: bool = False,
) -> tuple[_Stored, _Stored]:
    stored = []
    for planner in (_adk_planner, _pydantic_planner):
        scripted = ScriptedGenai(replies)
        specs = _specs(cast(ToolSurface, _Skills(delays or {})), *proposals, stuck=stop)
        stored.append(await _conversation(planner(specs, scripted), stop=stop))
    adk, pydantic = stored
    return adk, pydantic


# --- What is compared ------------------------------------------------------------------


def _dumped(events: Sequence[Event]) -> list[dict[str, Any]]:
    """Each event as stored, less what differs between any two runs: its id, its
    timestamp and the invocation's id."""
    return [
        event.model_dump(
            mode="json", exclude_none=True, exclude={"id", "timestamp", "invocation_id"}
        )
        for event in events
    ]


def _frames(events: Sequence[Event]) -> list[dict[str, Any]]:
    """The frames Studio's translator makes of `events`, less each line's duration."""
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    return [
        frame.model_dump(mode="json", exclude={"duration_ms"})
        for event in events
        for frame in translator.translate(event)
    ]


async def _no_executor(
    *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
    del on_progress
    raise AssertionError(f"no executor call expected: {tool}({args})")


def _assert_same(adk: _Stored, pydantic: _Stored) -> None:
    assert _dumped(pydantic.events) == _dumped(adk.events)
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


async def test_a_text_response_is_the_runs_output_and_a_thought_only_one_carries_adks_error() -> (
    None
):
    """A response with no call is the node's output, as ADK marks an agent's final
    turn, and one with nothing to act on carries ADK's no-content error."""
    adk, pydantic = await _both([reply(thought_part("Only thinking.", signature=None))])

    _assert_same(adk, pydantic)
    (event,) = _planner_events(pydantic)
    assert event.error_code == NO_CONTENT_ERROR_CODE
    assert event.node_info.message_as_output is True


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
    nor those of the call beside it that answered, as ADK stores neither; the
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
    stored events differ only in the random part of the id."""
    adk, pydantic = await _both([reply(call_part(PROPOSE_PLAN, {"question": "q"}))], _ACCEPTED)

    adk_ids, pydantic_ids = (_call_ids(stored) for stored in (adk, pydantic))
    assert len(adk_ids) == len(pydantic_ids) == 1
    assert adk_ids[0].startswith("adk-")
    assert pydantic_ids[0].startswith("adk-")
    renamed = [_renamed(event, pydantic_ids[0], adk_ids[0]) for event in pydantic.events]
    assert _dumped(renamed) == _dumped(adk.events)


def _call_ids(stored: _Stored) -> list[str]:
    return [call.id or "" for event in stored.events for call in event.get_function_calls()]


def _renamed(event: Event, old: str, new: str) -> Event:
    dumped = event.model_dump_json().replace(old, new)
    return Event.model_validate_json(dumped)


# --- The builders on their own --------------------------------------------------------


def _response(*parts: types.Part) -> PlannerResponse:
    return PlannerResponse(parts=parts)


def test_answers_are_ordered_by_call_and_their_writes_merged_as_adk_merges_them() -> None:
    response = _response(_ask("a1"), _proposal("p1"))
    answers = [
        PlannerToolAnswer(PROPOSE_PLAN, "p1", {"outcome": "ok"}, {"n": 2, "d": {"y": 2}}),
        PlannerToolAnswer(PLANNER_ASK, "a1", {"outcome": "ok"}, {"n": 1, "d": {"x": 1}}),
    ]

    assert answered_every_call(response, answers)
    assert not answered_every_call(response, answers[:1])
    event = tool_answers_event(response, answers, ended=False)

    assert [r.id for r in event.get_function_responses()] == ["a1", "p1"]
    # In call order a scalar is replaced and two dicts merge; the dict two calls
    # wrote then merges in the value of the call that answered last.
    assert event.actions.state_delta == {"n": 2, "d": {"x": 1, "y": 2}}
    assert event.actions.skip_summarization is None
    assert tool_answers_event(response, answers, ended=True).actions.skip_summarization is True


def test_an_answer_to_no_call_of_the_response_is_refused() -> None:
    with pytest.raises(ValueError, match="did not make"):
        tool_answers_event(
            _response(_ask("a1")), [PlannerToolAnswer(PLANNER_ASK, "x", {}, {})], ended=False
        )
