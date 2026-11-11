"""The planner seam on the Pydantic AI planner equals the seam on the ADK planner.

Each entry runs twice on the same script of a scripted google-genai client:
once with `PlannerSeam`'s ADK planner on `Gemini` over the client, once with
`runner="pydantic_ai"` over the same client, which runs `harness.planner_node`
in the ADK planner's place. Both run through the workflow node
`tests.test_planner_seam._SeamHarness` drives, each in a session of its own,
under Studio's streaming run config, and are compared on what the seam
answers, the progress it reports, the requests the model received, the events
the session stores, the steps `session_steps` reads from them, and the stored
state. The Studio tests run a whole turn through `/api/studio/turn` on each
planner and compare the frames it streams.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from unittest import mock

import pytest
from compiler.surface import FixtureSurface
from google.adk import Context, Event
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.models import Gemini
from google.adk.runners import Runner as AdkRunner
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from harness import planner_run
from harness.planner_run import PlannerSeamError
from harness.planner_seam import PlannerSeam
from harness.scripted import _compiled_proposal
from harness.sentences import SeamCode, refusal_sentence
from harness.settings import Settings
from harness.turn_budget import BoundedGemini, TurnBudget, current_turn_budget
from harness.turn_events import PLANNER_ASK
from planner.agent import AGENT_NAME, SpecTool, session_steps
from planner.call import PlannerCall, elected_spec_ids
from planner.plan import (
    DatasetRequirement,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    load_plan,
    save_pin_sources,
    save_plan,
)
from planner.settings import AgentSettings
from planner.specs import PROPOSE_PLAN
from planner.tool_results import Outcome, ProblemCode
from planner.tool_spec import ToolSpec
from planner.tools import EndRun
from submission.service import scope_choice_decisions
from translator.tool import Translate

from .orchestrator_fakes import plan_call, stream, studio_client, turn_body
from .planner.planner_pairs import assert_pinned_wire_differences
from .scripted_genai import (
    Reply,
    Request,
    ScriptedGenai,
    blocked_reply,
    call_part,
    empty_reply,
    reply,
    text_part,
    usage_metadata,
)
from .test_planner_seam import (
    _APP,
    _USER,
    _pending_ask_interrupt_id,
    _SeamHarness,
    _sector_switched,
    _translated_plan,
)
from .test_studio import _app, _interrupt_body, _pick, _text
from .test_turn_records import _switchable_scope

type Runner = Literal["adk", "pydantic_ai"]

_MODEL = AgentSettings().planner_model_name
_USAGE = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)
_QUESTION = "kenya debt"
_SSE = RunConfig(streaming_mode=StreamingMode.SSE)
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}
_SPEC = "spec_" + "c" * 32
_SPEC_NEW = "spec_" + "d" * 32
# Where the test's `propose_plan` records the spec_ids the user picked.
_ELECTED_KEY = "planner:elected_spec_ids"
# Gemini signs the first function call of a response it thought before.
_SIGNED = b"scripted-signature"


@contextlib.contextmanager
def _numbered_call_ids() -> Iterator[None]:
    """Number the ids ADK gives a call that has none from 1, so a turn put in the
    model's place carries the same call id on both planners."""
    counter = itertools.count(1)
    with mock.patch.object(
        adk_functions, "generate_client_function_call_id", lambda: f"adk-{next(counter)}"
    ):
        yield


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


# --- The seam on each planner ------------------------------------------------------------


def _seam(
    runner: Runner,
    scripted: ScriptedGenai,
    *,
    bounded: bool = False,
    translate: Translate | None = None,
) -> PlannerSeam:
    """The seam on `runner`, its model calls answered by `scripted`. `bounded` keeps the
    ADK planner's model to the turn's budget, as the served planner's is; the
    Pydantic AI planner's model always keeps to it."""
    surface = FixtureSurface(schemas=[])
    if runner == "adk":
        model = (BoundedGemini if bounded else Gemini)(model=_MODEL, client=scripted.client)
        return PlannerSeam(settings=Settings(), surface=surface, translate=translate, model=model)
    return PlannerSeam(
        settings=Settings(),
        surface=surface,
        translate=translate,
        runner="pydantic_ai",
        client=scripted.client,
    )


def _proposals(
    *answers: dict[str, Any], records: Plan | None = None
) -> Callable[[PlannerSeam], None]:
    """Give the seam's planner a `propose_plan` that answers `answers` in turn, and
    then as a proposal whose translation compiled the plan on record
    (`harness.scripted.use_compiled_proposals`), recording `records` first when
    the session holds no plan. Every call records the spec_ids the user picked,
    read from its steps, under `_ELECTED_KEY`. Each seam set up gets `answers`
    afresh."""

    def setup(seam: PlannerSeam) -> None:
        left = list(answers)

        def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
            call.state[_ELECTED_KEY] = sorted(elected_spec_ids(call))
            if left:
                return left.pop(0)
            if records is not None and load_plan(call.state) is None:
                save_plan(call.state, records)
            return _compiled_proposal(call, end_run, arguments)

        _serve_proposals(seam, body)

    return setup


def _serve_proposals(seam: PlannerSeam, body: Callable[..., object]) -> None:
    """Replace the seam's planner's `propose_plan` with one that runs `body`."""
    spec = ToolSpec(name=PROPOSE_PLAN, description="Propose the plan.", parameters=(), body=body)
    seam._agent.tools = [
        SpecTool(spec) if tool.name == PROPOSE_PLAN else tool for tool in seam._agent.tools
    ]


@dataclass
class _Entry:
    """What one planner's entries answered, reported, sent and left."""

    outputs: list[dict[str, Any] | None]
    reports: list[dict[str, Any]]
    requests: list[Request]
    events: list[Event]
    state: dict[str, Any]


@dataclass
class _Script:
    """An entry's script: the model's replies, the seam's arguments for each run,
    which the user's `answer` to the pending ask resumes, how the seam is set up,
    and the budget each run is given."""

    replies: Sequence[Reply]
    runs: Sequence[Mapping[str, Any]]
    setup: Callable[[PlannerSeam], None]
    budget: Callable[[], TurnBudget] | None = None
    translate: Translate | None = None


async def _entries(
    runner: Runner,
    script: _Script,
    *,
    then: Callable[[PlannerSeam, _SeamHarness], Any] | None = None,
) -> _Entry:
    scripted = ScriptedGenai(script.replies)
    seam = _seam(runner, scripted, bounded=script.budget is not None, translate=script.translate)
    script.setup(seam)
    harness = _SeamHarness()
    reports: list[dict[str, Any]] = []
    outputs: list[dict[str, Any] | None] = []
    for arguments in script.runs:
        kwargs = {"question": _QUESTION, "readings": [], **arguments}
        answer = kwargs.pop("answer", None)
        if answer is not None:
            interrupt_id = _pending_ask_interrupt_id(await harness.events())
            kwargs["new_message"] = types.Content(
                role="user", parts=[create_request_input_response(interrupt_id, answer)]
            )
        token = current_turn_budget.set(script.budget() if script.budget is not None else None)
        try:
            outputs.append(
                await harness.run(seam, on_progress=reports.append, run_config=_SSE, **kwargs)
            )
        finally:
            current_turn_budget.reset(token)
    if then is not None:
        outputs.append(await then(seam, harness))
    return _Entry(
        outputs=outputs,
        reports=[{k: v for k, v in report.items() if k != "duration_ms"} for report in reports],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


async def _both(
    script: _Script, *, then: Callable[[PlannerSeam, _SeamHarness], Any] | None = None
) -> tuple[_Entry, _Entry]:
    with _numbered_call_ids():
        adk = await _entries("adk", script, then=then)
    with _numbered_call_ids():
        pydantic = await _entries("pydantic_ai", script, then=then)
    return adk, pydantic


def _dumped(events: Sequence[Event]) -> list[dict[str, Any]]:
    """Each event as stored, less what differs between any two runs: its id, its
    timestamp and the invocation's id."""
    return [
        event.model_dump(
            mode="json", exclude_none=True, exclude={"id", "timestamp", "invocation_id"}
        )
        for event in events
    ]


def _assert_same(adk: _Entry, pydantic: _Entry) -> None:
    assert pydantic.outputs == adk.outputs
    assert pydantic.reports == adk.reports
    assert len(pydantic.requests) == len(adk.requests)
    for adk_request, pydantic_request in zip(adk.requests, pydantic.requests, strict=True):
        assert_pinned_wire_differences(adk_request, pydantic_request)
    assert _dumped(pydantic.events) == _dumped(adk.events)
    assert list(session_steps(pydantic.events)) == list(session_steps(adk.events))
    assert pydantic.state == adk.state


def _planner_events(entry: _Entry) -> list[Event]:
    return [event for event in entry.events if event.author == AGENT_NAME]


def _planned(plan: Plan) -> dict[str, Any]:
    return {"question": plan.question, "prior_plan": plan.model_dump(mode="json")}


def _ask_call(id: str, **args: Any) -> types.Part:
    return call_part(
        PLANNER_ASK, {"message": "Which source did you mean?", **args}, id=id, signature=_SIGNED
    )


# --- Entries ------------------------------------------------------------------------------


async def test_a_planned_turn_is_the_same_on_both_planners(plan: Plan) -> None:
    adk, pydantic = await _both(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(),
        )
    )

    assert adk.outputs[0] is not None and "plan" in adk.outputs[0]
    assert len(_planner_events(adk)) == 2
    _assert_same(adk, pydantic)


async def test_a_refused_proposal_then_an_accepted_one_is_the_same_on_both_planners(
    plan: Plan,
) -> None:
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), usage=_USAGE),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        )
    )

    assert adk.outputs[0] is not None and "plan" in adk.outputs[0]
    assert len(adk.requests) == 2
    _assert_same(adk, pydantic)


async def test_an_ask_and_its_answer_are_the_same_on_both_planners(plan: Plan) -> None:
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(_ask_call("a1"), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[_planned(plan), {**_planned(plan), "answer": {"result": "2020"}}],
            setup=_proposals(),
        )
    )

    assert adk.outputs[0] is None, "the ask pauses the first run"
    assert adk.outputs[1] is not None and "plan" in adk.outputs[1]
    _assert_same(adk, pydantic)


async def test_an_ask_beside_a_proposal_stores_one_answers_event_on_both_planners(
    plan: Plan,
) -> None:
    """The answers of a response's calls are stored as one event once every call
    has answered, as the ADK planner stores them."""
    adk, pydantic = await _both(
        _Script(
            replies=[reply(_ask_call("a1"), call_part(PROPOSE_PLAN, {}, id="p1"), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(),
        )
    )

    (answers,) = [event for event in _planner_events(adk) if event.get_function_responses()]
    assert [response.id for response in answers.get_function_responses()] == ["a1", "p1"]
    _assert_same(adk, pydantic)


# --- Turns put in the model's place ------------------------------------------------------


async def test_a_proposal_refused_for_good_ends_the_run_on_the_same_stop_on_both(
    plan: Plan,
) -> None:
    """A refusal that ends the entry answers the next request with the stop's
    sentence in place of the model, which is never called again."""
    refused = {
        "outcome": "refused",
        "code": ProblemCode.BUDGET_EXHAUSTED.value,
        "message": "No more rounds.",
    }
    adk, pydantic = await _both(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1"), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(refused),
        )
    )

    assert len(adk.requests) == 1
    (stop,) = [
        event
        for event in _planner_events(adk)
        if event.content and any(part.text for part in event.content.parts or ())
    ]
    assert stop.partial is None and stop.usage_metadata is None
    _assert_same(adk, pydantic)


async def test_a_referral_the_user_decides_asks_in_place_of_the_model_on_both(
    plan: Plan,
) -> None:
    """A referral whose pick is the user's answers the next request with the
    question in place of the model."""
    referral = _referral({"spec_id": _SPEC, "sentence": "external debt"})
    referral["translation"]["kind"] = "source_name_ambiguous"
    referral["translation"]["ambiguous_sources"] = [
        {"candidates": [{"ref": _SPEC, "label": "external debt"}]}
    ]
    adk, pydantic = await _both(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1"), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(referral),
        )
    )

    assert adk.outputs == [None], "the question pauses the run"
    assert len(adk.requests) == 1
    _assert_same(adk, pydantic)


# --- The turn's budget ------------------------------------------------------------------


def _one_call_budget() -> TurnBudget:
    """A budget whose call cap closes every call after the first."""
    return TurnBudget(model_time_s=60.0, model_calls_max=1, wall_s=60.0, first_chunk_s=30.0)


def _referral(*candidates: dict[str, Any]) -> dict[str, Any]:
    """A proposal whose translation referred the plan back to choose a source."""
    return {
        "outcome": Outcome.INVALID_INPUT.value,
        "translation": {
            "outcome": Outcome.CHOICE_REQUIRED.value,
            "kind": "source_ambiguous",
            "requirement_id": "debt",
            "candidates": list(candidates),
        },
    }


async def test_a_budget_closing_a_call_with_pinnable_candidates_asks_the_same_on_both(
    plan: Plan,
) -> None:
    """The second call is past the cap. The referral before it names a source to
    pin, so both planners answer the closed call with the same question."""
    adk, pydantic = await _both(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(_referral({"spec_id": _SPEC, "sentence": "external debt"})),
            budget=_one_call_budget,
        )
    )

    assert adk.outputs == [None], "the closed call's question pauses the run"
    (asked,) = [
        call
        for event in _planner_events(adk)
        for call in event.get_function_calls()
        if call.name == PLANNER_ASK
    ]
    assert asked.id == "adk-1"
    assert (asked.args or {})["candidates"] == [{"spec_id": _SPEC, "label": "external debt"}]
    _assert_same(adk, pydantic)


async def test_a_budget_closing_a_call_with_nothing_to_pin_refuses_the_same_on_both(
    plan: Plan,
) -> None:
    adk, pydantic = await _both(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(_referral({"sentence": "a nearby series"})),
            budget=_one_call_budget,
        )
    )

    code = SeamCode.TURN_MODEL_CALLS_SPENT
    assert adk.outputs == [
        {"refused": {"text": refusal_sentence({"code": code}), "final": True, "code": code}}
    ]
    _assert_same(adk, pydantic)


# --- Responses that are no turn, and a run that raises ----------------------------------


@pytest.mark.parametrize(
    "response",
    [
        reply(text_part("Cut off"), usage=_USAGE, finish_reason=types.FinishReason.MAX_TOKENS),
        blocked_reply(),
        empty_reply(),
    ],
    ids=["max-tokens", "blocked", "empty"],
)
async def test_a_response_that_errs_or_holds_nothing_is_stored_the_same_on_both(
    plan: Plan, response: Reply
) -> None:
    """A response cut at its token limit stores its finish reason as its error, a
    blocked prompt stores an event of its block reason and no content, and an
    empty response stores nothing; each ends the run with no plan."""
    adk, pydantic = await _both(
        _Script(replies=[response], runs=[_planned(plan)], setup=_proposals())
    )

    assert (
        adk.outputs[0] is not None and adk.outputs[0]["refused"]["code"] == SeamCode.PLAN_NOT_READY
    )
    _assert_same(adk, pydantic)


async def _raising(runner: Runner, plan: Plan) -> _Entry:
    """An entry whose second tool answer of one response is past the run's event
    cap (`harness.planner_run._MAX_EVENTS`, set to 1), which raises out of the run."""
    scripted = ScriptedGenai(
        [reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), _ask_call("a1"))]
    )
    seam = _seam(runner, scripted)
    _proposals(_REFUSED)(seam)
    harness = _SeamHarness()
    with (
        _numbered_call_ids(),
        mock.patch.object(planner_run, "_MAX_EVENTS", 1),
        pytest.raises(PlannerSeamError),
    ):
        await harness.run(seam, run_config=_SSE, readings=[], **_planned(plan))
    return _Entry(
        outputs=[],
        reports=[],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


async def test_a_run_that_raises_while_calls_run_keeps_the_answered_calls_on_both(
    plan: Plan,
) -> None:
    """The ADK planner keeps the answer of the call that finished when its sibling's
    raises; the Pydantic AI planner yields the answer it held."""
    adk = await _raising("adk", plan)
    pydantic = await _raising("pydantic_ai", plan)

    (answered,) = [
        response for event in _planner_events(adk) for response in event.get_function_responses()
    ]
    assert answered.id == "p1"
    _assert_same(adk, pydantic)


# --- A Stop ------------------------------------------------------------------------------


async def _stopped(runner: Runner, plan: Plan) -> _Entry:
    """An entry stopped while `propose_plan` hangs, its first call answered."""
    started = asyncio.Event()

    async def hangs(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run, arguments
        call.state["planner:hanging"] = "started"
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    calls = itertools.count()

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        if next(calls) == 0:
            call.state["planner:first"] = "answered"
            return _REFUSED
        return hangs(call, end_run, arguments)

    scripted = ScriptedGenai(
        [
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), _ask_call("a1")),
        ]
    )
    seam = _seam(runner, scripted)
    _serve_proposals(seam, body)
    harness = _SeamHarness()
    running = asyncio.create_task(harness.run(seam, run_config=_SSE, readings=[], **_planned(plan)))
    await asyncio.wait_for(started.wait(), timeout=5)
    running.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await running
    return _Entry(
        outputs=[],
        reports=[],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


async def test_a_stop_while_a_planner_tool_hangs_leaves_the_same_session_on_both(
    plan: Plan,
) -> None:
    adk = await _stopped("adk", plan)
    pydantic = await _stopped("pydantic_ai", plan)

    assert adk.state["planner:first"] == "answered"
    assert "planner:hanging" not in adk.state
    _assert_same(adk, pydantic)


# --- A revision after the entry -----------------------------------------------------------


def _pinned_plan(plan: Plan) -> tuple[Plan, DatasetRequirement, dict[str, PinSource]]:
    """`plan` with its dataset pinned to `_SPEC`, whose scope offers a sector
    switch, the requirement re-pinned to `_SPEC_NEW` with the sector switched,
    and the pin sources of both."""
    scope = _switchable_scope()
    switched = _sector_switched(scope)
    blocks = {
        _SPEC: {"spec_id": _SPEC, "resolved": {"scope_choices": scope}},
        _SPEC_NEW: {"spec_id": _SPEC_NEW, "resolved": {"scope_choices": switched}},
    }
    shown = {spec_id: PinSource(block=block, settled=True) for spec_id, block in blocks.items()}
    scoped = scope_choice_decisions({_SPEC: shown[_SPEC]})[_SPEC]
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    pinned = dataset.model_copy(
        update={
            "source_spec_id": _SPEC,
            "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
        }
    )
    pinned_plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [*plan.decisions, *scoped],
            "evidence_requirements": [
                pinned if r.id == dataset.id else r for r in plan.evidence_requirements
            ],
        }
    )
    return pinned_plan, pinned.model_copy(update={"source_spec_id": _SPEC_NEW}), shown


async def test_a_revision_after_an_entry_reads_the_same_steps_on_both_planners(
    plan: Plan,
) -> None:
    """The user picked `_SPEC` answering the planner's ask. A sector switch then
    re-pins it to `_SPEC_NEW` with no model call, and the decision the switch
    makes is the user's on both planners, read from the entry's stored ask and
    answer."""
    pinned, repinned, shown = _pinned_plan(plan)

    async def revise(seam: PlannerSeam, harness: _SeamHarness) -> dict[str, Any]:
        async def _node(ctx: Context, node_input: Any) -> Any:
            del node_input
            save_pin_sources(cast(SessionState, ctx.state), shown)
            return await seam.revise(
                ctx=ctx,
                prior_plan=pinned.model_dump(mode="json"),
                evidence_requirements=[repinned.model_dump(mode="json")],
                inherits={_SPEC_NEW: _SPEC},
            )

        runner = AdkRunner(
            node=Workflow(name="revise_wf", edges=[(START, node(_node, name="revise"))]),
            app_name=_APP,
            session_service=harness._sessions,
        )
        assert harness._session_id is not None
        (output,) = [
            event.output
            async for event in runner.run_async(
                user_id=_USER,
                session_id=harness._session_id,
                new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
            )
            if event.output is not None
        ]
        return cast(dict[str, Any], output)

    async def translate(call: PlannerCall) -> dict[str, object]:
        draft = load_plan(cast(SessionState, call.state))
        assert draft is not None
        assert plan.translation is not None and plan.workflow is not None
        save_plan(
            cast(SessionState, call.state),
            draft.with_translation(translation=plan.translation, workflow=plan.workflow),
        )
        return {"outcome": "ok"}

    candidates = [{"spec_id": _SPEC, "label": "external debt"}]
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(_ask_call("a1", candidates=candidates), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[_planned(pinned), {**_planned(pinned), "answer": {"result": {"spec_id": _SPEC}}}],
            setup=_proposals(),
            translate=translate,
        ),
        then=revise,
    )

    assert adk.state[_ELECTED_KEY] == [_SPEC]
    revised = adk.outputs[-1]
    assert revised is not None and "plan" in revised, revised
    revision = Plan.model_validate(revised["plan"])
    assert revision.status is PlanStatus.PENDING_APPROVAL
    sector = next(d for d in revision.decisions if d.slot == f"scope_sector_{_SPEC_NEW}")
    assert sector.resolved_by is not None and sector.resolved_by.value == "user"
    _assert_same(adk, pydantic)


# --- A Studio turn ------------------------------------------------------------------------


# The frame fields Studio fills with ids of its own drawing.
_DRAWN_IDS = ("interrupt_id", "plan_version_id")


def _frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The frames, less each line's duration, with each id Studio draws numbered in
    the order it first appears."""
    numbered: dict[str, str] = {}
    shown = []
    for frame in frames:
        kept = {k: v for k, v in frame.items() if k != "duration_ms"}
        for key in _DRAWN_IDS:
            if key in kept:
                kept[key] = numbered.setdefault(kept[key], f"{key}-{len(numbered)}")
        shown.append(kept)
    return shown


async def _studio(
    runner: Runner, plan: Plan, replies: Sequence[Reply], *, answer: dict[str, Any] | None = None
) -> list[list[dict[str, Any]]]:
    """The frames of a Studio turn whose orchestrator hands the question to the
    planner on `runner`, and of the answer that resumes it when `answer` is
    given."""
    scripted = ScriptedGenai(replies)
    seam = _seam(runner, scripted)
    _proposals(records=plan)(seam)
    app = _app(plan_call(_QUESTION), _text("Planned."), planner=seam)
    async with studio_client(app) as client:
        with _numbered_call_ids():
            first = await stream(client, "/api/studio/turn", turn_body(_QUESTION))
        streamed = [_frames(first)]
        if answer is not None:
            (elicitation,) = [f for f in first if f["event"] == "elicitation.request"]
            second = await stream(
                client,
                "/api/studio/interrupt",
                _interrupt_body(elicitation["interrupt_id"], "answer", answers={"q1": answer}),
            )
            streamed.append(_frames(second))
    return streamed


async def test_a_planned_studio_turn_streams_the_same_frames_on_both_planners(
    plan: Plan,
) -> None:
    replies = [reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)]

    adk = await _studio("adk", plan, replies)
    pydantic = await _studio("pydantic_ai", plan, replies)

    assert "plan.proposed" in [frame["event"] for frame in adk[0]]
    assert pydantic == adk


async def test_a_studio_ask_and_its_answer_stream_the_same_frames_on_both_planners(
    plan: Plan,
) -> None:
    candidates = [
        {"spec_id": "spec_a", "label": "reading A"},
        {"spec_id": "spec_b", "label": "reading B"},
    ]
    replies = [
        reply(_ask_call("a1", candidates=candidates), usage=_USAGE),
        reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
    ]

    adk = await _studio("adk", plan, replies, answer=_pick("spec_b"))
    pydantic = await _studio("pydantic_ai", plan, replies, answer=_pick("spec_b"))

    assert adk[0][-1]["event"] == "elicitation.request"
    assert "plan.proposed" in [frame["event"] for frame in adk[1]]
    assert pydantic == adk
