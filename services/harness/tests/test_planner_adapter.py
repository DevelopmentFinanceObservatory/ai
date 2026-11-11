"""Tests for `harness.planner_adapter`, the adapter over the planner
(`harness.planner_node`) with translation attached to its `propose_plan`.

Model-free. A scripted google-genai client (`tests.scripted_genai.ScriptedGenai`)
answers the planner's model calls (`tests.scripted_planner.scripted_planner`). Most
scenarios either hand the seam an already-compiled `Plan` (built the way
`tests/planner/test_agent.py`'s `_translate` helper builds one, through the
real `Plan.with_translation`) and give the planner the `propose_plan`
`harness.scripted.use_compiled_proposals` installs, so the scripted model's
one call declares it ready, or leave the plan a draft. The referral-stall tests below
script a bare `propose_plan` amendment, whose translation runs the real
translation loop against a canned surface. Whether the real model plans WELL is the eval sets' question
(`evals/README.md`).

`PlannerAdapter` runs the planner through `ctx.run_node`, in the graph's own
session, so every test below needs a real `Context`, not a bare call.
`_run_adapter` drives one, through a minimal one-node `Workflow`, the same
primitive `harness.graph`'s own planner node uses.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast, override

import pytest
from compiler.evidence import resolve_name
from compiler.sources import bound_refs
from compiler.surface import FixtureSurface, ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk import Context, Event
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.state import State
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from harness import plan_display, planner_adapter, planner_node
from harness.decision_pick import DecisionPick
from harness.graph import switch
from harness.graph.planning import planner_ask_interrupt
from harness.graph.state import (
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.orchestrator_tools import _search_context
from harness.planner_adapter import (
    _ASKED_REFERRAL_INTRO,
    _PLANNER_BRANCH,
    _SHARE_UNIT_CODES,
    PlannerAdapter,
    PlannerAdapterError,
    _check_branches,
    _plan_context,
    _skills_section,
)
from harness.planner_run import (
    _WHICH_INSTEAD,
    _WHICH_MEANT,
    PlannerRun,
    StallKey,
    _clarifying_question_args,
    _is_terminal_refusal,
    _pinnable_candidates,
    _stalled_referral,
)
from harness.scripted import use_compiled_proposals
from harness.sentences import (
    StopCode,
    referral_reason,
    referral_sentence,
    refusal_sentence,
)
from harness.studio_state import ConversationState
from harness.tool_contracts import reading_facts, resolver_digest
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.turn_events import PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME
from harness.turn_records import SearchReading
from harness.wiring import handing_back_planner_failures
from planner.call import elected_spec_ids
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    ALIGNMENTS_STATE_KEY,
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    CUT_MARKER,
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    OPTIONAL_DECLINES_STATE_KEY,
    REFUSED_SPECS_STATE_KEY,
    TERMINAL_DECLINES_STATE_KEY,
    UNSUPPORTED_STATE_KEY,
    WINDOWED_PINS_STATE_KEY,
    AggregateRequirement,
    AlignedInput,
    AnswerMode,
    AskAnswer,
    AskBreakdown,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    FullHistory,
    GrowthRequirement,
    JoinRequirement,
    Period,
    PinSource,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RefusedSpec,
    RequirementSide,
    ResolvedBy,
    SessionState,
    ShareOfRequirement,
    Unit,
    WholeOf,
    alignments,
    amendment_entry,
    current_discovery_time_left,
    current_earlier_block,
    decision_pick,
    discovery_time_left,
    full_histories,
    held_sources,
    load_plan,
    load_refused_specs,
    pin_sources,
    plan_revision,
    rolls_from_its_start,
    save_pin_sources,
    save_plan,
    save_refused_specs,
)
from planner.records import GroupBy, Reduction, RelativePeriod
from planner.session import SessionCall
from planner.settings import (
    AGENT_NAME,
    AGENT_THINKING_CONFIG,
)
from planner.skills import SKILLS_READ_STATE_KEY, skill_page_key
from planner.tool_results import ProblemCode, ok
from planner.tool_spec import ToolSpec
from planner.tools import load_pending_ask
from planner.workflow import (
    EvidenceBinding,
    ResolvedEntity,
    SelectedSource,
    ToolId,
    WorkflowStep,
)
from planner.workflow import Workflow as PlanWorkflow
from submission.service import (
    AlignedPin,
    save_aligned_pin,
    scope_choice_decisions,
)
from translator.discovery import current_cut_calls
from translator.mechanical import COMPILE_FAILURES_STATE_KEY
from translator.referral import ReferralKind

from .basis_alignment import (
    HEALTH,
    TOTAL,
    TOTAL_GROSS,
    TOTAL_NET,
    US_2023,
    BasisCompiles,
    WholeCompiles,
    basis_block,
)
from .scripted_genai import (
    Reply,
    Request,
    ScriptedGenai,
    Step,
    call_part,
    held,
    reply,
    text_part,
)
from .scripted_planner import SIGNED, SSE, scripted_planner
from .search_results import search_result_parts
from .split_plan import SPEC_DEBT, SPEC_DEBT_HISTORY, stored_split_plan
from .test_turn_records import _switchable_scope

SPEC = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32


# The call a scripted planner makes to have the plan on record declared ready,
# through the `propose_plan` `use_compiled_proposals` installs.
_DECLARE = [("propose_plan", {})]

_APP = "planner_adapter_test"
_USER = "u"


def _contents(request: Request) -> list[types.Content]:
    """The contents of one request the planner's model received."""
    return request.content_list


def _prompt_text(request: Any) -> str:
    """Every user-authored word the model was shown, joined."""
    return "\n".join(
        part.text or ""
        for content in _contents(request)
        if content.role == "user"
        for part in content.parts or []
    )


def _answers(request: Any, tool: str) -> list[dict[str, Any]]:
    """The answers to `tool`'s calls the model was shown in `request`, in order."""
    return [
        part.function_response.response or {}
        for content in _contents(request)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == tool
    ]


def _call_and_response_names(request: Any) -> list[str]:
    """`FC:<tool>` / `FR:<tool>` labels, in order, for every function call and
    function response part `request.contents` carries -- what tool-call
    history a generation actually saw, not just its text.
    """
    names: list[str] = []
    for content in _contents(request):
        for part in content.parts or []:
            if part.function_call is not None:
                names.append(f"FC:{part.function_call.name}")
            elif part.function_response is not None:
                names.append(f"FR:{part.function_response.name}")
    return names


def _last_amended_version(request: Any, *, default: int) -> int:
    """The version the most recent `propose_plan` answer in `request` recorded,
    or `default` when this generation has not seen one.

    `propose_plan` answers with `version`, and that number is where a model
    reads the `amends_version` its next call has to state. A request with no
    such answer in it yet is the turn's first proposal: the model has the
    version from the seam brief's own "version N" line, which `default`
    stands in for.
    """
    for content in reversed(_contents(request)):
        for part in content.parts or []:
            response = part.function_response
            if response is None or response.name != "propose_plan":
                continue
            version = (response.response or {}).get("version")
            assert isinstance(version, int), f"propose_plan answered without a version: {response}"
            return version
    return default


def _scripted_planner(
    scripted: ScriptedGenai,
    *,
    surface: FixtureSurface | None = None,
    compiled_proposals: bool = False,
) -> PlannerAdapter:
    """The `PlannerAdapter`, its model calls answered by `scripted`, over
    `surface` (an empty fixture surface by default). With `compiled_proposals`,
    `propose_plan` declares the plan on record ready
    (`harness.scripted.use_compiled_proposals`)."""
    return scripted_planner(
        scripted,
        surface=surface,
        tools=use_compiled_proposals if compiled_proposals else None,
    )


# Call ids for `_calls`, unique across the module's scripts.
_CALL_IDS = itertools.count(1)


def _calls(*calls: tuple[str, Mapping[str, Any]]) -> Reply:
    """A reply of one function call per `(tool, args)` pair, each with a call id
    of its own and the first signed, as Gemini answers after thinking."""
    return reply(
        *(
            call_part(
                name,
                dict(args),
                id=f"call-{next(_CALL_IDS)}",
                signature=SIGNED if position == 0 else None,
            )
            for position, (name, args) in enumerate(calls)
        )
    )


def _says(text: str) -> Reply:
    """A reply of answer text, which ends a planner run."""
    return reply(text_part(text))


def _computed(step: Callable[[Request], Sequence[tuple[str, Mapping[str, Any]]] | str]) -> Step:
    """A reply `step` computes from the request it answers: `(tool, args)`
    calls, as `_calls` takes them, or answer text."""

    def answer(request: Request) -> Reply:
        made = step(request)
        return _says(made) if isinstance(made, str) else _calls(*made)

    return answer


class _AdapterHarness:
    """Drives `PlannerAdapter.__call__` through a real, dynamically-scheduled
    `Context` -- the same primitive `harness.graph`'s own planner node
    drives `PlannerAdapter` through, and the only way to get a `Context`
    `ctx.run_node` will accept, since this seam has no `Runner` of its
    own to call directly.

    One `_AdapterHarness` reuses the same session across every `run` call,
    the way the graph's own session persists across a turn's several
    entries and across separate turns of one conversation alike -- so a
    test can inspect `ctx.state` (`state()`) after a call, without losing
    what it left behind.
    """

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._sessions = InMemorySessionService()
        self._session_id: str | None = None
        self._initial_state = dict(state or {})
        self._live_state: Mapping[str, Any] = {}

    async def run(
        self,
        adapter: PlannerAdapter,
        *,
        resume_inputs: dict[str, Any] | None = None,
        new_message: types.Content | None = None,
        stoppable: bool = False,
        run_config: RunConfig = SSE,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Run `adapter` once in this harness's session, under the switch's
        `_unless_stopped` when `stoppable`, as the planner node runs a pick's
        replan, and with `run_config`, Studio's streaming one by default, which
        a model on `ScriptedGenai` needs."""

        async def _node(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
            del node_input
            self._live_state = ctx.session.state
            if resume_inputs is not None:
                # `ctx.resume_inputs` reflects a genuine pause-and-resume of
                # THIS node's own interrupt, the planner's ask this node
                # pauses on (below) included. Setting it directly here is
                # only for a test that needs the OTHER resume condition --
                # "has `__call__` already been through this session once" --
                # without paying for a real interrupt round trip, the same
                # way a hand-built `Context` elsewhere in this suite stands
                # in for one the framework would otherwise build.
                ctx._resume_inputs = resume_inputs
            drafting = adapter(ctx=ctx, **kwargs)
            drafted = await (switch._unless_stopped(ctx, drafting) if stoppable else drafting)
            # The pause the graph's planner node makes on an accepted ask.
            if isinstance(drafted, dict) and "asked" in drafted:
                yield planner_ask_interrupt(drafted["asked"])
                return
            yield Event(output=drafted)

        node_fn = node(_node, name="adapter_under_test", rerun_on_resume=True)
        wf = Workflow(name="adapter_test_wf", edges=[(START, node_fn)])
        runner = Runner(node=wf, app_name=_APP, session_service=self._sessions)
        if self._session_id is None:
            session = await self._sessions.create_session(
                app_name=_APP, user_id=_USER, state=self._initial_state
            )
            self._session_id = session.id
        content = new_message or types.Content(role="user", parts=[types.Part(text="probe")])
        output: Any = None
        async for event in runner.run_async(
            user_id=_USER, session_id=self._session_id, new_message=content, run_config=run_config
        ):
            assert isinstance(event, Event)
            if event.output is not None:
                output = event.output
        return cast(dict[str, Any] | None, output)

    async def restore(self, events: Sequence[Event], state: Mapping[str, Any]) -> None:
        """Make `events`, on `state`, this harness's session, for a test that
        resumes a session another build stored."""
        assert self._session_id is None, "run() has been called already"
        session = await self._sessions.create_session(
            app_name=_APP, user_id=_USER, state=dict(state)
        )
        self._session_id = session.id
        for event in events:
            await self._sessions.append_event(session, event)

    async def state(self) -> SessionState:
        assert self._session_id is not None, "run() has not been called yet"
        session = await self._sessions.get_session(
            app_name=_APP, user_id=_USER, session_id=self._session_id
        )
        assert session is not None
        return cast(SessionState, session.state)

    async def events(self) -> list[Event]:
        """This session's own event history -- what `planner.call.
        elected_spec_ids` reads to find what the user actually chose
        answering the planner's asks, and what `ctx.run_node`'s own
        pause-and-resume depends on surviving (`harness.planner_adapter`'s
        module docstring)."""
        assert self._session_id is not None, "run() has not been called yet"
        session = await self._sessions.get_session(
            app_name=_APP, user_id=_USER, session_id=self._session_id
        )
        assert session is not None
        return session.events

    def live_state(self) -> Mapping[str, Any]:
        """The session state the last `run()` read and wrote in process, which
        holds writes no event carried to the store."""
        return self._live_state

    async def append_event(self, ctx: Context, event: Event) -> None:
        """Append `event` to the session a running `ctx` holds, for a test that
        needs an event on record no planner builds."""
        await self._sessions.append_event(ctx.session, event)

    def seed_state(self, key: str, value: Any) -> None:
        """Write directly into the canonical stored session, for a test that
        needs to set up state no real call would produce -- `get_session`
        (`state`, above) hands back a shallow copy (`InMemorySessionService`'s
        own `_copy_session`), so a write through it is invisible to every
        later `run()`.
        """
        assert self._session_id is not None, "run() has not been called yet"
        self._sessions.sessions[_APP][_USER][self._session_id].state[key] = value


async def _run_adapter(adapter: PlannerAdapter, **kwargs: Any) -> dict[str, Any]:
    """One-shot `_AdapterHarness.run`, for a test that only makes one call and
    does not need to inspect `ctx.state` afterward."""
    output = await _AdapterHarness().run(adapter, **kwargs)
    assert output is not None, "the seam node produced no output -- did it raise silently?"
    return output


def _translated_plan(
    decision: Any, requirement: Any, derived_requirement: Any, *, version: int = 1
) -> Plan:
    """A `PENDING_APPROVAL` plan, built through the real `with_translation`
    transition -- the minimal shape a graph node and `declare_ready` alike
    accept -- mirroring `tests/planner/test_agent.py`'s `_translate` helper.
    """
    made_decision: Decision = decision()
    dataset = requirement()
    derived = derived_requirement()
    draft = Plan(
        question="kenya debt",
        decisions=[made_decision],
        evidence_requirements=[dataset, derived],
        version=version,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id=dataset.id, spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id=derived.id,
                input="debt",
                reduction=derived.reduction,
                group_by=derived.group_by,
            ),
        ],
        terminal=["totals"],
    )
    workflow = PlanWorkflow(
        steps=[
            WorkflowStep(
                id="debt",
                operation_id="debt",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "debt", "output": "frame_ref"}},
                input_schema_hash="hash",
            ),
        ],
        sources=[SelectedSource(spec_id=SPEC, sentence="external debt service", has_data=True)],
        evidence_bindings=[
            EvidenceBinding(requirement_id=dataset.id, step_id="debt", spec_id=SPEC)
        ],
        terminal_steps=["totals"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


async def test_the_adapter_returns_a_plan_a_graph_node_can_validate(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai([_calls(*_DECLARE)])
    adapter = _scripted_planner(scripted, compiled_proposals=True)

    drafted = await _run_adapter(
        adapter, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
    )

    # The node's own validation (`Plan.model_validate(drafted["plan"])` in
    # `harness.graph.planning`'s `planner` node), repeated here as the contract this seam owes it.
    validated = Plan.model_validate(drafted["plan"])
    assert validated.version == plan.version
    assert validated.status is PlanStatus.PENDING_APPROVAL
    assert drafted["summary"]["revision"] == plan_revision(plan)
    assert drafted["summary"]["version"] == plan.version


async def test_on_progress_reports_the_time_since_the_previous_round_trip(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Each `on_progress` report's `duration_ms` covers everything since the
    previous report (or since the run started, for the first) -- the
    number that lets a ten-thousand-token thinking round read differently
    from a one-second lookup on a feed where every line otherwise prints
    the same way the instant it arrives."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai([_calls(*_DECLARE)])
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    reports: list[dict[str, Any]] = []

    await _run_adapter(
        adapter,
        question=plan.question,
        readings=[],
        prior_plan=plan.model_dump(mode="json"),
        on_progress=reports.append,
    )

    assert [(report["agent"], report["tools"]) for report in reports] == [
        ("dfo_planner", ["propose_plan"])
    ]
    assert isinstance(reports[0]["duration_ms"], int)
    assert reports[0]["duration_ms"] >= 0


async def test_a_prior_plan_is_what_propose_plan_amends_from(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment starts from the graph's own record, not whatever this
    seam last left in the session: seeding `prior_plan` through
    `save_plan` on every call means the second call's version-5 record,
    not the first call's version-1 draft, is what the amendment builds on
    -- even though both calls share the same session.

    The second call states `amends_version=5`, the version the seed put on
    record. `propose_plan` reads intent off the call, never off a question's
    wording, so the seed is what makes that version the right one to name: a
    session still holding the first call's version-1 draft refuses this call
    as amending the wrong version. What records version 6 is therefore the
    graph's own seeded record, not what this seam left in the session.
    """
    harness = _AdapterHarness()
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }
    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", proposal)),
            _says("done"),
            _calls(("propose_plan", {**proposal, "amends_version": 5})),
            _says("done"),
        ]
    )
    adapter = _scripted_planner(scripted)

    # A draft with no translation is never declared ready, so this call
    # ends with no plan -- it exists only to leave a version-1 plan in the
    # session, to prove the second call does not read it back.
    await harness.run(adapter, question="kenya debt", readings=[])
    left_behind = load_plan(await harness.state())
    assert left_behind is not None
    assert left_behind.version == 1

    seeded = _translated_plan(decision, requirement, derived_requirement, version=5)
    await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="only since 2015",
    )

    amended = load_plan(await harness.state())
    assert amended is not None
    assert amended.version == 6
    assert amended.previous_version == 5


async def test_a_model_that_stops_without_a_plan_ends_the_entry_on_plan_not_ready() -> None:
    """A run that ends with no plan declared ready and no referral on record
    ends the entry at once, on a refusal another plan may still answer."""
    scripted = ScriptedGenai([_says("Which years did you have in mind?")])
    adapter = _scripted_planner(scripted)

    output = await _run_adapter(adapter, question="kenya debt", readings=[])

    assert output["refused"] == {
        "text": refusal_sentence({"code": StopCode.PLAN_NOT_READY}),
        "final": False,
        "code": StopCode.PLAN_NOT_READY,
    }
    assert len(scripted.requests) == 1


class _FailingSkillSurface(FixtureSurface):
    """A data-mcp surface whose every call raises an exception `get_skill`
    does not anticipate, the way a bug below a planner tool would."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        del kwargs
        raise RuntimeError("http://internal.example/skills exploded")


async def test_a_planner_tool_that_raises_ends_the_entry_on_a_final_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run stops at the failed call and the orchestrator gets a final
    refusal naming the tool, with the traceback logged at ERROR."""
    scripted = ScriptedGenai([_calls(("get_skill", {"skill_id": "aid"})), _says("unreachable")])
    adapter = _scripted_planner(scripted, surface=_FailingSkillSurface(schemas=[]))

    harness = _AdapterHarness()

    with caplog.at_level(logging.ERROR):
        output = await harness.run(
            cast(PlannerAdapter, handing_back_planner_failures(adapter)),
            question="kenya debt",
            readings=[],
        )

    assert output is not None
    assert output["refused"]["final"] is True
    assert "get_skill" in output["refused"]["text"]
    assert len(scripted.requests) == 1
    (response,) = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content else None) or []
        if part.function_response is not None and part.function_response.name == "get_skill"
    ]
    assert "internal.example" not in json.dumps(response)
    (record,) = [
        r
        for r in caplog.records
        if r.name == planner_node.logger.name and r.levelno == logging.ERROR
    ]
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    assert record.args is not None and record.args[:2] == ("get_skill", "RuntimeError")


async def test_an_unknown_tool_name_is_answered_with_the_tools_the_model_can_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scripted = ScriptedGenai([_calls(("no_such_tool", {})), _says("done")])
    adapter = _scripted_planner(scripted)

    with caplog.at_level(logging.ERROR):
        output = await _run_adapter(adapter, question="kenya debt", readings=[])

    assert output["refused"]["final"] is False

    (response,) = _answers(scripted.requests[1], "no_such_tool")
    assert "The tools you can call are" in response["error"]
    assert not [
        r
        for r in caplog.records
        if r.name == planner_node.logger.name and r.levelno == logging.ERROR
    ]


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("entry", ["first-draft", "amendment", "replan"])
async def test_an_entry_owes_nothing_an_earlier_entry_recorded(resumed: bool, entry: str) -> None:
    """Fresh entries clear old declines and failures; only first drafts permit an
    unearned terminal decline. A user-input resume keeps the entry's state."""
    stale = {
        UNSUPPORTED_STATE_KEY: ["calculation_over_combined"],
        OPTIONAL_DECLINES_STATE_KEY: ["no_operation_covers"],
        TERMINAL_DECLINES_STATE_KEY: ["no_operation_covers"],
        COMPILE_FAILURES_STATE_KEY: ["fingerprint"],
        REFUSED_SPECS_STATE_KEY: [SPEC],
    }
    harness = _AdapterHarness(state=stale)
    adapter = _scripted_planner(ScriptedGenai([_says("Which years did you have in mind?")]))
    kwargs: dict[str, Any] = {}
    if entry == "amendment":
        kwargs["amendment"] = "show it"
    elif entry == "replan":
        kwargs["refusal"] = _unit_refusal("cross_frame_unit_mismatch")

    await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        resume_inputs={"ask": {"result": "since 2015"}} if resumed else None,
        **kwargs,
    )

    state = await harness.state()
    kept = {key: state.get(key) for key in stale}
    fresh = {
        UNSUPPORTED_STATE_KEY: [],
        OPTIONAL_DECLINES_STATE_KEY: ["no_operation_covers"] if entry == "first-draft" else [],
        TERMINAL_DECLINES_STATE_KEY: ["no_operation_covers"] if entry == "first-draft" else [],
        COMPILE_FAILURES_STATE_KEY: [],
        REFUSED_SPECS_STATE_KEY: [] if entry == "first-draft" else [SPEC],
    }
    assert kept == (stale if resumed else fresh)


async def test_a_question_with_nothing_left_after_a_decline_ends_on_the_decline(
    decision: Any, requirement: Any
) -> None:
    """The planner hears `unsupported` for a total over a join, is refused a plan
    that drops the total without a decline, then proposes the decline alone. That
    ends the entry with a final refusal naming the declined ask: no further
    generation and no plan."""
    made_decision = decision().model_dump(mode="json")
    bilateral = requirement(id="bilateral").model_dump(mode="json")
    multilateral = requirement(id="multilateral", measure="imputed multilateral aid")
    summed_join = [
        bilateral,
        multilateral.model_dump(mode="json"),
        {
            "kind": "join",
            "id": "joined",
            "decision_ids": ["measure"],
            "left_id": "bilateral",
            "right_id": "multilateral",
        },
        {
            "kind": "aggregate",
            "id": "total",
            "decision_ids": ["measure"],
            "input_id": "joined",
            "reduction": "sum",
            "group_by": ["period"],
        },
    ]
    first_draft = {"question": "total aid", "answer_mode": "computed", "decisions": [made_decision]}
    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", {**first_draft, "evidence_requirements": summed_join})),
            _calls(
                (
                    "propose_plan",
                    {**first_draft, "answer_mode": "series", "evidence_requirements": [bilateral]},
                )
            ),
            _calls(
                (
                    "propose_plan",
                    {
                        **first_draft,
                        "declined": [{"ask": "the total aid", "code": "calculation_over_combined"}],
                    },
                )
            ),
        ]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    result = await harness.run(adapter, question="total aid", readings=[])

    assert result is not None
    assert result["refused"]["final"] is True
    assert "the total aid" in result["refused"]["text"]
    assert len(scripted.requests) == 3
    answered = _answers(scripted.requests[-1], "propose_plan")
    assert [(r["outcome"], r["code"]) for r in answered] == [
        ("unsupported", "operation_unsupported"),
        ("invalid_input", "decline_missing"),
    ]
    assert load_plan(await harness.state()) is None


async def test_the_brief_lists_the_asks_and_propose_plan_holds_the_plan_to_them(
    decision: Any, requirement: Any
) -> None:
    """The asks in state render numbered after the question; a draft that leaves one
    unaccounted for is refused naming it, a decline no `unsupported` answer earned is
    refused, and the redraft that declines it once one has records the mapping on
    the plan."""
    asks = ["bilateral aid to Guatemala", "the total of bilateral and multilateral aid"]
    first_draft = {
        "question": "total aid",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [requirement(id="bilateral").model_dump(mode="json")],
        "answered": [{"ask": 0, "outputs": ["bilateral"]}],
    }
    decline = {"ask": asks[1], "code": "calculation_over_combined", "ask_ref": 1}
    multilateral = requirement(id="multilateral", measure="imputed multilateral aid")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="bilateral", right_id="multilateral"
    )
    summed = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id="joined",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    summed_join = {
        **first_draft,
        "evidence_requirements": [
            *first_draft["evidence_requirements"],
            *(r.model_dump(mode="json") for r in (multilateral, joined, summed)),
        ],
    }
    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", first_draft)),
            _calls(("propose_plan", {**first_draft, "declined": [decline]})),
            _calls(("propose_plan", summed_join)),
            _calls(("propose_plan", {**first_draft, "declined": [decline]})),
            _says("Planned."),
        ]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness(state={ASKS_STATE_KEY: asks})

    output = await harness.run(adapter, question="total aid", readings=[])

    assert output is not None and output["refused"]["final"] is False

    assert "0. bilateral aid to Guatemala\n1. the total of bilateral" in _prompt_text(
        scripted.requests[0]
    )
    answered = _answers(scripted.requests[-1], "propose_plan")
    assert [(r["outcome"], r.get("code")) for r in answered] == [
        ("invalid_input", "asks_unaccounted"),
        ("invalid_input", "decline_unearned"),
        ("unsupported", "operation_unsupported"),
        ("ok", None),
    ]
    assert answered[0]["detail"] == {"uncovered": [1], "repeated": []}
    recorded = load_plan(await harness.state())
    assert recorded is not None
    assert recorded.asks == tuple(asks)
    assert [(a.ask, a.outputs) for a in recorded.answered] == [(0, ("bilateral",))]
    assert [d.ask_ref for d in recorded.declined] == [1]


def _unit_refusal(code: str, *, tool: str = "compare") -> dict[str, Any]:
    """A `MethodChoiceNeeded` replan payload (`harness.graph.execute`'s `_replan_refusal`)
    for `tool` refusing a pair of sides on their units."""
    return {
        "kind": "method_choice_needed",
        "step_id": "per_person",
        "tool": tool,
        "code": code,
        "candidates": None,
        "refusal": {"code": code, "message": "refused", "retry_mode": "correct_and_retry"},
        "also_stopped": [],
    }


@pytest.mark.parametrize(
    ("refusal", "outcome"),
    [
        pytest.param(_unit_refusal("cross_frame_unit_unbound"), ("ok", None), id="no-operation"),
        pytest.param(
            {
                **_unit_refusal("growth_gap_year"),
                "also_stopped": [_unit_refusal("cross_frame_unit_unbound")],
            },
            ("ok", None),
            id="sibling",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_mismatch"),
            ("invalid_input", "decline_unearned"),
            id="method-choice",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_unbound", tool="sum_of"),
            ("ok", None),
            id="no-operation-sum",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_unbound", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-unbound",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_mismatch", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-mismatch",
        ),
        pytest.param(
            {
                **_unit_refusal("cross_frame_unit_unbound", tool="share_of"),
                "also_stopped": [_unit_refusal("cross_frame_unit_unbound")],
            },
            ("ok", None),
            id="share-with-a-compare-sibling",
        ),
        pytest.param(
            _unit_refusal("cross_frame_facet_mismatch", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-facet-mismatch",
        ),
        pytest.param(
            _unit_refusal("cross_frame_facet_mismatch"),
            ("invalid_input", "decline_unearned"),
            id="compare-facet-mismatch",
        ),
    ],
)
async def test_a_replan_over_a_refusal_no_operation_covers_earns_the_decline(
    decision: Any,
    requirement: Any,
    refusal: dict[str, Any],
    outcome: tuple[str, Any],
) -> None:
    """A replan entry whose refusal means no operation computes the step may
    decline the ask it served as `no_operation_covers`. With a refusal that
    offers a choice, the decline is unearned, and the refusal steers to
    no other output. A `share_of` step refusing its sides on their units earns
    none either: the replan names `quotient_of` as the repair. A step reading one
    measure from two sources earns none, and the replan names one source for it."""
    asks = ["GDP", "population", "GDP divided by population"]
    draft = {
        "question": "GDP per person",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement(id="gdp").model_dump(mode="json"),
            requirement(id="people", measure="population").model_dump(mode="json"),
        ],
        "answered": [{"ask": 0, "outputs": ["gdp"]}, {"ask": 1, "outputs": ["people"]}],
        "declined": [{"ask": asks[2], "code": "no_operation_covers", "ask_ref": 2}],
    }
    scripted = ScriptedGenai([_calls(("propose_plan", draft)), _says("Planned.")])
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness(state={ASKS_STATE_KEY: asks})

    await harness.run(adapter, question="GDP per person", readings=[], refusal=refusal)

    answered = _answers(scripted.requests[-1], "propose_plan")
    assert [(r["outcome"], r.get("code")) for r in answered] == [outcome]
    prompt = _prompt_text(scripted.requests[0])
    assert ("`no_operation_covers`" in prompt) is (outcome[0] == "ok")
    shares = refusal["tool"] == "share_of"
    assert ("`quotient_of`" in prompt) is (shares and refusal["code"] in _SHARE_UNIT_CODES)
    facets = refusal["code"] == "cross_frame_facet_mismatch"
    assert ("give every requirement on that measure the same source" in prompt) is facets
    if outcome[0] != "ok":
        assert "action" not in answered[0]
        assert "answer the ask" not in answered[0]["message"]


@pytest.mark.parametrize("entry", ["first-draft", "amendment"])
async def test_a_first_draft_may_decline_an_ask_no_requirement_kind_states(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    entry: str,
) -> None:
    """A first draft may decline as `no_operation_covers` an ask no requirement
    kind states, with no tool's `unsupported` answer before it, and plan what
    can still be shown beside it. An amendment entry earns no such decline."""
    asks = ["the donors whose aid rose in every one of the last five years"]
    draft = {
        "question": "donors whose aid rose every year",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [requirement(id="aid").model_dump(mode="json")],
        "answered": [{"ask": 0, "outputs": ["aid"]}],
        "declined": [{"ask": asks[0], "code": "no_operation_covers", "ask_ref": 0}],
    }
    kwargs: dict[str, Any] = {}
    if entry == "amendment":
        seeded = _translated_plan(decision, requirement, derived_requirement, version=1)
        kwargs = {"prior_plan": seeded.model_dump(mode="json"), "amendment": "show it"}
        draft = {
            **{key: value for key, value in draft.items() if key != "question"},
            "amends_version": 1,
        }
    scripted = ScriptedGenai([_calls(("propose_plan", draft)), _says("Planned.")])
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness(state={ASKS_STATE_KEY: asks})

    await harness.run(adapter, question="donors whose aid rose every year", readings=[], **kwargs)

    answered = _answers(scripted.requests[-1], "propose_plan")
    if entry == "first-draft":
        assert [(r["outcome"], r.get("code")) for r in answered] == [("ok", None)]
        recorded = load_plan(await harness.state())
        assert recorded is not None
        assert [(d.code.value, d.ask_ref) for d in recorded.declined] == [
            ("no_operation_covers", 0)
        ]
        assert [(a.ask, a.outputs) for a in recorded.answered] == [(0, ("aid",))]
    else:
        assert [(r["outcome"], r.get("code")) for r in answered] == [
            ("invalid_input", "decline_unearned")
        ]


@pytest.mark.parametrize("paused", [False, True])
async def test_a_first_draft_with_only_declines_ends_on_the_declined_asks(paused: bool) -> None:
    """A first-draft decline can close the entry, including after a user-input pause."""
    asks = ["the donors whose aid rose in every one of the last five years"]
    draft = {
        "question": "donors whose aid rose every year",
        "answer_mode": "series",
        "declined": [{"ask": asks[0], "code": "no_operation_covers", "ask_ref": 0}],
    }
    script = [_calls(("propose_plan", draft))]
    if paused:
        script.insert(0, _calls((PLANNER_ASK, {"message": "Which period?"})))
    scripted = ScriptedGenai(script)
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness(state={ASKS_STATE_KEY: asks})

    result = await harness.run(adapter, question=draft["question"], readings=[])
    if paused:
        assert result is None
        interrupt_id = _pending_ask_interrupt_id(await harness.events())
        result = await harness.run(
            adapter,
            question=draft["question"],
            readings=[],
            new_message=types.Content(
                role="user",
                parts=[create_request_input_response(interrupt_id, {"result": "last five years"})],
            ),
        )

    assert result is not None
    assert result["refused"]["final"] is True
    assert asks[0] in result["refused"]["text"]
    assert len(scripted.requests) == (2 if paused else 1)
    assert load_plan(await harness.state()) is None


# -- asking the user (`ask_user_question`) ---------------------------------


async def test_a_planner_ask_pauses_the_run_and_elects_nothing_until_answered() -> None:
    """The planner's `ask_user_question` call, accepted, ends the
    planner's run, and the seam answers the ask, which the node pauses on, so
    the node resolves no output. The call's own function-call event is on
    record afterward, in the session's own event history
    (`planner.session.session_steps`'s own docstring). But an offer is not an
    election: with no answer from the user on record, nothing the call
    offered counts as something the user chose.
    """
    scripted = ScriptedGenai(
        [
            _calls(
                (
                    PLANNER_ASK,
                    {
                        "message": "Which source did you mean?",
                        "candidates": [
                            {"spec_id": SPEC, "label": "a candidate"},
                            {"spec_id": SPEC_B, "label": "another candidate"},
                        ],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            )
        ]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    output = await harness.run(adapter, question="kenya debt", readings=[])

    assert output is None, "a pausing turn resolves no output -- the run is suspended, not done"
    events = await harness.events()
    fake_tool_context = SimpleNamespace(
        state={}, function_call_id=None, session=SimpleNamespace(events=events)
    )
    assert elected_spec_ids(SessionCall.of(fake_tool_context)) == frozenset()


async def test_a_slow_call_beside_an_accepted_ask_answers_before_the_pause_and_runs_once() -> None:
    """The planner's run ends on its accepted ask only once every call of the
    same response has answered, with no model call after. A slow
    `propose_plan` made beside the ask answers before the pause, and the run
    the answer resumes does not run it again: one proposal is on record."""
    proposals: list[int] = []

    async def propose_plan(*_: Any) -> dict[str, object]:
        # Slower than the ask's own call, which answers at once.
        await asyncio.sleep(0.05)
        proposals.append(1)
        return ok(status="draft", version=1)

    def slow_proposals(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        slow = ToolSpec(
            name="propose_plan", description="Propose the plan.", parameters=(), body=propose_plan
        )
        return [slow if spec.name == "propose_plan" else spec for spec in specs]

    scripted = ScriptedGenai(
        [_calls((PLANNER_ASK, {"message": "Which years?"}), ("propose_plan", {})), _says("stop")]
    )
    adapter = scripted_planner(scripted, tools=slow_proposals)
    harness = _AdapterHarness()

    assert await harness.run(adapter, question="kenya debt", readings=[]) is None
    assert len(scripted.requests) == 1
    events = await harness.events()
    answered = [
        at
        for at, event in enumerate(events)
        for response in event.get_function_responses()
        if response.name == "propose_plan"
    ]
    (paused,) = [
        at
        for at, event in enumerate(events)
        for call in event.get_function_calls()
        if call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    ]
    assert proposals == [1]
    assert len(answered) == 1 and answered[0] < paused

    answer = types.Content(
        role="user",
        parts=[
            create_request_input_response(
                _pending_ask_interrupt_id(events), {"result": {"text": "since 2015"}}
            )
        ],
    )
    await harness.run(adapter, question="kenya debt", readings=[], new_message=answer)

    assert proposals == [1]
    assert len(scripted.requests) == 2


async def test_a_terminal_refusal_beside_an_accepted_ask_ends_the_entry_on_the_refusal(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A proposal refused `QUESTION_FIXED` beside an accepted ask ends the
    entry: no answer could undo the refusal, so the entry ends on it and
    leaves no ask waiting."""
    seeded = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai(
        [
            _calls(
                (PLANNER_ASK, {"message": "Which years?"}),
                ("propose_plan", {"starts_new_plan": True, "question": "a different question"}),
            )
        ]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    result = await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result is not None
    assert result["refused"]["code"] == ProblemCode.QUESTION_FIXED
    assert load_pending_ask(await harness.state()) is None


async def test_an_ask_after_a_refused_ask_in_the_same_response_is_refused_too() -> None:
    """The first ask of a response offers options it wrote itself, which its
    tool refuses. The second, valid, is refused as a second ask: the response
    puts no question to the user, and the entry ends without a plan."""
    authored = {
        "message": "Constant or current dollars?",
        "options": [{"spec_id": "constant_2024_usd", "label": "Constant 2024 US dollars"}],
    }
    scripted = ScriptedGenai(
        [
            _calls((PLANNER_ASK, authored), (PLANNER_ASK, {"message": "Which years?"})),
            _says("done"),
        ]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    result = await harness.run(adapter, question="kenya debt", readings=[])

    assert result is not None
    assert result["refused"]["code"] == StopCode.PLAN_NOT_READY
    assert load_pending_ask(await harness.state()) is None
    answers = [
        (response.response or {}).get("code", (response.response or {}).get("status"))
        for event in await harness.events()
        for response in event.get_function_responses()
        if response.name == PLANNER_ASK
    ]
    assert answers == [ProblemCode.INVALID_ARGUMENTS, ProblemCode.INVALID_ARGUMENTS]


async def test_an_entry_that_raises_after_an_accepted_ask_leaves_no_ask_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner's run accepts an ask, and the branch check on its events
    then raises, which the served planner hands back as a refusal
    (`harness.wiring.handing_back_planner_failures`). The entry ends on it, so
    the state it leaves holds no ask for a resume or a reader of the state to
    find."""

    def stray(events: Any, **kwargs: Any) -> None:
        del events, kwargs
        raise PlannerAdapterError("a planner event carries another branch")

    monkeypatch.setattr(planner_adapter, "_check_branches", stray)
    scripted = ScriptedGenai([_calls((PLANNER_ASK, {"message": "Which years?"}))])
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    result = await harness.run(
        cast(PlannerAdapter, handing_back_planner_failures(adapter)),
        question="kenya debt",
        readings=[],
    )

    assert result is not None and "refused" in result
    assert load_pending_ask(await harness.state()) is None


async def _pause_on_a_legacy_ask(harness: _AdapterHarness, question: dict[str, Any]) -> None:
    """Leave `harness`'s session paused on a long-running
    `REQUEST_INPUT_FUNCTION_CALL_NAME` call asking `question`, with no pending
    ask recorded. No planner builds that call, so this helper stands in for the
    seam and appends the call to the stored session."""
    call_id = "legacy-ask"
    call = types.FunctionCall(id=call_id, name=REQUEST_INPUT_FUNCTION_CALL_NAME, args=question)

    async def paused(*, ctx: Context, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        await harness.append_event(
            ctx,
            Event(
                author=AGENT_NAME,
                content=types.Content(role="model", parts=[types.Part(function_call=call)]),
                long_running_tool_ids={call_id},
            ),
        )
        return {"asked": {"call_id": call_id, "args": question}}

    assert (
        await harness.run(cast(PlannerAdapter, paused), question="kenya debt", readings=[]) is None
    )


async def test_an_answer_to_a_pause_from_before_the_node_level_ask_re_enters_the_planner() -> None:
    """A turn paused on a long-running `REQUEST_INPUT_FUNCTION_CALL_NAME` call
    holds no pending ask. The answer re-enters the planner in a fresh run with
    the question and the pick in its brief, as a node-level ask's does."""
    question = {
        "message": "Which source did you mean?",
        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
    }
    harness = _AdapterHarness()
    await _pause_on_a_legacy_ask(harness, question)
    assert load_pending_ask(await harness.state()) is None

    scripted = ScriptedGenai([_says("stop")])
    adapter = _scripted_planner(scripted)
    answer = types.Content(
        role="user",
        parts=[
            create_request_input_response(
                _pending_ask_interrupt_id(await harness.events()), {"result": {"spec_id": SPEC}}
            )
        ],
    )
    result = await harness.run(adapter, question="kenya debt", readings=[], new_message=answer)

    assert result is not None
    (request,) = scripted.requests
    assert _call_and_response_names(request) == []
    brief = _brief_follows_the_question(request)
    assert question["message"] in brief
    assert f"They picked 'a candidate', id `{SPEC}`." in brief


async def test_the_branch_check_runs_on_the_events_a_run_left_before_pausing_for_an_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that ends on a planner-side ask has the events it appended
    checked, as a run that ends on a plan does."""
    checked: list[list[Event]] = []
    real_check = planner_adapter._check_branches

    def spy(events: Any, **kwargs: Any) -> None:
        checked.append(list(events))
        real_check(events, **kwargs)

    monkeypatch.setattr(planner_adapter, "_check_branches", spy)
    scripted = ScriptedGenai([_calls((PLANNER_ASK, {"message": "Which source did you mean?"}))])
    adapter = _scripted_planner(scripted)

    output = await _AdapterHarness().run(adapter, question="kenya debt", readings=[])

    assert output is None
    assert len(checked) == 1, "the branch check did not run on the run that paused"
    assert {event.author for event in checked[0]} >= {AGENT_NAME}


def _referring_proposals(referral: dict[str, Any]) -> Callable[[Sequence[Any]], list[Any]]:
    """The planner's tools with a `propose_plan` that answers every proposal
    as recorded, its translation referring `referral` back."""

    def propose_plan(*_: Any) -> dict[str, object]:
        return ok(status="draft", version=1, translation=referral)

    def tools(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        referring = ToolSpec(
            name="propose_plan", description="Propose the plan.", parameters=(), body=propose_plan
        )
        return [referring if spec.name == "propose_plan" else spec for spec in specs]

    return tools


@pytest.mark.parametrize(
    ("referral", "candidates"),
    [
        (
            {
                "outcome": "invalid_input",
                "kind": "requirement_unmeetable",
                "source_mismatch": True,
                "requirement_id": "debt",
                "reason": "r",
                "candidates": [
                    {"spec_id": SPEC, "sentence": "debt service from IDS"},
                    {"spec_id": SPEC_B, "sentence": "debt service from the IMF"},
                ],
            },
            [
                {"spec_id": SPEC, "label": "debt service from IDS"},
                {"spec_id": SPEC_B, "label": "debt service from the IMF"},
            ],
        ),
        (
            {
                "outcome": "choice_required",
                "kind": "source_name_ambiguous",
                "requirement_id": "debt",
                "reason": "r",
                "ambiguous_sources": [
                    {
                        "text": "the OECD",
                        "candidates": [
                            {"ref": "oecd/crs", "label": "OECD CRS"},
                            {"ref": "oecd/dac1", "label": "OECD DAC1"},
                        ],
                    }
                ],
            },
            [
                {"spec_id": "oecd/crs", "label": "OECD CRS"},
                {"spec_id": "oecd/dac1", "label": "OECD DAC1"},
            ],
        ),
    ],
    ids=["unmeetable-with-candidates", "source-name-ambiguous"],
)
async def test_a_referral_the_user_decides_ends_the_run_on_the_adapters_own_ask(
    referral: dict[str, Any], candidates: list[dict[str, str]]
) -> None:
    """The translation refers a requirement back with candidates the user
    picks from, so the seam asks them itself: the run pauses on an
    `ask_user_question` call led by the referral's table sentence, and no
    model turn composes it."""
    scripted = ScriptedGenai([_calls(*_DECLARE), _says("never sent")])
    adapter = scripted_planner(scripted, tools=_referring_proposals(referral))
    harness = _AdapterHarness()

    output = await harness.run(adapter, question="kenya debt", readings=[])

    assert output is None, "the ask should pause this call, not resolve it"
    assert len(scripted.requests) == 1
    asks = [
        part.function_call.args
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_call is not None and part.function_call.name == PLANNER_ASK
    ]
    mismatch = referral.get("source_mismatch", False)
    lead = referral_sentence(referral["kind"], source_mismatch=mismatch)
    question = _WHICH_INSTEAD if mismatch else _WHICH_MEANT
    assert asks == [{"message": f"{lead} {question}", "candidates": candidates}]


def _pending_ask_interrupt_id(events: list[Event]) -> str:
    """The `interrupt_id` (the function call's own `id`) of the most recent
    `REQUEST_INPUT_FUNCTION_CALL_NAME` call on record, the pause the planner node makes on
    the planner's ask under the ask's call id -- what a real resume answers
    with `create_request_input_response`, the same way `harness.studio`'s own
    `/interrupt` route builds one for the wire equivalent."""
    for event in reversed(events):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            call = part.function_call
            if call is not None and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
                assert call.id is not None
                return call.id
    raise AssertionError("no request-input call on record")


async def test_a_picks_replan_run_as_a_task_still_pauses_on_an_ask_and_resumes(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The planner node runs a pick's replan under the switch's
    `_unless_stopped`, which runs the seam as a task so a Stop can reach it.
    A planner-side ask pauses the run through a real `ctx.run_node`,
    the pick survives the pause in durable state, the answer resumes and
    finishes the draft, and the run reports its proposal from inside the
    task."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai(
        [_calls((PLANNER_ASK, {"message": "Which source did you mean?"})), _calls(*_DECLARE)]
    )
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    harness = _AdapterHarness()
    reports: list[dict[str, Any]] = []
    entry: dict[str, Any] = {
        "question": plan.question,
        "readings": [],
        "prior_plan": plan.model_dump(mode="json"),
        "pick": _PICKED,
        "on_progress": reports.append,
        "stoppable": True,
    }

    paused = await harness.run(adapter, **entry)

    assert paused is None, "the ask should pause the replan, not resolve it"
    assert decision_pick(await harness.state()) == _PICKED
    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume = types.Content(
        role="user", parts=[create_request_input_response(interrupt_id, {"result": "2020"})]
    )
    resumed = await harness.run(adapter, new_message=resume, **entry)
    assert resumed is not None and resumed["plan"]["version"] == plan.version
    assert decision_pick(await harness.state()) == _PICKED
    assert [report["tools"] for report in reports] == [[PLANNER_ASK], ["propose_plan"]]


_BRIEF_MARKER = "Context for this question."


def _brief_follows_the_question(request: Any) -> str:
    """The one brief in `request`, after checking it directly follows the
    turn's question. That is the latest "probe", the message `_AdapterHarness`
    opens every run on."""
    texts = [
        "".join(part.text or "" for part in content.parts or []) for content in _contents(request)
    ]
    (at,) = [i for i, text in enumerate(texts) if _BRIEF_MARKER in text]
    question = max(i for i, text in enumerate(texts) if text == "probe")
    assert at == question + 1, f"the brief does not follow the question: {texts}"
    return texts[at]


async def test_a_resumed_ask_and_the_amend_after_it_each_carry_the_brief_once(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """ADK hands a node its input only on a live run, so a planner-side ask's
    resume would send the model no brief if the brief travelled as node
    input. The seam gives the brief to the planner's node
    (`harness.planner_node.planner_node`), which puts it after the turn's
    question on every request: the first request after the answer,
    which runs the planner again with the user's pick in its brief, and the
    amend call's, which renders the plan it changes and the amendment after
    the readings. No request carries a brief twice."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai(
        [
            _calls(
                (
                    PLANNER_ASK,
                    {
                        "message": "Which source did you mean?",
                        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ),
            _calls(*_DECLARE),
            _says("noted, but not done yet"),
        ]
    )
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    harness = _AdapterHarness()

    paused = await harness.run(
        adapter, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
    )
    assert paused is None, "the ask should pause this call, not resolve it"
    asked = len(scripted.requests)

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC}})],
    )
    resumed = await harness.run(
        adapter,
        question=plan.question,
        readings=[],
        prior_plan=plan.model_dump(mode="json"),
        new_message=resume_message,
    )
    assert resumed is not None, "a real answer should let the same turn finish"
    assert resumed["plan"]["version"] == plan.version

    answered = scripted.requests[asked]
    brief = _brief_follows_the_question(answered)
    assert f"Question: {plan.question}" in brief
    assert f"They picked 'a candidate', id `{SPEC}`." in brief
    assert _call_and_response_names(answered) == []

    amending = len(scripted.requests)
    await harness.run(
        adapter,
        question=plan.question,
        readings=[],
        prior_plan=resumed["plan"],
        amendment="add a caveat about revisions",
    )

    brief = _brief_follows_the_question(scripted.requests[amending])
    assert brief.index(f"Question: {plan.question}") < brief.index("add a caveat about revisions")
    assert f"version {plan.version}" in brief, "the amend brief should show the plan it changes"


# -- a replan's refusal ---------------------------------------------------


def _rediscovery_refusal(spec_id: str, code: str = "over_ceiling") -> dict[str, Any]:
    """A `RediscoveryNeeded` replan payload, the shape `harness.graph.execute`'s
    `_replan_refusal` builds, refusing `spec_id` for `code` with the
    `over_ceiling`/`member_cardinality` message -- the one refusal shape that
    names a `spec_id` at the top level today."""
    return {
        "kind": "rediscovery_needed",
        "step_id": "malaria_kenya",
        "tool": "data-mcp.get_data",
        "requirement_id": "malaria_kenya",
        "spec_id": spec_id,
        "refusal": {
            "code": code,
            "message": "resolved.stat_vars.member_count = 872 against a cap of 700",
            "retry_mode": "correct_and_retry",
            "fix": {"ceiling_kind": "member_cardinality"},
        },
    }


def test_a_wave_with_two_refused_specs_names_both() -> None:
    """`_replan_refusal`'s `also_stopped` carries a wave's other stops
    (`executor.run._Run.execute`'s siblings); a `RediscoveryNeeded` among
    them names a second refused spec this seam must treat the same as the
    primary's own: named in the `avoid` sentence, not just the first."""
    adapter = _scripted_planner(ScriptedGenai([]))
    refusal = _rediscovery_refusal(SPEC)
    refusal["also_stopped"] = [_rediscovery_refusal(SPEC_B)]

    message = adapter._message(
        question="malaria grants and population for kenya",
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    avoid = message.split("Call propose_plan again with a plan that avoids it.")[1]
    assert SPEC in avoid and SPEC_B in avoid
    assert "again" in avoid and "refused" in avoid


def test_a_replan_names_the_asks_the_failed_run_computed() -> None:
    """`computed_asks` (`harness.graph.execute._computed_asks`) becomes the sentence
    the replan opens with, and stays out of the refusal it quotes."""
    refusal = {**_rediscovery_refusal(SPEC), "computed_asks": [0, 2]}

    message = _scripted_planner(ScriptedGenai([]))._message(
        question="malaria grants for kenya",
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "failed after it computed asks 0, 2." in message
    assert "computed_asks" not in message


@pytest.mark.parametrize(
    ("entry", "dropped", "said", "unsaid"),
    [
        (
            "replan",
            [1],
            [
                "leave it out of `answered` and `declined`",
                "still answers every ask not marked (removed by the user).",
            ],
            ["to put it back"],
        ),
        (
            "amendment",
            [1],
            ["unless the user's change asks for it again; answer it in `answered` to put it back"],
            ["leave it out of `answered`"],
        ),
        ("replan_nothing_removed", [], ["still answers every ask."], ["(removed by the user)"]),
    ],
    ids=["replan", "amendment", "replan-nothing-removed"],
)
async def test_the_brief_words_each_ask_the_user_removed_for_its_entry(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    entry: str,
    dropped: list[int],
    said: list[str],
    unsaid: list[str],
) -> None:
    """The asks section marks the ask the user's amendment removed, and no
    other. A replan leaves it out, and its failed-run sentence holds the new
    plan to the asks left unmarked. The user's next amendment may put it back
    by answering it. With no ask removed, the failed-run sentence names no mark."""
    asks = ["Kenya's debt totals", "Uganda's debt totals"]
    plan = Plan.model_validate(
        {
            **_translated_plan(decision, requirement, derived_requirement).model_dump(),
            "asks": asks,
            "answered": [
                AskAnswer(ask=number, outputs=("annual_totals",))
                for number in range(len(asks))
                if number not in dropped
            ],
            "dropped_asks": dropped,
        }
    )
    replan = entry != "amendment"

    text, _ = await _first_request_with_readings(
        _turn_readings(),
        plan=plan,
        stored={ASKS_STATE_KEY: asks},
        refusal={**_rediscovery_refusal(SPEC), "computed_asks": [0]} if replan else None,
        amendment=None if replan else "add Uganda back",
    )

    marked = [line for line in text.splitlines() if line.endswith("(removed by the user)")]
    assert marked == (["1. Uganda's debt totals (removed by the user)"] if dropped else [])
    for clause in said:
        assert clause in text
    for clause in unsaid:
        assert clause not in text


# -- the conversation's search readings ------------------------------------

SPEC_C = "spec_" + "c" * 32


def _turn_readings() -> list[dict[str, Any]]:
    """A definite, a no_data, a candidates and a fetched chosen reading, as `search` stores them."""
    readings = [
        SearchReading.model_validate(
            {
                "text": "aid",
                "status": "definite",
                "spec_id": SPEC,
                "sentence": "Gross ODA to Guatemala, 2020 to 2024",
                "window": {"start": 2020, "end": 2024, "source": "coverage"},
                "recent_years": [
                    {"year": 2024, "observations": 734},
                    {"year": 2023, "observations": 1210},
                ],
                "warnings": [{"code": "role_scope_defaulted"}],
                "asked_roles": ["donor", "recipient"],
            }
        ),
        SearchReading(text="aid to Atlantis", status="no_data", refusal_code="role_not_resolved"),
        SearchReading.model_validate(
            {
                "text": "health aid",
                "status": "candidates",
                "candidates": [{"spec_id": SPEC_B, "label": "Health ODA"}],
            }
        ),
        SearchReading(text="health aid", status="chosen", spec_id=SPEC_C).with_fetch(
            {"spec_id": SPEC_C, "summary": "14 observations, 2010-2023"}
        ),
    ]
    return [reading.model_dump(mode="json") for reading in readings]


async def _first_request_with_readings(
    readings: list[dict[str, Any]],
    *,
    plan: Plan,
    stored: Mapping[str, Any] | None = None,
    refusal: dict[str, Any] | None = None,
    amendment: str | None = None,
) -> tuple[str, SessionState]:
    """Run one fresh orchestrator entry over `readings`, with `stored` (the
    blocks and warnings `search` keeps) already in session state, answering
    `refusal` or the user's `amendment`."""
    scripted = ScriptedGenai([_calls(*_DECLARE)])
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    harness = _AdapterHarness(stored)
    await harness.run(
        adapter,
        question=plan.question,
        prior_plan=plan.model_dump(mode="json"),
        readings=readings,
        refusal=refusal,
        amendment=amendment,
    )
    return _prompt_text(scripted.requests[0]), await harness.state()


async def test_the_orchestrator_planner_first_request_carries_every_search_reading(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)

    text, state = await _first_request_with_readings(_turn_readings(), plan=plan)

    for shown in (
        f'"aid" definite {SPEC}',
        "window 2020-2024 (coverage)",
        "warnings: role_scope_defaulted",
        '"aid to Atlantis" no data: role_not_resolved',
        f"candidate {SPEC_B}",
        f'"health aid" chosen {SPEC_C}',
        "fetched: 14 observations, 2010-2023",
    ):
        assert shown in text
    # The request omits the stored reading's year counts.
    assert "734" not in text
    assert list(pin_sources(state)) == [SPEC, SPEC_B, SPEC_C]


# When `SPEC`'s definite reading was observed, and a day before.
_SEARCHED_AT = datetime(2000, 1, 2, tzinfo=UTC)
_REFUSED_BEFORE = datetime(2000, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("refusal", "earlier", "kept"),
    [
        (None, None, [SPEC, SPEC_B, SPEC_C]),
        (_rediscovery_refusal(SPEC), None, [SPEC_B, SPEC_C]),
        (None, SPEC, [SPEC_B, SPEC_C]),
        (None, "open_role_unavailable", [SPEC_B, SPEC_C]),
        (None, "upstream_version_changed", [SPEC, SPEC_B, SPEC_C]),
    ],
    ids=[
        "no_refusal",
        "executor_refused_spec",
        "stored_bare",
        "refused_for_good",
        "refreshed_since",
    ],
)
async def test_every_shown_id_is_a_pin_source_settled_only_by_its_own_reading(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    refusal: dict[str, Any] | None,
    earlier: str | None,
    kept: list[str],
) -> None:
    """The definite and the fetched chosen reading settle their own ids; an id
    only offered as a candidate is shown unsettled. Each carries the block and
    warning codes `search` stored for it, and the parties its settling reading asked
    for. A spec the executor refused is left out and named in the brief's avoid
    sentence. `earlier` is `SPEC` refused before its definite reading was
    observed, stored bare or for a code, answered by an amendment: one refused
    for a code a refreshed definition recovers from is kept and named nowhere."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    stored: dict[str, Any] = {
        ORCHESTRATOR_SPECS_KEY: {
            spec_id: {"spec_id": spec_id} for spec_id in (SPEC, SPEC_B, SPEC_C)
        },
        ORCHESTRATOR_SPEC_WARNINGS_KEY: {SPEC: [{"code": "role_scope_defaulted"}]},
    }
    if earlier is not None:
        save_refused_specs(
            cast(SessionState, stored),
            [
                RefusedSpec(spec_id=SPEC)
                if earlier == SPEC
                else RefusedSpec(spec_id=SPEC, code=earlier, refused_at=_REFUSED_BEFORE)
            ],
        )
    readings = _turn_readings()
    readings[0]["observed_at"] = _SEARCHED_AT.isoformat()

    text, state = await _first_request_with_readings(
        readings,
        plan=plan,
        stored=stored,
        refusal=refusal,
        amendment="only since 2015" if earlier is not None else None,
    )

    expected = {
        SPEC: PinSource(
            block={"spec_id": SPEC},
            warnings=("role_scope_defaulted",),
            settled=True,
            asked_roles=("donor", "recipient"),
        ),
        SPEC_B: PinSource(block={"spec_id": SPEC_B}),
        SPEC_C: PinSource(block={"spec_id": SPEC_C}, settled=True),
    }
    assert pin_sources(state) == {spec_id: expected[spec_id] for spec_id in kept}
    assert ("Do not propose" in text) is (SPEC not in kept)


@pytest.mark.parametrize(
    ("code", "also", "observed", "between", "kept"),
    [
        ("definition_not_held", None, "after", None, True),
        ("upstream_version_changed", None, "after", None, True),
        ("definition_not_held", None, "before", None, False),
        ("over_ceiling", None, "after", None, False),
        ("open_role_unavailable", "upstream_version_changed", "after", None, False),
        ("upstream_version_changed", None, "after", "refused-again", False),
        ("upstream_version_changed", None, "after", "resumed", True),
    ],
    ids=[
        "re-registered",
        "refreshed",
        "searched-before-the-refusal",
        "refused-for-good",
        "refused-for-good-and-for-a-refresh",
        "refused-again-after-the-search",
        "resume-keeps-the-refusal-time",
    ],
)
async def test_a_refusal_payload_is_held_with_its_code_until_a_later_search_lifts_it(
    code: str,
    also: str | None,
    observed: str,
    between: str | None,
    kept: bool,
) -> None:
    """A replan answers the executor refusing `SPEC` for `code`, which is held
    as a bare spec_id with its code and the entry's time beside it. A later
    amendment's readings settle on `SPEC`, observed before or after that
    replan. A refusal a refreshed or re-registered definition recovers from is
    lifted by a reading observed after it: `SPEC` is a pin source again and the
    brief names nothing to avoid. One observed before it, or a refusal for
    good, keeps `SPEC` out and named. Refused again after the reading
    (`between`), `SPEC` is held from the second refusal; a resume of the
    replan keeps the first refusal's time. A wave that also stops on `SPEC`
    for another code (`also`) holds it for good when either code is."""
    runs = 3 if between is not None else 2
    scripted = ScriptedGenai([_says("still nothing")] * runs)
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()
    refusal = _rediscovery_refusal(SPEC, code)
    if also is not None:
        refusal["also_stopped"] = [_rediscovery_refusal(SPEC, also)]
    before = datetime.now(UTC)

    await harness.run(
        adapter, question="aid to guatemala", readings=_turn_readings(), refusal=refusal
    )
    state = await harness.state()
    assert state[REFUSED_SPECS_STATE_KEY] == [SPEC]
    (held,) = load_refused_specs(state)
    assert held.code == code
    assert held.refused_at is not None and held.refused_at >= before
    after = datetime.now(UTC)
    if between is not None:
        await harness.run(
            adapter,
            question="aid to guatemala",
            readings=_turn_readings(),
            refusal=refusal,
            resume_inputs={"ask": {"result": "since 2015"}} if between == "resumed" else None,
        )
    readings = _turn_readings()
    readings[0]["observed_at"] = (before if observed == "before" else after).isoformat()
    calls = len(scripted.requests)
    await harness.run(
        adapter, question="aid to guatemala", readings=readings, amendment="only since 2015"
    )

    state = await harness.state()
    assert (SPEC in pin_sources(state)) is kept
    assert ("Do not propose" in _prompt_text(scripted.requests[calls])) is not kept


# A spec Search offered as another count of `SPEC`'s reading.
_SIBLING = "spec_" + "d" * 32


def _offering_readings(siblings: str) -> list[dict[str, Any]]:
    """`_turn_readings` with `SPEC`'s reading offering no other count, or
    offering `_SIBLING` unfetched, or offering it and the reading a `fetch`
    of it gained."""
    readings = _turn_readings()
    if siblings == "none":
        return readings
    readings[0] |= {
        "counted_as": "gross disbursements",
        "alternatives": [{"spec_id": _SIBLING, "label": "net disbursements"}],
    }
    if siblings == "fetched":
        fetched = SearchReading.model_validate(
            {
                "text": "aid",
                "status": "chosen",
                "spec_id": _SIBLING,
                "counted_as": "net disbursements",
                "sibling_of": {"spec_id": SPEC, "axis": "what"},
            }
        ).with_fetch({"spec_id": _SIBLING, "summary": "5 observations, 2020-2024"})
        readings.append(fetched.model_dump(mode="json"))
    return readings


def _avoided(brief: str) -> str:
    """The line of `brief` holding the sentence that names the refused specs."""
    (line,) = [line for line in brief.splitlines() if "Do not propose" in line]
    return line


@pytest.mark.parametrize("siblings", ["none", "unfetched", "fetched"])
async def test_every_spec_refused_since_the_first_draft_is_named_and_unpinnable(
    siblings: str,
) -> None:
    """Two replans and an amendment after one first draft, the replans
    refusing `SPEC` and then `SPEC_C`. Every call renders the same readings,
    refused ones and the other counts they offered included. From the first
    refusal on, each call's brief names every spec refused so far in one
    sentence and leaves them out of the pin sources. The next first-draft
    entry starts clean."""
    scripted = ScriptedGenai([_says("still nothing")] * 5)
    adapter = _scripted_planner(scripted)
    readings = _offering_readings(siblings)
    calls: list[dict[str, Any]] = [
        {},
        {"refusal": _rediscovery_refusal(SPEC)},
        {"refusal": _rediscovery_refusal(SPEC_C)},
        {"amendment": "only since 2015"},
        {},
    ]
    pinnable: list[list[str]] = []
    briefs: list[str] = []

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        for call in calls:
            before = len(scripted.requests)
            await adapter(ctx=ctx, question="aid to guatemala", readings=readings, **call)
            pinnable.append(list(pin_sources(cast(SessionState, ctx.session.state))))
            briefs.append(_brief_follows_the_question(scripted.requests[before]))

    wf = Workflow(
        name="adapter_test_wf",
        edges=[(START, node(_node, name="adapter_under_test", rerun_on_resume=True))],
    )
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for _ in runner.run_async(
        user_id=_USER, session_id=session.id, new_message=content, run_config=SSE
    ):
        pass

    shown = [SPEC, SPEC_B, SPEC_C, *([_SIBLING] if siblings == "fetched" else [])]
    refused_by_call = [(), (SPEC,), (SPEC, SPEC_C), (SPEC, SPEC_C), ()]
    assert pinnable == [
        [spec_id for spec_id in shown if spec_id not in refused] for refused in refused_by_call
    ]
    (readings_text,) = {brief.split("\nQuestion: ")[0] for brief in briefs}
    for spec_id in (SPEC, SPEC_C, *([_SIBLING] if siblings != "none" else [])):
        assert spec_id in readings_text
    for brief, refused in zip(briefs, refused_by_call, strict=True):
        if not refused:
            assert "Do not propose" not in brief
            continue
        avoided = _avoided(brief)
        assert all(spec_id in avoided for spec_id in refused)
        assert _SIBLING not in avoided and SPEC_B not in avoided
    # The second replan names its own refusal's reason and the earlier one's.
    assert "member_count = 872" in _avoided(briefs[2])
    assert planner_adapter._REFUSED_EARLIER in _avoided(briefs[3])


@pytest.mark.parametrize("siblings", ["unfetched", "fetched"])
async def test_a_replan_admits_a_pin_of_another_count_the_refused_reading_offered(
    decision: Any, requirement: Any, siblings: str
) -> None:
    """A replan after the executor refused France's gross disbursements to
    Kenya pins the net count Search offered beside them. Admission records
    the draft with the pin as stated, whether a `fetch` of the net count gave
    it a reading of its own, which makes it a pin source, or not."""
    (part,) = resolver_digest(_FRANCE_KENYA)["part_digests"]
    block = part["spec_block"]
    gross = block["spec_id"]
    net = block["resolved"]["alternative_measures"][0]["definition"]["spec_id"]
    stored = {k: v for k, v in part.items() if k not in ("spec_block", "candidate_blocks")}
    readings = [SearchReading.from_part({**stored, **reading_facts(block)})]
    assert net in {alternative.spec_id for alternative in readings[0].alternatives}
    if siblings == "fetched":
        sibling = SearchReading.model_validate(
            {
                "text": "aid",
                "status": "chosen",
                "spec_id": net,
                "counted_as": "net disbursements",
                "sibling_of": {"spec_id": gross, "axis": "what"},
            }
        )
        readings.append(sibling.with_fetch({"spec_id": net, "summary": "5 observations"}))
    dumped = [reading.model_dump(mode="json") for reading in readings]
    dataset = requirement()
    first_draft = {
        "question": "aid",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            dataset.model_copy(update={"source_spec_id": gross}).model_dump(mode="json")
        ],
    }
    repinned = dataset.model_copy(update={"source_spec_id": net}).model_dump(mode="json")
    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", first_draft)),
            _says("done"),
            _calls(("propose_plan", {"evidence_requirements": [repinned], "amends_version": 1})),
            _says("done"),
        ]
    )
    adapter = _scripted_planner(scripted)
    after: dict[str, Any] = {}

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        await adapter(ctx=ctx, question="aid", readings=dumped)
        first = load_plan(ctx.state)
        assert first is not None
        after["replanned_from"] = len(scripted.requests)
        await adapter(
            ctx=ctx,
            question="aid",
            readings=dumped,
            prior_plan=first.model_dump(mode="json"),
            refusal=_rediscovery_refusal(gross),
        )
        after["plan"] = load_plan(ctx.state)
        after["pinnable"] = list(pin_sources(cast(SessionState, ctx.session.state)))

    wf = Workflow(
        name="adapter_test_wf",
        edges=[(START, node(_node, name="adapter_under_test", rerun_on_resume=True))],
    )
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(
        app_name=_APP, user_id=_USER, state={ORCHESTRATOR_SPECS_KEY: {gross: block}}
    )
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for _ in runner.run_async(
        user_id=_USER, session_id=session.id, new_message=content, run_config=SSE
    ):
        pass

    replanning = scripted.requests[after["replanned_from"] :]
    (answered,) = _answers(replanning[-1], "propose_plan")
    assert (answered["outcome"], answered["status"], answered["version"]) == ("ok", "draft", 2)
    assert after["pinnable"] == ([net] if siblings == "fetched" else [])
    plan = after["plan"]
    assert plan is not None
    assert [r.source_spec_id for r in plan.evidence_requirements] == [net]  # type: ignore[union-attr]
    assert gross in _avoided(_brief_follows_the_question(replanning[0]))


async def test_the_shown_ids_are_exactly_the_ids_the_readings_text_names(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A hundred readings overflow the budget, so the oldest are dropped and their ids go unshown."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    readings = [
        SearchReading(
            text=f"measure {number}",
            status="definite",
            spec_id=f"spec_{number:032x}",
            sentence="x" * 400,
        ).model_dump(mode="json")
        for number in range(100)
    ]

    text, state = await _first_request_with_readings(readings, plan=plan)

    shown = list(pin_sources(state))
    named = [r["spec_id"] for r in readings if r["spec_id"] in text]
    assert "earlier readings not shown" in text
    assert shown == named
    assert 0 < len(shown) < len(readings)


async def test_an_orchestrator_turn_with_no_search_says_so(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)

    text, state = await _first_request_with_readings([], plan=plan)

    assert "(no search has run in this conversation)" in text
    assert pin_sources(state) == {}


def _prior_ledger(*turns: tuple[list[SearchReading], Mapping[str, Any]]) -> dict[str, Any]:
    """The prior ledger a conversation whose turns read `turns`, each its
    readings and the spec blocks its record stored, seeds the next turn with.
    The conversation's `earlier_block` becomes the current one, as
    `harness.studio` sets it for a segment."""
    conversation = ConversationState(conversation_id="conv_prior")
    conversation.turns = [
        {
            "user": "asked",
            "assistant": "answered",
            "readings": [reading.model_dump(mode="json") for reading in readings],
            "specs": dict(specs),
        }
        for readings, specs in turns
    ]
    prior = conversation.prior_ledger_state()
    assert prior is not None
    current_earlier_block.set(conversation.earlier_block)
    return prior


def _aid_reading(
    *, window_kind: str | None = None, sentence: str = "Gross ODA", origin: str = "search"
) -> SearchReading:
    return SearchReading.model_validate(
        {
            "text": "aid",
            "status": "definite",
            "spec_id": SPEC,
            "sentence": sentence,
            "requested_window_kind": window_kind,
            "origin": origin,
            "warnings": [{"code": "role_scope_defaulted"}],
        }
    )


@pytest.mark.parametrize(
    ("reading", "stored", "block"),
    [
        (_aid_reading(window_kind="range"), {SPEC: {"spec_id": SPEC}}, {"spec_id": SPEC}),
        (_aid_reading(window_kind="latest"), {SPEC: {"spec_id": SPEC}}, None),
        (_aid_reading(origin="translator"), {}, None),
    ],
    ids=[
        "range_pins_from_its_block",
        "latest_falls_through_to_a_search",
        "translator_found_has_no_block",
    ],
)
async def test_a_later_turn_renders_and_pins_an_earlier_turns_reading(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    reading: SearchReading,
    stored: dict[str, Any],
    block: dict[str, Any] | None,
) -> None:
    """Turn 1's reading reaches turn 2's ledger and pin map as one of turn 2's
    own would, from turn 1, with the block turn 1 stored; a `latest` reading
    or a source only the translator found carries no block, so a pin of it
    is confirmed by a fresh search."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    prior = _prior_ledger(([reading], {**stored, SPEC_B: {"x": 1}}))

    text, state = await _first_request_with_readings(
        prior["readings"],
        plan=plan,
        stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior},
    )

    assert f'"aid" definite {SPEC}' in text
    assert "(no search has run in this conversation)" not in text
    # The session's pin map holds no earlier turn's block; a pin reads it
    # through the conversation.
    assert pin_sources(state) == {
        SPEC: PinSource(warnings=("role_scope_defaulted",), settled=True, earlier_turn=True)
    }
    assert pin_sources(state)[SPEC].spec_block(SPEC) == block


async def test_the_newest_reading_of_a_spec_id_wins_across_turns(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A spec_id turn 1 and this turn both read renders once, as this turn
    read it, and pins from this turn's block; turn 2's `latest` reading of
    SPEC_C leaves turn 1's block of it behind."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    chosen = SearchReading(text="health aid", status="chosen", spec_id=SPEC_C)
    prior = _prior_ledger(
        ([_aid_reading(sentence="turn one"), chosen], {SPEC: {"turn": 1}, SPEC_C: {"turn": 1}}),
        ([chosen.model_copy(update={"requested_window_kind": "latest"})], {SPEC_C: {"turn": 2}}),
    )
    current = [_aid_reading(sentence="turn three").model_dump(mode="json")]

    text, state = await _first_request_with_readings(
        [*prior["readings"], *current],
        plan=plan,
        stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior, ORCHESTRATOR_SPECS_KEY: {SPEC: {"turn": 3}}},
    )

    ledger = text.split("Search readings (oldest first):")[1].split("\n\n")[0]
    assert "turn three" in ledger and "turn one" not in ledger
    assert ledger.count(SPEC) == 1 and ledger.count(SPEC_C) == 1
    assert ledger.index(SPEC_C) < ledger.index(SPEC)
    sources = pin_sources(state)
    assert (sources[SPEC].block, sources[SPEC_C].block) == ({"turn": 3}, None)
    assert not sources[SPEC].earlier_turn


async def test_the_cross_turn_ledger_cuts_the_oldest_readings_and_counts_them(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Earlier turns' readings past the budget are dropped oldest first with a
    count, and this turn's reading stays shown and pinnable."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    earlier = [
        SearchReading(
            text=f"measure {n}", status="definite", spec_id=f"spec_{n:032x}", sentence="x" * 400
        )
        for n in range(60)
    ]
    prior = _prior_ledger((earlier[:30], {}), (earlier[30:], {}))
    current = [_aid_reading().model_dump(mode="json")]

    text, state = await _first_request_with_readings(
        [*prior["readings"], *current],
        plan=plan,
        stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior},
    )

    assert "earlier readings not shown" in text
    assert earlier[0].spec_id not in text
    shown = list(pin_sources(state))
    assert shown[-1] == SPEC and earlier[0].spec_id not in shown


# -- the plan being changed, as a digest ----------------------------------


def _distinct_plan() -> dict[str, Any]:
    """A prior plan whose every field carries a value this test can look for.

    Each field says what an amendment has to preserve: `decisions` name the
    alternatives that were weighed, `evidence_requirements` say what the
    answer rests on, and `workflow.sources` name the dataset the user
    approved. The values are distinctive strings rather than realistic ones,
    so a failure names the field that went missing instead of printing a
    plan.
    """
    spec_id = "spec_" + "c" * 32
    decision = Decision(
        slot="measure_slot",
        description="DESCRIPTION_TEXT",
        options=["OPTION_ONE", "OPTION_TWO"],
        chosen="OPTION_TWO",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="RATIONALE_TEXT",
    )
    dataset = DatasetRequirement(
        id="dataset_req",
        decision_ids=["measure_slot"],
        measure="MEASURE_TEXT",
        entities=[Entity(name="ENTITY_NAME", role="ENTITY_ROLE")],
        period=Period(start=2015, end=2024),
        unit=Unit(currency="USD", prices="current"),
        source="SOURCE_TEXT",
    )
    derived = AggregateRequirement(
        id="derived_req",
        decision_ids=["measure_slot"],
        input_id="dataset_req",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    draft = Plan(
        question="QUESTION_TEXT",
        decisions=[decision],
        evidence_requirements=[dataset, derived],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="dataset_req", requirement_id="dataset_req", spec_id=spec_id),
            AggregateOperation(
                id="derived_req",
                requirement_id="derived_req",
                input="dataset_req",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["derived_req"],
    )
    workflow = PlanWorkflow(
        steps=[
            WorkflowStep(
                id="dataset_req",
                operation_id="dataset_req",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": spec_id},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="derived_req",
                operation_id="derived_req",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "dataset_req", "output": "frame_ref"}},
                input_schema_hash="hash",
            ),
        ],
        sources=[SelectedSource(spec_id=spec_id, sentence="SOURCE_SENTENCE", has_data=True)],
        evidence_bindings=[
            EvidenceBinding(requirement_id="dataset_req", step_id="dataset_req", spec_id=spec_id)
        ],
        terminal_steps=["derived_req"],
    )
    return draft.with_translation(translation=translation, workflow=workflow).model_dump(
        mode="json"
    )


def test_the_digest_carries_what_an_amendment_has_to_preserve() -> None:
    """The prior plan's digest keeps every field a next proposal depends on.

    This is what makes the digest honest as well as smaller. The planner's model reads the
    plan here and decides an amendment from it. It cannot read session state,
    so a field this renderer drops is a field the next proposal cannot
    preserve, and dropping one would be a regression dressed as a saving.

    Every value asserted below is a decision input rather than decoration:
    `chosen` and `options` are what a change is weighed against, the
    requirement's own fields are what code checks a retrieved frame against,
    `decision_ids` is what ties a requirement to the decision that fixed it,
    and the compiled source's `spec_id` is what a `source_spec_id` pin has to
    rest on. The plan's `question`, `version` and `status` say what is being
    changed and from where.
    """
    adapter = _scripted_planner(ScriptedGenai([]))
    plan = _distinct_plan()
    message = adapter._message(
        question="QUESTION_TEXT",
        prior_plan=plan,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    for preserved in (
        "QUESTION_TEXT",
        "measure_slot",
        "OPTION_ONE",
        "OPTION_TWO",
        "DESCRIPTION_TEXT",
        "RATIONALE_TEXT",
        "planner_default",
        "dataset_req",
        "derived_req",
        "MEASURE_TEXT",
        "ENTITY_NAME",
        "USD",
        "SOURCE_TEXT",
        "cites measure_slot",
        "reduction=sum",
        "SOURCE_SENTENCE",
        "pending_approval",
    ):
        assert preserved in message, (
            f"the prior plan's digest dropped {preserved!r}, which an amendment has to see "
            "to preserve it. `_plan_context` in `harness.planner_adapter` is what renders "
            "this section. The raw plan itself never appears in the message."
        )


def test_the_plan_section_writes_structured_values_with_non_ascii_text_as_written() -> None:
    """Entities and predicates render as JSON, and JSON's default writes each
    non-ASCII character as one to two `\\uXXXX` escapes.

    `Plan` limits the serialised size of the record with non-ASCII as written
    (`planner.plan.MAX_PLAN_RECORD_CHARS`), so the section has to render it the
    same way for that limit to bound it. A decision's `fixes` renders as the
    names of the fields it fixes, the form `propose_plan` takes, so the value
    appears once, on the requirement.
    """
    name = "C\u00f4te d'Ivoire \U0001f600"
    plan = _distinct_plan()
    plan["evidence_requirements"][0]["entities"][0]["name"] = name
    plan["decisions"][0]["fixes"] = {"measure": name, "unit": None}

    rendered = _plan_context(plan)

    assert rendered.count(name) == 1
    assert "  fixes: measure\n" in rendered
    assert "\\u" not in rendered


def test_the_plan_section_shows_the_plan_the_model_wrote(decision: Any) -> None:
    """The stored plan holds `debt_history`, code's copy of `debt` for a rolling growth.

    The model patches requirements by id and did not write the copy, so the
    digest names `debt`, its one source and its one binding, and omits both
    fields code writes.
    """
    stored = stored_split_plan(decision)

    message = _scripted_planner(ScriptedGenai([]))._message(
        question="QUESTION_TEXT",
        prior_plan=planner_adapter._as_authored(stored),
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "debt_history" not in message
    assert "rolling_copies" not in message
    assert "first_years" not in message
    assert f"- evidence: debt from {SPEC_DEBT} (step debt)" in message
    assert message.count("- evidence:") == 1
    assert message.count(" -- external debt service paid") == 1
    assert SPEC_DEBT_HISTORY not in message


async def test_the_brief_seeded_with_a_split_plan_shows_the_plan_the_model_wrote(
    decision: Any,
) -> None:
    """The graph hands the seam the stored plan, copies and all. The brief the
    planner reads names `debt` and its source, and neither the copy nor the
    copy's pin."""
    stored = stored_split_plan(decision)
    scripted = ScriptedGenai([_says("still nothing")])
    adapter = _scripted_planner(scripted)

    await _run_adapter(
        adapter, question=stored.question, readings=[], prior_plan=stored.model_dump(mode="json")
    )

    brief = _prompt_text(scripted.requests[0])
    assert f"- evidence: debt from {SPEC_DEBT} (step debt)" in brief
    assert "debt_history" not in brief
    assert SPEC_DEBT_HISTORY not in brief


def test_the_plan_section_names_the_source_a_requirement_s_evidence_came_from(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Sources are listed in retrieval order, so position cannot bind them.

    `Workflow.sources` is in the compiler's canonical order
    (`compiler.compile._index_operations`), not in the order requirements are
    declared. Here the single source is listed after a marker source that
    belongs to no requirement, so a reader taking the nth source as the nth
    requirement's source reads the wrong one. The binding line is what makes
    the association readable regardless of the source order.
    """
    other_spec = "spec_" + "d" * 32
    plan = _translated_plan(decision, requirement, derived_requirement)
    dumped = plan.model_dump(mode="json")
    # The source the dataset requirement's evidence actually came from sits
    # second. The first belongs to nothing in this plan.
    dumped["workflow"]["sources"] = [
        {"spec_id": other_spec, "sentence": "an unrelated series", "has_data": True},
        {"spec_id": SPEC, "sentence": "external debt service", "has_data": True},
    ]
    dumped["workflow"]["evidence_bindings"] = [
        {"requirement_id": "annual_debt_service", "step_id": "annual_debt_service", "spec_id": SPEC}
    ]

    rendered = _plan_context(dumped)

    assert f"- evidence: annual_debt_service from {SPEC}" in rendered, (
        "the plan section does not state which source the dataset requirement's evidence "
        "came from, and the source list is in retrieval order, so no position in it names "
        "that requirement"
    )


def test_the_plan_section_names_each_other_count_and_prices_of_a_source_with_its_spec_id(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment asking for net disbursements pins the spec_id named here,
    unless the search readings above already name it. Other prices are named
    the same way."""
    net = "spec_" + "e" * 32
    constant = "spec_" + "f" * 32
    dumped = _translated_plan(decision, requirement, derived_requirement).model_dump(mode="json")
    dumped["workflow"]["sources"] = [
        {
            "spec_id": SPEC,
            "sentence": "official development assistance",
            "elected_measure": {
                "label": "gross disbursements",
                "alternatives": [
                    {"spec_id": net, "label": "net disbursements", "axis": "what"},
                    {"spec_id": constant, "label": "constant prices", "axis": "unit"},
                ],
            },
        }
    ]

    rendered = _plan_context(dumped)

    assert (
        f"{SPEC} -- official development assistance; counted as gross disbursements; "
        f"other counts: net disbursements [{net}]; other prices: constant prices [{constant}]\n"
    ) in rendered
    assert "other counts" not in _plan_context(dumped, stated={net, constant})


# Search's answer to "Bilateral plus imputed multilateral aid from France to
# Kenya": gross disbursements elected, net disbursements offered beside it.
_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)


async def test_an_amend_call_shows_the_other_count_and_the_plan_pins_it(
    decision: Any, requirement: Any
) -> None:
    """ "Use net disbursements" arrives on an amend call, whose brief names the net
    sibling's spec_id beside the gross reading, so the model pins it and the
    measure decision follows the pin."""
    (part,) = resolver_digest(_FRANCE_KENYA)["part_digests"]
    block = part["spec_block"]
    gross = block["spec_id"]
    stored = {k: v for k, v in part.items() if k not in ("spec_block", "candidate_blocks")}
    readings = [SearchReading.from_part({**stored, **reading_facts(block)}).model_dump(mode="json")]
    dataset = requirement()
    first_draft = {
        "question": "aid",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            dataset.model_copy(update={"source_spec_id": gross}).model_dump(mode="json")
        ],
    }

    def _pin_the_net_count(request: Any) -> list[tuple[str, dict[str, Any]]]:
        named = re.search(r"other counts: net disbursements \[(spec_\w+)\]", _prompt_text(request))
        pin = named.group(1) if named else None
        repinned = dataset.model_copy(update={"source_spec_id": pin}).model_dump(mode="json")
        return [("propose_plan", {"evidence_requirements": [repinned], "amends_version": 1})]

    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", first_draft)),
            _says("done"),
            _computed(_pin_the_net_count),
            _says("done"),
        ]
    )
    adapter = _scripted_planner(scripted)
    plans: list[Plan | None] = []
    amended_from: list[int] = []

    # Both entries run in one node, as the graph's planner node re-enters the
    # seam over the `amend` edge in the same session. No translation
    # compiles, so each ends with no plan ready, leaving its draft on record.
    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        await adapter(ctx=ctx, question="aid", readings=readings)
        first = load_plan(ctx.state)
        assert first is not None
        amended_from.append(len(scripted.requests))
        await adapter(
            ctx=ctx,
            question="aid",
            readings=readings,
            prior_plan=first.model_dump(mode="json"),
            amendment="use net disbursements",
        )
        plans.append(load_plan(ctx.state))

    wf = Workflow(
        name="adapter_test_wf",
        edges=[(START, node(_node, name="adapter_under_test", rerun_on_resume=True))],
    )
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(
        app_name=_APP, user_id=_USER, state={ORCHESTRATOR_SPECS_KEY: {gross: block}}
    )
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for _ in runner.run_async(
        user_id=_USER, session_id=session.id, new_message=content, run_config=SSE
    ):
        pass

    net = block["resolved"]["alternative_measures"][0]["definition"]["spec_id"]
    assert f"other counts: net disbursements [{net}]" in _prompt_text(
        scripted.requests[amended_from[0]]
    )
    (plan,) = plans
    assert plan is not None and plan.version == 2
    (pinned,) = plan.evidence_requirements
    assert isinstance(pinned, DatasetRequirement)
    assert pinned.source_spec_id == net
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == net]
    assert measure.options == ("gross disbursements", "net disbursements")
    assert measure.chosen == "net disbursements"
    assert measure.slot in pinned.decision_ids


async def test_an_entry_records_whether_the_users_amendment_opened_it() -> None:
    """An entry carrying the user's amendment says so in state, for `propose_plan`
    to honour a decision they make in it. The next entry without one, a replan
    here, clears it, and a resume inside an entry keeps what the entry recorded."""
    harness = _AdapterHarness()
    scripted = ScriptedGenai([_says("Planned.")] * 4)
    adapter = _scripted_planner(scripted)
    recorded = []

    await harness.run(adapter, question="kenya debt", readings=[])
    recorded.append(amendment_entry(await harness.state()))
    await harness.run(adapter, question="kenya debt", readings=[], amendment="one total")
    recorded.append(amendment_entry(await harness.state()))
    await harness.run(adapter, question="kenya debt", readings=[], resume_inputs={"ask": {}})
    recorded.append(amendment_entry(await harness.state()))
    await harness.run(
        adapter, question="kenya debt", readings=[], refusal=_rediscovery_refusal(SPEC)
    )
    recorded.append(amendment_entry(await harness.state()))

    assert recorded == [False, True, True, False]


_PICKED = DecisionPick(
    slot="donor_scope",
    chosen="the top 10 donors",
    origin_fixes=DecisionFixes(entities=(Entity(name="DAC countries", role="donor"),)),
    replaced_readings=("0" * 64,),
    replaced_kinds=("dataset",),
    cited_by_every=("donor_scope",),
    cited_by_any=(),
    kept_requirements={},
    kept_decisions={},
)


async def test_an_entry_records_the_pick_it_replans_for_and_says_it_on_its_own_line() -> None:
    """An entry a pick on the plan card opens records the pick in state for
    `propose_plan` to hold, and tells the model on a line of its own, apart
    from the user's amendment. Its refusal carries the sentence the card shows
    under the decision. A resume inside the entry keeps the pick, and the
    next entry without one clears it."""
    harness = _AdapterHarness()
    scripted = ScriptedGenai([_says("Planned.")] * 3)
    adapter = _scripted_planner(scripted)
    pick = _PICKED
    recorded = []

    refused = await harness.run(adapter, question="kenya aid", readings=[], pick=pick)
    state = await harness.state()
    recorded.append((decision_pick(state), amendment_entry(state)))
    (last,) = scripted.requests[-1].content_list[-1:]
    asked = "\n".join(part.text or "" for part in last.parts or [])
    await harness.run(
        adapter, question="kenya aid", readings=[], pick=pick, resume_inputs={"ask": {}}
    )
    recorded.append(decision_pick(await harness.state()))
    amended = await harness.run(adapter, question="kenya aid", readings=[], amendment="one total")
    recorded.append(decision_pick(await harness.state()))

    assert recorded == [(_PICKED, False), _PICKED, None]
    assert (
        "\nOn the plan card the user picked 'the top 10 donors' for decision 'donor_scope'. "
        in asked
    )
    assert (
        "Change only that decision and the requirements that cite it. Carry every other "
        "requirement, with its id, its content and its citations, and every other decision "
        "exactly as the plan card has them. Keep the id of any replaced requirement that "
        "another requirement reads. Cite that decision, and every decision of the user's "
        "that all the replaced requirements cited, from each requirement you add, and cite "
        "any other decision of the user's a replaced requirement cited from at least one of "
        "them.\n" in asked
    )
    assert "The user asks for this change" not in asked
    assert refused is not None and amended is not None
    assert refused["refused"]["code"] == amended["refused"]["code"] == StopCode.PLAN_NOT_READY
    assert refused["refused"]["reason"]
    assert "reason" not in amended["refused"]


async def test_an_amendment_still_works_through_the_adapter(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment proposed against the digest amends the stored plan.

    The digest is what the planner's model reads to decide an amendment, so
    the end-to-end property is that a `propose_plan` call made after a
    digest-rendered brief still amends rather than resets: the stored plan
    moves version 5 -> 6 with `previous_version == 5`, and the requirements
    it does not restate are carried through. That is exactly what
    `Plan.amend` does when the model re-proposes, and it is the behaviour a
    digest that lost the prior plan would break.

    The call states `amends_version=5`, read off the "version 5" line of the
    digest this brief rendered. A brief that lost the prior plan would leave
    the model with no version to name.
    """
    harness = _AdapterHarness()
    seeded = _translated_plan(decision, requirement, derived_requirement, version=5).model_dump(
        mode="json"
    )
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
        "amends_version": 5,
    }
    scripted = ScriptedGenai([_calls(("propose_plan", proposal)), _says("done")])
    adapter = _scripted_planner(scripted)

    await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        prior_plan=seeded,
        amendment="only since 2015",
    )

    # What the model actually read on its one generation: the digest, not JSON.
    brief = _prompt_text(scripted.requests[0])
    for preserved in ("kenya debt", "annual_debt_service", "annual_totals", SPEC):
        assert preserved in brief, (
            f"the brief the amendment ran against is missing {preserved!r}, so the model "
            "had no way to preserve it"
        )

    amended = load_plan(await harness.state())
    assert amended is not None
    assert amended.version == 6
    assert amended.previous_version == 5


@pytest.mark.parametrize(
    "ask",
    [{"amendment": "only since 2015"}, {"refusal": _rediscovery_refusal(SPEC)}],
    ids=["amend", "replan"],
)
def test_an_amend_or_replan_brief_is_the_first_draft_brief_then_its_ask(
    ask: dict[str, Any],
) -> None:
    """An amend or replan call renders the whole brief, readings, digests and
    question as a first draft renders them, and closes on its ask, so the
    model sees the ask in context and the brief's text up to the ask is the
    same on every call of a turn."""
    adapter = _scripted_planner(ScriptedGenai([]))
    brief = {
        "question": "gdp guatemala",
        "prior_plan": None,
        "canvas_digest": {"blocks": ["a block"]},
        "conversation_digest": {"turns": ["an earlier turn"]},
        "ledger": planner_adapter._readings_ledger([]),
    }
    first_draft = adapter._message(**brief, refusal=None, amendment=None)
    closing = planner_adapter._ask_text(
        amendment=ask.get("amendment"), refusal=ask.get("refusal"), pick=None
    )

    message = adapter._message(**brief, refusal=ask.get("refusal"), amendment=ask.get("amendment"))

    assert message == first_draft + closing
    assert closing and "Question: gdp guatemala" not in closing
    for shown in ("Context for this question", "already on the user's canvas", "Earlier turns"):
        assert shown in first_draft


def test_terminal_refusal_covers_budget_exhausted_and_question_fixed_but_not_invalid_plan() -> None:
    """`_is_terminal_refusal` is what `PlannerRun.tool_answered` uses to stop the
    planner's turn on a `propose_plan` refusal that recorded nothing and
    left nothing to correct within this entry -- `BUDGET_EXHAUSTED`,
    `QUESTION_FIXED`, `NOTHING_TO_PLAN` and `PRICES_UNSERVED` alike. `INVALID_PLAN` -- a
    question-changing amendment under `question_fixed`, say -- is not among them: it is
    correctable, so the model gets to retry rather than the turn ending on it."""
    for code in ("budget_exhausted", "question_fixed", "nothing_to_plan", "prices_unserved"):
        assert _is_terminal_refusal({"outcome": "refused", "code": code})
    assert not _is_terminal_refusal({"outcome": "refused", "code": "invalid_plan"})
    assert not _is_terminal_refusal({"outcome": "ok", "code": "question_fixed"})
    assert not _is_terminal_refusal(None)


# -- user text in the brief is cut at the question cap ---------------------------


def _render(**overrides: Any) -> str:
    """One `_message` render with no readings, with `overrides` for what a test varies."""
    arguments: dict[str, Any] = {
        "question": "gdp guatemala",
        "prior_plan": None,
        "refusal": None,
        "amendment": None,
        "canvas_digest": None,
        "conversation_digest": None,
    }
    return _scripted_planner(ScriptedGenai([]))._message(**{**arguments, **overrides})


def test_a_question_past_the_cap_is_cut_in_the_brief_and_says_so() -> None:
    """`TurnRequest.message` has no length limit, and a clarify reply lengthens the question."""
    message = _render(question="q" * 1_000_000)

    (line,) = [line for line in message.splitlines() if line.startswith("Question: ")]
    assert line == "Question: " + "q" * (MAX_QUESTION_CHARS - len(CUT_MARKER)) + CUT_MARKER
    assert len(message) < 2 * MAX_QUESTION_CHARS


def test_a_question_at_the_cap_is_rendered_whole() -> None:
    question = "q" * MAX_QUESTION_CHARS

    message = _render(question=question)

    assert f"Question: {question}\n" in message
    assert CUT_MARKER not in message


def test_an_amendment_past_the_cap_is_cut_and_says_so() -> None:
    message = _render(amendment="a" * 1_000_000)

    (line,) = [
        line
        for line in message.splitlines()
        if line.startswith("The user asks for this change to the plan: ")
    ]
    asked = line.removeprefix("The user asks for this change to the plan: ")
    assert asked == "a" * (MAX_QUESTION_CHARS - len(CUT_MARKER)) + CUT_MARKER


def test_a_refusal_past_the_cap_is_cut_and_says_so() -> None:
    refusal = _rediscovery_refusal(SPEC)
    refusal["refusal"]["message"] = "m" * 1_000_000
    refusal["also_stopped"] = [_rediscovery_refusal(SPEC_B) for _ in range(1_000)]

    message = _render(refusal=refusal)

    assert "Executing the approved plan failed" in message
    assert message.count(CUT_MARKER) >= 1
    # Two cuts at most: the refusal and the sentence naming the specs to avoid.
    assert len(message) < 8 * MAX_QUESTION_CHARS


def _window_short(ref: str, magnitude: float) -> dict[str, Any]:
    """A `window_short` warning for a donor series that ends a year before the requirement."""
    stat_var = "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1010_Nominal"
    return {
        "code": "window_short",
        "severity": "warning",
        "requirement_id": "donor_oda",
        "detail": (
            f"series {stat_var} for [{ref!r}] covers 2024-2024, the requirement asks for 2024-2025"
        ),
        "found": {"start": 2024, "end": 2024},
        "required": {"start": 2024, "end": 2025},
        "stat_var": stat_var,
        "entities": [{"role": "financingProvider", "ref": ref, "label": None}],
        "magnitude": magnitude,
    }


_DONORS = (
    "United States",
    "Germany",
    "EU Institutions",
    "United Kingdom",
    "Japan",
    "France",
    "Canada",
    "Netherlands",
    "Italy",
    "Turkey",
)


def _entities_unmet_refusal() -> dict[str, Any]:
    """A `requirement_unmet` refusal with four `window_short` warnings for donors outside
    the ten, listed before an `entity_missing` error for each of the ten donors the plan
    named on a source that left the donor side open."""
    return {
        "kind": "refused",
        "refusal": "requirement_unmet",
        "detail": "requirement 'donor_oda' is not met by what 'get_data' returned",
        "step_id": "donor_oda",
        "requirement_id": "donor_oda",
        "issues": [
            _window_short("country/CYP", 12894524.0),
            _window_short("country/KAZ", 50351002.0),
            _window_short("country/SAU", 5553675122.0),
            _window_short("country/THA", 21450000.0),  # codespell:ignore
            *(
                {
                    "code": "entity_missing",
                    "severity": "error",
                    "requirement_id": "donor_oda",
                    "detail": f"the source resolves no entity named {name!r}",
                    "found": [],
                    "required": name,
                }
                for name in _DONORS
            ),
        ],
        "also_stopped": [],
    }


def test_a_refusal_shows_the_planner_every_error_before_its_warnings() -> None:
    """Four `window_short` warnings listed ahead of the entity errors would fill the cap
    and cut the errors off. Every error reaches the planner, and the warnings that do
    not fit are counted by code."""
    message = _render(refusal=_entities_unmet_refusal())

    shown, listed = _refusal_shown(message)
    errors = [*shown["issues"], *listed]
    assert [issue["required"] for issue in errors] == list(_DONORS)
    assert {issue["code"] for issue in errors} == {"entity_missing"}
    assert shown["warnings_left_out"] == {"window_short": 4}
    assert CUT_MARKER not in message


def test_a_refusal_lists_its_errors_before_its_warnings_when_all_fit() -> None:
    refusal = _entities_unmet_refusal()
    refusal["issues"] = [*refusal["issues"][:2], refusal["issues"][4]]

    shown, listed = _refusal_shown(_render(refusal=refusal))

    assert [issue["code"] for issue in shown["issues"]] == [
        "entity_missing",
        "window_short",
        "window_short",
    ]
    assert listed == []


def _refusal_shown(message: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The refusal JSON a replan message carries, and the errors listed after it."""
    refusal = message.split("Refusal: ", 1)[1].split("\nCall ", 1)[0]
    shown, _, listed = refusal.partition("\nErrors: ")
    return json.loads(shown), _errors_and_left_out(listed)[0] if listed else []


def _errors_and_left_out(listed: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The errors an `Errors:` line lists, and the counts its `Errors left out:` line gives."""
    errors, _, left_out = listed.split("\nCall ", 1)[0].partition("\nErrors left out: ")
    return json.loads(errors), json.loads(left_out) if left_out else {}


def _grain_ambiguous(requirement_id: str, series: int) -> dict[str, Any]:
    """A `grain_ambiguous` error, which the evidence check raises once per series."""
    return {
        "code": "grain_ambiguous",
        "severity": "error",
        "requirement_id": requirement_id,
        "detail": f"series {series} of {requirement_id!r} holds more than one row per period",
        "found": None,
        "required": None,
    }


@pytest.mark.parametrize(
    ("extra", "left_out"),
    [
        pytest.param([], set(), id="every-error-fits"),
        pytest.param(
            [_grain_ambiguous(rid, n) for rid in ("donor_oda", "gdp") for n in range(300)],
            {"grain_ambiguous"},
            id="hundreds-of-series-errors",
        ),
    ],
)
def test_errors_past_the_cap_follow_the_cut_refusal(
    extra: list[dict[str, Any]], left_out: set[str]
) -> None:
    """Past the cap the errors follow the cut refusal. The first of each requirement and
    code is always listed, more while a bound allows, and the rest are counted by code."""
    refusal = _entities_unmet_refusal()
    refusal["detail"] = "d" * 1_000_000
    refusal["issues"] = [*refusal["issues"], *extra]
    errors = [issue for issue in refusal["issues"] if issue["severity"] == "error"]
    for issue in errors[: len(_DONORS)]:
        issue["found"] = ["a donor label"] * 1_000

    message = _render(refusal=refusal)

    cut, _, listed = message.partition("\nErrors: ")
    shown, counted = _errors_and_left_out(listed)
    fields = ("code", "requirement_id", "detail", "required")
    assert shown[: len(_DONORS)] == [
        {key: issue[key] for key in fields} for issue in errors[: len(_DONORS)]
    ]
    assert {(one["requirement_id"], one["code"]) for one in shown} == {
        (issue["requirement_id"], issue["code"]) for issue in errors
    }
    assert len(shown) + sum(counted.values()) == len(errors)
    assert set(counted) == left_out
    assert cut.endswith(CUT_MARKER)
    assert len(message) < 5 * MAX_QUESTION_CHARS


class _Uncomparable(dict[str, Any]):
    """An issue that fails the test when compared with another, as only a pairwise scan does."""

    @override
    def __eq__(self, other: object) -> bool:
        raise AssertionError("the refusal's issues were compared pairwise")

    __hash__ = None  # type: ignore[assignment]


def test_a_refusal_of_thousands_of_errors_renders_as_its_small_counterpart() -> None:
    """Twelve thousand series errors render as three hundred do, their counts aside, and
    the issues are split by severity in one pass, with no pairwise comparison."""

    def rendered(series: int) -> str:
        refusal = _entities_unmet_refusal()
        refusal["issues"] = [
            _Uncomparable(issue)
            for issue in [*refusal["issues"], *(_grain_ambiguous("gdp", n) for n in range(series))]
        ]
        return _render(refusal=refusal)

    small, large = rendered(300), rendered(12_000)

    listed, counted = _errors_and_left_out(small.partition("\nErrors: ")[2])
    assert len(listed) + sum(counted.values()) == len(_DONORS) + 300
    more = {**counted, "grain_ambiguous": counted["grain_ambiguous"] + 11_700}
    assert large == small.replace(
        json.dumps({"entity_missing": len(_DONORS), "grain_ambiguous": 300}),
        json.dumps({"entity_missing": len(_DONORS), "grain_ambiguous": 12_000}),
    ).replace(f"Errors left out: {json.dumps(counted)}", f"Errors left out: {json.dumps(more)}")


def test_a_short_refusal_is_rendered_whole() -> None:
    message = _render(refusal=_rediscovery_refusal(SPEC))

    assert CUT_MARKER not in message
    assert "resolved.stat_vars.member_count = 872" in message


# -- a referral that repeats without new information ----------


def _repair(constraint: str) -> dict[str, Any]:
    return {
        "outcome": "invalid_input",
        "kind": "requirement_needs_repair",
        "requirement_id": "debt",
        "reason": "r",
        "constraint": constraint,
    }


_AMBIGUOUS = {
    "outcome": "choice_required",
    "kind": "source_ambiguous",
    "requirement_id": "gdp_gtm",
    "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
}


@pytest.mark.parametrize(
    ("first", "second", "stalled"),
    [
        (_AMBIGUOUS, _AMBIGUOUS, True),
        (_repair("family_unsummable"), _repair("family_unsummable"), True),
        (_repair("input_shape_unsupported"), _repair("family_unsummable"), False),
    ],
    ids=["same-choice", "same-repair", "different-repairs"],
)
def test_stalled_referral_stops_on_the_second_identical_referral_not_the_first(
    first: dict[str, Any], second: dict[str, Any], stalled: bool
) -> None:
    counts: dict[tuple[str, Any, Any], int] = {}

    assert _stalled_referral(first, counts) is False
    assert _stalled_referral(second, counts) is stalled


def _congo(*roles: str) -> dict[str, Any]:
    """An `entity_name_ambiguous` referral with one reading per "Congo" of each role."""
    return {
        "outcome": "choice_required",
        "kind": "entity_name_ambiguous",
        "requirement_id": "aid",
        "reason": "r",
        "ambiguous_entities": [
            {
                "text": "Congo",
                "candidates": [
                    {"ref": "country/COD", "label": "Democratic Republic of the Congo"},
                    {"ref": "country/COG", "label": "Republic of the Congo"},
                ],
                "entity": {"name": "Congo", "role": role},
            }
            for role in roles
        ],
    }


@pytest.mark.parametrize(
    ("rounds", "stalled"),
    [
        ((_congo("donor", "recipient"), _congo("recipient"), _congo("donor")), False),
        ((_congo("donor", "recipient"), _congo("recipient"), _congo("recipient")), True),
    ],
    ids=["one-end-settled-per-round", "same-end-again"],
)
def test_an_entity_referral_covering_fewer_entities_is_progress(
    rounds: tuple[dict[str, Any], ...], stalled: bool
) -> None:
    counts: dict[StallKey, int] = {}

    *earlier, last = rounds
    assert [_stalled_referral(one, counts) for one in earlier] == [False] * len(earlier)
    assert _stalled_referral(last, counts) is stalled


@pytest.mark.parametrize("kind", ["discovery_unavailable", "surface_unavailable"])
def test_a_transient_referral_never_stalls(kind: str) -> None:
    counts: dict[tuple[str, Any, Any], int] = {}
    referred = {
        "outcome": "transient",
        "kind": kind,
        "requirement_id": "gdp_gtm" if kind == "discovery_unavailable" else None,
        "reason": "r",
    }

    for _ in range(3):
        assert _stalled_referral(referred, counts) is False
    assert counts == {}


def test_stalled_referral_ignores_a_response_that_is_not_a_referral() -> None:
    counts: dict[tuple[str, Any, Any], int] = {}
    problem_response = {
        "outcome": "invalid_input",
        "code": "requirement_unknown",
        "message": "no such requirement",
    }

    assert _stalled_referral(problem_response, counts) is False
    assert _stalled_referral(problem_response, counts) is False
    assert counts == {}


def _track_proposal(tracker: PlannerRun, translation: dict[str, Any]) -> None:
    """Track one accepted `propose_plan` whose translation answered `translation`."""
    tracker.tool_answered(
        "propose_plan",
        {"outcome": "ok", "status": "draft", "version": 1, "translation": translation},
    )


def test_a_stalled_referral_ends_the_entry_on_plan_not_ready_carrying_its_constraint() -> None:
    """The second identical repair referral ends the entry with no further
    round: the refusal carries the compile issue the referral names, which
    the user's sentence states (`harness.sentences.referral_reason`)."""
    tracker = PlannerRun(agent="dfo_planner")
    referred = {
        "outcome": "invalid_input",
        "kind": "requirement_needs_repair",
        "requirement_id": "gdp_gtm",
        "reason": "operation 'gap' takes a whole side",
        "constraint": "entity_required",
        "repair": "name the entity",
    }

    _track_proposal(tracker, referred)
    assert not tracker.ready_to_break
    _track_proposal(tracker, referred)

    assert tracker.ready_to_break
    assert tracker.refused == {
        "code": StopCode.PLAN_NOT_READY,
        "detail": {
            "kind": "requirement_needs_repair",
            "requirement_id": "gdp_gtm",
            "reason": "operation 'gap' takes a whole side",
            "constraint": "entity_required",
        },
    }
    assert tracker.submitted is None


@pytest.mark.parametrize(
    ("referral", "named"),
    [
        (
            {
                "kind": "internal_failure",
                "outcome": "internal",
                "reason": "cycle: loop",
                "reference": "0123456789ab",
            },
            "0123456789ab",
        ),
        (
            {
                "kind": "requirement_needs_repair",
                "outcome": "invalid_input",
                "requirement_id": "cmp",
                "reason": "entity_required",
                "repeated": True,
            },
            "'cmp'",
        ),
        (
            {
                "kind": "discovery_unavailable",
                "outcome": "transient",
                "requirement_id": "gdp_gtm",
                "reason": "used 4 discovery calls",
                "budget_spent": True,
            },
            "'gdp_gtm'",
        ),
    ],
    ids=["internal", "repeated", "budget-spent"],
)
def test_a_closing_referral_ends_the_entry_on_the_proposal_carrying_it(
    referral: dict[str, Any], named: str
) -> None:
    """An internal failure, a translation that failed the way it failed
    before, or a data service that timed out until the discovery budget was
    spent ends the entry as a final refusal naming what stopped it, as soon
    as a proposal's translation answers with it."""
    tracker = PlannerRun(agent="dfo_planner")
    _track_proposal(tracker, referral)

    assert tracker.ready_to_break
    assert tracker.submitted is None
    assert tracker.refused is not None
    assert named in refusal_sentence(tracker.refused)
    assert PlannerAdapter._refused(tracker.refused)["refused"]["final"] is True


_NOT_CALLED = "'search' was not called: the planner entry has no time left for it"
_CUT_OFF = "'search' did not answer in the 54 s the planner entry had left"


@pytest.mark.parametrize(
    ("marks", "time_left", "code", "reason"),
    [
        ({"not_issued": True}, 0.0, "turn_wall_time_spent", _NOT_CALLED),
        ({}, 0.0, "discovery_unavailable", _CUT_OFF),
        ({}, 5.0, None, _CUT_OFF),
        ({}, None, None, _CUT_OFF),
        ({"budget_spent": True, "not_issued": True}, 0.0, "discovery_unavailable", _NOT_CALLED),
        ({"budget_spent": True}, 0.0, "discovery_unavailable", _CUT_OFF),
        ({"budget_spent": True}, 5.0, "discovery_unavailable", _CUT_OFF),
    ],
    ids=[
        "never-issued",
        "cut-off-no-time-left",
        "cut-off-time-left",
        "outside-an-entry",
        "budget-spent-before-never-issued",
        "budget-spent-no-time-left",
        "budget-spent-time-left",
    ],
)
def test_a_transient_referral_ends_the_entry_as_the_wall_bound_only_for_a_call_never_issued(
    marks: dict[str, bool], time_left: float | None, code: str | None, reason: str
) -> None:
    """Only a call the request's wall-clock bound left no time to make ends
    the entry as that bound spent.

    A search that was made and cut off with no time left means the data
    service did not answer, so the entry ends as the service unavailable;
    a spent budget says the same whatever the clock reads. With time left the
    planner may propose again. The referral's reason stays in the detail.
    """
    referral = {
        "kind": "discovery_unavailable",
        "outcome": "transient",
        "requirement_id": "oda_france",
        "reason": reason,
        **marks,
    }
    tracker = PlannerRun(agent="dfo_planner")
    token = current_discovery_time_left.set(None if time_left is None else lambda: time_left)
    try:
        _track_proposal(tracker, referral)
    finally:
        current_discovery_time_left.reset(token)

    if code is None:
        assert tracker.refused is None
        return
    assert tracker.ready_to_break
    assert tracker.refused is not None
    assert tracker.refused["code"] == code
    assert tracker.refused["detail"]["reason"] == reason
    assert tracker.refused["detail"]["requirement_id"] == "oda_france"


def test_last_referral_is_cleared_once_a_later_translation_resolves_the_requirement() -> None:
    """`PlannerRun.tool_answered` writes `tracker.last_referral` from every translation
    carrying an `outcome`, referral or not, replacing whatever the
    last one recorded: a later `ok` response clears it, so an entry that
    ends after that clear carries no referral the plan no longer has
    (`not_ready`)."""
    tracker = PlannerRun(agent="dfo_planner")
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    _track_proposal(tracker, referred)
    assert tracker.last_referral is not None

    resolved = {"outcome": "ok", "requirement_id": "gdp_gtm"}
    _track_proposal(tracker, resolved)

    assert tracker.last_referral is None


def test_last_referral_survives_a_translation_carrying_no_outcome() -> None:
    """A translation missing its own `outcome` key -- not a shape
    translation itself ever returns, but the defensive case `referral_of`
    already reads for -- leaves `last_referral` as it was, so an entry that
    ends right after still carries the last real referral."""
    tracker = PlannerRun(agent="dfo_planner")
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}],
    }
    _track_proposal(tracker, referred)

    _track_proposal(tracker, {"malformed": True})

    assert tracker.last_referral is not None
    assert tracker.last_referral["requirement_id"] == "gdp_gtm"


# A `no_data` `search` answer, canned so every `discover_source` call inside
# translation's own mechanical loop gets the identical answer -- no nearest
# series, no judgement to pause on, so translation refers the plan back on
# its own, in Python, the same way every time.
_NO_DATA_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "no_data",
    "parts": [
        {
            "status": "no_data",
            "variable_text": "asked",
            "warnings": [],
            "no_data": {"reason": "entity_not_resolved"},
        }
    ],
}

_NO_DATA_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_NO_DATA_SEARCH_RESPONSE_ENVELOPE)
)


def _propose_again_step(*, start_version: int = 1) -> Any:
    """A script entry that proposes the recorded plan again unchanged, which
    translates it.

    Reads the `amends_version` off the last `propose_plan` answer, as
    `_amend_step` does, from `start_version` for the turn's first proposal.
    """

    def step(request: Any) -> list[tuple[str, dict[str, Any]]]:
        version = _last_amended_version(request, default=start_version)
        return [("propose_plan", {"amends_version": version})]

    return step


def _repeating_proposal_script(rounds: int) -> list[Any]:
    """The planner's calls for `rounds` pairs of unchanged proposals.

    Each proposal's translation runs the real mechanical loop against
    `_NO_DATA_SEARCH_RESPONSE`, so both proposals in a pair refer the same
    `(kind, requirement_id)` back. That is the shape that stops one
    `ctx.run_node` call on its second identical referral.
    """
    return [_propose_again_step() for _ in range(2 * rounds)]


class _SkillPagesSurface(FixtureSurface):
    """A surface answering the planner's one `get_skill` from the `get_skill` response, or
    from the `get_skill_path` one when the call gives a `path`, so a test scripts a
    skill's first page and its files apart."""

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        if tool.name == "get_skill" and "path" in arguments:
            tool = ToolId(server=tool.server, name="get_skill_path")
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


def _gdp_plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return Plan(
        question="gdp guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )


async def test_the_planner_runs_on_the_model_and_translation_the_adapter_is_given(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    translated: list[Plan | None] = []

    async def translate(call: Any) -> dict[str, object]:
        translated.append(load_plan(call.state))
        return {"translated_by": "the caller"}

    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }
    scripted = ScriptedGenai([_calls(("propose_plan", proposal)), _says("done")])
    adapter = scripted_planner(scripted, translate=translate)

    result = await _run_adapter(adapter, question="kenya debt", readings=[])

    # The caller's translation gives no verdict, so the plan is never declared ready.
    assert result["refused"]["code"] == StopCode.PLAN_NOT_READY
    assert [plan.version if plan else None for plan in translated] == [1]
    (answered,) = _answers(scripted.requests[1], "propose_plan")
    assert answered["translation"] == {"translated_by": "the caller"}


async def test_every_planner_generation_runs_at_the_low_thinking_config(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    scripted = ScriptedGenai([_computed(_propose_again_step()), _says("done")])
    adapter = _scripted_planner(
        scripted,
        surface=_SkillPagesSurface(schemas=[], responses={"search": _NO_DATA_SEARCH_RESPONSE}),
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)

    await _run_adapter(
        adapter, question=draft.question, readings=[], prior_plan=draft.model_dump(mode="json")
    )

    assert len(scripted.requests) > 1
    assert all(
        request.generate_config.thinking_config == AGENT_THINKING_CONFIG
        and request.generate_config.automatic_continuation is True
        for request in scripted.requests
    )


def _skill_surface(responses: dict[str, ToolCallResult]) -> FixtureSurface:
    """A surface that answers `get_skill` from `responses`. A call with a `path`
    gets the `get_skill_path` response instead."""
    return _SkillPagesSurface(schemas=[], responses=responses)


async def test_a_later_entry_sees_the_brief_and_none_of_the_earlier_entrys_calls(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Two entries of one turn, each its own `run_node` child as the graph's
    planner node runs them: the later entry's first request holds the turn's
    question, the brief after it with the amendment, and none of the earlier
    entry's tool calls, since a `single_turn` run reads only its own node path.
    The skill the earlier entry read is in the later brief instead, starred on
    the Skills named line."""
    scripted = ScriptedGenai(
        [
            _calls(("get_skill", {"skill_id": "dfo-faq"})),
            _computed(_propose_again_step()),
            _says("stop"),
            _says("noted"),
        ]
    )
    adapter = _scripted_planner(
        scripted,
        surface=_skill_surface(
            {
                "search": _NO_DATA_SEARCH_RESPONSE,
                "get_skill": ToolCallResult(is_error=False, text="# DFO FAQ"),
            }
        ),
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    plan_json = draft.model_dump(mode="json")
    entries: list[dict[str, Any]] = [{}, {"amendment": "only since 2015"}]
    readings = _skill_readings("dfo-faq")

    async def _turn(ctx: Context, node_input: Any) -> None:
        del node_input
        for entry in entries:

            async def _entry(ctx: Context, node_input: Any, entry: dict[str, Any] = entry) -> Any:
                del node_input
                return await adapter(
                    ctx=ctx,
                    question="gdp guatemala",
                    readings=readings,
                    prior_plan=plan_json,
                    **entry,
                )

            await ctx.run_node(node(_entry, name="planner", rerun_on_resume=True))

    wf = Workflow(
        name="adapter_test_wf", edges=[(START, node(_turn, name="turn", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for _ in runner.run_async(
        user_id=_USER, session_id=session.id, new_message=content, run_config=SSE
    ):
        pass

    # `scripted.requests[0..2]` are the first entry: `get_skill`,
    # `propose_plan`, and the closing text. `[3]` is the later entry's first
    # generation.
    earlier = _call_and_response_names(scripted.requests[2])
    assert {"FC:get_skill", "FR:get_skill", "FC:propose_plan", "FR:propose_plan"} <= set(earlier)
    first_brief = _brief_follows_the_question(scripted.requests[0])
    assert _skills_named(first_brief) == "dfo-faq"
    assert _SKILL_PAGE not in first_brief
    later = scripted.requests[3]
    brief = _brief_follows_the_question(later)
    assert "only since 2015" in brief
    assert _skills_named(brief) == "dfo-faq*"
    assert f"get_skill('dfo-faq'):\n{_SKILL_PAGE}\n" in brief
    assert _call_and_response_names(later) == []


_SKILL_PAGE = "# DFO FAQ"


def _skill_readings(*skills: str) -> list[dict[str, Any]]:
    """One definite reading naming `skills`, as `search` stores it."""
    reading = SearchReading(text="aid", status="definite", spec_id=SPEC, skills=skills)
    return [reading.model_dump(mode="json")]


def _skills_named(brief: str) -> str:
    """The skills the brief's Skills named line lists, starred as it stars them."""
    (line,) = [line for line in brief.splitlines() if line.startswith("Skills named")]
    return line.removeprefix("Skills named (* = already read): ")


def _skill_page_surface() -> FixtureSurface:
    """A surface whose `get_skill` answers one page for a skill and one for a file."""
    return _skill_surface(
        {
            "get_skill": ToolCallResult(is_error=False, text=_SKILL_PAGE),
            "get_skill_path": ToolCallResult(is_error=False, text="# Methods"),
        }
    )


async def test_a_first_draft_carries_no_skill_an_earlier_plan_read() -> None:
    """A first draft starts a new plan, so the skills the last plan's entries
    read leave the brief with it, while an amend of that last plan still
    carried them."""
    scripted = ScriptedGenai(
        [
            _calls(("get_skill", {"skill_id": "dfo-faq"})),
            _says("stop"),
            _says("done"),
            _says("done"),
        ]
    )
    adapter = _scripted_planner(scripted, surface=_skill_page_surface())
    harness = _AdapterHarness()
    entry = {"question": "kenya debt", "readings": _skill_readings("dfo-faq")}

    await harness.run(adapter, **entry)
    amending = len(scripted.requests)
    await harness.run(adapter, amendment="only since 2015", **entry)
    drafting = len(scripted.requests)
    await harness.run(adapter, **entry)

    amend = _brief_follows_the_question(scripted.requests[amending])
    assert _skills_named(amend) == "dfo-faq*" and _SKILL_PAGE in amend
    draft = _brief_follows_the_question(scripted.requests[drafting])
    assert _skills_named(draft) == "dfo-faq"
    assert _SKILL_PAGE not in draft


_OLD_FIRST_PAGE = (
    "# DFO FAQ\nSee get_skill_file in prose.\n\nFiles in this skill; read one with "
    'get_skill_file(skill_id="dfo-faq", path=\u2026):\n- methods.md (5.5 KB)'
)
_OLD_FILE_PAGE = (
    "# Methods\n\nThis page ends at character 31,844 of 40,000. Read the rest with "
    'get_skill_file(skill_id="dfo-faq", path="methods.md", offset=31844, digest="sha256:a").'
)


async def test_a_page_stored_with_the_old_tools_call_lines_is_shown_naming_get_skill() -> None:
    """Pages stored before the planner held one skill tool print `get_skill_file(`
    call lines; the brief names `get_skill` in each, and leaves prose alone."""
    pages = [
        {"skill_id": "dfo-faq", "path": None, "offset": 0},
        {"skill_id": "dfo-faq", "path": "methods.md", "offset": 0},
    ]
    state = {
        SKILLS_READ_STATE_KEY: pages,
        skill_page_key(pages[0]): _OLD_FIRST_PAGE,
        skill_page_key(pages[1]): _OLD_FILE_PAGE,
    }
    scripted = ScriptedGenai([_says("stop")])
    adapter = _scripted_planner(scripted, surface=_skill_page_surface())
    harness = _AdapterHarness(state)

    await harness.run(
        adapter,
        question="kenya debt",
        readings=_skill_readings("dfo-faq"),
        amendment="only since 2015",
    )

    brief = _brief_follows_the_question(scripted.requests[0])
    assert 'get_skill(skill_id="dfo-faq", path=\u2026)' in brief
    assert (
        'get_skill(skill_id="dfo-faq", path="methods.md", offset=31844, digest="sha256:a")' in brief
    )
    assert "See get_skill_file in prose." in brief
    assert "get_skill_file(skill_id=" not in brief


async def test_a_page_stored_in_the_current_form_is_shown_as_stored() -> None:
    page = {"skill_id": "dfo-faq", "path": None, "offset": 0}
    text = 'Read the rest with get_skill(skill_id="dfo-faq", path="a.md", offset=9, digest="d").'
    section = _skills_section([{**page, "text": text}])

    assert f"\n{text}\n" in section


async def test_each_skill_page_text_is_stored_once_across_the_plans_events() -> None:
    """A read stores its own page's text and a short index, so a later read's
    state delta carries no earlier page and each text appears in one event,
    however many reads and entries follow."""
    first, second = "# FAQ " + "a" * 400, "# Methods " + "b" * 400
    scripted = ScriptedGenai(
        [
            _calls(("get_skill", {"skill_id": "dfo-faq"})),
            _calls(("get_skill", {"skill_id": "dfo-faq", "path": "methods.md"})),
            _says("stop"),
            _says("stop"),
        ]
    )
    adapter = _scripted_planner(
        scripted,
        surface=_skill_surface(
            {
                "get_skill": ToolCallResult(is_error=False, text=first),
                "get_skill_path": ToolCallResult(is_error=False, text=second),
            }
        ),
    )
    harness = _AdapterHarness()
    entry = {"question": "kenya debt", "readings": _skill_readings("dfo-faq")}

    await harness.run(adapter, **entry)
    await harness.run(adapter, amendment="only since 2015", **entry)

    deltas = [
        json.dumps(event.actions.state_delta)
        for event in await harness.events()
        if event.actions and event.actions.state_delta
    ]
    # The earliest delta holding the second page is its file-read event.
    read_second = next(delta for delta in deltas if second in delta)
    assert first not in read_second
    assert sum(delta.count(first) for delta in deltas) == 1


async def test_the_skills_an_entry_shows_survive_a_planner_ask_and_reach_the_next_entry() -> None:
    """An amend reads a skill file, then pauses on a planner-side ask. The
    answer runs the planner again, which sees none of the paused run's calls,
    so its brief carries the skill the first draft read and the file the
    paused run read, each once. The next entry's brief carries both, read back
    from state the resume persisted."""
    ask = (
        PLANNER_ASK,
        {
            "message": "Which source did you mean?",
            "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
            "recommended_spec_id": SPEC,
            "rationale": "broader coverage",
        },
    )
    scripted = ScriptedGenai(
        [
            _calls(("get_skill", {"skill_id": "dfo-faq"})),
            _says("stop"),
            _calls(("get_skill", {"skill_id": "dfo-faq", "path": "methods.md"})),
            _calls(ask),
            _says("stop"),
            _says("stop"),
        ]
    )
    adapter = _scripted_planner(scripted, surface=_skill_page_surface())
    harness = _AdapterHarness()
    entry = {"question": "kenya debt", "readings": _skill_readings("dfo-faq")}

    await harness.run(adapter, **entry)
    paused = await harness.run(adapter, amendment="only since 2015", **entry)
    assert paused is None, "the ask should pause the amend"
    resuming = len(scripted.requests)
    answer = types.Content(
        role="user",
        parts=[
            create_request_input_response(
                _pending_ask_interrupt_id(await harness.events()), {"result": {"spec_id": SPEC}}
            )
        ],
    )
    await harness.run(adapter, amendment="only since 2015", new_message=answer, **entry)
    replanning = len(scripted.requests)
    await harness.run(adapter, amendment="and drop Kenya", **entry)

    resumed = scripted.requests[resuming]
    brief = _brief_follows_the_question(resumed)
    assert brief.count(_SKILL_PAGE) == 1 and brief.count("# Methods") == 1
    assert "get_skill('dfo-faq', path='methods.md', offset=0):\n# Methods\n" in brief
    assert _call_and_response_names(resumed) == []
    later = _brief_follows_the_question(scripted.requests[replanning])
    assert f"get_skill('dfo-faq'):\n{_SKILL_PAGE}\n" in later
    assert "get_skill('dfo-faq', path='methods.md', offset=0):\n# Methods\n" in later


# An `unsupported` `search` answer: translation refers the plan back as
# `unsupported_ask`, a referral only the user's answer settles.
_UNSUPPORTED_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "unsupported",
    "parts": [
        {
            "status": "unsupported",
            "variable_text": "asked",
            "warnings": [],
            "unsupported": {"code": "out_of_scope", "detail": "not a data question"},
        }
    ],
}

_UNSUPPORTED_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_UNSUPPORTED_SEARCH_RESPONSE_ENVELOPE)
)


async def test_an_answer_that_resumes_a_run_after_a_translated_proposal_is_read(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The user's answer resumes a run whose earlier proposal was translated.

    The planner proposes the plan again, whose translation refers
    `unsupported_ask`, then asks the user through `ask_user_question` and
    pauses. The answer runs the planner again, and `propose_plan` reads the
    answer off the session and passes the referral guard. The `unsupported_ask`
    decline it owes survives the resume, so the amendment carries it. The
    amendment's translation refers again, so the entry ends with no plan
    ready.
    """
    draft = _gdp_plan(decision, requirement, derived_requirement)
    surface = _SkillPagesSurface(schemas=[], responses={"search": _UNSUPPORTED_SEARCH_RESPONSE})
    # The resumed entry keeps the version the entry drafted before it asked,
    # so the amendment patches that one.
    amend = [
        (
            "propose_plan",
            {
                "question": draft.question,
                "amends_version": draft.version + 1,
                "decisions": [draft.decisions[0].model_dump(mode="json")],
                "evidence_requirements": [],
                "declined": [{"ask": draft.question, "code": "unsupported_ask"}],
            },
        )
    ]
    scripted = ScriptedGenai(
        [
            _computed(_propose_again_step(start_version=draft.version)),
            _calls((PLANNER_ASK, {"message": "what can be measured instead?", "candidates": []})),
            _calls(*amend),
            _says("stop"),
        ]
    )
    adapter = _scripted_planner(scripted, surface=surface)
    harness = _AdapterHarness()
    plan_json = draft.model_dump(mode="json")

    paused = await harness.run(adapter, question=draft.question, readings=[], prior_plan=plan_json)
    assert paused is None, "the ask should pause this call, not resolve it"

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    answer = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"text": "use debt stock"}})],
    )
    output = await harness.run(
        adapter,
        question=draft.question,
        readings=[],
        prior_plan=plan_json,
        new_message=answer,
    )
    assert output is not None and output["refused"]["final"] is False
    proposals = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert [(p or {}).get("outcome") for p in proposals] == ["ok", "ok"], proposals
    assert (proposals[1] or {})["version"] == draft.version + 2


_CONGO_ROWS = [{"text": "Congo", "role": "donor"}, {"text": "Congo", "role": "recipient"}]
_CONGO_NODES = {"COD": "Democratic Republic of the Congo", "COG": "Republic of the Congo"}

# "Aid from Congo to Congo": Search's two "Congo" rows share one identity, so it
# answers one group whose options pin both rows, read as a donor and a recipient
# reading.
_CONGO_SEARCH_RESPONSE = ToolCallResult(
    is_error=False,
    **search_result_parts(
        {
            "schema_version": 1,
            "status": "no_data",
            "parts": [
                {
                    "status": "no_data",
                    "variable_text": "aid",
                    "warnings": [],
                    "no_data": {
                        "category": "unresolved",
                        "reason": "entity_ambiguous",
                        "ambiguous_entities": [
                            {
                                "text": "Congo",
                                "candidates": [
                                    {"ref": f"country/{code}", "label": label, "kind": "country"}
                                    for code, label in _CONGO_NODES.items()
                                ],
                            }
                        ],
                        "options": [
                            {
                                "mentions": ["Congo"],
                                "label": label,
                                "entity_choice": 0,
                                "data_query": {
                                    "parts": [
                                        {
                                            "variable": "aid",
                                            "entities": [
                                                {**row, "dcid": f"country/{code}"}
                                                for row in _CONGO_ROWS
                                            ],
                                        }
                                    ]
                                },
                            }
                            for code, label in _CONGO_NODES.items()
                        ],
                    },
                }
            ],
        }
    ),
)


class _RecordingSearch(FixtureSurface):
    """A surface whose `search` answers from `answers` in turn, recording what each call sent."""

    def __init__(self, answers: list[ToolCallResult]) -> None:
        super().__init__(schemas=[])
        self._answers = answers
        self.sent: list[Mapping[str, object]] = []

    @override
    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: Any = None,
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "search":
            raise ToolSurfaceUnavailableError(f"no canned answer for {tool.name!r}", retryable=True)
        self.sent.append(arguments)
        return self._answers[min(len(self.sent), len(self._answers)) - 1]


def _translation_seen(request: Any) -> dict[str, Any]:
    """The one translation `request` shows the model: a `propose_plan` answer's
    in the run's own history, or else the referral the brief of a run an
    answer resumed renders after `_ASKED_REFERRAL_INTRO`."""
    seen = [
        part.function_response.response["translation"]
        for content in _contents(request)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    brief = _brief_follows_the_question(request).splitlines()
    if _ASKED_REFERRAL_INTRO in brief:
        seen.append(json.loads(brief[brief.index(_ASKED_REFERRAL_INTRO) + 1]))
    (translation,) = seen
    return translation


def _ask_about_congo(role: str) -> Any:
    """A script entry asking about one end's Congo, offering the picks the referral carries."""

    def step(request: Any) -> list[tuple[str, dict[str, Any]]]:
        translation = _translation_seen(request)
        (reading,) = [
            reading
            for reading in translation["ambiguous_entities"]
            if reading["entity"]["role"] == role
        ]
        candidates = [
            {"spec_id": candidate["pick"], "label": candidate["label"]}
            for candidate in reading["candidates"]
        ]
        return [(PLANNER_ASK, {"message": f"Which Congo, as {role}?", "candidates": candidates})]

    return step


@pytest.mark.parametrize(
    ("stated", "admitted"),
    [(("COD", "COG"), True), (("COG", "COD"), False)],
    ids=["as_picked", "swapped"],
)
async def test_each_congo_is_settled_by_its_own_answer_and_reaches_search_as_picked(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    source: Any,
    stated: tuple[str, str],
    admitted: bool,
) -> None:
    """Propose, take the entity referral, ask about the donor, then the recipient,
    each answered on its own resume, then propose the nodes.

    The user picks the Democratic Republic as donor and the Republic as
    recipient. A proposal stating them so is admitted and its translation sends
    each row pinned to its own node, which evidence resolves per role. One
    swapping them is refused, so nothing is searched again.
    """
    congos = [Entity(name="Congo", role="donor"), Entity(name="Congo", role="recipient")]
    dataset = requirement(measure="aid", entities=congos)
    draft = Plan(
        question="aid from Congo to Congo",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement()],
    )
    pinned = [
        {"name": "Congo", "role": role, "dcid": f"country/{code}"}
        for role, code in zip(("donor", "recipient"), stated, strict=True)
    ]

    def settle(request: Any) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                "propose_plan",
                {
                    "amends_version": _last_amended_version(request, default=1),
                    "evidence_requirements": [
                        dataset.model_dump(mode="json") | {"entities": pinned}
                    ],
                },
            )
        ]

    surface = _RecordingSearch([_CONGO_SEARCH_RESPONSE, _NO_DATA_SEARCH_RESPONSE])
    scripted = ScriptedGenai(
        [
            _calls(
                (
                    "propose_plan",
                    {
                        "question": draft.question,
                        "answer_mode": "computed",
                        "decisions": [d.model_dump(mode="json") for d in draft.decisions],
                        "evidence_requirements": [
                            r.model_dump(mode="json") for r in draft.evidence_requirements
                        ],
                    },
                )
            ),
            _computed(_ask_about_congo("donor")),
            _computed(_ask_about_congo("recipient")),
            _computed(settle),
            _says("stop"),
        ]
    )
    adapter = _scripted_planner(scripted, surface=surface)
    harness = _AdapterHarness()

    assert await harness.run(adapter, question=draft.question, readings=[]) is None
    for pick in (
        f"{dataset.id}|donor|Congo|country/COD",
        f"{dataset.id}|recipient|Congo|country/COG",
    ):
        interrupt_id = _pending_ask_interrupt_id(await harness.events())
        answer = types.Content(
            role="user",
            parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": pick}})],
        )
        await harness.run(adapter, question=draft.question, readings=[], new_message=answer)

    proposals = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    asked = [
        part.function_call.args["candidates"]
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_call is not None and part.function_call.name == PLANNER_ASK
    ]
    assert [[c["spec_id"] for c in candidates] for candidates in asked] == [
        [f"{dataset.id}|{role}|Congo|country/{code}" for code in ("COD", "COG")]
        for role in ("donor", "recipient")
    ]
    assert len(proposals) == 2
    assert proposals[0]["translation"]["kind"] == "entity_name_ambiguous"
    if not admitted:
        assert proposals[-1]["code"] == ProblemCode.INVALID_PLAN.value
        assert len(surface.sent) == 1
        return
    assert proposals[-1]["outcome"] == "ok"
    sent = surface.sent[-1]["parts"][0]["entities"]
    assert sent == [
        {"text": "Congo", "dcid": "country/COD", "role": "donor"},
        {"text": "Congo", "dcid": "country/COG", "role": "recipient"},
    ]

    # Search restates each row as sent and binds it its own node; the recorded
    # plan's entities resolve to them per role.
    found = source(
        entities=(
            ResolvedEntity(role="origin", ref="country/COD", label="Congo, Dem. Rep."),
            ResolvedEntity(role="destination", ref="country/COG", label="Congo"),
        ),
        bound_refs=bound_refs(
            {
                "restatement": {"entities": sent},
                "entity_refs": [{"text": row["text"], "refs": [row["dcid"]]} for row in sent],
            }
        ),
        identity=None,
    )
    recorded = load_plan(await harness.state())
    assert recorded is not None
    (stored,) = [r for r in recorded.evidence_requirements if isinstance(r, DatasetRequirement)]
    resolved = {entity.role: resolve_name(entity, found) for entity in stored.entities}
    assert {role: entity.ref if entity else None for role, entity in resolved.items()} == {
        "donor": "country/COD",
        "recipient": "country/COG",
    }


async def test_a_repeating_referral_ends_the_entry_on_its_second_identical_referral(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Each proposal is translated, and every `search` answers `no_data`, so
    the second unchanged proposal refers the same `(kind, requirement_id,
    constraint)` back. That ends the entry with no further generation, on a refusal whose
    sentence states the referral, and another plan may still answer."""
    surface = _SkillPagesSurface(schemas=[], responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai([_computed(_propose_again_step())] * 2)
    adapter = _scripted_planner(scripted, surface=surface)

    result = await _run_adapter(
        adapter,
        question="gdp guatemala",
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    referred = {"kind": "no_data", "requirement_id": "gdp_gtm"}
    assert result["refused"] == {
        "text": refusal_sentence({"code": StopCode.PLAN_NOT_READY, "detail": referred}),
        "final": False,
        "code": StopCode.PLAN_NOT_READY,
    }
    assert len(scripted.requests) == 2


async def test_a_run_leaves_every_event_it_appended_scoped_to_the_planner(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The planner's own events carry `_PLANNER_BRANCH`, and no other agent runs.

    ADK's `_is_event_belongs_to_branch` returns True when either side's branch
    is empty, so keeping the planner's turns out of another agent's request
    (the orchestrator's, on its own branch) rests entirely on the branch the
    run is stamped with. `ctx.run_node` takes it as
    `override_branch=_PLANNER_BRANCH`. This asserts what that bought, rather
    than the argument being passed.

    Translation runs inside the planner's own `propose_plan` call, so every function
    call and response the run appended is the planner's: no second agent
    makes a model call in a planner run.
    """
    scripted = ScriptedGenai([_computed(step) for step in _repeating_proposal_script(1)])
    adapter = _scripted_planner(
        scripted,
        surface=_SkillPagesSurface(schemas=[], responses={"search": _NO_DATA_SEARCH_RESPONSE}),
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    harness = _AdapterHarness()

    await harness.run(
        adapter,
        question=draft.question,
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    events = await harness.events()
    planner_events = [event for event in events if event.author == "dfo_planner"]
    tool_authors = {
        event.author
        for event in events
        if event.get_function_calls() or event.get_function_responses()
    }

    assert any(event.get_function_responses() for event in planner_events), (
        "the run answered no planner tool call, so there is nothing to assert on"
    )
    assert tool_authors == {"dfo_planner"}
    assert {event.branch for event in planner_events} == {_PLANNER_BRANCH}, (
        "a planner event this run appended is not scoped to "
        f"{_PLANNER_BRANCH!r}: "
        f"{sorted({str(event.branch) for event in planner_events})!r}. Without "
        "`override_branch=_PLANNER_BRANCH` on the planner's `ctx.run_node` call, ADK "
        "treats an unbranched event as belonging to every branch."
    )


def _branch_check(*branches: tuple[str, str | None]) -> None:
    """Run `_check_branches` over one event per `(author, branch)` pair."""
    _check_branches(
        [Event(author=author, branch=branch) for author, branch in branches],
        planner="dfo_planner",
    )


@pytest.mark.parametrize("branch", [None, "", "dfo_orchestrator", "dfo_planner.x", "dfo_planner@1"])
def test_a_planner_event_off_the_planner_branch_fails_the_branch_check(
    branch: str | None,
) -> None:
    """The planner's events carry exactly `_PLANNER_BRANCH`. Any other value fails, an empty one included."""
    with pytest.raises(PlannerAdapterError, match="other than"):
        _branch_check(("dfo_planner", branch))


@pytest.mark.parametrize(
    "branches",
    [(), (("user", None),), (("dfo_orchestrator", "dfo_orchestrator"),)],
)
def test_a_slice_with_no_planner_event_passes_the_branch_check(
    branches: tuple[tuple[str, str | None], ...],
) -> None:
    """A run may leave no planner-authored event. A resumed run that finishes the
    model turn the user's answer completed appends none, and the check judges the
    events it is given."""
    _branch_check(*branches)


def test_the_branch_check_ignores_events_from_other_authors() -> None:
    """Only the planner's events are judged, so a user event may be bare."""
    _branch_check(
        ("user", None),
        ("dfo_planner", _PLANNER_BRANCH),
    )


async def test_translation_finds_the_plan_a_bare_amendment_records(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The plan round-trips through session state alone.

    The planner proposes with nothing but `amends_version`, and
    `run_translate` reaching the draft through `current_draft(tool_context)`
    is enough to conclude: the stub answers `no_data`, and the referral
    names `gdp_gtm`, the plan's own requirement id, which no argument
    carried. The planner's next generation reads that referral off the
    proposal's `translation`.
    """
    surface = _SkillPagesSurface(schemas=[], responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai([_computed(_propose_again_step()), _says("stop")])
    adapter = _scripted_planner(scripted, surface=surface)

    await _run_adapter(
        adapter,
        question=draft.question,
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    answers = [
        (part.function_response.response or {}).get("translation")
        for request in scripted.requests
        for content in _contents(request)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert any(
        isinstance(answer, dict) and answer.get("requirement_id") == "gdp_gtm" for answer in answers
    ), f"translation never reached the plan's own requirement: {answers!r}"


def _candidate_spec(spec_id: str, sentence: str) -> dict[str, Any]:
    """One discovered source, in the shape `search` serves it."""
    return {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": sentence,
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 2010, "end_year": 2024},
            },
            "stat_vars": {
                "items": [{"ref": "AidDisbursed", "label": "Aid disbursed", "slot_values": []}],
                "member_count": 1,
                "truncated": False,
            },
            "entities": [{"role": "recipient", "label": "Guatemala", "ref": "country/GTM"}],
            "facets": {"unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"}},
        },
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {"stat_var": "x"}},
    }


_TWO_CANDIDATES_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "candidates",
    "parts": [
        {
            "status": "candidates",
            "variable_text": "asked",
            "warnings": [],
            "candidates": {
                "specs": [
                    _candidate_spec(SPEC, "aid from France"),
                    _candidate_spec(SPEC_B, "aid to Togo"),
                ],
                "truncated": False,
            },
        }
    ],
}

_TWO_CANDIDATES_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_TWO_CANDIDATES_SEARCH_RESPONSE_ENVELOPE)
)


async def test_the_planners_proposal_carries_the_referral_with_its_outcome(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """On the path the planner is served from, the `translation` of the
    proposal's function response is the referral itself: its `outcome` on top
    (`choice_required` for two candidates), its `kind`, its requirement, and
    both candidates with the `spec_id` each may be pinned by.
    """
    surface = _SkillPagesSurface(schemas=[], responses={"search": _TWO_CANDIDATES_SEARCH_RESPONSE})
    draft = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm", measure="aid disbursed"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )
    scripted = ScriptedGenai([_computed(_propose_again_step()), _says("stop")])
    adapter = _scripted_planner(scripted, surface=surface)

    await _run_adapter(
        adapter,
        question="aid to guatemala",
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    (response,) = [
        part.function_response
        for part in _contents(scripted.requests[1])[-1].parts or []
        if part.function_response is not None
    ]
    assert response.name == "propose_plan"
    proposal = cast(dict[str, Any], response.response)
    assert proposal["outcome"] == "ok", proposal
    answer = proposal["translation"]
    assert answer["outcome"] == "choice_required", answer
    assert answer["kind"] == "source_ambiguous"
    assert answer["requirement_id"] == "gdp_gtm"
    assert {candidate["spec_id"] for candidate in answer["candidates"]} == {SPEC, SPEC_B}


async def test_two_conversations_on_one_planner_each_report_into_their_own_stream(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """One `PlannerAdapter` serves every conversation, and each run reports to
    the call it belongs to. Conversation A pauses on its first planner generation,
    conversation B runs its whole turn, and only then does A call `propose_plan`:
    A's report has to land in A's own `on_progress` stream, and B's stream
    must not grow once A resumes.

    One script answers both conversations in the order their calls arrive:
    A's held first generation, B's one generation, then A's generation after
    its proposal.
    """
    started_a = asyncio.Event()
    release_a = asyncio.Event()
    scripted = ScriptedGenai(
        [
            held(_computed(_propose_again_step()), paused=started_a, release=release_a),
            _says("nothing at all"),
            _says("nothing yet"),
        ]
    )
    adapter = _scripted_planner(
        scripted,
        surface=_SkillPagesSurface(schemas=[], responses={"search": _NO_DATA_SEARCH_RESPONSE}),
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    reports_a: list[dict[str, Any]] = []
    reports_b: list[dict[str, Any]] = []

    harness_a = _AdapterHarness()
    harness_b = _AdapterHarness()

    async def _run_a() -> None:
        await harness_a.run(
            adapter,
            question="gdp guatemala",
            readings=[],
            prior_plan=draft.model_dump(mode="json"),
            on_progress=reports_a.append,
        )

    task_a = asyncio.create_task(_run_a())
    await started_a.wait()

    # Conversation B's whole turn, in this coroutine's own task, while A
    # sits suspended above, before its `propose_plan` call.
    await harness_b.run(
        adapter, question="unrelated question", readings=[], on_progress=reports_b.append
    )
    reports_b_after_b = list(reports_b)

    release_a.set()
    await task_a

    assert reports_b == reports_b_after_b, (
        "conversation B's on_progress stream grew after B's own turn had already "
        f"finished -- conversation A's own propose_plan call reported into it: {reports_b}"
    )
    assert any(report["tools"] == ["propose_plan"] for report in reports_a), (
        "conversation A's own propose_plan call never reported into A's own on_progress stream"
    )


def test_planner_adapter_error_carries_a_do_not_retry_refusal_by_default() -> None:
    exc = PlannerAdapterError("the pair did not converge")

    assert exc.refusal["retry_mode"] == "do_not_retry"
    assert exc.refusal["fix"]["detail"] == "the pair did not converge"


# -- the plan-round budget resets per turn, not per call -------------------


def _amendment_proposal(
    decision: Any, requirement: Any, derived_requirement: Any
) -> dict[str, Any]:
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    return {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }


def _amend_step(proposal: dict[str, Any], *, start_version: int) -> Any:
    """A script entry that calls `propose_plan` as an amendment of whatever
    version this turn's recorded plan stands at.

    `propose_plan` refuses a call that states neither `amends_version` nor
    `starts_new_plan` once a plan is recorded, and refuses a version that is
    not the recorded one, so a script that proposes round after round within
    one turn has to read each answer's own `version`. `start_version` is the
    version the turn opens on: the version of a plan the test seeded or an
    earlier turn recorded. It goes unconsulted on the first call of a turn
    whose session holds no plan, since that call is a first draft and its
    intent is not in question.
    """

    def step(request: Any) -> list[tuple[str, dict[str, Any]]]:
        version = _last_amended_version(request, default=start_version)
        return [("propose_plan", {**proposal, "amends_version": version})]

    return step


async def test_a_restart_on_the_amend_edge_is_refused_as_question_fixed(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The `amend` edge (`approve_n -> planner_n`) seeds `prior_plan` from
    the graph's own record, so a plan is already recorded when this call's
    one generation runs. This seam's pair is always built with
    `question_fixed=True` (`PlannerAdapter.__init__`), so a `starts_new_plan`
    call there is refused as `QUESTION_FIXED` -- terminal, recording
    nothing -- rather than beginning a second plan for a different
    question out from under the graph's own `question` state.
    """
    seeded = _translated_plan(decision, requirement, derived_requirement, version=3)
    scripted = ScriptedGenai(
        [_calls(("propose_plan", {"starts_new_plan": True, "question": "a different question"}))]
    )
    adapter = _scripted_planner(scripted)
    harness = _AdapterHarness()

    result = await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result is not None and "refused" in result
    assert "own message" in result["refused"]["text"]
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 3
    assert plan.question == "kenya debt"


async def test_a_second_turn_on_one_session_clears_the_first_drafts_plan_and_briefed_flag(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The orchestrator can call `plan` again in one session after an
    earlier entry handed back, and the planner node passes no `prior_plan`
    on that later first-draft entry even though `ctx.state["plan"]` still
    holds the earlier entry's. `prior_plan is None` clears
    `planner.plan.PLAN_STATE_KEY` and the briefed flag at the top of
    `__call__` (see its own comment), so the second entry both renders the
    full brief again (checked against the rendered prompt, not a flag) and
    drafts its own plan. It is not refused for restating no `amends_version`
    against a plan still on record from the first entry.
    """
    harness = _AdapterHarness()
    second_proposal = {
        "question": "kenya oda 2023",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }
    # Both entries' generations, in order: the first entry's declaration, then
    # the second's proposal and the generation its answer asks for.
    scripted = ScriptedGenai(
        [_calls(*_DECLARE), _calls(("propose_plan", second_proposal)), _says("done")]
    )
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    first_plan = _translated_plan(decision, requirement, derived_requirement, version=1)

    first = await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        prior_plan=first_plan.model_dump(mode="json"),
    )
    assert first is not None and "plan" in first
    first_requests = len(scripted.requests)

    # A seam with the real `propose_plan`, on the same session.
    adapter = _scripted_planner(scripted)

    # This second entry's draft carries no translation, so it is never
    # declared ready and the entry ends with no plan -- the recorded plan
    # `propose_plan` itself left behind is what this test is about.
    await harness.run(adapter, question="kenya oda 2023", readings=[])

    # The second call's own generation carries its own question.
    brief = _prompt_text(scripted.requests[first_requests])
    assert "Question: kenya oda 2023" in brief

    # If the clear had not fired, `propose_plan`'s first-draft call above
    # (stating neither `amends_version` nor `starts_new_plan`) would have
    # been refused as `INVALID_PLAN` against the first turn's still-
    # recorded plan, recording nothing -- so the plan on record would
    # still be the first turn's own, at "kenya debt". It is instead the
    # second turn's own first draft.
    second_plan = load_plan(await harness.state())
    assert second_plan is not None
    assert second_plan.question == "kenya oda 2023"
    assert second_plan.version == 1
    assert second_plan.previous_version is None


async def test_a_later_first_draft_reads_none_of_the_earlier_plans_alignments(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The earlier entry's plan moved its `second` onto a compile. A later
    first-draft entry of the turn clears that plan, and its alignments go with
    it, so they are gone even when the entry records no plan of its own."""
    harness = _AdapterHarness()
    # The first entry's declaration, then the later entry's call and the
    # generation its answer asks for.
    scripted = ScriptedGenai([_calls(*_DECLARE), _calls(*_DECLARE), _says("done")])
    adapter = _scripted_planner(scripted, compiled_proposals=True)
    first_plan = _translated_plan(decision, requirement, derived_requirement, version=1)
    first = await harness.run(
        adapter, question="kenya debt", readings=[], prior_plan=first_plan.model_dump(mode="json")
    )
    assert first is not None and "plan" in first
    moved = AlignedInput(
        requirement_id="second",
        spec_id=f"spec_{'3' * 32}",
        origin=f"spec_{'4' * 32}",
        option_ids=("measure_basis:gross",),
        stated_pin=f"spec_{'4' * 32}",
    )
    harness.seed_state(ALIGNMENTS_STATE_KEY, [moved.model_dump(mode="json")])

    adapter = _scripted_planner(scripted)
    await harness.run(adapter, question="kenya oda 2023", readings=[])

    state = await harness.state()
    assert load_plan(state) is None
    assert alignments(state) == {}


_ASK_A_SOURCE = (
    PLANNER_ASK,
    {
        "message": "Which source did you mean?",
        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
        "recommended_spec_id": SPEC,
        "rationale": "broader coverage",
    },
)

# A translation's referral the planner's run ends on asking about.
_NEEDS_REPAIR = {
    "outcome": "invalid_input",
    "kind": "requirement_needs_repair",
    "requirement_id": "debt",
    "constraint": "requirement_unmeetable",
    "reason": "no source states it",
}


# What a key the stored session lacks reads as.
_ABSENT = object()


class _Drafts(NamedTuple):
    """What a scenario proposes, and the translated plan an entry amends."""

    proposal: dict[str, Any]
    plan: Plan


async def _assert_stored_as_live(harness: _AdapterHarness) -> None:
    """Every key the last run left in the live session state, but a `temp:`
    one, is in the session reloaded from the store with the same value. A key
    the live state holds as `None` may be absent from the store."""
    stored = await harness.state()
    lost: dict[str, tuple[Any, Any]] = {}
    for key, value in harness.live_state().items():
        held = stored.get(key, _ABSENT)
        if key.startswith(State.TEMP_PREFIX) or held == value:
            continue
        if value is None and held is _ABSENT:
            continue
        lost[key] = (value, held)
    assert lost == {}


def _answer_to_the_last_ask(events: list[Event]) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            create_request_input_response(
                _pending_ask_interrupt_id(events), {"result": {"spec_id": SPEC}}
            )
        ],
    )


async def _first_draft_proposes(harness: _AdapterHarness, drafts: _Drafts) -> None:
    scripted = ScriptedGenai([_calls(("propose_plan", drafts.proposal)), _says("stop")])
    adapter = _scripted_planner(scripted)
    await harness.run(adapter, question="kenya debt", readings=[])
    await _assert_stored_as_live(harness)


async def _first_draft_asks(harness: _AdapterHarness, drafts: _Drafts) -> None:
    del drafts
    adapter = _scripted_planner(ScriptedGenai([_calls(_ASK_A_SOURCE)]))
    assert await harness.run(adapter, question="kenya debt", readings=[]) is None
    await _assert_stored_as_live(harness)


async def _asks_after_a_referral(harness: _AdapterHarness, drafts: _Drafts) -> None:
    async def referring(call: Any) -> dict[str, object]:
        del call
        return dict(_NEEDS_REPAIR)

    scripted = ScriptedGenai([_calls(("propose_plan", drafts.proposal)), _calls(_ASK_A_SOURCE)])
    adapter = scripted_planner(scripted, translate=referring)
    assert await harness.run(adapter, question="kenya debt", readings=[]) is None
    await _assert_stored_as_live(harness)


async def _resumed_after_an_answer(harness: _AdapterHarness, drafts: _Drafts) -> None:
    scripted = ScriptedGenai(
        [_calls(_ASK_A_SOURCE), _calls(("propose_plan", drafts.proposal)), _says("stop")]
    )
    adapter = _scripted_planner(scripted)
    assert await harness.run(adapter, question="kenya debt", readings=[]) is None
    answer = _answer_to_the_last_ask(await harness.events())
    await harness.run(adapter, question="kenya debt", readings=[], new_message=answer)
    await _assert_stored_as_live(harness)


async def _resumed_proposes_and_asks_again(harness: _AdapterHarness, drafts: _Drafts) -> None:
    scripted = ScriptedGenai(
        [_calls(_ASK_A_SOURCE), _calls(("propose_plan", drafts.proposal)), _calls(_ASK_A_SOURCE)]
    )
    adapter = _scripted_planner(scripted)
    assert await harness.run(adapter, question="kenya debt", readings=[]) is None
    answer = _answer_to_the_last_ask(await harness.events())
    assert (
        await harness.run(adapter, question="kenya debt", readings=[], new_message=answer) is None
    )
    await _assert_stored_as_live(harness)


async def _resumed_from_a_legacy_ask(harness: _AdapterHarness, drafts: _Drafts) -> None:
    del drafts
    await _pause_on_a_legacy_ask(harness, {"message": "Which source did you mean?"})
    adapter = _scripted_planner(ScriptedGenai([_says("stop")]))
    answer = _answer_to_the_last_ask(await harness.events())
    await harness.run(adapter, question="kenya debt", readings=[], new_message=answer)
    await _assert_stored_as_live(harness)


async def _opened_by_a_card_pick(harness: _AdapterHarness, drafts: _Drafts) -> None:
    adapter = _scripted_planner(ScriptedGenai([_calls(_ASK_A_SOURCE)]))
    assert (
        await harness.run(
            adapter,
            question=drafts.plan.question,
            readings=[],
            prior_plan=drafts.plan.model_dump(mode="json"),
            pick=_PICKED,
        )
        is None
    )
    await _assert_stored_as_live(harness)


async def _amended(harness: _AdapterHarness, drafts: _Drafts) -> None:
    await _first_draft_proposes(harness, drafts)
    adapter = _scripted_planner(ScriptedGenai([_says("stop")]))
    await harness.run(adapter, question="kenya debt", readings=[], amendment="one total")
    await _assert_stored_as_live(harness)


async def _replanned_on_a_refusal(harness: _AdapterHarness, drafts: _Drafts) -> None:
    await _first_draft_proposes(harness, drafts)
    adapter = _scripted_planner(ScriptedGenai([_says("stop")]))
    await harness.run(
        adapter, question="kenya debt", readings=[], refusal=_rediscovery_refusal(SPEC)
    )
    await _assert_stored_as_live(harness)


async def _replanned_on_a_refusal_earning_a_decline(
    harness: _AdapterHarness, drafts: _Drafts
) -> None:
    adapter = _scripted_planner(ScriptedGenai([_says("stop")]))
    await harness.run(
        adapter,
        question=drafts.plan.question,
        readings=[],
        prior_plan=drafts.plan.model_dump(mode="json"),
        refusal=_unit_refusal("cross_frame_unit_unbound"),
    )
    await _assert_stored_as_live(harness)


async def _a_second_first_draft(harness: _AdapterHarness, drafts: _Drafts) -> None:
    await _first_draft_proposes(harness, drafts)
    await _first_draft_proposes(harness, drafts)


@pytest.mark.parametrize(
    "scenario",
    [
        _first_draft_proposes,
        _first_draft_asks,
        _asks_after_a_referral,
        _resumed_after_an_answer,
        _resumed_proposes_and_asks_again,
        _resumed_from_a_legacy_ask,
        _opened_by_a_card_pick,
        _amended,
        _replanned_on_a_refusal,
        _replanned_on_a_refusal_earning_a_decline,
        _a_second_first_draft,
    ],
)
async def test_an_entry_leaves_no_session_write_out_of_the_stored_session(
    scenario: Callable[[_AdapterHarness, _Drafts], Awaitable[None]],
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
) -> None:
    """Every key a planner entry leaves in the live session state, when it
    pauses on an ask or finishes, holds the same value in the stored session,
    except `temp:` keys. Nothing the seam or the planner's tools write is lost
    at the next `Runner.run_async`."""
    drafts = _Drafts(
        _amendment_proposal(decision, requirement, derived_requirement),
        _translated_plan(decision, requirement, derived_requirement),
    )
    await scenario(_AdapterHarness(), drafts)


async def test_a_sixth_amendment_turn_still_records_its_own_version(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Reproduces the production sequence: six separate amendment turns on
    the same question, each its own `PlannerAdapter.__call__`. `__call__`
    resets the counter at the top of every fresh turn (never on a resume --
    see the resume test below), so every turn gets its own four-round
    budget and a sixth amendment still lands as its own new version instead
    of silently repeating the fourth.

    No turn here compiles (the script only ever proposes), so every turn
    ends the same way, with no plan ready -- the version `propose_plan`
    recorded along the way is what this test is about.

    Each turn's one proposal amends the version the last turn recorded
    (`_amend_step`), read off that call's own answer, so the plan walks
    1-2-3-4-5-6 across the six turns rather than any turn resetting it at
    version 1.
    """
    harness = _AdapterHarness()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)

    # The version the recorded plan stands at as each turn opens. Turn 1
    # opens on a session with no plan recorded, so its default goes
    # unconsulted: that call is the first draft, recorded at version 1.
    version = 1

    def amend(request: Request) -> Reply:
        return _calls(*_amend_step(proposal, start_version=version)(request))

    # Each turn's proposal, then the generation its draft's answer asks for.
    scripted = ScriptedGenai([amend, _says("done")] * 6)
    adapter = _scripted_planner(scripted)
    for turn in range(1, 7):
        await harness.run(adapter, question="kenya debt", readings=[], amendment=f"turn {turn}")
        plan = load_plan(await harness.state())
        assert plan is not None
        version = plan.version

    assert version == 6


async def test_a_second_calls_brief_extends_the_first_byte_for_byte() -> None:
    """Two calls in one node, the shape the graph's `amend` and `replan` edges
    take inside one `Runner.run_async`. Each call's requests carry its own
    brief once, after the turn's question. The second brief opens with the
    first byte for byte, since `question`, `readings`, `canvas_digest` and
    `conversation_digest` do not change mid-turn, and only the ask is added
    after them.
    """
    question = "kenya debt"
    readings: list[dict[str, Any]] = []
    canvas_digest = {"blocks": ["a block"]}
    conversation_digest = {"turns": ["an earlier turn"]}

    scripted = ScriptedGenai([_says("still nothing"), _says("still nothing")])
    adapter = _scripted_planner(scripted)

    expected_first_message = adapter._message(
        question=question,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
        ledger=planner_adapter._readings_ledger(readings),
    )

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        await adapter(
            ctx=ctx,
            question=question,
            readings=readings,
            canvas_digest=canvas_digest,
            conversation_digest=conversation_digest,
        )
        await adapter(
            ctx=ctx,
            question=question,
            readings=readings,
            canvas_digest=canvas_digest,
            conversation_digest=conversation_digest,
            amendment="only since 2015",
        )

    node_fn = node(_node, name="adapter_under_test", rerun_on_resume=True)
    wf = Workflow(name="adapter_test_wf", edges=[(START, node_fn)])
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for event in runner.run_async(
        user_id=_USER, session_id=session.id, new_message=content, run_config=SSE
    ):
        assert isinstance(event, Event)

    assert _brief_follows_the_question(scripted.requests[0]) == expected_first_message
    second = _brief_follows_the_question(scripted.requests[1])
    assert second == expected_first_message + planner_adapter._ask_text(
        amendment="only since 2015", refusal=None, pick=None
    )


async def test_a_turn_that_loops_past_the_round_cap_is_refused_not_raised(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`MAX_PLAN_ROUNDS` bounds the cumulative `propose_plan` calls made
    *within* one turn, since the model can call it more than once inside a
    single `_propose`/`ctx.run_node` call and nothing but the round cap counts
    that. This scripts five `propose_plan` calls in one continuous planner
    turn (one `__call__`, one `_propose`, and no second turn), so the fifth is the one that goes past the cap. The seam returns a
    refusal and does not raise `PlannerAdapterError`: raising it would end the
    turn through `_stream_fatal`'s `close_turn(error=...)`, which `begin_turn`
    documents as abandoning the paused session and its plan, so the last
    recorded version could no longer be approved.

    Every call after the first amends what the call before it recorded
    (`_amend_step`), so the rounds actually accrue: the first drafts
    version 1, the next three amend 1, 2 and 3, and the fifth, amending
    version 4, is the one past the cap.
    """
    proposal = _amendment_proposal(decision, requirement, derived_requirement)
    step = _computed(_amend_step(proposal, start_version=1))
    scripted = ScriptedGenai([_calls(("propose_plan", proposal)), step, step, step, step])
    adapter = _scripted_planner(scripted)

    result = await _run_adapter(adapter, question="kenya debt", readings=[])

    assert "refused" in result
    text = result["refused"]["text"]
    assert f"budget of {MAX_PLAN_ROUNDS} plan drafts" in text
    assert "fresh budget" in text


async def test_a_refused_loop_leaves_the_last_recorded_version_on_record(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The fifth `propose_plan` call's own refusal writes nothing
    (`planner.tools.propose` returns its `problem(...)` before ever
    calling `save_plan`), so whatever the fourth call actually recorded
    stays exactly as it was -- proven here directly against session state
    rather than inferred from the refusal text.

    The four calls before it amend what each recorded (`_amend_step`), so
    the plan on record is the fourth: version 4, still a draft.
    """
    harness = _AdapterHarness()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)
    step = _computed(_amend_step(proposal, start_version=1))
    scripted = ScriptedGenai([_calls(("propose_plan", proposal)), step, step, step, step])
    adapter = _scripted_planner(scripted)

    result = await harness.run(adapter, question="kenya debt", readings=[])

    assert result is not None and "refused" in result
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 4
    assert plan.status is PlanStatus.DRAFT


async def test_the_round_budget_is_not_reset_on_a_resume_of_the_same_turn(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A mid-turn resume re-enters `__call__` with `ctx.resume_inputs`
    carrying the interrupt's own answer. Driven through a real planner ask
    and a real `create_request_input_response` answer, so this is the same
    boundary a live pause-and-resume crosses, not a double standing in for
    one.

    Three rounds are spent for real, through `propose_plan`'s own
    persisted write, in the same continuous planner turn that then asks
    and pauses. They are not seeded directly onto the stored session: a
    direct, non-delta-tracked write (what an ungated reset itself would
    do) proves nothing about whether the *reset* fired, only about what
    was written after it. Scripting a resumed turn that calls
    `propose_plan` twice more reads the count the way `propose_plan`
    itself does, through its own persisted write, which is the only
    reliable way to observe it. The fifth call is refused only if the
    three rounds and the fourth survived the resume. An ungated reset
    would land the fifth call within a fresh 1-2 budget, succeeding
    rather than refusing at all.
    """
    harness = _AdapterHarness()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)

    # One continuous planner turn: three real `propose_plan` rounds (no
    # `prior_plan` seeded, so the first call starts a brand-new plan and
    # each later one amends the version the call before it durably recorded,
    # through `_amend_step` -- `test_a_sixth_amendment_turn_still_records_
    # its_own_version`'s same pattern), then an ask that pauses mid-turn.
    # The two calls of the resumed entry come last.
    step = _computed(_amend_step(proposal, start_version=1))
    resumed_step = _computed(_amend_step(proposal, start_version=3))
    scripted = ScriptedGenai(
        [
            _calls(("propose_plan", proposal)),
            step,
            step,
            _calls(
                (
                    PLANNER_ASK,
                    {
                        "message": "Which source did you mean?",
                        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ),
            resumed_step,
            resumed_step,
        ]
    )
    adapter = _scripted_planner(scripted)
    paused = await harness.run(adapter, question="kenya debt", readings=[])
    paused_requests = len(scripted.requests)
    assert paused is None, "the ask should pause this call, not resolve it"

    # The resumed entry: `ctx.resume_inputs` carries the interrupt's own
    # answer, populated by ADK's own resume machinery, not set by hand.
    # `propose_plan` runs twice more -- the fourth and fifth cumulative
    # rounds if the three spent before the ask survived, only the first and
    # second if an ungated reset wiped them. Gated correctly, the fourth is
    # still within the cap and the fifth goes past it. The answer runs the
    # planner again, whose brief shows the version the paused run recorded,
    # version 3, which the fourth call amends.
    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC}})],
    )
    result = await harness.run(
        adapter,
        question="kenya debt",
        readings=[],
        amendment="one more change",
        new_message=resume_message,
    )

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert f"budget of {MAX_PLAN_ROUNDS} plan drafts" in text
    assert "fresh budget" in text
    # The resumed run sees none of the paused run's calls, so its brief shows
    # the draft the paused run recorded as the plan being changed.
    resumed = scripted.requests[paused_requests]
    assert "version 3" in _brief_follows_the_question(resumed)


# -- the request's budget ---------------------------------------------------


def _request_budget(**overrides: Any) -> TurnBudget:
    bounds: dict[str, Any] = {
        "model_time_s": 90.0,
        "model_calls_max": 30,
        "wall_s": 600.0,
        "first_chunk_s": 12.0,
    }
    return TurnBudget(**{**bounds, **overrides})


async def test_discovery_inside_an_entry_reads_the_request_s_wall_bound() -> None:
    """Inside the run a discovery call reads what is left of the request's
    wall-clock bound; outside a request, and outside the run, none is set."""
    seen: list[float | None] = []

    def _read_time_left(_request: Request) -> Reply:
        seen.append(discovery_time_left())
        return _says("done")

    scripted = ScriptedGenai([_read_time_left, _read_time_left])
    adapter = _scripted_planner(scripted)
    budget = _request_budget(wall_s=600.0, clock=lambda: 5.0)

    token = current_turn_budget.set(budget)
    try:
        await _run_adapter(adapter, question="kenya debt", readings=[])
    finally:
        current_turn_budget.reset(token)
    await _run_adapter(adapter, question="kenya debt", readings=[])

    assert seen == [600.0, None]
    assert discovery_time_left() is None


async def test_a_call_cut_inside_an_entry_and_still_running_is_cancelled_when_the_entry_ends() -> (
    None
):
    kept: list[asyncio.Task[Any]] = []

    def _keep_a_cut_call(_request: Request) -> Reply:
        cut_calls = current_cut_calls.get()
        assert cut_calls is not None
        task = asyncio.ensure_future(asyncio.sleep(3600))
        cut_calls.keep(f"call {len(kept)}".encode(), task)
        kept.append(task)
        return _says("done")

    scripted = ScriptedGenai([_keep_a_cut_call])
    adapter = _scripted_planner(scripted)

    await _run_adapter(adapter, question="kenya debt", readings=[])

    assert kept
    assert all(task.cancelled() for task in kept)
    assert current_cut_calls.get() is None


@pytest.mark.parametrize(
    ("spent", "code"),
    [
        ({"model_time_spent": 90.0}, StopCode.TURN_MODEL_TIME_SPENT),
        ({"model_calls": 30}, StopCode.TURN_MODEL_CALLS_SPENT),
        ({"wall_s": 0.0}, StopCode.TURN_WALL_TIME_SPENT),
    ],
)
async def test_a_run_whose_model_call_the_budget_closed_ends_the_entry_on_that_bound(
    spent: dict[str, float], code: StopCode
) -> None:
    """The model answers a closed call with the bound's sentence
    (`harness.turn_budget.bounded`) and makes no request. With no referral to
    ask about, that ends the planner's run, and the entry refuses on the
    bound, final for the request, in place of `StopCode.PLAN_NOT_READY`."""
    budget = _request_budget()
    for name, value in spent.items():
        setattr(budget, name, value)
    scripted = ScriptedGenai([_says("never sent")])
    adapter = _scripted_planner(scripted)

    token = current_turn_budget.set(budget)
    try:
        result = await _run_adapter(adapter, question="kenya debt", readings=[])
    finally:
        current_turn_budget.reset(token)

    assert scripted.requests == []
    assert result["refused"] == {
        "text": refusal_sentence({"code": code}),
        "final": True,
        "code": code,
    }


def test_pinnable_candidates_filters_to_spec_id_bearing_entries() -> None:
    """A `no_data` referral's own candidates (`translator.discovery.
    compact_source(selectable=False)`) carry no `spec_id`; the same filter
    `_stalled_referral`'s own `named` text applies drops them here too."""
    referral = {
        "kind": "source_ambiguous",
        "candidates": [
            {"spec_id": SPEC, "sentence": "aid from France"},
            {"sentence": "a nearby series with no spec_id"},
            {"spec_id": SPEC_B, "sentence": "aid to Togo"},
        ],
    }

    assert _pinnable_candidates(referral) == [
        {"spec_id": SPEC, "label": "aid from France"},
        {"spec_id": SPEC_B, "label": "aid to Togo"},
    ]


def test_pinnable_candidates_is_empty_for_no_referral_or_no_candidates() -> None:
    assert _pinnable_candidates(None) == []
    assert _pinnable_candidates({"kind": "no_operation_covers"}) == []
    assert _pinnable_candidates({"kind": "no_data", "candidates": "not a list"}) == []


def test_clarifying_question_args_offers_the_candidates_and_recommends_none() -> None:
    candidates = [{"spec_id": SPEC, "label": "a"}, {"spec_id": SPEC_B, "label": "b"}]

    args = _clarifying_question_args(candidates, lead="Why the user is asked.")

    assert args == {"message": args["message"], "candidates": candidates}
    # The message reaches the user: the candidates' labels name the sources.
    assert "spec_" not in args["message"]


# A translation referring the plan back to choose between two sources.
_SOURCE_AMBIGUOUS = {
    "outcome": "choice_required",
    "kind": "source_ambiguous",
    "requirement_id": "gdp_gtm",
    "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
}


async def _budget_closes_after(
    *translations: dict[str, Any],
    spend: Callable[[TurnBudget], None],
) -> tuple[dict[str, Any] | None, list[Event]]:
    """One entry whose planner proposes once for each of `translations`, each
    proposal recorded and translated as it names, and whose next call the
    request's budget closes, `spend` having spent it during the last proposal's
    generation. Returns the entry's output and the session's events."""
    budget = _request_budget()
    left = list(translations)

    def answer(call: Any, end_run: Any, arguments: dict[str, Any]) -> object:
        del call, end_run, arguments
        return {"outcome": "ok", "status": "draft", "version": 1, "translation": left.pop(0)}

    def tools(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        proposal = ToolSpec(
            name="propose_plan", description="Propose the plan.", parameters=(), body=answer
        )
        return [proposal if spec.name == "propose_plan" else spec for spec in specs]

    def last_proposal(_request: Request) -> Reply:
        spend(budget)
        return _calls(("propose_plan", {}))

    replies: list[Step] = [_calls(("propose_plan", {})) for _ in translations[1:]]
    scripted = ScriptedGenai([*replies, last_proposal])
    adapter = scripted_planner(scripted, tools=tools)
    harness = _AdapterHarness()
    token = current_turn_budget.set(budget)
    try:
        output = await harness.run(adapter, question="gdp guatemala", readings=[])
    finally:
        current_turn_budget.reset(token)
    assert len(scripted.requests) == len(translations), "the closed call was sent"
    return output, await harness.events()


def _asks(events: list[Event]) -> list[types.FunctionCall]:
    """The planner's `ask_user_question` calls among `events`."""
    return [
        call for event in events for call in event.get_function_calls() if call.name == PLANNER_ASK
    ]


def _spend_model_time(budget: TurnBudget) -> None:
    budget.model_time_spent = budget.model_time_s


def _spend_model_calls(budget: TurnBudget) -> None:
    budget.model_calls = budget.model_calls_max


async def test_a_referral_resolved_by_a_later_ok_translation_lets_the_budget_refuse_instead_of_asking() -> (
    None
):
    """Once a later `ok` translation has cleared the run's last referral
    (`test_last_referral_is_cleared_once_a_later_translation_resolves_the_requirement`),
    a call the budget closes finds no pinnable candidates: its closing text
    stands and the entry refuses on the bound, never the `ask_user_question`
    ask a still-open referral would get."""
    output, events = await _budget_closes_after(
        _SOURCE_AMBIGUOUS,
        {"outcome": "ok", "requirement_id": "gdp_gtm"},
        spend=_spend_model_time,
    )

    assert _asks(events) == [], "a cleared referral must not surface an ask"
    code = StopCode.TURN_MODEL_TIME_SPENT
    assert output == {
        "refused": {"text": refusal_sentence({"code": code}), "final": True, "code": code}
    }


async def test_a_referral_surviving_a_malformed_translation_still_gets_asked_when_the_budget_closes() -> (
    None
):
    """A translation with no `outcome` settles nothing
    (`test_last_referral_survives_a_translation_carrying_no_outcome`), so a
    call the budget closes after it still finds the earlier referral's
    candidates and asks through `ask_user_question`, the same as if the
    malformed response had never arrived."""
    output, events = await _budget_closes_after(
        _SOURCE_AMBIGUOUS, {"malformed": True}, spend=_spend_model_calls
    )

    assert output is None, "the budget's question pauses the entry, which refuses nothing"
    (ask,) = _asks(events)
    assert {c["spec_id"] for c in (ask.args or {})["candidates"]} == {
        "spec_nominal",
        "spec_per_capita",
    }


def test_a_closing_code_outside_stop_code_still_asks_on_a_referral_with_candidates() -> None:
    """The closing code is read as a `StopCode` only when the entry refuses on
    it, so a code no member names still asks the user over a referral that
    names sources to pin."""
    tracker = PlannerRun(agent="dfo_planner")
    _track_proposal(
        tracker,
        {
            "outcome": "choice_required",
            "kind": "source_ambiguous",
            "requirement_id": "gdp_gtm",
            "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
        },
    )

    turn = planner_node.directive_turn(tracker.model_answered("closed_by_a_newer_bound"))

    assert turn is not None and turn.parts is not None
    (part,) = turn.parts
    assert part.function_call is not None and part.function_call.name == PLANNER_ASK
    assert tracker.refused is None


def test_a_closing_code_outside_stop_code_raises_when_nothing_is_pinnable() -> None:
    """With nothing to pin the entry refuses on the closing code, so a code no
    `StopCode` member names raises and records no refusal."""
    tracker = PlannerRun(agent="dfo_planner")

    with pytest.raises(ValueError, match="closed_by_a_newer_bound"):
        tracker.model_answered("closed_by_a_newer_bound")
    assert tracker.refused is None


async def test_a_budget_closing_on_a_referral_with_candidates_asks_the_user_and_the_answer_elects(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A call the request's budget closes on a referral carrying pinnable
    candidates ends the entry the way the planner's own ask does: an
    `ask_user_question` pause, built from the referral's own
    candidates with no model call. The user's answer resumes the run in a
    request of its own and pairs by id (`planner.call.elected_spec_ids`).

    The cap allows one model call: it proposes the plan again, and its
    translation records a `source_ambiguous` referral naming `SPEC` and
    `SPEC_B` (`_TWO_CANDIDATES_SEARCH_RESPONSE`). The next call finds the cap
    spent.
    """
    draft = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm", measure="aid disbursed"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )
    scripted = ScriptedGenai([_computed(_propose_again_step()), _says("noted, but not done yet")])
    adapter = _scripted_planner(
        scripted,
        surface=_SkillPagesSurface(
            schemas=[], responses={"search": _TWO_CANDIDATES_SEARCH_RESPONSE}
        ),
    )
    harness = _AdapterHarness()

    token = current_turn_budget.set(_request_budget(model_calls_max=1))
    try:
        paused = await harness.run(
            adapter, question=draft.question, readings=[], prior_plan=draft.model_dump(mode="json")
        )
    finally:
        current_turn_budget.reset(token)
    assert paused is None, "the budget's clarifying question should pause this call"
    assert len(scripted.requests) == 1

    events = await harness.events()
    interrupt_id = _pending_ask_interrupt_id(events)
    (ask_call,) = [
        part.function_call
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_call is not None and part.function_call.name == PLANNER_ASK
    ]
    assert {c["spec_id"] for c in ask_call.args["candidates"]} == {SPEC, SPEC_B}
    assert "recommended_spec_id" not in ask_call.args
    assert "gdp_gtm" not in ask_call.args["message"]

    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC_B}})],
    )
    await harness.run(
        adapter,
        question=draft.question,
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
        new_message=resume_message,
    )

    fake_tool_context = SimpleNamespace(
        state={}, function_call_id=None, session=SimpleNamespace(events=await harness.events())
    )
    assert elected_spec_ids(SessionCall.of(fake_tool_context)) == frozenset({SPEC_B})


async def test_a_submitted_plan_ends_the_planner_run_with_no_further_model_call(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A compiled proposal declared ready is the planner's last event: the planner
    asks the model for nothing after it, so the submission stands."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    scripted = ScriptedGenai([_calls(*_DECLARE)])
    adapter = _scripted_planner(scripted, compiled_proposals=True)

    result = await _run_adapter(
        adapter, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
    )

    assert len(scripted.requests) == 1
    assert result is not None and result["plan"]["version"] == plan.version


# -- the first year a pinned dataset holds, read when a plan needs it -------------

# The dataset Search found for "the last five years": answered over 2020-2024.
_AID_2020_2024 = {
    "identity_version": "spec1",
    "measure": "education aid",
    "requested_window": {"kind": "range", "start_year": 2020, "end_year": 2024},
}
_AID_SPEC = spec_id_for(canonicalize(_AID_2020_2024))
_LAST_FIVE_YEARS = "How has education aid to Kenya grown over the last five years?"
_AID_EVERY_YEAR_SPEC, _AID_EVERY_YEAR = windowed_identity(_AID_2020_2024, {"kind": "all"})


class _DefinitionReads:
    """A tool surface whose `resolve_definition` answers `answer()`, recording each call it takes.

    Every other tool is unreachable, so the translation each recorded plan
    gets stops on a transient referral. Translation still resolves a pinned
    copy over the plan's period through `resolve_definition`, so a test
    about the first year reads `first_year_reads`.
    """

    def __init__(self, answer: Callable[[], ToolCallResult]) -> None:
        self._answer = answer
        self.calls: list[Mapping[str, object]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            raise ToolSurfaceUnavailableError(f"{tool.name} is not served here", retryable=True)
        self.calls.append(arguments)
        return self._answer()

    @property
    def first_year_reads(self) -> list[Mapping[str, object]]:
        """The calls resolving a dataset over no window, the read `propose_plan` takes."""
        return [
            call
            for call in self.calls
            if cast(dict[str, Any], call["identity"])["requested_window"] == {"kind": "all"}
        ]


def _held_since_1995() -> ToolCallResult:
    """data-mcp's answer for the dataset over no window: held from 1995, 870 rows."""
    block = {
        "spec_id": _AID_EVERY_YEAR_SPEC,
        "definition": {"identity": _AID_EVERY_YEAR},
        "resolved": {"download": {"rows": 870, "window": {"start_year": 1995, "end_year": 2030}}},
    }
    part = {"status": "definite", "spec": block}
    return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _unreachable() -> ToolCallResult:
    raise ToolSurfaceUnavailableError("data-mcp did not answer", retryable=True)


def _unreachable_for_good() -> ToolCallResult:
    raise ToolSurfaceUnavailableError("data-mcp cannot be reached from here", retryable=False)


def _last_five_years_plan() -> dict[str, Any]:
    """What a model proposes for "the last five years" and "the latest year" of the shown dataset.

    The growth and the filter both read from a start that follows the newest data.
    """
    latest = {"anchor": "latest", "offset": 0}
    return {
        "question": _LAST_FIVE_YEARS,
        "answer_mode": "series",
        "evidence_requirements": [
            {
                "kind": "dataset",
                "id": "aid",
                "measure": "education aid",
                "entities": [{"name": "Kenya"}],
                "period": {"start": 2020, "end": 2024},
                "period_end_open": True,
                "source_spec_id": _AID_SPEC,
            },
            {
                "kind": "growth",
                "id": "growth",
                "input_id": "aid",
                "window_start": 2020,
                "window_end": 2024,
                "rolling": {
                    "window_start": {"anchor": "latest", "offset": -4},
                    "window_end": latest,
                },
            },
            {
                "kind": "filter",
                "id": "newest",
                "input_id": "aid",
                "predicates": [
                    {"kind": "period_threshold", "op": "eq", "year": 2024, "rolling": latest}
                ],
            },
        ],
    }


_AID_BLOCK = {"spec_id": _AID_SPEC, "definition": {"identity": _AID_2020_2024}, "resolved": {}}


async def _run_the_planner(
    reads: _DefinitionReads, script: list[Step]
) -> tuple[ScriptedGenai, SessionState]:
    """One first-turn entry showing the planner the dataset Search answered over
    2020-2024, as a search reading whose block the turn's state holds."""
    scripted = ScriptedGenai(script)
    adapter = scripted_planner(scripted, surface=reads)
    reading = SearchReading.model_validate(
        {
            "text": "education aid over the last five years",
            "status": "definite",
            "spec_id": _AID_SPEC,
            "sentence": "Education aid to Kenya, 2020 to 2024",
            "window": {"start": 2020, "end": 2024, "source": "requested"},
        }
    )
    harness = _AdapterHarness({ORCHESTRATOR_SPECS_KEY: {_AID_SPEC: _AID_BLOCK}})
    await harness.run(
        adapter, question=_LAST_FIVE_YEARS, readings=[reading.model_dump(mode="json")]
    )
    return scripted, await harness.state()


async def _propose_the_last_five_years(
    reads: _DefinitionReads,
) -> tuple[ScriptedGenai, SessionState]:
    return await _run_the_planner(
        reads, [_calls(("propose_plan", _last_five_years_plan())), _says("done")]
    )


def _the_last_three_years(request: Any) -> list[tuple[str, dict[str, Any]]]:
    """A second proposal of the turn: the growth amended to the last three years."""
    growth = copy.deepcopy(_last_five_years_plan()["evidence_requirements"][1])
    growth["window_start"] = 2021
    growth["rolling"]["window_start"]["offset"] = -3
    amended = {
        "evidence_requirements": [growth],
        "amends_version": _last_amended_version(request, default=1),
    }
    return [("propose_plan", amended)]


async def test_a_last_five_years_plan_reads_the_first_year_once_and_pins_from_it() -> None:
    """Search answered over 2020-2024, which states no earlier year. The model's
    `propose_plan` call is recorded all the same: the dataset is resolved once
    over no window for its first year, 1995, and the pin moves to the dataset
    from 1995 with no end year. The growth and the latest-year filter both
    roll from their start on that one source, and one read serves both, as it
    does the second proposal of the turn."""
    reads = _DefinitionReads(_held_since_1995)

    scripted, state = await _run_the_planner(
        reads,
        [
            _calls(("propose_plan", _last_five_years_plan())),
            _computed(_the_last_three_years),
            _says("done"),
        ],
    )

    assert reads.first_year_reads == [
        {"spec_id": _AID_EVERY_YEAR_SPEC, "identity": _AID_EVERY_YEAR}
    ]
    assert full_histories(state) == {_AID_SPEC: FullHistory(first_year=1995, rows=870)}
    plan = load_plan(state)
    assert plan is not None, _contents(scripted.requests[-1])[-1]
    assert plan.version == 2
    opened, _ = windowed_identity(_AID_2020_2024, {"kind": "range", "start_year": 1995})
    dataset, growth, newest = plan.evidence_requirements
    assert isinstance(dataset, DatasetRequirement)
    assert (dataset.source_spec_id, dataset.period_end_open) == (opened, True)
    assert dataset.period == Period(start=2020, end=2024)
    assert isinstance(growth, GrowthRequirement)
    assert growth.rolling is not None
    assert growth.rolling.window_start == RelativePeriod(offset=-3)
    assert rolls_from_its_start(newest)


async def test_a_plan_of_fixed_years_over_a_windowed_reading_reads_nothing() -> None:
    """The reading is over 2020-2024 and the plan states those years as fixed,
    so no first year is needed and none is read."""
    reads = _DefinitionReads(_held_since_1995)
    fixed = _last_five_years_plan()
    dataset, growth, _ = fixed["evidence_requirements"]
    del dataset["period_end_open"], growth["rolling"]
    fixed["evidence_requirements"] = [dataset, growth]

    _, state = await _run_the_planner(reads, [_calls(("propose_plan", fixed)), _says("done")])

    assert reads.first_year_reads == []
    assert full_histories(state) == {}
    plan = load_plan(state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == _AID_SPEC  # type: ignore[union-attr]


def _no_first_year() -> ToolCallResult:
    """data-mcp's answer for the dataset over no window, stating no coverage."""
    block = {
        "spec_id": _AID_EVERY_YEAR_SPEC,
        "definition": {"identity": _AID_EVERY_YEAR},
        "resolved": {},
    }
    part = {"status": "definite", "spec": block}
    return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (_unreachable, "surface_unavailable"),
        (_unreachable_for_good, "invalid_plan"),
        (_no_first_year, "invalid_plan"),
    ],
    ids=["read-did-not-complete", "read-cannot-complete-on-a-retry", "answer-states-no-first-year"],
)
async def test_a_first_year_that_cannot_be_read_refuses_saying_the_read_failed(
    answer: Callable[[], ToolCallResult], code: str
) -> None:
    reads = _DefinitionReads(answer)

    scripted, state = await _propose_the_last_five_years(reads)

    assert len(reads.calls) == 1
    assert load_plan(state) is None
    (refusal,) = _answers(scripted.requests[-1], "propose_plan")
    assert refusal["code"] == code
    assert refusal["detail"] == {
        "issue": "rolling_needs_open_period",
        "fault": "first_year_read_failed",
        "requirement_ids": ["aid"],
        "spec_id": _AID_SPEC,
    }


_ACROSS_YEARS_RULE = "An ask marked (across years) asks for one figure, or one ranking"


@pytest.mark.parametrize("across_years", [[], [0, 1]], ids=["none", "marked"])
async def test_the_brief_marks_each_ask_with_the_parties_it_breaks_down_by(
    across_years: list[int],
) -> None:
    scripted = ScriptedGenai([_says("Planned.")])
    adapter = _scripted_planner(scripted)
    asks = ["the total", "Debt service by creditor"]
    harness = _AdapterHarness(
        state={
            ASKS_STATE_KEY: asks,
            ASK_BREAKDOWNS_STATE_KEY: [AskBreakdown(ask=1, by=("creditor",)).model_dump()],
            ASKS_ACROSS_YEARS_STATE_KEY: across_years,
        }
    )

    await harness.run(adapter, question="debt service by creditor", readings=[])

    prompt = _prompt_text(scripted.requests[0])
    if across_years:
        assert (
            "0. the total (across years)\n1. Debt service by creditor (by creditor) (across years)\n"
            in prompt
        )
        assert _ACROSS_YEARS_RULE in prompt
        assert "No other ask does: its answer keeps each year or reads one year." in prompt
    else:
        assert "0. the total\n1. Debt service by creditor (by creditor)\n" in prompt
        assert "(across years)" not in prompt
        assert _ACROSS_YEARS_RULE not in prompt
    assert (
        "'breakdown_not_kept' when the source this ask's answer pins cannot hold the party open"
        in prompt
    )


# --- revise: a plan amended in code, with no model call ------------------------


def _sector_switched(scope: dict[str, Any]) -> dict[str, Any]:
    """`scope` with its sector switched on the plan card to CRS 120 and 130."""
    switched = copy.deepcopy(scope)
    sector = switched["choices"][0]
    sector["active"] = "sector:120+130"
    switched["entries"] = [
        switched["entries"][1],
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": sector["id"]},
        },
    ]
    return switched


async def test_a_revision_repins_the_plan_and_its_scope_decisions_follow_with_no_model_call(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The US health aid plan, its sector switched: the re-pinned requirement
    runs through `propose_plan`'s chain and the translation, the old pin's
    scope decisions go, the new pin's are written and cited, the switched
    one the user's, and the planner's model is never asked."""
    old, new = "spec_" + "c" * 32, "spec_" + "d" * 32
    scope = _switchable_scope()
    blocks = {
        old: {"spec_id": old, "resolved": {"scope_choices": scope}},
        new: {"spec_id": new, "resolved": {"scope_choices": _sector_switched(scope)}},
    }
    shown = {spec_id: PinSource(block=block, settled=True) for spec_id, block in blocks.items()}
    scoped = scope_choice_decisions({old: shown[old]})[old]
    translated = _translated_plan(decision, requirement, derived_requirement)
    dataset = next(r for r in translated.evidence_requirements if isinstance(r, DatasetRequirement))
    pinned = dataset.model_copy(
        update={
            "source_spec_id": old,
            "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
        }
    )
    plan = Plan.model_validate(
        {
            **translated.model_dump(),
            "decisions": [*translated.decisions, *scoped],
            "evidence_requirements": [
                pinned if r.id == dataset.id else r for r in translated.evidence_requirements
            ],
        }
    )
    drafted: list[Plan] = []

    async def translate(call: Any) -> dict[str, object]:
        draft = load_plan(call.state)
        assert draft is not None
        drafted.append(draft)
        assert translated.translation is not None and translated.workflow is not None
        save_plan(
            call.state,
            draft.with_translation(
                translation=translated.translation, workflow=translated.workflow
            ),
        )
        return {"outcome": "ok"}

    scripted = ScriptedGenai([])
    adapter = scripted_planner(scripted, translate=translate)
    revised_pin = pinned.model_copy(update={"source_spec_id": new})

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        save_pin_sources(cast(SessionState, ctx.state), shown)
        return await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[revised_pin.model_dump(mode="json")],
            inherits={new: old},
        )

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(name="revise_wf", edges=[(START, node(_node, name="revise"))]),
        app_name=_APP,
        session_service=sessions,
    )
    outputs = [
        event.output
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
        )
        if event.output is not None
    ]

    (result,) = outputs
    assert scripted.requests == []
    revised = Plan.model_validate(result["plan"])
    assert (revised.version, revised.status) == (2, PlanStatus.PENDING_APPROVAL)
    assert [d.version for d in drafted] == [2]
    (dataset_after,) = [
        r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)
    ]
    assert dataset_after.source_spec_id == new
    by_slot = {d.slot: d for d in revised.decisions}
    assert not [slot for slot in by_slot if slot.endswith(old)]
    sector = by_slot[f"scope_sector_{new}"]
    assert (sector.chosen, sector.resolved_by, sector.rationale) == (
        "health (CRS 120) and population and reproductive health (CRS 130)",
        ResolvedBy.USER,
        "The user chose health (CRS 120) and population and reproductive health (CRS 130) "
        "on the plan.",
    )
    assert {f"scope_{axis}_{new}" for axis in ("sector", "region", "measure_basis")} <= set(
        dataset_after.decision_ids
    )
    assert result["summary"]["revision"] == plan_revision(revised)


def _region_switched(scope: dict[str, Any]) -> dict[str, Any]:
    """`scope` with its region switched on the plan card to Africa including
    its regional buckets."""
    switched = copy.deepcopy(scope)
    region = next(choice for choice in switched["choices"] if choice["axis"] == "region")
    region["active"] = "region:members_and_residuals"
    region["provenance"] = "explicit"
    switched["entries"] = [
        switched["entries"][0],
        {
            "axis": "region",
            "option_id": "region:members_and_residuals",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": region["id"]},
        },
    ]
    return switched


async def test_a_region_switch_and_its_users_decision_carry_through_an_amendment(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The US health aid plan, its region switched to Africa including its
    regional buckets, then amended with a new retrieval: the
    amended plan keeps the switched pin and the region decision the user
    made on it."""
    old, new, added = ("spec_" + c * 32 for c in "cde")
    scope = _switchable_scope()
    blocks = {
        old: {"spec_id": old, "resolved": {"scope_choices": scope}},
        new: {"spec_id": new, "resolved": {"scope_choices": _region_switched(scope)}},
    }
    shown = {spec_id: PinSource(block=block, settled=True) for spec_id, block in blocks.items()}
    scoped = scope_choice_decisions({old: shown[old]})[old]
    translated = _translated_plan(decision, requirement, derived_requirement)
    dataset = next(r for r in translated.evidence_requirements if isinstance(r, DatasetRequirement))
    pinned = dataset.model_copy(
        update={
            "source_spec_id": old,
            "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
        }
    )
    plan = Plan.model_validate(
        {
            **translated.model_dump(),
            "decisions": [*translated.decisions, *scoped],
            "evidence_requirements": [
                pinned if r.id == dataset.id else r for r in translated.evidence_requirements
            ],
        }
    )
    drafted: list[Plan] = []

    async def translate(call: Any) -> dict[str, object]:
        draft = load_plan(call.state)
        assert draft is not None
        drafted.append(draft)
        assert translated.translation is not None and translated.workflow is not None
        if {r.id for r in draft.evidence_requirements} == {
            r.id for r in translated.evidence_requirements
        }:
            save_plan(
                call.state,
                draft.with_translation(
                    translation=translated.translation, workflow=translated.workflow
                ),
            )
        return {"outcome": "ok"}

    def _amend(request: Any) -> list[tuple[str, dict[str, Any]]]:
        del request
        (switched,) = [
            r for r in drafted[-1].evidence_requirements if isinstance(r, DatasetRequirement)
        ]
        restated = [
            r.model_dump(mode="json")
            for r in drafted[-1].evidence_requirements
            if r is not switched
        ]
        more = dataset.model_copy(
            update={"id": "more", "source_spec_id": added, "decision_ids": dataset.decision_ids}
        )
        return [
            (
                "propose_plan",
                {
                    "evidence_requirements": [
                        switched.model_dump(mode="json"),
                        *restated,
                        more.model_dump(mode="json"),
                    ],
                    "amends_version": 2,
                },
            )
        ]

    scripted = ScriptedGenai([_computed(_amend), _says("done")])
    adapter = scripted_planner(scripted, translate=translate)
    revised_pin = pinned.model_copy(update={"source_spec_id": new})

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        state = cast(SessionState, ctx.state)
        save_pin_sources(state, shown)
        switched = await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[revised_pin.model_dump(mode="json")],
            inherits={new: old},
        )
        await adapter(
            ctx=ctx,
            question="kenya debt",
            readings=[],
            prior_plan=switched["plan"],
            amendment="add the total for the same years",
        )
        return {"done": True}

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(
            name="amend_wf", edges=[(START, node(_node, name="amend", rerun_on_resume=True))]
        ),
        app_name=_APP,
        session_service=sessions,
    )
    async for _ in runner.run_async(
        user_id=_USER,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
        run_config=SSE,
    ):
        pass

    switched_plan, amended = drafted[0], drafted[-1]
    assert (switched_plan.version, amended.version) == (2, 3)
    pins = {
        r.id: r.source_spec_id
        for r in amended.evidence_requirements
        if isinstance(r, DatasetRequirement)
    }
    assert pins == {dataset.id: new, "more": added}
    by_slot = {d.slot: d for d in amended.decisions}
    region = by_slot[f"scope_region_{new}"]
    assert (region.chosen, region.resolved_by, region.rationale) == (
        "Africa incl. regional buckets",
        ResolvedBy.USER,
        "The user chose Africa incl. regional buckets on the plan.",
    )
    assert region == next(d for d in switched_plan.decisions if d.slot == f"scope_region_{new}")


@pytest.mark.parametrize(
    ("kind", "outcome", "code", "final", "reason", "retryable"),
    [
        (
            ReferralKind.PINNED_SOURCE_NOT_FOUND,
            "choice_required",
            StopCode.PLAN_NOT_READY,
            False,
            "For health (CRS 120) and population and reproductive health (CRS 130), "
            f"{referral_reason(ReferralKind.PINNED_SOURCE_NOT_FOUND)}.",
            False,
        ),
        (
            ReferralKind.DISCOVERY_UNAVAILABLE,
            "transient",
            StopCode.DISCOVERY_UNAVAILABLE,
            True,
            "The data service did not answer while finding data for health (CRS 120) "
            "and population and reproductive health (CRS 130).",
            True,
        ),
        (
            ReferralKind.SURFACE_UNAVAILABLE,
            "transient",
            ProblemCode.SURFACE_UNAVAILABLE,
            False,
            "A service the plan needs could not be reached.",
            True,
        ),
    ],
    ids=["not_ready", "search_unanswered", "service_unanswered"],
)
async def test_a_revision_the_translation_refers_back_refuses_with_its_code_and_reason(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    kind: ReferralKind,
    outcome: str,
    code: str,
    final: bool,
    reason: str,
    retryable: bool,
) -> None:
    """A translation that refers the revised plan back ends the revision as a
    planner entry ending on that referral would. A pin the revision moved to
    that was not found again ends on `plan_not_ready` and the referral's
    reason, since no model is there to settle it, and a switch to the same
    option repeats it. A transient referral, with time left, ends on the
    service that did not answer, which a later switch can escape."""
    old, new = "spec_" + "c" * 32, "spec_" + "d" * 32
    scope = _switchable_scope()
    shown = {
        spec_id: PinSource(
            block={"spec_id": spec_id, "resolved": {"scope_choices": s}}, settled=True
        )
        for spec_id, s in ((old, scope), (new, _sector_switched(scope)))
    }
    translated = _translated_plan(decision, requirement, derived_requirement)
    dataset = next(r for r in translated.evidence_requirements if isinstance(r, DatasetRequirement))
    plan = Plan.model_validate(
        {
            **translated.model_dump(),
            "evidence_requirements": [
                dataset.model_copy(update={"source_spec_id": old}) if r.id == dataset.id else r
                for r in translated.evidence_requirements
            ],
        }
    )

    async def translate(call: Any) -> dict[str, object]:
        del call
        return {
            "kind": kind.value,
            "outcome": outcome,
            "requirement_id": dataset.id,
            "reason": f"{dataset.id!r} pins {new!r}, which discovery did not find.",
        }

    adapter = scripted_planner(ScriptedGenai([]), translate=translate)

    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": new, "scope": scope}
    )

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        save_pin_sources(cast(SessionState, ctx.state), shown)
        ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]
        return await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[
                dataset.model_copy(update={"source_spec_id": new}).model_dump(mode="json")
            ],
            inherits={new: old},
        )

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(name="referred_wf", edges=[(START, node(_node, name="revise"))]),
        app_name=_APP,
        session_service=sessions,
    )
    (result,) = [
        event.output
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
        )
        if event.output is not None
    ]

    refused = result["refused"]
    assert (refused["code"], refused["final"], refused["reason"]) == (code, final, reason)
    assert (refused["code"] in switch._RETRYABLE_REVISION_CODES) is retryable
    assert f"for requirement {dataset.id!r}" in refused["text"]


def test_a_declared_wholes_compile_leaves_the_readings_unchanged() -> None:
    """US health aid and the whole of it over every donor, compiled from the part:
    the compile is no reading of the conversation, so storing the turn's readings
    leaves them and their blocks as the search stored them. A later turn names the
    whole by its own requirement (`tests.test_aligned_turns`)."""
    origin = SearchReading.from_part(
        {
            "variable": "US health aid",
            "status": "definite",
            "spec_id": HEALTH,
            "scope": _switchable_scope(),
        }
    ).model_dump(mode="json")
    block = {"spec_id": TOTAL, "resolved": {"scope_choices": None}}
    whole = AlignedPin(
        origin=HEALTH,
        option_ids=(),
        spec_id=TOTAL,
        digest={
            "spec_block": block,
            "part_digests": [
                {"variable": TOTAL, "status": "definite", "spec_id": TOTAL, "spec_block": block}
            ],
        },
        whole_axis="donor",
    )
    ctx = SimpleNamespace(state={ORCHESTRATOR_SEARCHES_KEY: [origin]})
    planner_state: dict[str, Any] = {}
    save_aligned_pin(cast(SessionState, planner_state), whole, warnings=())

    planner_adapter.store_searched_pin_readings(cast(Any, ctx), cast(SessionState, planner_state))

    assert ctx.state == {ORCHESTRATOR_SEARCHES_KEY: [origin]}


_WINDOWED_HEALTH = "spec_" + "e" * 32


@pytest.mark.parametrize(
    "held", ["earlier_turn", "this_turn", "windowed_copy", "shown_and_compiled"]
)
def test_a_revision_names_a_pin_by_the_block_it_is_held_in(
    decision: Any, requirement: Any, derived_requirement: Any, held: str
) -> None:
    """A revision that did not compile names a retrieval by the concept of the
    block its pin reads: one an earlier turn stored, held through
    `earlier_block` alone; one this turn's search stored; for a pin
    copied over its window, its origin's block; or, for a user's own pin of a
    dataset the planner was shown that alignment also compiled another input
    onto, an average say, the block it was shown with, not the compile's."""
    block = {"spec_id": HEALTH, "resolved": {"concept_label": "health"}}
    state: dict[str, Any] = {}
    pinned = HEALTH
    if held == "earlier_turn":
        save_pin_sources(
            cast(SessionState, state), {HEALTH: PinSource(settled=True, earlier_turn=True)}
        )
    elif held == "this_turn":
        state[ORCHESTRATOR_SPECS_KEY] = {HEALTH: block}
    elif held == "windowed_copy":
        save_pin_sources(cast(SessionState, state), {HEALTH: PinSource(block=block, settled=True)})
        state[WINDOWED_PINS_STATE_KEY] = [
            {"spec_id": _WINDOWED_HEALTH, "identity": {}, "windowed_from": HEALTH}
        ]
        pinned = _WINDOWED_HEALTH
    else:
        save_pin_sources(cast(SessionState, state), {HEALTH: PinSource(block=block, settled=True)})
        compiled = {"spec_id": HEALTH, "resolved": {"concept_label": "average health aid"}}
        save_aligned_pin(
            cast(SessionState, state),
            AlignedPin(
                origin=TOTAL,
                option_ids=("measure_basis:gross",),
                spec_id=HEALTH,
                digest={"spec_block": compiled},
            ),
            warnings=(),
        )
    plan = Plan(
        question="health aid",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="health", source_spec_id=pinned),
            derived_requirement(input_id="health"),
        ],
    )
    token = current_earlier_block.set({HEALTH: block}.get)
    try:
        sources = planner_adapter._sources_by_spec_id(cast(SessionState, state), plan)
    finally:
        current_earlier_block.reset(token)

    assert plan_display.plan_words(plan, pinned=sources).names["health"] == "health"


async def test_a_pin_compiled_onto_a_shared_basis_is_no_reading_and_its_row_switches_from_it(
    decision: Any,
) -> None:
    """A revision of US health aid as a share of the US total compiles the total on
    gross disbursements. The compile joins no reading: the turn's readings are the
    search's alone. The total's basis row is its own (`scope_measure_basis_of_total`),
    and the `switch` node resolves its target from the compile, under the words
    and asked roles of the total's own reading, so the switch moves the total
    alone; a later search carries no basis from it."""
    blocks = (
        basis_block(HEALTH, narrow=True, active="measure_basis:gross"),
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
    )
    shown = {block["spec_id"]: PinSource(block=block, settled=True) for block in blocks}
    origin = SearchReading.model_validate(
        {"text": "US total ODA", "status": "definite", "spec_id": TOTAL, "asked_roles": ["donor"]}
    )

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=requirement_id,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    health = dataset("health", HEALTH)
    plan = Plan(
        question="What share of US aid goes to health?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[
            health,
            dataset("total", TOTAL),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="total"),
            ),
        ],
    )

    async def translate(call: Any) -> dict[str, object]:
        del call
        return {"outcome": "ok"}

    surface = BasisCompiles()
    adapter = scripted_planner(ScriptedGenai([]), surface=cast(Any, surface), translate=translate)

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        save_pin_sources(cast(SessionState, ctx.state), shown)
        ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [origin.model_dump(mode="json")]
        await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[health.model_dump(mode="json")],
            inherits={},
        )
        revised = load_plan(cast(SessionState, ctx.state))
        assert revised is not None
        state = cast(SessionState, ctx.state)
        (row,) = [d for d in revised.decisions if d.slot == "scope_measure_basis_of_total"]
        readings = switch._readings(ctx)
        target = switch._target(
            revised,
            row,
            state=state,
            on_record=held_sources(state),
            stated=switch._statement_of(revised, state),
            held=alignments(state),
            to_origin={},
            readings=readings,
        )
        return {
            "stored": [(r.spec_id, r.planner_axes) for r in readings],
            "target": [
                target.pin,
                target.moved_for,
                target.reading.text,
                list(target.reading.asked_roles),
                target.scope.id,
                target.choice.active,
            ],
            "carried": _search_context(cast(SessionState, ctx.state)).carried_entries,
        }

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(name="align_wf", edges=[(START, node(_node, name="revise"))]),
        app_name=_APP,
        session_service=sessions,
    )
    (result,) = [
        event.output
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="align")]),
        )
        if event.output is not None
    ]

    assert len(surface.calls) == 1
    assert result["stored"] == [(TOTAL, ())]
    assert result["target"] == [
        TOTAL,
        "total",
        "US total ODA",
        ["donor"],
        f"{TOTAL}_base",
        "measure_basis:gross",
    ]
    # A later "US total ODA" search carries no basis: the planner chose it.
    assert result["carried"] == []


_HEALTH_SWITCHED = f"spec_{'5' * 32}"


_NET_SWITCH = basis_block(
    _HEALTH_SWITCHED, narrow=True, active="measure_basis:net", provenance="explicit"
)


@pytest.mark.parametrize(
    ("total_pin", "ratio", "switched", "total_after", "tools"),
    [
        (
            TOTAL,
            ShareOfRequirement,
            _NET_SWITCH,
            TOTAL_NET,
            ["resolve_definition", "resolve_definition"],
        ),
        (
            TOTAL,
            ShareOfRequirement,
            basis_block(_HEALTH_SWITCHED, narrow=True, active="measure_basis:gross"),
            TOTAL_GROSS,
            ["resolve_definition"],
        ),
        (
            None,
            ShareOfRequirement,
            _NET_SWITCH,
            TOTAL_NET,
            ["search", "resolve_definition", "resolve_definition"],
        ),
        (
            None,
            QuotientRequirement,
            _NET_SWITCH,
            TOTAL_NET,
            ["search", "resolve_definition", "resolve_definition"],
        ),
    ],
    ids=[
        "basis-moves-both-inputs",
        "sector-moves-health-alone",
        "unpinned-share-total-moves-too",
        "unpinned-quotient-total-moves-too",
    ],
)
async def test_a_switch_on_a_share_plan_keeps_its_inputs_on_one_basis(
    decision: Any,
    total_pin: str | None,
    ratio: type[ShareOfRequirement | QuotientRequirement],
    switched: dict[str, Any],
    total_after: str,
    tools: list[str],
) -> None:
    """US health aid in 2023 as a share of the US total, the total compiled on
    the health aid's gross basis. A switch of the health aid's basis to net
    compiles the total on net too; a switch of its sector alone leaves the
    total on the gross pin compiled for it. A total the planner pinned no
    dataset for, of the United States as a donor in a share or as a country in
    a quotient, such as its GDP, is searched for in the translator's words and
    pinned before the plan records, so the switch moves it too, and the search
    joins the turn's readings."""

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=requirement_id,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=pin,
        )

    health = dataset("health", HEALTH)
    role = "donor" if ratio is ShareOfRequirement else None
    plan = Plan(
        question="What share of US aid went to health in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[
            health,
            dataset("total", TOTAL).model_copy(
                update={"source_spec_id": total_pin}
                if total_pin is not None
                else {
                    "source_spec_id": None,
                    "entities": (Entity(name="United States", role=role),),
                }
            ),
            ratio(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="total"),
            ),
        ],
    )
    blocks = (
        basis_block(HEALTH, narrow=True, active="measure_basis:gross"),
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
    )
    shown = {
        block["spec_id"]: PinSource(block=block, settled=True)
        for block in blocks
        if total_pin is not None or block["spec_id"] != TOTAL
    }

    async def translate(call: Any) -> dict[str, object]:
        del call
        return {"outcome": "ok"}

    surface = BasisCompiles()
    adapter = scripted_planner(ScriptedGenai([]), surface=cast(Any, surface), translate=translate)

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        state = cast(SessionState, ctx.state)
        save_pin_sources(state, shown)
        await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[health.model_dump(mode="json")],
            inherits={},
        )
        aligned = load_plan(state)
        assert aligned is not None
        save_pin_sources(
            state,
            {**pin_sources(state), _HEALTH_SWITCHED: PinSource(block=switched, settled=True)},
        )
        await adapter.revise(
            ctx=ctx,
            prior_plan=aligned.model_dump(mode="json"),
            evidence_requirements=[
                health.model_copy(update={"source_spec_id": _HEALTH_SWITCHED}).model_dump(
                    mode="json"
                )
            ],
            inherits={_HEALTH_SWITCHED: HEALTH},
        )
        after = load_plan(state)
        assert after is not None and after.version == aligned.version + 1
        return {
            "read": [r["spec_id"] for r in ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or []],
            "pins": {
                r.id: r.source_spec_id
                for r in after.evidence_requirements
                if isinstance(r, DatasetRequirement)
            },
            "bases": sorted(
                (d.fixes.source_spec_id, d.chosen)
                for d in after.decisions
                if d.scope is not None and d.scope.axis == "measure_basis" and d.fixes is not None
            ),
        }

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(name="share_switch_wf", edges=[(START, node(_node, name="revise"))]),
        app_name=_APP,
        session_service=sessions,
    )
    (result,) = [
        event.output
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
        )
        if event.output is not None
    ]

    assert result["pins"] == {"health": _HEALTH_SWITCHED, "total": total_after}
    assert surface.tools == tools
    if total_pin is None:
        (part,) = surface.calls[0]["parts"]
        assert part["variable"] == "total"
        assert part["entities"] == [{"text": "United States", **({"role": role} if role else {})}]
        assert TOTAL in result["read"]
    (health_basis, total_basis) = result["bases"]
    assert health_basis[1] == total_basis[1]


class _PartAndWholeCompiles(WholeCompiles):
    """`WholeCompiles`, whose resubmit of the US health part itself (its sector
    bound) answers the part as dev Search recorded it on the provider posted."""

    @override
    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        (sector,) = [s for s in arguments["identity"]["slots"] if s["property"] == "crsPurpose"]
        if sector["binding"]["state"] == "unbound":
            return await super().call_tool(tool=tool, arguments=arguments)
        self.calls.append(arguments)
        provider = [e["option_id"] for e in arguments["scope_entries"] if e["axis"] == "provider"]
        part = US_2023["health_inclusive" if provider == ["provider:inclusive"] else "health"]
        return ToolCallResult(
            is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [self.answered(part)]}}
        )

    def answered(self, part: dict[str, Any]) -> dict[str, Any]:
        """The part Search answers for the resubmit of `part`: `part` itself."""
        return part


class _PartMovedCompiles(_PartAndWholeCompiles):
    """`_PartAndWholeCompiles`, whose Search answers the part with a scope other than
    the one posted: its provider entry carries the concept "health"."""

    @override
    def answered(self, part: dict[str, Any]) -> dict[str, Any]:
        moved = copy.deepcopy(part)
        scope = moved["spec"]["resolved"]["scope_choices"]
        scope["entries"] = [
            {**entry, "concept": "health"} if entry["axis"] == "provider" else entry
            for entry in scope["entries"]
        ]
        return moved


@pytest.mark.parametrize(
    ("fake", "options", "calls", "concept"),
    [
        (
            _PartAndWholeCompiles,
            ("provider:inclusive", "provider:bilateral", "provider:inclusive"),
            [1, 3, 4, 4],
            None,
        ),
        (_PartMovedCompiles, ("provider:inclusive",), [1, 3], "health"),
    ],
    ids=["scope-kept", "scope-moved"],
)
async def test_a_provider_switch_compiles_the_parts_whole_once_a_turn(
    decision: Any,
    monkeypatch: pytest.MonkeyPatch,
    fake: type[_PartAndWholeCompiles],
    options: tuple[str, ...],
    calls: list[int],
    concept: str | None,
) -> None:
    """US health aid in 2023 as a share of its declared whole, switched on the plan
    card to bilateral plus imputed multilateral aid, back to bilateral, and to the
    inclusive scope again. Each switch posts the part, and the revision compiles
    the whole from Search's answer. A resubmit the turn already made is not posted
    again. The whole on bilateral flows is the one the first draft compiled, and
    the second inclusive switch repeats the first, so it posts nothing. Where
    Search answers the part with another scope, the revision compiles the whole
    from that scope."""
    # The user's words asked for bilateral aid, so the card shows the provider row.
    searched = copy.deepcopy(US_2023["health"])
    scope = searched["spec"]["resolved"]["scope_choices"]
    (provider,) = [c for c in scope["choices"] if c["axis"] == "provider"]
    provider["provenance"] = "explicit"
    scope["entries"] = [
        {
            "axis": "provider",
            "option_id": "provider:bilateral",
            "provenance": "explicit",
            "origin": {"kind": "text", "text": "bilateral", "start": 0, "end": 9},
        }
    ]
    health = searched["spec"]
    part_pin = health["spec_id"]

    def dataset(requirement_id: str, **fields: Any) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=requirement_id,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            **fields,
        )

    part = dataset("health", source_spec_id=part_pin)
    plan = Plan(
        question="What share of US aid went to health in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[
            part,
            dataset("total", whole_of=WholeOf(part="health", axis="sector")),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="total"),
            ),
        ],
    )

    async def translate(call: Any) -> dict[str, object]:
        del call
        return {"outcome": "ok"}

    # Each revision is offered as recorded: what it compiles is not this test's.
    monkeypatch.setattr(
        planner_adapter, "end_run_when_ready", lambda answer, ctx: {**answer, "summary": {}}
    )
    compiles = fake()
    adapter = scripted_planner(ScriptedGenai([]), surface=cast(Any, compiles), translate=translate)
    switching = switch.switch_node(planner=adapter, definition_surface=cast(Any, compiles))

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        state = cast(SessionState, ctx.state)
        save_pin_sources(state, {part_pin: PinSource(block=health, settled=True)})
        reading = SearchReading.from_part(
            {"variable": "US health aid", "status": "definite", "spec_id": part_pin, "scope": scope}
        )
        ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]
        drafted = await adapter.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[part.model_dump(mode="json")],
            inherits={},
        )
        assert "plan" in drafted, drafted
        ctx.state["plan"] = json.dumps(drafted["plan"])
        counted = [len(compiles.calls)]
        for option_id in options:
            shown = Plan.model_validate_json(ctx.state["plan"])
            (row,) = [
                d
                for d in shown.decisions
                if d.scope is not None
                and d.scope.axis == "provider"
                and d.fixes is not None
                and d.fixes.source_spec_id is not None
                and d.fixes.source_spec_id
                in {part_pin, US_2023["health_inclusive"]["spec"]["spec_id"]}
            ]
            pick = {"choice_id": plan_display.choice_id(shown, row), "option_id": option_id}
            async for event in switching(ctx, {"switch": pick}):
                assert "switch_failed" not in event.output, event.output["switch_failed"]
            counted.append(len(compiles.calls))
        shown = Plan.model_validate_json(ctx.state["plan"])
        return {
            "calls": counted,
            "whole": shown.requirement("total").source_spec_id,  # type: ignore[union-attr]
        }

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    runner = Runner(
        node=Workflow(name="whole_switch_wf", edges=[(START, node(_node, name="switch"))]),
        app_name=_APP,
        session_service=sessions,
    )
    (result,) = [
        event.output
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
        )
        if event.output is not None
    ]

    # The draft compiles the whole. Each switch posts the part, and a new whole once.
    assert result["calls"] == calls
    # The plan reads the inclusive whole, compiled from the part Search answered.
    assert result["whole"] == US_2023["whole_inclusive"]["spec"]["spec_id"]
    whole = [
        call
        for call in compiles.calls
        if any(
            slot["property"] == "crsPurpose" and slot["binding"]["state"] == "unbound"
            for slot in call["identity"]["slots"]
        )
    ][-1]
    (provider,) = [e for e in whole["scope_entries"] if e["axis"] == "provider"]
    assert provider.get("concept") == concept
