"""The orchestrator's direct `submit_analysis`: eligibility from recorded evidence,
the planner's own compile, and a run under a request authorisation.

The tool runs against a surface that declares the real servers' schemas, so a
recipe compiles as it does when served, and answers `get_data` and `growth`
from `harness.scripted`'s canned results, so the run is accepted and
published without an upstream.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import FrameRef
from compiler.surface import McpSurface, ToolCallResult, ToolSchema
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from executor.completion import FramePublished, PublishFault
from executor.ledger import (
    InMemoryRunLedger,
    LedgerUnavailableError,
    ParameterOrigin,
    RequestAuthorisation,
    RunStatus,
)
from executor.outcomes import RefusalKind, Refused
from executor.run import RunAbandonedError
from google.adk.events import Event, EventActions
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from harness import direct_recipes, planner_seam, studio_state
from harness.conversation import current_conversation_id
from harness.direct_recipes import (
    TOOL_NAME,
    DirectRecipes,
    acknowledge_deliveries,
    direct_recipe_tools,
    source_evidence,
)
from harness.graph.plan_done import HandBack
from harness.graph.resume import run_resumer
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    PendingPlan,
)
from harness.orchestrator_tools import build_orchestrator_tools
from harness.scripted import ScriptedExecutor, ScriptedSurface, scripted_publish_result
from harness.scripted_plan import scripted_aggregate_result, scripted_fetch_series_stats
from harness.studio_state import ConversationState
from harness.turn_records import PlanRecord
from orchestrator.agent import INSTRUCTION, create_orchestrator_agent
from planner.agent import build_planner_tools
from planner.plan import (
    PlanStatus,
    ResolvedBy,
    current_earlier_block,
    current_recorded_identity,
    load_plan,
    save_pin_sources,
)
from planner.records import GrowthChange
from planner.settings import SchemaDelivery
from planner.tool_results import Action, Outcome, ProblemCode, problem
from submission.direct import Compiled, compile_direct, graph_digest
from submission.recipes import GROWTH, GrowthParameters
from submission.service import Proposer
from translator.tool import build_translate

from .orchestrator_fakes import no_executor, no_resolver

SPEC = "spec_" + "2" * 32
OTHER = "spec_" + "3" * 32
SENTENCE = "life expectancy at birth in Kenya"


def _block(spec_id: str = SPEC, **resolved: Any) -> dict[str, Any]:
    """A definite part's spec block over 2015-2022, one StatVar for Kenya."""
    return {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": SENTENCE,
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 2015, "end_year": 2022},
            },
            "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
            "entities": [{"role": "observationAbout", "label": "Kenya", "ref": "country/KEN"}],
            "facets": {},
            **resolved,
        },
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
    }


def _reading(spec_id: str = SPEC, **reading: Any) -> dict[str, Any]:
    return {
        "text": "life expectancy",
        "status": "definite",
        "spec_id": spec_id,
        "sentence": SENTENCE,
        **reading,
    }


def _state(**overrides: Any) -> dict[str, Any]:
    """A turn whose search settled on `SPEC`."""
    return {
        ORCHESTRATOR_SPECS_KEY: {SPEC: _block()},
        ORCHESTRATOR_SEARCHES_KEY: [_reading()],
        **overrides,
    }


_SCHEMAS: list[ToolSchema] = []


class _Surface(ScriptedSurface):
    """The real servers' schemas; `get_data` and `growth` answered from canned
    results, every call counted."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def list_tools(self) -> list[ToolSchema]:
        if not _SCHEMAS:
            _SCHEMAS.extend(
                await McpSurface(
                    sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
                ).list_tools()
            )
        return _SCHEMAS

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        self.calls.append(tool.name)
        if tool.name == "growth":
            result = {**scripted_aggregate_result(), "role_keys": ["observationAbout"]}
            return ToolCallResult(is_error=False, structured_content=result, text="grew 5%")
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


class _Session:
    def __init__(self) -> None:
        self.id = "conversation-1"
        self.events: list[Any] = []


class _ToolContext:
    """What the orchestrator's and the planner's tools read of an ADK `ToolContext`."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.user_id = "user-1"
        self.session = _Session()
        self.actions = EventActions()
        self.invocation_id = "turn-1"


def _tool(
    surface: _Surface,
    *,
    ledger: InMemoryRunLedger | None = None,
    publisher: Callable[..., Any] | None = scripted_publish_result,
) -> Callable[..., Any]:
    (tool,) = direct_recipe_tools(
        DirectRecipes(
            recipes=("growth",),
            run_ledger=ledger or InMemoryRunLedger(),
            publisher=publisher,
            tool_surface=surface,
            translation_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
    )
    return tool


_GROWTH = {
    "recipe": "growth",
    "spec_id": SPEC,
    "start_year": 2015,
    "end_year": 2022,
    "change": GrowthChange.RELATIVE,
}

# A block's measure counted as net disbursements, offering gross under `OTHER`.
_COUNTED = {
    "measure_label": "net disbursements",
    "alternative_measures": [
        {"label": "gross disbursements", "axis": "what", "definition": {"spec_id": OTHER}}
    ],
}


async def test_an_eligible_growth_runs_under_a_request_authorisation_and_registers_its_frame():
    surface, ledger = _Surface(), InMemoryRunLedger()
    ctx = _ToolContext(_state())

    answer = await _tool(surface, ledger=ledger)(tool_context=ctx, **_GROWTH)

    assert answer["outcome"] == "completed"
    assert answer["run"]["status"] == "completed"
    (frame,) = answer["frames"]
    registered = ctx.state[ORCHESTRATOR_FRAMES_KEY][frame["frame"]]
    assert registered["ref"]["spec_id"].startswith("ops_")
    assert registered["store_ref"] == registered["ref"]
    assert surface.calls == ["get_data", "growth"]
    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == 1
    run = await ledger.get(answer["run"]["run_id"])
    assert run is not None and run.status is RunStatus.COMPLETED
    authorisation = run.authorisation
    assert isinstance(authorisation, RequestAuthorisation)
    assert (authorisation.recipe_id, authorisation.recipe_revision) == ("growth", GROWTH.revision)
    assert authorisation.source_policy.sources == (SPEC,)
    assert authorisation.parameter_origins == {
        "spec_id": ParameterOrigin.RECORDED,
        "start_year": ParameterOrigin.RECORDED,
        "end_year": ParameterOrigin.RECORDED,
        "change": ParameterOrigin.MODEL_STATED,
    }


async def test_the_same_request_in_a_later_turn_carries_on_the_same_run():
    """The run's key names the studio's conversation and the recipe with its
    parameters, so each turn's own session id leaves it unchanged."""
    ledger = InMemoryRunLedger()
    answers = []
    current_conversation_id.set("user-1/conversation-1")
    for turn in (1, 2):
        ctx = _ToolContext(_state())
        ctx.session.id = f"conversation-1#t{turn}"
        answers.append(await _tool(_Surface(), ledger=ledger)(tool_context=ctx, **_GROWTH))

    assert [a["outcome"] for a in answers] == ["completed", "completed"]
    assert answers[0]["run"]["run_id"] == answers[1]["run"]["run_id"]


@pytest.mark.parametrize(
    ("reading_this_turn", "block_this_turn"),
    [(False, False), (True, False), (False, True)],
    ids=["both-earlier", "reading-this-turn", "block-this-turn"],
)
async def test_an_earlier_turn_fills_whichever_of_reading_and_block_this_turn_lacks(
    reading_this_turn: bool, block_this_turn: bool
):
    """Each of the reading and the block is this turn's when the turn holds
    it, and an earlier turn's otherwise."""
    state: dict[str, Any] = {ORCHESTRATOR_PRIOR_LEDGER_KEY: {"spec_ids": [SPEC]}}
    if reading_this_turn:
        state[ORCHESTRATOR_SEARCHES_KEY] = [_reading()]
    else:
        state[ORCHESTRATOR_PRIOR_LEDGER_KEY]["readings"] = [_reading()]
    if block_this_turn:
        state[ORCHESTRATOR_SPECS_KEY] = {SPEC: _block()}
    token = current_earlier_block.set(
        lambda spec_id: _block() if spec_id == SPEC and not block_this_turn else None
    )
    try:
        found = source_evidence(state, SPEC)
        answer = await _tool(_Surface())(tool_context=_ToolContext(state), **_GROWTH)
    finally:
        current_earlier_block.reset(token)

    assert found is not None
    assert found[1].earlier_turn is not block_this_turn
    assert answer["outcome"] == "completed", answer


def _unsettled(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_SEARCHES_KEY] = [_reading(status="candidates")]


def _pending_plan(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_PENDING_KEY] = PendingPlan(brief="b", asks=[]).model_dump()


def _plan_spent(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_CALLS_KEY] = {"plan": 1}


def _open_donor_side(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_SPECS_KEY] = {
        SPEC: _block(role_axis_facts=[{"role": "donor", "member_count": 30}])
    }


def _blocking_warning(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_SEARCHES_KEY] = [_reading(warnings=[{"code": "time_text_not_parsed"}])]


def _another_count_offered(state: dict[str, Any]) -> None:
    state[ORCHESTRATOR_SPECS_KEY] = {SPEC: _block(**_COUNTED)}


def _unchanged(state: dict[str, Any]) -> None:
    del state


@pytest.mark.parametrize(
    ("perturb", "arguments", "outcome", "code"),
    [
        (_unchanged, {"spec_id": OTHER}, "needs_fields", "unverified_source"),
        (_unsettled, {}, "needs_fields", "unverified_source"),
        (_unchanged, {"start_year": None}, "needs_fields", "missing"),
        (_unchanged, {"end_year": 2015}, "needs_fields", "window_order"),
        (_unchanged, {"change": None}, "needs_decision", "method_unresolved"),
        (_unchanged, {"recipe": "quotient"}, "requires_planning", "recipe_not_enabled"),
        (_pending_plan, {}, "requires_planning", "plan_pending"),
        (_plan_spent, {}, "requires_planning", "plan_budget_spent"),
        (_unchanged, {"start_year": 2010}, "requires_planning", "outside_coverage"),
        (_open_donor_side, {}, "requires_planning", "not_one_series"),
        (_blocking_warning, {}, "requires_planning", "search_warning"),
        (_another_count_offered, {}, "needs_decision", "measure_unresolved"),
    ],
)
async def test_an_ineligible_request_is_refused_before_anything_runs(
    perturb: Callable[[dict[str, Any]], None], arguments: dict[str, Any], outcome: str, code: str
):
    surface = _Surface()
    state = _state()
    perturb(state)
    calls_before = dict(state.get(ORCHESTRATOR_CALLS_KEY) or {})

    answer = await _tool(surface)(tool_context=_ToolContext(state), **{**_GROWTH, **arguments})

    assert answer["outcome"] == outcome
    codes = [error["code"] for error in answer.get("errors", ())] or [answer.get("code")]
    assert code in codes
    assert surface.calls == []
    assert state.get(ORCHESTRATOR_CALLS_KEY, {}) == calls_before
    if outcome == "needs_fields":
        assert answer["contract"]["recipe"] == "growth"


async def _unreachable(frame: FrameRef) -> FramePublished | PublishFault:
    del frame
    return PublishFault(
        code="transient_unavailable", reason="down", retryable=True, retry_after_s=0.01
    )


async def _refusing(frame: FrameRef) -> FramePublished | PublishFault:
    del frame
    return PublishFault(code="recipe_too_large", reason="too large", retryable=False)


@pytest.mark.parametrize(
    ("publisher", "outcome", "status"),
    [
        (_unreachable, "pending", "pending"),
        (None, "pending", "publishing"),
        (_refusing, "failed", "failed"),
    ],
    ids=["retryable-fault", "no-publisher", "final-fault"],
)
async def test_a_publication_left_unfinished_answers_failed_only_for_a_failed_run(
    publisher: Any, outcome: str, status: str
):
    answer = await _tool(_Surface(), publisher=publisher)(
        tool_context=_ToolContext(_state()), **_GROWTH
    )

    assert (answer["outcome"], answer["run"]["status"]) == (outcome, status)
    assert "frames" not in answer


@pytest.mark.parametrize(
    "warnings", [[], [{"code": "retrieval_score_weak"}]], ids=["no-warning", "warning"]
)
async def test_the_direct_growth_compiles_to_the_planners_workflow_for_the_same_requirements(
    warnings: list[dict[str, str]],
):
    """The planner's `propose_plan`, with its translation, and the direct route
    compile one set of requirements to one workflow, each over the pin its
    own route builds from the same search readings and blocks."""
    surface = _Surface()
    state = _state(**{ORCHESTRATOR_SEARCHES_KEY: [_reading(warnings=warnings)]})
    found = source_evidence(state, SPEC)
    assert found is not None
    evidence, direct_pin = found
    parameters = GrowthParameters(
        spec_id=SPEC, start_year=2015, end_year=2022, change=GrowthChange.RELATIVE
    )
    requirements = GROWTH.build(parameters, evidence)
    assert isinstance(requirements, tuple)
    question = "How much did life expectancy in Kenya grow from 2015 to 2022?"

    direct = await compile_direct(
        question=question,
        requirements=requirements,
        pins={SPEC: direct_pin},
        surface=surface,
        proposer=Proposer(call=TOOL_NAME),
    )

    planner = _ToolContext({})
    save_pin_sources(
        planner.state,
        planner_seam._reading_pin_sources(
            state[ORCHESTRATOR_SEARCHES_KEY],
            [SPEC],
            blocks=state[ORCHESTRATOR_SPECS_KEY],
            earlier=(),
            warnings={SPEC: warnings},
        ),
    )
    (propose,) = [
        tool
        for tool in build_planner_tools(
            translate=build_translate(surface=surface), schema_delivery=SchemaDelivery.EAGER
        )
        if tool.name == "propose_plan"
    ]
    answer = await propose.run_async(
        args={
            "question": question,
            "answer_mode": "computed",
            "decisions": [],
            "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
        },
        tool_context=planner,  # type: ignore[arg-type]
    )
    planned = load_plan(planner.state)

    assert isinstance(direct, Compiled), direct
    assert planned is not None and planned.workflow is not None, answer
    assert direct.digest == graph_digest(planned.workflow)


async def test_two_calls_in_one_response_run_one_calculation():
    """ADK runs one response's calls concurrently. The first takes the
    turn's plan call before its first await, so the second finds it spent."""
    surface, ctx = _Surface(), _ToolContext(_state())
    tool = _tool(surface)

    answers = await asyncio.gather(
        tool(tool_context=ctx, **_GROWTH),
        tool(tool_context=ctx, **{**_GROWTH, "change": GrowthChange.ABSOLUTE}),
    )

    assert [a["outcome"] for a in answers] == ["completed", "requires_planning"]
    assert answers[1]["code"] == "plan_budget_spent"
    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == 1
    assert surface.calls == ["get_data", "growth"]


async def test_a_plan_beside_a_direct_calculation_finds_the_plan_call_taken():
    ctx = _ToolContext(_state())
    tools = {
        tool.__name__: tool
        for tool in build_orchestrator_tools(resolver=no_resolver, executor=no_executor)
    }
    direct = asyncio.create_task(_tool(_Surface())(tool_context=ctx, **_GROWTH))
    await asyncio.sleep(0)  # the direct call runs to its first await

    planned = tools["plan"]("Kenya's life expectancy growth", [], ctx)

    assert "error" in planned
    assert ctx.state.get(ORCHESTRATOR_PENDING_KEY) is None
    assert (await direct)["outcome"] == "completed"
    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == 1


class _Refusing(_Surface):
    """`tool` refused with `code` and `retry_mode`."""

    def __init__(self, tool: str, code: str, retry_mode: str) -> None:
        super().__init__()
        self.refused = (tool, code, retry_mode)

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        name, code, retry_mode = self.refused
        if tool.name != name:
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.calls.append(tool.name)
        refusal = {"code": code, "message": "refused", "retry_mode": retry_mode}
        return ToolCallResult(is_error=True, structured_content=refusal, text="refused")


@pytest.mark.parametrize(
    ("tool", "code", "retry_mode", "outcome", "plan_calls"),
    [
        ("growth", "growth_gap_year", "correct_and_retry", "needs_decision", 0),
        ("get_data", "definition_not_held", "refresh_then_retry", "requires_planning", 0),
        ("growth", "growth_failed", "do_not_retry", "failed", 1),
    ],
    ids=["method-choice", "rediscovery", "failed"],
)
async def test_a_stopped_run_keeps_the_plan_call_only_when_it_failed(
    tool: str, code: str, retry_mode: str, outcome: str, plan_calls: int
):
    """A run stopped for a method choice or a source to find again gives the
    call back for the plan its remedy needs; a failed run keeps it."""
    ctx = _ToolContext(_state())

    answer = await _tool(_Refusing(tool, code, retry_mode))(tool_context=ctx, **_GROWTH)

    assert answer["outcome"] == outcome, answer
    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == plan_calls


class _BrokenError(Exception):
    pass


async def _raise(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise _BrokenError


async def _expired(submission: Any, **kwargs: Any) -> Refused:
    del kwargs
    return Refused(
        revision=submission.revision, refusal=RefusalKind.AUTHORISATION_EXPIRED, detail="expired"
    )


async def _ledger_down(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise LedgerUnavailableError("run_admitted")


async def _not_admitted(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise LedgerUnavailableError(None)


async def _abandoned(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RunAbandonedError("run_admitted") from _BrokenError("secret detail")


@pytest.mark.parametrize(
    ("target", "replacement", "outcome", "code", "plan_calls"),
    [
        ("compile_direct", _raise, None, None, 0),
        ("execute_authorised", _raise, None, None, 0),
        ("execute_authorised", _expired, "failed", "authorisation_expired", 0),
        ("describe", _raise, None, None, 1),
        ("execute_authorised", _ledger_down, "pending", "ledger_unavailable", 1),
        ("link_accepted_run", _ledger_down, "pending", "ledger_unavailable", 1),
        ("execute_authorised", _abandoned, "failed", "run_abandoned", 1),
        ("execute_authorised", _not_admitted, "failed", "run_not_started", 0),
    ],
    ids=[
        "compile-raises",
        "run-raises",
        "never-admitted",
        "admitted-then-raises",
        "ledger-down-mid-run",
        "ledger-down-at-hand-back",
        "admitted-then-defect",
        "ledger-down-at-admission",
    ],
)
async def test_the_plan_call_is_given_back_unless_a_run_was_admitted(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    replacement: Callable[..., Any],
    outcome: str | None,
    code: str | None,
    plan_calls: int,
):
    ledger = InMemoryRunLedger()
    if target == "describe":
        monkeypatch.setattr(direct_recipes.HandBack, "describe", replacement)
    else:
        monkeypatch.setattr(direct_recipes, target, replacement)
    ctx = _ToolContext(_state())

    if outcome is None:
        with pytest.raises(_BrokenError):
            await _tool(_Surface(), ledger=ledger)(tool_context=ctx, **_GROWTH)
    else:
        answer = await _tool(_Surface(), ledger=ledger)(tool_context=ctx, **_GROWTH)
        assert (answer["outcome"], answer.get("code") or answer["run"]["code"]) == (outcome, code)
        assert "secret detail" not in json.dumps(answer)

    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == plan_calls


_SECRET = "ValueError: secret detail at gs://bucket/spec_0123"


@pytest.mark.parametrize(
    "refusal", [RefusalKind.RUN_ABANDONED, RefusalKind.TRANSPORT_FAILED, RefusalKind.EMPTY_ANSWER]
)
async def test_a_stopped_run_answers_its_code_and_none_of_its_detail(
    monkeypatch: pytest.MonkeyPatch, refusal: RefusalKind
):
    async def stopped(submission: Any, **kwargs: Any) -> Refused:
        del kwargs
        return Refused(
            revision=submission.revision, run_id="run_x", refusal=refusal, detail=_SECRET
        )

    monkeypatch.setattr(direct_recipes, "execute_authorised", stopped)

    answer = await _tool(_Surface())(tool_context=_ToolContext(_state()), **_GROWTH)

    assert (answer["outcome"], answer["code"]) == ("failed", refusal.value)
    assert answer["message"]
    assert "secret detail" not in json.dumps(answer)


def _compile_failed() -> dict[str, object]:
    return problem(
        ProblemCode.COMPILE_FAILED,
        f"This translation did not compile: {_SECRET}",
        outcome=Outcome.INVALID_INPUT,
        action=Action(call="return_to_planner", arguments={"reason": _SECRET}),
        detail={"issues": [{"code": "dangling_input", "detail": _SECRET}], "warnings": []},
    )


def _unsupported() -> dict[str, object]:
    return problem(
        ProblemCode.OPERATION_UNSUPPORTED,
        f"Unsupported: {_SECRET}",
        outcome=Outcome.UNSUPPORTED,
        action=Action(
            call="propose_plan",
            arguments={"declined": [{"code": "growth"}]},
            needs=("declined.ask",),
        ),
        detail={"operation": "growth"},
    )


def _referred() -> dict[str, object]:
    return {"outcome": "referred", "referral": {"kind": "no_data", "reason": _SECRET}}


@pytest.mark.parametrize(
    ("verdict", "codes"),
    [
        (
            _compile_failed,
            {
                "outcome": "invalid_input",
                "code": "compile_failed",
                "issue_codes": ["dangling_input"],
            },
        ),
        (_unsupported, {"outcome": "unsupported", "code": "operation_unsupported"}),
        (_referred, {"outcome": "referred", "code": "no_data"}),
    ],
    ids=["compile-failed", "unsupported", "referred"],
)
async def test_a_request_the_planner_chain_refuses_answers_its_codes_only(
    monkeypatch: pytest.MonkeyPatch, verdict: Callable[[], dict[str, object]], codes: dict
):
    """The chain's messages, details and actions are written for a planner
    and can quote an exception; the answer sends the model to `plan` with the
    codes alone."""

    async def refused(**kwargs: Any) -> dict[str, object]:
        del kwargs
        return verdict()

    monkeypatch.setattr(direct_recipes, "compile_direct", refused)

    answer = await _tool(_Surface())(tool_context=_ToolContext(_state()), **_GROWTH)

    assert (answer["outcome"], answer["code"]) == ("requires_planning", "not_compiled")
    assert answer["translation"] == codes
    assert "secret detail" not in json.dumps(answer)


def _picked(spec_id: str) -> list[Event]:
    """An ask offering net and gross by spec_id, and the user's pick of `spec_id`."""
    offered = [{"spec_id": SPEC}, {"spec_id": OTHER}]
    return [
        Event(
            author="dfo_orchestrator",
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="ask-1",
                            name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                            args={"payload": {"candidates": offered}},
                        )
                    )
                ]
            ),
        ),
        Event(
            author="user",
            content=types.Content(
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="ask-1",
                            name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                            response={"result": {"spec_id": spec_id}},
                        )
                    )
                ]
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("picked", "outcome"),
    [(None, "needs_decision"), (OTHER, "needs_decision"), (SPEC, "completed")],
    ids=["not-asked", "picked-the-other", "picked-this"],
)
async def test_a_source_offering_another_count_runs_only_on_the_users_pick(
    picked: str | None, outcome: str
):
    ctx = _ToolContext(_state(**{ORCHESTRATOR_SPECS_KEY: {SPEC: _block(**_COUNTED)}}))
    if picked is not None:
        ctx.session.events = _picked(picked)

    answer = await _tool(_Surface())(tool_context=ctx, **_GROWTH)

    assert answer["outcome"] == outcome
    if outcome == "needs_decision":
        assert answer["choices"] == [
            {"label": "net disbursements", "spec_id": SPEC},
            {"label": "gross disbursements", "spec_id": OTHER},
        ]


# Search's blocks for growth questions from Japan to Kenya, by the variable
# searched: gross disbursements elected, net offered beside it.
_MEASURE_NAMED = json.loads(
    (Path(__file__).parent / "fixtures" / "search_measure_named.json").read_text()
)["blocks"]


def _recorded(variable: str) -> dict[str, Any]:
    """`_block` stating the measure, its alternative and the trace search recorded for `variable`."""
    return _block(**_MEASURE_NAMED[variable]["resolved"])


# A source counting the flow as net disbursements, offering gross under `SPEC`.
_OFFERING_GROSS = {
    "measure_label": "net disbursements",
    "alternative_measures": [
        {"label": "gross disbursements", "axis": "what", "definition": {"spec_id": SPEC}}
    ],
}


@pytest.mark.parametrize(
    ("variable", "another_count", "outcome"),
    [
        ("gross ODA disbursements", False, "completed"),
        ("ODA disbursements", False, "needs_decision"),
        ("gross ODA disbursements", True, "needs_decision"),
    ],
    ids=["named", "elected", "another-count"],
)
async def test_a_source_offering_another_count_runs_when_the_question_named_its_measure(
    variable: str, another_count: bool, outcome: str
):
    """A source settled as another count of one (`another_count`), offered
    by `OTHER`, takes its choice from `OTHER` whatever its own trace says."""
    specs = {SPEC: _recorded(variable)}
    readings = [_reading()]
    if another_count:
        specs[OTHER] = _block(OTHER, **_OFFERING_GROSS)
        readings = [_reading(OTHER), _reading(other_count_of=OTHER)]
    ctx = _ToolContext(
        _state(**{ORCHESTRATOR_SPECS_KEY: specs, ORCHESTRATOR_SEARCHES_KEY: readings})
    )

    answer = await _tool(_Surface())(tool_context=ctx, **_GROWTH)

    assert answer["outcome"] == outcome, answer
    if outcome == "needs_decision":
        assert answer["code"] == "measure_unresolved"
    else:
        (record,) = ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
        assert [
            (d["resolved_by"], d["rationale"])
            for d in record["decisions"]
            if d["chosen"] == "gross disbursements"
        ] == [(ResolvedBy.USER.value, "The user's question named gross disbursements.")]


async def test_the_hand_back_carries_the_computation_notes_and_each_frames_status():
    ledger = InMemoryRunLedger()
    ctx = _ToolContext(_state())

    answer = await _tool(_Surface(), ledger=ledger)(tool_context=ctx, **_GROWTH)

    assert answer["outcome"] == "completed"
    # Delivered once the answer is in the session, so no later turn lists it.
    assert len(await ledger.unfinished()) == 1
    await acknowledge_deliveries(ctx)  # type: ignore[arg-type] -- reads only `invocation_id`
    assert await ledger.unfinished() == ()
    assert isinstance(answer["totals"], list)
    ((ask,),) = [a["answered_by"] for a in answer["asks"]]
    assert ask["status"] == "data"
    assert ask["frame"] == answer["frames"][0]["frame"]


class _Undescribable(ScriptedExecutor):
    async def __call__(self, *, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "describe_frame":
            return {"outcome": "refusal", "refusal": {"retry_mode": "do_not_retry"}}
        return await super().__call__(tool=tool, args=args)


async def test_a_result_describe_frame_cannot_view_is_not_completed():
    (tool,) = direct_recipe_tools(
        DirectRecipes(
            recipes=("growth",),
            run_ledger=InMemoryRunLedger(),
            publisher=scripted_publish_result,
            tool_surface=_Surface(),
            translation_surface=_Surface(),
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=_Undescribable(),
    )

    answer = await tool(tool_context=_ToolContext(_state()), **_GROWTH)

    assert answer["outcome"] == "not_chartable"
    assert answer["frames"] == []
    assert answer["frames_not_chartable"] == [
        {"step": "growth", "reason": "this step's data view could not be prepared"}
    ]


@pytest.mark.parametrize("change", list(GrowthChange))
async def test_a_later_turn_grows_the_series_a_direct_run_used(change: GrowthChange):
    """The direct run records its plan as `plan_done` does, so the turn keeps
    its source's block and the next turn finds the series verified. The next
    turn pins the source from the earlier turn's record, and its run is keyed
    by what that pin compiles to, so the same parameters never name a run of
    other work."""
    conversation = ConversationState(conversation_id="c1")
    ledger = InMemoryRunLedger()
    current_conversation_id.set("user-1/conversation-1")
    first = _ToolContext(_state())
    answer = await _tool(_Surface(), ledger=ledger)(tool_context=first, **_GROWTH)
    assert answer["outcome"] == "completed"
    conversation.turn_plans = list(first.state[ORCHESTRATOR_PLAN_RECORDS_KEY])
    conversation.turn_readings = list(first.state[ORCHESTRATOR_SEARCHES_KEY])
    conversation.turn_specs = dict(first.state[ORCHESTRATOR_SPECS_KEY])
    conversation.close_turn(final_text="Life expectancy grew 5%.")

    later = _ToolContext({ORCHESTRATOR_PRIOR_LEDGER_KEY: conversation.prior_ledger_state()})
    token = current_earlier_block.set(conversation.earlier_block)
    try:
        answer = await _tool(_Surface(), ledger=ledger)(
            tool_context=later, **{**_GROWTH, "change": change}
        )
    finally:
        current_earlier_block.reset(token)

    assert answer["outcome"] == "completed", answer


def test_with_no_recipe_enabled_there_is_no_tool_and_no_instruction_for_it():
    off = DirectRecipes(
        recipes=(),
        run_ledger=InMemoryRunLedger(),
        publisher=None,
        tool_surface=_Surface(),
        translation_surface=_Surface(),
    )
    on = dataclasses.replace(off, recipes=("growth",))

    def agent(direct: DirectRecipes) -> Any:
        tools = build_orchestrator_tools(resolver=no_resolver, executor=no_executor, direct=direct)
        return create_orchestrator_agent(model_name="gemini-test", tools=tools)

    assert TOOL_NAME not in {tool.name for tool in agent(off).tools}
    assert agent(off).instruction == INSTRUCTION
    assert TOOL_NAME in {tool.name for tool in agent(on).tools}
    assert f"{TOOL_NAME} with recipe" in agent(on).instruction


async def test_a_direct_run_whose_frame_times_out_is_listed_next_turn():
    """A frame refused for a retry leaves the run undelivered, and the next
    turn lists it again."""

    class _TimesOut(ScriptedExecutor):
        async def __call__(self, *, tool: str, args: dict[str, Any]) -> dict[str, Any]:
            if tool == "describe_frame":
                refusal = {"code": "timeout", "retry_mode": "retry_after_delay"}
                return {"outcome": "refusal", "refusal": refusal}
            return await super().__call__(tool=tool, args=args)

    ledger = InMemoryRunLedger()
    ctx = _ToolContext(_state())
    (tool,) = direct_recipe_tools(
        DirectRecipes(
            recipes=("growth",),
            run_ledger=ledger,
            publisher=scripted_publish_result,
            tool_surface=_Surface(),
            translation_surface=_Surface(),
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=_TimesOut(),
    )

    answer = await tool(tool_context=ctx, **_GROWTH)
    await acknowledge_deliveries(ctx)  # type: ignore[arg-type] -- reads only `invocation_id`

    assert answer["run"]["code"] == "result_undelivered"
    resume = run_resumer(
        run_ledger=ledger,
        publisher=scripted_publish_result,
        executor=ScriptedExecutor(),
        tool_surface=_Surface(),
        max_concurrent_steps=4,
        fetch_series_stats=scripted_fetch_series_stats,
    )
    handed_back = await resume(ctx)  # type: ignore[arg-type] -- reads only `state`
    assert handed_back is not None
    (resumed,) = json.loads(handed_back.hand_back["text"].split("\n", 1)[1])["runs"]
    assert (resumed["run_id"], resumed["status"]) == (answer["run"]["run_id"], "completed")
    assert resumed["frames"]


async def test_a_run_carried_on_next_turn_is_listed_as_plan_done_lists_one():
    """The resumed run's frames go through the same `HandBack`: its step
    summaries, and a frame no view was prepared for under
    `frames_not_chartable`. The turn that left it pending recorded its plan
    executing; once the run completes, recall states that turn's plan
    completed, read from the run ledger."""

    async def unreachable(frame: FrameRef) -> FramePublished | PublishFault:
        del frame
        return PublishFault(
            code="transient_unavailable", reason="down", retryable=True, retry_after_s=0.01
        )

    ledger = InMemoryRunLedger()
    ctx = _ToolContext(_state())
    left = await _tool(_Surface(), ledger=ledger, publisher=unreachable)(
        tool_context=ctx, **_GROWTH
    )
    assert left["outcome"] == "pending"
    conversation = ConversationState(conversation_id="c1")
    _closed(conversation, ctx)
    (recorded,) = (PlanRecord.model_validate(raw) for raw in conversation.turns[0]["plans"])
    assert (recorded.status, recorded.run_id) == (PlanStatus.EXECUTING, left["run"]["run_id"])
    assert conversation.awaited_runs() == {recorded.run_id}
    resume = run_resumer(
        run_ledger=ledger,
        publisher=scripted_publish_result,
        executor=_Undescribable(),
        tool_surface=_Surface(),
        max_concurrent_steps=4,
        fetch_series_stats=scripted_fetch_series_stats,
    )

    handed_back = await resume(ctx)  # type: ignore[arg-type] -- reads only `state`

    assert handed_back is not None
    runs = json.loads(handed_back.hand_back["text"].split("\n", 1)[1])
    (resumed,) = runs["runs"]
    assert resumed["status"] == "completed"
    assert resumed["frames"] == []
    assert resumed["frames_not_chartable"] == [
        {"step": "growth", "reason": "this step's data view could not be prepared"}
    ]
    assert resumed["step_summaries"]
    run = await ledger.get(recorded.run_id)
    assert run is not None
    runs = {run.run_id: run.status}
    completed = recorded.model_copy(update={"status": PlanStatus.COMPLETED})
    recalled = conversation.recall_part(part="plan", turn=1, runs=runs)
    assert recalled["text"] == [completed.render(budget=studio_state._RECALL_TEXT_BUDGET)]


@pytest.mark.parametrize(
    ("searched_as", "arguments", "same_run"),
    [
        ("life expectancy at birth", {}, True),
        ("life expectancy", {"change": GrowthChange.ABSOLUTE}, False),
    ],
    ids=["reworded-search", "other-change"],
)
async def test_a_request_in_a_later_turn_reuses_the_run_only_for_the_same_work(
    searched_as: str, arguments: dict[str, Any], same_run: bool
):
    """The run is named by what it computes: the recorded source, the recipe
    and its parameters. The words a search was sent are not part of it."""
    ledger = InMemoryRunLedger()
    current_conversation_id.set("user-1/conversation-1")
    answers = []
    for text, request in (("life expectancy", _GROWTH), (searched_as, {**_GROWTH, **arguments})):
        ctx = _ToolContext(_state(**{ORCHESTRATOR_SEARCHES_KEY: [_reading(text=text)]}))
        answers.append(await _tool(_Surface(), ledger=ledger)(tool_context=ctx, **request))
        assert ctx.state[ORCHESTRATOR_CALLS_KEY]["plan"] == 1

    assert [a["outcome"] for a in answers] == ["completed", "completed"]
    assert (answers[0]["run"]["run_id"] == answers[1]["run"]["run_id"]) is same_run


def _closed(conversation: ConversationState, ctx: _ToolContext) -> None:
    """Close the turn `ctx` ran, as `harness.studio` does with what the turn wrote."""
    conversation.turn_plans = list(ctx.state.get(ORCHESTRATOR_PLAN_RECORDS_KEY) or ())
    conversation.turn_readings = list(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or ())
    conversation.turn_specs = dict(ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {})
    conversation.close_turn(final_text="answered")


async def _next_turn(
    conversation: ConversationState, surface: _Surface, **arguments: Any
) -> tuple[dict[str, Any], _ToolContext]:
    """`submit_analysis` in a turn after those `conversation` holds."""
    ctx = _ToolContext({ORCHESTRATOR_PRIOR_LEDGER_KEY: conversation.prior_ledger_state()})
    blocks = current_earlier_block.set(conversation.earlier_block)
    identities = current_recorded_identity.set(conversation.recorded_identity)
    try:
        answer = await _tool(surface)(tool_context=ctx, **{**_GROWTH, **arguments})
    finally:
        current_earlier_block.reset(blocks)
        current_recorded_identity.reset(identities)
    return answer, ctx


def _measure_decisions(ctx: _ToolContext) -> list[dict[str, Any]]:
    (record,) = ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    return [d for d in record["decisions"] if d["chosen"] == "net disbursements"]


async def test_the_users_measure_pick_is_the_users_in_the_plan_and_in_a_later_turn():
    conversation = ConversationState(conversation_id="c1")
    first = _ToolContext(_state(**{ORCHESTRATOR_SPECS_KEY: {SPEC: _block(**_COUNTED)}}))
    first.session.events = _picked(SPEC)

    answer = await _tool(_Surface())(tool_context=first, **_GROWTH)

    assert answer["outcome"] == "completed", answer
    assert [d["resolved_by"] for d in _measure_decisions(first)] == [ResolvedBy.USER.value]
    _closed(conversation, first)

    later, ctx = await _next_turn(conversation, _Surface(), change=GrowthChange.ABSOLUTE)

    assert later["outcome"] == "completed", later
    assert [d["resolved_by"] for d in _measure_decisions(ctx)] == [ResolvedBy.USER.value]


# A source whose identity hashes to its spec_id, found by a search over 2015-2022.
_FOUND = {
    "stat_var": "LifeExpectancy_Person",
    "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2022},
}
_FOUND_SPEC = spec_id_for(canonicalize(_FOUND))
# The same source over 2017-2020, the copy a plan of those years retrieves.
_COPY_SPEC, _COPY = windowed_identity(
    _FOUND, {"kind": "range", "start_year": 2017, "end_year": 2020}
)


def _found_block(spec_id: str, identity: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    block = _block(
        spec_id,
        coverage={
            "kind": "exact",
            "has_data": True,
            "window": {"start_year": start, "end_year": end},
        },
    )
    block["definition"]["identity"] = identity
    return block


class _Resolving(_Surface):
    """`_Surface`, with data-mcp's `resolve_definition` answering the copy over
    2017-2020, and `get_data` the identity each spec_id hashes from."""

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        if tool.name == "get_data":
            answer = await super().call_tool(
                tool=tool, arguments=arguments, on_progress=on_progress
            )
            manifest = dict(answer.structured_content or {})
            identity = {_FOUND_SPEC: _FOUND, _COPY_SPEC: _COPY}[arguments["spec_id"]]
            manifest["definition"] = {**manifest["definition"], "identity": identity}
            return answer.model_copy(update={"structured_content": manifest})
        if tool.name != "resolve_definition":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.calls.append(tool.name)
        assert arguments["spec_id"] == _COPY_SPEC
        part = {"status": "definite", "spec": _found_block(_COPY_SPEC, _COPY, 2017, 2020)}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


@pytest.mark.parametrize("spec_id", [_FOUND_SPEC, _COPY_SPEC], ids=["origin", "copy"])
async def test_a_later_turn_grows_either_source_a_windowed_direct_run_used(spec_id: str):
    """Growth over fewer years than the search found retrieves a copy of the
    source over those years. The turn's record keeps the source and the
    copy's relation to it, so the next turn finds both verified."""
    conversation = ConversationState(conversation_id="c1")
    first = _ToolContext(
        {
            ORCHESTRATOR_SPECS_KEY: {_FOUND_SPEC: _found_block(_FOUND_SPEC, _FOUND, 2015, 2022)},
            ORCHESTRATOR_SEARCHES_KEY: [_reading(_FOUND_SPEC)],
        }
    )
    conversation.turn_specs = dict(first.state[ORCHESTRATOR_SPECS_KEY])
    token = current_recorded_identity.set(conversation.recorded_identity)
    try:
        answer = await _tool(_Resolving())(
            tool_context=first,
            **{**_GROWTH, "spec_id": _FOUND_SPEC, "start_year": 2017, "end_year": 2020},
        )
    finally:
        current_recorded_identity.reset(token)
    assert answer["outcome"] == "completed", answer
    (record,) = first.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    assert [source["spec_id"] for source in record["sources"]] == [_COPY_SPEC]
    _closed(conversation, first)

    later, _ = await _next_turn(
        conversation, _Resolving(), spec_id=spec_id, start_year=2017, end_year=2020
    )

    assert later["outcome"] == "completed", later


class _DescribingFirst(ScriptedExecutor):
    """Describes the first frame it is asked for and refuses every other."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = 0

    async def __call__(self, *, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "describe_frame":
            self.asked += 1
            if self.asked > 1:
                return {"outcome": "refusal", "refusal": {"retry_mode": "do_not_retry"}}
            return {"outcome": "ok", "view": {"series_count": 1}}
        return await super().__call__(tool=tool, args=args)


@pytest.mark.parametrize(
    ("executor", "frames", "nothing_chartable"),
    [(_DescribingFirst, 1, False), (_DescribingFirst, 2, False), (_Undescribable, 1, True)],
    ids=["one-described", "one-described-one-refused", "only-refused"],
)
async def test_a_result_is_told_in_words_only_when_no_frame_was_described(
    executor: Callable[[], ScriptedExecutor], frames: int, nothing_chartable: bool
):
    """`plan_done` and the direct route both read this of `HandBack`."""
    hand_back = HandBack(executor())
    for number in range(frames):
        ref = {"spec_id": f"ops_{number}"}
        await hand_back.describe(ref, tool="growth", store_ref=ref)

    assert hand_back.nothing_chartable is nothing_chartable


async def test_a_request_is_read_and_named_by_its_recipe(monkeypatch: pytest.MonkeyPatch):
    """The tool reads the arguments as the recipe's own parameters and names
    the run's task with the recipe's own `task`."""
    seen: list[type] = []

    def task(parameters: Any, evidence: Any) -> str:
        seen.append(type(parameters))
        return f"the recipe's own task for {evidence.spec_id}"

    monkeypatch.setitem(direct_recipes.RECIPES, "growth", dataclasses.replace(GROWTH, task=task))
    ledger = InMemoryRunLedger()

    answer = await _tool(_Surface(), ledger=ledger)(tool_context=_ToolContext(_state()), **_GROWTH)

    assert answer["outcome"] == "completed", answer
    assert seen == [GrowthParameters]
    run = await ledger.get(answer["run"]["run_id"])
    assert run is not None and isinstance(run.authorisation, RequestAuthorisation)
    assert run.authorisation.task == f"the recipe's own task for {SPEC}"
