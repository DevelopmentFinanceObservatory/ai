"""Tests for `translator.mechanical` -- the discover-then-compile loop as code.

`build_translation` is exercised first, on its own: the claim under test is
that it is a pure, deterministic function of a plan's own requirements (plus
the sources discovery found), never a fresh choice. `run_translate` is then
driven end to end through `translator.agent.build_translator_core`, against
an `McpSurface` combining the shared canned `data_server` (data-mcp's
`search`/`describe`, so `discover_source` answers stay under this suite's
control) with the real `get_data` tool object (`data_mcp.server`, added onto
the same server so its capability schema matches production and
`propose_operations` compiles for real) and the real `analysis_mcp` app
(every verb's own real capability schema). Nothing here calls `get_data`:
`translator.agent`'s module docstring states the translator never wraps it
either, so only its schema, never its implementation, has to be real.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.sources import DiscoveryStatus, source_from_spec_block
from compiler.surface import McpSurface
from data_mcp.definition_store import DefinitionStoreUnavailable, InMemoryDefinitionStore
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.definition_writes import windowed_identity
from data_mcp.server import mcp as real_data_mcp
from fastmcp import FastMCP
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# The key set and the walker that hold the model-facing boundary, shared
# with the module that established it on the discovery and refusal paths.
from harness.wire import (
    MAX_SEED_RECORD_CHARS,
    MAX_SEED_SOURCE_CHARS,
    MAX_SEED_STAT_VARS,
    MAX_SEED_TEXT_CHARS,
    MAX_SEEDS,
)
from planner.agent import _window_seeded_pins
from planner.operations import (
    FilterOperation,
    JoinOperation,
    RankOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    MAX_MEASURE_CHARS,
    MAX_PLAN_RECORD_CHARS,
    MAX_PLAN_REQUIREMENTS,
    PLAN_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    DatasetRequirement,
    Decision,
    Entity,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    SeededSource,
    ShareOfRequirement,
    plan_revision,
    save_dropped_seeds,
    save_plan,
    save_seeded_sources,
)
from planner.records import DenominatorScope, EntityIn, GroupBy, RankAmong, RankDirection
from planner.tool_results import ProblemCode
from planner.workflow import SelectedSource
from pydantic import Field as PydanticField
from translator import mechanical
from translator.agent import build_translator_core, create_translator_agent
from translator.discovery import DiscoveryRecord, load_log, requirement_revision, save_log
from translator.mechanical import (
    _BUILDERS,
    TranslatorCore,
    _load_pending,
    _operation_for,
    _unsupported_requirement,
    build_translation,
    run_translate,
)
from translator.referral import Referral, ReferralKind, load_referral, save_referral

from ..bounded_seeds import fullest_described_seed, read_seed
from ..data_mcp_definitions import SearchStub, search_api
from .test_model_facing_tool_results import _ENVELOPE_KEYS, _flat_keys

RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]

# A seeded turn request whose one seed is a real Search API Spec block.
SEEDED_REQUEST_FIXTURE = (
    Path(__file__).parents[1] / "wire_fixtures" / "studio" / "request-11-turn-seeded.json"
)
DraftFactory = Callable[..., Plan]
DecisionFactory = Callable[..., Decision]
PinningDecisionFactory = Callable[..., Decision]
ContextHolding = Callable[[Plan], Any]
SourceFactory = Callable[..., SelectedSource]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32


@pytest.fixture
async def combined_surface(data_server: FastMCP) -> McpSurface:
    """The canned `data_server` (search, describe) plus the real `get_data` and
    `resolve_definition` tool objects.

    `discover_source` answers from `data_server`'s canned `search`, and reads
    a seed's copy over a plan's period through the real `resolve_definition`,
    over the Search API a test installs (`search_api`);
    `propose_operations` needs `get_data`'s and every analysis-mcp verb's
    real capability schema to compile, or every attempt reports
    `capability_schema_drift` against a stand-in schema no real deploy would
    ever serve.
    """
    data_server.local_provider.remove_tool("get_data")
    data_server.add_tool(await real_data_mcp.get_tool("get_data"))
    data_server.add_tool(await real_data_mcp.get_tool("resolve_definition"))
    return McpSurface(sources={"data-mcp": data_server, "analysis-mcp": analysis_mcp})


@pytest.fixture
def core(combined_surface: McpSurface) -> TranslatorCore:
    return build_translator_core(surface=combined_surface)


# --- build_translation: a pure function of the plan, never a fresh choice ---


def test_build_translation_matches_a_retrieve_and_aggregate_pair(draft: DraftFactory) -> None:
    plan = draft()

    translation = build_translation(plan, {"annual_debt_service": SPEC_KENYA})

    assert translation.model_dump(mode="json") == {
        "operations": [
            {
                "kind": "retrieve",
                "id": "annual_debt_service",
                "requirement_id": "annual_debt_service",
                "spec_id": SPEC_KENYA,
            },
            {
                "kind": "aggregate",
                "id": "annual_totals",
                "requirement_id": "annual_totals",
                "input": "annual_debt_service",
                "reduction": "sum",
                "group_by": ["period"],
            },
        ],
        "terminal": ["annual_totals"],
    }


@pytest.mark.parametrize(
    "requirement,expected",
    [
        pytest.param(
            RankRequirement(
                id="rank_by_debt",
                decision_ids=["measure"],
                input_id="annual_debt_service",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.NAMED,
                axis="debtor",
            ),
            RankOperation(
                id="rank_by_debt",
                requirement_id="rank_by_debt",
                input="annual_debt_service",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.NAMED,
                axis="debtor",
            ),
            id="rank",
        ),
        pytest.param(
            RankRequirement(
                id="top_three",
                decision_ids=["measure"],
                input_id="annual_debt_service",
                direction=RankDirection.LOWEST_FIRST,
                among=RankAmong.POPULATION,
                axis="debtor",
                per_period=True,
                limit=3,
            ),
            RankOperation(
                id="top_three",
                requirement_id="top_three",
                input="annual_debt_service",
                direction=RankDirection.LOWEST_FIRST,
                among=RankAmong.POPULATION,
                axis="debtor",
                per_period=True,
                limit=3,
            ),
            id="rank_with_limit",
        ),
        pytest.param(
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="a", entity="Kenya"),
                denominator=RequirementSide(input_id="b"),
                axis="debtor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
            ShareOfOperation(
                id="share",
                requirement_id="share",
                numerator=Side(input="a", entity="Kenya"),
                denominator=Side(input="b"),
                axis="debtor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
            id="share_of",
        ),
        pytest.param(
            JoinRequirement(id="joined", decision_ids=["measure"], left_id="a", right_id="b"),
            JoinOperation(id="joined", requirement_id="joined", left="a", right="b"),
            id="join",
        ),
        pytest.param(
            FilterRequirement(
                id="filtered",
                decision_ids=["measure"],
                input_id="a",
                predicates=(EntityIn(names=("Kenya",)),),
            ),
            FilterOperation(
                id="filtered",
                requirement_id="filtered",
                input="a",
                predicates=(EntityIn(names=("Kenya",)),),
            ),
            id="filter",
        ),
    ],
)
def test_operation_for_copies_every_derived_kind_verbatim(
    requirement: object, expected: object
) -> None:
    """Every derived requirement kind builds the one operation `planner.proof.mismatches`
    would accept for it, with no field a plan did not already state."""
    built = _operation_for(requirement, {})  # type: ignore[arg-type]

    assert built == expected


def test_build_translation_covers_a_two_level_derived_chain(
    requirement: RequirementFactory, decision: DecisionFactory
) -> None:
    """`growth_a` reads `totals_a`'s own output, not the dataset directly --
    the chain shape the prototype never exercised. Both operations still
    build from their own requirement's fields alone, whichever requirement
    they read."""
    dataset = requirement(id="dataset_a")
    totals = AggregateRequirement(
        id="totals_a",
        decision_ids=["measure"],
        input_id="dataset_a",
        reduction="sum",
        group_by=("period",),
    )
    growth = GrowthRequirement(
        id="growth_a",
        decision_ids=["measure"],
        input_id="totals_a",
        window_start=2016,
        window_end=2020,
    )
    plan = Plan(
        question="Kenya's debt service growth",
        decisions=[decision()],
        evidence_requirements=[dataset, totals, growth],
    )

    translation = build_translation(plan, {"dataset_a": SPEC_KENYA})

    ids = {op.id: op for op in translation.operations}
    assert ids["growth_a"].input == "totals_a"  # reads the aggregate's own operation id
    assert ids["totals_a"].input == "dataset_a"
    # Only growth_a is terminal: both dataset_a and totals_a feed something.
    assert translation.terminal == ("growth_a",)


def test_build_translation_names_an_unread_dataset_requirement_its_own_terminal(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
) -> None:
    """A dataset requirement no calculation reads is a terminal answer in
    its own right, and a plan may state any number of them. This plan
    pairs one calculation over one dataset requirement with a second,
    separate dataset requirement nothing reads, and both the aggregate
    and the lone retrieval come back as terminal."""
    population = requirement(id="population", measure="population")
    plan = Plan(
        question="How much does Kenya pay to service its debt, and what is its population?",
        answer_mode=AnswerMode.SERIES,
        decisions=[decision()],
        evidence_requirements=[requirement(), derived_requirement(), population],
    )

    translation = build_translation(
        plan, {"annual_debt_service": SPEC_KENYA, "population": SPEC_OTHER}
    )

    assert translation.terminal == ("annual_totals", "population")


# --- run_translate: the whole loop, no model turn --------------------------


async def test_run_translate_compiles_a_clean_plan_with_no_pending_judgement(
    draft: DraftFactory, core: TranslatorCore, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert result["status"] == PlanStatus.PENDING_APPROVAL.value


async def test_a_compiled_plan_reaches_the_model_carrying_no_upstream_envelope(
    draft: DraftFactory, core: TranslatorCore, context_holding: ContextHolding
) -> None:
    """A successful compile answers the model with the same typed shape a
    referral does.

    `tests/translator/test_model_facing_tool_results.py` holds this boundary
    on the discovery and refusal paths. A compile runs the most upstream
    calls of any path (`search` and `describe` per requirement, then every
    capability schema `propose_operations` reads), so it is the path with
    the most envelopes to leak, and it is checked on the same key set.
    """
    context = context_holding(draft())

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert _flat_keys(result) & _ENVELOPE_KEYS == set()


class _ScriptedLlm(BaseLlm):
    """Answers each generation from `script`, in order: a list entry is
    either a list of `(tool_name, args)` function calls, a string of final
    text, or a callable that reads the `LlmRequest` and returns one of the
    other two. Past the end of the script it answers "done", so an
    unscripted extra generation ends the turn rather than looping.
    """

    model: str = "scripted-mechanical-test"
    script: list[Any] = PydanticField(default_factory=list)
    captured_requests: list[Any] = PydanticField(default_factory=list)

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        self.captured_requests.append(llm_request)
        index = len(self.captured_requests) - 1
        step = self.script[index] if index < len(self.script) else "done"
        if callable(step):
            step = step(llm_request)
        if isinstance(step, str):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part.from_text(text=step)])
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name=name, args=args) for name, args in step],
            )
        )


async def test_a_clean_plan_costs_the_translators_model_exactly_one_generation(
    draft: DraftFactory, combined_surface: McpSurface
) -> None:
    """A clean plan's only model call closes the turn.

    `translate` drives the whole loop as code and never touches a model
    (`translator.mechanical` imports nothing from
    `google.genai`/`google.adk.models`), and the generation that would decide
    to call it never reaches the model either: `_before_translator_model`
    answers that generation with `translate`'s own call. Asserted on the
    translator's own scripted model's call count, so a reintroduced
    generation fails here even where the result stays correct.
    """
    plan = draft()
    translator = create_translator_agent(
        model_name="m", surface=combined_surface, mode="single_turn"
    )
    translator.model = _ScriptedLlm(script=['{"outcome": "ok"}'])
    root_llm = _ScriptedLlm(script=[[("dfo_translator", {"request": "irrelevant here"})]])
    root = LlmAgent(name="root", model=root_llm, sub_agents=[translator])
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="app", user_id="u", session_id="s")
    session = await session_service.get_session(app_name="app", user_id="u", session_id="s")
    assert session is not None
    save_plan(cast("Any", session.state), plan)
    runner = Runner(agent=root, app_name="app", session_service=session_service)
    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        pass

    translator_llm = cast("_ScriptedLlm", translator.model)
    assert len(translator_llm.captured_requests) == 1
    (response,) = [
        part.function_response
        for part in translator_llm.captured_requests[0].contents[-1].parts
        if part.function_response
    ]
    assert response.name == "translate"


def _last_function_response(request: Any) -> Any:
    """The function response the model's next turn would read, from `request`."""
    (response,) = [p.function_response for p in request.contents[-1].parts if p.function_response]
    return response


def _retype_the_referral_translate_reported(request: Any) -> list[tuple[str, dict[str, Any]]]:
    """Close as a model does: hand `translate`'s own `referral` back through `set_model_response`."""
    referral = _last_function_response(request).response["referral"]
    return [("set_model_response", {"response": referral})]


async def _planner_reads(
    plan: Plan,
    surface: McpSurface,
    *,
    translator_script: list[Any],
) -> dict[str, Any]:
    """Run the translator as the planner's sub-agent; return what the planner's next turn reads.

    The translator's model follows `translator_script`; the planner's model
    calls `dfo_translator` once. The plan is written into the session at
    creation, so `translate` finds it.
    """
    translator = create_translator_agent(model_name="m", surface=surface, mode="single_turn")
    translator.model = _ScriptedLlm(script=translator_script)
    root_llm = _ScriptedLlm(script=[[("dfo_translator", {"request": "irrelevant here"})]])
    root = LlmAgent(name="root", model=root_llm, sub_agents=[translator])
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app",
        user_id="u",
        session_id="s",
        state={PLAN_STATE_KEY: plan.model_dump_json()},
    )
    runner = Runner(agent=root, app_name="app", session_service=session_service)
    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        pass

    assert len(root_llm.captured_requests) == 2
    response = _last_function_response(root_llm.captured_requests[1])
    assert response.name == "dfo_translator"
    return cast("dict[str, Any]", response.response)


def _ranking_plan(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    *,
    measure: str,
) -> Plan:
    dataset = requirement(id="gdp_gtm", measure=measure)
    return Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )


# Referral-shaped, but naming no requirement: what `Referral` refused before
# `Referred` accepted the wire shape.
_FABRICATED: dict[str, Any] = {"kind": "no_data", "reason": "guess"}

_RANK_THEN_REFER: list[Any] = [
    [("translate", {})],
    [("translate", {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"})],
    _retype_the_referral_translate_reported,
]


async def test_a_referral_the_model_hands_back_reaches_the_planner_with_its_candidates(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """The model closes with the `referral` `translate` answered, candidates and all.

    That payload renders each candidate as `compact_source` does (`measures`,
    entities without a `ref`), which is not the `SelectedSource` shape
    `Referral.candidates` validates, so the closing turn cannot be checked
    against `Referral`. The planner still has to read the referral, with
    its candidates, rather than an `Error running sub-agent` string.
    """
    plan = _ranking_plan(requirement, derived_requirement, decision, measure="aid disbursed")

    answer = await _planner_reads(plan, combined_surface, translator_script=_RANK_THEN_REFER)

    assert "result" not in answer, answer
    assert answer["kind"] == "source_ambiguous"
    assert answer["requirement_id"] == "gdp_gtm"
    assert {c["spec_id"] for c in answer["candidates"]} == {SPEC_OTHER, SPEC_THIRD}
    assert answer["recommendation"] == {"spec_id": SPEC_OTHER, "rationale": "closer"}


async def test_a_no_data_referral_reaches_the_planner_with_its_nearest_sources_unnamed(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    plan = _ranking_plan(requirement, derived_requirement, decision, measure="nearly aid")

    answer = await _planner_reads(
        plan,
        combined_surface,
        translator_script=[[("translate", {})], _retype_the_referral_translate_reported],
    )

    assert "result" not in answer, answer
    assert answer["kind"] == "no_data"
    assert answer["no_data_reason"] == "no_observations"
    (nearest,) = answer["candidates"]
    assert "spec_id" not in nearest


async def test_the_recorded_referral_is_what_the_planner_reads_whatever_the_model_typed(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """A model that garbles its closing turn cannot change which referral the planner acts on."""
    plan = _ranking_plan(requirement, derived_requirement, decision, measure="aid disbursed")
    garbled = [("set_model_response", {"response": {"kind": "no_data", "reason": "guess"}})]

    answer = await _planner_reads(
        plan,
        combined_surface,
        translator_script=[*_RANK_THEN_REFER[:2], garbled],
    )

    assert answer["kind"] == "source_ambiguous"
    assert answer["requirement_id"] == "gdp_gtm"
    assert len(answer["candidates"]) == 2


@pytest.mark.parametrize(
    "translator_script",
    [
        pytest.param(
            [[("translate", {})], [("set_model_response", {"response": _FABRICATED})]],
            id="after_a_clean_compile",
        ),
        pytest.param(
            [[("set_model_response", {"response": _FABRICATED})]], id="without_calling_translate"
        ),
    ],
)
async def test_a_referral_no_translate_call_recorded_never_reaches_the_planner(
    draft: DraftFactory, combined_surface: McpSurface, translator_script: list[Any]
) -> None:
    """With nothing recorded, only `Compiled` may close the turn.

    `Referred` carries none of `Referral`'s cross-field rules (this one names
    no requirement), so accepting it would hand the planner a referral no
    code produced. The closing turn fails instead, and the planner reads the
    sub-agent error rather than the invented referral.
    """
    answer = await _planner_reads(draft(), combined_surface, translator_script=translator_script)

    assert "kind" not in answer, answer
    assert str(answer.get("result", "")).startswith("Error running sub-agent"), answer


async def test_a_compiled_plan_closes_on_the_outcome_alone(
    draft: DraftFactory, combined_surface: McpSurface
) -> None:
    answer = await _planner_reads(
        draft(),
        combined_surface,
        translator_script=[
            [("translate", {})],
            [("set_model_response", {"response": {"outcome": "ok"}})],
        ],
    )

    assert answer == {"outcome": "ok"}


async def test_run_translate_compiles_a_two_level_derived_chain(
    requirement: RequirementFactory,
    decision: DecisionFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
) -> None:
    dataset = requirement(id="dataset_a")
    totals = AggregateRequirement(
        id="totals_a",
        decision_ids=["measure"],
        input_id="dataset_a",
        reduction="sum",
        group_by=("period",),
    )
    growth = GrowthRequirement(
        id="growth_a",
        decision_ids=["measure"],
        input_id="totals_a",
        window_start=2016,
        window_end=2020,
    )
    plan = Plan(
        question="Kenya's debt service growth",
        decisions=[decision()],
        evidence_requirements=[dataset, totals, growth],
        status=PlanStatus.DRAFT,
    )
    context = context_holding(plan)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert [s["tool"] for s in result["steps"]] == [
        "data-mcp.get_data",
        "analysis-mcp.aggregate",
        "analysis-mcp.growth",
    ]


async def test_run_translate_refers_an_ambiguous_requirement_after_ranking(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
) -> None:
    """A Guatemala-shaped plan: one requirement resolves `definite`, the other `candidates`.

    `population_gtm` resolves clean, `gdp_gtm` comes back with two
    candidates and no pin -- `run_translate` stops there for the one
    judgement it still asks a model to make, and the referral it eventually
    raises carries the model's own `choice` and `rationale` as a
    recommendation, never a pick this loop made itself (`discover_source`'s
    own answer already forbids that).
    """
    population = requirement(id="population_gtm", measure="population total")
    population_total = derived_requirement(id="population_total", input_id="population_gtm")
    gdp = requirement(id="gdp_gtm", measure="aid disbursed")
    gdp_total = derived_requirement(id="gdp_total", input_id="gdp_gtm")
    plan = Plan(
        question="What is Guatemala's population and aid received?",
        decisions=[decision()],
        evidence_requirements=[population, population_total, gdp, gdp_total],
        status=PlanStatus.DRAFT,
    )
    context = context_holding(plan)

    needs_decision = await run_translate(context, core=core)

    assert needs_decision["outcome"] == "needs_decision", needs_decision
    assert needs_decision["judgement"] == "candidate_ranking"
    assert needs_decision["requirement_id"] == "gdp_gtm"
    candidate_ids = {c["spec_id"] for c in needs_decision["candidates"]}
    assert candidate_ids == {SPEC_OTHER, SPEC_THIRD}
    # Each candidate is described, not just named -- describe ran from code.
    assert all(c["definition"] for c in needs_decision["candidates"])
    assert needs_decision["decisions"], "the decisions citing gdp_gtm ride along, unrequested"

    referred = await run_translate(
        context,
        core=core,
        requirement_id="gdp_gtm",
        choice=SPEC_OTHER,
        rationale="closer to the ask",
    )

    assert referred["outcome"] == "referred", referred
    referral = referred["referral"]
    assert referral["kind"] == "source_ambiguous"
    recommendation = referral["recommendation"]
    assert recommendation["spec_id"] == SPEC_OTHER
    assert recommendation["rationale"] == "closer to the ask"


async def test_run_translate_refuses_a_ranking_answer_naming_an_unshown_candidate(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
) -> None:
    gdp = requirement(id="gdp_gtm", measure="aid disbursed")
    gdp_total = derived_requirement(id="gdp_total", input_id="gdp_gtm")
    plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[gdp, gdp_total],
        status=PlanStatus.DRAFT,
    )
    context = context_holding(plan)
    await run_translate(context, core=core)

    result = await run_translate(
        context, core=core, requirement_id="gdp_gtm", choice=SPEC_KENYA, rationale="r"
    )

    assert result["outcome"] == "invalid_input"
    assert result["code"] == "invalid_arguments"


async def test_run_translate_refuses_a_ranking_answer_with_no_pending_judgement(
    draft: DraftFactory, core: TranslatorCore, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())

    result = await run_translate(
        context, core=core, requirement_id="annual_debt_service", choice=SPEC_KENYA, rationale="r"
    )

    assert result["outcome"] == "invalid_input"
    assert result["code"] == "no_pending_judgement"


async def test_run_translate_refers_a_timed_out_search_as_discovery_unavailable(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    """A `search` that never reaches the server must not end `translate` on
    a `needs_decision` no `translate` argument can answer -- `_resume` has
    only `NO_PENDING_JUDGEMENT` to offer any `requirement_id`/`choice`/
    `decision_id` a model might guess, since `_discover_one` saves no
    pending judgement for this shape at all.

    Nor may it end on `discover_source`'s own raw refusal handed back
    unwrapped: the translator's `output_schema` is `Compiled | Referral`
    (`translator.agent.create_translator_agent`), and a raw problem dict
    matches neither, so the model would have no legal shape to close the
    turn on either way -- confirmed directly against
    `test_a_closing_turn_that_does_not_match_the_schema_hard_fails`
    (`test_agent.py`), which shows ADK's `process_llm_agent_output` raises on
    exactly that mismatch, caught only as an opaque "Error running sub-agent"
    the planner cannot act on structurally.

    `run_translate` raises a real `discovery_unavailable` referral
    itself, mechanically, so `translate`'s own result is already the
    schema-conforming `Referral` the model only has to restate -- the same
    shape every other stopping path in this loop already produces.
    """
    unreachable = McpSurface(sources={"data-mcp": "http://127.0.0.1:9/mcp"})
    core = build_translator_core(surface=unreachable)
    context = context_holding(draft())

    result = await run_translate(context, core=core)

    assert result["outcome"] == "referred", result
    referral = result["referral"]
    assert referral["kind"] == "discovery_unavailable"
    assert referral["requirement_id"] == "annual_debt_service"
    assert "could not call 'search'" in referral["reason"]
    assert load_referral(context.state).kind is ReferralKind.DISCOVERY_UNAVAILABLE


async def test_run_translate_refers_a_search_that_finds_a_dropped_dataset_as_seed_dropped(
    draft: DraftFactory, core: TranslatorCore, context_holding: ContextHolding
) -> None:
    """The planner reads why the requirement has no source: the turn dropped the dataset
    its search found, and the plan is not compiled around it."""
    context = context_holding(draft())
    save_dropped_seeds(
        context.state,
        [SeededSource(source=SelectedSource(spec_id=SPEC_KENYA, sentence="s", has_data=True))],
    )

    result = await run_translate(context, core=core)

    assert result["outcome"] == "referred", result
    referral = result["referral"]
    assert referral["kind"] == "seed_dropped"
    assert referral["requirement_id"] == "annual_debt_service"
    assert "Remove this requirement" in referral["reason"]
    assert load_referral(context.state).kind is ReferralKind.SEED_DROPPED


async def test_run_translate_takes_no_source_from_a_search_answered_before_the_drop(
    draft: DraftFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
    search_calls: list[dict[str, Any]],
) -> None:
    """A search that settled the requirement before the turn dropped its dataset
    is not reused: the requirement is searched again and referred as `seed_dropped`."""
    context = context_holding(draft())
    found = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert found["status"] == "definite"
    assert len(search_calls) == 1
    save_dropped_seeds(
        context.state,
        [SeededSource(source=SelectedSource(spec_id=SPEC_KENYA, sentence="s", has_data=True))],
    )

    proposals: list[object] = []

    async def propose_operations(**kwargs: object) -> dict[str, object]:
        proposals.append(kwargs)
        return await core.propose_operations(**kwargs)

    result = await run_translate(
        context, core=dataclasses.replace(core, propose_operations=propose_operations)
    )

    # Searched again at once: no translation is proposed around a missing source.
    assert proposals == []
    assert len(search_calls) == 2
    assert result["outcome"] == "referred", result
    assert result["referral"]["kind"] == "seed_dropped"
    assert result["referral"]["requirement_id"] == "annual_debt_service"


async def test_run_translate_compiles_a_seed_whose_sibling_over_other_years_was_dropped(
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    core: TranslatorCore,
    search_calls: list[dict[str, Any]],
) -> None:
    """Two seeds over one dataset and different years, one dropped. The plan
    pins the one the turn still holds, which is honoured from the seeded
    source and compiled: the pin check and the source filter agree. The seed
    leaves its 193 countries open, so the requirement covers every country
    and the total keeps each one apart."""
    set_definition_store(InMemoryDefinitionStore())
    try:
        block = json.loads(SEEDED_REQUEST_FIXTURE.read_text())["seed_spec"][0]["spec"]
        seed = source_from_spec_block(block)
        identity = block["definition"]["identity"]

        def over(start: int, end: int) -> SelectedSource:
            window = {"kind": "range", "start_year": start, "end_year": end}
            spec_id, windowed = windowed_identity(identity, window)
            return seed.model_copy(update={"spec_id": spec_id, "identity": windowed})

        dropped, held = over(2010, 2010), over(2015, 2024)
        pinned = requirement(
            id="annual_debt_service",
            entities=[],
            across="country",
            source_spec_id=held.spec_id,
            decision_ids=["measure", "source"],
            period=Period(start=2015, end=2024),
        )
        source_decision = pinning_decision(
            held.spec_id,
            slot="source",
            description="Which source.",
            options=[held.spec_id, "x"],
            chosen=held.spec_id,
        )
        context = context_holding(
            draft(
                decisions=[decision(), source_decision],
                evidence_requirements=[
                    pinned,
                    derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD)),
                ],
            )
        )
        save_seeded_sources(context.state, [SeededSource(source=held)])
        save_dropped_seeds(context.state, [SeededSource(source=dropped)])

        result = await run_translate(context, core=core)
    finally:
        set_definition_store(None)

    assert result["outcome"] == "ok", result
    assert search_calls == []


# What one further retrieval of a dataset adds to a `translate` result: its
# spec_id and the phrase it was searched as (`translator.discovery.described_sources`).
_ALSO_RETRIEVED_CHARS = (
    len(json.dumps({"spec_id": "spec_" + "0" * 32, "searched_as": ""}))
    + MAX_MEASURE_CHARS
    + len(", ")
)
# The `sources` of a `translate` result for a plan over seeds: each seed
# described once, at most `MAX_SEED_SOURCE_CHARS` with its stamped phrase, and
# one `_ALSO_RETRIEVED_CHARS` for every other requirement that retrieves one.
TRANSLATE_SEEDED_SOURCES_CEILING = (
    MAX_SEEDS
    * (
        MAX_SEED_SOURCE_CHARS
        + len('""')
        + MAX_MEASURE_CHARS
        + len(', "also_retrieved_as": []')
        + len(", ")
    )
    + (MAX_PLAN_REQUIREMENTS - MAX_SEEDS) * _ALSO_RETRIEVED_CHARS
    + len("[]")
)
# The rest of the result is per requirement: one step, an id and a tool in
# 128 characters, and the compiler's warnings about its source, which quote
# the source's unit and window. Three warnings a requirement measure 2,787
# characters with every quoted string at `MAX_SEED_TEXT_CHARS`; 4,096 is
# allowed.
TRANSLATE_SEEDED_RESULT_CEILING = (
    TRANSLATE_SEEDED_SOURCES_CEILING + 1_024 + MAX_PLAN_REQUIREMENTS * (128 + 4_096)
)
# `planner:plan` holds the workflow, which keeps one source record and one
# `get_data` step per retrieval: the record at most `MAX_SEED_RECORD_CHARS`
# with its stamped phrase, the step's posted StatVars at most
# `MAX_SEED_STAT_VARS` dcids, and 4,096 for the step, its binding and its
# operation. The plan's own text is at most `MAX_PLAN_RECORD_CHARS`.
PLAN_STATE_SEEDED_CEILING = MAX_PLAN_RECORD_CHARS + MAX_PLAN_REQUIREMENTS * (
    MAX_SEED_RECORD_CHARS
    + MAX_MEASURE_CHARS
    + MAX_SEED_STAT_VARS * (MAX_SEED_TEXT_CHARS + len('"", '))
    + 4_096
)


def _answered_over_its_window(seed_block: dict[str, Any], copy: SeededSource) -> dict[str, Any]:
    """Search's block for `copy`: the seed's own, under the copy's spec_id and identity,
    with exact coverage over the copy's years."""
    identity = copy.source.identity
    assert identity is not None
    window = identity["requested_window"]
    return {
        "spec_id": copy.source.spec_id,
        "definition": {
            "schema_version": 1,
            "spec_id": copy.source.spec_id,
            "identity": identity,
        },
        "resolved": {
            **seed_block["resolved"],
            "coverage": {
                **seed_block["resolved"]["coverage"],
                "window": {"start_year": window["start_year"], "end_year": window["end_year"]},
            },
        },
    }


@pytest.mark.parametrize("a_period_each", [False, True], ids=["one-period", "a-period-each"])
async def test_translate_describes_each_seed_once_however_many_requirements_retrieve_it(
    a_period_each: bool,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    context_holding: ContextHolding,
    core: TranslatorCore,
) -> None:
    """`MAX_SEEDS` seeds at the description bound, retrieved by `MAX_PLAN_REQUIREMENTS`
    requirements: under one period per seed, or under a period per requirement, which
    pins a copy of the seed for each, whose coverage Search states for its own years.
    The result and the plan in state stay under ceilings derived from the seed bounds."""
    set_definition_store(InMemoryDefinitionStore())
    try:
        seeds: list[SeededSource] = []
        blocks: dict[str, dict[str, Any]] = {}
        decisions: list[Decision] = []
        requirements: list[DatasetRequirement] = []
        # A plan's own text is capped (`MAX_PLAN_RECORD_CHARS`), which a pin
        # decision per copy reaches at two periods a seed.
        per_seed = 2 if a_period_each else MAX_PLAN_REQUIREMENTS // MAX_SEEDS
        for number in range(MAX_SEEDS):
            wire_seed = fullest_described_seed(number)[0]
            # The entity every requirement here names.
            wire_seed["spec"]["resolved"]["entities"][0] = {
                "role": "destination",
                "ref": "country/KEN",
                "label": "Kenya",
            }
            seed = read_seed(number, wire_seed).source
            seeds.append(seed)
            blocks[seed.source.spec_id] = wire_seed["spec"]
            for index in range(per_seed):
                slot = f"source_{number}_{index if a_period_each else 0}"
                if all(decision.slot != slot for decision in decisions):
                    decisions.append(
                        pinning_decision(
                            seed.source.spec_id,
                            slot=slot,
                            description="Which source.",
                            options=[seed.source.spec_id, "x"],
                            chosen=seed.source.spec_id,
                        )
                    )
                requirements.append(
                    requirement(
                        id=f"req_{number}_{index}",
                        source_spec_id=seed.source.spec_id,
                        decision_ids=[slot],
                        measure=f"measure {number} {index}",
                        period=Period(start=2000 + (index if a_period_each else 0), end=2024),
                    )
                )
        assert len(requirements) == MAX_SEEDS * per_seed
        # As `propose_plan` records it: each pin moved to the seed over the
        # requirement's period, and a copy for each.
        plan, copies = _window_seeded_pins(
            draft(decisions=decisions, evidence_requirements=requirements), seeds
        )
        assert len(copies) == MAX_SEEDS * (per_seed if a_period_each else 1)
        pinned = {
            item.source_spec_id
            for item in plan.evidence_requirements
            if isinstance(item, DatasetRequirement)
        }
        assert pinned == {copy.source.spec_id for copy in copies}
        context = context_holding(plan)
        save_seeded_sources(context.state, [*seeds, *copies])
        search = SearchStub(
            _answered_over_its_window(blocks[cast(str, copy.windowed_from)], copy)
            for copy in copies
        )

        with search_api(search):
            result = await run_translate(context, core=core)
    finally:
        set_definition_store(None)

    assert result["outcome"] == "ok", str(result.get("referral"))[:1500]
    sources = cast(list[dict[str, Any]], result["sources"])
    assert len(sources) == MAX_SEEDS
    named = [
        retrieval["spec_id"]
        for source in sources
        for retrieval in (source, *source.get("also_retrieved_as", []))
    ]
    assert sorted(set(named)) == sorted(pinned)
    assert len(json.dumps(sources)) <= TRANSLATE_SEEDED_SOURCES_CEILING
    assert len(json.dumps(result)) <= TRANSLATE_SEEDED_RESULT_CEILING
    assert len(json.dumps(sources)) > MAX_SEEDS * MAX_SEED_SOURCE_CHARS * 0.98
    assert len(context.state[PLAN_STATE_KEY]) <= PLAN_STATE_SEEDED_CEILING


# Search's own answers for one dataset over three periods, keyed by the years.
_KENYA_OVER: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "resubmit_education_oda_kenya.json").read_text()
)["specs"]


def _kenya_seed_pinned_over_2010_to_2023(
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> tuple[Any, SeededSource]:
    """A context whose plan pins the education-aid-to-Kenya seed over 2015-2023 by a
    requirement over 2010-2023, as `propose_plan` records it, and the copy it retrieves."""
    seed = source_from_spec_block(_KENYA_OVER["2015-2023"])
    assert seed.coverage_window == Period(start=2015, end=2023)
    pinned = requirement(
        source_spec_id=seed.spec_id,
        decision_ids=["measure", "source"],
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2010, end=2023),
    )
    source_decision = pinning_decision(
        seed.spec_id,
        slot="source",
        description="Which source.",
        options=[seed.spec_id, "x"],
        chosen=seed.spec_id,
    )
    plan, copies = _window_seeded_pins(
        draft(
            decisions=[decision(), source_decision],
            evidence_requirements=[pinned, derived_requirement()],
        ),
        [SeededSource(source=seed)],
    )
    (copy,) = copies
    assert copy.source.spec_id == _KENYA_OVER["2010-2023"]["spec_id"]
    context = context_holding(plan)
    save_seeded_sources(context.state, [SeededSource(source=seed), copy])
    return context, copy


@pytest.mark.parametrize(
    ("stated", "short"),
    [
        ({"start_year": 2010, "end_year": 2023}, False),
        ({"start_year": 2015, "end_year": 2023}, True),
    ],
    ids=["search-states-2010-2023", "search-states-2015-2023"],
)
async def test_a_seed_pinned_over_a_wider_period_compiles_against_searchs_coverage_for_it(
    stated: dict[str, int],
    short: bool,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    core: TranslatorCore,
    search_calls: list[dict[str, Any]],
) -> None:
    """A seed over 2015-2023 pinned by a requirement over 2010-2023. The plan retrieves
    the seed's copy over 2010-2023, and the compiler checks the period against the
    coverage Search states for the copy: no `window_short` when that is 2010-2023, and
    `window_short` when Search states the data starts in 2015."""
    answered = _KENYA_OVER["2010-2023"]
    assert answered["resolved"]["coverage"]["window"] == {"start_year": 2010, "end_year": 2023}
    answered = {
        **answered,
        "resolved": {
            **answered["resolved"],
            "coverage": {**answered["resolved"]["coverage"], "window": stated},
        },
    }
    context, copy = _kenya_seed_pinned_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )
    set_definition_store(InMemoryDefinitionStore())
    try:
        with search_api(SearchStub([answered])):
            result = await run_translate(context, core=core)
    finally:
        set_definition_store(None)

    assert search_calls == []
    assert result["outcome"] == "ok", result
    codes = [warning["code"] for warning in cast(list[dict[str, Any]], result["warnings"])]
    assert ("window_short" in codes) is short
    # The source the plan holds is Search's for the copy.
    (described,) = cast(list[dict[str, Any]], result["sources"])
    assert described["spec_id"] == copy.source.spec_id
    assert described["sentence"] == answered["resolved"]["rendered_sentence"]
    assert "2010 to 2023" in described["sentence"]
    assert described["coverage_window"] == {"start": stated["start_year"], "end": 2023}


def _kenya_without_its_definition() -> dict[str, Any]:
    block = dict(_KENYA_OVER["2010-2023"])
    del block["definition"]
    return block


def _answering(part: dict[str, Any]) -> dict[str, Any]:
    """A one-part Search envelope answering `part`."""
    return {"schema_version": 1, "status": part["status"], "parts": [part]}


# What Search answers for the copy's identity (`None`: it refuses it), the referral
# the planner reads, and the reason code that referral carries.
_COPY_REFERRALS: dict[str, tuple[dict[str, Any] | None, ReferralKind, str | None]] = {
    "no-data": (
        _answering({"status": "no_data", "no_data": {"reason": "no_series"}}),
        ReferralKind.NO_DATA,
        "no_series",
    ),
    "refused": (None, ReferralKind.REQUIREMENT_UNMEETABLE, None),
    "candidates": (
        _answering({"status": "candidates", "candidates": {"specs": []}}),
        ReferralKind.REQUIREMENT_UNMEETABLE,
        None,
    ),
    "another-spec-id": (
        _answering({"status": "definite", "spec": _KENYA_OVER["2015-2023"]}),
        ReferralKind.REQUIREMENT_UNMEETABLE,
        None,
    ),
    "no-definition": (
        _answering({"status": "definite", "spec": _kenya_without_its_definition()}),
        ReferralKind.REQUIREMENT_UNMEETABLE,
        None,
    ),
}


@pytest.mark.parametrize("answer", sorted(_COPY_REFERRALS))
async def test_a_seed_copy_search_answers_for_good_is_referred_as_a_fact_about_the_requirement(
    answer: str,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    core: TranslatorCore,
    search_calls: list[dict[str, Any]],
) -> None:
    """Search's answer for the copy's period is final, so the planner is told what it
    means for the requirement: no data over those years, or a pinned dataset that cannot
    serve them. A `discovery_unavailable` referral would have the planner propose the
    same plan again, and Search would answer it the same way."""
    envelope, kind, reason_code = _COPY_REFERRALS[answer]
    context, copy = _kenya_seed_pinned_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )
    answers = {copy.source.spec_id: envelope} if envelope is not None else None
    set_definition_store(InMemoryDefinitionStore())
    try:
        with search_api(SearchStub(answer=answers)):
            result = await run_translate(context, core=core)
    finally:
        set_definition_store(None)

    assert search_calls == []
    assert result["outcome"] == "referred", result
    referral = Referral.model_validate(result["referral"])
    assert referral.kind is kind
    assert referral.requirement_id == "annual_debt_service"
    assert "over 2010 to 2023" in referral.reason
    assert referral.no_data_reason == reason_code
    assert referral.candidates == ()
    assert load_referral(context.state) == referral


class _UnreachableDefinitionStore:
    """A definition store whose backend cannot be reached."""

    async def put_many(self, definitions: Any) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


async def test_a_seed_copy_data_mcp_cannot_store_is_referred_as_discovery_unavailable(
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    core: TranslatorCore,
) -> None:
    """A service that could not answer just now is the one answer the same plan can
    succeed on later."""
    context, _ = _kenya_seed_pinned_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )
    set_definition_store(_UnreachableDefinitionStore())
    try:
        with search_api(SearchStub([_KENYA_OVER["2010-2023"]])):
            result = await run_translate(context, core=core)
    finally:
        set_definition_store(None)

    assert result["outcome"] == "referred", result
    referral = Referral.model_validate(result["referral"])
    assert referral.kind is ReferralKind.DISCOVERY_UNAVAILABLE
    assert referral.requirement_id == "annual_debt_service"
    assert "the definition store is unreachable" in referral.reason


async def test_run_translate_refuses_an_unmodeled_requirement_kind_rather_than_raising(
    requirement: RequirementFactory,
    decision: DecisionFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requirement kind no builder covers refers the plan back, the same
    way `requirement_unmeetable` already does for a source no operation can
    use -- never a bare `KeyError` out of `build_translation`.

    `JoinRequirement` is dropped from `_BUILDERS` for this test alone, since
    every kind `planner.plan` actually defines is covered today; this is
    the shape a *future* uncovered kind takes, not a live gap.
    """
    monkeypatch.delitem(_BUILDERS, JoinRequirement)
    dataset_a = requirement(id="dataset_a")
    dataset_b = requirement(id="dataset_b", measure="government revenue")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="dataset_a", right_id="dataset_b"
    )
    plan = Plan(
        question="joined",
        decisions=[decision()],
        evidence_requirements=[dataset_a, dataset_b, joined],
        status=PlanStatus.DRAFT,
    )
    context = context_holding(plan)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "referred", result
    referral = result["referral"]
    assert referral["kind"] == "requirement_unmeetable"
    assert referral["requirement_id"] == "joined"
    unsupported = _unsupported_requirement(plan)
    assert unsupported is not None
    assert unsupported.id == "joined"


async def test_run_translate_refuses_the_unmodeled_kind_before_any_discovery_call(
    requirement: RequirementFactory,
    decision: DecisionFactory,
    core: TranslatorCore,
    context_holding: ContextHolding,
    monkeypatch: pytest.MonkeyPatch,
    search_calls: list[dict[str, Any]],
) -> None:
    monkeypatch.delitem(_BUILDERS, JoinRequirement)
    dataset_a = requirement(id="dataset_a")
    dataset_b = requirement(id="dataset_b", measure="government revenue")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="dataset_a", right_id="dataset_b"
    )
    plan = Plan(
        question="joined",
        decisions=[decision()],
        evidence_requirements=[dataset_a, dataset_b, joined],
        status=PlanStatus.DRAFT,
    )
    context = context_holding(plan)

    await run_translate(context, core=core)

    assert search_calls == []


async def test_run_translate_settles_an_unmet_capability_judgement_with_a_decision_id(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    """`_compile_failure_action`'s own `no_capability` branch names `needs=
    ("decision_id",)`: `run_translate` cannot supply it on its own, so it
    pauses for the model's judgement, the same as candidate ranking pauses
    for the model's choice, and settles once the model answers with a
    `decision_id` -- naming `NO_OPERATION_COVERS` where the pending
    judgement said to, not a kind the caller gets to pick.
    """
    referred_with: dict[str, object] = {}

    async def discover_source(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"outcome": "ok", "status": "definite", "sources": [{"spec_id": SPEC_KENYA}]}

    async def propose_operations(*, translation: object, tool_context: Any) -> dict[str, object]:
        del translation, tool_context
        return {
            "outcome": "invalid_input",
            "code": "compile_failed",
            "message": "no operation covers this",
            "action": {
                "call": "return_to_planner",
                "arguments": {"kind": "no_operation_covers", "reason": "no tool implements this"},
                "needs": ["decision_id"],
            },
        }

    def return_to_planner(**kwargs: object) -> dict[str, object]:
        referred_with.update(kwargs)
        return {
            "outcome": "referred",
            "referral": {"kind": kwargs["kind"], "reason": kwargs["reason"]},
        }

    core = TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=None,  # type: ignore[arg-type]
    )
    context = context_holding(draft())

    needs_decision = await run_translate(context, core=core)

    assert needs_decision["outcome"] == "needs_decision"
    assert needs_decision["judgement"] == "unmet_capability"
    assert needs_decision["decisions"], "the plan's decisions ride along for the model to pick from"

    referred = await run_translate(context, core=core, decision_id="measure")

    assert referred["outcome"] == "referred", referred
    assert referred_with["kind"] == "no_operation_covers"
    assert referred_with["decision_id"] == "measure"


async def test_run_translate_gives_up_after_max_rounds(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    """A `propose_operations` that keeps naming the same `discover_source`
    retry forever is bounded by `MAX_ROUNDS`, not left to loop."""

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        return {
            "outcome": "ok",
            "requirement_id": requirement_id,
            "status": "definite",
            "sources": [{"spec_id": SPEC_KENYA}],
        }

    async def propose_operations(*, translation: object, tool_context: Any) -> dict[str, object]:
        del translation, tool_context
        return {
            "outcome": "invalid_input",
            "code": "compile_failed",
            "message": "never compiles",
            "action": {
                "call": "discover_source",
                "arguments": {"requirement_id": "annual_debt_service"},
            },
        }

    def return_to_planner(**kwargs: object) -> dict[str, object]:
        raise AssertionError(
            "never reached: the retry table keeps pointing back at discover_source"
        )

    core = TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=None,  # type: ignore[arg-type]
    )
    context = context_holding(draft())

    result = await run_translate(context, core=core)

    assert result["outcome"] == "needs_decision"
    assert result["judgement"] == "unmet_capability"
    assert "4" in result["reason"]


# --- _resolve_dataset_sources: a settled requirement is not re-spent -------


def _seed_settled(
    context: Any,
    plan: Plan,
    requirement: DatasetRequirement,
    source: SelectedSource,
    *,
    status: DiscoveryStatus,
    spec_ids: tuple[str, ...],
) -> None:
    """Seed the log with a settled record for `requirement`, as a real
    `discover_source` call would have left it: `source`'s own `spec_id`
    among `spec_ids`, keyed by `requirement`'s own `requirement_revision`
    so `selectable` matches it regardless of `plan`'s revision.
    """
    record = DiscoveryRecord(
        revision=plan_revision(plan),
        tool="search",
        requirement_id=requirement.id,
        requirement_key=requirement_revision(requirement),
        status=status,
        spec_ids=spec_ids,
    )
    save_log(context.state, load_log(context.state).with_record(record, (source,)))


def _spying_core(
    *, discover_source: Callable[..., Any], captured: list[Translation]
) -> TranslatorCore:
    """A `TranslatorCore` whose `propose_operations` records the `Translation`
    it was built with, and always reports `ok` -- `_resolve_dataset_sources`
    is what these tests exercise, not compilation.
    """

    async def propose_operations(
        *, translation: Translation, tool_context: Any
    ) -> dict[str, object]:
        del tool_context
        captured.append(translation)
        return {"outcome": "ok"}

    def return_to_planner(**kwargs: object) -> dict[str, object]:
        raise AssertionError("not exercised by this test")

    return TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=None,  # type: ignore[arg-type]
    )


def _retrieved_spec_id(translation: Translation, requirement_id: str) -> str | None:
    (op,) = [o for o in translation.operations if o.requirement_id == requirement_id]
    assert op.kind == "retrieve"
    return cast(Any, op).spec_id


async def test_a_settled_requirement_is_not_rediscovered_on_a_later_round(
    draft: DraftFactory, source: SourceFactory, context_holding: ContextHolding
) -> None:
    """A dataset requirement discovery already settled -- a `DEFINITE`
    record `DiscoveryLog.selectable` can hand back -- costs
    `_resolve_dataset_sources` no fresh `discover_source` call, the same
    reuse `propose_operations` already gets from `selectable`. Confirmed
    against the unfixed `_resolve_dataset_sources` (which called
    `discover_source` unconditionally, for every requirement, every time)
    that this test fails there: `discover_source` is called and the spy
    below raises.
    """
    plan = draft()
    requirement = cast(DatasetRequirement, plan.evidence_requirements[0])
    context = context_holding(plan)
    _seed_settled(
        context,
        plan,
        requirement,
        source(spec_id=SPEC_OTHER),
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_OTHER,),
    )

    async def discover_source(**kwargs: object) -> dict[str, object]:
        raise AssertionError(
            f"discover_source must not be called for a settled requirement: {kwargs}"
        )

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert len(captured) == 1
    assert _retrieved_spec_id(captured[0], requirement.id) == SPEC_OTHER


async def test_a_requirement_whose_searched_content_changed_is_rediscovered(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    """A settled record only covers the requirement's own searched content
    (`requirement_revision`): a requirement sharing the same id but asking
    about a different `measure` earns a fresh `discover_source` call rather
    than reusing a record settled for the old content.
    """
    stale = requirement()
    plan = draft(evidence_requirements=[stale, derived_requirement()])
    context = context_holding(plan)
    _seed_settled(
        context,
        plan,
        stale,
        source(spec_id=SPEC_OTHER),
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_OTHER,),
    )

    changed = requirement(measure="a different ask, never searched before")
    save_plan(context.state, draft(evidence_requirements=[changed, derived_requirement()]))

    calls: list[str] = []

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        calls.append(requirement_id)
        return {
            "outcome": "ok",
            "requirement_id": requirement_id,
            "status": "definite",
            "sources": [{"spec_id": SPEC_KENYA}],
        }

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert calls == [changed.id]
    assert _retrieved_spec_id(captured[0], changed.id) == SPEC_KENYA


@pytest.mark.parametrize(
    ("recorded_axes", "rediscovered"),
    [(None, True), ((), False)],
    ids=["axes-never-read", "axes-read-none-open"],
)
async def test_a_requirement_stating_across_rediscovers_a_source_recorded_without_its_open_axes(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
    recorded_axes: tuple[()] | None,
    rediscovered: bool,
) -> None:
    """`across` is no part of `requirement_revision`, so a record settled
    for the same searched content matches a requirement that states it. A
    source whose open entity axes were never read cannot answer `across`, so
    it is discovered again. A source that read them and found none open is
    reused: the compiler refuses it with what it states.
    """
    across = requirement(across="creditor")
    plan = draft(evidence_requirements=[across, derived_requirement()])
    context = context_holding(plan)
    _seed_settled(
        context,
        plan,
        across,
        source(spec_id=SPEC_OTHER, open_entity_axes=recorded_axes),
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_OTHER,),
    )
    assert requirement_revision(across) == requirement_revision(requirement())
    calls: list[str] = []

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        calls.append(requirement_id)
        return _ok_answer(requirement_id, SPEC_KENYA)

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert calls == ([across.id] if rediscovered else [])
    assert _retrieved_spec_id(captured[0], across.id) == (
        SPEC_KENYA if rediscovered else SPEC_OTHER
    )


async def test_a_pinned_requirement_resolves_through_the_pin_path(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    pinning_decision: PinningDecisionFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    """A requirement pinned to a `spec_id` a settled `CANDIDATES` record
    named resolves through `selectable`'s pin branch -- honoured out of a
    `CANDIDATES` record naming the pinned `spec_id`, not the `DEFINITE`-only
    branch -- without spending a fresh `discover_source` call.
    """
    pinned = requirement(source_spec_id=SPEC_THIRD)
    plan = draft(
        decisions=[pinning_decision(SPEC_THIRD)],
        evidence_requirements=[pinned, derived_requirement()],
    )
    context = context_holding(plan)
    _seed_settled(
        context,
        plan,
        pinned,
        source(spec_id=SPEC_THIRD),
        status=DiscoveryStatus.CANDIDATES,
        spec_ids=(SPEC_OTHER, SPEC_THIRD),
    )

    async def discover_source(**kwargs: object) -> dict[str, object]:
        raise AssertionError(
            f"discover_source must not be called for a settled, pinned requirement: {kwargs}"
        )

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert _retrieved_spec_id(captured[0], pinned.id) == SPEC_THIRD


# --- _resolve_dataset_sources: every unsettled requirement discovers at once


def _two_requirement_plan(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
) -> Plan:
    """A draft plan with two independent dataset requirements, `req_a` and `req_b`."""
    first = requirement(id="req_a")
    second = requirement(id="req_b")
    return Plan(
        question="How much does Kenya pay to service its debt, and how much does it borrow?",
        decisions=[decision()],
        evidence_requirements=[
            first,
            derived_requirement(id="req_a_total", input_id="req_a"),
            second,
            derived_requirement(id="req_b_total", input_id="req_b"),
        ],
        status=PlanStatus.DRAFT,
    )


def _ok_answer(requirement_id: str, spec_id: str) -> dict[str, object]:
    return {
        "outcome": "ok",
        "requirement_id": requirement_id,
        "status": "definite",
        "sources": [{"spec_id": spec_id}],
    }


async def test_unsettled_requirements_discover_concurrently(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    context_holding: ContextHolding,
) -> None:
    """Both unsettled requirements' `discover_source` calls are in flight together.

    Against a sequential loop, `in_flight` never rises past 1, since the
    second call is not started until the first has already returned; this
    test would fail there.
    """
    plan = _two_requirement_plan(requirement, derived_requirement, decision)
    context = context_holding(plan)
    in_flight = 0
    peak_in_flight = 0

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        nonlocal in_flight, peak_in_flight
        del tool_context
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return _ok_answer(requirement_id, SPEC_KENYA)

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "ok", result
    assert peak_in_flight == 2


async def test_a_later_requirements_judgement_wins_over_an_earlier_ones_resolution(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    context_holding: ContextHolding,
) -> None:
    """`req_b`'s `needs_decision` is what `run_translate` returns, in requirement order,
    even though `req_b`'s own call answers before `req_a`'s slower one resolves.
    """
    plan = _two_requirement_plan(requirement, derived_requirement, decision)
    context = context_holding(plan)

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        if requirement_id == "req_a":
            await asyncio.sleep(0.05)
            return _ok_answer(requirement_id, SPEC_KENYA)
        # Answers at once, with no action and no single source: the shape
        # `_discover_one` cannot resolve on its own, so it stops on a
        # `needs_decision` rather than guessing.
        return {"outcome": "candidates", "requirement_id": requirement_id}

    captured: list[Translation] = []
    core = _spying_core(discover_source=discover_source, captured=captured)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "needs_decision", result
    assert result["judgement"] == "unmet_capability"
    assert result["requirement_id"] == "req_b"
    assert not captured


async def test_a_stop_on_the_first_requirement_leaves_the_seconds_record_in_the_log(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    """`req_a`'s referral is what `run_translate` returns, but `req_b`'s own
    sibling call has already written its record and sources to the log by
    then -- the same log `discover_source` itself writes to, in production
    -- so a later re-entry finds `req_b` `selectable` and spends no fresh
    call on it.
    """
    plan = _two_requirement_plan(requirement, derived_requirement, decision)
    second = cast(DatasetRequirement, plan.requirement("req_b"))
    context = context_holding(plan)

    def return_to_planner(**kwargs: object) -> dict[str, object]:
        return {"outcome": "referred", "referral": kwargs}

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        if requirement_id == "req_a":
            return {
                "outcome": "no_data",
                "code": ProblemCode.SURFACE_UNAVAILABLE.value,
                "message": "the data service timed out",
            }
        _seed_settled(
            tool_context,
            plan,
            second,
            source(spec_id=SPEC_OTHER),
            status=DiscoveryStatus.DEFINITE,
            spec_ids=(SPEC_OTHER,),
        )
        return _ok_answer(requirement_id, SPEC_OTHER)

    async def propose_operations(**kwargs: object) -> dict[str, object]:
        raise AssertionError(f"not exercised: {kwargs!r} -- req_a's referral stops the loop first")

    core = TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=None,  # type: ignore[arg-type]
    )

    result = await run_translate(context, core=core)

    assert result["outcome"] == "referred", result
    assert result["referral"]["requirement_id"] == "req_a"

    log = load_log(context.state)
    settled = log.selectable(requirement_keys={"req_b": requirement_revision(second)}, pinned={})
    assert "req_b" in settled
    assert settled["req_b"].spec_id == SPEC_OTHER


def _spying_return_to_planner(
    calls: list[dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """A `return_to_planner` fake that records every call and saves the referral for real."""

    def return_to_planner(*, tool_context: Any, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        referral = Referral(**kwargs)
        save_referral(tool_context.state, referral)
        return {"outcome": "referred", "referral": referral.model_dump(mode="json")}

    return return_to_planner


async def test_two_referrals_settle_only_the_first_ones_requirement(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    context_holding: ContextHolding,
) -> None:
    """`req_a` and `req_b` both answer with a complete `return_to_planner`
    action, of different referral kinds. Only `req_a`'s -- the first in
    requirement order -- is ever settled: `return_to_planner` is called once,
    for `req_a`, and the referral saved to state names `req_a`, never
    whichever of the two sibling `discover_source` calls happened to settle
    last.
    """
    plan = _two_requirement_plan(requirement, derived_requirement, decision)
    context = context_holding(plan)

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        kind = "no_data" if requirement_id == "req_a" else "unsupported_ask"
        return {
            "action": {
                "call": "return_to_planner",
                "arguments": {
                    "kind": kind,
                    "reason": f"{requirement_id} cannot be met",
                    "requirement_id": requirement_id,
                },
            }
        }

    async def propose_operations(**kwargs: object) -> dict[str, object]:
        raise AssertionError(f"not exercised: {kwargs!r} -- a referral stops the loop first")

    calls: list[dict[str, object]] = []
    core = TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=_spying_return_to_planner(calls),
        surface=None,  # type: ignore[arg-type]
    )

    result = await run_translate(context, core=core)

    assert result["outcome"] == "referred", result
    assert result["referral"]["requirement_id"] == "req_a"
    assert len(calls) == 1, calls
    assert calls[0]["requirement_id"] == "req_a"

    referral = load_referral(context.state)
    assert referral is not None
    assert referral.requirement_id == "req_a"


async def test_two_ambiguities_settle_only_the_first_ones_pending_judgement(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`req_a` and `req_b` both answer `source_ambiguous`. Only `req_a`'s --
    the first in requirement order -- is ever settled: `_begin_candidate_ranking`
    runs once, for `req_a`, and the pending judgement saved to state names
    `req_a` and its own candidates, never whichever of the two sibling
    `discover_source` calls happened to settle last.
    """
    plan = _two_requirement_plan(requirement, derived_requirement, decision)
    first = cast(DatasetRequirement, plan.requirement("req_a"))
    second = cast(DatasetRequirement, plan.requirement("req_b"))
    context = context_holding(plan)
    _seed_settled(
        context,
        plan,
        first,
        source(spec_id=SPEC_OTHER, stat_vars=()),
        status=DiscoveryStatus.CANDIDATES,
        spec_ids=(SPEC_OTHER,),
    )
    _seed_settled(
        context,
        plan,
        second,
        source(spec_id=SPEC_THIRD, stat_vars=()),
        status=DiscoveryStatus.CANDIDATES,
        spec_ids=(SPEC_THIRD,),
    )

    async def discover_source(*, requirement_id: str, tool_context: Any) -> dict[str, object]:
        del tool_context
        return {
            "action": {
                "call": "return_to_planner",
                "arguments": {
                    "kind": "source_ambiguous",
                    "reason": f"{requirement_id} is ambiguous",
                },
            }
        }

    def return_to_planner(**kwargs: object) -> dict[str, object]:
        raise AssertionError(f"not exercised: {kwargs!r} -- both requirements pause for a decision")

    async def propose_operations(**kwargs: object) -> dict[str, object]:
        raise AssertionError(f"not exercised: {kwargs!r} -- both requirements pause for a decision")

    core = TranslatorCore(
        discover_source=discover_source,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=None,  # type: ignore[arg-type]
    )

    ranking_calls: list[str] = []
    original_begin_ranking = mechanical._begin_candidate_ranking

    async def spying_begin_ranking(*args: Any, **kwargs: Any) -> dict[str, object]:
        ranking_calls.append(cast(str, kwargs["requirement_id"]))
        return await original_begin_ranking(*args, **kwargs)

    monkeypatch.setattr(mechanical, "_begin_candidate_ranking", spying_begin_ranking)

    result = await run_translate(context, core=core)

    assert result["outcome"] == "needs_decision", result
    assert result["judgement"] == "candidate_ranking"
    assert result["requirement_id"] == "req_a"
    assert ranking_calls == ["req_a"], ranking_calls

    pending = _load_pending(context.state)
    assert pending is not None
    assert pending.requirement_id == "req_a"
    assert pending.candidate_spec_ids == (SPEC_OTHER,)
