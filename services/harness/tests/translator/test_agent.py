"""Tests for the translator agent: proposing, referring, and agent construction.

`propose_operations` and `return_to_planner` are exercised directly, through
`translator.agent.build_translator_core`, against a fake tool context: they
are no longer model-facing tools in their own right, but the same functions
`translate` (`translator.mechanical.run_translate`) drives, and this is what
belongs to this package -- the logic each one runs, independent of who
calls it. `translate`'s own loop has its own suite,
`tests/translator/test_mechanical.py`; the compiler's own checks have
theirs.

Every assertion reads `result["outcome"]`, `result["code"]` or
`result["action"]` -- the typed shape `planner.tool_results` defines -- never
a message substring, since the message is prose for a model to read, not a
contract a test pins.

There is no confirmation gate to test here. The translator discovers and
compiles; it retrieves and computes nothing, so the gate stays in the harness
graph, outside this package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

import pytest
from compiler.compile import CompileIssue, CompileIssueCode
from compiler.sources import DiscoveryStatus
from compiler.surface import (
    FixtureSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from google.adk.agents import LlmAgent
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Plan,
    PlanStatus,
    SeededSource,
    Unit,
    plan_revision,
    save_dropped_seeds,
)
from planner.settings import override_thinking_on_sampled_turns
from planner.workflow import SelectedSource, ToolId
from translator.agent import (
    Compiled,
    Referred,
    _before_translator_model,
    _compile_failure_action,
    build_translator_core,
    build_translator_tools,
    create_translator_agent,
)
from translator.discovery import DiscoveryRecord, load_log, requirement_revision, save_log
from translator.mechanical import TranslatorCore
from translator.referral import Referral, ReferralKind, load_referral, save_referral

DraftFactory = Callable[..., Plan]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]
SourceFactory = Callable[..., SelectedSource]

# Fixtures from this suite's conftest, annotated structurally.
ContextHolding = Callable[[Plan], Any]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32

TOOL_NAMES = ("translate",)

# A translation mirroring the `requirements` pair: retrieve the dataset
# requirement, total it per period for the aggregate that reads it.
GOOD_TRANSLATION: dict[str, Any] = {
    "operations": [
        {
            "kind": "retrieve",
            "id": "debt",
            "requirement_id": "annual_debt_service",
            "spec_id": SPEC_KENYA,
        },
        {
            "kind": "aggregate",
            "id": "totals",
            "requirement_id": "annual_totals",
            "input": "debt",
            "reduction": "sum",
            "group_by": ["period"],
        },
    ],
    "terminal": ["totals"],
}


class _UnreadableSurface(FixtureSurface):
    async def list_tools(self) -> list[ToolSchema]:
        raise ToolSurfaceUnavailableError("could not read the tool surface at 'data-mcp'")


class _CountingSurface(FixtureSurface):
    """A `FixtureSurface` that counts each tool it was asked to call.

    Proves `propose_operations`'s `describe` fetch is memoized on a
    same-revision retry, rather than only checking the result.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: dict[str, int] = {}

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: dict[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls[tool.name] = self.calls.get(tool.name, 0) + 1
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


# The default `source()` fixture's own `stat_vars` is `("DebtServicePaid",)`.
_DESCRIBE_RESPONSE = ToolCallResult(
    is_error=False,
    structured_content={
        "nodes": [
            {
                "shape": "stat_var",
                "dcid": "DebtServicePaid",
                "label": "DebtServicePaid",
                "definition": "External debt service a country pays to a foreign creditor.",
                "roles": ["measure"],
            }
        ]
    },
)


def _core(real_schemas: list[ToolSchema]) -> TranslatorCore:
    surface = FixtureSurface(schemas=real_schemas)
    return build_translator_core(surface=surface)


def _discovered(
    context: Any,
    *sources: SelectedSource,
    status: DiscoveryStatus,
    requirement_id: str = "annual_debt_service",
    variable_sent: str | None = None,
    reason: str | None = None,
) -> None:
    """Seed the log as a discovery answer of `status` naming `sources` would have.

    Each source is stamped `searched_as=variable_sent`, as `discover_source` does.
    """
    plan = Plan.model_validate_json(context.state[PLAN_STATE_KEY])
    requirement = plan.requirement(requirement_id)
    assert isinstance(requirement, DatasetRequirement)
    record = DiscoveryRecord(
        revision=plan_revision(plan),
        tool="search",
        requirement_id=requirement_id,
        requirement_key=requirement_revision(requirement),
        status=status,
        spec_ids=tuple(s.spec_id for s in sources),
        variable_sent=variable_sent,
        reason=reason,
    )
    stamped = tuple(s.model_copy(update={"searched_as": variable_sent}) for s in sources)
    save_log(context.state, load_log(context.state).with_record(record, stamped))


def _stored(context: Any) -> Plan:
    return Plan.model_validate_json(context.state[PLAN_STATE_KEY])


def test_build_translator_tools_registers_one_ungated_tool(real_schemas: list[ToolSchema]) -> None:
    # No `read_plan`: the plan travels in the `dfo_translator` tool call's own
    # `request` string (`create_translator_agent`'s module docstring), so
    # there is nothing left to fetch back out of session state. No
    # `discover_source`/`propose_operations`/`return_to_planner` either:
    # `translate` drives all three directly, in Python.
    tools = {
        tool.name: tool
        for tool in build_translator_tools(surface=FixtureSurface(schemas=real_schemas))
    }

    assert tuple(tools) == TOOL_NAMES
    assert all(tool._require_confirmation is False for tool in tools.values())


async def test_propose_operations_compiles_and_moves_the_plan_to_pending_approval(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(
        context,
        source(),
        status=DiscoveryStatus.DEFINITE,
        variable_sent="external debt service paid",
    )

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["outcome"] == "ok", result
    assert result["status"] == PlanStatus.PENDING_APPROVAL.value
    assert result["version"] == 1
    assert len(result["revision"]) == 64
    (only_source,) = result["sources"]
    assert only_source["spec_id"] == SPEC_KENYA
    # Always shown, own words or not: the workflow is the record of what was approved.
    assert only_source["searched_as"] == "external debt service paid"
    assert [s["tool"] for s in result["steps"]] == ["data-mcp.get_data", "analysis-mcp.aggregate"]

    stored = _stored(context)
    assert stored.status == PlanStatus.PENDING_APPROVAL
    assert stored.workflow is not None
    assert stored.workflow.sources[0].searched_as == "external debt service paid"
    assert stored.workflow.steps[1].bindings["frame_ref"].from_step == "debt"


async def test_propose_operations_compiles_no_source_the_turn_has_since_dropped(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    """The log's record predates the drop, so it carries no block of its own."""
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    save_dropped_seeds(context.state, [SeededSource(source=source())])

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["outcome"] != "ok", result
    assert _stored(context).status == PlanStatus.DRAFT
    assert _stored(context).workflow is None


async def test_a_definite_sources_describe_definition_reaches_the_workflow(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    surface = FixtureSurface(schemas=real_schemas, responses={"describe": _DESCRIBE_RESPONSE})
    core = build_translator_core(surface=surface)

    result = await core.propose_operations(translation=GOOD_TRANSLATION, tool_context=context)

    assert result["outcome"] == "ok", result
    stored = _stored(context)
    assert stored.workflow is not None
    (only_source,) = stored.workflow.sources
    assert only_source.definition == "External debt service a country pays to a foreign creditor."


async def test_a_missing_describe_answer_leaves_the_definition_unset_and_still_compiles(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # `_core` builds a `FixtureSurface` with no canned "describe" answer, so
    # the fetch behind `propose_operations` sees `ToolSurfaceUnavailableError`
    # -- a missing or failed describe must not fail the compile it enriches.
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["outcome"] == "ok", result
    stored = _stored(context)
    assert stored.workflow is not None
    (only_source,) = stored.workflow.sources
    assert only_source.definition is None


async def test_a_same_revision_retry_reads_the_describe_memo(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # `reduction="mean"` mismatches the plan's own `sum` aggregate, so compile
    # fails and the plan stays a `DRAFT` -- exactly the `COMPILE_FAILED`
    # shape a model retries within one revision. The describe fetch runs
    # before `compile_plan` either way, so it runs on both calls; only the
    # second call's network hit should be skipped.
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    surface = _CountingSurface(schemas=real_schemas, responses={"describe": _DESCRIBE_RESPONSE})
    core = build_translator_core(surface=surface)
    bad_translation = {
        **GOOD_TRANSLATION,
        "operations": [
            GOOD_TRANSLATION["operations"][0],
            {**GOOD_TRANSLATION["operations"][1], "reduction": "mean"},
        ],
    }

    first = await core.propose_operations(translation=bad_translation, tool_context=context)
    assert first["outcome"] != "ok", first
    assert _stored(context).status == PlanStatus.DRAFT

    second = await core.propose_operations(translation=bad_translation, tool_context=context)
    assert second["outcome"] != "ok", second

    assert surface.calls.get("describe") == 1


async def test_propose_operations_refuses_while_a_referral_is_pending(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # No translator tool settles a referral, so no action is offered.
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    save_referral(
        context.state,
        Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="annual_debt_service"),
    )

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["outcome"] == "refused"
    assert result["code"] == "referral_pending"
    assert "action" not in result
    assert _stored(context).status == PlanStatus.DRAFT


async def test_propose_operations_notes_a_source_found_under_other_words(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(
        context, source(), status=DiscoveryStatus.DEFINITE, variable_sent="debt paid abroad"
    )

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    (only_source,) = result["sources"]
    assert only_source["searched_as"] == "debt paid abroad"


async def test_a_source_discovery_never_confirmed_does_not_compile(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # Candidates were recorded, so the source is known to the log; it is still
    # not selectable, because no definite answer named it.
    context = context_holding(draft())
    _discovered(context, source(), source(spec_id=SPEC_OTHER), status=DiscoveryStatus.CANDIDATES)

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["outcome"] == "invalid_input"
    assert result["code"] == "compile_failed"
    assert [issue["code"] for issue in result["detail"]["issues"]] == [
        "source_not_discovered_for_requirement"
    ]
    assert result["action"] == {
        "call": "discover_source",
        "arguments": {"requirement_id": "annual_debt_service"},
    }
    assert _stored(context).status == PlanStatus.DRAFT


async def test_a_mismatched_operation_comes_back_as_operation_mismatch(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    translation = {
        **GOOD_TRANSLATION,
        "operations": [
            GOOD_TRANSLATION["operations"][0],
            {**GOOD_TRANSLATION["operations"][1], "reduction": "mean"},
        ],
    }

    result = await _core(real_schemas).propose_operations(
        translation=translation, tool_context=context
    )

    assert result["code"] == "compile_failed"
    assert "operation_mismatch" in {issue["code"] for issue in result["detail"]["issues"]}


async def test_a_failed_proposal_reports_every_issue_at_once(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    translation = {
        "operations": [
            {
                "kind": "retrieve",
                "id": "debt",
                "requirement_id": "annual_debt_service",
                "spec_id": SPEC_KENYA,
            },
            {
                "kind": "aggregate",
                "id": "totals",
                "requirement_id": "annual_totals",
                "input": "ghost",
                "reduction": "sum",
                "group_by": ["period"],
            },
        ],
        "terminal": ["answer"],
    }

    result = await _core(real_schemas).propose_operations(
        translation=translation, tool_context=context
    )

    assert result["code"] == "compile_failed"
    assert {issue["code"] for issue in result["detail"]["issues"]} >= {
        "dangling_input",
        "unknown_terminal",
    }


async def test_an_unmeetable_requirement_comes_back_pointing_at_the_referral(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2015))
    context = context_holding(draft(evidence_requirements=[constant, derived_requirement()]))
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert [issue["code"] for issue in result["detail"]["issues"]] == ["requirement_unmeetable"]
    assert result["action"] == {
        "call": "return_to_planner",
        "arguments": {
            "kind": "requirement_unmeetable",
            "requirement_id": "annual_debt_service",
            "reason": result["detail"]["issues"][0]["reason"],
        },
    }


async def test_a_proposal_failing_only_on_structure_carries_no_action(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    translation = {**GOOD_TRANSLATION, "terminal": ["answer"]}

    result = await _core(real_schemas).propose_operations(
        translation=translation, tool_context=context
    )

    assert result["code"] == "compile_failed"
    assert "action" not in result


@pytest.mark.parametrize(
    ("plan_overrides", "expected_code"),
    [
        ({"status": PlanStatus.PENDING_APPROVAL}, "plan_not_draft"),
        ({"evidence_requirements": []}, "invalid_plan"),
    ],
    ids=["not-a-draft", "no-requirements"],
)
async def test_propose_operations_refuses_a_plan_it_cannot_translate(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    plan_overrides: dict[str, Any],
    expected_code: str,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft(**plan_overrides))

    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=context
    )

    assert result["code"] == expected_code


async def test_propose_operations_without_a_plan_returns_an_error(
    real_schemas: list[ToolSchema],
    empty_context: Any,
) -> None:
    result = await _core(real_schemas).propose_operations(
        translation=GOOD_TRANSLATION, tool_context=empty_context
    )

    assert result["code"] == "plan_missing"


@pytest.mark.parametrize(
    "translation",
    [
        {"operations": [{"kind": "deflate", "id": "real", "input": "debt"}], "terminal": ["real"]},
        {"operations": [], "terminal": []},
        ["not", "a", "translation"],
    ],
    ids=["unknown-kind", "empty", "not-a-mapping"],
)
async def test_propose_operations_rejects_a_malformed_translation(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    translation: object,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())

    result = await _core(real_schemas).propose_operations(
        translation=translation, tool_context=context
    )

    assert result["code"] == "invalid_translation"
    assert _stored(context).status == PlanStatus.DRAFT


async def test_an_unreadable_surface_is_an_error_and_keeps_the_draft(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    core = build_translator_core(surface=_UnreadableSurface(schemas=real_schemas))

    result = await core.propose_operations(translation=GOOD_TRANSLATION, tool_context=context)

    assert result["code"] == "surface_unavailable"
    assert _stored(context).status == PlanStatus.DRAFT


def test_return_to_planner_attaches_the_candidates_discovery_recorded(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(
        context,
        source(spec_id=SPEC_OTHER),
        source(spec_id=SPEC_THIRD),
        status=DiscoveryStatus.CANDIDATES,
    )

    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.SOURCE_AMBIGUOUS,
        reason="two series report debt service for Kenya",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["outcome"] == "referred"
    referral = result["referral"]
    assert referral["kind"] == "source_ambiguous"
    assert [c["spec_id"] for c in referral["candidates"]] == [SPEC_OTHER, SPEC_THIRD]
    assert _stored(context).status == PlanStatus.DRAFT
    assert load_referral(context.state).kind is ReferralKind.SOURCE_AMBIGUOUS


def test_return_to_planner_attaches_the_candidates_for_a_kind_sent_as_a_string(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # ADK hands a tool the raw JSON value, so a model naming this kind sends
    # the string, not the member. The sibling test above passes the member and
    # so never exercised the path a real call takes: the candidates were left
    # off and `Referral` then refused the referral for carrying none, which no
    # amount of retrying by the model could fix.
    context = context_holding(draft())
    _discovered(
        context,
        source(spec_id=SPEC_OTHER),
        source(spec_id=SPEC_THIRD),
        status=DiscoveryStatus.CANDIDATES,
    )

    result = _core(real_schemas).return_to_planner(
        kind="source_ambiguous",
        reason="two series report debt service for Kenya",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["outcome"] == "referred", result
    assert [c["spec_id"] for c in result["referral"]["candidates"]] == [SPEC_OTHER, SPEC_THIRD]
    assert load_referral(context.state).kind is ReferralKind.SOURCE_AMBIGUOUS


def test_return_to_planner_attaches_no_datas_nearest_sources_and_reason(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(
        context,
        source(spec_id=SPEC_OTHER),
        status=DiscoveryStatus.NO_DATA,
        reason="no_observations",
    )

    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.NO_DATA,
        reason="no series matches the ask as stated",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["outcome"] == "referred"
    referral = result["referral"]
    assert referral["kind"] == "no_data"
    assert referral["no_data_reason"] == "no_observations"
    # Described, so the planner can read it for context; not named, so it
    # cannot be pinned without a fresh search of its own.
    (candidate,) = referral["candidates"]
    assert "spec_id" not in candidate
    assert load_referral(context.state).no_data_reason == "no_observations"


def test_return_to_planner_carries_no_candidates_when_no_data_found_no_nearest_series(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    # The nearest-series relaxation can come back empty: a `no_data` referral
    # is telling the truth then, not failing to look.
    context = context_holding(draft())
    _discovered(context, status=DiscoveryStatus.NO_DATA, reason="entity_not_resolved")

    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.NO_DATA,
        reason="the entity itself never resolved",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["outcome"] == "referred"
    assert result["referral"]["candidates"] == []
    assert result["referral"]["no_data_reason"] == "entity_not_resolved"


def test_a_kind_the_vocabulary_does_not_hold_is_refused(
    real_schemas: list[ToolSchema], draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())

    result = _core(real_schemas).return_to_planner(
        kind="source_is_a_bit_odd",
        reason="something the vocabulary has no word for",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["outcome"] == "invalid_input"
    assert result["code"] == "invalid_referral"
    assert load_referral(context.state) is None


def test_a_source_is_ambiguous_only_when_discovery_said_so(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)

    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.SOURCE_AMBIGUOUS,
        reason="r",
        requirement_id="annual_debt_service",
        tool_context=context,
    )

    assert result["code"] == "no_candidates_recorded"


def test_return_to_planner_names_a_decision_the_plan_makes(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    core = _core(real_schemas)

    ok = core.return_to_planner(
        kind=ReferralKind.NO_OPERATION_COVERS,
        reason="no operation divides one series by another",
        decision_id="measure",
        tool_context=context,
    )
    unknown = core.return_to_planner(
        kind=ReferralKind.NO_OPERATION_COVERS,
        reason="r",
        decision_id="normalisation",
        tool_context=context,
    )

    assert ok["referral"]["decision_id"] == "measure"
    assert unknown["code"] == "decision_unknown"
    assert "measure" in unknown["detail"]["known"]


def test_return_to_planner_refuses_a_requirement_the_plan_lacks(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.NO_DATA,
        reason="r",
        requirement_id="population",
        tool_context=context_holding(draft()),
    )

    assert result["code"] == "requirement_unknown"


def test_return_to_planner_refuses_a_referral_that_names_the_wrong_thing(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.NO_DATA, reason="r", tool_context=context_holding(draft())
    )

    assert result["code"] == "invalid_referral"


def test_return_to_planner_without_a_plan_returns_an_error(
    real_schemas: list[ToolSchema], empty_context: Any
) -> None:
    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, reason="r", tool_context=empty_context
    )

    assert result["code"] == "plan_missing"


def test_return_to_planner_refuses_a_plan_that_is_no_longer_a_draft(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    # A plan a successful compile moved on cannot have a referral written over it.
    context = context_holding(draft(status=PlanStatus.PENDING_APPROVAL))

    result = _core(real_schemas).return_to_planner(
        kind=ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, reason="r", tool_context=context
    )

    assert result["outcome"] == "refused"
    assert result["code"] == "plan_not_draft"
    assert load_referral(context.state) is None


def test_a_source_issue_points_at_discovery() -> None:
    issues = [
        CompileIssue(
            code=CompileIssueCode.SOURCE_UNSELECTED,
            reason="r",
            requirement_id="annual_debt_service",
        )
    ]

    action = _compile_failure_action(issues)

    assert action is not None
    assert action.call == "discover_source"
    assert action.arguments == {"requirement_id": "annual_debt_service"}


def test_an_uncovered_operation_kind_points_at_the_referral() -> None:
    issues = [CompileIssue(code=CompileIssueCode.NO_CAPABILITY, reason="r", operation_id="op")]

    action = _compile_failure_action(issues)

    assert action is not None
    assert action.call == "return_to_planner"
    assert action.arguments == {"kind": "no_operation_covers", "reason": "r"}
    # decision_id is genuine model judgement `_compile_failure_action` cannot
    # supply: named in `needs`, not guessed at or left silently missing.
    assert action.needs == ("decision_id",)


def test_a_source_issue_is_tried_before_an_unmeetable_one() -> None:
    issues = [
        CompileIssue(code=CompileIssueCode.REQUIREMENT_UNMEETABLE, reason="r", requirement_id="a"),
        CompileIssue(code=CompileIssueCode.SOURCE_UNSELECTED, reason="r", requirement_id="b"),
    ]

    action = _compile_failure_action(issues)

    assert action is not None
    assert action.call == "discover_source"
    assert action.arguments == {"requirement_id": "b"}


def test_an_uncorrelated_issue_gets_no_action() -> None:
    issues = [CompileIssue(code=CompileIssueCode.CYCLE, reason="r", operation_id="op")]

    assert _compile_failure_action(issues) is None


def test_create_translator_agent_wires_name_model_description_and_tools(
    real_schemas: list[ToolSchema],
) -> None:
    agent = create_translator_agent(
        model_name="gemini-test", surface=FixtureSurface(schemas=real_schemas)
    )

    assert isinstance(agent, LlmAgent)
    assert agent.name == "dfo_translator"
    assert agent.model == "gemini-test"
    assert "PENDING_APPROVAL" in agent.description
    assert tuple(tool.name for tool in agent.tools) == TOOL_NAMES  # type: ignore[union-attr]


def test_create_translator_agent_thinks_at_low_and_carries_thought_summaries(
    real_schemas: list[ToolSchema],
) -> None:
    agent = create_translator_agent(
        model_name="gemini-test", surface=FixtureSurface(schemas=real_schemas)
    )

    assert isinstance(agent.planner, BuiltInPlanner)
    assert agent.planner.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert agent.planner.thinking_config.include_thoughts is True


def test_create_translator_agent_composes_the_thinking_override_ahead_of_the_opener(
    real_schemas: list[ToolSchema],
) -> None:
    """A single assignment here would silently drop whichever callback lost
    the slot: `override_thinking_on_sampled_turns` always returns `None`, so
    it runs first without ending `_before_translator_model`'s own turn to
    decide the opener."""
    agent = create_translator_agent(
        model_name="gemini-test", surface=FixtureSurface(schemas=real_schemas)
    )

    assert agent.before_model_callback == [
        override_thinking_on_sampled_turns,
        _before_translator_model,
    ]


def test_create_translator_agent_takes_a_mode(real_schemas: list[ToolSchema]) -> None:
    surface = FixtureSurface(schemas=real_schemas)

    assert create_translator_agent(model_name="m", surface=surface).mode is None
    assert (
        create_translator_agent(model_name="m", surface=surface, mode="single_turn").mode
        == "single_turn"
    )


def test_create_translator_agent_closes_the_transfer_to_agent_escape_hatch(
    real_schemas: list[ToolSchema],
) -> None:
    """ADK declares a `transfer_to_agent` tool to any `LlmAgent` with a
    parent unless told not to, regardless of `mode` -- a second,
    unmonitored way for this agent's model to hand control back to the
    planner that never builds a `Referral` and never calls
    `return_to_planner`, the only response
    `harness.planner_seam.PlannerSeam._stalled_referral` inspects.

    Asserting the flags alone would not prove the escape is closed: what
    actually stops ADK from declaring `transfer_to_agent` is
    `LlmAgent._llm_flow` choosing `SingleFlow` over `AutoFlow`
    (`google.adk.agents.llm_agent`), which only happens when both transfer
    flags are set AND this agent has no sub-agents of its own -- so this
    checks the flow ADK actually builds, not just the flags that feed it.
    """
    agent = create_translator_agent(
        model_name="gemini-test", surface=FixtureSurface(schemas=real_schemas)
    )

    assert agent.disallow_transfer_to_parent is True
    assert agent.disallow_transfer_to_peers is True
    assert agent.sub_agents == []
    assert isinstance(agent._llm_flow, SingleFlow)


def test_create_translator_agent_carries_an_output_schema_admitting_compiled_or_referred(
    real_schemas: list[ToolSchema],
) -> None:
    # A plain union, not a discriminated one: ADK's own `validate_schema`
    # (`google.adk.utils._schema_utils`) does not re-validate a closing turn
    # against either half, so nothing here checks that -- only that both
    # halves are the ones `create_translator_agent`'s own module docstring
    # names, `Compiled` and `Referred`, in that order.
    agent = create_translator_agent(
        model_name="gemini-test", surface=FixtureSurface(schemas=real_schemas)
    )

    assert get_args(agent.output_schema) == (Compiled, Referred)


async def test_a_closing_turn_that_does_not_match_the_schema_hard_fails(
    real_schemas: list[ToolSchema], scripted_llm: type[Any], run_to_completion: Any
) -> None:
    """The one real behaviour change `output_schema` brings: a model that
    calls `translate` but does not also close with a schema-conforming turn
    no longer hands the planner an unhelpful sentence -- ADK's
    `process_llm_agent_output` (`google.adk.workflow._llm_agent_wrapper`)
    tries to parse that closing text against `Compiled | Referral`, and a
    plain-text turn like `ScriptedLlm`'s own default ("done") is not valid
    JSON, so it raises. `_SingleTurnAgentTool.run_async`
    (`google.adk.tools.agent_tool`) catches that and answers the caller with
    a legible `"Error running sub-agent: ..."` string instead of silently
    keeping the model's own words.

    Runs the translator as a real sub-agent of a throwaway root, through a
    real `Runner` (`scripted_llm`/`run_to_completion`, `conftest.py`) --
    `.func()` alone never reaches ADK's own schema enforcement, which lives
    entirely in the flow that runs a `single_turn` agent as a node.
    """
    translator = create_translator_agent(
        model_name="m", surface=FixtureSurface(schemas=real_schemas), mode="single_turn"
    )
    translator.model = scripted_llm(tool_name="translate", tool_args={})
    root_llm = scripted_llm(tool_name="dfo_translator", tool_args={"request": "irrelevant here"})
    root = LlmAgent(name="root", model=root_llm, sub_agents=[translator])

    await run_to_completion(agent=root, session_service=InMemorySessionService(), app_name="app")

    assert len(root_llm.captured_requests) == 2
    parts = root_llm.captured_requests[1].contents[-1].parts
    (function_response,) = [p.function_response for p in parts if p.function_response]
    assert function_response.name == "dfo_translator"
    assert "Error running sub-agent" in function_response.response["result"]
