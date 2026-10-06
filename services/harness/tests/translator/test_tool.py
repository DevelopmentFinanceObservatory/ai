"""Tests for `translator.tool`: proposing, referring, and the `translate` tool.

`propose_operations` and `return_to_planner` are exercised directly, through
`translator.tool.build_translator_core`, against a fake tool context: they
are not model-facing tools in their own right, but the same functions
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
from typing import Any

import pytest
import translator.tool
from compiler.compile import CompileIssue, CompileIssueCode
from compiler.sources import DiscoveryStatus
from compiler.surface import (
    FixtureSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from planner.operations import Translation
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Plan,
    PlanStatus,
    SeededSource,
    Unit,
    UnsupportedOperation,
    load_unsupported,
    plan_revision,
    save_dropped_seeds,
)
from planner.tool_results import Outcome, referred
from planner.workflow import SelectedSource, ToolId
from translator.discovery import DiscoveryRecord, load_log, requirement_revision, save_log
from translator.mechanical import TranslatorCore
from translator.referral import Referral, ReferralKind, load_referral, save_referral
from translator.tool import (
    TRANSLATE_TOOL_NAME,
    IssueResolution,
    _compile_failure_action,
    _referral_payload,
    build_translate_tool,
    build_translator_core,
    issue_resolution,
)

DraftFactory = Callable[..., Plan]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]
SourceFactory = Callable[..., SelectedSource]

# Fixtures from this suite's conftest, annotated structurally.
ContextHolding = Callable[[Plan], Any]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32

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


# A source whose identity pins `DebtServicePaid`, the one StatVar `describe` names.
_PINNED = {"identity": {"shape": {"kind": "stat_var", "id": "DebtServicePaid"}}}
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


def test_translate_is_one_ungated_tool_taking_no_arguments(real_schemas: list[ToolSchema]) -> None:
    # The plan is read from session state, so the model passes nothing. No
    # `discover_source`/`propose_operations`/`return_to_planner` tool either:
    # `translate` drives all three directly, in Python.
    tool = build_translate_tool(surface=FixtureSurface(schemas=real_schemas))
    declaration = tool._get_declaration()

    assert tool.name == TRANSLATE_TOOL_NAME == "translate"
    assert tool._require_confirmation is False
    assert declaration is not None
    assert (declaration.parameters, declaration.parameters_json_schema) == (None, None)


@pytest.mark.parametrize(
    ("kind", "recorded"),
    [
        (ReferralKind.NO_OPERATION_COVERS, {UnsupportedOperation.NO_OPERATION_COVERS}),
        (ReferralKind.UNSUPPORTED_ASK, {UnsupportedOperation.UNSUPPORTED_ASK}),
        (ReferralKind.NO_DATA, set()),
    ],
)
async def test_translate_records_an_unsupported_referral_the_plan_must_decline(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
    monkeypatch: pytest.MonkeyPatch,
    kind: ReferralKind,
    recorded: set[UnsupportedOperation],
) -> None:
    referral = Referral(kind=kind, reason="r", requirement_id="annual_debt_service")

    async def run_translate(tool_context: Any, *, core: TranslatorCore) -> dict[str, object]:
        return referred(_referral_payload(referral))

    monkeypatch.setattr(translator.tool, "run_translate", run_translate)
    context = context_holding(draft())

    answer = await build_translate_tool(surface=FixtureSurface(schemas=real_schemas)).func(
        tool_context=context
    )

    assert answer["kind"] == kind.value
    assert load_unsupported(context.state) == recorded


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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
    _discovered(context, source(**_PINNED), status=DiscoveryStatus.DEFINITE)
    surface = FixtureSurface(schemas=real_schemas, responses={"describe": _DESCRIBE_RESPONSE})
    core = build_translator_core(surface=surface)

    result = await core.propose_operations(
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
    )

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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
    _discovered(context, source(**_PINNED), status=DiscoveryStatus.DEFINITE)
    surface = _CountingSurface(schemas=real_schemas, responses={"describe": _DESCRIBE_RESPONSE})
    core = build_translator_core(surface=surface)
    bad_translation = {
        **GOOD_TRANSLATION,
        "operations": [
            GOOD_TRANSLATION["operations"][0],
            {**GOOD_TRANSLATION["operations"][1], "reduction": "mean"},
        ],
    }

    first = await core.propose_operations(
        translation=Translation.model_validate(bad_translation), tool_context=context
    )
    assert first["outcome"] != "ok", first
    assert _stored(context).status == PlanStatus.DRAFT

    second = await core.propose_operations(
        translation=Translation.model_validate(bad_translation), tool_context=context
    )
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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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
        translation=Translation.model_validate(translation), tool_context=context
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
        translation=Translation.model_validate(translation), tool_context=context
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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
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


async def test_a_proposal_failing_only_on_structure_closes_as_an_internal_failure(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    translation = {**GOOD_TRANSLATION, "terminal": ["answer"]}

    result = await _core(real_schemas).propose_operations(
        translation=Translation.model_validate(translation), tool_context=context
    )

    assert result["code"] == "compile_failed"
    assert result["action"]["arguments"]["kind"] == "internal_failure"


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
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
    )

    assert result["code"] == expected_code


async def test_propose_operations_without_a_plan_returns_an_error(
    real_schemas: list[ToolSchema],
    empty_context: Any,
) -> None:
    result = await _core(real_schemas).propose_operations(
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=empty_context
    )

    assert result["code"] == "plan_missing"


async def test_an_unreadable_surface_is_an_error_and_keeps_the_draft(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    _discovered(context, source(), status=DiscoveryStatus.DEFINITE)
    core = build_translator_core(surface=_UnreadableSurface(schemas=real_schemas))

    result = await core.propose_operations(
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
    )

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


def test_return_to_planner_logs_an_internal_failure_under_the_reference_it_carries(
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    context_holding: ContextHolding,
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = context_holding(draft())

    with caplog.at_level("ERROR", logger="translator.tool"):
        result = _core(real_schemas).return_to_planner(
            kind=ReferralKind.INTERNAL_FAILURE, reason="cycle: r", tool_context=context
        )

    referral = result["referral"]
    assert referral["outcome"] == "internal"
    assert referral["reference"] in caplog.text
    assert "cycle: r" in caplog.text
    assert load_referral(context.state).reference == referral["reference"]


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

    action = _compile_failure_action(issues, requirement_of={})

    assert action is not None
    assert action.call == "discover_source"
    assert action.arguments == {"requirement_id": "annual_debt_service"}


@pytest.mark.parametrize(
    "code", [CompileIssueCode.NO_CAPABILITY, CompileIssueCode.FRAME_UNIT_UNBOUND]
)
def test_an_uncovered_operation_refers_its_requirement_as_unsupported_before_a_repair(
    code: CompileIssueCode,
) -> None:
    issues = [
        CompileIssue(code=CompileIssueCode.FRAME_UNITS_DIFFER, reason="repair", operation_id="op"),
        CompileIssue(code=code, reason="r", operation_id="op", requirement_id="side"),
    ]

    action = _compile_failure_action(issues, requirement_of={"op": "req"})

    assert action.call == "return_to_planner"
    assert action.arguments == {
        "kind": "no_operation_covers",
        "requirement_id": "req",
        "reason": "r",
    }
    assert action.needs == ()


@pytest.mark.parametrize("code", list(CompileIssueCode))
def test_every_compile_issue_code_closes_on_one_well_formed_outcome(
    code: CompileIssueCode,
) -> None:
    """Each code resolves, alone, to a discovery retry or to a referral that
    validates with one of the closed outcomes; a repair carries its permitted
    change."""
    issue = CompileIssue(code=code, reason="r", operation_id="op", requirement_id="req")

    action = _compile_failure_action([issue], requirement_of={})

    if issue_resolution(code) is IssueResolution.DISCOVER_AGAIN:
        assert action.call == "discover_source"
        return
    arguments = dict(action.arguments or {})
    if arguments["kind"] == ReferralKind.INTERNAL_FAILURE:
        arguments["reference"] = "0123456789ab"
    referral = Referral.model_validate(arguments)
    expected = {
        IssueResolution.INTERNAL: Outcome.INTERNAL,
        IssueResolution.UNSUPPORTED: Outcome.UNSUPPORTED,
        IssueResolution.UNMEETABLE: Outcome.INVALID_INPUT,
        IssueResolution.REPAIR: Outcome.INVALID_INPUT,
    }
    assert referral.outcome is expected[issue_resolution(code)]
    assert (referral.kind is ReferralKind.REQUIREMENT_NEEDS_REPAIR) == (
        issue_resolution(code) is IssueResolution.REPAIR
    )


def test_a_source_issue_is_tried_before_an_unmeetable_one() -> None:
    issues = [
        CompileIssue(code=CompileIssueCode.REQUIREMENT_UNMEETABLE, reason="r", requirement_id="a"),
        CompileIssue(code=CompileIssueCode.SOURCE_UNSELECTED, reason="r", requirement_id="b"),
    ]

    action = _compile_failure_action(issues, requirement_of={})

    assert action is not None
    assert action.call == "discover_source"
    assert action.arguments == {"requirement_id": "b"}


def test_a_structural_issue_closes_as_an_internal_failure_before_a_repair() -> None:
    issues = [
        CompileIssue(code=CompileIssueCode.ENTITY_REQUIRED, reason="r", operation_id="cmp"),
        CompileIssue(code=CompileIssueCode.CYCLE, reason="loop", operation_id="op"),
    ]

    action = _compile_failure_action(issues, requirement_of={})

    assert action.arguments == {"kind": "internal_failure", "reason": "cycle: loop"}
