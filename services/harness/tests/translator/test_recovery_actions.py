"""Regression: every advertised recovery `Action` is a complete, executable call.

A `return_to_planner` Action's `arguments` carries `reason`, and, for
`no_operation_covers`, `decision_id`, so a model that calls
`return_to_planner` with an Action's `arguments` as the whole call is
accepted rather than hitting an uncaught `TypeError`. This package's own
design says a tool answers with a typed problem instead of raising
(`agent.py`'s module docstring).

One test per Action-producing path drives the tool that produces it, then
replays the Action exactly as advertised: its `arguments`, plus a value for
every name `needs` lists. Each asserts the replayed call is accepted
(`outcome: "referred"`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from compiler.compile import compile_plan
from compiler.sources import DiscoveryStatus
from compiler.surface import FixtureSurface, McpSurface, ToolSchema
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Plan,
    Unit,
    plan_revision,
)
from planner.workflow import SelectedSource
from translator.agent import build_translator_core
from translator.discovery import (
    DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
    DiscoveryRecord,
    load_log,
    requirement_revision,
    save_log,
)
from translator.mechanical import TranslatorCore

DraftFactory = Callable[..., Plan]
DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]
SourceFactory = Callable[..., SelectedSource]
ContextHolding = Callable[[Plan], Any]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32

# Mirrors the shared `requirements` pair: retrieve the dataset requirement,
# total it per period for the aggregate that reads it.
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


def _core(surface: Any) -> TranslatorCore:
    return build_translator_core(surface=surface)


def _discovered(
    context: Any,
    *sources: SelectedSource,
    status: DiscoveryStatus,
    requirement_id: str = "annual_debt_service",
) -> None:
    """Seed the log as a discovery answer of `status` naming `sources` would have."""
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
    )
    save_log(context.state, load_log(context.state).with_record(record, sources))


async def _replay_action(
    core: TranslatorCore, action: dict[str, Any], context: Any, **needs_values: Any
) -> dict[str, Any]:
    """Call `return_to_planner` exactly as `action` advertises it.

    `action["arguments"]` as the call's keyword arguments, plus a value for
    every name `action["needs"]` lists -- what a model follows the Action
    hint to do, not a test reaching past it into `return_to_planner`'s own
    full signature.
    """
    kwargs = dict(action.get("arguments") or {})
    for name in action.get("needs", ()):
        kwargs[name] = needs_values[name]
    return core.return_to_planner(tool_context=context, **kwargs)


async def test_budget_exhausted_action_refers(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    # One requirement alone reaches its own per-requirement ceiling well
    # before the revision's larger total, so that is what stops it here.
    context = context_holding(draft())
    core = _core(surface)
    for _ in range(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT):
        result = await core.discover_source(
            requirement_id="annual_debt_service", tool_context=context
        )
        assert result["outcome"] == "ok"
    blocked = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert blocked["code"] == "budget_exhausted"

    referred = await _replay_action(core, blocked["action"], context)

    assert referred["outcome"] == "referred"


async def test_ask_not_understood_action_refers(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(
            evidence_requirements=[
                requirement(measure="dropped role debt service"),
                derived_requirement(),
            ]
        )
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["code"] == "ask_not_understood"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"


async def test_source_ambiguous_action_refers(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(
            evidence_requirements=[requirement(measure="aid disbursements"), derived_requirement()]
        )
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["action"]["arguments"]["kind"] == "source_ambiguous"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"
    assert len(referred["referral"]["candidates"]) == 2


async def test_no_data_action_refers(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(evidence_requirements=[requirement(measure="unicorn horns"), derived_requirement()])
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["action"]["arguments"]["kind"] == "no_data"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"


async def test_unsupported_ask_action_refers(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(evidence_requirements=[requirement(measure="unsupported ask"), derived_requirement()])
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["action"]["arguments"]["kind"] == "unsupported_ask"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"


async def test_pinned_source_not_found_action_refers(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    pinned = requirement(source_spec_id=SPEC_OTHER)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_OTHER)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["action"]["arguments"]["kind"] == "pinned_source_not_found"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"


async def test_pinned_source_not_found_action_carries_the_logs_candidates(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The fresh search is ambiguous and neither real candidate is the pinned
    # id: the referral carries both, by their real spec ids, rather than the
    # dead end a bare "not found" would leave the planner with.
    pinned = requirement(measure="aid disbursements", source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    core = _core(surface)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    assert result["action"]["arguments"]["kind"] == "pinned_source_not_found"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"
    assert referred["referral"]["kind"] == "pinned_source_not_found"
    assert {c["spec_id"] for c in referred["referral"]["candidates"]} == {SPEC_OTHER, SPEC_THIRD}


async def test_requirement_unmeetable_action_refers(
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
    core = _core(FixtureSurface(schemas=real_schemas))

    result = await core.propose_operations(translation=GOOD_TRANSLATION, tool_context=context)
    assert result["action"]["arguments"]["kind"] == "requirement_unmeetable"

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"


async def test_no_operation_covers_action_refers(
    monkeypatch: pytest.MonkeyPatch,
    real_schemas: list[ToolSchema],
    draft: DraftFactory,
    source: SourceFactory,
    context_holding: ContextHolding,
) -> None:
    # No row in the production capability table covers any operation kind,
    # so `_check_capabilities` reports `no_capability` -- the one path
    # `_compile_failure_action` cannot compute a `decision_id` for, hence
    # `needs=("decision_id",)` rather than a guessed argument.
    context = context_holding(draft())
    source_selected = source()
    _discovered(context, source_selected, status=DiscoveryStatus.DEFINITE)
    core = _core(FixtureSurface(schemas=real_schemas))

    original_compile_plan = compile_plan

    def _compile_with_no_capabilities(**kwargs: Any) -> Any:
        return original_compile_plan(capabilities=(), **kwargs)

    monkeypatch.setattr("translator.agent.compile_plan", _compile_with_no_capabilities)

    result = await core.propose_operations(translation=GOOD_TRANSLATION, tool_context=context)
    assert result["action"]["arguments"]["kind"] == "no_operation_covers"
    assert result["action"]["needs"] == ["decision_id"]

    referred = await _replay_action(core, result["action"], context, decision_id="measure")

    assert referred["outcome"] == "referred"
