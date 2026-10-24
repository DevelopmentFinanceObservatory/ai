"""Regression: every advertised recovery `Action` is a complete, executable call.

A `return_to_planner` Action's `arguments` carries `reason`, and the
requirement or repair its kind names, so a caller that calls
`return_to_planner` with an Action's `arguments` as the whole call is
accepted rather than hitting an uncaught `TypeError`. This package's own
design says a tool answers with a typed problem instead of raising
(`tool.py`'s module docstring).

One test per Action-producing path drives the tool that produces it, then
replays the Action exactly as advertised: its `arguments`, plus a value for
every name `needs` lists. Each asserts the replayed call is accepted
(`outcome: "referred"`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from compiler.compile import compile_plan
from compiler.sources import DiscoveryStatus, source_from_spec_block
from compiler.surface import FixtureSurface, McpSurface, ToolSchema
from data_mcp.digest import render_answer
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from planner.operations import Translation
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    Period,
    PinSource,
    Plan,
    RankRequirement,
    Unit,
    plan_revision,
    save_pin_sources,
)
from planner.records import GroupBy, RankAmong, RankDirection, Reduction
from planner.tool_results import ProblemCode
from planner.workflow import SelectedSource
from submission.service import _choice_not_offered
from translator.discovery import (
    DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
    DiscoveryRecord,
    _sums_its_sides_away,
    load_log,
    requirement_revision,
    save_log,
)
from translator.mechanical import TranslatorCore
from translator.referral import load_referral
from translator.tool import build_translator_core

from ..split_plan import stored_split_plan
from .conftest import _envelope, _spec

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
    # The fresh search answered `definite` with another source: the referral
    # offers it, so a re-pin to it is one the pending choice accepts.
    assert [c["spec_id"] for c in referred["referral"]["candidates"]] == [SPEC_KENYA]


async def test_pinned_source_not_found_carries_every_stale_pin_of_the_revision(
    surface: McpSurface,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Two pins a moved period left stale: one referral offers each the source
    # found in its place, so one planner round repairs both.
    first = requirement(decision_ids=["measure", "first"], source_spec_id=SPEC_OTHER)
    second = requirement(
        id="second_debt_service", decision_ids=["measure", "second"], source_spec_id=SPEC_THIRD
    )
    context = context_holding(
        draft(
            decisions=[
                decision(),
                pinning_decision(SPEC_OTHER, slot="first"),
                pinning_decision(SPEC_THIRD, slot="second"),
            ],
            evidence_requirements=[
                first,
                second,
                derived_requirement(),
                derived_requirement(id="second_totals", input_id="second_debt_service"),
            ],
        )
    )
    core = _core(surface)
    await core.discover_source(requirement_id="second_debt_service", tool_context=context)
    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)

    referred = await _replay_action(core, result["action"], context)

    referral = referred["referral"]
    assert referral["requirement_id"] == "annual_debt_service"
    assert [c["spec_id"] for c in referral["candidates"]] == [SPEC_KENYA]
    assert [
        (pin["requirement_id"], [c["spec_id"] for c in pin["candidates"]])
        for pin in referral["stale_pins"]
    ] == [("second_debt_service", [SPEC_KENYA])]


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


# The France to Africa turn: the pin (fb3c) is France's aid to Africa, its
# recipients closed to the group; the fresh search, sent with `across:
# recipient` and no Africa, answers France's aid to all 266 recipients.
SPEC_AFRICA = "spec_fb3cc27e" + "0" * 24
SPEC_EVERY_RECIPIENT = "spec_4c330b1c" + "0" * 24
_FRANCE = {"role": "origin", "label": "France", "ref": "country/FRA"}
_WINDOW = {"kind": "exact", "has_data": True, "window": {"start_year": 2015, "end_year": 2024}}


def _every_recipient_server() -> FastMCP:
    """A `data-mcp` whose `search` answers every ask with France's aid to 266 recipients."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        """Find what data exists for the ask."""
        del parts, raw_query
        spec = _spec(
            SPEC_EVERY_RECIPIENT,
            "France's gross disbursements to 266 recipients",
            coverage=_WINDOW,
            entities=[_FRANCE],
            role_axis_facts=[
                {"role": "destination", "member_count": 266, "members_available": True}
            ],
        )
        envelope = _envelope("definite", [], spec=spec)
        return ToolResult(
            content="digest",
            structured_content=render_answer(envelope),
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )

    return server


@pytest.mark.parametrize("held_by", ["shown", "found"])
@pytest.mark.parametrize("named_source", [None, "IDS"])
async def test_a_source_dropping_the_pinned_group_is_withheld_and_cannot_be_pinned(
    held_by: str,
    named_source: str | None,
    draft: DraftFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    africa = _spec(
        SPEC_AFRICA,
        "France's gross disbursements to the 59 countries and territories in Africa",
        coverage=_WINDOW,
        entities=[
            _FRANCE,
            {"role": "destination", "label": "Ghana", "ref": "country/GHA"},
            {"role": "destination", "label": "Kenya", "ref": "country/KEN"},
        ],
        entity_groups={"destination": {"label": "Africa", "ref": "africa"}},
    )
    france = [Entity(name="France", role="donor"), Entity(name="Germany", role="donor")]
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_AFRICA)],
            evidence_requirements=[
                requirement(
                    measure="official development assistance",
                    entities=france,
                    across=("recipient",),
                    source_spec_id=SPEC_AFRICA,
                    source=named_source,
                ),
                derived_requirement(),
            ],
        )
    )
    if held_by == "shown":
        save_pin_sources(context.state, {SPEC_AFRICA: PinSource(block=africa, settled=True)})
    else:
        # The G7 turn: an earlier proposal, naming Africa and stating no
        # `across`, had the translator find the pin's source itself.
        found = DiscoveryRecord(
            revision="earlier",
            tool="search",
            requirement_id="annual_debt_service",
            requirement_key="earlier",
            status=DiscoveryStatus.DEFINITE,
            spec_ids=(SPEC_AFRICA,),
        )
        log = load_log(context.state).with_record(found, (source_from_spec_block(africa),))
        save_log(context.state, log)
    core = _core(McpSurface(sources={"data-mcp": _every_recipient_server()}))

    result = await core.discover_source(requirement_id="annual_debt_service", tool_context=context)
    referred = await _replay_action(core, result["action"], context)

    referral = referred["referral"]
    assert (referral["kind"], referral["candidates"]) == ("pinned_source_not_found", [])
    (withheld,) = referral["withheld"]
    assert "spec_id" not in withheld
    assert withheld["sentence"] == "France's gross disbursements to 266 recipients"
    (dropped,) = withheld["dropped"]
    assert "Africa" in dropped and "266" in dropped
    repair = "name 'Africa' as recipient in `entities` and drop 'recipient' from `across`"
    assert referral["repair"] == repair
    assert f"none keeps it, so {repair}" in referral["reason"]
    stored = load_referral(context.state)
    assert stored is not None
    assert stored.offered() == {"annual_debt_service": ()}

    repinned = draft(
        decisions=[pinning_decision(SPEC_EVERY_RECIPIENT)],
        evidence_requirements=[
            requirement(
                measure="official development assistance",
                entities=france,
                across=("recipient",),
                source_spec_id=SPEC_EVERY_RECIPIENT,
            ),
            derived_requirement(),
        ],
    )
    refused = _choice_not_offered(repinned, load_referral(context.state), context)

    assert refused is not None
    assert refused["code"] == ProblemCode.SOURCE_NOT_OFFERED.value

    # Unpinned, asking `search` for what it asked before, the requirement would
    # have the fresh search settle on the withheld source, whatever order it
    # names its entities in and whichever word it uses for the open role; the
    # repair passes.
    def unpinned(
        entities: list[Entity], across: str | None, source: str | None = named_source
    ) -> Plan:
        return draft(
            evidence_requirements=[
                requirement(
                    measure="official development assistance",
                    entities=entities,
                    across=() if across is None else (across,),
                    source=source,
                ),
                derived_requirement(),
            ]
        )

    for entities, across in [
        (france, "recipient"),
        (france[::-1], "recipient"),
        (france, "recipients"),
    ]:
        dropped_by_unpinning = _choice_not_offered(
            unpinned(entities, across), load_referral(context.state), context
        )
        assert dropped_by_unpinning is not None, (entities, across)
        assert dropped_by_unpinning["code"] == ProblemCode.SOURCE_NOT_OFFERED.value
        assert dropped_by_unpinning["detail"]["repair"] == repair
    if named_source is not None:
        for variant in ["  ids  ", "the IDS", "THE (IDS)."]:
            assert (
                _choice_not_offered(
                    unpinned(france, "recipient", variant), load_referral(context.state), context
                )
                is not None
            )
    # A different named table changes the search too, so the translator may
    # try it rather than rejecting the proposal before discovery runs.
    assert (
        _choice_not_offered(
            unpinned(france, "recipient", "DAC2a"), load_referral(context.state), context
        )
        is None
    )
    africa_named = [*france, Entity(name="Africa", role="recipient")]
    assert (
        _choice_not_offered(unpinned(africa_named, None), load_referral(context.state), context)
        is None
    )


# The planner-entry eval's recorded Kenya answers: the DAC official-donors
# total Search binds when no donor is named, and the 143 providers it leaves
# open when the donor side is asked for open.
_KENYA_ODA = json.loads(
    (
        Path(__file__).parents[1] / "fixtures" / "search_spec_dac_total_and_donor_open.json"
    ).read_text()
)


def _donor_open_server() -> FastMCP:
    """A `data-mcp` whose `search` answers every ask with ODA to Kenya from 143 providers."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        """Find what data exists for the ask."""
        del parts, raw_query
        envelope = _envelope("definite", [], spec=_KENYA_ODA["donor_open"])
        return ToolResult(
            content="digest",
            structured_content=render_answer(envelope),
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )

    return server


_TOP_DONORS = RankRequirement(
    id="top_donors",
    decision_ids=["every_donor"],
    input_id="oda",
    direction=RankDirection.HIGHEST_FIRST,
    among=RankAmong.POPULATION,
    axis="donor",
    reduction=Reduction.SUM,
    limit=10,
    window_start=2022,
    window_end=2022,
)
_TOTAL = AggregateRequirement(
    id="total",
    decision_ids=["measure"],
    input_id="oda",
    reduction=Reduction.SUM,
    group_by=(GroupBy.PERIOD,),
)
_NAME_THE_TOTAL = (
    "name 'official donors (DAC)' as donor in `entities` and drop 'donor' from `across`"
)


@pytest.mark.parametrize(
    ("reader", "repair"),
    [(_TOP_DONORS, None), (_TOTAL, _NAME_THE_TOTAL)],
    ids=["rank", "total"],
)
async def test_a_pin_closing_the_side_a_reader_reads_per_member_gives_way_to_the_open_source(
    reader: EvidenceRequirement,
    repair: str | None,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The pin is the DAC official-donors total and `across` asks for every
    # donor. A ranking reads each donor, so the donor-open source search finds
    # is offered and the pin gives way to it: naming the total would leave one
    # donor to rank. A total over donors sums the side away, so the pin's
    # scope is the figure asked for, the open source is withheld, and the
    # repair names the total.
    dac_total = _KENYA_ODA["dac_total"]["spec_id"]
    donor_open = _KENYA_ODA["donor_open"]["spec_id"]

    def oda(**overrides: Any) -> DatasetRequirement:
        return requirement(
            id="oda",
            measure="official development assistance",
            entities=[Entity(name="Kenya", role="recipient")],
            across=("donor",),
            period=Period(start=2015, end=2022),
            **overrides,
        )

    every_donor = (
        [decision(slot="every_donor", fixes=DecisionFixes(among=RankAmong.POPULATION))]
        if isinstance(reader, RankRequirement)
        else []
    )
    context = context_holding(
        draft(
            decisions=[pinning_decision(dac_total), *every_donor],
            evidence_requirements=[oda(source_spec_id=dac_total), reader],
        )
    )
    save_pin_sources(
        context.state, {dac_total: PinSource(block=_KENYA_ODA["dac_total"], settled=True)}
    )
    core = _core(McpSurface(sources={"data-mcp": _donor_open_server()}))

    result = await core.discover_source(requirement_id="oda", tool_context=context)
    referred = await _replay_action(core, result["action"], context)

    referral = referred["referral"]
    assert referral["kind"] == "pinned_source_not_found"
    assert referral.get("repair") == repair
    offered = [] if repair else [donor_open]
    assert [candidate["spec_id"] for candidate in referral["candidates"]] == offered
    assert len(referral.get("withheld", [])) == (1 if repair else 0)
    repinned = draft(
        decisions=[pinning_decision(donor_open), *every_donor],
        evidence_requirements=[oda(source_spec_id=donor_open), reader],
    )
    refused = _choice_not_offered(repinned, load_referral(context.state), context)
    assert (refused is None) == (repair is None)


@pytest.mark.parametrize("requirement_id", ["debt", "debt_history"])
def test_a_copy_of_a_whole_history_is_judged_by_its_origins_readers(
    decision: DecisionFactory, requirement_id: str
) -> None:
    # `last_five`, a growth, reads `debt_history`, the copy the split added;
    # the plan as the model wrote it has `last_five` read `debt`.
    assert _sums_its_sides_away(stored_split_plan(decision), requirement_id) is False


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
    # A source in current prices cannot serve a requirement in constant ones.
    _discovered(context, source(unit_ref="SDG_CU_USD"), status=DiscoveryStatus.DEFINITE)
    core = _core(FixtureSurface(schemas=real_schemas))

    result = await core.propose_operations(
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
    )
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
    # No row in the capability table covers any operation kind, so
    # `_check_capabilities` reports `no_capability`, a capability gap the
    # action refers as unsupported, naming the requirement.
    context = context_holding(draft())
    source_selected = source()
    _discovered(context, source_selected, status=DiscoveryStatus.DEFINITE)
    core = _core(FixtureSurface(schemas=real_schemas))

    original_compile_plan = compile_plan

    def _compile_with_no_capabilities(**kwargs: Any) -> Any:
        return original_compile_plan(capabilities=(), **kwargs)

    monkeypatch.setattr("translator.tool.compile_plan", _compile_with_no_capabilities)

    result = await core.propose_operations(
        translation=Translation.model_validate(GOOD_TRANSLATION), tool_context=context
    )
    assert result["action"]["arguments"]["kind"] == "no_operation_covers"
    assert "needs" not in result["action"]

    referred = await _replay_action(core, result["action"], context)

    assert referred["outcome"] == "referred"
    assert referred["referral"]["outcome"] == "unsupported"
