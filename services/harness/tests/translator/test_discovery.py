"""Tests for `translator.discovery`: bounded per revision, built from the requirement.

The tools are exercised through the MCP protocol against the canned
`data-mcp` server, so what is asserted is what a real server would see.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import DiscoveryStatus, pinned_stat_var, source_from_spec_block
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSurface,
    ToolSurfaceUnavailableError,
)
from data_mcp import resolution
from data_mcp.definition_store import DefinitionStoreUnavailable, InMemoryDefinitionStore
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.definition_writes import windowed_identity
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import CodeNode
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk.tools.function_tool import FunctionTool
from harness.turn_records import RecordedIdentity
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    FirstYear,
    GrowthRequirement,
    PinSource,
    Plan,
    PlanStatus,
    Unit,
    WindowedPin,
    current_discovery_time_left,
    current_earlier_block,
    current_recorded_identity,
    pin_sources,
    plan_revision,
    save_pin_sources,
    save_windowed_pins,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    Period,
    RelativePeriod,
    ResolvedSourceMention,
    RollingWindow,
    StatVarReading,
)
from planner.tool_results import ProblemCode
from planner.workflow import OpenEntityAxis, ResolvedEntity, SelectedSource, ToolId
from translator.discovery import (
    DISCOVERY_CALL_BUDGET,
    DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
    DISCOVERY_STATE_KEY,
    CutCalls,
    DefinitionNotResolvedError,
    DiscoveryLog,
    DiscoveryRecord,
    PartFault,
    PartNotResolvedError,
    build_discovery_tools,
    compact_source,
    current_cut_calls,
    described_sources,
    fetch_code_paths,
    fetch_definitions,
    load_log,
    requirement_revision,
    resolve_definition,
    resolve_entity_refs,
    resolved_spec,
    save_log,
)
from translator.referral import ReferralKind

from ..data_mcp_definitions import DataMcpDefinitions, GroupsSearchStub
from .conftest import _spec

DraftFactory = Callable[..., Plan]
DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]

# Fixtures from this suite's conftest, annotated structurally.
ContextHolding = Callable[[Plan], Any]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32

# Search's own answers for one dataset over three periods, keyed by the years.
_KENYA_OVER: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "resubmit_education_oda_kenya.json").read_text()
)["specs"]


def _tools(surface: ToolSurface) -> dict[str, FunctionTool]:
    return {tool.name: tool for tool in build_discovery_tools(surface=surface)}


def _revision(context: Any) -> str:
    return plan_revision(Plan.model_validate_json(context.state[PLAN_STATE_KEY]))


def _requirement_keys(context: Any, *requirement_ids: str) -> dict[str, str]:
    """Each of `requirement_ids`' own `requirement_revision`, as `selectable` takes it."""
    plan = Plan.model_validate_json(context.state[PLAN_STATE_KEY])
    keys: dict[str, str] = {}
    for requirement_id in requirement_ids:
        requirement = plan.requirement(requirement_id)
        assert isinstance(requirement, DatasetRequirement)
        keys[requirement_id] = requirement_revision(requirement)
    return keys


def _requirement_key(context: Any, requirement_id: str = "annual_debt_service") -> str:
    return _requirement_keys(context, requirement_id)[requirement_id]


async def _discover(
    surface: ToolSurface,
    context: Any,
    requirement_id: str = "annual_debt_service",
    **args: Any,
) -> dict[str, Any]:
    return await _tools(surface)["discover_source"].func(
        requirement_id=requirement_id, tool_context=context, **args
    )


def _apply(log: DiscoveryLog, pairs: Any) -> DiscoveryLog:
    """Apply each `fetch_definitions` pair onto `log`, the shape a caller applies under the lock."""
    for record, sources in pairs:
        log = log.with_record(record, sources)
    return log


class _CountingSurface:
    """Wraps a `ToolSurface`, counting each tool it was asked to call.

    Proves a memoized `fetch_definitions` skips the network on a
    same-`requirement_key` retry, rather than only checking the result.
    """

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface
        self.calls: dict[str, int] = {}

    async def list_tools(self) -> list[Any]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: Any,
        arguments: dict[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> Any:
        self.calls[tool.name] = self.calls.get(tool.name, 0) + 1
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


def test_the_discovery_tools_wrap_search_and_nothing_else(surface: McpSurface) -> None:
    # `describe` is not a model tool any more: `fetch_definitions` calls it
    # from code. `get_data` and the verbs are unreachable by construction:
    # no wrapper, no call.
    assert set(_tools(surface)) == {"discover_source"}


async def test_a_definite_answer_names_one_source_the_compiler_may_use(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())

    result = await _discover(surface, context)

    assert result["outcome"] == "ok"
    assert result["status"] == "definite"
    (found,) = result["sources"]
    assert found["spec_id"] == SPEC_KENYA
    assert found["searched_as"] == "external debt service paid"
    assert [e["label"] for e in found["entities"]] == ["Kenya"]
    assert found["coverage_window"] == {"start": 2010, "end": 2024}
    assert found["unit"] == "US dollars"
    assert result["warnings"] == []
    assert result["budget_remaining"] == DISCOVERY_CALL_BUDGET - 1
    assert "action" not in result
    keys = _requirement_keys(context, "annual_debt_service")
    assert list(load_log(context.state).selectable(requirement_keys=keys, pinned={})) == [
        "annual_debt_service"
    ]


async def test_the_ask_reaches_search_in_the_requirements_own_fields(
    surface: McpSurface,
    draft: DraftFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"] == [
        {
            "variable": "external debt service paid",
            "entities": [{"text": "Kenya", "role": "debtor"}],
            "time": "2015 to 2024",
            "unit": "current USD",
        }
    ]


async def test_a_named_source_reaches_search_as_an_included_source_and_moves_the_revision(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
) -> None:
    named = requirement(source="IDS")
    context = context_holding(draft(evidence_requirements=[named, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"][0]["sources"] == [{"text": "IDS"}]
    assert requirement_revision(named) != requirement_revision(requirement())


async def test_an_open_period_reaches_search_from_its_first_year_on(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
) -> None:
    # "since 2015" is the form Search binds as a range with no end year, so
    # the definition it finds reaches years published after the plan.
    opened = requirement(period_end_open=True)
    context = context_holding(draft(evidence_requirements=[opened, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"][0]["time"] == "since 2015"


@pytest.mark.parametrize(
    ("role", "sent"),
    [("origin", {"role": "donor"}), ("Lenders", {"role": "creditor"}), ("country", {})],
    ids=["axis-name", "plural-synonym", "place-noun"],
)
async def test_a_stored_role_reaches_search_as_the_party_word_it_stands_for(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
    role: str,
    sent: dict[str, str],
) -> None:
    # `propose_plan` holds a model to the party words; a plan read back from
    # session state may still carry a word `search`'s lexicon refuses.
    stored = requirement(entities=[Entity(name="Kenya", role=role)])
    context = context_holding(draft(evidence_requirements=[stored, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"][0]["entities"] == [{"text": "Kenya", **sent}]


@pytest.mark.parametrize(
    ("entity", "sent"),
    [
        (Entity(name="Africa"), {"text": "Africa"}),
        (Entity(name="Africa", members=True), {"text": "Africa", "members": True}),
        (
            Entity(name="G7", role="donor", members=True),
            {"text": "G7", "role": "donor", "members": True},
        ),
    ],
    ids=["set-alone", "set-members", "party-set-members"],
)
async def test_a_set_reaches_search_as_its_members_only_when_the_plan_asks_for_them(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
    entity: Entity,
    sent: dict[str, object],
) -> None:
    # "Africa" alone is the region's own figure, which search binds to its
    # published total; "African countries" asks for each member.
    named = requirement(entities=[entity])
    context = context_holding(draft(evidence_requirements=[named, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"][0]["entities"] == [sent]


async def test_the_unit_sent_carries_the_price_basis_and_base_year(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
) -> None:
    # "USD" alone would let a current-price source answer a constant-price ask.
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2015))
    context = context_holding(draft(evidence_requirements=[constant, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    assert call["parts"][0]["unit"] == "constant 2015 USD"


_KENYA_DEBTOR = Entity(name="Kenya", role="debtor")


@pytest.mark.parametrize(
    ("entities", "across", "open_roles"),
    [
        ([_KENYA_DEBTOR], ("creditor",), ["creditor"]),
        ([], ("creditor",), ["creditor"]),
        ([_KENYA_DEBTOR], ("lenders",), ["creditor"]),
        ([Entity(name="Kenya")], ("donor",), ["donor"]),
        ([_KENYA_DEBTOR], ("countries",), None),
        ([_KENYA_DEBTOR], ("bilateral creditor",), None),
        ([], ("donors", "recipient"), ["donor", "recipient"]),
        ([_KENYA_DEBTOR], ("creditor", "country"), ["creditor"]),
    ],
    ids=[
        "beside-entity",
        "alone",
        "plural-synonym",
        "unroled-entity",
        "place",
        "narrower",
        "both-sides",
        "beside-the-place",
    ],
)
async def test_a_flow_party_stated_in_across_reaches_search_as_an_open_role(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
    entities: list[Entity],
    across: tuple[str, ...],
    open_roles: list[str] | None,
) -> None:
    # Search elects a source broken down by the open role over its total, so
    # `across` asks for every member of the side it names. The place, or a
    # population narrower than a party, is no role, and the entities go as named.
    # The plan answers with the series, which a total across places would refuse.
    stated = requirement(entities=entities, across=across)
    context = context_holding(draft(evidence_requirements=[stated]))

    await _discover(surface, context)

    (call,) = search_calls
    (part,) = call["parts"]
    assert part.get("open_roles") == open_roles
    assert part.get("entities") == (
        [
            {"text": entity.name, **({"role": entity.role} if entity.role else {})}
            for entity in entities
        ]
        or None
    )


async def test_a_source_dumped_without_its_open_axes_is_not_selectable_for_across(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # A session discovers the requirement, and its log is stored by a
    # release whose `SelectedSource` had no `open_entity_axes`. The same
    # requirement then states `across`: the stored source restores with its
    # axes unread, stays selectable for a requirement that names its
    # entities, and is withheld from the one that states `across` until a
    # fresh discovery reads them.
    context = context_holding(draft())
    await _discover(surface, context)
    stored = context.state[DISCOVERY_STATE_KEY]["sources"]["annual_debt_service"][SPEC_KENYA]
    del stored["open_entity_axes"]
    keys = _requirement_keys(context, "annual_debt_service")
    across = requirement(across=("bilateral creditor",))
    assert requirement_revision(across) == keys["annual_debt_service"]

    restored = load_log(context.state)

    assert restored.sources["annual_debt_service"][SPEC_KENYA].open_entity_axes is None
    assert list(restored.selectable(requirement_keys=keys, pinned={})) == ["annual_debt_service"]
    assert restored.selectable(requirement_keys=keys, pinned={}, across={across.id}) == {}

    context.state[PLAN_STATE_KEY] = draft(
        evidence_requirements=[across, derived_requirement()]
    ).model_dump_json()
    await _discover(surface, context)

    rediscovered = load_log(context.state).selectable(
        requirement_keys=keys, pinned={}, across={across.id}
    )
    assert rediscovered[across.id].open_entity_axes == ()


def test_a_source_shows_each_open_axis_with_its_member_count() -> None:
    source = SelectedSource(
        spec_id="spec_" + "c" * 32,
        sentence="education aid to Kenya",
        has_data=True,
        open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
    )

    assert compact_source(source)["open_entity_axes"] == [{"column": "origin", "members": 55}]


@pytest.mark.parametrize(
    ("stat_vars", "measure_count"),
    [
        # A settled family: the members the mint serves.
        ({"member_count": 96, "member_count_exact": True, "served_count": 96}, "96 StatVars"),
        # A recall sample: the members Search confirmed, a floor on the family.
        (
            {"member_count": 3, "member_count_exact": False, "served_count": None},
            "at least 3 StatVars",
        ),
        # A served count is exact, whatever the confirmed count.
        ({"member_count": 3, "member_count_exact": False, "served_count": 5}, "5 StatVars"),
    ],
    ids=["settled-family", "recall-sample", "served-over-a-floor"],
)
def test_a_source_names_its_measure_counts_its_stat_vars_and_names_none(
    stat_vars: dict[str, Any], measure_count: str
) -> None:
    """The description counts the StatVars and names the axes they vary along,
    and no StatVar."""
    block = copy.deepcopy(_KENYA_OVER["2010-2023"])
    block["resolved"]["stat_vars"].update(stat_vars)
    source = source_from_spec_block(block)

    described = compact_source(source)

    assert described["measure"] == block["resolved"]["shape"]["label"]
    assert described["measure_count"] == measure_count
    assert described["varies_by"] == [axis.property for axis in source.open_axes]
    assert described["varies_by"]


@pytest.mark.parametrize("open_entity_axes", [(), None])
def test_a_source_with_no_open_axis_states_no_open_entity_axes(
    open_entity_axes: tuple[()] | None,
) -> None:
    source = SelectedSource(
        spec_id="spec_" + "c" * 32,
        sentence="education aid to Kenya",
        has_data=True,
        open_entity_axes=open_entity_axes,
    )

    assert "open_entity_axes" not in compact_source(source)


async def test_a_derived_requirement_id_is_not_a_dataset(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())

    result = await _discover(surface, context, requirement_id="annual_totals")

    assert result["outcome"] == "invalid_input"
    assert result["code"] == "requirement_not_a_dataset"


@pytest.mark.parametrize(
    ("measure", "dropped"),
    [
        ("dropped role debt service", {"role_text_not_recognised": ["debtor"]}),
        ("dropped time debt service", {"time_text_not_parsed": ["2015 to 2024"]}),
    ],
    ids=["role", "time"],
)
async def test_an_answer_search_reached_by_dropping_a_phrase_is_refused(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    measure: str,
    dropped: dict[str, list[str]],
) -> None:
    # The server found a source, but for a narrower ask than the requirement
    # states: the phrase it did not recognise was left out of the search.
    context = context_holding(
        draft(evidence_requirements=[requirement(measure=measure), derived_requirement()])
    )

    result = await _discover(surface, context)

    assert result["outcome"] == "refused"
    assert result["code"] == "ask_not_understood"
    assert result["detail"]["dropped"] == dropped
    assert result["action"] == {
        "call": "return_to_planner",
        "arguments": {
            "kind": "ask_not_understood",
            "requirement_id": "annual_debt_service",
            "reason": result["message"],
        },
    }
    log = load_log(context.state)
    (record,) = log.records
    assert record.status is DiscoveryStatus.DEFINITE
    assert record.blocked_by == tuple(dropped)
    assert [w.code for w in record.warnings] == list(dropped)
    keys = _requirement_keys(context, "annual_debt_service")
    assert log.selectable(requirement_keys=keys, pinned={}) == {}
    assert result["detail"]["budget_remaining"] == DISCOVERY_CALL_BUDGET - 1


async def test_a_pin_cannot_resurrect_an_answer_search_reached_by_dropping_a_phrase(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    pinned = requirement(measure="dropped role debt service", source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["code"] == "ask_not_understood"
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    assert log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_KENYA}) == {}


async def test_an_advisory_warning_rides_out_with_the_answer(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # A default the server fell back to is worth knowing, not a dropped ask.
    context = context_holding(
        draft(
            evidence_requirements=[
                requirement(measure="undeclared default debt service"),
                derived_requirement(),
            ]
        )
    )

    result = await _discover(surface, context)

    assert result["outcome"] == "ok"
    assert result["status"] == "definite"
    assert [w["code"] for w in result["warnings"]] == ["default_not_declared"]
    log = load_log(context.state)
    (record,) = log.records
    assert [w.code for w in record.warnings] == ["default_not_declared"]
    assert record.blocked_by == ()
    keys = _requirement_keys(context, "annual_debt_service")
    assert list(log.selectable(requirement_keys=keys, pinned={})) == ["annual_debt_service"]


async def test_candidates_are_recorded_but_not_selectable(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Which of several sources to use is a methodological decision, so the
    # candidates are kept for the referral and kept away from the compiler.
    context = context_holding(
        draft(
            evidence_requirements=[requirement(measure="aid disbursements"), derived_requirement()]
        )
    )

    result = await _discover(surface, context)

    assert result["status"] == "candidates"
    # Described so the model can judge them, with no `spec_id` attached so it cannot pick one.
    assert [s["sentence"] for s in result["sources"]] == ["aid from France", "aid to Togo"]
    assert all("spec_id" not in s for s in result["sources"])
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    assert log.selectable(requirement_keys=keys, pinned={}) == {}
    assert [
        c.spec_id for c in log.candidates_for(keys["annual_debt_service"], "annual_debt_service")
    ] == [
        SPEC_OTHER,
        SPEC_THIRD,
    ]


async def test_an_answer_that_is_not_definite_arrives_with_the_action_to_take(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The action sits beside the candidates a model must not pick from, which is
    # where the temptation is. A definite answer carries none.
    candidates_context = context_holding(
        draft(
            evidence_requirements=[requirement(measure="aid disbursements"), derived_requirement()]
        )
    )
    no_data_context = context_holding(
        draft(evidence_requirements=[requirement(measure="unicorn horns"), derived_requirement()])
    )
    unsupported_context = context_holding(
        draft(evidence_requirements=[requirement(measure="unsupported ask"), derived_requirement()])
    )

    definite = await _discover(surface, context_holding(draft()))
    candidates = await _discover(surface, candidates_context)
    nothing = await _discover(surface, no_data_context)
    unsupported = await _discover(surface, unsupported_context)

    assert "action" not in definite
    assert candidates["action"] == {
        "call": "return_to_planner",
        "arguments": {
            "kind": "source_ambiguous",
            "requirement_id": "annual_debt_service",
            "reason": "discover_source answered candidates for 'annual_debt_service'",
        },
    }
    assert nothing["action"]["arguments"]["kind"] == "no_data"
    assert nothing["action"]["arguments"]["reason"].startswith(
        "discover_source answered no_data for 'annual_debt_service'"
    )
    assert unsupported["action"]["arguments"]["kind"] == "unsupported_ask"
    assert unsupported["action"]["arguments"]["reason"].startswith(
        "discover_source answered unsupported for 'annual_debt_service'"
    )


async def test_no_data_and_unsupported_carry_the_servers_detail(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    no_data_context = context_holding(
        draft(evidence_requirements=[requirement(measure="unicorn horns"), derived_requirement()])
    )
    unsupported_context = context_holding(
        draft(evidence_requirements=[requirement(measure="unsupported ask"), derived_requirement()])
    )

    nothing = await _discover(surface, no_data_context)
    unsupported = await _discover(surface, unsupported_context)

    assert (nothing["status"], nothing["detail"]) == ("no_data", "entity_not_resolved")
    assert (unsupported["status"], unsupported["detail"]) == (
        "unsupported",
        "cannot serve this ask",
    )
    assert nothing["sources"] == [] and unsupported["sources"] == []


async def test_a_pinned_source_found_among_candidates_is_selectable(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Pinning is how a user settles a `source_ambiguous` referral. Once
    # settled this way, a fresh search that answers `candidates` again must
    # not send the model back to refer the same requirement again: an
    # unconditional hint (any non-`definite` status getting one, honoured
    # pin or not) would do exactly that, since the requirement never stops
    # looking settled to `selectable` while looking unsettled to the
    # `action` the model was told to follow.
    pinned = requirement(measure="aid disbursements", source_spec_id=SPEC_THIRD)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_THIRD)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["status"] == "candidates"
    assert "action" not in result
    # Only the pinned candidate is named -- the other stays a candidate
    # among several, described without a `spec_id`, exactly as `compact_source`
    # keeps an unpinned candidate from being one a model could pick itself.
    by_sentence = {s["sentence"]: s for s in result["sources"]}
    assert by_sentence["aid to Togo"].get("spec_id") == SPEC_THIRD
    assert "spec_id" not in by_sentence["aid from France"]
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    selectable = log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_THIRD})
    assert selectable["annual_debt_service"].spec_id == SPEC_THIRD


async def test_a_pinned_source_a_definite_answer_names_is_selectable(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The other half of the rule: a pin is honoured from a definite answer as
    # well as from candidates. Without this, dropping `definite` from the
    # honoured statuses would pass every other pin test in this file.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["status"] == "definite"
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    selectable = log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_KENYA})
    assert selectable["annual_debt_service"].spec_id == SPEC_KENYA


def _matching_window() -> dict[str, Any]:
    """A `coverage` block whose window agrees with `requirement`'s own default `period` (2015-2024)."""
    return {"kind": "exact", "has_data": True, "window": {"start_year": 2015, "end_year": 2024}}


@pytest.mark.parametrize("earlier_turn", [False, True], ids=["this_turn", "an_earlier_turn"])
async def test_a_pin_of_a_shown_dataset_builds_the_source_with_no_search(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    earlier_turn: bool,
) -> None:
    # The pin names a dataset the planner was shown (`PIN_SOURCES_STATE_KEY`)
    # with no blocking warning, and the block's own window agrees with the
    # requirement's period: the source comes straight from that block, with
    # no `search` call at all. An earlier turn's dataset is shown with no
    # block, and the pin reads the one that turn stored.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    block = _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window())
    if earlier_turn:
        sources = pin_sources(context.state)
        sources[SPEC_KENYA] = PinSource(settled=True, earlier_turn=True)
        save_pin_sources(context.state, sources)
        current_earlier_block.set({SPEC_KENYA: block}.get)
    else:
        _shown(context, SPEC_KENYA, block)

    result = await _discover(surface, context)

    assert search_calls == []
    assert result["status"] == "definite"
    assert "action" not in result
    assert result["sources"][0]["spec_id"] == SPEC_KENYA
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    selectable = log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_KENYA})
    assert selectable["annual_debt_service"].spec_id == SPEC_KENYA
    assert log.records[0].tool == "pin"


def _shown(context: Any, spec_id: str, block: dict[str, Any] | None, *warnings: str) -> None:
    """Add `spec_id` to the datasets the planner was shown, with `block` and `warnings`."""
    sources = pin_sources(context.state)
    sources[spec_id] = PinSource(block=block, warnings=warnings, settled=True)
    save_pin_sources(context.state, sources)


async def test_two_requirements_pinning_one_shown_dataset_both_search(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Two requirements with different measures pin the same shown dataset:
    # nothing tells which one the search that returned it was about, so
    # neither reuses its block and each is confirmed by a fresh search.
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[
                requirement(source_spec_id=SPEC_KENYA),
                derived_requirement(),
                requirement(id="req_b", measure="debt stock", source_spec_id=SPEC_KENYA),
                derived_requirement(id="calc_b", input_id="req_b"),
            ],
        )
    )
    _shown(
        context,
        SPEC_KENYA,
        _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()),
    )

    await _discover(surface, context)
    await _discover(surface, context, requirement_id="req_b")

    assert len(search_calls) == 2
    assert "pin" not in [record.tool for record in load_log(context.state).records]


async def test_two_pins_of_shown_datasets_each_reuse_their_own_block(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Two dataset requirements pin two datasets the planner was shown: each
    # takes its source from its own block, and no discovery search is spent.
    req_b = requirement(
        id="req_b",
        measure="other debt figures",
        source_spec_id=SPEC_OTHER,
        decision_ids=["source_b"],
    )
    source_b = pinning_decision(
        SPEC_OTHER,
        slot="source_b",
        description="Which debt figures to use.",
        options=[SPEC_OTHER, "unpinned"],
        chosen=SPEC_OTHER,
    )
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA), source_b],
            evidence_requirements=[
                requirement(source_spec_id=SPEC_KENYA),
                derived_requirement(),
                req_b,
                derived_requirement(id="calc_b", input_id="req_b"),
            ],
        )
    )
    _shown(
        context,
        SPEC_KENYA,
        _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()),
    )
    _shown(
        context, SPEC_OTHER, _spec(SPEC_OTHER, "debt stock of Kenya", coverage=_matching_window())
    )

    first = await _discover(surface, context)
    second = await _discover(surface, context, requirement_id="req_b")

    assert search_calls == []
    assert [first["sources"][0]["spec_id"], second["sources"][0]["spec_id"]] == [
        SPEC_KENYA,
        SPEC_OTHER,
    ]
    assert [record.tool for record in load_log(context.state).records] == ["pin", "pin"]


def _two_dataset_context_pinning(
    spec_id: str,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> Any:
    """A two-dataset plan whose second requirement, `req_b`, pins `spec_id`."""
    source_decision = pinning_decision(
        spec_id,
        slot="source",
        description="Which source to use.",
        options=[spec_id, "unpinned"],
        chosen=spec_id,
    )
    req_b = requirement(
        id="req_b",
        measure="other debt figures",
        source_spec_id=spec_id,
        decision_ids=["measure", "source"],
    )
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    return context_holding(
        draft(
            decisions=[decision(), source_decision],
            evidence_requirements=[requirement(), derived_requirement(), req_b, calc_b],
        )
    )


@pytest.fixture
def definition_store() -> Any:
    """The definition store this process holds, apart from the one a test gives data-mcp."""
    store = InMemoryDefinitionStore()
    set_definition_store(store)
    yield store
    set_definition_store(None)


def _described(spec_id: str, sentence: str, searched_as: str | None) -> SelectedSource:
    return SelectedSource(
        spec_id=spec_id,
        sentence=sentence,
        has_data=True,
        served_count=2,
        searched_as=searched_as,
    )


def test_described_sources_lists_each_dataset_once_with_every_retrieval_of_it() -> None:
    """A workflow holds one source per retrieval. One dataset under one spec_id,
    under a copy over another period, and searched as other words is one entry;
    another dataset is its own."""
    debt = _described(SPEC_KENYA, "debt service paid by Kenya", "debt service")
    sources = [
        debt,
        debt,
        _described(SPEC_KENYA, "debt service paid by Kenya", "interest paid"),
        _described(SPEC_THIRD, "debt service paid by Kenya", "debt service"),
        _described(SPEC_OTHER, "government revenue of Kenya", "revenue"),
    ]

    described = described_sources(sources)

    assert described == [
        {
            **compact_source(debt),
            "also_retrieved_as": [
                {"spec_id": SPEC_KENYA, "searched_as": "interest paid", "coverage_window": None},
                {"spec_id": SPEC_THIRD, "searched_as": "debt service", "coverage_window": None},
            ],
        },
        compact_source(sources[-1]),
    ]
    assert "also_retrieved_as" not in described[1]


def test_described_sources_lists_a_dataset_once_over_every_period_it_is_retrieved_for() -> None:
    # Search states a sentence, a coverage window and a StatVar count for each
    # period, so the copies of one dataset describe differently and share only
    # their identity apart from its years. Each period's own coverage, and its
    # `has_data` where that differs, is listed with its retrieval.
    wider, narrower, recent = (
        source_from_spec_block(_KENYA_OVER[years]).model_copy(update={"searched_as": phrase})
        for years, phrase in (
            ("2010-2023", "education aid"),
            ("2018-2020", "aid in the pandemic"),
            ("2015-2023", "recent aid"),
        )
    )
    assert wider.sentence != narrower.sentence
    assert wider.coverage_window != narrower.coverage_window
    recent = recent.model_copy(update={"has_data": False})
    other = wider.model_copy(
        update={"spec_id": SPEC_OTHER, "identity": {"identity_version": "spec1", "seed": "other"}}
    )

    described = described_sources([wider, narrower, recent, other])

    assert described == [
        {
            **compact_source(wider),
            "also_retrieved_as": [
                {
                    "spec_id": narrower.spec_id,
                    "searched_as": "aid in the pandemic",
                    "coverage_window": {"start": 2018, "end": 2020},
                },
                {
                    "spec_id": recent.spec_id,
                    "searched_as": "recent aid",
                    "coverage_window": {"start": 2015, "end": 2023},
                    "has_data": False,
                },
            ],
        },
        compact_source(other),
    ]


def test_described_sources_keeps_the_definition_any_retrieval_of_a_dataset_fetched() -> None:
    bare = _described(SPEC_KENYA, "debt service paid by Kenya", "debt service")
    defined = bare.model_copy(update={"definition": "Debt service is ..."})

    (described,) = described_sources([bare, defined])

    assert described["definition"] == "Debt service is ..."
    assert "also_retrieved_as" not in described


@pytest.mark.parametrize(
    "members",
    [
        ({"shape_member_digest": "d1"}, {"shape_member_digest": "d2"}),
        ({"shape_member_digest": "d1"}, {"shape_member_digest": None}),
    ],
    ids=["two-digests", "one-digest"],
)
def test_described_sources_keeps_apart_sources_that_bind_other_members(
    members: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Two sources with no identity that describe alike are one dataset only
    when they state the same member digest."""
    debt = _described(SPEC_KENYA, "debt service paid by Kenya", "debt service").model_copy(
        update=members[0]
    )
    other = debt.model_copy(update={"spec_id": SPEC_OTHER, **members[1]})

    assert described_sources([debt, other]) == [compact_source(debt), compact_source(other)]


def test_described_sources_of_one_source_is_its_compact_form() -> None:
    source = _described(SPEC_KENYA, "debt service paid by Kenya", "debt service")

    assert described_sources([source]) == [compact_source(source)]
    assert described_sources([]) == []


class _SearchAndResolve:
    """A surface answering `search` from `searching` and `resolve_definition` from `resolving`."""

    def __init__(self, searching: ToolSurface, resolving: DataMcpDefinitions) -> None:
        self._searching = searching
        self._resolving = resolving

    async def list_tools(self) -> list[Any]:
        return await self._searching.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Any,
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        chosen = self._resolving if tool.name == "resolve_definition" else self._searching
        return await chosen.call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


@pytest.mark.parametrize("on_record", ["copy", "identity", None])
async def test_a_pin_whose_identity_is_on_record_is_read_from_searchs_answer_with_no_search(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    on_record: str | None,
) -> None:
    # The pin names a dataset over the requirement's period whose identity is
    # on record: the copy `propose_plan` pinned in place of an earlier
    # search's dataset over other years, or an earlier search's own identity.
    # Search's answer for that identity is the source, with no `search`
    # call and nothing referred back. With nothing on record the pin is
    # searched for, and a search that does not return it refers it back.
    answered = _KENYA_OVER["2010-2023"]
    spec_id = answered["spec_id"]
    identity = answered["definition"]["identity"]
    context = context_holding(
        draft(
            decisions=[pinning_decision(spec_id)],
            evidence_requirements=[
                requirement(source_spec_id=spec_id, period=Period(start=2010, end=2023)),
                derived_requirement(),
            ],
        )
    )
    recorded = RecordedIdentity(identity=identity)
    token = current_recorded_identity.set(
        {spec_id: recorded}.get if on_record == "identity" else None
    )
    if on_record == "copy":
        earlier = _KENYA_OVER["2015-2023"]["spec_id"]
        copied = WindowedPin(spec_id=spec_id, identity=identity, windowed_from=earlier)
        save_windowed_pins(context.state, [copied])
    resolving = DataMcpDefinitions(InMemoryDefinitionStore(), served=[answered])

    try:
        result = await _discover(_SearchAndResolve(surface, resolving), context)
    finally:
        current_recorded_identity.reset(token)

    if on_record is None:
        assert resolving.calls == []
        assert len(search_calls) == 1
        assert result["action"]["arguments"]["kind"] == "pinned_source_not_found"
        return
    assert search_calls == []
    assert resolving.calls == [{"spec_id": spec_id, "identity": identity}]
    assert result["status"] == "definite"
    assert "action" not in result
    log = load_log(context.state)
    assert log.records[0].tool == "pin"
    assert log.sources["annual_debt_service"][spec_id].coverage_window == Period(
        start=2010, end=2023
    )


@pytest.mark.parametrize("whole_history", [True, False], ids=["whole-history", "stated-period"])
async def test_a_copy_with_no_end_year_is_read_from_the_year_the_plan_retrieves_it_from(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    whole_history: bool,
) -> None:
    # `propose_plan` pinned the dataset over a range from 2010 with no end
    # year, for a requirement stating 2015-2023 with `period_end_open`. The
    # copy is the requirement's source when the plan reads that dataset from
    # its first year, 2010, for a reader that rolls from its start; read from
    # its stated 2015 instead, a fresh `search` finds the definition.
    answered = copy.deepcopy(_KENYA_OVER["2010-2023"])
    spec_id, identity = windowed_identity(
        answered["definition"]["identity"], {"kind": "range", "start_year": 2010}
    )
    answered["spec_id"] = answered["definition"]["spec_id"] = spec_id
    answered["definition"]["identity"] = identity
    reader = (
        GrowthRequirement(
            id="growth",
            decision_ids=["measure"],
            input_id="annual_debt_service",
            window_start=2019,
            window_end=2023,
            rolling=RollingWindow(
                window_start=RelativePeriod(offset=-4), window_end=RelativePeriod(offset=0)
            ),
        )
        if whole_history
        else derived_requirement()
    )
    context = context_holding(
        draft(
            decisions=[pinning_decision(spec_id)],
            evidence_requirements=[
                requirement(
                    source_spec_id=spec_id,
                    period=Period(start=2015, end=2023),
                    period_end_open=True,
                ),
                reader,
            ],
            first_years=(FirstYear(requirement_id="annual_debt_service", year=2010),)
            if whole_history
            else (),
        )
    )
    earlier = _KENYA_OVER["2015-2023"]["spec_id"]
    save_windowed_pins(
        context.state, [WindowedPin(spec_id=spec_id, identity=identity, windowed_from=earlier)]
    )
    resolving = DataMcpDefinitions(InMemoryDefinitionStore(), served=[answered])

    result = await _discover(_SearchAndResolve(surface, resolving), context)

    if not whole_history:
        assert resolving.calls == []
        assert len(search_calls) == 1
        return
    assert search_calls == []
    assert resolving.calls == [{"spec_id": spec_id, "identity": identity}]
    assert result["status"] == "definite"
    assert spec_id in load_log(context.state).sources["annual_debt_service"]


@pytest.mark.parametrize("axis_open", [True, False], ids=["axis-open", "axis-closed"])
@pytest.mark.parametrize("path", ["shown-block", "recorded-identity"])
async def test_a_pin_leaving_the_across_side_closed_is_searched_for_as_an_open_role(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    path: str,
    axis_open: bool,
) -> None:
    # The requirement ranks across creditors and pins a dataset the turn
    # already holds. One that leaves the creditor side open is honoured with
    # no `search`. One that totals it, as a search that named no open role
    # finds it, cannot hold every creditor, so it is set aside, recorded with
    # no status, and `search` is asked for the side as an open role.
    creditor_axis = [{"role": "origin", "member_count": 55, "members_available": True}]
    if path == "shown-block":
        spec_id, period = SPEC_KENYA, Period(start=2015, end=2024)
        block = _spec(
            spec_id,
            "debt service paid by Kenya",
            coverage=_matching_window(),
            **({"role_axis_facts": creditor_axis} if axis_open else {}),
        )
        surface_used: ToolSurface = surface
    else:
        answered = copy.deepcopy(_KENYA_OVER["2010-2023"])
        answered["resolved"]["role_axis_facts"] = creditor_axis if axis_open else []
        spec_id, period = answered["spec_id"], Period(start=2010, end=2023)
        recorded = RecordedIdentity(identity=answered["definition"]["identity"])
        surface_used = _SearchAndResolve(
            surface, DataMcpDefinitions(InMemoryDefinitionStore(), served=[answered])
        )
    context = context_holding(
        draft(
            decisions=[pinning_decision(spec_id)],
            evidence_requirements=[
                requirement(source_spec_id=spec_id, period=period, across=("creditors",)),
                derived_requirement(),
            ],
        )
    )
    if path == "shown-block":
        _shown(context, spec_id, block)
    token = current_recorded_identity.set(
        {spec_id: recorded}.get if path == "recorded-identity" else None
    )

    try:
        result = await _discover(surface_used, context)
    finally:
        current_recorded_identity.reset(token)

    tools = [record.tool for record in load_log(context.state).records]
    if axis_open:
        assert search_calls == []
        assert tools == ["pin"]
        assert result["sources"][0]["open_entity_axes"] == [{"column": "origin", "members": 55}]
        return
    assert tools == ["pin", "search"]
    assert load_log(context.state).records[0].status is None
    (call,) = search_calls
    (part,) = call["parts"]
    assert part["open_roles"] == ["creditor"]


class _UnreachableStore:
    """A definition store whose backend cannot be reached."""

    async def put_many(self, definitions: Any) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


class _UnreachableDataMcp:
    """A surface whose data-mcp cannot be reached."""

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Any,
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del arguments, on_progress
        raise ToolSurfaceUnavailableError(f"{tool.server} timed out")


@pytest.mark.parametrize(
    ("surface_down", "said"),
    [
        (
            lambda: DataMcpDefinitions(_UnreachableStore(), served=[_KENYA_OVER["2010-2023"]]),
            "the definition store is unreachable",
        ),
        (_UnreachableDataMcp, "data-mcp timed out"),
    ],
    ids=["definition-store", "data-mcp"],
)
async def test_a_windowed_pin_search_could_not_be_asked_about_is_retryable(
    surface_down: Callable[[], ToolSurface],
    said: str,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    answered = _KENYA_OVER["2010-2023"]
    spec_id = answered["spec_id"]
    context = context_holding(
        draft(
            decisions=[pinning_decision(spec_id)],
            evidence_requirements=[
                requirement(source_spec_id=spec_id, period=Period(start=2010, end=2023)),
                derived_requirement(),
            ],
        )
    )
    copied = WindowedPin(
        spec_id=spec_id,
        identity=answered["definition"]["identity"],
        windowed_from=_KENYA_OVER["2015-2023"]["spec_id"],
    )
    save_windowed_pins(context.state, [copied])

    result = await _discover(surface_down(), context)

    assert result["outcome"] == "refused"
    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value
    assert result["message"].startswith(
        f"Search could not be asked for {spec_id!r} over 2010 to 2023: "
    )
    assert said in result["message"]
    assert "annual_debt_service" not in load_log(context.state).sources


def _kenya_without_its_definition() -> dict[str, Any]:
    block = dict(_KENYA_OVER["2010-2023"])
    del block["definition"]
    return block


class _Refusing:
    """A surface whose data-mcp refuses every call with `refusal`."""

    def __init__(self, refusal: dict[str, object]) -> None:
        self._refusal = refusal

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Any,
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del tool, arguments, on_progress
        return ToolCallResult(is_error=True, structured_content=self._refusal, text="refused")


@pytest.mark.parametrize(
    ("refusal", "code", "said"),
    [
        (
            {"code": "request_invalid"},
            "request_invalid",
            "Search no longer resolves it (request_invalid)",
        ),
        (
            {"code": "upstream_error", "upstream_code": "no_spec"},
            "no_spec",
            "Search no longer resolves it (no_spec)",
        ),
        ({"message": "refused"}, None, "Search no longer resolves it"),
    ],
    ids=["code", "upstream-code", "no-code"],
)
async def test_a_refused_resolve_names_the_refusals_code_when_it_states_one(
    refusal: dict[str, object], code: str | None, said: str
) -> None:
    with pytest.raises(DefinitionNotResolvedError) as raised:
        await resolve_definition(_Refusing(refusal), spec_id=SPEC_KENYA, identity={"seed": "x"})

    assert str(raised.value) == said
    assert raised.value.code == code
    assert raised.value.refused is True
    assert raised.value.retryable is False


def test_resolved_spec_is_the_block_of_a_definite_part_naming_the_spec_id_with_its_identity() -> (
    None
):
    block = _KENYA_OVER["2010-2023"]
    part = {"status": "definite", "spec": block}

    assert resolved_spec(part, spec_id=block["spec_id"]) == block


@pytest.mark.parametrize(
    ("part", "fault", "answered"),
    [
        ({"status": "no_data", "no_data": {"reason": "no_series"}}, PartFault.NOT_DEFINITE, None),
        (
            {"status": "definite", "spec": _KENYA_OVER["2015-2023"]},
            PartFault.ANOTHER_DATASET,
            _KENYA_OVER["2015-2023"]["spec_id"],
        ),
        (
            {"status": "definite", "spec": _kenya_without_its_definition()},
            PartFault.NO_DEFINITION,
            None,
        ),
        (
            {
                "status": "definite",
                "spec": {**_KENYA_OVER["2010-2023"], "definition": {"schema_version": 1}},
            },
            PartFault.NO_DEFINITION,
            None,
        ),
        ({"status": "definite"}, PartFault.NO_DEFINITION, None),
        (
            {"status": "definite", "spec": {**_KENYA_OVER["2010-2023"], "spec_id": 7}},
            PartFault.ANOTHER_DATASET,
            None,
        ),
        (
            {
                "status": "definite",
                "spec": {
                    key: value
                    for key, value in _KENYA_OVER["2010-2023"].items()
                    if key != "spec_id"
                },
            },
            PartFault.ANOTHER_DATASET,
            None,
        ),
    ],
    ids=[
        "not-definite",
        "another-spec-id",
        "no-definition",
        "no-identity",
        "no-spec",
        "spec-id-not-a-string",
        "no-spec-id",
    ],
)
def test_resolved_spec_names_how_a_part_is_not_the_dataset(
    part: dict[str, Any], fault: PartFault, answered: str | None
) -> None:
    with pytest.raises(PartNotResolvedError) as raised:
        resolved_spec(part, spec_id=_KENYA_OVER["2010-2023"]["spec_id"])

    assert raised.value.fault is fault
    assert raised.value.answered == answered
    assert raised.value.part == part


async def test_a_pin_matching_the_resolve_id_still_searches_when_the_resolve_dropped_time_text(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The resolve's own part carried a blocking warning: `search` dropped
    # the time text and answered a narrower ask than the requirement states,
    # so reusing the block would relabel that narrower answer as DEFINITE
    # for the whole requirement. Falls through to the existing search path,
    # which applies `ask_not_understood` itself.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    _shown(
        context,
        SPEC_KENYA,
        _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()),
        "time_text_not_parsed",
    )

    await _discover(surface, context)

    assert len(search_calls) == 1


@pytest.mark.parametrize(
    "coverage",
    [
        {"kind": "exact", "has_data": True, "window": {"start_year": 2010, "end_year": 2020}},
        {"kind": "breadth", "has_data": True},
        {"kind": "bare", "has_data": True},
    ],
    ids=["exact-other-window", "breadth", "bare"],
)
async def test_a_pin_matching_the_resolve_id_still_searches_over_a_different_period(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    coverage: dict[str, Any],
) -> None:
    # The block's window is not the requirement's own period (2015-2024,
    # `requirement`'s default), or the block states no window at all: a
    # source resolved as covering other years, or years nobody stated, is not
    # a source for this requirement's window, so this falls through to a
    # fresh search.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    _shown(
        context,
        SPEC_KENYA,
        _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=coverage),
    )

    await _discover(surface, context)

    assert len(search_calls) == 1


async def test_a_pin_to_a_different_id_than_the_resolve_still_searches(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The resolve settled on SPEC_KENYA this turn, but the pin names a
    # different id: the resolve's own block is not honoured, and the
    # existing search path confirms (or fails to confirm) the pin instead.
    pinned = requirement(source_spec_id=SPEC_OTHER)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_OTHER)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    _shown(context, SPEC_KENYA, _spec(SPEC_KENYA, "debt service paid by Kenya"))

    result = await _discover(surface, context)

    assert len(search_calls) == 1
    assert result["status"] == "definite"
    assert result["action"]["arguments"]["kind"] == "pinned_source_not_found"


async def test_a_pin_matching_the_resolve_id_with_no_block_falls_back_to_search(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The pin names the resolve's own id, but no block was carried forward
    # for it (a clarify re-entry carries none): nothing to build a source
    # from, so this falls through to the existing search path.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    _shown(context, SPEC_KENYA, None)

    result = await _discover(surface, context)

    assert len(search_calls) == 1
    assert result["status"] == "definite"
    assert "action" not in result
    assert result["sources"][0]["spec_id"] == SPEC_KENYA


async def test_a_pin_cannot_resurrect_a_no_data_answers_nearest_series(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The search said there is no data for the ask. Offering the nearest
    # real series instead does not count as finding the pinned source.
    pinned = requirement(measure="nearly debt service", source_spec_id=SPEC_OTHER)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_OTHER)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["status"] == "no_data"
    assert [s["sentence"] for s in result["sources"]] == ["total debt service paid by Kenya"]
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    assert SPEC_OTHER in log.records[0].spec_ids
    assert log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_OTHER}) == {}
    # The action agrees with `selectable`: a pin offered as a near miss has
    # not been found, so the model is sent back rather than told to retrieve
    # a source discovery will refuse to hand it.
    assert result["action"]["arguments"]["kind"] == ReferralKind.PINNED_SOURCE_NOT_FOUND.value


async def test_a_pinned_source_not_found_yields_the_referral_action(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
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

    result = await _discover(surface, context)

    assert result["status"] == "definite"
    assert result["action"] == {
        "call": "return_to_planner",
        "arguments": {
            "kind": "pinned_source_not_found",
            "requirement_id": "annual_debt_service",
            "reason": (
                f"'annual_debt_service' pins {SPEC_OTHER!r}, which this revision's "
                "discovery did not find. This search found 1 source(s) instead."
            ),
        },
    }


async def test_a_pin_that_misses_an_ambiguous_answer_names_what_was_found(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Both facts are true here: the pin missed, and the fresh search is
    # itself ambiguous. The action still names `pinned_source_not_found` --
    # `return_to_planner` is what attaches the two real candidates to the
    # referral -- but the reason says a search found something, rather than
    # only that the pin did not.
    pinned = requirement(measure="aid disbursements", source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["status"] == "candidates"
    assert result["action"]["arguments"]["kind"] == "pinned_source_not_found"
    assert result["action"]["arguments"]["reason"] == (
        f"'annual_debt_service' pins {SPEC_KENYA!r}, which this revision's discovery "
        "did not find. This search found 2 source(s) instead."
    )


async def test_a_requirement_the_plan_lacks_is_refused_with_the_plans_own(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())

    result = await _discover(surface, context, requirement_id="population")

    assert result["code"] == "requirement_unknown"
    assert "annual_debt_service" in result["detail"]["known"]
    assert load_log(context.state).records == ()


async def test_discovery_needs_a_draft_plan(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding, empty_context: Any
) -> None:
    assert (await _discover(surface, empty_context))["code"] == "plan_missing"

    approved = context_holding(draft(status=PlanStatus.PENDING_APPROVAL))
    assert (await _discover(surface, approved))["code"] == "plan_not_draft"


def _three_requirement_plan(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> Plan:
    """A draft with three dataset requirements, none of which alone reaches
    `DISCOVERY_CALL_BUDGET_PER_REQUIREMENT` across `DISCOVERY_CALL_BUDGET` calls."""
    req_b = requirement(id="req_b", measure="aid disbursements", entities=[Entity(name="Togo")])
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    req_c = requirement(id="req_c", measure="undeclared default debt service")
    calc_c = derived_requirement(id="calc_c", input_id="req_c")
    return draft(
        evidence_requirements=[
            requirement(),
            derived_requirement(),
            req_b,
            calc_b,
            req_c,
            calc_c,
        ]
    )


async def test_the_budget_stops_the_ninth_call_and_points_at_the_referral(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Round-robin across three requirements, each well under its own
    # per-requirement ceiling, so what stops the ninth call is the
    # revision's own total budget, not any one requirement's share of it.
    context = context_holding(_three_requirement_plan(draft, requirement, derived_requirement))
    order = ["annual_debt_service", "req_b", "req_c"]
    for i in range(DISCOVERY_CALL_BUDGET):
        assert (await _discover(surface, context, requirement_id=order[i % 3]))["outcome"] == "ok"

    blocked = await _discover(surface, context, requirement_id=order[DISCOVERY_CALL_BUDGET % 3])

    assert blocked["code"] == "budget_exhausted"
    assert blocked["detail"]["used"] == DISCOVERY_CALL_BUDGET
    assert blocked["action"] == {
        "call": "return_to_planner",
        "arguments": {"kind": "discovery_budget_exhausted", "reason": blocked["message"]},
    }


@pytest.mark.parametrize(
    ("timed_out", "arguments"),
    [
        # An early call timed out, the most recent one was answered.
        ({1}, {"kind": "discovery_budget_exhausted"}),
        # The most recent call, on `req_b`, timed out; the refused call is on `req_c`.
        (
            {DISCOVERY_CALL_BUDGET - 1},
            {"kind": "discovery_unavailable", "requirement_id": "req_b", "budget_spent": True},
        ),
    ],
)
async def test_a_spent_budget_refers_an_outage_only_when_its_last_call_timed_out(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    timed_out: set[int],
    arguments: dict[str, Any],
) -> None:
    context = context_holding(_three_requirement_plan(draft, requirement, derived_requirement))
    order = ["annual_debt_service", "req_b", "req_c"]
    log = DiscoveryLog()
    for i in range(DISCOVERY_CALL_BUDGET):
        log = log.with_record(
            DiscoveryRecord(
                revision=_revision(context),
                tool="search",
                requirement_id=order[i % 3],
                transport_failed=i in timed_out,
            )
        )
    save_log(context.state, log)

    blocked = await _discover(surface, context, requirement_id=order[DISCOVERY_CALL_BUDGET % 3])

    assert blocked["code"] == "budget_exhausted"
    assert blocked["action"] == {
        "call": "return_to_planner",
        "arguments": {**arguments, "reason": blocked["message"]},
    }


async def test_the_budget_counts_per_revision_not_per_session(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(_three_requirement_plan(draft, requirement, derived_requirement))
    order = ["annual_debt_service", "req_b", "req_c"]
    for i in range(DISCOVERY_CALL_BUDGET):
        assert (await _discover(surface, context, requirement_id=order[i % 3]))["outcome"] == "ok"
    assert (await _discover(surface, context, requirement_id=order[DISCOVERY_CALL_BUDGET % 3]))[
        "code"
    ] == "budget_exhausted"

    other = draft(question="A wholly different question about something else?")
    context.state[PLAN_STATE_KEY] = other.model_dump_json()

    result = await _discover(surface, context)

    assert result["outcome"] == "ok"
    assert result["budget_remaining"] == DISCOVERY_CALL_BUDGET - 1


async def test_a_single_requirement_stops_at_its_own_share_of_the_budget(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    # One requirement alone, called more times than its own ceiling allows,
    # well short of the revision's own total: the requirement's own share
    # stops it, not the revision.
    context = context_holding(draft())
    for _ in range(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT):
        assert (await _discover(surface, context))["outcome"] == "ok"

    blocked = await _discover(surface, context)

    assert blocked["code"] == "budget_exhausted"
    assert blocked["detail"]["used"] == DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    assert blocked["detail"]["requirement_id"] == "annual_debt_service"
    assert blocked["action"] == {
        "call": "return_to_planner",
        "arguments": {"kind": "discovery_budget_exhausted", "reason": blocked["message"]},
    }
    revision = _revision(context)
    log = load_log(context.state)
    assert log.calls(revision) == DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    assert log.calls(revision) < DISCOVERY_CALL_BUDGET


async def test_a_describe_record_does_not_spend_the_search_budget(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    # `fetch_definitions` writes its `describe` record from the compile
    # loop, not from a model's own choice, so it must not shave a real
    # search off `DISCOVERY_CALL_BUDGET_PER_REQUIREMENT`. One `describe`
    # interleaved among `DISCOVERY_CALL_BUDGET_PER_REQUIREMENT` real
    # searches must still let the last of those searches through.
    context = context_holding(draft())
    revision = _revision(context)
    key = _requirement_key(context)

    first = await _discover(surface, context)
    assert first["outcome"] == "ok"
    log = load_log(context.state)
    source = log.sources["annual_debt_service"][SPEC_KENYA]
    pairs = await fetch_definitions(
        surface,
        log,
        revision=revision,
        requirement_keys={"annual_debt_service": key},
        sources={"annual_debt_service": source},
    )
    log = _apply(log, pairs)
    assert log.records[-1].tool == "describe"
    save_log(context.state, log)

    for _ in range(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT - 1):
        result = await _discover(surface, context)
        assert result["outcome"] == "ok"

    log = load_log(context.state)
    assert (
        log.requirement_calls(revision, "annual_debt_service")
        == DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    )


async def test_an_amended_plan_has_a_fresh_budget_and_keeps_what_was_found(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    await _discover(surface, context)  # one definite search, before the amend

    amended = Plan.model_validate_json(context.state[PLAN_STATE_KEY]).amend()
    context.state[PLAN_STATE_KEY] = amended.model_dump_json()

    # The requirement itself is untouched by the amend, so what was already
    # found for it is selectable again with no fresh call at all: an edit
    # elsewhere in the plan does not discard this requirement's own
    # selectable view.
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    assert list(log.selectable(requirement_keys=keys, pinned={})) == ["annual_debt_service"]
    # The revision's own call budget is fresh independently of that.
    revision = _revision(context)
    assert log.calls(revision) == 0

    result = await _discover(surface, context)

    assert result["budget_remaining"] == DISCOVERY_CALL_BUDGET - 1


async def test_an_unrelated_decision_edit_leaves_a_settled_requirements_discovery_intact(
    surface: McpSurface,
    draft: DraftFactory,
    decision: DecisionFactory,
    context_holding: ContextHolding,
) -> None:
    # A decision edit changes the plan's own `revision` (`plan_revision`
    # covers every decision) without touching the requirement's own searched
    # fields, so its `requirement_revision` -- and what is selectable for it
    # -- does not move.
    context = context_holding(draft())
    await _discover(surface, context)

    plan = Plan.model_validate_json(context.state[PLAN_STATE_KEY])
    amended = plan.amend(
        decisions=[decision(rationale="Debt service is what the user actually asked about.")]
    )
    assert plan_revision(amended) != plan_revision(plan)
    context.state[PLAN_STATE_KEY] = amended.model_dump_json()

    keys = _requirement_keys(context, "annual_debt_service")
    selectable = load_log(context.state).selectable(requirement_keys=keys, pinned={})
    assert selectable["annual_debt_service"].spec_id == SPEC_KENYA


async def test_a_source_found_for_one_requirement_is_not_selectable_for_another(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    req_b = requirement(id="req_b", measure="aid disbursements", entities=[Entity(name="Togo")])
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    plan = draft()
    context = context_holding(
        plan.amend(evidence_requirements=[*plan.evidence_requirements, req_b, calc_b]).model_copy(
            update={"status": PlanStatus.DRAFT}
        )
    )

    await _discover(surface, context, requirement_id="annual_debt_service")

    keys = _requirement_keys(context, "annual_debt_service", "req_b")
    selectable = load_log(context.state).selectable(requirement_keys=keys, pinned={})
    assert "annual_debt_service" in selectable
    assert "req_b" not in selectable


async def test_each_requirement_keeps_its_own_view_of_a_shared_source(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # Both requirements resolve to the same spec; the later search must not
    # overwrite the phrase the earlier requirement found it under.
    req_b = requirement(id="req_b", measure="debt service paid abroad")
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    plan = draft()
    context = context_holding(
        plan.amend(evidence_requirements=[*plan.evidence_requirements, req_b, calc_b]).model_copy(
            update={"status": PlanStatus.DRAFT}
        )
    )

    await _discover(surface, context, requirement_id="annual_debt_service")
    await _discover(surface, context, requirement_id="req_b")

    keys = _requirement_keys(context, "annual_debt_service", "req_b")
    log = load_log(context.state)
    selectable = log.selectable(requirement_keys=keys, pinned={})
    assert selectable["annual_debt_service"].spec_id == SPEC_KENYA
    assert selectable["req_b"].spec_id == SPEC_KENYA
    assert selectable["annual_debt_service"].searched_as == "external debt service paid"
    assert selectable["req_b"].searched_as == "debt service paid abroad"
    assert (
        log.sources["annual_debt_service"][SPEC_KENYA].searched_as == "external debt service paid"
    )


async def test_a_pin_is_looked_up_only_among_the_requirements_own_sources(
    surface: McpSurface,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # req_b pins the source annual_debt_service found; its own search found
    # nothing. Selection neither borrows the source nor raises.
    #
    # req_b cites its own "source" decision for the pin, apart from the
    # shared "measure" decision `annual_debt_service` also cites: fixing
    # source_spec_id on "measure" would hold the unpinned requirement to a
    # pin it never made.
    source = pinning_decision(
        SPEC_KENYA,
        slot="source",
        description="Which source to use.",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
    )
    req_b = requirement(
        id="req_b",
        measure="unicorn horns",
        source_spec_id=SPEC_KENYA,
        decision_ids=["measure", "source"],
    )
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    plan = draft(
        decisions=[decision(), source],
        evidence_requirements=[requirement(), derived_requirement(), req_b, calc_b],
    )
    context = context_holding(plan)

    await _discover(surface, context, requirement_id="annual_debt_service")
    await _discover(surface, context, requirement_id="req_b")

    keys = _requirement_keys(context, "annual_debt_service", "req_b")
    selectable = load_log(context.state).selectable(
        requirement_keys=keys, pinned={"req_b": SPEC_KENYA}
    )
    assert list(selectable) == ["annual_debt_service"]


def test_a_source_is_recorded_against_the_requirement_that_found_it(
    source: Callable[..., SelectedSource],
) -> None:
    orphan = DiscoveryRecord(revision="r", tool="describe")

    with pytest.raises(ValueError, match="recorded against the requirement"):
        DiscoveryLog().with_record(orphan, (source(),))


async def test_a_refused_search_is_an_error_that_still_counts(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(evidence_requirements=[requirement(measure="refuse"), derived_requirement()])
    )

    result = await _discover(surface, context)

    assert result["code"] == "search_refused"
    assert result["detail"]["refusal"]["code"] == "request_invalid"
    assert result["detail"]["budget_remaining"] == DISCOVERY_CALL_BUDGET - 1


async def test_an_unrecognised_answer_is_an_error_that_still_counts(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(
        draft(evidence_requirements=[requirement(measure="garbage"), derived_requirement()])
    )

    result = await _discover(surface, context)

    assert result["code"] == "search_refused"
    assert "unrecognised status 'partial'" in result["detail"]["parse_error"]
    revision = _revision(context)
    assert load_log(context.state).calls(revision) == 1


async def test_a_success_with_no_envelope_in_meta_is_an_error_that_still_counts(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    """The envelope rides in `_meta`, so a success carrying only the small answer has nothing to read."""
    context = context_holding(draft())
    answer_only = ToolCallResult(is_error=False, structured_content={"status": "definite"})
    surface = FixtureSurface(schemas=[], responses={"search": answer_only})

    result = await _discover(surface, context)

    assert result["code"] == "search_refused"
    assert "no envelope" in result["detail"]["parse_error"]
    assert load_log(context.state).calls(_revision(context)) == 1


async def test_an_unreachable_server_is_an_error_that_counts_against_the_budget(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    unreachable = McpSurface(sources={"data-mcp": "http://127.0.0.1:9/mcp"})

    result = await _discover(unreachable, context)

    assert result["code"] == "surface_unavailable"
    assert "could not call 'search'" in result["message"]
    revision = _revision(context)
    log = load_log(context.state)
    assert log.calls(revision) == 1
    (record,) = log.records
    assert record.transport_failed is True
    # A transport failure names no source: nothing it left behind is selectable.
    keys = _requirement_keys(context, "annual_debt_service")
    assert log.selectable(requirement_keys=keys, pinned={}) == {}


class _SlowSurface:
    """A surface whose every call takes far longer than any planner entry has left."""

    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Any,
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del tool, arguments, on_progress
        self.calls += 1
        await asyncio.sleep(30)
        raise AssertionError("the call outlived the planner entry's time")


@pytest.mark.parametrize(("time_left", "issued"), [(0.05, 1), (0.0, 0)], ids=["little", "none"])
async def test_a_search_inside_a_planner_entry_takes_only_the_time_the_entry_has_left(
    time_left: float, issued: int, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    slow = _SlowSurface()
    token = current_discovery_time_left.set(lambda: time_left)
    try:
        async with asyncio.timeout(5):
            result = await _discover(slow, context)
    finally:
        current_discovery_time_left.reset(token)

    assert slow.calls == issued
    assert result["outcome"] == "refused"
    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value
    # A search cut off is spent, marked `transport_failed`; one never issued
    # spends nothing.
    records = load_log(context.state).records
    assert [record.transport_failed for record in records] == [True] * issued
    assert result["detail"]["budget_remaining"] == DISCOVERY_CALL_BUDGET - issued


class _HeldResolve:
    """A surface whose `resolve_definition` answers a definite part once `release` is set."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = False

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Any,
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition"
        self.calls += 1
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        part = {"status": "definite", "spec": {"spec_id": arguments["spec_id"]}}
        envelope = {"parts": [part]}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: envelope})


async def _cut_resolve(surface: _HeldResolve) -> None:
    """One `resolve_definition` the entry's deadline cuts while `surface` still holds it."""
    token = current_discovery_time_left.set(lambda: 0.01)
    try:
        with pytest.raises(DefinitionNotResolvedError, match="did not answer"):
            await resolve_definition(surface, spec_id=SPEC_KENYA, identity={"seed": "x"})
    finally:
        current_discovery_time_left.reset(token)


@pytest.mark.parametrize(
    ("answered_first", "time_left"),
    [(True, -1.0), (False, 5.0)],
    ids=["answered-before-retry", "awaited-by-retry"],
)
async def test_a_resolve_cut_by_the_entry_deadline_serves_its_late_answer_to_the_retry(
    answered_first: bool, time_left: float
) -> None:
    """The traced resolve answered 0.7 s after the entry cut it. A retry
    1 s past the deadline is served that answer rather than refused, and a
    retry with time left waits on the same call rather than issuing another."""
    held = _HeldResolve()
    cut_calls = CutCalls()
    cut_calls_token = current_cut_calls.set(cut_calls)
    try:
        await _cut_resolve(held)
        if answered_first:
            held.release.set()
            await asyncio.sleep(0)
        else:
            asyncio.get_running_loop().call_later(0.01, held.release.set)
        token = current_discovery_time_left.set(lambda: time_left)
        try:
            part = await resolve_definition(held, spec_id=SPEC_KENYA, identity={"seed": "x"})
        finally:
            current_discovery_time_left.reset(token)
    finally:
        current_cut_calls.reset(cut_calls_token)
        await cut_calls.close()

    assert part["spec"]["spec_id"] == SPEC_KENYA
    assert held.calls == 1


async def test_a_cut_resolve_still_running_when_the_entry_ends_is_cancelled() -> None:
    held = _HeldResolve()
    cut_calls = CutCalls()
    cut_calls_token = current_cut_calls.set(cut_calls)
    try:
        await _cut_resolve(held)
        await asyncio.sleep(0)
        running_after_cut = not held.cancelled
    finally:
        current_cut_calls.reset(cut_calls_token)
        await cut_calls.close()

    assert running_after_cut
    assert held.cancelled is True
    assert held.calls == 1


async def test_a_resolve_whose_caller_is_cancelled_mid_await_is_cancelled_when_the_entry_ends() -> (
    None
):
    held = _HeldResolve()
    cut_calls = CutCalls()
    cut_calls_token = current_cut_calls.set(cut_calls)
    time_token = current_discovery_time_left.set(lambda: 5.0)
    try:
        caller = asyncio.ensure_future(
            resolve_definition(held, spec_id=SPEC_KENYA, identity={"seed": "x"})
        )
        while not held.calls:
            await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        running_after_cancel = not held.cancelled
    finally:
        current_discovery_time_left.reset(time_token)
        current_cut_calls.reset(cut_calls_token)
        await cut_calls.close()

    assert running_after_cancel
    assert held.cancelled is True


async def test_keeping_a_second_task_under_a_key_cancels_the_first() -> None:
    cut_calls = CutCalls()
    first = asyncio.ensure_future(asyncio.sleep(3600))
    second = asyncio.ensure_future(asyncio.sleep(3600))

    cut_calls.keep(b"key", first)
    cut_calls.keep(b"key", second)
    await asyncio.sleep(0)
    first_cancelled = first.cancelled()
    await cut_calls.close()

    assert first_cancelled
    assert second.cancelled()


async def test_the_log_survives_a_round_trip_through_session_state(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    await _discover(surface, context)

    log = load_log(context.state)

    assert log.records[0].status is DiscoveryStatus.DEFINITE
    assert log.sources["annual_debt_service"][SPEC_KENYA].entities[0].ref == "country/KEN"


@pytest.mark.parametrize("as_string", [False, True], ids=["dict form", "string form"])
def test_load_log_leaves_out_a_requirement_stored_with_member_lists(
    source: Callable[..., SelectedSource],
    stored_with_member_list: Callable[[SelectedSource], dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    as_string: bool,
) -> None:
    # A session that crosses the release holds sources with fields
    # `SelectedSource` no longer has. Their requirement loads as never
    # discovered, its source-less `no_data` record included, since that
    # answer would otherwise stand in for the later `definite` one. The
    # requirement beside it, stored in the current shape, is untouched.
    log = DiscoveryLog()
    for requirement_id, status, found in (
        ("annual_debt_service", DiscoveryStatus.NO_DATA, ()),
        ("annual_debt_service", DiscoveryStatus.DEFINITE, (source(),)),
        ("exports", DiscoveryStatus.DEFINITE, (source(spec_id=SPEC_OTHER),)),
    ):
        record = DiscoveryRecord(
            revision="r1",
            tool="search",
            requirement_id=requirement_id,
            requirement_key="k",
            status=status,
            spec_ids=tuple(item.spec_id for item in found),
        )
        log = log.with_record(record, found)
    state: dict[str, Any] = {}
    save_log(state, log)
    stored = state[DISCOVERY_STATE_KEY]
    stored["sources"]["annual_debt_service"][SPEC_KENYA] = stored_with_member_list(source())
    if as_string:
        stored = {**stored, "records": [r.model_dump(mode="json") for r in log.records]}
        state[DISCOVERY_STATE_KEY] = json.dumps(stored)

    with caplog.at_level(logging.WARNING, logger="translator.discovery"):
        loaded = load_log(state)

    assert [record.requirement_id for record in loaded.records] == ["exports"]
    assert loaded.sources == {"exports": {SPEC_OTHER: source(spec_id=SPEC_OTHER)}}
    assert loaded.nearest_for("k", "annual_debt_service") == ()
    assert list(loaded.selectable(requirement_keys={"exports": "k"}, pinned={})) == ["exports"]
    (logged,) = caplog.records
    assert logged.levelno == logging.WARNING
    assert logged.args == (
        DISCOVERY_STATE_KEY,
        3,
        2,
        1,
        1,
        "pinned, stat_var_truncated, stat_vars",
    )


def test_load_log_reads_the_string_form_an_earlier_release_wrote(
    source: Callable[..., SelectedSource],
) -> None:
    # An earlier release persisted `DiscoveryLog.model_dump_json()` at
    # `DISCOVERY_STATE_KEY`; a `DatabaseSessionService`-backed session
    # written by that release and read by this one must not raise.
    first = DiscoveryRecord(
        revision="r1",
        tool="search",
        requirement_id="annual_debt_service",
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_KENYA,),
    )
    seeded = DiscoveryLog().with_record(first, (source(),))
    second = DiscoveryRecord(
        revision="r1",
        tool="describe",
        requirement_id="annual_debt_service",
        spec_ids=(SPEC_KENYA,),
    )
    seeded = seeded.with_record(second)
    state: dict[str, Any] = {DISCOVERY_STATE_KEY: seeded.model_dump_json()}

    loaded = load_log(state)

    assert [r.sequence for r in loaded.records] == [0, 1]
    assert [r.tool for r in loaded.records] == ["search", "describe"]
    assert loaded.sources["annual_debt_service"][SPEC_KENYA].spec_id == SPEC_KENYA

    third = DiscoveryRecord(
        revision="r1",
        tool="search",
        requirement_id="annual_debt_service",
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_KENYA,),
    )
    appended = loaded.with_record(third)
    save_log(state, appended)

    reloaded = load_log(state)
    assert [r.sequence for r in reloaded.records] == [0, 1, 2]
    assert [r.tool for r in reloaded.records] == ["search", "describe", "search"]
    assert reloaded.sources["annual_debt_service"][SPEC_KENYA].spec_id == SPEC_KENYA


def _pinning(spec_id: str, stat_var: str) -> SelectedSource:
    """A source whose identity pins `stat_var`, the one StatVar `describe` can name for it."""
    return SelectedSource(
        spec_id=spec_id,
        sentence="s",
        has_data=True,
        served_count=1,
        identity={"shape": {"kind": "stat_var", "id": stat_var}},
    )


def _log_with_known_source(revision: str, requirement_id: str, ref: str) -> DiscoveryLog:
    """A log recording one `definite` source for `requirement_id` naming `ref`.

    Lets a `fetch_definitions` test drive an id `data_server`'s canned
    `describe` recognises, without depending on what its `search` happens
    to resolve.
    """
    record = DiscoveryRecord(
        revision=revision,
        tool="search",
        requirement_id=requirement_id,
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_KENYA,),
    )
    source = _pinning(SPEC_KENYA, ref)
    return DiscoveryLog().with_record(record, (source,))


async def test_fetch_definitions_enriches_a_definite_sources_own_definition(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    await _discover(surface, context)  # records the source naming "DebtServicePaid"
    revision = _revision(context)
    key = _requirement_key(context)
    log = load_log(context.state)
    source = log.sources["annual_debt_service"][SPEC_KENYA]

    pairs = await fetch_definitions(
        surface,
        log,
        revision=revision,
        requirement_keys={"annual_debt_service": key},
        sources={"annual_debt_service": source},
    )
    enriched = _apply(log, pairs)

    assert enriched.sources["annual_debt_service"][SPEC_KENYA].definition == "d"
    describe_record = enriched.records[-1]
    assert describe_record.tool == "describe"
    assert describe_record.requirement_id == "annual_debt_service"
    assert describe_record.spec_ids == (SPEC_KENYA,)


@pytest.mark.parametrize(
    "family",
    [
        {"served_count": 2, "identity": {"shape": {"kind": "statvar_schema"}}},
        # A described identity serving one StatVar names no dcid to describe.
        {"identity": {"shape": {"kind": "statvar_schema"}}},
    ],
    ids=["a-described-family", "one-served-by-a-described-identity"],
)
async def test_fetch_definitions_describes_only_a_pinned_identity(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
    family: dict[str, Any],
) -> None:
    """A family's definition would be one member's, which a model would read
    as the whole family's, and a described identity names no dcid, so only a
    source whose identity pins its StatVar is described."""
    context = context_holding(draft())
    await _discover(surface, context)
    log = load_log(context.state)
    source = log.sources["annual_debt_service"][SPEC_KENYA]
    assert pinned_stat_var(source.identity) == "DebtServicePaid"
    family = source.model_copy(update=family)

    pairs = await fetch_definitions(
        surface,
        log,
        revision=_revision(context),
        requirement_keys={"annual_debt_service": _requirement_key(context)},
        sources={"annual_debt_service": family},
    )

    assert pairs == ()


async def test_fetch_definitions_reads_the_memo_on_a_second_call_for_the_same_key(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    # Stands in for a `COMPILE_FAILED` retry within one plan revision: the
    # requirement's own `requirement_key` has not changed, so the second call
    # must read `enriched`'s memo rather than describing "DebtServicePaid"
    # again.
    context = context_holding(draft())
    await _discover(surface, context)
    revision = _revision(context)
    key = _requirement_key(context)
    log = load_log(context.state)
    counting = _CountingSurface(surface)

    once = _apply(
        log,
        await fetch_definitions(
            counting,
            log,
            revision=revision,
            requirement_keys={"annual_debt_service": key},
            sources={"annual_debt_service": log.sources["annual_debt_service"][SPEC_KENYA]},
        ),
    )
    twice = _apply(
        once,
        await fetch_definitions(
            counting,
            once,
            revision=revision,
            requirement_keys={"annual_debt_service": key},
            sources={"annual_debt_service": once.sources["annual_debt_service"][SPEC_KENYA]},
        ),
    )

    assert counting.calls.get("describe") == 1
    assert twice.sources["annual_debt_service"][SPEC_KENYA].definition == "d"


async def test_fetch_definitions_leaves_the_definition_unset_on_a_refused_describe(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    revision = _revision(context)
    key = _requirement_key(context)
    log = _log_with_known_source(revision, "annual_debt_service", "refuse")
    source = log.sources["annual_debt_service"][SPEC_KENYA]

    pairs = await fetch_definitions(
        surface,
        log,
        revision=revision,
        requirement_keys={"annual_debt_service": key},
        sources={"annual_debt_service": source},
    )
    enriched = _apply(log, pairs)

    assert enriched.sources["annual_debt_service"][SPEC_KENYA].definition is None
    describe_record = enriched.records[-1]
    assert describe_record.tool == "describe"
    assert describe_record.detail == "describe refused"
    assert describe_record.spec_ids == (SPEC_KENYA,)


async def test_fetch_definitions_records_an_unreachable_describe_without_raising(
    draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    revision = _revision(context)
    key = _requirement_key(context)
    log = _log_with_known_source(revision, "annual_debt_service", "DebtServicePaid")
    source = log.sources["annual_debt_service"][SPEC_KENYA]
    unreachable = McpSurface(sources={"data-mcp": "http://127.0.0.1:9/mcp"})

    pairs = await fetch_definitions(
        unreachable,
        log,
        revision=revision,
        requirement_keys={"annual_debt_service": key},
        sources={"annual_debt_service": source},
    )
    enriched = _apply(log, pairs)

    assert enriched.sources["annual_debt_service"][SPEC_KENYA].definition is None
    describe_record = enriched.records[-1]
    assert describe_record.tool == "describe"
    assert describe_record.transport_failed is True


def _valid_spec_id(n: int) -> str:
    return f"spec_{n:032x}"


async def test_fetch_definitions_retries_a_batch_that_refuses_over_one_unheld_dcid(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    """`describe`'s own batch cap is `DESCRIBE_ID_BATCH_MAX` (8), and a batch
    mixes dcids across several requirements' own sources
    (`fetch_definitions`'s own docstring). A single unheld dcid among the
    other seven must not zero every one of their definitions: `describe`
    refuses the whole call `is_error=True` but names the offending dcid in
    `structured_content["fix"]["dcids"]`, which `fetch_definitions` has to
    read and retry without, rather than treating the whole chunk as
    undescribable.
    """
    context = context_holding(draft())
    revision = _revision(context)
    requirement_ids = [f"req_{i}" for i in range(8)]
    keys = {requirement_id: f"key_{requirement_id}" for requirement_id in requirement_ids}
    sources = {
        requirement_id: _pinning(_valid_spec_id(i), "refuse" if i == 3 else f"ok_{i}")
        for i, requirement_id in enumerate(requirement_ids)
    }
    counting = _CountingSurface(surface)

    pairs = await fetch_definitions(
        counting, DiscoveryLog(), revision=revision, requirement_keys=keys, sources=sources
    )
    enriched = _apply(DiscoveryLog(), pairs)

    assert counting.calls.get("describe") == 2
    for i, requirement_id in enumerate(requirement_ids):
        if i == 3:
            # `with_record` records no source at all when `fetch_definitions`
            # found no definition for it -- unlike every other requirement
            # here, "req_3" never gets an entry in `sources`.
            assert requirement_id not in enriched.sources
        else:
            spec_id = sources[requirement_id].spec_id
            assert enriched.sources[requirement_id][spec_id].definition == "d"
    refused_record = next(r for r in enriched.records if r.requirement_id == "req_3")
    assert refused_record.detail is None  # an overall-successful retry, not "describe refused"


async def test_definite_source_ids_includes_an_unpinned_definite_answer(
    surface: McpSurface, draft: DraftFactory, context_holding: ContextHolding
) -> None:
    context = context_holding(draft())
    await _discover(surface, context)
    keys = _requirement_keys(context, "annual_debt_service")

    definite_ids = load_log(context.state).definite_source_ids(requirement_keys=keys, pinned={})

    assert definite_ids == {"annual_debt_service"}


async def test_definite_source_ids_includes_a_pin_honoured_from_a_definite_answer(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    await _discover(surface, context)
    keys = _requirement_keys(context, "annual_debt_service")

    definite_ids = load_log(context.state).definite_source_ids(
        requirement_keys=keys, pinned={"annual_debt_service": SPEC_KENYA}
    )

    assert definite_ids == {"annual_debt_service"}


async def test_definite_source_ids_excludes_a_pin_honoured_from_candidates(
    surface: McpSurface,
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # `selectable` honours this pin from a `candidates` answer -- see
    # `test_a_pinned_source_found_among_candidates_is_selectable` -- so it is
    # not `definite`-origin: it got no definition-based confirmation, only a
    # referral the user answered without seeing the source's own definition.
    pinned = requirement(measure="aid disbursements", source_spec_id=SPEC_THIRD)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_THIRD)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    await _discover(surface, context)
    keys = _requirement_keys(context, "annual_debt_service")

    definite_ids = load_log(context.state).definite_source_ids(
        requirement_keys=keys, pinned={"annual_debt_service": SPEC_THIRD}
    )

    assert definite_ids == frozenset()


async def test_definite_source_ids_scans_back_past_a_later_transport_failure(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    """A `DEFINITE` search, then a later re-search for the *same* requirement
    -- nothing stops a redundant re-search of an already-definite requirement
    -- that never reaches the server. `selectable` finds the earlier `DEFINITE`
    record by scanning back past the failed one; `definite_source_ids` has
    to do the same, or `fetch_definitions` is skipped for the requirement
    and the compiled source ships with `definition=None` despite a definite
    source having been found.
    """
    context = context_holding(draft())
    first = await _discover(surface, context)
    assert first["outcome"] == "ok"

    unreachable = McpSurface(sources={"data-mcp": "http://127.0.0.1:9/mcp"})
    second = await _discover(unreachable, context)
    assert second["code"] == "surface_unavailable"

    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    (record_a, record_b) = log.records
    assert record_a.status is DiscoveryStatus.DEFINITE
    assert record_b.status is None and record_b.transport_failed is True
    assert list(log.selectable(requirement_keys=keys, pinned={})) == ["annual_debt_service"]

    definite_ids = log.definite_source_ids(requirement_keys=keys, pinned={})

    assert definite_ids == {"annual_debt_service"}


@pytest.mark.parametrize("posted", [True, False], ids=["with-groups", "identity-alone"])
async def test_a_spec_over_a_sets_places_resolved_again_names_the_set_only_when_posted(
    posted: bool,
) -> None:
    # A recorded identity or a windowed copy of "African countries" is resolved
    # again through `resolve_definition`. Posted with its groups, Search names
    # Africa again; posted alone, Africa is gone from the answer.
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    recorded = source_from_spec_block(block)
    assert recorded.identity is not None
    assert recorded.groups
    resolving = DataMcpDefinitions(InMemoryDefinitionStore(), search=GroupsSearchStub([block]))

    part = await resolve_definition(
        resolving,
        spec_id=recorded.spec_id,
        identity=recorded.identity,
        groups=recorded.groups if posted else (),
    )

    again = source_from_spec_block(resolved_spec(part, spec_id=recorded.spec_id))
    assert again.groups == (recorded.groups if posted else ())


# --- Entity refs ---------------------------------------------------------------


def _g7_members_source() -> SelectedSource:
    """G7 donors to Kenya as `search` binds them, labelled as the graph labels them."""
    return SelectedSource(
        spec_id="spec_" + "c" * 32,
        sentence="ODA from the G7 to Kenya",
        has_data=True,
        entities=(
            ResolvedEntity(role="origin", ref="country/USA", label="United States of America"),
            ResolvedEntity(role="origin", ref="country/JPN", label="Japan"),
            ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
        ),
    )


def _members_requirement(*names: str) -> DatasetRequirement:
    return DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="official development assistance",
        entities=[Entity(name=name, role="donor") for name in names],
        period=Period(start=2015, end=2024),
    )


async def test_names_no_label_matches_are_resolved_once_and_only_a_resolved_one_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[dict[str, list[str]]] = []

    async def resolve(scope: dict[str, list[str]]) -> dict[str, dict[str, object]]:
        asked.append({role: list(names) for role, names in scope.items()})
        outcomes: dict[str, object] = {
            "United States": resolution.ResolvedEntity(
                dcid="country/USA", name="United States", entity_type="Country"
            ),
            "Bermuda": resolution.NotFound(near_matches=[]),
        }
        return {role: {name: outcomes[name] for name in names} for role, names in scope.items()}

    monkeypatch.setattr(resolution, "resolve_entities", resolve)
    requirement = _members_requirement("United States", "Japan", "Bermuda")
    key = requirement_revision(requirement)
    source = _g7_members_source()

    pairs = await resolve_entity_refs(
        DiscoveryLog(),
        revision="rev",
        requirements=[requirement],
        requirement_keys={"oda": key},
        sources={"oda": source},
    )

    # Japan matches its label, so only the two that do not are asked.
    assert asked == [{"names": ["United States", "Bermuda"]}]
    ((record, (updated,)),) = pairs
    assert (record.tool, record.requirement_key, record.spec_ids) == (
        "resolve",
        key,
        (source.spec_id,),
    )
    assert updated.entity_refs == {"united states": "country/USA"}

    log = DiscoveryLog().with_record(record, (updated,))
    again = await resolve_entity_refs(
        log,
        revision="rev",
        requirements=[requirement],
        requirement_keys={"oda": key},
        sources={"oda": updated},
    )

    assert again == ()
    assert len(asked) == 1


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError(status=None, code=None, detail="the graph door timed out"),
        KeyError("results"),
        json.JSONDecodeError("Expecting value", "<html>", 0),
        ValueError("unrecognised resolution status 'merged'"),
    ],
    ids=["transport", "no-results", "not-json", "unknown-status"],
)
async def test_a_failed_resolve_records_the_attempt_adds_no_ref_and_is_asked_again(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    calls: list[dict[str, list[str]]] = []

    async def resolve(scope: dict[str, list[str]]) -> dict[str, dict[str, object]]:
        calls.append(scope)
        raise error

    monkeypatch.setattr(resolution, "resolve_entities", resolve)
    requirement = _members_requirement("United States")
    keys = {"oda": requirement_revision(requirement)}
    source = _g7_members_source()

    ((record, sources),) = await resolve_entity_refs(
        DiscoveryLog(),
        revision="rev",
        requirements=[requirement],
        requirement_keys=keys,
        sources={"oda": source},
    )

    assert record.transport_failed is True
    assert sources == ()
    # A failed resolve is not a tried one: the next translate asks again.
    await resolve_entity_refs(
        DiscoveryLog().with_record(record, sources),
        revision="rev",
        requirements=[requirement],
        requirement_keys=keys,
        sources={"oda": source},
    )
    assert len(calls) == 2


# The CRS purpose hierarchy above basic health care (12220) and population policy
# (13010), both leaves, by parent.
_PURPOSE_PARENTS = {
    "CRSPurposeCode12220": "CRSPurposeCode122",
    "CRSPurposeCode13010": "CRSPurposeCode130",
    "CRSPurposeCode122": "CRSPurposeCode120",
    "CRSPurposeCode120": "CRSPurposeCode100",
    "CRSPurposeCode130": "CRSPurposeCode100",
    "CRSPurposeCode100": None,
}


def _code_node(dcid: str) -> CodeNode:
    """`dcid` as the graph's node door answers a CRS purpose code, with its parent and,
    for a code with children, its `children` link."""
    parent = _PURPOSE_PARENTS[dcid]
    children = dcid in _PURPOSE_PARENTS.values()
    return CodeNode.model_validate(
        {
            "built_from": {},
            "code": {
                "code": dcid.removeprefix("CRSPurposeCode"),
                "code_system": {"ref": "CRSPurposeCodeEnum", "label": "CRS Purpose"},
                "parent": None if parent is None else {"ref": parent, "label": None},
                "property": "crsPurpose",
                "synonyms": [],
            },
            "description": None,
            "gaps": [],
            "kind": "code",
            "label": None,
            "links": {"children": f"/api/graph/node/{dcid}/children"} if children else {},
            "ref": dcid,
            "types": ["CRSPurposeCodeEnum"],
        }
    )


def _sectors_requirement(rid: str, *, breakdown: Breakdown | None) -> DatasetRequirement:
    return DatasetRequirement(
        id=rid,
        decision_ids=["measure"],
        measure="health and population ODA",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2018, end=2022),
        stat_vars=StatVarReading.SEPARATE if breakdown is not None else None,
        breakdown=breakdown,
    )


def _health_and_population_source() -> SelectedSource:
    """ODA to Kenya for health and population, its purpose slot bound as Search binds it."""
    return SelectedSource(
        spec_id="spec_" + "d" * 32,
        sentence="ODA to Kenya for health and population",
        has_data=True,
        identity={
            "slots": [
                {
                    "property": "crsPurpose",
                    "binding": {
                        "state": "set",
                        "refs": ["CRSPurposeCode120", "CRSPurposeCode130"],
                    },
                }
            ]
        },
    )


async def test_a_breakdowns_named_codes_are_read_up_to_their_root_once() -> None:
    """Health and population climb to social infrastructure (100) in two reads, and
    the requirement with no breakdown asks for nothing."""
    asked: list[list[str]] = []

    async def describe_nodes(dcids: Any) -> dict[str, CodeNode]:
        asked.append(list(dcids))
        return {dcid: _code_node(dcid) for dcid in dcids}

    by_sector = _sectors_requirement(
        "by_sector", breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0)
    )
    total = _sectors_requirement("total", breakdown=None)
    keys = {r.id: requirement_revision(r) for r in (by_sector, total)}
    source = _health_and_population_source()

    ((record, (updated,)),) = await fetch_code_paths(
        revision="rev",
        requirements=[by_sector, total],
        requirement_keys=keys,
        sources={"by_sector": source, "total": source},
        describe_nodes=describe_nodes,
    )

    assert asked == [["CRSPurposeCode120", "CRSPurposeCode130"], ["CRSPurposeCode100"]]
    assert (record.tool, record.requirement_id, record.transport_failed) == (
        "code_paths",
        "by_sector",
        False,
    )
    assert updated.code_paths == {
        "CRSPurposeCode120": ("CRSPurposeCode100", "CRSPurposeCode120"),
        "CRSPurposeCode130": ("CRSPurposeCode100", "CRSPurposeCode130"),
    }
    assert updated.leaf_codes == ()
    again = await fetch_code_paths(
        revision="rev",
        requirements=[by_sector],
        requirement_keys=keys,
        sources={"by_sector": updated},
        describe_nodes=describe_nodes,
    )
    assert again == ()
    assert len(asked) == 2


async def test_a_named_code_with_no_children_link_reads_as_a_leaf() -> None:
    """Basic health care (12220) states no `children` link and health (120) does, so
    only basic health care is recorded as a leaf."""

    async def describe_nodes(dcids: Any) -> dict[str, CodeNode]:
        return {dcid: _code_node(dcid) for dcid in dcids}

    by_sector = _sectors_requirement(
        "by_sector", breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0)
    )
    named = _health_and_population_source().model_copy(
        update={
            "identity": {
                "slots": [
                    {
                        "property": "crsPurpose",
                        "binding": {
                            "state": "set",
                            "refs": ["CRSPurposeCode120", "CRSPurposeCode12220"],
                        },
                    }
                ]
            }
        }
    )

    ((_, (updated,)),) = await fetch_code_paths(
        revision="rev",
        requirements=[by_sector],
        requirement_keys={"by_sector": requirement_revision(by_sector)},
        sources={"by_sector": named},
        describe_nodes=describe_nodes,
    )

    assert updated.leaf_codes == ("CRSPurposeCode12220",)


async def test_a_code_read_with_no_time_left_asks_for_no_node() -> None:
    asked: list[list[str]] = []

    async def describe_nodes(dcids: Any) -> dict[str, CodeNode]:
        asked.append(list(dcids))
        return {dcid: _code_node(dcid) for dcid in dcids}

    by_sector = _sectors_requirement(
        "by_sector", breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0)
    )
    token = current_discovery_time_left.set(lambda: 0.0)
    try:
        ((record, sources),) = await fetch_code_paths(
            revision="rev",
            requirements=[by_sector],
            requirement_keys={"by_sector": requirement_revision(by_sector)},
            sources={"by_sector": _health_and_population_source()},
            describe_nodes=describe_nodes,
        )
    finally:
        current_discovery_time_left.reset(token)

    assert asked == []
    assert (record.transport_failed, sources) == (True, ())


async def test_a_failed_code_read_records_the_attempt_and_adds_no_path() -> None:
    async def describe_nodes(dcids: Any) -> dict[str, CodeNode]:
        raise UpstreamError(status=None, code=None, detail="the graph door timed out")

    by_sector = _sectors_requirement(
        "by_sector", breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0)
    )

    ((record, sources),) = await fetch_code_paths(
        revision="rev",
        requirements=[by_sector],
        requirement_keys={"by_sector": requirement_revision(by_sector)},
        sources={"by_sector": _health_and_population_source()},
        describe_nodes=describe_nodes,
    )

    assert record.transport_failed is True
    assert sources == ()


async def test_excluded_source_and_chosen_reference_reach_search(
    surface, draft, requirement, derived_requirement, search_calls, context_holding
):
    ask = requirement(source="CRS", source_exclude=True, source_ref="provenance/CRS")
    context = context_holding(draft(evidence_requirements=[ask, derived_requirement()]))
    await _discover(surface, context)
    assert search_calls[0]["parts"][0]["sources"] == [
        {"text": "CRS", "exclude": True, "ref": "provenance/CRS"}
    ]
    assert requirement_revision(ask) != requirement_revision(
        ask.model_copy(update={"source_exclude": False})
    )
    assert requirement_revision(ask) != requirement_revision(
        ask.model_copy(update={"source_ref": None})
    )


async def test_a_source_code_reaches_search_upper_cased(
    surface, draft, requirement, derived_requirement, search_calls, context_holding
):
    ask = requirement(source_code=" dt.tds.degg.cd ")
    context = context_holding(draft(evidence_requirements=[ask, derived_requirement()]))
    await _discover(surface, context)
    assert search_calls[0]["parts"][0]["source_code"] == "DT.TDS.DEGG.CD"
    assert requirement_revision(ask) != requirement_revision(
        ask.model_copy(update={"source_code": None})
    )


_DEGG = "DT.TDS.DEGG.CD"
_MIDA = "DT.NFL.MIDA.CD"
_IMFC = "DT.NFL.IMFC.CD"


@pytest.mark.parametrize(
    ("held", "asked", "searched"),
    [
        ([{"code": _DEGG, "exact": True}], " dt.tds.degg.cd ", None),
        (
            [{"code": "dt.dod.dect.cd", "exact": False}, {"code": _DEGG, "exact": True}],
            _DEGG,
            None,
        ),
        ([{"code": _DEGG, "exact": True}], "DT.DOD.DECT.CD", "DT.DOD.DECT.CD"),
        ([], _DEGG, _DEGG),
        # A family listing two codes, neither marked exact, is neither series.
        ([{"code": _MIDA}, {"code": _IMFC}], _MIDA, _MIDA),
        ([{"code": _DEGG}], _DEGG, _DEGG),
        ([{"code": _DEGG, "exact": False}], _DEGG, _DEGG),
        ([{"code": _DEGG, "exact": "true"}], _DEGG, _DEGG),
        ([{"code": _DEGG, "exact": 1}], _DEGG, _DEGG),
    ],
    ids=[
        "exact",
        "exact_among_several",
        "another_code",
        "no_codes_on_record",
        "listed_not_exact",
        "no_exact_key",
        "exact_false",
        "exact_string",
        "exact_integer",
    ],
)
async def test_a_pin_is_reused_only_when_its_dataset_is_the_source_code_series(
    surface,
    draft,
    pinning_decision,
    requirement,
    derived_requirement,
    search_calls,
    context_holding,
    held,
    asked,
    searched,
):
    ask = requirement(source_code=asked, source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[ask, derived_requirement()],
        )
    )
    stat_vars = {"member_count": 1, "member_count_exact": True, "served_count": 1}
    if held:
        stat_vars["source_codes"] = [{"provenance": "provenance/IDS", **entry} for entry in held]
    _shown(
        context,
        SPEC_KENYA,
        _spec(
            SPEC_KENYA,
            "debt service paid by Kenya",
            coverage=_matching_window(),
            stat_vars=stat_vars,
        ),
    )

    await _discover(surface, context)

    assert [call["parts"][0].get("source_code") for call in search_calls] == (
        [searched] if searched else []
    )
    assert [record.tool for record in load_log(context.state).records] == (
        ["search"] if searched else ["pin"]
    )


@pytest.mark.parametrize(
    ("code", "variable", "kind"),
    [
        (_MIDA, "debt service", "no_data"),
        ("DT.NFL.NONE.CD", "debt service", "no_data"),
        # A refusal that does not name the code leaves the pin not found.
        (None, "unicorn debt service", "pinned_source_not_found"),
        (_DEGG, "unicorn debt service", "pinned_source_not_found"),
    ],
    ids=["shares_variable", "not_held", "no_code", "code_with_another_refusal"],
)
async def test_a_pinned_requirement_is_told_its_pin_is_missing_unless_search_refused_its_code(
    surface,
    draft,
    pinning_decision,
    requirement,
    derived_requirement,
    search_calls,
    context_holding,
    code,
    variable,
    kind,
):
    ask = requirement(source_code=code, measure=variable, source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[ask, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert len(search_calls) == 1
    assert result["action"]["arguments"]["kind"] == kind


async def test_a_pinned_requirement_is_told_its_code_is_ambiguous(
    draft, pinning_decision, requirement, derived_requirement, context_holding
):
    groups = [
        {
            "text": _DEGG,
            "candidates": [
                {"ref": "provenance/a", "label": "A", "kind": "provenance"},
                {"ref": "provenance/b", "label": "B", "kind": "provenance"},
            ],
        }
    ]
    envelope = {
        "parts": [
            {
                "status": "no_data",
                "no_data": {
                    "category": "unresolved",
                    "reason": "source_code_ambiguous",
                    "ambiguous_sources": groups,
                },
            }
        ]
    }
    surface = FixtureSurface(
        schemas=[],
        responses={
            "search": ToolCallResult(
                is_error=False, structured_content={}, meta={SEARCH_ENVELOPE_META_KEY: envelope}
            )
        },
    )
    ask = requirement(source_code=_DEGG, source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[ask, derived_requirement()],
        )
    )

    result = await _discover(surface, context)

    assert result["action"]["arguments"]["kind"] == "source_name_ambiguous"


@pytest.mark.parametrize(
    ("reason", "text"),
    [("source_ambiguous", "shared"), ("source_code_ambiguous", "DT.TDS.DEGG.CD")],
)
async def test_discovery_preserves_source_name_candidates_for_the_referral(
    draft, context_holding, reason, text
):
    groups = [
        {
            "text": text,
            "candidates": [
                {"ref": "provenance/a", "label": "A", "kind": "provenance"},
                {"ref": "provenance/b", "label": "B", "kind": "provenance"},
            ],
        }
    ]
    envelope = {
        "parts": [
            {
                "status": "no_data",
                "no_data": {
                    "category": "unresolved",
                    "reason": reason,
                    "ambiguous_sources": groups,
                },
            }
        ]
    }
    surface = FixtureSurface(
        schemas=[],
        responses={
            "search": ToolCallResult(
                is_error=False, structured_content={}, meta={SEARCH_ENVELOPE_META_KEY: envelope}
            )
        },
    )
    context = context_holding(draft())
    result = await _discover(surface, context)
    assert result["ambiguous_sources"] == groups
    assert result["action"]["arguments"]["kind"] == "source_name_ambiguous"
    assert (
        load_log(context.state).records[-1].ambiguous_sources[0].candidates[0].ref == "provenance/a"
    )


async def test_saved_alias_confirmation_survives_identity_resolution(
    surface, draft, pinning_decision, requirement, derived_requirement, context_holding
):
    answered = _KENYA_OVER["2010-2023"]
    spec_id = answered["spec_id"]
    mention = ResolvedSourceMention(text="catalogue alias", ref="provenance/IDS")
    recorded = RecordedIdentity(
        identity=answered["definition"]["identity"], resolved_sources=(mention,)
    )
    ask = requirement(
        source_spec_id=spec_id, source="catalogue alias", period=Period(start=2010, end=2023)
    )
    context = context_holding(
        draft(
            decisions=[pinning_decision(spec_id)],
            evidence_requirements=[ask, derived_requirement()],
        )
    )
    token = current_recorded_identity.set({spec_id: recorded}.get)
    resolving = DataMcpDefinitions(InMemoryDefinitionStore(), served=[answered])
    try:
        await _discover(_SearchAndResolve(surface, resolving), context)
    finally:
        current_recorded_identity.reset(token)
    selected = load_log(context.state).sources[ask.id][spec_id]
    assert mention in selected.resolved_sources


@pytest.mark.parametrize(
    ("exclude", "ref"), [(True, None), (False, "provenance/IDS"), (False, None)]
)
async def test_a_new_source_constraint_refreshes_an_existing_pin(
    draft, pinning_decision, requirement, derived_requirement, context_holding, exclude, ref
):
    ask = requirement(
        source="IDS", source_exclude=exclude, source_ref=ref, source_spec_id=SPEC_KENYA
    )
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[ask, derived_requirement()],
        )
    )
    block = _spec(SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window())
    _shown(context, SPEC_KENYA, block)
    mention = {"text": "IDS", "ref": ref, "polarity": "exclude" if exclude else "include"}
    envelope = {
        "parts": [{"status": "definite", "spec": block, "restatement": {"sources": [mention]}}]
    }
    surface = FixtureSurface(
        schemas=[],
        responses={
            "search": ToolCallResult(
                is_error=False, structured_content={}, meta={SEARCH_ENVELOPE_META_KEY: envelope}
            )
        },
    )
    result = await _discover(surface, context)
    assert result["status"] == "definite"
    log = load_log(context.state)
    assert [record.tool for record in log.records] == ["search"]
    assert log.sources[ask.id][SPEC_KENYA].resolved_sources == (
        ResolvedSourceMention.model_validate(mention),
    )
