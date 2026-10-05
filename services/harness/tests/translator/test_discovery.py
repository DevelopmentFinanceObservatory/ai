"""Tests for `translator.discovery`: bounded per revision, built from the requirement.

The tools are exercised through the MCP protocol against the canned
`data-mcp` server, so what is asserted is what a real server would see.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from compiler.sources import DiscoveryStatus, source_from_spec_block
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSurface,
    ToolSurfaceUnavailableError,
)
from data_mcp.definition_store import DefinitionStoreUnavailable, InMemoryDefinitionStore
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.definition_writes import identity_definition
from google.adk.tools.function_tool import FunctionTool
from planner.plan import (
    ENVELOPE_SPEC_BLOCK_STATE_KEY,
    ENVELOPE_SPEC_STATE_KEY,
    ENVELOPE_SPEC_WARNINGS_STATE_KEY,
    PLAN_STATE_KEY,
    SEEDED_SOURCES_STATE_KEY,
    SEEDS_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Plan,
    PlanStatus,
    SeededSource,
    Unit,
    plan_revision,
    save_dropped_seeds,
    save_seeded_sources,
    seeded_sources,
)
from planner.records import Period
from planner.tool_results import ProblemCode
from planner.workflow import OpenEntityAxis, SelectedSource, ToolId
from translator.discovery import (
    DISCOVERY_CALL_BUDGET,
    DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
    DISCOVERY_STATE_KEY,
    DefinitionNotResolvedError,
    DiscoveryLog,
    DiscoveryRecord,
    PartFault,
    PartNotResolvedError,
    build_discovery_tools,
    compact_source,
    described_sources,
    fetch_definitions,
    load_log,
    requirement_revision,
    resolve_definition,
    resolved_spec,
    save_log,
    settled_sources,
)
from translator.referral import ReferralKind

from ..data_mcp_definitions import DataMcpDefinitions, SearchStub
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

SEEDED_REQUEST_FIXTURE = (
    Path(__file__).parents[1] / "wire_fixtures" / "studio" / "request-11-turn-seeded.json"
)


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


@pytest.mark.parametrize(
    ("entities", "sent"),
    [
        ([Entity(name="Kenya", role="debtor")], [{"text": "Kenya", "role": "debtor"}]),
        ([], None),
    ],
)
async def test_a_party_stated_in_across_reaches_search_as_no_entity(
    surface: McpSurface,
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    search_calls: list[dict[str, Any]],
    context_holding: ContextHolding,
    entities: list[Entity],
    sent: list[dict[str, str]] | None,
) -> None:
    # Search defaults the scope of a side the part names nobody on, as it
    # does for the raw question, so `across` sends nothing of its own.
    across = requirement(entities=entities, across="creditor")
    context = context_holding(draft(evidence_requirements=[across, derived_requirement()]))

    await _discover(surface, context)

    (call,) = search_calls
    (part,) = call["parts"]
    assert part.get("entities") == sent
    assert "creditor" not in str(part)


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
    across = requirement(across="creditor")
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


async def test_a_pin_matching_the_resolve_id_builds_the_source_with_no_search(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The pin names the exact id this turn's own resolve settled on
    # (`ENVELOPE_SPEC_STATE_KEY`), the plan states one dataset requirement,
    # the resolve carried no blocking warning, and the block's own window
    # agrees with the requirement's period: the source comes straight from
    # the resolve's own block (`ENVELOPE_SPEC_BLOCK_STATE_KEY`), with no
    # `search` call at all.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA
    context.state[ENVELOPE_SPEC_BLOCK_STATE_KEY] = _spec(
        SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()
    )

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


async def test_a_pin_matching_the_resolve_id_still_searches_with_two_dataset_requirements(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The resolve's own search read one variable. A plan stating two dataset
    # requirements gives no way to tell which one that search answered, so
    # the pin still confirms through a fresh `search` rather than reusing
    # the resolve's block for either one.
    source_decision = pinning_decision(
        SPEC_KENYA,
        slot="source",
        description="Which source to use.",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
    )
    req_b = requirement(
        id="req_b",
        measure="other debt figures",
        source_spec_id=SPEC_KENYA,
        decision_ids=["measure", "source"],
    )
    calc_b = derived_requirement(id="calc_b", input_id="req_b")
    context = context_holding(
        draft(
            decisions=[decision(), source_decision],
            evidence_requirements=[requirement(), derived_requirement(), req_b, calc_b],
        )
    )
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA
    context.state[ENVELOPE_SPEC_BLOCK_STATE_KEY] = _spec(
        SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()
    )

    result = await _discover(surface, context, requirement_id="req_b")

    assert len(search_calls) == 1
    assert result["status"] == "definite"


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
        stat_vars=("Var_a", "Var_b"),
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
    # period, so the copies of one seed describe differently and share only
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


def test_described_sources_of_one_source_is_its_compact_form() -> None:
    source = _described(SPEC_KENYA, "debt service paid by Kenya", "debt service")

    assert described_sources([source]) == [compact_source(source)]
    assert described_sources([]) == []


def _seed_block() -> dict[str, Any]:
    """The seeded request fixture's Spec block, a real Search API block that
    names no year: exact coverage and no window."""
    block = json.loads(SEEDED_REQUEST_FIXTURE.read_text())["seed_spec"][0]["spec"]
    assert block["resolved"]["coverage"]["kind"] == "exact"
    assert "window" not in block["resolved"]["coverage"]
    return block


async def test_a_pin_naming_a_seeded_spec_is_honoured_from_its_source_whatever_its_window(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # `harness.graph`'s `seed` node records each seed's source, read from the
    # Spec block the user saw. Each seed is one dataset the user chose, so a
    # pin naming one is honoured from it with no `search`, whatever else the
    # plan states, and a block with no window is every year with data.
    block = _seed_block()
    context = _two_dataset_context_pinning(
        block["spec_id"],
        draft,
        decision,
        pinning_decision,
        requirement,
        derived_requirement,
        context_holding,
    )
    save_seeded_sources(context.state, [SeededSource(source=source_from_spec_block(block))])

    result = await _discover(surface, context, requirement_id="req_b")

    assert search_calls == []
    assert result["status"] == "definite"
    assert "action" not in result
    assert result["sources"][0]["spec_id"] == block["spec_id"]
    assert load_log(context.state).records[0].tool == "pin"
    # The seed's whole part of the result is its description, stamped with the
    # requirement's measure: what `tests/test_context_budget.py` holds to a ceiling.
    plan = Plan.model_validate_json(context.state[PLAN_STATE_KEY])
    measure = cast(DatasetRequirement, plan.requirement("req_b")).measure
    seeded = source_from_spec_block(block).model_copy(update={"searched_as": measure})
    assert result["sources"] == [compact_source(seeded)]


def _copy_over(seed: SelectedSource, block: dict[str, Any]) -> SeededSource:
    """The copy of `seed` over `block`'s years, as `planner.agent.propose_plan` records it."""
    return SeededSource(
        source=SelectedSource(
            spec_id=block["spec_id"],
            sentence=seed.sentence,
            has_data=None,
            stat_vars=seed.stat_vars,
            identity=block["definition"]["identity"],
        ),
        windowed_from=seed.spec_id,
    )


def _context_pinning_the_copy_over_2010_to_2023(
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> tuple[Any, SelectedSource, SeededSource]:
    """A context whose `req_b` pins the copy over 2010-2023 of a seed over 2015-2023, with both."""
    seed = source_from_spec_block(_KENYA_OVER["2015-2023"])
    copy = _copy_over(seed, _KENYA_OVER["2010-2023"])
    context = _two_dataset_context_pinning(
        copy.source.spec_id,
        draft,
        decision,
        pinning_decision,
        requirement,
        derived_requirement,
        context_holding,
    )
    save_seeded_sources(context.state, [SeededSource(source=seed), copy])
    return context, seed, copy


async def test_a_pin_naming_a_windowed_seed_is_read_from_searchs_answer_for_its_period(
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    definition_store: InMemoryDefinitionStore,
) -> None:
    # `propose_plan` pins a seed over the requirement's own period and records
    # the copy with no fact about that period. Search states them: its answer
    # for the copy's identity is the whole source, and data-mcp stores the
    # copy's definition as it answers, in its own store, apart from the one
    # this process holds, as when it runs as its own service.
    context, seed, copy = _context_pinning_the_copy_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )
    answered = _KENYA_OVER["2010-2023"]
    windowed = copy.source.spec_id
    held_before = context.state[SEEDED_SOURCES_STATE_KEY]
    service = InMemoryDefinitionStore()
    data_mcp = DataMcpDefinitions(service, served=[answered])

    result = await _discover(data_mcp, context, requirement_id="req_b")

    assert result["status"] == "definite"
    recorded = load_log(context.state).sources["req_b"][windowed]
    assert recorded == source_from_spec_block(answered).model_copy(
        update={"searched_as": recorded.searched_as}
    )
    assert result["sources"] == [compact_source(recorded)]
    # Search's facts for 2010-2023, where the seed's are for 2015-2023.
    assert seed.coverage_window == Period(start=2015, end=2023)
    assert recorded.coverage_window == Period(start=2010, end=2023)
    assert recorded.has_data is True
    assert "2010 to 2023" in recorded.sentence
    assert recorded.sentence != seed.sentence
    assert (seed.stat_var_member_count, recorded.stat_var_member_count) == (89, 96)
    # Search is asked with the copy's identity and the seed's elected StatVars.
    assert data_mcp.calls == [
        {
            "spec_id": windowed,
            "identity": copy.source.identity,
            "stat_vars": list(seed.stat_vars),
        }
    ]
    assert await service.get(windowed, schema_version=1) == identity_definition(
        answered["definition"]
    )
    assert await definition_store.get(windowed, schema_version=1) is None
    # The copy in session state is the record `propose_plan` made.
    assert context.state[SEEDED_SOURCES_STATE_KEY] == held_before
    assert seeded_sources(context.state)[1] == copy


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
async def test_a_windowed_seed_search_could_not_be_asked_about_is_retryable(
    surface_down: Callable[[], ToolSurface],
    said: str,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    context, seed, _ = _context_pinning_the_copy_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )

    result = await _discover(surface_down(), context, requirement_id="req_b")

    assert result["outcome"] == "refused"
    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value
    assert result["message"].startswith(
        f'Search could not be asked for "{seed.sentence}" over 2010 to 2023: '
    )
    assert said in result["message"]
    assert "req_b" not in load_log(context.state).sources


def _kenya_without_its_definition() -> dict[str, Any]:
    block = dict(_KENYA_OVER["2010-2023"])
    del block["definition"]
    return block


# What Search answers for the copy's identity, how the refusal words it, and the
# referral its action names.
_COPY_ANSWERS: dict[str, tuple[dict[str, Any] | None, str, ReferralKind]] = {
    "refused": (
        None,
        "it refuses the period (request_invalid)",
        ReferralKind.REQUIREMENT_UNMEETABLE,
    ),
    "no-data": (
        {"status": "no_data", "parts": [{"status": "no_data", "no_data": {"reason": "no_series"}}]},
        "it finds no data (no_series)",
        ReferralKind.NO_DATA,
    ),
    "candidates": (
        {"status": "candidates", "parts": [{"status": "candidates", "candidates": {"specs": []}}]},
        "it answers 'candidates'",
        ReferralKind.REQUIREMENT_UNMEETABLE,
    ),
    "another-spec-id": (
        {
            "status": "definite",
            "parts": [{"status": "definite", "spec": _KENYA_OVER["2015-2023"]}],
        },
        f"it resolves the period to another dataset ({_KENYA_OVER['2015-2023']['spec_id']})",
        ReferralKind.REQUIREMENT_UNMEETABLE,
    ),
    "no-identity": (
        {
            "status": "definite",
            "parts": [{"status": "definite", "spec": _kenya_without_its_definition()}],
        },
        "its answer carries no definition to retrieve",
        ReferralKind.REQUIREMENT_UNMEETABLE,
    ),
}


@pytest.mark.parametrize("answer", sorted(_COPY_ANSWERS))
async def test_a_windowed_seed_search_does_not_resolve_to_that_dataset_is_refused(
    answer: str,
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
    definition_store: InMemoryDefinitionStore,
) -> None:
    # The seed resolved when the turn started; the copy is another identity,
    # and Search's answer for it is not that one dataset. The refusal names
    # the dataset and the period and refers the requirement back, and no
    # source is recorded for the pin.
    del definition_store
    context, seed, copy = _context_pinning_the_copy_over_2010_to_2023(
        draft, decision, pinning_decision, requirement, derived_requirement, context_holding
    )
    envelope, fault, kind = _COPY_ANSWERS[answer]
    search = SearchStub(answer={copy.source.spec_id: envelope} if envelope is not None else None)

    result = await _discover(DataMcpDefinitions(search=search), context, requirement_id="req_b")

    message = f'Search finds no single dataset for "{seed.sentence}" over 2010 to 2023: {fault}.'
    assert result["outcome"] == "refused"
    assert result["code"] == ProblemCode.SEARCH_REFUSED.value
    assert result["message"] == message
    assert result["action"] == {
        "call": "return_to_planner",
        "arguments": {"kind": kind.value, "requirement_id": "req_b", "reason": message},
    }
    log = load_log(context.state)
    assert "req_b" not in log.sources
    # A period with no data is recorded with Search's reason code, which the
    # `no_data` referral carries; it spends no search.
    plan = Plan.model_validate_json(context.state[PLAN_STATE_KEY])
    key = requirement_revision(cast(DatasetRequirement, plan.requirement("req_b")))
    assert log.no_data_reason(key, "req_b") == ("no_series" if answer == "no-data" else None)
    assert log.calls(plan_revision(plan)) == 0


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
        await resolve_definition(
            _Refusing(refusal), spec_id=SPEC_KENYA, identity={"seed": "x"}, stat_vars=()
        )

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


def _dropped(spec_id: str, identity: dict[str, Any]) -> SeededSource:
    return SeededSource(
        source=SelectedSource(
            spec_id=spec_id, sentence="debt service paid by Kenya", has_data=True, identity=identity
        )
    )


@pytest.mark.parametrize(
    "dropped",
    [
        _dropped(SPEC_KENYA, {"stat_var": "x"}),
        _dropped(SPEC_THIRD, {"stat_var": "x", "requested_window": {"kind": "all"}}),
    ],
    ids=["same-spec-id", "same-dataset-over-other-years"],
)
async def test_a_search_that_finds_a_dataset_the_turn_dropped_names_no_source(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
    dropped: SeededSource,
) -> None:
    # The turn dropped the dataset and told the user the plan leaves it out.
    # An unpinned requirement for the same part of the question finds it
    # again, under its own spec_id or the one its identity hashes to over
    # other years; the answer is recorded and no retrieval may take it.
    context = context_holding(draft())
    save_dropped_seeds(context.state, [dropped])

    result = await _discover(surface, context)

    assert result["code"] == ProblemCode.SEED_DROPPED.value
    assert result["detail"]["spec_ids"] == [SPEC_KENYA]
    assert result["action"]["arguments"] == {
        "kind": ReferralKind.SEED_DROPPED.value,
        "requirement_id": "annual_debt_service",
        "reason": result["message"],
    }
    assert "Remove this requirement" in result["message"]
    log = load_log(context.state)
    assert log.records[-1].blocked_by == ("seed_dropped",)
    keys = _requirement_keys(context, "annual_debt_service")
    assert log.selectable(requirement_keys=keys, pinned={}) == {}
    assert log.selectable(requirement_keys=keys, pinned={"annual_debt_service": SPEC_KENYA}) == {}


@pytest.mark.parametrize(
    "dropped",
    [
        _dropped(SPEC_KENYA, {"stat_var": "x"}),
        _dropped(SPEC_THIRD, {"stat_var": "x", "requested_window": {"kind": "all"}}),
    ],
    ids=["same-spec-id", "same-dataset-over-other-years"],
)
async def test_a_search_answered_before_the_drop_settles_nothing_once_its_dataset_is_dropped(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
    dropped: SeededSource,
) -> None:
    """The record carries no block, since the search ran before the drop, and it
    matches the requirement across plan revisions; the drop is read where a
    source is taken."""
    context = context_holding(draft())
    first = await _discover(surface, context)
    assert first["status"] == "definite"
    keys = _requirement_keys(context, "annual_debt_service")
    log = load_log(context.state)
    settled = settled_sources(context.state, log, requirement_keys=keys, pinned={})
    assert {key: source.spec_id for key, source in settled.items()} == {
        "annual_debt_service": SPEC_KENYA
    }

    save_dropped_seeds(context.state, [dropped])

    assert log.records[-1].blocked_by == ()
    assert settled_sources(context.state, log, requirement_keys=keys, pinned={}) == {}
    pinned = {"annual_debt_service": SPEC_KENYA}
    assert settled_sources(context.state, log, requirement_keys=keys, pinned=pinned) == {}


async def test_a_dropped_dataset_does_not_block_a_search_that_finds_another(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    save_dropped_seeds(context.state, [_dropped(SPEC_OTHER, {"stat_var": "y"})])

    result = await _discover(surface, context)

    assert result["status"] == "definite"
    assert "action" not in result


async def test_a_seed_the_seed_node_never_stored_is_not_a_pin_the_translator_honours(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The host's own `SEEDS_STATE_KEY` holds what the request carried; a turn
    # that fell back to the question never wrote `SEEDED_SOURCES_STATE_KEY`, so
    # the pin is confirmed by a fresh search like any other.
    context = _two_dataset_context_pinning(
        SPEC_KENYA,
        draft,
        decision,
        pinning_decision,
        requirement,
        derived_requirement,
        context_holding,
    )
    context.state[SEEDS_STATE_KEY] = json.dumps(
        [{"spec": _spec(SPEC_KENYA, "debt service paid by Kenya"), "operations": []}]
    )

    await _discover(surface, context, requirement_id="req_b")

    assert len(search_calls) == 1


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
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA
    context.state[ENVELOPE_SPEC_BLOCK_STATE_KEY] = _spec(
        SPEC_KENYA, "debt service paid by Kenya", coverage=_matching_window()
    )
    context.state[ENVELOPE_SPEC_WARNINGS_STATE_KEY] = [
        {"code": "time_text_not_parsed", "severity": "warning", "message": "dropped the time text"}
    ]

    await _discover(surface, context)

    assert len(search_calls) == 1


async def test_a_pin_matching_the_resolve_id_still_searches_over_a_different_period(
    surface: McpSurface,
    search_calls: list[dict[str, Any]],
    draft: DraftFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    context_holding: ContextHolding,
) -> None:
    # The block's own coverage is `exact`, but its window (2010-2020) is not
    # the requirement's own period (2015-2024, `requirement`'s default): a
    # source resolved as covering different years is not a source for this
    # requirement's window, so this falls through to a fresh search.
    pinned = requirement(source_spec_id=SPEC_KENYA)
    context = context_holding(
        draft(
            decisions=[pinning_decision(SPEC_KENYA)],
            evidence_requirements=[pinned, derived_requirement()],
        )
    )
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA
    context.state[ENVELOPE_SPEC_BLOCK_STATE_KEY] = _spec(
        SPEC_KENYA,
        "debt service paid by Kenya",
        coverage={
            "kind": "exact",
            "has_data": True,
            "window": {"start_year": 2010, "end_year": 2020},
        },
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
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA
    context.state[ENVELOPE_SPEC_BLOCK_STATE_KEY] = _spec(SPEC_KENYA, "debt service paid by Kenya")

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
    context.state[ENVELOPE_SPEC_STATE_KEY] = SPEC_KENYA

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
    source = SelectedSource(spec_id=SPEC_KENYA, sentence="s", has_data=True, stat_vars=(ref,))
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
        requirement_id: SelectedSource(
            spec_id=_valid_spec_id(i),
            sentence="s",
            has_data=True,
            stat_vars=("refuse" if i == 3 else f"ok_{i}",),
        )
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
