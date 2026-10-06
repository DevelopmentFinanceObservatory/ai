"""The boundary where an upstream envelope stops and a typed tool result starts.

The planner and the translator do not expose data-mcp or analysis-mcp to a
model. Their tools call `compiler.surface.ToolSurface` from Python and answer
the model with one of `planner.tool_results`' shapes (`ok`, `problem`,
`referred`): a flat `outcome` plus named fields.

`translator.discovery.discover_source` is where an upstream envelope actually
arrives and is read by code: `outcome_from_envelope` over the envelope in
`answer.meta` turns a `search` answer into a `DiscoveryOutcome`, and the tool
answers the model with `compact_source`'s own field set, one entry per source,
and with a typed `problem` on a refusal. A `describe` answer never reaches a
model at all: `fetch_definitions` keeps one `definition` string per source out
of it.

What holds the boundary is that the models' tools are `FunctionTool`s over
Python functions. An ADK `McpTool`, whose dumped `CallToolResult` would carry
`structuredContent` into a prompt, is built nowhere in this repo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compiler.sources import DiscoveryStatus
from compiler.surface import FixtureSurface, McpSurface, ToolSurface
from planner.plan import PLAN_STATE_KEY, DatasetRequirement, Plan, plan_revision
from planner.tool_results import Outcome, ProblemCode, ToolProblem, ok, problem, referred
from planner.workflow import SelectedSource
from translator.discovery import (
    DiscoveryLog,
    DiscoveryRecord,
    apply_definitions,
    build_discovery_tools,
    fetch_definitions,
    requirement_revision,
    save_log,
)

ContextHolding = Callable[[Plan], Any]
DraftFactory = Callable[..., Plan]
RequirementFactory = Callable[..., DatasetRequirement]

REQUIREMENT_ID = "annual_debt_service"
SPEC_KENYA = "spec_" + "a" * 32

# A `ToolCallResult`'s own field names, and the key an upstream payload would
# arrive under if a tool returned its answer object rather than a typed shape.
_ENVELOPE_KEYS = frozenset(
    {"structured_content", "structuredContent", "meta", "_meta", "is_error", "isError", "text"}
)

# `compact_source`'s fields: what a model may see of a source.
_SOURCE_FIELDS = frozenset(
    {
        "spec_id",
        "sentence",
        "searched_as",
        "confidence",
        "has_data",
        "measure",
        "measure_count",
        "entities",
        "open_entity_axes",
        "unit",
        "coverage_window",
        "definition",
    }
)


def _flat_keys(value: Any) -> set[str]:
    """Every mapping key reachable in `value`, at any depth."""
    keys: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            keys.update(str(key) for key in current)
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return keys


def _discovery_tools(surface: ToolSurface) -> dict[str, Any]:
    return {tool.name: tool for tool in build_discovery_tools(surface=surface)}


async def _discover(
    surface: ToolSurface, context: Any, requirement_id: str = REQUIREMENT_ID
) -> dict[str, Any]:
    return await _discovery_tools(surface)["discover_source"].func(
        requirement_id=requirement_id, tool_context=context
    )


def _log_with_known_source(revision: str, ref: str) -> DiscoveryLog:
    """A log recording one `definite` source naming `ref`, as a `search` answer would.

    Lets a `fetch_definitions` test drive an id the canned `describe`
    recognises without depending on what the canned `search` resolves.
    """
    record = DiscoveryRecord(
        revision=revision,
        tool="search",
        requirement_id=REQUIREMENT_ID,
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC_KENYA,),
    )
    source = SelectedSource(
        spec_id=SPEC_KENYA,
        sentence="s",
        has_data=True,
        served_count=1,
        identity={"shape": {"kind": "stat_var", "id": ref}},
    )
    return DiscoveryLog().with_record(record, (source,))


async def test_a_search_envelope_reaches_the_model_only_as_the_typed_shape(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    context = context_holding(draft())
    result = await _discover(surface, context)

    assert result["outcome"] == Outcome.OK.value
    leaked = _flat_keys(result) & _ENVELOPE_KEYS
    assert leaked == set(), f"an upstream key reached the model-facing result: {sorted(leaked)}"
    assert isinstance(result["sources"], list) and result["sources"]
    for source in result["sources"]:
        assert set(source) <= _SOURCE_FIELDS, (
            f"a source reached the model with fields `compact_source` never names: "
            f"{sorted(set(source) - _SOURCE_FIELDS)}"
        )


async def test_a_refused_search_reaches_the_model_as_a_typed_problem(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
    requirement: RequirementFactory,
    derived_requirement: Callable[..., Any],
) -> None:
    # The canned `search` refuses whenever the requirement's own measure is
    # the literal "refuse" (`translator/tests/conftest.py`). The derived
    # requirement keeps the plan valid, since it reads the dataset one.
    plan = draft(evidence_requirements=[requirement(measure="refuse"), derived_requirement()])
    context = context_holding(plan)

    result = await _discover(surface, context)

    assert result["outcome"] == Outcome.REFUSED.value
    assert set(result) <= set(ToolProblem.model_fields) | {"outcome"}
    # The refusal body is nested in `detail` as data, not merged into the
    # top-level shape, so the model still reads one typed problem.
    assert "structured_content" not in _flat_keys(result)


async def test_a_describe_envelope_becomes_one_definition_string_and_nothing_else(
    surface: McpSurface,
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    """`describe`'s `nodes` reach a model only as `SelectedSource.definition`."""
    context = context_holding(draft())
    revision = plan_revision(Plan.model_validate_json(context.state[PLAN_STATE_KEY]))
    requirement = Plan.model_validate_json(context.state[PLAN_STATE_KEY]).requirement(
        REQUIREMENT_ID
    )
    assert isinstance(requirement, DatasetRequirement)
    log = _log_with_known_source(revision, "DebtServicePaid")
    source = log.sources[REQUIREMENT_ID][SPEC_KENYA]
    save_log(context.state, log)

    # The two calls `propose_operations` and `_enrich_candidates` both make:
    # `fetch_definitions` answers with one pair per source described, and
    # `apply_definitions` folds those pairs into the session's log.
    pairs = await fetch_definitions(
        surface,
        log,
        revision=revision,
        requirement_keys={REQUIREMENT_ID: requirement_revision(requirement)},
        sources={REQUIREMENT_ID: source},
    )
    enriched = await apply_definitions(context.state, pairs)

    # The whole `describe` answer was read down to one `definition` string on
    # the stored source, whose model-facing projection is `compact_source`.
    assert enriched.sources[REQUIREMENT_ID][SPEC_KENYA].definition == "d"
    assert enriched.records[-1].tool == "describe"
    assert _flat_keys(enriched.model_dump(mode="json")) & _ENVELOPE_KEYS == set()


async def test_a_unavailable_surface_reaches_the_model_as_a_typed_problem(
    draft: DraftFactory,
    context_holding: ContextHolding,
) -> None:
    # A `FixtureSurface` with no canned answers treats every tool as a server
    # that cannot be reached.
    context = context_holding(draft())

    result = await _discover(FixtureSurface(schemas=[]), context)

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value
    assert _flat_keys(result) & _ENVELOPE_KEYS == set()


def test_the_three_typed_shapes_carry_only_named_fields() -> None:
    """`ok`, `problem` and `referred` set `outcome` themselves and pass nothing through."""
    assert set(ok(a=1, b=2)) == {"outcome", "a", "b"}
    assert set(problem(ProblemCode.SURFACE_UNAVAILABLE, "x", outcome=Outcome.REFUSED)) == {
        "outcome",
        "code",
        "message",
    }
    assert set(referred({"kind": "no_data"})) == {"outcome", "referral"}


def test_the_discovery_module_wraps_search_and_describe_and_nothing_else(
    surface: McpSurface,
) -> None:
    """`get_data` and the analysis verbs have no wrapper here, so no model can call them."""
    assert set(_discovery_tools(surface)) == {"discover_source"}
