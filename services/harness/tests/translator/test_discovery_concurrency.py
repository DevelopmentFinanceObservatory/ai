"""Regression tests: discovery's concurrency invariants under ADK's concurrent tool dispatch.

ADK dispatches a turn's several function calls with `asyncio.gather` over one
shared `session.state` dict (see `translator.discovery`'s module docstring).
`discovery_lock` serialises only the budget check, the reservation it makes,
and the log's own read-modify-write; the `search` call itself, and
`fetch_definitions`'s own `describe` round trip, both run with the lock
released, so several concurrent calls overlap in flight while the budget
they draw on stays exact and no sibling's record is lost to another's write.

`_slow_server` stands in for the `data_server` fixture with a `search` that
sleeps, so every concurrent call is in flight before any of them return --
the same race window a model's several `discover_source` calls in one turn
open on a real ADK turn.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest
from compiler.sources import DiscoveryStatus
from compiler.surface import McpSurface, ProgressCallback, ToolCallResult
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    EvidenceRequirement,
    GroupBy,
    Period,
    Plan,
    PlanStatus,
    Reduction,
    ResolvedBy,
    Unit,
    plan_revision,
)
from planner.workflow import SelectedSource
from translator.discovery import (
    DESCRIBE,
    DISCOVERY_CALL_BUDGET,
    DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
    SEARCH,
    DiscoveryRecord,
    build_discovery_tools,
    fetch_definitions,
    load_log,
    requirement_revision,
    save_log,
)

from ..described_nodes import stat_var_node
from ..search_results import search_result_parts

SPEC = "spec_" + "a" * 32

ContextHolding = Callable[[Plan], Any]

# More than `DISCOVERY_CALL_BUDGET` (8): the race this module guards against
# is a plan whose several dataset requirements a model discovers in one turn.
N_CONCURRENT = 12


def _spec() -> dict[str, Any]:
    return {
        "spec_id": SPEC,
        "resolved": {
            "rendered_sentence": "debt service paid by Kenya",
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 2010, "end_year": 2024},
            },
            "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
            "entities": [{"role": "recipient", "label": "Kenya", "ref": "country/KEN"}],
            "facets": {"unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"}},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": SPEC,
            "identity": {"shape": {"kind": "stat_var", "id": "DebtServicePaid"}},
        },
    }


def _slow_server(*, calls: list[dict[str, Any]], concurrency: list[int]) -> FastMCP:
    """A `data-mcp` stand-in whose `search` sleeps and records how many calls overlap.

    Every concurrent caller reaches the `await asyncio.sleep` before any of
    them return, so `calls` and `concurrency` see exactly what a real
    concurrent turn would produce.
    """
    server = FastMCP(name="data-mcp")
    in_flight = 0

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        nonlocal in_flight
        calls.append({"parts": parts})
        in_flight += 1
        concurrency.append(in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        envelope = {
            "schema_version": 1,
            "status": "definite",
            "parts": [
                {"status": "definite", "variable_text": "asked", "warnings": [], "spec": _spec()}
            ],
        }
        return ToolResult(content="digest", **search_result_parts(envelope))

    @server.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        nodes = [stat_var_node(dcid, label=dcid, description="d", roles=()) for dcid in ids]
        return ToolResult(content="nodes", structured_content={"nodes": nodes})

    return server


def _plan(requirement_ids: list[str]) -> Plan:
    """A draft plan with one dataset requirement, and its reading aggregate, per id."""
    decision = Decision(
        slot="measure",
        description="What to measure Kenya's dependence by.",
        options=["annual debt service", "total debt stock"],
        chosen="annual debt service",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Debt service is the standard measure for near-term dependence.",
    )
    requirements: list[EvidenceRequirement] = []
    for requirement_id in requirement_ids:
        requirements.append(
            DatasetRequirement(
                id=requirement_id,
                decision_ids=["measure"],
                measure="external debt service paid",
                entities=[Entity(name="Kenya", role="debtor")],
                period=Period(start=2015, end=2024),
                unit=Unit(currency="USD", prices="current"),
            )
        )
        requirements.append(
            AggregateRequirement(
                id=f"{requirement_id}_total",
                decision_ids=["measure"],
                input_id=requirement_id,
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            )
        )
    return Plan(
        question="How much does Kenya pay to service its debt each year?",
        decisions=[decision],
        evidence_requirements=requirements,
        status=PlanStatus.DRAFT,
    )


def _discover_source(surface: McpSurface) -> Any:
    return {t.name: t for t in build_discovery_tools(surface=surface)}["discover_source"].func


def _never_answers_server() -> FastMCP:
    """A `data-mcp` stand-in whose `search` and `describe` never return.

    Slow rather than down: every call reaches the server and hangs, so the
    only thing that ever ends the call is the surface's own deadline turning
    it into a `ToolSurfaceUnavailableError`.
    """
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        await asyncio.sleep(3600)
        raise AssertionError("never reached: the deadline should have cut this off first")

    @server.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        await asyncio.sleep(3600)
        raise AssertionError("never reached: the deadline should have cut this off first")

    return server


def _fast_search_hangs_describe_server() -> FastMCP:
    """A `data-mcp` stand-in whose `search` answers at once and whose `describe` never does."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        envelope = {
            "schema_version": 1,
            "status": "definite",
            "parts": [
                {"status": "definite", "variable_text": "asked", "warnings": [], "spec": _spec()}
            ],
        }
        return ToolResult(content="digest", **search_result_parts(envelope))

    @server.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        await asyncio.sleep(3600)
        raise AssertionError("never reached: the deadline should have cut this off first")

    return server


# Four requirements, three concurrent calls each: `N_CONCURRENT` calls in
# total, none of them alone reaching its own per-requirement ceiling
# (`DISCOVERY_CALL_BUDGET_PER_REQUIREMENT`), so what the next two tests cap
# successes at is the revision's own total budget -- the race they guard.
_FOUR_REQUIREMENTS = [f"requirement_{i}" for i in range(4)]


async def test_concurrent_discovery_keeps_the_budget(context_holding: ContextHolding) -> None:
    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    context = context_holding(_plan(_FOUR_REQUIREMENTS))

    results = await asyncio.gather(
        *[
            discover_source(
                requirement_id=_FOUR_REQUIREMENTS[i % len(_FOUR_REQUIREMENTS)], tool_context=context
            )
            for i in range(N_CONCURRENT)
        ]
    )

    ok_count = sum(1 for r in results if r["outcome"] == "ok")
    budget_exhausted_count = sum(1 for r in results if r.get("code") == "budget_exhausted")

    assert len(calls) == DISCOVERY_CALL_BUDGET
    assert ok_count == DISCOVERY_CALL_BUDGET
    assert budget_exhausted_count == N_CONCURRENT - DISCOVERY_CALL_BUDGET


async def test_concurrent_discovery_keeps_every_record(context_holding: ContextHolding) -> None:
    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    plan = _plan(_FOUR_REQUIREMENTS)
    context = context_holding(plan)

    await asyncio.gather(
        *[
            discover_source(
                requirement_id=_FOUR_REQUIREMENTS[i % len(_FOUR_REQUIREMENTS)], tool_context=context
            )
            for i in range(N_CONCURRENT)
        ]
    )

    revision = plan_revision(plan)
    log = load_log(context.state)
    assert log.calls(revision) == DISCOVERY_CALL_BUDGET
    assert len(log.records) == DISCOVERY_CALL_BUDGET


async def test_concurrent_discovery_across_distinct_requirements_all_land(
    context_holding: ContextHolding,
) -> None:
    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    requirement_ids = [f"requirement_{i}" for i in range(5)]  # within DISCOVERY_CALL_BUDGET
    plan = _plan(requirement_ids)
    context = context_holding(plan)

    await asyncio.gather(
        *[
            discover_source(requirement_id=requirement_id, tool_context=context)
            for requirement_id in requirement_ids
        ]
    )

    revision = plan_revision(plan)
    log = load_log(context.state)
    assert set(log.sources) == set(requirement_ids)
    assert log.calls(revision) == len(requirement_ids)


async def test_concurrent_searches_overlap_within_the_budget(
    context_holding: ContextHolding,
) -> None:
    # Twelve concurrent `discover_source` calls against a budget of eight
    # let exactly eight searches reach the server, and several of them
    # overlap in flight. Only the budget check and the log write are
    # serialised. Holding `discovery_lock` across `discover_source`'s whole
    # body instead makes `max(concurrency)` fall back to 1 here.
    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    context = context_holding(_plan(_FOUR_REQUIREMENTS))

    await asyncio.gather(
        *[
            discover_source(
                requirement_id=_FOUR_REQUIREMENTS[i % len(_FOUR_REQUIREMENTS)], tool_context=context
            )
            for i in range(N_CONCURRENT)
        ]
    )

    assert len(calls) == DISCOVERY_CALL_BUDGET
    assert max(concurrency) > 1


async def test_a_concurrent_search_and_fetch_definitions_both_land_in_the_log(
    context_holding: ContextHolding,
) -> None:
    """A concurrent `discover_source` search and a concurrent `fetch_definitions`
    commit -- the read-release-await-reload-save shape `propose_operations`
    and `_enrich_candidates` both apply -- write to the same log at once.
    Both land: the search's own record and the describe's own enrichment
    each survive the other's write, rather than one replacing the other.
    """
    from translator.discovery import discovery_lock  # noqa: PLC0415

    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    plan = _plan(["annual_debt_service", "already_found"])
    context = context_holding(plan)
    revision = plan_revision(plan)
    requirement = plan.requirement("already_found")
    assert isinstance(requirement, DatasetRequirement)
    key = requirement_revision(requirement)
    known_source = SelectedSource(
        spec_id=SPEC,
        sentence="s",
        has_data=True,
        served_count=1,
        identity={"shape": {"kind": "stat_var", "id": "DebtServicePaid"}},
    )
    seed = DiscoveryRecord(
        revision=revision,
        tool="search",
        requirement_id="already_found",
        requirement_key=key,
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(SPEC,),
    )
    async with discovery_lock(context.state):
        save_log(context.state, load_log(context.state).with_record(seed, (known_source,)))

    async def _describe_already_found() -> None:
        async with discovery_lock(context.state):
            log = load_log(context.state)
        pairs = await fetch_definitions(
            surface,
            log,
            revision=revision,
            requirement_keys={"already_found": key},
            sources={"already_found": known_source},
        )
        async with discovery_lock(context.state):
            log = load_log(context.state)
            for record, sources in pairs:
                log = log.with_record(record, sources)
            save_log(context.state, log)

    await asyncio.gather(
        discover_source(requirement_id="annual_debt_service", tool_context=context),
        _describe_already_found(),
    )

    log = load_log(context.state)
    assert any(
        r.requirement_id == "annual_debt_service" and r.tool == "search" for r in log.records
    )
    assert any(r.requirement_id == "already_found" and r.tool == "describe" for r in log.records)
    assert log.sources["already_found"][SPEC].definition == "d"


class _RaisesNTimesThenAnswers:
    """A `ToolSurface` stand-in whose first `n` calls raise, unlike `ToolSurfaceUnavailableError`
    -- an exception `discover_source` was never taught to catch and record, so
    nothing about these calls is ever spent from `DISCOVERY_CALL_BUDGET` if
    each one's own reservation is released regardless.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self.calls = 0

    async def call_tool(
        self,
        *,
        tool: Any,
        arguments: dict[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del tool, arguments
        self.calls += 1
        if self.calls <= self._n:
            raise RuntimeError("boom")
        envelope = {
            "schema_version": 1,
            "status": "definite",
            "parts": [
                {"status": "definite", "variable_text": "asked", "warnings": [], "spec": _spec()}
            ],
        }
        return ToolCallResult(is_error=False, **search_result_parts(envelope))


async def test_a_reservation_is_released_when_the_search_raises(
    context_holding: ContextHolding,
) -> None:
    # `DISCOVERY_CALL_BUDGET_PER_REQUIREMENT` raises in a row: if the
    # reservation each one made were not released, they alone would exhaust
    # this requirement's own share of the budget with nothing ever recorded,
    # and the next call would be refused for a budget nothing was ever spent
    # from.
    surface = _RaisesNTimesThenAnswers(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT)
    discover_source = _discover_source(surface)
    context = context_holding(_plan(["annual_debt_service"]))

    for _ in range(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT):
        with pytest.raises(RuntimeError, match="boom"):
            await discover_source(requirement_id="annual_debt_service", tool_context=context)

    assert load_log(context.state).records == ()

    result = await discover_source(requirement_id="annual_debt_service", tool_context=context)

    assert result["outcome"] == "ok"


def _reordering_server() -> FastMCP:
    """A `data-mcp` stand-in whose first `search` call finishes after its second.

    Sleeps on its first call only, so two calls launched in list order
    finish in the reverse order: the same divergence between list order and
    completion order `merge_parallel_function_response_events` merges by.
    """
    server = FastMCP(name="data-mcp")
    calls_made = {"n": 0}

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        calls_made["n"] += 1
        if calls_made["n"] == 1:
            await asyncio.sleep(0.05)
        envelope = {
            "schema_version": 1,
            "status": "definite",
            "parts": [
                {"status": "definite", "variable_text": "asked", "warnings": [], "spec": _spec()}
            ],
        }
        return ToolResult(content="digest", **search_result_parts(envelope))

    return server


class _FakeContext:
    """A minimal stand-in for ADK's `ToolContext`, wrapping a real `google.adk.sessions.state.State`."""

    def __init__(self, state: Any) -> None:
        self.state = state


async def test_the_adk_merge_of_two_concurrent_calls_keeps_both_records() -> None:
    """Reproduces ADK's own merge, with ADK's own `State` and `deep_merge_dicts`.

    Two `discover_source` calls share one session's underlying state dict but
    each get their own `State.delta`, exactly as `Context.__init__` builds
    one `ToolContext` per concurrent call over `invocation_context.session.state`
    (`google.adk.agents.context`). `req_b`'s call finishes before `req_a`'s,
    the reverse of the order they are listed in below --
    `merge_parallel_function_response_events` merges `state_delta`s in list
    order, not completion order, so this is the case where the two diverge.
    `load_log` on the merged, committed result has to see both records for
    `save_log`'s dict shape to be doing its job here: keying the persisted
    log by one JSON string instead of by record id would lose `req_a`'s own
    record, since a plain string leaf is replaced outright by whichever
    delta is merged last rather than unioned with it.
    """
    from google.adk.flows.llm_flows.tools._batch_executor import deep_merge_dicts  # noqa: PLC0415
    from google.adk.sessions.state import State  # noqa: PLC0415
    from planner.plan import PLAN_STATE_KEY  # noqa: PLC0415

    plan = _plan(["req_a", "req_b"])
    shared_value: dict[str, Any] = {PLAN_STATE_KEY: plan.model_dump_json()}
    delta_a: dict[str, Any] = {}
    delta_b: dict[str, Any] = {}
    context_a = _FakeContext(State(value=shared_value, delta=delta_a))
    context_b = _FakeContext(State(value=shared_value, delta=delta_b))
    surface = McpSurface(sources={"data-mcp": _reordering_server()})
    discover_source = _discover_source(surface)

    await asyncio.gather(
        discover_source(requirement_id="req_a", tool_context=context_a),
        discover_source(requirement_id="req_b", tool_context=context_b),
    )

    merged: dict[str, Any] = deep_merge_dicts(deep_merge_dicts({}, delta_a), delta_b)
    committed = _State(merged)

    log = load_log(committed)
    assert {record.requirement_id for record in log.records} == {"req_a", "req_b"}


def test_discovery_lock_is_synchronous() -> None:
    # Load-bearing: an `await` inside `discovery_lock` would let two
    # concurrent callers each mint a token and take a different lock before
    # either finishes. Stated as a test, so a later `await` fails here
    # rather than in production. Imported inside the test, not at module
    # level, so the rest of this file still runs -- and demonstrates the
    # race -- against a module old enough not to define `discovery_lock` at
    # all.
    from translator.discovery import discovery_lock  # noqa: PLC0415

    assert inspect.iscoroutinefunction(discovery_lock) is False


class _State:
    """A minimal `SessionState`: a plain `dict` fails `ty`'s structural check here.

    `dict.__setitem__`'s `key` and `value` are positional-only, and
    `SessionState.__setitem__` (`planner/plan.py`) does not mark its own
    parameters positional-only, so a `dict` does not structurally satisfy the
    protocol under `ty` even though it works at runtime. This wrapper matches
    the protocol's own signature instead. `data`, when given, seeds it --
    reading back what a merged, committed session state would hold.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data) if data else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value


def test_one_session_takes_one_lock() -> None:
    from translator.discovery import discovery_lock  # noqa: PLC0415

    state = _State()

    assert discovery_lock(state) is discovery_lock(state)


async def test_a_timed_out_search_is_counted_against_the_budget(
    context_holding: ContextHolding,
) -> None:
    # The server hangs, so only the surface's own deadline ends each call.
    # Every call here targets the same requirement, so the requirement's own
    # share of the budget (DISCOVERY_CALL_BUDGET_PER_REQUIREMENT) runs out
    # before the revision's larger total does -- see
    # `test_the_budget_stops_the_ninth_call_...` in `test_discovery.py` for
    # the total-budget half of this behaviour.
    surface = McpSurface(
        sources={"data-mcp": _never_answers_server()},
        call_timeouts={SEARCH: 0.02, DESCRIBE: 0.02},
    )
    discover_source = _discover_source(surface)
    plan = _plan(["annual_debt_service"])
    context = context_holding(plan)
    revision = plan_revision(plan)
    requirement = plan.requirement("annual_debt_service")
    assert isinstance(requirement, DatasetRequirement)
    key = requirement_revision(requirement)

    results = [
        await discover_source(requirement_id="annual_debt_service", tool_context=context)
        for _ in range(DISCOVERY_CALL_BUDGET_PER_REQUIREMENT + 2)
    ]

    codes = [r.get("code") for r in results]
    budget = DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    assert codes[:budget] == ["surface_unavailable"] * budget
    assert codes[budget:] == ["budget_exhausted"] * 2

    log = load_log(context.state)
    assert log.calls(revision) == DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    assert (
        log.requirement_calls(revision, "annual_debt_service")
        == DISCOVERY_CALL_BUDGET_PER_REQUIREMENT
    )
    assert log.selectable(requirement_keys={"annual_debt_service": key}, pinned={}) == {}
    assert log.sources == {}


async def test_fetch_definitions_records_a_timed_out_describe_without_raising(
    context_holding: ContextHolding,
) -> None:
    # A real search finds `DebtServicePaid`, so there is a source to
    # describe; only `describe` hangs, and only its deadline ends the call.
    # `fetch_definitions` runs as plain code, so this exercises the deadline
    # alone.
    surface = McpSurface(
        sources={"data-mcp": _fast_search_hangs_describe_server()},
        call_timeouts={DESCRIBE: 0.02},
    )
    discover_source = _discover_source(surface)
    plan = _plan(["annual_debt_service"])
    context = context_holding(plan)
    revision = plan_revision(plan)
    requirement = plan.requirement("annual_debt_service")
    assert isinstance(requirement, DatasetRequirement)
    key = requirement_revision(requirement)

    await discover_source(requirement_id="annual_debt_service", tool_context=context)
    log = load_log(context.state)
    source = log.sources["annual_debt_service"][SPEC]

    pairs = await fetch_definitions(
        surface,
        log,
        revision=revision,
        requirement_keys={"annual_debt_service": key},
        sources={"annual_debt_service": source},
    )

    # The timed-out describe is returned for the caller to record. Only the
    # search spends the budget, though both records end up kept once the
    # caller applies this pair.
    ((describe_record, described_sources),) = pairs
    assert describe_record.tool == "describe"
    assert describe_record.transport_failed is True
    assert described_sources == ()
    enriched = log.with_record(describe_record, described_sources)
    assert len(enriched.records) == 2
    assert enriched.calls(revision) == 1
    assert enriched.sources["annual_debt_service"][SPEC].definition is None


async def test_a_sibling_check_between_release_and_commit_cannot_overspend_the_budget(
    context_holding: ContextHolding, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling's own budget check must never land in a gap between a
    reservation's release and its record's commit: were the two ever split
    across separate `discovery_lock` acquisitions, a call landing in that gap
    would see the slot as free before any record said it was spent.

    Patches `_record_search` to run one extra `discover_source` call before
    delegating to the real implementation, landing that sibling's own check
    inside whatever gap `_record_search` leaves between releasing the
    reservation its own caller made and committing that caller's record. With
    `DISCOVERY_CALL_BUDGET` calls already in flight, the sibling's check must
    still see the budget spent, since release and commit are one locked
    section, so exactly `DISCOVERY_CALL_BUDGET` searches ever reach the
    server, not `DISCOVERY_CALL_BUDGET + 1`.
    """
    import translator.discovery as discovery_module  # noqa: PLC0415

    calls: list[dict[str, Any]] = []
    concurrency: list[int] = []
    surface = McpSurface(sources={"data-mcp": _slow_server(calls=calls, concurrency=concurrency)})
    discover_source = _discover_source(surface)
    context = context_holding(_plan(_FOUR_REQUIREMENTS))

    original = discovery_module._record_search
    fired = {"done": False}

    async def _record_search_with_sibling_first(
        tool_context: Any, record: Any, sources: Any = (), **kwargs: Any
    ) -> Any:
        if not fired["done"]:
            fired["done"] = True
            await discover_source(requirement_id=_FOUR_REQUIREMENTS[0], tool_context=tool_context)
        return await original(tool_context, record, sources, **kwargs)

    monkeypatch.setattr(discovery_module, "_record_search", _record_search_with_sibling_first)

    await asyncio.gather(
        *[
            discover_source(
                requirement_id=_FOUR_REQUIREMENTS[i % len(_FOUR_REQUIREMENTS)], tool_context=context
            )
            for i in range(DISCOVERY_CALL_BUDGET)
        ]
    )

    assert len(calls) == DISCOVERY_CALL_BUDGET
