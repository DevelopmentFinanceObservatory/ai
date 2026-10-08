"""Fixtures for the executor suite: a scripted surface and approved plans.

`ScriptedSurface` answers each tool from a queue of canned results, repeating
the last one, and records every call; an `Exception` in a queue is raised in
place of an answer, which is how a transport failure is scripted. Its schemas
are the real servers', so the capability table's pinned hashes hold and the
executor's own surface check passes. `delays` optionally holds one tool's
answer for a fixed number of seconds after it is called, and `max_concurrency`
is the highest number of that surface's calls ever in flight at once -- a
test proving two steps run concurrently delays one tool and reads this back.

`approved_plan` compiles a retrieve-then-aggregate translation against those
schemas and approves it, the state a run starts from. `two_source_plan` adds
a second, independent retrieve-then-aggregate pair reading a population
requirement -- for tests that check a step's evidence is bound to the source
it was compiled to retrieve, not whichever one a tool happens to answer with.
`join_plan` retrieves a debt and a GDP series and joins them into one terminal
frame, and `two_input_plan` does the same for a compare, a share_of or a
quotient_of -- for tests of a terminal frame that rests on more than one retrieval.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import compile_plan
from compiler.sources import member_digest
from compiler.surface import (
    McpSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from data_mcp.manifest import SeriesStatisticsModel
from data_mcp.series_statistics import series_statistics_table
from data_mcp.server import mcp as data_mcp
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    GroupBy,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DerivedRequirement,
    Entity,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RequirementSide,
    ShareOfRequirement,
)
from planner.workflow import ResolvedEntity, SelectedSource, ToolId

SPEC_KENYA = "spec_" + "a" * 32
SPEC_POPULATION = "spec_" + "c" * 32
SPEC_GDP = "spec_" + "d" * 32
CONTENT_VERSION = "cv_" + "1" * 32
RESULT_FRAME: dict[str, Any] = {
    "spec_id": "ops_" + "2" * 32,
    "content_version": "ocv_" + "3" * 32,
    "rows": 1,
    "role_keys": [],
    "stat_var_count": 1,
}


def _fixture_stat_var(source: SelectedSource) -> str:
    """The one StatVar a fixture source's identity names (`{"stat_var": ...}`)."""
    return str((source.identity or {}).get("stat_var") or "Stat")


def _resolved_stat_vars_trace(source: SelectedSource) -> dict[str, Any]:
    """A manifest `trace` serving the fixture source's one StatVar, and stating the family
    digest the source was approved with, so membership is confirmed by default."""
    return {
        "resolved_stat_vars": [{"ref": f"dcid:{_fixture_stat_var(source)}", "roles": []}],
        "family_member_digest": source.shape_member_digest,
        "family_member_count": source.shape_member_count,
    }


_STATISTICS_ARTIFACT_DIGEST = "sha256:" + "9" * 64


def _statistics_artifact() -> dict[str, Any]:
    """A `rendering: {"statistics": "series"}` artifact entry -- the manifest's own
    reference to its statistics artifact, matched by `find_statistics_artifact` and never
    actually fetched over the network in this suite (see `_stub_fetch_series_statistics`,
    which reads `_stats_by_manifest` instead).
    """
    return {
        "artifact_digest": _STATISTICS_ARTIFACT_DIGEST,
        "rendering": {"statistics": "series"},
        "bytes": 512,
        "media_type": "application/vnd.apache.parquet",
        "created_at": "2026-08-11T00:00:00+00:00",
        "tier": "working",
    }


# `(spec_id, content_version) -> statistics`, the raw list each test's own `manifest`/
# `manifest_for` factory built -- registered by reference, so a test that mutates the
# list after the factory returns it (`two["statistics"].append(...)`,
# `sparse["statistics"][0]["observations"] = 7`) is seen by `_stub_fetch_series_statistics`
# too, without either side re-deriving the other. Cleared and rebuilt by
# `_stats_by_manifest` every test, so entries never survive across tests.
_StatsRegistry = dict[tuple[str, str], list[dict[str, Any]]]


@pytest.fixture
def _stats_by_manifest() -> _StatsRegistry:
    return {}


@pytest.fixture(autouse=True)
def _stub_fetch_series_statistics(
    monkeypatch: pytest.MonkeyPatch, _stats_by_manifest: _StatsRegistry
) -> None:
    """Stand in for `executor.run.fetch_series_statistics_table`: every test in this suite
    scripts `get_data`'s own manifest dict directly (`ScriptedSurface`), never a real
    signed-URL round trip, so the real fetch -- which needs a live `client_holder` client
    and a real Parquet GET -- has nothing to reach here. Reads `_stats_by_manifest`, which
    the `manifest`/`manifest_for` factories populate by `(spec_id, content_version)`, the
    same pair `executor.run._fetch_series_stats` looks the artifact up under.
    """

    async def _stub(*, subject_id: str, content_version: str, artifact: Any, **_: Any) -> Any:
        del artifact
        rows = _stats_by_manifest.get((subject_id, content_version), [])
        return series_statistics_table([SeriesStatisticsModel.model_validate(row) for row in rows])

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", _stub)


class ScriptedSurface:
    """A `ToolSurface` answering from canned results and recording every call.

    `delays` optionally names, per tool, how many seconds `call_tool` waits
    after picking its scripted answer and before returning (or raising) it.
    The answer is picked before the wait, so two concurrent callers of the
    same tool still get their answers in the order they called in, the same
    order a serial loop would have called in. `max_concurrency` counts the
    wait alone: it is the highest number of calls this surface ever held in
    that wait at once, across every tool, so a test can prove two steps
    overlapped rather than merely both having run.
    """

    def __init__(
        self,
        *,
        schemas: Sequence[ToolSchema],
        responses: Mapping[str, Sequence[ToolCallResult | Exception]],
        delays: Mapping[str, float] | None = None,
    ) -> None:
        self.schemas = list(schemas)
        self.queues = {name: deque(items) for name, items in responses.items()}
        self.delays = dict(delays or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.max_concurrency = 0
        self._in_flight = 0

    async def list_tools(self) -> list[ToolSchema]:
        return list(self.schemas)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append((tool.name, dict(arguments)))
        queue = self.queues.get(tool.name)
        if not queue:
            raise ToolSurfaceUnavailableError(f"no scripted answer for {tool.name!r}")
        item = queue.popleft() if len(queue) > 1 else queue[0]
        delay = self.delays.get(tool.name)
        if delay:
            self._in_flight += 1
            self.max_concurrency = max(self.max_concurrency, self._in_flight)
            try:
                await asyncio.sleep(delay)
            finally:
                self._in_flight -= 1
        if isinstance(item, Exception):
            raise item
        return item


def ok(structured_content: Mapping[str, Any], text: str = "digest") -> ToolCallResult:
    return ToolCallResult(is_error=False, structured_content=dict(structured_content), text=text)


def refusal(code: str, retry_mode: str, **extra: Any) -> ToolCallResult:
    return ToolCallResult(
        is_error=True,
        structured_content={"code": code, "retry_mode": retry_mode, **extra},
        text=f"refused: {code}",
    )


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas, which the capability table is pinned to."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


def with_frame_ref(result: dict[str, Any]) -> dict[str, Any]:
    """`result` with the `frame_ref` `get_data` states beside its manifest: the
    manifest's own pair, as for a frame minted with the default columns, unless
    `result` states one already or names no definition."""
    definition = result.get("definition")
    if "frame_ref" in result or not isinstance(definition, dict):
        return result
    return result | {
        "frame_ref": {
            "spec_id": definition.get("spec_id"),
            "content_version": result.get("content_version"),
        }
    }


@pytest.fixture
def manifest(
    source: Callable[..., SelectedSource], _stats_by_manifest: _StatsRegistry
) -> Callable[..., dict[str, Any]]:
    """Return a factory for a `get_data` manifest matching the `source` fixture.

    Ten annual observations for Kenya, 2015 to 2024, in USD, under the same
    identity the source was approved with. `statistics` still rides the
    returned dict under its own key -- tests read and mutate it directly
    (`two["statistics"].append(...)`, `sparse["statistics"][0]["observations"]
    = 7`) -- but the manifest itself carries only `series_count` plus a
    statistics artifact reference, per the real wire shape; `statistics`'s
    same list object is registered into `_stats_by_manifest`
    (`_stub_fetch_series_statistics` reads it back), so a mutation after this
    call still reaches `evidence_from_manifest` the way a real fetch would.
    """

    def _manifest(**overrides: Any) -> dict[str, Any]:
        base = source()
        statistics: list[dict[str, Any]] = [
            {
                "stat_var": "dcid:DebtServicePaid",
                "entities": [{"role": "recipient", "ref": "country/KEN"}],
                "facet": {"unit": "USD"},
                "observations": 10,
                "first": {"value": 100.0, "period": "2015"},
                "latest": {"value": 109.0, "period": "2024"},
                "minimum": {"value": 100.0, "period": "2015"},
                "maximum": {"value": 109.0, "period": "2024"},
            }
        ]
        content_version = overrides.get("content_version", CONTENT_VERSION)
        spec_id = (overrides.get("definition") or {}).get("spec_id", SPEC_KENYA)
        _stats_by_manifest[(spec_id, content_version)] = statistics
        fields: dict[str, Any] = {
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": SPEC_KENYA,
                "identity": dict(base.identity),
            },
            "content_version": CONTENT_VERSION,
            "coverage": {"rows": 10, "dimensions": []},
            "citation": {"text": "World Bank, International Debt Statistics", "sources": []},
            "trace": _resolved_stat_vars_trace(base),
            "warnings": [],
            "artifacts": [_statistics_artifact()],
            "series_count": len(statistics),
            "statistics": statistics,
        }
        return with_frame_ref(fields | overrides)

    return _manifest


@pytest.fixture
def manifest_for(_stats_by_manifest: _StatsRegistry) -> Callable[..., dict[str, Any]]:
    """Return a factory for a `get_data` manifest under a given source's own spec_id and identity.

    Ten annual observations, 2015 to 2024, under the source's first entity and
    unit -- for tests that hand one step's retrieval the manifest a different
    source would produce. See `manifest`'s own docstring for `statistics` /
    `_stats_by_manifest`.
    """

    def _manifest_for(source: SelectedSource, **overrides: Any) -> dict[str, Any]:
        entity = (
            source.entities[0]
            if source.entities
            else ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya")
        )
        stat_var = _fixture_stat_var(source)
        statistics: list[dict[str, Any]] = [
            {
                "stat_var": f"dcid:{stat_var}",
                "entities": [{"role": entity.role, "ref": entity.ref}],
                "facet": {"unit": source.unit_ref} if source.unit_ref else {},
                "observations": 10,
                "first": {"value": 100.0, "period": "2015"},
                "latest": {"value": 109.0, "period": "2024"},
                "minimum": {"value": 100.0, "period": "2015"},
                "maximum": {"value": 109.0, "period": "2024"},
            }
        ]
        content_version = overrides.get("content_version", CONTENT_VERSION)
        spec_id = (overrides.get("definition") or {}).get("spec_id", source.spec_id)
        _stats_by_manifest[(spec_id, content_version)] = statistics
        fields: dict[str, Any] = {
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": source.spec_id,
                "identity": dict(source.identity),
            },
            "content_version": CONTENT_VERSION,
            "coverage": {"rows": 10, "dimensions": []},
            "citation": {"text": "World Bank, International Debt Statistics", "sources": []},
            "trace": _resolved_stat_vars_trace(source),
            "warnings": [],
            "artifacts": [_statistics_artifact()],
            "series_count": len(statistics),
            "statistics": statistics,
        }
        return with_frame_ref(fields | overrides)

    return _manifest_for


@pytest.fixture
def population_source() -> Callable[..., SelectedSource]:
    """Return a factory for a discovered population source, distinct from `source`'s debt one."""

    def _population_source(**overrides: Any) -> SelectedSource:
        fields: dict[str, Any] = {
            "spec_id": SPEC_POPULATION,
            "sentence": "total population of Kenya, 2015 to 2024",
            "confidence": "high",
            "has_data": True,
            "served_count": 1,
            "shape_member_count": 1,
            "shape_member_digest": member_digest(["Population"]),
            "entities": (ResolvedEntity(role="country", ref="country/KEN", label="Kenya"),),
            "unit_ref": None,
            "unit_label": None,
            "coverage_window": Period(start=2015, end=2024),
            "identity": {"stat_var": "Population"},
        }
        return SelectedSource(**(fields | overrides))

    return _population_source


@pytest.fixture
def approved_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Callable[..., Plan]:
    """Return a factory for an approved retrieve-then-aggregate plan."""

    def _approved(**overrides: Any) -> Plan:
        dataset = overrides.pop("requirement", requirement())
        draft = Plan(
            question="How much does Kenya pay to service its debt each year?",
            decisions=[decision()],
            evidence_requirements=[dataset, derived_requirement()],
            status=PlanStatus.DRAFT,
        )
        translation = Translation(
            operations=[
                RetrieveOperation(
                    id="debt", requirement_id="annual_debt_service", spec_id=SPEC_KENYA
                ),
                AggregateOperation(
                    id="totals",
                    requirement_id="annual_totals",
                    input="debt",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
            ],
            terminal=["totals"],
        )
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered={"annual_debt_service": overrides.pop("source", source())},
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()

    return _approved


@pytest.fixture
def two_source_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
    source: Callable[..., SelectedSource],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Callable[..., Plan]:
    """Return a factory for an approved plan retrieving two independent sources.

    A debt requirement (`annual_debt_service`) feeding `totals`, and a
    population requirement (`population`) feeding `population_totals` -- for
    tests that check a step's evidence is bound to the source it was compiled
    to retrieve, not whichever one a tool happens to answer with.
    """

    def _two_source(**overrides: Any) -> Plan:
        population = DatasetRequirement(
            id="population",
            decision_ids=["measure"],
            measure="total population",
            entities=[Entity(name="Kenya")],
            period=Period(start=2015, end=2024),
        )
        population_totals = AggregateRequirement(
            id="population_totals",
            decision_ids=["measure"],
            input_id="population",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        )
        draft = Plan(
            question=(
                "How much does Kenya pay to service its debt each year, and what is its population?"
            ),
            decisions=[decision()],
            evidence_requirements=[
                requirement(),
                derived_requirement(),
                population,
                population_totals,
            ],
            status=PlanStatus.DRAFT,
        )
        translation = Translation(
            operations=[
                RetrieveOperation(
                    id="debt", requirement_id="annual_debt_service", spec_id=SPEC_KENYA
                ),
                AggregateOperation(
                    id="totals",
                    requirement_id="annual_totals",
                    input="debt",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
                RetrieveOperation(id="pop", requirement_id="population", spec_id=SPEC_POPULATION),
                AggregateOperation(
                    id="population_totals_op",
                    requirement_id="population_totals",
                    input="pop",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
            ],
            terminal=["totals", "population_totals_op"],
        )
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered={
                "annual_debt_service": overrides.pop("source", source()),
                "population": overrides.pop("population_source", population_source()),
            },
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()

    return _two_source


@pytest.fixture
def gdp_source(source: Callable[..., SelectedSource]) -> Callable[..., SelectedSource]:
    """Return a factory for a discovered GDP source that shares `source`'s entity column.

    The same Kenya entity and unit as the debt `source`, so a join of the two
    has a role column to key on and a compare or share_of reads both sides.
    """

    def _gdp_source(**overrides: Any) -> SelectedSource:
        fields: dict[str, Any] = {
            "spec_id": SPEC_GDP,
            "sentence": "gross domestic product of Kenya, 2010 to 2024",
            "served_count": 1,
            "shape_member_digest": member_digest(["Gdp"]),
            "identity": {"stat_var": "Gdp"},
        }
        return source(**(fields | overrides))

    return _gdp_source


@pytest.fixture
def two_input_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    gdp_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Callable[..., Plan]:
    """Return a factory for an approved plan whose one terminal frame rests on two retrievals.

    A debt retrieval (`debt`) and a GDP retrieval (`gdp`) feed a single
    terminal step named `combined`, a `join` by default or a `compare`, a
    `share_of` or a `quotient_of` when `kind` says so. Either side reads its own retrieval
    directly, so the terminal frame's lineage is exactly those two.
    `retrieval_ids` renames the two retrieval steps, debt first.
    """

    def _two_input(
        *, kind: str = "join", retrieval_ids: tuple[str, str] = ("debt", "gdp"), **overrides: Any
    ) -> Plan:
        debt_id, gdp_id = retrieval_ids
        gdp = requirement(id="annual_gdp", measure="gross domestic product")
        derived: DerivedRequirement
        operation: Operation
        match kind:
            case "join":
                derived = JoinRequirement(
                    id="combined",
                    decision_ids=["measure"],
                    left_id="annual_debt_service",
                    right_id="annual_gdp",
                )
                operation = JoinOperation(
                    id="combined", requirement_id="combined", left=debt_id, right=gdp_id
                )
            case "compare":
                derived = ComparisonRequirement(
                    id="combined",
                    decision_ids=["measure"],
                    left=RequirementSide(input_id="annual_debt_service"),
                    right=RequirementSide(input_id="annual_gdp"),
                )
                operation = CompareOperation(
                    id="combined",
                    requirement_id="combined",
                    left=Side(input=debt_id),
                    right=Side(input=gdp_id),
                )
            case "share_of":
                derived = ShareOfRequirement(
                    id="combined",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="annual_debt_service"),
                    denominator=RequirementSide(input_id="annual_gdp"),
                )
                operation = ShareOfOperation(
                    id="combined",
                    requirement_id="combined",
                    numerator=Side(input=debt_id),
                    denominator=Side(input=gdp_id),
                )
            case "quotient_of":
                derived = QuotientRequirement(
                    id="combined",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="annual_debt_service"),
                    denominator=RequirementSide(input_id="annual_gdp"),
                )
                operation = QuotientOfOperation(
                    id="combined",
                    requirement_id="combined",
                    numerator=Side(input=debt_id),
                    denominator=Side(input=gdp_id),
                )
            case _:
                raise AssertionError(f"unknown terminal kind: {kind!r}")
        draft = Plan(
            question="How does Kenya's debt service compare with its GDP?",
            decisions=[decision()],
            evidence_requirements=[requirement(), gdp, derived],
            status=PlanStatus.DRAFT,
        )
        translation = Translation(
            operations=[
                RetrieveOperation(
                    id=debt_id, requirement_id="annual_debt_service", spec_id=SPEC_KENYA
                ),
                RetrieveOperation(id=gdp_id, requirement_id="annual_gdp", spec_id=SPEC_GDP),
                operation,
            ],
            terminal=["combined"],
        )
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered={
                "annual_debt_service": overrides.pop("source", source()),
                "annual_gdp": overrides.pop("gdp_source", gdp_source()),
            },
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()

    return _two_input


@pytest.fixture
def join_plan(two_input_plan: Callable[..., Plan]) -> Callable[..., Plan]:
    """Return a factory for an approved plan joining a debt and a GDP retrieval into one frame."""

    def _join(**overrides: Any) -> Plan:
        return two_input_plan(kind="join", **overrides)

    return _join
