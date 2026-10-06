"""A CRS family frame through compile and execute, for every verb.

The compiler names role columns by flow axis (`origin`, `destination`), the
way search resolves a flow dataset's entities. The frame `get_data` registers
names them by declared property (`financingProvider`, `financingRecipient`),
carries four `stat_var_*` columns besides, and holds one StatVar per family
member the source has data for. Each test compiles a real plan over that
source, holds a real frame in the store analysis-mcp reads, scripts
`get_data` to report that frame's `role_keys` and `stat_var_count` and each
coverage dimension's axis, and runs every analysis step on the real
analysis-mcp server (`FrameSurface`). A verb answer is scripted only where a
test needs a result no real verb returns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

import pytest
from analysis_mcp import server as analysis_server
from common.materialisation import FrameRef, InMemoryMaterialisationStore, MaterialisationTrace
from compiler.compile import compile_plan
from compiler.sources import member_digest, pinned_stat_var
from compiler.surface import McpSurface, ProgressCallback, ToolCallResult, ToolSchema
from data_mcp.frame_materialisation import observed_shape
from dfo_frame_grammar import Frame
from executor.outcomes import Accepted, MethodChoiceNeeded, RefusalKind, Refused
from executor.run import execute_approved_plan
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    SumOfRequirement,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    DenominatorScope,
    EntityIn,
    PeriodThreshold,
    RankAmong,
    RankDirection,
    StatVarReading,
)
from planner.workflow import OpenAxis, OpenEntityAxis, ResolvedEntity, SelectedSource, ToolId

from .conftest import ScriptedSurface, ok

SPEC_CRS = "spec_" + "e" * 32
SPEC_CRS_HEALTH = "spec_" + "f" * 32
CONTENT_VERSION = "cv_" + "4" * 32

GERMANY = ResolvedEntity(role="origin", ref="country/DEU", label="Germany")
FRANCE = ResolvedEntity(role="origin", ref="country/FRA", label="France")
KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")

PROVIDER = "financingProvider"
RECIPIENT = "financingRecipient"
_SLOT_COLUMNS = (
    "stat_var_crsCooperationModality",
    "stat_var_crsPurpose",
    "stat_var_crsTypeOfFinance",
    "stat_var_crsTypeOfFlow",
)
FRAME_ROLE_KEYS = (PROVIDER, RECIPIENT, *_SLOT_COLUMNS)
# The coverage dimensions dfo-apis reports for this frame: each declared role
# column with its flow axis, then `years` with none.
DIMENSIONS = [
    {"label": PROVIDER, "count": 2, "axis": "origin"},
    {"label": RECIPIENT, "count": 1, "axis": "destination"},
    {"label": "years", "count": 10},
]
# The same dimensions from a dfo-apis that reports no axis.
DIMENSIONS_WITHOUT_AXES = [{"label": d["label"], "count": d["count"]} for d in DIMENSIONS]

# Three members of one CRS family. `OdaEquity` reports through 2004 only, so
# a filter from 2005 keeps two of the three.
FAMILY = ("OdaGrants", "OdaLoans", "OdaEquity")
_LAST_YEAR = {"OdaEquity": 2004}

_STATISTICS_DIGEST = "sha256:" + "8" * 64

PlanFactory = Callable[..., Plan]
Retrieved = Callable[..., Awaitable[dict[str, Any]]]


def _source(
    *,
    spec_id: str = SPEC_CRS,
    entities: tuple[ResolvedEntity, ...] = (GERMANY, FRANCE, KENYA),
    open_entity_axes: tuple[OpenEntityAxis, ...] | None = None,
) -> SelectedSource:
    return SelectedSource(
        spec_id=spec_id,
        open_entity_axes=open_entity_axes,
        sentence="bilateral ODA from Germany and France to Kenya, 2000 to 2023",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["Oda"]),
        entities=entities,
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2000, end=2023),
        identity=_pinning("Oda", spec_id),
    )


def _pinning(stat_var: str, spec_id: str) -> dict[str, Any]:
    """An identity pinning `stat_var`, told apart per spec_id."""
    return {"shape": {"kind": "stat_var", "id": stat_var}, "spec": spec_id}


def _pinned(source: SelectedSource) -> str:
    """The StatVar `source`'s identity pins."""
    stat_var = pinned_stat_var(source.identity)
    assert stat_var is not None
    return stat_var


def _requirement(
    *,
    id: str = "oda",
    donors: Sequence[str] = ("Germany", "France"),
    recipients: Sequence[str] = ("Kenya",),
) -> DatasetRequirement:
    return DatasetRequirement(
        id=id,
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[
            *(Entity(name=name, role="donor") for name in donors),
            *(Entity(name=name, role="recipient") for name in recipients),
        ],
        period=Period(start=2000, end=2023),
    )


def _family_frame(
    *,
    stat_vars: Sequence[str] = ("Oda",),
    providers: Sequence[ResolvedEntity] = (GERMANY, FRANCE),
    provider_column: str = PROVIDER,
    unit: str = "usd",
) -> Frame:
    """A CRS family frame: each of `stat_vars` for each provider to Kenya, yearly from 2000.

    Two declared role columns and the four `stat_var_*` slot columns, whose
    purpose slot tells the family's members apart.
    """
    return Frame.build(
        role_keys=[provider_column, RECIPIENT, *_SLOT_COLUMNS],
        rows=[
            {
                "stat_var": f"dcid:{stat_var}",
                provider_column: provider.ref,
                RECIPIENT: KENYA.ref,
                "stat_var_crsCooperationModality": "crsCooperationModality/Project",
                "stat_var_crsPurpose": f"crsPurpose/{stat_var}",
                "stat_var_crsTypeOfFinance": "crsTypeOfFinance/Standard",
                "stat_var_crsTypeOfFlow": "crsTypeOfFlow/Oda",
                "time_period": str(year),
                "value": 100.0 + member + year,
                "unit": unit,
            }
            for member, stat_var in enumerate(stat_vars)
            for provider in providers
            for year in range(2000, _LAST_YEAR.get(stat_var, 2023) + 1)
        ],
    )


def _stat_vars(frame: Frame) -> int:
    return len(set(frame.table.column("stat_var").to_pylist()))


class FrameSurface(ScriptedSurface):
    """A `ScriptedSurface` whose unscripted tools run on the real analysis-mcp server."""

    def __init__(
        self,
        *,
        schemas: Sequence[ToolSchema],
        responses: Mapping[str, Sequence[ToolCallResult | Exception]],
        engines: McpSurface,
    ) -> None:
        super().__init__(schemas=schemas, responses=responses)
        self._engines = engines

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name in self.queues:
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.calls.append((tool.name, dict(arguments)))
        return await self._engines.call_tool(tool=tool, arguments=arguments)


@pytest.fixture
def frame_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryMaterialisationStore:
    """The store analysis-mcp reads frames from and writes its results to."""
    store = InMemoryMaterialisationStore()

    async def _build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(analysis_server, "_build_store", _build_store)
    return store


@pytest.fixture
async def surface_for(
    frame_store: InMemoryMaterialisationStore, real_schemas: list[ToolSchema]
) -> AsyncIterator[Callable[..., FrameSurface]]:
    """Return a factory for a `FrameSurface` answering `get_data` (and any other
    tool named) from `responses`."""
    del frame_store
    engines = McpSurface(sources={"analysis-mcp": analysis_server.mcp})

    def _surface(**responses: Sequence[ToolCallResult | Exception]) -> FrameSurface:
        return FrameSurface(schemas=real_schemas, responses=responses, engines=engines)

    yield _surface
    await engines.aclose()


@pytest.fixture
def retrieved(
    frame_store: InMemoryMaterialisationStore,
    _stats_by_manifest: dict[tuple[str, str], list[dict[str, Any]]],
) -> Retrieved:
    """Return a factory holding a frame for `source` and giving `get_data`'s result over it.

    The frame is a family frame of one StatVar for each of `providers`, by
    default the donors `source` names, unless `frame` gives another.
    `role_keys` and `stat_var_count` are read off the frame the way `get_data`
    reads them (`data_mcp.frame_materialisation.observed_shape`), and
    `dimensions` are what dfo-apis reports. `role_keys` overrides the
    reported keys. `frame_read=False` leaves both out, as `get_data` does
    when it could not read the frame.
    """

    async def _retrieved(
        source: SelectedSource,
        *,
        frame: Frame | None = None,
        providers: Sequence[ResolvedEntity] | None = None,
        role_keys: Sequence[str] | None = None,
        dimensions: list[dict[str, Any]] | None = None,
        frame_read: bool = True,
    ) -> dict[str, Any]:
        if providers is None:
            providers = [entity for entity in source.entities if entity.role == "origin"]
        held = _family_frame(providers=providers) if frame is None else frame
        ref = FrameRef(spec_id=source.spec_id, content_version=CONTENT_VERSION)
        trace = {
            "applied_window": None,
            "graph_revision": "rev_2026_10_01",
            "source_revisions": [{"source": "oecd_crs", "revision": "2026-10-01"}],
            "resolved_stat_vars": [{"ref": f"dcid:{_pinned(source)}", "roles": []}],
            "family_member_digest": source.shape_member_digest,
            "family_member_count": 1,
            "facets": [{"axis": "unit", "ref": "usd", "basis": "declared_default"}],
        }
        await frame_store.put(
            ref,
            frame=held,
            trace=MaterialisationTrace(
                upstream_trace=trace, spec_id=ref.spec_id, content_version=ref.content_version
            ),
            writer="tests",
        )
        statistics = [
            {
                "stat_var": f"dcid:{_pinned(source)}",
                "entities": [
                    {"role": PROVIDER, "ref": provider.ref},
                    {"role": RECIPIENT, "ref": KENYA.ref},
                ],
                "facet": {"unit": "usd"},
                "observations": 24,
                "first": {"value": 100.0, "period": "2000"},
                "latest": {"value": 123.0, "period": "2023"},
                "minimum": {"value": 100.0, "period": "2000"},
                "maximum": {"value": 123.0, "period": "2023"},
            }
            for provider in providers
        ]
        _stats_by_manifest[(source.spec_id, CONTENT_VERSION)] = statistics
        result: dict[str, Any] = {
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": source.spec_id,
                "identity": dict(source.identity or {}),
            },
            "content_version": CONTENT_VERSION,
            "coverage": {
                "rows": held.num_rows,
                "dimensions": DIMENSIONS if dimensions is None else dimensions,
            },
            "citation": {"text": "OECD, Creditor Reporting System", "sources": []},
            "trace": trace,
            "warnings": [],
            "artifacts": [
                {
                    "artifact_digest": _STATISTICS_DIGEST,
                    "rendering": {"statistics": "series"},
                    "bytes": 512,
                    "media_type": "application/vnd.apache.parquet",
                    "created_at": "2026-10-01T00:00:00+00:00",
                    "tier": "working",
                }
            ],
            "series_count": len(statistics),
            "frame_ref": ref.model_dump(),
        }
        if frame_read:
            frame_role_keys, stat_var_count = observed_shape(held)
            result["role_keys"] = list(frame_role_keys if role_keys is None else role_keys)
            result["stat_var_count"] = stat_var_count
        return result

    return _retrieved


def _scripted_result(*role_keys: str, stat_var_count: int = 1) -> dict[str, Any]:
    """A scripted verb result, for a shape no real verb returns over the frame it reads."""
    return {
        "spec_id": "ops_" + "5" * 32,
        "content_version": "ocv_" + "6" * 32,
        "rows": 3,
        "role_keys": list(role_keys),
        "stat_var_count": stat_var_count,
    }


@pytest.fixture
def plan_for(decision: Callable[..., Decision], real_schemas: list[ToolSchema]) -> PlanFactory:
    """Return a factory compiling and approving a plan over CRS family sources."""

    def _plan(
        requirements: Sequence[EvidenceRequirement],
        operations: Sequence[Operation],
        *,
        terminal: str,
        discovered: dict[str, SelectedSource] | None = None,
        decisions: Sequence[Decision] = (),
    ) -> Plan:
        draft = Plan(
            question="How much bilateral ODA do Germany and France give Kenya?",
            decisions=[decision(), *decisions],
            evidence_requirements=list(requirements),
            status=PlanStatus.DRAFT,
        )
        translation = Translation(operations=list(operations), terminal=[terminal])
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered=discovered or {"oda": _source()},
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()

    return _plan


_RETRIEVE = RetrieveOperation(id="crs", requirement_id="oda", spec_id=SPEC_CRS)
_RETRIEVE_HEALTH = RetrieveOperation(
    id="crs_health", requirement_id="oda_health", spec_id=SPEC_CRS_HEALTH
)
_GERMANY_ONLY = (GERMANY,)


def _strings(value: Any) -> set[str]:
    """Every string anywhere in `value`, object keys included."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set(value) | {s for item in value.values() for s in _strings(item)}
    if isinstance(value, list):
        return {s for item in value for s in _strings(item)}
    return set()


def _get_data_answers(plan: Plan, results: dict[str, dict[str, Any]]) -> list[Any]:
    """`get_data`'s answers in the order the plan's retrievals call it, by spec_id."""
    assert plan.workflow is not None
    return [
        ok(results[str(step.args["spec_id"])])
        for step in plan.workflow.steps
        if step.tool.name == "get_data"
    ]


async def _run(plan: Plan, surface: ScriptedSurface) -> Any:
    _, outcome = await execute_approved_plan(plan, surface=surface)
    return outcome


def _analysis_call(surface: ScriptedSurface, tool: str) -> dict[str, Any]:
    (call,) = [arguments for name, arguments in surface.calls if name == tool]
    assert not {"origin", "destination"} & _strings(call), call
    return {key: value for key, value in call.items() if not key.endswith("frame_ref")}


def _since_2005_plan(plan_for: PlanFactory, source: SelectedSource) -> Plan:
    since = FilterRequirement(
        id="since",
        decision_ids=["measure"],
        input_id="oda",
        predicates=(PeriodThreshold(op="gte", year=2005),),
    )
    return plan_for(
        [_requirement(donors=("Germany",), recipients=()), since],
        [
            _RETRIEVE,
            FilterOperation(
                id="since_op",
                requirement_id="since",
                input="crs",
                predicates=(PeriodThreshold(op="gte", year=2005),),
            ),
        ],
        terminal="since_op",
        discovered={"oda": source},
    )


def _aggregate_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    del decision
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )
    return plan_for(
        [_requirement(), totals],
        [
            _RETRIEVE,
            AggregateOperation(
                id="totals_op",
                requirement_id="totals",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            ),
        ],
        terminal="totals_op",
    )


def _rank_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    fields: dict[str, Any] = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "donor",
        "reduction": Reduction.MEAN,
        "window_start": 2020,
        "window_end": 2023,
    }
    return plan_for(
        [
            _requirement(),
            RankRequirement(
                id="ranked", decision_ids=["measure", "population"], input_id="oda", **fields
            ),
        ],
        [_RETRIEVE, RankOperation(id="ranked_op", requirement_id="ranked", input="crs", **fields)],
        terminal="ranked_op",
        decisions=[decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))],
    )


def _growth_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    del decision
    window: dict[str, Any] = {"window_start": 2010, "window_end": 2020}
    return plan_for(
        [
            _requirement(),
            GrowthRequirement(id="growth", decision_ids=["measure"], input_id="oda", **window),
        ],
        [
            _RETRIEVE,
            GrowthOperation(id="growth_op", requirement_id="growth", input="crs", **window),
        ],
        terminal="growth_op",
    )


def _share_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    """Germany's ODA to health as a share of its total ODA, per donor."""
    del decision
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="oda_health"),
        denominator=RequirementSide(input_id="oda"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    return plan_for(
        [
            _requirement(donors=("Germany",), recipients=()),
            _requirement(id="oda_health", donors=("Germany",), recipients=()),
            share,
        ],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            ShareOfOperation(
                id="share_op",
                requirement_id="share",
                numerator=Side(input="crs_health"),
                denominator=Side(input="crs"),
                axis="donor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
        ],
        terminal="share_op",
        discovered={
            "oda": _source(entities=_GERMANY_ONLY),
            "oda_health": _source(spec_id=SPEC_CRS_HEALTH, entities=_GERMANY_ONLY),
        },
    )


def _compare_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    del decision
    compared = ComparisonRequirement(
        id="compared",
        decision_ids=["measure"],
        left=RequirementSide(input_id="oda", entity="Germany"),
        right=RequirementSide(input_id="oda", entity="France"),
    )
    return plan_for(
        [_requirement(), compared],
        [
            _RETRIEVE,
            CompareOperation(
                id="compared_op",
                requirement_id="compared",
                left=Side(input="crs", entity="Germany"),
                right=Side(input="crs", entity="France"),
            ),
        ],
        terminal="compared_op",
    )


def _join_plan(plan_for: PlanFactory, decision: Callable[..., Decision] | None = None) -> Plan:
    del decision
    health = _requirement(id="oda_health")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="oda", right_id="oda_health"
    )
    return plan_for(
        [_requirement(), health, joined],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            JoinOperation(id="joined_op", requirement_id="joined", left="crs", right="crs_health"),
        ],
        terminal="joined_op",
        discovered={"oda": _source(), "oda_health": _source(spec_id=SPEC_CRS_HEALTH)},
    )


def _sources_of(plan: Plan) -> dict[str, SelectedSource]:
    assert plan.workflow is not None
    return {source.spec_id: source for source in plan.workflow.sources}


# --- Role columns: every verb is sent the frame's own spelling ---------------


@pytest.mark.parametrize("dimensions", [DIMENSIONS, DIMENSIONS_WITHOUT_AXES])
async def test_a_period_filter_on_a_family_frame_declares_every_frame_column(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    dimensions: list[dict[str, Any]],
) -> None:
    # Germany's bilateral ODA since 2000, filtered from 2005: the plan names
    # Germany alone, under the `origin` column, and the frame carries both
    # declared roles and four `stat_var_*` columns. A period filter names no
    # role column, so it runs whether or not dfo-apis reports axes.
    source = _source(entities=_GERMANY_ONLY)
    plan = _since_2005_plan(plan_for, source)
    assert plan.workflow is not None
    assert plan.workflow.steps[-1].expected_role_keys == ("origin",)
    surface = surface_for(get_data=[ok(await retrieved(source, dimensions=dimensions))])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "filter") == {
        "predicates": [
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2005"}
        ]
    }


async def test_a_filter_that_drops_a_frame_column_is_still_a_shape_mismatch(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    source = _source(entities=_GERMANY_ONLY)
    surface = surface_for(
        get_data=[ok(await retrieved(source))], filter=[ok(_scripted_result(PROVIDER))]
    )

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SHAPE_MISMATCH
    assert outcome.step_id == "since_op"


async def test_an_entity_filter_names_the_frames_recipient_column(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    to_kenya = FilterRequirement(
        id="to_kenya",
        decision_ids=["measure"],
        input_id="oda",
        predicates=(EntityIn(names=("Kenya",)),),
    )
    plan = plan_for(
        [_requirement(), to_kenya],
        [
            _RETRIEVE,
            FilterOperation(
                id="to_kenya_op",
                requirement_id="to_kenya",
                input="crs",
                predicates=(EntityIn(names=("Kenya",)),),
            ),
        ],
        terminal="to_kenya_op",
    )
    surface = surface_for(get_data=[ok(await retrieved(_source()))])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[-1].row_count == 48
    assert _analysis_call(surface, "filter") == {
        "predicates": [{"kind": "category_match", "column": RECIPIENT, "values": ["country/KEN"]}]
    }


async def test_an_aggregate_by_entity_groups_by_the_frames_role_columns(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    surface = surface_for(get_data=[ok(await retrieved(_source()))])

    outcome = await _run(_aggregate_plan(plan_for, decision), surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "aggregate") == {
        "aggregation": "sum",
        "group_by": [PROVIDER, RECIPIENT, "time_period"],
    }


async def test_a_rank_ranks_by_the_frames_provider_column(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    surface = surface_for(get_data=[ok(await retrieved(_source()))])

    outcome = await _run(_rank_plan(plan_for, decision), surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "rank") == {
        "rank_by": [PROVIDER],
        "ascending": False,
        "aggregation": "mean",
        "window_start": "2020",
        "window_end": "2023",
    }


async def test_a_rank_over_every_donor_the_source_leaves_open_ranks_the_frames_provider_column(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # "Top donors of ODA to Kenya": the source binds Kenya and leaves the
    # donor side open as `origin`, and the frame carries the donors under
    # `financingProvider`, which the manifest places on the `origin` axis.
    source = _source(
        entities=(KENYA,), open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),)
    )
    every_donor = DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[Entity(name="Kenya", role="recipient")],
        across="donor",
        period=Period(start=2000, end=2023),
    )
    fields: dict[str, Any] = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "donor",
        "reduction": Reduction.SUM,
    }
    plan = plan_for(
        [
            every_donor,
            RankRequirement(
                id="ranked", decision_ids=["measure", "population"], input_id="oda", **fields
            ),
        ],
        [_RETRIEVE, RankOperation(id="ranked_op", requirement_id="ranked", input="crs", **fields)],
        terminal="ranked_op",
        discovered={"oda": source},
        decisions=[decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))],
    )
    surface = surface_for(get_data=[ok(await retrieved(source, providers=(GERMANY, FRANCE)))])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "rank") == {
        "rank_by": [PROVIDER],
        "ascending": False,
        "aggregation": "sum",
    }


def _sector_frame(*, spec_id: str, totals: Mapping[tuple[ResolvedEntity, str], float]) -> Frame:
    """A frame as dfo-apis rolls a family up to sector groups: `stat_var` is the spec's own
    ref on every row, the role columns are kept, and `stat_var_crsPurpose` names each
    row's group. A group with no records for a donor has no row, never a zero."""
    return Frame.build(
        role_keys=[PROVIDER, RECIPIENT, "stat_var_crsPurpose"],
        rows=[
            {
                "stat_var": f"dcid:{spec_id}",
                PROVIDER: provider.ref,
                RECIPIENT: KENYA.ref,
                "stat_var_crsPurpose": sector,
                "time_period": "2024",
                "value": value,
                "unit": "usd",
            }
            for (provider, sector), value in totals.items()
        ],
    )


async def test_which_sectors_received_the_most_aid_in_kenya_ranks_the_frames_sector_column(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    frame_store: InMemoryMaterialisationStore,
) -> None:
    # "Which sectors received the most aid in Kenya in 2024?": the retrieval
    # sums the family within each top-level purpose code, the aggregate totals
    # every donor within each sector, and the rank orders the sectors. Neither
    # donor gave to the production sector, so it has no row and no rank.
    source = _source(
        entities=(KENYA,), open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),)
    ).model_copy(
        update={
            "served_count": None,
            "open_axes": (OpenAxis(property="crsPurpose", summable=True),),
        }
    )
    every_donor = DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[Entity(name="Kenya", role="recipient")],
        across="donor",
        period=Period(start=2024, end=2024),
        stat_vars=StatVarReading.SEPARATE,
        breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0),
    )
    by_sector: dict[str, Any] = {
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.PERIOD, GroupBy.BREAKDOWN),
    }
    rank: dict[str, Any] = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "sector",
        "limit": 10,
    }
    plan = plan_for(
        [
            every_donor,
            AggregateRequirement(
                id="by_sector", decision_ids=["measure"], input_id="oda", **by_sector
            ),
            RankRequirement(
                id="top", decision_ids=["measure", "population"], input_id="by_sector", **rank
            ),
        ],
        [
            _RETRIEVE,
            AggregateOperation(
                id="by_sector_op", requirement_id="by_sector", input="crs", **by_sector
            ),
            RankOperation(id="top_op", requirement_id="top", input="by_sector_op", **rank),
        ],
        terminal="top_op",
        discovered={"oda": source},
        decisions=[decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))],
    )
    frame = _sector_frame(
        spec_id=SPEC_CRS,
        totals={
            (GERMANY, "Social infrastructure"): 300.0,
            (FRANCE, "Social infrastructure"): 200.0,
            (GERMANY, "Economic infrastructure"): 400.0,
            (FRANCE, "Humanitarian aid"): 50.0,
        },
    )
    surface = surface_for(
        get_data=[ok(await retrieved(source, frame=frame, providers=(GERMANY, FRANCE)))]
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    ((_, get_data),) = [call for call in surface.calls if call[0] == "get_data"]
    assert get_data["breakdown"] == {"code_system": "crsPurpose", "level": 0}
    assert _analysis_call(surface, "aggregate") == {
        "aggregation": "sum",
        "group_by": ["time_period", "stat_var_crsPurpose"],
    }
    assert _analysis_call(surface, "rank")["rank_by"] == ["stat_var_crsPurpose"]
    held = await frame_store.get(outcome.results["top_op"])
    assert held is not None
    ranked = held[0].table.to_pylist()
    assert {row["stat_var_crsPurpose"]: row["value"] for row in ranked} == {
        "Social infrastructure": 500.0,
        "Economic infrastructure": 400.0,
        "Humanitarian aid": 50.0,
    }


async def test_a_growth_keys_each_series_by_the_frames_role_columns(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    surface = surface_for(get_data=[ok(await retrieved(_source()))])

    outcome = await _run(_growth_plan(plan_for, decision), surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "growth") == {
        "window_start": "2010",
        "window_end": "2020",
        "entity_by": [PROVIDER, RECIPIENT],
    }


async def test_a_share_of_keys_each_side_by_its_own_frames_provider_column(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # The denominator's frame spells the provider column `provider`, so each
    # side of `denominator_by` binds against its own frame.
    plan = _share_plan(plan_for, decision)
    results = {
        SPEC_CRS: await retrieved(
            _source(entities=_GERMANY_ONLY),
            frame=_family_frame(providers=_GERMANY_ONLY, provider_column="provider"),
            dimensions=[{"label": "provider", "count": 1, "axis": "origin"}, *DIMENSIONS[1:]],
        ),
        SPEC_CRS_HEALTH: await retrieved(_source(spec_id=SPEC_CRS_HEALTH, entities=_GERMANY_ONLY)),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "share_of") == {
        "numerator_by": [PROVIDER],
        "denominator_by": {PROVIDER: "provider"},
    }


async def test_a_compare_selects_each_side_by_the_frames_provider_column(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    surface = surface_for(get_data=[ok(await retrieved(_source()))])

    outcome = await _run(_compare_plan(plan_for, decision), surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "compare") == {
        "left_entity": {PROVIDER: "country/DEU"},
        "right_entity": {PROVIDER: "country/FRA"},
    }


async def test_a_sum_of_two_sources_pins_each_sides_unit_and_yields_one_series(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # Bilateral ODA plus a second source's imputed ODA, each from Germany
    # to Kenya: each side names its donor to read one series, and the two
    # are added year by year.
    bilateral = _source(entities=(GERMANY, KENYA))
    imputed = _source(spec_id=SPEC_CRS_HEALTH, entities=(GERMANY, KENYA)).model_copy(
        update={
            "identity": _pinning("OdaImputed", SPEC_CRS_HEALTH),
            "shape_member_digest": member_digest(["OdaImputed"]),
        }
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id="oda", entity="Germany"),
        right=RequirementSide(input_id="oda_health", entity="Germany"),
    )
    germany_to_kenya = {"donors": ("Germany",)}
    plan = plan_for(
        [
            _requirement(**germany_to_kenya),
            _requirement(id="oda_health", **germany_to_kenya),
            total,
        ],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            SumOfOperation(
                id="total_op",
                requirement_id="total",
                left=Side(input="crs", entity="Germany"),
                right=Side(input="crs_health", entity="Germany"),
            ),
        ],
        terminal="total_op",
        discovered={"oda": bilateral, "oda_health": imputed},
    )
    results = {
        SPEC_CRS: await retrieved(bilateral),
        SPEC_CRS_HEALTH: await retrieved(
            imputed, frame=_family_frame(stat_vars=("OdaImputed",), providers=_GERMANY_ONLY)
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[-1].row_count == 24
    assert outcome.steps[-1].role_keys == ()
    assert outcome.steps[-1].stat_var_count == 1
    assert _analysis_call(surface, "sum_of") == {
        "left_entity": {PROVIDER: "country/DEU"},
        "right_entity": {PROVIDER: "country/DEU"},
        "left_unit": "usd",
        "right_unit": "usd",
    }


async def test_a_join_keys_both_frames_by_their_shared_role_columns(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    plan = _join_plan(plan_for)
    results = {
        SPEC_CRS: await retrieved(_source()),
        SPEC_CRS_HEALTH: await retrieved(
            _source(spec_id=SPEC_CRS_HEALTH), frame=_family_frame(stat_vars=("OdaHealth",))
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[-1].stat_var_count == 2
    assert _analysis_call(surface, "join") == {"join_keys": [PROVIDER, RECIPIENT]}


async def test_a_join_key_the_two_frames_spell_differently_refuses_before_the_call(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    plan = _join_plan(plan_for)
    results = {
        SPEC_CRS: await retrieved(_source()),
        SPEC_CRS_HEALTH: await retrieved(
            _source(spec_id=SPEC_CRS_HEALTH),
            frame=_family_frame(stat_vars=("OdaHealth",), provider_column="provider"),
            dimensions=[{"label": "provider", "count": 2, "axis": "origin"}, *DIMENSIONS[1:]],
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.COLUMN_UNBOUND
    assert outcome.step_id == "joined_op"
    assert "origin" in outcome.detail
    assert [name for name, _ in surface.calls] == ["get_data", "get_data"]


async def test_a_role_the_frame_has_no_column_for_refuses_before_the_call(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # A dfo-apis that reports no axis leaves `origin` and `destination` with
    # no column to bind to; grouping by them cannot be sent.
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), dimensions=DIMENSIONS_WITHOUT_AXES))]
    )

    outcome = await _run(_aggregate_plan(plan_for, decision), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.COLUMN_UNBOUND
    assert outcome.step_id == "totals_op"
    assert [name for name, _ in surface.calls] == ["get_data"]


async def test_a_retrieval_reporting_a_reserved_role_key_is_refused_not_raised(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # `value` is a reserved frame column, which `normalize_role_keys` refuses
    # as a role key. The run stops on the retrieval before the filter is called.
    source = _source(entities=_GERMANY_ONLY)
    surface = surface_for(get_data=[ok(await retrieved(source, role_keys=(PROVIDER, "value")))])

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_RESULT_UNUSABLE
    assert outcome.step_id == "crs"
    assert "not valid role columns" in outcome.detail
    assert [tool for tool, _arguments in surface.calls] == ["get_data"]


async def test_a_verb_declaring_a_reserved_role_key_is_refused_not_raised(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    source = _source(entities=_GERMANY_ONLY)
    surface = surface_for(
        get_data=[ok(await retrieved(source))],
        filter=[ok(_scripted_result(*FRAME_ROLE_KEYS, "value"))],
    )

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_RESULT_UNUSABLE
    assert outcome.step_id == "since_op"
    assert "not valid role columns" in outcome.detail


async def test_a_retrieval_reporting_no_role_keys_keeps_the_plans_own_spelling(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # The frame carries no `origin` column, so the real `aggregate` would
    # refuse the plan's own spelling; its answer is scripted to read the call.
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), frame_read=False))],
        aggregate=[ok(_scripted_result("destination", "origin"))],
    )

    outcome = await _run(_aggregate_plan(plan_for, decision), surface)

    assert isinstance(outcome, Accepted), outcome
    ((_, aggregate_call),) = [call for call in surface.calls if call[0] == "aggregate"]
    assert aggregate_call["group_by"] == ["origin", "destination", "time_period"]


# --- StatVars: a family frame holds several, and each verb's count follows its input ---


async def test_a_period_filter_over_a_family_frame_declares_the_stat_vars_it_kept(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # Three family members, one of which reports through 2004 only. The
    # retrieval declares three StatVars and the filter from 2005 two.
    source = _source(entities=_GERMANY_ONLY)
    frame = _family_frame(stat_vars=FAMILY, providers=_GERMANY_ONLY)
    assert _stat_vars(frame) == 3
    surface = surface_for(
        get_data=[ok(await retrieved(source, frame=frame, providers=(GERMANY, FRANCE)))]
    )

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Accepted), outcome
    retrieval, filtered = outcome.steps
    assert retrieval.stat_var_count == 3
    assert filtered.stat_var_count == 2
    assert filtered.row_count == 2 * 19


async def test_a_filter_declaring_more_stat_vars_than_its_input_holds_is_a_shape_mismatch(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    source = _source(entities=_GERMANY_ONLY)
    frame = _family_frame(stat_vars=FAMILY, providers=_GERMANY_ONLY)
    surface = surface_for(
        get_data=[ok(await retrieved(source, frame=frame))],
        filter=[ok(_scripted_result(*FRAME_ROLE_KEYS, stat_var_count=4))],
    )

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SHAPE_MISMATCH
    assert outcome.step_id == "since_op"
    assert "declared 4 distinct stat_vars over an input declaring 3" in outcome.detail


async def test_a_filter_over_a_frame_of_unknown_stat_var_count_declares_its_own(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # `get_data` did not read the frame, so the filter's count has no bound
    # above and its role keys are the plan's own. The real filter declares
    # the frame's columns, so its answer is scripted.
    source = _source(entities=_GERMANY_ONLY)
    frame = _family_frame(stat_vars=FAMILY, providers=_GERMANY_ONLY)
    surface = surface_for(
        get_data=[ok(await retrieved(source, frame=frame, frame_read=False))],
        filter=[ok(_scripted_result("origin", stat_var_count=2))],
    )

    outcome = await _run(_since_2005_plan(plan_for, source), surface)

    assert isinstance(outcome, Accepted), outcome
    assert [step.stat_var_count for step in outcome.steps] == [None, 2]


_SINGLE_STAT_VAR_PLANS: dict[str, tuple[Callable[..., Plan], str]] = {
    "aggregate": (_aggregate_plan, "stat_var_unresolved"),
    "growth": (_growth_plan, "stat_var_unresolved"),
    "rank": (_rank_plan, "stat_var_unresolved"),
    "share_of": (_share_plan, "stat_var_unresolved"),
    "join": (_join_plan, "stat_var_unresolved"),
    "compare": (_compare_plan, "role_ambiguous"),
}


@pytest.mark.parametrize("verb", sorted(_SINGLE_STAT_VAR_PLANS))
async def test_a_verb_reading_one_stat_var_refuses_a_family_frame_itself(
    verb: str,
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # The real verb runs over a frame holding three StatVars, and the run
    # stops on the verb's own refusal code.
    build, code = _SINGLE_STAT_VAR_PLANS[verb]
    plan = build(plan_for, decision)
    results = {
        spec_id: await retrieved(
            source,
            frame=_family_frame(
                stat_vars=FAMILY,
                providers=[entity for entity in source.entities if entity.role == "origin"],
            ),
        )
        for spec_id, source in _sources_of(plan).items()
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused | MethodChoiceNeeded), outcome
    refusal = outcome.payload if isinstance(outcome, Refused) else outcome.refusal
    assert refusal is not None, outcome
    assert refusal.code == code
    assert outcome.step_id == plan.workflow.terminal_steps[0]
    assert surface.calls[-1][0] == verb


_OVER_DECLARED: dict[str, tuple[Callable[..., Plan], tuple[str, ...], int]] = {
    "aggregate": (_aggregate_plan, (PROVIDER, RECIPIENT), 2),
    "growth": (_growth_plan, (PROVIDER, RECIPIENT), 2),
    "rank": (_rank_plan, (PROVIDER,), 2),
    "share_of": (_share_plan, (PROVIDER,), 2),
    "compare": (_compare_plan, (), 2),
    "join": (_join_plan, (PROVIDER, RECIPIENT), 3),
}


@pytest.mark.parametrize("verb", sorted(_OVER_DECLARED))
async def test_a_verb_declaring_more_stat_vars_than_it_can_yield_is_a_shape_mismatch(
    verb: str,
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # Each input holds one StatVar. The verb's role keys are the ones the
    # plan expects, so the declared StatVar count alone is out of bounds.
    build, role_keys, declared = _OVER_DECLARED[verb]
    plan = build(plan_for, decision)
    results = {spec_id: await retrieved(source) for spec_id, source in _sources_of(plan).items()}
    surface = surface_for(
        get_data=_get_data_answers(plan, results),
        **{verb: [ok(_scripted_result(*role_keys, stat_var_count=declared))]},
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SHAPE_MISMATCH
    assert f"declared {declared} distinct stat_vars" in outcome.detail


async def test_a_join_of_one_stat_var_under_two_units_declares_one(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    plan = _join_plan(plan_for)
    results = {
        SPEC_CRS: await retrieved(_source()),
        SPEC_CRS_HEALTH: await retrieved(
            _source(spec_id=SPEC_CRS_HEALTH), frame=_family_frame(unit="eur")
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[-1].stat_var_count == 1
    assert outcome.steps[-1].row_count == 2 * 48
