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
from datetime import UTC, datetime
from functools import partial
from typing import Any

import pytest
from analysis_mcp import server as analysis_server
from common.materialisation import FrameRef, InMemoryMaterialisationStore, MaterialisationTrace
from compiler.compile import CompileResult, compile_plan
from compiler.sources import member_digest, pinned_stat_var
from compiler.surface import McpSurface, ProgressCallback, ToolCallResult, ToolSchema
from data_mcp.frame_materialisation import observed_shape
from dfo_frame_grammar import Frame
from executor.outcomes import (
    Accepted,
    AnchorResult,
    MethodChoiceNeeded,
    RefusalKind,
    Refused,
)
from executor.run import AnchorRead, execute_approved_plan, read_anchors
from harness.graph.answer import answer_notes
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
    FirstYear,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    SumOfRequirement,
    plan_revision,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    DenominatorScope,
    EntityIn,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    RelativePeriod,
    RollingWindow,
    StatVarReading,
)
from planner.workflow import OpenAxis, OpenEntityAxis, ResolvedEntity, SelectedSource, ToolId
from pydantic import JsonValue
from translator.tool import restated_to_the_data

from ..runner_frames import hold_root, result_table
from .conftest import ScriptedSurface, ok

SPEC_CRS = "spec_" + "e" * 32
SPEC_CRS_HEALTH = "spec_" + "f" * 32
CONTENT_VERSION = "cv_" + "4" * 32

GERMANY = ResolvedEntity(role="origin", ref="country/DEU", label="Germany")
FRANCE = ResolvedEntity(role="origin", ref="country/FRA", label="France")
KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
UGANDA = ResolvedEntity(role="destination", ref="country/UGA", label="Uganda")
UNITED_STATES = ResolvedEntity(role="origin", ref="country/USA", label="United States")

PROVIDER = "financingProvider"
RECIPIENT = "financingRecipient"
_SLOT_COLUMNS = (
    "stat_var_crsCooperationModality",
    "stat_var_crsPurpose",
    "stat_var_crsTypeOfFinance",
    "stat_var_crsTypeOfFlow",
)
FRAME_ROLE_KEYS = (PROVIDER, RECIPIENT, *_SLOT_COLUMNS)
# The coverage dimensions apis reports for this frame: each declared role
# column with its flow axis, then `years` with none.
DIMENSIONS = [
    {"label": PROVIDER, "count": 2, "axis": "origin"},
    {"label": RECIPIENT, "count": 1, "axis": "destination"},
    {"label": "years", "count": 10},
]
# The same dimensions from an apis service that reports no axis.
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
    recipients: Sequence[ResolvedEntity] = (KENYA,),
    provider_column: str = PROVIDER,
    unit: str = "usd",
    last_year: int = 2023,
    last_years: Mapping[ResolvedEntity, int] | None = None,
    kinds: bool = False,
    types: bool = False,
) -> Frame:
    """A CRS family frame for each provider and recipient, yearly from 2000 through `last_year`,
    or through a provider's own year in `last_years`, else its recipient's.

    Two declared role columns and the four `stat_var_*` slot columns, whose
    purpose slot tells the family's members apart. With `kinds`, each role's kind
    column too, every party of kind `party`, as `get_data(place_types=true)` serves it.
    """
    kind_columns = [f"{provider_column}_party_kind", f"{RECIPIENT}_party_kind"] if kinds else []
    type_columns = [f"{provider_column}_type", f"{RECIPIENT}_type"] if types else []
    return Frame.build(
        role_keys=[provider_column, RECIPIENT, *kind_columns, *type_columns, *_SLOT_COLUMNS],
        rows=[
            {
                "stat_var": f"dcid:{stat_var}",
                provider_column: provider.ref,
                RECIPIENT: recipient.ref,
                **dict.fromkeys(kind_columns, "party"),
                **dict.fromkeys(type_columns, "Country"),
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
            for recipient in recipients
            for year in range(
                2000,
                (last_years or {}).get(
                    provider,
                    (last_years or {}).get(recipient, _LAST_YEAR.get(stat_var, last_year)),
                )
                + 1,
            )
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
    `dimensions` are what apis reports. `role_keys` overrides the
    reported keys. `frame_read=False` leaves both out, as `get_data` does
    when it could not read the frame. `statistics` replaces the per-series
    statistics, one per provider and recipient by default.
    """

    async def _retrieved(
        source: SelectedSource,
        *,
        frame: Frame | None = None,
        providers: Sequence[ResolvedEntity] | None = None,
        recipients: Sequence[ResolvedEntity] = (KENYA,),
        role_keys: Sequence[str] | None = None,
        dimensions: list[dict[str, Any]] | None = None,
        frame_read: bool = True,
        statistics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if providers is None:
            providers = [entity for entity in source.entities if entity.role == "origin"]
        held = _family_frame(providers=providers, recipients=recipients) if frame is None else frame
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
        await hold_root(
            frame_store,
            ref,
            frame=held,
            trace=MaterialisationTrace(
                upstream_trace=trace, spec_id=ref.spec_id, content_version=ref.content_version
            ),
        )
        statistics = statistics or [
            {
                "stat_var": f"dcid:{_pinned(source)}",
                "entities": [
                    {"role": PROVIDER, "ref": provider.ref},
                    {"role": RECIPIENT, "ref": recipient.ref},
                ],
                "facet": {"unit": "usd"},
                "observations": 24,
                "first": {"value": 100.0, "period": "2000"},
                "latest": {"value": 123.0, "period": "2023"},
                "minimum": {"value": 100.0, "period": "2000"},
                "maximum": {"value": 123.0, "period": "2023"},
            }
            for provider in providers
            for recipient in recipients
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
    """Return a factory compiling and approving a plan over CRS family sources.

    With `anchors_read`, the newest years `executor.run.read_anchors` read over
    the compiled workflow, the plan and translation are restated from them as
    translation restates them (`translator.tool.restated_to_the_data`) and
    compiled again before approval.
    """

    def _plan(
        requirements: Sequence[EvidenceRequirement],
        operations: Sequence[Operation],
        *,
        terminal: str | Sequence[str],
        discovered: dict[str, SelectedSource] | None = None,
        decisions: Sequence[Decision] = (),
        anchors_read: Sequence[AnchorRead] = (),
    ) -> Plan:
        draft = Plan(
            question="How much bilateral ODA do Germany and France give Kenya?",
            decisions=[decision(), *decisions],
            evidence_requirements=list(requirements),
            first_years=[
                FirstYear(requirement_id=r.id, year=r.period.start)
                for r in requirements
                if isinstance(r, DatasetRequirement) and r.period_end_open
            ],
            status=PlanStatus.DRAFT,
        )
        terminals = [terminal] if isinstance(terminal, str) else list(terminal)
        translation = Translation(operations=list(operations), terminal=terminals)

        def compiled_from(plan: Plan, translation: Translation) -> CompileResult:
            result = compile_plan(
                plan=plan,
                translation=translation,
                schemas={schema.name: schema for schema in real_schemas},
                discovered=discovered or {"oda": _source()},
            )
            assert result.workflow is not None, result.issues
            return result

        compiled = compiled_from(draft, translation)
        if anchors_read:
            assert compiled.workflow is not None
            restated = restated_to_the_data(draft, translation, compiled.workflow, anchors_read)
            assert not isinstance(restated, CompileResult), restated.issues
            draft, translation = restated
            compiled = compiled_from(draft, translation)
        assert compiled.workflow is not None
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
            at=datetime.now(UTC)
        )

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


def _strict_destination_country_filter_plan(
    plan_for: PlanFactory,
    source: SelectedSource,
    *,
    rolling: bool = False,
) -> Plan:
    """Keep country recipients on an open destination axis.

    The compiled filter names ``destination_type``. The retrieved CRS frame
    declares the same enrichment as ``financingRecipient_type``.
    """
    predicates = (
        PlaceLevel(level=PlaceType.COUNTRY, role="country", strict=True),
        *(
            (PeriodThreshold(op="eq", year=2023, rolling=RelativePeriod(offset=0)),)
            if rolling
            else ()
        ),
    )
    countries = FilterRequirement(
        id="countries", decision_ids=["measure"], input_id="oda", predicates=predicates
    )
    requirement = DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[Entity(name="Germany", role="donor")],
        across=("country",),
        period=Period(start=2000, end=2023),
        period_end_open=rolling,
    )
    return plan_for(
        [requirement, countries],
        [
            _RETRIEVE,
            FilterOperation(
                id="countries_op",
                requirement_id="countries",
                input="crs",
                predicates=predicates,
            ),
        ],
        terminal="countries_op",
        discovered={"oda": source},
    )


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


def _aggregate_plan(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    *,
    requirement: DatasetRequirement | None = None,
) -> Plan:
    del decision
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )
    return plan_for(
        [requirement or _requirement(), totals],
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


def _rolling_growth_plan(plan_for: PlanFactory, *, newest: int = 2023) -> Plan:
    """Growth over the last five years, planned against `newest`."""
    window: dict[str, Any] = {
        "window_start": newest - 5,
        "window_end": newest,
        "rolling": RollingWindow(
            window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
        ),
    }
    return plan_for(
        [
            _requirement().model_copy(update={"period_end_open": True}),
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
        minuend=RequirementSide(input_id="oda", entity="France"),
        subtrahend=RequirementSide(input_id="oda", entity="Germany"),
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
    # role column, so it runs whether or not apis reports axes.
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


async def test_a_strict_country_filter_binds_destination_type_to_the_frames_enrichment(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    source = _source(
        entities=(GERMANY,),
        open_entity_axes=(OpenEntityAxis(column="destination", member_count=193),),
    )
    plan = _strict_destination_country_filter_plan(plan_for, source)
    assert plan.workflow is not None
    assert plan.workflow.steps[-1].args == {
        "predicates": [
            {
                "kind": "place_level",
                "column": "destination_type",
                "level": "Country",
                "strict": True,
            }
        ]
    }
    destination_coverage = [
        *DIMENSIONS[:1],
        {"label": RECIPIENT, "count": 193, "axis": "destination"},
        *DIMENSIONS[2:],
    ]
    surface = surface_for(
        get_data=[
            ok(
                await retrieved(
                    source,
                    frame=_family_frame(types=True, recipients=(KENYA, UGANDA)),
                    providers=(GERMANY,),
                    recipients=(KENYA, UGANDA),
                    dimensions=destination_coverage,
                )
            )
        ]
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "filter") == {
        "predicates": [
            {
                "kind": "place_level",
                "column": f"{RECIPIENT}_type",
                "level": "Country",
                "strict": True,
            }
        ]
    }


async def test_a_rolling_country_filter_rebinds_destination_type_in_its_population(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    source = _source(
        entities=(GERMANY,),
        open_entity_axes=(OpenEntityAxis(column="destination", member_count=193),),
    )
    plan = _strict_destination_country_filter_plan(plan_for, source, rolling=True)
    assert plan.workflow is not None
    (filter_step,) = plan.workflow.steps[-1:]
    assert filter_step.args["rolling"] == {
        "population": [
            {
                "spec_id": SPEC_CRS,
                "bindings": {"origin": [GERMANY.ref]},
                "levels": {"destination_type": "Country"},
                "strict_levels": ["destination_type"],
            }
        ]
    }
    destination_coverage = [
        *DIMENSIONS[:1],
        {"label": RECIPIENT, "count": 193, "axis": "destination"},
        *DIMENSIONS[2:],
    ]
    surface = surface_for(
        get_data=[
            ok(
                await retrieved(
                    source,
                    frame=_family_frame(types=True, recipients=(KENYA, UGANDA)),
                    providers=(GERMANY,),
                    recipients=(KENYA, UGANDA),
                    dimensions=destination_coverage,
                )
            )
        ]
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert [name for name, _ in surface.calls] == ["get_data", "resolve_anchor", "filter"]
    population = [
        {
            "spec_id": SPEC_CRS,
            "bindings": {PROVIDER: [GERMANY.ref]},
            "levels": {f"{RECIPIENT}_type": "Country"},
            "strict_levels": [f"{RECIPIENT}_type"],
        }
    ]
    assert _analysis_call(surface, "resolve_anchor") == {
        "populations": population,
        "roots": [{"spec_id": SPEC_CRS, "content_version": CONTENT_VERSION}],
    }
    sent = _analysis_call(surface, "filter")
    assert sent["predicates"][0] == {
        "kind": "place_level",
        "column": f"{RECIPIENT}_type",
        "level": "Country",
        "strict": True,
    }
    period_predicate = dict(sent["predicates"][1])
    assert period_predicate.pop("rolling") == {"anchor": "latest", "offset": 0, "rule": "half1"}
    assert period_predicate == {
        "kind": "ordered_threshold",
        "column": "time_period",
        "op": "eq",
        "value": "2023",
    }
    assert sent["rolling"]["population"] == population
    assert sent["anchor_key"] == outcome.anchors[0].anchor_key


async def test_a_strict_country_filter_without_the_destination_enrichment_refuses_before_call(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    source = _source(
        entities=(GERMANY,),
        open_entity_axes=(OpenEntityAxis(column="destination", member_count=193),),
    )
    destination_coverage = [
        *DIMENSIONS[:1],
        {"label": RECIPIENT, "count": 193, "axis": "destination"},
        *DIMENSIONS[2:],
    ]
    surface = surface_for(
        get_data=[
            ok(
                await retrieved(
                    source,
                    providers=(GERMANY,),
                    recipients=(KENYA, UGANDA),
                    dimensions=destination_coverage,
                )
            )
        ]
    )

    outcome = await _run(_strict_destination_country_filter_plan(plan_for, source), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.COLUMN_UNBOUND
    assert outcome.step_id == "countries_op"
    assert "destination_type" in outcome.detail
    assert [name for name, _ in surface.calls] == ["get_data"]


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
    # A rank across donors reads their kinds, which `get_data(without_party_totals=True)` serves.
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), frame=_family_frame(kinds=True)))]
    )

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
        across=("donor",),
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
    surface = surface_for(
        get_data=[
            ok(
                await retrieved(
                    source, providers=(GERMANY, FRANCE), frame=_family_frame(kinds=True)
                )
            )
        ]
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "rank") == {
        "rank_by": [PROVIDER],
        "ascending": False,
        "aggregation": "sum",
    }


def _sector_frame(*, spec_id: str, totals: Mapping[tuple[ResolvedEntity, str], float]) -> Frame:
    """A frame as apis rolls a family up to sector groups: `stat_var` is the spec's own
    ref on every row, the role columns are kept with each party's kind, and
    `stat_var_crsPurpose` names each row's group. A group with no records for a donor has
    no row, never a zero."""
    return Frame.build(
        role_keys=[
            PROVIDER,
            RECIPIENT,
            f"{PROVIDER}_party_kind",
            f"{RECIPIENT}_party_kind",
            "stat_var_crsPurpose",
        ],
        rows=[
            {
                "stat_var": f"dcid:{spec_id}",
                PROVIDER: provider.ref,
                RECIPIENT: KENYA.ref,
                f"{PROVIDER}_party_kind": "party",
                f"{RECIPIENT}_party_kind": "party",
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
        across=("donor",),
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
    ranked = (await result_table(outcome.results["top_op"])).to_pylist()
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


async def test_a_rolling_growth_is_resolved_and_run_over_the_frames_own_columns(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # Planned against 2021, over a frame whose newest year is 2022. The plan's
    # population names `origin` and `destination`; the frame's columns are the
    # declared properties, and analysis-mcp checks each rolling call's
    # population against the one the newest period was resolved over.
    plan = _rolling_growth_plan(plan_for, newest=2021)
    assert plan.workflow is not None
    assert plan.workflow.steps[-1].args["rolling"]["population"] == [
        {
            "spec_id": SPEC_CRS,
            "bindings": {
                "destination": [KENYA.ref],
                "origin": [GERMANY.ref, FRANCE.ref],
            },
        }
    ]
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), frame=_family_frame(last_year=2022)))]
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert [name for name, _ in surface.calls] == ["get_data", "resolve_anchor", "growth"]
    population = [
        {
            "spec_id": SPEC_CRS,
            "bindings": {PROVIDER: [GERMANY.ref, FRANCE.ref], RECIPIENT: [KENYA.ref]},
        }
    ]
    assert _analysis_call(surface, "resolve_anchor") == {
        "populations": population,
        "roots": [{"spec_id": SPEC_CRS, "content_version": CONTENT_VERSION}],
    }
    (anchor,) = outcome.anchors
    assert (anchor.period, anchor.left_out) == ("2022", ())
    sent = _analysis_call(surface, "growth")
    assert sent == {
        "window_start": "2017",
        "window_end": "2022",
        "entity_by": [PROVIDER, RECIPIENT],
        "rolling": {**sent["rolling"], "population": population},
        "anchor_key": anchor.anchor_key,
    }


async def test_a_newest_year_before_the_plans_stops_the_run_before_the_growth(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # Planned against 2023; analysis-mcp resolves 2022 over a frame ending there.
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), frame=_family_frame(last_year=2022)))]
    )

    outcome = await _run(_rolling_growth_plan(plan_for), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.ROLLING_ANCHOR_BEHIND_PLAN
    assert outcome.step_id == "growth_op"
    assert [name for name, _ in surface.calls] == ["get_data", "resolve_anchor"]
    assert [anchor.period for anchor in outcome.anchors] == ["2022"]


async def test_a_rolling_population_the_frame_cannot_spell_refuses_before_it_is_resolved(
    plan_for: PlanFactory, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> None:
    # With no axis reported, nothing says which frame column carries `origin`.
    surface = surface_for(
        get_data=[ok(await retrieved(_source(), dimensions=DIMENSIONS_WITHOUT_AXES))]
    )

    outcome = await _run(_rolling_growth_plan(plan_for), surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.COLUMN_UNBOUND
    assert outcome.step_id == "growth_op"
    assert [name for name, _ in surface.calls] == ["get_data"]


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


async def test_a_share_of_two_one_figure_totals_is_one_row(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    frame_store: InMemoryMaterialisationStore,
) -> None:
    # "What share of the ODA Kenya received came from Germany and France
    # together?": each side totals every row it reads, donors and years
    # alike, and the share divides one figure by the other.
    named = _source()
    every_donor = _source(
        spec_id=SPEC_CRS_HEALTH,
        entities=(KENYA,),
        open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
    )
    every_donor_requirement = DatasetRequirement(
        id="oda_health",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2000, end=2023),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="named_total"),
        denominator=RequirementSide(input_id="every_total"),
    )
    totals = [
        AggregateRequirement(
            id=total, decision_ids=["measure"], input_id=read, reduction=Reduction.SUM, group_by=()
        )
        for total, read in (("named_total", "oda"), ("every_total", "oda_health"))
    ]
    plan = plan_for(
        [_requirement(), every_donor_requirement, *totals, share],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            *(
                AggregateOperation(
                    id=f"{total}_op",
                    requirement_id=total,
                    input=read,
                    reduction=Reduction.SUM,
                    group_by=(),
                )
                for total, read in (("named_total", "crs"), ("every_total", "crs_health"))
            ),
            ShareOfOperation(
                id="share_op",
                requirement_id="share",
                numerator=Side(input="named_total_op"),
                denominator=Side(input="every_total_op"),
            ),
        ],
        terminal="share_op",
        discovered={"oda": named, "oda_health": every_donor},
    )
    results = {
        SPEC_CRS: await retrieved(named, frame=_family_frame(kinds=True)),
        SPEC_CRS_HEALTH: await retrieved(
            every_donor,
            providers=(GERMANY, FRANCE, UNITED_STATES),
            frame=_family_frame(providers=(GERMANY, FRANCE, UNITED_STATES), kinds=True),
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    aggregates = [arguments for name, arguments in surface.calls if name == "aggregate"]
    assert [call["group_by"] for call in aggregates] == [[], []]
    assert _analysis_call(surface, "share_of") == {}
    assert outcome.steps[-1].role_keys == ()
    # Every donor gives alike each year, so two of the three hold two thirds.
    held = await result_table(outcome.results["share_op"])
    assert held.column("value").to_pylist() == [pytest.approx(200 / 3)]


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
    assert _analysis_call(surface, "join") == {
        "join_keys": [PROVIDER, RECIPIENT],
        "right_keys": {PROVIDER: PROVIDER, RECIPIENT: RECIPIENT},
    }


async def test_a_join_key_the_two_frames_spell_differently_is_sent_in_each_frames_spelling(
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

    assert isinstance(outcome, Accepted), outcome
    assert _analysis_call(surface, "join") == {
        "join_keys": [PROVIDER, RECIPIENT],
        "right_keys": {PROVIDER: "provider", RECIPIENT: RECIPIENT},
    }


@pytest.mark.parametrize("health_side", ["right", "left"])
async def test_a_join_that_matched_a_donor_in_no_year_says_which_sides_data_it_lacked(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    health_side: str,
) -> None:
    # Health ODA comes from Germany alone, so France's ODA to Kenya matches in
    # no year and its rows are left out, on whichever side ODA is joined.
    health = _requirement(id="oda_health", donors=("Germany",)).model_copy(
        update={"measure": "health ODA"}
    )
    sides = ("oda", "oda_health") if health_side == "right" else ("oda_health", "oda")
    operations = ("crs", "crs_health") if health_side == "right" else ("crs_health", "crs")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id=sides[0], right_id=sides[1]
    )
    health_source = _source(spec_id=SPEC_CRS_HEALTH, entities=(GERMANY, KENYA))
    plan = plan_for(
        [_requirement(), health, joined],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            JoinOperation(
                id="joined_op", requirement_id="joined", left=operations[0], right=operations[1]
            ),
        ],
        terminal="joined_op",
        discovered={"oda": _source(), "oda_health": health_source},
    )
    results = {
        SPEC_CRS: await retrieved(_source()),
        SPEC_CRS_HEALTH: await retrieved(
            health_source, frame=_family_frame(stat_vars=("OdaHealth",), providers=(GERMANY,))
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    join = outcome.steps[-1]
    assert (join.dropped_key_count, join.right_dropped_key_count) == (
        (1, 0) if health_side == "right" else (0, 1)
    )
    # The two sides join on donor and on recipient, so a key is one pair.
    assert answer_notes(None, outcome, plan=plan) == [
        "1 donor and recipient pair had no health ODA figure for the same year as its "
        "bilateral ODA and was left out."
    ]


_WHO_KENYA = ResolvedEntity(role="observationAbout", ref="country/KEN", label="Kenya")
_WHO_UGANDA = ResolvedEntity(role="observationAbout", ref="country/UGA", label="Uganda")


async def test_a_place_series_joins_the_aid_its_place_receives_by_value(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    frame_store: InMemoryMaterialisationStore,
) -> None:
    # A place series on `observationAbout` beside the aid Kenya receives,
    # summed over its donors on `financingRecipient`, matched by value: Kenya
    # matches Kenya each year, so no key is dropped.
    place = _WHO_KENYA
    spending = DatasetRequirement(
        id="spending",
        decision_ids=["measure"],
        measure="domestic health spending",
        entities=[Entity(name=place.label)],
        period=Period(start=2000, end=2023),
    )
    oda = _requirement(donors=())
    total = AggregateRequirement(
        id="oda_total",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="spending", right_id="oda_total"
    )
    spending_source = _source(spec_id=SPEC_CRS_HEALTH, entities=(place,)).model_copy(
        update={"identity": _pinning("HealthSpending", SPEC_CRS_HEALTH)}
    )
    plan = plan_for(
        [spending, oda, total, joined],
        [
            RetrieveOperation(id="who", requirement_id="spending", spec_id=SPEC_CRS_HEALTH),
            _RETRIEVE,
            AggregateOperation(
                id="oda_total_op",
                requirement_id="oda_total",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            ),
            JoinOperation(
                id="joined_op", requirement_id="joined", left="who", right="oda_total_op"
            ),
        ],
        terminal="joined_op",
        discovered={"spending": spending_source, "oda": _source(entities=(KENYA,))},
    )
    place_frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:HealthSpending",
                "observationAbout": place.ref,
                "time_period": str(year),
                "value": 10.0 + year,
                "unit": "usd",
            }
            for year in range(2000, 2024)
        ],
    )
    results = {
        SPEC_CRS_HEALTH: await retrieved(
            spending_source,
            frame=place_frame,
            dimensions=[{"label": "observationAbout", "count": 1}, DIMENSIONS[-1]],
            statistics=[
                {
                    "stat_var": "dcid:HealthSpending",
                    "entities": [{"role": "observationAbout", "ref": place.ref}],
                    "facet": {"unit": "usd"},
                    "observations": 24,
                    "first": {"value": 2010.0, "period": "2000"},
                    "latest": {"value": 2033.0, "period": "2023"},
                    "minimum": {"value": 2010.0, "period": "2000"},
                    "maximum": {"value": 2033.0, "period": "2023"},
                }
            ],
        ),
        SPEC_CRS: await retrieved(
            _source(entities=(KENYA,)), providers=(GERMANY, FRANCE), frame=_family_frame(kinds=True)
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert _analysis_call(surface, "join") == {
        "join_keys": ["observationAbout"],
        "right_keys": {"observationAbout": RECIPIENT},
    }
    records = outcome.steps if isinstance(outcome, Accepted) else outcome.completed
    (join,) = [record for record in records if record.step_id == "joined_op"]
    dropped = [line for line in join.digest.splitlines() if "unmatched_key" in line]
    assert isinstance(outcome, Accepted), outcome
    held = await result_table(outcome.results["joined_op"])
    assert held.num_rows == 48
    assert dropped == []


@pytest.mark.parametrize(
    ("kenya_spending_from", "aid_recipients", "kenya_aid_to", "notes"),
    [
        # Health spending for Kenya and Uganda beside the aid Kenya alone
        # receives: Uganda matches in no year.
        pytest.param(
            2000,
            (KENYA,),
            2023,
            [
                "1 country had no bilateral ODA figure for the same year as its domestic "
                "health spending and was left out."
            ],
            id="one-sided",
        ),
        # Kenya's aid ends in 2009 and its spending starts in 2010, so Kenya
        # is on both sides in years that never overlap. Uganda matches.
        pytest.param(
            2010,
            (KENYA, UGANDA),
            2009,
            [
                "1 country had no bilateral ODA figure for the same year as its domestic "
                "health spending and was left out.",
                "1 country had no domestic health spending figure for the same year as its "
                "bilateral ODA and was left out.",
            ],
            id="disjoint-years",
        ),
    ],
)
async def test_a_place_series_joined_to_aid_names_the_place_the_aid_missed_as_a_country(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    kenya_spending_from: int,
    aid_recipients: tuple[ResolvedEntity, ...],
    kenya_aid_to: int,
    notes: list[str],
) -> None:
    # The one key pair is the place.
    places = (_WHO_KENYA, _WHO_UGANDA)
    spending = DatasetRequirement(
        id="spending",
        decision_ids=["measure"],
        measure="domestic health spending",
        entities=[Entity(name=place.label) for place in places],
        period=Period(start=2000, end=2023),
    )
    oda = _requirement(donors=(), recipients=[entity.label for entity in aid_recipients])
    total = AggregateRequirement(
        id="oda_total",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="spending", right_id="oda_total"
    )
    spending_source = _source(spec_id=SPEC_CRS_HEALTH, entities=places).model_copy(
        update={"identity": _pinning("HealthSpending", SPEC_CRS_HEALTH)}
    )
    plan = plan_for(
        [spending, oda, total, joined],
        [
            RetrieveOperation(id="who", requirement_id="spending", spec_id=SPEC_CRS_HEALTH),
            _RETRIEVE,
            AggregateOperation(
                id="oda_total_op",
                requirement_id="oda_total",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            ),
            JoinOperation(
                id="joined_op", requirement_id="joined", left="who", right="oda_total_op"
            ),
        ],
        terminal="joined_op",
        discovered={"spending": spending_source, "oda": _source(entities=aid_recipients)},
    )
    place_frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:HealthSpending",
                "observationAbout": place.ref,
                "time_period": str(year),
                "value": 10.0 + year,
                "unit": "usd",
            }
            for place in places
            for year in range(kenya_spending_from if place is _WHO_KENYA else 2000, 2024)
        ],
    )
    results = {
        SPEC_CRS_HEALTH: await retrieved(
            spending_source,
            frame=place_frame,
            dimensions=[{"label": "observationAbout", "count": len(places)}, DIMENSIONS[-1]],
            statistics=[
                {
                    "stat_var": "dcid:HealthSpending",
                    "entities": [{"role": "observationAbout", "ref": place.ref}],
                    "facet": {"unit": "usd"},
                    "observations": 24,
                    "first": {"value": 2010.0, "period": "2000"},
                    "latest": {"value": 2033.0, "period": "2023"},
                    "minimum": {"value": 2010.0, "period": "2000"},
                    "maximum": {"value": 2033.0, "period": "2023"},
                }
                for place in places
            ],
        ),
        SPEC_CRS: await retrieved(
            _source(entities=aid_recipients),
            providers=(GERMANY, FRANCE),
            recipients=aid_recipients,
            frame=_family_frame(
                kinds=True, recipients=aid_recipients, last_years={KENYA: kenya_aid_to}
            ),
        ),
    }
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert answer_notes(None, outcome, plan=plan) == notes


async def test_a_role_the_frame_has_no_column_for_refuses_before_the_call(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
) -> None:
    # An apis service that reports no axis leaves `origin` and `destination` with
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


# --- Party totals: a flow retrieval is served without the parties totalling others -----

# DAC2A's donors to Kenya with the kind apis serves for each: countries, an
# organisation and a residual add up; the DAC total and the G7 hold the countries' figures
# again, and `get_data(without_party_totals=True)` leaves them out.
_DONORS_TO_KENYA: dict[str, tuple[str, float]] = {
    "country/DEU": ("party", 10.0),
    "country/USA": ("party", 20.0),
    "EuropeanUnionInstitutions": ("party", 5.0),
    "DACOtherMultilaterals": ("residual", 3.0),
    "DACCountriesTotal": ("total", 30.0),
    "GroupOfSeven": ("total", 30.0),
}
_WITHOUT_TOTALS_CONTENT_VERSION = "cv_" + "7" * 32


def _donors_frame(*, without_totals: bool) -> Frame:
    """Every donor's yearly aid to Kenya since 2000, each with its kind, less the donors
    that total others when `without_totals`."""
    provider_kind = f"{PROVIDER}_party_kind"
    return Frame.build(
        role_keys=[PROVIDER, RECIPIENT, provider_kind],
        rows=[
            {
                "stat_var": "dcid:Oda",
                PROVIDER: donor,
                RECIPIENT: KENYA.ref,
                provider_kind: kind,
                "time_period": str(year),
                "value": value,
                "unit": "usd",
            }
            for donor, (kind, value) in _DONORS_TO_KENYA.items()
            if not (without_totals and kind == "total")
            for year in range(2000, 2024)
        ],
    )


class _WithoutTotalsOnRequest(FrameSurface):
    """Answers `get_data` with the frame without donor totals when the call asks for it."""

    def __init__(self, *, answers: Mapping[bool, dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.calls.append((tool.name, dict(arguments)))
        return ok(self._answers[bool(arguments.get("without_party_totals"))])


async def test_a_yearly_total_over_every_donor_sums_the_frame_served_without_donor_totals(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    frame_store: InMemoryMaterialisationStore,
    _stats_by_manifest: dict[tuple[str, str], list[dict[str, Any]]],
    real_schemas: list[ToolSchema],
) -> None:
    # "Total aid to Kenya each year": the source binds Kenya and leaves the donors open.
    # The compiled retrieval asks for the frame without donor totals, and each year's
    # total sums the countries, the organisation and the residual, 38, where every donor
    # sums to 98.
    source = _source(
        entities=(KENYA,), open_entity_axes=(OpenEntityAxis(column="origin", member_count=6),)
    )
    plan = plan_for(
        [
            DatasetRequirement(
                id="oda",
                decision_ids=["measure"],
                measure="bilateral ODA",
                entities=[Entity(name="Kenya", role="recipient")],
                across=("donor",),
                period=Period(start=2000, end=2023),
            ),
            AggregateRequirement(
                id="totals",
                decision_ids=["measure"],
                input_id="oda",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        [
            _RETRIEVE,
            AggregateOperation(
                id="totals_op",
                requirement_id="totals",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal="totals_op",
        discovered={"oda": source},
    )
    donors = tuple(ResolvedEntity(role="origin", ref=ref, label=ref) for ref in _DONORS_TO_KENYA)
    plain = await retrieved(source, frame=_donors_frame(without_totals=False), providers=donors)
    without_totals_frame = _donors_frame(without_totals=True)
    without_totals_ref = FrameRef(
        spec_id=source.spec_id, content_version=_WITHOUT_TOTALS_CONTENT_VERSION
    )
    plain_ref = FrameRef(spec_id=source.spec_id, content_version=CONTENT_VERSION)
    _held, trace = await frame_store.get(plain_ref)
    await hold_root(frame_store, without_totals_ref, frame=without_totals_frame, trace=trace)
    _stats_by_manifest[(source.spec_id, _WITHOUT_TOTALS_CONTENT_VERSION)] = _stats_by_manifest[
        (source.spec_id, CONTENT_VERSION)
    ]
    without_totals = plain | {
        "content_version": _WITHOUT_TOTALS_CONTENT_VERSION,
        "frame_ref": without_totals_ref.model_dump(),
        "role_keys": list(observed_shape(without_totals_frame)[0]),
    }
    engines = McpSurface(sources={"analysis-mcp": analysis_server.mcp})
    surface = _WithoutTotalsOnRequest(
        answers={False: plain, True: without_totals},
        schemas=real_schemas,
        responses={},
        engines=engines,
    )

    try:
        outcome = await _run(plan, surface)
    finally:
        await engines.aclose()

    assert isinstance(outcome, Accepted), outcome
    total = await result_table(outcome.results["totals_op"])
    assert set(total.column("value").to_pylist()) == {38.0}
    ((_, aggregated),) = [call for call in surface.calls if call[0] == "aggregate"]
    assert aggregated["frame_ref"]["content_version"] == _WITHOUT_TOTALS_CONTENT_VERSION


# --- A keyed pairing under a rolling period: the matched keys' newest year ----

_MATCHED_DONORS = (
    GERMANY,
    FRANCE,
    UNITED_STATES,
    ResolvedEntity(role="origin", ref="country/GBR", label="United Kingdom"),
)
_MATCHED_WINDOW = Period(start=2000, end=2024)


def _matched_source(spec_id: str) -> SelectedSource:
    return _source(spec_id=spec_id, entities=(*_MATCHED_DONORS, KENYA)).model_copy(
        update={"coverage_window": _MATCHED_WINDOW}
    )


def _matched_requirement(id: str) -> DatasetRequirement:
    return DatasetRequirement(
        id=id,
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[
            *(Entity(name=donor.label, role="donor") for donor in _MATCHED_DONORS),
            Entity(name="Kenya", role="recipient"),
        ],
        period=_MATCHED_WINDOW,
        period_end_open=True,
    )


_MATCHED_GROWTH: dict[str, Any] = {
    "change": GrowthChange.ABSOLUTE,
    "window_start": 2019,
    "window_end": 2024,
    "rolling": RollingWindow(
        window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
    ),
}


async def _restated_from_the_data(
    planned: Callable[..., Plan],
    *,
    results: dict[str, dict[str, Any]],
    surface_for: Callable[..., FrameSurface],
) -> tuple[AnchorRead, FrameSurface, Plan]:
    """The one group's newest year as translation reads it over the workflow
    `planned()` compiles (`read_anchors`), the surface it read on, and the plan
    `planned` compiles restated from it."""
    plan = planned()
    assert plan.workflow is not None
    reading = surface_for(get_data=_get_data_answers(plan, results))
    read = await read_anchors(
        plan.workflow,
        revision=plan_revision(plan),
        requirements=[r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)],
        surface=reading,
    )
    assert isinstance(read, tuple), read
    (group,) = read
    return group, reading, planned(anchors_read=read)


def _rolling_share_plan(plan_for: PlanFactory, **restated: Any) -> Plan:
    """Growth over the last five years of each donor's health share of its ODA,
    planned against 2024."""
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
            _matched_requirement("oda"),
            _matched_requirement("oda_health"),
            share,
            GrowthRequirement(
                id="growth", decision_ids=["measure"], input_id="share", **_MATCHED_GROWTH
            ),
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
            GrowthOperation(
                id="growth_op", requirement_id="growth", input="share_op", **_MATCHED_GROWTH
            ),
        ],
        terminal="growth_op",
        discovered={
            "oda": _matched_source(SPEC_CRS),
            "oda_health": _matched_source(SPEC_CRS_HEALTH),
        },
        **restated,
    )


async def test_a_rolling_share_of_disjoint_halves_answers_from_the_year_its_keys_share(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    frame_store: InMemoryMaterialisationStore,
) -> None:
    # Half the donors report 2024 on each side, and they are different halves,
    # so no donor's share holds 2024. Counted over the donors both sides hold,
    # the newest year is 2023, so translation restates the plan's 2019 to 2024
    # as 2018 to 2023 and the run of it answers. Counted per side, as at
    # cd1f7289, it was 2024, and the run stopped at the growth with
    # `growth_gap_year`, since no share has a row in 2024.
    first, second = _MATCHED_DONORS[:2], _MATCHED_DONORS[2:]
    results = {
        SPEC_CRS: await retrieved(
            _matched_source(SPEC_CRS),
            frame=_family_frame(
                providers=_MATCHED_DONORS,
                last_years={**dict.fromkeys(first, 2023), **dict.fromkeys(second, 2024)},
                kinds=True,
            ),
            providers=_MATCHED_DONORS,
        ),
        SPEC_CRS_HEALTH: await retrieved(
            _matched_source(SPEC_CRS_HEALTH),
            frame=_family_frame(
                stat_vars=("OdaHealth",),
                providers=_MATCHED_DONORS,
                last_years={**dict.fromkeys(first, 2024), **dict.fromkeys(second, 2023)},
                kinds=True,
            ),
            providers=_MATCHED_DONORS,
        ),
    }
    read, reading, plan = await _restated_from_the_data(
        lambda **restated: _rolling_share_plan(plan_for, **restated),
        results=results,
        surface_for=surface_for,
    )
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(read.result, AnchorResult)
    assert (read.planned, read.result.period) == ("2024", "2023")
    # Translation's read and the run each name the group's retrievals.
    assert read.result.roots == outcome.anchors[0].roots == ("crs", "crs_health")
    bindings = {PROVIDER: sorted(donor.ref for donor in _MATCHED_DONORS), RECIPIENT: [KENYA.ref]}
    assert _analysis_call(reading, "resolve_anchor")["populations"] == [
        {
            "pair": [
                [{"spec_id": SPEC_CRS_HEALTH, "bindings": bindings}],
                [{"spec_id": SPEC_CRS, "bindings": bindings}],
            ],
            "keys": {PROVIDER: PROVIDER},
        }
    ]
    assert isinstance(outcome, Accepted), outcome
    assert [anchor.period for anchor in outcome.anchors] == ["2023"]
    sent = _analysis_call(surface, "growth")
    assert (sent["window_start"], sent["window_end"]) == ("2018", "2023")
    held = await result_table(outcome.results["growth_op"])
    assert held.num_rows == len(_MATCHED_DONORS)


_RELATIVE_GROWTH = {**_MATCHED_GROWTH, "change": GrowthChange.RELATIVE}
_MATCHED_PLACES = (
    KENYA,
    UGANDA,
    ResolvedEntity(role="destination", ref="country/TZA", label="Tanzania"),
    ResolvedEntity(role="destination", ref="country/RWA", label="Rwanda"),
)


async def test_a_rolling_join_through_right_keys_answers_from_the_year_its_places_share(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    frame_store: InMemoryMaterialisationStore,
) -> None:
    # The growth of a place series on `observationAbout` joined to the growth of
    # the aid each place receives, summed over donors on `financingRecipient`.
    # Half the places report 2024 on each side, and they are different halves.
    # Counted over the places both sides hold, the newest year is 2023, so
    # translation restates 2019 to 2024 as 2018 to 2023 and the run answers.
    # Counted per side, as at cd1f7289, it was 2024, and the join matched no
    # place (`empty_answer`).
    first, second = _MATCHED_PLACES[:2], _MATCHED_PLACES[2:]
    spending = DatasetRequirement(
        id="spending",
        decision_ids=["measure"],
        measure="domestic health spending",
        entities=[Entity(name=place.label) for place in _MATCHED_PLACES],
        period=_MATCHED_WINDOW,
        period_end_open=True,
    )
    oda = DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[Entity(name=place.label, role="recipient") for place in _MATCHED_PLACES],
        period=_MATCHED_WINDOW,
        period_end_open=True,
    )
    total = AggregateRequirement(
        id="oda_total",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )
    spending_growth = GrowthRequirement(
        id="spending_growth", decision_ids=["measure"], input_id="spending", **_RELATIVE_GROWTH
    )
    oda_growth = GrowthRequirement(
        id="oda_growth", decision_ids=["measure"], input_id="oda_total", **_RELATIVE_GROWTH
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="spending_growth", right_id="oda_growth"
    )
    places = tuple(
        ResolvedEntity(role="observationAbout", ref=place.ref, label=place.label)
        for place in _MATCHED_PLACES
    )
    spending_source = _source(spec_id=SPEC_CRS_HEALTH, entities=places).model_copy(
        update={
            "identity": _pinning("HealthSpending", SPEC_CRS_HEALTH),
            "coverage_window": _MATCHED_WINDOW,
        }
    )
    oda_source = _source(entities=_MATCHED_PLACES).model_copy(
        update={"coverage_window": _MATCHED_WINDOW}
    )
    planned = partial(
        plan_for,
        [spending, oda, total, spending_growth, oda_growth, joined],
        [
            RetrieveOperation(id="who", requirement_id="spending", spec_id=SPEC_CRS_HEALTH),
            _RETRIEVE,
            AggregateOperation(
                id="oda_total_op",
                requirement_id="oda_total",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            ),
            GrowthOperation(
                id="spending_growth_op",
                requirement_id="spending_growth",
                input="who",
                **_RELATIVE_GROWTH,
            ),
            GrowthOperation(
                id="oda_growth_op",
                requirement_id="oda_growth",
                input="oda_total_op",
                **_RELATIVE_GROWTH,
            ),
            JoinOperation(
                id="joined_op",
                requirement_id="joined",
                left="spending_growth_op",
                right="oda_growth_op",
            ),
        ],
        terminal="joined_op",
        discovered={"spending": spending_source, "oda": oda_source},
    )
    last = {**dict.fromkeys(first, 2024), **dict.fromkeys(second, 2023)}
    place_frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:HealthSpending",
                "observationAbout": place.ref,
                "time_period": str(year),
                "value": 10.0 + year,
                "unit": "usd",
            }
            for place in _MATCHED_PLACES
            for year in range(2000, last[place] + 1)
        ],
    )
    results = {
        SPEC_CRS_HEALTH: await retrieved(
            spending_source,
            frame=place_frame,
            dimensions=[{"label": "observationAbout", "count": 4}, DIMENSIONS[-1]],
            statistics=[
                {
                    "stat_var": "dcid:HealthSpending",
                    "entities": [{"role": "observationAbout", "ref": place.ref}],
                    "facet": {"unit": "usd"},
                    "observations": 24,
                    "first": {"value": 2010.0, "period": "2000"},
                    "latest": {"value": 2033.0, "period": "2023"},
                    "minimum": {"value": 2010.0, "period": "2000"},
                    "maximum": {"value": 2033.0, "period": "2023"},
                }
                for place in _MATCHED_PLACES
            ],
        ),
        SPEC_CRS: await retrieved(
            oda_source,
            providers=(GERMANY, FRANCE),
            recipients=_MATCHED_PLACES,
            frame=_family_frame(
                recipients=_MATCHED_PLACES,
                last_years={**dict.fromkeys(first, 2023), **dict.fromkeys(second, 2024)},
                kinds=True,
            ),
        ),
    }
    read, reading, plan = await _restated_from_the_data(
        planned, results=results, surface_for=surface_for
    )
    surface = surface_for(get_data=_get_data_answers(plan, results))

    outcome = await _run(plan, surface)

    assert isinstance(read.result, AnchorResult)
    assert (read.planned, read.result.period) == ("2024", "2023")
    refs = sorted(place.ref for place in _MATCHED_PLACES)
    assert _analysis_call(reading, "resolve_anchor")["populations"] == [
        {
            "pair": [
                [{"spec_id": SPEC_CRS_HEALTH, "bindings": {"observationAbout": refs}}],
                [{"spec_id": SPEC_CRS, "bindings": {RECIPIENT: refs}}],
            ],
            "keys": {"observationAbout": RECIPIENT},
        }
    ]
    assert isinstance(outcome, Accepted), outcome
    assert [anchor.period for anchor in outcome.anchors] == ["2023"]
    windows = [
        (sent["window_start"], sent["window_end"])
        for name, sent in surface.calls
        if name == "growth"
    ]
    assert windows == [("2018", "2023")] * 2
    held = await result_table(outcome.results["joined_op"])
    # Each place's growth on each side, every place matched.
    assert held.num_rows == 2 * len(_MATCHED_PLACES)


# --- A donor whose series stopped: left out, and the answer names it ---------
#
# Germany's series stops before the requirement's period, so the retrieval
# holds France's alone.

_LEFT_OUT = "Germany has no data from 2000 to 2023 and was left out."


def _entity_filter_plan(*names: str) -> Callable[..., Plan]:
    """A plan keeping the rows of `names` from both donors' ODA to Kenya."""

    def _plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
        del decision
        kept = FilterRequirement(
            id="kept",
            decision_ids=["measure"],
            input_id="oda",
            predicates=(EntityIn(names=names),),
        )
        return plan_for(
            [_requirement(), kept],
            [
                _RETRIEVE,
                FilterOperation(
                    id="kept_op",
                    requirement_id="kept",
                    input="crs",
                    predicates=(EntityIn(names=names),),
                ),
            ],
            terminal="kept_op",
        )

    return _plan


def _provider_aggregate_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    """`_aggregate_plan` naming Germany a donor and France a provider: one side of the flow."""
    named = DatasetRequirement(
        id="oda",
        decision_ids=["measure"],
        measure="bilateral ODA",
        entities=[
            Entity(name="Germany", role="donor"),
            Entity(name="France", role="provider"),
            Entity(name="Kenya", role="recipient"),
        ],
        period=Period(start=2000, end=2023),
    )
    return _aggregate_plan(plan_for, decision, requirement=named)


def _germany_share_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    """Germany's share of both donors' total."""
    del decision
    pooled = AggregateRequirement(
        id="pooled",
        decision_ids=["measure"],
        input_id="oda",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="oda", entity="Germany"),
        denominator=RequirementSide(input_id="pooled"),
    )
    return plan_for(
        [_requirement(), pooled, share],
        [
            _RETRIEVE,
            AggregateOperation(
                id="pooled_op",
                requirement_id="pooled",
                input="crs",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            ShareOfOperation(
                id="share_op",
                requirement_id="share",
                numerator=Side(input="crs", entity="Germany"),
                denominator=Side(input="pooled_op"),
            ),
        ],
        terminal="share_op",
    )


def _germany_sum_plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
    """Germany's bilateral ODA plus its imputed ODA, each source naming both donors."""
    del decision
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id="oda", entity="Germany"),
        right=RequirementSide(input_id="oda_health", entity="Germany"),
    )
    return plan_for(
        [_requirement(), _requirement(id="oda_health"), total],
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
        discovered={"oda": _source(), "oda_health": _imputed_source()},
    )


def _imputed_source() -> SelectedSource:
    """A second source of both donors' ODA to Kenya: imputed multilateral ODA."""
    return _source(spec_id=SPEC_CRS_HEALTH).model_copy(
        update={
            "identity": _pinning("OdaImputed", SPEC_CRS_HEALTH),
            "shape_member_digest": member_digest(["OdaImputed"]),
        }
    )


def _shared_germany_plan(*, bilateral_numerator: bool) -> Callable[..., Plan]:
    """Each donor's bilateral ODA as a share of its health ODA, or the other
    way round, then Germany's share kept.

    The share rests on both retrievals and keeps the donors both sides
    hold, so Germany, which the bilateral retrieval left out, is in it on
    neither reading.
    """
    numerator, denominator = ("oda", "oda_health") if bilateral_numerator else ("oda_health", "oda")
    steps = {"oda": "crs", "oda_health": "crs_health"}

    def _plan(plan_for: PlanFactory, decision: Callable[..., Decision]) -> Plan:
        del decision
        return _share_then_germany(plan_for, numerator, denominator, steps)

    return _plan


def _share_then_germany(
    plan_for: PlanFactory, numerator: str, denominator: str, steps: Mapping[str, str]
) -> Plan:
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator),
        denominator=RequirementSide(input_id=denominator),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    germany = FilterRequirement(
        id="germany",
        decision_ids=["measure"],
        input_id="share",
        predicates=(EntityIn(names=("Germany",)),),
    )
    return plan_for(
        [_requirement(), _requirement(id="oda_health"), share, germany],
        [
            _RETRIEVE,
            _RETRIEVE_HEALTH,
            ShareOfOperation(
                id="share_op",
                requirement_id="share",
                numerator=Side(input=steps[numerator]),
                denominator=Side(input=steps[denominator]),
                axis="donor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
            FilterOperation(
                id="germany_op",
                requirement_id="germany",
                input="share_op",
                predicates=(EntityIn(names=("Germany",)),),
            ),
        ],
        terminal="germany_op",
        discovered={"oda": _source(), "oda_health": _source(spec_id=SPEC_CRS_HEALTH)},
    )


async def _stopped_bilateral(
    plan: Plan, retrieved: Retrieved, surface_for: Callable[..., FrameSurface]
) -> FrameSurface:
    """A surface whose bilateral retrieval holds France alone and any other
    retrieval both donors, each frame carrying its parties' kinds, which a
    share across donors pools by."""
    results = {}
    for spec_id, source in _sources_of(plan).items():
        providers = (FRANCE,) if spec_id == SPEC_CRS else (GERMANY, FRANCE)
        results[spec_id] = await retrieved(
            source,
            frame=_family_frame(stat_vars=(_pinned(source),), providers=providers, kinds=True),
            providers=providers,
        )
    return surface_for(get_data=_get_data_answers(plan, results))


def _left_out(outcome: Accepted) -> list[tuple[JsonValue, str]]:
    return [
        (gap.required, gap.severity.value)
        for record in outcome.evidence
        for gap in record.gaps
        if gap.code.value == "entity_missing"
    ]


@pytest.mark.parametrize(
    ("plan_of", "rows"),
    [
        pytest.param(_aggregate_plan, 24, id="aggregate"),
        pytest.param(_provider_aggregate_plan, 24, id="donor-and-provider"),
        pytest.param(_rank_plan, 1, id="rank"),
        pytest.param(_growth_plan, 1, id="growth"),
        pytest.param(_entity_filter_plan("Germany", "France"), 24, id="filter"),
    ],
)
async def test_a_donor_with_no_series_in_the_period_is_left_out_and_named(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    plan_of: Callable[..., Plan],
    rows: int | None,
) -> None:
    plan = plan_of(plan_for, decision)
    surface = await _stopped_bilateral(plan, retrieved, surface_for)

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    if rows is not None:
        assert outcome.steps[-1].row_count == rows
    assert _left_out(outcome) == [("Germany", "warning")]
    assert answer_notes(surface, outcome, plan=plan) == [_LEFT_OUT]


@pytest.mark.parametrize(
    ("plan_of", "step_id"),
    [
        pytest.param(_compare_plan, "compared_op", id="compare"),
        pytest.param(_germany_share_plan, "share_op", id="share"),
        pytest.param(_entity_filter_plan("Germany"), "kept_op", id="filter"),
        pytest.param(_germany_sum_plan, "total_op", id="sum"),
        pytest.param(
            _shared_germany_plan(bilateral_numerator=True), "germany_op", id="share-then-filter"
        ),
        pytest.param(
            _shared_germany_plan(bilateral_numerator=False),
            "germany_op",
            id="share-over-it-then-filter",
        ),
    ],
)
async def test_a_step_reading_the_left_out_donor_alone_is_refused_before_any_verb_runs(
    plan_for: PlanFactory,
    decision: Callable[..., Decision],
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    plan_of: Callable[..., Plan],
    step_id: str,
) -> None:
    plan = plan_of(plan_for, decision)
    surface = await _stopped_bilateral(plan, retrieved, surface_for)

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert (outcome.step_id, outcome.requirement_id) == (step_id, "oda")
    assert [
        (issue.code.value, issue.required, issue.severity.value) for issue in outcome.issues
    ] == [("entity_missing", "Germany", "error")]
    assert plan.workflow is not None
    assert [name for name, _ in surface.calls] == ["get_data"] * len(
        plan.workflow.evidence_bindings
    )


TANZANIA = ResolvedEntity(role="destination", ref="country/TZA", label="Tanzania")
# Germany's ODA to each recipient ends in its own year.
_LATEST_ODA = {KENYA: 2021, UGANDA: 2024, TANZANIA: 2022}


@pytest.mark.parametrize("with_kenya", [True, False], ids=["beside-kenya", "alone"])
async def test_an_answer_states_the_short_series_of_its_own_newest_year(
    plan_for: PlanFactory,
    retrieved: Retrieved,
    surface_for: Callable[..., FrameSurface],
    with_kenya: bool,
) -> None:
    # One retrieval of Germany's ODA to Kenya, Uganda and Tanzania, open to
    # the newest year, under the latest year for Uganda and Tanzania and,
    # beside it, the latest year for Kenya. Kenya's newest year is 2021, and
    # Uganda and Tanzania's 2024, which Tanzania's series falls short of.
    # Each answer states the series of its own newest year: the Kenya branch
    # neither ends Uganda and Tanzania's window at 2021 nor puts Kenya's
    # series in it.
    recipients = tuple(_LATEST_ODA)
    source = _source(entities=(GERMANY, *recipients)).model_copy(
        update={"coverage_window": Period(start=2000, end=2024)}
    )
    oda = _requirement(donors=("Germany",), recipients=[r.label for r in recipients]).model_copy(
        update={"period": Period(start=2000, end=2024), "period_end_open": True}
    )
    branches = [("kenya", ("Kenya",)), ("others", ("Uganda", "Tanzania"))][not with_kenya :]
    latest = {
        id: (
            EntityIn(names=names, role="recipient"),
            PeriodThreshold(op="eq", year=2024, rolling=RelativePeriod(offset=0)),
        )
        for id, names in branches
    }

    def planned(**restated: Any) -> Plan:
        return plan_for(
            [
                oda,
                *(
                    FilterRequirement(
                        id=id, decision_ids=["measure"], input_id="oda", predicates=predicates
                    )
                    for id, predicates in latest.items()
                ),
            ],
            [
                _RETRIEVE,
                *(
                    FilterOperation(
                        id=f"{id}_op", requirement_id=id, input="crs", predicates=predicates
                    )
                    for id, predicates in latest.items()
                ),
            ],
            terminal=[f"{id}_op" for id in latest],
            discovered={"oda": source},
            **restated,
        )

    results = {
        SPEC_CRS: await retrieved(
            source,
            frame=_family_frame(
                providers=(GERMANY,), recipients=recipients, last_years=_LATEST_ODA
            ),
            providers=(GERMANY,),
            recipients=recipients,
            statistics=[
                {
                    "stat_var": "dcid:Oda",
                    "entities": [
                        {"role": PROVIDER, "ref": GERMANY.ref},
                        {"role": RECIPIENT, "ref": recipient.ref},
                    ],
                    "facet": {"unit": "usd"},
                    "observations": end - 2000 + 1,
                    "first": {"value": 100.0, "period": "2000"},
                    "latest": {"value": 123.0, "period": str(end)},
                    "minimum": {"value": 100.0, "period": "2000"},
                    "maximum": {"value": 123.0, "period": str(end)},
                }
                for recipient, end in _LATEST_ODA.items()
            ],
            dimensions=[
                {"label": PROVIDER, "count": 1, "axis": "origin"},
                {"label": RECIPIENT, "count": len(recipients), "axis": "destination"},
                {"label": "years", "count": 25},
            ],
        )
    }
    draft = planned()
    assert draft.workflow is not None
    read = await read_anchors(
        draft.workflow,
        revision=plan_revision(draft),
        requirements=[oda],
        surface=surface_for(get_data=_get_data_answers(draft, results)),
    )
    assert isinstance(read, tuple), read
    plan = planned(anchors_read=read)

    outcome = await _run(plan, surface_for(get_data=_get_data_answers(plan, results)))

    assert isinstance(outcome, Accepted), outcome
    assert [anchor.period for anchor in outcome.anchors] == ["2021", "2024"][not with_kenya :]
    assert answer_notes(None, outcome, plan=plan, terminals=["others_op"]) == [
        "Germany and Tanzania's series ends in 2022."
    ]
    if with_kenya:
        assert answer_notes(None, outcome, plan=plan, terminals=["kenya_op"]) == []
