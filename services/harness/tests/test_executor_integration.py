"""The executor against the real services: a plan is compiled, approved and run.

`get_data`, called through the mounted `data_mcp` app, registers a frame in the
shared store; `aggregate`, through the mounted `analysis_mcp` app, resolves that
frame; the executor binds the one to the other, checks the retrieval against
its requirement, and accepts the run. The upstream mint is a canned response
and the artifact fetch is patched at the one seam nothing threads a transport
into, so no socket opens (see `test_materialisation_seam.py`).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation import FrameRef
from common.materialisation import get_store as get_frame_store
from common.recipe import build_recipe
from common.row_budget import RowBudget
from compiler.compile import compile_plan
from compiler.sources import member_digest
from compiler.surface import (
    McpSurface,
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from data_mcp import client_holder, frame_materialisation
from data_mcp.deadline import CallDeadline
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.frame_materialisation import DEFAULT_MINT_OPTIONS, frame_ref_for
from data_mcp.manifest import SeriesStatisticsModel
from data_mcp.series_statistics import series_statistics_table
from executor.completion import ResultPublisher
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from harness.graph.answer import answer_frames, link_evidence_to_terminal_frames
from harness.graph.state import ORCHESTRATOR_FRAMES_KEY, frame_id_for, register_frames
from harness.mcp_client import McpExecutor
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.orchestrator_tools import build_orchestrator_tools
from harness.provenance import ProvenanceStore, data_artifact
from harness.recipe_replay import Pending, Replayed, replay_recipe
from harness.recipe_replay import Refused as ReplayRefused
from harness.server import create_app
from harness.settings import ANALYSIS_MCP_MOUNT_PATH, DATA_MCP_MOUNT_PATH, Settings
from harness.studio_translate import _RecordingSurface
from harness.tool_contracts import HarnessCode
from planner.operations import (
    AggregateOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    FilterRequirement,
    GrowthRequirement,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
)
from planner.records import (
    DenominatorScope,
    EntityIn,
    GrowthChange,
    Period,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    RelativePeriod,
    RollingWindow,
)
from planner.workflow import OpenEntityAxis, ResolvedEntity, SelectedSource, ToolId
from presentation_mcp.platform_contract import RESOLVED_PAYLOAD_META_KEY
from translator.mechanical import build_translation

from .canonical_versions import serve_canonical_frames, version_of
from .replay_support import FakeApis, data_mcp_client
from .resolved_revision import resolved_revision

SPEC_ID = "spec_" + "5" * 32
YEARS = [str(year) for year in range(2015, 2025)]

IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "DebtServicePaid", "kind": "stat_var"},
    "entities": {"recipient": ["country/KEN"]},
    "facets": {"unit": {"ref": "USD", "state": "value"}},
    "requested_window": {"kind": "all"},
    "slots": [],
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DATA_MCP_URL", "ANALYSIS_MCP_URL", "PRESENTATION_MCP_URL", "DATABASE_URL", "PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
async def publisher() -> AsyncIterator[ResultPublisher]:
    """The completion gate's publisher as the deployed app has it,
    `McpExecutor.publish_result`, publishing each derived frame with the recipe
    the store's traces build to a fake dfo-apis."""
    assembler = FakeApis().assembler()
    yield McpExecutor(settings=Settings(), publish_frame=assembler.publish_frame).publish_result
    await assembler.aclose()


@pytest.fixture(autouse=True)
async def _reset_data_mcp_client() -> AsyncIterator[None]:
    """`data_mcp.server`'s lifespan leaves `client_holder` alone, so a client set
    here would leak into later tests across the whole workspace suite."""
    yield
    client_holder.set_client(None)


_StatsRegistry = dict[tuple[str, str], list[dict[str, Any]]]


@pytest.fixture
def _stats_registry() -> _StatsRegistry:
    return {}


@pytest.fixture(autouse=True)
def _stub_fetch_series_statistics(
    monkeypatch: pytest.MonkeyPatch, _stats_registry: _StatsRegistry
) -> None:
    """Stand in for `fetch_series_statistics` and `fetch_series_statistics_table`, reading `_stats_registry`
    (`_register_statistics` populates it) instead of a real signed-URL Parquet fetch. Two
    call sites reach it in this suite's real, mounted-in-process apps -- the harness's own
    `executor.run` (the evidence check) and, since `_mint_response`'s own `series_count`
    makes `get_data`'s digest fetch it too, `data_mcp.get_data_tool` -- so both module
    bindings are patched, the same way `frame_materialisation._fetch_artifact_bytes` already
    is for the frame artifact. This suite proves the executor's evidence-check wiring across
    the real mounted MCP transport, not the artifact-fetch mechanics themselves -- those are
    `test_series_statistics.py`'s.
    """

    def _models(subject_id: str, content_version: str) -> list[SeriesStatisticsModel]:
        rows = _stats_registry.get((subject_id, content_version), [])
        return [SeriesStatisticsModel.model_validate(row) for row in rows]

    async def _stub(*, subject_id: str, content_version: str, artifact: Any, **_: Any) -> Any:
        del artifact
        return _models(subject_id, content_version)

    async def _table_stub(*, subject_id: str, content_version: str, artifact: Any, **_: Any) -> Any:
        del artifact
        return series_statistics_table(_models(subject_id, content_version))

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", _table_stub)
    monkeypatch.setattr("data_mcp.get_data_tool.fetch_series_statistics", _stub)


def _register_statistics(registry: _StatsRegistry, response: dict[str, Any]) -> dict[str, Any]:
    """Add a `series_count` and a statistics-artifact reference to `response`, and register
    its own `statistics` rows into `registry` under `(spec_id, content_version)` for
    `_stub_fetch_series_statistics` to read back -- the real wire shape, minus the real
    Parquet round trip. Mutates and returns `response`.
    """
    statistics = response["statistics"]
    response["series_count"] = len(statistics)
    response["artifacts"] = [
        *response["artifacts"],
        {
            "artifact_digest": "sha256:" + "8" * 64,
            "rendering": {"statistics": "series"},
            "bytes": 512,
            "media_type": "application/vnd.apache.parquet",
            "created_at": "2026-08-11T00:00:00+00:00",
            "tier": "working",
            "url": "/links/should-not-be-fetched",
        },
    ]
    key = (response["definition"]["spec_id"], response["content_version"])
    registry[key] = statistics
    return response


def _parquet_bytes() -> bytes:
    """Ten annual rows for Kenya, one per year of the requirement's period."""
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:DebtServicePaid"] * len(YEARS), type=pa.string()),
            "recipient": pa.array(["country/KEN"] * len(YEARS), type=pa.string()),
            "time_period": pa.array(YEARS, type=pa.string()),
            "value": pa.array([float(100 + i) for i in range(len(YEARS))], type=pa.float64()),
            "unit": pa.array(["USD"] * len(YEARS), type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


# The version whose frame file is `_parquet_bytes()`.
CONTENT_VERSION = version_of(_parquet_bytes())


def _mint_response() -> dict[str, Any]:
    """A mint response whose manifest carries the statistics the evidence check reads."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": SPEC_ID, "identity": IDENTITY},
        "content_version": CONTENT_VERSION,
        "coverage": {"rows": len(YEARS), "dimensions": [{"label": "years", "count": len(YEARS)}]},
        "citation": {"text": "World Bank, International Debt Statistics", "sources": []},
        "trace": {
            "graph_revision": "g1",
            "facets": [],
            "resolved_stat_vars": [{"ref": "dcid:DebtServicePaid", "roles": ["recipient"]}],
        },
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(_parquet_bytes()).hexdigest()}",
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": len(_parquet_bytes()),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        "statistics": [
            {
                "stat_var": "dcid:DebtServicePaid",
                "entities": [{"role": "recipient", "ref": "country/KEN"}],
                "facet": {"unit": "USD"},
                "observations": len(YEARS),
                "first": {"value": 100.0, "period": YEARS[0]},
                "latest": {"value": 109.0, "period": YEARS[-1]},
                "minimum": {"value": 100.0, "period": YEARS[0]},
                "maximum": {"value": 109.0, "period": YEARS[-1]},
            }
        ],
    }


def _approved_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
    schemas: dict[str, Any],
) -> Plan:
    source = SelectedSource(
        spec_id=SPEC_ID,
        sentence="external debt service paid by Kenya, 2015 to 2024",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["DebtServicePaid"]),
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2015, end=2024),
        identity=IDENTITY,
    )
    draft = Plan(
        question="How much has Kenya paid to service its debt each year since 2015?",
        decisions=[decision()],
        evidence_requirements=[requirement(), derived_requirement()],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC_ID),
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
        schemas=schemas,
        discovered={"annual_debt_service": source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_an_approved_plan_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _parquet_bytes()

    serve_canonical_frames(monkeypatch, fake_fetch)

    mint_response = _register_statistics(_stats_registry, _mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _approved_plan(decision, requirement, derived_requirement, schemas)

        finished, outcome = await execute_approved_plan(
            approved, surface=surface, publisher=publisher
        )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == ["debt", "totals"]
    # The retrieval's frame is what the aggregate was called with.
    assert outcome.steps[0].frame_ref.spec_id == SPEC_ID
    assert outcome.steps[0].frame_ref.content_version == CONTENT_VERSION
    assert outcome.results["totals"].spec_id.startswith("ops_")
    (evidence,) = outcome.evidence
    assert evidence.requirement_id == "annual_debt_service"
    assert evidence.stat_vars == ("DebtServicePaid",)
    assert evidence.window == Period(start=2015, end=2024)
    assert [gap.code.value for gap in evidence.gaps] == ["unit_price_basis_unverified"]
    assert evidence.citation.startswith("World Bank")


HEALTH_SPEC = "spec_" + "7" * 32
TOTAL_SPEC = "spec_" + "8" * 32

RANK_SPEC = "spec_" + "e" * 32
RANK_YEARS = ["2010", "2011", "2012"]

KENYA_RECIPIENT = ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya")
UGANDA_RECIPIENT = ResolvedEntity(role="recipient", ref="country/UGA", label="Uganda")
TANZANIA_RECIPIENT = ResolvedEntity(role="recipient", ref="country/TZA", label="Tanzania")


def _donor_parquet_bytes(*, spec_id: str, values: dict[str, float]) -> bytes:
    """One row per donor for 2023, under `spec_id`'s own stat_var."""
    stat_var = (
        "dcid:OfficialDevelopmentAssistanceToHealth"
        if spec_id == HEALTH_SPEC
        else "dcid:TotalOfficialDevelopmentAssistance"
    )
    donors = sorted(values)
    table = pa.table(
        {
            "stat_var": pa.array([stat_var] * len(donors), type=pa.string()),
            "donor": pa.array(donors, type=pa.string()),
            "time_period": pa.array(["2023"] * len(donors), type=pa.string()),
            "value": pa.array([values[donor] for donor in donors], type=pa.float64()),
            "unit": pa.array(["USD"] * len(donors), type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _donor_mint_response(*, spec_id: str, values: dict[str, float]) -> dict[str, Any]:
    parquet = _donor_parquet_bytes(spec_id=spec_id, values=values)
    content_version = version_of(parquet)
    stat_var = (
        "OfficialDevelopmentAssistanceToHealth"
        if spec_id == HEALTH_SPEC
        else "TotalOfficialDevelopmentAssistance"
    )
    donors = sorted(values)
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {
                "identity_version": "spec1",
                "shape": {"id": stat_var, "kind": "stat_var"},
                "entities": {"donor": [f"country/{code}" for code in ("USA", "GBR", "FRA")]},
                "facets": {"unit": {"ref": "USD", "state": "value"}},
                "requested_window": {"kind": "all"},
                "slots": [],
            },
        },
        "content_version": content_version,
        "coverage": {"rows": len(donors), "dimensions": [{"label": "years", "count": 1}]},
        "citation": {"text": "OECD DAC", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        # One series per entity, matching a real manifest's own shape: a
        # combined entry naming all three donors reads as one series with
        # more observations than its own one-year span (`grain_ambiguous`).
        "statistics": [
            {
                "stat_var": f"dcid:{stat_var}",
                "entities": [{"role": "donor", "ref": f"country/{code}"}],
                "facet": {"unit": "USD"},
                "observations": 1,
                "first": {"value": values[f"country/{code}"], "period": "2023"},
                "latest": {"value": values[f"country/{code}"], "period": "2023"},
                "minimum": {"value": values[f"country/{code}"], "period": "2023"},
                "maximum": {"value": values[f"country/{code}"], "period": "2023"},
            }
            for code in ("USA", "GBR", "FRA")
        ],
    }


HEALTH_VALUES = {
    "country/USA": 6_000_000_000.0,
    "country/GBR": 3_000_000_000.0,
    "country/FRA": 9_000_000_000.0,
}
TOTAL_VALUES = {
    "country/USA": 20_000_000_000.0,
    "country/GBR": 15_000_000_000.0,
    "country/FRA": 12_000_000_000.0,
}


def _share_approved_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """`retrieve(health) + retrieve(total) -> share_of(axis=donor, per_axis) -> rank`,
    run through the real compiler and the real mounted services rather than
    the bare tool functions."""
    # The compiler's own width check (`_check_side`) reasons about
    # `SelectedSource.entities`, which is what discovery resolved for the
    # requirement's own named focus -- one donor -- not the wider
    # population the real retrieved frame will carry once `get_data` runs.
    # Naming all three here would trip `ENTITY_REQUIRED` the same way the
    # compiler's own keyed-share precedent avoids it
    # (`test_compile.py::test_rank_donors_by_their_share_of_oda_to_health_compiles_end_to_end`).
    focus = (ResolvedEntity(role="donor", ref="country/USA", label="United States"),)
    health_source = SelectedSource(
        spec_id=HEALTH_SPEC,
        sentence="ODA to health from the United States, the United Kingdom and France, 2023",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["OfficialDevelopmentAssistanceToHealth"]),
        entities=focus,
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2023, end=2023),
        identity=_donor_mint_response(spec_id=HEALTH_SPEC, values=HEALTH_VALUES)["definition"][
            "identity"
        ],
    )
    total_source = SelectedSource(
        spec_id=TOTAL_SPEC,
        sentence="total ODA from the United States, the United Kingdom and France, 2023",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["TotalOfficialDevelopmentAssistance"]),
        entities=focus,
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2023, end=2023),
        identity=_donor_mint_response(spec_id=TOTAL_SPEC, values=TOTAL_VALUES)["definition"][
            "identity"
        ],
    )
    # `among='population'` means the *source* holds every donor; the
    # requirement itself names one focus entity, exactly like the compiler's
    # own keyed-share precedent (`test_compile.py::
    # test_rank_donors_by_their_share_of_oda_to_health_compiles_end_to_end`)
    # -- naming all three here would trip `ENTITY_REQUIRED`, since a whole
    # side over several named entities has no defined single series.
    health_req = DatasetRequirement(
        id="health_oda",
        decision_ids=["measure"],
        measure="ODA to health",
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2023, end=2023),
    )
    total_req = DatasetRequirement(
        id="total_oda",
        decision_ids=["measure"],
        measure="total ODA",
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2023, end=2023),
    )
    share_req = ShareOfRequirement(
        id="health_share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="health_oda"),
        denominator=RequirementSide(input_id="total_oda"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population_scope"],
        input_id="health_share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="donor",
    )
    population_decision = decision(
        slot="population_scope",
        description="Whether the rank compares only the three named donors or every donor the source holds.",
        options=["the three named donors only", "every donor the source holds"],
        chosen="every donor the source holds",
        resolved_by=ResolvedBy.USER,
        rationale="The question asks how each donor's own share ranks, over the whole population.",
        fixes=DecisionFixes(among=RankAmong.POPULATION),
    )
    draft = Plan(
        question="Rank donors by their share of official development assistance to health.",
        decisions=[decision(), population_decision],
        evidence_requirements=[health_req, total_req, share_req, rank_req],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="health", requirement_id="health_oda", spec_id=HEALTH_SPEC),
            RetrieveOperation(id="total", requirement_id="total_oda", spec_id=TOTAL_SPEC),
            ShareOfOperation(
                id="health_share",
                requirement_id="health_share",
                numerator=Side(input="health"),
                denominator=Side(input="total"),
                axis="donor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
            RankOperation(
                id="ranked",
                requirement_id="ranked",
                input="health_share",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis="donor",
            ),
        ],
        terminal=["ranked"],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=schemas,
        discovered={"health_oda": health_source, "total_oda": total_source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_the_keyed_share_chain_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    """`retrieve(health) + retrieve(total) -> share_of(axis=donor) -> rank`, run
    through the real compiler and the real mounted services. France spends
    less on health than Germany would (not modelled here) but has the
    largest *share* of its own total, so a rank that read the raw values
    instead of the shares would put France last, not first."""

    async def fake_fetch(*, subject_id: str, **_kwargs: Any) -> bytes:
        values = HEALTH_VALUES if subject_id == HEALTH_SPEC else TOTAL_VALUES
        return _donor_parquet_bytes(spec_id=subject_id, values=values)

    serve_canonical_frames(monkeypatch, fake_fetch)

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path != "/api/datasets":
            raise AssertionError(f"unexpected call to {request.url.path}")
        spec_id = json.loads(request.content)["input"]["spec_id"]
        values = HEALTH_VALUES if spec_id == HEALTH_SPEC else TOTAL_VALUES
        response = _register_statistics(
            _stats_registry,
            _donor_mint_response(spec_id=spec_id, values=values),
        )
        return httpx2.Response(200, json=response)

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _share_approved_plan(decision, schemas)

        finished, outcome = await execute_approved_plan(
            approved, surface=surface, publisher=publisher
        )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == ["health", "total", "health_share", "ranked"]
    rank_digest = outcome.steps[-1].digest
    # France's share of its own total, 0.75, in per cent.
    assert "#1 donor=country/FRA: 75 [" in rank_digest


RANK_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "AidReceived", "kind": "stat_var"},
    "entities": {"recipient": ["country/KEN", "country/UGA", "country/TZA"]},
    "facets": {"unit": {"ref": "USD", "state": "value"}},
    "requested_window": {"kind": "all"},
    "slots": [],
}


def _rank_parquet_bytes() -> bytes:
    """Kenya and Tanzania across all three years; Uganda has no row in 2011
    -- a genuine hole in the population's own series, the fixture the
    plan-level narrowing rule (a filter between the rank and the growth)
    exists to keep a growth from tripping over at run time."""
    rows = [
        ("country/KEN", "2010", 50.0),
        ("country/UGA", "2010", 30.0),
        ("country/TZA", "2010", 70.0),
        ("country/KEN", "2011", 60.0),
        ("country/TZA", "2011", 40.0),
        ("country/KEN", "2012", 90.0),
        ("country/UGA", "2012", 80.0),
        ("country/TZA", "2012", 20.0),
    ]
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:AidReceived"] * len(rows), type=pa.string()),
            "recipient": pa.array([row[0] for row in rows], type=pa.string()),
            "time_period": pa.array([row[1] for row in rows], type=pa.string()),
            "value": pa.array([row[2] for row in rows], type=pa.float64()),
            "unit": pa.array(["USD"] * len(rows), type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _rank_mint_response() -> dict[str, Any]:
    parquet = _rank_parquet_bytes()
    rank_content_version = version_of(parquet)
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": RANK_SPEC, "identity": RANK_IDENTITY},
        "content_version": rank_content_version,
        "coverage": {"rows": 8, "dimensions": [{"label": "years", "count": 3}]},
        "citation": {"text": "OECD DAC", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        # One series per entity -- Uganda's own span is 2010-2012 (three
        # years) with only two observations, the hole this fixture exists
        # to carry; that reads as `coverage_gaps` (a warning under the
        # default `on_missing="report_gaps"`), not `grain_ambiguous`, which
        # is what a single entry naming all three entities would trip.
        "statistics": [
            {
                "stat_var": "dcid:AidReceived",
                "entities": [{"role": "recipient", "ref": "country/KEN"}],
                "facet": {"unit": "USD"},
                "observations": 3,
                "first": {"value": 50.0, "period": "2010"},
                "latest": {"value": 90.0, "period": "2012"},
                "minimum": {"value": 50.0, "period": "2010"},
                "maximum": {"value": 90.0, "period": "2012"},
            },
            {
                "stat_var": "dcid:AidReceived",
                "entities": [{"role": "recipient", "ref": "country/UGA"}],
                "facet": {"unit": "USD"},
                "observations": 2,
                "first": {"value": 30.0, "period": "2010"},
                "latest": {"value": 80.0, "period": "2012"},
                "minimum": {"value": 30.0, "period": "2010"},
                "maximum": {"value": 80.0, "period": "2012"},
            },
            {
                "stat_var": "dcid:AidReceived",
                "entities": [{"role": "recipient", "ref": "country/TZA"}],
                "facet": {"unit": "USD"},
                "observations": 3,
                "first": {"value": 70.0, "period": "2010"},
                "latest": {"value": 20.0, "period": "2012"},
                "minimum": {"value": 20.0, "period": "2012"},
                "maximum": {"value": 70.0, "period": "2010"},
            },
        ],
    }


def _rank_approved_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """`retrieve -> rank(per_period=True) -> filter(EntityIn=[Kenya]) -> growth(absolute)`:
    the motivating per-period-rank chain, run through the real compiler and
    the real mounted services."""
    source = SelectedSource(
        spec_id=RANK_SPEC,
        sentence="aid received by Kenya, Uganda and Tanzania, 2010 to 2012",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["AidReceived"]),
        entities=(KENYA_RECIPIENT, UGANDA_RECIPIENT, TANZANIA_RECIPIENT),
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2012),
        identity=RANK_IDENTITY,
    )
    population_decision = decision(
        slot="population_scope",
        description="Whether the rank compares Kenya alone or every recipient the source holds.",
        options=["Kenya alone", "every recipient the source holds"],
        chosen="every recipient the source holds",
        resolved_by=ResolvedBy.USER,
        rationale="The question asks how Kenya's own position has moved among every recipient.",
        fixes=DecisionFixes(among=RankAmong.POPULATION),
    )
    aid_req = DatasetRequirement(
        id="aid_received",
        decision_ids=["measure", "population_scope"],
        measure="aid received",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2010, end=2012),
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population_scope"],
        input_id="aid_received",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        per_period=True,
        window_start=2010,
        window_end=2012,
    )
    filter_req = FilterRequirement(
        id="kenya_only",
        decision_ids=["population_scope"],
        input_id="ranked",
        predicates=(EntityIn(names=("Kenya",), role="recipient"),),
    )
    growth_req = GrowthRequirement(
        id="growth_of_rank",
        decision_ids=["measure"],
        input_id="kenya_only",
        window_start=2010,
        window_end=2012,
        change=GrowthChange.ABSOLUTE,
    )
    draft = Plan(
        question="How has Kenya's rank among aid recipients changed since 2010?",
        decisions=[decision(), population_decision],
        evidence_requirements=[aid_req, rank_req, filter_req, growth_req],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="aid", requirement_id="aid_received", spec_id=RANK_SPEC),
            RankOperation(
                id="ranked",
                requirement_id="ranked",
                input="aid",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis="recipient",
                per_period=True,
                window_start=2010,
                window_end=2012,
            ),
            FilterOperation(
                id="kenya_only",
                requirement_id="kenya_only",
                input="ranked",
                predicates=(EntityIn(names=("Kenya",), role="recipient"),),
            ),
            GrowthOperation(
                id="growth_of_rank",
                requirement_id="growth_of_rank",
                input="kenya_only",
                window_start=2010,
                window_end=2012,
                change=GrowthChange.ABSOLUTE,
            ),
        ],
        terminal=["growth_of_rank"],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=schemas,
        discovered={"aid_received": source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_kenyas_rank_series_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    """The motivating per-period-rank chain, over a fixture with a genuine
    hole (Uganda has no row in 2011), so each year's rank positions depend
    on which entities reported that year. A population-wide change in rank
    would mix movement with reporting, which is why the plan narrows to
    Kenya between the rank and the growth. A fixture with every entity in
    every period would hide that hole. See
    `test_verb_coverage_corpus.py::_run_kenya_rank_change_since_2010` for
    what growth computes straight off the population-wide series, against
    the bare tool functions."""

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _rank_parquet_bytes()

    serve_canonical_frames(monkeypatch, fake_fetch)

    rank_response = _register_statistics(_stats_registry, _rank_mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=rank_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _rank_approved_plan(decision, schemas)

        finished, outcome = await execute_approved_plan(
            approved, surface=surface, publisher=publisher
        )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == [
        "aid",
        "ranked",
        "kenya_only",
        "growth_of_rank",
    ]
    growth_digest = outcome.steps[-1].digest
    assert "1 place better" in growth_digest
    assert "2 -> 1" in growth_digest


TOP_DONOR_SPEC = "spec_" + "c" * 32

# ODA disbursed in 2023, in billions of US dollars, keyed by donor ref.
TOP_DONOR_VALUES = {
    "country/USA": 64.7,
    "country/DEU": 37.9,
    "country/JPN": 19.6,
    "country/GBR": 15.4,
    "country/FRA": 15.1,
    "country/CAN": 6.9,
}
TOP_DONORS = (
    ResolvedEntity(role="donor", ref="country/USA", label="United States"),
    ResolvedEntity(role="donor", ref="country/DEU", label="Germany"),
    ResolvedEntity(role="donor", ref="country/JPN", label="Japan"),
    ResolvedEntity(role="donor", ref="country/GBR", label="United Kingdom"),
    ResolvedEntity(role="donor", ref="country/FRA", label="France"),
    ResolvedEntity(role="donor", ref="country/CAN", label="Canada"),
)
TOP_DONOR_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "OdaDisbursed", "kind": "stat_var"},
    "entities": {"donor": [donor.ref for donor in TOP_DONORS]},
    "facets": {"unit": {"ref": "USD", "state": "value"}},
    "requested_window": {"kind": "all"},
    "slots": [],
}


def _top_donor_parquet_bytes() -> bytes:
    """One 2023 row per donor in `TOP_DONOR_VALUES`."""
    donors = sorted(TOP_DONOR_VALUES)
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:OdaDisbursed"] * len(donors), type=pa.string()),
            "donor": pa.array(donors, type=pa.string()),
            "time_period": pa.array(["2023"] * len(donors), type=pa.string()),
            "value": pa.array([TOP_DONOR_VALUES[d] for d in donors], type=pa.float64()),
            "unit": pa.array(["USD"] * len(donors), type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _top_donor_mint_response() -> dict[str, Any]:
    parquet = _top_donor_parquet_bytes()
    top_donor_content_version = version_of(parquet)
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": TOP_DONOR_SPEC,
            "identity": TOP_DONOR_IDENTITY,
        },
        "content_version": top_donor_content_version,
        "coverage": {
            "rows": len(TOP_DONOR_VALUES),
            "dimensions": [{"label": "years", "count": 1}],
        },
        "citation": {"text": "OECD DAC", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        "statistics": [
            {
                "stat_var": "dcid:OdaDisbursed",
                "entities": [{"role": "donor", "ref": ref}],
                "facet": {"unit": "USD"},
                "observations": 1,
                "first": {"value": value, "period": "2023"},
                "latest": {"value": value, "period": "2023"},
                "minimum": {"value": value, "period": "2023"},
                "maximum": {"value": value, "period": "2023"},
            }
            for ref, value in TOP_DONOR_VALUES.items()
        ],
    }


def _top_donors_approved_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """`retrieve -> rank(limit=3)` over every donor the source holds, run
    through the translator, the real compiler and the real mounted services."""
    source = SelectedSource(
        spec_id=TOP_DONOR_SPEC,
        sentence="ODA disbursed by six DAC donors, 2023",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["OdaDisbursed"]),
        entities=TOP_DONORS,
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2023, end=2023),
        identity=TOP_DONOR_IDENTITY,
    )
    oda_req = DatasetRequirement(
        id="oda_disbursed",
        decision_ids=["measure"],
        measure="ODA disbursed",
        entities=[Entity(name=donor.label, role="donor") for donor in TOP_DONORS],
        period=Period(start=2023, end=2023),
    )
    rank_req = RankRequirement(
        id="top_three",
        decision_ids=["measure", "population_scope"],
        input_id="oda_disbursed",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="donor",
        limit=3,
    )
    population_decision = decision(
        slot="population_scope",
        description="Whether the rank compares the named donors or every donor the source holds.",
        options=["the named donors only", "every donor the source holds"],
        chosen="every donor the source holds",
        resolved_by=ResolvedBy.USER,
        rationale="The question asks for the top three donors overall.",
        fixes=DecisionFixes(among=RankAmong.POPULATION),
    )
    draft = Plan(
        question="Which three donors disbursed the most ODA in 2023?",
        decisions=[decision(), population_decision],
        evidence_requirements=[oda_req, rank_req],
        status=PlanStatus.DRAFT,
    )
    translation = build_translation(draft, {"oda_disbursed": TOP_DONOR_SPEC})
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=schemas,
        discovered={"oda_disbursed": source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_a_top_three_rank_keeps_three_donors_and_reports_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    """A planned `limit=3` reaches `rank` as an argument, so the stored frame
    holds the three highest donors at their measured amounts, and the three
    cut donors reach the digest and the trace as a `beyond_limit` exclusion."""

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _top_donor_parquet_bytes()

    serve_canonical_frames(monkeypatch, fake_fetch)

    mint_response = _register_statistics(_stats_registry, _top_donor_mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _top_donors_approved_plan(decision, schemas)
        assert approved.workflow is not None
        assert approved.workflow.steps[-1].args["limit"] == 3

        finished, outcome = await execute_approved_plan(
            approved, surface=surface, publisher=publisher
        )
        assert isinstance(outcome, Accepted), outcome
        stored_frame, stored_trace = await get_frame_store().get(outcome.results["top_three"])

    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == ["oda_disbursed", "top_three"]
    donor = stored_frame.columns.index("donor")
    value = stored_frame.columns.index("value")
    assert {row[donor]: row[value] for row in stored_frame.rows} == {
        "country/USA": 64.7,
        "country/DEU": 37.9,
        "country/JPN": 19.6,
    }
    (cut,) = [e for e in stored_trace.exclusions if e.reason == "beyond_limit"]
    assert cut.rows == 3
    assert cut.entities == {"donor": ("country/CAN", "country/FRA", "country/GBR")}
    rank_step = outcome.steps[-1]
    assert rank_step.row_count == 3
    assert "limit 3 (kept 3 of 6 ranked entities)" in rank_step.digest
    assert "beyond_limit (ranked beyond the limit): 3 row(s)" in rank_step.digest


ROLLING_SPEC = "spec_" + "9" * 32
ROLLING_YEARS = range(2015, 2025)


def _rolling_last_years(*, reporting_2024: int) -> dict[str, int]:
    """Each donor's last reported year: 2024 for the first `reporting_2024`, 2023 for the rest."""
    return {
        donor.ref: 2024 if position < reporting_2024 else 2023
        for position, donor in enumerate(TOP_DONORS)
    }


def _rolling_parquet_bytes(last_years: Mapping[str, int]) -> bytes:
    """One row a year for each donor, from 2015 through its last reported year."""
    rows = [
        (ref, str(year), 10.0 * (position + 1) + (year - 2015))
        for position, (ref, last) in enumerate(last_years.items())
        for year in ROLLING_YEARS
        if year <= last
    ]
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:OdaDisbursed"] * len(rows), type=pa.string()),
            "donor": pa.array([row[0] for row in rows], type=pa.string()),
            "time_period": pa.array([row[1] for row in rows], type=pa.string()),
            "value": pa.array([row[2] for row in rows], type=pa.float64()),
            "unit": pa.array(["USD"] * len(rows), type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _rolling_mint_response(last_years: Mapping[str, int]) -> dict[str, Any]:
    parquet = _rolling_parquet_bytes(last_years)
    content_version = version_of(parquet)
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": ROLLING_SPEC,
            "identity": TOP_DONOR_IDENTITY,
        },
        "content_version": content_version,
        "coverage": {
            "rows": sum(last - 2014 for last in last_years.values()),
            "dimensions": [{"label": "years", "count": len(ROLLING_YEARS)}],
        },
        "citation": {"text": "OECD DAC", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        "statistics": [
            {
                "stat_var": "dcid:OdaDisbursed",
                "entities": [{"role": "donor", "ref": ref}],
                "facet": {"unit": "USD"},
                "observations": last - 2014,
                "first": {"value": 10.0, "period": "2015"},
                "latest": {"value": 20.0, "period": str(last)},
                "minimum": {"value": 10.0, "period": "2015"},
                "maximum": {"value": 20.0, "period": str(last)},
            }
            for ref, last in last_years.items()
        ],
    }


def _rolling_approved_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """`retrieve -> filter(the last five years) -> growth(over them)`, both planned against 2024."""
    source = SelectedSource(
        spec_id=ROLLING_SPEC,
        sentence="ODA disbursed by six DAC donors, 2015 to 2024",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["OdaDisbursed"]),
        entities=TOP_DONORS,
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2015, end=2024),
        identity=TOP_DONOR_IDENTITY,
    )
    since = (PeriodThreshold(op="gte", year=2019, rolling=RelativePeriod(offset=-5)),)
    window: dict[str, Any] = {
        "window_start": 2019,
        "window_end": 2024,
        "rolling": RollingWindow(
            window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
        ),
    }
    draft = Plan(
        question="How has each donor's ODA changed over the last five years?",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="oda_disbursed",
                decision_ids=["measure"],
                measure="ODA disbursed",
                entities=[Entity(name=donor.label, role="donor") for donor in TOP_DONORS],
                period=Period(start=2015, end=2024),
                period_end_open=True,
            ),
            FilterRequirement(
                id="recent_years",
                decision_ids=["measure"],
                input_id="oda_disbursed",
                predicates=since,
            ),
            GrowthRequirement(
                id="recent_growth", decision_ids=["measure"], input_id="recent_years", **window
            ),
        ],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="oda", requirement_id="oda_disbursed", spec_id=ROLLING_SPEC),
            FilterOperation(
                id="recent", requirement_id="recent_years", input="oda", predicates=since
            ),
            GrowthOperation(id="change", requirement_id="recent_growth", input="recent", **window),
        ],
        terminal=["change"],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=schemas,
        discovered={"oda_disbursed": source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


class _CountingSurface:
    """A surface that passes every call through and keeps each call's tool and arguments."""

    def __init__(self, surface: McpSurface) -> None:
        self._surface = surface
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append((tool.name, dict(arguments)))
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


@pytest.fixture
def run_rolling_plan(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> Callable[..., Any]:
    """Return a coroutine function running the rolling plan twice over one version of its data.

    `reporting_2024` donors of six report 2024 in the version served under
    version its frame file names. Returns the first run's outcome, the recipe of its
    final frame, the second run's outcome, and every call both runs made.
    """

    async def _run(
        *, reporting_2024: int
    ) -> tuple[Accepted, dict[str, Any], Accepted, list[tuple[str, dict[str, Any]]]]:
        last_years = _rolling_last_years(reporting_2024=reporting_2024)

        async def fake_fetch(**_kwargs: Any) -> bytes:
            return _rolling_parquet_bytes(last_years)

        serve_canonical_frames(monkeypatch, fake_fetch)
        mint_response = _register_statistics(_stats_registry, _rolling_mint_response(last_years))

        def mint_handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/api/datasets":
                return httpx2.Response(200, json=mint_response)
            raise AssertionError(f"unexpected call to {request.url.path}")

        client_holder.set_client(data_mcp_client(mint_handler))
        app = create_app(settings=Settings(mcp_api_token="test-token"))
        async with running_lifespan(app):
            surface = _CountingSurface(
                McpSurface(
                    sources={
                        "data-mcp": mounted_mcp_transport(
                            app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                        ),
                        "analysis-mcp": mounted_mcp_transport(
                            app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                        ),
                    }
                )
            )
            schemas = {schema.name: schema for schema in await surface.list_tools()}
            _, outcome = await execute_approved_plan(
                _rolling_approved_plan(decision, schemas), surface=surface
            )
            assert isinstance(outcome, Accepted), outcome
            recipe = await build_recipe(outcome.results["change"], store=get_frame_store())
            _, again = await execute_approved_plan(
                _rolling_approved_plan(decision, schemas), surface=surface
            )
            assert isinstance(again, Accepted), again
        return outcome, recipe, again, surface.calls

    return _run


async def test_a_rolling_plan_runs_from_the_newest_year_half_of_the_series_hold(
    run_rolling_plan: Callable[..., Any],
) -> None:
    """The plan states 2019 to 2024. The run takes the newest year at least half
    of the retrieved series hold, writes the filter's and the growth's years
    from it, and analysis-mcp accepts both calls against the stored resolution,
    which the final frame's recipe then names."""
    # Two of six donors report 2024, so the newest year half of them hold is 2023.
    outcome, recipe, again, calls = await run_rolling_plan(reporting_2024=2)

    assert [name for name, _ in calls[:4]] == ["get_data", "resolve_anchor", "filter", "growth"]
    (_, kept), (_, growth) = calls[2:4]
    assert [predicate["value"] for predicate in kept["predicates"]] == ["2018"]
    assert (growth["window_start"], growth["window_end"]) == ("2018", "2023")
    (anchor,) = outcome.anchors
    assert (anchor.period, anchor.left_out) == ("2023", ())
    assert recipe["anchor"]["period"] == "2023"
    # A second run of the plan over the same frames resolves once more, to the
    # same stored resolution, and so to the same final frame.
    assert [name for name, _ in calls].count("resolve_anchor") == 2
    assert again.anchors == outcome.anchors
    assert again.results == outcome.results

    # Three of six report 2024 in the next version: 2024 stands, the three
    # donors without it are named, and the final frame is another pair.
    later, later_recipe, _, later_calls = await run_rolling_plan(reporting_2024=3)

    _, later_growth = later_calls[3]
    assert (later_growth["window_start"], later_growth["window_end"]) == ("2019", "2024")
    (later_anchor,) = later.anchors
    assert later_anchor.period == later_recipe["anchor"]["period"] == "2024"
    assert sorted(series.coordinates["donor"] for series in later_anchor.left_out) == sorted(
        donor.ref for donor in TOP_DONORS[3:]
    )
    assert later_anchor.anchor_key != anchor.anchor_key
    assert later.results["change"] != outcome.results["change"]


# --- the cards of a rolling plan, drawn by the orchestrator's make_charts --------

_GROWTH_TITLE = "ODA growth over the last five years"
# The run's frame covers 2018 to 2023, and the frame of the two donors 2019 to 2024.
_RUN_YEARS_TITLE = "ODA growth, 2018-2023"
_YEAR_TITLE = "ODA growth to 2024"


class _Platform:
    """Every MCP client `McpExecutor` holds, as one stand-in. `get_data` goes to
    the mounted data-mcp. The presentation tools answer from the store and keep
    what was written: `describe_frame` gives the view of the frame the store
    holds, `write_component` stores the intent and answers with a resolved
    revision on the frame the intent names, and `describe_component` reads the
    stored intent back."""

    def __init__(self, surface: McpSurface) -> None:
        self._surface = surface
        self.writes: list[dict[str, Any]] = []
        self.stored: dict[str, dict[str, Any]] = {}

    async def __aenter__(self) -> _Platform:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def call_tool(self, tool: str, arguments: dict[str, Any], **_: Any) -> SimpleNamespace:
        if tool == "get_data":
            result = await self._surface.call_tool(
                tool=ToolId(server="data-mcp", name="get_data"), arguments=arguments
            )
            return self._result(result.structured_content or {}, is_error=result.is_error)
        if tool == "describe_frame":
            return self._result(await self._view(arguments["frame_ref"]))
        if tool == "describe_component":
            component_id = arguments["component_id"]
            return self._result(
                {
                    "component_id": component_id,
                    "revision": 1,
                    "stored_intent": self.stored[component_id],
                }
            )
        assert tool == "write_component", tool
        self.writes.append(arguments)
        intent = arguments["intent"]
        component_id = arguments.get("component_id") or f"cmp_{len(self.stored) + 1}"
        self.stored[component_id] = {**self.stored.get(component_id, {}), **intent}
        stored = self.stored[component_id]
        envelope = resolved_revision(
            stored["type"], component_id=component_id, frame_ref=stored["frame_ref"]
        )
        return self._result(
            {"component_id": component_id, "revision": 1, "fact_summary": {}},
            meta={RESOLVED_PAYLOAD_META_KEY: envelope},
        )

    @staticmethod
    def _result(
        body: Mapping[str, Any], *, is_error: bool = False, meta: dict[str, Any] | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            is_error=is_error, structured_content=dict(body), content=[], meta=meta
        )

    @staticmethod
    async def _view(frame_ref: Mapping[str, Any]) -> dict[str, Any]:
        held = await get_frame_store().get(FrameRef.model_validate(frame_ref))
        assert held is not None
        donors = sorted(set(held[0].table.column("donor").to_pylist()))
        return {
            "dimensions": [
                {"name": "stat_var", "plural": False, "values": ["dcid:OdaDisbursed"]},
                {"name": "donor", "plural": True, "values": donors, "total_count": len(donors)},
            ]
        }

    def created(self) -> list[dict[str, Any]]:
        """The intent of every create the platform received."""
        return [write["intent"] for write in self.writes if "component_id" not in write]


@dataclass
class _Studio:
    """The orchestrator's tools over a real `McpExecutor`, after one run of the
    rolling plan: the run's final frame under its chart id, the platform
    stand-in, and each replay's populations and analysis calls."""

    tools: dict[str, Any]
    context: SimpleNamespace
    executor: McpExecutor
    platform: _Platform
    run_frame: dict[str, Any]
    replays: list[Sequence[Mapping[str, Any]]]
    replay_calls: list[str]
    pending: list[bool]
    closed_end: list[int]

    async def charts(self, *charts: dict[str, Any]) -> list[dict[str, Any]]:
        drawn = await self.tools["make_charts"](list(charts), self.context)
        return drawn["charts"]

    def chart(self, donors: Sequence[str], *, title: str = _GROWTH_TITLE, **members: Any) -> Any:
        return {
            "frame": frame_id_for(self.run_frame),
            "type": "bar_chart",
            "title": title,
            "select_json": json.dumps({"donor": list(donors)}),
            **members,
        }

    async def newest_year(self, frame_ref: Mapping[str, Any]) -> str:
        recipe = await build_recipe(FrameRef.model_validate(frame_ref), store=get_frame_store())
        return recipe["anchor"]["period"]


@pytest.fixture
async def studio(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> AsyncIterator[_Studio]:
    """The orchestrator's tools after the rolling plan ran on the mounted services.

    Two of six donors report 2024, so the run's newest year is 2023. The executor
    is the deployed `McpExecutor` with `_Platform` for its clients, and its
    `card_replay` runs `replay_recipe` over the mounted analysis-mcp and
    data-mcp's own mint, publishing to a fake dfo-apis.
    """
    last_years = _rolling_last_years(reporting_2024=2)

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _rolling_parquet_bytes(last_years)

    serve_canonical_frames(monkeypatch, fake_fetch)
    mint_response = _register_statistics(_stats_registry, _rolling_mint_response(last_years))

    closed_end: list[int] = []

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path != "/api/datasets":
            raise AssertionError(f"unexpected call to {request.url.path}")
        if not closed_end:
            return httpx2.Response(200, json=mint_response)
        # The same data under a definition that ends at a stated year.
        window = {"kind": "range", "start_year": 2015, "end_year": closed_end[0]}
        identity = {**TOP_DONOR_IDENTITY, "requested_window": window}
        definition = {**mint_response["definition"], "identity": identity}
        return httpx2.Response(200, json={**mint_response, "definition": definition})

    client_holder.set_client(data_mcp_client(mint_handler))
    apis = FakeApis()
    assembler = apis.assembler()
    replays: list[Sequence[Mapping[str, Any]]] = []
    replay_calls: list[str] = []
    pending: list[bool] = []
    app = create_app(settings=Settings(mcp_api_token="test-token"))
    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )

        async def analysis(*, name: str, arguments: Mapping[str, object]) -> ToolCallResult:
            replay_calls.append(name)
            return await surface.call_tool(
                tool=ToolId(server="analysis-mcp", name=name), arguments=arguments
            )

        async def card_replay(
            recipe: Mapping[str, Any],
            *,
            deadline: CallDeadline,
            populations: Sequence[Mapping[str, Any]],
        ) -> Replayed | Pending | ReplayRefused:
            replays.append(populations)
            if pending:
                return Pending(retry_after_s=1.5, reason="mint_queued")
            async with apis.client() as client:
                return await replay_recipe(
                    recipe,
                    deadline=deadline,
                    call_tool=analysis,
                    store=get_frame_store(),
                    apis=client,
                    publish=assembler.publish_frame,
                    row_budget=RowBudget(max_rows=100_000),
                    populations=populations,
                )

        platform = _Platform(surface)
        monkeypatch.setattr(
            McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: platform
        )
        provenance = ProvenanceStore()
        executor = McpExecutor(settings=Settings(), provenance=provenance, card_replay=card_replay)
        # The plan's steps run as the Studio runs them, so the final frame has
        # the provenance a card on it must cite.
        recording = _RecordingSurface(surface, provenance=provenance)
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        plan = _rolling_approved_plan(decision, schemas)
        _, outcome = await execute_approved_plan(plan, surface=recording)
        assert isinstance(outcome, Accepted), outcome
        assert plan.workflow is not None
        link_evidence_to_terminal_frames(recording, outcome, workflow=plan.workflow)
        run_frame = outcome.results["change"].model_dump()

        # `plan_done` lists a plan's final frames to the agent this way.
        described = await executor(tool="describe_frame", args={"frame_ref": run_frame})
        frames: dict[str, Any] = {}
        register_frames(frames, [(run_frame, described["view"], run_frame)])
        tools = build_orchestrator_tools(resolver=_no_search, executor=executor)
        yield _Studio(
            tools={tool.__name__: tool for tool in tools},
            context=SimpleNamespace(
                state={"surface": "studio", ORCHESTRATOR_FRAMES_KEY: frames},
                invocation_id="inv_1",
            ),
            executor=executor,
            platform=platform,
            run_frame=run_frame,
            replays=replays,
            replay_calls=replay_calls,
            pending=pending,
            closed_end=closed_end,
        )
        await assembler.aclose()


async def _no_search(**_kwargs: Any) -> dict[str, Any]:
    raise AssertionError("these tests search for nothing")


async def test_charts_of_two_donors_are_drawn_on_a_frame_of_their_own_latest_year(
    studio: _Studio,
) -> None:
    """The run takes 2023, the newest year half of the six donors hold. The
    charts draw the two donors that report 2024, so the recipe is replayed once
    over those two and both charts stand on the frame that ends in 2024. The
    chart titled with years is refused and the agent is told why."""
    usa, germany = (donor.ref for donor in TOP_DONORS[:2])
    frame_view = studio.context.state[ORCHESTRATOR_FRAMES_KEY][frame_id_for(studio.run_frame)]

    chart, table, titled = await studio.charts(
        studio.chart([usa, germany]),
        studio.chart([usa, germany], type="table", title="ODA growth by donor"),
        studio.chart([usa, germany], title=_YEAR_TITLE),
    )

    # The agent reads that the frame rolls in the view it charts from.
    assert frame_view["view"]["rolling"] is True
    assert "reply_figures" in chart
    assert "reply_figures" in table
    # 2024 is the newest year of the two donors' frame, not of the run's.
    assert "reply_figures" not in titled
    assert "names 2024" in titled["error"]
    assert titled["code"] == HarnessCode.TITLE_QUOTES_ROLLING_PERIOD
    assert str(titled["fix"]["period"]) == "2024"
    # One replay for the two charts' one selection, with the rolling steps run once.
    assert studio.replays == [
        [{"spec_id": ROLLING_SPEC, "bindings": {"donor": sorted([usa, germany])}}]
    ]
    assert studio.replay_calls == ["resolve_anchor", "filter", "growth"]
    first, second = studio.platform.created()
    assert first["frame_ref"] == second["frame_ref"] != studio.run_frame
    assert await studio.newest_year(studio.run_frame) == "2023"
    assert await studio.newest_year(first["frame_ref"]) == "2024"
    # No chart states a period, so each shows whatever its frame holds.
    assert [sorted(intent) for intent in (first, second)] == [
        ["frame_ref", "select", "title", "type"]
    ] * 2


async def test_charts_of_the_plans_own_donors_are_drawn_on_the_runs_frame(
    studio: _Studio,
) -> None:
    (drawn,) = await studio.charts(studio.chart([donor.ref for donor in TOP_DONORS]))

    assert "reply_figures" in drawn
    assert studio.replays == []
    assert [intent["frame_ref"] for intent in studio.platform.created()] == [studio.run_frame]


async def test_a_replay_still_pending_draws_no_chart_on_the_runs_frame(studio: _Studio) -> None:
    usa, germany = (donor.ref for donor in TOP_DONORS[:2])
    studio.pending.append(True)

    chart, table = await studio.charts(
        studio.chart([usa, germany]), studio.chart([usa, germany], type="table")
    )

    assert [drawn["code"] for drawn in (chart, table)] == [HarnessCode.ROLLING_REPLAY_PENDING] * 2
    assert not any("reply_figures" in drawn for drawn in (chart, table))
    assert "Asking again shortly may work" in chart["error"]
    assert studio.platform.writes == []
    # The second chart of the same series waits out the first one's refusal.
    assert len(studio.replays) == 1


async def test_a_revise_to_other_donors_is_refused_and_its_card_rebuilt_on_their_frame(
    studio: _Studio,
) -> None:
    """The Edit Card route's two writes, at the executor every route shares: a
    revise that selects other series is refused, and the same card written as a
    new one lands on the frame of those series, after one replay."""
    usa, germany = (donor.ref for donor in TOP_DONORS[:2])
    await studio.charts(studio.chart([donor.ref for donor in TOP_DONORS]))
    stored = studio.platform.stored["cmp_1"]

    async def revise(patch: dict[str, Any]) -> dict[str, Any]:
        return await studio.executor(
            tool="write_component", args={"component_id": "cmp_1", "intent": patch}
        )

    retitled = await revise({"title": "How each donor's ODA changed"})
    year = await revise({"title": _RUN_YEARS_TITLE})
    narrowed = await revise({"select": {"donor": [usa, germany]}})
    rebuilt = await studio.executor(
        tool="write_component",
        args={"intent": {**stored, "select": {"donor": [usa, germany]}}},
    )

    assert retitled["outcome"] == "ok"
    assert year["refusal"]["code"] == "title_quotes_rolling_period"
    assert narrowed["refusal"]["code"] == "rolling_select_needs_new_frame"
    assert rebuilt["outcome"] == "ok"
    assert len(studio.replays) == 1
    # Of the two refused revises nothing reached the platform.
    assert [write.get("component_id") for write in studio.platform.writes] == [
        None,
        "cmp_1",
        None,
    ]
    moved = studio.platform.created()[-1]["frame_ref"]
    assert await studio.newest_year(moved) == "2024"
    assert rebuilt["component"]["provenance"]["inputs"][0] == moved


async def test_a_chart_of_fetched_data_counts_its_period_back_from_the_latest_data(
    studio: _Studio,
) -> None:
    """No plan: the agent fetches a dataset and charts it. `latest_periods`
    writes a relative window, whose card may name no year, and a chart of every
    period the frame covers may."""
    fetched = await studio.tools["fetch"](ROLLING_SPEC, studio.context)
    (frame,) = fetched["frames"]
    usa = TOP_DONORS[0].ref

    def chart(title: str, **members: Any) -> dict[str, Any]:
        return {
            "frame": frame["frame"],
            "type": "line_chart",
            "title": title,
            "select_json": json.dumps({"donor": usa}),
            **members,
        }

    moving, titled, fixed = await studio.charts(
        chart("US ODA over the last five years", latest_periods=5),
        chart("US ODA, 2020-2024", latest_periods=5),
        chart("US ODA, 2015-2024"),
    )

    assert "rolling" not in frame["view"]
    assert "reply_figures" in moving
    assert "reply_figures" in fixed
    assert "reply_figures" not in titled
    assert "names 2024" in titled["error"]
    assert titled["code"] == HarnessCode.TITLE_QUOTES_ROLLING_PERIOD
    assert str(titled["fix"]["period"]) == "2024"
    # The platform resolves a relative window as it stores the card, so the
    # titled chart was stored before its title could be judged, and not drawn.
    relative, _, whole = studio.platform.created()
    assert relative["window"] == {
        "start": {"anchor": "latest", "offset": -4},
        "end": {"anchor": "latest", "offset": 0},
    }
    assert relative["frame_ref"]["spec_id"] == ROLLING_SPEC
    assert "window" not in whole
    assert studio.replays == []


async def test_a_chart_counting_back_on_data_fetched_to_a_stated_end_year_is_refused(
    studio: _Studio,
) -> None:
    """A definition that ends at 2024 is rebuilt over the same years for good, so
    a chart counting back from the latest data on it could never move. The agent
    is told to fetch with an open end; a chart of the whole frame is drawn."""
    studio.closed_end.append(2024)
    fetched = await studio.tools["fetch"](ROLLING_SPEC, studio.context)
    (frame,) = fetched["frames"]
    chart = {
        "frame": frame["frame"],
        "type": "line_chart",
        "title": "US ODA",
        "select_json": json.dumps({"donor": TOP_DONORS[0].ref}),
    }

    moving, whole = await studio.charts({**chart, "latest_periods": 5}, chart)

    assert "reply_figures" not in moving
    assert moving["code"] == HarnessCode.RELATIVE_PERIOD_ON_CLOSED_WINDOW
    assert '{"kind": "range", "start_year": Y}' in moving["error"]
    assert "reply_figures" in whole
    assert [("window" in intent) for intent in studio.platform.created()] == [False]


async def test_a_surface_reading_the_mounted_services_without_the_token_is_refused(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
) -> None:
    """The same mount the previous test reads, but through a transport that
    carries no bearer -- proving `McpSurface` doesn't quietly bypass the gate
    `common.auth.apply_gate` put on it (`common.auth`'s own suite proves the
    gate itself; this proves the compiler's own client reaching it obeys it)."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    statuses: list[int] = []
    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", statuses=statuses
                )
            }
        )

        with pytest.raises(ToolSurfaceUnavailableError):
            await surface.list_tools()

    assert set(statuses) == {401}


# "Top 2 by GDP" with no place named: the mint covers every place with data, at every
# level, and carries each place's types when asked (dfo-apis `materialise.enrichment`).
GDP_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "Amount_EconomicActivity_GrossDomesticProduction_Nominal", "kind": "stat_var"},
    "entities": {},
    "requested_window": {"kind": "all"},
    "slots": [],
}
# The id dfo-apis' registry holds the identity under: `get_data` reads an identity the
# definition store lacks from the registry and keeps it only when it hashes to its id.
GDP_SPEC = spec_id_for(canonicalize(GDP_IDENTITY))
GDP_BY_PLACE: dict[str, tuple[str, float]] = {
    "Earth": ("Place", 105.0),
    "country/HIC": ("ProvisionalNode", 64.0),
    "country/USA": ("Country", 27.0),
    "country/CHN": ("Country", 18.0),
    "country/FIN": ("Country", 0.3),
    "nuts/FI2": ("AdministrativeArea1,Country,EurostatNUTS1,Region", 0.002),
    "geoId/06": ("AdministrativeArea1,State", 3.9),
    "undata-geo/G00158000": ("", 2.0),
}


def _gdp_parquet_bytes(*, place_types: bool = True) -> bytes:
    """GDP in 2023 at every place, in the column order dfo-apis renders `options.columns`,
    with each place's types when `place_types`."""
    places = list(GDP_BY_PLACE)
    table = pa.table(
        {
            "stat_var": pa.array(
                ["dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"] * len(places)
            ),
            "observationAbout": pa.array(places),
            "time_period": pa.array(["2023"] * len(places)),
            "value": pa.array([GDP_BY_PLACE[place][1] for place in places]),
            "unit": pa.array(["USDollar"] * len(places)),
            "measurementMethod": pa.array([None] * len(places), type=pa.string()),
            "provenance": pa.array(["dc/base/WorldDevelopmentIndicators"] * len(places)),
            "observationAbout_type": pa.array([GDP_BY_PLACE[place][0] for place in places]),
        }
    )
    if not place_types:
        table = table.drop_columns("observationAbout_type")
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


# When dfo-apis stored each GDP rendering, by whether it carries place types.
_GDP_CREATED_AT = {False: "2026-10-01T00:00:00+00:00", True: "2026-10-02T00:00:00+00:00"}


# Both GDP renderings come back under the version whose canonical frame is the
# plain one.
GDP_CONTENT_VERSION = version_of(_gdp_parquet_bytes(place_types=False))


def _gdp_mint_response(*, place_types: bool = True) -> dict[str, Any]:
    """dfo-apis' manifest for GDP. Both renderings come back under one
    `content_version`, each with its own artifact and the request's options as its
    `rendering`."""
    parquet = _gdp_parquet_bytes(place_types=place_types)
    rendering: dict[str, Any] = {"bundle": False, "format": "parquet", "layout": "frame"}
    if place_types:
        rendering["columns"] = ["observationAbout_type"]
    stat_var = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": GDP_SPEC, "identity": GDP_IDENTITY},
        "content_version": GDP_CONTENT_VERSION,
        "coverage": {
            "rows": len(GDP_BY_PLACE),
            "dimensions": [{"label": "years", "count": 1}],
        },
        "citation": {"text": "World Bank, World Development Indicators", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": rendering,
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": _GDP_CREATED_AT[place_types],
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        "statistics": [
            {
                "stat_var": stat_var,
                "entities": [{"role": "observationAbout", "ref": place}],
                "facet": {"unit": "USDollar"},
                "observations": 1,
                "first": {"value": value, "period": "2023"},
                "latest": {"value": value, "period": "2023"},
                "minimum": {"value": value, "period": "2023"},
                "maximum": {"value": value, "period": "2023"},
            }
            for place, (_types, value) in GDP_BY_PLACE.items()
        ],
    }


def _top_countries_by_gdp_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """ "Top 2 by GDP": no place named, a level decision keeping the countries, then the rank."""
    source = SelectedSource(
        spec_id=GDP_SPEC,
        sentence="Nominal GDP",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(
            ["Amount_EconomicActivity_GrossDomesticProduction_Nominal"]
        ),
        open_entity_axes=(OpenEntityAxis(column="observationAbout", member_count=1996),),
        unit_ref="USDollar",
        unit_label="US dollars",
        coverage_window=Period(start=2023, end=2023),
        identity=GDP_IDENTITY,
    )
    countries = (PlaceLevel(level=PlaceType.COUNTRY),)
    level = decision(
        slot="level",
        description="Which level of place to rank.",
        options=["countries", "continents"],
        chosen="countries",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Top GDP reads as a comparison of countries.",
        fixes=DecisionFixes(predicates=countries, among=RankAmong.POPULATION),
    )
    draft = Plan(
        question="Which economies have the highest GDP?",
        decisions=[level],
        evidence_requirements=[
            DatasetRequirement(
                id="gdp",
                decision_ids=["level"],
                measure="GDP",
                across=("country",),
                period=Period(start=2023, end=2023),
            ),
            FilterRequirement(
                id="countries", decision_ids=["level"], input_id="gdp", predicates=countries
            ),
            RankRequirement(
                id="top_two",
                decision_ids=["level"],
                input_id="countries",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                limit=2,
            ),
        ],
        status=PlanStatus.DRAFT,
    )
    translation = build_translation(draft, {"gdp": GDP_SPEC})
    compiled = compile_plan(
        plan=draft, translation=translation, schemas=schemas, discovered={"gdp": source}
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_a_place_less_rank_keeps_the_countries_before_ranking_them(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    """The retrieval asks dfo-apis for each place's types, the filter keeps the
    places typed Country, and the top two are the two largest countries: Earth
    and the World Bank's high-income group, both larger, are left out."""

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _gdp_parquet_bytes()

    serve_canonical_frames(monkeypatch, fake_fetch)

    mint_response = _register_statistics(_stats_registry, _gdp_mint_response())
    options: list[dict[str, Any]] = []

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            options.append(json.loads(request.content)["options"])
            return httpx2.Response(200, json=mint_response)
        if request.url.path.endswith("/options"):
            # The spec is not in the definition store, so `get_data` reads its identity here.
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _top_countries_by_gdp_plan(decision, schemas)

        finished, outcome = await execute_approved_plan(
            approved, surface=surface, publisher=publisher
        )
        assert isinstance(outcome, Accepted), outcome
        stored_frame, _trace = await get_frame_store().get(outcome.results["top_two"])

    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == ["gdp", "countries", "top_two"]
    assert options[0]["columns"][-1] == "observationAbout_type"
    place = stored_frame.columns.index("observationAbout")
    value = stored_frame.columns.index("value")
    assert {row[place]: row[value] for row in stored_frame.rows} == {
        "country/USA": 27.0,
        "country/CHN": 18.0,
    }


def _serve_gdp_in_both_renderings(
    monkeypatch: pytest.MonkeyPatch, stats_registry: _StatsRegistry
) -> list[tuple[str, bool]]:
    """Answer GDP's mint with the type column when `options.columns` asks for it, and
    without otherwise, under one `content_version`. Each manifest lists both renderings,
    the one not requested first, as dfo-apis lists every rendering stored under a pair.
    Returns the frame fetches as `(content_version, typed)`, in order: the plain read
    fetches the version's canonical frame, and the typed read its rendering."""
    renderings = {
        typed: _register_statistics(stats_registry, _gdp_mint_response(place_types=typed))
        for typed in (False, True)
    }
    data_artifacts = {typed: response["artifacts"][0] for typed, response in renderings.items()}
    typed_by_digest = {
        artifact["artifact_digest"]: typed for typed, artifact in data_artifacts.items()
    }
    for typed, response in renderings.items():
        response["artifacts"].insert(0, data_artifacts[not typed])
    fetched: list[tuple[str, bool]] = []

    async def fake_fetch(*, content_version: str, artifact_digest: str, **_: Any) -> bytes:
        typed = typed_by_digest[artifact_digest]
        fetched.append((content_version, typed))
        return _gdp_parquet_bytes(place_types=typed)

    async def record_canonical(**kwargs: Any) -> bytes | None:
        data = await fetch_any(**kwargs)
        if kwargs["link"] is not None:
            fetched.append((kwargs["content_version"], False))
        return data

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            typed = "columns" in json.loads(request.content)["options"]
            return httpx2.Response(200, json=renderings[typed])
        if request.url.path.endswith("/options"):
            # The spec is not in the definition store, so `get_data` reads its identity here.
            return httpx2.Response(200, json=renderings[True])
        raise AssertionError(f"unexpected call to {request.url.path}")

    serve_canonical_frames(monkeypatch, fake_fetch)
    fetch_any = frame_materialisation._fetch_artifact_bytes
    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", record_canonical)
    client_holder.set_client(data_mcp_client(mint_handler))
    return fetched


_GET_DATA = ToolId(server="data-mcp", name="get_data")


async def test_a_level_filtered_rank_after_a_plain_read_of_the_same_mint_keeps_the_countries(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> None:
    """A plain read of GDP is stored first. The rank's retrieval asks for place types,
    and dfo-apis answers under the same `{spec_id, content_version}`; the typed frame is
    a frame of its own, so the level filter finds its type column."""
    fetched = _serve_gdp_in_both_renderings(monkeypatch, _stats_registry)
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = _gdp_surface(app, mounted_mcp_transport)
        plain = await surface.call_tool(tool=_GET_DATA, arguments={"spec_id": GDP_SPEC})
        assert not plain.is_error, plain.text
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _top_countries_by_gdp_plan(decision, schemas)

        _finished, outcome = await execute_approved_plan(approved, surface=surface)
        assert isinstance(outcome, Accepted), outcome
        stored_frame, _trace = await get_frame_store().get(outcome.results["top_two"])

    assert fetched == [(GDP_CONTENT_VERSION, False), (GDP_CONTENT_VERSION, True)]
    place = stored_frame.columns.index("observationAbout")
    assert {row[place] for row in stored_frame.rows} == {"country/USA", "country/CHN"}


async def test_a_plain_total_after_a_level_filtered_rank_of_the_same_mint_reads_no_type_column(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> None:
    """The rank stores GDP's typed frame first. A later plain read of the same mint
    gets the plain frame, so a period total over it runs: over the typed frame, whose
    places span several levels, the total refuses `place_levels_mixed`."""
    fetched = _serve_gdp_in_both_renderings(monkeypatch, _stats_registry)
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = _gdp_surface(app, mounted_mcp_transport)
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        approved = _top_countries_by_gdp_plan(decision, schemas)
        _finished, outcome = await execute_approved_plan(approved, surface=surface)
        assert isinstance(outcome, Accepted), outcome

        plain = await surface.call_tool(tool=_GET_DATA, arguments={"spec_id": GDP_SPEC})
        assert not plain.is_error, plain.text
        plain_content: dict[str, Any] = plain.structured_content or {}
        total = await surface.call_tool(
            tool=ToolId(server="analysis-mcp", name="aggregate"),
            arguments={
                "frame_ref": plain_content["frame_ref"],
                "aggregation": "sum",
                "group_by": ["time_period"],
            },
        )

    assert not total.is_error, total.text
    assert plain_content["role_keys"] == ["observationAbout"]
    assert fetched == [(GDP_CONTENT_VERSION, True), (GDP_CONTENT_VERSION, False)]


@pytest.mark.parametrize("typed", [False, True], ids=["plain", "typed"])
async def test_a_reads_freshness_date_is_its_own_renderings_beside_a_sibling(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    _stats_registry: _StatsRegistry,
    typed: bool,
) -> None:
    """The harness dates a retrieval by its manifest's data artifact
    (`harness.provenance.data_artifact`). dfo-apis lists the other rendering first
    with its own `created_at`, and get_data's manifest keeps only the requested one."""
    _serve_gdp_in_both_renderings(monkeypatch, _stats_registry)
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = _gdp_surface(app, mounted_mcp_transport)
        read = await surface.call_tool(
            tool=_GET_DATA, arguments={"spec_id": GDP_SPEC, "place_types": typed}
        )

    assert not read.is_error, read.text
    artifacts = (read.structured_content or {})["artifacts"]
    assert data_artifact(artifacts)["created_at"] == _GDP_CREATED_AT[typed]


def _gdp_surface(app: Any, mounted_mcp_transport: Callable[..., Any]) -> McpSurface:
    return McpSurface(
        sources={
            "data-mcp": mounted_mcp_transport(
                app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
            ),
            "analysis-mcp": mounted_mcp_transport(
                app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
            ),
        }
    )


# --- a flow retrieval that ends a plan, served by dfo-apis without its group totals ---

# "Aid to Kenya from every donor": the definition binds Kenya and leaves the donors open.
KENYA_DONORS_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "Oda", "kind": "stat_var"},
    "entities": {"destination": ["country/KEN"]},
    "open_roles": ["origin"],
    "facets": {"unit": {"ref": "USD", "state": "value"}},
    "requested_window": {"kind": "all"},
    "slots": [],
}
KENYA_DONORS_SPEC = spec_id_for(canonicalize(KENYA_DONORS_IDENTITY))
# Each donor's 2023 aid to Kenya and the kind dfo-apis' code index declares for it. The
# DAC total and the G7 hold the two countries' figures again.
KENYA_DONORS: dict[str, tuple[str, float]] = {
    "country/DEU": ("party", 20.0),
    "country/FRA": ("party", 30.0),
    "DACTotal": ("total", 50.0),
    "GroupOfSeven": ("total", 50.0),
}
KENYA_DONOR_TOTALS = sorted(ref for ref, (kind, _) in KENYA_DONORS.items() if kind == "total")
# dfo-apis' base version of the frame, and the version its cut of the group totals mints.
KENYA_DONORS_BASE = "cv_" + "b" * 32
KENYA_DONORS_CUT = "cv_" + "c" * 32


def _kenya_donors(*, cut: bool) -> dict[str, tuple[str, float]]:
    return {
        ref: kept for ref, kept in KENYA_DONORS.items() if not (cut and ref in KENYA_DONOR_TOTALS)
    }


def _kenya_donors_parquet_bytes(*, cut: bool) -> bytes:
    """The frame dfo-apis renders with the party kinds, of the base or the cut version."""
    donors = _kenya_donors(cut=cut)
    count = len(donors)
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Oda"] * count, type=pa.string()),
            "financingProvider": pa.array(list(donors), type=pa.string()),
            "financingRecipient": pa.array(["country/KEN"] * count, type=pa.string()),
            "time_period": pa.array(["2023"] * count, type=pa.string()),
            "value": pa.array([value for _, value in donors.values()], type=pa.float64()),
            "unit": pa.array(["USD"] * count, type=pa.string()),
            "financingProvider_party_kind": pa.array([kind for kind, _ in donors.values()]),
            "financingRecipient_party_kind": pa.array(["party"] * count),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _kenya_donors_mint_response(*, cut: bool, options: Mapping[str, Any]) -> dict[str, Any]:
    """dfo-apis' answer to a mint with `options`: the base version, or the version of the
    frame cut of the donors' group totals, whose coverage, statistics and trace are its
    own."""
    donors = _kenya_donors(cut=cut)
    parquet = _kenya_donors_parquet_bytes(cut=cut)
    trace: dict[str, Any] = {"graph_revision": "g1", "facets": []}
    if cut:
        trace["party_totals_left_out"] = [
            {"role": "financingProvider", "entities": KENYA_DONOR_TOTALS}
        ]
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": KENYA_DONORS_SPEC,
            "identity": KENYA_DONORS_IDENTITY,
        },
        "content_version": KENYA_DONORS_CUT if cut else KENYA_DONORS_BASE,
        "coverage": {
            "rows": len(donors),
            "dimensions": [
                {"label": "financingProvider", "count": len(donors), "axis": "origin"},
                {"label": "financingRecipient", "count": 1, "axis": "destination"},
                {"label": "years", "count": 1},
            ],
        },
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": trace,
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": dict(options),
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
        "statistics": [
            {
                "stat_var": "dcid:Oda",
                "entities": [
                    {"role": "financingProvider", "ref": ref},
                    {"role": "financingRecipient", "ref": "country/KEN"},
                ],
                "facet": {"unit": "USD"},
                "observations": 1,
                "first": {"value": value, "period": "2023"},
                "latest": {"value": value, "period": "2023"},
                "minimum": {"value": value, "period": "2023"},
                "maximum": {"value": value, "period": "2023"},
            }
            for ref, (_, value) in donors.items()
        ],
    }


def _kenya_donors_approved_plan(decision: Callable[..., Decision], schemas: dict[str, Any]) -> Plan:
    """A plan whose answer is the retrieval itself: every donor's aid to Kenya in 2023."""
    source = SelectedSource(
        spec_id=KENYA_DONORS_SPEC,
        sentence="ODA to Kenya from every donor, 2023",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["Oda"]),
        entities=(ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),),
        open_entity_axes=(OpenEntityAxis(column="origin", member_count=len(KENYA_DONORS)),),
        unit_ref="USD",
        unit_label="US dollars",
        coverage_window=Period(start=2023, end=2023),
        identity=KENYA_DONORS_IDENTITY,
    )
    draft = Plan(
        question="Which donors gave Kenya aid in 2023?",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="oda_to_kenya",
                decision_ids=["measure"],
                measure="ODA to Kenya",
                entities=[Entity(name="Kenya", role="recipient")],
                across=("donor",),
                period=Period(start=2023, end=2023),
            )
        ],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="oda", requirement_id="oda_to_kenya", spec_id=KENYA_DONORS_SPEC)
        ],
        terminal=["oda"],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=schemas,
        discovered={"oda_to_kenya": source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


class _ServedFramePlatform:
    """The presentation tools `McpExecutor` calls, reading each frame as dfo-apis serves it:
    by its pair, from `served` (a `content_version` to the donors its frame holds), as
    presentation-mcp reads a retrieval's frame from dfo-apis. Records the pair of every
    `describe_frame` and every created component."""

    def __init__(self, served: Mapping[str, Sequence[str]]) -> None:
        self._served = served
        self.described: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    async def __aenter__(self) -> _ServedFramePlatform:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def call_tool(self, tool: str, arguments: dict[str, Any], **_: Any) -> SimpleNamespace:
        if tool == "describe_frame":
            self.described.append(arguments["frame_ref"])
            donors = sorted(self._served[arguments["frame_ref"]["content_version"]])
            view = {
                "dimensions": [
                    {"name": "stat_var", "plural": False, "values": ["dcid:Oda"]},
                    {
                        "name": "financingProvider",
                        "plural": True,
                        "values": donors,
                        "total_count": len(donors),
                    },
                ]
            }
            return SimpleNamespace(is_error=False, structured_content=view, content=[], meta=None)
        assert tool == "write_component", tool
        intent = arguments["intent"]
        self.created.append(intent)
        envelope = resolved_revision(
            intent["type"], component_id="cmp_1", frame_ref=intent["frame_ref"]
        )
        return SimpleNamespace(
            is_error=False,
            structured_content={"component_id": "cmp_1", "revision": 1, "fact_summary": {}},
            content=[],
            meta={RESOLVED_PAYLOAD_META_KEY: envelope},
        )


async def test_a_retrieval_that_ends_a_plan_is_described_and_drawn_on_dfo_apis_cut_version(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
    publisher: ResultPublisher,
) -> None:
    """The compiled retrieval asks get_data for the frame without party totals, which
    data-mcp asks dfo-apis for. dfo-apis answers with the version of the cut frame, so the
    frame the answer is described and charted by is that version's pair, holding the two
    countries alone, and the digest names the totals left out."""
    minted: list[dict[str, Any]] = []

    async def fake_fetch(*, artifact_digest: str, **_kwargs: Any) -> bytes:
        for cut in (False, True):
            parquet = _kenya_donors_parquet_bytes(cut=cut)
            if artifact_digest == f"sha256:{hashlib.sha256(parquet).hexdigest()}":
                return parquet
        raise AssertionError(f"unexpected fetch of {artifact_digest}")

    serve_canonical_frames(monkeypatch, fake_fetch)

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/options"):
            # The spec is not in the definition store, so `get_data` reads its identity here.
            return httpx2.Response(
                200, json=_kenya_donors_mint_response(cut=False, options=DEFAULT_MINT_OPTIONS)
            )
        if request.url.path != "/api/datasets":
            raise AssertionError(f"unexpected call to {request.url.path}")
        options = json.loads(request.content)["options"]
        minted.append(options)
        response = _kenya_donors_mint_response(
            cut=options.get("without_party_totals") is True, options=options
        )
        return httpx2.Response(200, json=_register_statistics(_stats_registry, response))

    client_holder.set_client(data_mcp_client(mint_handler))
    app = create_app(settings=Settings(mcp_api_token="test-token"))
    served = {
        version: list(_kenya_donors(cut=cut))
        for version, cut in ((KENYA_DONORS_BASE, False), (KENYA_DONORS_CUT, True))
    }
    platform = _ServedFramePlatform(served)
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: platform
    )

    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        provenance = ProvenanceStore()
        recording = _RecordingSurface(surface, provenance=provenance)
        schemas = {schema.name: schema for schema in await surface.list_tools()}
        plan = _kenya_donors_approved_plan(decision, schemas)
        _, outcome = await execute_approved_plan(plan, surface=recording, publisher=publisher)
        assert isinstance(outcome, Accepted), outcome
        assert plan.workflow is not None
        link_evidence_to_terminal_frames(recording, outcome, workflow=plan.workflow)

        # As `harness.graph.execute` hands the answer frames to `plan_done`, which
        # describes each and registers it for `make_charts`.
        (answer,) = answer_frames(outcome).values()
        executor = McpExecutor(settings=Settings(), provenance=provenance)
        described = await executor(
            tool="describe_frame", args={"frame_ref": answer.presented.model_dump()}
        )
        frames: dict[str, Any] = {}
        register_frames(
            frames,
            [(answer.presented.model_dump(), described["view"], answer.store.model_dump())],
        )
        tools = {
            tool.__name__: tool
            for tool in build_orchestrator_tools(resolver=_no_search, executor=executor)
        }
        drawn = await tools["make_charts"](
            [
                {
                    "frame": frame_id_for(answer.presented.model_dump()),
                    "type": "bar_chart",
                    "title": "ODA to Kenya by donor, 2023",
                    "select_json": json.dumps(
                        {"financingProvider": ["country/DEU", "country/FRA"]}
                    ),
                }
            ],
            SimpleNamespace(
                state={"surface": "studio", ORCHESTRATOR_FRAMES_KEY: frames},
                invocation_id="inv_1",
            ),
        )
        held = await get_frame_store().get(answer.store)
        assert held is not None

    cut = FrameRef(spec_id=KENYA_DONORS_SPEC, content_version=KENYA_DONORS_CUT)
    assert answer.presented == cut
    assert answer.store == frame_ref_for(
        spec_id=KENYA_DONORS_SPEC, content_version=KENYA_DONORS_CUT, columns=("party_kind",)
    )
    # The frame the answer is described and drawn by is the cut version's.
    assert platform.described == [cut.model_dump()]
    assert described["view"]["dimensions"][1]["values"] == ["country/DEU", "country/FRA"]
    assert "holds" in drawn["charts"][0], drawn
    assert [intent["frame_ref"] for intent in platform.created] == [cut.model_dump()]
    # The steps reading the retrieval hold the same rows.
    assert sorted(held[0].table.column("financingProvider").to_pylist()) == [
        "country/DEU",
        "country/FRA",
    ]
    # The digest the agent reads names the totals dfo-apis left out.
    (step,) = outcome.steps
    assert (
        "Left out of the frame: 2 group totals on financingProvider (DACTotal, GroupOfSeven)"
        in step.digest
    )
    # data-mcp asked dfo-apis for the cut, with the kinds.
    (options,) = minted
    assert options["without_party_totals"] is True
    assert options["columns"] == ["party_kind"]


# --- an outside client's path to the OECD's all-donors figures ---

# Official Donors, Total ("DACOfficialDonors") and Private Donors, Total ("DACPrivateDonors") are
# DAC2a rows that each hold a whole population of donors: Ethiopia, 2021.
OFFICIAL_DONORS_2021 = 4_267_588_068.0
PRIVATE_DONORS_2021 = 310_811_932.0
WHOLE_TOTALS: dict[str, float] = {
    "DACOfficialDonors": OFFICIAL_DONORS_2021,
    "DACPrivateDonors": PRIVATE_DONORS_2021,
}
OFFICIAL_ONLY_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "Oda", "kind": "stat_var"},
    "entities": {"origin": ["DACOfficialDonors"], "destination": ["country/ETH"]},
    "facets": {"unit": {"ref": "USD", "state": "value"}},
    "requested_window": {"kind": "all"},
    "slots": [],
}
BOTH_TOTALS_IDENTITY: dict[str, Any] = {
    **OFFICIAL_ONLY_IDENTITY,
    "entities": {
        "origin": ["DACOfficialDonors", "DACPrivateDonors"],
        "destination": ["country/ETH"],
    },
}
OFFICIAL_ONLY_SPEC = spec_id_for(canonicalize(OFFICIAL_ONLY_IDENTITY))
BOTH_TOTALS_SPEC = spec_id_for(canonicalize(BOTH_TOTALS_IDENTITY))
_WHOLE_TOTAL_IDENTITIES = {
    OFFICIAL_ONLY_SPEC: (OFFICIAL_ONLY_IDENTITY, ["DACOfficialDonors"]),
    BOTH_TOTALS_SPEC: (BOTH_TOTALS_IDENTITY, ["DACOfficialDonors", "DACPrivateDonors"]),
}


def _whole_totals_parquet_bytes(donors: Sequence[str], *, kinds: bool) -> bytes:
    """Each of `donors`' 2021 aid to Ethiopia as dfo-apis renders it, with the party kinds
    (`whole_total` for both) when `kinds`."""
    count = len(donors)
    columns: dict[str, Any] = {
        "stat_var": pa.array(["dcid:Oda"] * count, type=pa.string()),
        "financingProvider": pa.array(list(donors), type=pa.string()),
        "financingRecipient": pa.array(["country/ETH"] * count, type=pa.string()),
        "time_period": pa.array(["2021"] * count, type=pa.string()),
        "value": pa.array([WHOLE_TOTALS[donor] for donor in donors], type=pa.float64()),
        "unit": pa.array(["USD"] * count, type=pa.string()),
    }
    if kinds:
        columns["financingProvider_party_kind"] = pa.array(["whole_total"] * count)
        columns["financingRecipient_party_kind"] = pa.array(["party"] * count)
    buf = io.BytesIO()
    pq.write_table(pa.table(columns), buf)
    return buf.getvalue()


_WHOLE_TOTALS_STAMP = "ens_" + "1" * 16


def _whole_totals_mint_response(spec_id: str, *, kinds: bool) -> dict[str, Any]:
    """dfo-apis' manifest for `spec_id`: one `content_version` for both renderings, the
    request's options as the artifact's `rendering`, and on the kind file the stamp of
    the code index its kind cells come from."""
    identity, donors = _WHOLE_TOTAL_IDENTITIES[spec_id]
    parquet = _whole_totals_parquet_bytes(donors, kinds=kinds)
    rendering: dict[str, Any] = {"bundle": False, "format": "parquet", "layout": "frame"}
    if kinds:
        rendering["columns"] = ["party_kind"]
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
        "content_version": version_of(_whole_totals_parquet_bytes(donors, kinds=False)),
        "coverage": {
            "rows": len(donors),
            "dimensions": [
                {"label": "financingProvider", "count": len(donors), "axis": "origin"},
                {"label": "financingRecipient", "count": 1, "axis": "destination"},
                {"label": "years", "count": 1},
            ],
        },
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": f"sha256:{hashlib.sha256(parquet).hexdigest()}",
                "rendering": rendering,
                "bytes": len(parquet),
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-10-01T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
                **({"enrichment_stamp": _WHOLE_TOTALS_STAMP} if kinds else {}),
            }
        ],
        "statistics": [
            {
                "stat_var": "dcid:Oda",
                "entities": [
                    {"role": "financingProvider", "ref": donor},
                    {"role": "financingRecipient", "ref": "country/ETH"},
                ],
                "facet": {"unit": "USD"},
                "observations": 1,
                "first": {"value": WHOLE_TOTALS[donor], "period": "2021"},
                "latest": {"value": WHOLE_TOTALS[donor], "period": "2021"},
                "minimum": {"value": WHOLE_TOTALS[donor], "period": "2021"},
                "maximum": {"value": WHOLE_TOTALS[donor], "period": "2021"},
            }
            for donor in donors
        ],
    }


def _serve_whole_totals(
    monkeypatch: pytest.MonkeyPatch, stats_registry: _StatsRegistry
) -> list[dict[str, Any]]:
    """Answer the whole-total specs' mints with the kind columns when `options.columns`
    asks for them, as dfo-apis does. Returns the options of every mint."""
    renderings = {
        (spec_id, kinds): _register_statistics(
            stats_registry, _whole_totals_mint_response(spec_id, kinds=kinds)
        )
        for spec_id in _WHOLE_TOTAL_IDENTITIES
        for kinds in (False, True)
    }
    fetchable = {
        f"sha256:{hashlib.sha256(data).hexdigest()}": data
        for spec_id, (_identity, donors) in _WHOLE_TOTAL_IDENTITIES.items()
        for data in (_whole_totals_parquet_bytes(donors, kinds=True),)
    }
    minted: list[dict[str, Any]] = []

    async def fake_fetch(*, artifact_digest: str, **_: Any) -> bytes:
        return fetchable[artifact_digest]

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            body = json.loads(request.content)
            minted.append(body["options"])
            spec_id = body["input"]["spec_id"]
            return httpx2.Response(200, json=renderings[spec_id, "columns" in body["options"]])
        if request.url.path.endswith("/options"):
            # The spec is not in the definition store, so `get_data` reads its identity here.
            spec_id = request.url.path.split("/")[-2]
            return httpx2.Response(200, json=renderings[spec_id, False])
        raise AssertionError(f"unexpected call to {request.url.path}")

    serve_canonical_frames(monkeypatch, fake_fetch)
    client_holder.set_client(data_mcp_client(mint_handler))
    return minted


def _flow_surface(app: Any, mounted_mcp_transport: Callable[..., Any]) -> McpSurface:
    return _gdp_surface(app, mounted_mcp_transport)


async def test_get_data_with_default_flags_on_a_spec_binding_the_official_total_returns_one_row(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    _stats_registry: _StatsRegistry,
) -> None:
    """The OECD's figure for every official donor, read by an outside model that sets no
    flag, is the one row it sums."""
    minted = _serve_whole_totals(monkeypatch, _stats_registry)
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with running_lifespan(app):
        surface = _flow_surface(app, mounted_mcp_transport)
        read = await surface.call_tool(tool=_GET_DATA, arguments={"spec_id": OFFICIAL_ONLY_SPEC})
        assert not read.is_error, read.text
        held = await get_frame_store().get(
            FrameRef.model_validate((read.structured_content or {})["frame_ref"])
        )

    assert held is not None
    assert "columns" not in minted[0]
    table = held[0].table
    assert table.column("financingProvider").to_pylist() == ["DACOfficialDonors"]
    assert table.column("value").to_pylist() == [OFFICIAL_DONORS_2021]


async def test_an_outside_client_sums_a_pair_of_whole_totals_after_asking_for_place_types(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    _stats_registry: _StatsRegistry,
) -> None:
    """`search` for "all donors including private" returns one spec binding both totals.
    `get_data` with default flags gives two rows, and no kind column says they are
    whole totals, so `aggregate` refuses `place_levels_mixed`. After a retry with
    `place_types`, both cells are `whole_total` and the sum is both populations."""
    minted = _serve_whole_totals(monkeypatch, _stats_registry)
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async def total(surface: McpSurface, frame_ref: dict[str, Any]) -> Any:
        return await surface.call_tool(
            tool=ToolId(server="analysis-mcp", name="aggregate"),
            arguments={
                "frame_ref": frame_ref,
                "aggregation": "sum",
                "group_by": ["time_period"],
            },
        )

    async with running_lifespan(app):
        surface = _flow_surface(app, mounted_mcp_transport)
        plain = await surface.call_tool(tool=_GET_DATA, arguments={"spec_id": BOTH_TOTALS_SPEC})
        assert not plain.is_error, plain.text
        plain_content: dict[str, Any] = plain.structured_content or {}
        plain_held = await get_frame_store().get(
            FrameRef.model_validate(plain_content["frame_ref"])
        )
        refused = await total(surface, plain_content["frame_ref"])
        typed = await surface.call_tool(
            tool=_GET_DATA, arguments={"spec_id": BOTH_TOTALS_SPEC, "place_types": True}
        )
        assert not typed.is_error, typed.text
        typed_content: dict[str, Any] = typed.structured_content or {}
        summed = await total(surface, typed_content["frame_ref"])
        assert not summed.is_error, summed.text
        sum_pair = summed.structured_content or {}
        summed_held = await get_frame_store().get(
            FrameRef(spec_id=sum_pair["spec_id"], content_version=sum_pair["content_version"])
        )

    assert plain_held is not None
    assert plain_held[0].table.column("value").to_pylist() == list(WHOLE_TOTALS.values())
    assert sum(plain_held[0].table.column("value").to_pylist()) == pytest.approx(4_578_400_000.0)
    assert refused.is_error
    assert (refused.structured_content or {})["code"] == "place_levels_mixed"
    assert [options.get("columns") for options in minted] == [None, ["party_kind"]]
    # The frame holding the kinds is named apart from the plain one, under its file's stamp.
    assert typed_content["frame_ref"] != plain_content["frame_ref"]
    assert typed_content["frame_ref"] == frame_ref_for(
        spec_id=BOTH_TOTALS_SPEC,
        content_version=typed_content["content_version"],
        columns=("party_kind",),
        enrichment_stamp=_WHOLE_TOTALS_STAMP,
    ).model_dump(mode="json")
    assert summed_held is not None
    assert summed_held[0].table.column("value").to_pylist() == pytest.approx([4_578_400_000.0])
