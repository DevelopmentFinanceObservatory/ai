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
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation import get_store as get_frame_store
from compiler.compile import compile_plan
from compiler.sources import member_digest
from compiler.surface import McpSurface, ToolSurfaceUnavailableError
from data_mcp import client_holder, frame_materialisation
from data_mcp.client import UpstreamClient
from data_mcp.manifest import SeriesStatisticsModel
from data_mcp.series_statistics import series_statistics_table
from data_mcp.settings import Settings as DataMcpSettings
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from harness.provenance import data_artifact
from harness.server import create_app
from harness.settings import ANALYSIS_MCP_MOUNT_PATH, DATA_MCP_MOUNT_PATH, Settings
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
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
)
from planner.workflow import OpenEntityAxis, ResolvedEntity, SelectedSource, ToolId
from translator.mechanical import build_translation

SPEC_ID = "spec_" + "5" * 32
CONTENT_VERSION = "cv_" + "6" * 32
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
            "url": "https://signed.example/should-not-be-fetched",
            "url_expires_at": "2026-08-11T00:15:00+00:00",
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
                "url": "https://signed.example/should-not-be-fetched",
                "url_expires_at": "2026-08-11T00:15:00+00:00",
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
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_an_approved_plan_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
    _stats_registry: _StatsRegistry,
) -> None:
    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _parquet_bytes()

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    mint_response = _register_statistics(_stats_registry, _mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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

        finished, outcome = await execute_approved_plan(approved, surface=surface)

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
SHARE_CONTENT_VERSION = "cv_" + "9" * 32

RANK_SPEC = "spec_" + "e" * 32
RANK_CONTENT_VERSION = "cv_" + "f" * 32
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


def _donor_mint_response(
    *, spec_id: str, content_version: str, values: dict[str, float]
) -> dict[str, Any]:
    parquet = _donor_parquet_bytes(spec_id=spec_id, values=values)
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
                "url": "https://signed.example/should-not-be-fetched",
                "url_expires_at": "2026-08-11T00:15:00+00:00",
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
        identity=_donor_mint_response(
            spec_id=HEALTH_SPEC, content_version=SHARE_CONTENT_VERSION, values=HEALTH_VALUES
        )["definition"]["identity"],
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
        identity=_donor_mint_response(
            spec_id=TOTAL_SPEC, content_version=SHARE_CONTENT_VERSION, values=TOTAL_VALUES
        )["definition"]["identity"],
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
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_the_keyed_share_chain_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> None:
    """`retrieve(health) + retrieve(total) -> share_of(axis=donor) -> rank`, run
    through the real compiler and the real mounted services. France spends
    less on health than Germany would (not modelled here) but has the
    largest *share* of its own total, so a rank that read the raw values
    instead of the shares would put France last, not first."""

    async def fake_fetch(*, subject_id: str, **_kwargs: Any) -> bytes:
        values = HEALTH_VALUES if subject_id == HEALTH_SPEC else TOTAL_VALUES
        return _donor_parquet_bytes(spec_id=subject_id, values=values)

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path != "/api/datasets":
            raise AssertionError(f"unexpected call to {request.url.path}")
        spec_id = json.loads(request.content)["input"]["spec_id"]
        values = HEALTH_VALUES if spec_id == HEALTH_SPEC else TOTAL_VALUES
        response = _register_statistics(
            _stats_registry,
            _donor_mint_response(
                spec_id=spec_id, content_version=SHARE_CONTENT_VERSION, values=values
            ),
        )
        return httpx2.Response(200, json=response)

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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

        finished, outcome = await execute_approved_plan(approved, surface=surface)

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED
    assert [step.step_id for step in outcome.steps] == ["health", "total", "health_share", "ranked"]
    rank_digest = outcome.steps[-1].digest
    assert "#1 donor=country/FRA" in rank_digest
    assert "0.75" in rank_digest


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
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": RANK_SPEC, "identity": RANK_IDENTITY},
        "content_version": RANK_CONTENT_VERSION,
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
                "url": "https://signed.example/should-not-be-fetched",
                "url_expires_at": "2026-08-11T00:15:00+00:00",
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
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_kenyas_rank_series_runs_end_to_end_through_the_mounted_services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
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

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    rank_response = _register_statistics(_stats_registry, _rank_mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=rank_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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

        finished, outcome = await execute_approved_plan(approved, surface=surface)

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
TOP_DONOR_CONTENT_VERSION = "cv_" + "d" * 32

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
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": TOP_DONOR_SPEC,
            "identity": TOP_DONOR_IDENTITY,
        },
        "content_version": TOP_DONOR_CONTENT_VERSION,
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
                "url": "https://signed.example/should-not-be-fetched",
                "url_expires_at": "2026-08-11T00:15:00+00:00",
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
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_a_top_three_rank_keeps_three_donors_and_reports_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> None:
    """A planned `limit=3` reaches `rank` as an argument, so the stored frame
    holds the three highest donors at their measured amounts, and the three
    cut donors reach the digest and the trace as a `beyond_limit` exclusion."""

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _top_donor_parquet_bytes()

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    mint_response = _register_statistics(_stats_registry, _top_donor_mint_response())

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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

        finished, outcome = await execute_approved_plan(approved, surface=surface)
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
GDP_SPEC = "spec_" + "4" * 32
GDP_CONTENT_VERSION = "cv_" + "3" * 32
GDP_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "shape": {"id": "Amount_EconomicActivity_GrossDomesticProduction_Nominal", "kind": "stat_var"},
    "entities": {},
    "requested_window": {"kind": "all"},
    "slots": [],
}
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
                "url": "https://signed.example/should-not-be-fetched",
                "url_expires_at": "2026-10-02T00:15:00+00:00",
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
                across="country",
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
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_a_place_less_rank_keeps_the_countries_before_ranking_them(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
    decision: Callable[..., Decision],
    _stats_registry: _StatsRegistry,
) -> None:
    """The retrieval asks dfo-apis for each place's types, the filter keeps the
    places typed Country, and the top two are the two largest countries: Earth
    and the World Bank's high-income group, both larger, are left out."""

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _gdp_parquet_bytes()

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    mint_response = _register_statistics(_stats_registry, _gdp_mint_response())
    options: list[dict[str, Any]] = []

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            options.append(json.loads(request.content)["options"])
            return httpx2.Response(200, json=mint_response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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

        finished, outcome = await execute_approved_plan(approved, surface=surface)
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
    Returns the artifact fetches as `(content_version, typed)`, in order."""
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

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            typed = "columns" in json.loads(request.content)["options"]
            return httpx2.Response(200, json=renderings[typed])
        raise AssertionError(f"unexpected call to {request.url.path}")

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )
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
        plain_content = plain.structured_content or {}
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
