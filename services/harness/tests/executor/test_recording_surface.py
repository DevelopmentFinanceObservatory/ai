"""Tests for `harness.studio_translate._RecordingSurface`: the approved-plan
path's own writer into `harness.provenance.ProvenanceStore` (shared with
`harness.mcp_client.McpExecutor`'s direct-fetch writer, so a component
assembled from a plan step's frame carries real provenance), and the two
gaps in its own activity recording -- a call whose transport never comes
back leaves no record at all, and a call that succeeds but is refused
afterward by the executor's own post-call checks (shape, evidence, frame)
stays recorded as "complete" forever.

The multi-input tests use `join_plan` and `two_input_plan` (a compare or a
share_of) in place of `approved_plan`. Their one terminal frame rests on two
retrievals, so it gets a two-input lineage, and a card built from it names
both sources. `test_a_retrieval_terminal_plan_links_its_own_frame` in
`tests/executor/test_retrieval_terminal_surface.py` covers the plan whose
terminal is the retrieval itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from compiler.compile import compile_plan
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from data_mcp.frame_materialisation import frame_ref_for
from executor.outcomes import Accepted, RefusalKind, Refused
from executor.run import execute_approved_plan
from harness import graph
from harness.conversation import current_conversation_id
from harness.graph import _failed_step, _link_evidence_to_terminal_frames, _mark_masked_step_failed
from harness.mcp_client import McpExecutor, _wire_provenance
from harness.provenance import ProvenanceInput, ProvenanceStore
from harness.scripted import SCRIPTED_SPEC
from harness.scripted import ScriptedSurface as ScriptedToolSurface
from harness.settings import Settings
from harness.studio_translate import _RecordingSurface, _refusal_detail
from planner.plan import (
    DatasetRequirement,
    Decision,
    DecisionFixes,
    FilterRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
)
from planner.records import PlaceLevel, PlaceType
from planner.workflow import OpenAxis, OpenEntityAxis, SelectedSource, ToolId
from translator.mechanical import build_translation

from ..resolved_revision import resolved_revision
from .conftest import (
    CONTENT_VERSION,
    RESULT_FRAME,
    SPEC_GDP,
    SPEC_KENYA,
    ScriptedSurface,
    _statistics_artifact,
    ok,
)

ManifestFactory = Callable[..., dict[str, Any]]
PlanFactory = Callable[..., Plan]


def _today_utc() -> str:
    """Today's date in UTC, the zone `_wire_provenance`'s fallback
    `last_refreshed` is computed in."""
    return datetime.now(UTC).date().isoformat()


# ---------------------------------------------------------------------------
# The planned path's manifest reaches component assembly
# ---------------------------------------------------------------------------


async def test_a_component_from_a_planned_run_carries_its_retrievals_real_provenance(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    store = ProvenanceStore()
    rich_manifest = manifest(
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank", "license": "CC BY 4.0"}],
        },
        warnings=[
            {"code": "coverage_gap", "severity": "warn", "message": "2023's figure is provisional"}
        ],
        artifacts=_artifacts("2019-05-01T00:00:00+00:00"),
    )
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(rich_manifest)], "aggregate": [ok(RESULT_FRAME)]},
    )
    surface = _RecordingSurface(scripted, provenance=store)

    plan = approved_plan()
    _finished, outcome = await execute_approved_plan(plan, surface=surface)

    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    # What `harness.graph`'s `execute` node does on an `Accepted` run,
    # before routing on to `compose` -- see this module's docstring for why
    # it is needed here too.
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["totals"]

    # A second, independent `McpExecutor`, standing in for the one
    # `harness.server.create_app` builds -- sharing only `store`, exactly
    # what `create_app` wires `McpExecutor` and `_RecordingSurface` to share
    # (`provenance=provenance` on both).
    executor = McpExecutor(settings=Settings(), provenance=store)
    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1},
        envelope=resolved_revision("stat_card", frame_ref=frame_ref.model_dump()),
    )

    assert component is not None
    result = component["provenance"]
    assert result["citation"] == "World Bank, International Debt Statistics"
    assert result["sources"] == [{"id": "worldbank", "name": "World Bank", "licence": "CC BY 4.0"}]
    assert result["warnings"] == [
        {"code": "coverage_gap", "severity": "warn", "message": "2023's figure is provisional"}
    ]
    assert result["freshness"]["last_refreshed"] == "2019-05-01"
    assert result["freshness"]["last_refreshed"] != _today_utc()


async def test_a_retrieval_held_under_its_own_content_version_still_carries_its_provenance(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """A frame minted with extra columns is held under a `content_version` of its own,
    which `get_data` states as `frame_ref` beside the manifest. The retrieval's
    manifest is recorded under that reference, the one the run links the terminal
    frame to, so the card keeps its source."""
    typed = {"spec_id": SPEC_KENYA, "content_version": "cv_" + "9" * 32}
    stated = manifest(
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank"}],
        },
        frame_ref=typed,
    )
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(stated)], "aggregate": [ok(RESULT_FRAME)]},
    )
    store = ProvenanceStore()
    surface = _RecordingSurface(scripted, provenance=store)
    plan = approved_plan()

    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    component = await _assemble(store, outcome.results["totals"].model_dump())

    assert scripted.calls[1][1]["frame_ref"] == typed
    assert surface.retrieval_manifest(frame_ref=typed) == stated
    assert component is not None
    assert component["provenance"]["citation"] == "World Bank, International Debt Statistics"


def test_create_app_shares_one_provenance_store_between_executor_and_surface_recorder() -> None:
    """`harness.server.create_app` must build ONE `ProvenanceStore` and hand
    it to both `McpExecutor` and `_RecordingSurface` -- two independent
    stores would leave a planned run's manifests unreadable to component
    assembly even with both writers wired."""
    from harness.server import create_app  # noqa: PLC0415 -- isolate this suite's own import cost

    app = create_app(settings=Settings())

    executor = app.state.studio.recorder._executor
    assert isinstance(executor, McpExecutor)
    surface_recorder = app.state.studio.surface_recorder
    assert isinstance(surface_recorder, _RecordingSurface)
    assert surface_recorder._provenance is executor._provenance


# ---------------------------------------------------------------------------
# Per-frame attribution: each terminal frame carries the sources it rests on
# ---------------------------------------------------------------------------


async def test_two_independent_retrievals_each_attribute_to_their_own_frame(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    population_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    """Attribution is per frame. `two_source_plan` runs two independent
    retrieve-then-aggregate branches, one dataset behind each, so the run
    holds two `EvidenceRecord`s while each terminal frame is fed by exactly
    one retrieval of its own. `_link_evidence_to_terminal_frames` links each
    terminal frame to its own retrieval, and `McpExecutor._assemble_component`
    builds each card from that retrieval's real source and date.
    """
    debt_manifest = manifest_for(
        source(),
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank"}],
        },
        artifacts=[
            {
                "artifact_digest": "sha256:" + "1" * 64,
                "rendering": {"bundle": False, "format": "parquet"},
                "bytes": 1024,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2019-05-01T00:00:00+00:00",
                "tier": "working",
            },
            _statistics_artifact(),
        ],
    )
    population_manifest = manifest_for(
        population_source(),
        citation={
            "text": "UN Population Division, World Population Prospects",
            "sources": [{"ref": "un-pop", "label": "UN Population Division"}],
        },
        artifacts=[
            {
                "artifact_digest": "sha256:" + "2" * 64,
                "rendering": {"bundle": False, "format": "parquet"},
                "bytes": 2048,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2021-11-20T00:00:00+00:00",
                "tier": "working",
            },
            _statistics_artifact(),
        ],
    )
    debt_totals = {
        **RESULT_FRAME,
        "spec_id": "ops_" + "6" * 32,
        "content_version": "ocv_" + "7" * 32,
    }
    population_totals = {
        **RESULT_FRAME,
        "spec_id": "ops_" + "8" * 32,
        "content_version": "ocv_" + "9" * 32,
    }
    store = ProvenanceStore()
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(debt_manifest), ok(population_manifest)],
            "aggregate": [ok(debt_totals), ok(population_totals)],
        },
    )
    surface = _RecordingSurface(scripted, provenance=store)

    plan = two_source_plan()
    _finished, outcome = await execute_approved_plan(plan, surface=surface)

    assert isinstance(outcome, Accepted), outcome
    # The exact shape of the bug: more than one retrieval in the run.
    assert len(outcome.evidence) == 2
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)

    executor = McpExecutor(settings=Settings(), provenance=store)
    debt_ref = outcome.results["totals"].model_dump()
    population_ref = outcome.results["population_totals_op"].model_dump()
    debt_component = await executor._assemble_component(
        reply={"component_id": "cmp_debt", "revision": 1},
        envelope=resolved_revision("stat_card", frame_ref=debt_ref),
    )
    population_component = await executor._assemble_component(
        reply={"component_id": "cmp_population", "revision": 1},
        envelope=resolved_revision("stat_card", frame_ref=population_ref),
    )

    assert debt_component is not None
    assert population_component is not None
    debt_provenance = debt_component["provenance"]
    population_provenance = population_component["provenance"]
    assert debt_provenance["citation"] == "World Bank, International Debt Statistics"
    assert debt_provenance["freshness"]["last_refreshed"] == "2019-05-01"
    assert population_provenance["citation"] == (
        "UN Population Division, World Population Prospects"
    )
    assert population_provenance["freshness"]["last_refreshed"] == "2021-11-20"
    for provenance in (debt_provenance, population_provenance):
        assert provenance["citation"] != "unknown"
        assert provenance["sources"][0]["id"] != "unknown"
        assert provenance["freshness"]["last_refreshed"] != _today_utc()


async def test_a_frame_with_no_linked_manifest_is_withheld_not_fabricated() -> None:
    """`_assemble_component` withholds the component when the frame has no
    lineage in the store, the state of a frame nothing was linked to (a
    retrieval it rests on has no recorded manifest, or the store evicted it).
    An empty `ProvenanceStore` reproduces that "no lineage found" state
    directly."""
    frame_ref = {"spec_id": "ops_" + "f" * 32, "content_version": "ocv_" + "f" * 32}
    executor = McpExecutor(settings=Settings(), provenance=ProvenanceStore())

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1},
        envelope=resolved_revision("stat_card", frame_ref=frame_ref),
    )

    assert component is None


async def test_a_single_source_chain_states_exactly_its_one_retrievals_provenance(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """A frame resting on one retrieval lists itself as the one input and
    carries its retrieval's sources and warnings verbatim, with no `spec_id`
    tag on any warning. That is the provenance the manifest alone gives."""
    stated = manifest(
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank"}],
        },
        coverage={"rows": 10, "dimensions": [], "window": {"start_year": 2015, "end_year": 2024}},
        warnings=[
            {"code": "coverage_gap", "severity": "warn", "message": "2023's figure is provisional"}
        ],
        artifacts=_artifacts("2019-05-01T00:00:00+00:00"),
    )
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(stated)], "aggregate": [ok(RESULT_FRAME)]},
    )
    store = ProvenanceStore()
    surface = _RecordingSurface(scripted, provenance=store)
    plan = approved_plan()
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["totals"].model_dump()

    component = await _assemble(store, frame_ref)

    assert component is not None
    provenance = dict(component["provenance"])
    del provenance["generated_at"]
    assert provenance == {
        "inputs": [frame_ref],
        "sources": [
            {"id": "worldbank", "name": "World Bank", "licence": "not stated by the platform"}
        ],
        "freshness": {
            "data_through": "2024",
            "last_refreshed": "2019-05-01",
            "partial_period": False,
        },
        "warnings": [
            {"code": "coverage_gap", "severity": "warn", "message": "2023's figure is provisional"}
        ],
        "citation": "World Bank, International Debt Statistics",
    }


async def test_a_card_counts_the_stat_vars_of_the_frame_it_draws_from(
    approved_plan: PlanFactory,
    source: Callable[..., SelectedSource],
    manifest_for: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """The retrieval sums a family of three StatVars, so its frame and the
    aggregate over it each declare one StatVar, the family ref. A card over
    either frame counts the three the sum added, which the summed retrieval's
    manifest states and the lineage link carries to the aggregate's frame."""
    family = source(
        stat_vars=("Debt_0", "Debt_1", "Debt_2"),
        open_axes=(OpenAxis(property="crsPurpose", summable=True),),
    )
    summed = manifest_for(family) | {"stat_var_count": 1, "summed_stat_var_count": 3}
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(summed)], "aggregate": [ok(RESULT_FRAME)]},
    )
    store = ProvenanceStore()
    surface = _RecordingSurface(scripted, provenance=store)
    plan = approved_plan(source=family)
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    assert [record.stat_var_count for record in outcome.steps] == [1, 1]
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)

    over_aggregate = await _assemble(store, outcome.results["totals"].model_dump())
    over_retrieval = await _assemble(store, outcome.evidence[0].frame_ref.model_dump())

    assert over_aggregate is not None
    assert over_retrieval is not None
    assert over_aggregate["fact_summary"]["attribution"]["indicator_count"] == 3
    assert over_retrieval["fact_summary"]["attribution"]["indicator_count"] == 3


async def test_a_single_retrieval_with_no_creation_time_reads_as_refreshed_today(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """A frame resting on one retrieval whose manifest states no `created_at`
    reads `last_refreshed` as today's date. A frame resting on several
    retrievals reads it as unknown when any of them states none."""
    stated = manifest(
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank"}],
        },
        coverage={"rows": 10, "dimensions": [], "window": {"start_year": 2015, "end_year": 2024}},
        artifacts=_artifacts(None),
    )
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(stated)], "aggregate": [ok(RESULT_FRAME)]},
    )
    store = ProvenanceStore()
    surface = _RecordingSurface(scripted, provenance=store)
    plan = approved_plan()
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["totals"].model_dump()

    component = await _assemble(store, frame_ref)

    assert component is not None
    assert component["provenance"]["freshness"] == {
        "data_through": "2024",
        "last_refreshed": _today_utc(),
        "partial_period": False,
    }


# ---------------------------------------------------------------------------
# A terminal frame that rests on more than one retrieval
# ---------------------------------------------------------------------------

_GDP_CONTENT_VERSION = "cv_" + "4" * 32
_DEBT_FRAME = {"spec_id": SPEC_KENYA, "content_version": CONTENT_VERSION}
_GDP_FRAME = {"spec_id": SPEC_GDP, "content_version": _GDP_CONTENT_VERSION}
# What each terminal tool declares about its frame, as the plan is compiled to
# expect it: a join keeps the two inputs' shared `destination` column and both
# their stat vars; a compare or share_of yields one series with no role column.
_TERMINAL_FRAMES = {
    "join": {**RESULT_FRAME, "role_keys": ["destination"], "stat_var_count": 2},
    "compare": RESULT_FRAME,
    "share_of": RESULT_FRAME,
}


def _artifacts(created_at: str | None) -> list[dict[str, Any]]:
    # Always carries its own statistics artifact -- `manifest`/`manifest_for`'s own default
    # `series_count` (1) still applies to a caller passing `artifacts=_artifacts(...)`, and
    # `find_statistics_artifact` needs an entry to find, whatever this frame artifact's own
    # creation time says.
    frame_artifact = (
        []
        if created_at is None
        else [
            {
                "artifact_digest": "sha256:" + "0" * 64,
                "rendering": {"bundle": False, "format": "parquet"},
                "bytes": 1024,
                "media_type": "application/vnd.apache.parquet",
                "created_at": created_at,
                "tier": "working",
            }
        ]
    )
    return [*frame_artifact, _statistics_artifact()]


def _stated(
    manifest_for: Callable[..., dict[str, Any]],
    retrieved: Any,
    *,
    text: str,
    ref: str,
    label: str,
    created_at: str | None,
    end_year: int | None,
    licence: str | None = None,
    content_version: str = CONTENT_VERSION,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """A `get_data` manifest for `retrieved` that states its own source, freshness and coverage.

    The source carries a `license` only when `licence` is given."""
    window = {"start_year": 2015, "end_year": end_year} if end_year is not None else None
    return manifest_for(
        retrieved,
        citation={
            "text": text,
            "sources": [{"ref": ref, "label": label, **({"license": licence} if licence else {})}],
        },
        coverage={"rows": 10, "dimensions": [], **({"window": window} if window else {})},
        artifacts=_artifacts(created_at),
        warnings=warnings or [],
        content_version=content_version,
    )


@dataclass
class _TwoInputRun:
    """A two-retrieval plan run to `Accepted` through a `_RecordingSurface`, not yet linked."""

    plan: Plan
    store: ProvenanceStore
    surface: _RecordingSurface
    scripted: ScriptedSurface
    outcome: Accepted

    @property
    def workflow(self) -> Any:
        assert self.plan.workflow is not None
        return self.plan.workflow

    @property
    def terminal_frame(self) -> dict[str, str]:
        return self.outcome.results["combined"].model_dump()

    def link(self) -> None:
        _link_evidence_to_terminal_frames(self.surface, self.outcome, workflow=self.workflow)


async def _run_two_input_plan(
    plan: Plan,
    *,
    schemas: list[ToolSchema],
    debt: dict[str, Any],
    gdp: dict[str, Any],
    terminal_tool: str = "join",
    delays: dict[str, float] | None = None,
) -> _TwoInputRun:
    store = ProvenanceStore()
    scripted = ScriptedSurface(
        schemas=schemas,
        responses={
            "get_data": [ok(debt), ok(gdp)],
            terminal_tool: [ok(_TERMINAL_FRAMES[terminal_tool])],
        },
        delays=delays,
    )
    surface = _RecordingSurface(scripted, provenance=store)
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    return _TwoInputRun(plan=plan, store=store, surface=surface, scripted=scripted, outcome=outcome)


async def _assemble(
    store: ProvenanceStore, frame_ref: dict[str, str], *, facts: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    executor = McpExecutor(settings=Settings(), provenance=store)
    return await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts or {}},
        envelope=resolved_revision("stat_card", frame_ref=frame_ref),
    )


async def test_a_join_of_two_different_source_retrievals_states_both(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    """A join reads two retrievals from two publishers. Its card cites both
    sources and both citations, is as fresh and as current as its stalest
    input, lists both retrievals as inputs, and attributes each warning to
    the retrieval it came from."""
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
        licence="OECD Terms and Conditions",
        warnings=[{"code": "coverage_gap", "severity": "warn", "message": "2023 is provisional"}],
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
        warnings=[{"code": "estimate", "severity": "info", "message": "2022 is an estimate"}],
    )
    run = await _run_two_input_plan(join_plan(), schemas=real_schemas, debt=debt, gdp=gdp)
    run.link()

    drawn = {
        "series": [
            {"id": "debt", "coordinates": {"stat_var": "dcid:DebtServicePaid"}},
            {"id": "gdp", "coordinates": {"stat_var": "dcid:Gdp"}},
        ]
    }
    component = await _assemble(run.store, run.terminal_frame, facts=drawn)

    assert component is not None
    provenance = component["provenance"]
    assert provenance["inputs"] == [run.terminal_frame, _DEBT_FRAME, _GDP_FRAME]
    assert provenance["sources"] == [
        {"id": "worldbank", "name": "World Bank", "licence": "OECD Terms and Conditions"},
        {"id": "imf", "name": "IMF", "licence": "not stated by the platform"},
    ]
    assert provenance["citation"] == (
        "World Bank, International Debt Statistics; IMF, World Economic Outlook"
    )
    assert provenance["freshness"] == {
        "data_through": "2022",
        "last_refreshed": "2019-05-01",
        "partial_period": False,
    }
    assert provenance["warnings"] == [
        {
            "code": "coverage_gap",
            "severity": "warn",
            "message": "2023 is provisional",
            "spec_id": SPEC_KENYA,
        },
        {
            "code": "estimate",
            "severity": "info",
            "message": "2022 is an estimate",
            "spec_id": SPEC_GDP,
        },
    ]
    assert component["fact_summary"]["attribution"]["source_count"] == 2
    assert {row["id"] for row in component["fact_summary"]["indicators"]} == {
        "dcid:DebtServicePaid",
        "dcid:Gdp",
    }
    assert component["fact_summary"]["attribution"]["indicator_count"] == 2


async def test_a_join_of_two_retrievals_from_one_source_cites_it_once(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    shared = {
        "text": "World Bank, World Development Indicators",
        "ref": "worldbank",
        "label": "World Bank",
    }
    debt = _stated(
        manifest_for, source(), created_at="2021-01-01T00:00:00+00:00", end_year=2024, **shared
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        created_at="2021-01-01T00:00:00+00:00",
        end_year=2024,
        content_version=_GDP_CONTENT_VERSION,
        **shared,
    )
    run = await _run_two_input_plan(join_plan(), schemas=real_schemas, debt=debt, gdp=gdp)
    run.link()

    component = await _assemble(run.store, run.terminal_frame)

    assert component is not None
    provenance = component["provenance"]
    assert provenance["sources"] == [
        {"id": "worldbank", "name": "World Bank", "licence": "not stated by the platform"}
    ]
    assert provenance["citation"] == "World Bank, World Development Indicators"
    assert provenance["inputs"] == [run.terminal_frame, _DEBT_FRAME, _GDP_FRAME]
    assert component["fact_summary"]["attribution"]["source_count"] == 1


async def test_a_join_input_that_states_no_creation_date_or_end_year_leaves_them_unknown(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    """One input's undated manifest leaves the join's `last_refreshed`
    unknown, and one input's unstated coverage leaves its `data_through`
    unknown, whatever the other input states."""
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at=None,
        end_year=None,
        content_version=_GDP_CONTENT_VERSION,
    )
    run = await _run_two_input_plan(join_plan(), schemas=real_schemas, debt=debt, gdp=gdp)
    run.link()

    component = await _assemble(run.store, run.terminal_frame)

    assert component is not None
    assert component["provenance"]["freshness"] == {
        "data_through": "",
        "last_refreshed": "",
        "partial_period": False,
    }


async def test_a_frame_missing_one_retrievals_manifest_is_not_linked_and_its_card_is_withheld(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Linking is all-or-nothing. With the GDP retrieval's manifest gone from
    the store, the join frame gets no lineage at all, because the debt
    retrieval alone would state a narrower provenance than the join rests on."""
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
    )
    run = await _run_two_input_plan(join_plan(), schemas=real_schemas, debt=debt, gdp=gdp)
    conversation = current_conversation_id.get()
    del run.store._by_conversation[conversation][(SPEC_GDP, _GDP_CONTENT_VERSION)]

    with caplog.at_level(logging.WARNING, logger="harness.studio_translate"):
        run.link()

    frame = run.terminal_frame
    assert (
        run.store.get(
            conversation=conversation,
            spec_id=frame["spec_id"],
            content_version=frame["content_version"],
        )
        is None
    )
    assert "no lineage written" in caplog.text
    assert await _assemble(run.store, frame) is None


async def test_a_frame_whose_retrieval_has_no_evidence_record_is_not_linked(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
    caplog: pytest.LogCaptureFixture,
) -> None:
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
    )
    run = await _run_two_input_plan(join_plan(), schemas=real_schemas, debt=debt, gdp=gdp)
    partial = run.outcome.model_copy(
        update={"evidence": tuple(r for r in run.outcome.evidence if r.step_id != "gdp")}
    )

    with caplog.at_level(logging.WARNING, logger="harness.graph"):
        _link_evidence_to_terminal_frames(run.surface, partial, workflow=run.workflow)

    assert await _assemble(run.store, run.terminal_frame) is None
    assert "without an evidence record" in caplog.text


class _IteratesAscending(frozenset[str]):
    """A frozenset that iterates in sorted order."""

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(frozenset.__iter__(self)))


class _IteratesDescending(frozenset[str]):
    """A frozenset that iterates in reverse sorted order."""

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(frozenset.__iter__(self), reverse=True))


@pytest.mark.parametrize("ordered_set", [_IteratesAscending, _IteratesDescending])
async def test_the_linked_retrievals_follow_workflow_order_whatever_the_set_iterates_as(
    ordered_set: type[frozenset[str]],
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_upstream_retrievals` returns a frozenset, whose order changes with
    the process hash seed. Both sorted directions are forced here. The two
    retrieval steps are named so that workflow order (`z_debt`, `a_gdp`) is the
    reverse of lexical order, so the linked retrievals come out in workflow
    order only when the sort key is the step's position in the workflow, and the
    citation, sources and inputs cannot depend on the seed."""
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
    )
    run = await _run_two_input_plan(
        join_plan(retrieval_ids=("z_debt", "a_gdp")), schemas=real_schemas, debt=debt, gdp=gdp
    )
    original = graph._upstream_retrievals

    def _forced_order(*args: Any, **kwargs: Any) -> frozenset[str]:
        found = original(*args, **kwargs)
        return ordered_set(found)

    monkeypatch.setattr(graph, "_upstream_retrievals", _forced_order)
    calls: list[dict[str, Any]] = []

    class _Spy:
        def link_evidence(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    spy: Any = _Spy()
    _link_evidence_to_terminal_frames(spy, run.outcome, workflow=run.workflow)

    assert calls == [
        {
            "frame_ref": run.terminal_frame,
            "evidence_frame_refs": [_DEBT_FRAME, _GDP_FRAME],
            "stat_var_count": _TERMINAL_FRAMES["join"]["stat_var_count"],
            "carried_frame_refs": [_DEBT_FRAME, _GDP_FRAME],
        }
    ]


async def test_a_join_whose_two_retrievals_overlap_still_states_both_in_workflow_order(
    join_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    """The two retrievals share a wave and are in flight together. Each writes
    its own manifest under its own frame, and the terminal frame's lineage names
    both, debt first, as it does when they run one after the other."""
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
    )
    run = await _run_two_input_plan(
        join_plan(), schemas=real_schemas, debt=debt, gdp=gdp, delays={"get_data": 0.02}
    )
    run.link()

    component = await _assemble(run.store, run.terminal_frame)

    assert run.scripted.max_concurrency == 2
    assert component is not None
    assert component["provenance"]["inputs"] == [run.terminal_frame, _DEBT_FRAME, _GDP_FRAME]
    assert component["provenance"]["citation"] == (
        "World Bank, International Debt Statistics; IMF, World Economic Outlook"
    )


@pytest.mark.parametrize(
    ("kind", "tool"),
    [("compare", "compare"), ("share_of", "share_of")],
    ids=["compare", "share_of"],
)
async def test_a_compare_or_share_of_over_two_retrievals_states_both(
    kind: str,
    tool: str,
    two_input_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., Any],
    gdp_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    debt = _stated(
        manifest_for,
        source(),
        text="World Bank, International Debt Statistics",
        ref="worldbank",
        label="World Bank",
        created_at="2019-05-01T00:00:00+00:00",
        end_year=2024,
    )
    gdp = _stated(
        manifest_for,
        gdp_source(),
        text="IMF, World Economic Outlook",
        ref="imf",
        label="IMF",
        created_at="2021-11-20T00:00:00+00:00",
        end_year=2022,
        content_version=_GDP_CONTENT_VERSION,
    )
    plan = two_input_plan(kind=kind)
    assert plan.workflow is not None
    assert plan.workflow.steps[-1].tool.name == tool
    run = await _run_two_input_plan(
        plan, schemas=real_schemas, debt=debt, gdp=gdp, terminal_tool=tool
    )
    run.link()

    component = await _assemble(run.store, run.terminal_frame)

    assert component is not None
    provenance = component["provenance"]
    assert [row["id"] for row in provenance["sources"]] == ["worldbank", "imf"]
    assert provenance["inputs"] == [run.terminal_frame, _DEBT_FRAME, _GDP_FRAME]
    assert provenance["freshness"] == {
        "data_through": "2022",
        "last_refreshed": "2019-05-01",
        "partial_period": False,
    }


def _level_filtered_join_plan(
    decision: Callable[..., Decision],
    debt_source: SelectedSource,
    gdp_source: SelectedSource,
    schemas: list[ToolSchema],
) -> Plan:
    """A join of two place-less datasets, each kept to countries."""
    countries = (PlaceLevel(level=PlaceType.COUNTRY),)
    level = decision(
        slot="level",
        description="Which level of place to compare.",
        options=["countries", "continents"],
        chosen="countries",
        rationale="Debt service against GDP reads as a comparison of countries.",
        fixes=DecisionFixes(predicates=countries),
    )
    period = Period(start=2015, end=2024)
    draft = Plan(
        question="How does each country's debt service compare with its GDP?",
        decisions=[level],
        evidence_requirements=[
            DatasetRequirement(
                id="debt",
                decision_ids=["level"],
                measure="external debt service paid",
                across="country",
                period=period,
            ),
            DatasetRequirement(
                id="gdp",
                decision_ids=["level"],
                measure="gross domestic product",
                across="country",
                period=period,
            ),
            FilterRequirement(
                id="debt_countries", decision_ids=["level"], input_id="debt", predicates=countries
            ),
            FilterRequirement(
                id="gdp_countries", decision_ids=["level"], input_id="gdp", predicates=countries
            ),
            JoinRequirement(
                id="combined",
                decision_ids=["level"],
                left_id="debt_countries",
                right_id="gdp_countries",
            ),
        ],
        status=PlanStatus.DRAFT,
    )
    translation = build_translation(draft, {"debt": debt_source.spec_id, "gdp": gdp_source.spec_id})
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas={schema.name: schema for schema in schemas},
        discovered={"debt": debt_source, "gdp": gdp_source},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_a_join_of_two_level_filtered_retrievals_lists_the_pairs_dfo_apis_serves(
    decision: Callable[..., Decision],
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    """Each retrieval reads place types for its level filter, so `get_data` holds its
    frame under a `content_version` of its own (`frame_ref_for`), and the lineage is
    keyed by that frame. dfo-apis minted and serves the manifest's pair, so the card
    lists each retrieval under it."""
    every_country = (OpenEntityAxis(column="observationAbout", member_count=217),)
    debt_source = source(entities=(), open_entity_axes=every_country)
    gdp_source = source(
        spec_id=SPEC_GDP,
        sentence="gross domestic product, 2010 to 2024",
        stat_vars=("Gdp",),
        identity={"stat_var": "Gdp"},
        entities=(),
        open_entity_axes=every_country,
    )
    plan = _level_filtered_join_plan(decision, debt_source, gdp_source, real_schemas)
    by_place = {"role_keys": ["observationAbout"]}
    retrievals = [
        manifest_for(
            side,
            content_version=minted["content_version"],
            frame_ref=frame_ref_for(**minted, columns=("observationAbout_type",)).model_dump(),
            **by_place,
        )
        for side, minted in ((debt_source, _DEBT_FRAME), (gdp_source, _GDP_FRAME))
    ]
    for retrieval in retrievals:
        # Statistics the evidence check reads: one series per place the source holds.
        (series,) = retrieval["statistics"]
        retrieval["statistics"][:] = [
            {**series, "entities": [{"role": "observationAbout", "ref": place}]}
            for place in ("country/KEN", "country/GHA", "africa")
        ]
    scripted = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(retrieval) for retrieval in retrievals],
            "filter": [
                ok({**RESULT_FRAME, **by_place, "spec_id": "ops_" + digit * 32}) for digit in "56"
            ],
            "join": [ok({**RESULT_FRAME, **by_place, "stat_var_count": 2})],
        },
    )
    store = ProvenanceStore()
    surface = _RecordingSurface(scripted, provenance=store)

    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    assert [call[1].get("place_types") for call in scripted.calls if call[0] == "get_data"] == [
        True,
        True,
    ]
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    terminal = outcome.results["combined"].model_dump()
    component = await _assemble(store, terminal)

    assert component is not None
    assert component["provenance"]["inputs"] == [terminal, _DEBT_FRAME, _GDP_FRAME]


# ---------------------------------------------------------------------------
# A failed step is recorded as failed
# ---------------------------------------------------------------------------


async def test_a_transport_failure_is_recorded_as_a_failed_activity(
    real_schemas: list[ToolSchema],
) -> None:
    class _Unreachable:
        async def list_tools(self) -> list[ToolSchema]:
            return list(real_schemas)

        async def call_tool(
            self,
            *,
            tool: ToolId,
            arguments: dict[str, Any],
            on_progress: ProgressCallback | None = None,
        ) -> Any:
            del tool, arguments
            raise ToolSurfaceUnavailableError("data-mcp is unreachable")

    surface = _RecordingSurface(_Unreachable())
    tool = ToolId(server="data-mcp", name="get_data")

    with pytest.raises(ToolSurfaceUnavailableError):
        await surface.call_tool(tool=tool, arguments={"spec_id": SPEC_KENYA})

    records = surface.drain(current_conversation_id.get())
    assert len(records) == 1
    assert records[0].result == {
        "outcome": "refusal",
        "refusal": {
            "code": "transport_failed",
            "retry_mode": "retry_after_delay",
            "fix": None,
            "message": "data-mcp is unreachable",
        },
    }


async def test_a_calls_progress_reaches_the_callback_through_the_recording_surface() -> None:
    report = ProgressReport(progress=3, message="running")
    surface = _RecordingSurface(ScriptedToolSurface(get_data_progress=[report]))
    reports: list[ProgressReport] = []

    await surface.call_tool(
        tool=ToolId(server="data-mcp", name="get_data"),
        arguments={"spec_id": SCRIPTED_SPEC},
        on_progress=reports.append,
    )

    assert reports == [report]


async def test_a_step_refused_after_its_own_call_succeeded_is_amended_to_failed(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """`_check_evidence` (`executor.run._Run`) refuses `REQUIREMENT_UNMET`
    after the retrieval's own call already succeeded -- the same source
    drift `executor/test_run.py::
    test_a_retrieval_reminted_under_another_id_is_still_checked_against_its_source`
    drives at the outcome level, checked here at the Activity feed's own
    record: `_RecordingSurface` alone leaves it "complete" forever
    (`call_tool` records the moment the call returns, before that check
    runs); `harness.graph`'s `execute` node is what corrects it, once it
    has read the run's real outcome (`_mark_masked_step_failed`)."""
    drifted = manifest()
    drifted["definition"]["spec_id"] = "spec_" + "e" * 32
    drifted["definition"]["identity"] = {"stat_var": "SomethingElse"}
    scripted = ScriptedSurface(
        schemas=real_schemas, responses={"get_data": [ok(drifted)], "aggregate": [ok(RESULT_FRAME)]}
    )
    surface = _RecordingSurface(scripted)

    _finished, outcome = await execute_approved_plan(approved_plan(), surface=surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    # `check_evidence` reports the drift as its own `EvidenceIssue`, not
    # just the generic `outcome.detail` -- the amended record below has to
    # carry that specific code, not a fixed one.
    assert outcome.issues, "check_evidence should have reported the spec_id drift"
    # The bad retrieval stops the run before `aggregate` ever runs.
    assert [name for name, _ in scripted.calls] == ["get_data"]

    # Before the amendment: the raw call succeeded, and that is all
    # `_RecordingSurface` knows -- the exact defect.
    assert [record.result["outcome"] for record in surface._records] == ["ok"]

    _mark_masked_step_failed(surface, outcome, tools={"debt": "get_data"})

    records = surface.drain(current_conversation_id.get())
    assert len(records) == 1
    assert records[0].result["outcome"] == "refusal"
    failed = _failed_step(outcome)
    assert records[0].result["refusal"] == {
        "code": failed["code"],
        "retry_mode": "do_not_retry",
        "fix": None,
        "message": failed["message"],
    }
    # The amended detail names the drifted spec_id, not the generic
    # "requirement X is not met by what Y returned" `outcome.detail` alone
    # would have shown before this fix.
    assert failed["code"] == "source_drift"
    assert "spec_" + "e" * 32 in failed["message"]

    # What `_drain_recorder` actually renders onto the wire's `activity.step`
    # `detail` (`studio_translate._refusal_detail`, unchanged by this fix):
    # the real code, not the fixed `post_call_validation_failed` every
    # masked failure used to report regardless of what refused it.
    rendered = _refusal_detail(records[0].result["refusal"])
    assert rendered.startswith("source_drift: ")
    assert "spec_" + "e" * 32 in rendered


@pytest.mark.parametrize(
    ("stated", "wire"),
    [
        ({"license": "  CC BY 4.0 "}, "CC BY 4.0"),
        ({"license": ""}, "not stated by the platform"),
        ({"license": "   "}, "not stated by the platform"),
        ({"license": None}, "not stated by the platform"),
        ({}, "not stated by the platform"),
    ],
)
def test_a_sources_licence_is_what_the_citation_states_when_it_states_one(
    stated: dict[str, Any], wire: str
) -> None:
    manifest = {"citation": {"text": "T", "sources": [{"ref": "a", "label": "A", **stated}]}}
    frame_ref = {"spec_id": "spec_a", "content_version": "v1"}
    lineage = (ProvenanceInput("spec_a", "v1", manifest),)

    provenance = _wire_provenance(
        frame_ref=frame_ref, lineage=lineage, manifest=manifest, platform_facts={}
    )

    assert provenance["sources"] == [{"id": "a", "name": "A", "licence": wire}]
