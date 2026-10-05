"""A plan whose terminal step is the retrieval itself, run end to end.

A question that asks to see a series for named entities is answered by the
retrieved frame, so the plan states one dataset requirement and the
mechanical translation names the retrieval as its terminal. The retrieval's
frame is therefore the key `harness.studio_translate._RecordingSurface`
wrote its manifest under, and `harness.graph._link_evidence_to_terminal_frames`
links that frame to itself. `_upstream_retrievals` returns a retrieval
step as its own source. The card built from the run's own terminal frame
carries the retrieval's real citation, as it does for a transform's `ops_`
frame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from common.materialisation import FrameRef
from compiler.compile import compile_plan
from compiler.surface import ToolSchema
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from harness.conversation import current_conversation_id
from harness.graph import _link_evidence_to_terminal_frames
from harness.mcp_client import McpExecutor
from harness.provenance import ProvenanceStore
from harness.settings import Settings
from harness.studio_translate import _RecordingSurface
from planner.operations import RetrieveOperation, Translation
from planner.plan import AnswerMode, DatasetRequirement, Decision, Plan, PlanStatus
from planner.workflow import SelectedSource

from ..resolved_revision import resolved_revision
from .conftest import CONTENT_VERSION, SPEC_KENYA, ScriptedSurface, ok

ManifestFactory = Callable[..., dict[str, Any]]
SourceFactory = Callable[..., SelectedSource]


def _retrieval_terminal_plan(
    *,
    decisions: list[Decision],
    requirement: DatasetRequirement,
    source: SelectedSource,
    schemas: list[ToolSchema],
) -> Plan:
    """Compile and approve a plan whose only requirement the retrieval answers itself."""
    draft = Plan(
        question="What has Kenya paid to service its debt each year?",
        answer_mode=AnswerMode.SERIES,
        decisions=decisions,
        evidence_requirements=[requirement],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(
                id=requirement.id, requirement_id=requirement.id, spec_id=source.spec_id
            )
        ],
        terminal=[requirement.id],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas={schema.name: schema for schema in schemas},
        discovered={requirement.id: source},
    )
    assert compiled.workflow is not None, compiled.issues
    (step,) = compiled.workflow.steps
    assert step.tool.name == "get_data"
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve()


async def test_a_retrieval_terminal_plan_links_its_own_frame(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: SourceFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    store = ProvenanceStore()
    surface = _RecordingSurface(
        ScriptedSurface(schemas=real_schemas, responses={"get_data": [ok(manifest())]}),
        provenance=store,
    )
    plan = _retrieval_terminal_plan(
        decisions=[decision()], requirement=requirement(), source=source(), schemas=real_schemas
    )

    _finished, outcome = await execute_approved_plan(plan, surface=surface)

    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["annual_debt_service"]
    assert frame_ref == FrameRef(spec_id=SPEC_KENYA, content_version=CONTENT_VERSION)
    assert (
        store.get(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
        )
        is not None
    )


async def test_a_retrieval_terminal_plan_assembles_a_card_from_its_own_source(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: SourceFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    store = ProvenanceStore()
    cited = manifest(
        citation={
            "text": "World Bank, International Debt Statistics",
            "sources": [{"ref": "worldbank", "label": "World Bank"}],
        }
    )
    surface = _RecordingSurface(
        ScriptedSurface(schemas=real_schemas, responses={"get_data": [ok(cited)]}),
        provenance=store,
    )
    plan = _retrieval_terminal_plan(
        decisions=[decision()], requirement=requirement(), source=source(), schemas=real_schemas
    )
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["annual_debt_service"].model_dump()

    executor = McpExecutor(settings=Settings(), provenance=store)
    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1},
        envelope=resolved_revision("line_chart", frame_ref=frame_ref),
    )

    assert component is not None
    assert component["provenance"]["citation"] == "World Bank, International Debt Statistics"
    assert component["provenance"]["sources"] == [
        {"id": "worldbank", "name": "World Bank", "licence": "not stated by the platform"}
    ]


async def test_a_retrieval_terminal_card_counts_the_stat_vars_its_frame_holds(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: SourceFactory,
    manifest_for: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    # The manifest resolves three StatVars and the frame `get_data` read holds
    # two of them, which is the count the card states.
    family = source(stat_vars=("Debt_0", "Debt_1", "Debt_2"))
    store = ProvenanceStore()
    surface = _RecordingSurface(
        ScriptedSurface(
            schemas=real_schemas,
            responses={"get_data": [ok(manifest_for(family, stat_var_count=2))]},
        ),
        provenance=store,
    )
    plan = _retrieval_terminal_plan(
        decisions=[decision()], requirement=requirement(), source=family, schemas=real_schemas
    )
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    assert isinstance(outcome, Accepted), outcome
    assert [record.stat_var_count for record in outcome.steps] == [2]
    assert plan.workflow is not None
    _link_evidence_to_terminal_frames(surface, outcome, workflow=plan.workflow)
    frame_ref = outcome.results["annual_debt_service"].model_dump()

    executor = McpExecutor(settings=Settings(), provenance=store)
    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1},
        envelope=resolved_revision("line_chart", frame_ref=frame_ref),
    )

    assert component is not None
    assert component["fact_summary"]["attribution"]["indicator_count"] == 2
