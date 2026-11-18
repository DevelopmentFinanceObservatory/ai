"""Plans, planners and surfaces the plan-path tests share.

`_planner_plan` is a real, translated plan over the real servers' schemas
(`real_schemas` in each test module), `_scoped_plan` the same plan pinned to
a scope's compiled source, and `_row_id` a scope row's id on its card.
`_Planner` hands back scripted plans, `_fixture_surface` answers the plan's
`get_data` and analysis steps, and `ScriptedExecutor` describes the plan's
final frames.
"""

from __future__ import annotations

from typing import Any

from compiler.surface import FixtureSurface, ProgressCallback, ToolCallResult, ToolSchema
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    PinSource,
    ResolvedBy,
)
from planner.plan import Plan as PlannerPlan
from planner.records import GroupBy, Reduction
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from submission.service import scope_choice_decisions

SPEC_A = "spec_" + "a" * 32


def _get_data_manifest(*, spec_id: str = SPEC_A) -> dict[str, Any]:
    """A minimal valid `get_data` manifest. The hand-built workflows below
    carry no `evidence_bindings`, so only its shape (a definition manifest and
    the `frame_ref` beside it, which `extract_frame_ref` reads) matters."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": "cv_" + "1" * 32,
        "coverage": {"rows": 10, "dimensions": []},
        "citation": {"text": "Test source", "sources": []},
        "trace": {},
        "artifacts": [],
        "frame_ref": {"spec_id": spec_id, "content_version": "cv_" + "1" * 32},
    }


def _analysis_result(*, spec_id: str = "ops_" + "2" * 32) -> dict[str, Any]:
    """A minimal analysis-mcp result: one row of one StatVar, matching a
    `WorkflowStep`'s default `expected_role_keys=()`."""
    return {
        "spec_id": spec_id,
        "content_version": "ocv_" + "3" * 32,
        "rows": 1,
        "role_keys": [],
        "stat_var_count": 1,
    }


def _fixture_surface(
    schemas: list[ToolSchema],
    *,
    analysis_tool: str = "aggregate",
    analysis_result: dict[str, Any] | None = None,
    get_data_result: ToolCallResult | None = None,
    analysis_call_result: ToolCallResult | None = None,
) -> FixtureSurface:
    """A `ToolSurface` over the real schemas, answering `get_data` and one
    analysis tool. A caller wanting a refusal from either passes its own
    `ToolCallResult`."""
    named = {schema.name: schema for schema in schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named[analysis_tool]],
        responses={
            "get_data": get_data_result
            or ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            analysis_tool: analysis_call_result
            or ToolCallResult(
                is_error=False,
                structured_content=analysis_result or _analysis_result(),
                text="computed",
            ),
        },
    )


class ScriptedExecutor:
    """The executor `plan_done` describes the plan's final frames through:
    the scripted results in order, then a described frame with an empty view."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        if len(self.calls) <= len(self.results):
            return self.results[len(self.calls) - 1]
        return {"outcome": "ok", "view": {}}


def _planner_plan(
    *,
    question: str = "kenya debt",
    version: int = 1,
    schemas: list[ToolSchema],
    bind_evidence: bool = False,
    stat_vars: str | None = None,
    terminal: tuple[str, ...] = ("totals",),
) -> PlannerPlan:
    """A real, translated `planner.plan.Plan` -- the shape a `PlannerModel` hands back.

    `stat_vars` is the reading its `get_data` step sends, for a plan whose
    source is a family of StatVars; `None` sends none. `terminal` names the
    steps that answer: `totals` by default, and `get_data` with it for a plan
    that shows the series as well as its total.
    """
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    draft = PlannerPlan(
        question=question,
        decisions=[decision],
        evidence_requirements=[dataset, totals],
        version=version,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=list(terminal),
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A} | ({"stat_vars": stat_vars} if stat_vars else {}),
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        # No evidence bindings by default: most of these tests are about
        # approval/execution wiring, not the evidence check, and carry no
        # real source discovery to check a manifest against. A caller
        # testing a refusal on the bound step (which never reaches the
        # check, since it only runs on success) opts in.
        evidence_bindings=(
            [EvidenceBinding(requirement_id=dataset.id, step_id="get_data", spec_id=SPEC_A)]
            if bind_evidence
            else []
        ),
        terminal_steps=list(terminal),
    )
    return draft.with_translation(translation=translation, workflow=workflow)


class _Planner:
    """A `PlannerModel` returning `{"plan": ..., "summary": ...}` verbatim."""

    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = plans
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._plans) - 1)
        return self._plans[index]


# The window the plan of `_scoped_plan` reads its dataset over.
_PLAN_WINDOW = {"kind": "range", "start_year": 2015, "end_year": 2024}


def _scoped_plan(
    pin: str, scope: dict[str, Any], *, schemas: list[ToolSchema], version: int = 1
) -> PlannerPlan:
    """`_planner_plan` with its dataset pinned to `pin`, citing the scope
    decisions a pin of `scope`'s part states."""
    plan = _planner_plan(schemas=schemas)
    block = {"spec_id": pin, "resolved": {"scope_choices": scope}}
    scoped = scope_choice_decisions({pin: PinSource(block=block, settled=True)})[pin]
    return PlannerPlan.model_validate(
        {
            **plan.model_dump(),
            "version": version,
            "previous_version": version - 1 if version > 1 else None,
            "decisions": [*plan.decisions, *scoped],
            "evidence_requirements": [
                r.model_copy(
                    update={
                        "source_spec_id": pin,
                        "decision_ids": (*r.decision_ids, *(d.slot for d in scoped)),
                    }
                )
                if isinstance(r, DatasetRequirement)
                else r
                for r in plan.evidence_requirements
            ],
        }
    )


def _row_id(plan: PlannerPlan, axis: str) -> str:
    """The plan card's id for the row of `plan`'s one dataset's choice on `axis`
    (`harness.plan_display.choice_id`)."""
    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    return f"{dataset.id}:{axis}"
