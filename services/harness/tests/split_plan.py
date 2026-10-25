"""A compiled plan as `propose_plan` stores it once a dataset is read from its first year.

`debt` is read beside a copy, `debt_history`. `last_five` rolls from its
start, so it reads the copy from the dataset's first year (2000), where
`total` keeps the years the plan states (2015-2024). The model wrote
`debt`, `total` and `last_five`. Code added `rolling_copies` and
`first_years`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from planner.operations import (
    AggregateOperation,
    GrowthOperation,
    RetrieveOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    FirstYear,
    GrowthRequirement,
    Period,
    Plan,
    split_of,
)
from planner.records import (
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    Reduction,
    RelativePeriod,
    RollingWindow,
    Severity,
)
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)

SPEC_DEBT = "spec_" + "a" * 32
SPEC_DEBT_HISTORY = "spec_" + "b" * 32
FIRST_YEAR = 2000

_LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)


def stored_split_plan(decision: Callable[..., Decision]) -> Plan:
    """The compiled, split plan described in the module docstring, with a warning on each read of the dataset."""
    return split_draft(decision).with_translation(translation=_translation(), workflow=_workflow())


def split_draft(decision: Callable[..., Decision]) -> Plan:
    """The split plan described in the module docstring, as `propose_plan` records it before translation."""
    draft = Plan(
        question="How has Kenya's debt service changed?",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="debt",
                decision_ids=["measure"],
                measure="external debt service paid",
                entities=[Entity(name="Kenya", role="debtor")],
                period=Period(start=2015, end=2024),
                period_end_open=True,
            ),
            AggregateRequirement(
                id="total",
                decision_ids=["measure"],
                input_id="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthRequirement(
                id="last_five",
                input_id="debt",
                window_start=2019,
                window_end=2024,
                rolling=_LAST_FIVE,
            ),
        ],
        first_years=[FirstYear(requirement_id="debt", year=FIRST_YEAR)],
    )
    split = split_of(draft)
    assert [copy.copy_id for copy in split.copies] == ["debt_history"]
    return Plan.model_validate(
        {
            **draft.model_dump(),
            "evidence_requirements": split.requirements,
            "rolling_copies": split.copies,
        }
    )


def _translation() -> Translation:
    return Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="debt", spec_id=SPEC_DEBT),
            RetrieveOperation(
                id="debt_history", requirement_id="debt_history", spec_id=SPEC_DEBT_HISTORY
            ),
            AggregateOperation(
                id="total",
                requirement_id="total",
                input="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="last_five",
                requirement_id="last_five",
                input="debt_history",
                window_start=2019,
                window_end=2024,
                rolling=_LAST_FIVE,
            ),
        ],
        terminal=["total", "last_five"],
    )


def _workflow() -> Workflow:
    def retrieval(step_id: str, spec_id: str) -> WorkflowStep:
        return WorkflowStep(
            id=step_id,
            operation_id=step_id,
            tool=ToolId(server="data-mcp", name="get_data"),
            args={"spec_id": spec_id},
            input_schema_hash="hash",
        )

    def reading(step_id: str, tool: str, from_step: str, **args: Any) -> WorkflowStep:
        return WorkflowStep(
            id=step_id,
            operation_id=step_id,
            tool=ToolId(server="analysis-mcp", name=tool),
            args=args,
            bindings={"frame_ref": Binding(from_step=from_step, output=ArtifactKind.FRAME_REF)},
            input_schema_hash="hash",
        )

    def warning(requirement_id: str) -> EvidenceIssue:
        return EvidenceIssue(
            code=EvidenceIssueCode.WINDOW_SHORT,
            severity=Severity.WARNING,
            requirement_id=requirement_id,
            detail="the source starts after the years asked.",
        )

    return Workflow(
        steps=[
            retrieval("debt", SPEC_DEBT),
            retrieval("debt_history", SPEC_DEBT_HISTORY),
            reading("total", "aggregate", "debt", aggregation="sum", group_by=["time_period"]),
            reading("last_five", "growth", "debt_history"),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_DEBT, sentence="external debt service paid", has_data=True),
            SelectedSource(
                spec_id=SPEC_DEBT_HISTORY, sentence="external debt service paid", has_data=True
            ),
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id="debt", step_id="debt", spec_id=SPEC_DEBT),
            EvidenceBinding(
                requirement_id="debt_history", step_id="debt_history", spec_id=SPEC_DEBT_HISTORY
            ),
        ],
        terminal_steps=["total", "last_five"],
        warnings=[warning("debt"), warning("debt_history")],
    )
