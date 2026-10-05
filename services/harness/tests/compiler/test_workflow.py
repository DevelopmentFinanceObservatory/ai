"""Tests for `planner.workflow`: the record of exactly what runs."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from compiler import evidence as compiler_evidence
from planner import records as planner_records
from planner.records import EvidenceIssue, EvidenceIssueCode, Severity
from planner.workflow import (
    ArtifactKind,
    Binding,
    OpenEntityAxis,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import ValidationError

FRAME = Binding(from_step="debt", output=ArtifactKind.FRAME_REF)


def _step(step_id: str = "debt", **overrides: object) -> WorkflowStep:
    fields: dict[str, object] = {
        "id": step_id,
        "operation_id": step_id,
        "tool": ToolId(server="data-mcp", name="get_data"),
        "args": {"spec_id": "spec_" + "a" * 32},
        "input_schema_hash": "hash",
    }
    return WorkflowStep(**(fields | overrides))  # type: ignore[arg-type]


def test_an_argument_is_literal_or_bound_and_never_both() -> None:
    with pytest.raises(ValidationError, match="both literally and by binding: \\['frame_ref'\\]"):
        _step(args={"frame_ref": {}}, bindings={"frame_ref": FRAME})


def test_a_tool_belongs_to_a_known_server() -> None:
    with pytest.raises(ValidationError, match="server"):
        ToolId(server="presentation-mcp", name="write_component")


def test_a_workflow_refers_only_to_steps_it_holds() -> None:
    totals = _step(
        "totals",
        tool=ToolId(server="analysis-mcp", name="aggregate"),
        args={},
        bindings={"frame_ref": FRAME},
    )

    with pytest.raises(ValidationError, match="does not hold: \\['debt'\\]"):
        Workflow(steps=[totals], sources=[], evidence_bindings=[], terminal_steps=["totals"])
    with pytest.raises(ValidationError, match="does not hold: \\['answer'\\]"):
        Workflow(steps=[_step()], sources=[], evidence_bindings=[], terminal_steps=["answer"])


def test_step_ids_are_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Workflow(
            steps=[_step(), _step()], sources=[], evidence_bindings=[], terminal_steps=["debt"]
        )


def test_a_step_names_the_operation_it_compiles() -> None:
    with pytest.raises(ValidationError, match="'debt' must equal its operation_id 'totals'"):
        _step(operation_id="totals")


def test_a_workflow_records_the_phrase_each_source_was_searched_as(
    source: Callable[..., SelectedSource],
) -> None:
    found = source(searched_as="debt service payments")

    workflow = Workflow(
        steps=[_step()], sources=[found], evidence_bindings=[], terminal_steps=["debt"]
    )

    (recorded,) = workflow.sources
    assert recorded.searched_as == "debt service payments"
    assert source().searched_as is None


def test_a_source_permits_unknown_coverage_and_an_absent_identity(
    source: Callable[..., SelectedSource],
) -> None:
    unknown = source(has_data=None, identity=None, coverage_kind="breadth")

    assert unknown.has_data is None
    assert unknown.identity is None
    assert unknown.coverage_kind == "breadth"


def test_a_source_names_each_open_axis_once(source: Callable[..., SelectedSource]) -> None:
    with pytest.raises(ValidationError, match="its own column: \\['origin', 'origin'\\]"):
        source(
            open_entity_axes=[
                OpenEntityAxis(column="origin", member_count=55),
                OpenEntityAxis(column="origin"),
            ]
        )


def test_a_column_a_resolved_entity_binds_is_not_open(
    source: Callable[..., SelectedSource],
) -> None:
    # The default source resolves Kenya under `destination`.
    with pytest.raises(ValidationError, match="not open: \\['destination'\\]"):
        source(open_entity_axes=[OpenEntityAxis(column="destination", member_count=1)])


def test_a_source_opens_a_column_beside_the_ones_it_binds(
    source: Callable[..., SelectedSource],
) -> None:
    donors = OpenEntityAxis(column="origin", member_count=55)

    assert source(open_entity_axes=[donors]).open_entity_axes == (donors,)


def test_a_source_states_whether_its_open_axes_were_read(
    source: Callable[..., SelectedSource],
) -> None:
    # `None` is "never read"; an empty tuple is "read, none open".
    assert source().open_entity_axes is None
    assert source(open_entity_axes=[]).open_entity_axes == ()


def test_a_workflow_carries_the_caveats_it_was_compiled_with() -> None:
    caveat = EvidenceIssue(
        code=EvidenceIssueCode.WINDOW_SHORT,
        severity=Severity.WARNING,
        requirement_id="annual_debt_service",
        detail="the source covers 2010-2020, the requirement asks for 2010-2024",
    )
    quiet = Workflow(steps=[_step()], sources=[], evidence_bindings=[], terminal_steps=["debt"])
    caveated = Workflow(
        steps=[_step()],
        sources=[],
        evidence_bindings=[],
        terminal_steps=["debt"],
        warnings=[caveat],
    )

    assert quiet.warnings == ()
    assert caveated.warnings == (caveat,)
    assert caveated != quiet


def test_evidence_types_are_importable_from_records_and_from_the_compiler() -> None:
    assert compiler_evidence.EvidenceIssue is planner_records.EvidenceIssue
    assert compiler_evidence.EvidenceIssueCode is planner_records.EvidenceIssueCode
    assert compiler_evidence.Severity is planner_records.Severity
