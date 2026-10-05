"""A retrieval asked to sum its family of StatVars must return one StatVar.

dfo-apis collapses a summed family into one series under the family's own
ref, so `get_data` declares `stat_var_count == 1` for it. A count above one
means the sum did not happen; the executor refuses `SHAPE_MISMATCH` at the
retrieval rather than letting the aggregate reading it fail as
`stat_var_unresolved`, with a `family_not_summed` payload that stops the turn.
No count at all is a fetch that degraded, and runs on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from compiler.surface import ToolSchema
from executor.classify import FAMILY_NOT_SUMMED, Disposition, classify_refusal
from executor.outcomes import Accepted, RefusalKind, Refused
from executor.run import execute_approved_plan
from planner.plan import Plan
from planner.workflow import OpenAxis, SelectedSource

from .conftest import RESULT_FRAME, ScriptedSurface, ok

PlanFactory = Callable[..., Plan]
ManifestFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def summed_family(source: Callable[..., SelectedSource]) -> SelectedSource:
    """A family of three StatVars open only on a summable axis, which the rule reads `sum`."""
    return source(
        stat_vars=("Debt_0", "Debt_1", "Debt_2"),
        open_axes=(OpenAxis(property="crsPurpose", summable=True),),
    )


async def _run_with_declared_count(
    plan: Plan,
    manifest: dict[str, Any],
    schemas: list[ToolSchema],
    declared: dict[str, Any],
) -> Accepted | Refused | Any:
    surface = ScriptedSurface(
        schemas=schemas,
        responses={"get_data": [ok(manifest | declared)], "aggregate": [ok(RESULT_FRAME)]},
    )
    _finished, outcome = await execute_approved_plan(plan, surface=surface)
    return outcome


async def test_a_summed_retrieval_declaring_one_stat_var_runs(
    approved_plan: PlanFactory,
    summed_family: SelectedSource,
    manifest_for: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    plan = approved_plan(source=summed_family)
    assert plan.workflow is not None
    assert plan.workflow.steps[0].args["stat_vars"] == "sum"

    outcome = await _run_with_declared_count(
        plan, manifest_for(summed_family), real_schemas, {"stat_var_count": 1}
    )

    assert isinstance(outcome, Accepted), outcome
    assert [step.stat_var_count for step in outcome.steps] == [1, 1]
    # Each record carries the role keys its result declared, which narration
    # compares to tell a total across recipients (`harness.graph`).
    assert [step.role_keys for step in outcome.steps] == [None, ()]


async def test_a_summed_retrieval_declaring_several_stat_vars_is_refused(
    approved_plan: PlanFactory,
    summed_family: SelectedSource,
    manifest_for: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    outcome = await _run_with_declared_count(
        approved_plan(source=summed_family),
        manifest_for(summed_family),
        real_schemas,
        {"stat_var_count": 3},
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SHAPE_MISMATCH
    assert outcome.step_id == "debt"
    assert "asked to sum its family" in outcome.detail
    # The same plan draws the same unsummed frame, so the payload classifies
    # as a stop and words the refusal for the user.
    assert outcome.payload is not None
    assert outcome.payload.code == FAMILY_NOT_SUMMED
    assert classify_refusal(outcome.payload) is Disposition.STOP
    assert "did not add them up" in outcome.payload.message
    assert outcome.payload.fix["remedy"] not in outcome.payload.message


async def test_a_summed_retrieval_declaring_no_count_runs_on(
    approved_plan: PlanFactory,
    summed_family: SelectedSource,
    manifest_for: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    # `get_data` declares no `stat_var_count` when its fetch of the frame
    # degraded; the sum still happened upstream.
    outcome = await _run_with_declared_count(
        approved_plan(source=summed_family), manifest_for(summed_family), real_schemas, {}
    )

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[0].stat_var_count is None


async def test_a_retrieval_that_does_not_sum_declares_whatever_its_frame_holds(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    # One StatVar, no reading, so no bound: the count is the frame's own.
    plan = approved_plan()
    assert plan.workflow is not None
    assert "stat_vars" not in plan.workflow.steps[0].args

    outcome = await _run_with_declared_count(plan, manifest(), real_schemas, {})

    assert isinstance(outcome, Accepted), outcome
