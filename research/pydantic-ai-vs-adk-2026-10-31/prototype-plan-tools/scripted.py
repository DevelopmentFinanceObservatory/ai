"""The scripted half of a run that needs no data: dev's canned plan, proposed through
`propose_plan` and run on `harness.scripted`'s surface and executor.

`PROPOSAL` is the canned plan `harness.scripted_plan.scripted_plan` builds, as
`propose_plan` arguments. `scripted_translate` stands in for production's
translation: it translates and compiles the plan on record the way
`scripted_plan` does, against `SCRIPTED_TOOL_SCHEMAS`, retrieving the dataset
the session's search returned, so the plan runs on `ScriptedSurface`. `FixedPlanner` is a `PlannerModel` seam that drafts a given
plan, so dev's own root workflow runs the very plan this server accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compiler.compile import compile_plan
from harness.scripted_plan import (
    _AGGREGATE_STEP_ID,
    _RETRIEVE_STEP_ID,
    _SCHEMAS_BY_NAME,
    _SCRIPTED_CAPABILITIES,
    _scripted_source,
)
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Plan,
    load_plan,
    pin_sources,
    plan_revision,
    save_plan,
)
from planner.records import GroupBy, Reduction
from planner.workflow import SelectedSource

BRIEF = "Life expectancy in Kenya since 1960"
SEARCH = {"query": BRIEF, "parts": [{"variable": "Life expectancy in Kenya"}]}

PROPOSAL: dict[str, Any] = {
    "question": BRIEF,
    "answer_mode": "computed",
    "decisions": [
        {
            "slot": "measure",
            "description": "Which measure answers the scripted question.",
            "options": ["life expectancy at birth", "healthy life expectancy"],
            "chosen": "life expectancy at birth",
            "resolved_by": "planner_default",
            "rationale": "Life expectancy at birth is the measure scripted mode answers with.",
        }
    ],
    "evidence_requirements": [
        {
            "kind": "dataset",
            "id": "life_expectancy",
            "decision_ids": ["measure"],
            "measure": "life expectancy at birth",
            "entities": [{"name": "Kenya"}],
            "period": {"start": 1960, "end": 2023},
        },
        {
            "kind": "aggregate",
            "id": "life_expectancy_by_period",
            "decision_ids": ["measure"],
            "input_id": "life_expectancy",
            "reduction": "mean",
            "group_by": ["period"],
        },
    ],
    "answered": [{"ask": 0, "outputs": ["life_expectancy_by_period"]}],
}


def amended(version: int) -> dict[str, Any]:
    """A patch of `version` that narrows the period to 1990 onwards."""
    dataset = {**PROPOSAL["evidence_requirements"][0], "period": {"start": 1990, "end": 2023}}
    return {"amends_version": version, "evidence_requirements": [dataset]}


def scripted_source(spec_id: str) -> SelectedSource:
    return _scripted_source(spec_id)


async def scripted_translate(tool_context: Any) -> dict[str, object]:
    """Translate and compile the plan on record as `scripted_plan` does, and answer `ok`."""
    plan = load_plan(tool_context.state)
    assert plan is not None
    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    (derived,) = [r for r in plan.evidence_requirements if isinstance(r, AggregateRequirement)]
    # The dataset the session's search returned, as production's discovery pins
    # a requirement stating no source to the source its pin shows.
    spec_id = next(iter(pin_sources(tool_context.state)), "spec_" + "f" * 32)
    translation = Translation(
        operations=[
            RetrieveOperation(id=_RETRIEVE_STEP_ID, requirement_id=dataset.id, spec_id=spec_id),
            AggregateOperation(
                id=_AGGREGATE_STEP_ID,
                requirement_id=derived.id,
                input=_RETRIEVE_STEP_ID,
                reduction=Reduction.MEAN,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=[_AGGREGATE_STEP_ID],
    )
    compiled = compile_plan(
        plan=plan,
        translation=translation,
        schemas=_SCHEMAS_BY_NAME,
        discovered={dataset.id: scripted_source(spec_id)},
        capabilities=_SCRIPTED_CAPABILITIES,
    )
    assert compiled.workflow is not None, compiled.issues
    save_plan(
        tool_context.state,
        plan.with_translation(translation=translation, workflow=compiled.workflow),
    )
    return {"outcome": "ok"}


class FixedPlanner:
    """A `PlannerModel` seam that drafts `plan` and revises nothing."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "plan": self._plan.model_dump(mode="json"),
            "summary": {"version": self._plan.version, "revision": plan_revision(self._plan)},
        }

    async def revise(self, **kwargs: Mapping[str, Any]) -> dict[str, Any]:
        del kwargs
        return {"refused": {"text": "The fixed planner revises no plan.", "code": None}}
