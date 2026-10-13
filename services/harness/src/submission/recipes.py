"""Direct recipes: reviewed requirement templates the orchestrator runs without a plan.

A recipe is a fixed calculation over sources the conversation has already
verified. It has an id, typed parameters, preconditions stated in words and
checked in code, and a canonical requirement template: the dataset
requirement and the derived requirement a planner would write for the same
calculation, which compile through the planner's own translator and
compiler (`submission.direct`). Its revision is the digest of what it
promises: its parameters' schema, its preconditions, and the requirements
its template writes for a fixed reference input. Editing any of them moves
the revision.

`RECIPES` holds every recipe this harness ships. `growth` is the only one:
the growth of one verified series between two stated years, relative or
absolute. A quotient of two series is left to the planner, since nothing the
conversation records settles that a denominator is compatible with its
numerator (unit, entity alignment, period). The planner settles that with
decisions the user approves.

A recipe's `build` checks a request's source and years against the evidence
the harness read for its source (`SourceEvidence`), and answers the
requirements to compile or a `DirectRefusal` whose `outcome` says what the
caller does next: `needs_fields` to fill or correct
a value already established in the conversation, `needs_decision` for a
choice only the user can make, `requires_planning` for work beyond the
recipe. A parameter nothing recorded can check, such as the growth method,
is the model's reading of the user's words, and `Recipe.origins` says so for
the run's `executor.ledger.RequestAuthorisation`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Any

import rfc8785
from executor.ledger import ParameterOrigin
from planner.plan import (
    MAX_MEASURE_CHARS,
    DatasetRequirement,
    Entity,
    GrowthRequirement,
)
from planner.records import GrowthChange, Period, StrictModel, Year
from pydantic import Field


class DirectOutcome(StrEnum):
    """The `outcome` a direct submission answers with."""

    NEEDS_FIELDS = "needs_fields"
    NEEDS_DECISION = "needs_decision"
    REQUIRES_PLANNING = "requires_planning"
    COMPLETED = "completed"
    NOT_CHARTABLE = "not_chartable"
    PENDING = "pending"
    FAILED = "failed"


class FieldCode(StrEnum):
    """Why a parameter is refused under `needs_fields`."""

    MISSING = "missing"
    UNVERIFIED_SOURCE = "unverified_source"
    WINDOW_ORDER = "window_order"


class DecisionCode(StrEnum):
    """Which choice a `needs_decision` answer leaves to the user."""

    METHOD_UNRESOLVED = "method_unresolved"
    MEASURE_UNRESOLVED = "measure_unresolved"


class PlanningCode(StrEnum):
    """Why a request needs the planner (`requires_planning`)."""

    RECIPE_NOT_ENABLED = "recipe_not_enabled"
    PLAN_PENDING = "plan_pending"
    PLAN_BUDGET_SPENT = "plan_budget_spent"
    SEARCH_WARNING = "search_warning"
    COVERAGE_UNKNOWN = "coverage_unknown"
    OUTSIDE_COVERAGE = "outside_coverage"
    NOT_ONE_SERIES = "not_one_series"
    NOT_COMPILED = "not_compiled"
    BEYOND_LIMITS = "beyond_limits"


@dataclass(frozen=True)
class SourceEvidence:
    """What the conversation recorded about one source, as the harness read it.

    `measure` is the sentence search rendered for it (the search part's own
    text when it rendered none), which the words search was sent do not
    change; `entities` the names of the entities its block resolves,
    `coverage` the window its block reports for an exact coverage (`None`
    for any other), `one_series` whether a retrieval of it holds a single
    series, and `blocking_warnings` the codes of its search part's warnings
    that mean search answered a narrower question than was asked.
    """

    spec_id: str
    measure: str
    entities: tuple[str, ...]
    coverage: Period | None
    one_series: bool
    blocking_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectRefusal:
    """A direct submission the harness will not run, and what the caller does next.

    `errors` are `needs_fields`' per-parameter problems, each
    `{"field", "code", "message"}`; `code` names the reason for the other
    outcomes; `options` are a decision's choices.
    """

    outcome: DirectOutcome
    message: str
    code: str | None = None
    errors: tuple[Mapping[str, str], ...] = ()
    options: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def payload(self, recipe: Recipe | None = None) -> dict[str, Any]:
        """The refusal as the orchestrator reads it, with `recipe`'s contract
        on a `needs_fields` answer."""
        answer: dict[str, Any] = {"outcome": self.outcome.value, "message": self.message}
        if self.code is not None:
            answer["code"] = self.code
        if self.errors:
            answer["errors"] = [dict(error) for error in self.errors]
        if self.options:
            answer["options"] = list(self.options)
        answer.update(self.detail)
        if self.outcome is DirectOutcome.NEEDS_FIELDS and recipe is not None:
            answer["contract"] = recipe.contract()
        return answer


def requires_planning(code: PlanningCode, message: str, **detail: Any) -> DirectRefusal:
    return DirectRefusal(
        outcome=DirectOutcome.REQUIRES_PLANNING, message=message, code=code.value, detail=detail
    )


# The requirements a recipe compiles: its dataset requirements first.
type Requirements = tuple[DatasetRequirement | GrowthRequirement, ...]


class RecipeParameters(StrictModel):
    """What every recipe's parameters hold: the one source it reads."""

    spec_id: str | None = Field(
        default=None,
        description="the spec_id of the series, one a search this conversation settled on",
    )


@dataclass(frozen=True, kw_only=True)
class Recipe:
    """One direct recipe.

    `parameters` is the type a request's arguments are read as. `build`
    checks a request against the evidence for its source (the parameters'
    `spec_id`) and answers the requirements to compile or the refusal.
    `task` names, in words, the work of a request `build` accepted: the
    run's authorisation records it and the planner's translation reads it
    as the question. `origins` names where each
    parameter comes from (`executor.ledger.ParameterOrigin`). `max_steps`
    bounds the workflow its requirements may compile to. `reference` is the
    fixed input the revision digests the template's output for.
    """

    id: str
    purpose: str
    parameters: type[RecipeParameters]
    preconditions: tuple[str, ...]
    origins: Mapping[str, ParameterOrigin]
    max_steps: int
    build: Callable[[Any, SourceEvidence | None], Requirements | DirectRefusal]
    task: Callable[[Any, SourceEvidence], str]
    reference: tuple[RecipeParameters, SourceEvidence]

    @cached_property
    def revision(self) -> str:
        """The digest of the parameters' schema, the preconditions, the
        parameters' origins and the requirements the template writes for
        `reference`."""
        parameters, evidence = self.reference
        built = self.build(parameters, evidence)
        if isinstance(built, DirectRefusal):
            raise RuntimeError(f"recipe {self.id!r} refuses its own reference input: {built}")
        content = {
            "id": self.id,
            "parameters": self.parameters.model_json_schema(),
            "preconditions": list(self.preconditions),
            "origins": dict(self.origins),
            "template": [requirement.model_dump(mode="json") for requirement in built],
        }
        return hashlib.sha256(rfc8785.dumps(content)).hexdigest()[:16]

    def contract(self) -> dict[str, Any]:
        """What a caller fills the recipe from: each parameter with its
        description, the required ones, and the preconditions."""
        schema = self.parameters.model_json_schema()
        return {
            "recipe": self.id,
            "revision": self.revision,
            "purpose": self.purpose,
            "parameters": {
                name: prop.get("description", "") for name, prop in schema["properties"].items()
            },
            "preconditions": list(self.preconditions),
        }


# --- growth ------------------------------------------------------------------

# The requirement ids the growth template writes.
SERIES_ID = "series"
GROWTH_ID = "growth"


class GrowthParameters(RecipeParameters):
    """The growth recipe's parameters. Every one is optional here so a
    missing one reaches `build`, which answers `needs_fields`."""

    start_year: Year | None = Field(default=None, description="the year growth is measured from")
    end_year: Year | None = Field(default=None, description="the year growth is measured to")
    change: GrowthChange | None = Field(
        default=None,
        description=(
            "relative: the change as a share of the start value (per cent); absolute: the "
            "difference in the series' own unit. Set only when the user's words settle it"
        ),
    )


_GROWTH_PRECONDITIONS = (
    "the series is a spec_id a search in this conversation settled on, with no warning that "
    "search answered a narrower question",
    "a retrieval of it holds one series: no side of a flow left open, no family read apart",
    "both years lie inside the window its search reported",
    "the change, relative or absolute, is stated from the user's words; the harness records "
    "it as the model's reading and does not check it",
)

_GROWTH_ORIGINS = {
    "spec_id": ParameterOrigin.RECORDED,
    "start_year": ParameterOrigin.RECORDED,
    "end_year": ParameterOrigin.RECORDED,
    "change": ParameterOrigin.MODEL_STATED,
}


def _missing(field_name: str) -> dict[str, str]:
    return {
        "field": field_name,
        "code": FieldCode.MISSING.value,
        "message": f"{field_name} is not set: fill it from the conversation",
    }


def _source_refusal(evidence: SourceEvidence, asked: Period) -> DirectRefusal | None:
    """The refusal of a source the growth recipe's preconditions do not hold
    for over the `asked` years, or `None`."""
    if evidence.blocking_warnings:
        return requires_planning(
            PlanningCode.SEARCH_WARNING,
            "Search answered a narrower question than was asked for this series.",
            warnings=list(evidence.blocking_warnings),
        )
    if not evidence.one_series:
        return requires_planning(
            PlanningCode.NOT_ONE_SERIES,
            "This source holds more than one series, so its growth needs a plan.",
        )
    if evidence.coverage is None:
        return requires_planning(
            PlanningCode.COVERAGE_UNKNOWN,
            "Search reported no window of years for this series.",
        )
    if not evidence.coverage.covers(asked):
        return requires_planning(
            PlanningCode.OUTSIDE_COVERAGE,
            f"The series covers {evidence.coverage.start} to {evidence.coverage.end}, which "
            f"does not hold {asked.start} to {asked.end}.",
            coverage=evidence.coverage.model_dump(),
        )
    return None


def build_growth(
    parameters: GrowthParameters, evidence: SourceEvidence | None
) -> Requirements | DirectRefusal:
    """The growth template's two requirements for `parameters`, or the refusal.

    Missing values come first, all at once, then a source the conversation
    did not verify, then the method, then the source's own preconditions.
    """
    errors = [
        _missing(name)
        for name in ("spec_id", "start_year", "end_year")
        if getattr(parameters, name) is None
    ]
    if parameters.spec_id is not None and evidence is None:
        errors.append(
            {
                "field": "spec_id",
                "code": FieldCode.UNVERIFIED_SOURCE.value,
                "message": f"no search in this conversation settled on {parameters.spec_id}: "
                "use a spec_id a search result gave as definite or a candidate you fetched",
            }
        )
    start, end = parameters.start_year, parameters.end_year
    if start is not None and end is not None and end <= start:
        errors.append(
            {
                "field": "end_year",
                "code": FieldCode.WINDOW_ORDER.value,
                "message": f"end_year {end} must come after start_year {start}",
            }
        )
    if errors:
        return DirectRefusal(
            outcome=DirectOutcome.NEEDS_FIELDS,
            message="The growth recipe cannot run until these fields are corrected.",
            errors=tuple(errors),
        )
    assert evidence is not None and start is not None and end is not None
    if parameters.change is None:
        return DirectRefusal(
            outcome=DirectOutcome.NEEDS_DECISION,
            message="Whether growth is relative (per cent) or absolute (the difference in the "
            "series' unit) is not settled: ask the user, or call again with the change their "
            "words state.",
            code=DecisionCode.METHOD_UNRESOLVED.value,
            options=tuple(change.value for change in GrowthChange),
        )
    asked = Period(start=start, end=end)
    refused = _source_refusal(evidence, asked)
    if refused is not None:
        return refused
    series = DatasetRequirement(
        id=SERIES_ID,
        measure=evidence.measure[:MAX_MEASURE_CHARS],
        entities=tuple(Entity(name=name) for name in evidence.entities),
        period=asked,
        source_spec_id=evidence.spec_id,
    )
    growth = GrowthRequirement(
        id=GROWTH_ID,
        input_id=SERIES_ID,
        window_start=start,
        window_end=end,
        change=parameters.change,
    )
    return (series, growth)


def growth_task(parameters: GrowthParameters, evidence: SourceEvidence) -> str:
    """The growth of `evidence`'s measure over the years `parameters` ask."""
    change = parameters.change.value if parameters.change is not None else "unstated"
    return (
        f"{change} growth of {evidence.measure} from {parameters.start_year} to "
        f"{parameters.end_year}"
    )


_REFERENCE_SPEC = "spec_" + "0" * 32

GROWTH = Recipe(
    id="growth",
    purpose="the growth of one verified series between two stated years",
    parameters=GrowthParameters,
    preconditions=_GROWTH_PRECONDITIONS,
    origins=_GROWTH_ORIGINS,
    max_steps=2,
    build=build_growth,
    task=growth_task,
    reference=(
        GrowthParameters(
            spec_id=_REFERENCE_SPEC, start_year=2015, end_year=2020, change=GrowthChange.RELATIVE
        ),
        SourceEvidence(
            spec_id=_REFERENCE_SPEC,
            measure="reference measure",
            entities=("Reference place",),
            coverage=Period(start=2010, end=2024),
            one_series=True,
        ),
    ),
)

RECIPES: Mapping[str, Recipe] = {GROWTH.id: GROWTH}
