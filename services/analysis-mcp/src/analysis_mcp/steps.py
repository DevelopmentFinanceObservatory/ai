"""Step records: what an operation did to its inputs, one record per operation.

The models follow dfo-spec-kit's `steps.schema.json` (`dfo_descriptions`) for
every member this toolkit writes: an id, the operation, its inputs by role, the
operation's one fixed policy, the caller's own choices, what it kept, the data
gaps it left out by reason, per-period or per-entity coverage, the unit
it produces, the new variable it writes its rows under, and the entities each
of its inputs holds. A member the schema makes optional is absent until set,
and a model dumps only the members it holds. A unit is `None` only where the data
declares none and the values are no ratio.

A stored frame that an in-process verb wrote carries its step on its `Trace`,
naming each input by the frame pair it read (`VersionRef`).
`common.steps.assemble_steps` turns the steps along a frame's ancestry into one
list whose inputs name earlier steps. The reason names in `left_out` follow the
exclusion reasons (`analysis_mcp.artifact.ExclusionReason`), which the kit's
operation list uses too.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from dfo_frame_grammar.grammar import (
    DeflationBasis,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)
from pydantic.experimental.missing_sentinel import MISSING

Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
"""A lower-snake-case name: an operation, a policy, a role, a reason or a count."""

StepId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
"""A step's id, unique within its list."""

TEXT_MAX_LENGTH = 512
"""The most characters a step's text holds, a `Ref`'s `ref` among them (the kit's bound)."""

PERIOD_MAX_LENGTH = 32
"""The most characters a `PeriodCoverage` period holds (the kit's bound)."""

Text = Annotated[str, StringConstraints(min_length=1, max_length=TEXT_MAX_LENGTH)]
Count = Annotated[int, Field(ge=0, strict=True)]

RoleName = Annotated[str, StringConstraints(pattern=r"^[a-z][A-Za-z0-9]{0,63}$")]
"""A role's graph name, the bare lower-camel-case property such as `financingProvider`
(the kit's `$defs/role` in `steps.schema.json`)."""

ENTITY_COLUMN_MAX_LENGTH = 64
"""The most characters an entity column in a step's `sides` holds (the kit's bound)."""

SIDE_MAX_COLUMNS = 16
"""The most entity columns one side in a step's `sides` holds (the kit's bound)."""

EntityColumn = Annotated[str, StringConstraints(min_length=1, max_length=ENTITY_COLUMN_MAX_LENGTH)]
"""A frame's entity column, such as `financingRecipient` or `donor`."""

EntitiesHeld = Annotated[dict[EntityColumn, Text], Field(max_length=SIDE_MAX_COLUMNS)]
"""The entities one input holds: each entity column holding one value across
the input's rows, with that value as the input's rows name it (`dcid:...`)."""

type CountOf = Literal["entities", "periods", "entity_periods", "pooled_entities"]
"""What a `LeftOut` count counts. `pooled_entities` counts the parties of one
role, which the step's kept `entities` never include."""

_DCID_PREFIX = "dcid:"


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionRef(_Step):
    """A version a step reads: a `spec_` id with its `cv_` content version, or
    an `ops_` id with its `ocv_` content version."""

    id: Annotated[str, StringConstraints(pattern=r"^(spec_[0-9a-f]{32}|ops_[0-9a-f]{6,64})$")]
    content_version: Annotated[
        str, StringConstraints(pattern=r"^(cv_[0-9a-f]{32}|ocv_[0-9a-f]{6,64})$")
    ]
    label: Text | MISSING = MISSING

    @model_validator(mode="after")
    def _one_family(self) -> Self:
        if self.id.startswith("spec_") != self.content_version.startswith("cv_"):
            raise ValueError(f"{self.id} cannot carry content version {self.content_version}")
        return self


StepInput = StepId | VersionRef
"""One input: an earlier step's id, or a version."""


class SimpleUnit(_Step):
    """A unit by reference, by currency, or both. `ref` is the unit's bare
    dcid; `currency`, `prices`, `base_year` and `deflation_basis` name a
    converted unit."""

    ref: Text | MISSING = MISSING
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | MISSING = MISSING
    label: Text | MISSING = MISSING
    prices: Name | MISSING = MISSING
    base_year: Annotated[int, Field(ge=1900, le=2100, strict=True)] | MISSING = MISSING
    deflation_basis: DeflationBasis | MISSING = MISSING

    @model_validator(mode="after")
    def _ref_or_currency(self) -> Self:
        if self.ref is MISSING and self.currency is MISSING:
            raise ValueError("a unit names a ref, a currency, or both")
        return self


def _bare_dcid(value: str) -> str:
    if value.startswith(_DCID_PREFIX):
        raise ValueError("a variable is named by its bare dcid, without `dcid:`")
    return value


class VariableSide(_Step):
    """One side of a composed unit that counts a variable, named by its bare
    dcid, in place of a unit."""

    stat_var: Annotated[Text, AfterValidator(_bare_dcid)]
    label: Text | MISSING = MISSING


class ComposedStepUnit(_Step):
    """A quotient's unit: the numerator's unit per the denominator's."""

    numerator: SimpleUnit | VariableSide
    denominator: SimpleUnit | VariableSide


StepUnit = SimpleUnit | ComposedStepUnit | None
"""The unit a step produces; `None` when the data declares none."""


class LeftOut(_Step):
    """One data gap: why some data was left out, how many, and of what. A
    count of `pooled_entities` names the `role` whose parties it counts, and
    no other count names one."""

    reason: Name
    count: Count
    of: CountOf
    role: RoleName | MISSING = MISSING

    @model_validator(mode="after")
    def _role_of_pooled_entities(self) -> Self:
        if (self.of == "pooled_entities") == (self.role is MISSING):
            raise ValueError("a count names a role exactly when it counts pooled entities")
        return self


class PeriodCoverage(_Step):
    """How many entities one period's total covers."""

    period: Annotated[str, StringConstraints(min_length=1, max_length=PERIOD_MAX_LENGTH)]
    entities: Count


class Ref(_Step):
    """A reference by the dcid or id it names, with optional words for it."""

    ref: Text
    label: Text | MISSING = MISSING


class EntityCoverage(_Step):
    """How many of the window's `of` periods one ranked entity holds."""

    entity: Ref
    periods: Count
    of: Count


MAX_COVERAGE = 500
"""The most entries a step's `period_coverage` or `entity_coverage` holds (the
kit's bound)."""


class Step(_Step):
    """One operation over its inputs. A step takes `input` or `inputs`, or
    neither, and never both."""

    id: StepId
    op: Name
    input: StepInput | MISSING = MISSING
    inputs: Annotated[dict[Name, StepInput], Field(min_length=1, max_length=8)] | MISSING = MISSING
    policy: Name | MISSING = MISSING
    params: Annotated[dict[Name, JsonValue], Field(max_length=32)] | MISSING = MISSING
    options: Annotated[dict[Name, JsonValue], Field(max_length=32)] | MISSING = MISSING
    kept: Annotated[dict[Name, Count], Field(max_length=16)] | MISSING = MISSING
    left_out: Annotated[list[LeftOut], Field(max_length=32)] | MISSING = MISSING
    period_coverage: Annotated[list[PeriodCoverage], Field(max_length=MAX_COVERAGE)] | MISSING = (
        MISSING
    )
    entity_coverage: Annotated[list[EntityCoverage], Field(max_length=MAX_COVERAGE)] | MISSING = (
        MISSING
    )
    unit: StepUnit | MISSING = MISSING
    stat_var: Annotated[Text, AfterValidator(_bare_dcid)] | MISSING = MISSING
    sides: Annotated[dict[Name, EntitiesHeld], Field(min_length=1, max_length=8)] | MISSING = (
        MISSING
    )

    @model_validator(mode="after")
    def _input_or_inputs(self) -> Self:
        if self.input is not MISSING and self.inputs is not MISSING:
            raise ValueError("a step takes `input` or `inputs`, not both")
        return self
