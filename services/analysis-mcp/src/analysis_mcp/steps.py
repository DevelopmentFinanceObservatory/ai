"""Step records: what an operation did to its inputs, one record per operation.

The models follow dfo-spec-kit's `steps.schema.json` (`dfo_descriptions`) for
every member this toolkit writes: an id, the operation, its inputs by role, the
operation's one fixed policy, the caller's own choices, what it kept, the data
gaps it left out by reason, per-period or per-entity coverage, the unit
it produces, the new variable it writes its rows under, and the entities each
of its inputs holds. A member the schema makes optional is absent until set,
and a model dumps only the members it holds. A unit is `None` only where the data
declares none and the values are no ratio.

Each verb tool builds its own step where its engine result meets its `Trace`,
naming each input by the frame pair it read (`VersionRef`).
`common.steps.assemble_steps` turns the steps along a frame's ancestry into one
list whose inputs name earlier steps. The reason names in `left_out` follow the
engine's exclusion reasons (`analysis_mcp.engine.common.ExclusionReason`),
which the kit's operation list uses too.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence, Sized
from typing import TYPE_CHECKING, Annotated, Literal, Self

from common.materialisation import FrameRef
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitCell, UnitPart, UnitRequest
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

from analysis_mcp.engine.common import ExclusionReason, Exclusions

if TYPE_CHECKING:
    from analysis_mcp.trace import Trace

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
    dcid; `currency`, `prices` and `base_year` name a converted unit."""

    ref: Text | MISSING = MISSING
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | MISSING = MISSING
    label: Text | MISSING = MISSING
    prices: Name | MISSING = MISSING
    base_year: Annotated[int, Field(ge=1900, le=2100, strict=True)] | MISSING = MISSING

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


def _representable(names: Sequence[str], *, max_length: int) -> bool:
    """Whether a coverage list whose entries name `names`, one per entry,
    fits its step: at most `MAX_COVERAGE` entries, each name at most
    `max_length` characters."""
    return len(names) <= MAX_COVERAGE and all(len(name) <= max_length for name in names)


def _sides_representable(sides: Mapping[str, Mapping[str, str]]) -> bool:
    """Whether every side of `sides` fits its step: at most `SIDE_MAX_COLUMNS`
    columns, each named in 1 to `ENTITY_COLUMN_MAX_LENGTH` characters and
    holding a value of 1 to `TEXT_MAX_LENGTH`."""
    return all(
        len(held) <= SIDE_MAX_COLUMNS
        and all(
            0 < len(column) <= ENTITY_COLUMN_MAX_LENGTH and 0 < len(value) <= TEXT_MAX_LENGTH
            for column, value in held.items()
        )
        for held in sides.values()
    )


NEW_VARIABLE_OPS = frozenset({"compare", "sum", "quotient", "share"})
"""The operations that write their rows under a new variable, `dcid:<op_spec_hash>`,
which their step names in `stat_var` (`new_variable` in the kit's operation list)."""


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


def _bare(ref: str) -> str:
    return ref.removeprefix(_DCID_PREFIX)


def _simple_unit(unit: str | UnitRequest) -> SimpleUnit:
    if isinstance(unit, str):
        return SimpleUnit(ref=_bare(unit))
    if unit.base_year is None:
        return SimpleUnit(currency=unit.currency, prices=unit.prices)
    return SimpleUnit(currency=unit.currency, prices=unit.prices, base_year=unit.base_year)


def _unit_side(part: UnitPart) -> SimpleUnit | VariableSide:
    if isinstance(part, StatVarPart):
        return VariableSide(stat_var=_bare(part.stat_var))
    return _simple_unit(part)


RATIO_UNIT = SimpleUnit(ref="dimensionless")
"""The unit of a dimensionless ratio, such as a relative growth rate."""


def step_unit(unit: UnitCell | None, *, ratio: bool = False) -> StepUnit:
    """`unit`, a frame's `unit` cell, as a step names it: a ref by its bare
    dcid, a converted unit by its currency and price basis, and a composed
    unit side by side, a side that declares no unit by the variable it
    counts. No unit is `RATIO_UNIT` when the values are a ratio (`ratio`,
    the trace's `dimensionless` marker), and `None` otherwise."""
    if unit is None:
        return RATIO_UNIT if ratio else None
    if isinstance(unit, ComposedUnit):
        return ComposedStepUnit(
            numerator=_unit_side(unit.numerator), denominator=_unit_side(unit.denominator)
        )
    return _simple_unit(unit)


def shared_step_unit(
    left: UnitCell | None, right: UnitCell | None, *, ratio: tuple[bool, bool]
) -> StepUnit | MISSING:
    """The unit a two-sided verb's step names: each side's `unit` cell as
    `step_unit` names it under that side's `ratio` flag, when both sides name
    the same unit, and `MISSING` when they differ."""
    left_unit = step_unit(left, ratio=ratio[0])
    return left_unit if left_unit == step_unit(right, ratio=ratio[1]) else MISSING


def entity_ref(entity_key: Sequence[tuple[str, str]]) -> str:
    """An entity keyed by `entity_key`, its role columns with their values,
    as a step's `Ref` names it: one role's value by its bare dcid, and
    several as `role=value` pairs joined by commas."""
    if len(entity_key) == 1:
        ((_, value),) = entity_key
        return _bare(value)
    return ", ".join(f"{role}={_bare(value)}" for role, value in entity_key)


_FRAME_SLOT = "frame_ref"


def caller_params(canonical_op_spec: str) -> dict[str, JsonValue]:
    """The parameters the caller set in `canonical_op_spec`, an op-spec's
    canonical JSON text (`Trace.execution_record.content`): every parameter
    that is neither null nor empty, less the frame slots a step names as its
    inputs."""
    parameters: dict[str, JsonValue] = json.loads(canonical_op_spec)["parameters"]
    return {
        name: value
        for name, value in parameters.items()
        if not name.endswith(_FRAME_SLOT) and value not in (None, {}, [])
    }


_WINDOW_PARAM_NAMES = {"window_start": "start", "window_end": "end"}


def window_params(params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """`params` with a window's bounds under the kit's names, `start` and
    `end`."""
    return {_WINDOW_PARAM_NAMES.get(name, name): value for name, value in params.items()}


def build_step(
    *,
    op: str,
    trace: Trace,
    inputs: Mapping[str, FrameRef],
    kept: Mapping[str, int | None],
    left_out: Sequence[LeftOut],
    unit: StepUnit | MISSING,
    params: Mapping[str, JsonValue] | None = None,
    period_entities: Iterable[tuple[str, int]] | None = None,
    entity_coverage: Iterable[tuple[str, int, int]] | None = None,
    sides: Mapping[str, Mapping[str, str]] | None = None,
) -> Step:
    """The step a verb's own operation `op` records on `trace`, the result's
    own trace.

    The step's id is the result's `op_spec_hash`, until
    `common.steps.assemble_steps` renumbers it. Its policy is the trace's
    `applied_policy.policy_name`, and its `params` the op-spec parameters
    the caller set (`caller_params`), unless the verb names them itself. An
    operation in `NEW_VARIABLE_OPS` names the variable its rows are written
    under, the bare `op_spec_hash`, in `stat_var`.

    Args:
        op: The operation's name in the kit's list (`quotient`, `share`, ...).
        trace: The result's trace, with its identity and policy set.
        inputs: Each input role and the frame it read. One role named
            `input` is the step's single `input`.
        kept: What the step kept, by count name; a `None` count is left out.
        left_out: The data gaps the step left out.
        unit: The unit the step produces, or `MISSING` when its rows hold
            several units or none, so no one unit names it.
        params: The caller's choices under the names the kit's step vectors
            give them, for a verb whose op-spec names them otherwise.
        period_entities: Each result period with how many entities its total
            covers, for an operation that records coverage.
        entity_coverage: Each ranked entity's ref (`entity_ref`) with how
            many of the window's periods it holds and the window's period
            count, for rank.
        sides: The entities each input role holds (`EntitiesHeld`), for
            compare.

    A `sides` with any side over its bounds (`_sides_representable`) is
    written with every side empty, which a reader takes as entities unknown,
    since dropping some columns could leave a different one telling the
    sides apart.

    A coverage list over `MAX_COVERAGE` entries, or naming a period over
    `PERIOD_MAX_LENGTH` characters or an entity over `TEXT_MAX_LENGTH`, is
    left out whole, since a part of it would not name each period or
    entity, and the step keeps its counts. An entry invalid otherwise
    raises `ValidationError`.
    """
    refs = {
        role: VersionRef(id=ref.spec_id, content_version=ref.content_version)
        for role, ref in inputs.items()
    }
    members: dict[str, object] = {
        "id": trace.op_spec_hash,
        "op": op,
        **({"input": refs["input"]} if set(refs) == {"input"} else {"inputs": refs}),
        "policy": trace.applied_policy.policy_name,
        "params": (
            caller_params(trace.execution_record.content) if params is None else dict(params)
        ),
        "kept": {name: count for name, count in kept.items() if count is not None},
        "left_out": list(left_out),
    }
    if unit is not MISSING:
        members["unit"] = unit
    if op in NEW_VARIABLE_OPS:
        members["stat_var"] = trace.op_spec_hash
    if sides is not None:
        members["sides"] = (
            {role: dict(held) for role, held in sides.items()}
            if _sides_representable(sides)
            else {role: {} for role in sides}
        )
    if period_entities is not None:
        periods = list(period_entities)
        if _representable([period for period, _ in periods], max_length=PERIOD_MAX_LENGTH):
            members["period_coverage"] = [
                PeriodCoverage(period=period, entities=count) for period, count in periods
            ]
    if entity_coverage is not None:
        ranked = list(entity_coverage)
        if _representable([ref for ref, _, _ in ranked], max_length=TEXT_MAX_LENGTH):
            members["entity_coverage"] = [
                EntityCoverage(entity=Ref(ref=ref), periods=periods, of=of)
                for ref, periods, of in ranked
            ]
    return Step.model_validate(members)


def left_out_keys(dropped: Iterable[tuple[ExclusionReason, Sized]]) -> list[LeftOut]:
    """The numerator keys a two-sided result holds no row for, counted as
    entities: one count for each `(reason, keys)` of `dropped` that has
    keys, in the order given."""
    return [
        LeftOut(reason=reason, count=len(keys), of="entities") for reason, keys in dropped if keys
    ]


# The reasons that count rows a verb's inputs held and the step left out as
# data gaps, read from the engine's exclusion records. The caller's own
# selections (`entity_not_selected`, `unit_not_selected`) are parameters.
_ROW_REASONS: tuple[ExclusionReason, ...] = ("unit_unbound", "period_unmatched")


def left_out_from_exclusions(
    exclusions: Exclusions, *, row_reasons: Iterable[ExclusionReason] = _ROW_REASONS
) -> list[LeftOut]:
    """The data gaps `exclusions` records, in first-seen order: each of
    `row_reasons` once, as the values it dropped (`entity_periods`, one per
    row), and `party_total` once per role, as the distinct flow parties of
    that role left out across every input (`pooled_entities`). A total both
    sides of a share over one frame leave out counts once. Other reasons are
    not counted here."""
    counted = set(row_reasons)
    rows: dict[ExclusionReason, int] = {}
    parties: dict[str, set[str]] = {}
    order: dict[tuple[ExclusionReason, str | None], None] = {}
    for record in exclusions.records:
        if record.reason in counted:
            rows[record.reason] = rows.get(record.reason, 0) + record.row_count
            order[record.reason, None] = None
        elif record.reason == "party_total" and record.parties is not None:
            (role,) = record.entities
            parties.setdefault(role, set()).update(record.parties)
            order[record.reason, role] = None
    return [
        LeftOut(reason=reason, count=rows[reason], of="entity_periods")
        if role is None
        else LeftOut(reason=reason, count=len(parties[role]), of="pooled_entities", role=role)
        for reason, role in order
    ]
