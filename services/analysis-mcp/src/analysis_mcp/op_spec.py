"""The op-spec schema: the closed verb set every toolkit tool call expands into.

Eight verbs, closed at the `Verb` enum: `aggregate`, `compare`, `growth`,
`rank`, `share_of`, `join`, `filter`, and `sum_of`. Each verb is its own pydantic model
carrying `frame_ref` plus that verb's own keyword-only parameters -- there is
no single "parameters" bag a caller could stuff arbitrary keys into, because
every model sets `extra="forbid"` and pydantic's own `BaseModel.__init__`
already refuses positional arguments. `OpSpec` unions the eight, discriminated
on `verb`, so parsing a verb outside the closed set (or a positional call, or
an unknown key) refuses at parse time rather than reaching an engine.

Per-verb parameter *content* is deliberately minimal here -- each verb's own
correctness policy (aggregate's null-skip rule, share_of's coverage-match
rule, and so on) belongs to that verb's own implementation, not to this
schema. This schema locks the outer shape: the verb field, the
keyword-only/no-positional/no-unknown-key discipline, and the one piece of
parameter vocabulary shared across verbs today -- `filter`'s row-value
predicates, closed to `OrderedThresholdPredicate`, `CategoryMatchPredicate` and
`PlaceLevelPredicate`.

`frame_ref` is `common.materialisation.FrameRef`: one pair type,
`{spec_id, content_version}`, naming either a Datasets snapshot or a prior
toolkit result -- see that model's own docstring for the two families it
accepts.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, NotRequired, TypedDict, cast

from common.hashed_fields import HASHED_WHEN_SET
from common.materialisation import FrameRef
from common.place_type import PlaceLevel
from dfo_frame_grammar.grammar import (
    TIME_PERIOD_COLUMN,
    UnitRequest,
    validate_time_period,
    validate_unit_request,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
    field_validator,
    model_validator,
    with_config,
)

from analysis_mcp.units import unit_json

_MAX_COLUMN_NAME_CHARS = 200
_MAX_CATEGORY_VALUE_CHARS = 500
_MAX_AGGREGATION_NAME_CHARS = 64
# The longest period the frame grammar accepts is a day-grain span,
# YYYY-MM-DD/YYYY-MM-DD, so an interval operand at any grain reaches the
# filter's typed interval refusal.
_MAX_ORDERED_THRESHOLD_STRING_CHARS = 21
MAX_RANK_LIMIT = 1000


class Verb(StrEnum):
    """The closed verb set. No ninth verb, no synonym, no free-text escape hatch."""

    AGGREGATE = "aggregate"
    COMPARE = "compare"
    GROWTH = "growth"
    RANK = "rank"
    SHARE_OF = "share_of"
    JOIN = "join"
    FILTER = "filter"
    SUM_OF = "sum_of"


class OrderedThresholdPredicate(BaseModel):
    """`column <op> value` over an ordered column: `value >= 1_000_000`, or
    `time_period >= "2015"`, comparing the period string lexically against
    `time_period`'s own zero-padded `YYYY`/`YYYY-MM`/`YYYY-MM-DD` forms,
    which sort in true chronological order at every grain. A coarser period
    compares at its own start, not its own span, so a `"2015"` row satisfies
    `gt "2014"` but fails `gte "2015-06"`.

    `value` is a number against the numeric `value` column, or a string
    against `time_period` or any other column. A string against
    `time_period` must be a well-formed period string, checked here ahead
    of the engine; a string against any other column passes op-spec
    validation unchecked and reaches the engine's own type check instead.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ordered_threshold"] = "ordered_threshold"
    column: str = Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)
    op: Literal["gt", "gte", "lt", "lte", "eq", "ne"]
    value: float | Annotated[str, Field(max_length=_MAX_ORDERED_THRESHOLD_STRING_CHARS)]

    @model_validator(mode="after")
    def _validate_value(self) -> OrderedThresholdPredicate:
        """A numeric operand must be finite. A string operand against
        `time_period` must be a well-formed period string; a string operand
        against any other column passes through unchecked here."""
        if isinstance(self.value, str):
            if self.column == TIME_PERIOD_COLUMN:
                validate_time_period(self.value)
        elif not math.isfinite(self.value):
            raise ValueError("value must be a finite number")
        return self


class CategoryMatchPredicate(BaseModel):
    """`column in values`, over a closed, caller-supplied set of category values."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["category_match"] = "category_match"
    column: str = Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)
    values: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_CATEGORY_VALUE_CHARS)], ...
    ] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _normalise_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """NFC-normalise every value, the same normalisation `Frame.build`
        already applies to every cell it stores. `run_filter` matches a
        `category_match` verbatim against already-NFC cells, and
        `analysis_mcp.identity` separately NFC-normalises this same field
        before hashing -- normalising here, at construction, is the one
        place that feeds both, so an NFD- and an NFC-spelled predicate carry
        identical values into each.
        """
        return tuple(unicodedata.normalize("NFC", value) for value in values)


class PlaceLevelPredicate(BaseModel):
    """`column`, a place's type enrichment, names a place at `level`.

    Reads a `{role}_type` column such as the `observationAbout_type` column a
    `get_data(place_types=true)` frame carries (`Country`,
    `AdministrativeArea1,State`), by `common.place_type.levels_of`: a place sits
    at each of its types, except that one carrying a Eurostat NUTS type is not
    at `Country`. `level="Country"` keeps every country and territory and
    leaves out Åland (`nuts/FI2`), whose figures Finland's include.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["place_level"] = "place_level"
    column: str = Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)
    level: PlaceLevel


# The closed predicate vocabulary: a row-value predicate is one of these
# shapes, discriminated on `kind`, and nothing else. Scope-narrowing
# predicates (roles, windows, facet bindings) are out of this grammar
# entirely. They are Datasets-door input, and a verb's own parameters
# cannot express them (see `FilterOpSpec`'s docstring).
Predicate = Annotated[
    OrderedThresholdPredicate | CategoryMatchPredicate | PlaceLevelPredicate,
    Field(discriminator="kind"),
]

# Rebuilds one predicate from its canonical dict, whichever kind it is.
PredicateAdapter: TypeAdapter[Predicate] = TypeAdapter(Predicate)


@with_config(ConfigDict(extra="forbid"))
class ConvertedUnit(TypedDict):
    """A converted unit, as `get_data` writes it into a frame's `unit` cells:
    `currency` is an upper-case ISO 4217 code, `prices` is `current` or
    `constant`, and `base_year` is present exactly when prices are constant."""

    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    prices: Literal["current", "constant"]
    base_year: NotRequired[int]


def _parse_pinned_unit(value: object) -> str | UnitRequest:
    """A unit pin as the engines compare it against a frame's `unit` cells:
    a unit ref stays a string, and a converted-unit object becomes the
    `UnitRequest` a converted cell holds. `validate_unit_request` decides
    which objects are converted units, the same check a frame's own cells
    pass, so a pin and a cell can never disagree on the shape."""
    if isinstance(value, UnitRequest):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_COLUMN_NAME_CHARS:
            raise ValueError(f"a unit ref has at most {_MAX_COLUMN_NAME_CHARS} characters")
        return value
    if isinstance(value, Mapping):
        request = cast("Mapping[str, object]", value)
        validate_unit_request(request)
        return UnitRequest(
            currency=cast("str", request["currency"]),
            prices=cast("str", request["prices"]),
            base_year=cast("int | None", request.get("base_year")),
        )
    raise ValueError("a unit pin is a unit ref string or a converted-unit object")


# A verb's unit pin: a unit ref, or a converted unit. Both forms are exactly
# what a `unit_heterogeneous` refusal's `fix["units"]` lists, and the op-spec
# dumps a converted unit as that same request object.
PinnedUnit = Annotated[
    str | UnitRequest,
    PlainValidator(
        _parse_pinned_unit,
        json_schema_input_type=Annotated[str, Field(max_length=_MAX_COLUMN_NAME_CHARS)]
        | ConvertedUnit,
    ),
    PlainSerializer(unit_json),
]


class _OpSpecBase(BaseModel):
    """Fields every verb's op-spec carries."""

    model_config = ConfigDict(extra="forbid")

    frame_ref: FrameRef


class AggregateOpSpec(_OpSpecBase):
    """`aggregate`'s own reduction parameters.

    `aggregation` is deliberately a plain, length-bounded `str` rather than a
    pydantic `Literal`. The closed vocabulary (`analysis_mcp.engine.
    aggregate.AGGREGATIONS`) is enforced by the engine at runtime, so a
    caller naming an aggregation outside the closed set reaches the tool's
    own `operation_uncovered` refusal, with the vocabulary listed in `fix`,
    rather than a generic pydantic schema error indistinguishable from a
    malformed request.

    `unit` disambiguates a heterogeneous input frame. Left `None`, aggregate
    refuses over more than one distinct unit (`unit_heterogeneous`, with the
    candidate units in the refusal's `fix["units"]`). A retry supplying one
    of those values here, a unit ref or a converted-unit object, pins the
    aggregate to that unit's rows.
    """

    verb: Literal[Verb.AGGREGATE] = Verb.AGGREGATE
    group_by: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)], ...
    ] = ()
    aggregation: str = Field(min_length=1, max_length=_MAX_AGGREGATION_NAME_CHARS)
    unit: PinnedUnit | None = None


class _SlicePairOpSpec(_OpSpecBase):
    """The two-slice parameters `compare` and `sum_of` share.

    Each side is a *slice*: `frame_ref` (the base class's
    field) plus `left_entity`, a `{role_column: value}` binding narrowing it
    to one entity's own time series -- empty when `frame_ref` already names
    exactly one entity. `right_frame_ref` names the other side; left `None`,
    the other side is a *second slice of the same frame_ref* (`right_entity`
    then does the narrowing), which is what makes "two frame_refs" and "two
    slices of one frame" the same shape with one field left unset rather
    than two different op-specs.

    `left_unit`/`right_unit` pin each side's unit independently, the same
    disambiguation-retry shape `AggregateOpSpec.unit` uses -- one side's
    heterogeneous unit refusal never blocks resolving the other.
    """

    right_frame_ref: FrameRef | None = None
    left_entity: dict[
        Annotated[str, Field(max_length=_MAX_COLUMN_NAME_CHARS)],
        Annotated[str, Field(max_length=_MAX_CATEGORY_VALUE_CHARS)],
    ] = Field(default_factory=dict)
    right_entity: dict[
        Annotated[str, Field(max_length=_MAX_COLUMN_NAME_CHARS)],
        Annotated[str, Field(max_length=_MAX_CATEGORY_VALUE_CHARS)],
    ] = Field(default_factory=dict)
    left_unit: PinnedUnit | None = None
    right_unit: PinnedUnit | None = None


class CompareOpSpec(_SlicePairOpSpec):
    """`compare`'s parameters: two slices, compared over their intersected periods."""

    verb: Literal[Verb.COMPARE] = Verb.COMPARE


class SumOfOpSpec(_SlicePairOpSpec):
    """`sum_of`'s parameters: two slices, added over their intersected periods."""

    verb: Literal[Verb.SUM_OF] = Verb.SUM_OF


class GrowthOpSpec(_OpSpecBase):
    """`growth`'s own endpoint policy parameters.

    `window_start`/`window_end` are whole calendar years (`YYYY`) -- the
    endpoint policy (`analysis_mcp.engine.growth`'s own docstring) reads each
    entity's rows in those two years and counts coverage over the integer
    calendar span between them, so both are pinned to that
    format at the schema layer rather than the general `time_period` grammar
    (which also allows `YYYY-MM`/`YYYY-MM-DD`).

    `entity_by` names the role column(s) identifying each entity whose own
    growth rate is computed -- the same "grouping columns" shape
    `AggregateOpSpec.group_by` uses, empty meaning the whole frame is one
    entity's own series.

    `change` picks the arithmetic: `relative` (the default) computes
    `(value_end - value_start) / value_start`, undefined -- and refused --
    over a zero or sign-crossing base; `absolute` computes
    `value_end - value_start` directly, defined over any base including
    zero or a sign crossing (`analysis_mcp.engine.growth`'s own docstring).
    A `Literal` rather than `AggregateOpSpec.aggregation`'s open,
    engine-enforced `str`: the vocabulary is fixed at two members forever,
    so there is no retry-friendlier engine refusal being traded away by
    catching an unknown mode at the schema layer instead.
    """

    verb: Literal[Verb.GROWTH] = Verb.GROWTH
    window_start: str = Field(pattern=r"^[0-9]{4}$")
    window_end: str = Field(pattern=r"^[0-9]{4}$")
    entity_by: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)], ...
    ] = ()
    unit: PinnedUnit | None = None
    change: Literal["relative", "absolute"] = "relative"

    @model_validator(mode="after")
    def _window_is_ordered(self) -> GrowthOpSpec:
        if int(self.window_end) <= int(self.window_start):
            raise ValueError("window_end must be a later year than window_start")
        return self


class RankOpSpec(_OpSpecBase):
    """`rank`'s own coverage-floor parameters.

    `rank_by` names the role column(s) identifying each ranked entity --
    required and non-empty, since ranking with no entity-identifying column
    would rank a single, undifferentiated row. `aggregation`, left `None`,
    ranks each entity's own single already-present value directly; set to
    one of `analysis_mcp.engine.aggregate.AGGREGATIONS`, it reduces each
    entity's rows first (optionally windowed by `window_start`/`window_end`,
    which -- like `GrowthOpSpec`'s own fields -- must both be set or both be
    `None`) via the aggregate engine before ranking the reduced values.

    `per_period`, left `False` (the default), collapses the window into one
    ranking, one row per entity. Set to
    `True`, the window's periods stay apart: each period is ranked on its
    own, and the output carries one row per entity per period whose own
    `value` is that entity's rank *position* in that period, not the
    measured value the collapsing mode emits (`analysis_mcp.engine.rank`'s
    own module docstring). `aggregation` still means "reduce to one value
    before ranking", now per (entity, period) cell rather than per entity,
    needed only when a cell carries more than one row.

    `limit`, left `None`, keeps every ranked entity. Set, it keeps every
    entity whose rank number is at most `limit`, within each period under
    `per_period`; a value shared at the boundary keeps every entity that
    shares it. A limited per-period ranking's `value` is the measured value,
    not the rank position.
    """

    verb: Literal[Verb.RANK] = Verb.RANK
    rank_by: tuple[Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)], ...] = (
        Field(min_length=1)
    )
    aggregation: str | None = Field(default=None, max_length=_MAX_AGGREGATION_NAME_CHARS)
    window_start: str | None = Field(default=None, pattern=r"^[0-9]{4}$")
    window_end: str | None = Field(default=None, pattern=r"^[0-9]{4}$")
    unit: PinnedUnit | None = None
    ascending: bool = False
    per_period: bool = False
    limit: Annotated[int | None, HASHED_WHEN_SET] = Field(default=None, ge=1, le=MAX_RANK_LIMIT)

    @model_validator(mode="after")
    def _window_bounds_are_paired(self) -> RankOpSpec:
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("window_start and window_end must both be set, or both be None")
        if (
            self.window_start is not None
            and self.window_end is not None
            and int(self.window_end) <= int(self.window_start)
        ):
            raise ValueError("window_end must be a later year than window_start")
        return self


class ShareOfOpSpec(_OpSpecBase):
    """`share_of`'s own numerator/denominator parameters.

    Mirrors `CompareOpSpec`'s two-side shape exactly, renamed left/right to
    numerator/denominator: `frame_ref` (the base class's field) is the
    numerator's own slice, `denominator_frame_ref` left `None` means "the
    other side is a second slice of the same frame_ref" (the same "two
    frame_refs, or two slices of one frame" unification `CompareOpSpec`'s
    own docstring explains). `numerator_entity`/`denominator_entity` narrow
    each side; empty binds nothing, and -- unlike `compare`'s ambiguous-
    slice refusal -- more than one row per period on a side is not an
    error here, it is summed (`analysis_mcp.engine.share_of`'s own
    docstring: "fsum for any summed side").

    `numerator_unit`/`denominator_unit` pin each side's unit independently,
    the same disambiguation-retry shape `CompareOpSpec.left_unit`/
    `right_unit` use.

    `numerator_by` names role column(s), on the numerator's own frame, to
    key the output by -- one row per (key, period) instead of one row per
    period. `denominator_by` pairs each `numerator_by` column with the
    denominator's own column carrying that same role, **by name, never by
    position**: `analysis_mcp.identity._normalise_parameters` and
    `frame_ref_resolution.canonical_columns_or_refusal` both sort every
    array-shaped field, so two independently sorted column lists would
    silently pair the wrong columns whenever the two sides' own
    alphabetical orders disagree. `denominator_by` is a JSON
    object precisely so JCS sorts its *members*, not the pairing itself --
    the same reason `numerator_entity`/`denominator_entity` need no
    canonicalisation of their own beyond key order (see
    `frame_ref_resolution.canonical_entity_binding`'s own docstring).
    `denominator_by` left empty means a pooled denominator, the total every
    numerator key divides by; when non-empty it must name every
    `numerator_by` column, checked below -- a partial mapping is refused at
    the engine, not here, since "every `numerator_by` column has an entry"
    is a claim about the two lists together that only reads naturally once
    both are in hand.
    """

    verb: Literal[Verb.SHARE_OF] = Verb.SHARE_OF
    denominator_frame_ref: FrameRef | None = None
    numerator_entity: dict[
        Annotated[str, Field(max_length=_MAX_COLUMN_NAME_CHARS)],
        Annotated[str, Field(max_length=_MAX_CATEGORY_VALUE_CHARS)],
    ] = Field(default_factory=dict)
    denominator_entity: dict[
        Annotated[str, Field(max_length=_MAX_COLUMN_NAME_CHARS)],
        Annotated[str, Field(max_length=_MAX_CATEGORY_VALUE_CHARS)],
    ] = Field(default_factory=dict)
    numerator_unit: PinnedUnit | None = None
    denominator_unit: PinnedUnit | None = None
    numerator_by: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)], ...
    ] = ()
    denominator_by: dict[
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)],
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)],
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _denominator_by_keys_are_known(self) -> ShareOfOpSpec:
        """Every `denominator_by` key must be a column `numerator_by` itself
        names -- a key outside `numerator_by` names nothing this call could
        ever key its output by. The reverse (`numerator_by` naming a column
        `denominator_by` leaves out) is a partial mapping, refused instead
        at the engine (`ShareOfKeyUnknownError`) -- see this class's own
        docstring for why.
        """
        unknown = sorted(set(self.denominator_by) - set(self.numerator_by))
        if unknown:
            raise ValueError(
                f"denominator_by names column(s) numerator_by does not carry: {unknown}. "
                f"numerator_by: {list(self.numerator_by)}."
            )
        return self


class JoinOpSpec(_OpSpecBase):
    """`join`'s own key parameters.

    Unlike `compare`/`share_of`, `join` always needs two genuinely distinct
    frames -- there is no "two slices of one frame_ref" reading of joining a
    frame with itself, so `right_frame_ref` (the frame joined against
    `frame_ref`, the left side) is required rather than defaulting to
    `None`.

    `join_keys` names the role columns to match rows on, beyond the
    `time_period` every join also matches on implicitly
    (`analysis_mcp.engine.join`'s own docstring). Left empty (the default),
    the engine auto-detects the role columns both frames declare in common;
    a non-empty tuple pins the key list explicitly instead of relying on
    that auto-detection. The key list stays minimal and closed: no facet
    or unit column may be named here, only a role column present on both
    sides.

    `left_unit`/`right_unit` pin each side's unit independently, the same
    shape `CompareOpSpec.left_unit`/`right_unit` use.
    """

    verb: Literal[Verb.JOIN] = Verb.JOIN
    right_frame_ref: FrameRef
    join_keys: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_COLUMN_NAME_CHARS)], ...
    ] = ()
    left_unit: PinnedUnit | None = None
    right_unit: PinnedUnit | None = None


class FilterOpSpec(_OpSpecBase):
    """Row-value predicates only.

    `filter` operates on a frame the materialisation library has already
    narrowed by entity, window, and facet scope. Narrowing that scope is
    Datasets-door input that this verb's parameters cannot express. This
    schema enforces that boundary structurally. `predicates` accepts only the
    closed `Predicate` vocabulary, so a role, a window, or a facet binding has
    no field to occupy even if a caller tried to supply one.
    """

    verb: Literal[Verb.FILTER] = Verb.FILTER
    predicates: tuple[Predicate, ...] = ()


OpSpec = Annotated[
    AggregateOpSpec
    | CompareOpSpec
    | GrowthOpSpec
    | RankOpSpec
    | ShareOfOpSpec
    | JoinOpSpec
    | FilterOpSpec
    | SumOfOpSpec,
    Field(discriminator="verb"),
]

# One shared adapter for parsing an op-spec from wire JSON -- refuses an
# unrecognised `verb` (outside the closed `Literal` set on every branch) the
# same way it refuses any other schema violation.
OpSpecAdapter: TypeAdapter[OpSpec] = TypeAdapter(OpSpec)
