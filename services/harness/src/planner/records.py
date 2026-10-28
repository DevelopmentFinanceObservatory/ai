"""Value types every plan record is built from.

`planner.plan`, `planner.operations` and `planner.workflow` all build on
these, so they live below all three: a `StrictModel` base, the identifier and
year types the records reference each other with, `Period`, which a
requirement states and a discovered source reports, and `EvidenceIssue`, the
finding a workflow carries about a requirement it was approved against.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Literal, override

from common.hashed_fields import HASHED_WHEN_SET
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    JsonValue,
    ValidationError,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema


def _strip_titles(schema: JsonSchemaValue) -> None:
    """Remove every `title` pydantic wrote into one definition's schema.

    A title repeats the property or class name it sits above, spelled out for
    display ("decision_ids" becomes "Decision Ids"), so it tells a model
    reading the schema nothing the name it is keyed by does not. Every model in
    this package carries this as `json_schema_extra`, and pydantic runs it over
    each definition as that definition is finished, before `$defs` is
    assembled, so nothing reads a title back.
    """
    if isinstance(schema, dict):
        schema.pop("title", None)
        for value in schema.values():
            _strip_titles(value)
    elif isinstance(schema, list):
        for value in schema:
            _strip_titles(value)


# A name a plan uses to refer to one of its own parts -- a decision's slot, a
# requirement's id, an operation's id. Lower-case identifiers, so a reference
# compares exactly.
MAX_IDENTIFIER_CHARS = 64
Identifier = Annotated[str, Field(pattern=rf"^[a-z][a-z0-9_]{{0,{MAX_IDENTIFIER_CHARS - 1}}}$")]

# The longest entity name and role phrase a plan or an operation states, and the
# most entities one field lists. A name is the entity in the question's own words
# and a role is a short relational phrase, so both sit several times above any
# value in the plans the eval sets under `evals/` record. The plan renders them
# in full into the planner's brief, which is what these limits bound. The entity
# count is data-mcp's own limit. `get_data` takes 50 entity names per call
# (`_MAX_ENTITY_NAMES` in `data_mcp/get_data_tool.py`), so a plan can state any
# requirement the tool can carry out, including one that names all 27 EU members.
MAX_ENTITY_NAME_CHARS = 200
MAX_ENTITY_ROLE_CHARS = 100
MAX_ENTITIES = 50

# A calendar year, as a requirement states its period.
Year = Annotated[int, Field(ge=1900, le=2100)]

# How many places a rank keeps, as a rank requirement and its operation state
# it. A component draws at most 24 series (the apis service's `MAX_SERIES_PER_COMPONENT`),
# while the narration reads a longer ranking, and a question past 100 places asks
# for a listing rather than a ranking.
MAX_RANK_LIMIT = 100
RankLimit = Annotated[int, Field(ge=1, le=MAX_RANK_LIMIT)]

# A dataset definition's id as data-mcp's `search` mints it: a hash of the
# definition's canonical identity, which is what makes one stable enough to
# approve.
SpecId = Annotated[str, Field(pattern=r"^spec_[0-9a-f]{32}$")]

# Whether one of a Search part's scope choice options can be served
# (`harness.scope_choices`). A plan's scope decision states it too
# (`planner.plan.DecisionScope`), beside the choice's axis
# (`data_mcp.scope.ScopeAxis`).
OptionStatus = Literal["available", "partial", "unavailable"]


class StrictModel(BaseModel):
    """A frozen record that rejects fields it does not declare.

    Every part of a plan is one, so that a plan's own JSON schema is the whole
    of what a model reads to write one, and carries nothing the model cannot
    act on. A revision is a value that `planner.plan.plan_revision` hashes, so
    nothing may change in place and nothing may ride along unnamed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_strip_titles)


def rejected_fields(errors: Iterable[ValidationError]) -> list[str]:
    """The fields `errors` reject, each as its dotted path, sorted and named once.

    A list index is left out of a path, so one field rejected in several
    entries is one name. A path names a field and never its value, so a
    caller may log it for stored state whose content it must not log.
    """
    return sorted(
        {
            ".".join(part for part in error["loc"] if isinstance(part, str))
            for exc in errors
            for error in exc.errors(include_input=False, include_url=False)
        }
    )


def warn_left_out(
    logger: logging.Logger,
    state_key: str,
    errors: Sequence[ValidationError],
    *,
    entries: int = 1,
    detail: str = "",
    counts: Sequence[int] = (),
) -> None:
    """Log on `logger` that a reader left `entries` stored entries out of `state_key`.

    The warning names the state key and the fields `errors` reject
    (`rejected_fields`), and no stored value. `detail` is a `%d` format that
    breaks `entries` down, filled from `counts`.
    """
    noun = "entry" if entries == 1 else "entries"
    breakdown = f" ({detail})" if detail else ""
    logger.warning(
        f"%s: left out %d stored {noun}{breakdown} whose shape is not the current one; "
        "rejected fields: %s",
        state_key,
        entries,
        *counts,
        ", ".join(rejected_fields(errors)),
    )


class TaggedModel(StrictModel):
    """A `StrictModel` whose JSON schema lists its `kind` tag as required.

    Each member of a discriminated union declares `kind` with a default, so
    code constructing one gets the tag for free. Pydantic drops a defaulted
    field from the schema's `required`, though, and a discriminated union
    genuinely needs its tag: a model proposing from that schema omits `kind`
    and gets `union_tag_not_found` back.
    """

    @classmethod
    @override
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler, /
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        schema = handler.resolve_ref_schema(json_schema)
        if "kind" in schema.get("properties", {}):
            others = [name for name in schema.get("required", []) if name != "kind"]
            schema["required"] = ["kind", *others]
        return json_schema


class Period(StrictModel):
    """An inclusive range of calendar years."""

    start: Year = Field(description="The first year this period covers.")
    end: Year = Field(description="The last year this period covers.")

    @model_validator(mode="after")
    def _ends_no_earlier_than_it_starts(self) -> Period:
        if self.end < self.start:
            raise ValueError(f"period ends ({self.end}) before it starts ({self.start})")
        return self

    def covers(self, other: Period) -> bool:
        """Whether every year of `other` lies within this period."""
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: Period) -> bool:
        """Whether this period and `other` share at least one year."""
        return self.start <= other.end and other.start <= self.end


# --- Calculation vocabulary ---------------------------------------------------
#
# Shared by the plan's derived requirements and the translator's operations, so
# that an operation's fields compare with a requirement's without a translation
# table. Each names what analysis-mcp computes; the compiler writes the columns.


class TitlelessEnum(StrEnum):
    """A `StrEnum` whose JSON schema omits pydantic's title for it.

    `GenerateJsonSchema.enum_schema` opens an enum's schema with the class name
    as a `title`, which a model's `json_schema_extra` cannot reach. Every enum the
    planner's schema reaches inherits this, in this module, `planner.plan` and
    `planner.workflow`, so the key naming the enum in `$defs` is the only place
    its name appears.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler, /
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema.pop("title", None)
        return json_schema


class Reduction(TitlelessEnum):
    """How analysis-mcp collapses several observations into one figure.

    `sum` and `mean` total or average the values in a group, and `max` and
    `min` take the extreme.
    """

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class GroupBy(TitlelessEnum):
    """A column an aggregate groups by: `period` by year, `entity` by entity."""

    PERIOD = "period"
    ENTITY = "entity"
    BREAKDOWN = "breakdown"


class BreakdownDimension(TitlelessEnum):
    """A hierarchical classification a retrieval can break its total down by.

    `sector` is the OECD CRS purpose codes.
    """

    SECTOR = "sector"


class Breakdown(StrictModel):
    """One classification a retrieval keeps, its codes rolled up to one level."""

    dimension: BreakdownDimension = Field(
        description="The classification. `sector` is the OECD CRS purpose codes."
    )
    level: int = Field(
        ge=0,
        description="The level codes roll up to: 0 is the top (the ten sector groups).",
    )


class StatVarReading(TitlelessEnum):
    """How a retrieval reads a source that binds a family of StatVars.

    `sum` reads the family as one series, its members summed by apis at
    retrieval. `separate` keeps one series per member StatVar.
    """

    SUM = "sum"
    SEPARATE = "separate"


class RankDirection(TitlelessEnum):
    """Which end of a ranking comes first: `highest_first` ranks the largest value 1st, `lowest_first` the smallest."""

    HIGHEST_FIRST = "highest_first"
    LOWEST_FIRST = "lowest_first"


class RankAmong(TitlelessEnum):
    """Who a rank compares an entity against.

    `named` ranks exactly the entities the upstream requirement names on the
    axis, the visible list a filter has narrowed to. `population` ranks every
    entity the retrieved definition holds on the axis, with the named entities
    as the focus.
    """

    NAMED = "named"
    POPULATION = "population"


class GrowthChange(TitlelessEnum):
    """How a growth reads the difference between two endpoint values.

    `relative` divides by the start value, so a share moving 10% to 15% reads
    +50%. `absolute` subtracts, in the input's own unit, so the same move reads
    +5 percentage points.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class DenominatorScope(TitlelessEnum):
    """Whose total a keyed `share_of` divides by.

    `per_axis` divides each key by that same key's own denominator value, a
    donor's share of its own total. `pooled` divides every key by one
    denominator value summed across the whole population, a creditor's lending
    over one GDP series.
    """

    PER_AXIS = "per_axis"
    POOLED = "pooled"


# --- Rolling periods ----------------------------------------------------------
#
# A requirement states its years as calendar years, read from the data Search
# found. `RelativePeriod` says beside a year that it follows the newest period
# the data holds, so a card drawn from the plan moves as new years arrive. The
# year stays on the record as the reading at planning time, and every rule that
# compares years reads it.

# The furthest back a relative period reaches, which is the span `Year` allows.
MIN_ROLLING_OFFSET = -200


class RollingIssue(TitlelessEnum):
    """Why a plan's rolling periods cannot be recorded, as `RollingPlanError` names it."""

    # A period that follows the newest data reads a dataset retrieved with a
    # closed end, or one that cannot be retrieved with an open one.
    NEEDS_OPEN_PERIOD = "rolling_needs_open_period"
    # Rolling years a run reads from one newest year stand for different
    # newest years, or for one after the last year the datasets they read
    # hold, so the years stated beside them are not the ones it reads.
    YEARS_DISAGREE = "rolling_years_disagree"
    # A first year that follows the newest data meets a fixed last year, so
    # newer data leaves no year (`planner.shapes.years_run_out`), or newer
    # data takes a growth window's first or last year out of its input
    # (`planner.shapes.growth_endpoint_leaves`).
    YEARS_RUN_OUT = "rolling_years_run_out"
    # A two-sided step pairs a side whose years follow the newest year with
    # a side open to each series' newest year that the newest year is not
    # resolved over (`planner.shapes.anchor_misses`).
    SIDE_UNANCHORED = "rolling_side_unanchored"
    # A filter's `ne` follows the newest data, or removes a year from years
    # newer data moves (a dataset retrieved with `period_end_open`).
    EXCLUSION_ROLLS = "rolling_exclusion"
    # The dataset's whole history holds more rows than one frame may.
    FRAME_TOO_LARGE = "rolling_frame_too_large"


class OpenPeriodFault(TitlelessEnum):
    """Which part of `RollingIssue.NEEDS_OPEN_PERIOD` holds."""

    # The dataset the relative period reads does not state `period_end_open`.
    DATASET_CLOSED = "dataset_closed"
    # The dataset is pinned to no definition on record, so it is found by a
    # search over its stated years.
    NO_SOURCE_ON_RECORD = "no_source_on_record"
    # No Search answer on record states the first year the dataset holds.
    FIRST_YEAR_UNKNOWN = "first_year_unknown"
    # The read of the dataset over no window did not complete, or its answer
    # states no first year.
    FIRST_YEAR_READ_FAILED = "first_year_read_failed"


class RollingPlanError(ValueError):
    """A plan rule refused a rolling period.

    `issue` names the rule, `requirement_ids` the requirements to change, and
    `fault` the part of `RollingIssue.NEEDS_OPEN_PERIOD` that holds, so
    `planner.agent.propose_plan` answers with a code a caller can branch on.
    `spec_id` is the pinned dataset whose first year is missing, under
    `OpenPeriodFault.FIRST_YEAR_UNKNOWN` and `FIRST_YEAR_READ_FAILED`.
    """

    def __init__(
        self,
        issue: RollingIssue,
        message: str,
        *,
        requirement_ids: Sequence[str],
        fault: OpenPeriodFault | None = None,
        spec_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = issue
        self.requirement_ids = tuple(requirement_ids)
        self.fault = fault
        self.spec_id = spec_id


class RelativePeriod(StrictModel):
    """A year stated against the newest year the data holds."""

    anchor: Literal["latest"] = Field(
        default="latest",
        description="Always 'latest': the newest year the data holds.",
    )
    offset: int = Field(
        ge=MIN_ROLLING_OFFSET,
        le=0,
        strict=True,
        description="Years back from the newest: 0 is the newest year, -5 five years before it.",
    )

    def latest_implied_by(self, year: int) -> int:
        """The newest year under which this relative period is `year`."""
        return year - self.offset


class RollingWindow(StrictModel):
    """Which ends of a window of years follow the newest data, and how far back each sits."""

    window_start: RelativePeriod | None = Field(
        default=None,
        description="The start as a relative period; unset for a fixed year.",
    )
    window_end: RelativePeriod | None = Field(
        default=None,
        description="The end as a relative period; unset for a fixed year.",
    )

    @model_validator(mode="after")
    def _names_a_bound(self) -> RollingWindow:
        if self.window_start is None and self.window_end is None:
            raise ValueError(
                "a rolling window names window_start, window_end or both; leave `rolling` "
                "unset for a window of fixed years"
            )
        return self


def check_rolling_window(
    rolling: RollingWindow | None,
    *,
    window_start: int | None,
    window_end: int | None,
    error: Callable[[str], Exception],
) -> None:
    """Raise `error(fault)` when `rolling` names a bound of a window that states no years.

    The newest year each relative bound stands for is checked across the
    plan, every rolling year a run reads together against one newest year
    (`planner.plan._check_rolling_years_agree`).
    """
    if rolling is not None and (window_start is None or window_end is None):
        raise error("its `rolling` names a bound of a window it does not state")


ThresholdOp = Literal["gt", "gte", "lt", "lte", "eq", "ne"]


class ValueThreshold(TaggedModel):
    """Keep rows whose value compares to `value` under `op`."""

    kind: Literal["value_threshold"] = Field(
        default="value_threshold", description="Tags this predicate as a value threshold."
    )
    op: ThresholdOp = Field(
        description="How a row's value compares to `value`: gt, gte, lt, lte, eq or ne."
    )
    value: float = Field(description="The value a row's value is compared against.")


class PeriodThreshold(TaggedModel):
    """Keep rows whose year compares to `year` under `op`."""

    kind: Literal["period_threshold"] = Field(
        default="period_threshold", description="Tags this predicate as a period threshold."
    )
    op: ThresholdOp = Field(
        description="How a row's year compares to `year`: gt, gte, lt, lte, eq or ne."
    )
    year: Year = Field(description="The year a row's year is compared against.")
    rolling: Annotated[RelativePeriod | None, HASHED_WHEN_SET] = Field(
        default=None,
        description=(
            "Set when `year` follows the newest data; `year` still states the year. Needs "
            "`period_end_open` on the dataset requirement read."
        ),
    )

    @model_validator(mode="after")
    def _keeps_the_newest_year_reachable(self) -> PeriodThreshold:
        """Refuse `gt` at offset 0, which keeps only the years after the newest.

        The run's newest year is the newest at least half the series hold
        (`executor.run`), so the years after it belong to the fewer series
        reporting past it, and to none when every series ends on it.
        Offsets are 0 or less, so no other rolling bound sets a first
        year after the newest.
        """
        if self.op == "gt" and self.rolling is not None and self.rolling.offset == 0:
            raise ValueError(
                "op `gt` with `rolling` at offset 0 keeps only the years after the newest year, "
                "which most series do not hold; keep the newest year on with op `gte` at offset 0"
            )
        return self


class EntityIn(TaggedModel):
    """Keep rows whose entity name is one of `names`."""

    kind: Literal["entity_in"] = Field(
        default="entity_in", description="Tags this predicate as an entity membership filter."
    )
    names: tuple[Annotated[str, Field(max_length=MAX_ENTITY_NAME_CHARS)], ...] = Field(
        min_length=1,
        max_length=MAX_ENTITIES,
        description="The entity names a row's entity has to be one of, to survive.",
    )
    role: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description="Narrows the match to entities of this role, for an input carrying more than one.",
    )


class PlaceType(TitlelessEnum):
    """A level of place, named by its Data Commons type.

    A requirement that names no place retrieves every place with data at
    every level: the world, UN regions (`UNGeoRegion`), World Bank groups
    (`ProvisionalNode`), countries and their subdivisions. Regions and
    groups contain one another, so their types name no level. A place is at
    a level when this type is one of its types, except that a place
    carrying a Eurostat NUTS type is not a country.
    """

    COUNTRY = "Country"
    CONTINENT = "Continent"
    ADMINISTRATIVE_AREA_1 = "AdministrativeArea1"
    ADMINISTRATIVE_AREA_2 = "AdministrativeArea2"
    CITY = "City"
    EUROSTAT_NUTS_1 = "EurostatNUTS1"
    EUROSTAT_NUTS_2 = "EurostatNUTS2"
    EUROSTAT_NUTS_3 = "EurostatNUTS3"


class PlaceLevel(TaggedModel):
    """Keep rows whose role's entities are at `level`."""

    kind: Literal["place_level"] = Field(
        default="place_level", description="Tags this predicate as a level of place."
    )
    level: PlaceType = Field(
        description=(
            "The one level of place to keep, by Data Commons type: 'Country', 'Continent', "
            "'AdministrativeArea1' (states, provinces), 'AdministrativeArea2' (districts, "
            "counties), 'City', or 'EurostatNUTS1' to 'EurostatNUTS3'. A requirement that "
            "names no place retrieves every place with data at every level: the world, UN "
            "regions ('UNGeoRegion'), World Bank groups ('ProvisionalNode'), countries and "
            "their subdivisions. Regions and groups contain one another, so their types name "
            "no level. A place is at a level when this type is one of its types, except that "
            "a place carrying a Eurostat NUTS type is not a country."
        )
    )
    role: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description=(
            "The role whose entities' types this keeps. Omit only when the input has one "
            "resolvable entity role; old place filters omit it."
        ),
    )
    strict: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        description=(
            "Refuse if any entity on this role has unknown typing. New country-population "
            "filters set this true; omitted legacy filters retain their exclude-unknown policy."
        ),
    )


Predicate = Annotated[
    ValueThreshold | PeriodThreshold | EntityIn | PlaceLevel, Field(discriminator="kind")
]


# --- Evidence findings --------------------------------------------------------
#
# What `compiler.evidence.check_evidence` reports about one requirement. Defined
# here rather than in the compiler because a `Workflow` carries the non-blocking
# findings it was approved with, and `planner.workflow` cannot import the
# compiler back. `compiler.evidence` re-exports all three.


class Severity(TitlelessEnum):
    """How a finding bears on the requirement."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class EvidenceIssueCode(TitlelessEnum):
    """What `check_evidence` found."""

    NO_DATA_AT_SOURCE = "no_data_at_source"
    DATA_UNVERIFIED = "data_unverified"
    IDENTITY_UNVERIFIED = "identity_unverified"
    ENTITY_MISSING = "entity_missing"
    ENTITY_UNVERIFIED = "entity_unverified"
    ENTITY_ROLE_AMBIGUOUS = "entity_role_ambiguous"
    ENTITY_ROLE_MISMATCH = "entity_role_mismatch"
    ENTITY_ROLE_UNVERIFIED = "entity_role_unverified"
    GROUP_READ_AS_TOTAL = "group_read_as_total"
    OPEN_AXIS_UNRESOLVED = "open_axis_unresolved"
    WINDOW_SHORT = "window_short"
    PERIOD_UNVERIFIED = "period_unverified"
    SCOPE_ENTITIES_UNBOUND = "scope_entities_unbound"
    SCOPE_ENTITIES_WIDER = "scope_entities_wider"
    SCOPE_WINDOW_SHORT = "scope_window_short"
    UNIT_MATCHES = "unit_matches"
    UNIT_MISMATCH = "unit_mismatch"
    UNIT_UNVERIFIED = "unit_unverified"
    UNIT_PRICE_BASIS_MISMATCH = "unit_price_basis_mismatch"
    UNIT_PRICE_BASIS_UNVERIFIED = "unit_price_basis_unverified"
    SOURCE_PIN_MISMATCH = "source_pin_mismatch"
    SOURCE_MISMATCH = "source_mismatch"
    SOURCE_UNVERIFIED = "source_unverified"
    # What a retrieval's own manifest reports, once a run has one to read.
    SOURCE_DRIFT = "source_drift"
    STAT_VAR_MEMBERSHIP_DRIFT = "stat_var_membership_drift"
    # No check reports this code. A run recorded while `get_data` answered a
    # described identity under a second spec_id holds it among its evidence
    # findings (`executor.outcomes.EvidenceRecord.gaps`), so the member stays
    # for a saved session to load.
    SOURCE_SUBSTITUTED = "source_substituted"
    ROWS_EMPTY = "rows_empty"
    STATISTICS_UNAVAILABLE = "statistics_unavailable"
    COVERAGE_GAPS = "coverage_gaps"
    GRAIN_AMBIGUOUS = "grain_ambiguous"


class SeriesEntity(StrictModel):
    """One entity of a retrieved series, as its statistics name it.

    `role` and `ref` are the series' own; `label` is the name the approved
    source gives `ref`, `None` where it gives none.
    """

    role: str = Field(description="The series' role column for this entity.")
    ref: str = Field(description="The entity's value in that column.")
    label: str | None = Field(
        default=None, description="The approved source's name for `ref`, when it names one."
    )


class EvidenceIssue(StrictModel):
    """One finding about one requirement."""

    code: EvidenceIssueCode = Field(description="What `check_evidence` found.")
    severity: Severity = Field(
        description="How this finding bears on the requirement: error, warning or info."
    )
    requirement_id: Identifier = Field(description="The requirement this finding is about.")
    detail: str = Field(description="A human-readable explanation of this finding.")
    found: JsonValue = Field(
        default=None,
        description="What was actually found, when this finding compares an expectation to an observation.",
    )
    required: JsonValue = Field(
        default=None,
        description="What was required, when this finding compares an expectation to an observation.",
    )
    # `stat_var`, `entities` and `magnitude` are left out of a dump while unset, which
    # keeps the findings a model reads, and every stored one, unchanged.
    stat_var: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        exclude_if=lambda stat_var: stat_var is None,
        description=(
            "The StatVar of the one retrieved series a `window_short` or `coverage_gaps` "
            "finding is about."
        ),
    )
    entities: Annotated[tuple[SeriesEntity, ...], HASHED_WHEN_SET] = Field(
        default=(),
        exclude_if=lambda entities: not entities,
        description=(
            "The entities of the one retrieved series a `window_short` or `coverage_gaps` "
            "finding is about."
        ),
    )
    magnitude: Annotated[float | None, HASHED_WHEN_SET] = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "The largest absolute value the one retrieved series a `window_short` or "
            "`coverage_gaps` finding is about holds in any period, which ranks the series "
            "an answer names."
        ),
    )


class ResolvedSourceMention(StrictModel):
    """The source constraint Search resolved for the part producing a source."""

    text: str | None = None
    ref: str | None = None
    polarity: Literal["include", "exclude"] = "include"


class SourceNameCandidate(StrictModel):
    """An admitted source reference offered for ambiguous source wording."""

    ref: str
    label: str
    kind: str


class AmbiguousSourceReading(StrictModel):
    """One ambiguous wording and the source references Search offers for it."""

    text: str
    candidates: tuple[SourceNameCandidate, ...]


class EntityCandidate(StrictModel):
    """A graph node Search offers for ambiguous entity wording: its dcid and label.

    `restate_as` is set when the candidate is a group's own node on a row
    asked for its members: Search's option posts the group's registered name
    with `members` and no dcid, since a pinned group node cannot ask for its
    members, so the entity states that name in place of the `ref`. It is left
    out of a dump while unset.

    `pick` is the id an ask offers the candidate under, naming the
    requirement and the reading as well as the candidate (`entity_pick`). The
    referral holding the reading fills it (`AmbiguousEntityReading.picked_under`),
    and it is left out of a dump while unset.
    """

    ref: str
    label: str
    restate_as: str | None = Field(default=None, exclude_if=lambda name: name is None)
    pick: str = Field(default="", exclude_if=lambda pick: not pick)


class EntityKey(StrictModel):
    """A requirement entity as `(name, role)`, the entity an ambiguous reading is about."""

    name: str
    role: str | None = None


def entity_pick(requirement_id: str, entity: EntityKey, choice: str) -> str:
    """The id an ask offers a candidate under for the reading about `entity` in `requirement_id`.

    The requirement, then the entity's role, then its name, then `choice`,
    the candidate's node or the group name it is restated as
    (`EntityCandidate.restate_as`), so an answer naming it says which entity
    of which requirement the user settled as well as how:
    "flows|donor|Congo|country/COD" and "flows|recipient|Congo|country/COD"
    are two different answers, and a later referral about the same
    requirement offers the same ids for the same readings. An entity with no
    role leaves its field empty.
    """
    return f"{requirement_id}|{entity.role or ''}|{entity.name}|{choice}"


class AmbiguousEntityReading(StrictModel):
    """One requirement entity whose words Search cannot tie to one graph node,
    with the nodes it offers.

    Each reading is one decision. Discovery sends rows by text alone, so rows
    written alike share one identity and one Search group: "aid from Congo to
    Congo" is one group, read as two readings, one per end.
    """

    text: str
    candidates: tuple[EntityCandidate, ...]
    entity: EntityKey

    def picked_under(self, requirement_id: str) -> AmbiguousEntityReading:
        """This reading with each candidate's `pick` naming it within `requirement_id` (`entity_pick`)."""
        return self.model_copy(
            update={
                "candidates": tuple(
                    candidate.model_copy(
                        update={
                            "pick": entity_pick(
                                requirement_id, self.entity, candidate.restate_as or candidate.ref
                            )
                        }
                    )
                    for candidate in self.candidates
                )
            }
        )
