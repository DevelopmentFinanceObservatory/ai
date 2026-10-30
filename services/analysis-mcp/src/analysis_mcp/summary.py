"""The facts of one verb call, as dfo-apis' analysis runner states them.

`POST /api/datasets/analysis` answers a computed tree with the `summary` of
its top step (`dfo_analysis_engine.calls.VerbCall.summary` in dfo-apis): the
verb, its canonical op-spec `parameters`, every `result.VerbResult` field,
the execution identity, the derived marker, and `result`, every field of the
engine's result dataclass but its frame, as JSON. A unit is its pin form
(`dfo_frame_grammar.grammar.unit_cell_to_json`), and join's matched keys are
their rows.

`VerbSummary.parse` reads that object into the typed facts `digest` renders
from. Each `*Facts` model names the result fields its digest reads, under the
engine dataclass's own field names, and ignores the rest, so a field the
engine adds reaches no digest until a renderer reads it. The exclusions and
the time the summary carries ignore added fields the same way, and the
parameters a digest reads (`*Parameters`) are checked as the facts are, so a
summary that lacks one is refused as an answer outside the contract.

A join's matched keys and a filter's kept rows can number in the hundreds of
thousands, and a digest shows `LIST_CAP` of them. `JoinFacts` and `FilterFacts`
keep only the entries the digest shows (`capped`), chosen from the summary's
JSON before any is read into a model, beside the count of every entry.

A result in the paired form a `compare` or `sum_of` with `on` gives
(`dfo_analysis_engine.engine.paired.PairedResult`) has no facts model here,
since these tools send no `on`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal

from dfo_frame_grammar.grammar import ComposedUnit, UnitCell, unit_cell_from_json
from pydantic import BaseModel, ConfigDict, Field, PlainValidator, create_model, model_validator

from analysis_mcp.artifact import Exclusion, Interval, Observation
from analysis_mcp.execution_identity import ExecutionIdentity
from analysis_mcp.result import VerbResult

type KeyPairs = tuple[tuple[str, str], ...]
type KeyPeriods = tuple[KeyPairs, tuple[str, ...]]


def _unit(value: object) -> UnitCell | None:
    return unit_cell_from_json(value)


def _composed_unit(value: object) -> ComposedUnit:
    unit = unit_cell_from_json(value)
    if not isinstance(unit, ComposedUnit):
        raise ValueError(f"a composed unit is a {{numerator, denominator}} object, not {value!r}")
    return unit


# A unit cell read from its pin form.
Unit = Annotated[UnitCell | None, PlainValidator(_unit)]

# The most entries of one list a digest shows.
LIST_CAP = 20


def capped[T](items: Sequence[T], *, magnitude: Callable[[T], float]) -> Sequence[T]:
    """Up to `LIST_CAP` entries of `items`, chosen by descending
    `abs(magnitude(entry))` rather than by position -- for a renderer whose
    order is not the caller's stated criterion, the largest-magnitude
    entries are the most likely to be what a model needs from a truncated
    digest. Ties go to the earlier entry.

    Kept entries are returned in `items`' own original order, not sorted by
    magnitude, so the shown block still reads in the order its verb
    promises.
    """
    if len(items) <= LIST_CAP:
        return items
    kept_positions = sorted(
        sorted(range(len(items)), key=lambda i: abs(magnitude(items[i])), reverse=True)[:LIST_CAP]
    )
    return [items[i] for i in kept_positions]


def _shown(data: object, *, entries: str, count: str, magnitude: Callable[[Any], float]) -> object:
    """`data`, a facts object as JSON, with its list `entries` cut to the entries a
    digest shows (`capped`) and `count` set to the length of the whole list."""
    if not isinstance(data, Mapping) or not isinstance(listed := data.get(entries), list):
        return data
    try:
        shown = capped(listed, magnitude=magnitude)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"an entry of {entries} has no number to order it by") from exc
    return {**data, entries: list(shown), count: len(listed)}


class _Facts(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class SummaryExclusion(Exclusion):
    """An `Exclusion` as a summary states it, ignoring fields dfo-apis adds."""

    model_config = ConfigDict(extra="ignore")


class SummaryObservation(Observation):
    """An `Observation` as a summary states it, ignoring fields dfo-apis adds."""

    model_config = ConfigDict(extra="ignore")


class SummaryInterval(Interval):
    """An `Interval` as a summary states it, ignoring fields dfo-apis adds."""

    model_config = ConfigDict(extra="ignore")


SummaryTime = Annotated[SummaryObservation | SummaryInterval, Field(discriminator="kind")]


class PooledParties(_Facts):
    role: str
    count: int
    total_count: int
    periods: int
    private_count: int | None = None
    private_total_count: int | None = None
    private_value: float | None = None


class GroupSummary(_Facts):
    group_key: KeyPairs
    value: float
    row_count: int
    time_period_min: str
    time_period_max: str
    value_period: str | None = None


class AggregateFacts(_Facts):
    resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    realised_coverage_fraction: float
    skipped_cell_count: int
    expected_cell_count: int
    present_cell_count: int
    group_summaries: tuple[GroupSummary, ...]
    period_label: str | None = None
    pooled_parties: tuple[PooledParties, ...] = ()


class EntityGrowth(_Facts):
    entity_key: KeyPairs
    value_start: float
    value_end: float
    growth_rate: float


class GrowthFacts(_Facts):
    resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    entity_growths: tuple[EntityGrowth, ...]
    change: Literal["relative", "absolute"]


class EntityRank(_Facts):
    entity_key: KeyPairs
    value: float
    rank: int
    row_count: int
    coverage_fraction: float
    time_period: str


class RankFacts(_Facts):
    resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    aggregation: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    periods_in_window: tuple[str, ...]
    overall_coverage_fraction: float
    entity_ranks: tuple[EntityRank, ...]
    per_period: bool
    limit: int | None = None
    private_parties: KeyPairs = ()

    @property
    def kept_ranks(self) -> tuple[EntityRank, ...]:
        """The ranks within `limit`, which the output frame holds."""
        return tuple(r for r in self.entity_ranks if self.limit is None or r.rank <= self.limit)


class PeriodComparison(_Facts):
    time_period: str
    left_value: float
    right_value: float
    delta: float


class CompareFacts(_Facts):
    stat_var: str
    left_stat_vars: tuple[str, ...]
    right_stat_vars: tuple[str, ...]
    left_entities: dict[str, str]
    right_entities: dict[str, str]
    left_resolved_unit: Unit = None
    right_resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    intersected_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    period_comparisons: tuple[PeriodComparison, ...]


class PeriodSum(_Facts):
    time_period: str
    left_value: float
    right_value: float
    value: float


class PeriodLeftOut(_Facts):
    time_period: str
    side: Literal["left", "right"]
    value: float


class SumOfFacts(_Facts):
    stat_var: str
    left_stat_vars: tuple[str, ...]
    right_stat_vars: tuple[str, ...]
    unit: Unit = None
    warnings: tuple[str, ...] = ()
    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    intersected_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    period_sums: tuple[PeriodSum, ...]
    left_out: tuple[PeriodLeftOut, ...] = ()


class PeriodShare(_Facts):
    time_period: str
    numerator_value: float
    denominator_value: float
    percent: float
    key: KeyPairs = ()


class _TwoSidedFacts(_Facts):
    """The fields `share_of` and `quotient_of` digests read alike."""

    stat_var: str
    numerator_stat_var: str
    denominator_stat_var: str
    numerator_resolved_unit: Unit = None
    denominator_resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    numerator_periods: tuple[str, ...]
    denominator_periods: tuple[str, ...]
    matched_periods: tuple[str, ...]
    numerator_coverage_fraction: float
    denominator_coverage_fraction: float
    unmatched_keys: tuple[KeyPairs, ...] = ()
    keys_without_shared_period: tuple[KeyPairs, ...] = ()
    numerator_only_periods: tuple[KeyPeriods, ...] = ()
    denominator_only_periods: tuple[KeyPeriods, ...] = ()
    numerator_pooled: tuple[PooledParties, ...] = ()
    denominator_pooled: tuple[PooledParties, ...] = ()


class ShareOfFacts(_TwoSidedFacts):
    period_shares: tuple[PeriodShare, ...]


class PeriodQuotient(_Facts):
    time_period: str
    numerator_value: float
    denominator_value: float
    quotient: float
    key: KeyPairs = ()


class ZeroDenominatorCell(_Facts):
    time_period: str
    numerator_value: float
    key: KeyPairs = ()


class QuotientOfFacts(_TwoSidedFacts):
    unit: Annotated[ComposedUnit, PlainValidator(_composed_unit)]
    period_quotients: tuple[PeriodQuotient, ...]
    keys_with_zero_denominator: tuple[KeyPairs, ...] = ()
    zero_denominator_cells: tuple[ZeroDenominatorCell, ...] = ()


class MatchedKey(_Facts):
    key: KeyPairs
    time_period: str
    left_value: float
    right_value: float


class JoinFacts(_Facts):
    """`matched_keys` holds the keys a digest shows, and `matched_key_count` counts
    every matched key."""

    left_resolved_unit: Unit = None
    right_resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    join_keys: tuple[str, ...]
    right_key_columns: tuple[str, ...]
    matched_keys: tuple[MatchedKey, ...]
    matched_key_count: int = Field(ge=0)
    left_dropped_key_count: int
    right_dropped_key_count: int

    @model_validator(mode="before")
    @classmethod
    def _shown_keys(cls, data: object) -> object:
        return _shown(
            data,
            entries="matched_keys",
            count="matched_key_count",
            magnitude=lambda key: max(abs(key["left_value"]), abs(key["right_value"])),
        )


class FilterRowSummary(_Facts):
    row_key: KeyPairs
    value: float
    unit: Unit = None


class FilterFacts(_Facts):
    """`kept_row_summaries` holds the rows a digest shows, and
    `kept_row_summary_count` counts every kept row's summary."""

    total_row_count: int
    kept_row_count: int
    dropped_row_count: int
    kept_row_summaries: tuple[FilterRowSummary, ...]
    kept_row_summary_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _shown_rows(cls, data: object) -> object:
        return _shown(
            data,
            entries="kept_row_summaries",
            count="kept_row_summary_count",
            magnitude=lambda row: abs(row["value"]),
        )


type ResultFacts = (
    AggregateFacts
    | GrowthFacts
    | RankFacts
    | CompareFacts
    | SumOfFacts
    | ShareOfFacts
    | QuotientOfFacts
    | JoinFacts
    | FilterFacts
)


class AggregateParameters(_Facts):
    aggregation: str
    group_by: tuple[str, ...] = ()


class GrowthParameters(_Facts):
    window_start: str
    window_end: str
    entity_by: tuple[str, ...] = ()


class RankParameters(_Facts):
    rank_by: tuple[str, ...] = ()


class FilterParameters(_Facts):
    predicates: tuple[Any, ...] = ()


class _NoParameters(_Facts):
    pass


# The canonical op-spec parameters a verb's digest reads.
_PARAMETERS_BY_VERB: Mapping[str, type[_Facts]] = {
    "aggregate": AggregateParameters,
    "growth": GrowthParameters,
    "rank": RankParameters,
    "filter": FilterParameters,
}

FACTS_BY_VERB: Mapping[str, type[ResultFacts]] = {
    "aggregate": AggregateFacts,
    "growth": GrowthFacts,
    "rank": RankFacts,
    "compare": CompareFacts,
    "sum_of": SumOfFacts,
    "share_of": ShareOfFacts,
    "quotient_of": QuotientOfFacts,
    "join": JoinFacts,
    "filter": FilterFacts,
}


class _Call(_Facts):
    """The part of a summary checked against its verb's facts and parameters, so a
    field error is placed under `result` or `parameters`."""

    result: Any
    parameters: Any


_CALL_BY_VERB: Mapping[str, type[_Call]] = {
    verb: create_model(
        f"{facts.__name__}Call",
        __base__=_Call,
        result=(facts, ...),
        parameters=(_PARAMETERS_BY_VERB.get(verb, _NoParameters), ...),
    )
    for verb, facts in FACTS_BY_VERB.items()
}


class ResolvedPeriod(BaseModel):
    """One relative period of a computed recipe and the literal it was filled
    with, as the answer's `resolved_periods` lists it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    offset: int = Field(le=0)
    period: str


class RecomputedInput(BaseModel):
    """A derived input the runner computed again under the call's latest year:
    its pair, its top verb, and the latest year it took alone, `None` when its
    manifest could not be read."""

    model_config = ConfigDict(frozen=True)

    spec_id: str
    content_version: str
    verb: str | None
    latest: str | None


class Rolling(BaseModel):
    """The latest year a call's relative periods were filled from: `latest`,
    each relative period with its literal (`periods`, by offset), and the
    derived inputs computed again under `latest` (`recomputed`)."""

    model_config = ConfigDict(frozen=True)

    latest: str
    periods: tuple[ResolvedPeriod, ...]
    recomputed: tuple[RecomputedInput, ...] = ()


class VerbSummary(BaseModel):
    """A verb call's summary, from which its digest and its `VerbResult` are built.

    `result` is the verb's `*Facts`. `derived` is the derived marker's detail
    (`frame_ref_resolution.DerivedMarker`), `None` when the result carries none.
    `rolling` is the latest year the call's relative periods were filled from,
    which the answer states beside the summary, `None` for a call with none.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, arbitrary_types_allowed=True)

    verb: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    spec_id: str
    content_version: str
    frame_content_version: str
    rows: int = Field(ge=0)
    role_keys: tuple[str, ...] = ()
    stat_var_count: int = Field(ge=0)
    dropped_key_count: int = Field(default=0, ge=0)
    right_dropped_key_count: int = Field(default=0, ge=0)
    ranked_before_limit: int | None = Field(default=None, ge=0)
    exclusions: tuple[SummaryExclusion, ...] = ()
    time: SummaryTime = Field(default_factory=SummaryObservation)
    execution_identity: ExecutionIdentity
    derived: dict[str, Any] | None = None
    result: Any
    rolling: Rolling | None = None

    @classmethod
    def parse(cls, summary: Mapping[str, Any]) -> VerbSummary:
        """`summary` read into typed facts.

        Raises:
            pydantic.ValidationError: `summary` is not a verb summary of a verb
                these tools call, or its `result` or `parameters` do not fit
                its verb's facts or the parameters its digest reads.
            KeyError: `summary` names a verb with no facts model.
        """
        verb = str(summary.get("verb"))
        if verb not in _CALL_BY_VERB:
            raise KeyError(verb)
        call = _CALL_BY_VERB[verb].model_validate(
            {"result": summary.get("result"), "parameters": summary.get("parameters", {})}
        )
        return cls.model_validate({**summary, "result": call.result})

    def verb_result(self) -> VerbResult:
        """The `VerbResult` a tool returns as `structuredContent`."""
        return VerbResult.model_validate(
            self.model_dump(include=set(VerbResult.model_fields), mode="json")
        )
