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
engine adds reaches no digest until a renderer reads it.

A result in the paired form a `compare` or `sum_of` with `on` gives
(`dfo_analysis_engine.engine.paired.PairedResult`) has no facts model here,
since these tools send no `on`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from dfo_frame_grammar.grammar import ComposedUnit, UnitCell, unit_cell_from_json
from pydantic import BaseModel, ConfigDict, Field, PlainValidator

from analysis_mcp.artifact import Exclusion, Observation, Time
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


class _Facts(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


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
    left_resolved_unit: Unit = None
    right_resolved_unit: Unit = None
    warnings: tuple[str, ...] = ()
    join_keys: tuple[str, ...]
    right_key_columns: tuple[str, ...]
    matched_keys: tuple[MatchedKey, ...]
    left_dropped_key_count: int
    right_dropped_key_count: int


class FilterRowSummary(_Facts):
    row_key: KeyPairs
    value: float
    unit: Unit = None


class FilterFacts(_Facts):
    total_row_count: int
    kept_row_count: int
    dropped_row_count: int
    kept_row_summaries: tuple[FilterRowSummary, ...]


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


class VerbSummary(BaseModel):
    """A verb call's summary, from which its digest and its `VerbResult` are built.

    `result` is the verb's `*Facts`. `derived` is the derived marker's detail
    (`frame_ref_resolution.DerivedMarker`), `None` when the result carries none.
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
    exclusions: tuple[Exclusion, ...] = ()
    time: Time = Field(default_factory=Observation)
    execution_identity: ExecutionIdentity
    derived: dict[str, Any] | None = None
    result: Any

    @classmethod
    def parse(cls, summary: Mapping[str, Any]) -> VerbSummary:
        """`summary` read into typed facts.

        Raises:
            pydantic.ValidationError: `summary` is not a verb summary of a verb
                these tools call, or its `result` does not fit its verb's facts.
            KeyError: `summary` names a verb with no facts model.
        """
        facts = FACTS_BY_VERB[str(summary["verb"])].model_validate(summary["result"])
        return cls.model_validate({**summary, "result": facts})

    def verb_result(self) -> VerbResult:
        """The `VerbResult` a tool returns as `structuredContent`."""
        return VerbResult.model_validate(
            self.model_dump(include=set(VerbResult.model_fields), mode="json")
        )
