"""The artifact facts a trace carries beyond identity: which rows a verb's
narrowing dropped and why (`Exclusion`), and whether the output's
`time_period` names a genuine single observation or stands for a collapsed
span of more than one real period (`Observation`/`Interval`).

`ExclusionReason` is the closed set of narrowing steps found across every
engine (`analysis_mcp.engine`): `narrow_and_resolve`'s unit narrowing
(`unit_unbound`, `unit_not_selected`) and entity narrowing
(`entity_not_selected`), `growth`'s and `rank`'s window filters
(`outside_window`), `growth`'s entities lacking a row in either endpoint
year (`endpoint_missing`), with a zero or sign-crossing base under a relative
change (`zero_or_sign_crossing_base`), or overflowing to a non-finite value
(`non_finite`), `filter`'s predicate loop (`predicate`), `join`'s
unmatched-key sets (`unmatched_key`), `compare`'s unintersected periods
(`period_unmatched`), `rank`'s limit (`beyond_limit`), and `quotient_of`'s
matched cells whose denominator sums to zero (`zero_denominator`). A
heterogeneous facet always refuses
(`analysis_mcp.engine.common.FacetHeterogeneousError`); this set carries no
facet reason.

`entity_not_selected` and `period_unmatched` are recorded here and in
`structuredContent` but never rendered in the digest's own exclusions block
(`analysis_mcp.digest`): a caller's own `entity` binding asked for those rows
to go, and an unmatched period already has its own single digest
representation (`compare`'s coverage line).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from analysis_mcp.engine.common import ExclusionReason, ExclusionSide

# Per-role-column cap on `Exclusion.entities`, mirroring
# `analysis_mcp.engine.common._ENTITY_CAP`.
ENTITY_CAP = 20


class Exclusion(BaseModel):
    """Rows one narrowing step dropped, and why.

    `entities` names, per role column the output frame declares, every
    distinct value a dropped row carried on that column, capped at
    `ENTITY_CAP` values per role. `entities_truncated` is set when a role's
    distinct-value count exceeded that cap; `rows` still names the true
    total dropped. A per-period `rank`'s `beyond_limit` names only the
    entities cut from every period they were ranked in; an entity kept in
    some period is still in the output, though its other periods' rows
    count in `rows`.

    `side` names which of a two-input verb's two sides these rows came from
    (`compare`/`join`'s `left`/`right`, `share_of`'s
    `numerator`/`denominator`). `None` for a single-input verb (`aggregate`,
    `growth`, `rank`, `filter`).
    """

    model_config = ConfigDict(extra="forbid")

    reason: ExclusionReason
    rows: int = Field(ge=0)
    entities: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    entities_truncated: bool = False
    side: ExclusionSide | None = None


class Observation(BaseModel):
    """The output's own `time_period` names a genuine single observation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["observation"] = "observation"


class Interval(BaseModel):
    """The output's `time_period` stands for a span from `start` to `end`:
    the earliest and latest real periods the output collapsed, unioned
    across every group or entity the result carries. A verb builds one when
    it collapses a frame of points into one span (`aggregate` under sum or
    mean outside time grouping, a collapsing `rank`, `growth` over its
    window); a verb reading a span frame carries the input's interval on,
    and a raw materialisation labelled with one span reads as that span.

    `contiguous` is set when every period of the result's grain (year,
    month, or day) between `start` and `end` appears in at least one
    collapsed input row. `absent` names up to 10 of the periods in that
    span that do not, sorted; `absent_count` is the true total, so a
    10-entry `absent` list never has to be read as exhaustive. `contiguous`
    is `True`, `absent` is `()`, and `absent_count` is `0` for a span of
    exactly one period.

    `contiguous` is `None`, with `absent` and `absent_count` left at their
    defaults, when contiguity cannot be computed -- unknown, not `False`:
    `start` or `end` is a calendar-invalid day the source frame carried (the
    frame grammar admits a source's own out-of-range date; see
    `dfo_frame_grammar.grammar`), or the span is a raw materialisation's own
    label, whose trace records no periods it collapsed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["interval"] = "interval"
    start: str
    end: str
    contiguous: bool | None = True
    absent: tuple[str, ...] = ()
    absent_count: int = Field(default=0, ge=0)


Time = Annotated[Observation | Interval, Field(discriminator="kind")]
