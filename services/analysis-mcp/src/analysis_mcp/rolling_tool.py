"""Rolling periods at the tool boundary: the `resolve_anchor` tool, and the
check `growth`, `rank` and `filter` run on a rolling call.

A rolling call states a period relative to the latest period of the data and
carries the literal period that stood for. Nothing in such a call says what
the latest period was, so it is resolved once, by `resolve_anchor`, and every
rolling call is checked against the stored result.

`resolve_anchor` takes the populations of a question (one for each root
dataset, `analysis_mcp.op_spec.Populations`) and the root frames. It loads
each root frame, narrows it by each population of its dataset with the
`filter` engine (a binding as a `category_match`, a level as a `place_level`),
and reads the narrowed rows as series: one for each distinct
combination of every column but `time_period`, `value` and `status`, mapped to
the periods it has a row at. `dfo_frame_grammar.latest_anchor` then gives the
newest period at least half of the series of every population hold, and the
series that lack it. The result is stored as an anchor record
(`common.materialisation.AnchorRecord`) under `common.rolling.anchor_key` and
returned with that key. The tool writes no frame and mints no pair.

A population names its dataset and no frame, so it is counted over every root
frame given with its `spec_id`. Every population needs at least one such
frame and every root frame at least one such population. A binding or a level
on a column the frame lacks, or a binding on a column that is no part of a
series, refuses.
Periods other than calendar years refuse (`rolling_grain_unsupported`).

The series are those `apis`' assembler groups a frame's rows into
(`assembler.resolve._group_by_coordinate` over `frame_view.dimension_columns`),
with the same three columns left out. The assembler counts a period for a
series only on a row whose `status` is null. Every frame this service reads
or writes holds a null `status` on every row, so each row counts here.

`anchor_refusal` is the check on a rolling verb call, which each of the three
verbs runs as it resolves its input (`anchored_input`). It reads the record
under the call's `anchor_key` and refuses `rolling_anchor_unresolved` unless
the record's populations are the op-spec's, every root frame the call's input
was computed from is one the record was resolved over, and each rolling
literal is the record's period moved by its offset. A fixed call reads no
record.

Finding the roots of a derived input reads the stored trace of each distinct
derived frame between the input and its roots, once (`common.recipe
.reached_roots`). The input's own trace is the one the verb already read with
its rows. On the bucket-backed store each further trace is one metadata read
and one download of that frame's object, checked against its digest, with no
column decoded. A root frame is recognised by its `spec_` id and is not read.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterator, Mapping, Sequence
from typing import Annotated, Any, NamedTuple, cast

import pyarrow as pa
from common.materialisation import (
    AnchorRecord,
    AnchorRecordIntegrityError,
    FrameRef,
    LeftOutSeries,
    StoredFrameTrace,
    get_store,
)
from common.recipe import RecipeError, reached_roots
from common.rolling import ANCHOR_KEY_PATTERN, NON_SERIES_COLUMNS, canonical_roots
from common.time_grain import is_interval, time_grain
from dfo_frame_grammar import (
    ANCHOR_RULE,
    Anchor,
    FrameError,
    RollingAnchorUnresolved,
    RollingError,
    RollingGrainUnsupported,
    latest_anchor,
    shift_period,
)
from dfo_frame_grammar.grammar import (
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    ComposedUnit,
    UnitCell,
    UnitRequest,
)
from dfo_spec_kit.identity import canonicalize
from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict, Field

from analysis_mcp.engine.columnar import group_rows
from analysis_mcp.engine.common import Units
from analysis_mcp.engine.filter import (
    FilterPredicateTypeMismatchError,
    FilterPredicateValueTypeMismatchError,
    FilterUnitConversionUnsupportedError,
    FilterUnknownColumnError,
    run_filter,
)
from analysis_mcp.frame_ref_resolution import (
    AdmittedFrame,
    holds_input_rows,
    resolve_input_frames,
)
from analysis_mcp.op_spec import (
    CategoryMatchPredicate,
    FilterOpSpec,
    GrowthOpSpec,
    OrderedThresholdPredicate,
    PlaceLevelPredicate,
    Populations,
    RankOpSpec,
    RollingRef,
    RootPopulation,
)
from analysis_mcp.settings import Settings
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal

RESOLVE_ANCHOR_VERSION = "1"

# How many periods a `rolling_grain_unsupported` refusal names.
_NAMED_PERIODS = 3


class _AnchorRequest(BaseModel):
    """`resolve_anchor`'s arguments, the populations in their one spelling."""

    model_config = ConfigDict(extra="forbid")

    populations: Populations
    roots: tuple[FrameRef, ...] = Field(min_length=1)


class ResolvedAnchor(BaseModel):
    """What `resolve_anchor` returns as `structuredContent`.

    - `anchor_key`: the handle of the stored record, passed to each rolling
      `growth`, `rank` and `filter` call as `anchor_key`.
    - `period`: the latest period, which a rolling literal at offset 0 is.
    - `left_out`: each series counted that holds no value at `period`, as the
      index of its population in the canonical populations and its
      `coordinates`, the series' value in each of its columns.
    """

    model_config = ConfigDict(extra="forbid")

    anchor_key: str = Field(pattern=ANCHOR_KEY_PATTERN)
    period: str
    left_out: tuple[LeftOutSeries, ...] = ()


RESOLVE_ANCHOR_OUTPUT_SCHEMA: dict[str, Any] = ResolvedAnchor.model_json_schema(
    mode="serialization"
)


def _unresolved(detail: str, reason: str, **context: object) -> ToolResult:
    return to_tool_result(
        ToolRefusal.rolling_anchor_unresolved(detail=detail, reason=reason, context=context)
    )


def _roots_by_population(request: _AnchorRequest) -> list[tuple[int, FrameRef]] | ToolResult:
    """Each population's index beside each root frame of its dataset, or the
    refusal for a population or a root frame with no counterpart."""
    pairs = [
        (index, root)
        for index, population in enumerate(request.populations)
        for root in request.roots
        if root.spec_id == population.spec_id
    ]
    for population in request.populations:
        if all(root.spec_id != population.spec_id for root in request.roots):
            return _unresolved(
                "population_without_root",
                f"no root frame was given for the population over {population.spec_id}",
                spec_id=population.spec_id,
            )
    for root in request.roots:
        if all(root.spec_id != population.spec_id for population in request.populations):
            return _unresolved(
                "root_without_population",
                f"the root frame {root.spec_id} matches the spec_id of no population",
                spec_id=root.spec_id,
            )
    return pairs


def _narrowed(population: RootPopulation, admitted: AdmittedFrame) -> pa.Table | ToolResult:
    """The rows of `admitted` that `population`'s bindings keep."""
    frame = admitted.frame
    narrowing = {*population.bindings, *population.levels}
    unknown = sorted(column for column in narrowing if column not in frame.columns)
    if unknown:
        return _unresolved(
            "binding_column_unknown",
            f"the root frame of {population.spec_id} has no column named {unknown[0]!r}",
            spec_id=population.spec_id,
            columns=unknown,
            available=list(frame.columns),
        )
    if not narrowing:
        return frame.table
    if off_series := sorted(set(population.bindings) & NON_SERIES_COLUMNS):
        return _unresolved(
            "binding_unusable",
            f"{off_series[0]!r} is no part of a series, so no population binds it",
            spec_id=population.spec_id,
            columns=off_series,
        )
    predicates: list[CategoryMatchPredicate | PlaceLevelPredicate] = [
        CategoryMatchPredicate(column=column, values=values)
        for column, values in population.bindings.items()
    ]
    predicates += [
        PlaceLevelPredicate(column=column, level=level)
        for column, level in population.levels.items()
    ]
    try:
        return run_filter(frame, predicates=predicates, span=admitted.span).frame.table
    except (
        FilterUnknownColumnError,
        FilterPredicateTypeMismatchError,
        FilterPredicateValueTypeMismatchError,
        FilterUnitConversionUnsupportedError,
    ) as exc:
        return _unresolved(
            "binding_unusable",
            f"a binding of the population over {population.spec_id} cannot narrow its "
            f"root frame: {exc}",
            spec_id=population.spec_id,
            columns=sorted(narrowing),
        )


def coordinate_text(cell: str | UnitCell | None) -> str | None:
    """One cell of a series' coordinates as text, in the form `apis`' assembler
    groups it under: a converted unit as `currency/prices[/base_year]`, a composed
    unit as the RFC 8785 text of its JSON object, and any other cell as itself."""
    if isinstance(cell, UnitRequest):
        text = f"{cell.currency}/{cell.prices}"
        return text if cell.base_year is None else f"{text}/{cell.base_year}"
    if isinstance(cell, ComposedUnit):
        return canonicalize(cell.as_json()).decode()
    return cell


def _series(table: pa.Table) -> Iterator[tuple[dict[str, str], set[str]]]:
    """Each series of `table`, in order of its first row: its coordinates, a
    column it holds no value in left out, and the periods it has a row at."""
    dimensions = [column for column in table.column_names if column not in NON_SERIES_COLUMNS]
    groups = group_rows(table, dimensions)
    periods = cast("list[str]", table.column(TIME_PERIOD_COLUMN).to_pylist())
    indices = cast("list[int]", groups.indices.to_pylist())
    plain = [column for column in dimensions if column != UNIT_COLUMN]
    keys = cast("list[dict[str, str | None]]", groups.keys.select(plain).to_pylist())
    units = Units.of(groups.keys) if UNIT_COLUMN in dimensions else None
    unit_codes = cast("list[int]", units.codes.to_pylist()) if units is not None else []
    for group in range(groups.count):
        cells: dict[str, str | UnitCell | None] = dict(keys[group]) if plain else {}
        if units is not None:
            cells[UNIT_COLUMN] = units.cells[unit_codes[group]]
        coordinates = {
            column: text
            for column in dimensions
            if (text := coordinate_text(cells[column])) is not None
        }
        rows = indices[groups.offsets[group] : groups.offsets[group + 1]]
        yield coordinates, {periods[row] for row in rows}


# Each series of one population over one root frame, keyed by its place in the
# call, with the periods it has a row at.
type _Counted = dict[tuple[int, int], set[str]]


def _counted(
    request: _AnchorRequest,
    pairs: Sequence[tuple[int, FrameRef]],
    admitted: Mapping[tuple[str, str], AdmittedFrame],
) -> tuple[list[_Counted], dict[tuple[int, int], dict[str, str]]] | ToolResult:
    """One `_Counted` for each of `pairs`, and every series' coordinates by its
    key. Two series with one set of coordinates, in two frames, stay apart."""
    counted: list[_Counted] = []
    coordinates: dict[tuple[int, int], dict[str, str]] = {}
    for position, (index, root) in enumerate(pairs):
        table = _narrowed(request.populations[index], admitted[root.spec_id, root.content_version])
        if isinstance(table, ToolResult):
            return table
        population: _Counted = {}
        for number, (series, periods) in enumerate(_series(table)):
            population[position, number] = periods
            coordinates[position, number] = series
        counted.append(population)
    return counted, coordinates


def _anchor_or_refusal(counted: Sequence[_Counted]) -> Anchor[tuple[int, int]] | ToolResult:
    """`latest_anchor` over `counted`, whose periods must be calendar years."""
    held = sorted(
        {period for population in counted for periods in population.values() for period in periods}
    )
    off_grain = [period for period in held if is_interval(period) or time_grain(period) != "year"]
    if off_grain:
        return to_tool_result(
            ToolRefusal.rolling_grain_unsupported(
                reason="a root frame holds periods that are no calendar year",
                periods=off_grain[:_NAMED_PERIODS],
            )
        )
    try:
        return latest_anchor(counted)
    except RollingAnchorUnresolved as exc:
        return _unresolved("no_qualifying_period", str(exc))
    except RollingGrainUnsupported as exc:
        return to_tool_result(
            ToolRefusal.rolling_grain_unsupported(reason=str(exc), periods=held[:_NAMED_PERIODS])
        )


@holds_input_rows
async def resolve_anchor(
    populations: Annotated[
        Populations,
        Field(
            description=(
                "The populations the latest period is taken over: one {spec_id, bindings} "
                "for each root dataset of the question, bindings narrowing its series as "
                "{column: [values]}, and levels, when set, holding a place-type column to "
                "one level of place as {column: level}. The same value goes on every "
                "rolling call as rolling.population."
            )
        ),
    ],
    roots: Annotated[
        tuple[FrameRef, ...],
        Field(
            min_length=1,
            description=(
                "The {spec_id, content_version} pair of every root frame of the question, "
                "as get_data returned it. Each population is counted over every frame "
                "given with its spec_id."
            ),
        ),
    ],
) -> ToolResult:
    """Find the latest period of the data a question's rolling periods follow.

    The latest period is the newest calendar year at least half of the series
    of every population hold. A series is one combination of a frame's columns
    other than `time_period`, `value` and `status`.

    Use it before a rolling `growth`, `rank` or `filter` call, to find the
    latest period the data holds.

    The result is stored, and its `anchor_key` goes on every rolling `growth`,
    `rank` and `filter` call of the question, whose literal periods are
    `period` moved back by each rolling offset. This tool computes no frame
    and returns no {spec_id, content_version} pair.

    Args:
        populations: The populations the latest period is taken over.
        roots: The pair of every root frame of the question.

    Returns:
        The `anchor_key`, the latest `period`, and `left_out`: each series
        counted that has no value at that period. On refusal, a typed error
        with a machine-readable fix.
    """
    request = build_op_spec_or_refusal(
        lambda: _AnchorRequest(populations=populations, roots=tuple(roots))
    )
    if isinstance(request, ToolResult):
        return request

    pairs = _roots_by_population(request)
    if isinstance(pairs, ToolResult):
        return pairs

    distinct = [
        FrameRef(**root) for root in canonical_roots(root.model_dump() for root in request.roots)
    ]
    resolved = await resolve_input_frames(distinct, settings=Settings())
    if isinstance(resolved, ToolResult):
        return resolved
    admitted = {
        (root.spec_id, root.content_version): frame
        for root, frame in zip(distinct, resolved, strict=True)
    }
    series = _counted(request, pairs, admitted)
    if isinstance(series, ToolResult):
        return series
    counted, coordinates = series
    anchor = _anchor_or_refusal(counted)
    if isinstance(anchor, ToolResult):
        return anchor

    record = AnchorRecord(
        populations=[population.model_dump(mode="json") for population in request.populations],
        roots=request.roots,
        rule=ANCHOR_RULE,
        period=anchor.period,
        left_out=[
            LeftOutSeries(root=pairs[position][0], coordinates=coordinates[key])
            for position, key in anchor.left_out
        ],
    )
    store = get_store()
    await store.put_anchor_record(record)
    # The record the store keeps is the one every later call is checked against.
    kept = await store.get_anchor_record(record.anchor_key) or record
    result = ResolvedAnchor(
        anchor_key=record.anchor_key, period=kept.period, left_out=kept.left_out
    )
    return ToolResult(
        content=(
            f"The latest period is {result.period} (anchor_key {result.anchor_key}). "
            f"{len(result.left_out)} of the series counted hold no value at it."
        ),
        structured_content=result.model_dump(mode="json"),
        is_error=False,
    )


def _rolling_literals(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec,
) -> list[tuple[str, str, RollingRef]]:
    """Each literal of `op_spec` a rolling reference governs: where it sits,
    the literal, and the reference."""
    if isinstance(op_spec, FilterOpSpec):
        return [
            (
                f"the {predicate.op} threshold on {predicate.column}",
                predicate.value,
                predicate.rolling,
            )
            for predicate in op_spec.predicates
            if isinstance(predicate, OrderedThresholdPredicate)
            and predicate.rolling is not None
            and isinstance(predicate.value, str)
        ]
    rolling = op_spec.rolling
    assert rolling is not None
    return [
        (name, literal, reference)
        for name, literal, reference in (
            ("window_start", op_spec.window_start, rolling.window_start),
            ("window_end", op_spec.window_end, rolling.window_end),
        )
        if reference is not None and literal is not None
    ]


class _Failure(NamedTuple):
    """One failed check of a rolling call: its `fix["detail"]`, the reason in
    words, and the values the check read."""

    detail: str
    reason: str
    context: dict[str, object]


async def _uncovered_root(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec,
    record: AnchorRecord,
    *,
    input_trace: StoredFrameTrace,
) -> _Failure | None:
    """The first root frame `op_spec`'s input was computed from that `record`
    was not resolved over."""
    try:
        reached = await reached_roots(op_spec.frame_ref, store=get_store(), trace=input_trace)
    except RecipeError as exc:
        return _Failure("root_walk_failed", str(exc), {})
    for root in reached:
        if root not in record.roots:
            return _Failure(
                "root_not_in_record",
                f"the input was computed from the root frame {root.spec_id}, "
                f"{root.content_version}, which the record was not resolved over",
                {"root": root.model_dump()},
            )
    return None


def _differing_literal(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec, record: AnchorRecord
) -> _Failure | None:
    """The first rolling literal of `op_spec` that is not `record.period` moved
    by its offset."""
    for place, literal, reference in _rolling_literals(op_spec):
        try:
            expected = shift_period(record.period, int(reference.offset))
        except (RollingError, FrameError):
            expected = None
        if literal != expected:
            return _Failure(
                "literal_differs",
                f"{place} is {literal}, and the latest period {record.period} at offset "
                f"{reference.offset} is {expected}",
                {
                    "period": record.period,
                    "offset": reference.offset,
                    "literal": literal,
                    "expected": expected,
                },
            )
    return None


async def _failed_check(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec,
    populations: Populations,
    anchor_key: str,
    *,
    input_trace: StoredFrameTrace,
) -> _Failure | None:
    """The first check the rolling `op_spec` fails against the record under
    `anchor_key`. `populations` is `op_spec.rolling.population`."""
    try:
        record = await get_store().get_anchor_record(anchor_key)
    except AnchorRecordIntegrityError as exc:
        return _Failure("anchor_record_unreadable", exc.reason, {})
    if record is None:
        return _Failure("anchor_record_missing", f"no record is stored under {anchor_key}", {})
    stated = [population.model_dump(mode="json") for population in populations]
    if stated != [population.model_dump(mode="json") for population in record.populations]:
        return _Failure(
            "populations_differ",
            f"rolling.population is not the populations {anchor_key} was resolved over",
            {},
        )
    return await _uncovered_root(op_spec, record, input_trace=input_trace) or _differing_literal(
        op_spec, record
    )


async def anchor_refusal(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec,
    anchor_key: str | None,
    *,
    input_trace: StoredFrameTrace,
) -> ToolResult | None:
    """The `rolling_anchor_unresolved` refusal for a rolling `op_spec` its
    anchor record does not cover, or `None` when the call may run. See the
    module docstring. A fixed `op_spec` is `None` at once and reads no record.

    Args:
        op_spec: The call's op-spec, already through `with_anchor_key`.
        anchor_key: The call's `anchor_key`.
        input_trace: The stored trace of the frame `op_spec.frame_ref` names.
    """
    if op_spec.rolling is None or anchor_key is None:
        return None
    failure = await _failed_check(
        op_spec, op_spec.rolling.population, anchor_key, input_trace=input_trace
    )
    if failure is None:
        return None
    return _unresolved(failure.detail, failure.reason, anchor_key=anchor_key, **failure.context)


async def anchored_input(
    op_spec: GrowthOpSpec | RankOpSpec | FilterOpSpec,
    anchor_key: str | None,
    resolving: Awaitable[AdmittedFrame | ToolResult],
) -> AdmittedFrame | ToolResult:
    """The input `resolving` admits, or a refusal: `resolving`'s own, or
    `anchor_refusal`'s for a rolling `op_spec`. `growth`, `rank` and `filter`
    resolve their input through this, so each checks a rolling call before its
    engine runs."""
    admitted = await resolving
    if isinstance(admitted, ToolResult):
        return admitted
    refusal = await anchor_refusal(op_spec, anchor_key, input_trace=admitted.trace)
    return admitted if refusal is None else refusal
