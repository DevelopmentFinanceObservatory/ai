"""The `join` engine: inner join of two frames' Arrow tables on their shared
role columns plus `time_period`. `join` is an inner join that reports
dropped-key counts on both sides.

A joined pair of rows never collapses into one output row: `Frame` allows
exactly one `value` column (`dfo_frame_grammar.frame.Frame.build`'s own
required-column set has one `value` position, full stop), so a row carrying
both sides' own values side by side is not a shape this grammar can
represent. The honest output is long-format instead -- one row per matched
key *per side*, `stat_var` distinguishing which side a row's own value came
from, exactly the shape `analysis_mcp.fixtures.join_compatible_frame_pair`'s
own two frames (ODA disbursed, GNI) already model as two distinct StatVars
sharing one `(reporter, time_period)` grain.

This is a "proceed" instance of `analysis_mcp.engine.common`'s own
partial-overlap policy -- see that module's docstring for the reasoning.

The join key is always `time_period` plus a set of role columns: either
`join_keys` (validated against both frames, and against a small disallowed
set -- `stat_var`/`time_period` may not be named explicitly, since the
former would defeat the point of joining across two StatVars and the latter
is already matched implicitly), or, left empty, every role column both
frames declare in common, auto-detected. `right_keys` names the right
frame's own column for each `join_keys` column, so two columns under
different names (a WHO series on `observationAbout`, a CRS series on
`destination`) pair by value; the output's key column takes the left side's
name. Auto-detection pairs same-name columns only. A key value repeated
within one side is ambiguous -- there is no single row to join for that key -- and
refuses (`JoinAmbiguousKeyError`) rather than silently picking one or
summing (unlike `share_of`, a join's own per-side value is never reduced).

Two output rows sharing a grain is refused too, before `Frame.build` ever
sees them: if both sides resolve to the identical `stat_var`, unit, and
facets, every matched key's two emitted rows would be indistinguishable
(`JoinOutputCollisionError`) -- this can only happen when a caller joins a
frame against a differently-scoped slice of the same StatVar, rather than
the two-distinct-StatVars shape this engine is built for.

`join` performs no reduction of its own -- every emitted `value` is copied
verbatim from an already-finite input cell (`Frame.build` guarantees that at
construction) -- so this engine never calls `math.fsum`: single-op verbs
need no reduction, and this one carries no non-finite refusal. Each
emitted `unit` is copied the same way, from the matched row, which after
narrowing declares its side's resolved unit (a ref, a converted
`grammar.UnitRequest`, a `grammar.ComposedUnit`, or none).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
from dfo_frame_grammar import Frame, array_kernel
from dfo_frame_grammar.grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitCell,
)

from analysis_mcp.engine.columnar import group_rows, keys_unique, repeated
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    available_entity_columns,
    combine_exclusions,
    frame_keeping_role_types,
    narrow_and_resolve,
    record_exclusion,
    require_single_stat_var,
    role_columns,
)

# Never a valid join key: `value`/`status` are reduction/reserved columns (the
# same reasoning `analysis_mcp.engine.aggregate._DISALLOWED_GROUP_BY_COLUMNS`
# already applies), `unit` and the facet axes are resolved to a single value
# before this engine runs, `stat_var` naming a join key would only ever match
# rows sharing one StatVar (defeating the point of joining two different
# StatVars together), and `time_period` is already matched implicitly by
# every join, never optionally.
_DISALLOWED_JOIN_KEY_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, STATUS_COLUMN, UNIT_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)

# Row-position columns `run_join` carries through the key match. The match
# tables name their key columns by position (`_match_key_names`), so these
# names never meet a role name.
_LEFT_ROW = "left_row"
_RIGHT_ROW = "right_row"


class JoinUnknownKeyError(Exception):
    """One or more `join_keys` (or `right_keys`) entries name a column the
    frame they key has no grain for -- absent from that frame's own columns,
    or one of the disallowed reduction/reserved/implicit-match columns.
    `parameter` is the parameter at fault and `available` the columns it
    may name."""

    def __init__(
        self, columns: Sequence[str], *, available: Sequence[str], parameter: str = "join_keys"
    ) -> None:
        self.columns = tuple(columns)
        self.available = tuple(available)
        self.parameter = parameter
        super().__init__(f"{parameter} names unknown/disallowed column(s): {self.columns!r}")


class JoinRightKeysPartialError(Exception):
    """`right_keys` is non-empty but names no right column for the
    `join_keys` columns in `columns`, which therefore pair with nothing.
    `JoinOpSpec` refuses this first."""

    def __init__(self, *, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        super().__init__(f"right_keys leaves out join_keys column(s): {self.columns!r}")


class JoinAmbiguousKeyError(Exception):
    """More than one row on one side shares the identical join key -- there
    is no single row to join for that key."""

    def __init__(self, *, side: str, key_columns: Sequence[str], available: Sequence[str]) -> None:
        self.side = side
        self.key_columns = tuple(key_columns)
        self.available = tuple(available)
        super().__init__(f"{side} side carries more than one row for one join key")


class JoinNoCommonKeyColumnsError(Exception):
    """`join_keys` was left empty (auto-detect), and the two frames declare
    no role column in common -- while *both* sides declare at least one
    role column of their own. Proceeding would join on `time_period` alone,
    silently pairing every row on one side that shares a period with every
    row on the other, regardless of role (a USA disbursement matched to a
    Kenya receipt that merely shares a year). Asymmetric case (one side
    declares no role column at all): joining on `time_period` alone is safe
    there. A role-less side is typically already narrowed to one row per
    period (a world total, or an already-single-entity aggregate), carrying
    no role vocabulary for the other side's own role columns to conflict
    with; `_resolve_join_keys` returns an empty key list there without
    raising. The role-less side's grain *is* `time_period` so `Frame.build`
    already refuses multi-row-per-period there, while the *other*, role-bearing
    side can still have multiple rows per period despite declaring a role
    column; `run_join`'s own `JoinAmbiguousKeyError` catches that once
    matching on `time_period` alone collapses duplicates onto one key.
    """

    def __init__(self, *, left_columns: Sequence[str], right_columns: Sequence[str]) -> None:
        self.left_columns = tuple(left_columns)
        self.right_columns = tuple(right_columns)
        super().__init__(
            f"auto-detected no role column in common (left declares {self.left_columns!r}, "
            f"right declares {self.right_columns!r}); joining on time_period alone would "
            "pair unrelated rows"
        )


class JoinResultTooLargeError(Exception):
    """The joined output would carry more rows than `max_output_rows`. Each
    matched key emits one row per side, so the output has twice as many rows
    as there are matched keys, and inputs that each fit the ceiling can still
    produce a result over it. Raised once the matched count is known, before
    any output column is built."""

    def __init__(self, *, row_count: int, ceiling: int) -> None:
        self.row_count = row_count
        self.ceiling = ceiling
        super().__init__(f"the joined output would have {row_count} rows, over {ceiling}")


class JoinOutputCollisionError(Exception):
    """Both sides resolved to the identical `stat_var`, unit, and facets --
    every matched key would then emit two output rows sharing one grain
    (every cell except `value` and `status`), which `Frame.build` cannot
    represent. Distinct from `JoinAmbiguousKeyError`: each side's own row is
    unambiguous here, it's the *pairing* of the two sides that collides.
    """

    def __init__(self, *, left_stat_var: str, right_stat_var: str) -> None:
        self.left_stat_var = left_stat_var
        self.right_stat_var = right_stat_var
        super().__init__(
            f"both sides resolved to stat_var {left_stat_var!r} with the same unit and "
            "facets; the joined output would collide"
        )


@dataclass(frozen=True, kw_only=True)
class MatchedKey:
    """One matched join key's own left/right values, addressed by its own
    key (role values plus `time_period`) -- the same "never trust row
    order" discipline every other engine result here follows."""

    key: tuple[tuple[str, str], ...]
    time_period: str
    left_value: float
    right_value: float


@dataclass(frozen=True, kw_only=True)
class MatchedKeys:
    """Every matched join key in match order, held as Arrow arrays of equal
    length: `role_cells` maps each join key column to its values, beside the
    key's `time_periods` and both sides' values. `rows` converts only the
    positions it is given, so a digest that shows a few keys builds a
    `MatchedKey` for those alone."""

    role_cells: Mapping[str, pa.Array]
    time_periods: pa.Array
    left_values: pa.Array
    right_values: pa.Array

    def __len__(self) -> int:
        return len(self.time_periods)

    def rows(self, positions: Sequence[int] | None = None) -> tuple[MatchedKey, ...]:
        """The matched keys at `positions`, in the order given, or every
        matched key when `positions` is `None`."""
        indices = pa.array(range(len(self)) if positions is None else positions, type=pa.int64())

        def cells(array: pa.Array) -> list[Any]:
            return array.take(indices).to_pylist()

        role_values = [cells(values) for values in self.role_cells.values()]
        return tuple(
            MatchedKey(
                key=tuple(zip(self.role_cells, key_values, strict=True)),
                time_period=time_period,
                left_value=left_value,
                right_value=right_value,
            )
            for *key_values, time_period, left_value, right_value in zip(
                *role_values,
                cells(self.time_periods),
                cells(self.left_values),
                cells(self.right_values),
                strict=True,
            )
        )


@dataclass(frozen=True, kw_only=True)
class JoinResult:
    """Everything `join_tool.py` needs to mint identity, write the store,
    and render the digest -- the frame is here for the store write and
    replay; the digest itself renders from `matched_keys`, never
    `frame.rows`.

    `left_dropped_key_count` and `right_dropped_key_count` count each
    side's (role columns, `time_period`) keys that matched nothing.
    `left_unmatched_entity_count` and `right_unmatched_entity_count` count
    each side's role-column values that matched in no period, the entities
    the output holds no row for; 0 for a join on `time_period` alone.
    """

    frame: Frame
    left_resolved_unit: UnitCell | None
    right_resolved_unit: UnitCell | None
    left_facet_selections: tuple[FacetSelection, ...]
    right_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    join_keys: tuple[str, ...]
    right_key_columns: tuple[str, ...]
    matched_keys: MatchedKeys
    left_dropped_key_count: int
    right_dropped_key_count: int
    left_unmatched_entity_count: int
    right_unmatched_entity_count: int
    exclusions: Exclusions = field(default_factory=Exclusions)


def _resolve_join_keys(
    left_frame: Frame,
    right_frame: Frame,
    *,
    join_keys: Sequence[str],
    right_keys: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The role columns to join on, beyond `time_period`, as the left
    side's columns and the right side's column for each. These are
    `join_keys` (paired through `right_keys` when it is set, under the same
    name when not) validated against both frames if given, or every role
    column both frames declare in common if not.

    Auto-detection (`join_keys` empty) refuses (`JoinNoCommonKeyColumnsError`)
    when the intersection it would return is empty *and both* sides
    actually declare at least one role column -- see that exception's own
    docstring for why an empty intersection is only sometimes the
    legitimate "at least one side has no role column at all" case.
    """
    left_role_columns = set(role_columns(left_frame)) - _DISALLOWED_JOIN_KEY_COLUMNS
    right_role_columns = set(role_columns(right_frame)) - _DISALLOWED_JOIN_KEY_COLUMNS
    if right_keys:
        missing = sorted(set(join_keys) - set(right_keys))
        if missing:
            raise JoinRightKeysPartialError(columns=missing)
        right = tuple(right_keys[column] for column in join_keys)
        for parameter, columns, held in (
            ("join_keys", join_keys, left_role_columns),
            ("right_keys", right, right_role_columns),
        ):
            invalid = [column for column in columns if column not in held]
            if invalid:
                raise JoinUnknownKeyError(invalid, available=sorted(held), parameter=parameter)
        return tuple(join_keys), right
    if join_keys:
        invalid = [
            column
            for column in join_keys
            if column not in left_role_columns or column not in right_role_columns
        ]
        if invalid:
            available = sorted(left_role_columns & right_role_columns)
            raise JoinUnknownKeyError(invalid, available=available)
        return tuple(join_keys), tuple(join_keys)
    common = left_role_columns & right_role_columns
    # `and`, not `or`: refuse only the symmetric case (both sides declare
    # their own role columns, but share no name -- donor vs recipient,
    # genuinely nothing to auto-detect safely). The asymmetric case (one
    # side declares no role column at all) is not the same hazard --
    # see `JoinNoCommonKeyColumnsError`'s own docstring for why a role-less
    # side has no role vocabulary to lose by matching on time_period alone.
    if not common and (left_role_columns and right_role_columns):
        raise JoinNoCommonKeyColumnsError(
            left_columns=sorted(left_role_columns), right_columns=sorted(right_role_columns)
        )
    return tuple(sorted(common)), tuple(sorted(common))


def _keyed_table(
    frame: Frame,
    *,
    side: Literal["left", "right"],
    key_columns: Sequence[str],
    pinned_unit: UnitCell | None,
) -> tuple[pa.Table, UnitCell | None, tuple[FacetSelection, ...], Exclusions]:
    """Resolve `frame`'s unit/facets and narrow its table to the rows that
    take part in the join. Refuses (`JoinAmbiguousKeyError`) if any join key
    (`key_columns` values plus `time_period`) repeats.
    """
    # `exclude_unitless=True`: emitted rows copy this side's resolved unit
    # onto matched values. A row leaving `unit` unbound is excluded; it would
    # otherwise be mislabelled with a unit it never declared. Excluded rows
    # leave the other side's matching key with no partner.
    table, resolved, exclusions = narrow_and_resolve(
        frame, pinned_unit=pinned_unit, exclude_unitless=True, side=side
    )
    if not keys_unique(table, [*key_columns, TIME_PERIOD_COLUMN]):
        raise JoinAmbiguousKeyError(
            side=side, key_columns=key_columns, available=available_entity_columns(frame)
        )
    return table, resolved.unit, resolved.facet_selections, exclusions


def _match_key_names(count: int) -> list[str]:
    """The names a match table gives its `count` key columns, by position."""
    return [f"key_{position}" for position in range(count)]


def _match_table(table: pa.Table, key: Sequence[str], *, row: str) -> pa.Table:
    """`table`'s `key` columns named by position (`_match_key_names`), and its
    rows numbered from 0 in a column `row`."""
    return pa.Table.from_arrays(
        [*table.select(key).columns, pa.array(range(table.num_rows), type=pa.int64())],
        names=[*_match_key_names(len(key)), row],
    )


def _unmatched(table: pa.Table, matched_rows: pa.Array) -> pa.Table:
    """The rows of `table` whose position is not in `matched_rows`."""
    positions = pa.array(range(table.num_rows), type=pa.int64())
    return table.filter(
        array_kernel("invert", array_kernel("is_in", positions, value_set=matched_rows))
    )


def run_join(
    left_frame: Frame,
    right_frame: Frame,
    *,
    join_keys: Sequence[str] = (),
    right_keys: Mapping[str, str] | None = None,
    left_pinned_unit: UnitCell | None = None,
    right_pinned_unit: UnitCell | None = None,
    max_output_rows: int | None = None,
) -> JoinResult:
    """Inner-join `left_frame` and `right_frame` on their shared role
    columns (or `join_keys`, if given, each paired with its `right_keys`
    column when that is set) plus `time_period`. A non-empty `right_keys`
    that leaves a `join_keys` column out raises `JoinRightKeysPartialError`,
    which `JoinOpSpec` refuses first. See the module
    docstring for the long-format output shape and the key-resolution
    rules; see each raised exception's own docstring for its refusal
    condition. `max_output_rows`, when given, bounds the output frame's row
    count (`JoinResultTooLargeError`).
    """
    left_stat_var = require_single_stat_var(left_frame)
    right_stat_var = require_single_stat_var(right_frame)
    key_columns, right_key_columns = _resolve_join_keys(
        left_frame, right_frame, join_keys=join_keys, right_keys=right_keys or {}
    )
    key = [*key_columns, TIME_PERIOD_COLUMN]
    right_key = [*right_key_columns, TIME_PERIOD_COLUMN]

    left_table, left_unit, left_facets, left_exclusions = _keyed_table(
        left_frame, side="left", key_columns=key_columns, pinned_unit=left_pinned_unit
    )
    right_table, right_unit, right_facets, right_exclusions = _keyed_table(
        right_frame, side="right", key_columns=right_key_columns, pinned_unit=right_pinned_unit
    )

    # Every key is unique on its side, so the key columns order the matches
    # totally. Arrow compares strings by UTF-8 bytes, which is the code-point
    # order Python's `str` comparison uses.
    match_key = _match_key_names(len(key))
    matched = _match_table(left_table, key, row=_LEFT_ROW).join(
        _match_table(right_table, right_key, row=_RIGHT_ROW), keys=match_key, join_type="inner"
    )
    count = matched.num_rows
    if max_output_rows is not None and 2 * count > max_output_rows:
        raise JoinResultTooLargeError(row_count=2 * count, ceiling=max_output_rows)
    matched = matched.take(
        array_kernel(
            "sort_indices", matched, sort_keys=[(column, "ascending") for column in match_key]
        )
    )
    left_rows = matched.column(_LEFT_ROW).combine_chunks()
    right_rows = matched.column(_RIGHT_ROW).combine_chunks()

    left_dropped = _unmatched(left_table, left_rows)
    right_dropped = _unmatched(right_table, right_rows)
    # Every matched entity is one value on both sides, so each side's
    # entities less the matched ones are those that matched in no period.
    if key_columns:
        matched_entities = group_rows(matched, match_key[: len(key_columns)]).count
        left_unmatched_entities = group_rows(left_table, key_columns).count - matched_entities
        right_unmatched_entities = (
            group_rows(right_table, right_key_columns).count - matched_entities
        )
    else:
        left_unmatched_entities = right_unmatched_entities = 0
    exclusions = combine_exclusions(left_exclusions, right_exclusions)
    exclusions = record_exclusion(
        exclusions, frame=left_frame, reason="unmatched_key", dropped=left_dropped, side="left"
    )
    exclusions = record_exclusion(
        exclusions, frame=right_frame, reason="unmatched_key", dropped=right_dropped, side="right"
    )

    if count:
        left_facet_set = {(selection.axis, selection.ref) for selection in left_facets}
        right_facet_set = {(selection.axis, selection.ref) for selection in right_facets}
        if (
            left_stat_var == right_stat_var
            and left_unit == right_unit
            and left_facet_set == right_facet_set
        ):
            raise JoinOutputCollisionError(
                left_stat_var=left_stat_var, right_stat_var=right_stat_var
            )

    left_values = left_table.column(VALUE_COLUMN).take(left_rows).combine_chunks()
    right_values = right_table.column(VALUE_COLUMN).take(right_rows).combine_chunks()
    key_cells = {
        column: matched.column(match_column).combine_chunks()
        for column, match_column in zip(key, match_key, strict=True)
    }
    left_facet_refs = {selection.axis: selection.ref for selection in left_facets}
    right_facet_refs = {selection.axis: selection.ref for selection in right_facets}
    # Long format: every matched key's left row, then its right row, as two
    # blocks the output frame's canonical sort interleaves.
    output_frame = frame_keeping_role_types(
        {
            STAT_VAR_COLUMN: pa.concat_arrays(
                [repeated(left_stat_var, count), repeated(right_stat_var, count)]
            ),
            **{column: pa.concat_arrays([cells, cells]) for column, cells in key_cells.items()},
            VALUE_COLUMN: pa.concat_arrays([left_values, right_values]),
            UNIT_COLUMN: pa.concat_arrays(
                [
                    left_table.column(UNIT_COLUMN).take(left_rows).combine_chunks(),
                    right_table.column(UNIT_COLUMN).take(right_rows).combine_chunks(),
                ]
            ),
            **{
                axis: pa.concat_arrays(
                    [
                        repeated(left_facet_refs.get(axis), count),
                        repeated(right_facet_refs.get(axis), count),
                    ]
                )
                for axis in FACET_IDENTITY_COLUMNS
            },
        },
        role_keys=key_columns,
        source=left_frame,
    )

    warnings: list[str] = []
    if left_dropped.num_rows or right_dropped.num_rows:
        warnings.append(
            f"{left_dropped.num_rows} left-side and {right_dropped.num_rows} right-side key(s) "
            "had no match on the other side and were dropped (inner join)."
        )

    return JoinResult(
        frame=output_frame,
        left_resolved_unit=left_unit,
        right_resolved_unit=right_unit,
        left_facet_selections=left_facets,
        right_facet_selections=right_facets,
        warnings=tuple(warnings),
        join_keys=key_columns,
        right_key_columns=right_key_columns,
        matched_keys=MatchedKeys(
            role_cells={column: key_cells[column] for column in key_columns},
            time_periods=key_cells[TIME_PERIOD_COLUMN],
            left_values=left_values,
            right_values=right_values,
        ),
        left_dropped_key_count=left_dropped.num_rows,
        right_dropped_key_count=right_dropped.num_rows,
        left_unmatched_entity_count=left_unmatched_entities,
        right_unmatched_entity_count=right_unmatched_entities,
        exclusions=exclusions,
    )
