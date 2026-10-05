"""Bind what a compiled plan says of its frames to the frame a retrieval returns.

The compiler names a role column the way the approved source's entities do
(`SelectedSource.entities[*].role`): a flow axis (`origin`, `destination`) on a
flow dataset, the declared role (`observationAbout`) otherwise. The frame
`get_data` registers names its columns by the dataset's declared observation
properties (`financingProvider`, `financingRecipient`) and adds a
`stat_var_<code system>` column for each open family slot. The source does not
carry either set, so the executor learns the frame's columns from `get_data`'s
result once a retrieval returns (`frame_columns`). Before each step downstream
of it runs, the executor rewrites the step's column-naming arguments
(`bound_arguments`) and derives the role keys the step must declare
(`bound_role_keys`).

How many StatVars a frame holds is a property of the minted frame too: a
family retrieval holds one per family member the source has data for, and
one when it asks for the family summed. The executor reads the count off
each step's result (`get_data` and every analysis verb declare
`stat_var_count`) and bounds it: a summed retrieval's by one, and each later
step's by the count of the frame it reads (`bound_stat_var_count`).

Only a column's spelling changes. The roles a step reads, groups and keys by
are the ones the plan was approved with, and a role the frame carries no
column for raises `UnboundColumnError` before the call.

`_COLUMN_SITES` lists every argument `compiler.compile` writes a role column
into, per tool. `tests/executor/test_wide_frame.py` compiles and runs every
verb over a flow frame and fails if a flow-axis name reaches a call, so a
column-naming argument missing from the table shows there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial

from dfo_frame_grammar.grammar import STAT_VAR_COLUMN, normalize_role_keys
from planner.workflow import ToolId
from pydantic import JsonValue

# The frame parameter every analysis tool reads its first (or only) input
# through. A two-sided tool that leaves its second frame parameter out reads
# this one on both sides, as `compare`, `sum_of` and `share_of` document.
FRAME_PARAMETER = "frame_ref"


class UnboundColumnError(ValueError):
    """A step names a role the frame it reads carries no column for."""


@dataclass(frozen=True, slots=True)
class FrameColumns:
    """The role columns one retrieved frame carries, and the column behind each compiled name.

    `role_keys` is the frame's own, in `normalize_role_keys` order. `columns`
    maps every role column the compiler may have written for this retrieval
    to the frame column carrying it, or to `None` when the frame carries
    none.
    """

    role_keys: tuple[str, ...]
    columns: Mapping[str, str | None]

    def column(self, name: str) -> str:
        """`name` as this frame spells it.

        A name outside `columns` is not a role column the compiler wrote
        (`time_period` in a `group_by`, `value` in a threshold) and comes
        back unchanged.

        Raises:
            UnboundColumnError: If `name` is a role the frame carries no
                column for.
        """
        if name not in self.columns:
            return name
        bound = self.columns[name]
        if bound is None:
            raise UnboundColumnError(
                f"the frame carries no column for the {name!r} role; its role columns are "
                f"{list(self.role_keys)!r}"
            )
        return bound


def frame_columns(
    *,
    role_keys: Sequence[str],
    axis_by_column: Mapping[str, str | None],
    compiled: Iterable[str],
) -> FrameColumns:
    """Bind each of `compiled` to a column of a frame carrying `role_keys`.

    A compiled name the frame carries as a column binds to itself. Otherwise
    it binds to the one role column whose flow axis (`axis_by_column`, from
    the manifest's coverage dimensions) it names. An axis two columns share
    binds neither, and a name that is neither a column nor an axis binds to
    `None`.

    Args:
        role_keys: The frame's role columns, as `get_data` reports them.
        axis_by_column: Each coverage dimension's label and flow axis.
        compiled: Every role column the compiler may have written for the
            retrieval.
    """
    keys = normalize_role_keys(role_keys)
    by_axis = columns_on_axes(axis_by_column, keys)
    return FrameColumns(
        role_keys=keys,
        columns={name: name if name in keys else by_axis.get(name) for name in compiled},
    )


def columns_on_axes(
    axis_by_column: Mapping[str, str | None], columns: Iterable[str]
) -> dict[str, str | None]:
    """The one column of `columns` on each flow axis `axis_by_column` states.

    An axis two of `columns` share maps to `None`, since neither can stand for
    it. A column outside `columns`, and one stating no axis, map nothing.

    Args:
        axis_by_column: Each coverage dimension's label and flow axis.
        columns: The role columns a retrieval carries.
    """
    present = set(columns)
    by_axis: dict[str, str | None] = {}
    for column, axis in axis_by_column.items():
        if axis is None or column not in present:
            continue
        by_axis[axis] = None if axis in by_axis else column
    return by_axis


class _Form(Enum):
    """Where in one argument's value its column names sit."""

    NAMES = auto()  # a list of column names
    KEYS = auto()  # an object keyed by column name
    VALUES = auto()  # an object whose values are column names
    PREDICATES = auto()  # a list of predicate objects, each naming its `column`


@dataclass(frozen=True, slots=True)
class _ColumnSite:
    """One argument naming role columns, and the frame parameters whose columns it names.

    A join key names a column on both sides, so it lists both frames and must
    bind to one spelling on each.
    """

    parameter: str
    form: _Form
    frames: tuple[str, ...]


def _analysis(name: str) -> ToolId:
    return ToolId(server="analysis-mcp", name=name)


_COLUMN_SITES: Mapping[ToolId, tuple[_ColumnSite, ...]] = {
    _analysis("aggregate"): (_ColumnSite("group_by", _Form.NAMES, (FRAME_PARAMETER,)),),
    _analysis("growth"): (_ColumnSite("entity_by", _Form.NAMES, (FRAME_PARAMETER,)),),
    _analysis("rank"): (_ColumnSite("rank_by", _Form.NAMES, (FRAME_PARAMETER,)),),
    _analysis("filter"): (_ColumnSite("predicates", _Form.PREDICATES, (FRAME_PARAMETER,)),),
    _analysis("join"): (
        _ColumnSite("join_keys", _Form.NAMES, (FRAME_PARAMETER, "right_frame_ref")),
    ),
    _analysis("compare"): (
        _ColumnSite("left_entity", _Form.KEYS, (FRAME_PARAMETER,)),
        _ColumnSite("right_entity", _Form.KEYS, ("right_frame_ref",)),
    ),
    _analysis("sum_of"): (
        _ColumnSite("left_entity", _Form.KEYS, (FRAME_PARAMETER,)),
        _ColumnSite("right_entity", _Form.KEYS, ("right_frame_ref",)),
    ),
    _analysis("share_of"): (
        _ColumnSite("numerator_entity", _Form.KEYS, (FRAME_PARAMETER,)),
        _ColumnSite("denominator_entity", _Form.KEYS, ("denominator_frame_ref",)),
        _ColumnSite("numerator_by", _Form.NAMES, (FRAME_PARAMETER,)),
        _ColumnSite("denominator_by", _Form.KEYS, (FRAME_PARAMETER,)),
        _ColumnSite("denominator_by", _Form.VALUES, ("denominator_frame_ref",)),
    ),
}

# The verbs whose output keeps every role column of their input. Every other
# verb's output role columns are the ones its arguments name.
_KEEPS_INPUT_ROLES: frozenset[ToolId] = frozenset({_analysis("filter")})


def bound_arguments(
    tool: ToolId,
    arguments: Mapping[str, JsonValue],
    *,
    frames: Mapping[str, FrameColumns | None],
) -> dict[str, JsonValue]:
    """`arguments` with every role column they name spelled the way the frame read spells it.

    Args:
        tool: The step's tool.
        arguments: The step's arguments, frames resolved.
        frames: The columns of the frame bound to each of the step's frame
            parameters. `None` stands for a frame whose columns are unknown
            (a retrieval whose result carried no `role_keys`), whose names
            pass unchanged. A frame parameter the step leaves out reads
            `FRAME_PARAMETER`'s frame.

    Raises:
        UnboundColumnError: If an argument names a role its frame carries no
            column for, or a join key the two frames spell differently.
    """
    rewritten = dict(arguments)
    for site in _COLUMN_SITES.get(tool, ()):
        value = rewritten.get(site.parameter)
        if value is None:
            continue
        sides = tuple(
            frames[parameter] if parameter in frames else frames.get(FRAME_PARAMETER)
            for parameter in site.frames
        )
        rewritten[site.parameter] = _rewrite(value, site.form, partial(_spelling, sides=sides))
    return rewritten


def bound_role_keys(
    tool: ToolId,
    expected: Sequence[str],
    *,
    frame: FrameColumns | None,
    input_role_keys: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """The role keys a step must declare, given the frame it reads.

    A filter keeps every role column its input carries, so it must declare
    `input_role_keys`, the keys the step that produced its input declared.
    Every other verb's output role columns are the ones its arguments name,
    which the compiler predicted as `expected` in its own spelling; each is
    spelled the way `frame` spells it. With neither known, `expected` stands
    as compiled.

    Args:
        tool: The step's tool.
        expected: The step's compiled `expected_role_keys`.
        frame: The columns of the frame bound to `FRAME_PARAMETER`.
        input_role_keys: The role keys that frame was declared with.

    Raises:
        UnboundColumnError: If `expected` names a role `frame` carries no
            column for.
    """
    if tool in _KEEPS_INPUT_ROLES and input_role_keys is not None:
        return normalize_role_keys(input_role_keys)
    if frame is None:
        return tuple(expected)
    return normalize_role_keys(frame.column(name) for name in expected)


def _one_stat_var(arguments: Mapping[str, JsonValue], input_count: int | None) -> int | None:
    del arguments, input_count
    return 1


def _one_per_side(arguments: Mapping[str, JsonValue], input_count: int | None) -> int | None:
    del arguments, input_count
    return 2


def _of_the_input(arguments: Mapping[str, JsonValue], input_count: int | None) -> int | None:
    del arguments
    return input_count


def _one_when_summed(arguments: Mapping[str, JsonValue], input_count: int | None) -> int | None:
    del input_count
    return 1 if arguments.get("stat_vars") == "sum" else None


def _one_unless_grouped(arguments: Mapping[str, JsonValue], input_count: int | None) -> int | None:
    group_by = arguments.get("group_by")
    grouped = isinstance(group_by, list) and STAT_VAR_COLUMN in group_by
    return input_count if grouped else 1


# The most distinct StatVars each verb's non-empty result holds, from its
# arguments and the StatVar count of its `FRAME_PARAMETER` input, as
# `analysis_mcp.engine` computes it:
#
# - `get_data` reads no input. Asked to sum a family (`stat_vars="sum"`),
#   dfo-apis collapses every member into one series under the family's own
#   ref; otherwise the frame holds what the definition binds, unbounded here.
# - `growth`, `rank` and `share_of` refuse an input holding several StatVars
#   (`require_single_stat_var`) and stamp the one they read on every row.
# - `aggregate` does the same unless `group_by` names `stat_var`, which keeps
#   one group per StatVar of its input.
# - `join` refuses a side holding several and writes each side's rows under
#   that side's StatVar, so its result holds two, or one when both sides
#   carry the same StatVar under different units or facets.
# - `filter` keeps a subset of its input's rows.
# - `compare` writes the rows of its left side's slice, one per period.
# - `sum_of` writes one series under its own StatVar, one row per period.
_MOST_STAT_VARS: Mapping[ToolId, Callable[[Mapping[str, JsonValue], int | None], int | None]] = {
    ToolId(server="data-mcp", name="get_data"): _one_when_summed,
    _analysis("aggregate"): _one_unless_grouped,
    _analysis("growth"): _one_stat_var,
    _analysis("rank"): _one_stat_var,
    _analysis("share_of"): _one_stat_var,
    _analysis("join"): _one_per_side,
    _analysis("filter"): _of_the_input,
    _analysis("compare"): _of_the_input,
    _analysis("sum_of"): _one_stat_var,
}


def bound_stat_var_count(
    tool: ToolId,
    arguments: Mapping[str, JsonValue],
    *,
    input_stat_var_count: int | None,
) -> int | None:
    """The most distinct StatVars a non-empty result of `tool` may declare.

    Such a result declares at least one, so a bound of 1 fixes the count.
    `None` when the bound is the input's count and that count is unknown (a
    retrieval whose result carried no `stat_var_count`), and for a
    retrieval that does not sum its family.

    Args:
        tool: The step's tool, `get_data` or one of the analysis verbs.
        arguments: The step's arguments as sent.
        input_stat_var_count: The distinct StatVars the step that produced
            the frame bound to `FRAME_PARAMETER` declared.

    Raises:
        KeyError: If `tool` is not a verb `_MOST_STAT_VARS` covers.
    """
    return _MOST_STAT_VARS[tool](arguments, input_stat_var_count)


def _spelling(name: str, *, sides: Sequence[FrameColumns | None]) -> str:
    """`name` as every frame in `sides` spells it, which must be one spelling."""
    spellings = {name if side is None else side.column(name) for side in sides}
    if len(spellings) > 1:
        raise UnboundColumnError(
            f"the {name!r} role is {sorted(spellings)!r} across the frames this step joins; "
            "a key needs one column name on every side"
        )
    (spelling,) = spellings
    return spelling


def _rewrite(value: JsonValue, form: _Form, column: Callable[[str], str]) -> JsonValue:
    """`value` with the column names `form` locates passed through `column`.

    A value not of the shape `form` expects is returned unchanged, for the
    tool's own schema check to refuse.
    """
    if form is _Form.NAMES and isinstance(value, list):
        return [column(item) if isinstance(item, str) else item for item in value]
    if form is _Form.KEYS and isinstance(value, dict):
        return {column(key): item for key, item in value.items()}
    if form is _Form.VALUES and isinstance(value, dict):
        return {key: column(item) if isinstance(item, str) else item for key, item in value.items()}
    if form is _Form.PREDICATES and isinstance(value, list):
        return [
            {**item, "column": column(name)}
            if isinstance(item, dict) and isinstance(name := item.get("column"), str)
            else item
            for item in value
        ]
    return value
