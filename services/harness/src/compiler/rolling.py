"""Rolling periods in a compiled workflow: what a step carries and what is filled at run time.

A `growth`, `rank` or `filter` step whose plan states a period against the
newest data carries the tool's `rolling` argument beside its literal years.
`rolling.window_start` and `rolling.window_end` (a filter: `rolling` on each
period predicate) hold a reference `{anchor, offset, rule}`, and
`rolling.population` the series the newest period is taken over. The literal
years are the ones the plan read from Search.

`is_rolling` tells such a step from a fixed one: its arguments hold `rolling`.
The compiler writes no `anchor_key`, which names a stored resolution of the
newest period and exists only once the retrievals it is taken over have run.

A newest period belongs to a group of rolling steps (`anchor_groups`). Two
rolling steps are in one group when some step reads both, directly or through
other steps, and the relation is transitive. A frame computed from two rolling
steps states one period. A group's `roots` are the retrievals its rolling steps
rest on, and its `population` the series of those retrievals that count. A
retrieval no rolling step of the group reads is in neither, so a dataset of
fixed years joined above a rolling step, and a dataset only another card reads,
do not move the group's period. At run time the caller resolves each group's
period over its population and the frames of its roots, after those retrievals
and before the group's steps, and sends each step `resolved_arguments`: the
literals rewritten from the period, and the key. `rolling_bounds` lists those
literals.

`step_populations` builds each group's population from the arguments the
compiler wrote (`common.rolling.anchor_populations`): one entry for each chain
from a terminal step down to a retrieval that passes through a rolling step,
narrowed by the filters and entity bindings on the way and by the entities the
retrieval's requirement names. Every rolling step of one group carries the same
value. Its columns are spelled the way the approved source spells its roles,
like every other column the compiler writes.
`compiler.frame_columns.bound_populations` spells them the way the retrieved
frames do.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from common.rolling import anchor_populations, canonical_populations
from dfo_frame_grammar import ANCHOR_RULE, shift_period
from planner.records import RelativePeriod, RollingWindow
from planner.workflow import Binding, Workflow, WorkflowStep
from pydantic import JsonValue

from compiler.frame_columns import FRAME_PARAMETER, ROLLING_PARAMETER, rolling_population

# The argument naming the stored resolution of the newest period a rolling
# call's literals were written from.
ANCHOR_KEY_PARAMETER = "anchor_key"
_WINDOW_BOUNDS = ("window_start", "window_end")
_PREDICATES = "predicates"


class PopulationsDifferError(RuntimeError):
    """The rolling steps of one group carry different populations.

    The newest period is resolved once for a group, over one population, so
    steps that disagree cannot all be checked against it. The compiler writes
    one value on every step of a group, so a difference is a fault in the
    workflow.
    """


@dataclass(frozen=True, slots=True)
class AnchorGroup:
    """The rolling steps of a workflow that stand on one newest period.

    `steps` are the group's rolling steps and `roots` the retrieval steps they
    rest on, both in the workflow's step order. `population` is the value every
    step of the group carries.
    """

    steps: tuple[str, ...]
    roots: tuple[str, ...]
    population: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True)
class Root:
    """One retrieval as a population sees it.

    `columns` are the columns a binding may name on its frame, in the approved
    source's spelling. `named` is each role column to the refs of the entities
    the retrieval's requirement names on it, which stand for a card's selection
    until a card states one.
    """

    spec_id: str
    columns: frozenset[str]
    named: Mapping[str, tuple[str, ...]]


def rolling_ref(period: RelativePeriod) -> dict[str, JsonValue]:
    """`period` as the tools take it, under the rule that chooses the newest period."""
    return {"anchor": period.anchor, "offset": period.offset, "rule": ANCHOR_RULE}


def window_overlay(rolling: RollingWindow) -> dict[str, JsonValue]:
    """The `rolling` argument of a `growth` or `rank` step, before its population is known."""
    return {
        bound: rolling_ref(period)
        for bound in _WINDOW_BOUNDS
        if (period := getattr(rolling, bound)) is not None
    }


def with_population(
    arguments: Mapping[str, JsonValue], population: Sequence[Mapping[str, Any]]
) -> dict[str, JsonValue]:
    """`arguments` with `population` written on the `rolling` they hold."""
    rolling = arguments[ROLLING_PARAMETER]
    assert isinstance(rolling, dict)
    return {**arguments, ROLLING_PARAMETER: {**rolling, "population": list(population)}}


def is_rolling(step: WorkflowStep) -> bool:
    """Whether `step` states a period against the newest data, and so needs an `anchor_key`."""
    return step.args.get(ROLLING_PARAMETER) is not None


def anchor_groups(workflow: Workflow) -> tuple[AnchorGroup, ...]:
    """The groups `workflow`'s rolling steps form, in the order of each group's
    first step, or an empty tuple when no step rolls. See the module docstring.

    Raises:
        PopulationsDifferError: Two rolling steps of one group carry different
            populations, or one carries none.
    """
    reads = {
        step.id: [binding.from_step for binding in step.bindings.values()]
        for step in workflow.steps
    }
    retrievals = {binding.step_id for binding in workflow.evidence_bindings}
    args = {step.id: step.args for step in workflow.steps}
    groups: list[AnchorGroup] = []
    for members in _grouped(
        reads, rolling=[step.id for step in workflow.steps if is_rolling(step)]
    ):
        reached = set().union(*(_under(step_id, reads) for step_id in members))
        populations = [rolling_population(args[step_id]) for step_id in members]
        first = populations[0]
        if first is None or any(population != first for population in populations):
            raise PopulationsDifferError(
                f"the rolling steps {members!r} stand on one newest period and do not carry "
                "one population"
            )
        groups.append(
            AnchorGroup(
                steps=tuple(members),
                roots=tuple(step_id for step_id in reads if step_id in reached & retrievals),
                population=tuple(first),
            )
        )
    return tuple(groups)


def _under(step_id: str, reads: Mapping[str, Iterable[str]]) -> set[str]:
    """`step_id` and every step it reads, directly or through other steps."""
    found: set[str] = set()
    pending = [step_id]
    while pending:
        current = pending.pop()
        if current not in found:
            found.add(current)
            pending.extend(reads[current])
    return found


def _grouped(reads: Mapping[str, Iterable[str]], *, rolling: Sequence[str]) -> list[list[str]]:
    """`rolling` split into groups. Two steps share one when some step of `reads`
    reads both, directly or through other steps, transitively. Each group and
    the list of them keep the order of `rolling`."""
    group_of = {step_id: {step_id} for step_id in rolling}
    for step_id in reads:
        below = _under(step_id, reads)
        together = [member for member in rolling if member in below]
        merged = set().union(*(group_of[member] for member in together))
        for member in merged:
            group_of[member] = merged
    return [
        [member for member in rolling if member in group_of[step_id]]
        for step_id in rolling
        if step_id == min(group_of[step_id], key=rolling.index)
    ]


def step_populations(
    written: Mapping[str, tuple[Mapping[str, JsonValue], Mapping[str, Binding]]],
    *,
    terminal: Sequence[str],
    roots: Mapping[str, Root],
    frame_slots: Mapping[str, Collection[str]],
) -> dict[str, list[dict[str, Any]]]:
    """The population of each rolling step, from the arguments written for the
    workflow's steps: its group's, by step id.

    Args:
        written: Each step's literal arguments and bindings, by step id. A
            rolling step's arguments hold `rolling`, with no population yet.
        terminal: The workflow's terminal steps. Every step feeds one.
        roots: Each retrieval step, by step id.
        frame_slots: Each other step's frame parameters, bound or not. A
            two-sided verb reads `frame_ref` on both sides when its second
            frame parameter is left out.

    Raises:
        common.rolling.PopulationEmptyError: The bindings on one chain keep no
            series of a retrieval.
        common.rolling.PopulationLevelColumnMissingError: A place-level filter
            reads a retrieval that carries no place types.
    """
    index = {step_id: position for position, step_id in enumerate(roots)}

    def node(step_id: str) -> dict[str, Any]:
        if step_id not in index:
            arguments, bindings = written[step_id]
            return {
                "op_spec": {"parameters": {**dict.fromkeys(frame_slots[step_id]), **arguments}},
                "inputs": {
                    parameter: node(binding.from_step) for parameter, binding in bindings.items()
                },
            }
        root: dict[str, Any] = {"root": index[step_id]}
        named = roots[step_id].named
        if not named:
            return root
        # `anchor_populations` takes one selection for every root. Each
        # retrieval names its own entities, so they are stated as a filter
        # over that root alone.
        predicates = [
            {"kind": "category_match", "column": column, "values": list(refs)}
            for column, refs in named.items()
        ]
        return {
            "op_spec": {"parameters": {_PREDICATES: predicates}},
            "inputs": {FRAME_PARAMETER: root},
        }

    datasets = [{"dataset": {"spec_id": root.spec_id}} for root in roots.values()]
    columns = [root.columns for root in roots.values()]
    reads = {
        step_id: [binding.from_step for binding in bindings.values()]
        for step_id, (_, bindings) in written.items()
    }
    rolling = [
        step_id for step_id, (arguments, _) in written.items() if ROLLING_PARAMETER in arguments
    ]
    # A terminal's rolling steps are all of one group, since it reads each of
    # them, so every population of its tree belongs to that group.
    above = {
        step_id: anchor_populations(node(step_id), datasets, root_columns=columns)
        for step_id in terminal
    }
    populations: dict[str, list[dict[str, Any]]] = {}
    for members in _grouped(reads, rolling=rolling):
        group = canonical_populations(
            population
            for step_id in terminal
            if not _under(step_id, reads).isdisjoint(members)
            for population in above[step_id]
        )
        populations.update(dict.fromkeys(members, group))
    return populations


@dataclass(frozen=True, slots=True)
class RollingBound:
    """One literal of a rolling step's arguments that a reference governs.

    `bound` is `window_start`, `window_end`, or `predicates/<index>` for the
    predicate at that index of `predicates`. `offset` is the reference's.
    `period` is the literal the arguments hold for it. `op` is the
    predicate's comparison, and `None` for a window bound.
    """

    bound: str
    offset: int
    period: str
    op: str | None


def rolling_bounds(arguments: Mapping[str, JsonValue]) -> tuple[RollingBound, ...]:
    """Every literal of `arguments` a reference governs.

    The window bounds come first, start then end, then the predicates in
    order. Empty for arguments that hold no `rolling`. Read from
    `resolved_arguments`' result, each `period` is the resolved one.
    """
    rolling = arguments.get(ROLLING_PARAMETER)
    if not isinstance(rolling, dict):
        return ()
    found = [
        _bound(bound, reference, arguments.get(bound), op=None)
        for bound in _WINDOW_BOUNDS
        if isinstance(reference := rolling.get(bound), dict)
    ]
    predicates = arguments.get(_PREDICATES)
    for index, predicate in enumerate(predicates if isinstance(predicates, list) else ()):
        if isinstance(predicate, dict) and isinstance(
            reference := predicate.get(ROLLING_PARAMETER), dict
        ):
            op = predicate.get("op")
            found.append(
                _bound(
                    f"{_PREDICATES}/{index}",
                    reference,
                    predicate.get("value"),
                    op=op if isinstance(op, str) else None,
                )
            )
    return tuple(found)


def resolved_arguments(
    arguments: Mapping[str, JsonValue], *, period: str, anchor_key: str
) -> dict[str, JsonValue]:
    """A rolling step's arguments once the newest period is known.

    Each literal a reference governs (`rolling_bounds`) becomes `period` moved
    by the reference's offset: `window_start` and `window_end` of a `growth`
    or a `rank`, and the `value` of each period predicate of a `filter`.
    `anchor_key` is added.

    Args:
        arguments: The step's arguments, holding `rolling`.
        period: The newest period, as `resolve_anchor` returned it.
        anchor_key: The key of the stored resolution `period` was read from.
    """
    resolved = {**arguments, ANCHOR_KEY_PARAMETER: anchor_key}
    predicates = arguments.get(_PREDICATES)
    rewritten = list(predicates) if isinstance(predicates, list) else []
    for governed in rolling_bounds(arguments):
        literal = shift_period(period, governed.offset)
        site, _, index = governed.bound.partition("/")
        if site != _PREDICATES:
            resolved[site] = literal
            continue
        predicate = rewritten[int(index)]
        assert isinstance(predicate, dict)
        rewritten[int(index)] = {**predicate, "value": literal}
    if rewritten:
        resolved[_PREDICATES] = rewritten
    return resolved


def _bound(
    bound: str, reference: Mapping[str, JsonValue], literal: JsonValue, *, op: str | None
) -> RollingBound:
    offset = reference["offset"]
    assert isinstance(offset, int)
    assert isinstance(literal, str)
    return RollingBound(bound=bound, offset=offset, period=literal, op=op)
