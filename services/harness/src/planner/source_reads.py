"""The reads a planner proposal can ask for, made without ADK.

`with_source_reads` makes the reads a refused `propose_plan` asks for (the
first-year history, a scope, a compile on a basis or of a whole, a pin by
search) and applies them in order, and the helpers it calls: the read
functions, the pins' containment checks and their refusals.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any, NamedTuple, cast

from compiler.sources import block_group, party_class, readable_wire_groups
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.scope import ChoiceOrigin, ScopeEntry
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.scope_choices import (
    PROVIDER_AXIS,
    EntitiesBinding,
    PartScope,
    ScopeChoice,
    mark_planner_axes,
    mark_searched_axes,
    option_axis,
    planner_axes,
    switch_entries,
)
from harness.tool_contracts import resolver_digest
from pydantic import JsonValue
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from submission.service import (
    WHOLE_COLUMNS,
    AlignedPin,
    Realignment,
    ScopeRead,
    SearchedPin,
    Unpinned,
    aligned_pins,
    axis_slot,
    basis_unservable,
    block_parties,
    bound_parties,
    first_year_identity,
    part_scope,
    realignments_wanted,
    save_aligned_pin,
    save_searched_pin,
    scopes_wanted,
    share_whole_underivable,
    unpinned_wanted,
    unread_part,
    whole_axis_held,
)
from translator.discovery import (
    SEARCH,
    DefinitionNotResolvedError,
    PartNotResolvedError,
    answered_once_at,
    call_within_entry,
    pin_warnings,
    read_found_scope,
    refusal_retryable,
    resolve_definition,
    resolve_definition_at,
    resolved_spec,
)

from planner.plan import (
    FullHistory,
    PinSource,
    SessionState,
    compile_warnings,
    full_histories,
    full_history,
    held_sources,
    pin_sources,
    record_spec_searched,
    record_unsupported,
    save_found_sources,
    save_full_history,
    save_pin_sources,
    warning_codes,
)
from planner.records import (
    OpenPeriodFault,
)
from planner.tool_results import Outcome, ProblemCode, problem

logger = logging.getLogger(__name__)


def _first_year_wanted(result: Mapping[str, object]) -> str | None:
    """The pinned dataset a refusal says no first year is on record for; `None` for any other answer."""
    detail = result.get("detail")
    if (
        not isinstance(detail, Mapping)
        or detail.get("fault") != OpenPeriodFault.FIRST_YEAR_UNKNOWN.value
    ):
        return None
    spec_id = detail.get("spec_id")
    return spec_id if isinstance(spec_id, str) else None


# The seconds one first-year read may take (`read_full_history`). It runs
# inside a `propose_plan` call, so inside the planner entry's remaining time,
# which `translator.discovery.resolve_definition` already holds it to; this
# keeps a slow Search from spending that whole remainder on one read.
FIRST_YEAR_READ_DEADLINE_SECONDS = 10.0


async def read_full_history(
    surface: ToolSurface,
    identity: Mapping[str, Any],
    *,
    deadline_s: float = FIRST_YEAR_READ_DEADLINE_SECONDS,
) -> FullHistory | None:
    """What Search states about every year the dataset of `identity` holds, read now.

    One `resolve_definition` call for `identity` over no window, the same
    read the translator makes for a pin's copy
    (`translator.discovery.resolve_definition`): data-mcp posts the identity
    to Search once and stores the definition it answers. `None` when Search
    does not answer with that one dataset, or its answer states no first
    year (`planner.plan.full_history`). The window is read for its first
    year only and is never retrieved.

    Raises:
        DefinitionNotResolvedError: If the call did not complete, or did not
            answer within `deadline_s`, and can succeed later (`retryable`).
    """
    spec_id, unwindowed = windowed_identity(identity, {"kind": "all"})
    try:
        async with asyncio.timeout(deadline_s):
            part = await resolve_definition(surface, spec_id=spec_id, identity=unwindowed)
        return full_history(resolved_spec(part, spec_id=spec_id))
    except TimeoutError as exc:
        raise DefinitionNotResolvedError(
            f"Search did not answer in {deadline_s:.0f} s", retryable=True
        ) from exc
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            raise
        return None
    except PartNotResolvedError:
        return None


def _first_year_unreadable(
    refused: Mapping[str, object], exc: DefinitionNotResolvedError
) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a first-year read that did not complete.

    `refused` is the `first_year_unknown` refusal the read was made for; its
    requirements and dataset carry over, under the fault
    `first_year_read_failed`. The same proposal can succeed later, so
    nothing in it needs changing.
    """
    detail = dict(cast("Mapping[str, JsonValue]", refused["detail"]))
    detail["fault"] = OpenPeriodFault.FIRST_YEAR_READ_FAILED.value
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"The first year {detail['spec_id']} holds could not be read: {exc}. The plan is "
        "sound and nothing was recorded; propose it again, or, where the window covers "
        "several years, state the start as a fixed year and roll only the end; a window of "
        "one year cannot, so drop `rolling` and `period_end_open` there for fixed years.",
        outcome=Outcome.REFUSED,
        detail=detail,
    )


# A read of a source, made: applied to the session state, it makes the read's
# writes and answers the refusal the proposal is given, or `None` once the
# read is held. `with_source_reads` makes the reads one refusal asks for at
# once and applies them in order, so a read after a refused one writes nothing.
type SourceRead = Callable[[SessionState], dict[str, object] | None]


def _refusing(refused: dict[str, object]) -> SourceRead:
    """The read answering `refused`, writing nothing."""
    return lambda _: refused


async def compile_on_basis(  # noqa: PLR0911 -- one distinct refusal per way the compile fails
    surface: ToolSurface,
    realignment: Realignment,
    *,
    state: SessionState,
) -> SourceRead:
    """Compile `realignment`'s pin on its calculation's options and hold the pin Search answers.

    Posts the pin's base identity with its scope entries and each option, on
    the axis it names (`harness.scope_choices.option_axis`), as an explicit
    entry from its choice in place of the entry on that axis, the resubmit
    a switch on the plan card posts (`harness.scope_choices.switch_entries`)
    once per option, with what the resubmit posts of the part's entity groups
    (`compiler.sources.block_wire_groups`), through data-mcp's
    `resolve_definition`, which stores the definition it answers. One
    resubmit carries every option, so an input moving on the measure basis
    and the provider scope is compiled once. The answer is held as a
    `planner.plan.AlignedPin` with the answer's digest
    (`harness.tool_contracts.resolver_digest`), its block marking those
    axes as the planner's, beside any axis the origin's block marks
    (`harness.scope_choices.mark_planner_axes`). The datasets the planner was
    shown (`planner.plan.pin_sources`) are left as shown, so a requirement
    pinning the answer's spec_id directly takes none of the origin's
    `planner.plan.PinSource` fields (settled, warnings, asked roles,
    provenance ref). Where it was not shown that spec_id it reads the
    compile's block (`planner.plan.held_sources`), choice provenance and
    planner marks included. The next proposal pins the input to it
    (`submission.service.admit`).

    The compile takes the time a resubmit has
    (`translator.discovery.call_within_entry`): the capability row's call
    timeout, above data-mcp's resubmit budget, within the planner entry's
    time left.

    `state` is read; the writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held, or the refusal
    the proposal answers: `SURFACE_UNAVAILABLE` when Search did not answer
    and can later, `OPERATION_UNSUPPORTED` when it refuses the options or
    answers with no one dataset, or when what it posts of the part's entity
    groups is not readable, since a resubmit posted without them can bind
    another population.

    A declared whole (`Realignment.whole_axis`) is compiled by `compile_whole`.
    """
    if realignment.whole_axis is not None:
        return await compile_whole(surface, realignment, state=state)
    source = held_sources(state).get(realignment.spec_id)
    block = source.spec_block(realignment.spec_id) if source is not None else None
    scope = part_scope(realignment.spec_id, block)
    choices: dict[str, ScopeChoice] = (
        {choice.axis: choice for choice in scope.choices} if scope is not None else {}
    )
    axes = [option_axis(option_id) for option_id in realignment.option_ids]

    def unservable(reason: str) -> SourceRead:
        return lambda into: basis_unservable(
            spec_id=realignment.spec_id,
            option_ids=realignment.option_ids,
            requirement_id=realignment.requirement_id,
            step_id=realignment.step_id,
            reason=reason,
            record=lambda operation: record_unsupported(into, operation),
            proposer=PROPOSE_PLAN_PROPOSER,
        )

    missing = [axis for axis in axes if axis not in choices]
    if block is None or scope is None or missing:
        return unservable(
            f"{realignment.requirement_id!r}'s dataset has no "
            f"{' or '.join(missing or axes)} choice on record"
        )
    switched = scope.entries
    for option_id in realignment.option_ids:
        switched = tuple(
            switch_entries(
                scope.model_copy(update={"entries": switched}),
                choice=choices[option_axis(option_id)],
                option_id=option_id,
            )
        )
    entries = [entry.model_dump(mode="json", exclude_none=True) for entry in switched]
    groups = readable_wire_groups(block)
    if groups is None:
        return unservable(
            f"Search's named sets for {realignment.requirement_id!r}'s dataset are not readable"
        )
    try:
        part, resolved_at = await resolve_definition_at(
            surface,
            spec_id=scope.base.spec_id,
            identity=scope.base.identity,
            groups=groups,
            scope_entries=entries,
        )
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_basis_uncompiled(realignment, str(exc)))
        return unservable(
            f"Search does not count {realignment.requirement_id!r}'s dataset the way its "
            f"calculation's other inputs share ({exc.code or 'refused'})"
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    if not isinstance(spec, Mapping) or not isinstance(spec_id, str):
        return unservable(
            f"Search holds no one dataset counting {realignment.requirement_id!r} the way "
            "its calculation's other inputs share"
        )
    digest = resolver_digest({"parts": [dict(part)]})
    mark_planner_axes(digest, planner_axes(block) | set(axes))

    def hold(into: SessionState) -> None:
        save_aligned_pin(
            into,
            AlignedPin(
                origin=realignment.spec_id,
                option_ids=realignment.option_ids,
                spec_id=spec_id,
                digest=digest,
            ),
            warnings=_origin_warnings(into, realignment.spec_id),
        )
        record_spec_searched(into, spec_id, resolved_at)

    return hold


async def read_scope(
    surface: ToolSurface, read: ScopeRead, *, held: Mapping[str, PinSource], state: SessionState
) -> SourceRead:
    """Read how `read`'s pin, a candidate Search offered with no scope choices,
    is counted, and hold the block Search answers in its place.

    `held` is the blocks on record (`planner.plan.held_sources`), taken
    before any read of the refusal waits. Resolves the pin's identity once
    (`translator.discovery.read_found_scope`) and holds the definite block
    Search answers with its scope choices (`planner.plan.save_found_sources`),
    so the next proposal aligns the input as any other
    (`submission.service._aligned_bases`). It answers `None` once the block
    is held, or the refusal the proposal answers: `SURFACE_UNAVAILABLE` when
    Search did not answer and can later, and `OPERATION_UNSUPPORTED` when it
    refuses the identity or answers with no one dataset, since the input
    cannot then be counted as its calculation's other inputs are
    (`_scope_unservable`).
    """
    source = held.get(read.spec_id)
    block = source.spec_block(read.spec_id) if source is not None else None
    if block is None:
        return _refusing(
            _scope_unservable(read, f"{read.requirement_id!r}'s dataset has no block on record")
        )
    try:
        found = await read_found_scope(surface, read.spec_id, held=held, state=state)
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_scope_not_read(read, str(exc)))
        return _refusing(
            _scope_unservable(
                read,
                f"Search does not say how {read.requirement_id!r}'s dataset is counted "
                f"({exc.code or 'refused'})",
            )
        )
    except (PartNotResolvedError, ValueError) as exc:
        return _refusing(
            _scope_unservable(
                read, f"Search does not say how {read.requirement_id!r}'s dataset is counted: {exc}"
            )
        )
    return lambda into: save_found_sources(into, found)


def _scope_unservable(read: ScopeRead, reason: str) -> dict[str, object]:
    """Return the `OPERATION_UNSUPPORTED` problem for `read`'s input, whose scope
    Search does not state, so it cannot be counted as its calculation's other
    inputs are. `reason` says why, as a clause.

    It refuses this proposal alone and records nothing: the pin cannot be
    aligned, and another series of the input, or a plan leaving it out of
    the calculation, can be, so the ask stays answerable.
    `detail.basis` names the input as `submission.service.basis_unservable`
    does, for the refusal's sentence (`harness.sentences`).
    """
    return problem(
        ProblemCode.OPERATION_UNSUPPORTED,
        f"{reason}, so {read.step_id!r} would mix figures counted different ways. Pin "
        f"{read.requirement_id!r} to another series a search answered in full, or leave it "
        f"out of {read.step_id!r}.",
        outcome=Outcome.UNSUPPORTED,
        action=PROPOSE_PLAN_PROPOSER.action(needs=("evidence_requirements",)),
        detail={
            "basis": {
                "spec_id": read.spec_id,
                "option_ids": [],
                "requirement_id": read.requirement_id,
                "step_id": read.step_id,
            }
        },
    )


def _origin_warnings(state: SessionState, origin: str) -> tuple[str, ...]:
    """The warning codes on the record of `origin`, a pin a compile is made from (`translator.discovery.pin_warnings`), which the compile keeps."""
    return pin_warnings(
        origin,
        entry=None,
        shown=pin_sources(state),
        compiled=aligned_pins(state),
        recorded=compile_warnings(state),
    )


async def compile_whole(  # noqa: PLR0911 -- one distinct refusal per way the compile fails
    surface: ToolSurface,
    realignment: Realignment,
    *,
    state: SessionState,
) -> SourceRead:
    """Compile the declared whole `realignment` names from its part, and hold it as a pin.

    The part is the pin `realignment.spec_id` and the axis left open
    `realignment.whole_axis` (`planner.plan.WholeOf.axis`). Posts the part's
    base identity with that axis opened, under the spec_id that identity
    hashes to, the part's entity groups, and one explicit entry from each of
    its other choices holding its active option, with the concept of the
    part's entry on that axis (`_opened`). The sector is opened by
    unbinding the slot the part's sector choice binds
    (`submission.service.axis_slot`). A donor or recipient is opened by
    dropping that role's column from the identity
    (`submission.service.WHOLE_COLUMNS`), with the set named on it and the
    choice adding entities to it, the region, so Search answers with the
    total over every party there that holds the part, and what population
    that is stays Search's answer. Search keeps the part's measure basis,
    and its region on another role, which an option the whole cannot serve
    would otherwise leave to Search's default. The provider scope of a
    donor whole is posted as a carried entry: Search counts the whole on it
    where the whole offers it, and otherwise drops it and counts the whole
    on its own default, since the scope names whose spending a named donor
    counts and Search offers the inclusive scope on no total naming no
    donor. The part keeps its own scope, and the whole's caveat says what
    each side counts (`submission.service._whole_shares`). A recipient
    whole names the part's donor, so Search can count it on the part's
    scope, and its provider scope is posted as an explicit entry that
    Search refuses where it cannot hold it, as the other choices are.
    No model call is made. The part is read off the blocks on record
    (`planner.plan.held_sources`), so a part alignment moved reads its
    compile. The whole takes its choices
    from its part. Each the whole holds on the part's option carries the
    part's provenance, and the block marks as the planner's
    (`harness.scope_choices.mark_planner_axes`) those of them the part's
    block marks, so the whole reads as settled the way the part was and a
    later search carries what the part's carries. It is held as a
    `planner.plan.AlignedPin` from the part on that axis, which every
    whole declaring the part and the axis pins. Like a compile from
    `compile_on_basis`, it is held there alone.

    `state` is read; the writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held,
    `SURFACE_UNAVAILABLE` when Search did not answer and can later, or
    `SHARE_WHOLE_UNDERIVABLE` under a cause
    naming why: the part's pin is not among the datasets shown
    (`part_not_shown`), is shown with no block (`part_block_missing`), holds
    scope choices this release cannot read (`scope_unreadable`), binds
    nothing on the axis to leave open (`axis_not_held`), or names a set on a
    role the whole keeps whose posted identity (`compiler.sources.block_wire_groups`)
    is not readable, which the whole would otherwise be posted without
    (`groups_unreadable`); or Search refuses
    the whole (as `requested_option_unavailable` where it cannot hold an
    option the part holds), or answers with no one dataset (`whole_refused`),
    or with a donor or recipient whole that Search shows does not hold a party
    the part names on that role (`part_outside_whole`), or bound to one total
    that states neither its members nor a whole class of party, so nothing
    shows it does (`whole_members_unstated`) (`_parties_outside`).
    """
    part_pin, axis = realignment.spec_id, cast(str, realignment.whole_axis)

    def underivable(reason: str, cause: str = "whole_refused") -> SourceRead:
        return _refusing(
            share_whole_underivable(
                whole_id=realignment.requirement_id,
                part_spec_id=part_pin,
                reason=reason,
                cause=cause,
            )
        )

    source = held_sources(state).get(part_pin)
    block = source.spec_block(part_pin) if source is not None else None
    scope = part_scope(part_pin, block)
    if source is None or block is None or scope is None or not whole_axis_held(scope, axis):
        return underivable(*unread_part(part_pin, axis, source=source, block=block, scope=scope))
    column = WHOLE_COLUMNS.get(axis)
    groups = readable_wire_groups(block, leaving_out=column)
    if groups is None:
        return underivable(
            f"Search's named sets for {part_pin} are not readable", "groups_unreadable"
        )
    identity, choices = _opened(scope, axis)
    concepts = {entry.axis: entry.concept for entry in scope.entries}
    entries = [
        ScopeEntry(
            axis=choice.axis,
            option_id=choice.active,
            provenance="carried"
            if axis == "donor" and choice.axis == PROVIDER_AXIS
            else "explicit",
            concept=concepts.get(choice.axis),
            origin=ChoiceOrigin(choice_id=choice.id),
        )
        for choice in choices
    ]
    try:
        part, resolved_at = await resolve_definition_at(
            surface,
            spec_id=spec_id_for(canonicalize(identity)),
            identity=identity,
            groups=groups,
            scope_entries=[entry.model_dump(mode="json", exclude_none=True) for entry in entries],
        )
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_basis_uncompiled(realignment, str(exc)))
        return underivable(
            f"Search does not count {part_pin}'s dataset over every {axis} the way the part is "
            f"counted ({exc.code or 'refused'}): {exc}"
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    if not isinstance(spec, Mapping) or not isinstance(spec_id, str):
        return underivable(f"Search holds no one dataset counting every {axis} of {part_pin}")
    held = (
        _parties_outside(scope, spec, column, part_block=block)
        if column is not None
        else _Containment()
    )
    if held.outside or held.unshown:
        return underivable(*_uncontained(part_pin, axis, held))
    answered = _answered_options(spec_id, spec)
    inherited: dict[str, str] = {
        choice.axis: choice.provenance
        for choice in choices
        if answered.get(choice.axis) == choice.active
    }
    digest = resolver_digest({"parts": [_as_whole(part, inherited, entries, answered)]})
    mark_planner_axes(digest, planner_axes(block) & inherited.keys())

    def hold(into: SessionState) -> None:
        save_aligned_pin(
            into,
            AlignedPin(
                origin=part_pin,
                option_ids=(),
                spec_id=spec_id,
                digest=digest,
                whole_axis=axis,
            ),
            warnings=_origin_warnings(into, part_pin),
        )
        record_spec_searched(into, spec_id, resolved_at)

    return hold


async def run_together[T](calls: Iterable[Awaitable[T]]) -> list[T]:
    """Each of `calls`' results in order, run at once.

    The first to raise cancels the rest, which are waited for before it is
    raised, so none is left running.
    """
    tasks = [asyncio.ensure_future(call) for call in calls]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        await _cancel_and_drain(tasks)


async def _cancel_and_drain(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancel each of `tasks` still running, wait for every one, and retrieve each
    exception, so none is reported as never retrieved."""
    running = [task for task in tasks if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running)
    for task in tasks:
        if not task.cancelled():
            task.exception()


def _opened(scope: PartScope, axis: str) -> tuple[dict[str, Any], list[ScopeChoice]]:
    """The identity and scope choices the whole of the part `scope` records over
    every member of `axis` posts (`compile_whole`).

    On the sector, the base identity with the slot the sector choice binds
    unbound (`axis_slot`), and every choice but the sector. On a donor or
    recipient, the base identity without that role's column
    (`WHOLE_COLUMNS`), and every choice but one adding entities to that role,
    the region's. The whole posts the part's groups on other roles
    (`compiler.sources.block_wire_groups`).
    """
    identity = copy.deepcopy(scope.base.identity)
    column = WHOLE_COLUMNS.get(axis)
    if column is None:
        slot = axis_slot(scope, axis)
        for held in identity.get("slots") or ():
            if held.get("property") == slot:
                held["binding"] = {"state": "unbound"}
        return identity, [choice for choice in scope.choices if choice.axis != axis]
    identity["entities"] = {
        role: refs for role, refs in (identity.get("entities") or {}).items() if role != column
    }
    return identity, [
        choice
        for choice in scope.choices
        if not any(
            isinstance(option.binding, EntitiesBinding) and option.binding.role == column
            for option in choice.options
        )
    ]


def _answered_options(spec_id: str, spec: Mapping[str, Any]) -> dict[str, str]:
    """The option each scope choice of `spec`, the Search spec block of `spec_id`,
    holds active, by axis (`compile_whole`). Empty where it records no choices this
    release reads (`submission.service.part_scope`)."""
    scope = part_scope(spec_id, spec)
    return {choice.axis: choice.active for choice in scope.choices} if scope is not None else {}


class _Containment(NamedTuple):
    """The parties a part names on a donor or recipient whole's role that Search's
    answer shows are outside the whole (`outside`), and those it does not show are
    inside or outside (`unshown`) (`_parties_outside`)."""

    outside: tuple[str, ...] = ()
    unshown: tuple[str, ...] = ()


def _uncontained(part_pin: str, axis: str, held: _Containment) -> tuple[str, str]:
    """Why the whole over every `axis` of the part pinned `part_pin` does not contain
    it, `held` its containment (`_parties_outside`), as a clause and its
    `share_whole_underivable` cause: `part_outside_whole` where Search shows a
    party the part names is outside, else `whole_members_unstated`."""
    if held.outside:
        return (
            f"Search counts {part_pin}'s dataset over a {axis} total that does not hold "
            f"{', '.join(held.outside)}, which the part names, so the total would not "
            "contain it",
            "part_outside_whole",
        )
    return (
        f"Search counts {part_pin}'s dataset over a {axis} total that lists no members "
        f"and no class of party holding {', '.join(held.unshown)}, which the part "
        "names, so nothing shows the total holds it",
        "whole_members_unstated",
    )


# Search's cause for an entry dropped because the part has no choice on its axis.
_AXIS_ABSENT = "axis_absent"

# The class of party a total holding a whole population can leave out and still
# hold every party the part names: a residual names no party (`common.place_type`).
_NOT_A_PARTY_CLASS = "residual"


def _parties_outside(
    scope: PartScope, spec: Mapping[str, Any], column: str, *, part_block: Mapping[str, Any]
) -> _Containment:
    """Whether the whole Search answered with, `spec`, holds each party the part
    `scope` records names on the role `column` (`compile_whole`), `part_block`
    the part's spec block.

    A whole binding nothing on `column` counts every party its dataset holds
    there, the part's own dataset, so it holds every party the part names.
    One binding parties there holds those it binds, and of a total it binds,
    the members Search lists for the group naming it on `column`
    (`ResolvedGroup.members`, DAC countries' total). A total holding a whole
    population (`ResolvedGroup.whole_of`) holds every party of its classes and
    no other: the part's party where its ref shows it is of one
    (`compiler.sources.party_class`, a country official), and not where its
    ref shows another. It holds a party whose ref shows no class where its
    classes are every class the side publishes (`ResolvedGroup.side_classes`),
    residuals aside; where the side publishes another class, or Search states
    no classes for the side but residuals, nothing here says which class that
    party is, so it is unshown.
    Any other party is outside where Search binds parties or lists a total's
    members there, and unshown where the total it binds states neither its
    members nor a whole class. Where Search's group on `column` is not
    readable (`compiler.sources.block_group`), as one naming a class of party
    outside `PartyClass`, every party it does not bind is unshown; a group on
    another role is not read. The parties the part and the whole bind on
    `column` are those their scope records bind
    (`submission.service.bound_parties`), a choice's among them; where the
    whole's record does not read, those its definition binds
    (`submission.service.block_parties`).

    A party of the part the whole neither binds nor lists is read through the
    part's own group on `column` where it is that group's total (the G7's): the
    whole holds it where it holds every member Search lists for the group, does
    not where one is outside, and nothing shows it does where Search lists no
    members, or the part's group there is not readable.
    """
    named = bound_parties(scope, column)
    bound = block_parties(spec, column, scope=part_scope(str(spec.get("spec_id")), spec))
    if not bound:
        return _Containment()
    try:
        group = block_group(spec, column)
    except ValueError:
        return _Containment(unshown=tuple(party for party in named if party not in bound))
    members = set(group.members) if group is not None else set()

    def holds(party: str) -> bool | None:
        """Whether the whole holds `party` as itself, `None` where nothing shows it."""
        if party in bound or party in members:
            return True
        if group is not None and group.whole_of:
            if (held := party_class(party)) is not None:
                return held in group.whole_of
            classes = set(group.side_classes) - {_NOT_A_PARTY_CLASS}
            return True if classes and classes <= set(group.whole_of) else None
        if members or group is None or group.total not in bound:
            return False
        return None

    try:
        part_group = block_group(part_block, column)
    except ValueError:
        part_group, part_readable = None, False
    else:
        part_readable = True
    outside: list[str] = []
    unshown: list[str] = []
    for party in named:
        held = holds(party)
        if held:
            continue
        if not part_readable:
            # The party may be the total of a group nothing here reads.
            held = None
        elif part_group is not None and party == part_group.total:
            each = [holds(member) for member in part_group.members]
            if not each:
                held = None
            elif all(each):
                continue
            else:
                held = False if False in each else None
        (outside if held is False else unshown).append(party)
    return _Containment(outside=tuple(outside), unshown=tuple(unshown))


def _as_whole(
    part: Mapping[str, Any],
    provenance: Mapping[str, str],
    posted: Iterable[ScopeEntry],
    answered: Mapping[str, str],
) -> dict[str, Any]:
    """A copy of `part`, the `definite` Search part answering a whole, whose scope
    choice on each axis of `provenance` carries the provenance given for that axis,
    and whose dropped entries leave out each carried entry of `posted`, the entries
    the whole was posted with, that changed nothing (`compile_whole`). `answered` is
    the option the whole holds on each axis (`_answered_options`).

    A carried entry the harness posted changed nothing where the whole holds its
    option all the same, or where Search dropped it because the whole has no
    choice on its axis (`axis_absent`), as a DAC2a total naming no donor has no
    provider axis, so the user chose nothing there that failed to apply. One Search
    dropped for a whole that counts another option, as DAC countries' total ODA
    beside a part on bilateral ODA, stays, so the whole's reading says it did not
    apply. The whole's caveat says what each side counts
    (`submission.service._whole_shares`). Any other dropped entry stays."""
    copied = copy.deepcopy(dict(part))
    resolved = copied["spec"].get("resolved")
    scope = resolved.get("scope_choices") if isinstance(resolved, dict) else None
    if not isinstance(scope, dict):
        return copied
    for choice in scope.get("choices") or ():
        if isinstance(choice, dict) and choice.get("axis") in provenance:
            choice["provenance"] = provenance[choice["axis"]]
    carried = {
        (entry.axis, entry.option_id, entry.origin.choice_id)
        for entry in posted
        if entry.provenance == "carried" and isinstance(entry.origin, ChoiceOrigin)
    }
    if carried and isinstance(scope.get("dropped_entries"), list):
        scope["dropped_entries"] = [
            dropped
            for dropped in scope["dropped_entries"]
            if not _changed_nothing(dropped, carried, answered)
        ]
    return copied


def _changed_nothing(
    dropped: object, carried: Collection[tuple[str, str, str]], answered: Mapping[str, str]
) -> bool:
    """Whether `dropped`, one of Search's dropped entries, is a carried entry the
    harness posted, `carried` keying each by axis, option and choice, that changed
    nothing because the whole holds its option on its axis (`answered`) or has no
    choice there (`_as_whole`)."""
    if not isinstance(dropped, Mapping):
        return False
    entry = dropped.get("entry")
    if not isinstance(entry, Mapping) or entry.get("provenance") != "carried":
        return False
    origin = entry.get("origin")
    choice_id = origin.get("choice_id") if isinstance(origin, Mapping) else None
    axis, option_id = entry.get("axis"), entry.get("option_id")
    if (axis, option_id, choice_id) not in carried:
        return False
    return dropped.get("cause") == _AXIS_ABSENT or answered.get(str(axis)) == option_id


async def pin_by_search(surface: ToolSurface, unpinned: Unpinned) -> SourceRead:
    """Search for `unpinned`, a calculation's input the planner pinned no dataset for, and hold what Search settles on.

    Posts the `search` part the translator's own discovery sends for the
    requirement (`translator.discovery.search_part`), in the planner's words
    and with no message of the user's, so every axis an entry settled on
    Search's answer is marked as the planner's
    (`harness.scope_choices.mark_planner_axes`). A `definite` answer is
    shown to the planner as a source of its own, with the codes of its
    part's warnings and the request's
    (`harness.tool_contracts.graph_envelope`; `planner.plan.save_pin_sources`),
    and held as a
    `submission.service.SearchedPin` with its digest, and the time the
    search was made as one settling on its spec_id
    (`planner.plan.record_spec_searched`), so the next proposal
    pins it and the planner seam stores it as a reading of the conversation
    (`harness.planner_seam.store_searched_pin_readings`). Any other answer is
    held as a search that settled on nothing, under Search's part status, or
    `refused` for a refusal, and the next proposal is refused
    (`submission.service._aligned_bases`). The search takes the time the
    translator's own discovery gives one
    (`translator.discovery.call_within_entry`), and is made once an
    invocation (`translator.discovery.answered_once_at`), so a read the
    proposal's other refused search discarded (`with_source_reads`) is not
    repeated.

    The holds are made when the read returned is applied (`SourceRead`). It
    answers `None` once the search is held, or `SURFACE_UNAVAILABLE` when
    Search did not answer, or refused in a way that can succeed later
    (`translator.discovery.refusal_retryable`); nothing is held then, so the
    next proposal searches again. A call that failed for a reason no retry
    changes (a `ToolSurfaceUnavailableError` not marked `retryable`) is held
    as a refused search.
    """
    arguments = {"parts": [unpinned.part]}
    try:
        answer, searched_at = await answered_once_at(
            SEARCH,
            arguments,
            lambda: call_within_entry(surface, tool=SEARCH, arguments=arguments),
        )
    except ToolSurfaceUnavailableError as exc:
        if exc.retryable:
            return _refusing(_search_unavailable(unpinned, str(exc)))
        # A search no later attempt can make is held as a refused one.
        refused = SearchedPin(
            requirement_key=unpinned.requirement_key,
            status="refused",
            asked_roles=unpinned.asked_roles,
            open_roles=unpinned.open_roles,
        )
        return lambda into: save_searched_pin(into, refused)
    if answer.is_error and refusal_retryable(answer):
        refusal = answer.structured_content or {}
        cause = refusal.get("message") or answer.text or "Search refused it for now"
        return _refusing(_search_unavailable(unpinned, str(cause)))
    envelope = None if answer.is_error else (answer.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
    parts = envelope.get("parts") if isinstance(envelope, dict) else None
    part = parts[0] if isinstance(parts, list) and len(parts) == 1 else None
    status = part.get("status") if isinstance(part, dict) else None
    spec = part.get("spec") if isinstance(part, dict) and status == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, dict) else None
    searched = SearchedPin(
        requirement_key=unpinned.requirement_key,
        status="refused"
        if answer.is_error
        else status
        if isinstance(status, str)
        else "unreadable",
        asked_roles=unpinned.asked_roles,
        open_roles=unpinned.open_roles,
    )
    if not isinstance(spec_id, str) or not isinstance(envelope, dict):
        return lambda into: save_searched_pin(into, searched)
    digest = resolver_digest(cast(dict[str, Any], envelope))
    mark_searched_axes(digest, spec_id=spec_id)
    shown = PinSource(
        block=digest["spec_block"],
        warnings=warning_codes(digest["warnings"]),
        settled=True,
        asked_roles=unpinned.asked_roles,
    )

    def hold(into: SessionState) -> None:
        save_pin_sources(into, {**pin_sources(into), spec_id: shown})
        save_searched_pin(into, searched.model_copy(update={"spec_id": spec_id, "digest": digest}))
        record_spec_searched(into, spec_id, searched_at)

    return hold


def _search_unavailable(unpinned: Unpinned, cause: str) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a search of `unpinned` that can succeed later.

    The same proposal can succeed later, so nothing in it needs changing.
    """
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"{unpinned.requirement_id!r}, an input of {unpinned.step_id!r}, pins no dataset, "
        f"and Search could not be asked for one: {cause}. The plan is sound and nothing was "
        "recorded; propose it again.",
        outcome=Outcome.REFUSED,
        detail={"search": unpinned.model_dump(mode="json")},
    )


def _scope_not_read(read: ScopeRead, cause: str) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a read of a candidate's scope that did not complete.

    The same proposal can succeed later, so nothing in it needs changing.
    """
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"How {read.requirement_id!r} is counted could not be read from Search: {cause}. "
        "The plan is sound and nothing was recorded; propose it again.",
        outcome=Outcome.REFUSED,
        detail={"read": read._asdict()},
    )


def _basis_uncompiled(realignment: Realignment, cause: str) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a compile on another basis or provider scope that did not complete.

    The same proposal can succeed later, so nothing in it needs changing.
    """
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"{realignment.requirement_id!r} could not be counted the way the other inputs of "
        f"{realignment.step_id!r} are: {cause}. The plan is sound and nothing was recorded; "
        "propose it again.",
        outcome=Outcome.REFUSED,
        detail={"basis": realignment.detail()},
    )


async def with_source_reads(  # noqa: PLR0911 -- one return per kind of read a refusal asks for
    propose: Callable[[], dict[str, object]], *, state: SessionState, surface: ToolSurface
) -> dict[str, object]:
    """`propose`'s answer, once every read of a source it was refused for is made.

    A refusal records nothing, so `propose` runs again once the read is held.
    Two refusals ask for one. A plan rolling from a dataset's first year
    needs that first year (`read_full_history`, held by
    `planner.plan.save_full_history`): one read serves every requirement, and
    every later proposal of the turn, that pins the dataset; an answer
    stating no first year is held too, and the next refusal says so
    (`first_year_read_failed`). A calculation one of whose inputs pins no
    dataset while another carries a basis choice needs that input searched
    for and pinned (`BASIS_UNKNOWN`, `pin_by_search`), as does the part of
    a declared whole; a search held
    for the requirement's content is made once. A calculation whose inputs are
    counted on different measure bases or provider scopes needs each input
    it names compiled on their shared options, every option of one input in
    one compile (`BASES_UNALIGNED`, `compile_on_basis`), and a declared
    whole needs its part compiled with the axis left open (`compile_whole`).
    An input pinned to a candidate Search offered with no scope choices needs
    them read first (`SCOPE_UNREAD`, `read_scope`), once a pin, however many
    inputs pin it, and once a call: a refusal naming a pin already read
    answers `OPERATION_UNSUPPORTED`, since Search's answer left it unread;
    a pin already compiled on a set of options, or a whole already compiled,
    is asked for once, so a refusal naming one
    is answered as it stands, and inputs sharing a pin and a set of options
    compile it once (`_alignment_key`).

    The reads one refusal asks for run at once. Their writes are made in the
    order the refusal names them, up to the first read refused, whose
    refusal is answered as soon as it and every read before it are made,
    and the reads after it are cancelled (`_held_in_order`), so the state is
    the one reading them one after another leaves. A search the refused read's writes discarded keeps
    its answer for the invocation (`pin_by_search`), so the next proposal
    searches only for what was refused.
    """
    scopes_read: set[str] = set()
    while True:
        result = propose()
        pin = _first_year_wanted(result)
        if pin is not None:
            if pin in full_histories(state):
                return result
            identity = first_year_identity(pin, state)
            if identity is None:
                return result
            try:
                history = await read_full_history(surface, identity)
            except DefinitionNotResolvedError as exc:
                return _first_year_unreadable(result, exc)
            save_full_history(state, pin, history)
            continue
        unpinned = unpinned_wanted(result)
        if unpinned:
            refused = await _held_in_order(
                (pin_by_search(surface, item) for item in unpinned), state=state
            )
            if refused is not None:
                return refused
            continue
        reads = scopes_wanted(result)
        if reads:
            # A pin is read once a call: Search answered a read that left it
            # unread, and will answer it so again.
            again = next((item for item in reads if item.spec_id in scopes_read), None)
            if again is not None:
                return _scope_unservable(
                    again,
                    f"Search's answer does not say how {again.requirement_id!r}'s dataset "
                    "is counted",
                )
            by_pin: dict[str, ScopeRead] = {}
            for item in reads:
                by_pin.setdefault(item.spec_id, item)
            held = held_sources(state)
            refused = await _held_in_order(
                (read_scope(surface, item, held=held, state=state) for item in by_pin.values()),
                state=state,
            )
            scopes_read.update(by_pin)
            if refused is not None:
                return refused
            continue
        wanted = realignments_wanted(result)
        done = {
            _alignment_key(pin.origin, pin.option_ids, pin.whole_axis)
            for pin in aligned_pins(state)
        }
        keys = [_alignment_key(item.spec_id, item.option_ids, item.whole_axis) for item in wanted]
        if not wanted or any(key in done for key in keys):
            return result
        compiles: dict[tuple[str, tuple[str, ...], str | None], Realignment] = {}
        for key, item in zip(keys, wanted, strict=True):
            compiles.setdefault(key, item)
        refused = await _held_in_order(
            (compile_on_basis(surface, item, state=state) for item in compiles.values()),
            state=state,
        )
        if refused is not None:
            return refused


async def _held_in_order(
    reads: Iterable[Awaitable[SourceRead]], *, state: SessionState
) -> dict[str, object] | None:
    """Run `reads` at once and apply each to `state` in order, answering the first refused.

    Each read is applied once it and every read before it are made, so a
    refusal is answered without waiting for the reads after it, which are
    cancelled. A read that raises raises once every read before it is
    applied without a refusal, and cancels the reads after it. Every read
    takes `state` before its first wait, so none reads what another wrote.
    """
    tasks = [asyncio.ensure_future(read) for read in reads]
    try:
        for task in tasks:
            refused = (await task)(state)
            if refused is not None:
                return refused
        return None
    finally:
        await _cancel_and_drain(tasks)


def _alignment_key(
    spec_id: str, option_ids: Iterable[str], whole_axis: str | None
) -> tuple[str, tuple[str, ...], str | None]:
    """The compile that puts the pin `spec_id` on the options `option_ids`, or leaves its choice on `whole_axis` open, as a key.

    The options are sorted and deduplicated, so one compile naming the same
    options in another order is the same work.
    """
    return spec_id, tuple(sorted(set(option_ids))), whole_axis
