"""The reads a planner proposal can ask for, made outside a model turn.

`with_source_reads` makes the reads a refused `propose_plan` asks for (the
first-year history, a scope, a compile on a basis or of a whole, a pin by
search) and applies them in order, and the helpers it calls: the read
functions, the pins' containment checks and their refusals.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from datetime import datetime
from typing import Any, NamedTuple, cast

from compiler.sources import (
    WHOLE_COLUMNS,
    open_entity_axes_of,
    open_side_column,
    readable_wire_groups,
)
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.scope_choices import (
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
from submission.service import (
    GROUPS_UNREADABLE,
    HARNESS_WHOLE_CAUSES,
    PART_MOVED,
    PART_UNREAD,
    SUM_OF_MEMBERS,
    WHOLE_REFUSED,
    AlignedPin,
    Realignment,
    ScopeRead,
    SearchedPin,
    Unpinned,
    aligned_pins,
    basis_unservable,
    first_year_identity,
    part_scope,
    realignments_wanted,
    save_aligned_pin,
    save_searched_pin,
    scopes_wanted,
    share_whole_underivable,
    stated_alone_unanswered,
    stated_compile_unanswered,
    unpinned_wanted,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
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
    UnsupportedOperation,
    compile_warnings,
    full_histories,
    full_history,
    held_sources,
    pin_sources,
    record_optional_decline,
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
from planner.stated_scope import (
    CombinedRefusal,
    combined_refused,
    record_combined_refused,
    record_stated_uncompiled,
    stated_compile_unreachable,
)
from planner.tool_results import Outcome, ProblemCode, problem
from planner.workflow import OpenEntityAxis

logger = logging.getLogger(__name__)


def _first_year_wanted(result: Mapping[str, object]) -> str | None:
    """The pinned dataset a first-year refusal names, or `None` for any other answer."""
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
# which `translator.discovery.resolve_definition` already holds it to. The
# deadline keeps a slow Search from spending that whole remainder on one read.
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

    `refused` is the `first_year_unknown` refusal the read was made for. Its
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


class _Compiled(NamedTuple):
    """Search's compile of a pin (`compile_on_basis`): its part, spec_id and when it
    was resolved."""

    part: Mapping[str, Any]
    spec_id: str
    resolved_at: datetime | None


# The cause `compile_on_basis` gives where Search refuses stating no code, where it
# does not answer (which states none), where it answers with no one dataset, where
# the call cannot reach it for good, and where the pin's record cannot be posted (no
# choice on the axis, unreadable named sets: `GROUPS_UNREADABLE`).
_REFUSED = "refused"
_UNANSWERED = "unanswered"
_NO_ONE_DATASET = "no_one_dataset"
_UNREACHABLE = "unreachable"
_NO_CHOICE = "no_choice_on_record"


class _Uncompiled(NamedTuple):
    """Why Search answered no compile of a pin (`compile_on_basis`): its answer in
    words (`cause`) and as a code (`code`), whether it can answer later (`retryable`),
    the refusal's reason (`reason`), and whether the call cannot reach Search for good,
    which is no answer of Search's about the pin (`unreachable`)."""

    cause: str
    code: str
    retryable: bool
    reason: str
    unreachable: bool = False


def _alone_unanswered(realignment: Realignment, option_id: str, cause: str) -> SourceRead:
    """The read answering `submission.service.stated_alone_unanswered` for
    `realignment`'s pin, whose compile on `option_id` alone Search did not answer."""
    return lambda into: stated_alone_unanswered(
        realignment, option_id=option_id, shown=held_sources(into), cause=cause
    )


def _stated_refused(spec_id: str, option_id: str, cause: str) -> SourceRead:
    """The read recording that Search refused for good to compile `spec_id` on the
    stated provider option `option_id` (`planner.stated_scope.record_stated_uncompiled`),
    so the next proposal is refused for a pin that cannot count it, with the ask."""

    def record(into: SessionState) -> None:
        record_stated_uncompiled(into, spec_id=spec_id, option_id=option_id, cause=cause)

    return record


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
    (`translator.discovery.call_within_entry`). That is the capability row's
    call timeout, above data-mcp's resubmit budget, within the planner
    entry's time left.

    `state` is read. The writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held, or the refusal
    the proposal answers: `SURFACE_UNAVAILABLE` when Search did not answer
    and can later, `OPERATION_UNSUPPORTED` when it refuses the options or
    answers with no one dataset, or when what it posts of the part's entity
    groups is not readable, since a resubmit posted without them can bind
    another population. A pin compiled on the provider option its
    requirement declares, and nothing else
    (`Realignment.stated`), answers `submission.service.stated_compile_unanswered`
    where Search can answer later, and otherwise records Search's code
    (`planner.stated_scope.record_stated_uncompiled`), so the next proposal is
    refused for a pin that cannot count the option.

    A compile moving a calculation's axis beside the stated option
    (`Realignment.stated_option`) that Search refuses for good is held for the turn
    (`planner.stated_scope.record_combined_refused`), and the pin is compiled on the
    stated option alone, since Search may refuse either move. A later proposal posts
    only that second compile. Where Search refuses it for good too, its code is
    recorded as above, since the pin cannot count what the user asked for. Where
    Search does not answer it, the proposal answers `stated_alone_unanswered`, asking
    to propose again once. A second non-answer, or an alone compile that succeeds,
    leaves the first compile's refusal standing (`OPERATION_UNSUPPORTED`).

    A compile carrying a stated option that cannot reach Search for good, either of
    the two, is a fault of the system and no answer about the pin: it records
    nothing against the option and ends the entry
    (`planner.stated_scope.stated_compile_unreachable`).

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

    def unservable(reason: str, *, code: str) -> SourceRead:
        if realignment.stated:
            return stated_uncompiled(code, retryable=False)
        return lambda into: basis_unservable(
            spec_id=realignment.spec_id,
            option_ids=realignment.option_ids,
            requirement_id=realignment.requirement_id,
            step_id=realignment.step_id,
            reason=reason,
            record=lambda operation: record_unsupported(into, operation),
            proposer=PROPOSE_PLAN_PROPOSER,
        )

    def stated_uncompiled(cause: str, *, retryable: bool) -> SourceRead:
        if retryable:
            return lambda into: stated_compile_unanswered(
                realignment, shown=held_sources(into), cause=cause
            )
        (option_id,) = realignment.option_ids
        return _stated_refused(realignment.spec_id, option_id, cause)

    missing = [axis for axis in axes if axis not in choices]
    if block is None or scope is None or missing:
        return unservable(
            f"{realignment.requirement_id!r}'s dataset has no "
            f"{' or '.join(missing or axes)} choice on record",
            code=_NO_CHOICE,
        )
    groups = readable_wire_groups(block)
    if groups is None:
        return unservable(
            f"Search's named sets for {realignment.requirement_id!r}'s dataset are not readable",
            code=GROUPS_UNREADABLE,
        )
    wire_groups = groups

    async def attempt(option_ids: tuple[str, ...]) -> _Compiled | _Uncompiled:
        """Search's compile of the pin on `option_ids`, or why it answered none."""
        switched = scope.entries
        for option_id in option_ids:
            switched = tuple(
                switch_entries(
                    scope.model_copy(update={"entries": switched}),
                    choice=choices[option_axis(option_id)],
                    option_id=option_id,
                )
            )
        try:
            part, resolved_at = await resolve_definition_at(
                surface,
                spec_id=scope.base.spec_id,
                identity=scope.base.identity,
                groups=wire_groups,
                scope_entries=[e.model_dump(mode="json", exclude_none=True) for e in switched],
            )
        except DefinitionNotResolvedError as exc:
            code = exc.code or (
                _UNANSWERED
                if exc.retryable
                else _REFUSED
                if exc.refused
                else _NO_ONE_DATASET
                if exc.no_part
                else _UNREACHABLE
            )
            return _Uncompiled(
                cause=str(exc),
                code=code,
                retryable=exc.retryable,
                reason=(
                    f"Search does not count {realignment.requirement_id!r}'s dataset the way "
                    f"its calculation's other inputs share ({code})"
                ),
                unreachable=code == _UNREACHABLE,
            )
        spec = part.get("spec") if part.get("status") == "definite" else None
        if not isinstance(spec, Mapping) or not isinstance(spec.get("spec_id"), str):
            return _Uncompiled(
                cause="no one dataset",
                code=_NO_ONE_DATASET,
                retryable=False,
                reason=(
                    f"Search holds no one dataset counting {realignment.requirement_id!r} the "
                    "way its calculation's other inputs share"
                ),
            )
        return _Compiled(part=part, spec_id=spec["spec_id"], resolved_at=resolved_at)

    stated_option = None if realignment.stated else realignment.stated_option
    key = (realignment.spec_id, realignment.option_ids)
    held = combined_refused(state).get(key) if stated_option is not None else None
    compiled = (
        await attempt(realignment.option_ids)
        if held is None
        else _Uncompiled(cause=held.code, code=held.code, retryable=False, reason=held.reason)
    )
    if (
        isinstance(compiled, _Uncompiled)
        and compiled.unreachable
        and (realignment.stated or stated_option is not None)
    ):
        return _refusing(
            stated_compile_unreachable(
                spec_id=realignment.spec_id, option_ids=realignment.option_ids
            )
        )
    if isinstance(compiled, _Uncompiled):
        if compiled.retryable and realignment.stated:
            return stated_uncompiled(compiled.code, retryable=True)
        if compiled.retryable:
            return _refusing(_basis_uncompiled(realignment, compiled.cause))
        if stated_option is None:
            return unservable(compiled.reason, code=compiled.code)
        # The compile also carries a calculation's move, and Search may refuse either.
        # The stated option alone decides whether the pin can count what the user
        # asked for. The refused compile is held for the turn, so proposing again
        # posts only the stated option's.
        refusal = held or CombinedRefusal(code=compiled.code, reason=compiled.reason)
        alone = await attempt((stated_option,))
        if isinstance(alone, _Uncompiled) and alone.unreachable:
            return _refusing(
                stated_compile_unreachable(spec_id=realignment.spec_id, option_ids=(stated_option,))
            )
        if isinstance(alone, _Uncompiled) and alone.retryable and refusal.unanswered:
            # Search did not answer it once already: the calculation's refusal stands.
            answer = unservable(refusal.reason, code=refusal.code)
        elif isinstance(alone, _Uncompiled) and alone.retryable:
            refusal = refusal._replace(unanswered=refusal.unanswered + 1)
            answer = _alone_unanswered(realignment, stated_option, alone.code)
        elif isinstance(alone, _Uncompiled):
            answer = _stated_refused(realignment.spec_id, stated_option, alone.code)
        else:
            answer = unservable(compiled.reason, code=compiled.code)

        def holding_refused(into: SessionState) -> dict[str, object] | None:
            record_combined_refused(
                into,
                spec_id=realignment.spec_id,
                option_ids=realignment.option_ids,
                refusal=refusal,
            )
            return answer(into)

        return holding_refused
    part, spec_id, resolved_at = compiled.part, compiled.spec_id, compiled.resolved_at
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

    It refuses this proposal alone and records nothing. The pin cannot be
    aligned, and another series of the input, or a plan that leaves it out of
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
    """The warning codes on the record of `origin`, a pin a compile is made from.

    They are read through `translator.discovery.pin_warnings`, and the compile
    keeps the codes.
    """
    return pin_warnings(
        origin,
        entry=None,
        shown=pin_sources(state),
        compiled=aligned_pins(state),
        recorded=compile_warnings(state),
    )


async def compile_whole(
    surface: ToolSurface,
    realignment: Realignment,
    *,
    state: SessionState,
) -> SourceRead:
    """Ask Search for the declared whole `realignment` names, and hold it as a pin.

    The part is the pin `realignment.spec_id`, read off the blocks on record
    (`planner.plan.held_sources`), so a part alignment moved reads its compile,
    and the axis is `realignment.whole_axis` (`planner.plan.WholeOf.axis`).
    Posts the part's base identity and scope entries as its block records them
    (`PartScope.base`, `PartScope.entries`), else its definition with none, its
    entity groups, and `whole: {axis}`. Search compiles the part, then its whole
    with every other choice held, and answers the whole, whose scope states the
    part beside it (`PartScope.derived`). No identity is edited here, and no model
    call is made. The block marks as the planner's
    (`harness.scope_choices.mark_planner_axes`) each axis the part's block marks
    that the whole holds, other than the one widened, so the whole reads as
    settled the way the part was. It is held as a `planner.plan.AlignedPin` from
    the part on that axis, which every whole declaring the part and the axis
    pins. Like a compile from `compile_on_basis`, it is held there alone.

    `state` is read. The writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held,
    `SURFACE_UNAVAILABLE` when Search did not answer and can later, or
    `SHARE_WHOLE_UNDERIVABLE` (`submission.service.share_whole_underivable`)
    under the cause Search states for refusing the whole, none where an older
    Search states none, with the options it offers in its place; under
    `part_unread` where no block of the part's pin is on record; under
    `groups_unreadable` where the part names a set whose posted form this
    release cannot read; under `whole_refused` where Search answers with no one
    dataset or with a block stating no derived whole; and under `part_moved`
    where the part Search derived the whole from is not the part's compile, or
    the whole widens another axis (`_whole_fault`).
    """
    part_pin, axis = realignment.spec_id, cast(str, realignment.whole_axis)

    def underivable(
        reason: str,
        cause: str | None = WHOLE_REFUSED,
        alternatives: Sequence[tuple[str, str]] = (),
    ) -> SourceRead:
        return _refusing(
            share_whole_underivable(
                whole_id=realignment.requirement_id,
                part_spec_id=part_pin,
                reason=reason,
                cause=cause,
                alternatives=alternatives,
            )
        )

    source = held_sources(state).get(part_pin)
    block = source.spec_block(part_pin) if source is not None else None
    scope = part_scope(part_pin, block)
    definition = (block or {}).get("definition")
    identity = (
        scope.base.identity
        if scope is not None
        else definition.get("identity")
        if isinstance(definition, Mapping)
        else None
    )
    if block is None or not isinstance(identity, Mapping):
        return underivable(f"no definition of {part_pin} from Search is held", PART_UNREAD)
    # The set on the side the whole widens is left out: Search drops that side's
    # parties for the whole, and compiles the part again from its identity.
    groups = readable_wire_groups(block, leaving_out=WHOLE_COLUMNS.get(axis))
    if groups is None:
        return underivable(
            f"Search's named sets for {part_pin} are not readable", GROUPS_UNREADABLE
        )
    try:
        part, resolved_at = await resolve_definition_at(
            surface,
            spec_id=scope.base.spec_id if scope is not None else part_pin,
            identity=identity,
            groups=groups,
            scope_entries=[
                entry.model_dump(mode="json", exclude_none=True)
                for entry in (scope.entries if scope is not None else ())
            ],
            whole={"axis": axis},
        )
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_basis_uncompiled(realignment, str(exc)))
        # An older Search refuses a whole stating no cause: the refusal says why in
        # its words alone (data-mcp's `fix.detail`), and its code is no cause.
        stated = (
            f" ({exc.cause})"
            if exc.cause is not None
            else f": {exc.detail.rstrip('.')}"
            if exc.detail is not None
            else ""
        )
        return underivable(
            f"Search does not derive the whole of {part_pin} over every {axis}{stated}",
            exc.cause,
            exc.alternatives,
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    whole_scope = part_scope(spec_id, spec) if isinstance(spec_id, str) else None
    digest = resolver_digest({"parts": [dict(part)]})
    answered = digest.get("spec_block")
    fault = _whole_fault(
        whole_scope if isinstance(answered, Mapping) else None,
        part_pin=part_pin,
        compiled=scope.compiled.spec_id if scope is not None else part_pin,
        axis=axis,
        open_axes=open_entity_axes_of(_resolved_of(answered)),
    )
    if fault is not None:
        # A whole is held only stating its derived whole, which `aligned_pins`
        # reads as current.
        return underivable(*fault)
    whole_scope, whole_id = cast(PartScope, whole_scope), cast(str, spec_id)
    # An axis reads as the planner's on the whole only where the whole holds the
    # option the part does: one Search moved, as a widened side's region, is
    # Search's default.
    part_active = {choice.axis: choice.active for choice in scope.choices} if scope else {}
    kept = {
        choice.axis
        for choice in whole_scope.choices
        if choice.axis != axis and part_active.get(choice.axis) == choice.active
    }
    mark_planner_axes(digest, planner_axes(block) & kept)

    def hold(into: SessionState) -> None:
        save_aligned_pin(
            into,
            AlignedPin(
                origin=part_pin,
                option_ids=(),
                spec_id=whole_id,
                digest=digest,
                whole_axis=axis,
            ),
            warnings=_origin_warnings(into, part_pin),
        )
        record_spec_searched(into, whole_id, resolved_at)

    return hold


def _whole_fault(
    whole: PartScope | None,
    *,
    part_pin: str,
    compiled: str,
    axis: str,
    open_axes: Sequence[OpenEntityAxis],
) -> tuple[str, str] | None:
    """Why `whole`, the scope record of Search's answer for the whole of the part
    pinned `part_pin` over every `axis`, is no whole of that part, as a clause and
    its `share_whole_underivable` cause; `None` where it is one. `compiled` is the
    spec_id the part's own record compiles to, and `open_axes` the entity axes the
    answer's block leaves open (`compiler.sources.open_entity_axes_of`).

    An answer with no block or no derived whole is `whole_refused`, and one
    deriving the whole of another part or on another axis, as a compile of the
    posted part landing elsewhere after a Search release, is `part_moved`. A
    donor or recipient whole Search states it summed over its members
    (`SUM_OF_MEMBERS`) leaves that side open on its block, and one it derived as
    a total binds it: the plan, the compiler and the whole's coverage read the
    side off the block's axes (`compiler.sources.open_side_column`), which every
    answer for the whole's spec_id carries, so an answer whose two statements
    disagree is `whole_refused`."""
    if whole is None or whole.derived is None:
        return f"Search answers no one derived whole of {part_pin}", WHOLE_REFUSED
    derived = whole.derived
    if derived.part.spec_id != compiled or derived.axis != axis:
        return (
            f"Search derived the whole of {derived.part.spec_id} over every "
            f"{derived.axis}, not of {part_pin} over every {axis}",
            PART_MOVED,
        )
    if axis in WHOLE_COLUMNS:
        summed = derived.kind == SUM_OF_MEMBERS
        if summed != (open_side_column(open_axes, axis) is not None):
            held = "binds" if summed else "leaves open"
            return (
                f"Search derived the whole of {part_pin} as "
                f"{'a sum over' if summed else 'one total for'} every {axis}, and its "
                f"answer {held} that side",
                WHOLE_REFUSED,
            )
    return None


def _resolved_of(block: object) -> Mapping[str, object]:
    """The `resolved` of `block`, a spec block, empty where it states none."""
    resolved = block.get("resolved") if isinstance(block, Mapping) else None
    return cast(Mapping[str, object], resolved) if isinstance(resolved, Mapping) else {}


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
    pins it and the planner adapter stores it as a reading of the conversation
    (`harness.planner_adapter.store_searched_pin_readings`). Any other answer is
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
    (`translator.discovery.refusal_retryable`). Nothing is held then, so the
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
    """The `SURFACE_UNAVAILABLE` problem for a search of `unpinned` that can succeed later.

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
    """The `SURFACE_UNAVAILABLE` problem for a read of a candidate's scope that did
    not complete.

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
    """The `SURFACE_UNAVAILABLE` problem for a compile on another basis or provider
    scope that did not complete.

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


async def with_source_reads(
    propose: Callable[[], dict[str, object]], *, state: SessionState, surface: ToolSurface
) -> dict[str, object]:
    """`propose`'s answer, once every read of a source it was refused for is made
    (`_after_source_reads`).

    A declared whole Search does not derive (`SHARE_WHOLE_UNDERIVABLE`) permits an
    `unsupported_ask` decline for the rest of the planner entry, the repair its
    message names beside a total pinned by search, so the decline is earned. A
    refusal the harness states (`HARNESS_WHOLE_CAUSES`) earns none.
    """
    result = await _after_source_reads(propose, state=state, surface=surface)
    if (
        result.get("code") == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value
        and _whole_cause(result) not in HARNESS_WHOLE_CAUSES
    ):
        record_optional_decline(state, UnsupportedOperation.UNSUPPORTED_ASK)
    return result


def _whole_cause(result: Mapping[str, object]) -> object:
    """The cause a `SHARE_WHOLE_UNDERIVABLE` answer states (`detail.whole.cause`)."""
    detail = result.get("detail")
    whole = detail.get("whole") if isinstance(detail, Mapping) else None
    return whole.get("cause") if isinstance(whole, Mapping) else None


async def _after_source_reads(  # noqa: PLR0911 -- one return per kind of read a refusal asks for
    propose: Callable[[], dict[str, object]], *, state: SessionState, surface: ToolSurface
) -> dict[str, object]:
    """`propose`'s answer, once every read of a source it was refused for is made.

    A refusal records nothing, so `propose` runs again once the read is held.
    Two refusals ask for one. A plan rolling from a dataset's first year
    needs that first year (`read_full_history`, held by
    `planner.plan.save_full_history`). One read serves every requirement, and
    every later proposal of the turn, that pins the dataset. An answer
    stating no first year is held too, and the next refusal says so
    (`first_year_read_failed`). A calculation one of whose inputs pins no
    dataset while another carries a basis choice needs that input searched
    for and pinned (`BASIS_UNKNOWN`, `pin_by_search`), as does the part of
    a declared whole. A search held for the requirement's content is made
    once. A calculation whose inputs are counted on different measure bases
    or provider scopes needs each input it names compiled on their shared
    options, every option of one input in one compile (`BASES_UNALIGNED`,
    `compile_on_basis`), and a declared whole needs Search's whole of its part
    on the declared axis (`compile_whole`).
    An input pinned to a candidate Search offered with no scope choices needs
    them read first (`SCOPE_UNREAD`, `read_scope`), once a pin, however many
    inputs pin it, and once a call. A refusal naming a pin already read
    answers `OPERATION_UNSUPPORTED`, since Search's answer left it unread.
    A pin already compiled on a set of options, or a whole already compiled,
    is asked for once, so a refusal naming one is answered as it stands, and
    inputs sharing a pin and a set of options compile it once
    (`_alignment_key`). A whole pin an earlier release compiled, whose block
    states no derived whole, is not on record (`planner.plan.aligned_pins`), so
    it is compiled again, once a call: a refusal naming a whole this call
    compiled answers `SHARE_WHOLE_UNDERIVABLE`.

    The reads one refusal asks for run at once. Their writes are made in the
    order the refusal names them, up to the first read refused. That refusal
    is answered as soon as it and every read before it are made, and the
    reads after it are cancelled (`_held_in_order`), so the state is the one
    reading them one after another leaves. A search the refused read's writes
    discarded keeps its answer for the invocation (`pin_by_search`), so the
    next proposal searches only for what was refused.
    """
    scopes_read: set[str] = set()
    compiled: set[tuple[str, tuple[str, ...], str | None]] = set()
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
            # A pin is read once a call. Search answered a read that left it
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
        again = next(
            (item for key, item in zip(keys, wanted, strict=True) if key in compiled), None
        )
        if again is not None:
            # Held and refused again for want of what it holds: Search's answer
            # will not change within the call.
            if again.whole_axis is None:
                return result
            return share_whole_underivable(
                whole_id=again.requirement_id,
                part_spec_id=again.spec_id,
                reason=(
                    f"Search's answer for the whole of {again.spec_id} does not say how it "
                    f"serves every {again.whole_axis}"
                ),
                cause=WHOLE_REFUSED,
            )
        compiles: dict[tuple[str, tuple[str, ...], str | None], Realignment] = {}
        for key, item in zip(keys, wanted, strict=True):
            compiles.setdefault(key, item)
        refused = await _held_in_order(
            (compile_on_basis(surface, item, state=state) for item in compiles.values()),
            state=state,
        )
        compiled.update(compiles)
        if refused is not None:
            return refused


async def _held_in_order(
    reads: Iterable[Awaitable[SourceRead]], *, state: SessionState
) -> dict[str, object] | None:
    """Run `reads` at once, apply each to `state` in order, and answer the first refused.

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
    """The key of the compile that puts pin `spec_id` on `option_ids`.

    A request for the pin's whole on `whole_axis` is keyed by that axis.

    The options are sorted and deduplicated, so one compile naming the same
    options in another order is the same work.
    """
    return spec_id, tuple(sorted(set(option_ids))), whole_axis
