"""Run the translator's discover-then-compile loop as code, behind the `translate` tool.

Finding each dataset requirement's source, reading the `action` a
non-`definite` answer carries and following it, building a `Translation`
that repeats each requirement's own calculation fields, and following
`propose_operations`'s own `action` when a proposal does not compile are
each a function of data the process already holds -- the plan's own
requirements, the discovered sources, `_compile_failure_action`'s
issue-to-action table. `run_translate` drives the translator's core
functions directly, in that order, spending no model turn on any of it.

`build_translation` is the piece that reads as if it needed judgement -- "a
`retrieve` naming each dataset requirement's discovered spec_id, and for
each derived requirement the operation of the same kind, naming that
requirement in `requirement_id` and repeating its calculation fields
exactly" -- and is not: `planner.proof.mismatches` already proves an
operation and its requirement agree field for field, so building the
operation *from* the requirement is the inverse of that same
correspondence, not a fresh choice. Setting each operation's own `id` to
the requirement id it carries out is what makes every `input`/`left`/
`right`/`numerator.input`/`denominator.input` resolve without a lookup:
those fields already name the *requirement* they read
(`AggregateRequirement.input_id`, `RequirementSide.input_id`,
`JoinRequirement.left_id`/`right_id`), which equals the upstream
operation's id under this scheme.

Several candidate sources for one requirement close the call on a
`source_ambiguous` referral carrying the candidates as the discovery log
recorded them. The planner records the choice as a decision or puts it to
the user, and `discover_source`'s own answer forbids this loop from picking
one.

Every stop is a typed referral, whose `outcome` is one of the closed
set `translator.referral.OUTCOME_OF` names: a requirement of a kind no
operation builder covers is unsupported, every compile issue resolves
through `translator.tool.issue_resolution`, a service that did not answer
is transient, and a shape this loop was not taught is an internal failure.
A translation that fails with the same issues it failed with before in the
session (`_failure_fingerprint`) is referred again marked `repeated`, so the
planner learns the plan did not change where it failed.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import rfc8785
from google.adk.tools.tool_context import ToolContext
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GrowthOperation,
    JoinOperation,
    Operation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    RankRequirement,
    ShareOfRequirement,
    SumOfRequirement,
    inputs_of,
)
from planner.tool_results import Action, Outcome, ProblemCode

from translator.discovery import (
    current_draft,
    dataset_keys,
    load_log,
    settled_sources,
)
from translator.referral import Referral, ReferralKind, clear_referral, load_referral

# How many propose-then-fix rounds this loop tries before it gives up and
# escalates. `translator.discovery.DISCOVERY_CALL_BUDGET` bounds how many
# *discovery* calls one plan revision may spend; this bounds this loop's own
# rounds of `propose_operations`, a different and much smaller resource,
# since a determinate fix from `_compile_failure_action` is applied at most
# once per issue class before either compiling or hitting an issue the table
# does not cover.
MAX_ROUNDS = 4

# `translate`'s own core: the three functions this loop drives directly,
# bypassing ADK's model-turn dispatch. `discover_source` and
# `propose_operations` are async (they call the tool surface);
# `return_to_planner` is not.
DiscoverSource = Callable[..., Awaitable[dict[str, object]]]
ProposeOperations = Callable[..., Awaitable[dict[str, object]]]
ReturnToPlanner = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class TranslatorCore:
    """The translator's three core functions.

    Built once by `translator.tool` and closed over by the `translate`
    tool.
    """

    discover_source: DiscoverSource
    propose_operations: ProposeOperations
    return_to_planner: ReturnToPlanner


# The session-state slot holding `_failure_fingerprint` of every translation
# that failed to compile with no fix this loop applies itself, in the
# planner's current entry. `harness.planner_seam` clears it when an entry
# starts (`clear_compile_failures`), so an ask-the-user resume keeps it.
COMPILE_FAILURES_STATE_KEY = "translator:compile_failures"


# --- build_translation: a pure function of the plan, never a fresh choice --


def build_translation(plan: Plan, discovered: Mapping[str, str]) -> Translation:
    """Build the `Translation` every dataset requirement's discovered source owes the plan.

    Every operation's calculation fields are copied verbatim from the
    requirement it carries out, and its `id` is set to that requirement's
    own `id` -- so any other requirement's `input_id`/`left_id`/`right_id`
    (or a `RequirementSide.input_id`) already names the right operation,
    with no id remapping needed. The terminal operations are the
    requirements `inputs_of` names as nobody's input.

    Args:
        plan: The plan whose requirements to build operations for. Every
            requirement's own type must be one `_BUILDERS` covers --
            `run_translate` checks this before calling here, so a plan
            reaching this function never names an unsupported kind.
        discovered: `requirement_id` -> `spec_id`, for every dataset
            requirement discovery has resolved definitely (or whose pin it
            honoured). Call this only once every dataset requirement is in
            `discovered`; a requirement missing from it raises `KeyError`.

    Returns:
        The mechanically-built translation.
    """
    operations: list[Operation] = [
        _operation_for(requirement, discovered) for requirement in plan.evidence_requirements
    ]
    read_ids = {
        input_id
        for requirement in plan.evidence_requirements
        for input_id in inputs_of(requirement)
    }
    terminal = tuple(
        requirement.id
        for requirement in plan.evidence_requirements
        if requirement.id not in read_ids
    )
    return Translation(operations=tuple(operations), terminal=terminal)


def _side(input_id: str, entity: str | None) -> Side:
    return Side(input=input_id, entity=entity)


def _retrieve(requirement: DatasetRequirement, discovered: Mapping[str, str]) -> Operation:
    return RetrieveOperation(
        id=requirement.id, requirement_id=requirement.id, spec_id=discovered[requirement.id]
    )


def _aggregate(requirement: AggregateRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return AggregateOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        input=requirement.input_id,
        reduction=requirement.reduction,
        group_by=requirement.group_by,
    )


def _growth(requirement: GrowthRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return GrowthOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        input=requirement.input_id,
        window_start=requirement.window_start,
        window_end=requirement.window_end,
        change=requirement.change,
    )


def _compare(requirement: ComparisonRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return CompareOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        left=_side(requirement.left.input_id, requirement.left.entity),
        right=_side(requirement.right.input_id, requirement.right.entity),
    )


def _sum_of(requirement: SumOfRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return SumOfOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        left=_side(requirement.left.input_id, requirement.left.entity),
        right=_side(requirement.right.input_id, requirement.right.entity),
    )


def _rank(requirement: RankRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return RankOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        input=requirement.input_id,
        direction=requirement.direction,
        among=requirement.among,
        axis=requirement.axis,
        reduction=requirement.reduction,
        window_start=requirement.window_start,
        window_end=requirement.window_end,
        per_period=requirement.per_period,
        limit=requirement.limit,
    )


def _share_of(requirement: ShareOfRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return ShareOfOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        numerator=_side(requirement.numerator.input_id, requirement.numerator.entity),
        denominator=_side(requirement.denominator.input_id, requirement.denominator.entity),
        axis=requirement.axis,
        denominator_scope=requirement.denominator_scope,
    )


def _join(requirement: JoinRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return JoinOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        left=requirement.left_id,
        right=requirement.right_id,
    )


def _filter(requirement: FilterRequirement, discovered: Mapping[str, str]) -> Operation:
    del discovered
    return FilterOperation(
        id=requirement.id,
        requirement_id=requirement.id,
        input=requirement.input_id,
        predicates=requirement.predicates,
    )


# One builder per requirement type, each copying that type's own calculation
# fields verbatim into the operation kind `planner.proof.KIND_OF_REQUIREMENT`
# says carries it out. Keyed by type rather than `kind` because a dataset
# requirement's `kind` is a plain `Literal`, not a member of the
# `EvidenceRequirement` discriminated union's own tag space this dict would
# otherwise have to duplicate.
_BUILDERS: dict[type, Callable[[Any, Mapping[str, str]], Operation]] = {
    DatasetRequirement: _retrieve,
    AggregateRequirement: _aggregate,
    GrowthRequirement: _growth,
    ComparisonRequirement: _compare,
    SumOfRequirement: _sum_of,
    RankRequirement: _rank,
    ShareOfRequirement: _share_of,
    JoinRequirement: _join,
    FilterRequirement: _filter,
}


def _operation_for(requirement: EvidenceRequirement, discovered: Mapping[str, str]) -> Operation:
    """The one operation `requirement`'s own fields determine, verbatim.

    `run_translate`'s own pre-flight check (`_unsupported_requirement`)
    refuses a plan naming a requirement type outside `_BUILDERS` before this
    is ever called, so the lookup here is never missing a builder in
    practice.
    """
    builder = _BUILDERS[type(requirement)]
    return builder(requirement, discovered)


def _unsupported_requirement(plan: Plan) -> EvidenceRequirement | None:
    """The first requirement of a plan whose type no builder covers, if any."""
    for requirement in plan.evidence_requirements:
        if type(requirement) not in _BUILDERS:
            return requirement
    return None


# --- discovery: every unsettled requirement at once, stopping at the first referral --


def _action_of(payload: Mapping[str, object]) -> Action | None:
    raw = payload.get("action")
    return Action.model_validate(raw) if raw else None


# `discover_source`'s own problem codes for a data-service failure rather
# than a data answer, answered with no `action`: `_settle_discovery` turns
# either into a transient `ReferralKind.DISCOVERY_UNAVAILABLE` referral
# rather than the internal failure a shape it cannot resolve becomes. A
# `SEARCH_REFUSED` that carries an `action`, Search's final answer for a seed's copy over a
# period (`translator.discovery._seeded_source`), follows its action.
_NAMED_FAILURE_CODES = frozenset(
    {ProblemCode.SURFACE_UNAVAILABLE.value, ProblemCode.SEARCH_REFUSED.value}
)


def _spec_id_of(answer: Mapping[str, object]) -> str | None:
    """The one source's `spec_id` a `definite` (or honoured-pin) answer names, if any."""
    if answer.get("outcome") != "ok":
        return None
    sources = answer.get("sources")
    if not isinstance(sources, list):
        return None
    spec_ids = [
        spec_id
        for s in sources
        if isinstance(s, dict) and isinstance(spec_id := s.get("spec_id"), str)
    ]
    # A `definite` answer, or a pin honoured out of a wider one, names
    # exactly one selectable source; anything else with no action attached
    # is not a shape this loop expects, and escalates rather than guessing.
    return spec_ids[0] if len(spec_ids) == 1 else None


@dataclass
class _Stopped:
    """A `_settle_discovery` call's own reason to stop the whole loop here."""

    result: dict[str, object]


def _internal_failure(
    core: TranslatorCore, tool_context: ToolContext, reason: str
) -> dict[str, object]:
    """Refer a step this loop was not taught to resolve as an internal failure."""
    return core.return_to_planner(
        tool_context=tool_context, kind=ReferralKind.INTERNAL_FAILURE, reason=reason
    )


async def _discover_answer(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Call `discover_source` once, settling nothing.

    Safe to run alongside its siblings for other requirements:
    `discovery_lock` guards only the budget check inside `discover_source`,
    and the log records each requirement under its own key.
    `_resolve_dataset_sources` gathers several of these calls concurrently,
    then settles each answer in requirement order, one at a time
    (`_settle_discovery`) -- settling is the part that is not safe to run
    for two requirements at once.
    """
    return await core.discover_source(**arguments, tool_context=tool_context)


def _settle_discovery(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    requirement_id: str,
    answer: dict[str, object],
) -> str | _Stopped:
    """Settle one `discover_source` answer, following its action or naming a stop.

    Returns the discovered `spec_id` to continue with, or a `_Stopped`
    naming the result the whole `translate` call ends on: the referral a
    complete `return_to_planner` action names, a `source_ambiguous` one
    included; a `discovery_unavailable` referral,
    raised here rather than by an `action`, when a data-service failure is
    what stopped it (`_NAMED_FAILURE_CODES`); or an internal failure for
    anything else the answer does not resolve to a single source with no
    action attached.

    Writes `translate`'s single-slot referral state when it stops, so a
    caller settling more than one requirement's answer must do it one call at a time, checking each
    result for a `_Stopped` before settling the next: two settlements in
    flight together would let the later one's write clobber the earlier
    one's, even though both `discover_source` calls that produced their
    answers were free to run concurrently.
    """
    action = _action_of(answer)
    if action is not None:
        if action.call == "return_to_planner" and not action.needs:
            referral = core.return_to_planner(tool_context=tool_context, **(action.arguments or {}))
            return _Stopped(referral)
        # An incomplete action (`needs` set), or one naming a call this loop
        # does not know how to drive itself: `discover_source` names no such
        # shape, and a future one is a fault until this loop is taught it.
        return _Stopped(
            _internal_failure(
                core,
                tool_context,
                f"discover_source named an action this loop cannot complete for "
                f"{requirement_id!r}: {action.model_dump_json()}",
            )
        )
    spec_id = _spec_id_of(answer)
    if spec_id is not None:
        return spec_id
    if answer.get("code") in _NAMED_FAILURE_CODES:
        # A service the search timed out on, or that refused the call
        # outright -- a fact about the data service right now, not about
        # the requirement's own content, so `requirement_unmeetable` would
        # mislabel it and send the planner off rewriting a requirement that
        # was never the problem.
        reason = str(answer.get("message") or "discover_source could not reach the data service")
        return _Stopped(
            core.return_to_planner(
                tool_context=tool_context,
                kind=ReferralKind.DISCOVERY_UNAVAILABLE,
                reason=reason,
                requirement_id=requirement_id,
            )
        )
    return _Stopped(
        _internal_failure(
            core,
            tool_context,
            f"discover_source answered {requirement_id!r} with no action and no single "
            f"source to use: {answer.get('code') or answer.get('outcome')}",
        )
    )


async def _discover_one(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    arguments: dict[str, object],
) -> str | _Stopped:
    """Call `discover_source` once and settle its answer immediately.

    For `_propose_until_settled`'s discover-again retry, where one
    requirement's discovery is in flight at a time, so calling and settling
    together is safe. `_resolve_dataset_sources` keeps the two apart for its
    own concurrent calls.
    """
    requirement_id = cast(str, arguments["requirement_id"])
    answer = await _discover_answer(core, tool_context, arguments=arguments)
    return _settle_discovery(core, tool_context, requirement_id=requirement_id, answer=answer)


async def _resolve_dataset_sources(
    core: TranslatorCore, tool_context: ToolContext, *, plan: Plan
) -> dict[str, str] | _Stopped:
    """Discover every dataset requirement's source, or the result the loop stops with.

    A requirement whose own searched content (`requirement_revision`) and pin
    already have a settled `DiscoveryLog` record -- from an earlier
    `translate()` call, an earlier round of this loop, or a plan amendment
    that left this requirement untouched -- takes that record's source
    directly, the same way `propose_operations` does
    (`requirement_keys`/`pinned`/`log.selectable`), rather than spending a
    fresh `discover_source` call to re-learn what discovery already settled.
    A requirement stating `across` whose source was recorded without its
    open entity axes is not settled, and is discovered again. Neither is one
    whose recorded source is a dataset the turn has since dropped
    (`translator.discovery.settled_sources`): its fresh search is refused as
    `seed_dropped`, which refers the requirement back to the planner.

    Every requirement still unsettled starts its `discover_source` call
    together, gathered with `return_exceptions=True`, since the call itself
    is already safe to run alongside its siblings -- `discovery_lock` guards
    only the budget check, and the log records each requirement under its
    own key. Settling what each call answered (`_settle_discovery`) is not:
    it is what writes the referral to `translate`'s
    single-slot state, so the answers are walked back in requirement order
    and settled one at a time -- the first genuine exception is re-raised,
    the first settlement that yields a `_Stopped` is returned at once, and
    no later answer is settled. Every sibling's `discover_source` call has
    already run and written its own record and sources to the log by the
    time this returns, so a re-entry after the returned referral still finds
    the untouched requirements `selectable` and spends no fresh call on them.
    """
    dataset_requirements = [
        requirement
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    ]
    keys = dataset_keys(plan)
    log = load_log(tool_context.state)
    settled = settled_sources(
        tool_context.state,
        log,
        requirement_keys=keys.requirement_keys,
        pinned=keys.pinned,
        across=keys.across,
    )

    discovered: dict[str, str] = {}
    unsettled: list[DatasetRequirement] = []
    for requirement in dataset_requirements:
        selected = settled.get(requirement.id)
        if selected is not None:
            discovered[requirement.id] = selected.spec_id
        else:
            unsettled.append(requirement)

    answers = await asyncio.gather(
        *(
            _discover_answer(core, tool_context, arguments={"requirement_id": r.id})
            for r in unsettled
        ),
        return_exceptions=True,
    )
    for requirement, answer in zip(unsettled, answers, strict=True):
        if isinstance(answer, BaseException):
            raise answer
        step = _settle_discovery(core, tool_context, requirement_id=requirement.id, answer=answer)
        if isinstance(step, _Stopped):
            return step
        discovered[requirement.id] = step
    return discovered


# --- propose-then-fix, following `_compile_failure_action` up to MAX_ROUNDS --


def _failure_fingerprint(translation: Translation, result: Mapping[str, object]) -> str:
    """The canonical hash of `translation` plus the sorted codes it failed to compile with."""
    detail = result.get("detail")
    issues = detail.get("issues") if isinstance(detail, dict) else None
    codes = sorted(
        str(issue.get("code"))
        for issue in (issues if isinstance(issues, list) else ())
        if isinstance(issue, dict)
    )
    digest = hashlib.sha256(rfc8785.dumps(translation.model_dump(mode="json"))).hexdigest()
    return f"{digest}:{','.join(codes)}"


def clear_compile_failures(state: Any) -> None:
    """Forget the compile-failure fingerprints, as a new planner entry starts."""
    state[COMPILE_FAILURES_STATE_KEY] = []


def _seen_before(state: Any, fingerprint: str) -> bool:
    """Whether `fingerprint` failed before in this planner entry, recording it if not."""
    seen = list(state.get(COMPILE_FAILURES_STATE_KEY) or ())
    if fingerprint in seen:
        return True
    state[COMPILE_FAILURES_STATE_KEY] = [*seen, fingerprint]
    return False


def _uncompiled(
    core: TranslatorCore, tool_context: ToolContext, result: Mapping[str, object]
) -> dict[str, object]:
    """Refer a `propose_operations` answer that is not a compile failure.

    A tool surface that could not be listed is transient. Anything else --
    a translation `build_translation` built in a shape `propose_operations`
    refuses, or a plan that stopped being a translatable draft mid-run -- is
    a fault in this loop.
    """
    message = str(result.get("message") or "")
    if result.get("code") == ProblemCode.SURFACE_UNAVAILABLE.value:
        return core.return_to_planner(
            tool_context=tool_context,
            kind=ReferralKind.SURFACE_UNAVAILABLE,
            reason=message or "the tool surface could not be listed",
        )
    return _internal_failure(
        core, tool_context, f"propose_operations answered {result.get('code')}: {message}"
    )


async def _propose_until_settled(
    core: TranslatorCore, tool_context: ToolContext, *, plan: Plan, discovered: dict[str, str]
) -> dict[str, object]:
    """Build and propose a translation, following `_compile_failure_action` up to `MAX_ROUNDS`.

    A source to discover again is the one fix this loop applies itself. Any
    other compile failure closes on the referral its action names; one whose
    `_failure_fingerprint` failed before in this session closes marked
    `repeated`. Only a compile failure counts towards the guard: a transient
    failure repeats legitimately.
    """
    for _round in range(MAX_ROUNDS):
        translation = build_translation(plan, discovered)
        result = await core.propose_operations(translation=translation, tool_context=tool_context)
        if result.get("outcome") == "ok":
            return result

        action = _action_of(result)
        if action is None or result.get("code") != ProblemCode.COMPILE_FAILED.value:
            return _uncompiled(core, tool_context, result)
        if action.call == "return_to_planner":
            repeated = _seen_before(tool_context.state, _failure_fingerprint(translation, result))
            return core.return_to_planner(
                tool_context=tool_context, **(action.arguments or {}), repeated=repeated
            )
        if action.call != "discover_source":
            return _internal_failure(
                core, tool_context, f"propose_operations named an unknown next call {action.call!r}"
            )

        step = await _discover_one(core, tool_context, arguments=dict(action.arguments or {}))
        if isinstance(step, _Stopped):
            return step.result
        requirement_id = cast(str, (action.arguments or {})["requirement_id"])
        discovered[requirement_id] = step

    return _internal_failure(
        core, tool_context, f"propose_operations did not settle within {MAX_ROUNDS} rounds"
    )


def _referred_again(
    core: TranslatorCore, tool_context: ToolContext, referral: Referral
) -> dict[str, object]:
    """Answer with the referral already pending on the plan, as `return_to_planner` built it.

    The plan is unchanged since it was referred (`propose_plan` clears the
    referral), so translating it again could only reach the same stop. A
    transient referral is the exception, and `run_translate` retries it.
    """
    return core.return_to_planner(
        tool_context=tool_context,
        **referral.model_dump(
            exclude={"candidates", "no_data_reason", "stale_pins"}, exclude_defaults=True
        ),
    )


# --- the public entry point --------------------------------------------------


async def run_translate(tool_context: ToolContext, *, core: TranslatorCore) -> dict[str, object]:
    """Run the translator's whole discover-then-compile loop without a model turn.

    Answers with the referral already pending on the plan, unless it is
    transient. Otherwise drives `discover_source` for every dataset
    requirement, then `propose_operations`, following each step's own
    `action`.

    Returns:
        `{"outcome": "ok", ...}` once `propose_operations` succeeds;
        `{"outcome": "referred", "referral": {...}}` the moment any step
        stops short of compiling, the referral's own `outcome` one of
        `translator.referral.OUTCOME_OF`'s; or the typed problem that stops
        a call on a plan that is not a translatable draft.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan

    pending = load_referral(tool_context.state)
    if pending is not None:
        if pending.outcome is not Outcome.TRANSIENT:
            return _referred_again(core, tool_context, pending)
        clear_referral(tool_context.state)

    unsupported = _unsupported_requirement(plan)
    if unsupported is not None:
        return core.return_to_planner(
            tool_context=tool_context,
            kind=ReferralKind.NO_OPERATION_COVERS,
            requirement_id=unsupported.id,
            reason=(
                f"no operation this translator builds carries out a {unsupported.kind!r} "
                "requirement."
            ),
        )

    resolved = await _resolve_dataset_sources(core, tool_context, plan=plan)
    if isinstance(resolved, _Stopped):
        return resolved.result

    return await _propose_until_settled(core, tool_context, plan=plan, discovered=resolved)
