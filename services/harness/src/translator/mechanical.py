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

One judgement survives, because nothing in the plan or the discovery log
settles it: which of several candidate sources best answers a requirement.
`run_translate` stops there and answers `needs_decision`, carrying the
candidates, their own descriptions, and the decisions citing the
requirement, so the translator's model can rank them. Its answer -- a
`choice` and a `rationale` -- rides into the `source_ambiguous` referral as
a recommendation for the person who settles the choice to read, not a
pick the translator makes on its own: `discover_source`'s own answer
already forbids that, and nothing here overrides it.

A requirement of a kind no operation builder covers, or a compile issue
`_compile_failure_action` cannot resolve into a complete `return_to_planner`
call, refers back or escalates rather than raising: a plan shape this loop
has not been taught refuses in a way the planner can act on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from compiler.surface import ToolSurface
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
    inputs_of,
    plan_revision,
)
from planner.records import Identifier, SpecId, StrictModel
from planner.tool_results import Action, Outcome, ProblemCode, problem
from planner.workflow import SelectedSource
from pydantic import JsonValue

from translator.discovery import (
    DescribedPair,
    apply_definitions,
    compact_source,
    current_draft,
    discovery_lock,
    fetch_definitions,
    load_log,
    requirement_revision,
    settled_sources,
)
from translator.referral import ReferralKind, SourceRecommendation

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
    """The translator's three tool functions, plus the surface they read.

    Built once by `translator.agent` and closed over by the `translate`
    tool; `surface` is carried alongside the functions because candidate
    enrichment (`_enrich_candidates`) calls `fetch_definitions` directly,
    the same describe fetch `propose_operations` runs for a genuinely
    `definite` source.
    """

    discover_source: DiscoverSource
    propose_operations: ProposeOperations
    return_to_planner: ReturnToPlanner
    surface: ToolSurface


# The session-state slot holding the judgement `translate` is waiting on an
# answer for, if any. One key, since a translator turn asks at most one
# question before it either settles or refers -- `_settle_discovery` and
# `_propose_until_settled` both stop the loop the moment they raise one, and
# `_resolve_dataset_sources` settles its own several concurrent answers one
# at a time precisely so that only the one it stops on ever writes here.
_PENDING_JUDGEMENT_STATE_KEY = "translator:pending_judgement"

Judgement = Literal["candidate_ranking", "unmet_capability"]


class _PendingJudgement(StrictModel):
    """What `translate` is waiting on an answer for, and enough to check one.

    `revision` is the plan's `plan_revision` at the moment this was asked:
    an amendment between this call and the answer changes the plan the
    judgement was about, so a stale answer is refused rather than applied
    to a plan it never saw. `requirement_id` and `candidate_spec_ids` are
    set only for `candidate_ranking`; `unmet_capability` needs neither, since
    `_compile_failure_action`'s own `no_operation_covers` action names no
    requirement, only the decision the model still has to pick.
    """

    judgement: Judgement
    revision: str
    reason: str
    requirement_id: Identifier | None = None
    candidate_spec_ids: tuple[SpecId, ...] = ()


def _save_pending(state: Any, pending: _PendingJudgement) -> None:
    state[_PENDING_JUDGEMENT_STATE_KEY] = pending.model_dump_json()


def _load_pending(state: Any) -> _PendingJudgement | None:
    raw = state.get(_PENDING_JUDGEMENT_STATE_KEY)
    return _PendingJudgement.model_validate_json(raw) if raw else None


def _clear_pending(state: Any) -> None:
    state[_PENDING_JUDGEMENT_STATE_KEY] = None


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


# --- discovery: every unsettled requirement at once, stopping at the first referral or judgement --


def _action_of(payload: Mapping[str, object]) -> Action | None:
    raw = payload.get("action")
    return Action.model_validate(raw) if raw else None


# `discover_source`'s own problem codes for a data-service failure rather
# than a data answer, answered with no `action`: without this list
# `_settle_discovery` would treat either as a shape it cannot resolve and
# hand back an unanswerable `needs_decision` -- see `_settle_discovery`'s
# final branch, which turns either into a
# `ReferralKind.DISCOVERY_UNAVAILABLE` referral instead. A `SEARCH_REFUSED`
# that carries an `action`, Search's final answer for a seed's copy over a
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


async def _enrich_candidates(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    revision: str,
    requirement_id: str,
    key: str,
    candidates: tuple[SelectedSource, ...],
) -> tuple[SelectedSource, ...]:
    """Fetch each candidate's own `describe` definition, memoized in the log.

    `fetch_definitions` enriches one source per requirement id per call, so
    several candidates for the same requirement are enriched one call each
    -- a plan's single-digit candidate count costs a matching handful of
    `describe` round trips, not a model turn either way. Every round trip
    runs with `discovery_lock` released: only the log read before them and
    the write after them need serialising, not the network wait between.
    """
    async with discovery_lock(tool_context.state):
        log = load_log(tool_context.state)

    pairs: list[DescribedPair] = []
    for candidate in candidates:
        pairs.extend(
            await fetch_definitions(
                core.surface,
                log,
                revision=revision,
                requirement_keys={requirement_id: key},
                sources={requirement_id: candidate},
            )
        )

    log = await apply_definitions(tool_context.state, pairs)
    enriched = log.sources.get(requirement_id, {})
    return tuple(enriched.get(c.spec_id, c) for c in candidates)


async def _begin_candidate_ranking(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement_id: str,
    reason: str,
) -> dict[str, object]:
    """Fetch what ranking needs, persist the pending judgement, and answer `needs_decision`.

    The translator keeps exactly one judgement: which candidate best
    answers `requirement_id`, given the requirement itself and the
    decisions citing it. Neither is decided here -- picking a candidate is
    the one thing `discover_source`'s own answer already forbids this loop
    from doing on its own.
    """
    requirement = plan.requirement(requirement_id)
    assert isinstance(requirement, DatasetRequirement)  # only a dataset requirement is discovered
    key = requirement_revision(requirement)
    revision = plan_revision(plan)
    log = load_log(tool_context.state)
    candidates = log.candidates_for(key, requirement_id)
    candidates = await _enrich_candidates(
        core,
        tool_context,
        revision=revision,
        requirement_id=requirement_id,
        key=key,
        candidates=candidates,
    )
    _save_pending(
        tool_context.state,
        _PendingJudgement(
            judgement="candidate_ranking",
            revision=revision,
            reason=reason,
            requirement_id=requirement_id,
            candidate_spec_ids=tuple(c.spec_id for c in candidates),
        ),
    )
    decisions = [
        decision.model_dump(mode="json")
        for decision in plan.decisions
        if decision.slot in requirement.decision_ids
    ]
    return {
        "outcome": "needs_decision",
        "judgement": "candidate_ranking",
        "requirement_id": requirement_id,
        "reason": reason,
        "candidates": [compact_source(candidate) for candidate in candidates],
        "decisions": decisions,
    }


@dataclass
class _Stopped:
    """A `_settle_discovery` call's own reason to stop the whole loop here."""

    result: dict[str, object]


def _unmet_capability(reason: str, **detail: object) -> dict[str, object]:
    """The generic `needs_decision` shape for a step this loop cannot resolve on its own.

    Only ever a real, resumable judgement when the caller has also saved a
    `_PendingJudgement` for it (`_apply_compile_action`'s `action.needs`
    branch is the one case that does, mirroring `_compile_failure_action`'s
    own `NO_CAPABILITY` branch, which always needed the model's `decision_id`).
    Every other caller reports a shape this loop was not taught a fix for --
    an answer no `translate` argument is defined to carry -- so it saves
    nothing to resume: a caller that answers one anyway is refused as having
    no pending judgement, honestly, rather than accepting an answer that
    would apply to nothing.
    """
    return {"outcome": "needs_decision", "judgement": "unmet_capability", "reason": reason} | detail


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


async def _settle_discovery(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement_id: str,
    answer: dict[str, object],
) -> str | _Stopped:
    """Settle one `discover_source` answer, following its action or naming a stop.

    Returns the discovered `spec_id` to continue with, or a `_Stopped`
    naming the result the whole `translate` call ends on: a referral this
    loop raised itself for a determinate, non-ambiguous action; a
    `needs_decision` for a `source_ambiguous` action, which pauses for the
    one judgement the translator keeps; a `discovery_unavailable` referral,
    raised here rather than by an `action`, when a data-service failure is
    what stopped it (`_NAMED_FAILURE_CODES`); or a `needs_decision` for
    anything else the answer does not resolve to a single source with no
    action attached.

    Writes `translate`'s single-slot state -- the referral or the pending
    judgement -- when it stops, so a caller settling more than one
    requirement's answer must do it one call at a time, checking each
    result for a `_Stopped` before settling the next: two settlements in
    flight together would let the later one's write clobber the earlier
    one's, even though both `discover_source` calls that produced their
    answers were free to run concurrently.
    """
    action = _action_of(answer)
    if action is not None:
        reason = str((action.arguments or {}).get("reason", ""))
        if action.call == "return_to_planner" and not action.needs:
            kind = (action.arguments or {}).get("kind")
            if kind == ReferralKind.SOURCE_AMBIGUOUS.value:
                return _Stopped(
                    await _begin_candidate_ranking(
                        core, tool_context, plan=plan, requirement_id=requirement_id, reason=reason
                    )
                )
            referral = core.return_to_planner(tool_context=tool_context, **(action.arguments or {}))
            return _Stopped(referral)
        # An incomplete action (`needs` set), or one naming a call this loop
        # does not know how to drive itself: `discover_source` names no such
        # shape today, but escalating rather than guessing keeps this loop
        # from mis-driving a future one.
        return _Stopped(
            _unmet_capability(
                reason or "discover_source named an action this loop cannot complete",
                requirement_id=requirement_id,
                action=action.model_dump(mode="json"),
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
        # was never the problem. Raised as a referral here, on the first
        # occurrence, rather than folded into `_unmet_capability`'s
        # `needs_decision`, which the translator's own `output_schema`
        # (`Compiled | Referral`, `translator.agent.create_translator_agent`)
        # gives the model no legal way to close a turn on: every branch in
        # this loop that stops without compiling produces a schema-conforming
        # `Referral` for the model to restate verbatim, this one included.
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
        _unmet_capability(
            "discover_source answered with no action and no single source to use",
            requirement_id=requirement_id,
            answer=answer,
        )
    )


async def _discover_one(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    plan: Plan,
    arguments: dict[str, object],
) -> str | _Stopped:
    """Call `discover_source` once and settle its answer immediately.

    For `_apply_compile_action`'s discover-again retry, where one
    requirement's discovery is in flight at a time, so calling and settling
    together is safe. `_resolve_dataset_sources` keeps the two apart for its
    own concurrent calls.
    """
    requirement_id = cast(str, arguments["requirement_id"])
    answer = await _discover_answer(core, tool_context, arguments=arguments)
    return await _settle_discovery(
        core, tool_context, plan=plan, requirement_id=requirement_id, answer=answer
    )


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
    it is what writes the referral or the pending judgement to `translate`'s
    single-slot state, so the answers are walked back in requirement order
    and settled one at a time -- the first genuine exception is re-raised,
    the first settlement that yields a `_Stopped` is returned at once, and
    no later answer is settled. Every sibling's `discover_source` call has
    already run and written its own record and sources to the log by the
    time this returns, so a re-entry after the returned referral or
    judgement still finds the untouched requirements `selectable` and
    spends no fresh call on them.
    """
    dataset_requirements = [
        requirement
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    ]
    requirement_keys = {
        requirement.id: requirement_revision(requirement) for requirement in dataset_requirements
    }
    pinned = {
        requirement.id: requirement.source_spec_id
        for requirement in dataset_requirements
        if requirement.source_spec_id is not None
    }
    across = {
        requirement.id for requirement in dataset_requirements if requirement.across is not None
    }
    log = load_log(tool_context.state)
    settled = settled_sources(
        tool_context.state, log, requirement_keys=requirement_keys, pinned=pinned, across=across
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
        step = await _settle_discovery(
            core, tool_context, plan=plan, requirement_id=requirement.id, answer=answer
        )
        if isinstance(step, _Stopped):
            return step
        discovered[requirement.id] = step
    return discovered


# --- propose-then-fix, following `_compile_failure_action` up to MAX_ROUNDS --


async def _apply_compile_action(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    plan: Plan,
    action: Action,
    result: dict[str, object],
) -> dict[str, object] | None:
    """Follow `action` past a failed `propose_operations`, or `None` to try again this round.

    `None` means `action` pointed back at `discover_source` and it resolved
    a new source to retry with -- the only case `_propose_until_settled`
    itself has to act on further, since its retry state (`discovered`) is
    not this function's to hold.
    """
    if action.needs:
        # The only `needs_decision` case with an answer this loop can apply
        # on its own once given: `_compile_failure_action`'s `NO_CAPABILITY`
        # branch names `needs=("decision_id",)`, so this is the pending
        # judgement `_settle_unmet_capability` resumes. The reason recorded
        # is the action's own, not the issue's raw `result`, since that is
        # what `return_to_planner` needs once the model answers.
        reason = str((action.arguments or {}).get("reason", "propose_operations named an issue"))
        _save_pending(
            tool_context.state,
            _PendingJudgement(
                judgement="unmet_capability", revision=plan_revision(plan), reason=reason
            ),
        )
        return _unmet_capability(
            f"propose_operations named an action needing {list(action.needs)}",
            action=action.model_dump(mode="json"),
            problem=result,
            decisions=[decision.model_dump(mode="json") for decision in plan.decisions],
        )
    if action.call == "return_to_planner":
        kind = (action.arguments or {}).get("kind")
        reason = str((action.arguments or {}).get("reason", ""))
        if kind == ReferralKind.SOURCE_AMBIGUOUS.value:
            requirement_id = cast(str, (action.arguments or {}).get("requirement_id"))
            return await _begin_candidate_ranking(
                core, tool_context, plan=plan, requirement_id=requirement_id, reason=reason
            )
        return core.return_to_planner(tool_context=tool_context, **(action.arguments or {}))
    if action.call != "discover_source":
        return _unmet_capability(
            f"propose_operations named an unknown next call {action.call!r}",
            action=action.model_dump(mode="json"),
        )
    return None


async def _propose_until_settled(
    core: TranslatorCore, tool_context: ToolContext, *, plan: Plan, discovered: dict[str, str]
) -> dict[str, object]:
    """Build and propose a translation, following `_compile_failure_action` up to `MAX_ROUNDS`."""
    for _round in range(MAX_ROUNDS):
        translation = build_translation(plan, discovered)
        result = await core.propose_operations(translation=translation, tool_context=tool_context)
        if result.get("outcome") == "ok":
            return result

        action = _action_of(result)
        if action is None:
            return _unmet_capability(
                "propose_operations reported issues with no action to follow", problem=result
            )
        settled = await _apply_compile_action(
            core, tool_context, plan=plan, action=action, result=result
        )
        if settled is not None:
            return settled

        step = await _discover_one(
            core, tool_context, plan=plan, arguments=dict(action.arguments or {})
        )
        if isinstance(step, _Stopped):
            return step.result
        requirement_id = cast(str, (action.arguments or {})["requirement_id"])
        discovered[requirement_id] = step

    return _unmet_capability(f"propose_operations did not settle within {MAX_ROUNDS} rounds")


# --- resuming a pending judgement --------------------------------------------


def _mismatch(message: str, **detail: JsonValue) -> dict[str, object]:
    return problem(
        ProblemCode.INVALID_ARGUMENTS, message, outcome=Outcome.INVALID_INPUT, detail=detail or None
    )


def _settle_candidate_ranking(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    pending: _PendingJudgement,
    requirement_id: str | None,
    choice: str | None,
    rationale: str | None,
) -> dict[str, object]:
    if requirement_id != pending.requirement_id:
        return _mismatch(
            f"translate is waiting on a candidate ranking for {pending.requirement_id!r}, not "
            f"{requirement_id!r}.",
            pending_requirement_id=pending.requirement_id,
        )
    if choice not in pending.candidate_spec_ids:
        return _mismatch(
            f"{choice!r} is not one of the candidates translate showed for {requirement_id!r}.",
            candidates=list(pending.candidate_spec_ids),
        )
    if not rationale:
        return _mismatch("a candidate ranking needs a rationale, not only a choice.")
    referral = core.return_to_planner(
        kind=ReferralKind.SOURCE_AMBIGUOUS,
        reason=pending.reason,
        requirement_id=requirement_id,
        recommendation=SourceRecommendation(spec_id=choice, rationale=rationale),
        tool_context=tool_context,
    )
    _clear_pending(tool_context.state)
    return referral


def _settle_unmet_capability(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    pending: _PendingJudgement,
    decision_id: str | None,
) -> dict[str, object]:
    if not decision_id:
        return _mismatch(
            "translate is waiting on which decision no operation covers; answer with decision_id."
        )
    referral = core.return_to_planner(
        kind=ReferralKind.NO_OPERATION_COVERS,
        reason=pending.reason,
        decision_id=decision_id,
        tool_context=tool_context,
    )
    _clear_pending(tool_context.state)
    return referral


def _resume(
    core: TranslatorCore,
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement_id: str | None,
    choice: str | None,
    rationale: str | None,
    decision_id: str | None,
) -> dict[str, object]:
    pending = _load_pending(tool_context.state)
    if pending is None:
        return problem(
            ProblemCode.NO_PENDING_JUDGEMENT,
            "translate has no pending judgement to answer; call it with no arguments to "
            "start a fresh run.",
            outcome=Outcome.INVALID_INPUT,
        )
    if pending.revision != plan_revision(plan):
        _clear_pending(tool_context.state)
        return problem(
            ProblemCode.NO_PENDING_JUDGEMENT,
            "the plan changed since translate asked this; call it again with no arguments.",
            outcome=Outcome.INVALID_INPUT,
        )
    if pending.judgement == "candidate_ranking":
        return _settle_candidate_ranking(
            core,
            tool_context,
            pending=pending,
            requirement_id=requirement_id,
            choice=choice,
            rationale=rationale,
        )
    return _settle_unmet_capability(core, tool_context, pending=pending, decision_id=decision_id)


# --- the public entry point --------------------------------------------------


async def run_translate(
    tool_context: ToolContext,
    *,
    core: TranslatorCore,
    requirement_id: str | None = None,
    choice: str | None = None,
    rationale: str | None = None,
    decision_id: str | None = None,
) -> dict[str, object]:
    """Run the translator's whole discover-then-compile loop without a model turn.

    Called with every argument `None`, drives `discover_source` for every
    dataset requirement, then `propose_operations`, following each step's
    own `action` the way `translator.agent.INSTRUCTION` once told a model
    to. Called with `requirement_id`/`choice`/`rationale` or `decision_id`
    set, applies the model's answer to whichever judgement `translate` last
    asked, and refuses one that does not match what is pending.

    Returns:
        `{"outcome": "ok", ...}` once `propose_operations` succeeds;
        `{"outcome": "referred", "referral": {...}}` the moment any step
        names a referral with a complete `Action`, or once a pending
        candidate ranking or unmet-capability judgement is answered;
        `{"outcome": "needs_decision", ...}` the moment a step's own answer
        carries no action, or one naming `needs`, pausing for the model to
        settle it; or the typed problem that stops a malformed resume.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan

    answer_given = (requirement_id, choice, rationale, decision_id) != (None, None, None, None)
    if answer_given:
        return _resume(
            core,
            tool_context,
            plan=plan,
            requirement_id=requirement_id,
            choice=choice,
            rationale=rationale,
            decision_id=decision_id,
        )

    unsupported = _unsupported_requirement(plan)
    if unsupported is not None:
        return core.return_to_planner(
            tool_context=tool_context,
            kind=ReferralKind.REQUIREMENT_UNMEETABLE,
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
