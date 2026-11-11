"""The planner's tool bodies, over a `planner.call.PlannerCall` and no agent framework.

`ask_user`, `propose` (with `propose_reading_sources` and `propose_and_translate`)
and `reject` are what the planner's `ask_user_question`, `propose_plan` and
`reject_plan` do. Each reads its call's session state and steps through the
`PlannerCall` it is given, and a tool that ends the planner's run does so
through an `EndRun` the framework supplies. `planner.specs` declares them as
the tools the model reads.

A `propose_plan` whose translation compiles declares the plan ready for the
user's answer in code (`declare_ready`), with no model call. It runs
`_execution_blocked`'s checks (a rejected plan, a pending referral, an
uncompiled workflow, a status other than `PENDING_APPROVAL`) and, past those,
answers `_approval_summary(plan)` (the version and revision that tie the
answer to one content hash, and the compiler's warnings, the one thing on it a
caller cannot read off the plan record) under `summary`. `ready_and_end` ends
the run on an answer carrying `summary`. Declaring a plan ready approves
nothing. Whatever runs the planner withholds execution until the user's
answer arrives, which for the served path is the harness's own graph.

An ask the planner makes (`ask_user`) is recorded under
`PENDING_ASK_STATE_KEY` and answers `ASK_ACCEPTED`. The graph's planner node
puts it to the user once the run ends.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import Any, NamedTuple

from compiler.surface import ToolSurface
from harness.turn_events import PLANNER_ASK
from submission.prices_choice import PRICES_OPTIONS
from submission.service import (
    COMBINATION_OPTIONS,
    Proposal,
    submit,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.referral import (
    USER_DECIDES,
    Referral,
    ReferralKind,
    clear_referral,
    load_referral,
)
from translator.tool import Translate, authored_answer

from planner.call import (
    CallSession,
    PlannerCall,
    answered_since_referral,
    candidate_ids,
)
from planner.plan import (
    Plan,
    PlanStatus,
    SessionState,
    load_plan,
    plan_revision,
    save_plan,
    split_of,
)
from planner.source_reads import with_source_reads
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem

logger = logging.getLogger(__name__)


_NO_PLAN = "No plan has been proposed yet."


def _approval_summary(plan: Plan) -> dict[str, object]:
    """Return what a next call needs to know about the compiled `plan`.

    The version and revision, which tie a later answer to one content hash,
    and every warning the compiler let through (`Workflow.warnings`). The
    plan's own decisions, requirements, sources and steps are deliberately
    absent. `harness.studio_translate` builds the card the user approves from
    the plan record, through `harness.plan_display`, and translation and
    `declare_ready` both read the record out of session state, so a copy here is
    a second description of one plan that can disagree with the first.

    Requires a plan that carries a workflow.
    """
    workflow = plan.workflow
    assert workflow is not None
    return {
        "version": plan.version,
        "revision": plan_revision(plan),
        "warnings": [warning.model_dump(mode="json") for warning in workflow.warnings],
    }


def _execution_blocked(plan: Plan, call: PlannerCall) -> dict[str, object] | None:
    """Return the problem that stops `declare_ready` on `plan`, or `None` to let it proceed.

    Checked in order: a rejected plan; a pending referral, which the planner
    settles by proposing again, after asking the user where the referral is of
    a kind in `USER_DECIDES` (the refusal's `action` names `PLANNER_ASK` until
    the user answers, `propose_plan` otherwise); a plan not yet translated and
    compiled; and, last, an allowlist
    on `plan.status` itself. Only `PENDING_APPROVAL` may be declared ready, so a
    plan already `APPROVED`, `EXECUTING`, `COMPLETED` or `FAILED` is refused
    too, even though none of those carries a message of its own above. A plan
    seeded back from a failed run (see `harness.graph`) still carries a workflow
    and the version it failed at, so it would otherwise pass every check above
    and reach the user as though it were still approvable.
    """
    if plan.status is PlanStatus.REJECTED:
        return problem(
            ProblemCode.PLAN_REJECTED,
            "The current plan was rejected. Propose a new one before executing.",
            outcome=Outcome.REFUSED,
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    referral = load_referral(call.state)
    if referral is not None:
        asks_user = referral.kind in USER_DECIDES and not answered_since_referral(call)
        return problem(
            ProblemCode.REFERRAL_PENDING,
            "The translator referred this plan back; settle the referral with the user "
            "and propose again before executing.",
            outcome=Outcome.REFUSED,
            detail={"referral": authored_answer(referral.model_dump(mode="json"), split_of(plan))},
            action=Action(call=PLANNER_ASK) if asks_user else PROPOSE_PLAN_PROPOSER.action(),
        )
    if plan.status is PlanStatus.DRAFT or plan.workflow is None:
        # A draft states the methodology. What runs arrives with translation,
        # so approving one would approve nothing to run.
        return problem(
            ProblemCode.WORKFLOW_MISSING,
            "This plan has not compiled into a workflow, so nothing can execute yet. "
            "Propose it again, with its `version` as `amends_version`, to have it "
            "translated.",
            outcome=Outcome.REFUSED,
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    if plan.status is not PlanStatus.PENDING_APPROVAL:
        return problem(
            ProblemCode.TRANSITION_REFUSED,
            f"This plan is {plan.status.value}, not awaiting approval, so it cannot be "
            "declared ready. Propose a new plan before executing.",
            outcome=Outcome.REFUSED,
            detail={"status": plan.status.value},
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    return None


def declare_ready(
    answer: dict[str, object],
    call: PlannerCall,
) -> dict[str, object]:
    """Declare the plan on record ready when `answer`'s translation compiled.

    `answer` is an accepted `propose_plan` answer carrying its `translation`.
    A verdict other than `ok` leaves it unchanged. Past `_execution_blocked`,
    it gains `summary` (`_approval_summary`), and the run ends there
    (`ready_and_end`). The harness asks the user, so no model call follows to
    restate the plan the user already sees. A plan the gate blocks gains the
    gate's problem under `submission` instead, its `action` naming
    `propose_plan`, and the run goes on. Declaring a plan ready approves
    nothing. Whatever runs this agent gates on the user's answer, keyed on the
    summary's revision.
    """
    translation = answer.get("translation")
    if not isinstance(translation, dict) or translation.get("outcome") != Outcome.OK.value:
        return answer
    plan = load_plan(call.state)
    assert plan is not None, "a translation answers ok only for the plan on record"
    blocked = _execution_blocked(plan, call)
    if blocked is not None:
        return {**answer, "submission": blocked}
    return {**answer, "summary": _approval_summary(plan)}


# The `propose_plan` arguments that patch a plan. A call naming none of them
# changes nothing about the version it amends.
_PATCH_ARGUMENTS = (
    "decisions",
    "evidence_requirements",
    "removed_decisions",
    "removed_evidence_requirements",
    "declined",
    "answered",
    "removed_asks",
)


def _transient_retry(arguments: Mapping[str, Any], state: SessionState) -> Plan | None:
    """The recorded plan a `propose_plan` call asks to have translated again, or `None`.

    A transient referral (the data service did not answer) leaves the
    recorded plan as it was. A call amending that version and changing
    nothing (`amends_version` alone, any restated `question` or
    `answer_mode` equal to the recorded one) asks only for that version to
    be translated again, so it records no new version and spends no plan
    round. Any other call is a proposal.
    """
    referral = load_referral(state)
    plan = load_plan(state)
    if (
        referral is None
        or referral.outcome is not Outcome.TRANSIENT
        or plan is None
        or arguments.get("amends_version") != plan.version
        or arguments.get("starts_new_plan")
        or any(arguments.get(name) for name in _PATCH_ARGUMENTS)
    ):
        return None
    if arguments.get("question") not in (None, plan.question):
        return None
    if arguments.get("answer_mode") not in (None, plan.answer_mode):
        return None
    return plan


def ask_refusal(args: Mapping[str, Any]) -> dict[str, object] | None:
    """The refusal an ask earns on its arguments alone, or `None` for one that pauses.

    ADK hands the model's arguments over unchecked. A refusal answers the
    call at once, so the run goes on and Studio asks nothing. An ask this
    accepts ends the planner's run, and the graph's planner node pauses on
    it with these arguments as its `RequestInput`'s `payload`, from which
    Studio builds its question (`harness.studio_translate._planner_question`).
    `message` is the question as text. Each of `candidates` is a
    mapping, read by its `spec_id` and `label`. `options` are
    `COMBINATION_OPTIONS` or `PRICES_OPTIONS` exactly, as a refusal's action
    carries them, offered without `candidates`. A `recommended_spec_id` names
    one of `candidates`.
    """
    if not isinstance(args.get("message"), str):
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "`message` must be the question to show the user, as text.",
            outcome=Outcome.INVALID_INPUT,
        )
    candidates = args.get("candidates") or []
    if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            '`candidates` must be a list of `{"spec_id": ..., "label": ...}`.',
            outcome=Outcome.INVALID_INPUT,
        )
    options = args.get("options") or []
    fixed = ([dict(o) for o in COMBINATION_OPTIONS], [dict(o) for o in PRICES_OPTIONS])
    if options and (candidates or options not in fixed):
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "`options` must be the ones a refusal's action carries, unchanged, "
            "and offered without `candidates`. With no such refusal, leave `options` "
            "out and ask in `message` alone.",
            outcome=Outcome.INVALID_INPUT,
        )
    offered = [str(c["spec_id"]) for c in candidates if c.get("spec_id")]
    recommended_spec_id = args.get("recommended_spec_id")
    if recommended_spec_id and recommended_spec_id not in offered:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            f"recommended_spec_id {recommended_spec_id!r} names no candidate in {offered!r}",
            outcome=Outcome.INVALID_INPUT,
        )
    return None


def _entity_ask_unbound(referral: Referral | None, candidates: object) -> dict[str, object] | None:
    """The refusal for an ask under an entity referral whose candidates are not one
    reading's picks, or `None`.

    While an `entity_name_ambiguous` referral is pending, an ask that offers
    candidates offers those of one reading in its `ambiguous_entities`, each
    under its `pick` (`planner.records.entity_pick`), and offers all of them.
    Studio shows an ask as one single-select question, so one answer settles
    one entity, an id that names no pick would come back as an answer nothing
    reads, and an ask leaving a candidate out would make the planner's choice
    the user's. An ask with no candidates, as for free text or a combination,
    passes.
    """
    if referral is None or referral.kind is not ReferralKind.ENTITY_NAME_AMBIGUOUS:
        return None
    offered = candidate_ids(candidates)
    readings = {
        candidate.pick: reading.entity
        for reading in referral.ambiguous_entities
        for candidate in reading.candidates
    }
    unknown = sorted(offered - readings.keys())
    if unknown:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            f"spec_id {unknown[0]!r} is no candidate's `pick` in the pending referral's "
            "`ambiguous_entities`. Offer one reading's candidates, each under its `pick` "
            "unchanged.",
            outcome=Outcome.INVALID_INPUT,
        )
    touched = {readings[spec_id] for spec_id in offered}
    if len(touched) > 1:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "These candidates come from more than one reading in `ambiguous_entities`. Ask "
            f"about one reading per `{PLANNER_ASK}` call, and the next once the user has "
            "answered.",
            outcome=Outcome.INVALID_INPUT,
        )
    if touched and offered != {pick for pick, entity in readings.items() if entity in touched}:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "These candidates leave out some of their reading's candidates in "
            "`ambiguous_entities`. Offer every candidate of the reading, each under its `pick`.",
            outcome=Outcome.INVALID_INPUT,
        )
    return None


def _ask_check(args: Mapping[str, Any], state: SessionState) -> dict[str, object] | None:
    """The refusal an ask with `args` earns on its arguments and the pending
    referral, or `None` for one its tool accepts unless it follows another ask
    in the same response (`_ask_beside_ask`).

    A history copy's referral is read as the model wrote the plan
    (`translator.tool.authored_answer`), so its picks are the origin's.
    """
    refusal = ask_refusal(args)
    if refusal is not None:
        return refusal
    referral = load_referral(state)
    plan = load_plan(state)
    if referral is not None and plan is not None and plan.rolling_copies:
        referral = Referral.model_validate(
            authored_answer(referral.model_dump(mode="json"), split_of(plan))
        )
    return _entity_ask_unbound(referral, args.get("candidates"))


def _ask_beside_ask(call: PlannerCall) -> dict[str, object] | None:
    """The refusal for an ask beside an earlier ask in its model response, or `None`.

    The user answers one question at a time. The calls of one response run
    concurrently, and an accepted ask ends the run before the model is
    called again, so the only way two asks can wait together is two in one
    response. The first ask in the response's call list is the only one that
    may reach the user. Every later one is refused, whatever became of the
    first. Deciding from the call list alone gives the same answer under any
    order the response's calls run in.

    `PlannerCall.response_calls` gives the response's call list. `steps` cannot,
    since an accepted ask of an earlier response that no answer followed stands
    next to this response's asks there.
    """
    asks = [c.call_id for c in call.response_calls() if c.name == PLANNER_ASK]
    if call.call_id not in asks or asks[0] == call.call_id:
        return None
    return problem(
        ProblemCode.INVALID_ARGUMENTS,
        f"This response calls `{PLANNER_ASK}` more than once, and the user answers one "
        "question at a time. Ask this one in your next response, if it still needs "
        "asking.",
        outcome=Outcome.INVALID_INPUT,
    )


# The planner tool that rejects the plan on record (`reject`).
REJECT_PLAN = "reject_plan"


# The planner tools that read and write `planner.plan.PLAN_STATE_KEY`, of which
# one runs per model response (`running_plan_write`).
PLAN_WRITES = frozenset({PROPOSE_PLAN_PROPOSER.call, REJECT_PLAN})


def running_plan_write(names: Sequence[str]) -> int | None:
    """The place, among one model response's call `names` in order, of the plan
    write that runs: the first `PLAN_WRITES` call, or `None` when there is none.
    Every later plan write of the response is refused
    (`_plan_write_beside_plan_write`)."""
    return next((i for i, name in enumerate(names) if name in PLAN_WRITES), None)


def _plan_write_beside_plan_write(call: PlannerCall) -> dict[str, object] | None:
    """The refusal for a plan write beside an earlier one in its model response, or `None`.

    `propose_plan` and `reject_plan` each read and write
    `planner.plan.PLAN_STATE_KEY`, and the calls of one response run
    concurrently, so two of them leave whichever finished last, and a
    proposal's translation can read the other's version. The first of the two
    kinds in the response's call list (`PlannerCall.response_calls`) runs,
    whatever its answer (`running_plan_write`). Every later one is refused
    before it reads or writes any state. Deciding from the call list alone
    gives the same answer under any order the response's calls run in. An ask
    beside a plan write is left alone.
    """
    calls = call.response_calls()
    place = next((i for i, c in enumerate(calls) if c.call_id == call.call_id), None)
    if (
        place is None
        or calls[place].name not in PLAN_WRITES
        or place == running_plan_write([c.name for c in calls])
    ):
        return None
    return problem(
        ProblemCode.PLAN_WRITE_REPEATED,
        f"Only the first `{PROPOSE_PLAN_PROPOSER.call}` or `{REJECT_PLAN}` call of a response "
        "runs. Make this call in your next response, if the plan still needs it.",
        outcome=Outcome.INVALID_INPUT,
    )


# Where an accepted ask waits for the graph's planner node to put it to the
# user (`AcceptedAsk`). `harness.planner_adapter.PlannerAdapter` reads it once the
# planner's run ends and clears it when the next entry starts.
PENDING_ASK_STATE_KEY = "planner:pending_ask"


class AcceptedAsk(NamedTuple):
    """An ask the planner made and its tool accepted, waiting for the user.

    `call_id` is the id of the planner's `PLANNER_ASK` call, which the graph's
    planner node pauses on as the interrupt id, so the user's answer carries
    it. `args` are the call's arguments, the question as Studio shows it
    (`harness.studio_translate._planner_question`).
    """

    call_id: str
    args: dict[str, Any]


def load_pending_ask(state: SessionState) -> AcceptedAsk | None:
    """The ask waiting for the user, or `None`."""
    held = state.get(PENDING_ASK_STATE_KEY)
    return AcceptedAsk(held["call_id"], dict(held["args"])) if isinstance(held, dict) else None


def clear_pending_ask(state: SessionState) -> None:
    """Forget the ask waiting for the user."""
    state[PENDING_ASK_STATE_KEY] = None


# What an accepted ask answers. Its call has a response, and the user's answer
# reaches the planner in the brief of the entry that resumes it.
ASK_ACCEPTED = ok(
    status="asked",
    message="The question is with the user. Their answer arrives with your next brief.",
)


# What a tool does to end the planner's run on its answer, so that no model
# call follows. The framework running the planner supplies it.
type EndRun = Callable[[], None]


def ready_and_end(
    answer: dict[str, object], call: PlannerCall, end_run: EndRun
) -> dict[str, object]:
    """`declare_ready` over `call`, ending the run once the plan is ready.

    An answer carrying `summary` is a plan declared ready, so `end_run` ends the
    run on it with no model call after. Any other answer leaves the run going.
    """
    declared = declare_ready(answer, call)
    if "summary" in declared:
        end_run()
    return declared


def ask_user(
    call: PlannerCall,
    end_run: EndRun,
    *,
    message: str,
    candidates: list[dict[str, str]],
    options: list[dict[str, str]],
    recommended_spec_id: str | None,
    rationale: str | None,
) -> dict[str, object]:
    """Record the planner's ask as pending and end the run, or refuse it.

    An accepted ask ends the run (`end_run`) once the calls made beside it
    have answered, as a plan declared ready does (`ready_and_end`).
    """
    # Copied here, so the durable record shares no list with the defaults
    # or the model's call.
    args: dict[str, Any] = {
        "message": message,
        "candidates": copy.deepcopy(candidates),
        "options": copy.deepcopy(options),
        "recommended_spec_id": recommended_spec_id,
        "rationale": rationale,
    }
    state = call.state
    refusal = _ask_check(args, state) or _ask_beside_ask(call)
    if refusal is not None:
        return refusal
    call_id = call.call_id
    assert call_id is not None, "every function call has an id"
    state[PENDING_ASK_STATE_KEY] = {"call_id": call_id, "args": args}
    end_run()
    return dict(ASK_ACCEPTED)


def propose(
    call: PlannerCall,
    arguments: Mapping[str, Any],
    *,
    reads_user_words: bool,
) -> dict[str, object]:
    """Record the plan a `propose_plan` call's `arguments` state, or patch the one
    on record, over `call`'s session. A call after another plan write in its response
    is refused (`_plan_write_beside_plan_write`)."""
    refusal = _plan_write_beside_plan_write(call)
    if refusal is not None:
        return refusal
    return submit(
        CallSession(call),
        Proposal(**arguments),
        reads_user_words=reads_user_words,
        proposer=PROPOSE_PLAN_PROPOSER,
    )


async def propose_reading_sources(
    call: PlannerCall,
    arguments: Mapping[str, Any],
    *,
    reads_user_words: bool,
    surface: ToolSurface | None,
) -> dict[str, object]:
    """`propose`. With a `surface`, the source reads a refusal asks for are made,
    and the proposal is made again (`with_source_reads`)."""

    def draft() -> dict[str, object]:
        return propose(call, arguments, reads_user_words=reads_user_words)

    if surface is None:
        return draft()
    return await with_source_reads(draft, state=call.state, surface=surface)


async def propose_and_translate(
    call: PlannerCall,
    end_run: EndRun,
    *,
    arguments: Mapping[str, Any],
    reads_user_words: bool,
    translate: Translate,
    surface: ToolSurface | None,
) -> dict[str, object]:
    """`propose_reading_sources` over `arguments`, translating the plan it records.

    Translation runs once, on the version the proposal recorded, after any
    source reads. The proposal's own `outcome` stays `ok` whatever the verdict,
    since the version is recorded either way, and a verdict of `ok` declares
    the plan ready (`ready_and_end`). A proposal that records nothing
    translates nothing. A bare re-proposal after a transient verdict
    (`_transient_retry`) translates the recorded version again and records
    nothing. A call after another plan write in its response is refused before
    either (`_plan_write_beside_plan_write`).
    """
    refusal = _plan_write_beside_plan_write(call)
    if refusal is not None:
        return refusal
    retried = _transient_retry(arguments, call.state)
    if retried is not None:
        clear_referral(call.state)
        answer = ok(
            status=retried.status.value,
            version=retried.version,
            previous_version=retried.previous_version,
            question=retried.question,
            translation=await translate(call),
        )
        return ready_and_end(answer, call, end_run)
    result = await propose_reading_sources(
        call,
        arguments,
        reads_user_words=reads_user_words,
        surface=surface,
    )
    if result.get("outcome") != Outcome.OK.value:
        return result
    return ready_and_end({**result, "translation": await translate(call)}, call, end_run)


def reject(call: PlannerCall) -> dict[str, object]:
    """Record that the user rejected the plan on record, unless it has run.

    A call after another plan write in its response is refused
    (`_plan_write_beside_plan_write`).
    """
    refusal = _plan_write_beside_plan_write(call)
    if refusal is not None:
        return refusal
    state = call.state
    plan = load_plan(state)
    if plan is None:
        return problem(ProblemCode.PLAN_MISSING, _NO_PLAN, outcome=Outcome.STATE_MISSING)
    try:
        plan = plan.reject()
    except ValueError as exc:
        # A run that has started or finished cannot be withdrawn.
        return problem(
            ProblemCode.TRANSITION_REFUSED,
            str(exc),
            outcome=Outcome.REFUSED,
            detail={"status": plan.status.value},
        )
    save_plan(state, plan)
    clear_referral(state)
    return ok(status=plan.status.value)
