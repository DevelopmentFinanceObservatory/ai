"""The orchestrator's `submit_analysis`: a reviewed recipe run without a plan.

The tool runs one recipe of `submission.recipes.RECIPES` that this server
enables (`DirectRecipes.recipes`, from `Settings.orchestrator_direct_recipes`).
Eligibility is the harness's to decide, from what the conversation recorded:
the source must be a spec_id a search settled on (a `definite` part, or a
candidate the turn fetched), read here from the turn's search readings and
spec blocks or from an earlier turn's (`source_evidence`), and the recipe's
own preconditions are checked against that record (`submission.recipes`).
The growth method is the model's reading of the user's words, which nothing
recorded can check; the run's `RequestAuthorisation` records it as
`model_stated` (`Recipe.origins`).

A request goes to the planner instead (`requires_planning`) when this turn
already holds a pending plan or has spent its one `plan` call, when it names
no enabled recipe, or when the recipe's preconditions do not hold. A missing
or unverified value is `needs_fields`. A method the user's words leave open,
or a source that counts its measure in more than one way when the user has
not picked this one (`submission.service.measure_choices`, the choice a plan
card shows), is `needs_decision`.

An eligible request takes the turn's `plan` call before its first await, so
a second call in the same model response finds it spent. The call stays
taken only when the request's run was admitted and stands: accepted or
failed (`_keeps_plan_call`), stopped by a defect and recorded failed
(`RunAbandonedError`), or left unfinished by a ledger whose store could not
record or read it (`LedgerUnavailableError`). Otherwise it is given back
whichever way the request ends, an exception included. A run that stopped on
a method choice or a source to find again gives it back, since its remedy is
a plan. It compiles through the planner's own chain
(`submission.direct.compile_direct`); a refusal there answers
`requires_planning` with the chain's codes (`_verdict_codes`). It runs under
a `RequestAuthorisation` as one run of the graph's ledger
(`executor.run.execute_authorised`), with the completion gate, its compiled
plan stored beforehand for a later turn that completes the run
(`harness.graph.answer.store_run_plan`). An accepted run's plan is recorded
as `plan_done` records a plan's (`harness.graph.plan_done.record_plan`), so a
later turn recalls it and keeps its source's block. A completed run's lineage
is restored and its terminal frames linked
(`harness.graph.answer.link_accepted_run`), described and listed as
`plan_done` lists them (`harness.graph.plan_done.HandBack`), so `make_charts`
draws them. Its delivery is decided by those frames
(`harness.graph.delivery.deliver`) and acknowledged once the answer is in the
session (`acknowledge_deliveries`). A failed run answers `failed` with its
code and a fixed sentence (`harness.graph.execute.STOP_MESSAGES`, or the
refusing tool's own), and a run the ledger's store could not admit answers
`failed` with `run_not_started`, giving the call back. Any other unfinished
run answers `pending` with its run ID, as `plan_done` states one, and is
carried on by the next turn's `seed` node (`harness.graph.resume`).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import FunctionType
from typing import Any

from compiler.sources import (
    EXACT_COVERAGE,
    default_reading,
    is_family,
    measure_named_by_query,
    source_from_spec_block,
)
from compiler.surface import ToolSurface
from executor.completion import ResultPublisher, run_hand_back, unrecorded_hand_back
from executor.ledger import (
    LedgerUnavailableError,
    RequestAuthorisation,
    RunLedger,
    RunRecord,
    RunStatus,
    idempotency_key,
    store_faults,
    submission_digest,
)
from executor.outcomes import (
    Accepted,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    RunOutcome,
)
from executor.run import (
    FetchSeriesStats,
    RunAbandonedError,
    Submission,
    default_fetch_series_stats,
    execute_authorised,
)
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import ToolContext
from planner.plan import PinSource, PlanStatus, earlier_block
from planner.records import GrowthChange, StatVarReading, Year
from submission.direct import Compiled, compile_direct
from submission.recipes import (
    RECIPES,
    DecisionCode,
    DirectOutcome,
    DirectRefusal,
    PlanningCode,
    Recipe,
    SourceEvidence,
    requires_planning,
)
from submission.service import Proposer, measure_choices
from translator.discovery import BLOCKING_WARNING_CODES

from harness.conversation import conversation_of
from harness.graph.answer import (
    answer_frames,
    answer_notes,
    link_accepted_run,
    presentable_steps,
    store_run_plan,
)
from harness.graph.delivery import deliver
from harness.graph.execute import STOP_MESSAGES
from harness.graph.plan_done import HandBack, record_plan
from harness.graph.seams import Executor
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    PendingPlan,
    count_call,
    pending_action,
    turn_budgets,
)
from harness.turn_records import SETTLED_STATUSES, SearchReading, plan_status_of

logger = logging.getLogger(__name__)

TOOL_NAME = "submit_analysis"

# How long the harness's authorisation of a direct request lets its run
# dispatch work. A run admitted under it may finish publication later.
DIRECT_AUTHORISATION_TTL = timedelta(hours=1)


@dataclass(frozen=True, kw_only=True)
class DirectRecipes:
    """What the direct route runs with: the recipes enabled, the graph's run
    ledger and publisher, the recorded surface steps run against, and the
    surface the planner's translation compiles against.

    `max_concurrent_steps` and `fetch_series_stats` are
    `executor.run.execute_authorised`'s own.
    """

    recipes: tuple[str, ...]
    run_ledger: RunLedger
    publisher: ResultPublisher | None
    tool_surface: ToolSurface
    translation_surface: ToolSurface
    max_concurrent_steps: int = 4
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats


def _settled_reading(readings: Any, spec_id: str) -> SearchReading | None:
    """The latest of `readings` (stored `SearchReading` JSON) that settled on `spec_id`."""
    for raw in reversed(list(readings or ())):
        reading = SearchReading.model_validate(raw)
        if reading.spec_id == spec_id and reading.status in SETTLED_STATUSES:
            return reading
    return None


def source_evidence(state: Any, spec_id: str) -> tuple[SourceEvidence, PinSource] | None:
    """What the conversation recorded about `spec_id`, and the pin of it the
    planner's chain reads, or `None` when no search settled on it.

    This turn's reading and block come first, each on its own; otherwise an
    earlier turn's reading (`ORCHESTRATOR_PRIOR_LEDGER_KEY`) and the block
    that turn kept (`planner.plan.earlier_block`). A windowed copy an earlier turn's plan
    retrieved is read as the source it copies (the ledger's
    `windowed_from`), whose spec_id the evidence then names: the recipe
    windows that source to the years asked. A source holds one series when
    no side of a flow is left open, each role names one entity, and a family
    of StatVars is read as its sum.
    """
    prior = state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}
    spec_id = (prior.get("windowed_from") or {}).get(spec_id, spec_id)
    reading = _settled_reading(state.get(ORCHESTRATOR_SEARCHES_KEY), spec_id) or (
        _settled_reading(prior.get("readings"), spec_id)
    )
    block = (state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(spec_id)
    earlier_turn = block is None
    if earlier_turn:
        block = earlier_block(spec_id)
    if reading is None or block is None:
        return None
    try:
        source = source_from_spec_block(block)
    except ValueError:
        logger.warning("submit_analysis: the block of %s is not readable as a source", spec_id)
        return None
    roles = Counter(entity.role for entity in source.entities)
    one_series = (
        not source.open_entity_axes
        and all(count == 1 for count in roles.values())
        and (not is_family(source) or default_reading(source) is StatVarReading.SUM)
    )
    codes = tuple(warning.code for warning in reading.warnings)
    evidence = SourceEvidence(
        spec_id=spec_id,
        measure=source.sentence or reading.sentence or reading.text or "",
        entities=tuple(dict.fromkeys(entity.label for entity in source.entities)),
        coverage=source.coverage_window if source.coverage_kind == EXACT_COVERAGE else None,
        one_series=one_series,
        blocking_warnings=tuple(code for code in codes if code in BLOCKING_WARNING_CODES),
    )
    pin = PinSource(
        provenance_ref=reading.provenance_ref,
        provenance_account=reading.provenance_account,
        block=None if earlier_turn else dict(block),
        warnings=codes,
        settled=True,
        earlier_turn=earlier_turn,
        other_count_of=reading.other_count_of,
    )
    return evidence, pin


def _plan_spent(state: Any) -> bool:
    calls = state.get(ORCHESTRATOR_CALLS_KEY) or {}
    return int(calls.get("plan", 0)) >= turn_budgets(state)["plan"]


def _gate(state: Any, recipe: Recipe | None, name: str) -> DirectRefusal | None:
    """The refusal of a request this turn cannot run directly, before its
    parameters are read: no enabled recipe by `name`, a plan pending, or the
    turn's `plan` call spent."""
    if recipe is None:
        return requires_planning(
            PlanningCode.RECIPE_NOT_ENABLED,
            f"No direct recipe {name!r} is enabled here: call plan with the question.",
        )
    if isinstance(pending_action(state), PendingPlan):
        return requires_planning(
            PlanningCode.PLAN_PENDING,
            "This turn already handed a plan to the planner; end your turn.",
        )
    if _plan_spent(state):
        return requires_planning(
            PlanningCode.PLAN_BUDGET_SPENT,
            "This turn has already run its one calculation or plan. Answer with what you "
            "have, and say plainly what was not calculated.",
        )
    return None


def _elected(state: Any, tool_context: ToolContext) -> frozenset[str]:
    """The spec_ids the user chose: picked answering an ask this turn
    (`planner.agent.ToolContextSession.elected_spec_ids`), or recorded as
    their choice in an earlier turn's plan (the ledger's `elected`)."""
    from planner.agent import (  # noqa: PLC0415 - planner.agent imports the orchestrator's tools, which import this module
        ToolContextSession,
    )

    prior = state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}
    return ToolContextSession(tool_context).elected_spec_ids() | frozenset(
        prior.get("elected") or ()
    )


def _measure_refusal(
    state: Any, spec_id: str, pin: PinSource, *, elected: frozenset[str]
) -> DirectRefusal | None:
    """`needs_decision` when `spec_id`'s source counts its measure in more
    than one way (`submission.service.measure_choices`, the measure decision
    a plan card shows for the same pin) and the user has not chosen it:
    `elected`, the spec_ids the user picked (`_elected`), does not hold
    `spec_id`, and the user's words did not name its measure
    (`compiler.sources.measure_named_by_query`).

    A source settled as another count of one (`PinSource.other_count_of`)
    takes its choice from the source that offered it, so that source is
    read beside it when the conversation recorded it. Its own block names
    its own measure, so its trace says nothing of the user's words."""
    shown = {spec_id: pin}
    if pin.other_count_of is not None:
        offering = source_evidence(state, pin.other_count_of)
        if offering is not None:
            shown[pin.other_count_of] = offering[1]
    choice = measure_choices(shown).get(spec_id)
    if choice is None or spec_id in elected:
        return None
    block = pin.spec_block(spec_id)
    if pin.other_count_of is None and block is not None and measure_named_by_query(block):
        return None
    return DirectRefusal(
        outcome=DirectOutcome.NEEDS_DECISION,
        message="This source counts the measure in more than one way and the user has not "
        "chosen one: ask the user, offering each option with its spec_id, and call again with "
        "the spec_id they pick.",
        code=DecisionCode.MEASURE_UNRESOLVED.value,
        options=choice.options,
        detail={
            "choices": [
                {"label": label, "spec_id": choice_spec_id}
                for label, choice_spec_id in zip(choice.options, choice.spec_ids, strict=True)
            ]
        },
    )


def _keeps_plan_call(outcome: RunOutcome) -> bool:
    """Whether `outcome` keeps the turn's `plan` call: a run the ledger
    admitted that was accepted or failed. A run that stopped on a method
    choice or a source to find again gives the call back for the plan its
    remedy needs, and a refusal with no run ran nothing."""
    if isinstance(outcome, Accepted):
        return True
    return isinstance(outcome, Refused) and outcome.run_id is not None


def _stopped(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """A run that stopped before it was accepted, as the orchestrator reads it."""
    if isinstance(outcome, MethodChoiceNeeded):
        return DirectRefusal(
            outcome=DirectOutcome.NEEDS_DECISION,
            message=outcome.refusal.message,
            code=outcome.code,
            detail={"candidates": outcome.candidates, "run_id": outcome.run_id},
        ).payload()
    if isinstance(outcome, RediscoveryNeeded):
        return requires_planning(
            PlanningCode.NOT_COMPILED,
            "The series needs finding again before it can be retrieved: call plan with the "
            "question.",
            run_id=outcome.run_id,
        ).payload()
    failure = outcome.payload
    if failure is not None:
        code, message = failure.code, failure.message
    else:
        # `detail` can quote an exception or the ledger's failure reason.
        logger.warning(
            "submit_analysis: run %s stopped as %s: %s",
            outcome.run_id,
            outcome.refusal.value,
            outcome.detail,
        )
        code, message = outcome.refusal.value, STOP_MESSAGES[outcome.refusal]
    return {
        "outcome": DirectOutcome.FAILED.value,
        "run_id": outcome.run_id,
        "code": code,
        "message": message,
    }


def _unfinished(run: RunRecord) -> dict[str, Any]:
    """An accepted run whose results are not all published with their replay
    records, as the orchestrator reads it: `failed` for a failed run, and
    `pending` for any other, as `plan_done` states one."""
    if run.status is RunStatus.FAILED:
        return {
            "outcome": DirectOutcome.FAILED.value,
            "run": run_hand_back(run),
            "message": f"The calculation ran, but run {run.run_id} could not store it with the "
            "record that lets it be reproduced, so it is not complete. State none of its "
            "figures and chart nothing from it: say plainly that the calculation could not be "
            "completed.",
        }
    return {
        "outcome": DirectOutcome.PENDING.value,
        "run": run_hand_back(run),
        "message": "The calculation ran, but storing it with the record that lets it be "
        f"reproduced is unfinished. State none of its figures and chart nothing from it: say "
        f"the calculation is pending under run {run.run_id}.",
    }


# What a request that did not compile is sent to: the orchestrator's `plan`
# tool, called with a brief and asks the model writes.
_PLANNING = Proposer(call="plan")


def _verdict_codes(verdict: dict[str, Any]) -> dict[str, Any]:
    """The planner chain's `verdict` on a direct request it could not
    compile, as the orchestrator reads it under `requires_planning`: its
    `outcome` and code (a problem's `code`, a referral's `kind`), and each
    compile issue's code. Its messages, details and `action` are written for
    a planner and can quote an exception, so they are logged instead."""
    codes: dict[str, Any] = {"outcome": verdict.get("outcome")}
    referral = verdict.get("referral")
    code = verdict.get("code") or (referral.get("kind") if isinstance(referral, dict) else None)
    if code is not None:
        codes["code"] = code
    issues = (verdict.get("detail") or {}).get("issues")
    if isinstance(issues, list):
        codes["issue_codes"] = [issue.get("code") for issue in issues if isinstance(issue, dict)]
    return codes


def _abandoned(run_id: str) -> dict[str, Any]:
    """An admitted run a defect stopped (`executor.run.RunAbandonedError`),
    recorded failed, as the orchestrator reads it: its code, and none of the
    defect's own text."""
    return {
        "outcome": DirectOutcome.FAILED.value,
        "run_id": run_id,
        "code": RefusalKind.RUN_ABANDONED.value,
        "message": "The calculation stopped on an internal fault and could not be completed. "
        "Say plainly that it could not be completed.",
    }


def _unrecorded(run_id: str) -> dict[str, Any]:
    """An admitted run whose progress the ledger's store could not record or
    read (`LedgerUnavailableError`), as the orchestrator reads it: `pending`,
    standing `unfinished` (`executor.completion.unrecorded_hand_back`), for
    the conversation's next turn to carry on."""
    return {
        "outcome": DirectOutcome.PENDING.value,
        "run": unrecorded_hand_back(run_id),
        "message": f"The calculation's run {run_id} could not be saved as it ran, so it is "
        "unfinished. State none of its figures and chart nothing from it: tell the user the "
        "work could not be saved and carries on when they next write.",
    }


def _not_started() -> dict[str, Any]:
    """A run the ledger's store could not admit (`LedgerUnavailableError`
    with no run id), as the orchestrator reads it: `failed`, standing
    `not_started` (`executor.completion.unrecorded_hand_back`). Nothing ran
    and nothing carries on."""
    return {
        "outcome": DirectOutcome.FAILED.value,
        "run": unrecorded_hand_back(None),
        "code": "run_not_started",
        "message": "The calculation could not be started, because its record could not be "
        "saved. State no figure from it: say it could not be run just now.",
    }


# The deliveries of completed direct runs awaiting acknowledgement, by the
# invocation whose `submit_analysis` answer lists them.
_ACKNOWLEDGEMENTS: dict[str, list[Callable[[], Awaitable[None]]]] = {}


async def acknowledge_deliveries(
    callback_context: CallbackContext, llm_request: LlmRequest | None = None
) -> None:
    """Acknowledge each delivery `submit_analysis` decided in this invocation
    (`harness.graph.delivery.Delivery.acknowledge`). The orchestrator calls
    it before each model call and once it finishes, when the tool's answer
    is in the session: ADK appends a non-partial event before the agent that
    yielded it goes on. A delivery an invocation never acknowledges, one cut
    short, leaves its run for the next turn to list (`harness.graph.resume`)."""
    del llm_request
    for acknowledge in _ACKNOWLEDGEMENTS.pop(callback_context.invocation_id, []):
        await acknowledge()


def direct_recipe_tools(direct: DirectRecipes | None, *, executor: Executor) -> list[FunctionType]:
    """The orchestrator's direct `submit_analysis`, or none when `direct` is
    unset or enables no recipe. `executor` describes the frames a run returns."""
    if direct is None or not direct.recipes:
        return []
    enabled = {name: RECIPES[name] for name in direct.recipes}

    async def run(
        recipe: Recipe, compiled: Compiled, *, task: str, tool_context: ToolContext
    ) -> RunOutcome:
        """Run `compiled` under the harness's authorisation of `recipe` for `task`.

        The run's idempotency key names the conversation
        (`harness.conversation.conversation_of`), the recipe's revision and
        the digest of the submission the run executes, the digest the ledger
        binds the key to, so the key never names a run of other work. A
        request whose recorded source, recipe and parameters are unchanged
        compiles to the same submission, so it carries on its run in a later
        turn of the conversation, and other work names another run."""
        now = datetime.now(UTC)
        submission = Submission(
            revision=compiled.digest, workflow=compiled.workflow, requirements=compiled.requirements
        )
        source_policy = submission.source_policy()
        await store_run_plan(direct.run_ledger, revision=submission.revision, plan=compiled.plan)
        authorisation = RequestAuthorisation(
            source_policy=source_policy,
            expires_at=now + DIRECT_AUTHORISATION_TTL,
            task=task,
            recipe_id=recipe.id,
            recipe_revision=recipe.revision,
            graph_digest=compiled.digest,
            limits={"steps": recipe.max_steps, "concurrent_steps": direct.max_concurrent_steps},
            parameter_origins=dict(recipe.origins),
            at=now,
        )
        return await execute_authorised(
            submission,
            authorisation=authorisation,
            idempotency_key=idempotency_key(
                caller=tool_context.user_id,
                conversation_id=conversation_of(tool_context.session.id),
                task=f"direct:{recipe.id}:{recipe.revision}:"
                + submission_digest(submission=submission, source_policy=source_policy),
            ),
            ledger=direct.run_ledger,
            surface=direct.tool_surface,
            publisher=direct.publisher,
            max_concurrent_steps=direct.max_concurrent_steps,
            fetch_series_stats=direct.fetch_series_stats,
        )

    async def answer(
        outcome: RunOutcome, compiled: Compiled, *, task: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """`outcome` handed back: the run's frames, or its standing. A
        completed run's delivery is decided by its frames
        (`harness.graph.delivery.deliver`) and acknowledged once the answer
        is in the session (`acknowledge_deliveries`)."""
        state = tool_context.state
        if not isinstance(outcome, Accepted):
            return _stopped(outcome)
        assert outcome.run_id is not None, "an accepted run is one the ledger admitted"
        async with store_faults(outcome.run_id):
            run = await direct.run_ledger.get(outcome.run_id)
        assert run is not None, "an accepted run is one the ledger admitted"
        if run.status is RunStatus.COMPLETED:
            await link_accepted_run(
                direct.tool_surface, outcome, workflow=compiled.workflow, ledger=direct.run_ledger
            )
        status = plan_status_of(run.status)
        record_plan(state, compiled.plan.model_copy(update={"status": status}), run_id=run.run_id)
        if run.status is not RunStatus.COMPLETED:
            return _unfinished(run)
        hand_back = HandBack(
            executor,
            summaries=[step.digest for step in presentable_steps(outcome) if step.digest],
            totals=answer_notes(direct.tool_surface, outcome, plan=compiled.plan),
        )
        tools = {step.id: step.tool.name for step in compiled.workflow.steps}
        answered_by = [
            {
                "step": step_id,
                **await hand_back.describe(
                    frame.presented.model_dump(),
                    tool=tools[step_id],
                    store_ref=frame.store.model_dump(),
                ),
            }
            for step_id, frame in answer_frames(outcome).items()
        ]
        listed = hand_back.listing(state, [{"ask": task, "answered_by": answered_by}])
        delivery = await deliver(direct.run_ledger, run.run_id, read=hand_back.read)
        _ACKNOWLEDGEMENTS.setdefault(tool_context.invocation_id, []).append(delivery.acknowledge)
        standing = {**run_hand_back(run), **delivery.standing}
        if hand_back.nothing_chartable:
            return {
                "outcome": DirectOutcome.NOT_CHARTABLE.value,
                "run": standing,
                "message": "The calculation ran, but its result cannot be drawn: tell the user "
                "what it computed, in words, from the step summaries.",
                **listed,
            }
        return {"outcome": DirectOutcome.COMPLETED.value, "run": standing, **listed}

    async def submit_analysis(
        recipe: str,
        tool_context: ToolContext,
        spec_id: str | None = None,
        start_year: Year | None = None,
        end_year: Year | None = None,
        change: GrowthChange | None = None,
    ) -> dict[str, Any]:
        """Run a reviewed calculation on a series a search settled on, with no plan.

        `recipe` is "growth": the growth of one series from `start_year` to
        `end_year`, `change` "relative" (per cent) or "absolute" (in the
        series' own unit). `spec_id` is the series, one a search result gave
        as definite or a candidate you fetched. Set every value from the
        user's words and the search results; leave `change` out when the
        user's words do not settle it.

        Returns `outcome`: `completed` with the result's frames, to chart and
        answer from, and the run; `not_chartable` when the result cannot be
        drawn; `pending` or `failed` with the run's standing; `needs_fields`
        with the values to correct and the recipe's contract;
        `needs_decision` with the options to ask the user about;
        `requires_planning` when the question needs a plan, with the reason.
        This call and plan share one use per turn.
        """
        chosen = enabled.get(recipe)
        state = tool_context.state
        gated = _gate(state, chosen, recipe)
        if gated is not None:
            return gated.payload()
        assert chosen is not None
        parameters = chosen.parameters.model_validate(
            {"spec_id": spec_id, "start_year": start_year, "end_year": end_year, "change": change}
        )
        found = (
            source_evidence(state, parameters.spec_id) if parameters.spec_id is not None else None
        )
        built = chosen.build(parameters, found[0] if found is not None else None)
        if isinstance(built, DirectRefusal):
            return built.payload(chosen)
        assert found is not None, "a recipe builds only on its source's evidence"
        evidence, pin = found
        elected = _elected(state, tool_context)
        undecided = _measure_refusal(state, evidence.spec_id, pin, elected=elected)
        if undecided is not None:
            return undecided.payload()
        task = chosen.task(parameters, evidence)
        # Every check above is synchronous, so taking the call here, before
        # the first await, leaves a concurrent call of the same model
        # response to find it spent.
        count_call(state, "plan")
        kept = False
        try:
            compiled = await compile_direct(
                question=task,
                requirements=built,
                pins={evidence.spec_id: pin},
                surface=direct.translation_surface,
                proposer=_PLANNING,
                elected=elected,
            )
            if not isinstance(compiled, Compiled):
                logger.info("submit_analysis: %s did not compile: %s", recipe, compiled)
                return requires_planning(
                    PlanningCode.NOT_COMPILED,
                    "This calculation could not be set up directly: call plan with the question.",
                    translation=_verdict_codes(compiled),
                ).payload()
            if len(compiled.workflow.steps) > chosen.max_steps:
                return requires_planning(
                    PlanningCode.BEYOND_LIMITS,
                    "This calculation needs more steps than the recipe allows: call plan with "
                    "the question.",
                ).payload()
            try:
                outcome = await run(chosen, compiled, task=task, tool_context=tool_context)
                kept = _keeps_plan_call(outcome)
                answered = await answer(outcome, compiled, task=task, tool_context=tool_context)
            except LedgerUnavailableError as exc:
                if exc.run_id is None:
                    # The store could not admit the run: nothing ran, and the
                    # call is given back.
                    answered = _not_started()
                else:
                    # The run was admitted and is unfinished, so it keeps the call.
                    kept = True
                    record_plan(
                        state,
                        compiled.plan.model_copy(update={"status": PlanStatus.EXECUTING}),
                        run_id=exc.run_id,
                    )
                    answered = _unrecorded(exc.run_id)
            except RunAbandonedError as exc:
                # The run was admitted and is recorded failed, so it keeps the call.
                logger.exception("submit_analysis: run %s was abandoned", exc.run_id)
                kept = True
                answered = _abandoned(exc.run_id)
            return answered
        finally:
            if not kept:
                count_call(state, "plan", -1)

    submit_analysis.__name__ = TOOL_NAME
    return [submit_analysis]


__all__ = [
    "DIRECT_AUTHORISATION_TTL",
    "TOOL_NAME",
    "DirectRecipes",
    "direct_recipe_tools",
    "source_evidence",
]
