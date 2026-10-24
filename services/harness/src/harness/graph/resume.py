"""Carry on the runs an earlier turn of the conversation left unfinished.

A run is left `pending` when publishing its results hit a fault worth
retrying, and `admitted` or `running` when the turn that ran it ended before
its computation did. The next turn of the same conversation carries each one
on before the agent runs (`executor.run.resume_run`): a pending run resumes
publication, an unfinished computation runs its stored workflow again, the
tools answering what the store already holds. A completed run's answer
frames (`harness.graph.answer.answer_frames`) are linked to the retrievals
under them, described and listed through `plan_done`'s own `HandBack`, as a
run that completes in its own turn is, so the agent can quote and chart them;
a frame `describe_frame` refuses is listed under `frames_not_chartable`. The
notes on how its figures were computed (`totals`,
`harness.graph.answer.answer_notes`) are read from the plan the ledger stored
for the run's work (`stored_run_plan`); a run stored without one lists none.
Its lineage is first restored from the `get_data` results it sealed its
inputs from (`harness.graph.answer.link_accepted_run`), so a card drawn on a
frame of a run carried on after a restart keeps its sources. Its plan record
is the one the turn that admitted it wrote.

A completed run's delivery is decided by `harness.graph.delivery.deliver`, as
`plan_done` decides it for a run that completes in its own turn:
`Resumed.acknowledge`, called once the hand-back is in the session, records
each run whose frames were all read `executor.ledger.Delivered`, and a run
with a frame left unread is recorded `executor.ledger.Undelivered`, its
standing carrying `harness.graph.delivery.RESULT_UNDELIVERED`, and is listed again by the next turn
until the delivery bound ends it.

Carrying a run on never ends the turn. A run whose stored submission is
missing or unreadable is recorded failed (`run_record_unreadable`); a fault
the computation raises again is recorded by `executor.run` as it was the
first time; anything the run ledger's store raises is
`executor.ledger.LedgerUnavailableError` (`executor.ledger.store_faults`),
which is logged and leaves the run as it is, for the next turn to carry on
until the resume bound (`executor.ledger.MAX_PUBLICATION_RESUMES`,
`MAX_PENDING_AGE`) ends it; any other exception is logged and the run
recorded failed (`resume_failed`), so no later turn tries it again. The
`seed` node calls this, so its `_turn_ran_before` guard keeps a resumed turn
from doing it twice.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from compiler.surface import ToolSurface
from executor.completion import ResultPublisher, run_hand_back
from executor.ledger import (
    TERMINAL_STATUSES,
    Fail,
    LedgerUnavailableError,
    RunLedger,
    RunRecord,
    RunStatus,
    record_change,
    store_faults,
)
from executor.outcomes import Accepted
from executor.run import (
    FetchSeriesStats,
    LabelRefs,
    UnreadableRunError,
    load_submission,
    resume_run,
)
from google.adk import Context

from harness.graph.answer import (
    answer_frames,
    answer_notes,
    link_accepted_run,
    presentable_steps,
    stored_run_plan,
)
from harness.graph.delivery import Delivery, deliver
from harness.graph.plan_done import HandBack, StepSummary
from harness.graph.seams import Executor
from harness.tool_contracts import HarnessCode

logger = logging.getLogger(__name__)

# What the agent reads when the turn opens on runs an earlier turn left
# unfinished, before their standings.
_RESUMED_LEAD = (
    "Calculations an earlier turn left unfinished were carried on before this turn. A "
    "completed run's frames are listed and can be charted and quoted; for a completed run "
    "with code result_undelivered, say it is computed and its result is listed when the "
    "user next writes, or, when it gives a reason, that its result could not be listed "
    "and why; for a run still pending, say it is pending under its run ID and is "
    "finished when the user next writes; for a failed one, say it could not be completed "
    "and why, in the user's terms. Then answer the user's message."
)


@dataclass(frozen=True)
class Resumed:
    """The runs carried on: `hand_back` is what the agent reads, and
    `acknowledge` records each completed run it delivers `Delivered`. Call
    it only once the conversation has recorded `hand_back`; a run it could
    not record is delivered again the next turn."""

    hand_back: dict[str, str]
    acknowledge: Callable[[], Awaitable[None]]


type ResumeRuns = Callable[[Context], Awaitable[Resumed | None]]


def run_resumer(
    *,
    run_ledger: RunLedger,
    publisher: ResultPublisher | None,
    executor: Executor,
    tool_surface: ToolSurface | None,
    max_concurrent_steps: int,
    fetch_series_stats: FetchSeriesStats,
    label_refs: LabelRefs,
) -> ResumeRuns:
    """Carry on every unfinished run of `run_ledger`, re-running computations
    against `tool_surface` and publishing through `publisher`, and describing
    a completed run's frames through `executor` so the agent can chart them.
    Without a `tool_surface`, no run is carried on."""

    async def resume(ctx: Context) -> Resumed | None:
        """The hand-back naming each run carried on and its standing, or
        `None` when the ledger holds no unfinished run or cannot be read."""
        if tool_surface is None:
            return None
        try:
            runs = await run_ledger.unfinished()
        except Exception:
            logger.exception("resume: the run ledger could not be read; no run is carried on")
            return None
        standings: list[dict[str, Any]] = []
        deliveries: list[Delivery] = []
        for run in runs:
            try:
                standing, delivery = await carry_on(ctx, run, tool_surface)
            except LedgerUnavailableError:
                logger.warning(
                    "resume: run %s left unfinished; the run ledger faulted",
                    run.run_id,
                    exc_info=True,
                )
                continue
            except Exception as exc:
                logger.exception("resume: run %s could not be carried on", run.run_id)
                standing, delivery = await _failed(run_ledger, run.run_id, exc), None
            if standing is not None:
                standings.append(standing)
            if delivery is not None:
                deliveries.append(delivery)
        if not standings:
            return None

        async def acknowledge() -> None:
            for delivery in deliveries:
                await delivery.acknowledge()

        hand_back = {
            "status": "Finishing an earlier calculation",
            "text": f"{_RESUMED_LEAD}\n{json.dumps({'runs': standings})}",
        }
        return Resumed(hand_back=hand_back, acknowledge=acknowledge)

    async def carry_on(
        ctx: Context, run: RunRecord, surface: ToolSurface
    ) -> tuple[dict[str, Any], Delivery | None]:
        """Carry `run` on, and return its standing as the agent reads it and,
        for a completed run, its delivery by that standing."""
        try:
            submission = await load_submission(run_ledger, run.run_id)
        except UnreadableRunError as exc:
            logger.error("resume: %s", exc)
            failed = await record_change(
                run_ledger, run.run_id, Fail(HarnessCode.RUN_RECORD_UNREADABLE.value, str(exc))
            )
            return run_hand_back(failed), None
        outcome = await resume_run(
            run,
            submission,
            ledger=run_ledger,
            surface=surface,
            publisher=publisher,
            max_concurrent_steps=max_concurrent_steps,
            fetch_series_stats=fetch_series_stats,
            label_refs=label_refs,
        )
        async with store_faults(run.run_id):
            finished = await run_ledger.get(run.run_id)
        assert finished is not None
        standing = run_hand_back(finished)
        if finished.status is not RunStatus.COMPLETED or not isinstance(outcome, Accepted):
            return standing, None
        await link_accepted_run(surface, outcome, workflow=submission.workflow, ledger=run_ledger)
        plan = await stored_run_plan(run_ledger, run_id=run.run_id, revision=submission.revision)
        hand_back = HandBack(
            executor,
            summaries=[
                StepSummary.of_record(step) for step in presentable_steps(outcome) if step.digest
            ],
            totals=answer_notes(surface, outcome, plan=plan) if plan is not None else [],
        )
        tools = {record.step_id: record.tool.name for record in outcome.steps}
        read = True
        for step_id, frame in answer_frames(outcome).items():
            try:
                await hand_back.describe(
                    frame.presented.model_dump(),
                    tool=tools[step_id],
                    store_ref=frame.store.model_dump(),
                )
            except Exception:
                logger.exception("resume: describing %s raised", frame.presented.spec_id)
                read = False
        standing.update(hand_back.listing(ctx.state, None))
        delivery = await deliver(run_ledger, run.run_id, read=read and hand_back.read)
        standing.update(delivery.standing)
        return standing, delivery

    return resume


async def _failed(ledger: RunLedger, run_id: str, exc: Exception) -> dict[str, Any] | None:
    """Run `run_id`'s standing once `exc` stopped it being carried on: recorded
    failed (`resume_failed`) unless it already finished. A completed run is
    undelivered (`deliver`), since none of its frames is listed. `None` when
    the ledger cannot be written or read."""
    try:
        run = await ledger.get(run_id)
        if run is not None and run.status not in TERMINAL_STATUSES:
            run = await ledger.record(
                run_id, Fail(HarnessCode.RESUME_FAILED.value, f"{type(exc).__name__}: {exc}")
            )
    except Exception:
        logger.exception("resume: run %s could not be recorded failed", run_id)
        return None
    if run is None:
        return None
    standing = run_hand_back(run)
    if run.status is RunStatus.COMPLETED:
        standing.update((await deliver(ledger, run_id, read=False)).standing)
    return standing
