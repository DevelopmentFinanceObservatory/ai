"""The `execute` node: runs the approved plan's compiled workflow and maps its
outcome onto the node's `ok` and `replan` edges.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Callable,
    Collection,
    Coroutine,
    Mapping,
)
from dataclasses import dataclass
from typing import Any, cast

from compiler.evidence import errors
from compiler.surface import ProgressReport, ToolSurface
from executor.classify import (
    Disposition,
    ToolFailure,
    candidates_of,
    classify_refusal,
)
from executor.outcomes import (
    Accepted,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
)
from executor.run import (
    FetchSeriesStats,
    StepProgressCallback,
    execute_approved_plan,
)
from google.adk import Context, Event
from planner.plan import Plan as PlannerPlan

from harness import plan_display
from harness.graph.answer import (
    answer_frames,
    answer_notes,
    link_evidence_to_terminal_frames,
)
from harness.graph.state import (
    ORCHESTRATOR_KEPT_ASKS_KEY,
    STEP_PROGRESS_KEY,
    readable_plan,
)

# Execution-triggered replans per turn. Each one is a new plan card and a new
# approval; past this the turn closes and names the step that could not run.
MAX_REPLANS = 1


def _replan_refusal(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """The refusal payload the planner node reads on a "replan" route:
    each outcome kind's own identity plus its diagnosis, never a numeric step
    index -- the compiled workflow steps are named, not ordinal. `also_stopped`
    names the same wave's other stops (`outcome.also_stopped`), each in this
    same per-kind shape, empty when the wave surfaced only this one."""
    return {
        **_refusal_shape(outcome),
        "also_stopped": [_refusal_shape(sibling) for sibling in outcome.also_stopped],
    }


def _is_dead_end(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> bool:
    """Whether `outcome` is a tool's refusal `classify_refusal` calls `Disposition.STOP`."""
    return (
        isinstance(outcome, Refused)
        and outcome.payload is not None
        and classify_refusal(outcome.payload) is Disposition.STOP
    )


def _refusal_shape(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """One outcome's own identity and diagnosis, with no `also_stopped` of its own."""
    if isinstance(outcome, MethodChoiceNeeded):
        return {
            "kind": outcome.kind,
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.code,
            "candidates": outcome.candidates,
            "refusal": outcome.refusal.model_dump(mode="json"),
        }
    if isinstance(outcome, RediscoveryNeeded):
        return {
            "kind": outcome.kind,
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "requirement_id": outcome.requirement_id,
            "spec_id": outcome.spec_id,
            # Whatever the refusal itself offered, verbatim -- same helper
            # `MethodChoiceNeeded.candidates` is minted from at
            # `executor/run.py`. A rediscovery with nothing attached sends
            # the planner back to search with nothing to choose from.
            "candidates": candidates_of(outcome.refusal),
            "refusal": outcome.refusal.model_dump(mode="json"),
        }
    return {
        "kind": outcome.kind,
        "refusal": outcome.refusal.value,
        "detail": outcome.detail,
        "step_id": outcome.step_id,
        "requirement_id": outcome.requirement_id,
        "issues": [issue.model_dump(mode="json") for issue in outcome.issues],
    }


@dataclass(frozen=True)
class _StepReport:
    """One progress report, tagged with the step it describes."""

    step_id: str
    report: ProgressReport


class _ReportRelay:
    """Holds the latest progress report per step, for one reader to drain.

    `put` replaces any report still pending for the same step rather than
    queueing another one, so a producer faster than the reader grows this
    at most one entry per distinct step, never per report -- only the
    latest report says where a step is now. `latest` wakes whenever there
    is something pending, yields it as one batch, and returns once
    `finish` has been called and nothing is left to deliver.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _StepReport] = {}
        self._ready = asyncio.Event()
        self._finished = False

    def put(self, step_id: str, progress: ProgressReport) -> None:
        self._pending[step_id] = _StepReport(step_id=step_id, report=progress)
        self._ready.set()

    def finish(self) -> None:
        self._finished = True
        self._ready.set()

    async def latest(self) -> AsyncIterator[_StepReport]:
        while True:
            await self._ready.wait()
            self._ready.clear()
            pending, self._pending = self._pending, {}
            for item in pending.values():
                yield item
            if self._finished and not self._pending:
                return


def _start_relayed[T](
    call: Callable[[StepProgressCallback], Coroutine[Any, Any, T]],
) -> tuple[asyncio.Task[T], _ReportRelay]:
    """Run `call` as its own task, relaying its progress reports.

    A node awaiting a long call directly could yield nothing until the call
    returns; run as a task, the call reports while the node relays
    (`_ReportRelay.latest`). The relay is told the task finished, however it
    finishes, so `latest` always returns. The caller awaits the task for its
    result and stops it (`_stop`) if the node exits first.
    """
    relay = _ReportRelay()
    task = asyncio.create_task(call(relay.put))
    task.add_done_callback(lambda _: relay.finish())
    return task, relay


async def _stop(task: asyncio.Task[Any]) -> None:
    """Cancel `task` if it is still running, and wait until it has stopped."""
    if not task.done():
        task.cancel()
        await asyncio.wait({task})


def _step_progress_event(report: _StepReport, *, tool: str) -> Event:
    """A relayed progress report as its own event, carried on `custom_metadata`.

    Never on `output`: a node run may set `output` once (`_node_runner.py`'s
    `ctx.output`), and the node's result needs it.
    """
    return Event(
        custom_metadata={
            STEP_PROGRESS_KEY: {
                "step_id": report.step_id,
                "tool": tool,
                "message": report.report.message,
            }
        }
    )


def _mark_masked_step_failed(
    tool_surface: ToolSurface | None,
    outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused,
    *,
    tools: Mapping[str, str],
) -> None:
    """Correct the activity record of every step a wave's stop touches --
    the primary outcome and each sibling in `outcome.also_stopped` -- when
    its raw call succeeded but the executor's own post-call validation
    (shape, evidence, frame) refused it afterward, and record one for a
    step whose budget ran out before its call returned. `tools` names each
    plan step's tool, by step id, for that record.

    `harness.studio_translate._RecordingSurface.call_tool` records a step's
    call the moment it returns, before `executor.run._Run` can run those
    checks on it, so a masked failure would otherwise read "complete" on the
    Activity feed forever, and a call cut off mid-flight would leave its
    running line open. A wave can surface more than one post-call
    refusal at once (`executor.run._Run.execute`), and each sibling in
    `also_stopped` carries its own `step_id` and gets its own record
    corrected. `mark_step_failed` is looked up via `getattr` (`tool_surface`
    is a bare `ToolSurface` in tests with no recorder around it, and a step
    whose own call already failed is a no-op there), the same duck-typed
    opt-in `_RecordingExecutor.aclose` uses. Each stop's own `_failed_step`
    supplies the `code`/`message` the Activity feed's refusal shows, naming
    the actual check that fired (e.g. `source_pin_mismatch`).
    """
    mark = getattr(tool_surface, "mark_step_failed", None)
    if mark is None:
        return
    for stopped in (outcome, *outcome.also_stopped):
        if stopped.step_id is None:
            continue
        failed = _failed_step(stopped)
        mark(
            step_id=stopped.step_id,
            tool=tools.get(stopped.step_id, ""),
            code=failed["code"],
            detail=failed["message"],
        )


def _computed_asks(plan: PlannerPlan, answers: Collection[str]) -> list[int]:
    """The number of each ask `plan` answers whose answering steps all
    finished as terminal steps (`answers`, `answer_frames`'s keys)."""
    return [
        number
        for number, row in enumerate(plan_display.ask_rows(plan))
        if row["answered_by"]
        and all(step in answers for step in cast("list[str]", row["answered_by"]))
    ]


def _kept_asks(
    kept: Mapping[str, Any] | None,
    tool_surface: ToolSurface | None,
    outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused,
    *,
    plan: PlannerPlan,
) -> dict[str, Any]:
    """`kept` with each ask `outcome`'s run computed (`_computed_asks`) set to
    that run's answer frames for it: per answering step its id, tool, frame
    ref and digest, the notes on how those frames were computed
    (`answer_notes`: family totals and keys left out), and for an ask
    `plan` answers in part, the part not computed with its reason, as
    `_ask_status` reports it. A plan call's asks are fixed (`plan_start`), so
    an ask's number names the same ask in every plan the call runs."""
    answers = answer_frames(outcome)
    records = {record.step_id: record for record in outcome.completed}
    rows = plan_display.ask_rows(plan)
    kept = dict(kept or {})
    for number in _computed_asks(plan, answers):
        row = rows[number]
        steps = cast("list[str]", row["answered_by"])
        kept[str(number)] = {
            "steps": [
                {
                    "step_id": step,
                    "tool": records[step].tool.name,
                    "frame_ref": answers[step].model_dump(),
                    "digest": records[step].digest,
                }
                for step in steps
            ],
            "totals": answer_notes(tool_surface, outcome, plan=plan, terminals=steps),
        }
        if row["declined"] is not None:
            kept[str(number)]["not_computed"] = {
                "part": row["not_computed"],
                "reason": row["reason"],
            }
    return kept


def _failed_step(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """`execute`'s `failed_step`, once the replan budget is spent."""
    if isinstance(outcome, MethodChoiceNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.code,
            "message": outcome.refusal.message,
        }
    if isinstance(outcome, RediscoveryNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": "rediscovery_needed",
            "message": outcome.refusal.message
            or f"the source for requirement {outcome.requirement_id!r} needs discovering again",
        }
    # `outcome.detail` wraps the tool's own text ("'aggregate' refused:
    # ..."), but only when the call left one -- with no `result.text`,
    # `_refused` falls back to `failure.code` instead of `failure.message`
    # (`executor.run._Run._stopped_by_refusal`), so `detail` is not a
    # reliable stand-in for what the tool actually said. `payload`, set on
    # every `STEP_REFUSED` stop, carries the tool's own `code` and
    # `message` straight from its `ToolFailure`; a `Refused` raised for any
    # other reason (`EMPTY_ANSWER`, `SHAPE_MISMATCH`, ...) has no tool call
    # to report and keeps today's `refusal.value`/`detail` fallback -- unless
    # it carries `issues` (`executor.run._Run._check_evidence`'s
    # `REQUIREMENT_UNMET`, today's only source of them): `outcome.detail`
    # there is the generic "requirement X is not met by what Y returned",
    # naming neither which of `check_evidence`'s ten sub-checks fired nor
    # what it found versus what the plan required, so the specific
    # `EvidenceIssueCode` and its own expected/actual detail lead instead.
    if outcome.payload is not None:
        return _refused_step(step_id=outcome.step_id, failure=outcome.payload)
    if outcome.issues:
        failing = errors(list(outcome.issues)) or list(outcome.issues)
        return {
            "step_id": outcome.step_id,
            "tool": None,
            "code": failing[0].code.value,
            "message": "; ".join(issue.detail for issue in failing),
        }
    return {
        "step_id": outcome.step_id,
        "tool": None,
        "code": outcome.refusal.value,
        "message": outcome.detail,
    }


def _refused_step(*, step_id: str | None, failure: ToolFailure) -> dict[str, Any]:
    """`execute`'s `failed_step` for a step the tool itself refused.

    A tool's `fix` can carry its own `remedy`: one sentence written for the
    end user (`data_mcp.tool_errors`: an `over_ceiling` on
    `member_cardinality`, and a mint dfo-apis refuses `member_not_resolvable`),
    not the code/detail prose the rest of `fix` holds for a retrying caller.
    `plan_done` hands it to the orchestrator under `failed_step`.
    """
    failed: dict[str, Any] = {
        "step_id": step_id,
        "tool": None,
        "code": failure.code,
        "message": failure.message,
    }
    remedy = failure.fix.get("remedy")
    if isinstance(remedy, str) and remedy:
        failed["remedy"] = remedy
    return failed


def execute_node(
    *,
    tool_surface: ToolSurface | None,
    max_concurrent_steps: int,
    fetch_series_stats: FetchSeriesStats,
) -> Callable[[Context, Any], AsyncIterator[Any]]:
    """The `execute` node, running the approved plan against `tool_surface`."""

    async def execute(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """Run the approved plan's compiled workflow through the tool surface.

        `execute_approved_plan` owns the run: per-step retry, the
        per-capability transport policy, and the checks that turn a step's
        own declared shape into a refusal. This node only reads what came
        back and maps it onto the graph's two outgoing edges.
        An `Accepted` run first aliases each terminal frame's provenance to
        the retrieval(s) that actually feed it, per frame
        (`link_evidence_to_terminal_frames`), then routes "ok" with the
        terminal steps' frame refs (`workflow.terminal_steps`).
        `MethodChoiceNeeded` and `RediscoveryNeeded` route "replan" while the budget allows and "ok"
        with `failed_step` once it is spent, the same as any other
        `Refused`. A stopped run's answer frames (`answer_frames`) have
        their provenance linked as an accepted run's are, and the asks they
        compute are kept for the plan call (`_kept_asks`), so `plan_done`
        can list them whichever later run or planner refusal ends the call.
        The "ok" route carries those frames as final; the "replan" refusal
        names the asks they compute (`computed_asks`), since the new plan
        runs from the start.
        The exception is a plain `Refused` whose own `payload` is a
        `ToolFailure` `executor.classify.classify_refusal` calls
        `Disposition.STOP`: that is a dead end no replan reaches, so the
        turn closes right there with the refusal's own remedy, no replan
        spent, rather than spending the budget reproducing the identical
        failing call. `Refused(NOT_APPROVED)` and
        `Refused(STALE_REVISION)` mean the workflow's own `approve` node let
        through a plan it should not have -- a replan here would loop on the
        same defect, so those raise instead. Any other outcome also corrects
        that step's own activity record, and every sibling's in
        `outcome.also_stopped`, in case a raw call had already succeeded
        before a post-call check refused it (`_mark_masked_step_failed`).

        The run goes on as its own task (`_start_relayed`) while this node
        relays each step's progress reports as their own events
        (`_step_progress_event`), one per report as it arrives. The task is
        awaited last, so its exceptions propagate here, and cancelled if the
        node exits first (a stop or a disconnect closing the stream).

        A plan stored in a shape this release does not read
        (`readable_plan`) runs nothing and spends no replan: it routes
        `"replan"` with `{"unreadable_plan": True}`, and `planner` drafts the
        question from the start.
        """
        del node_input
        if tool_surface is None:
            raise RuntimeError(
                "the execute node has no tool surface: create_root_workflow(tool_surface=...) "
                "was not given one, so the approved plan cannot be run"
            )
        plan = readable_plan(ctx, ctx.state["plan"])
        if plan is None:
            # Nothing to run, and no replan is spent on it.
            yield Event(output={"unreadable_plan": True}, route="replan")
            return
        tools = {step.id: step.tool.name for step in plan.workflow.steps} if plan.workflow else {}
        task, relay = _start_relayed(
            lambda on_progress: execute_approved_plan(
                plan,
                surface=tool_surface,
                max_concurrent_steps=max_concurrent_steps,
                fetch_series_stats=fetch_series_stats,
                on_progress=on_progress,
            )
        )
        try:
            async for report in relay.latest():
                yield _step_progress_event(report, tool=tools.get(report.step_id, ""))
            finished, outcome = await task
        finally:
            await _stop(task)
        ctx.state["plan"] = finished.model_dump_json()
        workflow = plan.workflow
        assert workflow is not None, "an approved plan always carries a workflow"
        if isinstance(outcome, Accepted):
            link_evidence_to_terminal_frames(tool_surface, outcome, workflow=workflow)
            step_order = {record.step_id: index for index, record in enumerate(outcome.steps)}
            answers = answer_frames(outcome)
            yield Event(
                output={
                    "frame_refs": [ref.model_dump() for ref in answers.values()],
                    # Parallel to frame_refs above: which plan step (id and
                    # tool) produced each ref, and that it is final.
                    # `outcome.results` is keyed over `workflow.terminal_steps`
                    # (`executor.run._Run.execute`), so every entry here is a
                    # terminal step's; a DAG can have more than one.
                    "frame_ref_steps": [
                        {
                            "step_id": step_id,
                            "tool": outcome.steps[step_order[step_id]].tool.name,
                            "is_final_step": True,
                        }
                        for step_id in answers
                    ],
                    "plan_version": finished.version,
                    "step_summaries": [step.digest for step in outcome.steps if step.digest],
                    "stat_var_totals": answer_notes(tool_surface, outcome, plan=finished),
                },
                route="ok",
            )
            return
        _mark_masked_step_failed(tool_surface, outcome, tools=tools)
        if isinstance(outcome, Refused) and outcome.refusal in (
            RefusalKind.NOT_APPROVED,
            RefusalKind.STALE_REVISION,
        ):
            raise RuntimeError(
                "the approve node let a plan reach execute that execute_approved_plan refused "
                f"as {outcome.refusal.value}: {outcome.detail}"
            )
        # A dead end: no replan can fix this, so none is spent, and the
        # planner's re-proposal (which would reproduce this exact failing
        # call) never runs.
        dead_end = _is_dead_end(outcome)
        answers = answer_frames(outcome)
        link_evidence_to_terminal_frames(tool_surface, outcome, workflow=workflow)
        if finished.asks:
            ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = _kept_asks(
                ctx.state.get(ORCHESTRATOR_KEPT_ASKS_KEY), tool_surface, outcome, plan=finished
            )
        replans = int(ctx.state.get("replans", 0))
        if not dead_end and replans < MAX_REPLANS:
            ctx.state["replans"] = replans + 1
            refusal = _replan_refusal(outcome)
            if finished.asks:
                refusal["computed_asks"] = _computed_asks(finished, answers)
            output: dict[str, Any] = {"refusal": refusal, "plan_version": finished.version}
            yield Event(output=output, route="replan")
            return
        completed = outcome.completed
        yield Event(
            output={
                "frame_refs": [step.presented_ref.model_dump() for step in completed],
                # Parallel to frame_refs above: which plan step (id and tool)
                # produced each ref, and whether it is final. A terminal step
                # that finished before the stop is final (`answer_frames`):
                # its frame answers the asks it serves, so `plan_done` takes it
                # as it would on an accepted run. Any other step's frame is an
                # intermediate, which `plan_done` does not list.
                "frame_ref_steps": [
                    {
                        "step_id": step.step_id,
                        "tool": step.tool.name,
                        "is_final_step": step.step_id in answers,
                    }
                    for step in completed
                ],
                "plan_version": finished.version,
                "step_summaries": [step.digest for step in completed if step.digest],
                "stat_var_totals": answer_notes(tool_surface, outcome, plan=finished),
                "failed_step": _failed_step(outcome),
            },
            route="ok",
        )

    return execute
