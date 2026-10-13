"""The `planner`, `approve` and `plan_start` nodes: the agent's brief handed to
the planner, the plan it drafts, and the approval gate.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
)
from datetime import datetime
from typing import Any

from common.telemetry import record_span
from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow._errors import NodeInterruptedError
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_STATE_KEY,
    PlanStatus,
    plan_revision,
)
from planner.plan import Plan as PlannerPlan

from harness.graph.seams import (
    PlannerModel,
)
from harness.graph.state import (
    APPROVE_INTERRUPT,
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    PLANNER_STARTED_KEY,
    UNREADABLE_PLAN_NOTE,
    UNREADABLE_PLAN_UNPLANNED_NOTE,
    interrupt_event,
    readable_plan,
    state_json,
)

logger = logging.getLogger(__name__)


def planner_node(*, planner: PlannerModel) -> Callable[[Context, Any], AsyncIterator[Any]]:
    """The `planner` node, drafting with `planner`."""

    async def draft_plan(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """The planner node.

        A `PlannerModel` seam drafts a compiled `planner.plan.Plan` from the
        conversation's search readings and this node only validates it
        (`PlannerModel`'s contract): versioning, `previous_version` lineage and the
        DRAFT/PENDING_APPROVAL transitions are enforced by `planner.plan.Plan`
        itself, inside the seam, never here.

        A draft can run long (the planner loads skills, proposes,
        translates, and corrects) with nothing to show for it until
        this returns. So this node queues each `on_progress` report as it
        arrives and relays the queued reports as their own `Event`s once the
        await settles (see the direct-await discussion below). The `execute`
        node relays its reports live instead, because its run never calls
        `ctx.run_node` and so can run as a task of its own. Carried on
        `custom_metadata`, never `output`: `ctx.output` (set
        by any yielded event whose `output` is not `None`,
        `_node_runner.py`'s `_track_event_in_context`) accepts exactly one
        write per node run and raises on a second, so only the final yield,
        the drafted plan (or the refusal, below), may use it.
        `custom_metadata` is untouched by that rule, so every progress event
        passes straight through as a plain activity update and
        `studio_translate._Translator.translate` reads
        `event.custom_metadata["progress"]` instead of `event.output` to
        tell the two apart.

        Three routes out, all explicit on the final yield: `"ok"`, to
        the `approve` node, for a drafted plan; `"pending_refusal"`, also to
        the `approve` node, for a seam that refuses to draft one at all -- either
        `PlannerSeam`'s own `MAX_PLAN_ROUNDS` cap, or `starts_new_plan`
        refused as `QUESTION_FIXED` because this seam's pair is built with
        the question fixed (`harness.planner_seam.PlannerSeam.__init__`) --
        surfaced as `{"refused": {"text": ...}}` rather than a raised
        `PlannerSeamError` -- while `ctx.state["plan"]` still holds a plan
        `PENDING_APPROVAL`; and `"refused"`, to `plan_done`, for the same
        two refusals when it does not. Raising `PlannerSeamError` here
        instead would end the turn through `_stream_fatal`'s own
        `close_turn(error=...)`, which `studio_state.begin_turn` documents
        as abandoning the paused session and its plan -- exactly the plan
        a refusal that still has one pending needs the user to be able to
        act on. `ctx.state["plan"]` is never written on either refusal
        route, so it is unchanged either way; what differs is only whether
        that unchanged record is worth routing back to the `approve` node for.
        `"refused"` covers a refusal on a first draft (nothing was ever
        submitted this turn) and on a replan (the record on file is already
        `APPROVED`, not something a fresh approval interrupt could act on).
        A seam that raises fails the turn.
        On the `"refused"` route `plan_done` lists the asks an earlier run
        of the plan call computed (`ORCHESTRATOR_KEPT_ASKS_KEY`).

        An amendment or a replan of a plan stored in a shape this release
        does not read (`readable_plan`) is drafted as a first draft, and
        the `"ok"` route's plan carries `UNREADABLE_PLAN_NOTE` as
        `refusal_note`, which `harness.studio_translate` shows before the
        plan card, and `"restarted": True`, since its version counts from 1
        again. The `"refused"` route's text opens with
        `UNREADABLE_PLAN_UNPLANNED_NOTE`.

        `planner(...)` is awaited directly here, with no
        `asyncio.ensure_future` wrapper of its own: ADK's own docstring
        warns against wrapping a `ctx.run_node`-calling call in a task like
        that, and this node has two reasons that matters. A real
        `PlannerModel` seam (`harness.planner_seam.PlannerSeam`) calls
        `ctx.run_node` from inside it, and that call can genuinely pause
        mid-turn -- a planner-side ask (`planner.agent`'s
        `adk_request_input` tool) suspends the whole run by raising
        `NodeInterruptedError` straight out of the awaited call, a
        `BaseException` `ctx.run_node` uses exclusively for this
        (`google.adk.workflow._errors`); wrapped in a `Task`, that exception
        would sit on the `Task` instead of propagating up through this
        node's own coroutine the way its `rerun_on_resume=True` re-entry
        depends on. The direct await also closes a real turn more simply:
        `studio.py`'s `_sse_stream` -> `_unwind` cancels the event stream on
        a stop or a disconnect, throwing into whichever call is suspended --
        with no separate `Task` to orphan, the throw propagates straight
        through `ctx.run_node` on its own, with no explicit `.cancel()`
        needed here. The cost is `on_progress`: a report queues rather than
        relaying as its own `Event` the instant it arrives, since yielding
        one while a nested plain `await` is still pending would need a
        second concurrent task to rendezvous with, which this design
        avoids creating. Reports queue as they happen and relay the moment
        the `await` returns. A pause drops them instead: ADK streams the
        ask before the pause reaches this node, so anything relayed then
        would follow the ask.
        """
        raw = ctx.state.get("plan")
        question = ctx.state.get(ORCHESTRATOR_BRIEF_KEY, "")
        payload = node_input if isinstance(node_input, dict) else {}
        refusal = payload.get("refusal")
        amendment_answer = payload.get("answer")
        amendment = (
            amendment_answer.get("amendment") if isinstance(amendment_answer, dict) else None
        )
        # A first-draft entry -- `plan_start`'s, the only one of the three
        # that lands here with neither a `"plan"` key (`approve`'s
        # own "amend" route, whether the answer carries amendment text or
        # merely names a stale revision -- `_resume`d with an
        # unrecognised one, say) nor a `"refusal"` key (`execute`'s own
        # "replan" route) -- passes no `prior_plan`, even when
        # `ctx.state["plan"]` already holds one. Keyed on which of those the
        # payload carries, not on whether `refusal`/`amendment` themselves
        # are `None`: a stale-revision "amend" carries no amendment text
        # either, and reading that shape as a first draft would wipe out a
        # plan still `PENDING_APPROVAL` right before the `approve` node re-offers
        # it. A later `plan` call in the same turn finds the earlier call's
        # plan on record -- withholding it here is what lets
        # `harness.planner_seam.PlannerSeam.__call__` tell this is a genuine
        # first draft and clear its own `planner:plan` and briefed-flag
        # state for it (see that module's docstring), rather than reading
        # as a continuing call on an already-briefed session that would
        # need `starts_new_plan` -- refused, this seam's pair being built
        # with the question fixed -- to draft anything but an amendment to
        # that stale plan.
        existing = readable_plan(ctx, raw)
        is_first_draft = "plan" not in payload and "refusal" not in payload
        # The plan this entry amends or replans could not be read: found so
        # by `approve` or `execute`, which cleared it, or here, on the resume
        # of a question asked while amending it. Nothing is left to amend or
        # to replan from, so the question is drafted from the start and the
        # card says why. A first draft never read the plan on record, and
        # says nothing of one it cannot read.
        unreadable = bool(payload.get("unreadable_plan")) or bool(
            not is_first_draft and raw and existing is None
        )
        if unreadable:
            refusal = amendment = None
            is_first_draft = True
        yield Event(custom_metadata={PLANNER_STARTED_KEY: {"revising": not is_first_draft}})

        progress: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        try:
            drafted = await planner(
                ctx=ctx,
                question=question,
                readings=[
                    *(ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("readings", []),
                    *(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or []),
                ],
                prior_plan=(
                    None
                    if is_first_draft
                    else (existing.model_dump(mode="json") if existing else None)
                ),
                refusal=refusal,
                amendment=amendment,
                canvas_digest=state_json(ctx, "canvas_digest"),
                conversation_digest=state_json(ctx, "conversation_digest"),
                on_progress=progress.put_nowait,
            )
        except NodeInterruptedError:
            # A planner-side ask paused the draft mid-turn, and ADK has
            # already streamed the ask, so the reports queued so far are
            # dropped with it: relayed now they would follow the ask.
            raise
        while not progress.empty():
            yield Event(custom_metadata={"progress": progress.get_nowait()})
        refused = drafted.get("refused") if isinstance(drafted, dict) else None
        if refused is not None:
            text = str(refused.get("text", ""))
            if existing is not None and existing.status is PlanStatus.PENDING_APPROVAL:
                # The plan the user was already looking at is untouched
                # (see the docstring: this refusal never writes
                # `ctx.state["plan"]`) and still `PENDING_APPROVAL`, so it
                # is a real thing to re-offer, not a stale claim -- route
                # back to the `approve` node with that same plan, carrying the
                # refusal's own sentence so the user reads why their
                # change did not land before seeing the unchanged plan
                # again.
                note = (
                    f"{text} Version {existing.version} is still there to "
                    "approve; anything past that needs a new question."
                )
                yield Event(
                    output={
                        **existing.model_dump(mode="json"),
                        "summary": note,
                        "revision": plan_revision(existing),
                        "refusal_note": note,
                    },
                    route="pending_refusal",
                )
                return
            if unreadable:
                text = f"{UNREADABLE_PLAN_UNPLANNED_NOTE} {text}".rstrip()
            yield Event(output={"text": text, "final": bool(refused.get("final"))}, route="refused")
            return
        planner_plan = PlannerPlan.model_validate(drafted["plan"])
        ctx.state["plan"] = planner_plan.model_dump_json()
        yield Event(
            output={
                **planner_plan.model_dump(mode="json"),
                "summary": drafted["summary"],
                "revision": plan_revision(planner_plan),
                **({"refusal_note": UNREADABLE_PLAN_NOTE} if unreadable else {}),
                # The draft's version counts from 1 again, on a line of its
                # own: no earlier card of the turn is a version of this plan.
                **({"restarted": True} if unreadable else {}),
            },
            route="ok",
        )

    return draft_plan


def approve(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
    """The approval gate, as a graph interrupt. Revision-bound.

    This is the only gate: nothing but the `execute` node reaches the
    executor.

    `ctx.state["plan"]` always holds a `planner.plan.Plan`, since every
    `PlannerModel` seam drafts one. A plan is keyed on `plan_revision(plan)`,
    a content hash that changes with every amendment. A plan stored in a
    shape this release does not read (`readable_plan`) has no revision
    for the answer to name: it routes `"amend"` with
    `{"unreadable_plan": True}`, and `planner` drafts the question from
    the start.

    An answer approves when it names the revision and says, under `at`, when
    the user gave it, as an ISO 8601 time with its offset: the studio's
    interrupt route sends the time it recorded the first answer to this
    interrupt (`harness.approval.ApprovalRecord.at`), so a retry resuming
    the same answer later approves as of the same moment. The `"approved"`
    route stamps that time on the plan (`Plan.approved_at`), from which the
    approval's lifetime runs. An answer with no such time approves nothing
    and routes `"amend"` as one naming another revision does. It
    records a `plan.approved` span (nesting under the turn span
    `harness.studio_state.begin_turn` opens) and a matching log line with
    the same facts, `plan_version` and `amended`, off the `Plan` just
    approved.
    """
    answer = (ctx.resume_inputs or {}).get(APPROVE_INTERRUPT)
    if not (ctx.state.get("awaiting_approve") and answer is not None):
        ctx.state["awaiting_approve"] = True
        yield interrupt_event(
            RequestInput(
                interrupt_id=APPROVE_INTERRUPT,
                message="Approve this plan before anything is retrieved?",
                payload=node_input,
            )
        )
        return
    ctx.state["awaiting_approve"] = False
    approved_answer = answer.get("approved") if isinstance(answer, dict) else None
    plan = readable_plan(ctx, ctx.state["plan"])
    if plan is None:
        # The card the user answered names a plan this release does not
        # read, so the answer approves nothing and amends nothing.
        yield Event(output={"unreadable_plan": True}, route="amend")
        return
    key: Any = plan_revision(plan)
    approved_at = _answered_at(answer)
    if approved_answer == key and approved_at is not None:
        approved = plan.approve(at=approved_at)
        ctx.state["plan"] = approved.model_dump_json()
        plan_version = approved.version
        amended = plan_version > 1
        record_span("plan.approved", {"dfo.plan_version": plan_version, "dfo.amended": amended})
        logger.info("plan approved: plan_version=%s amended=%s", plan_version, amended)
        yield Event(output=approved.model_dump(mode="json"), route="approved")
        return
    yield Event(
        output={"plan": plan.model_dump(mode="json"), "answer": answer},
        route="amend",
    )


def _answered_at(answer: object) -> datetime | None:
    """When an approve answer says the user gave it, or `None` when it states
    no timezone-aware ISO 8601 time under `at`."""
    raw = answer.get("at") if isinstance(answer, dict) else None
    if not isinstance(raw, str):
        return None
    try:
        at = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return at if at.tzinfo is not None else None


def plan_start(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
    """Hand the agent's brief to the planner under the orchestrator's own
    key. This call's asks go under `ASKS_STATE_KEY` and the parties they
    break down by under `ASK_BREAKDOWNS_STATE_KEY`, each cleared when it
    lists none so no plan is held to an earlier call's asks, and no
    earlier call's kept asks are listed. The planner's first-draft entry
    is an empty payload."""
    ctx.state[ORCHESTRATOR_BRIEF_KEY] = node_input["brief"]
    ctx.state[ASKS_STATE_KEY] = node_input.get("asks") or None
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = node_input.get("ask_breakdowns") or None
    ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
    return {}
