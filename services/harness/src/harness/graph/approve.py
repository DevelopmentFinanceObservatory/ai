"""The `approve` node: the plan card, and the user's answer to it.

`approve` pauses the turn on the card (`Pause` of kind `"approve"`) and,
once the user answers, routes the answer: `approved` to `execute`, `amend`
to `planner`, `switch` to `switch`. It is the only gate before the
executor: nothing but `execute` runs a plan, and `execute` runs only the
plan `approve` approved.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

from common.telemetry import record_span
from planner.plan import (
    PlanStatus,
    SessionState,
    alignments,
    plan_revision,
    stored_plan,
    windowed_origins,
)

from harness.decision_pick import decision_pick
from harness.early_mint import EarlyMints
from harness.graph.events import GraphEvent
from harness.graph.state import APPROVE_PAUSE, readable_plan
from harness.turn.node import Next, Node, NodeCtx, Pause

logger = logging.getLogger(__name__)

# The line the plan card's pause carries.
APPROVE_MESSAGE = "Approve this plan to run it? Its data may already be loading."


def approve_node(*, early_mints: EarlyMints | None) -> Node[GraphEvent]:
    """The `approve` node: `run` offers the card (`offer`), `resume` decides
    the user's answer to it (`decide`), each card it shows offered to
    `early_mints` when given."""

    async def run(ctx: NodeCtx[GraphEvent], node_input: dict[str, Any]) -> Pause:
        return offer(ctx, node_input, early_mints=early_mints)

    async def resume(
        ctx: NodeCtx[GraphEvent], node_input: dict[str, Any], answer: dict[str, Any]
    ) -> Next:
        del node_input
        return decide(ctx, answer, early_mints=early_mints)

    return Node(run=run, resume=resume)


def offer(
    ctx: NodeCtx[GraphEvent], node_input: dict[str, Any], *, early_mints: EarlyMints | None
) -> Pause:
    """Pause on the plan card `node_input`, the plan the node before drafted
    or revised.

    A card pending approval is offered to `early_mints` (`harness.early_mint`),
    which starts its retrievals unless it already started that revision. The
    offer runs after the pause commits (`NodeCtx.after_commit`), so a node
    that dies before the commit and runs again offers once, and one that dies
    between the commit and the offer offers nothing, and the run then asks
    for its frames itself. The answer reaches `decide`, which makes no offer.
    """
    if early_mints is not None:
        offered = stored_plan(ctx.state.get("plan"), state_key="plan")
        if offered is not None and offered.status is PlanStatus.PENDING_APPROVAL:

            async def start_retrievals() -> None:
                early_mints.offer(offered)

            ctx.after_commit(start_retrievals)
    return Pause(kind=APPROVE_PAUSE, payload=node_input, message=APPROVE_MESSAGE)


def decide(
    ctx: NodeCtx[GraphEvent], answer: dict[str, Any], *, early_mints: EarlyMints | None
) -> Next:
    """The user's `answer` to the plan card, as an edge. Revision-bound.

    `ctx.state["plan"]` always holds a `planner.plan.Plan`, since every
    `PlannerModel` drafts one. A plan is keyed on `plan_revision(plan)`, a
    content hash that changes with every amendment. A plan stored in a shape
    this release does not read (`readable_plan`) has no revision for the
    answer to name: it routes `amend` with `{"unreadable_plan": True}`, and
    `planner` drafts the question from the start.

    An answer approves when it names the revision and says, under `at`, when
    the user gave it, as an ISO 8601 time with its offset: the studio's
    interrupt route sends the time it recorded the first answer to this
    pause (`harness.approval.ApprovalRecord.at`), so a retry resuming the
    same answer later approves as of the same moment. The `approved` edge
    carries the plan stamped with that time (`Plan.approved_at`), from which
    the approval's lifetime runs. An answer with no such time approves
    nothing and routes `amend`, as one naming another revision does.

    An answer carrying `switch` (`{"choice_id", "option_id"}`, a scope choice
    the user moved on the card) routes `switch` to the `switch` node
    (`harness.graph.switch`), which offers the revised plan back here. A
    switch of a planner decision's row (`harness.decision_pick.decision_pick`)
    routes `amend` with the plan, the `pick` and the `switch`, and the
    `planner` node replans holding the pick.

    Once the approval commits (`NodeCtx.after_commit`), it records a
    `plan.approved` span (nesting under the turn span) and a log line with
    the same facts, `plan_version` and `amended`, and withdraws the card's
    early calls still waiting for a slot (`EarlyMints.withdraw`). A crash
    before the commit decides the answer again; one after it records
    neither.
    """
    plan = readable_plan(ctx, ctx.state["plan"])
    if plan is None:
        # The card the user answered names a plan this release does not
        # read, so the answer approves nothing and amends nothing.
        return Next("amend", {"unreadable_plan": True})
    switch = answer.get("switch")
    if isinstance(switch, dict):
        pick = decision_pick(
            plan,
            choice_id=str(switch.get("choice_id")),
            option_id=str(switch.get("option_id")),
            to_origin=windowed_origins(cast(SessionState, ctx.state)),
            alignments=alignments(cast(SessionState, ctx.state)),
        )
        if pick is not None:
            # A planner decision's pick changes what the plan states, so the
            # planner replans with the pick held.
            return Next(
                "amend",
                {
                    "plan": plan.model_dump(mode="json"),
                    "pick": pick.model_dump(mode="json"),
                    "switch": switch,
                },
            )
        # A switch moves a scope choice and approves nothing. The `switch`
        # node revises the plan in code and offers it here again. A row it
        # cannot find, such as a planner decision the plan dropped or one
        # switched to the option it already has, it offers back unchanged
        # with `switch_failed`.
        return Next("switch", {"switch": switch})
    approved_at = _answered_at(answer)
    if answer.get("approved") != plan_revision(plan) or approved_at is None:
        return Next("amend", {"plan": plan.model_dump(mode="json"), "answer": answer})
    approved = plan.approve(at=approved_at)
    ctx.state["plan"] = approved.model_dump_json()

    async def record_approval() -> None:
        amended = approved.version > 1
        record_span("plan.approved", {"dfo.plan_version": approved.version, "dfo.amended": amended})
        logger.info("plan approved: plan_version=%s amended=%s", approved.version, amended)
        if early_mints is not None:
            early_mints.withdraw()

    ctx.after_commit(record_approval)
    return Next("approved", approved.model_dump(mode="json"))


def _answered_at(answer: dict[str, Any]) -> datetime | None:
    """When an approve answer says the user gave it, or `None` when it states
    no timezone-aware ISO 8601 time under `at`."""
    raw = answer.get("at")
    if not isinstance(raw, str):
        return None
    try:
        at = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return at if at.tzinfo is not None else None
