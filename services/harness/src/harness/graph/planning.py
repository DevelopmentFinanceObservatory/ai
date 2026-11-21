"""The `plan_start` and `planner` nodes: the agent's brief handed to the
planner, and the plan it drafts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, cast

from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    MAX_QUESTION_CHARS,
    PlanStatus,
    SessionState,
    UnreadablePickError,
    cut_user_text,
    pick_of,
    plan_revision,
    save_decision_pick,
)
from planner.plan import Plan as PlannerPlan
from planner.session import ASKED_CALL_ID, PLANNER_ASK_PAUSE
from planner.tool_results import ProblemCode

from harness.decision_pick import DecisionPick
from harness.graph import switch as switch_module
from harness.graph.events import GraphEvent, PlannerStarted
from harness.graph.interfaces import PlannerModel
from harness.graph.state import (
    NEW_QUESTION_KEY,
    ORCHESTRATOR_AMENDMENTS_KEY,
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    PICK_REPLAN_KEY,
    UNREADABLE_PLAN_NOTE,
    UNREADABLE_PLAN_UNPLANNED_NOTE,
    conversation_readings,
    readable_plan,
)
from harness.graph.switch import SwitchFailedError
from harness.sentences import StopCode
from harness.turn.node import Next, Node, NodeCtx, Pause

logger = logging.getLogger(__name__)


def planner_node(*, planner: PlannerModel) -> Node[GraphEvent]:
    """The `planner` node, drafting with `planner`. `run` opens an entry;
    `resume` takes the user's answer to the ask the entry's planner ended
    on and runs the planner again with it."""

    async def run(ctx: NodeCtx[GraphEvent], node_input: Any) -> Next | Pause:
        return await _draft(ctx, node_input, planner=planner, answer=None)

    async def resume(
        ctx: NodeCtx[GraphEvent], node_input: Any, answer: dict[str, Any]
    ) -> Next | Pause:
        return await _draft(ctx, node_input, planner=planner, answer=answer)

    # A pick's replan the user stops offers the unchanged plan, whose pause
    # records the Stop.
    return Node(run=run, resume=resume, stop_edges=frozenset({"pending_refusal"}))


async def _draft(
    ctx: NodeCtx[GraphEvent],
    node_input: Any,
    *,
    planner: PlannerModel,
    answer: dict[str, Any] | None,
) -> Next | Pause:
    """One planner entry, or its resume by the user's `answer`.

    A `PlannerModel` drafts a compiled `planner.plan.Plan` from the
    conversation's search readings and this node only validates it
    (`PlannerModel`'s contract): versioning, `previous_version` lineage and the
    DRAFT/PENDING_APPROVAL transitions are enforced by `planner.plan.Plan`
    itself, inside the planner.

    Four edges leave the node. `"ok"` goes to the `approve` node with a
    drafted plan. `"new_question"` ends the turn when the planner read the
    user's amendment as a different question (`starts_new_plan`, refused as
    `QUESTION_FIXED` because the turn's question is fixed,
    `submission.service._resolve_intent`): the amendment's words go under
    `NEW_QUESTION_KEY` for the client to send as the next turn, and the plan
    card is set aside unanswered. `"pending_refusal"` also goes to
    `approve`, for a planner that refuses to draft a plan at all
    (`PlannerAdapter`'s own `MAX_PLAN_ROUNDS` cap, or `QUESTION_FIXED` with
    no amendment's words to hand on) while `ctx.state["plan"]` still holds a
    plan `PENDING_APPROVAL`. The refusal arrives as
    `{"refused": {"text": ...}}` rather than a raised `PlannerAdapterError`,
    which would fail the turn and abandon the plan the user still needs to
    act on. `"refused"` goes to `plan_done` for the same two refusals when
    no such plan is on record. `ctx.state["plan"]` is never written on
    either refusal edge, so it is unchanged either way. The edges differ in
    whether that unchanged record is worth offering again. `"refused"`
    covers a refusal on a first draft (nothing was ever submitted this turn)
    and on a replan (the record on file is already `APPROVED`, so a fresh
    approval cannot act on it). A planner that raises fails the
    turn. A planner that answers `"asked"`, for a planner run that ended on a
    question for the user (`ask_user_question`, `planner.specs`), pauses the
    entry on that question (`planner_ask_pause`); the resume runs the planner
    again with the answer. On the `"refused"` edge `plan_done` lists the asks
    an earlier run of the plan call computed (`ORCHESTRATOR_KEPT_ASKS_KEY`).

    An amendment or a replan of a plan stored in a shape this release does
    not read (`readable_plan`) is drafted as a first draft, and the `"ok"`
    edge's plan carries `UNREADABLE_PLAN_NOTE` as `refusal_note`, which Studio
    shows before the plan card, and `"restarted": True`, since its version
    counts from 1 again. The `"refused"` edge's text opens with
    `UNREADABLE_PLAN_UNPLANNED_NOTE`.

    A first draft's plan also carries `"new_plan": True`, which starts a new
    plan card lineage in the turn. A re-offer of the plan on record does not
    carry it.

    An `"amend"` entry carrying `pick` (`approve` routes a switch of a
    planner decision's row so) replans with the user's pick held: the
    planner is passed it, and admission refuses a plan that does not state
    it. The run can be stopped as a switch is, so it runs under the switch
    node's `_unless_stopped`. A stop or a refusal ends on
    `"pending_refusal"` with the unchanged plan and the `switch_failed` the
    switch node offers (`switch_failed_offer`). It puts back the state from
    before the pick (`_pick_snapshot`), as it does when the run raises.
    Every end of the entry but a planner-side ask clears the pick, so no
    later entry or revision of the plan holds it. An ask keeps the replan's
    state for its resume and records the snapshot under `PICK_REPLAN_KEY`,
    which the resume restores from, and the resumed replan runs under
    `_unless_stopped` again. The entry's `PlannerStarted` says it is a
    pick's.

    The planner runs in this node's task, so a Stop that cancels the node
    cancels the run, and the turn's budget
    (`harness.turn_budget.current_turn_budget`) and the
    `planner.plan.current_discovery_time_left` the run sets stay within it.
    A pick's replan runs as a task of its own under `_unless_stopped`, which
    copies the context it starts in.
    """
    raw = ctx.state.get("plan")
    question = ctx.state.get(ORCHESTRATOR_BRIEF_KEY, "")
    payload = node_input if isinstance(node_input, dict) else {}
    refusal = payload.get("refusal")
    amendment_answer = payload.get("answer")
    amendment = amendment_answer.get("amendment") if isinstance(amendment_answer, dict) else None
    # A first-draft entry passes no `prior_plan`, even when
    # `ctx.state["plan"]` already holds one. It is `plan_start`'s, the only
    # entry of the three that carries neither a `"plan"` key (`approve`'s
    # "amend" edge, whether the answer carries amendment text or merely
    # names a stale revision) nor a `"refusal"` key (`execute`'s "replan"
    # edge). The test is which keys the payload carries, because a
    # stale-revision "amend" carries no amendment text either, and reading
    # that shape as a first draft would wipe out a plan still
    # `PENDING_APPROVAL` right before the `approve` node offers it again.
    # A later `plan` call in the same turn finds the earlier call's plan on
    # record. Withholding it here lets
    # `harness.planner_adapter.PlannerAdapter.__call__` see a genuine first
    # draft and clear its own `planner:plan` drafting state. Otherwise it
    # would read a continuing call on a plan already drafted this turn,
    # which needs `starts_new_plan` (refused, the turn's question being
    # fixed) to draft anything but an amendment to that stale plan.
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
    resuming = answer is not None
    try:
        pick = _pick(payload) if existing is not None else None
    except SwitchFailedError as exc:
        # No replan runs for a pick this release cannot read: the state
        # goes back as it stood before the pick, and the card offers
        # the plan unchanged.
        assert existing is not None
        _end_pick_entry(ctx, before=_pick_snapshot(ctx, resuming=resuming))
        switch = payload.get("switch") or {}
        return Next(
            "pending_refusal",
            switch_module.switch_failed_offer(
                existing,
                choice_id=str(switch.get("choice_id")),
                option_id=str(switch.get("option_id")),
                error=exc,
            ),
        )
    ctx.emit(PlannerStarted(revising=not is_first_draft, pick=pick is not None))

    # The state before the pick, which a pick's replan that is stopped,
    # refused or raises puts back.
    before = _pick_snapshot(ctx, resuming=resuming) if pick is not None else None
    drafting = planner(
        ctx=ctx,
        question=question,
        readings=conversation_readings(ctx.state),
        prior_plan=(
            None if is_first_draft else (existing.model_dump(mode="json") if existing else None)
        ),
        refusal=refusal,
        amendment=amendment,
        canvas_digest=_digest(ctx, "canvas_digest"),
        conversation_digest=_digest(ctx, "conversation_digest"),
        pick=pick,
        answer=answer,
    )
    failed: SwitchFailedError | None = None
    try:
        drafted = await (drafting if pick is None else switch_module._unless_stopped(ctx, drafting))
    except SwitchFailedError as exc:
        # Only a pick's replan, stopped by the user.
        failed, drafted = exc, {}
    except BaseException:
        if before is not None:
            _end_pick_entry(ctx, before=before)
        raise
    asked = drafted.get("asked") if isinstance(drafted, dict) else None
    if asked is not None:
        # The planner's run ended on an ask it put to the user, so the
        # entry pauses on it, after every call the run made has answered.
        # The resume calls the planner again with the answer. A pick's
        # replan keeps its state for the resume, and records what to put
        # back should the resumed replan not land.
        if before is not None:
            ctx.state[PICK_REPLAN_KEY] = before
        return planner_ask_pause(asked)
    refused = drafted.get("refused") if isinstance(drafted, dict) else None
    if pick is not None and (failed is not None or refused is not None):
        assert existing is not None and before is not None
        _end_pick_entry(ctx, before=before)
        switch = payload.get("switch") or {}
        return Next(
            "pending_refusal",
            switch_module.switch_failed_offer(
                existing,
                choice_id=str(switch.get("choice_id")),
                option_id=str(switch.get("option_id")),
                error=failed or _pick_refused(refused or {}, pick),
            ),
        )
    if refused is not None:
        return _refused_entry(
            ctx, refused, existing=existing, amendment=amendment, unreadable=unreadable
        )
    planner_plan = PlannerPlan.model_validate(drafted["plan"])
    ctx.state["plan"] = planner_plan.model_dump_json()
    if amendment:
        ctx.state[ORCHESTRATOR_AMENDMENTS_KEY] = [
            *(ctx.state.get(ORCHESTRATOR_AMENDMENTS_KEY) or ()),
            cut_user_text(str(amendment), MAX_QUESTION_CHARS),
        ]
    if pick is not None:
        _end_pick_entry(ctx)
    return Next(
        "ok",
        {
            **planner_plan.model_dump(mode="json"),
            "summary": drafted["summary"],
            "revision": plan_revision(planner_plan),
            **({"refusal_note": UNREADABLE_PLAN_NOTE} if unreadable else {}),
            # The draft's version counts from 1 again, on a line of its
            # own: no earlier card of the turn is a version of this plan.
            **({"restarted": True} if unreadable else {}),
            # A first draft starts a new plan card lineage, as a second
            # `plan` call in a turn does. A re-offer of the plan on
            # record (`"pending_refusal"`) keeps the lineage it has.
            **({"new_plan": True} if is_first_draft and not unreadable else {}),
        },
    )


def _refused_entry(
    ctx: NodeCtx[GraphEvent],
    refused: Mapping[str, Any],
    *,
    existing: PlannerPlan | None,
    amendment: Any,
    unreadable: bool,
) -> Next:
    """Where an entry the planner refused goes (`_draft`): `new_question`
    for an amendment read as a different question, `pending_refusal` to
    offer the plan still awaiting approval again, else `refused`."""
    if refused.get("code") == ProblemCode.QUESTION_FIXED and amendment:
        # The user typed a different question on the plan card. The turn
        # ends here, and the client sends the words as the next turn.
        ctx.state[NEW_QUESTION_KEY] = str(amendment)
        return Next("new_question")
    text = str(refused.get("text", ""))
    if existing is not None and existing.status is PlanStatus.PENDING_APPROVAL:
        # The plan the user was already looking at is untouched (this
        # refusal never writes `ctx.state["plan"]`) and still
        # `PENDING_APPROVAL`, so it can be offered again as it stands.
        # Go back to the `approve` node with that same plan, carrying
        # the refusal's own sentence so the user reads why their change
        # did not land before seeing the unchanged plan again.
        note = (
            f"{text} Version {existing.version} is still there to "
            "approve; anything past that needs a new question."
        )
        return Next(
            "pending_refusal",
            {
                **existing.model_dump(mode="json"),
                "summary": note,
                "revision": plan_revision(existing),
                "refusal_note": note,
            },
        )
    if unreadable:
        text = f"{UNREADABLE_PLAN_UNPLANNED_NOTE} {text}".rstrip()
    return Next(
        "refused",
        {"text": text, "final": bool(refused.get("final")), "code": refused.get("code")},
    )


def _pick(payload: dict[str, Any]) -> DecisionPick | None:
    """The planner decision's option the user picked on the plan card, which
    `approve`'s `"amend"` route carries under `pick`, or `None`.

    Raises:
        SwitchFailedError: If the payload carries a pick this release cannot
            read, as one an earlier release wrote before a deploy. The user
            picks it again.
    """
    try:
        return pick_of(payload.get("pick"))
    except UnreadablePickError as exc:
        raise SwitchFailedError(
            "That choice could not be applied. Make it again.",
            code=switch_module.SCOPE_CHOICE_NOT_APPLIED,
            retryable=True,
        ) from exc


def planner_ask_pause(asked: Mapping[str, Any]) -> Pause:
    """The pause on the planner's accepted ask, `asked` as a `PlannerModel`
    answers it. Its payload is the call's arguments, which Studio renders the
    question from (`harness.studio_translate._planner_question`), and the
    call's id under `ASKED_CALL_ID`, by which the planner's tools read the
    user's answer to the pause as the answer to that call
    (`planner.session.session_steps`)."""
    return Pause(
        kind=PLANNER_ASK_PAUSE,
        payload={**asked["args"], ASKED_CALL_ID: asked["call_id"]},
        message=str(asked["args"]["message"]),
    )


def _digest(ctx: NodeCtx[GraphEvent], name: str) -> dict[str, Any] | None:
    """The turn's digest input `name`, decoded, or `None` when the turn
    opened with none. The turn holds it as the JSON text it was written as."""
    raw = ctx.inputs.get(name)
    return json.loads(raw) if raw else None


def _pick_snapshot(ctx: NodeCtx[GraphEvent], *, resuming: bool) -> dict[str, Any]:
    """The state a pick's replan puts back when it does not land.

    A fresh entry takes it as the state stands. The resume of a planner-side
    ask reads back the copy the pause recorded under `PICK_REPLAN_KEY`,
    since the asking run has written drafting state and readings since.
    """
    held = ctx.state.get(PICK_REPLAN_KEY)
    if resuming and isinstance(held, dict):
        return held
    return switch_module._restorable_state(ctx)


def _end_pick_entry(ctx: NodeCtx[GraphEvent], *, before: dict[str, Any] | None = None) -> None:
    """End a pick's planner entry other than by a planner-side ask: put the
    state back as `before` holds it, for a replan that did not land, clear
    the pick so no later entry or revision of the plan holds it, and clear
    the snapshot a pause recorded."""
    if before is not None:
        switch_module._restore(ctx, before)
    save_decision_pick(cast(SessionState, ctx.state), None)
    ctx.state[PICK_REPLAN_KEY] = None


def _pick_refused(refused: dict[str, Any], pick: DecisionPick) -> SwitchFailedError:
    """The `SwitchFailedError` for a pick's replan that the planner adapter
    refused: it takes the refusal's code, or the adapter's `PLAN_NOT_READY` for
    one stating none, and its sentence for the user (`reason`). It is
    retryable under the codes a scope switch's revision is
    (`harness.graph.switch._RETRYABLE_REVISION_CODES`): a later pick of the
    same option escapes only a refusal for time or an unreachable service."""
    code = str(refused.get("code") or StopCode.PLAN_NOT_READY)
    return SwitchFailedError(
        str(refused.get("reason") or refused.get("text") or "")
        or f"The plan could not be redrafted to use {pick.chosen}.",
        code=code,
        retryable=code in switch_module._RETRYABLE_REVISION_CODES,
    )


async def _plan_start(ctx: NodeCtx[GraphEvent], node_input: dict[str, Any]) -> Next:
    """Hand the agent's brief to the planner under the orchestrator's own
    key. This call's asks go under `ASKS_STATE_KEY`, the parties they
    break down by under `ASK_BREAKDOWNS_STATE_KEY`, and the numbers of the
    asks across years under `ASKS_ACROSS_YEARS_STATE_KEY`. The last two are
    cleared when the call lists none, so no plan is held to an earlier
    call's, and no earlier call's kept asks or amendments are listed. The
    planner's first-draft entry is an empty payload."""
    ctx.state[ORCHESTRATOR_BRIEF_KEY] = node_input["brief"]
    ctx.state[ASKS_STATE_KEY] = node_input["asks"]
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = node_input.get("ask_breakdowns") or None
    ctx.state[ASKS_ACROSS_YEARS_STATE_KEY] = node_input.get("asks_across_years") or None
    ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
    ctx.state[ORCHESTRATOR_AMENDMENTS_KEY] = None
    return Next("next", {})


plan_start: Node[GraphEvent] = Node(run=_plan_start)
