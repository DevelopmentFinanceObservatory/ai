"""The `plan_done` node: hands the agent what its plan computed, what failed,
and each ask's status, or the planner's refusal.
"""

from __future__ import annotations

import json
import re
from collections.abc import (
    Callable,
    Collection,
    Coroutine,
    Mapping,
)
from typing import Any, cast

from analysis_mcp import digest as analysis_digest
from data_mcp import get_data_digest
from data_mcp.tool_errors import ToolErrorCode
from google.adk import Context
from planner.plan import (
    load_asks,
)
from planner.workflow import Workflow as PlanWorkflow

from harness import plan_display
from harness.graph.seams import (
    Executor,
)
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    ORCHESTRATOR_PLAN_FAILED_STATUS,
    ORCHESTRATOR_PLAN_RAN_STATUS,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    frame_id_for,
    readable_plan,
    register_frames,
)
from harness.turn_records import PlanAsk, PlanRecord

# Characters of planner text one hand-back to the orchestrator may carry per
# field, and characters the plan record may take. The two recorded plans in
# `tests/fixtures/turn_records_recorded.json` render whole in 1,347 and 1,638
# characters. A larger plan folds steps first and loses decisions last
# (`PlanRecord.render`).
_HAND_BACK_CAP = 800
_PLAN_RECORD_CHARS = 2_000


# Lines of a step summary addressed to a tool caller handling frame references
# by hand: the frame reference and execution identity a digest ends with, and
# the note on a mint that ran as a queued operation, by the prefixes their
# renderers write them with. The orchestrator charts the frames `plan_done`
# registers under ids and fetches by a step's spec_id.
_SUMMARY_NOISE = (
    get_data_digest.FRAME_REFERENCE_PREFIX,
    get_data_digest.OPERATION_POLL_PREFIX,
    analysis_digest.FRAME_REFERENCE_PREFIX,
    analysis_digest.EXECUTION_IDENTITY_PREFIX,
)


def _without_noise(summary: str) -> str:
    """`summary` without its `_SUMMARY_NOISE` lines and the blank lines they leave."""
    kept = "\n".join(line for line in summary.split("\n") if not line.startswith(_SUMMARY_NOISE))
    return re.sub(r"\n{3,}", "\n\n", kept).strip()


def _give_plan_back(ctx: Context) -> bool:
    """The first plan that could not be made or run gives the orchestrator
    its plan call back (`BUDGETS`), for one brief that avoids the failure; a
    second failure does not. Answers whether it gave the call back. The
    workflow's own replan cap is untouched."""
    calls = dict(ctx.state.get(ORCHESTRATOR_CALLS_KEY) or {})
    if calls.get("plan_given_back"):
        return False
    calls["plan"] = 0
    calls["plan_given_back"] = True
    ctx.state[ORCHESTRATOR_CALLS_KEY] = calls
    return True


def _ask_status(
    ask: PlanAsk,
    ran: Mapping[str, dict[str, Any]],
    failed: Mapping[str, Any] | None,
    stopped: Collection[str],
) -> dict[str, Any]:
    """One ask as `plan_done` reports it: declined, with the plain reason, or
    each step answering it with its status, and for an ask answered in part
    the part not computed with its reason. `ran` holds each final step that
    produced a frame. `data` and `empty` name the frame, and `empty` means it
    holds no series. `not_chartable` gives the reason no view was prepared.
    A step outside `ran` is `failed` when it is in `stopped` (the failed step
    and every step whose inputs trace back to it, `_stopped_steps`), naming
    the failed step (`failed_at`) and its refusal `code`, and `not_run`
    otherwise."""
    if ask.declined is not None and not ask.outputs:
        return {"ask": ask.ask, "declined": plan_display.DECLINE_REASONS[ask.declined]}
    failure = {"status": "failed"}
    if failed is not None:
        failure["failed_at"] = str(failed.get("step_id"))
        if failed.get("code"):
            failure["code"] = str(failed["code"])
    status: dict[str, Any] = {
        "ask": ask.ask,
        "answered_by": [
            {
                "step": step,
                **ran.get(step, failure if step in stopped else {"status": "not_run"}),
            }
            for step in ask.outputs
        ],
    }
    if ask.declined is not None:
        status["not_computed"] = {
            "part": ask.not_computed or ask.ask,
            "reason": plan_display.DECLINE_REASONS[ask.declined],
        }
    return status


# The `_ask_status` statuses of a step that produced no frame.
_NOT_COMPUTED = frozenset({"failed", "not_run"})


def _computed(ask: Mapping[str, Any]) -> bool:
    """Whether every step answering `ask` (an `_ask_status` entry) produced its frame."""
    entries = ask.get("answered_by") or ()
    return bool(entries) and not any(entry["status"] in _NOT_COMPUTED for entry in entries)


def _partial_result(asks: list[dict[str, Any]] | None, *, frames: bool) -> str:
    """What a stopped plan computed, as `plan_done` states it: the asks whose
    every answering frame holds series (`data`) and can be charted, the asks
    computed with a frame that holds none or cannot be drawn, those an
    earlier plan computed, and the asks a failed or unrun step leaves not
    computed. An ask computed in part is named with the part not computed
    and its reason. Without a computed ask, whether any frame is listed."""
    computed = [ask for ask in asks or () if _computed(ask)]

    def computed_part(ask: Mapping[str, Any]) -> str:
        """`ask` quoted, with the part its plan declined and the reason."""
        declined = ask.get("not_computed")
        if declined is None:
            return json.dumps(ask["ask"])
        return (
            f"{json.dumps(ask['ask'])} except {json.dumps(declined['part'])}, "
            f"not computed because {declined['reason']}"
        )

    charted = [
        ask for ask in computed if all(entry["status"] == "data" for entry in ask["answered_by"])
    ]
    uncharted = [ask for ask in computed if ask not in charted]
    earlier = [ask["ask"] for ask in computed if ask.get("earlier_plan")]
    missing = [ask["ask"] for ask in asks or () if ask.get("answered_by") and not _computed(ask)]
    sentences = []
    if charted:
        sentences.append(
            "The frames listed compute these asks and can be charted: "
            + "; ".join(computed_part(ask) for ask in charted)
            + "."
        )
    if uncharted:
        sentences.append(
            "Computed, with a frame that holds no series or cannot be drawn, as each "
            "step's status says: " + "; ".join(computed_part(ask) for ask in uncharted) + "."
        )
    if earlier:
        sentences.append(
            "Of these, an earlier plan that stopped computed "
            + "; ".join(json.dumps(ask) for ask in earlier)
            + "; their steps are not in the plan record."
        )
    if not computed and frames:
        sentences.append("The frames listed are the results computed before the stop.")
    if missing:
        sentences.append("Not computed: " + "; ".join(json.dumps(ask) for ask in missing) + ".")
    return " ".join(sentences) or "Its result is not available."


def _stopped_steps(workflow: PlanWorkflow | None, failed_step: str | None) -> set[str]:
    """`failed_step` and every step of `workflow` whose bindings trace back to
    it. Steps are in run order, so one pass sees each step's inputs first."""
    if failed_step is None:
        return set()
    stopped = {failed_step}
    for step in workflow.steps if workflow is not None else ():
        if any(binding.from_step in stopped for binding in step.bindings.values()):
            stopped.add(step.id)
    return stopped


# Closed vocabulary for a describe_frame refusal, bucketed by retry_mode so
# no platform string -- not `detail`, not `message`, not the refusal `code`
# itself -- ever reaches the orchestrator through `plan_done`. Those carry
# internal tokens (literal spec_id/content_version values, upstream refusal
# codes).
_FRAME_UNAVAILABLE_REASONS: dict[str, str] = {
    "retry_after_delay": "this step's data view is temporarily unavailable",
    "do_not_retry": "this step's data view could not be prepared",
}


def _frame_unavailable_reason(refusal: dict[str, Any]) -> str:
    """A safe, closed-vocabulary reason a `describe_frame` refusal produced
    no view. Never the raw refusal text: see the module comment above."""
    retry_mode = str(refusal.get("retry_mode"))
    return _FRAME_UNAVAILABLE_REASONS.get(retry_mode, _FRAME_UNAVAILABLE_REASONS["do_not_retry"])


def plan_done_node(
    *, executor: Executor
) -> Callable[[Context, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """The `plan_done` node, describing each final frame through `executor`."""

    async def plan_done(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
        """What the agent reads once its plan has run, or could not be made:
        the final-step frames, described and registered under ids it can
        chart (never an intermediate step's; on a stopped plan, each
        terminal step that finished before the stop), the step summaries, the notes on how the figures were computed
        (`totals`, `answer_notes`), what failed, each cut, and
        each ask the brief listed with its outputs' status or its decline
        (`_ask_status`), then the plan record
        (`harness.turn_records.PlanRecord`). A step refused
        `capability_unavailable` marks its spec_id unavailable to `fetch`
        for the rest of the turn and does not give the plan call back.

        When the plan call's last run stopped, or the planner refused after
        a run stopped, each ask that run did not compute but an earlier run
        of the call did is listed from that earlier run's kept frames
        (`ORCHESTRATOR_KEPT_ASKS_KEY`), marked `earlier_plan`, with their
        step summaries and those notes, and an ask that earlier run
        computed in part with the part not computed and its reason."""
        kept: dict[str, Any] = ctx.state.get(ORCHESTRATOR_KEPT_ASKS_KEY) or {}
        ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
        described: list[tuple[dict[str, Any], dict[str, Any]]] = []
        not_chartable: list[dict[str, Any]] = []
        summaries: list[str] = list(node_input.get("step_summaries", []))
        totals: list[str] = list(node_input.get("stat_var_totals", []))
        frames: dict[str, Any] = dict(ctx.state.get(ORCHESTRATOR_FRAMES_KEY) or {})

        async def describe(ref: dict[str, Any], *, tool: str) -> dict[str, Any]:
            """Describe one final frame for listing, and return its status as
            `_ask_status` reports it."""
            view = await executor(tool="describe_frame", args={"frame_ref": ref})
            if view.get("outcome") == "ok":
                described.append((ref, view.get("view") or {}))
                empty = (view.get("view") or {}).get("series_count") == 0
                return {"status": "empty" if empty else "data", "frame": frame_id_for(ref)}
            # A closed vocabulary (`_FRAME_UNAVAILABLE_REASONS`): never the
            # platform's own text.
            reason = _frame_unavailable_reason(view.get("refusal") or {})
            not_chartable.append({"step": tool, "reason": reason})
            return {"status": "not_chartable", "reason": reason}

        async def earlier(number: int, ask: str) -> dict[str, Any]:
            """Ask `number` as an earlier run of this plan call computed it,
            from its kept frames, adding their summaries and notes."""
            entry = kept[str(number)]
            summaries.extend(step["digest"] for step in entry["steps"] if step["digest"])
            totals.extend(total for total in entry["totals"] if total not in totals)
            status: dict[str, Any] = {
                "ask": ask,
                "answered_by": [
                    {
                        "step": step["step_id"],
                        **await describe(step["frame_ref"], tool=step["tool"]),
                    }
                    for step in entry["steps"]
                ],
            }
            if "not_computed" in entry:
                status["not_computed"] = entry["not_computed"]
            status["earlier_plan"] = True
            return status

        def listing(asks: list[dict[str, Any]] | None) -> dict[str, Any]:
            """The frames, summaries, computation notes (`totals`) and asks
            the agent reads, the described frames registered under their ids."""
            result: dict[str, Any] = {
                "frames": register_frames(frames, described),
                "step_summaries": [
                    _without_noise(str(summary))[:_HAND_BACK_CAP] for summary in summaries
                ],
                "totals": totals,
            }
            if not_chartable:
                result["frames_not_chartable"] = not_chartable
            if asks is not None:
                result["asks"] = asks
            ctx.state[ORCHESTRATOR_FRAMES_KEY] = frames
            return result

        if "text" in node_input:
            # A final refusal (an internal or repeated failure, or a question
            # declined whole) fails the same way for any brief.
            again = not node_input.get("final") and _give_plan_back(ctx)
            computed = listed = ""
            if kept:
                brief_asks = load_asks(ctx.state)
                result = listing(
                    [
                        await earlier(int(number), brief_asks[int(number)])
                        for number in sorted(kept, key=int)
                    ]
                )
                computed = (
                    f"{_partial_result(result['asks'], frames=bool(result['frames']))} "
                    "Every other ask was not computed. "
                )
                listed = f"\n{json.dumps(result)}"
            return {
                "status": ORCHESTRATOR_PLAN_FAILED_STATUS,
                "text": "The planner could not make a plan: "
                f"{str(node_input['text'])[:_HAND_BACK_CAP]}\n{computed}Answer with search "
                "and fetch where you can, and say plainly what could not be calculated."
                + (" A brief that avoids the problem may be planned once more." if again else "")
                + listed,
            }
        # Final step id -> its output's status, as `_ask_status` reports it.
        ran: dict[str, dict[str, Any]] = {}
        steps = node_input.get("frame_ref_steps", [])
        for ref, step in zip(node_input.get("frame_refs", []), steps, strict=True):
            if step.get("is_final_step"):
                ran[step.get("step_id", "")] = await describe(ref, tool=step.get("tool", ""))
        asks: list[dict[str, Any]] | None = None
        failed = node_input.get("failed_step")
        missing_capability = bool(
            failed and failed.get("code") == ToolErrorCode.CAPABILITY_UNAVAILABLE.value
        )
        record = ""
        plan = readable_plan(ctx, ctx.state.get("plan"))
        if plan is not None:
            # The plan record names every source by spec_id, so the agent can
            # fetch a part the failure did not touch without searching again,
            # and the readings let `fetch` read such a spec the way the plan
            # did.
            readings = dict(ctx.state.get(ORCHESTRATOR_READINGS_KEY) or {})
            unavailable = dict(ctx.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})
            for row in plan_display.step_rows(plan):
                args = cast("dict[str, Any]", row["args"])
                if args.get("spec_id") and args.get("stat_vars"):
                    readings[args["spec_id"]] = args["stat_vars"]
                if (
                    missing_capability
                    and args.get("spec_id")
                    and row["id"] == failed.get("step_id")
                ):
                    unavailable[args["spec_id"]] = str(failed.get("message") or "")
            ctx.state[ORCHESTRATOR_READINGS_KEY] = readings
            ctx.state[ORCHESTRATOR_UNAVAILABLE_KEY] = unavailable
            plan_record = PlanRecord.from_plan(
                plan,
                # A source an earlier turn's search returned is a searched
                # one too, and keeps its block in that turn's record.
                searched={
                    *(ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}),
                    *(ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("spec_ids", []),
                },
                failed_step=failed.get("step_id") if failed else None,
            )
            ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY] = [
                *(ctx.state.get(ORCHESTRATOR_PLAN_RECORDS_KEY) or []),
                plan_record.model_dump(mode="json"),
            ]
            record = "\n" + plan_record.render(budget=_PLAN_RECORD_CHARS)
            if plan_record.asks:
                stopped = _stopped_steps(plan.workflow, failed.get("step_id") if failed else None)
                asks = [_ask_status(ask, ran, failed, stopped) for ask in plan_record.asks]
        if failed and asks:
            asks = [
                await earlier(number, status["ask"])
                if str(number) in kept and not _computed(status)
                else status
                for number, status in enumerate(asks)
            ]
        result = listing(asks)
        if failed:
            result["failed_step"] = {
                key: str(failed[key])[:_HAND_BACK_CAP]
                for key in ("step_id", "tool", "code", "message", "remedy")
                if failed.get(key)
            }
            partial = _partial_result(result.get("asks"), frames=bool(result["frames"]))
            if missing_capability:
                # No brief avoids a capability the question needs, so the
                # plan call is not given back.
                lead = (
                    "The approved plan stopped at the step marked failed: this deployment "
                    "cannot do what that step needs, so the part of the question it serves "
                    "is unavailable. Its data fetched without the refused option, or a chart "
                    "of other data, does not answer it. Say that this part is unavailable "
                    f"and why, in the user's terms. {partial}"
                )
            else:
                lead = (
                    f"The approved plan stopped at the step marked failed. {partial} Some "
                    "other parts of the question may still be retrievable on their own with "
                    "fetch, by the spec_id a step names: show what can be shown, labelled as "
                    "what it is, never as the derived result; do not repeat the failed "
                    "request unchanged; and say plainly what remains unanswered and why, in "
                    "the user's terms."
                )
                if _give_plan_back(ctx):
                    lead += " A brief that avoids the failure may be planned once more."
        elif not_chartable and not described:
            lead = (
                "The approved plan ran, but its result cannot be drawn: tell the user what "
                "the plan computed, in words, from the step summaries. Only a frame listed "
                "under frames can be charted."
            )
        else:
            lead = "The approved plan ran. Chart its frames and answer."
        return {
            "status": ORCHESTRATOR_PLAN_RAN_STATUS,
            "text": f"{lead}\n{json.dumps(result)}{record}",
        }

    return plan_done
