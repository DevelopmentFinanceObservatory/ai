"""The `plan_done` node: hands the agent what its plan computed, what failed,
and each ask's status, or the planner's refusal.
"""

from __future__ import annotations

import json
import re
from collections.abc import (
    AsyncIterator,
    Callable,
    Collection,
    Mapping,
)
from dataclasses import dataclass, field
from typing import Any, cast

from analysis_mcp import digest as analysis_digest
from data_mcp import get_data_digest
from data_mcp.tool_errors import ToolErrorCode
from executor.ledger import RunLedger
from google.adk import Context
from planner.plan import (
    Plan,
    SessionState,
    load_asks,
)
from planner.workflow import Workflow as PlanWorkflow

from harness import plan_display
from harness.graph.delivery import Delivery, deliver, frame_read
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
from harness.sentences import DECLINE_REASONS
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
    otherwise: a step the run never reached, or a derived step whose frame
    is withheld because it is not published (`presentable_steps`), as on a
    stopped run or one whose publication is unfinished."""
    if ask.declined is not None and not ask.outputs:
        return {"ask": ask.ask, "declined": DECLINE_REASONS[ask.declined]}
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
            "reason": DECLINE_REASONS[ask.declined],
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


def frame_unavailable_reason(refusal: dict[str, Any]) -> str:
    """A safe, closed-vocabulary reason a `describe_frame` refusal produced
    no view. Never the raw refusal text: see `_FRAME_UNAVAILABLE_REASONS`."""
    retry_mode = str(refusal.get("retry_mode"))
    return _FRAME_UNAVAILABLE_REASONS.get(retry_mode, _FRAME_UNAVAILABLE_REASONS["do_not_retry"])


def _unfinished_run_lead(run: Mapping[str, Any]) -> str:
    """What the agent is told of an accepted run whose derived results are not
    all published with their replay records (`executor.completion.run_hand_back`),
    whose retrieved frames `plan_done` still lists, or of a run whose progress
    could not be recorded or that could not be admitted at all
    (`executor.completion.unrecorded_hand_back`)."""
    if run["status"] == "not_started":
        return (
            "The approved plan's run could not be started: its record could not be saved, so "
            "nothing was run and nothing carries on. State no figure and chart nothing from "
            "it: tell the user the calculation could not be started and can be asked for again."
        )
    run_id = run["run_id"]
    if run["status"] == "unfinished":
        return (
            f"The approved plan's run {run_id} could not be saved as it ran, so the "
            "calculation is unfinished. State none of its figures and chart nothing from it "
            "this turn: tell the user the work could not be saved and carries on when they "
            "next write."
        )
    if run["status"] == "failed":
        return (
            f"The approved plan computed its result, but run {run_id} could not store it "
            f"with the record that lets it be reproduced ({run.get('code')}), so the "
            "calculation is not complete. State none of the figures it derives and chart "
            "none of them: say plainly that the calculation could not be completed, and "
            "why, in the user's terms."
        )
    return (
        f"The approved plan computed its result, but storing it with the record that "
        f"lets it be reproduced is unfinished: run {run_id} is pending. State none of the "
        "figures it derives and chart none of them this turn: tell the user the "
        f"calculation is pending under run {run_id} and is finished when they next write."
    )


@dataclass
class HandBack:
    """The final frames a run hands the orchestrator, as `plan_done` and the
    direct `submit_analysis` (`harness.direct_recipes`) both list them.

    `describe` describes one frame through `executor`. `listing` registers
    the described frames under ids the orchestrator charts by, each with the
    pair the materialisation store holds it under, and returns
    them with the step summaries, the notes on how the figures were computed
    (`totals`, `harness.graph.answer.answer_notes`), the frames no view was
    prepared for (`frames_not_chartable`), and the asks. `read` says whether
    every frame described so far was read for delivery
    (`harness.graph.delivery.frame_read`).
    """

    executor: Executor
    summaries: list[str] = field(default_factory=list)
    totals: list[str] = field(default_factory=list)
    described: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = field(
        default_factory=list
    )
    not_chartable: list[dict[str, Any]] = field(default_factory=list)
    reads: list[bool] = field(default_factory=list)

    @property
    def read(self) -> bool:
        """Whether every frame described so far was read (`frame_read`)."""
        return all(self.reads)

    @property
    def nothing_chartable(self) -> bool:
        """Whether no frame was described and at least one was refused a
        view, so the run's result can only be told in words."""
        return bool(self.not_chartable) and not self.described

    async def describe(
        self, ref: dict[str, Any], *, tool: str, store_ref: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Describe one final frame for listing, and return its status as
        `_ask_status` reports it: `data`, `empty` for a frame with no series,
        or `not_chartable` with a closed-vocabulary reason. `ref` is the pair
        the frame is described and drawn by, and `store_ref` the pair the
        materialisation store holds it under, or None when the step states no
        store pair."""
        view = await self.executor(tool="describe_frame", args={"frame_ref": ref})
        self.reads.append(frame_read(view))
        if view.get("outcome") == "ok":
            self.described.append((ref, view.get("view") or {}, store_ref))
            empty = (view.get("view") or {}).get("series_count") == 0
            return {"status": "empty" if empty else "data", "frame": frame_id_for(ref)}
        # The reason comes from a closed vocabulary
        # (`_FRAME_UNAVAILABLE_REASONS`), so the platform's text stays out.
        reason = frame_unavailable_reason(view.get("refusal") or {})
        self.not_chartable.append({"step": tool, "reason": reason})
        return {"status": "not_chartable", "reason": reason}

    def listing(self, state: SessionState, asks: list[dict[str, Any]] | None) -> dict[str, Any]:
        """The frames, summaries, computation notes and asks the agent
        reads, the described frames registered in `state` under their ids."""
        frames = dict(state.get(ORCHESTRATOR_FRAMES_KEY) or {})
        result: dict[str, Any] = {
            "frames": register_frames(frames, self.described),
            "step_summaries": [
                _without_noise(str(summary))[:_HAND_BACK_CAP] for summary in self.summaries
            ],
            "totals": self.totals,
        }
        if self.not_chartable:
            result["frames_not_chartable"] = self.not_chartable
        if asks is not None:
            result["asks"] = asks
        state[ORCHESTRATOR_FRAMES_KEY] = frames
        return result


def _missing_capability(failed: Mapping[str, Any] | None) -> bool:
    """Whether `failed` (a run's `failed_step`) was refused `capability_unavailable`."""
    return bool(failed and failed.get("code") == ToolErrorCode.CAPABILITY_UNAVAILABLE.value)


def record_plan(
    state: SessionState,
    plan: Plan,
    *,
    failed: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> PlanRecord:
    """Record `plan` as run this turn, and return its record.

    The record (`harness.turn_records.PlanRecord`) joins
    `ORCHESTRATOR_PLAN_RECORDS_KEY`, which `harness.studio` keeps with the
    turn, so the next turn can recall it and keep the spec blocks of the
    sources it used. Each step's StatVar reading joins
    `ORCHESTRATOR_READINGS_KEY`, so `fetch` reads such a spec the way the
    plan did. A `failed` step refused with `capability_unavailable` marks its
    spec_id unavailable to `fetch` for the rest of the turn. The record names
    every source by spec_id, so the agent can fetch a part a failure did not
    touch without searching again. `run_id` is the run the plan executed as,
    which a later turn reads the record's status from.
    """
    missing_capability = _missing_capability(failed)
    readings = dict(state.get(ORCHESTRATOR_READINGS_KEY) or {})
    unavailable = dict(state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})
    for row in plan_display.step_rows(plan):
        args = cast("dict[str, Any]", row["args"])
        if args.get("spec_id") and args.get("stat_vars"):
            readings[args["spec_id"]] = args["stat_vars"]
        if (
            failed is not None
            and missing_capability
            and args.get("spec_id")
            and row["id"] == failed.get("step_id")
        ):
            unavailable[args["spec_id"]] = str(failed.get("message") or "")
    state[ORCHESTRATOR_READINGS_KEY] = readings
    state[ORCHESTRATOR_UNAVAILABLE_KEY] = unavailable
    plan_record = PlanRecord.from_plan(
        plan,
        # A source an earlier turn's search returned is a searched one too,
        # and keeps its block in that turn's record.
        searched={
            *(state.get(ORCHESTRATOR_SPECS_KEY) or {}),
            *(state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("spec_ids", []),
        },
        failed_step=failed.get("step_id") if failed else None,
        run_id=run_id,
    )
    state[ORCHESTRATOR_PLAN_RECORDS_KEY] = [
        *(state.get(ORCHESTRATOR_PLAN_RECORDS_KEY) or []),
        plan_record.model_dump(mode="json"),
    ]
    return plan_record


def plan_done_node(
    *, executor: Executor, run_ledger: RunLedger
) -> Callable[[Context, dict[str, Any]], AsyncIterator[dict[str, Any]]]:
    """The `plan_done` node, describing each final frame through `executor`
    and deciding the delivery of a run completed in `run_ledger`."""

    async def plan_done(ctx: Context, node_input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """What the agent reads once its plan has run, or could not be made:
        the final-step frames, described and registered under ids it can
        chart (never an intermediate step's, nor a derived one that is not
        published: on a stopped plan, or a run pending or failed at
        publication, each retrieved terminal step that finished; none for a
        run whose progress could not be recorded; a run's standing `run`
        names and `_unfinished_run_lead` tells), the step summaries, the
        notes on how the figures were computed (`totals`, `answer_notes`), what failed, each cut, and
        each ask the brief listed with its outputs' status or its decline
        (`_ask_status`), then the plan record
        (`harness.turn_records.PlanRecord`). A step refused
        `capability_unavailable` marks its spec_id unavailable to `fetch`
        for the rest of the turn and does not give the plan call back.

        When the plan call's last run stopped or did not complete, or the
        planner refused after a run stopped, each ask that run did not
        compute but an earlier run of the call did is listed from that
        earlier run's kept frames (`ORCHESTRATOR_KEPT_ASKS_KEY`), marked
        `earlier_plan`, with their step summaries and those notes, and an
        ask that earlier run computed in part with the part not computed and
        its reason.

        A completed run's delivery is decided by its own final frames
        (`harness.graph.delivery.deliver`): once the session holds the
        output, which it does when the yield returns, a run whose frames
        were all read is recorded delivered; one with a frame left unread is
        recorded undelivered, says so in its standing, and the next turn
        lists it again (`harness.graph.resume`)."""
        kept: dict[str, Any] = ctx.state.get(ORCHESTRATOR_KEPT_ASKS_KEY) or {}
        ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
        hand_back = HandBack(
            executor,
            summaries=list(node_input.get("step_summaries", [])),
            totals=list(node_input.get("stat_var_totals", [])),
        )

        async def earlier(number: int, ask: str) -> dict[str, Any]:
            """Ask `number` as an earlier run of this plan call computed it,
            from its kept frames, adding their summaries and notes."""
            entry = kept[str(number)]
            hand_back.summaries.extend(step["digest"] for step in entry["steps"] if step["digest"])
            hand_back.totals.extend(
                total for total in entry["totals"] if total not in hand_back.totals
            )
            status: dict[str, Any] = {
                "ask": ask,
                "answered_by": [
                    {
                        "step": step["step_id"],
                        **await hand_back.describe(
                            step["frame_ref"],
                            tool=step["tool"],
                            store_ref=step.get("store_ref"),
                        ),
                    }
                    for step in entry["steps"]
                ],
            }
            if "not_computed" in entry:
                status["not_computed"] = entry["not_computed"]
            status["earlier_plan"] = True
            return status

        if "text" in node_input:
            # A final refusal (an internal or repeated failure, or a question
            # declined whole) fails the same way for any brief.
            again = not node_input.get("final") and _give_plan_back(ctx)
            computed = listed = ""
            if kept:
                brief_asks = load_asks(ctx.state)
                result = hand_back.listing(
                    ctx.state,
                    [
                        await earlier(int(number), brief_asks[int(number)])
                        for number in sorted(kept, key=int)
                    ],
                )
                computed = (
                    f"{_partial_result(result['asks'], frames=bool(result['frames']))} "
                    "Every other ask was not computed. "
                )
                listed = f"\n{json.dumps(result)}"
            yield {
                "status": ORCHESTRATOR_PLAN_FAILED_STATUS,
                "text": "The planner could not make a plan: "
                f"{str(node_input['text'])[:_HAND_BACK_CAP]}\n{computed}Answer with search "
                "and fetch where you can, and say plainly what could not be calculated."
                + (" A brief that avoids the problem may be planned once more." if again else "")
                + listed,
            }
            return
        # Final step id -> its output's status, as `_ask_status` reports it.
        ran: dict[str, dict[str, Any]] = {}
        steps = node_input.get("frame_ref_steps", [])
        for ref, step in zip(node_input.get("frame_refs", []), steps, strict=True):
            if step.get("is_final_step"):
                ran[step.get("step_id", "")] = await hand_back.describe(
                    ref, tool=step.get("tool", ""), store_ref=step.get("store_ref")
                )
        # The run's own frames decide its delivery; an earlier run's kept
        # frames, described below, do not.
        read = hand_back.read
        asks: list[dict[str, Any]] | None = None
        failed = node_input.get("failed_step")
        missing_capability = _missing_capability(failed)
        run = node_input.get("run")
        unfinished = run is not None and run.get("status") != "completed"
        delivery: Delivery | None = None
        if run is not None and not unfinished:
            delivery = await deliver(run_ledger, run["run_id"], read=read)
        unrecorded = unfinished and run.get("status") in ("unfinished", "not_started")
        record = ""
        plan = readable_plan(ctx, ctx.state.get("plan"))
        if plan is not None:
            plan_record = record_plan(
                ctx.state, plan, failed=failed, run_id=run.get("run_id") if run else None
            )
            record = "\n" + plan_record.render(budget=_PLAN_RECORD_CHARS)
            if plan_record.asks and not unrecorded:
                stopped = _stopped_steps(plan.workflow, failed.get("step_id") if failed else None)
                asks = [_ask_status(ask, ran, failed, stopped) for ask in plan_record.asks]
        if (failed or unfinished) and asks:
            asks = [
                await earlier(number, status["ask"])
                if str(number) in kept and not _computed(status)
                else status
                for number, status in enumerate(asks)
            ]
        result = hand_back.listing(ctx.state, asks)
        if run is not None:
            result["run"] = {**run, **(delivery.standing if delivery is not None else {})}
        if unfinished:
            lead = _unfinished_run_lead(run)
            if result["frames"]:
                lead += (
                    " The frames listed hold retrieved data, which may be charted and quoted "
                    "as what it is, never as the derived result."
                )
        elif failed:
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
        elif hand_back.nothing_chartable:
            lead = (
                "The approved plan ran, but its result cannot be drawn: tell the user what "
                "the plan computed, in words, from the step summaries. Only a frame listed "
                "under frames can be charted."
            )
        else:
            lead = "The approved plan ran. Chart its frames and answer."
        yield {
            "status": ORCHESTRATOR_PLAN_RAN_STATUS,
            "text": f"{lead}\n{json.dumps(result)}{record}",
        }
        if delivery is not None:
            await delivery.acknowledge()

    return plan_done
