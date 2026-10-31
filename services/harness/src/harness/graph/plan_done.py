"""The `plan_done` node: hands the agent what its plan computed, what failed,
and each ask's status, or the planner's refusal.
"""

from __future__ import annotations

import json
import re
from collections.abc import (
    AsyncIterator,
    Callable,
    Mapping,
)
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

from analysis_mcp import digest as analysis_digest
from data_mcp import get_data_digest
from data_mcp.tool_errors import ToolErrorCode
from executor.ledger import RunLedger
from executor.outcomes import StepRecord
from google.adk import Context
from planner.plan import (
    Plan,
    SessionState,
    aligned_pins,
    aligned_stated_pins,
    authored_text,
    copy_origins,
    load_asks,
    windowed_pins,
)

from harness import plan_display
from harness.chart_binding import MAX_SERIES_PER_COMPONENT
from harness.graph.delivery import Delivery, deliver, frame_read
from harness.graph.orchestrate import AgentHandBack
from harness.graph.seams import (
    Executor,
)
from harness.graph.state import (
    ORCHESTRATOR_ALIGNED_SERVED_KEY,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    ORCHESTRATOR_PLAN_FAILED_STATUS,
    ORCHESTRATOR_PLAN_RAN_STATUS,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    conversation_readings,
    frame_id_for,
    readable_plan,
    register_frames,
)
from harness.sentences import DECLINE_REASONS
from harness.turn_records import AlignedBasis, PlanAsk, PlanRecord, aligned_basis

# Characters of planner text one hand-back to the orchestrator may carry per
# field, and characters the plan record may take. The two recorded plans in
# `tests/fixtures/turn_records_recorded.json` render whole in 1,347 and 1,638
# characters. A larger plan folds steps first and loses decisions last
# (`PlanRecord.render`).
_HAND_BACK_CAP = 800
# Characters of a step summary that hands back a limited ranking whole
# (`_lists_kept_rows`): the digest's header, up to `_LIST_CAP` rank lines of
# at most about 120 characters each (a key of two role columns, a value and
# the `[N rows, coverage=]` tail), and its unit and coverage lines. A summary
# that still passes it is trimmed to the first and last row instead of cut
# mid-list.
_WHOLE_RANKING_CHARS = 3_000
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


class StepSummary(NamedTuple):
    """A step's digest, the tool that produced it, the row count of the
    frame it produced and, for a `rank` with a `limit`, how many entries it
    ranked before the limit cut them. `tool`, `rows` and `ranked_before_limit`
    are None for a summary stored without them. Its digest is handed back
    whole."""

    digest: str
    tool: str | None = None
    rows: int | None = None
    ranked_before_limit: int | None = None

    @classmethod
    def of_record(cls, record: StepRecord) -> StepSummary:
        """The summary of a step that ran. `_asdict` is the shape `of` reads."""
        return cls(record.digest, record.tool.name, record.row_count, record.ranked_before_limit)

    @classmethod
    def of(cls, stored: str | Mapping[str, Any]) -> StepSummary:
        """The summary `execute` hands over as `{"tool", "digest", "rows",
        "ranked_before_limit"}`, or a plain digest string."""
        if isinstance(stored, str):
            return cls(stored)
        return cls(
            str(stored["digest"]),
            stored.get("tool"),
            stored.get("rows"),
            stored.get("ranked_before_limit"),
        )


# The analysis tools whose digest (`analysis_mcp.digest`) prints one line per
# entity. A rank block keeps its first and last row, and the others keep the
# count.
_RANKING_TOOL = "rank"
_COUNT_ONLY_TOOLS = ("aggregate", "growth", "filter", "join", "quotient_of", "share_of")
# A block of lines opens with a heading that states its count, either leading
# ("12 ranked entit(y/ies):") or in brackets (a keyed share's "[3 period(s)]").
_COUNTED_HEADING = re.compile(r"^(\s*)(?:\d+ .*|.* \[\d+ period\(s\)\].*):$")
# A block of more lines than this is trimmed.
_LINES_KEPT_WHOLE = 4


def _is_entry(line: str, *, under: int) -> bool:
    """Whether `line` is a block line: indented deeper than its heading at
    `under`, and other than the `... more entries not shown` note."""
    text = line.lstrip()
    return bool(text) and len(line) - len(text) > under and not text.startswith("...")


def _counted_blocks(lines: list[str]) -> list[tuple[int, int, int]]:
    """`(depth, start, end)` of each block of `lines` that opens with a
    `_COUNTED_HEADING`: the heading's indent, and the slice of lines under it."""
    blocks: list[tuple[int, int, int]] = []
    index = 0
    while index < len(lines):
        heading = _COUNTED_HEADING.match(lines[index])
        index += 1
        if heading is None:
            continue
        depth = len(heading[1])
        end = index
        while end < len(lines) and _is_entry(lines[end], under=depth):
            end += 1
        blocks.append((depth, index, end))
        index = end
    return blocks


def _lists_kept_rows(summary: StepSummary, blocks: list[tuple[int, int, int]]) -> bool:
    """Whether `summary` is a ranking that ran with a `limit`, keeps no more
    rows than a card draws, and whose one block of rank lines holds every row
    the frame keeps. The digest lists at most `_LIST_CAP` rows, so a frame that
    keeps more is not listed whole."""
    if summary.tool != _RANKING_TOOL or summary.ranked_before_limit is None:
        return False
    if summary.rows is None or len(blocks) != 1:
        return False
    _, start, end = blocks[0]
    return summary.rows <= MAX_SERIES_PER_COMPONENT and end - start >= summary.rows


def _without_entity_lines(summary: StepSummary, *, whole_ranking: bool = True) -> str:
    """`summary`'s digest with, when its tool prints one line per entity, each
    block of more than `_LINES_KEPT_WHOLE` lines cut to its heading, which
    states the count, and a line saying how many lines were left out. The
    digest of any other tool is returned as it is.

    A rank digest lists the rows its frame keeps. When the ranking ran with
    a limit, keeps no more rows than a card draws and its one block lists
    every one (`_lists_kept_rows`), all rows are handed back if
    `whole_ranking`. Otherwise the block is cut to its first and last rows. A
    digest prints at most `analysis_mcp.digest._LIST_CAP` rows, so a frame that
    keeps more rows than the block holds keeps the first row and the count, and
    the marker says the last row is not listed. A limited rank digest with a
    block for each period has its blocks replaced by one marker, since its
    header states the kept and ranked counts, and a line per period can pass
    `_HAND_BACK_CAP` and lose the unit, coverage and warning lines after them.
    Any other rank digest with a block for each period, or of a frame whose row
    count is unknown, has each block of more than `_LINES_KEPT_WHOLE` lines cut
    to its count."""
    if summary.tool != _RANKING_TOOL and summary.tool not in _COUNT_ONLY_TOOLS:
        return summary.digest
    lines = summary.digest.split("\n")
    blocks = _counted_blocks(lines)
    ranking = summary.tool == _RANKING_TOOL and summary.rows is not None and len(blocks) == 1
    if summary.ranked_before_limit is not None and len(blocks) > 1:
        first = blocks[0][1] - 2  # The period heading over the first count heading.
        last = blocks[-1][2]
        while last < len(lines) and lines[last].lstrip().startswith("..."):
            last += 1
        marker = f"({len(blocks)} period blocks left out; the header states the counts)"
        return "\n".join([*lines[:first], marker, *lines[last:]])
    whole = whole_ranking and _lists_kept_rows(summary, blocks)
    kept: list[str] = []
    cursor = 0
    for depth, start, end in blocks:
        kept += lines[cursor:start]
        cursor = end
        block = lines[start:end]
        listed = len(block)
        unlisted_rows = ranking and (summary.rows or 0) > listed
        if ranking:
            block = block[: summary.rows]
        if (len(block) <= _LINES_KEPT_WHOLE and not unlisted_rows) or whole:
            kept += block
            continue
        if not ranking:
            edges = []
        elif unlisted_rows:
            edges = block[:1]
        else:
            edges = [block[0], block[-1]]
        indent = " " * (depth + 2)
        marker = f"{indent}({len(block) - len(edges)} lines left out)"
        if unlisted_rows:
            marker = (
                f"{indent}({len(block) - len(edges)} lines left out; the digest lists "
                f"{listed} of the frame's {summary.rows} rows, so its last row is not listed)"
            )
        kept += [*edges[:1], marker, *edges[1:]]
    kept += lines[cursor:]
    return "\n".join(kept)


def _hand_back_text(summary: StepSummary) -> str:
    """`summary` as the orchestrator reads it: without its noise lines and
    entity lines, and within `_HAND_BACK_CAP` characters. A ranking handed back
    whole has `_WHOLE_RANKING_CHARS` instead, and falls back to the first and
    last row where it passes them."""
    summary = summary._replace(digest=_without_noise(summary.digest))
    text = _without_entity_lines(summary)
    if _lists_kept_rows(summary, _counted_blocks(summary.digest.split("\n"))):
        if len(text) <= _WHOLE_RANKING_CHARS:
            return text
        text = _without_entity_lines(summary, whole_ranking=False)
    return text[:_HAND_BACK_CAP]


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
    stopped: Mapping[str, str],
) -> dict[str, Any]:
    """One ask as `plan_done` reports it: removed by the user's amendment
    (`dropped`), declined, with the plain reason, or
    each step answering it with its status, and for an ask answered in part
    the part not computed with its reason. `ran` holds each final step that
    produced a frame. `data` and `empty` name the frame, and `empty` means it
    holds no series. `not_chartable` gives the reason no view was prepared.
    A step outside `ran` is `failed` when it is in `stopped` (each step the
    stop stopped, mapped to the step it landed on, `_stop_causes`), naming
    that step (`failed_at`) and the refusal `code`, and `not_run`
    otherwise: a step the run never reached, or a derived step whose frame
    is withheld because it is not published (`presentable_steps`), as on a
    stopped run or one whose publication is unfinished."""
    if ask.dropped:
        return {"ask": ask.ask, "dropped": _DROPPED}
    if ask.declined is not None and not ask.outputs:
        return {"ask": ask.ask, "declined": DECLINE_REASONS[ask.declined]}

    def failure(step: str) -> dict[str, str]:
        entry = {"status": "failed"}
        if failed is not None:
            entry["failed_at"] = stopped[step]
            if failed.get("code"):
                entry["code"] = str(failed["code"])
        return entry

    status: dict[str, Any] = {
        "ask": ask.ask,
        "answered_by": [
            {
                "step": step,
                **ran.get(step, failure(step) if step in stopped else {"status": "not_run"}),
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


# What `_ask_status` says of an ask the user's amendment removed.
_DROPPED = "The user's amendment removed this ask from the plan; nothing computes it."

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


# The fields of a run's report that name a step or a requirement, or carry
# prose about them, at any depth. Inside a tool's `fix` that maps `remedy`,
# `detail` and a nested refusal's `message`. Every other field (a spec_id, a
# frame reference, a candidate source, the rest of `fix`) holds refs the
# split never renames.
_AUTHORED_FIELDS = frozenset({"step_id", "requirement_id", "detail", "message", "remedy"})


def authored_report(report: Any, plan: Plan) -> Any:
    """`report`, part of what a run of `plan` reports, with each copy named by the requirement it copies.

    A run executes `plan` as stored, beside the copies the split of
    whole-history reads added (`Plan.rolling_copies`), so a stop names a
    copy's step and requirement, which exist only in the stored plan. Each
    string under a field in `_AUTHORED_FIELDS`, at any depth, reads through
    `planner.plan.authored_text`; every other string is kept as it is. What
    `execute` hands the planner reads through this, and the failure
    `plan_done` hands the orchestrator reads through `shown_failure`. The
    run's outcome, and the `failed_step` `execute` hands `plan_done`, keep
    the copy's ids, which the run's lineage and `_stop_causes` look steps
    up by.
    """
    if not plan.rolling_copies:
        return report
    return _authored(report, copy_origins(plan.rolling_copies))


def _authored(value: Any, origins: Mapping[str, str], *, names: bool = False) -> Any:
    if isinstance(value, str):
        return authored_text(value, origins) if names else value
    if isinstance(value, list):
        return [_authored(item, origins, names=names) for item in value]
    if isinstance(value, dict):
        return {
            key: _authored(item, origins, names=key in _AUTHORED_FIELDS)
            for key, item in value.items()
        }
    return value


def _stop_causes(plan: Plan, step_id: str | None) -> dict[str, str]:
    """Each step of `plan` a stop at `step_id` stopped, mapped to the step the model wrote that the stop landed on.

    The stopped steps are `step_id` and every step whose bindings trace back
    to it. The stop lands on `step_id` when the model wrote it. A copy the
    split added is not shown (`plan_display.shown_plan`), so a stop on one
    lands on each shown step that reads it through copies only: one per
    rolling reader of the copy, whose results the stop withholds. The copy's
    origin ran on its own pin and is not among them. Every other stopped
    shown step maps to the step its first stopped input landed on, and a
    stopped copy maps to nothing and is left out. Steps are in run order, so
    one pass sees each step's inputs first, and the result keeps that order.
    """
    if step_id is None:
        return {}
    steps = plan.workflow.steps if plan.workflow is not None else ()
    hidden: set[str] = set()
    if plan.rolling_copies:
        shown = plan_display.shown_plan(plan).workflow
        kept = {step.id for step in shown.steps} if shown is not None else set()
        hidden = {step.id for step in steps} - kept
    landed: dict[str, str | None] = {step_id: None if step_id in hidden else step_id}
    for step in steps:
        inputs = [b.from_step for b in step.bindings.values() if b.from_step in landed]
        if step.id in landed or not inputs:
            continue
        cause = next((landed[i] for i in inputs if landed[i] is not None), None)
        landed[step.id] = cause if cause is not None or step.id in hidden else step.id
    return {step: cause for step, cause in landed.items() if cause is not None}


def shown_failure(
    failed: Mapping[str, Any] | None, plan: Plan, stopped: Mapping[str, str]
) -> dict[str, Any] | None:
    """`failed`, a run's `failed_step`, as the orchestrator and the user read it.

    Its `step_id` is the first step, in run order, the stop landed on
    (`_stop_causes`, given as `stopped`), or its own where it landed on
    none, and its prose names each copy by the requirement it copies
    (`authored_report`).
    """
    if failed is None:
        return None
    landed = next((step for step, cause in stopped.items() if step == cause), None)
    return {**authored_report(failed, plan), "step_id": landed or failed.get("step_id")}


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
    """The final frames a run hands the orchestrator, as `plan_done` and
    `harness.graph.resume` both list them.

    `describe` describes one frame through `executor`. `listing` registers
    the described frames under ids the orchestrator charts by, each with the
    pair the materialisation store holds it under and, for a frame of
    shares, the whole they divide by (`wholes`), and returns them with the
    step summaries, the notes on how the figures were computed
    (`totals`, `harness.graph.answer.answer_notes`), the frames no view was
    prepared for (`frames_not_chartable`), and the asks. `read` says whether
    every frame described so far was read for delivery
    (`harness.graph.delivery.frame_read`).
    """

    executor: Executor
    summaries: list[StepSummary] = field(default_factory=list)
    totals: list[str] = field(default_factory=list)
    described: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = field(
        default_factory=list
    )
    not_chartable: list[dict[str, Any]] = field(default_factory=list)
    reads: list[bool] = field(default_factory=list)
    wholes: dict[str, str] = field(default_factory=dict)

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
        self,
        ref: dict[str, Any],
        *,
        tool: str,
        store_ref: dict[str, Any] | None,
        whole: str | None = None,
    ) -> dict[str, Any]:
        """Describe one final frame for listing, and return its status as
        `_ask_status` reports it: `data`, `empty` for a frame with no series,
        or `not_chartable` with a closed-vocabulary reason. `ref` is the pair
        the frame is described and drawn by, and `store_ref` the pair the
        materialisation store holds it under, or None when the step states no
        store pair. `whole` is what the frame's shares divide by
        (`plan_display.share_wholes`), None for a frame of anything else."""
        view = await self.executor(tool="describe_frame", args={"frame_ref": ref})
        self.reads.append(frame_read(view))
        if view.get("outcome") == "ok":
            self.described.append((ref, view.get("view") or {}, store_ref))
            if whole is not None:
                self.wholes[frame_id_for(ref)] = whole
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
        result: dict[str, Any] = {
            "frames": register_frames(state, self.described, wholes=self.wholes),
            "step_summaries": [_hand_back_text(summary) for summary in self.summaries],
            "totals": self.totals,
        }
        if self.not_chartable:
            result["frames_not_chartable"] = self.not_chartable
        if asks is not None:
            result["asks"] = asks
        return result


def _missing_capability(failed: Mapping[str, Any] | None) -> bool:
    """Whether `failed` (a run's `failed_step`) was refused `capability_unavailable`."""
    return bool(failed and failed.get("code") == ToolErrorCode.CAPABILITY_UNAVAILABLE.value)


def record_plan(
    state: SessionState,
    plan: Plan,
    *,
    failed: Mapping[str, Any] | None = None,
    stopped: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> PlanRecord:
    """Record `plan` as run this turn, and return its record.

    The record (`harness.turn_records.PlanRecord`) joins
    `ORCHESTRATOR_PLAN_RECORDS_KEY`, which `harness.studio` keeps with the
    turn, so the next turn can recall it and keep the spec blocks of the
    sources it used. Each step's StatVar reading joins
    `ORCHESTRATOR_READINGS_KEY`, so `fetch` reads such a spec the way the
    plan did. `failed` is the run's `failed_step` as `execute` hands it,
    naming the step that failed, a copy the split added included. Refused
    with `capability_unavailable`, it marks that step's own spec_id
    unavailable to `fetch` for the rest of the turn. `stopped` maps each step
    the stop stopped to the step it landed on (`_stop_causes`, read off
    `failed` when not given), and the record marks failed each step it
    landed on. The record names every source
    by spec_id, so the agent can fetch a part a failure did not touch
    without searching again. `run_id` is the run the plan executed as, which
    a later turn reads the record's status from.
    """
    readings = dict(state.get(ORCHESTRATOR_READINGS_KEY) or {})
    unavailable = dict(state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})
    for row in plan_display.step_rows(plan):
        args = cast("dict[str, Any]", row["args"])
        if args.get("spec_id") and args.get("stat_vars"):
            readings[args["spec_id"]] = args["stat_vars"]
    failed_step = failed.get("step_id") if failed is not None else None
    if stopped is None:
        stopped = _stop_causes(plan, failed_step)
    if failed is not None and _missing_capability(failed) and plan.workflow is not None:
        for step in plan.workflow.steps:
            if step.id == failed_step and step.args.get("spec_id"):
                unavailable[str(step.args["spec_id"])] = str(
                    authored_report(failed, plan).get("message") or ""
                )
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
        failed_steps=frozenset(stopped.values()),
        run_id=run_id,
        windowed_from={copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()},
        aligned_from=aligned_stated_pins(state),
        aligned_bases=_aligned_bases(state),
    )
    state[ORCHESTRATOR_PLAN_RECORDS_KEY] = [
        *(state.get(ORCHESTRATOR_PLAN_RECORDS_KEY) or []),
        plan_record.model_dump(mode="json"),
    ]
    return plan_record


def _aligned_bases(state: SessionState) -> dict[str, AlignedBasis]:
    """How each compile `state` holds counts the inputs alignment moved onto
    it, by the compile's spec_id (`harness.turn_records.aligned_basis`): read
    off the compile's block (`AlignedPin.block`) under the words of the
    newest reading of the pin it was compiled from, with how the plan's
    retrieval of it was served (`ORCHESTRATOR_ALIGNED_SERVED_KEY`). A declared
    whole (`AlignedPin.whole_axis`) is no reading of its part's words and is
    read under none. A compile whose block is missing or has no scope choices
    has no basis."""
    words = {
        reading.get("spec_id"): reading.get("text") for reading in conversation_readings(state)
    }
    served = state.get(ORCHESTRATOR_ALIGNED_SERVED_KEY) or {}
    bases: dict[str, AlignedBasis] = {}
    for pin in aligned_pins(state):
        block = pin.block
        if block is None or pin.spec_id in bases:
            continue
        stated = None if pin.whole_axis is not None else words.get(pin.origin)
        basis = aligned_basis(block, words=stated, served=served.get(pin.spec_id))
        if basis is not None:
            bases[pin.spec_id] = basis
    return bases


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
        each ask the brief listed with its outputs' status, its decline, or
        its removal by the user's amendment (`_ask_status`), then the plan record
        (`harness.turn_records.PlanRecord`), under `plan_record`. It is the
        result of the agent's `plan` call (`AgentHandBack`): a sentence that
        says what to do with it as the result's text, and the rest as its
        structured fields. A step refused
        `capability_unavailable` marks its spec_id unavailable to `fetch`
        for the rest of the turn and does not give the plan call back.

        When the plan call's last run stopped or did not complete, or the
        planner refused after a run stopped, each ask that run did not
        compute but an earlier run of the call did, and the user's amendment
        did not remove, is listed from that
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
            summaries=[StepSummary.of(stored) for stored in node_input.get("step_summaries", [])],
            totals=list(node_input.get("stat_var_totals", [])),
        )

        async def earlier(number: int, ask: str) -> dict[str, Any]:
            """Ask `number` as an earlier run of this plan call computed it,
            from its kept frames, adding their summaries and notes."""
            entry = kept[str(number)]
            hand_back.summaries.extend(
                StepSummary.of(step) for step in entry["steps"] if step["digest"]
            )
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
            computed = ""
            listing: dict[str, Any] = {}
            if kept:
                brief_asks = load_asks(ctx.state)
                listing = hand_back.listing(
                    ctx.state,
                    [
                        await earlier(int(number), brief_asks[int(number)])
                        for number in sorted(kept, key=int)
                    ],
                )
                computed = (
                    f"{_partial_result(listing['asks'], frames=bool(listing['frames']))} "
                    "Every other ask was not computed. "
                )
            refused = AgentHandBack(
                answers="plan",
                status=ORCHESTRATOR_PLAN_FAILED_STATUS,
                # The refusal's code, which words the failed step the user sees.
                failure={"code": node_input.get("code")},
                text="The planner could not make a plan: "
                f"{str(node_input['text'])[:_HAND_BACK_CAP]}\n{computed}Answer with search "
                "and fetch where you can, and say plainly what could not be calculated."
                + (" A brief that avoids the problem may be planned once more." if again else ""),
                body=listing,
            )
            yield refused.model_dump(mode="json")
            return
        plan = readable_plan(ctx, ctx.state.get("plan"))
        wholes = plan_display.share_wholes(plan) if plan is not None else {}
        # Final step id -> its output's status, as `_ask_status` reports it.
        ran: dict[str, dict[str, Any]] = {}
        steps = node_input.get("frame_ref_steps", [])
        for ref, step in zip(node_input.get("frame_refs", []), steps, strict=True):
            if step.get("is_final_step"):
                step_id = step.get("step_id", "")
                ran[step_id] = await hand_back.describe(
                    ref,
                    tool=step.get("tool", ""),
                    store_ref=step.get("store_ref"),
                    whole=wholes.get(step_id),
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
        record: dict[str, str] = {}
        if plan is not None:
            stopped = _stop_causes(plan, failed.get("step_id") if failed else None)
            plan_record = record_plan(
                ctx.state,
                plan,
                failed=failed,
                stopped=stopped,
                run_id=run.get("run_id") if run else None,
            )
            failed = shown_failure(failed, plan, stopped)
            record["plan_record"] = plan_record.render(budget=_PLAN_RECORD_CHARS)
            if plan_record.asks and not unrecorded:
                asks = [_ask_status(ask, ran, failed, stopped) for ask in plan_record.asks]
        if (failed or unfinished) and asks:
            asks = [
                await earlier(number, status["ask"])
                if str(number) in kept and not _computed(status) and "dropped" not in status
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
            if "remedy" in result["failed_step"]:
                lead += (
                    " Tell the user what they can do about it, as the failed step's remedy says it."
                )
        elif hand_back.nothing_chartable:
            lead = (
                "The approved plan ran, but its result cannot be drawn: tell the user what "
                "the plan computed, in words, from the step summaries. Only a frame listed "
                "under frames can be charted."
            )
        else:
            lead = "The approved plan ran. Chart its frames and answer."
        ran_back = AgentHandBack(
            answers="plan",
            status=ORCHESTRATOR_PLAN_RAN_STATUS,
            text=lead,
            body={**result, **record},
        )
        yield ran_back.model_dump(mode="json")
        if delivery is not None:
            await delivery.acknowledge()

    return plan_done
