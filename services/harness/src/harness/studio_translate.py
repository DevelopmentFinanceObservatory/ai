"""ADK events and executor records, projected onto the v1 wire frames.

The translator keys on the graph's node names and on ADK's
`adk_request_input` interrupt events, so it depends only on
`harness.graph`'s topology. A pausing node's `RequestInput` carries its
payload in the interrupt event's function call args, which is everything
`plan.proposed` / `elicitation.request` need.

Tool-call progress (`activity.step`, `component.created`) comes from the
`_RecordingExecutor` wrapper around the injected executor and the
`_RecordingSurface` wrapper around the tool surface: the graph's nodes make
their calls in-node and yield their result at the end, so the recorders
capture each call as it happens and the translator drains them whenever the
next ADK event arrives. A result carrying a `"component"` member becomes a
`component.created` frame and a composition update.

While a plan step runs, `execute` relays each progress report the call
sends as its own event (`harness.graph.state.STEP_PROGRESS_KEY` on
`custom_metadata`), which becomes a `running` `activity.step` whose
`detail` is the report's message. A plan step keeps one wire `step_id`
from its first report to its final `complete` or `failed` frame, so the
Activity feed holds one line per step and updates it in place.

The planner node has no such recorder -- its work happens inside a
`PlannerModel` seam's own model round trips, not through the tool surface.
Its agent's function-call events reach the runner stream as each call is
made, so each call it makes becomes a status line ("Checking the plan
against the data", "Getting the plan ready for you", ...) the moment it
happens, after the "Working on a plan" line the node opens as a draft
starts. One status line runs at a time; the next one, the plan card, a
pause or the segment's end completes it. The seam's own `on_progress` reports, which
the node can only relay once the draft is done (`harness.graph`'s
`planner` node), say the same thing late and are not shown.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from common.mcp_telemetry import WRITE_TOKEN_KEY
from common.period_note import period_note
from common.refusal import refusal_code
from compiler.surface import (
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
    ToolSurfaceUnavailableError,
)
from data_mcp.catalogue_counts import COUNTS_UNAVAILABLE
from executor.artifacts import declared_ranked_before_limit
from executor.outcomes import RefusalKind
from executor.run import current_step_id
from google.adk.events import Event
from google.adk.workflow.utils._workflow_hitl_utils import (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)
from orchestrator.agent import AGENT_NAME as ORCHESTRATOR_AGENT_NAME
from planner.agent import AGENT_NAME as PLANNER_AGENT_NAME
from planner.agent import PROPOSE_PLAN
from planner.phrasing import entities_phrase
from planner.plan import PLAN_STATE_KEY, DatasetRequirement
from planner.plan import Plan as PlannerPlan
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import ToolId
from pydantic import ValidationError

from harness import plan_display
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph.seams import Executor
from harness.graph.state import (
    APPROVE_INTERRUPT,
    ORCHESTRATOR_PLAN_FAILED_STATUS,
    ORCHESTRATOR_PLAN_RAN_STATUS,
    ORCHESTRATOR_STATUS_FAILURE_KEY,
    ORCHESTRATOR_STATUS_KEY,
    PLANNER_STARTED_KEY,
    STEP_PROGRESS_KEY,
)
from harness.mcp_client import _manifest_summary
from harness.orchestrator_tools import HANDED_OFF
from harness.provenance import (
    ProvenanceStore,
    manifest_with_stat_vars,
    retrieval_frame_ref,
)
from harness.seeds import SEED_STEP_TOOL, SEEDS_STEP_ID
from harness.sentences import MAX_SENTENCE_CHARS, failure_sentence
from harness.studio_state import (
    ConversationState,
    StudioStore,
    _PendingInterrupt,
    _PendingQuestion,
    _wire_id,
)
from harness.studio_state import (
    land_component as state_land_component,
)
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.turn_records import loose_match_notes
from harness.wire import (
    ActivityStep,
    ComponentCreated,
    CompositionUpdated,
    Done,
    ElicitationInput,
    ElicitationOption,
    ElicitationQuestion,
    ElicitationRequest,
    ErrorCode,
    ErrorFrame,
    MessageComplete,
    MessageDelta,
    PlanAction,
    PlanAsk,
    PlanDecision,
    PlanOperation,
    PlanProposed,
    PlanRequirement,
    PlanStep,
    PlanVersioned,
    RetryMode,
    StepProduced,
    StepStatus,
    Suggestions,
    ToolError,
)

logger = logging.getLogger(__name__)

# The bound on a recorded refusal's `message` and on a running line's detail.
# Large enough to hold a real over_ceiling/member_cardinality refusal whole,
# including the advice sentence at the end that says scope cannot narrow it.
_STEP_DETAIL_CAP = 400


# The graph's planner states `resolved_by` as planner_default/user
# (`planner.plan.ResolvedBy`); the wire vocabulary is default/user.
_RESOLVED_BY_TO_WIRE = {"planner_default": "default", "user": "user"}

# The line a `propose_plan` call opens and its response closes. The call
# runs the translation (`planner.agent`'s `propose_plan_translating`), so
# the line's duration covers it. The label stays generic: a call's arguments
# are a patch on the recorded plan, so they name only the datasets the patch
# changes, and the recorded plan is not yet the one being checked. The
# closing frame's `detail` names the recorded plan's data (`_data_phrase`),
# which Studio shows as the line's summary.
_CHECKING = "Checking the plan against the data"

# The line a refused `propose_plan` opens while the planner redrafts, and the
# Activity detail for the refusal's code. The refusal's own message names
# plan fields, so only the code reaches the wire, and a code not listed
# here leaves the line without a detail.
_REWORKING = "Reworking the draft"
_REWORK_REASONS = {
    ProblemCode.INVALID_PLAN: "a step didn't fit the plan's rules",
    ProblemCode.INVALID_ARGUMENTS: "a step didn't fit the plan's rules",
    ProblemCode.OPERATION_UNSUPPORTED: "a step asked for something the tools can't do",
    ProblemCode.PLAN_NOT_DRAFT: "the draft was out of date",
}

# The line from the planner's start, or the agent's hand-off to it, until
# its first call.
_WORKING_ON_PLAN = "Working on a plan"

# The line from an approved plan's `approve` event until its first step
# reports, which covers the executor's start.
_STARTING_PLAN = "Starting the plan"

# The Activity feed's labels for a tool's calls. A plan step's line takes
# the plan's own label for that step instead (`_RecordingSurface.
# step_labels`); this is the fallback when the plan named none, such as a
# plan proposed before step labels were kept.
_TOOL_LABELS = {
    "get_data": "Fetching data",
    "aggregate": "Combining values",
    "compare": "Comparing",
    "growth": "Calculating growth",
    "rank": "Ranking",
    "share_of": "Calculating shares",
    "quotient_of": "Dividing values",
    "sum_of": "Adding values",
    "filter": "Filtering data",
    "join": "Joining data",
    "resolve_anchor": "Finding the latest year",
    "describe_frame": "Checking data",
    "describe_component": "Reading a card",
    "write_component": "Adding a chart",
    SEED_STEP_TOOL: "Receive the datasets from Search",
}
# A tool with no label above is never shown by its machine name.
_UNNAMED_TOOL_LABEL = "Working on your request"

# What a failed seed step says in place of its code's sentence. A seed is
# taken before the agent runs, and a turn that keeps none starts from the
# question with no word of the datasets the user brought: this line is where
# the user learns it.
_SEEDS_UNUSED = (
    "The datasets from Search could not be used, so the answer starts from your question"
)
_SEED_UNUSED = "This Search result was left out"


def failure_detail(failure: Mapping[str, Any]) -> str:
    """The sentence a user reads for `failure`, a refusal or a tool's failed
    answer: the one `harness.sentences.failure_sentence` gives its `code`,
    `retry_mode` and `fix`. Its `message`, which is written for the model,
    stays out of the sentence."""
    fix = failure.get("fix")
    return failure_sentence(
        failure.get("code"), failure.get("retry_mode"), fix if isinstance(fix, Mapping) else None
    )


def _timeout_refusal(conversation: ConversationState, *, tool: str, key: str) -> dict[str, Any]:
    """A synthetic refusal, in the same `{code, retry_mode, fix}` shape a
    real one carries, for an MCP call that hit its pinned timeout
    (`mcp_client._call_tool` tags the exception with `timeout_tool`/
    `timeout_key`). `key` identifies the call itself -- tool plus its own
    arguments -- not just the tool: the first time THIS call times out in
    the conversation it is a genuine transient failure, worth the pinned
    wait, and so is a DIFFERENT call to the same tool timing out (that is
    evidence the service itself is slow). Only a repeat of the exact same
    call means the wait already bought nothing, so this says that instead
    of advising the same delay again.

    A `write_component` timeout is never advised a retry: the call was sent,
    so the platform may have stored the component, and repeating it can
    store a second draft."""
    repeat = key in conversation.timed_out_calls
    conversation.timed_out_calls.add(key)
    if tool == "write_component":
        return harness_refusal(
            code=HarnessCode.WRITE_UNCONFIRMED,
            message="write_component timed out after it was sent, so the component may exist",
        )
    return {
        "code": ErrorCode.TRANSIENT_UNAVAILABLE.value,
        "retry_mode": (RetryMode.DO_NOT_RETRY if repeat else RetryMode.RETRY_AFTER_DELAY).value,
        "fix": None,
        "message": f"{tool} timed out again; the earlier wait did not help"
        if repeat
        else f"{tool} timed out",
    }


def _stream_fatal(conversation: ConversationState, exc: Exception) -> ErrorFrame:
    """A stream-fatal error frame. A resolver refusal keeps its real code
    when the wire grammar knows it (e.g. `capability_unavailable` on an
    identity-reproduction failure); a timed-out MCP call (`timeout_tool`/
    `timeout_key`, set by `mcp_client._call_tool`) becomes its own synthetic
    refusal so a repeat of the exact same call stops being told to wait, and
    reaches the wire as `transient_unavailable` whatever its own code.

    A code the wire grammar has no member for, e.g.
    `executor.outcomes.RefusalKind.SURFACE_CHANGED`, which fires when a
    plan step's tool has changed since the plan was compiled, still
    honours the refusal's own `retry_mode` rather than defaulting to a
    transient one: `retry_after_delay` (or no `retry_mode` at all, for an
    exception that carries no refusal shape) becomes `transient_unavailable`
    as before, anything else becomes `capability_unavailable`, the closest
    true member, so a do-not-retry refusal is never told to the user as a
    reason to retry.

    The frame's `detail`, which Studio shows as the reply, is the sentence
    for the refusal's own code, or `raised` for an exception that states
    none, under the retry mode the frame states (`failure_detail`). The
    refusal's `fix.detail` and `message` are written for a model or an
    operator and stay in the log the caller writes, as the turn's trace id
    does.
    """
    timeout_tool = getattr(exc, "timeout_tool", None)
    if timeout_tool is not None:
        timeout_key = getattr(exc, "timeout_key", None) or timeout_tool
        refusal = _timeout_refusal(conversation, tool=timeout_tool, key=timeout_key)
        wire_code = ErrorCode.TRANSIENT_UNAVAILABLE.value
    else:
        refusal = getattr(exc, "refusal", None) or {}
        wire_code = refusal.get("code")
    try:
        error_code = ErrorCode(wire_code)
        retry = RetryMode(refusal.get("retry_mode", RetryMode.DO_NOT_RETRY))
    except ValueError:
        try:
            retry = RetryMode(refusal.get("retry_mode", RetryMode.RETRY_AFTER_DELAY))
        except ValueError:
            retry = RetryMode.RETRY_AFTER_DELAY
        error_code = (
            ErrorCode.TRANSIENT_UNAVAILABLE
            if retry is RetryMode.RETRY_AFTER_DELAY
            else ErrorCode.CAPABILITY_UNAVAILABLE
        )
    stated = refusal if refusal.get("code") else {**refusal, "code": HarnessCode.RAISED.value}
    detail = failure_detail({**stated, "retry_mode": retry.value})
    conversation.close_turn(
        final_text=conversation.turn_final_text or detail, error=error_code.value
    )
    return ErrorFrame(
        event="error",
        seq=conversation.next_seq(),
        error=ToolError(
            error=error_code,
            retry=retry,
            detail=detail,
            retry_after_seconds=30 if retry is RetryMode.RETRY_AFTER_DELAY else None,
        ),
    )


@dataclass
class _ExecutorRecord:
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    duration_ms: int
    # Which conversation's turn made the call (the request-scoped identity,
    # captured at call time). The wrapper is per-runner and SHARED across
    # conversations, whose turns run concurrently. An untagged queue let
    # one conversation's translator drain another's records.
    conversation: str
    # The step this record is for, read off `executor.run.current_step_id`
    # at call time: a plan step's id, or `None` for a call outside a plan,
    # such as an orchestrator tool's. Lets `_RecordingSurface.mark_step_failed`
    # find and amend the one record a step's post-call validation refused,
    # by identity rather than by position, and lets the translator give a
    # step's final frame the wire id its progress frames already used.
    step_id: str | None = None
    # Stamped by `_RecordingExecutor.land_component` right after the write
    # it belongs to returns (`make_charts`, `harness.orchestrator_tools`), for
    # a component that passed `harness.wire.usable_component`. Not a
    # decision; the landing happens later, at drain time, when
    # `_component_frames` reads it. `False` (its default) for every record
    # that never became a `land_component` call: a plan-step surface record,
    # or a write whose component failed validation.
    land: bool = False


def _raised_refusal(exc: BaseException) -> dict[str, Any]:
    """The refusal a failed step shows for a call that raised, with the
    exception's class and text as its message, or `cancelled` for one stopped
    mid-flight."""
    if isinstance(exc, asyncio.CancelledError):
        return {
            "code": ErrorCode.CANCELLED.value,
            "retry_mode": "do_not_retry",
            "fix": None,
            "message": "the call was stopped before it returned",
        }
    return {
        "code": HarnessCode.RAISED.value,
        "retry_mode": "retry_after_delay",
        "fix": None,
        "message": f"{type(exc).__name__}: {exc}"[:_STEP_DETAIL_CAP],
    }


class _RecordingExecutor:
    """Wraps the injected executor so every call becomes wire progress.

    The graph's execute node loops in-node and yields once at the end, so
    per-call progress can't come from ADK events; it comes from here. The
    wrapper is per-runner (the workflow closes over it) while turns of
    DIFFERENT conversations run concurrently, so every record is tagged
    with the conversation that made it and drained by tag; within one
    conversation the lock serialises turns and the graph runs nodes
    sequentially.

    `store` is where `_keep_write_token` finds the current conversation.
    """

    def __init__(self, executor: Executor, *, store: StudioStore) -> None:
        self._executor = executor
        self._store = store
        self._records: list[_ExecutorRecord] = []

    def land_component(self, component: dict[str, Any]) -> None:
        """The synchronous stamping capability `make_charts` calls right after
        `write_component` returns, with no `await` in between. Stamps `land`
        onto this call's own recorded write (matched by identity: the record
        whose result carries this exact `component` object) so
        `_component_frames` can perform the landing later, at drain time.
        Never touches the conversation itself; a component with no matching
        record (this capability called directly, not through `__call__`
        first) leaves nothing to stamp and is silently a no-op."""
        for record in reversed(self._records):
            if record.result.get("component") is component:
                record.land = True
                break

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """Run the call and record it; a call that raises is recorded as failed, then re-raised.

        `BaseException`, so a call cancelled mid-flight (a sibling `seed`
        mint stopped by another's failure) closes its step too.
        """
        started = time.monotonic()
        try:
            result = await self._executor(tool=tool, args=args, on_progress=on_progress)
        except BaseException as exc:
            self._record(
                tool, args, {"outcome": "refusal", "refusal": _raised_refusal(exc)}, started=started
            )
            raise
        self._keep_write_token(tool=tool, args=args, result=result)
        self._record(tool, args, result, started=started)
        return result

    def record_step(
        self, *, tool: str, args: dict[str, Any], result: dict[str, Any], step_id: str
    ) -> None:
        """Record step `step_id`, work the caller did without calling the executor, with `result`.

        An optional capability `harness.graph`'s `seed` node probes for with
        `getattr`, as `make_charts` probes for `land_component`: the node
        resolves the seeds through data-mcp, off the executor, and refuses the
        ones it cannot use, and each of those still shows as a step.
        """
        self._record(tool, args, result, started=time.monotonic(), step_id=step_id)

    def _record(
        self,
        tool: str,
        args: dict[str, Any],
        result: dict[str, Any],
        *,
        started: float,
        step_id: str | None = None,
    ) -> None:
        """Record one call; its step is `step_id`, or the current step when that is `None`."""
        self._records.append(
            _ExecutorRecord(
                tool=tool,
                args={key: value for key, value in args.items() if key != WRITE_TOKEN_KEY},
                result=result,
                duration_ms=int((time.monotonic() - started) * 1000),
                conversation=current_conversation_id.get(),
                step_id=step_id if step_id is not None else current_step_id.get(),
            )
        )

    def _keep_write_token(self, *, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        """Move a create's write token off `result` into the current
        conversation's `write_tokens`, before the record, the graph or the
        Edit Card route can read the result. The token is the grant
        `harness.studio_edit` needs to revise that component; a token with
        no conversation to hold it is dropped, and that component's later
        edits recreate it instead. A successful create that returns no
        token is logged at error level, since that card is rebuilt as a
        new card on its next edit."""
        token = result.pop(WRITE_TOKEN_KEY, None)
        component = result.get("component")
        if not isinstance(component, dict):
            return
        if not token:
            if tool == "write_component" and "component_id" not in args:
                logger.error(
                    "write_component created component_id=%r in conversation=%r with no "
                    "write token; its next edit will rebuild it as a new card",
                    component.get("component_id"),
                    current_conversation_id.get(),
                )
            return
        key = current_conversation_key.get()
        conversation = self._store.current(*key) if key is not None else None
        if conversation is None:
            logger.warning(
                "dropping the write token for component_id=%r: no conversation is loaded "
                "to hold it",
                component.get("component_id"),
            )
            return
        conversation.write_tokens[str(component["component_id"])] = str(token)

    def drain(self, conversation: str) -> list[_ExecutorRecord]:
        """Take (only) this conversation's records, preserving order."""
        mine = [record for record in self._records if record.conversation == conversation]
        self._records = [record for record in self._records if record.conversation != conversation]
        return mine

    async def aclose(self) -> None:
        """Release the wrapped executor's resources, if it holds any.

        `Executor` (`harness.graph.seams.Executor`) is a bare-call protocol with
        no `aclose` of its own; `McpExecutor` adds one to release its
        long-lived per-conversation presentation-mcp clients
        (`harness.mcp_client.McpExecutor.aclose`), while the scripted
        stand-ins used in tests and the scripted eval hold nothing to
        release and define no such method. A narrow `hasattr` check here,
        rather than an `isinstance` ladder naming every executor kind, is
        what lets a future executor opt in just by defining the method.
        Raises whatever the wrapped `aclose` raises; the caller
        (`StudioRunner.aclose`, called from `_lifespan`'s shutdown) decides
        whether that can fail the process.
        """
        aclose = getattr(self._executor, "aclose", None)
        if aclose is not None:
            await aclose()


def _surface_call_outcome(tool: str, result: ToolCallResult) -> dict[str, Any]:
    """A `ToolCallResult` in the outcome grammar `_Translator._drain_recorder` reads.

    A successful `get_data` call's summary is its manifest's one line, the
    one a direct fetch shows (`harness.mcp_client._manifest_summary`), beside
    the `period_note` of its coverage (`common.period_note.period_note`)
    when there is one, as a direct fetch states it. Any other successful
    call has no summary, since its plan step's label already says what it
    did, and states the `rows` its result holds when its structured content
    counts them, as an analysis verb's does, with the `ranked_before_limit`
    of a limited ranking. The digest text is written for the model and names
    dcids and dimension keys, so it never becomes the summary.

    A refusal carries `code`, `retry_mode`, `fix` and `message`, read off
    `structured_content`, the machine-readable half of the result. The digest
    text serves only as the last-resort `message` for a refusal that carries
    no structured body at all (a fastmcp argument-validation error, which
    fails before the tool itself runs and so states nothing but the exception
    text). The refusal omits `origin` and `advice`, so it reads as
    unattributed; `McpExecutor._refusal_outcome` states both.
    """
    body: dict[str, Any] = result.structured_content or {}
    if not result.is_error and tool == "get_data":
        note = period_note(body.get("coverage") or {})
        return {
            "outcome": "ok",
            "frame_refs": [],
            "summary": _manifest_summary(body),
            **({"period_note": note} if note else {}),
        }
    if not result.is_error:
        counted: dict[str, int] = {}
        if isinstance(body.get("rows"), int):
            counted["rows"] = body["rows"]
        if (ranked := declared_ranked_before_limit(body)) is not None:
            counted["ranked_before_limit"] = ranked
        return {"outcome": "ok", "frame_refs": [], "summary": None, **counted}
    return {
        "outcome": "refusal",
        "refusal": {
            "code": refusal_code(body) or HarnessCode.UNSPECIFIED.value,
            "retry_mode": str(body.get("retry_mode") or body.get("retry") or "do_not_retry"),
            "fix": body.get("fix") if isinstance(body.get("fix"), dict) else None,
            "message": str(body.get("message") or body.get("detail") or result.text),
        },
    }


class _RecordingSurface:
    """Wraps a `ToolSurface` so every plan step becomes wire progress.

    Mirrors `_RecordingExecutor`: the `execute` node runs
    `executor.run.execute_approved_plan` against this wrapper instead of the
    bare `ToolSurface`, timing each `call_tool` and tagging the record with
    the conversation that made it, the same discipline `_RecordingExecutor`
    uses for the scripted `Executor` seam. `list_tools` passes straight
    through; only calls are wire progress.

    Two things beyond timing:

    - A `get_data` step's manifest is written into `provenance`
      (`harness.provenance.ProvenanceStore`), the same store
      `harness.mcp_client.McpExecutor._get_data` writes on the direct-fetch
      path, so a component assembled from a plan step's frame carries real
      provenance instead of being withheld for want of a manifest.
    - A call's own record is written the moment the call returns, before
      `executor.run._Run`'s post-call checks (shape, evidence, frame) can
      run on it, so a call that succeeded but was refused afterward would
      otherwise read "complete" forever. A call cut off by its step's
      budget never returns and writes no record at all. `mark_step_failed`
      lets the `execute` node correct the first and record the second once
      it has read the run's real outcome (`harness.graph`'s `execute` node
      looks it up via `getattr`, the same duck-typed opt-in
      `_RecordingExecutor.aclose` uses).
    """

    def __init__(self, surface: ToolSurface, *, provenance: ProvenanceStore | None = None) -> None:
        self._surface = surface
        self._provenance = provenance
        self._records: list[_ExecutorRecord] = []
        # The latest approved plan's workflow step order, by conversation
        # (this wrapper is per-runner and shared across conversations, the
        # same reason `_records` is conversation-tagged). Read by
        # `_Translator._drain_recorder` to sort a wave's records back into
        # plan order before numbering them.
        self._step_order: dict[str, list[str]] = {}
        # The same plan's label for each step, by conversation: the Activity
        # feed names a plan step's line with it.
        self._step_labels: dict[str, dict[str, str]] = {}
        # When each plan step's call in flight started, by (conversation,
        # step id). A call that returns or fails takes its entry with it
        # (`_record`); a cancelled one leaves it for `mark_step_failed`.
        self._in_flight: dict[tuple[str, str], float] = {}

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        started = time.monotonic()
        step_id = current_step_id.get()
        if step_id is not None:
            self._in_flight[(current_conversation_id.get(), step_id)] = started
        try:
            result = await self._surface.call_tool(
                tool=tool, arguments=arguments, on_progress=on_progress
            )
        except ToolSurfaceUnavailableError as exc:
            # The call never reached the server, or its answer never came
            # back: with no `ToolCallResult` there is nothing for
            # `_surface_call_outcome` to read, but the attempt still
            # belongs on the Activity feed -- otherwise a transport failure
            # shows nothing at all while the executor retries or stops.
            self._record(
                tool.name,
                arguments,
                {
                    "outcome": "refusal",
                    "refusal": {
                        "code": RefusalKind.TRANSPORT_FAILED.value,
                        "retry_mode": "retry_after_delay",
                        "fix": None,
                        "message": str(exc)[:_STEP_DETAIL_CAP],
                    },
                },
                started=started,
            )
            raise
        if self._provenance is not None and tool.name == "get_data" and not result.is_error:
            manifest = await manifest_with_stat_vars(dict(result.structured_content or {}))
            result = result.model_copy(update={"structured_content": manifest})
            self._remember_manifest(manifest)
        self._record(
            tool.name, arguments, _surface_call_outcome(tool.name, result), started=started
        )
        return result

    def _record(
        self, tool: str, arguments: Mapping[str, Any], result: dict[str, Any], *, started: float
    ) -> None:
        """Record one finished call, which is no longer in flight."""
        conversation = current_conversation_id.get()
        step_id = current_step_id.get()
        if step_id is not None:
            self._in_flight.pop((conversation, step_id), None)
        self._records.append(
            _ExecutorRecord(
                tool=tool,
                args=dict(arguments),
                result=result,
                duration_ms=int((time.monotonic() - started) * 1000),
                conversation=conversation,
                step_id=step_id,
            )
        )

    def _remember_manifest(self, manifest: dict[str, Any]) -> None:
        """Write a successful `get_data` step's manifest into `provenance`.

        Through `ProvenanceStore.put_retrieval`, as `harness.mcp_client.McpExecutor
        ._get_data` writes it, so the two writers agree on where a manifest lives
        regardless of which path retrieved it.
        """
        assert self._provenance is not None  # guarded by the caller
        self._provenance.put_retrieval(
            conversation=current_conversation_id.get(), manifest=manifest
        )

    async def restore_retrievals(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Write each `get_data` result in `results` into `provenance` as
        `call_tool` writes a step's own, its long StatVar list read back
        first. A run carried on from its record (`harness.graph.resume`)
        passes the results it sealed its inputs from
        (`executor.ledger.RunLedger.retrievals`), so its frames rest on the
        same lineage as a run that never left its turn. A no-op when
        `provenance` was never given, and for a result whose lineage
        `provenance` already holds, as a run computed in this process does."""
        if self._provenance is None:
            return
        conversation = current_conversation_id.get()
        for result in results:
            held = self._provenance.get(conversation=conversation, **retrieval_frame_ref(result))
            if held is not None:
                continue
            self._remember_manifest(await manifest_with_stat_vars(dict(result)))

    def mark_step_failed(self, *, step_id: str, tool: str, code: str, detail: str) -> None:
        """Make `step_id`'s last record read failed.

        Amends the step's own complete record when it has one. Finds it by
        (conversation, step_id) -- read off `executor.run.current_step_id`
        at record time -- rather than by position, since the failing step's
        raw call is not always the most recently recorded one (a terminal
        step's `_answer_frame` check runs only after every step in the
        workflow has already made its own call).

        Otherwise records a failed call to `tool` when the step's last call
        never returned (its budget ran out mid-call) or the step recorded
        nothing at all, so the step's line still closes on the Activity
        feed. A step whose own last call already failed is left untouched:
        its last record already reads "refusal".

        `code` is the caller's own diagnosis of what refused the step
        (`harness.graph.execute._failed_step`'s `code`: a tool's own code when one
        called it, the specific `EvidenceIssueCode` when `check_evidence`
        raised it, `outcome.refusal.value` otherwise) -- never a fixed
        string, so the step's record names the actual check that fired
        instead of one generic label for every masked failure.
        """
        conversation = current_conversation_id.get()
        failed = {
            "outcome": "refusal",
            "refusal": {
                "code": code,
                "retry_mode": "do_not_retry",
                "fix": None,
                "message": detail[:_STEP_DETAIL_CAP],
            },
        }
        records = [
            record
            for record in self._records
            if record.conversation == conversation and record.step_id == step_id
        ]
        for record in records:
            if record.result.get("outcome") == "ok":
                record.result = failed
                return
        started = self._in_flight.pop((conversation, step_id), None)
        if started is None and records:
            return
        self._records.append(
            _ExecutorRecord(
                tool=tool,
                args={},
                result=failed,
                duration_ms=int((time.monotonic() - started) * 1000) if started is not None else 0,
                conversation=conversation,
                step_id=step_id,
            )
        )

    def link_evidence(
        self,
        *,
        frame_ref: Mapping[str, str | None],
        evidence_frame_refs: Sequence[Mapping[str, str | None]],
        stat_var_count: int | None,
        carried_frame_refs: Sequence[Mapping[str, str | None]],
        summed_frame_refs: Sequence[Sequence[Mapping[str, str | None]]] = (),
    ) -> None:
        """Write `frame_ref`'s lineage from the retrieval frames in `evidence_frame_refs`.

        A terminal frame is usually a transform's (`aggregate`, `join`, ...),
        which mints its own `ops_` frame identity (`analysis_mcp.tool_pipeline`),
        distinct from the `spec_` one `_remember_manifest` wrote each
        retrieval's manifest under. Without this, `provenance.get` on the
        terminal frame's own key would find nothing. `evidence_frame_refs`
        names every retrieval the terminal frame rests on, in workflow order;
        the lineage written under `frame_ref` holds them all
        (`ProvenanceStore.alias`), beside `stat_var_count`, the distinct
        StatVars the terminal step declared its frame carries (`None` for a
        retrieval whose `get_data` call did not read the frame), counting a
        summed family ref from a retrieval in `carried_frame_refs` as the
        StatVars it sums, and each ref a `sum_of`, `quotient_of` or
        `share_of` wrote for its own result that the frame carries as the
        distinct StatVars of its entry in `summed_frame_refs`, the
        retrievals under that verb's sides. Called once per terminal frame from
        `harness.graph.answer.link_evidence_to_terminal_frames`.

        A no-op when `provenance` was never given. All-or-nothing: when any
        retrieval's manifest was never recorded (its tool was not `get_data`,
        it ran through a different `_RecordingSurface`, or the store evicted
        it), nothing is written, a warning names the frame, and the card for
        that frame stays withheld rather than citing fewer sources than it
        rests on.
        """
        if self._provenance is None:
            return
        linked = self._provenance.alias(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.get("spec_id"),
            content_version=frame_ref.get("content_version"),
            sources=[
                (ref.get("spec_id"), ref.get("content_version")) for ref in evidence_frame_refs
            ],
            stat_var_count=stat_var_count,
            carried=[
                (ref.get("spec_id"), ref.get("content_version")) for ref in carried_frame_refs
            ],
            summed=[
                [(ref.get("spec_id"), ref.get("content_version")) for ref in under]
                for under in summed_frame_refs
            ],
        )
        if not linked:
            logger.warning(
                "provenance: no lineage written for frame spec_id=%r content_version=%r; "
                "a retrieval frame it rests on has no recorded manifest (%s)",
                frame_ref.get("spec_id"),
                frame_ref.get("content_version"),
                ", ".join(str(ref.get("spec_id")) for ref in evidence_frame_refs) or "none named",
            )

    def retrieval_manifest(self, *, frame_ref: Mapping[str, str | None]) -> dict[str, Any] | None:
        """The `get_data` result recorded for the retrieval that minted
        `frame_ref`, or `None` when it was never recorded. Read by
        `harness.graph`'s `execute` node via `getattr`, like `link_evidence`.
        """
        if self._provenance is None:
            return None
        lineage = self._provenance.get(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.get("spec_id"),
            content_version=frame_ref.get("content_version"),
        )
        return lineage[0].manifest if lineage is not None else None

    def drain(self, conversation: str) -> list[_ExecutorRecord]:
        """Take (only) this conversation's records, preserving order, and
        forget its calls still in flight, whose run has ended."""
        mine = [record for record in self._records if record.conversation == conversation]
        self._records = [record for record in self._records if record.conversation != conversation]
        self._in_flight = {
            key: started for key, started in self._in_flight.items() if key[0] != conversation
        }
        return mine

    def set_step_order(
        self,
        conversation: str,
        order: Sequence[str],
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Remember `conversation`'s latest approved plan's workflow step order.

        Called from `_Translator._interrupt_frames` when it builds the
        `plan.proposed` frame. A workflow's own steps run as concurrent
        waves (`executor.run._Run.execute`), so two independent steps can
        finish, and get recorded, in the opposite order the plan lists
        them; `step_order` lets `_drain_recorder` put them back. `labels`
        are the plan's own label for each step, by step id.
        """
        self._step_order[conversation] = list(order)
        self._step_labels[conversation] = dict(labels or {})

    def step_order(self, conversation: str) -> list[str]:
        """`conversation`'s latest plan step order, or `[]` if none was set."""
        return self._step_order.get(conversation, [])

    def step_labels(self, conversation: str) -> dict[str, str]:
        """`conversation`'s latest plan step labels by step id, or `{}`."""
        return self._step_labels.get(conversation, {})


# ---------------------------------------------------------------------------
# ADK events -> wire frames
# ---------------------------------------------------------------------------


def _node_name(event: Event) -> str:
    path = event.node_info.path if event.node_info else ""
    return path.split("/")[-1].split("@")[0]


def _request_input_call(event: Event) -> tuple[str, dict[str, Any]] | None:
    """The (interrupt id, `RequestInput` dump) a pausing node's interrupt
    event carries. The dump is keyed by RequestInput's camelCase aliases;
    the id is the function call's own `id` (set to the interrupt id by
    ADK's `create_request_input_event`).

    None for a partial event. A streamed model call reaches the runner
    first as partial chunks and then as one aggregated non-partial event,
    and the runner saves only the non-partial one to the session. A pause
    frame built from a chunk would name a call the session may never hold,
    so `/interrupt`'s answer to it would match no call."""
    if event.partial or not (event.content and event.content.parts):
        return None
    for part in event.content.parts:
        call = part.function_call
        if call is not None and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
            return str(call.id or ""), dict(call.args or {})
    return None


def _facts(*facts: Any) -> str | None:
    """A completed step's detail: each of `facts` its result states, in
    order, or `None` when it states none. The Studio eval's grounding check
    reads a reply's numbers against it."""
    return " \u00b7 ".join(str(fact) for fact in facts if fact)[:_STEP_DETAIL_CAP] or None


def _rows_held(record: _ExecutorRecord) -> str | None:
    """How many rows the result of a recorded call other than a retrieval
    holds, when it counts them. A ranking's rows are the ones it keeps, of
    the entries it ranked when a limit cut them. A retrieval's summary
    states its own observations."""
    rows = record.result.get("rows")
    if record.tool == "get_data" or not isinstance(rows, int):
        return None
    ranked = declared_ranked_before_limit(record.result)
    total = rows if ranked is None else ranked
    counted = f"{rows:,}" if ranked is None else f"{rows:,} of {ranked:,}"
    counted += " row" if total == 1 else " rows"
    return (
        f"The ranking keeps {counted}." if record.tool == "rank" else f"The result holds {counted}."
    )


def _record_detail(record: _ExecutorRecord) -> str | None:
    """A successful recorded call's result: its one-line summary, the period
    note of a retrieval that carries one, and the rows an analysis result
    holds. A card's line names its kind and title already (`_record_label`),
    so it carries none."""
    if record.tool == "write_component":
        return None
    result = record.result
    return _facts(result.get("summary"), result.get("period_note"), _rows_held(record))


def _refused_detail(record: _ExecutorRecord) -> str:
    """A failed recorded call's line: a seed step's plain sentence, else the
    sentence for its refusal's code."""
    if record.tool == SEED_STEP_TOOL:
        return _SEEDS_UNUSED if record.step_id == SEEDS_STEP_ID else _SEED_UNUSED
    refusal = record.result.get("refusal")
    return failure_detail(refusal if isinstance(refusal, Mapping) else {})


def _answer_failure(answer: Mapping[str, Any]) -> str:
    """The line of an agent tool's failed answer. A tool states its `code`
    and `retry_mode` beside its `error` or `refusal` prose, or inside a
    `refusal` that is a refusal body."""
    refusal = answer.get("refusal")
    return failure_detail(refusal if isinstance(refusal, Mapping) else answer)


def _step_label(tool: str, plan_label: str | None = None) -> str:
    return plan_label or _TOOL_LABELS.get(tool, _UNNAMED_TOOL_LABEL)


def _record_label(record: _ExecutorRecord, plan_labels: Mapping[str, str]) -> str:
    """A recorded call's Activity line: a card's kind and title for a write
    (its summary reads `line_chart: <title>`), else the plan's label for
    its step, else the tool's own."""
    if record.tool == "write_component":
        kind, _, title = str(record.result.get("summary") or "").partition(": ")
        if title:
            return f"{kind.replace('_', ' ').capitalize()}: {title}"
    return _step_label(record.tool, plan_labels.get(record.step_id or ""))


def _data_phrase(plan: PlannerPlan) -> str | None:
    """The plan's data for the checking line's summary, naming its first
    dataset: "health spending per person for Kenya and Uganda", with "and N
    more" for the rest. None for a plan with no dataset."""
    datasets = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    if not datasets:
        return None
    first = datasets[0]
    phrase = f"{first.measure} for {entities_phrase(first.entities, across=first.across)}"
    return f"{phrase} and {len(datasets) - 1} more" if len(datasets) > 1 else phrase


def _plan_from_payload(payload: dict[str, Any]) -> PlannerPlan:
    """The planner node's payload as a `planner.plan.Plan`. The planner
    node itself validates a model-drafted plan (`harness.graph.planning`'s `planner` node
    raises on a draft the seam could not validate), so this only ever sees
    a plan already known good.

    `refusal_note` is the one field a `"pending_refusal"`-routed payload,
    or an `"ok"` draft made over a plan the node could not read, carries
    beside a normal `"ok"` one's own `summary`/`revision` (`harness.graph.planning`'s
    `planner` node): the payload still needs to say why the user is seeing
    a plan again, and `Plan`'s schema has no field for that. `restarted`
    marks an `"ok"` draft made over a plan the node could not read, whose version
    counts from 1 again; `_interrupt_frames` reads it to start a new line of
    plan version ids.
    """
    fields = {
        key: value
        for key, value in payload.items()
        if key not in ("summary", "revision", "refusal_note", "restarted")
    }
    return PlannerPlan.model_validate(fields)


def _planner_question(args: dict[str, Any]) -> ElicitationQuestion:
    """The planner's ask as the frame's one question, `q1`. Its
    `adk_request_input` call args are the question directly: `message`,
    `candidates`, `options`, `recommended_spec_id` and `rationale`.
    `options` are fixed choices that are no source, such as whether to add
    two datasets (`planner.agent.COMBINATION_OPTIONS`); the planner reads
    only a pick of one, so they take no free text."""
    candidates = args.get("candidates") or []
    fixed = args.get("options") or []
    recommended_spec_id = args.get("recommended_spec_id")
    options = [
        ElicitationOption(
            id=str(candidate.get("spec_id", "")),
            label=str(candidate.get("label", candidate.get("spec_id", ""))),
            recommended=bool(recommended_spec_id)
            and str(candidate.get("spec_id", "")) == recommended_spec_id,
        )
        for candidate in candidates
    ] + [ElicitationOption(id=str(option["id"]), label=str(option["label"])) for option in fixed]
    return ElicitationQuestion(
        id="q1",
        question=str(args.get("message", "Which reading did you mean?")),
        input=ElicitationInput.SINGLE_SELECT,
        options=options,
        allow_free_text=not fixed,
        rationale=(
            str(args["rationale"])
            if args.get("rationale") and any(option.recommended for option in options)
            else None
        ),
    )


class _Translator:
    """One turn segment's ADK-event-to-wire-frame translation.

    Stateful over the segment: `paused` is set when a pausing frame
    (plan.proposed / elicitation.request) is emitted, which tells the route
    the stream must end without `done` (contract delta 1).

    The agent's tool calls run through the recording executor and surface,
    so their steps and cards arrive through `_drain_recorder`; the plan
    path's events (the planner's status lines, the executor's step
    progress, the approval and clarify interrupts) go to
    `_plan_path_frames`. On top of those, the agent's own events stream as
    they happen:

    - one running status line (an `activity.step` with no `tool`) whenever
      the agent works and no tool step runs. After a batch of tool results
      it names what the model reads ("Reading what the search found",
      "Looking at <measure>, <years>", "Going over the charts", "Reading
      the earlier turn", "Going over the notes on DFO"), with the limits those
      results state as `detail`, and "Working out the next step" after a
      refused hand-off; a batch that mixes tools is named by its latest
      stage. A successful plan hand-off opens "Working on a plan", a
      successful ask hands over to the question, and neither opens a
      reading line. The line completes, with `duration_ms`, when anything else
      happens: a tool call, answer text, a hand-off, the segment's end. A skill
      read keeps it running. Thought text never shows;
    - each tool call, as an `activity.step` labelled from its arguments the
      moment the model makes it, completed when the tool returns. A
      hand-off to the planner and a skill read show only when they fail:
      the planner's own lines follow a hand-off, and a skill read is
      background, not progress;
    - answer text, as `message.delta` chunks, the last model call's text
      being the turn's `message.complete`: text before a tool call is
      interim, and text after a hand-off to the user or the planner would
      sit above the question or the plan card, so neither is the answer;
    - an orchestrator node's own status line ("Reading your question") as a
      running step until the next thing happens; a plan hand-back shows a
      failed step and then "Reading why no plan was made" when the planner
      made no plan, and "Reading the plan's results" when the plan ran,
      opened at the execute node's result so it covers `plan_done` too;
    - a `complete` frame on a failed step's own wire id once a later try of
      the same step succeeds (`_retry_key`): the same tool with the same
      arguments, a later hand-off the planner takes for a failed hand-off,
      and a plan made, ready for approval, for either plan failure.
    """

    def __init__(
        self,
        *,
        conversation: ConversationState,
        recorder: _RecordingExecutor,
        surface_recorder: _RecordingSurface | None = None,
    ) -> None:
        self._conversation = conversation
        self._recorder = recorder
        self._surface_recorder = surface_recorder
        # The wire `step_id` each step (`_ExecutorRecord.step_id`) was given
        # at its first frame, so its later frames update the same line.
        self._wire_step_ids: dict[str, str] = {}
        self._final_text = ""
        self.paused = False
        # The running status line, as (wire id, label, monotonic start).
        self._status: tuple[str, str, float] | None = None
        self._streamed_text = False
        # Each open call's wire step id, label, arguments and monotonic
        # start, by call id.
        self._open_calls: dict[str, tuple[str, str, dict[str, Any], float]] = {}
        # Calls that show only if they fail, by call id: the tool's name and
        # arguments.
        self._quiet_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        # The failed steps a later try can still recover, by `_retry_key`:
        # each one's wire step id, label and tool.
        self._failed: dict[tuple[str, str], list[tuple[str, str, str | None]]] = {}
        # The kind of hand-off ("plan", "ask") the agent's tools recorded.
        self._handed_off: str | None = None
        # The highest composition revision this stream has sent.
        self._sent_revision = 0

    def _plan_path_frames(self, event: Event) -> list[Any]:
        """Frames for an event of the plan path: the planner's status lines,
        a plan step's progress, the approval and clarify interrupts."""
        step_progress = (
            event.custom_metadata.get(STEP_PROGRESS_KEY) if event.custom_metadata else None
        )
        if isinstance(step_progress, dict):
            # Mid-call: records wait for the node's own result event. A
            # report can arrive after the call it describes has returned and
            # been recorded, so draining here could close a step's line
            # before this frame reopens it; and `harness.graph`'s `execute`
            # may still amend a finished step's record
            # (`_RecordingSurface.mark_step_failed`) once the whole run is over.
            return [*self._close_starting(), self._running_step(step_progress)]
        frames: list[Any] = list(self._drain_recorder())
        interrupt = _request_input_call(event)
        if interrupt is not None:
            adk_id, args = interrupt
            frames.extend(self._close_status())
            if adk_id == APPROVE_INTERRUPT:
                frames.extend(self._recover(_retry_key("plan", {}), _PLAN_MADE))
            frames.extend(self._interrupt_frames(adk_id, args))
            return frames
        if event.author == PLANNER_AGENT_NAME and not event.partial:
            for call in event.get_function_calls():
                frames.extend(self._planner_call_status(call.name or ""))
            for response in event.get_function_responses():
                if response.name == PROPOSE_PLAN:
                    recorded = event.actions.state_delta.get(PLAN_STATE_KEY)
                    frames.extend(self._proposal_answered(response.response or {}, recorded))
            return frames
        node = _node_name(event)
        # The planner node's other `custom_metadata` events, the seam's
        # late `progress` reports, carry no output and add no frame.
        started = event.custom_metadata.get(PLANNER_STARTED_KEY) if event.custom_metadata else None
        if node == "planner" and isinstance(started, dict):
            label = "Revising the plan" if started.get("revising") else _WORKING_ON_PLAN
            frames.extend(self._open_status(label))
        output = event.output if isinstance(event.output, dict) else None
        if output is None:
            return frames
        if node == "approve" and event.actions.route == "approved":
            frames.extend(self._recover(_retry_key("plan", {}), _PLAN_MADE))
            frames.extend(self._open_status(_STARTING_PLAN))
        if node == "planner":
            frames.extend(self._close_status())
        return frames

    def _planner_call_status(self, name: str) -> list[ActivityStep]:
        """The status line for one of the planner's own calls. A
        `propose_plan` call opens the checking line, since the tool translates
        the plan it records before it answers. Any other call leaves the
        current line running."""
        if name == PROPOSE_PLAN:
            return self._open_status(_CHECKING)
        return []

    def _proposal_answered(self, answer: dict[str, Any], recorded: object) -> list[ActivityStep]:
        """The line change a `propose_plan` response makes: an accepted
        proposal (`ok`) completes the checking line with the
        data of the plan it recorded as `detail` (`recorded` is the plan's
        JSON from the response's state delta), and a refused one (it recorded
        nothing, so ran no translation) makes the planner's redraft a separate
        "Reworking the draft" line. The refusal's code shows in plain words;
        its message, which names plan fields, never reaches the wire."""
        outcome = answer.get("outcome")
        if outcome == Outcome.OK:
            detail = (
                _data_phrase(PlannerPlan.model_validate_json(recorded))
                if isinstance(recorded, str)
                else None
            )
            return self._close_status(detail=detail)
        return self._open_status(_REWORKING, detail=_REWORK_REASONS.get(answer.get("code") or ""))

    def _open_status(self, label: str, *, detail: str | None = None) -> list[ActivityStep]:
        """Start a `running` status line for `label`, completing the one
        before it. The same label again keeps its line running."""
        if self._status is not None and self._status[1] == label:
            return []
        frames = self._close_status()
        seq = self._conversation.next_seq()
        step_id = self._wire_step_id(None, seq=seq)
        self._status = (step_id, label, time.monotonic())
        frames.append(
            ActivityStep(
                event="activity.step",
                seq=seq,
                step_id=step_id,
                label=label,
                status=StepStatus.RUNNING,
                detail=detail,
            )
        )
        return frames

    def _close_starting(self) -> list[ActivityStep]:
        """Complete "Starting the plan" ahead of the plan's first step frame."""
        if self._status is not None and self._status[1] == _STARTING_PLAN:
            return self._close_status()
        return []

    def _close_status(self, *, detail: str | None = None) -> list[ActivityStep]:
        """Complete the running status line, if there is one. The
        `duration_ms` is what tells Studio the line is done; `detail` is the
        summary Studio shows for it."""
        if self._status is None:
            return []
        step_id, label, started = self._status
        self._status = None
        return [
            ActivityStep(
                event="activity.step",
                seq=self._conversation.next_seq(),
                step_id=step_id,
                label=label,
                status=StepStatus.COMPLETE,
                duration_ms=int((time.monotonic() - started) * 1000),
                detail=detail,
            )
        ]

    def _wire_step_id(self, step: str | None, *, seq: int) -> str:
        """The wire `step_id` for `step`, minted from `seq`, the `seq` of its
        first frame; a fresh one for `None`.

        Studio resumes a paused turn into the same Activity entry and matches
        its lines by `step_id` alone, so an id must be unique across every
        segment of the turn. `seq` is: `ConversationState.seq` restarts only
        at `begin_turn`, and a resumed segment continues it.
        """
        if step is not None and step in self._wire_step_ids:
            return self._wire_step_ids[step]
        wire_id = f"s{seq}"
        if step is not None:
            self._wire_step_ids[step] = wire_id
        return wire_id

    def _running_step(self, progress: dict[str, Any]) -> ActivityStep:
        """A step's progress report (`harness.graph.execute._step_progress_event`) as a
        `running` frame on the step's own wire id, with the report's message
        as `detail`. The label matches the one the step's final frame
        carries (`_drain_recorder`), since the Activity feed names a line
        from its first frame."""
        tool = str(progress.get("tool") or "")
        message = progress.get("message")
        seq = self._conversation.next_seq()
        plan_labels = (
            self._surface_recorder.step_labels(current_conversation_id.get())
            if self._surface_recorder is not None
            else {}
        )
        return ActivityStep(
            event="activity.step",
            seq=seq,
            step_id=self._wire_step_id(str(progress.get("step_id")), seq=seq),
            label=_step_label(tool, plan_labels.get(str(progress.get("step_id")))),
            status=StepStatus.RUNNING,
            tool=tool or None,
            detail=message if isinstance(message, str) else None,
        )

    def _drain_recorder(self) -> list[Any]:
        """Frames for every call recorded since the last drain."""
        return self._record_frames(self._take_records())

    def _take_records(self) -> list[_ExecutorRecord]:
        """Every call recorded since the last drain, plan steps first."""
        conversation = current_conversation_id.get()
        records = list(self._recorder.drain(conversation))
        if self._surface_recorder is not None:
            # Plan steps (the surface) precede the executor's own calls
            # within any one segment, so this ordering matches
            # what actually ran; the two never both hold records at once in
            # practice, since draining happens between every ADK event. A
            # wave's steps finish, and get recorded, in whatever order the
            # event loop schedules them (`executor.run._Run.execute` gathers
            # a wave concurrently), so the surface's own records are
            # stable-sorted into the latest plan's step order before
            # numbering; a record whose `step_id` that order does not name
            # sorts after every named one, in the order it was recorded.
            surface_records = self._surface_recorder.drain(conversation)
            order = self._surface_recorder.step_order(conversation)
            position = {step_id: index for index, step_id in enumerate(order)}
            surface_records = sorted(
                surface_records, key=lambda record: position.get(record.step_id, len(order))
            )
            records = [*surface_records, *records]
        return records

    def _record_frames(self, records: list[_ExecutorRecord]) -> list[Any]:
        """Frames for `records`, after one `composition.updated` for each
        removal since the last drain. Every record of one step, retries
        included, shares that step's wire id."""
        plan_labels = (
            self._surface_recorder.step_labels(current_conversation_id.get())
            if self._surface_recorder is not None
            else {}
        )
        frames: list[Any] = self._close_starting() if records else []
        # A removal changed the canvas as its tool ran, before any landing
        # below, so its frame goes first. It carries the manifest stored now
        # rather than the one the removal wrote: `/composition` may have
        # committed a newer revision since (`StudioStore.mutate_composition`
        # rebuilds on the removal), and the client renders every update, so
        # an older one would undo that edit on screen. One frame stands for
        # every removal since the last drain.
        if self._conversation.take_unreported_removal():
            frames.extend(self._composition_update())
        for record in records:
            if self._unshown(record):
                continue
            seq = self._conversation.next_seq()
            step_id = self._wire_step_id(record.step_id, seq=seq)
            refused = record.result.get("outcome") == "refusal"
            component = record.result.get("component")
            frames.append(
                ActivityStep(
                    event="activity.step",
                    seq=seq,
                    step_id=step_id,
                    label=_record_label(record, plan_labels),
                    status=StepStatus.FAILED if refused else StepStatus.COMPLETE,
                    tool=record.tool,
                    duration_ms=record.duration_ms,
                    detail=_refused_detail(record) if refused else _record_detail(record),
                    produced=(
                        StepProduced(
                            component_id=component["component_id"],
                            revision=component["revision"],
                        )
                        if isinstance(component, dict)
                        else None
                    ),
                )
            )
            if isinstance(component, dict):
                frames.extend(
                    self._component_frames(
                        component,
                        step_id=step_id,
                        land=record.land,
                        intent=record.args.get("intent"),
                    )
                )
        return frames

    def _component_frames(
        self,
        component: dict[str, Any],
        *,
        step_id: str,
        land: bool,
        intent: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Validate `component` against the `component.created` wire
        contract, then, only for a component stamped `land=True`
        (`_RecordingExecutor.land_component`, called by `make_charts` right
        after the write returned), land it: append its block to the canvas
        and keep the `intent` it was written from with it
        (`harness.studio_state.land_component`). A component that fails
        validation here never lands and produces no frame at all. For an
        unstamped record (`land=False`) this is the expected outcome:
        `make_charts` already refused this write because the same component
        failed `harness.wire.usable_component`, so no frame is the right
        result and this is not logged. For a record stamped `land=True`,
        `make_charts` already ran `usable_component` on this exact component
        and it passed, so a validation failure here means the wire check and
        this method's own contract have diverged: a real bug, logged at
        ERROR. `land=False` with a valid component (a bare executor with no
        `land_component` capability at all) means no block exists to report:
        only `component.created` goes out. `land=True` against an
        `auto_add=False` conversation resolves to `"held"` inside
        `harness.studio_state.land_component` itself, also only
        `component.created`."""
        try:
            created = ComponentCreated.model_validate(
                {
                    **component,
                    "event": "component.created",
                    "seq": self._conversation.next_seq(),
                    "created_from": component.get("created_from") or {"step_id": step_id},
                }
            )
        except ValidationError:
            if land:
                logger.error(
                    "a component stamped land=True (make_charts' usable_component check "
                    "passed) failed component.created validation at drain time; "
                    "make_charts' wire check and this validation have diverged"
                )
            return []
        if not land:
            return [created]
        if state_land_component(self._conversation, component, intent=intent) != "created":
            return [created]
        # Read with no await since the landing, so this is the manifest it wrote.
        return [created, *self._composition_update()]

    def _composition_update(self) -> list[CompositionUpdated]:
        """The `composition.updated` frame for the conversation's stored
        manifest, or none when it stores none.

        The manifest is read from the conversation the store holds
        (`StudioStore.current`), which every writer assigns to, so its
        revision never falls along a stream. A manifest below a revision this
        stream already sent shows a read that did not come from the store: it
        is logged and not sent, since a client renders every update.
        """
        manifest = self._conversation.composition
        if manifest is None:
            return []
        if manifest.revision < self._sent_revision:
            logger.error(
                "composition.updated withheld: revision %d is below revision %d, which "
                "this stream already sent",
                manifest.revision,
                self._sent_revision,
            )
            return []
        self._sent_revision = manifest.revision
        return [
            CompositionUpdated(
                event="composition.updated", seq=self._conversation.next_seq(), manifest=manifest
            )
        ]

    def _interrupt_frames(self, adk_id: str, args: dict[str, Any]) -> list[Any]:
        """Every wire model either branch below returns is `extra="forbid"`,
        so building one is real work that can raise. Both branches write
        `conversation.pending[wire_id]` only after every model they are
        about to return has already been built successfully: written any
        earlier, a raise mid-construction would register an interrupt the
        client was never shown, an orphan nothing can ever answer (the
        mirrored rule on delete, at `harness.studio`'s `/interrupt` route,
        keeps a record from being lost the same way)."""
        conversation = self._conversation
        payload = args.get("payload") or {}
        wire_id = _wire_id("int")
        self.paused = True
        if adk_id == APPROVE_INTERRUPT:
            frames: list[Any] = []
            note = payload.get("refusal_note")
            if isinstance(note, str) and note:
                # `harness.graph.planning`'s `planner` node sets this in two cases. Its
                # `"pending_refusal"` route (BUDGET_EXHAUSTED or
                # QUESTION_FIXED), a refused amendment that still has a
                # `PENDING_APPROVAL` plan on record re-offered unmodified,
                # carries the refusal's sentence: the plan card below says
                # what is still approvable, and this says why the user is
                # seeing it again rather than the change they just asked
                # for. An `"ok"` draft made over a plan the node could not
                # read says so (`UNREADABLE_PLAN_NOTE`). Any other `"ok"`
                # draft carries no `refusal_note` and gets no message here.
                frames.append(
                    MessageDelta(event="message.delta", seq=conversation.next_seq(), text=note)
                )
                frames.append(
                    MessageComplete(
                        event="message.complete", seq=conversation.next_seq(), text=note
                    )
                )
            version = int(payload.get("version", 1))
            latest = conversation.latest_plan_version
            restarted_from: str | None = None
            if payload.get("restarted"):
                # `harness.graph.planning`'s `planner` node drafted this plan from the
                # start, over a plan it could not read, so its version
                # counts from 1 on a line of its own. The ids given out so
                # far name versions of the plan it replaces: they are
                # forgotten, and the last card shown is superseded below.
                if latest is not None:
                    restarted_from = conversation.plan_version_ids[latest]
                conversation.plan_version_ids.clear()
                conversation.latest_plan_version = latest = None
            if latest is not None and version < latest:
                # An amendment or a replan raises the version (see plan.py's
                # `Plan.amend`), and a plan drafted over an unreadable one
                # arrives `restarted` and is handled above. A lower version
                # arriving here is a bug elsewhere, not a real re-offer, so
                # this plan gets a fresh wire id rather than `setdefault`
                # handing back the stale one already given out for that
                # version number, and `latest_plan_version` does not move
                # backwards.
                logger.error(
                    "studio plan version regressed: version=%s < latest_plan_version=%s",
                    version,
                    latest,
                )
                plan_version_id = _wire_id("plan")
                conversation.plan_version_ids[version] = plan_version_id
            else:
                plan_version_id = conversation.plan_version_ids.setdefault(
                    version, _wire_id("plan")
                )
                supersedes = (
                    conversation.plan_version_ids[latest]
                    if latest is not None and version > latest
                    else restarted_from
                )
                if supersedes is not None:
                    frames.append(
                        PlanVersioned(
                            event="plan.versioned",
                            seq=conversation.next_seq(),
                            plan_version_id=plan_version_id,
                            version=version,
                            supersedes=supersedes,
                        )
                    )
                conversation.latest_plan_version = version
            plan = _plan_from_payload(payload)
            shown_slots = {decision.slot for decision in plan_display.shown_decisions(plan)}
            named = functools.partial(plan_display.with_source_names, plan=plan)
            revision = payload.get("revision")
            steps = [
                PlanStep(
                    id=str(row["id"]),
                    label=str(row["label"]),
                    description=str(row["description"]),
                    args=cast("dict[str, Any] | None", row["args"]),
                )
                for row in plan_display.step_rows(plan)
            ]
            # `plan.workflow.steps`' own ids are what `executor.run.
            # current_step_id` sets and `_ExecutorRecord.step_id` carries
            # (`plan_display.step_rows` reads the same ids off the same
            # steps to build `steps` above); the fallback covers a plan
            # whose workflow this payload's `_plan_from_payload` parse
            # somehow left unset. Recorded on the pending interrupt below so
            # `/interrupt` can restore it on whichever instance resumes the
            # turn, even a different one than the one that built this frame.
            step_order = (
                [step.id for step in plan.workflow.steps]
                if plan.workflow is not None
                else [step.id for step in steps]
            )
            step_labels = {step.id: step.label for step in steps}
            if self._surface_recorder is not None:
                self._surface_recorder.set_step_order(
                    current_conversation_id.get(), step_order, step_labels
                )
            requirements = [
                PlanRequirement(**cast("dict[str, Any]", row))
                for row in plan_display.requirement_rows(plan)
            ]
            operations = [
                PlanOperation(**cast("dict[str, Any]", row))
                for row in plan_display.operation_rows(plan)
            ]
            asks = [
                PlanAsk(
                    ask=str(row["ask"]),
                    answered_by=cast("list[str]", row["answered_by"]),
                    declined=cast("str | None", row["reason"]),
                    not_computed=cast("str | None", row["not_computed"])
                    if row["answered_by"]
                    else None,
                )
                for row in plan_display.ask_rows(plan)
            ]
            frames.append(
                PlanProposed(
                    event="plan.proposed",
                    seq=conversation.next_seq(),
                    interrupt_id=wire_id,
                    plan_version_id=plan_version_id,
                    version=version,
                    title="Research plan",
                    steps=steps,
                    decisions=[
                        PlanDecision(
                            id=decision.get("slot", f"d{index + 1}"),
                            question=named(decision.get("description", "")),
                            chosen=named(decision.get("chosen", "")),
                            alternatives=(
                                [named(option) for option in decision["options"]]
                                if decision.get("options") is not None
                                else None
                            ),
                            reason=named(decision.get("rationale", "")),
                            resolved_by=_RESOLVED_BY_TO_WIRE.get(
                                decision.get("resolved_by", ""), "default"
                            ),
                        )
                        for index, decision in enumerate(payload.get("decisions", []))
                        if decision.get("slot") in shown_slots
                    ],
                    actions=[PlanAction.APPROVE, PlanAction.AMEND, PlanAction.REJECT],
                    revision=revision,
                    requirements=requirements,
                    operations=operations,
                    asks=asks or None,
                )
            )
            conversation.pending[wire_id] = _PendingInterrupt(
                adk_id=adk_id,
                kind="plan",
                plan_version=version,
                plan_revision=revision,
                step_order=tuple(step_order),
                step_labels=step_labels,
            )
            return frames
        # The workflow's own `ask` node raises a `RequestInput` whose `payload`
        # carries one to three `questions`, each with its `candidates`. A
        # lone question is dismissed with the whole pause. Each of several
        # may be left to the agent instead. The planner's `adk_request_input`
        # call asks one question (`_planner_question`).
        asked = payload.get("questions")
        if asked:
            several = len(asked) > 1
            questions = [
                ElicitationQuestion(
                    id=str(question["id"]),
                    question=str(question["question"]),
                    input=ElicitationInput.SINGLE_SELECT,
                    options=[
                        ElicitationOption(
                            id=str(candidate["spec_id"]), label=str(candidate["label"])
                        )
                        for candidate in question["candidates"]
                    ],
                    allow_free_text=True,
                    skippable=several,
                )
                for question in asked
            ]
            skippable = not several
        else:
            questions = [_planner_question(args)]
            skippable = True
        frame = ElicitationRequest(
            event="elicitation.request",
            seq=conversation.next_seq(),
            interrupt_id=wire_id,
            questions=questions,
            skippable=skippable,
        )
        conversation.pending[wire_id] = _PendingInterrupt(
            adk_id=adk_id,
            kind="elicitation",
            questions=tuple(
                _PendingQuestion(
                    id=question.id,
                    question=question.question,
                    options={option.id: option.label for option in question.options or ()},
                    skippable=question.skippable,
                )
                for question in questions
            ),
            skippable=skippable,
        )
        return [frame]

    def _step(
        self, step_id: str | None, label: str, status: StepStatus, **extra: Any
    ) -> ActivityStep:
        """A step frame; with no `step_id`, a fresh one minted from the frame's
        own seq, unique across every segment of the turn (`_wire_step_id`)."""
        seq = self._conversation.next_seq()
        return ActivityStep(
            event="activity.step",
            seq=seq,
            step_id=step_id or self._wire_step_id(None, seq=seq),
            label=label,
            status=status,
            **extra,
        )

    def _failed_step(self, key: tuple[str, str], frame: ActivityStep) -> ActivityStep:
        """`frame`, a failed step, kept under `key` for a later try to recover."""
        self._failed.setdefault(key, []).append((frame.step_id, frame.label, frame.tool))
        return frame

    def _recover(self, *keys: tuple[str, str]) -> list[ActivityStep]:
        """A `complete` frame on each failed step kept under `keys`, which a
        later try just made good. Studio keeps a step's latest status, so the
        failed line settles instead of standing beside the answer."""
        return [
            self._step(step_id, label, StepStatus.COMPLETE, tool=tool, detail=_RECOVERED)
            for key in keys
            for step_id, label, tool in self._failed.pop(key, [])
        ]

    def _unshown(self, record: _ExecutorRecord) -> bool:
        """A successful call no plan step made, to a tool whose work an
        agent tool call's own line already stands for: the data a fetch
        retrieved and checked, a skill it read, or a card `make_charts` read
        back. Failures always show."""
        return (
            record.step_id is None
            and record.tool in _UNSHOWN_RECORDED_TOOLS
            and record.result.get("outcome") != "refusal"
        )

    def translate(self, event: Event) -> list[Any]:
        # A paused segment ends on its pause frame; the planner relays its
        # queued progress after pausing.
        if self.paused:
            return []
        status = (
            event.custom_metadata.get(ORCHESTRATOR_STATUS_KEY) if event.custom_metadata else None
        )
        if isinstance(status, str):
            # An agent run starts: whatever it wrote before was interim.
            self._final_text = ""
            self._handed_off = None
            frames = list(self._drain_recorder())
            if status == ORCHESTRATOR_PLAN_FAILED_STATUS:
                frames.extend(self._close_status())
                failure = (event.custom_metadata or {}).get(ORCHESTRATOR_STATUS_FAILURE_KEY)
                code = failure.get("code") if isinstance(failure, Mapping) else None
                detail = failure_detail({"code": code})
                frames.append(
                    self._failed_step(
                        _PLAN_MADE, self._step(None, _PLAN_FAILED, StepStatus.FAILED, detail=detail)
                    )
                )
                # The agent reads the planner's answer under its own line.
                frames.extend(self._open_status(status))
            else:
                # A plan hand-back's status is the line the execute node's
                # result opened, which keeps running.
                frames.extend(self._open_status(status))
            return frames
        if event.author != ORCHESTRATOR_AGENT_NAME:
            frames = self._plan_path_frames(event)
            if (
                _node_name(event) == "execute"
                and event.output is not None
                and event.actions.route == "ok"
            ):
                # The plan ran: `plan_done` describes its frames, then the
                # agent reads them.
                frames.extend(self._open_status(ORCHESTRATOR_PLAN_RAN_STATUS))
            return frames
        records = self._take_records()
        frames: list[Any] = self._record_frames(records)
        parts = event.content.parts if event.content and event.content.parts else []
        text = "".join(part.text or "" for part in parts if part.text and not part.thought)
        if event.partial:
            if text and not self._handed_off:
                frames.extend(self._close_status())
                self._streamed_text = True
                self._final_text += text
                frames.append(
                    MessageDelta(
                        event="message.delta", seq=self._conversation.next_seq(), text=text
                    )
                )
            return frames
        # A finished model call: its text arrived already as partials unless
        # the model sent it in one piece.
        if text and not self._streamed_text and not self._handed_off:
            frames.extend(self._close_status())
            self._final_text += text
            frames.append(
                MessageDelta(event="message.delta", seq=self._conversation.next_seq(), text=text)
            )
        self._streamed_text = False
        calls = event.get_function_calls()
        if calls and not self._handed_off:
            self._final_text = ""
            # A skill read shows no step, so the line it ran under keeps running.
            if any(call.name not in _SKILL_READS for call in calls):
                frames.extend(self._close_status())
        for call in event.get_function_calls():
            name = call.name or ""
            args = call.args or {}
            if name in _QUIET_CALLS:
                self._quiet_calls[call.id or ""] = (name, args)
                continue
            measure = self._measure(args.get("spec_id")) if name == "fetch" else None
            label = _tool_call_label(name, args, measure)
            query = args.get("query") if name == "search" else None
            frame = self._step(
                None, label, StepStatus.RUNNING, tool=name, detail=str(query) if query else None
            )
            self._open_calls[call.id or frame.step_id] = (
                frame.step_id,
                label,
                args,
                time.monotonic(),
            )
            frames.append(frame)
        # What the agent's next model call reads: (tool, arguments, result,
        # failed) for each response a status line can name.
        read: list[tuple[str, dict[str, Any], dict[str, Any], bool]] = []
        for response in event.get_function_responses():
            # A hand-off counts once the tool recorded it; a refused or
            # failed one leaves the model to answer, and that text shows.
            answer = response.response or {}
            if answer.get(HANDED_OFF):
                self._handed_off = str(answer[HANDED_OFF])
            # A failure's reason is written for the model, so its line
            # shows the sentence for its code.
            failed = "error" in answer or "refusal" in answer
            quiet = self._quiet_calls.pop(response.id or "", None)
            if quiet is not None:
                name, args = quiet
                if failed:
                    frames.append(
                        self._failed_step(
                            _retry_key(name, args),
                            self._step(
                                None,
                                _tool_call_label(name, {}),
                                StepStatus.FAILED,
                                tool=name,
                                detail=_answer_failure(answer),
                            ),
                        )
                    )
                else:
                    frames.extend(self._recover(_retry_key(name, args)))
                continue
            opened = self._open_calls.get(response.id or "")
            if opened is None:
                continue
            step_id, label, args, started = opened
            if failed:
                status, detail = StepStatus.FAILED, _answer_failure(answer)
            else:
                status, detail = self._outcome(response.name or "", args, answer)
            # Studio closes a `pending` line only on a frame with a duration;
            # every other status closes it by itself.
            timed = (
                {"duration_ms": int((time.monotonic() - started) * 1000)}
                if status is StepStatus.PENDING
                else {}
            )
            frame = self._step(step_id, label, status, tool=response.name, detail=detail, **timed)
            key = _retry_key(response.name or "", args)
            if status is StepStatus.FAILED:
                frames.append(self._failed_step(key, frame))
            else:
                frames.append(frame)
                if status is StepStatus.COMPLETE:
                    frames.extend(self._recover(key))
            if response.name in _READING_STAGES:
                read.append((response.name or "", args, answer, failed))
        if not event.get_function_responses():
            return frames
        if self._handed_off == "plan":
            # The planner node opens this same line as it starts, which
            # keeps this one running through the hand-off.
            frames.extend(self._open_status(_WORKING_ON_PLAN))
        elif self._handed_off is None:
            responded = {response.name or "" for response in event.get_function_responses()}
            frames.extend(self._reading_status(read, records, responded))
        return frames

    def _reading_status(
        self,
        read: list[tuple[str, dict[str, Any], dict[str, Any], bool]],
        records: list[_ExecutorRecord],
        responded: set[str],
    ) -> list[ActivityStep]:
        """The status line the agent's next model call runs under, naming
        what it reads: the batch's latest stage (`_READING_STAGES`), with the
        limits its results state as `detail`. A batch with no reading stage
        reads about DFO after `explain_dfo`, and otherwise, after a refused
        hand-off, works out the next step; a batch of skill reads alone opens
        none, so the line it ran under keeps running. `records` are the calls
        the batch's tools made, drained with its responses; `responded` names
        every tool the batch answered."""
        names = {name for name, *_ in read}
        stage = next((stage for stage in _READING_STAGES if stage in names), None)
        if stage is None:
            if "explain_dfo" in responded:
                return self._open_status(_READING_ABOUT_DFO)
            if responded - _SKILL_READS:
                return self._open_status(_WORKING_OUT_NEXT_STEP)
            return []
        results = [(args, answer, failed) for name, args, answer, failed in read if name == stage]
        if stage == "fetch":
            label, detail = self._looking_at(results, records)
        elif stage == "search":
            label = _READING_LABELS[stage]
            limits = (_search_limits(answer) for _, answer, failed in results if not failed)
            detail = "; ".join(limit for limit in limits if limit)
        elif stage == "recall" and all(_is_canvas_listing(stage, args) for args, *_ in results):
            label, detail = _READING_CANVAS, None
        else:
            label, detail = _READING_LABELS[stage], None
        return self._open_status(label, detail=detail[:_STEP_DETAIL_CAP] if detail else None)

    def _looking_at(
        self,
        fetched: list[tuple[dict[str, Any], dict[str, Any], bool]],
        records: list[_ExecutorRecord],
    ) -> tuple[str, str | None]:
        """The line and detail for a batch of fetches. One fetch names its
        measure, and the years its data covers when its retrieval states
        them, with its summary as detail; several list each measure and
        summary in the detail."""
        if len(fetched) == 1:
            args, answer, failed = fetched[0]
            measure = None if failed else self._measure(args.get("spec_id"))
            if measure is None:
                return _LOOKING_AT_DATA, None
            years = self._fetched_years(records).get(str(args.get("spec_id")))
            label = f"Looking at {measure}, {years}" if years else f"Looking at {measure}"
            summary = answer.get("summary")
            return label, str(summary) if summary else None
        lines = []
        for args, answer, failed in fetched:
            if failed:
                continue
            pieces = (self._measure(args.get("spec_id")), answer.get("summary"))
            lines.append(": ".join(str(piece) for piece in pieces if piece))
        return _LOOKING_AT_DATA, "; ".join(line for line in lines if line) or None

    def _fetched_years(self, records: list[_ExecutorRecord]) -> dict[str, str]:
        """The years each fetch's retrieval in `records` covers, such as 2000-2023,
        by spec_id, read off the manifest the retrieval recorded. Empty with
        no surface recorder to read manifests through."""
        if self._surface_recorder is None:
            return {}
        years: dict[str, str] = {}
        for record in records:
            if (
                record.tool != "get_data"
                or record.step_id is not None
                or record.result.get("outcome") != "ok"
            ):
                continue
            for ref in record.result.get("frame_refs") or []:
                manifest = self._surface_recorder.retrieval_manifest(frame_ref=ref) or {}
                window = (manifest.get("coverage") or {}).get("window") or {}
                start, end = window.get("start_year"), window.get("end_year")
                if isinstance(start, int) and isinstance(end, int):
                    years[str(record.args.get("spec_id"))] = _year_span(start, end)
        return years

    def _outcome(
        self, name: str, args: dict[str, Any], answer: dict[str, Any]
    ) -> tuple[StepStatus, str | None]:
        """A call that returned no error or refusal: its status and detail."""
        if name == "search":
            return StepStatus.COMPLETE, _search_detail(args.get("query"), answer)
        if name == "fetch":
            return StepStatus.COMPLETE, _facts(answer.get("summary"), answer.get("period_note"))
        if name == "make_charts":
            return _charts_outcome(args, answer)
        if name == "explain_dfo":
            return StepStatus.COMPLETE, _explain_detail(answer)
        return StepStatus.COMPLETE, None

    def _measure(self, spec_id: Any) -> str | None:
        """The measure a search in this conversation returned `spec_id` for,
        in the question's words: from this turn's readings, which a resumed
        segment keeps, then earlier turns', newest first."""
        turns = [self._conversation.turn_readings] + [
            turn.get("readings") or [] for turn in reversed(self._conversation.turns)
        ]
        for readings in turns:
            for reading in reversed(readings):
                offered = [reading.get("spec_id")] + [
                    candidate.get("spec_id") for candidate in reading.get("candidates") or []
                ]
                if spec_id in offered and reading.get("text"):
                    return str(reading["text"])
        return None

    def finish(self) -> list[Any]:
        """Frames that close a segment that ran to the end of its events. A
        segment that did not pause closes the turn on the agent's answer,
        followed by one sentence for each loose match the turn used
        (`harness.turn_records.loose_match_notes`), sent as a last
        `message.delta` so the deltas still add up to the `message.complete`
        text."""
        frames: list[Any] = [*self._drain_recorder(), *self._close_status()]
        if self.paused:
            return frames
        if self._final_text:
            notes = loose_match_notes(
                self._conversation.turn_readings,
                specs=self._conversation.turn_specs,
                plans=self._conversation.turn_plans,
            )
            if notes:
                coverage = "\n\n" + " ".join(notes)
                self._final_text += coverage
                frames.append(
                    MessageDelta(
                        event="message.delta", seq=self._conversation.next_seq(), text=coverage
                    )
                )
            self._conversation.turn_final_text = self._final_text
            frames.append(
                MessageComplete(
                    event="message.complete",
                    seq=self._conversation.next_seq(),
                    text=self._final_text,
                )
            )
        self._conversation.close_turn(final_text=self._final_text)
        frames.append(Suggestions(event="suggestions", seq=self._conversation.next_seq(), items=[]))
        frames.append(
            Done(
                event="done",
                seq=self._conversation.next_seq(),
                trace_id=self._conversation.trace_id,
                composition=self._conversation.composition,
            )
        )
        return frames


# The step a plan hand-back with no plan shows, detailed by the sentence the
# failure table has for the refusal's code. The closing message says what
# could not be calculated.
_PLAN_FAILED = "No plan could be made"

# The `detail` of a failed step a later try of the same step made good.
_RECOVERED = "This worked on a later try."
# The key of a plan hand-back with no plan, which a later plan made recovers.
_PLAN_MADE = ("plan", "made")


def _retry_key(name: str, args: Mapping[str, Any]) -> tuple[str, str]:
    """What makes a later call the same step tried again: the same tool with
    the same arguments. A plan hand-off is keyed by its tool alone, since a
    turn plans one question and the agent rewords its brief on a retry."""
    if name == "plan":
        return (name, "")
    return (name, json.dumps(args, sort_keys=True, default=str))


# Agent tools whose calls show only when they fail.
_QUIET_CALLS = frozenset({"plan", "get_skill", "get_skill_file"})
_SKILL_READS = frozenset({"get_skill", "get_skill_file"})
# The tools whose results the agent's next model call reads under a status
# line, latest stage first: a batch that mixes them is named by the first.
_READING_STAGES = ("make_charts", "fetch", "search", "recall")
_READING_LABELS = {
    "make_charts": "Going over the charts",
    "search": "Reading what the search found",
    "recall": "Reading the earlier turn",
}
_LOOKING_AT_DATA = "Looking at the data"
_READING_CANVAS = "Going over your canvas"
_READING_ABOUT_DFO = "Going over the notes on DFO"
_WORKING_OUT_NEXT_STEP = "Working out the next step"
# Recorded calls an agent tool makes on its own behalf (`fetch` retrieves
# and checks its data, a skill read goes through the surface, `make_charts`
# reads back a card on the canvas), shown only when they fail.
_UNSHOWN_RECORDED_TOOLS = frozenset(
    {"get_data", "describe_frame", "describe_component", "get_skill", "get_skill_file"}
)

_ORCHESTRATOR_TOOL_LABELS = {
    "search": "Searching for data",
    "fetch": "Fetching data",
    "ask_user": "Asking you a question",
    "plan": _WORKING_ON_PLAN,
    "explain_dfo": "Reading about DFO",
    "recall": "Looking back at an earlier turn",
    "remove_cards": "Removing cards from the canvas",
    "get_skill": "Reading background notes",
    "get_skill_file": "Reading background notes",
}


def _tool_call_label(name: str, args: dict[str, Any], measure: str | None = None) -> str:
    """A status line for one of the orchestrator's tool calls, from its
    arguments; a fetch names `measure`, what a search said its spec_id reads
    as."""
    if name == "fetch" and measure:
        return f"Fetching {measure}"
    if name == "make_charts":
        count = len(args.get("charts") or [])
        return f"Making {count} chart{'s' if count != 1 else ''}"
    if _is_canvas_listing(name, args):
        return "Looking at your canvas"
    return _ORCHESTRATOR_TOOL_LABELS.get(name) or _step_label(name)


def _is_canvas_listing(name: str, args: dict[str, Any]) -> bool:
    """A `recall` of the whole canvas, not of an earlier turn or one card."""
    return name == "recall" and args.get("part") == "canvas" and not args.get("component_id")


def _search_detail(query: Any, answer: dict[str, Any]) -> str | None:
    """A search's result in words: the query, what it found (a definite
    part by the measure it resolved to), and what it did not find with the
    reason Search gave."""
    found: list[str] = []
    missed: list[str] = []
    for part in answer.get("parts") or []:
        variable = str(part.get("variable") or "a measure")
        status = part.get("status")
        if status == "definite":
            found.append(str(part.get("resolved") or variable))
        elif status == "candidates":
            found.append(f"{variable} (more than one reading)")
        elif status == "no_data":
            missed.append(f"no data for {variable}")
        elif status == "unsupported":
            missed.append(f"no supported dataset for {variable}")
        else:
            missed.append(f"no match for {variable}")
    outcome = []
    if found:
        outcome.append("Found " + ", ".join(found))
    outcome.extend(missed)
    if not answer.get("parts") and answer.get("kind") not in (None, "resolution"):
        outcome.append(str(answer["kind"]))
    words = "; ".join(outcome)
    text = " · ".join(str(piece) for piece in (query, words) if piece)
    return text[:_STEP_DETAIL_CAP] or None


def _search_limits(answer: dict[str, Any]) -> str | None:
    """What a search result says limits the answer, in the words it already
    holds: each found measure's years, the parts with more than one reading,
    and the parts with no data. Codes, ids and the resolver's warnings,
    which can name dcids, stay out."""
    limits: list[str] = []
    for part in answer.get("parts") or []:
        variable = str(part.get("variable") or "a measure")
        status = part.get("status")
        if status == "definite":
            window = part.get("window") or {}
            start, end = window.get("start"), window.get("end")
            if isinstance(start, int) and isinstance(end, int):
                limits.append(f"{part.get('resolved') or variable}, {_year_span(start, end)}")
        elif status == "candidates":
            count = len(part.get("candidates") or [])
            limits.append(f"{variable}: {count} possible readings")
        elif status == "no_data":
            limits.append(f"no data for {variable}")
        elif status == "unsupported":
            limits.append(f"no supported dataset for {variable}")
    return "; ".join(limits) or None


def _year_span(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}\u2013{end}"


def _explain_detail(answer: dict[str, Any]) -> str | None:
    """The catalogue counts an `explain_dfo` answer carries, shown where the
    user and the Studio eval's grounding check read a step's facts, or
    `None` when it carries none."""
    counts = answer.get("counts")
    return None if counts in (None, COUNTS_UNAVAILABLE) else str(counts)


def _cut_sentence(cut: Mapping[str, Any]) -> str:
    """What a cut card draws and omits, from the counts its result
    states in `holds.cut` (`harness.reply_figures.card_holds`): how many
    members it draws of how many with a value at its period, then how many
    were left out as totals, unallocated amounts or other levels and how many
    for having no value there, each named only when it is more than none."""
    left_out = [
        f"{count:,} {what}"
        for count, what in (
            (cut.get("other_kinds"), "totals, unallocated amounts or other levels"),
            (cut.get("unvalued"), "with no value"),
        )
        if count
    ]
    drawn = (
        f"The chart shows the {cut['drawn']:,} largest of {cut['with_a_value']:,} "
        f"with a value in {cut['period']}"
    )
    return f"{drawn} and omits {' and '.join(left_out)}." if left_out else f"{drawn}."


def _charts_outcome(args: dict[str, Any], answer: dict[str, Any]) -> tuple[StepStatus, str | None]:
    """A make_charts result's status: failed when any chart asked for was not
    drawn, either refused or past the per-call limit.

    A call that drew every chart states what each cut card leaves out
    (`_cut_sentence`), one sentence a card in the order drawn, and nothing
    when no card was cut.

    A failed call's detail gives each reason once, as the sentence for the
    chart's code, then the titles it holds for, reasons in the order their
    first chart was asked: "The card could not be drawn as asked: Aid by
    donor; Aid by sector". A refusal's own message is written for the model,
    so it stays out. The first reason leads because the Activity log shows a
    detail's first `MAX_SENTENCE_CHARS` characters, and the detail is cut to
    that many here so the cut falls in a title."""
    asked = args.get("charts") or []
    charts = answer.get("charts") or []
    refused = [chart for chart in charts if "error" in chart]
    if not refused and not answer.get("not_drawn"):
        cuts = ((chart.get("holds") or {}).get("cut") for chart in charts)
        return StepStatus.COMPLETE, _facts(" ".join(_cut_sentence(cut) for cut in cuts if cut))
    reasons: dict[str, list[Any]] = {}
    for chart in refused:
        reasons.setdefault(failure_detail(chart), []).append(chart)
    if answer.get("not_drawn"):
        past = harness_refusal(code=HarnessCode.CHART_NOT_DRAWN, message=str(answer["not_drawn"]))
        reasons.setdefault(failure_detail(past), []).extend(asked[len(charts) :])
    titles = {
        sentence: [str(c["title"]) for c in held if isinstance(c, Mapping) and c.get("title")]
        for sentence, held in reasons.items()
    }
    detail = "; ".join(
        f"{sentence.removesuffix('.')}: {'; '.join(named)}" if named else sentence.removesuffix(".")
        for sentence, named in titles.items()
    )
    detail = " ".join(detail.split())
    if len(detail) > MAX_SENTENCE_CHARS:
        detail = detail[: MAX_SENTENCE_CHARS - 1] + "\u2026"
    return StepStatus.FAILED, detail
