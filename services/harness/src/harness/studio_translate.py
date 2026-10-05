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

While a call runs, `execute` and `fetch` relay each progress report the
call sends as its own event (`harness.graph.STEP_PROGRESS_KEY` on
`custom_metadata`), which becomes a `running` `activity.step` whose
`detail` is the report's message. A plan step keeps one wire `step_id`
from its first report to its final `complete` or `failed` frame, so the
Activity feed holds one line per step and updates it in place.

The planner node has no such recorder -- its work happens inside a
`PlannerModel` seam's own model round trips, not through the tool surface.
Its agent's function-call events reach the runner stream as each call is
made, so each call it makes becomes a status line ("Drafting a plan",
"Checking the plan against the data", ...) the moment it happens, after
the "Working on a plan" line the node opens as a draft starts. One
status line runs at a time; the next one, the plan card, a pause or the
segment's end completes it. The seam's own `on_progress` reports, which
the node can only relay once the draft is done (`harness.graph`'s
`planner` node), say the same thing late and are not shown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from common.mcp_telemetry import WRITE_TOKEN_KEY
from common.refusal import refusal_code
from compiler.surface import (
    ProgressCallback,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
    ToolSurfaceUnavailableError,
)
from executor.run import current_step_id
from google.adk.events import Event
from google.adk.workflow.utils._workflow_hitl_utils import (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)
from orchestrator.agent import AGENT_NAME as ORCHESTRATOR_AGENT_NAME
from planner.agent import AGENT_NAME as PLANNER_AGENT_NAME
from planner.plan import Plan as PlannerPlan
from planner.workflow import ToolId
from pydantic import ValidationError
from translator.agent import AGENT_NAME as PLAN_TRANSLATOR_AGENT_NAME

from harness import plan_display
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph import (
    APPROVE_INTERRUPT,
    ORCHESTRATOR_STATUS_KEY,
    PLANNER_STARTED_KEY,
    SEED_STEP_TOOL,
    STEP_PROGRESS_KEY,
    Executor,
)
from harness.planner_seam import PlannerSeamError
from harness.provenance import ProvenanceStore, retrieval_frame_ref
from harness.studio_state import (
    ConversationState,
    StudioStore,
    _PendingInterrupt,
    _wire_id,
)
from harness.studio_state import (
    land_component as state_land_component,
)
from harness.wire import (
    ActivityStep,
    ComponentCreated,
    CompositionUpdated,
    Done,
    ElicitationInput,
    ElicitationOption,
    ElicitationRequest,
    ErrorCode,
    ErrorFrame,
    MessageComplete,
    MessageDelta,
    PlanAction,
    PlanDecision,
    PlanDepth,
    PlanOperation,
    PlanProposed,
    PlanRequirement,
    PlanStep,
    PlanVersioned,
    RetryMode,
    StepProduced,
    StepProgress,
    StepStatus,
    Suggestions,
    ToolError,
)

logger = logging.getLogger(__name__)

# A failed step's one-line reason in the Activity feed, bounded. Large
# enough to hold a real over_ceiling/member_cardinality refusal whole,
# including the advice sentence at the end that tells the user scope
# cannot narrow it and to go back to search.
_STEP_DETAIL_CAP = 400


# The graph's planner states `resolved_by` as planner_default/user
# (`planner.plan.ResolvedBy`); the wire vocabulary is default/user.
_RESOLVED_BY_TO_WIRE = {"planner_default": "default", "user": "user"}

# Status lines for the planner's own calls. A call not named here leaves
# the current line running.
_PLANNER_CALL_LABELS = {
    PLAN_TRANSLATOR_AGENT_NAME: "Checking the plan against the data",
    "submit_plan": "Getting the plan ready for you",
}

# Human labels for plan steps and activity entries. The planner agent writes
# its own `label` / `description` per step (`step_display` on the planner
# node's output) and those win; this is the
# fallback for a scripted draft, which carries none. Also the Activity feed's
# labels, which are per-tool by nature.
_TOOL_LABELS = {
    "get_data": "Fetch the data",
    "aggregate": "Aggregate the series",
    "compare": "Compare the series",
    "growth": "Compute growth",
    "rank": "Rank the entities",
    "share_of": "Compute shares",
    "filter": "Filter the frame",
    "join": "Join the frames",
    "describe_frame": "Describe the frame",
    "write_component": "Create a canvas component",
    SEED_STEP_TOOL: "Receive the datasets from Search",
}


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
        return {
            "code": "transient_unavailable",
            "retry_mode": "do_not_retry",
            "fix": {
                "detail": (
                    "write_component timed out after it was sent, so the component may "
                    "already exist; check the canvas before asking for it again -- the "
                    f"team can trace it as {conversation.trace_id}"
                )
            },
        }
    if repeat:
        return {
            "code": "transient_unavailable",
            "retry_mode": "do_not_retry",
            "fix": {
                "detail": (
                    f"{tool} timed out again in this conversation; the earlier "
                    "wait did not help, so retrying now is unlikely to either -- "
                    f"the team can trace it as {conversation.trace_id}"
                )
            },
        }
    return {
        "code": "transient_unavailable",
        "retry_mode": "retry_after_delay",
        "fix": {"detail": f"{tool} timed out; the team can trace it as {conversation.trace_id}"},
    }


# What the user reads when the planner pair stops without a plan ready for
# them.
_PLANNER_FAILED_DETAIL = (
    "The assistant could not finish preparing a plan for this question. "
    "The team can investigate using reference {trace_id}."
)


def _stream_fatal(conversation: ConversationState, exc: Exception) -> ErrorFrame:
    """A stream-fatal error frame. A resolver refusal keeps its real code
    when the wire grammar knows it (e.g. `capability_unavailable` on an
    identity-reproduction failure); a timed-out MCP call (`timeout_tool`/
    `timeout_key`, set by `mcp_client._call_tool`) becomes its own synthetic
    refusal so a repeat of the exact same call stops being told to wait.

    A code the wire grammar has no member for, e.g.
    `executor.outcomes.RefusalKind.SURFACE_CHANGED`, which fires when a
    plan step's tool has changed since the plan was compiled, still
    honours the refusal's own `retry_mode` rather than defaulting to a
    transient one: `retry_after_delay` (or no `retry_mode` at all, for an
    exception that carries no refusal shape) becomes `transient_unavailable`
    as before, anything else becomes `capability_unavailable`, the closest
    true member, so a do-not-retry refusal is never told to the user as a
    reason to retry.

    A `PlannerSeamError` keeps its code and retry mode but not its detail:
    that text is the seam's message to the planner model ("Call
    `dfo_translator` first"), logged by the caller, and the user reads
    `_PLANNER_FAILED_DETAIL` with the trace id instead.
    """
    timeout_tool = getattr(exc, "timeout_tool", None)
    if timeout_tool is not None:
        timeout_key = getattr(exc, "timeout_key", None) or timeout_tool
        refusal = _timeout_refusal(conversation, tool=timeout_tool, key=timeout_key)
    else:
        refusal = getattr(exc, "refusal", None) or {}
    code = refusal.get("code")
    try:
        error_code = ErrorCode(code)
        retry = RetryMode(refusal.get("retry_mode", RetryMode.DO_NOT_RETRY))
        if isinstance(exc, PlannerSeamError):
            detail = _PLANNER_FAILED_DETAIL.format(trace_id=conversation.trace_id)
        else:
            detail = str(
                (refusal.get("fix") or {}).get("detail")
                or f"the turn failed server-side; the team can trace it as {conversation.trace_id}"
            )
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
        detail = f"the turn failed server-side; the team can trace it as {conversation.trace_id}"
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
    # at call time: a plan step's id, `harness.graph.FETCH_STEP_ID` for the
    # `fetch` node's call, and `None` for compose's presentation calls,
    # which are no step at all. Lets `_RecordingSurface.mark_step_failed`
    # find and amend the one record a step's post-call validation refused,
    # by identity rather than by position, and lets the translator give a
    # step's final frame the wire id its progress frames already used.
    step_id: str | None = None
    # Stamped by `_RecordingExecutor.land_component` right after the write
    # it belongs to returns (`harness.graph`'s `compose` node): `land=True`
    # for a component that passed `harness.wire.usable_component`,
    # `revises` for the turn's revise target, carried only on the first
    # usable write. Neither field is a decision; the replace-or-append
    # call happens later, at drain time, when `_component_frames` reads
    # them. `land` stays `False` (its default) for every record that never
    # became a `land_component` call: a plan-step surface record, or a
    # write whose component failed validation.
    land: bool = False
    revises: str | None = None


def _raised_refusal(exc: BaseException) -> dict[str, Any]:
    """The refusal a failed step shows for a call that raised: the exception's
    class as its code, or `cancelled` for one stopped mid-flight."""
    if isinstance(exc, asyncio.CancelledError):
        return {
            "code": "cancelled",
            "retry_mode": "do_not_retry",
            "fix": None,
            "message": "the call was stopped before it returned",
        }
    return {
        "code": type(exc).__name__,
        "retry_mode": "retry_after_delay",
        "fix": None,
        "message": (str(exc) or type(exc).__name__)[:_STEP_DETAIL_CAP],
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

    `store` also backs `components_replaced`: an optional capability
    `harness.graph`'s `narrate` node probes for with `getattr` (the same
    pattern `aclose` below uses), never a required part of the `Executor`
    protocol -- a bare unit-test executor built without this wrapper simply
    has none, and `narrate` then reads `components_replaced` as always
    False. Required here, not optional, because every real caller
    (`harness.studio.StudioRunner`) has a store to bind, and a wrapper
    built without one could never resolve the conversation
    `components_replaced` needs.
    """

    def __init__(self, executor: Executor, *, store: StudioStore) -> None:
        self._executor = executor
        self._store = store
        self._records: list[_ExecutorRecord] = []

    def land_component(self, component: dict[str, Any], *, land: bool, revises: str | None) -> None:
        """The synchronous stamping capability `compose` calls right after
        `write_component` returns, with no `await` in between. Stamps
        `land` and `revises` onto this call's own recorded write (matched
        by identity: the record whose result carries this exact `component`
        object) so `_component_frames` can decide and perform the real
        landing later, at drain time, from these two fields alone. Never
        decides anything and never touches the conversation itself; a
        component with no matching record (this capability called
        directly, not through `__call__` first) leaves nothing to stamp
        and is silently a no-op."""
        for record in reversed(self._records):
            if record.result.get("component") is component:
                record.land = land
                record.revises = revises
                break

    def components_replaced(self) -> bool:
        """Whether a component landed by the current turn actually replaced
        an existing block, read by `harness.graph`'s `narrate` node,
        after `compose` has finished and every write it attempted has
        already been drained (`_component_frames`, which is what actually
        calls `harness.studio_state.land_component` and records the
        outcome on `conversation.turn_components`). Resolves the current
        turn's conversation off the (store, contextvar) pair the same way
        `harness.studio.StudioRunner` binds it for the whole turn, through
        this instance's already-loaded live state (`StudioStore.current`).
        The route that owns this turn has already loaded or created the
        conversation before the graph ever runs."""
        key = current_conversation_key.get()
        if key is None:
            raise RuntimeError(
                "components_replaced needs a current conversation key; the caller must run "
                "within a studio turn"
            )
        conversation = self._store.current(*key)
        if conversation is None:
            raise RuntimeError(
                "components_replaced needs a conversation already loaded onto this store; "
                "the caller must run within a studio turn"
            )
        return any(card.get("action") == "replaced" for card in conversation.turn_components)

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
        `getattr`, as `compose` probes for `land_component`: the node stores
        the seeds through data-mcp, off the executor, and refuses the ones it
        cannot use, and each of those still shows as a step.
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

        `Executor` (`harness.graph.Executor`) is a bare-call protocol with
        no `aclose` of its own; `McpExecutor` adds one to release its
        long-lived per-conversation presentation-mcp clients
        (`harness.mcp_client.McpExecutor.aclose`), while the scripted
        stand-ins used in tests and `STUDIO_SCRIPTED` mode hold nothing to
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


def _surface_call_outcome(result: ToolCallResult) -> dict[str, Any]:
    """A `ToolCallResult` in the outcome grammar `_Translator._drain_recorder` reads.

    A refusal carries `code`, `retry_mode`, `fix` and `message`, read off
    `structured_content`, the machine-readable half of the result. The digest
    text serves only as the last-resort `message` for a refusal that carries
    no structured body at all (a fastmcp argument-validation error, which
    fails before the tool itself runs and so states nothing but the exception
    text). The refusal omits `origin` and `advice`, so it reads as
    unattributed; `McpExecutor._refusal_outcome` states both.
    """
    if not result.is_error:
        return {"outcome": "ok", "frame_refs": [], "summary": _first_line(result.text)}
    body = result.structured_content or {}
    return {
        "outcome": "refusal",
        "refusal": {
            "code": refusal_code(body) or "unspecified",
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
                        "code": "transport_failed",
                        "retry_mode": "retry_after_delay",
                        "fix": None,
                        "message": str(exc)[:_STEP_DETAIL_CAP],
                    },
                },
                started=started,
            )
            raise
        if self._provenance is not None and tool.name == "get_data" and not result.is_error:
            self._remember_manifest(result)
        self._record(tool.name, arguments, _surface_call_outcome(result), started=started)
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

    def _remember_manifest(self, result: ToolCallResult) -> None:
        """Write a successful `get_data` step's manifest into `provenance`.

        Keyed by `retrieval_frame_ref`, as `harness.mcp_client.McpExecutor._get_data`
        keys it, so the two writers agree on where a manifest lives regardless of
        which path retrieved it.
        """
        manifest: dict[str, Any] = dict(result.structured_content or {})
        ref = retrieval_frame_ref(manifest)
        assert self._provenance is not None  # guarded by the caller
        self._provenance.put(
            conversation=current_conversation_id.get(),
            spec_id=ref["spec_id"],
            content_version=ref["content_version"],
            manifest=manifest,
        )

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
        (`harness.graph._failed_step`'s `code`: a tool's own code when one
        called it, the specific `EvidenceIssueCode` when `check_evidence`
        raised it, `outcome.refusal.value` otherwise) -- never a fixed
        string, so the Activity feed names the actual check that fired
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
        StatVars it sums. Called once per terminal frame from
        `harness.graph._link_evidence_to_terminal_frames`.

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

    def set_step_order(self, conversation: str, order: Sequence[str]) -> None:
        """Remember `conversation`'s latest approved plan's workflow step order.

        Called from `_Translator._interrupt_frames` when it builds the
        `plan.proposed` frame. A workflow's own steps run as concurrent
        waves (`executor.run._Run.execute`), so two independent steps can
        finish, and get recorded, in the opposite order the plan lists
        them; `step_order` lets `_drain_recorder` put them back.
        """
        self._step_order[conversation] = list(order)

    def step_order(self, conversation: str) -> list[str]:
        """`conversation`'s latest plan step order, or `[]` if none was set."""
        return self._step_order.get(conversation, [])


def _first_line(text: str, *, limit: int = 160) -> str | None:
    """A result's digest text, as the Activity feed's one-line summary."""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line:
        return None
    return line if len(line) <= limit else line[: limit - 1] + "…"


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
    ADK's `create_request_input_event`)."""
    if not (event.content and event.content.parts):
        return None
    for part in event.content.parts:
        call = part.function_call
        if call is not None and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
            return str(call.id or ""), dict(call.args or {})
    return None


def _step_label(tool: str) -> str:
    return _TOOL_LABELS.get(tool, tool)


def _planner_call_label(name: str, args: dict[str, Any]) -> str | None:
    """The status line for one of the planner's own calls, or None to
    leave the current one running."""
    if name == "propose_plan":
        return "Revising the plan" if args.get("amends_version") is not None else "Drafting a plan"
    return _PLANNER_CALL_LABELS.get(name)


def _plan_from_payload(payload: dict[str, Any]) -> PlannerPlan:
    """The planner node's payload as a `planner.plan.Plan`. The planner
    node itself validates a model-drafted plan (`graph.py`'s `planner` node
    raises on a draft the seam could not validate), so this only ever sees
    a plan already known good.

    `refusal_note` is the one field a `"pending_refusal"`-routed payload,
    or a replan after a dropped seed, carries beside a normal `"ok"` one's
    own `summary`/`revision` (`graph.py`'s `planner` node): the payload
    still needs to say why the user is seeing a plan again, and `Plan`'s
    schema has no field for that. `depth` and `depth_notice` are
    the depth verdict `graph.py`'s `choose_depth` node reached earlier in
    the same turn, carried on both routes for `_interrupt_frames` to read;
    `Plan`'s own schema has no field for either.
    """
    fields = {
        key: value
        for key, value in payload.items()
        if key not in ("summary", "revision", "refusal_note", "depth", "depth_notice")
    }
    return PlannerPlan.model_validate(fields)


def _plan_depth(payload: dict[str, Any]) -> PlanDepth | None:
    """The planner node's `depth` value, as the wire enum -- `None` for a
    value the enum doesn't carry, `depth="none"` included. That verdict
    routes a turn to `fetch` when it resolves, so a first plan card never
    carries it; a card proposed after a dropped seed can, when the seeds
    left would have been fetched, and it renders with no depth. A payload
    without a recognised depth still renders a card rather than raising."""
    try:
        return PlanDepth(payload.get("depth"))
    except ValueError:
        return None


def _refusal_detail(refusal: dict[str, Any]) -> str:
    """A failed step's one-line reason, for the Activity feed.

    The message is included because a fastmcp argument-validation error
    carries no structured body: its `code` is "unspecified" and the
    explanation lives in `message`, so the user reads the reason itself.
    Cut on a word boundary with an ellipsis rather than mid-word: a
    refusal's final sentence -- often the one telling the user what to do
    next -- must read as a sentence even when the whole message doesn't fit.
    """
    code = str(refusal.get("code") or "refused")
    said = refusal.get("message") or refusal.get("fix") or ""
    if not isinstance(said, str):
        said = json.dumps(said, default=str)
    full = f"{code}: {said}".strip(": ")
    if len(full) <= _STEP_DETAIL_CAP:
        return full
    return full[:_STEP_DETAIL_CAP].rsplit(" ", 1)[0] + "..."


class _Translator:
    """One turn segment's ADK-event-to-wire-frame translation.

    Stateful over the segment: `paused` is set when a pausing frame
    (plan.proposed / elicitation.request) is emitted, which tells the route
    the stream must end without `done` (contract delta 1).
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
        # The dropped seeds the running draft's `PLANNER_STARTED_KEY` names,
        # which a question the planner asks opens with.
        self._untold_drops: str | None = None

    def translate(self, event: Event) -> list[Any]:
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
            return [self._running_step(step_progress)]
        frames: list[Any] = list(self._drain_recorder())
        interrupt = _request_input_call(event)
        if interrupt is not None:
            adk_id, args = interrupt
            frames.extend(self._close_status())
            frames.extend(self._interrupt_frames(adk_id, args))
            return frames
        if event.author == PLANNER_AGENT_NAME and not event.partial:
            for call in event.get_function_calls():
                label = _planner_call_label(call.name or "", call.args or {})
                if label is not None:
                    frames.extend(self._open_status(label))
            return frames
        node = _node_name(event)
        # The planner node's other `custom_metadata` events, the seam's
        # late `progress` reports, carry no output and add no frame.
        started = event.custom_metadata.get(PLANNER_STARTED_KEY) if event.custom_metadata else None
        if node == "planner" and isinstance(started, dict):
            label = "Revising the plan" if started.get("revising") else "Working on a plan"
            frames.extend(self._open_status(label))
            self._untold_drops = started.get("untold_drops")
        output = event.output if isinstance(event.output, dict) else None
        if output is None:
            return frames
        if node == "planner":
            self._untold_drops = None
            frames.extend(self._close_status())
        if node == "probe":
            frames.append(self._resolve_step(output))
        elif node in ("converse", "reply", "narrate", "deadend", "canvas_answer") and isinstance(
            output.get("text"), str
        ):
            frames.append(
                MessageDelta(
                    event="message.delta",
                    seq=self._conversation.next_seq(),
                    text=output["text"],
                )
            )
            if node in ("reply", "narrate", "deadend", "canvas_answer"):
                self._final_text = output["text"]
                self._conversation.turn_final_text = output["text"]
                frames.append(
                    MessageComplete(
                        event="message.complete",
                        seq=self._conversation.next_seq(),
                        text=output["text"],
                    )
                )
        return frames

    def finish(self) -> list[Any]:
        """Frames that close a segment that ran to the end of its events."""
        frames: list[Any] = [*self._drain_recorder(), *self._close_status()]
        if self.paused:
            return frames
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

    def _resolve_step(self, envelope: dict[str, Any]) -> ActivityStep:
        status = str(envelope.get("status", ""))
        # Resolver diagnostics belong on this frame (the Activity panel),
        # away from the user-facing chat close. Carrying them on a fact
        # frame lets a plain-language refusal echo one of their numbers
        # without tripping the grounding check.
        diagnostics = "; ".join(
            str(warning.get("message", ""))
            for warning in envelope.get("warnings", [])
            if warning.get("message")
        )
        detail = f"resolved: {status}" if status else None
        parts = envelope.get("part_digests") or []
        if detail and len(parts) > 1:
            listed = "; ".join(f"{part.get('variable')}: {part.get('status')}" for part in parts)
            detail = f"{detail} — {len(parts)} parts: {listed}"
        if detail and diagnostics:
            detail = f"{detail} — {diagnostics[:500]}"
        return ActivityStep(
            event="activity.step",
            seq=self._conversation.next_seq(),
            step_id="resolve",
            label="Resolve the question against the data graph",
            status=StepStatus.COMPLETE,
            tool="search",
            detail=detail,
        )

    def _open_status(self, label: str) -> list[ActivityStep]:
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
            )
        )
        return frames

    def _close_status(self) -> list[ActivityStep]:
        """Complete the running status line, if there is one. The
        `duration_ms` is what tells Studio the line is done."""
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
        """A step's progress report (`harness.graph._step_progress_event`) as a
        `running` frame on the step's own wire id, with the report's message
        as `detail`. The label matches the one the step's final frame
        carries (`_drain_recorder`), since the Activity feed names a line
        from its first frame."""
        tool = str(progress.get("tool") or "")
        message = progress.get("message")
        seq = self._conversation.next_seq()
        return ActivityStep(
            event="activity.step",
            seq=seq,
            step_id=self._wire_step_id(str(progress.get("step_id")), seq=seq),
            label=_step_label(tool),
            status=StepStatus.RUNNING,
            tool=tool or None,
            detail=message if isinstance(message, str) else None,
        )

    def _drain_recorder(self) -> list[Any]:
        """Frames for every call recorded since the last drain. Every record
        of one step, retries included, shares that step's wire id."""
        conversation = current_conversation_id.get()
        records = list(self._recorder.drain(conversation))
        if self._surface_recorder is not None:
            # Plan steps (the surface) precede compose's presentation calls
            # (the executor) within any one segment, so this ordering matches
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
        frames: list[Any] = []
        for record in records:
            seq = self._conversation.next_seq()
            step_id = self._wire_step_id(record.step_id, seq=seq)
            refused = record.result.get("outcome") == "refusal"
            refusal = record.result.get("refusal") or {}
            component = record.result.get("component")
            frames.append(
                ActivityStep(
                    event="activity.step",
                    seq=seq,
                    step_id=step_id,
                    label=_step_label(record.tool),
                    status=StepStatus.FAILED if refused else StepStatus.COMPLETE,
                    tool=record.tool,
                    duration_ms=record.duration_ms,
                    detail=_refusal_detail(refusal) if refused else record.result.get("summary"),
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
                        component, step_id=step_id, land=record.land, revises=record.revises
                    )
                )
        return frames

    def _component_frames(
        self, component: dict[str, Any], *, step_id: str, land: bool, revises: str | None
    ) -> list[Any]:
        """Validate `component` against the `component.created` wire
        contract, then, only for a component stamped `land=True`
        (`_RecordingExecutor.land_component`, called by `harness.graph`'s
        `compose` node right after the write returned), land it: the
        replace-or-append decision and the canvas mutation itself
        (`harness.studio_state.land_component`, appended, or swapped in
        place for the block `revises` named, the same op the Edit Card
        route's recreate uses, `_replace_block`, called directly by
        `harness.studio_edit`). This is the one place that decision is
        made and carried out. A component that fails validation here
        never lands and produces no frame at all. For an unstamped record
        (`land=False`) this is the expected outcome: `harness.graph`'s
        `compose` already refused this write (`component_invalid`, in
        `refused_writes`) because the same component failed
        `harness.wire.usable_component`, so no frame is the right result
        and this is not logged. For a record stamped `land=True`, `compose`
        already ran `usable_component` on this exact component and it
        passed, so a validation failure here means the wire check and this
        method's own contract have diverged: a real bug, logged at
        ERROR. `land=False` (a write compose never judged usable, or a
        bare executor with no `land_component` capability at all) means no
        block exists to report: only `component.created` goes out, for
        whichever component made it this far. `land=True` against an
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
                    "a component stamped land=True (compose's usable_component check "
                    "passed) failed component.created validation at drain time; "
                    "compose's wire check and this validation have diverged"
                )
            return []
        if not land:
            return [created]
        action = state_land_component(self._conversation, component, revises=revises)
        if action not in ("replaced", "created"):
            return [created]
        return [
            created,
            CompositionUpdated(
                event="composition.updated",
                seq=self._conversation.next_seq(),
                manifest=self._conversation.composition,
            ),
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
                # `graph.py`'s `planner` node sets this on its
                # `"pending_refusal"` route (BUDGET_EXHAUSTED or
                # QUESTION_FIXED), a refused amendment that still has a
                # `PENDING_APPROVAL` plan on record re-offered unmodified,
                # and on a replan after a dropped seed, which names the
                # seed left out. The plan card below says what is still
                # approvable; this says why the user is seeing it again
                # rather than the change they just asked for -- any other
                # `"ok"` draft carries no `refusal_note` and gets no
                # message here.
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
            if latest is not None and version < latest:
                # Versions only rise within a turn once the amend/replan
                # guard lands (see plan.py's `Plan.amend`); a lower version
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
                if latest is not None and version > latest:
                    frames.append(
                        PlanVersioned(
                            event="plan.versioned",
                            seq=conversation.next_seq(),
                            plan_version_id=plan_version_id,
                            version=version,
                            supersedes=conversation.plan_version_ids[latest],
                        )
                    )
                conversation.latest_plan_version = version
            plan = _plan_from_payload(payload)
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
            if self._surface_recorder is not None:
                self._surface_recorder.set_step_order(current_conversation_id.get(), step_order)
            requirements = [
                PlanRequirement(**cast("dict[str, Any]", row))
                for row in plan_display.requirement_rows(plan)
            ]
            operations = [
                PlanOperation(**cast("dict[str, Any]", row))
                for row in plan_display.operation_rows(plan)
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
                    depth=_plan_depth(payload),
                    depth_notice=cast("str | None", payload.get("depth_notice")),
                    decisions=[
                        PlanDecision(
                            id=decision.get("slot", f"d{index + 1}"),
                            question=decision.get("description", ""),
                            chosen=decision.get("chosen", ""),
                            alternatives=decision.get("options"),
                            reason=decision.get("rationale", ""),
                            resolved_by=_RESOLVED_BY_TO_WIRE.get(
                                decision.get("resolved_by", ""), "default"
                            ),
                        )
                        for index, decision in enumerate(payload.get("decisions", []))
                    ],
                    actions=[PlanAction.APPROVE, PlanAction.AMEND, PlanAction.REJECT],
                    revision=revision,
                    requirements=requirements,
                    operations=operations,
                )
            )
            conversation.pending[wire_id] = _PendingInterrupt(
                adk_id=adk_id,
                kind="plan",
                plan_version=version,
                plan_revision=revision,
                step_order=tuple(step_order),
            )
            return frames
        # The graph's own `clarify` node (`APPROVE_INTERRUPT`'s sibling
        # route) raises a `RequestInput` whose `payload` carries
        # `candidates`; the planner's `adk_request_input` tool call
        # (`planner.agent`'s `ask_user`) has no `payload` wrapper at all --
        # `candidates`, `recommended_spec_id` and `rationale` are its own
        # call args directly. Either shape lands here.
        candidates = payload.get("candidates") or args.get("candidates") or []
        recommended_spec_id = args.get("recommended_spec_id")
        options = [
            ElicitationOption(
                id=str(candidate.get("spec_id", "")),
                label=str(candidate.get("label", candidate.get("spec_id", ""))),
                recommended=bool(recommended_spec_id)
                and str(candidate.get("spec_id", "")) == recommended_spec_id,
            )
            for candidate in candidates
        ]
        progress = (
            StepProgress(completed=int(payload["round"]), total=int(payload["max_rounds"]))
            if payload.get("round") and payload.get("max_rounds")
            else None
        )
        question = str(args.get("message", "Which reading did you mean?"))
        if self._untold_drops:
            # A planner-side ask: the `planner` node has marked the drops it
            # names as told. `clarify` opens its own message with them.
            question = f"{self._untold_drops} {question}"
            self._untold_drops = None
        frame = ElicitationRequest(
            event="elicitation.request",
            seq=conversation.next_seq(),
            interrupt_id=wire_id,
            question=question,
            input=ElicitationInput.SINGLE_SELECT,
            options=options,
            allow_free_text=True,
            skippable=True,
            progress=progress,
            rationale=(
                str(args["rationale"])
                if args.get("rationale") and any(option.recommended for option in options)
                else None
            ),
        )
        conversation.pending[wire_id] = _PendingInterrupt(
            adk_id=adk_id,
            kind="elicitation",
            option_ids=frozenset(option.id for option in options),
        )
        return [frame]


# ---------------------------------------------------------------------------
# SSE plumbing
# ---------------------------------------------------------------------------


class _OrchestratorTranslator(_Translator):
    """Turns the orchestrator engine's events into wire frames.

    The agent's tool calls run through the recording executor, so their
    steps and cards arrive through `_drain_recorder` exactly as the graph's
    do; the plan path's events (the planner's status lines, the executor's
    step progress, the approval and clarify interrupts) are the graph's own
    and go to `_Translator.translate`. On top of those, the agent's own
    events stream as they happen:

    - a thought summary, as one running `activity.step` per model call
      labelled with the summary's latest heading (Gemini opens each
      thought with one; a short call thinks too little to send any);
    - each tool call, as an `activity.step` labelled from its arguments the
      moment the model makes it, completed when the tool returns;
    - answer text, as `message.delta` chunks, the last model call's text
      being the turn's `message.complete`: text before a tool call is
      interim, and text after a hand-off to the user or the planner would
      sit above the question or the plan card, so neither is the answer;
    - an orchestrator node's own status line ("Reading your question") as a
      running step until the next thing happens.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thought = ""
        self._thought_step: str | None = None
        self._streamed_text = False
        self._open_calls: dict[str, tuple[str, str]] = {}
        self._handed_off = False

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
            self._handed_off = False
            return [*self._drain_recorder(), *self._open_status(status)]
        if event.author != ORCHESTRATOR_AGENT_NAME:
            return super().translate(event)
        frames: list[Any] = list(self._drain_recorder())
        parts = event.content.parts if event.content and event.content.parts else []
        thought = "".join(part.text or "" for part in parts if part.thought)
        text = "".join(part.text or "" for part in parts if part.text and not part.thought)
        if event.partial:
            if thought:
                self._thought += thought
                if _thought_heading(self._thought) is not None:
                    # Close the status line before the step takes its seq.
                    frames.extend(self._close_status())
                    frames.extend(self._thinking())
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
        # A finished model call: its thinking is over, and its text arrived
        # already as partials unless the model sent it in one piece.
        frames.extend(self._close_thinking())
        if text and not self._streamed_text and not self._handed_off:
            frames.extend(self._close_status())
            self._final_text += text
            frames.append(
                MessageDelta(event="message.delta", seq=self._conversation.next_seq(), text=text)
            )
        self._streamed_text = False
        if event.get_function_calls():
            self._final_text = ""
            frames.extend(self._close_status())
        for call in event.get_function_calls():
            label = _tool_call_label(call.name or "", call.args or {})
            frame = self._step(None, label, StepStatus.RUNNING, tool=call.name)
            self._open_calls[call.id or frame.step_id] = (frame.step_id, label)
            frames.append(frame)
        for response in event.get_function_responses():
            # A hand-off counts once the tool took it; a refused one (its
            # budget spent) leaves the model to answer, and that text shows.
            answer = response.response or {}
            if response.name in ("plan", "ask_user") and "error" not in answer:
                self._handed_off = True
            opened = self._open_calls.get(response.id or "")
            if opened is not None:
                error = answer.get("error")
                frames.append(
                    self._step(
                        *opened,
                        StepStatus.FAILED if error else StepStatus.COMPLETE,
                        tool=response.name,
                        detail=str(error)[:_STEP_DETAIL_CAP] if error else None,
                    )
                )
        if event.get_function_responses():
            frames.extend(self._open_status("Reading the results"))
        return frames

    def _thinking(self) -> list[Any]:
        heading = _thought_heading(self._thought)
        if heading is None:
            return []
        frame = self._step(self._thought_step, heading, StepStatus.RUNNING, detail=self._thought)
        self._thought_step = frame.step_id
        return [frame]

    def _close_thinking(self) -> list[Any]:
        frames: list[Any] = []
        heading = _thought_heading(self._thought)
        if self._thought_step is not None and heading is not None:
            frames.append(
                self._step(self._thought_step, heading, StepStatus.COMPLETE, detail=self._thought)
            )
        self._thought = ""
        self._thought_step = None
        return frames

    def finish(self) -> list[Any]:
        frames: list[Any] = list(self._drain_recorder())
        if self._final_text and not self.paused:
            frames.extend(self._close_status())
            self._conversation.turn_final_text = self._final_text
            frames.append(
                MessageComplete(
                    event="message.complete",
                    seq=self._conversation.next_seq(),
                    text=self._final_text,
                )
            )
        return frames + super().finish()


# A thought summary's heading is a paragraph's first line, which Gemini
# writes in bold (or as a markdown heading); longer than this, it is prose.
_THOUGHT_HEADING_CHARS = 80


def _thought_heading(thought: str) -> str | None:
    """The latest heading in a thought summary, or None when it has none yet."""
    for paragraph in reversed(thought.split("\n\n")):
        first = paragraph.strip().split("\n", 1)[0].strip()
        if first.startswith(("**", "#")):
            heading = first.strip("*# ").strip()
            if heading and len(heading) <= _THOUGHT_HEADING_CHARS:
                return heading
    return None


_ORCHESTRATOR_TOOL_LABELS = {
    "fetch": "Fetching data",
    "ask_user": "Asking you a question",
    "plan": "Handing the question to the planner",
    "explain_dfo": "Reading about DFO",
}


def _tool_call_label(name: str, args: dict[str, Any]) -> str:
    """A status line for one of the orchestrator's tool calls, from its arguments."""
    if name == "search":
        return f"Searching: {args['query']}" if args.get("query") else "Searching"
    if name == "make_charts":
        count = len(args.get("charts") or [])
        return f"Drawing {count} chart{'s' if count != 1 else ''}"
    if name in ("get_skill", "get_skill_file"):
        return (
            f"Reading the {args['skill_id']} skill" if args.get("skill_id") else "Reading a skill"
        )
    return _ORCHESTRATOR_TOOL_LABELS.get(name) or _step_label(name)
