"""The graph's Resolver and Executor over the loopback MCP mounts.

`McpResolver` wraps data-mcp `search`; `McpExecutor` routes plan steps to
data-mcp (`get_data`) and presentation-mcp (`describe_frame` /
`write_component` / `describe_component`). The analysis verbs reach
analysis-mcp through the execute node's `ToolSurface` instead (see
`harness.agents.build_tool_surface`); this module never sees them. Both
speak MCP over streamable HTTP to the mounts `harness.server.create_app`
hosts in-process -- the same seam a split-out deployment would use, just
loopback.
Every `Client(...)` construction below goes through `self._policy`
(`harness.mcp_deadlines.McpTimeoutPolicy`), which carries `auth=self._token`
-- the same bearer `common.auth.bearer_token()` resolved when `create_app`
gated the mounts (`None` when the gate is disabled); the harness is the
mounts' one caller, so one token threaded through every call site is enough
-- and binds a finite call deadline (per-tool where `compiler.capabilities.
call_timeouts()` pins one, else `Settings.mcp_call_timeout_s`) and a finite
init deadline (`Settings.mcp_init_timeout_s`). FastMCP's own `Client`
defaults both to `None`, unbounded; on this, the harness's own served
request path, an unbounded MCP call holds the request open indefinitely.

Context discipline: what these return into graph state and node payloads is
digest-sized. The resolver maps the search envelope down to the graph's
decision inputs (status / spec_id / candidates / choice_points / parts /
warnings); the full envelope, with its dozens of unbound definition slots,
never leaves this module (`harness.tool_contracts.graph_envelope`). The executor returns frame REFERENCES; rows stay
in the materialisation store.

Clients: every mount gets one client per conversation, held in a
`compiler.caller_clients.CallerClients` registry keyed by the conversation's
caller (`harness.conversation.current_conversation_caller`). The client
names that caller in its `x-dfo-caller` header on every request, since
under the 2026-07-28 protocol revision the MCP transport session lasts one
request and cannot identify a conversation. data-mcp and analysis-mcp
record the caller as a stored frame's `writer`. presentation-mcp ignores
it: a component id grants reads, and the `write_token` a create returns
grants revises (`common.mcp_telemetry.WRITE_TOKEN_KEY`).

Holding a client for a whole conversation means a client that breaks mid-
conversation would otherwise keep failing every later call the same way,
with nothing to reconnect it. `_call_held` is where every call site (except
`_indicators_for`'s, which already treats any describe failure as
"degrade to a fallback name" and discards unconditionally) draws the line: a
read timeout says only that one call ran past its own deadline, not that the
client is broken, so the client stays cached; anything else discards it, so
the next call reconnects instead of repeating the same failure.

Refusal grammar: data refusals arrive as MCP tool errors whose
structured content is `{code, retry_mode, fix, ...}`; presentation refusals
arrive as SUCCESSFUL results shaped `{error, retry, detail, fix?}` (that
service's model-facing contract). `_refusal_outcome` normalises both into
the graph's `{"outcome": "refusal", "refusal": {code, retry_mode, fix,
advice, message, origin}}` so the execute node branches on one grammar,
plus the refusal's own `retry_after_s` when it names one.
`origin` says who declined: `"service"` for a refusal an MCP service
returned, `"harness"` for the three this module builds itself
(`unknown_tool`, and `resolved_revision_unavailable` or
`provenance_unavailable` for a card it withholds after the platform accepted
the write).

Retries: `McpExecutor` retries a `retry_after_delay` refusal on the plan
executor's own policy (`executor.classify.delay_before_retry`): at most
`MAX_ATTEMPTS_PER_STEP` attempts, each wait drawn with jitter from the
refusal's `retry_after_s`. A full frame row budget refuses every call that
arrives while it is full, and the jitter keeps those calls from retrying
together. The attempts and waits together stay within the tool's own call
deadline (`McpTimeoutPolicy.call_timeout_s`, 180 s for `get_data`): a retry
whose wait would carry the call past it is not made, so a refusal that
arrives at once is retried and one that arrived near the deadline, such as
`get_data`'s unanswered mint, comes back as it is. A retry still running when
the deadline arrives is cut off, so the attempts and waits never outlast the
deadline: a read returns the refusal before it, and `write_component` raises
the timeout a first attempt's deadline raises, since the platform may have
stored the write. `write_component` is retried only after the row budget's
refusal, which is raised before any platform write; `get_data` and
`describe_frame` are reads or idempotent mints and are retried on any
`retry_after_delay`. Every other refusal, and every transport
failure, comes back from the first attempt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from common.auth import bearer_token
from common.mcp_telemetry import WRITE_TOKEN_KEY
from common.refusal import refusal_code
from compiler.caller_clients import CallerClients, session_call_meta
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    deliver_progress,
    mcp_progress_handler,
)
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from dfo_frame_grammar.grammar import parse_time_period
from executor.classify import MAX_ATTEMPTS_PER_STEP, Jitter, delay_before_retry
from fastmcp import Client
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT, ErrorData
from presentation_mcp.platform_contract import RESOLVED_PAYLOAD_META_KEY

from harness.conversation import current_conversation_caller, current_conversation_id
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.provenance import (
    BOTH_SIDES_VERBS,
    Lineage,
    OwnStatVars,
    ProvenanceInput,
    ProvenanceStore,
    SidePart,
    data_artifact,
    manifest_with_stat_vars,
    merged_manifest,
    retrieval_frame_ref,
    summed_stat_var_count,
)
from harness.rolling_cards import (
    AdmittedWrite,
    CardReplay,
    CardWithheld,
    RollingCards,
    resolved_basis,
    written_title_refusal,
)
from harness.settings import Settings
from harness.tool_contracts import (
    _PRESENTATION_TOOLS,
    _SEVERITY_TO_WIRE,
    REFUSAL_ORIGIN_SERVICE,
    ResolverError,
    _float_or_none,
    harness_refusal,
    platform_trace_id_of,
    resolver_digest,
)

logger = logging.getLogger(__name__)


# `describe` caps ids per call at 8, and `_describe_into_cache` keeps at most
# `_DESCRIBE_CONCURRENCY` calls in flight. Each call reads up to four nodes at
# once (`data_mcp.node_lookup._MAX_CONCURRENT_NODE_READS`, a semaphore built
# per call), so the two bounds together allow eight node reads at a time
# against the node door's admission pool of four. The door refuses a read
# that finds every slot held with a retryable 503, which data-mcp's client
# retries inside the call.
_DESCRIBE_BATCH = 8
_DESCRIBE_CONCURRENCY = 2

# The source the card names when neither the manifest nor the chart names one.
_UNKNOWN_SOURCE = "Unknown source"

# The licence a card's source states when its manifest citation carries none.
_LICENCE_NOT_STATED = "not stated by the platform"

# The last line of the content block presentation-mcp appends to a create's
# text, the same token as the structured content's `WRITE_TOKEN_KEY`.
_WRITE_TOKEN_LINE = re.compile(rf"{WRITE_TOKEN_KEY}: \S+\s*$")


def _without_write_token_text(content: list[Any]) -> list[Any]:
    """`content` without the text block that carries a create's write token
    on its last line, so no summary or refusal text read from the content
    can repeat the token."""
    return [
        block for block in content if not _WRITE_TOKEN_LINE.search(getattr(block, "text", "") or "")
    ]


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    """A stable string for "the same call again": sorted keys, no
    whitespace, and no write token, which is a secret and names nothing
    about the call. Good enough to key a repeat-timeout memory by, not
    meant to be read back."""
    keyed = {key: value for key, value in arguments.items() if key != WRITE_TOKEN_KEY}
    return json.dumps(keyed, sort_keys=True, separators=(",", ":"), default=str)


class _McpTimeoutError(MCPError):
    """`MCPError` re-raised with which tool hit its pinned deadline, and a
    key identifying the call itself (tool + its own arguments).
    `harness.studio_translate._stream_fatal` reads `timeout_tool`/
    `timeout_key` (duck typed, the same way it reads
    `ResolverError`/`PlanInvalidError`'s `.refusal`) to tell a timeout apart
    from an unclassifiable crash and to remember, per conversation, whether
    the SAME call has timed out before -- two different `search` calls
    timing out is evidence the service itself is slow, still worth a wait;
    the identical call timing out twice means the wait already didn't help."""

    def __init__(self, error: ErrorData, *, tool: str, arguments: dict[str, Any]) -> None:
        super().__init__(code=error.code, message=error.message, data=error.data)
        self.timeout_tool = tool
        self.timeout_key = f"{tool}:{_canonical_arguments(arguments)}"


async def _call_tool(
    client: Client,
    tool: str,
    arguments: dict[str, Any],
    *,
    on_progress: ProgressCallback | None = None,
) -> Any:
    """`client.call_tool`, raising `_McpTimeoutError` in place of a bare
    `MCPError` when it is one. `raise_on_error=False` only suppresses a
    tool-level error result; the per-tool deadline
    (`compiler.capabilities.call_timeouts()`, else `Settings.
    mcp_call_timeout_s`) elapsing still raises `MCPError`, with mcp's
    `REQUEST_TIMEOUT` code and no refusal body. `on_progress` hears the
    call's progress notifications (`compiler.surface.mcp_progress_handler`)."""
    try:
        return await client.call_tool(
            tool,
            arguments,
            meta=session_call_meta(client),
            progress_handler=mcp_progress_handler(on_progress),
            raise_on_error=False,
        )
    except MCPError as exc:
        if exc.error.code == REQUEST_TIMEOUT:
            raise _McpTimeoutError(exc.error, tool=tool, arguments=arguments) from exc
        raise


async def _call_held(
    clients: CallerClients,
    client: Client,
    tool: str,
    arguments: dict[str, Any],
    *,
    on_progress: ProgressCallback | None = None,
) -> Any:
    """`_call_tool` against `client` -- `clients`' held client for the
    current conversation -- evicting it from `clients` when the call fails
    for any reason but its own read timeout.

    A timeout says only that this one call ran past its deadline
    (`get_data`'s mint can legitimately take up to 180s); it says nothing
    about whether the client underneath still works, so it is left alone
    and re-raised as `_McpTimeoutError`, the same client still cached for
    the next call. Any other exception -- a transport error, a protocol
    failure -- means the client itself is the problem: reusing it again
    would just repeat the same failure for the rest of the conversation, so
    it is evicted and the next `get()` reconnects instead.

    This is not the same as `client` being disconnected here: `discard` only
    marks it, since a sibling call sharing this conversation's client may
    still be in flight on it. Whichever call is holding `client` -- this one
    included -- releases it back to `clients` with `release`; the caller of
    `_call_held` owns that pairing, since it is the one that called `get()`
    in the first place.
    """
    try:
        return await _call_tool(client, tool, arguments, on_progress=on_progress)
    except _McpTimeoutError:
        raise
    except Exception:
        logger.warning(
            "%r failed on its held client for a reason other than its own timeout; "
            "discarding that client so the next call reconnects instead of repeating "
            "the same failure",
            tool,
            exc_info=True,
        )
        await clients.discard(client)
        raise


class McpResolver:
    """data-mcp `search` behind the graph's Resolver protocol."""

    def __init__(self, *, settings: Settings) -> None:
        self._url = settings.data_mcp_url
        # Resolved once, here. By the time a resolver/executor is
        # constructed, `harness.server.create_app` has already called
        # `common.auth.apply_gate` on every mount, so the process-wide
        # policy is settled (`None` when the gate is disabled).
        self._token = bearer_token()
        self._policy = McpTimeoutPolicy(settings=settings)
        # One held client per conversation -- see the module docstring's
        # "Clients" note.
        self._clients = CallerClients(
            lambda caller: self._policy.client(
                self._url, auth=self._token, caller=caller, tool="search"
            ),
            caller=current_conversation_caller,
        )

    async def aclose(self) -> None:
        """Release every conversation's held `search` client.

        Unlike `McpExecutor.aclose` -- routed to from `_lifespan` shutdown
        through `StudioRunner.aclose` -- no caller on the served path holds
        a reference to the `McpResolver` `harness.server.create_app` builds,
        so nothing calls this one yet. A caller that keeps its own
        `McpResolver` past a single conversation must call this to release
        its clients.
        """
        await self._clients.aclose()

    async def __call__(
        self,
        *,
        question: str,
        spec_id: str | None = None,
        wide: bool = False,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if spec_id is not None:
            # A clarify re-entry: the user (or the bound loop) already chose a
            # reading, and its definition was registered by the first search.
            # Re-searching would only reproduce the fork the choice settled.
            # `wide` cannot be recomputed without that search, so it is not
            # recomputed here: `graph.clarify` carries forward the chosen
            # candidate's own `wide` verdict (`harness.tool_contracts.
            # graph_envelope`'s `candidates` branch computes one per offered
            # spec, the same predicate a `definite` reading gets) and
            # `graph.probe` passes it through as this call's `wide` argument.
            # `parts` is the decomposition `graph.probe` carries forward from
            # the search that first produced the candidates; its length is
            # the same part count that search's envelope reported.
            return {
                "status": "definite",
                "spec_id": spec_id,
                "choice_points": 0,
                "parts": len(parts) if parts else 1,
                "warnings": [],
                "candidates": [],
                "wide": wide,
            }
        if not parts:
            # `graph.probe` omits `parts` when the decomposition came back
            # empty, and `create_root_workflow` can wire this resolver with
            # no generative seam to have re-decomposed first (unlike the
            # shipped server, which always wires `GeminiConverse` or
            # `ScriptedResolver`). Refuse here rather than sending `search`
            # a null/empty `parts`, which the tool's own schema rejects with
            # a raw MCP validation failure the graph cannot present. No
            # silent `{"variable": question}` stand-in -- an unscoped guess
            # would mask the gap instead of surfacing it.
            raise ResolverError(
                "search needs at least one decomposed part; none were supplied",
                refusal={
                    "code": "no_search_parts",
                    "retry_mode": "do_not_retry",
                    "fix": {"detail": "the question could not be decomposed into search parts"},
                },
            )
        arguments: dict[str, Any] = {"parts": parts, "raw_query": question}
        client = await self._clients.get()
        try:
            result = await _call_held(self._clients, client, "search", arguments)
        finally:
            await self._clients.release(client)
        if result.is_error:
            refusal = result.structured_content or {}
            raise ResolverError(f"search refused: {refusal}", refusal=refusal)
        # `structuredContent` is the small answer a host may show a model.
        # The upstream envelope this reads rides in `_meta`.
        envelope = (result.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
        if not isinstance(envelope, dict):
            raise ResolverError(
                f"search succeeded but carried no envelope under _meta[{SEARCH_ENVELOPE_META_KEY!r}]"
            )
        return resolver_digest(envelope)


def _held_to_its_resolved_periods(
    intent: Mapping[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    """`outcome`, a stored card whose intent states a relative period, or the
    refusal of its title when it names a period the platform resolved for it
    (`harness.rolling_cards.written_title_refusal`). `intent` is the card as
    stored (`AdmittedWrite.stored_card`): for a revise, the stored intent with
    the patch merged in, or the patch alone when the stored intent could not be
    read, judged against the title the new revision drew when it names none.

    The periods exist only once the platform has stored the card, so a refused
    card stays stored there as a revision no canvas shows, as a card withheld
    for its provenance does.
    """
    component = outcome["component"]
    drawn = (component.get("payload") or {}).get("component") or {}
    refusal = written_title_refusal(
        intent,
        title=intent.get("title") or (drawn.get("title") or {}).get("text"),
        window=(component.get("fact_summary") or {}).get("window"),
        basis=resolved_basis(component.get("payload")),
    )
    if refusal is None:
        return outcome
    logger.warning(
        "write_component withheld for component_id=%r: its title names %s, a period "
        "resolved from the latest data",
        component.get("component_id"),
        refusal["fix"]["period"],
    )
    return {"outcome": "refusal", "refusal": refusal}


class McpExecutor:
    """Plan steps against data-mcp and presentation-mcp, in the graph's outcome grammar.

    Writes every fetched manifest into `provenance` (`harness.provenance.
    ProvenanceStore`), keyed by frame reference, so a later `write_component`
    on that frame can assemble wire provenance without re-fetching. That
    store is shared with `harness.studio_translate._RecordingSurface`, the
    approved-plan path's own writer -- `_assemble_component` below reads
    from it regardless of which path retrieved the frame.

    Every `write_component` passes `harness.rolling_cards.RollingCards.admit`
    before it is sent (`_admitted_write`), whoever asked for it. A card whose
    period follows the latest data is written on the frame of the series it
    draws, replayed through `card_replay` when that is another frame than the
    one the intent names, and is refused with a harness refusal when its title
    names a year or no such frame can be had. A `describe_frame` view of a
    frame that rolls carries `rolling: true`, so whoever reads the view to
    title a card knows.

    `sleep` waits between retries, `jitter` draws each wait and `clock` times
    a call's attempts against its deadline (see the module docstring's
    "Retries" note); all three are injectable so tests neither wait, draw at
    random nor read the real clock.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provenance: ProvenanceStore | None = None,
        card_replay: CardReplay | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Jitter = random.uniform,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock
        self._data_url = settings.data_mcp_url
        self._presentation_url = settings.presentation_mcp_url
        # See `McpResolver.__init__` above. Resolved once, after
        # `create_app` has already gated every mount.
        self._token = bearer_token()
        self._policy = McpTimeoutPolicy(settings=settings)
        # `harness.server.create_app` builds one `ProvenanceStore` and passes
        # it here and to `_RecordingSurface` alike; a caller with no plan
        # path to share it with (a test, `McpResolver`'s doubles) gets a
        # private one.
        self._provenance = provenance if provenance is not None else ProvenanceStore()
        # The rules for a card whose period follows the latest data, with the
        # frames and replays they have found, shared by every conversation.
        self._rolling_cards = RollingCards(replay=card_replay, clock=clock)
        # Indicator metadata (label/definition) from data-mcp `describe`,
        # cached per dcid for the process life -- definitions do not move.
        self._indicator_cache: dict[str, dict[str, str]] = {}
        # One held client per conversation per mount/tool -- see the module
        # docstring's "Clients" note. `describe`
        # and `get_data` are separate held clients, not one shared client,
        # because each carries its own pinned call deadline
        # (`McpTimeoutPolicy.client`'s per-tool row): 180s for `get_data`'s
        # mint budget, 30s for `describe`.
        self._presentation_clients = CallerClients(
            # No single tool: this client is long-lived, serving
            # `describe_frame` / `write_component` / `describe_component`
            # across a whole conversation, so it gets the general default
            # (`McpTimeoutPolicy.client`'s `tool=None` case).
            lambda caller: self._policy.client(
                self._presentation_url, auth=self._token, caller=caller
            ),
            caller=current_conversation_caller,
        )
        self._get_data_clients = CallerClients(
            lambda caller: self._policy.client(
                self._data_url, auth=self._token, caller=caller, tool="get_data"
            ),
            caller=current_conversation_caller,
        )
        self._describe_clients = CallerClients(
            lambda caller: self._policy.client(
                self._data_url, auth=self._token, caller=caller, tool="describe"
            ),
            caller=current_conversation_caller,
        )

    async def aclose(self) -> None:
        """Release every conversation's held clients, across all three.

        Called once, from `harness.server._lifespan`'s shutdown, through
        `StudioRunner.aclose` -> `_RecordingExecutor.aclose` -> here.
        """
        await self._presentation_clients.aclose()
        await self._get_data_clients.aclose()
        await self._describe_clients.aclose()

    async def _presentation_client(self) -> Client:
        return await self._presentation_clients.get()

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """Run one plan step; `on_progress` hears its MCP progress notifications
        and a report naming each retry before its wait starts.

        A `retry_after_delay` refusal that `_safe_to_repeat` allows is retried
        while `executor.classify.delay_before_retry` names a wait and the time
        spent on this call's attempts so far, plus that wait, stays within the
        tool's call deadline. Each retry runs under the time left of that
        deadline, so the whole call never outlasts it. A read the deadline cuts
        short returns the refusal that preceded it. A `write_component` it cuts
        short raises `_McpTimeoutError`, as a first attempt's timeout does: the
        attempt was sent, so the platform may have stored the write, and the
        earlier refusal no longer says what happened. Any other outcome, and
        the refusal no retry follows, comes back as it is.
        """
        if tool == "write_component":
            try:
                admitted = await self._admitted_write(args)
            except CardWithheld as withheld:
                return {"outcome": "refusal", "refusal": withheld.refusal}
            outcome = await self._retried(tool, admitted.args, on_progress=on_progress)
            if admitted.stored_card is not None and outcome.get("outcome") == "ok":
                return _held_to_its_resolved_periods(admitted.stored_card, outcome)
            return outcome
        return await self._retried(tool, args, on_progress=on_progress)

    async def _retried(
        self, tool: str, args: dict[str, Any], *, on_progress: ProgressCallback | None
    ) -> dict[str, Any]:
        """One call of `tool` and the retries `__call__` describes."""
        deadline_s = self._policy.call_timeout_s(tool)
        started = self._clock()
        outcome = await self._attempt(tool, args, on_progress=on_progress)
        for attempt in range(MAX_ATTEMPTS_PER_STEP - 1):
            refusal = outcome.get("refusal") if outcome.get("outcome") == "refusal" else None
            if refusal is None or not _safe_to_repeat(tool, refusal):
                return outcome
            delay_s = delay_before_retry(refusal, attempt, jitter=self._jitter)
            if delay_s is None or self._clock() - started + delay_s > deadline_s:
                return outcome
            logger.info(
                "%s refused %s (retry_after_delay); retrying in %.2fs, attempt %d of %d",
                tool,
                refusal.get("code"),
                delay_s,
                attempt + 2,
                MAX_ATTEMPTS_PER_STEP,
            )
            if on_progress is not None:
                deliver_progress(
                    on_progress,
                    ProgressReport(
                        progress=attempt + 2,
                        total=MAX_ATTEMPTS_PER_STEP,
                        message=(
                            f"Retrying in {delay_s:.3g}s "
                            f"(attempt {attempt + 2} of {MAX_ATTEMPTS_PER_STEP})"
                        ),
                    ),
                )
            await self._sleep(delay_s)
            # The client's own timeout restarts on every request, so the
            # retry runs under what is left of the call's deadline.
            try:
                async with asyncio.timeout(deadline_s - (self._clock() - started)) as budget:
                    outcome = await self._attempt(tool, args, on_progress=on_progress)
            except TimeoutError as exc:
                if not budget.expired():
                    raise
                if tool == "write_component":
                    raise _McpTimeoutError(
                        ErrorData(
                            code=REQUEST_TIMEOUT,
                            message=f"{tool} ran past its {deadline_s:g}s call deadline",
                        ),
                        tool=tool,
                        arguments=args,
                    ) from exc
                return outcome
        return outcome

    async def _admitted_write(self, args: dict[str, Any]) -> AdmittedWrite:
        """The arguments to send for a `write_component` call, under the rules
        for a card whose period follows the latest data
        (`harness.rolling_cards.RollingCards.admit`).

        A write moved to a frame replayed for the card's own series gives that
        frame the lineage of the frame the intent named, because the replay
        computes the same recipe and so rests on the same retrievals, carries
        the same StatVars and names the same `sum_of` sides.

        Raises:
            CardWithheld: A rule refuses the write.
        """
        admitted = await self._rolling_cards.admit(
            args,
            read_view=self._frame_view,
            read_stored=self._stored_component,
            read_window=self._requested_window,
        )
        if admitted.replayed_from is not None:
            conversation = current_conversation_id.get()
            source = (
                admitted.replayed_from.get("spec_id"),
                admitted.replayed_from.get("content_version"),
            )
            target = admitted.args["intent"]["frame_ref"]
            self._provenance.alias(
                conversation=conversation,
                spec_id=target["spec_id"],
                content_version=target["content_version"],
                sources=[source],
                stat_var_count=self._provenance.stat_var_count(
                    conversation=conversation, spec_id=source[0], content_version=source[1]
                ),
                own_stat_vars=self._provenance.own_stat_vars(
                    conversation=conversation, spec_id=source[0], content_version=source[1]
                ),
            )
        return admitted

    def _requested_window(self, frame_ref: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The `requested_window` of the definition `frame_ref` was retrieved
        for, from the retrieval's manifest in the provenance store. `None` for
        a frame with no manifest there, a frame resting on several retrievals,
        and a manifest that states no window."""
        lineage = self._provenance.get(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.get("spec_id"),
            content_version=frame_ref.get("content_version"),
        )
        if lineage is None or len(lineage) != 1:
            return None
        definition = lineage[0].manifest.get("definition") or {}
        window = (definition.get("identity") or {}).get("requested_window")
        return window if isinstance(window, Mapping) else None

    async def _frame_view(self, frame_ref: Mapping[str, Any]) -> dict[str, Any] | None:
        """The view of `frame_ref`, or `None` when presentation-mcp refuses it."""
        described = await self._presentation("describe_frame", {"frame_ref": dict(frame_ref)})
        return described.get("view") if described.get("outcome") == "ok" else None

    async def _stored_component(self, component_id: str) -> dict[str, Any] | None:
        """`component_id`'s head revision as `describe_component` gives it, with
        the revision's resolved payload under `resolved_payload` when the read
        carried one, or `None` when presentation-mcp refuses the read."""
        described = await self._presentation("describe_component", {"component_id": component_id})
        body = described.get("component_body")
        if described.get("outcome") != "ok" or not isinstance(body, dict):
            return None
        return {**body, "resolved_payload": described.get("resolved_payload")}

    async def _attempt(
        self, tool: str, args: dict[str, Any], *, on_progress: ProgressCallback | None
    ) -> dict[str, Any]:
        """One call of `tool`, in the outcome grammar."""
        if tool == "get_data":
            return await self._get_data(args, on_progress=on_progress)
        if tool in _PRESENTATION_TOOLS:
            return await self._presentation(tool, args, on_progress=on_progress)
        return {
            "outcome": "refusal",
            "refusal": harness_refusal(
                code="unknown_tool",
                message=f"the plan named a tool this executor does not route: {tool!r}",
            ),
        }

    async def _get_data(
        self, args: dict[str, Any], *, on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        client = await self._get_data_clients.get()
        try:
            result = await _call_held(
                self._get_data_clients, client, "get_data", args, on_progress=on_progress
            )
        finally:
            await self._get_data_clients.release(client)
        if result.is_error:
            return _refusal_outcome(result.structured_content, fallback_text=_result_text(result))
        manifest = await manifest_with_stat_vars(result.structured_content or {})
        ref = retrieval_frame_ref(manifest)
        self._provenance.put(
            conversation=current_conversation_id.get(),
            spec_id=ref["spec_id"],
            content_version=ref["content_version"],
            manifest=manifest,
        )
        return {
            "outcome": "ok",
            "frame_refs": [ref],
            "summary": _manifest_summary(manifest),
            "rows": (manifest.get("coverage") or {}).get("rows"),
            "warnings": manifest.get("warnings") or [],
        }

    async def _presentation(
        self, tool: str, args: dict[str, Any], *, on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool in {"write_component", "describe_component"}:
            # The event stream carries the render, and a describe's result is
            # read from its structured content alone, so neither the text
            # projection nor the chart image presentation-mcp sends this
            # undeclared client would reach anybody.
            args = {**args, "include_projection": "none"}
        client = await self._presentation_client()
        try:
            result = await _call_held(
                self._presentation_clients, client, tool, args, on_progress=on_progress
            )
        finally:
            await self._presentation_clients.release(client)
        # A create's write token is a secret: taken off the result here,
        # before anything reads it, and handed on only beside the component.
        body = result.structured_content or {}
        write_token = body.pop(WRITE_TOKEN_KEY, None)
        result.content = _without_write_token_text(result.content)
        if result.is_error or ("error" in body and "retry" in body):
            # presentation-mcp returns its refusals as successful results
            # in its own {error, retry, detail} grammar (its model-facing
            # contract); a transport-level error still arrives as is_error.
            return _refusal_outcome(body, fallback_text=_result_text(result))
        if tool == "describe_frame":
            frame_ref = args.get("frame_ref")
            if isinstance(frame_ref, Mapping) and await self._rolling_cards.frame(frame_ref):
                body = {**body, "rolling": True}
            return {"outcome": "ok", "frame_refs": [], "view": body}
        envelope = (result.meta or {}).get(RESOLVED_PAYLOAD_META_KEY)
        if tool == "describe_component":
            # The body is the model vocabulary. The resolved payload holds the
            # labels the platform drew for the revision's relative periods.
            return {
                "outcome": "ok",
                "frame_refs": [],
                "component_body": body,
                "resolved_payload": envelope if isinstance(envelope, dict) else None,
            }
        if not isinstance(envelope, dict) or not isinstance(envelope.get("component"), dict):
            logger.warning(
                "write_component withheld for component_id=%r: the result carries no "
                "resolved revision under _meta[%r]",
                body.get("component_id"),
                RESOLVED_PAYLOAD_META_KEY,
            )
            return {
                "outcome": "refusal",
                "refusal": harness_refusal(
                    code="resolved_revision_unavailable",
                    message=(
                        "the platform stored the card but returned no resolved revision; "
                        "withholding the card rather than sending one with nothing to draw"
                    ),
                ),
            }
        component = await self._assemble_component(reply=body, envelope=envelope)
        if component is None:
            return {
                "outcome": "refusal",
                "refusal": harness_refusal(
                    code="provenance_unavailable",
                    message=(
                        "no source manifest is linked to this frame; withholding the card "
                        "rather than stating a source it cannot support"
                    ),
                ),
            }
        title = (args.get("intent") or {}).get("title") or _envelope_title(envelope)
        summary = f"{component['type']}: {title}" if title else None
        outcome = {"outcome": "ok", "frame_refs": [], "component": component, "summary": summary}
        if isinstance(write_token, str) and write_token:
            # `_RecordingExecutor.__call__` pops this into the conversation's
            # `write_tokens` before the outcome reaches any other reader.
            outcome[WRITE_TOKEN_KEY] = write_token
        return outcome

    async def _indicators_for(
        self, refs: list[str], *, named: Mapping[str, dict[str, str]] | None = None
    ) -> list[dict[str, str]]:
        """The More Info panel's indicator rows for `refs` (deduplicated),
        as real as local data allows: labels and definitions from data-mcp
        `describe`, with a mechanical prettifying of the dcid as the label
        fallback. A describe outage degrades to the fallback and the
        component still writes. A ref in `named` takes its row from there
        and is never described: a family ref (`_family_indicators`) is no
        graph node."""
        named = named or {}
        refs = list(dict.fromkeys(refs))
        missing = [ref for ref in refs if ref not in named and ref not in self._indicator_cache]
        if missing:
            await self._describe_into_cache(missing)
        return [
            {
                "id": ref,
                **(
                    named.get(ref)
                    or self._indicator_cache.get(ref)
                    or {"name": _indicator_name(ref), "description": ""}
                ),
            }
            for ref in refs
        ]

    async def _own_stat_var_indicators(
        self,
        refs: list[str],
        *,
        lineage: Lineage,
        family: Mapping[str, dict[str, str]],
        written: OwnStatVars,
    ) -> dict[str, dict[str, str]]:
        """The indicator row for each ref among `refs` that a verb wrote for
        its own result, named by its two sides in the verb's words
        (`BOTH_SIDES_VERBS`): "X plus Y" for a sum, "X per Y" for a quotient
        and "X as a share of Y" for a share. Two sides holding the same
        StatVars read "X", "Ratio of X" and "Share of X".

        A side's name is its parts' names joined by " and ": a summed
        family's from `family`, an earlier verb's own ref by that verb's
        sides in turn, as dfo-apis labels a published frame, and any other
        StatVar's from `describe`. A share of Debt per Population over GDP per
        Population reads "Debt per Population as a share of GDP per
        Population". A ref whose sides rest on a retrieval missing from
        `lineage` gets no row.
        """
        members = {(member.spec_id, member.content_version): member for member in lineage}
        named: dict[str, dict[str, str] | None] = {}
        for ref in refs:
            if ref in written:
                await self._name_own_stat_var(
                    ref, members=members, family=family, written=written, named=named
                )
        return {ref: row for ref in refs if (row := named.get(ref)) is not None}

    async def _name_own_stat_var(
        self,
        ref: str,
        *,
        members: Mapping[tuple[str | None, str | None], ProvenanceInput],
        family: Mapping[str, dict[str, str]],
        written: OwnStatVars,
        named: dict[str, dict[str, str] | None],
    ) -> dict[str, str] | None:
        """`ref`'s row for `_own_stat_var_indicators`, memoised in `named`,
        or `None` when a side cannot be named."""
        if ref in named:
            return named[ref]
        named[ref] = None
        tool, *sides = written[ref]
        wording = BOTH_SIDES_VERBS[tool].own_stat_var
        if wording is None:
            return None
        named_sides: list[tuple[str, list[str]]] = []
        for side in sides:
            named_side = await self._name_side(
                side, members=members, family=family, written=written, named=named
            )
            if named_side is None:
                return None
            named_sides.append(named_side)
        (first, first_refs), (second, second_refs) = named_sides
        named[ref] = {
            "name": f"{wording.equal_sides}{first}"
            if first_refs == second_refs
            else wording.joiner.join((first, second)),
            "description": wording.description,
        }
        return named[ref]

    async def _name_side(
        self,
        side: Sequence[SidePart],
        *,
        members: Mapping[tuple[str | None, str | None], ProvenanceInput],
        family: Mapping[str, dict[str, str]],
        written: OwnStatVars,
        named: dict[str, dict[str, str] | None],
    ) -> tuple[str, list[str]] | None:
        """One side's name for `_name_own_stat_var`, with the StatVar refs it
        names, or `None` when a part cannot be named or the side is empty."""
        refs: list[str] = []
        own_rows: dict[str, dict[str, str]] = {}
        for part in side:
            if isinstance(part, str):
                if part not in written:
                    return None
                row = await self._name_own_stat_var(
                    part, members=members, family=family, written=written, named=named
                )
                if row is None:
                    return None
                own_rows[part] = row
                refs.append(part)
            elif (member := members.get(part)) is not None:
                refs.extend(_member_stat_vars(member))
            else:
                return None
        if not refs:
            return None
        rows = await self._indicators_for(refs, named={**family, **own_rows})
        return " and ".join(dict.fromkeys(row["name"] for row in rows)), refs

    async def _describe_into_cache(self, refs: list[str]) -> None:
        """Describe `refs` in concurrent batches, filling `_indicator_cache`
        as each batch lands. A batch that fails leaves its unresolved refs
        uncached; answered refs and definitive misses stay cached."""
        client: Client | None = None
        try:
            client = await self._describe_clients.get()
            gate = asyncio.Semaphore(_DESCRIBE_CONCURRENCY)

            async def one(batch: list[str]) -> None:
                async with gate:
                    await self._describe_batch(client, batch)

            outcomes = await asyncio.gather(
                *(
                    one(refs[start : start + _DESCRIBE_BATCH])
                    for start in range(0, len(refs), _DESCRIBE_BATCH)
                ),
                return_exceptions=True,
            )
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            if failures:
                raise failures[0]
        except Exception:
            logger.warning("indicator describe failed; using fallback names", exc_info=True)
            # This degrades to a fallback either way, so it does not need to
            # tell a slow call apart from a broken client: either is reason
            # enough for the next call to reconnect and not repeat it. `client` is `None` only when `get()` itself failed to
            # connect, in which case there is nothing held to evict.
            if client is not None:
                await self._describe_clients.discard(client)
        finally:
            if client is not None:
                await self._describe_clients.release(client)

    async def _describe_batch(self, client: Client, refs: list[str]) -> None:
        """Describe one batch and cache each ref's answer: the node's label
        and definition, or the fallback name for a ref the refusal names as
        a definitive miss. Any other error, and a miss that names no ref of
        this batch, caches nothing."""
        # Refs are "dcid:"-prefixed; describe wants bare dcids.
        by_dcid = {ref.removeprefix("dcid:"): ref for ref in refs}
        result = await client.call_tool(
            "describe", {"ids": list(by_dcid)}, meta=session_call_meta(client), raise_on_error=False
        )
        nodes: list[dict[str, Any]] = []
        if result.is_error:
            # node_not_held is a definitive answer; anything else may be
            # transient -- do not poison the cache.
            body = result.structured_content or {}
            if body.get("code") != "node_not_held":
                return
            # The refusal names the unheld dcids in `fix.dcids` and carries
            # no `nodes`, so the batch's held refs are described again on
            # their own rather than cached under their fallback names.
            unheld = [
                by_dcid[dcid]
                for dcid in (body.get("fix") or {}).get("dcids") or []
                if dcid in by_dcid
            ]
            if not unheld:
                return
            held = [ref for ref in refs if ref not in unheld]
            if held:
                for ref in unheld:
                    self._indicator_cache[ref] = {"name": _indicator_name(ref), "description": ""}
                await self._describe_batch(client, held)
                return
        else:
            nodes = (result.structured_content or {}).get("nodes") or []
        for ref in refs:
            self._indicator_cache[ref] = {"name": _indicator_name(ref), "description": ""}
        for node in nodes:
            ref = by_dcid.get(str(node.get("dcid", "")))
            if ref:
                self._indicator_cache[ref] = {
                    "name": str(node.get("label") or _indicator_name(ref)),
                    "description": str(node.get("definition") or ""),
                }

    async def _assemble_component(
        self, *, reply: dict[str, Any], envelope: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build the wire `component.created` body from a `write_component`
        result.

        `reply` is the result's model vocabulary (`component_id`,
        `revision`, the platform's fact summary) and `envelope` the resolved
        revision presentation-mcp carries beside it in
        `_meta[RESOLVED_PAYLOAD_META_KEY]`. The envelope becomes the wire
        `payload` verbatim and answers `type`, `catalog_version` and the
        frame reference (`provenance.inputs[0]`, the frame the stored intent
        names, which a revise patch does not carry). The wire fact summary is
        mapped from the platform's, and the wire provenance is assembled
        from the cached mint manifest(s) of that frame.

        A frame resting on several retrievals (a join, a compare) has a
        lineage of several manifests (`harness.provenance.ProvenanceStore`),
        which `merged_manifest` folds into one before the wire shapes are
        built; `_wire_provenance` also lists each retrieval as an input.

        The indicators listed are the StatVars the chart draws
        (`_drawn_stat_vars`), a summed family's ref named as dfo-apis names
        it (`_family_indicators`) and a `sum_of`, `quotient_of` or `share_of`
        result's ref by its two sides (`_own_stat_var_indicators`).
        `indicator_count` counts the StatVars of the frame the card draws
        from, which is the count the step that produced the frame declared
        (`ProvenanceStore.stat_var_count`), or, for a
        retrieval's frame held without one, the StatVars its manifest
        summed or, for an unsummed one, resolved.

        Returns `None`, refusing to assemble, when the frame has no lineage.
        It was never linked
        (`harness.graph._link_evidence_to_terminal_frames`, which links
        nothing when a retrieval it rests on has no manifest), the store has
        evicted it, or the envelope names no input frame at all. A card
        built anyway would state a source it cannot support, which is worse
        than no card. `_presentation` turns a `None` here into a
        `provenance_unavailable` refusal.
        """
        component_id = reply["component_id"]
        revision = int(reply["revision"])
        catalog_version = str(envelope.get("catalog_version") or "unversioned")
        platform_facts = reply.get("fact_summary") or {}
        inputs = (envelope.get("provenance") or {}).get("inputs") or []
        frame_ref = inputs[0] if inputs else {}
        lineage = self._provenance.get(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.get("spec_id"),
            content_version=frame_ref.get("content_version"),
        )
        frame_stat_var_count = self._provenance.stat_var_count(
            conversation=current_conversation_id.get(),
            spec_id=frame_ref.get("spec_id"),
            content_version=frame_ref.get("content_version"),
        )
        if lineage is None:
            logger.warning(
                "write_component withheld for component_id=%r: no provenance is linked to "
                "frame spec_id=%r content_version=%r -- the card would have stated a source "
                "it cannot support",
                component_id,
                frame_ref.get("spec_id"),
                frame_ref.get("content_version"),
            )
            return None
        manifest = merged_manifest(lineage)
        drawn = _drawn_stat_vars(platform_facts)
        family = _family_indicators(lineage)
        written = await self._own_stat_var_indicators(
            drawn,
            lineage=lineage,
            family=family,
            written=self._provenance.own_stat_vars(
                conversation=current_conversation_id.get(),
                spec_id=frame_ref.get("spec_id"),
                content_version=frame_ref.get("content_version"),
            ),
        )
        indicators = await self._indicators_for(drawn, named={**family, **written})
        return {
            "component_id": component_id,
            "revision": revision,
            "type": str(envelope["component"].get("type", "")),
            "catalog_version": catalog_version,
            "min_catalog_version": catalog_version,
            "payload": envelope,
            "fact_summary": _wire_fact_summary(
                platform_facts,
                component=envelope["component"],
                component_id=component_id,
                revision=revision,
                manifest=manifest,
                indicators=indicators,
                indicator_count=(
                    frame_stat_var_count
                    if frame_stat_var_count is not None
                    else summed_stat_var_count(manifest) or len(_manifest_stat_vars(manifest))
                ),
            ),
            "provenance": _wire_provenance(
                frame_ref=frame_ref,
                lineage=lineage,
                manifest=manifest,
                platform_facts=platform_facts,
            ),
        }


def _safe_to_repeat(tool: str, refusal: dict[str, Any]) -> bool:
    """Whether a refused `tool` call may be made again.

    `write_component` mints a fresh write token per call, so repeating it after
    a refusal the platform may have committed (`transient_unavailable`, any
    other retryable code) can store a second draft. Only the row budget's
    refusal, whose `fix` carries `rows_requested`, is raised before any write.
    Reads and idempotent mints repeat freely.
    """
    if tool != "write_component":
        return True
    fix = refusal.get("fix")
    return isinstance(fix, dict) and "rows_requested" in fix


def _envelope_title(envelope: dict[str, Any]) -> str | None:
    """The resolved component's title text, or `None` for a member that
    carries no title (a note, a source card)."""
    title = envelope["component"].get("title")
    return title.get("text") if isinstance(title, dict) else None


def _refusal_outcome(
    structured: dict[str, Any] | None, *, fallback_text: str = ""
) -> dict[str, Any]:
    """A platform refusal as the executor's `{"outcome": "refusal", ...}`.

    `origin` is `"service"`, since an MCP service minted the refusal. The
    two refusals this module builds itself (`unknown_tool`, and
    `provenance_unavailable` after the platform accepted a write) carry
    `"harness"`, so a consumer can tell who declined. `advice` is a
    presentation service's display-text `fix` and the only key that holds
    the violated constraint id. `platform_trace_id` is the platform's own
    trace id for the refused call, present only when the body carries a
    non-empty string. It serves trace and log correlation; no user or model
    sees it.
    """
    body = structured or {}
    fix = body.get("fix")
    refusal: dict[str, Any] = {
        "code": refusal_code(body) or "unspecified",
        "retry_mode": str(body.get("retry_mode") or body.get("retry") or "do_not_retry"),
        "origin": REFUSAL_ORIGIN_SERVICE,
        # data/analysis refusals carry a machine-mergeable fix mapping;
        # presentation's fix is display text. The graph merges only
        # mapping fixes into retried args; a string rides along as `advice`.
        "fix": fix if isinstance(fix, dict) else None,
        "advice": fix if isinstance(fix, str) else None,
        # fastmcp's own arg-validation errors carry NO structured content;
        # without the raw text the replanner redrafts blind.
        "message": str(body.get("message") or body.get("detail") or fallback_text),
    }
    platform_trace_id = platform_trace_id_of(body)
    if platform_trace_id:
        refusal["platform_trace_id"] = platform_trace_id
    # Read by `McpExecutor.__call__`'s retry wait, which ignores an unusable value.
    if body.get("retry_after_s") is not None:
        refusal["retry_after_s"] = body["retry_after_s"]
    return {"outcome": "refusal", "refusal": refusal}


def _first_line(text: str, *, limit: int = 160) -> str | None:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line:
        return None
    return line if len(line) <= limit else line[: limit - 1] + "\u2026"


def _manifest_summary(manifest: dict[str, Any]) -> str | None:
    """A one-line human result for the Activity feed ("64 observations,
    1960\u20132023 \u00b7 WorldDevelopmentIndicators"), read off the wire
    manifest's coverage, series count and citation, with no model involved."""
    coverage = manifest.get("coverage") or {}
    window = coverage.get("window") or {}
    start, end = window.get("start_year"), window.get("end_year")
    citation = manifest.get("citation")
    citation_text = citation.get("text") if isinstance(citation, dict) else citation
    series_count = manifest.get("series_count")

    extent = []
    if rows := coverage.get("rows"):
        extent.append(f"{rows} observations")
    if start is not None and end is not None:
        extent.append(str(start) if start == end else f"{start}\u2013{end}")
    parts = [", ".join(extent)] if extent else []
    if citation_text:
        parts.append(str(citation_text))
    if series_count and series_count > 1:
        parts.append(f"{series_count} series")
    return _first_line(" \u00b7 ".join(parts)) if parts else None


def _result_text(result: Any) -> str:
    """Plain text of a tool result's content blocks, for refusal fallback."""
    return " ".join(
        text for block in (result.content or []) if (text := getattr(block, "text", ""))
    )


def _wire_fact_summary(
    platform_facts: dict[str, Any],
    *,
    component: dict[str, Any],
    component_id: str,
    revision: int,
    manifest: dict[str, Any] | None,
    indicators: list[dict[str, str]],
    indicator_count: int,
) -> dict[str, Any]:
    """The wire fact summary. `indicator_count` is the frame's StatVar count,
    which the label states. `indicators` lists the StatVars the chart draws,
    so a list shorter than the count is a partial one. Each series carries
    the periods of its first and last values, read from the drawn points:
    a series with gaps starts or ends inside the window. `unit` is the
    platform's, forwarded as it is: the unit slot of the card's payload."""
    platform_ids = Counter(entry.get("id") for entry in platform_facts.get("series") or [])
    periods = {
        series_id: dated
        for series_id, dated in _endpoint_periods(component).items()
        if platform_ids[series_id] == 1
    }
    series = [
        {
            "id": str(entry.get("id", "")),
            "label": str(entry.get("label") or entry.get("id", "")),
            "point_count": int(entry.get("point_count") or 0),
            "coordinates": {
                key: value
                for key, value in (entry.get("coordinates") or {}).items()
                if isinstance(value, str)
            },
            "first": _float_or_none(entry.get("first_value")),
            "last": _float_or_none(entry.get("last_value")),
            **periods.get(str(entry.get("id", "")), {}),
        }
        for entry in platform_facts.get("series") or []
    ]
    citation = (manifest or {}).get("citation") or {}
    source_label = citation.get("text") or _platform_source(platform_facts) or _UNKNOWN_SOURCE
    total = max(indicator_count, 1)
    summary: dict[str, Any] = {
        "component_id": component_id,
        "revision": revision,
        "series": series,
        "indicators": indicators,
        "attribution": {
            "label": f"{source_label} · {total} indicator(s)",
            "source_count": max(len(citation.get("sources") or []), 1),
            "indicator_count": total,
        },
    }
    window = platform_facts.get("window") or _drawn_window(component)
    if window.get("start") and window.get("end"):
        summary["window"] = {"start": str(window["start"]), "end": str(window["end"])}
    if (unit := platform_facts.get("unit")) is not None:
        summary["unit"] = unit
    if defaults := platform_facts.get("defaults"):
        summary["defaults"] = [
            {"pointer": str(entry["pointer"]), "detail": str(entry["detail"])} for entry in defaults
        ]
    return summary


def _endpoint_periods(component: dict[str, Any]) -> dict[str, dict[str, str]]:
    """`{first_period, last_period}` for each drawn series, by id: the
    periods of its first and last observed points, chosen as the platform
    chooses the points its `first_value` and `last_value` come from (a point
    holding a `value`; one marked missing holds none). Only these two
    periods leave the points, never the values between. An endpoint without
    a period gets none, and a series whose id is missing or repeated gets
    nothing, so a period is never attached to another value. Stat cards and
    tables draw no `series` and get nothing."""
    drawn_series = component.get("series") or []
    ids = Counter(drawn.get("id") for drawn in drawn_series)
    periods: dict[str, dict[str, str]] = {}
    for drawn in drawn_series:
        series_id = drawn.get("id")
        if not isinstance(series_id, str) or not series_id or ids[series_id] != 1:
            continue
        observed = [point for point in drawn.get("points") or [] if "value" in point]
        if not observed:
            continue
        endpoints = {
            key: str(point["period"])
            for key, point in (("first_period", observed[0]), ("last_period", observed[-1]))
            if point.get("period") is not None
        }
        if endpoints:
            periods[series_id] = endpoints
    return periods


def _manifest_stat_vars(manifest: dict[str, Any] | None) -> list[str]:
    """The distinct StatVar refs the manifest resolved (manifests may repeat one)."""
    entries = ((manifest or {}).get("trace") or {}).get("resolved_stat_vars") or []
    return list(dict.fromkeys(str(entry["ref"]) for entry in entries if entry.get("ref")))


def _family_indicators(lineage: Lineage) -> dict[str, dict[str, str]]:
    """The indicator row for each family ref a frame in `lineage` can carry.

    A summed retrieval's frame holds one StatVar ref, `dcid:<spec_id>` of
    its own definition. It is named by `get_data`'s `summed_stat_var_label`,
    the label dfo-apis's presentation door gives the same ref on the chart,
    or by the retrieval's citation text, the source the card's attribution
    names, when dfo-apis stated none. Every member gets a row, so no family
    ref reaches `describe`. Its description counts the StatVars the sum
    added, when known.
    """
    rows: dict[str, dict[str, str]] = {}
    for member in lineage:
        manifest = member.manifest
        if member.spec_id is None:
            continue
        label = manifest.get("summed_stat_var_label")
        citation = manifest.get("citation") or {}
        name = label if isinstance(label, str) and label else citation.get("text")
        count = summed_stat_var_count(manifest)
        rows[f"dcid:{member.spec_id}"] = {
            "name": str(name or _UNKNOWN_SOURCE),
            "description": f"A total over {count} indicators." if count is not None else "",
        }
    return rows


def _member_stat_vars(member: ProvenanceInput) -> list[str]:
    """The StatVar refs a retrieval's frame carries: its family ref when
    summed (`_family_indicators`), else the StatVars its manifest resolved."""
    if member.spec_id is not None and summed_stat_var_count(member.manifest) is not None:
        return [f"dcid:{member.spec_id}"]
    return _manifest_stat_vars(member.manifest)


def _drawn_stat_vars(platform_facts: dict[str, Any]) -> list[str]:
    """The distinct StatVars the chart's series carry in their `stat_var`
    coordinate. Empty when no series has one, as in a groupby rendering."""
    refs = (
        (entry.get("coordinates") or {}).get("stat_var")
        for entry in platform_facts.get("series") or []
    )
    return list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref))


def _drawn_window(component: dict[str, Any]) -> dict[str, Any]:
    """The periods a chart draws, read from its axis. The platform states no
    window for a bar chart of entities at one period, and that chart draws
    its `period` alone. A point is a one-period window, and an interval
    such as `2018/2022` draws from its start to its end. A table has no
    axis and no window."""
    axis = component.get("x_axis") or component.get("category_axis") or {}
    if window := axis.get("window"):
        return window
    period = axis.get("period")
    if not period:
        return {}
    parsed = parse_time_period(str(period))
    return {"start": parsed.start, "end": parsed.end}


def _indicator_name(ref: str) -> str:
    """A display name derived mechanically from the stat var id --
    "dcid:LifeExpectancy_Person" -> "Life Expectancy Person". Opaque codes
    (e.g. WDI's "BX.GRT.EXTA.CD.DT") pass through unchanged."""
    bare = ref.removeprefix("dcid:")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", bare.replace("_", " ")).strip()
    return spaced or ref


def _platform_source(platform_facts: dict[str, Any]) -> str | None:
    for entry in platform_facts.get("series") or []:
        source = (entry.get("coordinates") or {}).get("provenance")
        if isinstance(source, str) and source:
            return source
    return None


def _wire_provenance(
    *,
    frame_ref: dict[str, Any],
    lineage: Lineage,
    manifest: dict[str, Any],
    platform_facts: dict[str, Any],
) -> dict[str, Any]:
    """Wire provenance built from the manifests actually linked to `frame_ref`.

    `lineage` is never empty and `manifest` is `merged_manifest(lineage)`.
    `_assemble_component` calls this only for a frame that has a lineage,
    since a card that states a source it does not have is worse than no card.

    A frame resting on one retrieval lists itself as the one input, and a
    manifest with no creation time gets today's date as `last_refreshed`. A
    frame resting on several lists itself first, then each retrieval it rests
    on. Its `last_refreshed` is `""` when any retrieval's manifest states no
    creation time, as `data_through` is `""` when the end year is missing.

    Each source's `licence` is the `license` its manifest citation states, or
    `_LICENCE_NOT_STATED` when the field is absent or blank.

    A retrieval, the card's own frame among them, is listed under
    `ProvenanceInput.served_ref`, the pair the datasets door serves, while the
    lineage stays keyed by its frame key.
    """
    now = datetime.now(UTC)
    served = {(member.spec_id, member.content_version): member.served_ref for member in lineage}
    own = served.get((frame_ref.get("spec_id"), frame_ref.get("content_version")), frame_ref)
    inputs: list[dict[str, Any]] = (
        [own]
        if own.get("spec_id") and own.get("content_version")
        else [{"spec_id": "unknown", "content_version": "unknown"}]
    )
    if len(lineage) > 1:
        for member_ref in served.values():
            if member_ref["spec_id"] and member_ref["content_version"] and member_ref not in inputs:
                inputs.append(member_ref)
    citation = manifest.get("citation") or {}
    coverage_window = (manifest.get("coverage") or {}).get("window") or {}
    created_at = data_artifact(manifest.get("artifacts") or []).get("created_at")
    if created_at:
        last_refreshed = str(created_at).split("T", maxsplit=1)[0]
    elif len(lineage) > 1:
        last_refreshed = ""
    else:
        last_refreshed = now.isoformat().split("T", maxsplit=1)[0]
    return {
        "inputs": inputs,
        "sources": [
            {
                "id": str(source.get("ref") or source.get("label") or "unknown"),
                "name": str(source.get("label") or source.get("ref") or "unknown"),
                "licence": str(source.get("license") or "").strip() or _LICENCE_NOT_STATED,
            }
            for source in citation.get("sources") or [{}]
        ],
        "freshness": {
            "data_through": str(coverage_window.get("end_year") or ""),
            "last_refreshed": last_refreshed,
            "partial_period": False,
        },
        "warnings": [
            {
                "code": str(warning.get("code", "unspecified")),
                "severity": _SEVERITY_TO_WIRE.get(str(warning.get("severity", "")), "warn"),
                "message": str(warning.get("message", "")),
                **({"spec_id": str(warning["spec_id"])} if warning.get("spec_id") else {}),
            }
            for warning in manifest.get("warnings") or []
        ],
        "generated_at": now.isoformat(),
        "citation": str(citation.get("text") or "unknown"),
    }


__all__ = ["McpExecutor", "McpResolver"]
