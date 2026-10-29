"""Run an authorised workflow, step by step, and check what comes back.

`execute_authorised` admits a run to the run ledger (`executor.ledger`)
under a plan's approval (`PlanApproval`) before anything is dispatched,
records each tool attempt before it goes out, seals the retrieved inputs and
checks them against the approval before the first derived step, and after an
accepted run publishes each derived terminal frame with its replay recipe
(`executor.completion`). `execute_approved_plan` runs a plan through it: it
moves a plan `APPROVED` to `EXECUTING`, runs it under a `PlanApproval` of
its approved revision, and leaves it `COMPLETED` once the run completed,
`EXECUTING` while the run's publication is unfinished, and `FAILED` when the
run stopped or its publication failed. A plan that is not `APPROVED` is
refused (`NOT_APPROVED`) without being touched; one whose content no longer
hashes to what was approved is refused (`STALE_REVISION`) by
`start_execution`. `resume_run` carries on a run a restart left unfinished,
from the submission stored with it (`load_submission`). `_run_workflow` is
the step loop underneath them, private because a workflow runs only as an
admitted run.

`_run_workflow` runs the plan's compiled `Workflow` as the dependency graph
its own bindings state, not as a plain list: every step with no unmet frame
binding runs in one wave, and each further wave holds only the steps whose
bindings all name a step an earlier wave already ran. A wave's steps run
concurrently, bounded by `max_concurrent_steps` (`_Run`'s semaphore, held
for a step's own run including its retries), and the wave runs every one of
its steps to completion before a stop native to any of them ends the run,
so what a stop's outcome reports as `completed` names every step that
finished by then, in the plan's own step order rather than the order they
happened to finish in. A stop's own `also_stopped` names the wave's other
stops, wave order, so a wave that surfaces two stale sources lets one
replan fix both. Before the first call it re-reads
the tool surface and refuses to run if any step's tool has changed since the plan was
compiled -- its argument schema, its output schema, or its declared version.
For each step it resolves the bindings from the frames earlier steps
produced, validates the full arguments against the tool's live schema,
calls the tool, and reads the frame reference the step was compiled to
produce. A retrieval that is a requirement's evidence is then checked
against the source the plan was approved with -- the one the step's
`EvidenceBinding` names, never one looked up by whatever `spec_id` the tool
happened to return -- through the same `check_evidence` the compiler ran
before approval, this time on the manifest the tool returned.

Transient failures are retried, with the delay a refusal names (capped) or a
doubling backoff, spread by jitter (`executor.classify.jittered_retry_delay`)
so calls refused together retry apart. A `ToolSurfaceUnavailableError` -- the call never reached
the server, or its answer never came back -- is retried up to
`MAX_ATTEMPTS_PER_STEP` unless the step's tool capability sets
`retry_on_transport_failure=False`, in which case the first such failure is
final: `get_data`'s mint is not safely retryable, so a transport failure on
it is never retried, however early in the attempt count it happens. This is
a per-tool boolean read live from the capability table at run time. It
carries no attempt count or backoff of its own, and it is never hashed into
a plan's revision.
Nothing else is retried: a refusal that carries a choice
goes back to the planner as `MethodChoiceNeeded`; one only a fresh discovery
can fix -- the source's upstream version changed, or its approved `spec_id`
is no longer held -- comes back as `RediscoveryNeeded` naming the requirement
and source off the step's `EvidenceBinding`, since a bare retry reproduces it
identically and only a new discovery, under a new plan revision, can change
what gets retrieved; the same refusal on a step bound to no evidence has
nothing to rediscover and stops the run as `Refused`, as does any other
refusal. `executor.classify` places every server code. The executor never
corrects a call, because a corrected call is a different plan.

A step whose tool capability pins `call_timeout_s` (`get_data`, among the
workflow's tools) runs under one absolute budget, `step_budget_s`, covering
every attempt and every retry delay at their own worst case, so it never cuts
an attempt the retry policy would have made, plus `EVIDENCE_ALLOWANCE_S` on
an evidence-bearing step. Each call runs inside it, so its timeout is clipped
to what remains. The evidence check's own statistics fetch runs under a
deadline built from what is left of the step budget once the check starts,
capped at `EVIDENCE_ALLOWANCE_S` (`default_fetch_series_stats`). A fetch that
outruns it, or fails in transit, stops the step `TRANSPORT_FAILED`, the same
refusal a step that runs past its own budget gets, and a statistics artifact
the run cannot trust stops it `STEP_RESULT_UNUSABLE`; neither is a fact of
the data, so neither is reported as an unmet requirement. A tool with no
pinned timeout runs at the surface's default per call and has no statistics
fetch, so each of its attempts is already bounded and no step budget applies.

The compiler names role columns in the approved source's vocabulary (a flow
axis such as `origin`), while a retrieved frame names them by the dataset's
declared properties (`financingProvider`) and may carry more of them
(`stat_var_*` columns). `get_data`'s result reports the frame's `role_keys`
and each coverage dimension's flow axis, and the executor binds the plan's
columns to the frame's (`compiler.frame_columns`): before each step downstream
of a retrieval runs, its column-naming arguments are spelled the way the frame
spells them, and a role the frame carries no column for refuses
`COLUMN_UNBOUND` before the call. A retrieval whose result carries no
`role_keys` leaves the plan's own spelling in place.

The StatVars a frame holds are the minted frame's too: a family retrieval
holds one per member the source has data for, or one when it asks for the
family summed. `get_data` and every analysis verb declare their frame's
`stat_var_count`; a summed retrieval must declare one, and the count any
other step may declare is bounded by the count of the frame it reads
(`compiler.frame_columns.bound_stat_var_count`).

A step whose plan states a period against the newest data is rolling
(`compiler.rolling.is_rolling`): its literal years are the ones the plan read
from Search, and the run writes them again from the data it retrieved. The
rolling steps that some step reads together form a group
(`compiler.rolling.anchor_groups`), and a group stands on one newest period,
taken over the retrievals its steps rest on. A retrieval no step of the group
reads has no part in it. Every step of a group runs after every retrieval of
the group (`_waves`). Once those are in and before the group's first step, the
run calls analysis-mcp's `resolve_anchor` once for the group, with the
populations its steps carry, spelled the way the retrieved frames spell their
columns, and the frames of its retrievals. The answer is the newest year at
least half of the series of each population hold, and the key of its stored
resolution. Each of the group's steps has its periods written as that year
moved by the step's offsets (`compiler.rolling.resolved_arguments`), and the
key goes with the call. The call takes the path a step's call takes
(`_Run._call`), reports under the id of the group's first step to run, and its
tool is part of the surface check. A plan with no rolling step never calls it.
The outcome carries each group's answer in `anchors`.

A group's rolling literals, less their offsets, state the newest period the
plan was checked against. The run reads the plan's years moved later. A
resolved period before the stated one is refused
`ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN` before the group's first step, with
a message asking the user to ask again, so the plan is made from the period
the data holds. `read_anchors` takes the same path before a plan is
approved, so translation states the plan's rolling years from the period its
run will resolve. It runs the retrievals each group rests on and
`resolve_anchor`, with no ledger, no derived step and no check against the
stated period.

A rolled start before the first year its dataset holds is refused
`ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET` before the step's call, with the first year
read off the retrieval's own evidence. `resolve_anchor` finding no qualifying
year, or periods that are no calendar years, stops the run with the tool's
code and a message for the user. A `rolling_anchor_unresolved` of any other
detail says the run's own populations, roots, key or periods do not fit
(`executor.classify.is_anchor_fault`) and stops it as an `internal_error`.

A keyed `quotient_of` or `share_of` leaves out each numerator key it finds
nothing to divide by, and a `join` each entity of either side that matches
in no period. The result declares how many (`dropped_key_count`, and
`right_dropped_key_count` for a join's right side). The step's `StepRecord`
carries both counts, so an answer resting on the step can say its
population lost members.

Three checks run on what a step's own result declares, in `structured_content`
rather than the digest prose, which the executor never reads for anything
load-bearing:

- `_run_step` compares a successful analysis-mcp result's own `role_keys` and
  `stat_var_count` against the shape the step must produce, refusing
  `SHAPE_MISMATCH` on a difference. The role keys are the compiler's
  `expected_role_keys` spelled the way the frame read spells them, except for
  a filter, which must declare every role column of its input
  (`bound_role_keys`). A non-empty result declares at least one StatVar and
  at most the verb's bound (`bound_stat_var_count`): one for `growth`, `rank`,
  `share_of` and an `aggregate` that does not group by `stat_var`, two for a
  `join`, and the count its input declared for a `filter`, a `compare` and an
  `aggregate` grouping by `stat_var`. A retrieval's role keys and StatVar
  count are the frame's, so only a retrieval asked to sum its family is
  checked there: it must declare one StatVar, and one that declares more
  stops with an `ExecutorCode.FAMILY_NOT_SUMMED` payload, which ends the turn.
- `declared_shape` refuses `STEP_RESULT_UNUSABLE` when a result's declared
  `role_keys` are not valid role columns, so every role key a later step
  binds to has passed `normalize_role_keys`.
- `_answer_refusal` refuses `EMPTY_ANSWER` when a terminal step's own
  capability says an empty result is not an answer (`empty_is_an_answer`,
  `True` on `filter` alone) and the call declared zero rows, and refuses
  `STEP_RESULT_UNUSABLE` when the tool declares an output schema but the
  call carried no row count -- the schema promises the field, so its absence
  is a broken contract, not an unknown. A run that stops carries the
  finished terminal frames this check lets stand as its `answers`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial

import httpx2
import pyarrow as pa
from common.materialisation import FrameRef
from common.refusal import internal_error_message
from common.telemetry import current_trace_id
from compiler.arguments import literal_argument_errors
from compiler.capabilities import CAPABILITIES, Extract, ToolCapability
from compiler.evidence import (
    EvidenceIssue,
    Severity,
    check_evidence,
    errors,
    evidence_from_manifest,
    left_out_entities,
    unlabelled_series_refs,
    with_series_labels,
)
from compiler.frame_columns import (
    FRAME_PARAMETER,
    FrameColumns,
    UnboundColumnError,
    bound_arguments,
    bound_populations,
    bound_role_keys,
    bound_stat_var_count,
    entity_narrowings,
)
from compiler.rolling import (
    AnchorGroup,
    RollingBound,
    anchor_groups,
    is_rolling,
    resolved_arguments,
    rolling_bounds,
)
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
    ToolSurfaceUnavailableError,
    UnhashableSchemaError,
    deliver_progress,
)
from data_mcp.artifact_resource import ArtifactRefusalError
from data_mcp.deadline import CallDeadline
from data_mcp.manifest import Manifest
from data_mcp.node_lookup import describe_nodes
from data_mcp.series_statistics import (
    SERIES_STATISTICS_ARROW_SCHEMA,
    SeriesStatisticsSchemaError,
    continuing_method,
    fetch_series_statistics_table,
    find_statistics_artifact,
)
from data_mcp.stat_var_list import with_stat_var_list, without_long_stat_var_list
from dfo_frame_grammar import shift_period
from fastmcp.exceptions import ResourceError
from planner.plan import DatasetRequirement, Plan, PlanStatus, plan_revision
from planner.records import StrictModel
from planner.workflow import (
    ArtifactKind,
    EvidenceBinding,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import JsonValue, ValidationError

from executor.artifacts import (
    ArtifactError,
    declared_dropped_key_counts,
    declared_ranked_before_limit,
    declared_shape,
    extract_frame_ref,
    manifest_frame_ref,
    parse_manifest,
    retrieved_frame_columns,
)
from executor.classify import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS_PER_STEP,
    MAX_RETRY_AFTER_SECONDS,
    PAIRING_SHARES_NO_KEY,
    ROLLING_ANCHOR_UNRESOLVED,
    TRANSIENT_CODES,
    Disposition,
    ExecutorCode,
    Jitter,
    ToolFailure,
    candidates_of,
    classify_refusal,
    delay_before_retry,
    is_anchor_fault,
    jittered_retry_delay,
    parse_tool_failure,
)
from executor.completion import ResultPublisher, publish_results
from executor.ledger import (
    TERMINAL_STATUSES,
    Attempt,
    AuthorisationExpiredError,
    Fail,
    InMemoryRunLedger,
    LedgerUnavailableError,
    ManifestEntry,
    ManifestOutsideAuthorisationError,
    PlanApproval,
    Publishing,
    ResultFrame,
    Resume,
    RunChange,
    RunLedger,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    Seal,
    SourcePolicy,
    Start,
    idempotency_key,
    new_run,
    record_change,
    store_faults,
    submission_content,
    submission_digest,
)
from executor.outcomes import (
    Accepted,
    AnchorResult,
    EvidenceRecord,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    RunOutcome,
    StepRecord,
    outcome_from_json,
)

logger = logging.getLogger(__name__)

# The part of a step budget reserved for the evidence check after the last
# attempt: `fetch_series_statistics_table` signs the artifact's URL and GETs it
# under a deadline of its own -- what remains of the step budget when the
# check starts, capped at this many seconds (`default_fetch_series_stats`).
EVIDENCE_ALLOWANCE_S = 30.0

# Subtracted from that deadline so the fetch's own internal `asyncio.timeout`
# always expires before the step's outer one: `asyncio.timeout` only converts
# a cancellation into `TimeoutError` for the context whose own deadline fired,
# so two deadlines landing at the same instant could let the outer budget's
# cancellation reach `_check_evidence` as a bare `CancelledError` instead of
# the `TimeoutError` `default_fetch_series_stats` reports as
# `StatisticsUnreachableError`.
_EVIDENCE_DEADLINE_MARGIN_S = 1.0

# What the user reads when a retrieval asked to sum its family of StatVars
# returns them unsummed (`_Run._check_summed_retrieval`): why there is no
# total, and what to ask instead.
_FAMILY_NOT_SUMMED_MESSAGE = (
    "The data service returned this measure's component series side by side and did "
    "not add them up, so there is no total to report."
)
_FAMILY_NOT_SUMMED_REMEDY = (
    "Ask for the component series separately, or ask for the total again later."
)

# The tool that resolves the newest period a run's rolling steps stand on.
_RESOLVE_ANCHOR = ToolId(server="analysis-mcp", name="resolve_anchor")

# What the user reads when `resolve_anchor` cannot name a newest period, by the
# tool's code, and what to ask instead.
_ANCHOR_REFUSAL_MESSAGES = {
    ROLLING_ANCHOR_UNRESOLVED: (
        "No recent year is reported by at least half of the series this question reads, "
        "so there is no latest year to count the period from."
    ),
    "rolling_grain_unsupported": (
        "This data is not reported by calendar year, so a period counted from its latest "
        "year cannot be worked out."
    ),
}
_NAMED_YEARS_REMEDY = "Ask for the period in named years instead."
# The message and remedy for a `rolling_anchor_unresolved` whose `fix.detail`
# says more than its code.
_ANCHOR_DETAIL_WORDING = {
    PAIRING_SHARES_NO_KEY: (
        "The two sides this question pairs name no place or party in common, so "
        "nothing can be matched across them in any year.",
        "Ask about a place or party both sides report on.",
    ),
}

# The comparisons under which a rolled period threshold is where a filter's
# rows start, and how far past the threshold the first year kept is.
_YEARS_PAST_THRESHOLD = {"gte": 0, "eq": 0, "gt": 1}
_WINDOW_START = "window_start"


def _first_year_kept(bound: RollingBound) -> int | None:
    """The first year a rolled `bound` keeps, or `None` when it sets no start."""
    if bound.op is None:
        return int(bound.period) if bound.bound == _WINDOW_START else None
    past = _YEARS_PAST_THRESHOLD.get(bound.op)
    return None if past is None else int(bound.period) + past


def step_budget_s(call_timeout_s: float, *, evidence: bool) -> float:
    """The absolute budget for one step whose every call times out at `call_timeout_s`.

    `MAX_ATTEMPTS_PER_STEP` calls at their full timeout, the longest delay
    `jittered_retry_delay` can ask for before each retry (a wait capped at
    `MAX_RETRY_AFTER_SECONDS`, or the doubling backoff when that is longer),
    and `EVIDENCE_ALLOWANCE_S` when the step's result is checked as evidence.
    For `get_data`, `3 * 180 + 2 * 30 + 30 = 630` seconds.
    """
    delays = sum(
        max(MAX_RETRY_AFTER_SECONDS, BACKOFF_SECONDS * 2**attempt)
        for attempt in range(MAX_ATTEMPTS_PER_STEP - 1)
    )
    allowance = EVIDENCE_ALLOWANCE_S if evidence else 0.0
    return MAX_ATTEMPTS_PER_STEP * call_timeout_s + delays + allowance


# The id of the step whose tool call is in flight, for the duration of that
# step's own `_run_step` (retries included). This module has no notion of
# "recording" or "conversation" -- a caller-side recorder for a served turn
# (`harness.studio_translate._RecordingSurface`) reads it to tag its own
# per-call records, the same way it reads `harness.conversation.
# current_conversation_id`, without this module importing anything from the
# harness layer.
current_step_id: ContextVar[str | None] = ContextVar("executor_current_step_id", default=None)

# What remains of the running step's own budget, set for the whole of
# `_run_step`'s body -- calls and evidence check alike, unlike `current_step_id`
# above, which only covers `_call`. `default_fetch_series_stats` reads this
# to bound `fetch_series_statistics_table`'s own deadline without `FetchSeriesStats`
# naming a budget every injected implementation (`harness.scripted_plan
# .scripted_fetch_series_stats` among them) would otherwise have to accept
# and ignore. `None` when the step's tool pins no call timeout, the same
# case `_budget_s` answers `None` for.
current_step_deadline: ContextVar[CallDeadline | None] = ContextVar(
    "executor_current_step_deadline", default=None
)

# The running step's budget (`_Run._run_step`), which `_Run._call` stops
# while it records an attempt, so the budget covers the step's tool calls and
# the waits between them and a slow ledger write is never read as a call that
# ran past its time. `None` outside a step, or for a step with no budget.
_step_budget: ContextVar[asyncio.Timeout | None] = ContextVar("executor_step_budget", default=None)

Sleep = Callable[[float], Awaitable[None]]

# `evidence_from_manifest`'s `series_stats` argument, resolved for one manifest. Injectable
# so a caller whose tool surface is itself fully virtualised -- `harness.scripted`'s
# deterministic mode states "no upstream" as its whole point -- can answer this from its own
# canned data instead of `fetch_series_statistics_table`'s real signed-URL fetch, the same way it
# already supplies its own `surface`/`executor`/`resolver` in place of the real ones.
FetchSeriesStats = Callable[[Manifest], Awaitable[pa.Table | None]]

# The names the graph gives each of some entity refs, keyed by ref; a ref it
# names none for is absent. Injectable as `FetchSeriesStats` is. One that
# raises or runs past its deadline names nothing (`_series_labels`).
LabelRefs = Callable[[Sequence[str]], Awaitable[Mapping[str, str]]]

# The most entity refs one evidence check asks names for: two requests of
# `describe_nodes`' 50. The largest short series' come first
# (`unlabelled_series_refs`), and an answer names three of them.
LABELLED_REFS = 100

# The most seconds that lookup may take. Past it the series stay unlabelled,
# and an answer counts them as it would without the lookup.
LABEL_ALLOWANCE_S = 5.0


class StatisticsUnreachableError(Exception):
    """A manifest's statistics artifact could not be fetched this time.

    A timeout, a failed connection, an HTTP 429 or 5xx, or a link lookup the
    data service refused with a transient code (`TRANSIENT_CODES`). A later
    try can get past it, so `_Run._check_evidence` stops the step
    `TRANSPORT_FAILED`.
    """


class StatisticsUnusableError(Exception):
    """A manifest's statistics metadata cannot be trusted or read.

    The manifest names series but carries no statistics artifact, the
    artifact fails its digest, size, schema or series-count check, or the data
    service refused its link or download for good. Nothing the user asked for
    is wrong, so `_Run._check_evidence` stops the step
    `STEP_RESULT_UNUSABLE`.
    """


def _is_transient(exc: Exception) -> bool:
    """Whether a failed statistics fetch is one a later try can get past."""
    if isinstance(exc, ArtifactRefusalError):
        return exc.code in TRANSIENT_CODES
    if isinstance(exc, httpx2.HTTPStatusError):
        status = exc.response.status_code
        return status == httpx2.codes.TOO_MANY_REQUESTS or status >= 500
    return isinstance(exc, (TimeoutError, httpx2.TransportError))


# What `execute_approved_plan` hands each progress report to, with the id of
# the plan step it describes: a wave's steps run concurrently, so a report
# means nothing without its step. `_Run._call` binds the step id and passes
# the result to `ToolSurface.call_tool` as that call's `on_progress`, and
# reports its own retries through the same binding. It runs on the event
# loop, so it must return promptly.
type StepProgressCallback = Callable[[str, ProgressReport], None]


async def default_fetch_series_stats(manifest: Manifest) -> pa.Table | None:
    """The table `evidence_from_manifest` keeps as `RetrievalEvidence.series`.

    Fetched off `manifest`'s own statistics artifact through
    `data_mcp.series_statistics` -- the same package `get_data` (data-mcp)
    itself uses, live in this process since the deployed topology mounts
    data-mcp in-process with the harness today (`data_mcp.client_holder`'s
    own docstring). `execute_approved_plan`'s default `fetch_series_stats`;
    a caller whose tool surface allows no real upstream call passes its own.

    A manifest with neither `series_count` nor a statistics artifact (an
    old record) needs no fetch and answers `None`, matching a manifest that
    never carried statistics at all. One that carries the artifact but no
    `series_count` (apis mints an all-years definition that way) is
    fetched with no row count to check. `series_count == 0` answers an empty
    table directly, with no fetch either -- `_series_issues` treats `None`
    and no rows alike as
    `STATISTICS_UNAVAILABLE`, so there is nothing a fetch could add for
    a mint that resolved to no series. Otherwise this fetches every row,
    with no cap: unlike the MCP digest's own `_STATISTICS_SERIES_CAP`,
    which only bounds what a model reads, `_series_issues` is a bulk
    coverage check that reads every series a retrieval carries, as Arrow
    columns rather than one model a series. The fetch runs under its own
    deadline -- `current_step_deadline`'s remaining seconds, capped at
    `EVIDENCE_ALLOWANCE_S` and reduced by `_EVIDENCE_DEADLINE_MARGIN_S` -- so
    it gives up before the step's outer budget does and its failure is
    reported as its own.

    Raises:
        StatisticsUnreachableError: the fetch timed out or failed in transit
            (`_is_transient`).
        StatisticsUnusableError: the manifest names series but no statistics
            artifact, or the artifact's link, download or contents failed
            for any other reason.
    """
    if manifest.series_count == 0:
        return SERIES_STATISTICS_ARROW_SCHEMA.empty_table()
    if manifest.definition is None:
        return None
    artifact = find_statistics_artifact(manifest)
    if artifact is None:
        if manifest.series_count is None:
            return None
        raise StatisticsUnusableError(
            f"the manifest for {manifest.definition.spec_id}/{manifest.content_version} names "
            f"series_count={manifest.series_count} but carries no statistics artifact"
        )
    step_deadline = current_step_deadline.get()
    remaining_s = step_deadline.remaining() if step_deadline is not None else EVIDENCE_ALLOWANCE_S
    fetch_deadline = CallDeadline.after(
        max(0.0, min(remaining_s, EVIDENCE_ALLOWANCE_S) - _EVIDENCE_DEADLINE_MARGIN_S)
    )
    try:
        return await fetch_series_statistics_table(
            subject_id=manifest.definition.spec_id,
            content_version=manifest.content_version,
            artifact=artifact,
            expected_series=manifest.series_count,
            continuing=continuing_method(manifest),
            deadline=fetch_deadline,
        )
    except (
        ResourceError,
        SeriesStatisticsSchemaError,
        httpx2.HTTPError,
        pa.ArrowException,
        ValidationError,
        KeyError,
        TimeoutError,
    ) as exc:
        # The detail leaves `exc` out: an `httpx2` error names the artifact link it failed on.
        logger.warning(
            "executor: could not fetch or decode the statistics artifact for %s/%s: %s",
            manifest.definition.spec_id,
            manifest.content_version,
            exc,
        )
        detail = (
            f"the statistics artifact for {manifest.definition.spec_id}/"
            f"{manifest.content_version} could not be fetched or read"
        )
        if _is_transient(exc):
            raise StatisticsUnreachableError(detail) from exc
        raise StatisticsUnusableError(detail) from exc


async def default_label_refs(refs: Sequence[str]) -> Mapping[str, str]:
    """The names the graph gives `refs`, through `data_mcp.node_lookup.describe_nodes`:
    `execute_approved_plan`'s default `label_refs`."""
    nodes = await describe_nodes(refs)
    return {ref: label for ref, node in nodes.items() if (label := getattr(node, "label", None))}


async def no_label_refs(refs: Sequence[str]) -> Mapping[str, str]:
    """The `LabelRefs` that names nothing, for a read no answer is written from
    (`read_anchors`) and a caller that makes no upstream call."""
    del refs
    return {}


async def _series_labels(label_refs: LabelRefs, refs: Sequence[str]) -> Mapping[str, str]:
    """The names `label_refs` gives `refs`, or none where it fails or outruns its deadline.

    A name labels a series in an answer's notes and nothing else, so the
    step goes on without it, and its series are counted. The deadline is
    what remains of the step's budget (`current_step_deadline`), less
    `_EVIDENCE_DEADLINE_MARGIN_S`, capped at `LABEL_ALLOWANCE_S`, so the
    lookup gives up before the step's own budget does.
    """
    allowance_s = LABEL_ALLOWANCE_S
    if (step_deadline := current_step_deadline.get()) is not None:
        allowance_s = min(step_deadline.remaining() - _EVIDENCE_DEADLINE_MARGIN_S, allowance_s)
    if allowance_s <= 0:
        return {}
    # Whatever an injected seam raises, only the names go unread, so the catch is broad.
    try:
        async with asyncio.timeout(allowance_s):
            return await label_refs(refs)
    except Exception as exc:
        logger.warning("executor: could not read names for %d entity refs: %r", len(refs), exc)
        return {}


class _Stopped(Exception):  # noqa: N818 -- a control-flow signal carrying an outcome, not an error
    """Carries the outcome that ends a run out of the step loop.

    `siblings` holds the other stops the same wave produced, wave order,
    when `outcome` is the wave's first and so the one a run raises
    (`_Run.execute`); empty for every stop that is not a wave's first.
    """

    def __init__(
        self,
        outcome: MethodChoiceNeeded | Refused | RediscoveryNeeded,
        *,
        siblings: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = (),
    ) -> None:
        super().__init__(outcome.kind)
        self.outcome = outcome
        self.siblings = siblings


@dataclass(frozen=True, slots=True)
class _Anchored:
    """The newest period of one group of rolling steps, as the run resolved it.

    `columns` holds the columns of each retrieved frame of the group, by the
    `spec_id` its retrieval was approved for, which the population of each of
    the group's steps is spelled against.
    """

    result: AnchorResult
    columns: dict[str, list[FrameColumns | None]]


def _waves(workflow: Workflow) -> list[tuple[WorkflowStep, ...]]:
    """Group `workflow`'s steps into the dependency levels their bindings state.

    Level 0 holds every step with no frame binding; level `n` holds every
    step whose bindings all name a step in a level below `n`. Steps that
    share a level carry no dependency between them and are safe to run
    concurrently; within a level, steps keep the order `workflow.steps`
    gives them.

    A rolling step also runs after every retrieval of its group
    (`compiler.rolling.anchor_groups`), the ones it binds no frame from
    included, since the newest period its years are written from is taken over
    all of them.
    """
    steps_by_id = {step.id: step for step in workflow.steps}
    roots = {step_id: group.roots for group in anchor_groups(workflow) for step_id in group.steps}
    level_of: dict[str, int] = {}

    def level(step_id: str) -> int:
        if step_id in level_of:
            return level_of[step_id]
        step = steps_by_id.get(step_id)
        assert step is not None, (
            f"a binding names step {step_id!r}, which the workflow does not hold"
        )
        after = [binding.from_step for binding in step.bindings.values()]
        after.extend(roots.get(step_id, ()))
        computed = 1 + max(level(earlier) for earlier in after) if after else 0
        level_of[step_id] = computed
        return computed

    waves: dict[int, list[WorkflowStep]] = {}
    for step in workflow.steps:
        waves.setdefault(level(step.id), []).append(step)
    return [tuple(waves[index]) for index in sorted(waves)]


# How long a user's approval of a plan revision authorises the run it starts
# to dispatch work. A run admitted under it may finish publication later.
PLAN_APPROVAL_TTL = timedelta(hours=24)


class Submission(StrictModel):
    """What a run executes, stored with it at admission so a run a restart
    left unfinished can be carried on from it.

    `revision` names the work in every outcome: a plan's revision, or, for a
    stored run admitted under a `RequestAuthorisation`, its workflow's digest.
    `requirements` are the dataset requirements the workflow's retrievals are
    checked against as evidence.
    """

    revision: str
    workflow: Workflow
    requirements: tuple[DatasetRequirement, ...]

    def source_policy(self) -> SourcePolicy:
        """The policy naming every source the workflow retrieves, at its latest version."""
        return SourcePolicy(sources=tuple(source.spec_id for source in self.workflow.sources))


@dataclass(frozen=True, kw_only=True)
class _Runtime:
    """What every run of one call shares: where it dispatches, how it waits,
    and where it publishes."""

    surface: ToolSurface
    capabilities: Sequence[ToolCapability]
    sleep: Sleep
    jitter: Jitter
    max_concurrent_steps: int
    fetch_series_stats: FetchSeriesStats
    label_refs: LabelRefs
    on_progress: StepProgressCallback | None
    publisher: ResultPublisher | None


async def execute_authorised(
    submission: Submission,
    *,
    authorisation: PlanApproval,
    idempotency_key: str,
    ledger: RunLedger,
    surface: ToolSurface,
    publisher: ResultPublisher | None,
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    label_refs: LabelRefs = default_label_refs,
    on_progress: StepProgressCallback | None = None,
) -> RunOutcome:
    """Run `submission` under `authorisation`, a plan's approval, as one run of `ledger`.

    The run is admitted before anything is dispatched, under `idempotency_key`
    and the digest of `submission` and the approval's source policy
    (`executor.ledger`). A key that already names a run of the same submission
    carries that run on: one whose computation finished returns the outcome it
    recorded and only resumes publication, and one that was still computing
    runs its workflow again, the tools answering what the store already
    holds. A key whose run failed admits a new run of the work. Every tool call
    is recorded as an attempt before it goes out, and the retrieved inputs are
    sealed, and checked against the authorisation's sources, before the first
    derived step. An accepted run then publishes each derived terminal frame
    with its replay recipe through `publisher`
    (`executor.completion.publish_results`); a stopped one is recorded failed.

    `sleep`, `jitter`, `max_concurrent_steps`, `fetch_series_stats`,
    `label_refs` and `on_progress` are `execute_approved_plan`'s own. With no
    `publisher`, a run with a derived result stays `publishing`.

    `authorisation` must approve the work `submission` is: its approved
    revision is `submission.revision`. One that names other work admits no run.

    Returns:
        The run's outcome, carrying its `run_id`, or a `Refused` with none:
        `AUTHORISATION_MISMATCH` when `authorisation` names other work, and
        `AUTHORISATION_EXPIRED` when a new run's authorisation has expired.
        The run's standing (`completed`, `pending`, `failed`) is the ledger's:
        `ledger.get(outcome.run_id)`.

    Raises:
        IdempotencyConflictError: `idempotency_key` names a run of other work.
        LedgerUnavailableError: the ledger's store could not admit the run, and
            nothing was run (`run_id` `None`), or could not record its progress,
            and the run is left unfinished for a later turn to carry on
            (`harness.graph.resume`).
        RunAbandonedError: a defect stopped the run's computation; the run is
            recorded failed.
    """
    runtime = _Runtime(
        surface=surface,
        capabilities=capabilities,
        sleep=sleep,
        jitter=jitter,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
        label_refs=label_refs,
        on_progress=on_progress,
        publisher=publisher,
    )
    if (authorised := authorisation.approved_revision) != submission.revision:
        return Refused(
            revision=submission.revision,
            refusal=RefusalKind.AUTHORISATION_MISMATCH,
            detail=f"the authorisation names work {authorised}, not {submission.revision}",
        )
    candidate = new_run(
        idempotency_key=idempotency_key,
        submission_digest=submission_digest(
            submission=submission, source_policy=authorisation.source_policy
        ),
        authorisation=authorisation,
        now=datetime.now(UTC),
    )
    try:
        async with store_faults(None):
            admission = await ledger.admit(candidate, submission=submission_content(submission))
    except AuthorisationExpiredError as exc:
        return Refused(
            revision=submission.revision, refusal=RefusalKind.AUTHORISATION_EXPIRED, detail=str(exc)
        )
    return await _carry_out(
        admission.run, submission, ledger=ledger, runtime=runtime, resumed=admission.resumed
    )


class UnreadableRunError(Exception):
    """A run's stored submission is missing or is not one this release reads."""


class RunAbandonedError(Exception):
    """A defect stopped the computation of admitted run `run_id`, which is
    recorded failed (`internal_error`). The defect is the `__cause__`."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} was recorded failed when a defect stopped it")
        self.run_id = run_id


async def load_submission(ledger: RunLedger, run_id: str) -> Submission:
    """The submission stored with run `run_id` of `ledger`.

    Raises:
        UnreadableRunError: `ledger` holds none for the run, or one this
            release cannot read.
        LedgerUnavailableError: the ledger's store could not be read.
    """
    async with store_faults(run_id):
        raw = await ledger.submission(run_id)
    if raw is None:
        raise UnreadableRunError(f"run {run_id} has no stored submission to carry on from")
    try:
        return Submission.model_validate(raw)
    except ValidationError as exc:
        raise UnreadableRunError(f"run {run_id}'s stored submission is unreadable: {exc}") from exc


async def resume_run(
    run: RunRecord,
    submission: Submission,
    *,
    ledger: RunLedger,
    surface: ToolSurface,
    publisher: ResultPublisher | None,
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    label_refs: LabelRefs = default_label_refs,
) -> RunOutcome:
    """Carry an unfinished `run` of `ledger` on from `submission`, the one
    stored with it (`load_submission`), as `execute_authorised` carries on a
    run its key already names. Raises as `execute_authorised` does."""
    runtime = _Runtime(
        surface=surface,
        capabilities=capabilities,
        sleep=sleep,
        jitter=jitter,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
        label_refs=label_refs,
        on_progress=None,
        publisher=publisher,
    )
    return await _carry_out(run, submission, ledger=ledger, runtime=runtime, resumed=True)


# The runs this process is carrying out, by run id, each to the outcome it
# will come to. A second submission of a run's work in flight here, as two
# identical tool calls of one model response are, awaits that outcome.
_IN_FLIGHT: dict[str, asyncio.Future[RunOutcome]] = {}


async def _carry_out(
    run: RunRecord,
    submission: Submission,
    *,
    ledger: RunLedger,
    runtime: _Runtime,
    resumed: bool,
) -> RunOutcome:
    """Take `run` to its outcome, or, when this process is already carrying
    it out, wait for that outcome without driving the run a second time. The
    driver starts from the ledger's current record, since `run` may be a
    snapshot taken before an earlier driver finished. `resumed` says `run`
    was admitted before this call, so driving it carries it on (`_drive`).

    A waiter cancelled leaves the run going on for the caller driving it. A
    driver cancelled fails a computation it was running (`_abandoned`),
    whether or not another caller waits on it, and each waiter then takes
    the run from the ledger as a driver would: a run failed with no outcome
    answers `RUN_ABANDONED`, and one the driver left unfinished, cancelled
    before or after its computation, is carried on by the waiter."""
    while (in_flight := _IN_FLIGHT.get(run.run_id)) is not None:
        await asyncio.wait((in_flight,))
        if not in_flight.cancelled():
            return in_flight.result()
    outcome: asyncio.Future[RunOutcome] = asyncio.get_running_loop().create_future()
    _IN_FLIGHT[run.run_id] = outcome
    try:
        async with store_faults(run.run_id):
            current = await ledger.get(run.run_id)
        if current is None:
            raise RunNotFoundError(run.run_id)
        result = await _drive(current, submission, ledger=ledger, runtime=runtime, resumed=resumed)
    except asyncio.CancelledError:
        outcome.cancel()
        raise
    except BaseException as exc:
        outcome.set_exception(exc)
        outcome.exception()  # the driving caller raises it; a waiter need not exist
        raise
    else:
        outcome.set_result(result)
        return result
    finally:
        del _IN_FLIGHT[run.run_id]


async def _drive(
    run: RunRecord,
    submission: Submission,
    *,
    ledger: RunLedger,
    runtime: _Runtime,
    resumed: bool,
) -> RunOutcome:
    """Take `run` from where the ledger holds it to its outcome.

    An unfinished run admitted before (`resumed`) is first recorded carried
    on (`Resume`), which fails one resumed too often or too old. A run that
    already came to an outcome returns it, finishing its publication, and
    one failed before it came to any answers `RUN_ABANDONED`.

    Raises:
        LedgerUnavailableError: the ledger's store could not record the run's
            progress; the run is left unfinished, never failed for it.
        RunAbandonedError: a defect stopped the computation; the run is
            recorded failed (`_abandoned`).
    """
    if resumed and run.status not in TERMINAL_STATUSES:
        run = await record_change(ledger, run.run_id, Resume())
    if run.outcome is not None:
        if run.status in (RunStatus.PUBLISHING, RunStatus.PENDING):
            await publish_results(
                run,
                ledger=ledger,
                publisher=runtime.publisher,
                sleep=runtime.sleep,
                jitter=runtime.jitter,
            )
        return _recorded_outcome(run.outcome, workflow=submission.workflow)
    if run.status is RunStatus.FAILED:
        assert run.failure is not None, "a failed run records its failure"
        return Refused(
            revision=submission.revision,
            run_id=run.run_id,
            refusal=RefusalKind.RUN_ABANDONED,
            detail=f"the run failed before it came to an outcome: {run.failure.reason}",
        )
    if run.status is RunStatus.ADMITTED:
        try:
            run = await record_change(ledger, run.run_id, Start())
        except AuthorisationExpiredError as exc:
            return await _failed(
                Refused(
                    revision=submission.revision,
                    run_id=run.run_id,
                    refusal=RefusalKind.AUTHORISATION_EXPIRED,
                    detail=str(exc),
                ),
                ledger=ledger,
            )
    try:
        outcome = await _run_workflow(submission, run_id=run.run_id, ledger=ledger, runtime=runtime)
    except LedgerUnavailableError:
        raise
    except BaseException as exc:
        await _abandoned(run.run_id, exc, ledger=ledger)
        if isinstance(exc, Exception):
            raise RunAbandonedError(run.run_id) from exc
        raise
    if not isinstance(outcome, Accepted):
        return await _failed(outcome, ledger=ledger)
    results = tuple(
        ResultFrame(spec_id=frame.spec_id, content_version=frame.content_version)
        for frame in outcome.results.values()
    )
    run = await record_change(
        ledger, run.run_id, Publishing(results, outcome=outcome.model_dump(mode="json"))
    )
    await publish_results(
        run,
        ledger=ledger,
        publisher=runtime.publisher,
        sleep=runtime.sleep,
        jitter=runtime.jitter,
    )
    return outcome


def _recorded_outcome(raw: JsonValue, *, workflow: Workflow) -> RunOutcome:
    """The outcome a run record holds, its frames by terminal step in
    `workflow`'s terminal order, the order the run produced them in. A store
    may hand a JSON object back in another key order (Postgres `jsonb` does)."""

    def ordered(frames: Mapping[str, FrameRef]) -> dict[str, FrameRef]:
        return {step: frames[step] for step in workflow.terminal_steps if step in frames}

    outcome = outcome_from_json(raw)
    if isinstance(outcome, Accepted):
        return outcome.model_copy(update={"results": ordered(outcome.results)})
    also_stopped = tuple(
        stop.model_copy(update={"answers": ordered(stop.answers)}) for stop in outcome.also_stopped
    )
    return outcome.model_copy(
        update={"answers": ordered(outcome.answers), "also_stopped": also_stopped}
    )


async def _abandoned(run_id: str, exc: BaseException, *, ledger: RunLedger) -> None:
    """Record run `run_id` failed by `exc`, which ends its computation with no
    outcome: the turn running it was cancelled, or a defect raised. Only a
    process that died mid-run, or a ledger that could not record the run's
    progress (`LedgerUnavailableError`, which `_drive` never passes here),
    leaves a run unfinished for the next turn to carry on; a failed run is
    never carried on, and submitting its work again admits a new run. A
    write that fails here is logged, so `exc` still propagates."""
    cancelled = isinstance(exc, asyncio.CancelledError)
    code = ExecutorCode.CANCELLED if cancelled else ExecutorCode.INTERNAL_ERROR
    reason = "the turn running it was cancelled" if cancelled else f"{type(exc).__name__}: {exc}"
    try:
        await ledger.record(run_id, Fail(code, reason))
    except Exception:
        logger.exception("executor: could not record run %s as %s", run_id, code)


async def _failed(
    outcome: MethodChoiceNeeded | Refused | RediscoveryNeeded, *, ledger: RunLedger
) -> RunOutcome:
    """Record `outcome`'s run failed under the outcome's own code, and return it."""
    assert outcome.run_id is not None
    if isinstance(outcome, Refused):
        code, reason = outcome.refusal.value, outcome.detail
    else:
        code, reason = outcome.kind, outcome.refusal.message or outcome.refusal.code
    await record_change(
        ledger, outcome.run_id, Fail(code, reason, outcome=outcome.model_dump(mode="json"))
    )
    return outcome


async def execute_approved_plan(
    plan: Plan,
    *,
    surface: ToolSurface,
    ledger: RunLedger | None = None,
    publisher: ResultPublisher | None = None,
    approver: str = "unattributed",
    conversation_id: str = "",
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    label_refs: LabelRefs = default_label_refs,
    on_progress: StepProgressCallback | None = None,
) -> tuple[Plan, RunOutcome]:
    """Run an approved plan through `execute_authorised` and record the run in its status.

    The only public way to run a plan: a plan that is not `APPROVED` is
    refused (`NOT_APPROVED`) without being touched, and one whose content no
    longer hashes to `approved_revision` is refused (`STALE_REVISION`) by
    `Plan.start_execution`. Its authorisation is
    a `PlanApproval` of the approved revision by `approver`, over every source
    the workflow retrieves, in force for `PLAN_APPROVAL_TTL` from the plan's
    `approved_at`, so admitting a new run of it, a retry included, needs an
    approval still in force. A plan with no `approved_at` has no approval time
    to measure from and is refused (`AUTHORISATION_EXPIRED`) with nothing
    run, as is one whose approval has expired when a new run is admitted. The run's
    idempotency key names the approved revision (which covers the plan's
    version) for `approver` in `conversation_id`, so executing one revision
    again, in any turn of the conversation, carries on its run. `ledger` keeps the run, a
    fresh `InMemoryRunLedger` by default.

    The plan's status follows its run's: `COMPLETED` once the run completed,
    `EXECUTING` while an accepted run's publication is unfinished (publishing
    or pending), and `FAILED` when the run stopped or failed.

    `sleep` waits between retries and `jitter` draws each wait
    (`executor.classify.jittered_retry_delay`); both injectable so tests
    neither wait nor draw at random. `max_concurrent_steps` bounds how many
    of a wave's steps may have their tool call in flight at once (`_Run`'s
    semaphore). `fetch_series_stats`
    resolves a dataset step's manifest into the rows `evidence_from_manifest`
    checks coverage against; the real fetch by default, injectable for a
    caller whose `surface` is itself fully virtualised, which raises
    `StatisticsUnreachableError` or `StatisticsUnusableError` the way the real
    one does. `label_refs` names the entities of the series a retrieval holds
    short of its period that the approved source names none for, so an
    answer can name the largest (`default_label_refs` by default).
    `on_progress` hears every progress report a step's tool call sends, and
    a report naming the next attempt each time a step is retried, tagged
    with the step's id.

    Returns:
        The plan as the run left it, or unchanged when it could not start,
        and the outcome.

    Raises:
        LedgerUnavailableError: the ledger's store could not record or read
            the run (`execute_authorised`).
    """
    if plan.status is not PlanStatus.APPROVED:
        return plan, Refused(
            revision=plan_revision(plan),
            refusal=RefusalKind.NOT_APPROVED,
            detail=f"this plan is {plan.status.value}, not approved; nothing runs",
        )
    try:
        running = plan.start_execution()
    except ValueError as exc:
        # Approved as different content: nothing runs.
        return plan, Refused(
            revision=plan_revision(plan), refusal=RefusalKind.STALE_REVISION, detail=str(exc)
        )
    assert running.workflow is not None, "an EXECUTING plan always carries a workflow"
    assert running.approved_revision is not None, "an EXECUTING plan was approved"
    if running.approved_at is None:
        return running.fail(), Refused(
            revision=plan_revision(running),
            refusal=RefusalKind.AUTHORISATION_EXPIRED,
            detail="this plan carries no approval time, so it needs approving again",
        )
    # The plan validated its own workflow and requirements.
    submission = Submission.model_construct(
        revision=plan_revision(running),
        workflow=running.workflow,
        requirements=tuple(
            r for r in running.evidence_requirements if isinstance(r, DatasetRequirement)
        ),
    )
    ledger = ledger if ledger is not None else InMemoryRunLedger()
    approval = PlanApproval(
        source_policy=submission.source_policy(),
        expires_at=running.approved_at + PLAN_APPROVAL_TTL,
        plan_version=running.version,
        approved_revision=running.approved_revision,
        approver=approver,
        at=running.approved_at,
    )
    outcome = await execute_authorised(
        submission,
        authorisation=approval,
        idempotency_key=idempotency_key(
            caller=approver,
            conversation_id=conversation_id,
            task=f"plan:{running.approved_revision}",
        ),
        ledger=ledger,
        surface=surface,
        publisher=publisher,
        capabilities=capabilities,
        sleep=sleep,
        jitter=jitter,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
        label_refs=label_refs,
        on_progress=on_progress,
    )
    run = None
    if outcome.run_id is not None:
        async with store_faults(outcome.run_id):
            run = await ledger.get(outcome.run_id)
    if run is None or run.status is RunStatus.FAILED or not isinstance(outcome, Accepted):
        return running.fail(), outcome
    if run.status is RunStatus.COMPLETED:
        return running.complete(), outcome
    return running, outcome


async def _run_workflow(
    submission: Submission, *, run_id: str, ledger: RunLedger, runtime: _Runtime
) -> RunOutcome:
    """Run `submission`'s workflow as run `run_id` and return what came of it.

    Private: reachable only through `_carry_out`, once the run is admitted
    and `running`.

    Returns:
        `Accepted` when every step ran and every requirement is met,
        `MethodChoiceNeeded` when a tool refused with a choice for the user,
        `RediscoveryNeeded` when a tool said only a refresh could fix its
        refusal, `Refused` otherwise. A refusal's own `also_stopped` names
        the rest of its wave's stops, wave order, when there were any. Each
        carries `run_id`.
    """
    run = _Run(submission, run_id=run_id, ledger=ledger, runtime=runtime)
    try:
        return await run.execute()
    except _Stopped as stopped:
        return _stop_outcome(stopped)


def _stop_outcome(stopped: _Stopped) -> MethodChoiceNeeded | Refused | RediscoveryNeeded:
    """The outcome `stopped` carries, its own `also_stopped` naming the rest of its wave's stops."""
    if not stopped.siblings:
        return stopped.outcome
    return stopped.outcome.model_copy(update={"also_stopped": stopped.siblings})


class LeftOutEntity(StrictModel):
    """One entity a retrieval left out, by the role and ref its approved source
    resolved it to, with its finding (`compiler.evidence.left_out_entities`)."""

    role: str
    ref: str
    issue: EvidenceIssue


class HeldAnchor(StrictModel):
    """What a read of one group of rolling steps found, kept so a later read of the
    same group can stand on it without running its retrievals again.

    `key` is `_Run._group_key`: the group's population, and each retrieval's
    tool, arguments, approved source and requirement, so it names the same
    calls and checks whatever the steps are called. `result` is what
    `resolve_anchor` answered, and `left_out` the entities each retrieval left
    out, in the order of the group's `roots`, which the check of the steps
    narrowing to them reads (`_Run._check_left_out_narrowings`).
    """

    key: str
    result: AnchorResult
    left_out: tuple[tuple[LeftOutEntity, ...], ...]


@dataclass(frozen=True, slots=True)
class AnchorRead:
    """The newest period one group of a workflow's rolling steps stands on, read before a run.

    `steps` are the group's rolling steps (`compiler.rolling.AnchorGroup`),
    `planned` the newest period their literals state (`planned_anchor`) and
    `result` what `resolve_anchor` answered over the group's retrievals, or
    the stop its run would meet on one of them or on `resolve_anchor`.
    `held` is the read as a later `read_anchors` may be handed it, `None`
    when the group stopped.
    """

    steps: tuple[str, ...]
    planned: str
    result: AnchorResult | MethodChoiceNeeded | Refused | RediscoveryNeeded
    held: HeldAnchor | None = None

    @property
    def behind(self) -> bool:
        """Whether the newest period resolved is before the one the plan states."""
        return isinstance(self.result, AnchorResult) and self.result.period < self.planned


def planned_anchor(group: AnchorGroup, steps_by_id: Mapping[str, WorkflowStep]) -> str:
    """The newest period the rolling literals of `group` state.

    A rolling literal less its offset is the newest period the plan was
    checked against, one for the whole group
    (`planner.plan._check_rolling_years_agree`). Periods of one grain order
    by their spelling, so the largest stands for the group.
    """
    return max(
        shift_period(bound.period, -bound.offset)
        for step_id in group.steps
        for bound in rolling_bounds(steps_by_id[step_id].args)
    )


async def read_anchors(
    workflow: Workflow,
    *,
    revision: str,
    requirements: Sequence[DatasetRequirement],
    surface: ToolSurface,
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    schemas: Mapping[str, ToolSchema] | None = None,
    held: Iterable[HeldAnchor] = (),
) -> tuple[AnchorRead, ...] | MethodChoiceNeeded | Refused | RediscoveryNeeded:
    """Resolve the newest period of each group of `workflow`'s rolling steps, as its run would.

    Takes a run's own path (`_Run.read_anchors`): the surface check, the
    retrievals the groups rest on, checked against `requirements` as
    evidence, and `resolve_anchor` for each group, over the same
    populations and frames. Nothing is admitted to a ledger and nothing is
    recorded, and no derived step runs. `revision` names the work in a stop's
    outcome. The other arguments but the last two are
    `execute_approved_plan`'s. No answer is written from the read, so it asks
    no names for its series (`no_label_refs`).

    `schemas` are the tools `workflow` was just compiled against, by name;
    the surface check then reads them in place of listing the surface again.
    A group whose key (`HeldAnchor.key`) one of `held` carries reads as that
    one, and its retrievals run only when another group rests on them too.

    Returns:
        Each group's `AnchorRead`, in the order its run resolves them, empty
        for a workflow with no rolling step, each holding the stop a run
        would meet on the group's retrievals or its resolution, with no
        `run_id`; or the outcome a run of `workflow` stops on before any
        retrieval, on a retrieval whose stop names no step, or on a step
        reading only entities the retrievals left out.
    """
    runtime = _Runtime(
        surface=surface,
        capabilities=capabilities,
        sleep=sleep,
        jitter=jitter,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
        label_refs=no_label_refs,
        on_progress=None,
        publisher=None,
    )
    submission = Submission.model_construct(
        revision=revision, workflow=workflow, requirements=tuple(requirements)
    )
    run = _Run(submission, run_id=None, ledger=None, runtime=runtime)
    try:
        return await run.read_anchors(schemas=schemas, held={anchor.key: anchor for anchor in held})
    except _Stopped as stopped:
        return _stop_outcome(stopped)


class _Run:
    """One run's working state.

    A read made before any run is admitted (`read_anchors`) has no `run_id`
    and no `ledger`, and records nothing.
    """

    def __init__(
        self,
        submission: Submission,
        *,
        run_id: str | None,
        ledger: RunLedger | None,
        runtime: _Runtime,
    ) -> None:
        workflow = submission.workflow
        self.workflow = workflow
        self.run_id = run_id
        self.ledger = ledger
        self.surface = runtime.surface
        self.sleep = runtime.sleep
        self.jitter = runtime.jitter
        self.fetch_series_stats = runtime.fetch_series_stats
        self.label_refs = runtime.label_refs
        self.on_progress = runtime.on_progress
        # Bounds how many steps of one wave may have their tool call in
        # flight at once, acquired in `_attempt` around `_run_step`; a wave
        # wider than the bound queues at the point of call rather than
        # opening every call in the wave at once.
        self.concurrency = asyncio.Semaphore(runtime.max_concurrent_steps)
        self.revision = submission.revision
        self.capabilities = {c.tool: c for c in runtime.capabilities}
        self.requirements = {r.id: r for r in submission.requirements}
        self.sources = {s.spec_id: s for s in workflow.sources}
        self.evidence_bindings: dict[str, EvidenceBinding] = {
            b.step_id: b for b in workflow.evidence_bindings
        }
        # The source each evidence-bearing step is checked against: the one
        # the step was compiled from (`Workflow.source_of`), so two
        # retrievals of one `spec_id` each keep their own requirement's
        # reading of it. The `spec_id` the tool answers with
        # (`evidence.spec_id`) plays no part in the lookup.
        self.approved_source: dict[str, SelectedSource] = {
            b.step_id: workflow.source_of(b.step_id) or self.sources[b.spec_id]
            for b in workflow.evidence_bindings
        }
        self.steps_by_id: dict[str, WorkflowStep] = {s.id: s for s in workflow.steps}
        # The entities each retrieval left out, by the role and ref its
        # approved source resolved them to, with their findings
        # (`_check_left_out_narrowings`).
        self.left_out: dict[str, dict[tuple[str, str], EvidenceIssue]] = {}
        self.frames: dict[str, FrameRef] = {}
        self.row_counts: dict[str, int | None] = {}
        self.stat_var_counts: dict[str, int | None] = {}
        self.dropped_key_counts: dict[str, tuple[int, int]] = {}
        self.ranked_before_limit: dict[str, int | None] = {}
        # Per step that produced a frame: the role keys its result declared,
        # and the columns of the retrieval it rests on, which the plan's role
        # columns bind to (`compiler.frame_columns`). `None` where the result
        # declared none.
        self.role_keys: dict[str, tuple[str, ...] | None] = {}
        self.columns: dict[str, FrameColumns | None] = {}
        # Per-step slots, keyed by step id rather than appended to as steps
        # finish: a wave's steps finish in whatever order the event loop
        # schedules them, and `_completed_so_far`/`_evidence_so_far` read
        # these back out in the plan's own step order regardless.
        self.records: dict[str, StepRecord] = {}
        self.evidence_by_step: dict[str, EvidenceRecord] = {}
        # The groups of rolling steps that each stand on one newest period,
        # none when no step rolls; each rolling step's group; and each group's
        # period once the run has resolved it, under every step of the group.
        self.anchor_groups = anchor_groups(workflow)
        self.group_of: dict[str, AnchorGroup] = {
            step_id: group for group in self.anchor_groups for step_id in group.steps
        }
        self.anchored: dict[str, _Anchored] = {}
        self._resolutions: dict[str, _Anchored] = {}
        # The input each retrieval read and the `get_data` result it was
        # read from, its long StatVar list left out, by step id, and whether
        # the run has sealed them (`_seal`).
        self.inputs: dict[str, ManifestEntry] = {}
        self.retrievals: dict[str, JsonValue] = {}
        self.sealed = False

    async def execute(self) -> Accepted:
        schemas = await self._surface_as_compiled()
        for wave in _waves(self.workflow):
            # Every retrieval reads no frame, so it runs in the first wave;
            # the first wave holding a derived step seals what they read.
            if not self.sealed and not all(self._retrieves(s) for s in wave):
                await self._seal()
                self._check_left_out_narrowings()
            # Every retrieval of a group is in a wave before the group's first
            # step (`_waves`), and the wave's steps start only once these return.
            for group, first in self._groups_starting(wave):
                anchor = await self._resolve_group(group, first, schemas)
                try:
                    self._check_anchor_held(group, first, anchor)
                except _Stopped as stopped:
                    raise _Stopped(self._restamped(stopped.outcome)) from stopped
            await self._run_wave(wave, schemas)
        answers, refused = self._answers(self.workflow.terminal_steps)
        if refused is not None:
            raise _Stopped(self._restamped(refused.outcome))
        if not self.sealed:
            await self._seal()
        return Accepted(
            revision=self.revision,
            run_id=self.run_id,
            evidence=self._evidence_so_far(),
            steps=self._completed_so_far(),
            results=answers,
            anchors=self._anchors_so_far(),
        )

    async def read_anchors(
        self, *, schemas: Mapping[str, ToolSchema] | None, held: Mapping[str, HeldAnchor]
    ) -> tuple[AnchorRead, ...]:
        """Run the retrievals the anchor groups rest on, seal them, and resolve each group.

        The run's own steps for the part of it a group's newest period rests
        on (`execute`), with each group resolved under its first step to
        run, and no derived step run and no check of the period against the
        plan's (`_check_anchor_held`), which the caller makes. A retrieval
        that stops stops only the groups resting on it, which read as that
        stop, the first in step order. With no retrieval stopped, a step
        reading only entities they left out stops the whole read before any
        group is resolved, as it stops the run (`_check_left_out_narrowings`).
        A group whose resolution stops reads as that stop too. Every other
        group is resolved.

        A group `held` names by its key (`_group_key`) reads as the held
        result instead, renamed to the group's steps, and the entities its
        retrievals left out are taken from it; a retrieval runs only for a
        group `held` does not name. `schemas`, when given, stand in for the
        surface's own list (`_surface_as_compiled`).
        """
        schemas = await self._surface_as_compiled(schemas)
        keys = {group.steps: self._group_key(group) for group in self.anchor_groups}
        standing = {
            group.steps: held[keys[group.steps]]
            for group in self.anchor_groups
            if keys[group.steps] in held
        }
        roots = {
            root
            for group in self.anchor_groups
            if group.steps not in standing
            for root in group.roots
        }
        for group in self.anchor_groups:
            if (anchor := standing.get(group.steps)) is not None:
                for root, left_out in zip(group.roots, anchor.left_out, strict=True):
                    if root not in roots:
                        self.left_out[root] = {
                            (entity.role, entity.ref): entity.issue for entity in left_out
                        }
        stops: dict[str, MethodChoiceNeeded | Refused | RediscoveryNeeded] = {}
        try:
            await self._run_wave(tuple(s for s in self.workflow.steps if s.id in roots), schemas)
        except _Stopped as stopped:
            for outcome in (stopped.outcome, *stopped.siblings):
                if outcome.step_id is None:
                    raise
                stops.setdefault(outcome.step_id, outcome)
        await self._seal()
        # The run meets a retrieval's stop before this check, which it makes
        # once its retrievals return (`execute`).
        if not stops:
            self._check_left_out_narrowings()
        read: list[AnchorRead] = []
        for wave in _waves(self.workflow):
            for group, first in self._groups_starting(wave):
                if any(read_group.steps == group.steps for read_group in read):
                    continue
                planned = planned_anchor(group, self.steps_by_id)
                if (anchor := standing.get(group.steps)) is not None:
                    result = anchor.result.model_copy(
                        update={"steps": group.steps, "roots": group.roots}
                    )
                    read.append(AnchorRead(group.steps, planned, result, anchor))
                    continue
                stop = next((stops[step] for step in group.roots if step in stops), None)
                result = stop or await self._resolved_or_stop(group, first, schemas)
                kept = (
                    HeldAnchor(
                        key=keys[group.steps], result=result, left_out=self._left_out_of(group)
                    )
                    if isinstance(result, AnchorResult)
                    else None
                )
                read.append(AnchorRead(group.steps, planned, result, kept))
        return tuple(read)

    def _group_key(self, group: AnchorGroup) -> str:
        """What `group`'s read rests on, hashed: its population, and each retrieval's
        tool, arguments, approved source and requirement, in the group's order.
        Step ids play no part, so a group compiled again from the same plan keeps it."""
        roots: list[JsonValue] = []
        for root in group.roots:
            step = self.steps_by_id[root]
            requirement = self.requirements[self.evidence_bindings[root].requirement_id]
            roots.append(
                {
                    "tool": [step.tool.server, step.tool.name],
                    "args": step.args,
                    "source": self.approved_source[root].model_dump(mode="json"),
                    "requirement": requirement.model_dump(mode="json"),
                }
            )
        stated = json.dumps({"population": list(group.population), "roots": roots}, sort_keys=True)
        return hashlib.sha256(stated.encode()).hexdigest()

    def _left_out_of(self, group: AnchorGroup) -> tuple[tuple[LeftOutEntity, ...], ...]:
        """The entities each retrieval of `group` left out, in the group's order."""
        return tuple(
            tuple(
                LeftOutEntity(role=role, ref=ref, issue=issue)
                for (role, ref), issue in self.left_out.get(root, {}).items()
            )
            for root in group.roots
        )

    async def _resolved_or_stop(
        self, group: AnchorGroup, first: WorkflowStep, schemas: Mapping[str, ToolSchema]
    ) -> AnchorResult | MethodChoiceNeeded | Refused | RediscoveryNeeded:
        """`_resolve_group`'s period for `group`, or the stop it raised."""
        try:
            return await self._resolve_group(group, first, schemas)
        except _Stopped as stopped:
            return stopped.outcome

    def _groups_starting(
        self, wave: Sequence[WorkflowStep]
    ) -> list[tuple[AnchorGroup, WorkflowStep]]:
        """Each group not yet resolved whose first step to run is in `wave`, with that step."""
        starting: list[tuple[AnchorGroup, WorkflowStep]] = []
        for group in self.anchor_groups:
            first = next((step for step in wave if step.id in group.steps), None)
            if first is not None and first.id not in self.anchored:
                starting.append((group, first))
        return starting

    async def _resolve_group(
        self, group: AnchorGroup, first: WorkflowStep, schemas: Mapping[str, ToolSchema]
    ) -> AnchorResult:
        """`_resolve_anchor` for `group` under `first`, a stop carrying what the run finished."""
        try:
            return await self._resolve_anchor(group, first, schemas[_RESOLVE_ANCHOR.name])
        except _Stopped as stopped:
            raise _Stopped(self._restamped(stopped.outcome)) from stopped

    async def _run_wave(
        self, wave: Sequence[WorkflowStep], schemas: Mapping[str, ToolSchema]
    ) -> None:
        """Run every step of `wave` to completion, then raise the first stop among them."""
        # Each step runs as its own task. `asyncio.gather` copies the
        # calling context per task, so `current_step_id`, set inside
        # `_run_step`, stays with its step among concurrent siblings.
        # `return_exceptions=True` lets every step in the wave run to
        # completion when one of them raises something `_attempt` did not
        # catch. Such an exception is a defect, since `_attempt` returns
        # `_Stopped` and never raises it, and ending the wave at once would
        # abandon that wave's other in-flight steps unawaited. A
        # `BaseException` outside `Exception`, such as `CancelledError`,
        # is not captured by `return_exceptions` and propagates
        # regardless.
        outcomes = await asyncio.gather(
            *(self._attempt(step, schemas[step.tool.name]) for step in wave),
            return_exceptions=True,
        )
        stopped: _Stopped | None = None
        other_stops: list[MethodChoiceNeeded | Refused | RediscoveryNeeded] = []
        for outcome in outcomes:
            # `_attempt` returns `_Stopped`, an `Exception` subclass, and
            # never raises it, so it is read off first, before the general
            # `BaseException` check below that re-raises anything else.
            if isinstance(outcome, _Stopped):
                if stopped is None:
                    stopped = outcome
                else:
                    other_stops.append(outcome.outcome)
                continue
            if isinstance(outcome, BaseException):
                # The first such exception in the wave's own step order,
                # since `outcomes` keeps `asyncio.gather`'s input order
                # regardless of which task finished first.
                raise outcome
            record, evidence = outcome
            self.records[record.step_id] = record
            if evidence is not None:
                self.evidence_by_step[record.step_id] = evidence
        if stopped is not None:
            # The wave has already run every one of its steps to
            # completion, so `_completed_so_far` names every step that
            # finished in this wave and in every wave before it,
            # `_evidence_so_far` the evidence they recorded, and
            # `_answers` the finished terminal steps that stand as
            # answers. Re-stamp the outcome, and every sibling stop,
            # with those rather than whatever each saw at the moment it
            # stopped.
            raise _Stopped(
                self._restamped(stopped.outcome),
                siblings=tuple(self._restamped(sibling) for sibling in other_stops),
            )

    async def _record(self, change: RunChange, *, step_id: str | None = None) -> None:
        """Record `change` to the run, refusing the run when the ledger says
        the authorisation no longer covers it. A read with no run records
        nothing."""
        if self.ledger is None or self.run_id is None:
            return
        try:
            await record_change(self.ledger, self.run_id, change)
        except AuthorisationExpiredError as exc:
            raise self._refused(
                RefusalKind.AUTHORISATION_EXPIRED, str(exc), step_id=step_id
            ) from exc
        except ManifestOutsideAuthorisationError as exc:
            raise self._refused(RefusalKind.MANIFEST_OUTSIDE_AUTHORISATION, str(exc)) from exc

    def _retrieves(self, step: WorkflowStep) -> bool:
        """Whether `step`'s tool retrieves a root input, read off a manifest."""
        capability = self.capabilities.get(step.tool)
        return capability is not None and any(
            output.extract is Extract.FRAME_REF_FROM_MANIFEST for output in capability.outputs
        )

    async def _seal(self) -> None:
        """Seal the inputs the retrievals read, before any derived step runs."""
        await self._record(
            Seal(tuple(self.inputs.values()), retrievals=tuple(self.retrievals.values()))
        )
        self.sealed = True

    async def _attempt(
        self, step: WorkflowStep, schema: ToolSchema
    ) -> tuple[StepRecord, EvidenceRecord | None] | _Stopped:
        """Run one step, returning its own `_Stopped` rather than raising it.

        Lets `execute` gather a whole wave to completion before any one
        step's stop ends the run, so a wave's successes are recorded even
        when a sibling in the same wave refuses. Holds `self.concurrency`
        for the step's own run.
        """
        async with self.concurrency:
            try:
                return await self._run_step(step, schema)
            except _Stopped as stopped:
                return stopped

    def _completed_so_far(self) -> tuple[StepRecord, ...]:
        """Every step that has finished, in the plan's own step order.

        Read off `self.records` by walking `workflow.steps` rather than by
        when each step finished, so this reads the same whether the step
        that filled a slot ran alone or alongside others in its wave.
        """
        return tuple(
            self.records[step.id] for step in self.workflow.steps if step.id in self.records
        )

    def _evidence_so_far(self) -> tuple[EvidenceRecord, ...]:
        """Every requirement met so far, in the plan's own step order."""
        return tuple(
            self.evidence_by_step[step.id]
            for step in self.workflow.steps
            if step.id in self.evidence_by_step
        )

    def _anchors_so_far(self) -> tuple[AnchorResult, ...]:
        """Each newest period resolved so far, once, in the order it was resolved."""
        return tuple(anchored.result for anchored in self._resolutions.values())

    def _restamped(
        self, outcome: MethodChoiceNeeded | Refused | RediscoveryNeeded
    ) -> MethodChoiceNeeded | Refused | RediscoveryNeeded:
        """`outcome` carrying what the run finished: its steps, their evidence,
        the terminal frames that stand as answers (`_answers`) and the newest
        periods it resolved."""
        return outcome.model_copy(
            update={
                "completed": self._completed_so_far(),
                "evidence": self._evidence_so_far(),
                "anchors": self._anchors_so_far(),
                "answers": self._answers(
                    step_id for step_id in self.workflow.terminal_steps if step_id in self.records
                )[0],
            }
        )

    def _answers(self, step_ids: Iterable[str]) -> tuple[dict[str, FrameRef], _Stopped | None]:
        """The frame of each of `step_ids` (terminal steps) that stands as an
        answer (`_answer_refusal`), by step id in the order given, and the
        first refusal when one does not stand.
        """
        answers: dict[str, FrameRef] = {}
        refused: _Stopped | None = None
        for step_id in step_ids:
            refusal = self._answer_refusal(step_id)
            if refusal is None:
                answers[step_id] = self.frames[step_id]
            elif refused is None:
                refused = refusal
        return answers, refused

    def _answer_refusal(self, step_id: str) -> _Stopped | None:
        """Why a terminal step's frame does not stand as an answer: it
        produced none, its own declared row count is missing where its output
        schema promises one, or that count is zero from a tool whose empty
        result is not an answer. `None` when the frame stands.
        """
        if step_id not in self.frames:
            return self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"terminal step {step_id!r} produced no frame to answer with",
                step_id=step_id,
            )
        step = self.steps_by_id[step_id]
        row_count = self.row_counts.get(step_id)
        if step.output_schema_hash is not None and row_count is None:
            return self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"{step.tool.name!r} declares an output schema promising a row count, but "
                f"terminal step {step_id!r}'s result carried none",
                step_id=step_id,
            )
        if row_count == 0:
            capability = self.capabilities.get(step.tool)
            if capability is None or not capability.empty_is_an_answer:
                return self._refused(
                    RefusalKind.EMPTY_ANSWER,
                    f"terminal step {step_id!r} matched no rows, and {step.tool.name!r} is not "
                    "one whose empty result is itself an answer",
                    step_id=step_id,
                )
        return None

    # -- before the first call ------------------------------------------------------

    async def _surface_as_compiled(
        self, listed: Mapping[str, ToolSchema] | None = None
    ) -> dict[str, ToolSchema]:
        """Re-read the surface and refuse to run against tools that have changed.

        Each step's tool is compared with what the step was compiled against.
        A workflow with a rolling step also calls `resolve_anchor`, which no
        step names, so that tool is compared with its row of the capability
        table, under the first rolling step's id. `listed`, the surface a
        caller has just listed, by name, is compared in place of listing it
        again; a run lists it itself, since the surface may have moved since
        the plan was approved.
        """
        if listed is not None:
            schemas = dict(listed)
        else:
            try:
                schemas = {schema.name: schema for schema in await self.surface.list_tools()}
            except ToolSurfaceUnavailableError as exc:
                raise self._refused(RefusalKind.TRANSPORT_FAILED, str(exc)) from exc
        for step in self.workflow.steps:
            self._check_as_compiled(
                schemas.get(step.tool.name),
                tool=step.tool,
                input_schema_hash=step.input_schema_hash,
                output_schema_hash=step.output_schema_hash,
                implementation_version=step.implementation_version,
                step_id=step.id,
            )
        if self.anchor_groups:
            rolling = next(step for step in self.workflow.steps if is_rolling(step))
            capability = self.capabilities.get(_RESOLVE_ANCHOR)
            if capability is None:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"no capability pins {_RESOLVE_ANCHOR.name!r}, which a plan with a rolling "
                    "period calls",
                    step_id=rolling.id,
                )
            self._check_as_compiled(
                schemas.get(_RESOLVE_ANCHOR.name),
                tool=_RESOLVE_ANCHOR,
                input_schema_hash=capability.input_schema_hash,
                output_schema_hash=capability.output_schema_hash,
                implementation_version=capability.implementation_version,
                step_id=rolling.id,
            )
        return schemas

    def _check_as_compiled(
        self,
        schema: ToolSchema | None,
        *,
        tool: ToolId,
        input_schema_hash: str,
        output_schema_hash: str | None,
        implementation_version: str | None,
        step_id: str,
    ) -> None:
        """Refuse `SURFACE_CHANGED` unless `schema` is `tool` as it was pinned."""
        if schema is None or schema.server != tool.server:
            raise self._refused(
                RefusalKind.SURFACE_CHANGED,
                f"{tool.server} no longer serves {tool.name!r}",
                step_id=step_id,
            )
        try:
            input_hash = schema.input_schema_hash
            output_hash = schema.output_schema_hash
        except UnhashableSchemaError as exc:
            raise self._refused(
                RefusalKind.SURFACE_CHANGED,
                f"{tool.name!r} now declares a schema that cannot be compared with the "
                f"one the plan was compiled against: {exc}",
                step_id=step_id,
            ) from exc
        if input_hash != input_schema_hash:
            raise self._refused(
                RefusalKind.SURFACE_CHANGED,
                f"{tool.name!r} takes different arguments than the plan was compiled "
                "against; the plan needs compiling and approving again",
                step_id=step_id,
            )
        if output_hash != output_schema_hash:
            raise self._refused(
                RefusalKind.SURFACE_CHANGED,
                f"{tool.name!r} declares a different output schema than the plan was "
                "compiled against; the plan needs compiling and approving again",
                step_id=step_id,
            )
        if schema.version != implementation_version:
            raise self._refused(
                RefusalKind.SURFACE_CHANGED,
                f"{tool.name!r} is at version {schema.version!r}, not the "
                f"{implementation_version!r} the plan was compiled against",
                step_id=step_id,
            )

    # -- the newest period ----------------------------------------------------------

    async def _resolve_anchor(
        self, group: AnchorGroup, step: WorkflowStep, schema: ToolSchema
    ) -> AnchorResult:
        """Resolve the newest period the rolling steps of `group` stand on, once.

        Sends `resolve_anchor` the populations the group's steps carry, with
        every column spelled the way the retrieved frames spell it
        (`bound_populations`, the value `_arguments` writes on each of them),
        and the frame of each retrieval of the group. `step` is the group's
        first step about to run: the call reports and is refused under its id.

        Writes `self.anchored` for every step of the group, and returns the
        period resolved.
        """
        roots: dict[str, list[FrameColumns | None]] = {}
        for root in group.roots:
            roots.setdefault(self.approved_source[root].spec_id, []).append(self.columns.get(root))
        try:
            populations = bound_populations(group.population, roots=roots)
        except UnboundColumnError as exc:
            raise self._column_unbound(step, exc) from exc
        # Two retrievals that returned one frame name it once.
        frames = {
            (frame.spec_id, frame.content_version): frame.model_dump()
            for frame in (self.frames[root] for root in group.roots)
        }
        arguments: dict[str, JsonValue] = {
            "populations": populations,
            "roots": list(frames.values()),
        }
        # Groups over the same series and frames, such as two rolling steps
        # each over the one retrieval, share one resolution.
        sent = json.dumps({"populations": populations, "roots": sorted(frames)}, sort_keys=True)
        if (held := self._resolutions.get(sent)) is not None:
            steps = (*held.result.steps, *group.steps)
            shared = held.result.roots
            shared = (*shared, *(root for root in group.roots if root not in shared))
            held = replace(
                held, result=held.result.model_copy(update={"steps": steps, "roots": shared})
            )
            self._resolutions[sent] = held
            self.anchored.update(dict.fromkeys(group.steps, held))
            return held.result
        self._check_fit(step, _RESOLVE_ANCHOR, arguments, schema)
        token = current_step_id.set(step.id)
        try:
            result, _ = await self._call(step.id, _RESOLVE_ANCHOR, arguments)
        finally:
            current_step_id.reset(token)
        if result.is_error:
            raise self._stopped_by_anchor_refusal(step, result)
        try:
            anchor = AnchorResult.model_validate(result.structured_content).model_copy(
                update={
                    "steps": group.steps,
                    "roots": group.roots,
                    "populations": tuple(populations),
                }
            )
        except ValidationError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"{_RESOLVE_ANCHOR.name!r} answered with no newest period the run can read: {exc}",
                step_id=step.id,
            ) from exc
        anchored = _Anchored(result=anchor, columns=roots)
        self._resolutions[sent] = anchored
        self.anchored.update(dict.fromkeys(group.steps, anchored))
        return anchor

    def _check_anchor_held(
        self, group: AnchorGroup, step: WorkflowStep, anchor: AnchorResult
    ) -> None:
        """Refuse `ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN` for a newest period before the plan's.

        The plan's newest period is the one its literals state
        (`planned_anchor`). Every rule the plan met holds of its years moved later, so a period
        on it or after it runs. An earlier one reads years no rule was checked
        against, which can leave a step no rows or the wrong years.

        The refusal's payload is worded for the user and classifies as a
        stop, since the same plan resolves the same period over the same data.
        """
        planned = planned_anchor(group, self.steps_by_id)
        if not AnchorRead(steps=group.steps, planned=planned, result=anchor).behind:
            return
        raise self._refused(
            RefusalKind.ROLLING_ANCHOR_BEHIND_PLAN,
            f"step {step.id!r} counts from {anchor.period}, the newest period resolved, "
            f"before {planned}, the one the plan states",
            step_id=step.id,
            payload=ToolFailure(
                code=ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN,
                message=(
                    f"The plan counted the period from {planned}, but fewer than half of these "
                    f"series now report {planned}. The latest year at least half of them "
                    f"report is {anchor.period}."
                ),
                retry_mode="do_not_retry",
                fix={
                    "remedy": f"Ask again to plan the period from {anchor.period}.",
                    "anchor": anchor.period,
                    "planned_anchor": planned,
                },
            ),
        )

    def _stopped_by_anchor_refusal(self, step: WorkflowStep, result: ToolCallResult) -> _Stopped:
        """The stop for a `resolve_anchor` call the tool refused.

        A refusal that states a fact of the data keeps its code and carries a
        message and a `fix.remedy` written for the user, who reads them in the
        turn's closing text.
        """
        failure = parse_tool_failure(result.structured_content or {})
        if is_anchor_fault(failure):
            return self._anchor_fault(step, failure)
        detail = failure.fix.get("detail")
        worded = _ANCHOR_DETAIL_WORDING.get(detail) if isinstance(detail, str) else None
        if worded is None and (message := _ANCHOR_REFUSAL_MESSAGES.get(failure.code)):
            worded = (message, _NAMED_YEARS_REMEDY)
        if worded is not None:
            message, remedy = worded
            failure = failure.model_copy(
                update={"message": message, "fix": failure.fix | {"remedy": remedy}}
            )
        return self._step_refused(step.id, _RESOLVE_ANCHOR, result, failure)

    def _anchor_fault(self, step: WorkflowStep, failure: ToolFailure) -> _Stopped:
        """The stop for a refusal saying the run's own newest period is not in order.

        The populations, root frames, key and periods a rolling call carries
        are all the run's own, so the user is told of an internal error and
        the refusal is logged with its trace id.
        """
        trace_id = current_trace_id()
        logger.error(
            "executor: step %s was refused %s (%s) on the newest period the run resolved "
            "trace_id=%s",
            step.id,
            failure.code,
            failure.fix.get("detail"),
            trace_id,
        )
        return self._refused(
            RefusalKind.STEP_REFUSED,
            f"step {step.id!r} was refused {failure.code!r} on the newest period the run itself "
            f"resolved: {failure.fix.get('detail')!r}",
            step_id=step.id,
            payload=ToolFailure(
                code=ExecutorCode.INTERNAL_ERROR,
                message=internal_error_message(subject="Working out the latest period"),
                retry_mode="do_not_retry",
                fix={"refusal": failure.model_dump(mode="json")},
            ),
        )

    def _check_rolled_start(self, step: WorkflowStep, bounds: Sequence[RollingBound]) -> None:
        """Refuse `ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET` for a rolled start before the data.

        The start is the earliest year a rolled bound keeps (`_first_year_kept`).
        A dataset's first year is the start of the window its retrieval's
        evidence recorded, which is read off the retrieved series. A retrieval
        whose evidence states no window is not compared.

        The refusal's payload is worded for the user and classifies as a
        stop, since the same plan resolves the same start over the same data.
        """
        starts = [year for bound in bounds if (year := _first_year_kept(bound)) is not None]
        if not starts:
            return
        anchor = self.anchored[step.id].result
        start = min(starts)
        under = self._retrievals_under(step.id)
        for root in self.group_of[step.id].roots:
            evidence = self.evidence_by_step.get(root)
            if root not in under or evidence is None or evidence.window is None:
                continue
            first = evidence.window.start
            if start >= first:
                continue
            binding = self.evidence_bindings[root]
            requirement = self.requirements[binding.requirement_id]
            assert isinstance(requirement, DatasetRequirement)
            name = self.approved_source[root].measure_label or requirement.measure
            raise self._refused(
                RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET,
                f"step {step.id!r} would start in {start}, counted from {anchor.period}, "
                f"before {first}, the first year {binding.spec_id} holds",
                step_id=step.id,
                requirement_id=requirement.id,
                payload=ToolFailure(
                    code=ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET,
                    message=(
                        f"The data for {name} starts in {first}. The period asked for would "
                        f"start in {start}, counted back from {anchor.period}, the latest "
                        "year at least half of its series report."
                    ),
                    retry_mode="do_not_retry",
                    fix={
                        "remedy": f"Ask for a shorter period, or for named years from {first} on.",
                        "spec_id": binding.spec_id,
                        "first_year": first,
                        "window_start": start,
                        "anchor": anchor.period,
                    },
                ),
            )

    def _check_left_out_narrowings(self) -> None:
        """Refuse the run before its first derived step when a step narrows a
        frame to named entities only, every one of which a retrieval under
        that frame left out (`compiler.evidence.left_out_entities`).

        Every figure such a step gives for the entity rests on the retrieval
        that lacks it. A side or filter over that retrieval's own rows keeps
        no row. A keyed `share_of` or `quotient_of` holds no key for the
        entity when the retrieval lacking it is the numerator or a keyed
        denominator, and divides it by a total without the entity's own
        amounts when that retrieval is a denominator read whole. So the step
        is refused whichever side of the frame the left-out retrieval sits
        on. It stops as
        `REQUIREMENT_UNMET` naming the first such step in plan order, its
        requirement the left-out entity's, carrying each left-out entity's
        `ENTITY_NO_SERIES` finding as an error. A narrowing is read off the
        step's compiled arguments (`compiler.frame_columns.entity_narrowings`),
        whose columns are the approved source's, as the left-out entities'
        are.
        """
        refused: list[tuple[str, EvidenceIssue]] = []
        for step in self.workflow.steps:
            for narrowing in entity_narrowings(step.tool, step.args):
                binding = step.bindings.get(narrowing.frame) or step.bindings.get(FRAME_PARAMETER)
                if binding is None or not narrowing.refs:
                    continue
                left_out = {
                    key: issue
                    for retrieval in sorted(self._retrievals_under(binding.from_step))
                    for key, issue in self.left_out.get(retrieval, {}).items()
                }
                missing = [
                    left_out[(narrowing.column, ref)]
                    for ref in sorted(narrowing.refs)
                    if (narrowing.column, ref) in left_out
                ]
                if len(missing) == len(narrowing.refs):
                    refused.extend((step.id, issue) for issue in missing)
        if not refused:
            return
        step_id, first = refused[0]
        raise self._refused(
            RefusalKind.REQUIREMENT_UNMET,
            "; ".join(
                f"step {narrowing_step!r} reads {issue.required!r} alone, which the retrieval "
                "holds no series for"
                for narrowing_step, issue in refused
            ),
            step_id=step_id,
            requirement_id=first.requirement_id,
            issues=tuple(
                issue.model_copy(
                    update={
                        "severity": Severity.ERROR,
                        "detail": f"the retrieval holds no series for {issue.required!r}, "
                        f"which step {narrowing_step!r} reads alone",
                    }
                )
                for narrowing_step, issue in refused
            ),
        )

    def _retrievals_under(self, step_id: str) -> frozenset[str]:
        """The retrieval steps `step_id` rests on, through its bindings."""
        if step_id in self.evidence_bindings:
            return frozenset({step_id})
        return frozenset().union(
            *(
                self._retrievals_under(binding.from_step)
                for binding in self.steps_by_id[step_id].bindings.values()
            )
        )

    # -- one step -----------------------------------------------------------------------

    async def _run_step(
        self, step: WorkflowStep, schema: ToolSchema
    ) -> tuple[StepRecord, EvidenceRecord | None]:
        """Call `step`'s tool and return what it produced.

        Writes `self.frames[step.id]`, `self.row_counts[step.id]`,
        `self.stat_var_counts[step.id]`, `self.dropped_key_counts[step.id]`,
        `self.ranked_before_limit[step.id]`, `self.role_keys[step.id]` and
        `self.columns[step.id]`. These dicts are keyed by step id, so a step
        running concurrently with its siblings can write them safely. It
        leaves filling `self.records` and `self.evidence_by_step`'s own slots
        for this step to the caller,
        which does so once every step in this one's wave has run.

        Everything after the arguments resolve runs inside the step's
        budget (`_budget_s`), and running past it is `TRANSPORT_FAILED`.
        `current_step_deadline` carries that same budget as a `CallDeadline`,
        set for this whole method's body -- not only for `_call`, the way
        `current_step_id` is -- so `default_fetch_series_stats` can read it
        and give `_check_evidence`'s own evidence fetch a deadline of its own.
        """
        arguments = self._arguments(step, schema)
        expected_role_keys = self._expected_role_keys(step)
        budget_s = self._budget_s(step)
        step_deadline = CallDeadline.after(budget_s) if budget_s is not None else None
        budget = asyncio.timeout(step_deadline.remaining() if step_deadline is not None else None)
        try:
            async with budget:
                deadline_token = current_step_deadline.set(step_deadline)
                budget_token = _step_budget.set(budget if budget_s is not None else None)
                try:
                    token = current_step_id.set(step.id)
                    try:
                        result, attempts = await self._call(step.id, step.tool, arguments)
                    finally:
                        current_step_id.reset(token)
                    if result.is_error:
                        raise self._stopped_by_refusal(step, result)
                    frame, manifest_ref = self._frame(step, result)
                    if self._retrieves(step):
                        read = manifest_ref or frame
                        self.inputs[step.id] = ManifestEntry(
                            spec_id=read.spec_id, content_version=read.content_version
                        )
                        self.retrievals[step.id] = without_long_stat_var_list(
                            result.structured_content or {}
                        )
                    self._learn_columns(step, result)
                    self._check_declared_shape(step, result, arguments, expected_role_keys)
                    self.frames[step.id] = frame
                    record = StepRecord(
                        step_id=step.id,
                        tool=step.tool,
                        attempts=attempts,
                        frame_ref=frame,
                        manifest_ref=manifest_ref,
                        digest=result.text,
                        row_count=self.row_counts[step.id],
                        role_keys=self.role_keys[step.id],
                        stat_var_count=self.stat_var_counts[step.id],
                        dropped_key_count=self.dropped_key_counts[step.id][0],
                        right_dropped_key_count=self.dropped_key_counts[step.id][1],
                        ranked_before_limit=self.ranked_before_limit[step.id],
                    )
                    evidence = (
                        await self._check_evidence(step, result)
                        if step.id in self.evidence_bindings
                        else None
                    )
                    return record, evidence
                finally:
                    _step_budget.reset(budget_token)
                    current_step_deadline.reset(deadline_token)
        except TimeoutError as exc:
            if not budget.expired():
                raise
            raise self._refused(
                RefusalKind.TRANSPORT_FAILED,
                f"step {step.id!r} ran past its {budget_s:g}s budget for up to "
                f"{MAX_ATTEMPTS_PER_STEP} calls to {step.tool.name!r}, their retry delays "
                "and any evidence check",
                step_id=step.id,
            ) from exc

    def _budget_s(self, step: WorkflowStep) -> float | None:
        """`step_budget_s` for `step`, or `None` when its tool pins no call timeout."""
        capability = self.capabilities.get(step.tool)
        if capability is None or capability.call_timeout_s is None:
            return None
        return step_budget_s(capability.call_timeout_s, evidence=step.id in self.evidence_bindings)

    def _check_declared_shape(
        self,
        step: WorkflowStep,
        result: ToolCallResult,
        arguments: Mapping[str, JsonValue],
        expected_role_keys: tuple[str, ...] | None,
    ) -> None:
        """Compare a result's own declared shape against the shape `step` must
        produce, refusing `SHAPE_MISMATCH` on a difference, and record the
        declared row count, role keys and StatVar count for `_answer_refusal`,
        the steps reading this one and the step's `StepRecord`, with the keys
        the result declares it left out (`declared_dropped_key_counts`).

        `expected_role_keys` is `None` for a retrieval: the role keys and
        StatVar count its result declares are the frame's own, and stand as
        its shape, except that a retrieval asked to sum its family must
        declare one StatVar (`_check_summed_retrieval`). Any other step's
        declared StatVar count is checked against
        `bound_stat_var_count` for `arguments` and the count the step that
        produced its `FRAME_PARAMETER` input declared. With zero rows there
        is no stat_var to count, so only the role keys are compared. A result
        whose declared role keys are not valid role columns
        (`declared_shape`) refuses `STEP_RESULT_UNUSABLE`.
        """
        try:
            row_count, role_keys, stat_var_count = declared_shape(result.structured_content)
        except ArtifactError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE, str(exc), step_id=step.id
            ) from exc
        self.row_counts[step.id] = row_count
        self.stat_var_counts[step.id] = stat_var_count
        self.dropped_key_counts[step.id] = declared_dropped_key_counts(result.structured_content)
        self.ranked_before_limit[step.id] = declared_ranked_before_limit(result.structured_content)
        self.role_keys[step.id] = role_keys
        binding = step.bindings.get(FRAME_PARAMETER)
        if binding is None:
            self._check_summed_retrieval(step, arguments, stat_var_count)
            return
        if (
            expected_role_keys is not None
            and role_keys is not None
            and role_keys != expected_role_keys
        ):
            raise self._refused(
                RefusalKind.SHAPE_MISMATCH,
                f"{step.tool.name!r} declared role keys {role_keys!r}, not the "
                f"{expected_role_keys!r} the plan expects of it",
                step_id=step.id,
            )
        if row_count is None or row_count == 0:
            return
        input_count = self.stat_var_counts.get(binding.from_step)
        most = bound_stat_var_count(step.tool, arguments, input_stat_var_count=input_count)
        if (
            stat_var_count is None
            or stat_var_count < 1
            or (most is not None and stat_var_count > most)
        ):
            allowed = "at least 1" if most is None else "1" if most == 1 else f"1 to {most}"
            raise self._refused(
                RefusalKind.SHAPE_MISMATCH,
                f"{step.tool.name!r} declared {stat_var_count!r} distinct stat_vars over an "
                f"input declaring {input_count!r}; its non-empty result holds {allowed}",
                step_id=step.id,
            )

    def _check_summed_retrieval(
        self, step: WorkflowStep, arguments: Mapping[str, JsonValue], stat_var_count: int | None
    ) -> None:
        """Refuse `SHAPE_MISMATCH` when a retrieval asked to sum its family declares several StatVars.

        A summed family's frame holds one StatVar, the family's own ref, or
        none when it is empty. A count above one means the sum did not
        happen, and every single-measure verb reading the frame would refuse
        it at run time. No count passes: `get_data` declares none when its
        fetch of the frame degraded, which says nothing about the sum, and a
        retrieval that declares none goes on the way a separate one does. A
        retrieval that does not sum has no bound (`bound_stat_var_count`).

        The refusal carries an `ExecutorCode.FAMILY_NOT_SUMMED` payload, which
        `classify_refusal` reads as a stop: the same plan draws the same
        unsummed frame, so a replan would ask the user to approve it twice.
        The payload's message and `fix.remedy` are written for the user, who
        reads them in the turn's closing text.
        """
        most = bound_stat_var_count(step.tool, arguments, input_stat_var_count=None)
        if most is None or stat_var_count is None:
            return
        if stat_var_count > most:
            raise self._refused(
                RefusalKind.SHAPE_MISMATCH,
                f"{step.tool.name!r} was asked to sum its family of StatVars into one series "
                f"but declared {stat_var_count!r} distinct stat_vars",
                step_id=step.id,
                payload=ToolFailure(
                    code=ExecutorCode.FAMILY_NOT_SUMMED,
                    message=_FAMILY_NOT_SUMMED_MESSAGE,
                    retry_mode="do_not_retry",
                    fix={"remedy": _FAMILY_NOT_SUMMED_REMEDY},
                ),
            )

    def _expected_role_keys(self, step: WorkflowStep) -> tuple[str, ...] | None:
        """The role keys `step` must declare, or `None` for a retrieval, which
        reads no frame and whose result states its frame's own.

        Raises `COLUMN_UNBOUND`, before the call, when the prediction names a
        role the frame read carries no column for.
        """
        binding = step.bindings.get(FRAME_PARAMETER)
        if binding is None:
            return None
        try:
            return bound_role_keys(
                step.tool,
                step.expected_role_keys,
                frame=self.columns.get(binding.from_step),
                input_role_keys=self.role_keys.get(binding.from_step),
            )
        except UnboundColumnError as exc:
            raise self._column_unbound(step, exc) from exc

    def _learn_columns(self, step: WorkflowStep, result: ToolCallResult) -> None:
        """Record the columns the plan's role columns bind to on `step`'s output.

        A retrieval's are read off its result, for every role its approved
        source resolves under (`retrieved_frame_columns`). Any other step's
        output carries the binding of its `FRAME_PARAMETER` input, the one
        its own arguments were spelled against.
        """
        source = self.approved_source.get(step.id)
        if source is None:
            binding = step.bindings.get(FRAME_PARAMETER)
            self.columns[step.id] = (
                self.columns.get(binding.from_step) if binding is not None else None
            )
            return
        base_columns = {
            *step.expected_role_keys,
            *(entity.role for entity in source.entities),
            *(axis.column for axis in source.open_entity_axes or ()),
        }
        compiled = {
            *base_columns,
            *(
                f"{column}{suffix}"
                for column in base_columns
                for suffix in ("_type", "_party_kind")
            ),
        }
        try:
            self.columns[step.id] = retrieved_frame_columns(
                result.structured_content or {}, compiled=compiled
            )
        except ArtifactError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE, str(exc), step_id=step.id
            ) from exc

    def _column_unbound(self, step: WorkflowStep, exc: UnboundColumnError) -> _Stopped:
        return self._refused(
            RefusalKind.COLUMN_UNBOUND,
            f"step {step.id!r} cannot call {step.tool.name!r}: {exc}",
            step_id=step.id,
        )

    def _arguments(self, step: WorkflowStep, schema: ToolSchema) -> dict[str, JsonValue]:
        """The step's literal arguments plus every binding resolved to a frame,
        with every role column they name spelled the way that frame spells it
        (`compiler.frame_columns.bound_arguments`).

        A rolling step's population is spelled the way the workflow's
        retrieved frames spell it, its periods are written from the newest
        period the run resolved, and that resolution's key is added
        (`compiler.rolling.resolved_arguments`). A rolled start before the
        step's data refuses here (`_check_rolled_start`).
        """
        arguments: dict[str, JsonValue] = dict(step.args)
        for parameter, binding in step.bindings.items():
            frame = self.frames.get(binding.from_step)
            if frame is None or binding.output is not ArtifactKind.FRAME_REF:
                raise self._refused(
                    RefusalKind.BINDING_INVALID,
                    f"step {step.id!r} binds {parameter!r} to {binding.from_step!r}, which produced "
                    "no frame",
                    step_id=step.id,
                )
            arguments[parameter] = frame.model_dump()
        anchored = self.anchored.get(step.id)
        try:
            arguments = bound_arguments(
                step.tool,
                arguments,
                frames={
                    parameter: self.columns.get(binding.from_step)
                    for parameter, binding in step.bindings.items()
                },
                roots=anchored.columns if anchored is not None else None,
            )
        except UnboundColumnError as exc:
            raise self._column_unbound(step, exc) from exc
        if is_rolling(step):
            assert anchored is not None, "a rolling step runs once its newest period is resolved"
            arguments = resolved_arguments(
                arguments, period=anchored.result.period, anchor_key=anchored.result.anchor_key
            )
            self._check_rolled_start(step, rolling_bounds(arguments))
        self._check_fit(step, step.tool, arguments, schema)
        return arguments

    def _check_fit(
        self,
        step: WorkflowStep,
        tool: ToolId,
        arguments: Mapping[str, JsonValue],
        schema: ToolSchema,
    ) -> None:
        """Refuse `BINDING_INVALID` when `arguments` do not fit `tool`'s live schema."""
        problems = literal_argument_errors(arguments=arguments, schema=schema.parameters)
        if problems:
            raise self._refused(
                RefusalKind.BINDING_INVALID,
                f"step {step.id!r}'s arguments do not fit {tool.name!r}: {problems!r}",
                step_id=step.id,
            )

    async def _call(
        self, step_id: str, tool: ToolId, arguments: Mapping[str, JsonValue]
    ) -> tuple[ToolCallResult, int]:
        """Call `tool` for step `step_id`, retrying only what the tool or the transport says can be retried.

        Each call is recorded as the run's attempt before it goes out. Every
        call hands `on_progress`, bound to the step's id, to the surface. Each answer is reported as it arrives, so the step's line
        stops showing the server's last wait message once the call returns.
        Each retry reports the attempt it is about to make before its delay
        starts.
        """
        on_progress = partial(self.on_progress, step_id) if self.on_progress is not None else None
        for attempt in range(MAX_ATTEMPTS_PER_STEP):
            last = attempt + 1 == MAX_ATTEMPTS_PER_STEP
            await self._record_attempt(step_id, tool)
            try:
                result = await self.surface.call_tool(
                    tool=tool, arguments=arguments, on_progress=on_progress
                )
            except ToolSurfaceUnavailableError as exc:
                capability = self.capabilities.get(tool)
                final = last or (
                    capability is not None and not capability.retry_on_transport_failure
                )
                if final:
                    raise self._refused(
                        RefusalKind.TRANSPORT_FAILED, str(exc), step_id=step_id
                    ) from exc
                await self._before_retry(
                    jittered_retry_delay({}, attempt, jitter=self.jitter), attempt, on_progress
                )
                continue
            if on_progress is not None:
                deliver_progress(
                    on_progress,
                    ProgressReport(
                        progress=attempt + 1,
                        total=MAX_ATTEMPTS_PER_STEP,
                        message="Result received; checking it",
                    ),
                )
            if not result.is_error:
                return result, attempt + 1
            delay_s = delay_before_retry(
                result.structured_content or {}, attempt, jitter=self.jitter
            )
            if delay_s is None:
                return result, attempt + 1
            await self._before_retry(delay_s, attempt, on_progress)
        raise AssertionError("unreachable: every attempt returns or raises")

    async def _record_attempt(self, step_id: str, tool: ToolId) -> None:
        """Record an attempt of `tool` for step `step_id`, with the step's
        budget, and the deadline `current_step_deadline` carries, stopped
        while the ledger writes."""
        budget = _step_budget.get()
        if budget is None or (expires := budget.when()) is None:
            await self._record(Attempt(step_id, tool.name), step_id=step_id)
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        budget.reschedule(None)
        try:
            await self._record(Attempt(step_id, tool.name), step_id=step_id)
        finally:
            spent = loop.time() - started
            budget.reschedule(expires + spent)
            if (deadline := current_step_deadline.get()) is not None:
                current_step_deadline.set(replace(deadline, expires_at=deadline.expires_at + spent))

    async def _before_retry(
        self, delay_s: float, attempt: int, on_progress: ProgressCallback | None
    ) -> None:
        """Report the attempt after `attempt` (zero-based), then wait `delay_s` for it."""
        if on_progress is not None:
            following = attempt + 2
            deliver_progress(
                on_progress,
                ProgressReport(
                    progress=following,
                    total=MAX_ATTEMPTS_PER_STEP,
                    message=(
                        f"Retrying in {delay_s:.3g}s (attempt {following} of {MAX_ATTEMPTS_PER_STEP})"
                    ),
                ),
            )
        await self.sleep(delay_s)

    def _stopped_by_refusal(self, step: WorkflowStep, result: ToolCallResult) -> _Stopped:
        payload = result.structured_content or {}
        failure = parse_tool_failure(payload)
        if is_anchor_fault(failure):
            return self._anchor_fault(step, failure)
        disposition = classify_refusal(failure)
        if disposition is Disposition.METHOD_CHOICE:
            return _Stopped(
                MethodChoiceNeeded(
                    revision=self.revision,
                    run_id=self.run_id,
                    step_id=step.id,
                    tool=step.tool,
                    code=failure.code,
                    candidates=candidates_of(failure),
                    refusal=failure,
                    completed=self._completed_so_far(),
                    evidence=self._evidence_so_far(),
                )
            )
        binding = self.evidence_bindings.get(step.id)
        if disposition is Disposition.REDISCOVER and binding is not None:
            return _Stopped(
                RediscoveryNeeded(
                    revision=self.revision,
                    run_id=self.run_id,
                    step_id=step.id,
                    tool=step.tool,
                    requirement_id=binding.requirement_id,
                    spec_id=binding.spec_id,
                    refusal=failure,
                    completed=self._completed_so_far(),
                    evidence=self._evidence_so_far(),
                )
            )
        # A rediscovery on a step bound to no evidence would name nothing to
        # rediscover, so it is a plain stop like any other refusal.
        return self._step_refused(step.id, step.tool, result, failure)

    def _step_refused(
        self, step_id: str, tool: ToolId, result: ToolCallResult, failure: ToolFailure
    ) -> _Stopped:
        """The plain stop for a call `tool` refused, carrying `failure` as its payload.

        The detail reaches the planner's replan prompt. A refusal's text ends
        in its JSON envelope (`common.refusal.refusal_text`), so the detail
        takes the structured `message`. A result with no refusal payload,
        such as a tool that raised, has only its text, which carries no
        envelope.
        """
        reason = result.text if failure.code == ExecutorCode.MALFORMED_REFUSAL else failure.message
        return self._refused(
            RefusalKind.STEP_REFUSED,
            f"{tool.name!r} refused: {reason or failure.code}",
            step_id=step_id,
            payload=failure,
        )

    def _frame(
        self, step: WorkflowStep, result: ToolCallResult
    ) -> tuple[FrameRef, FrameRef | None]:
        """The frame `step` produced, and the manifest's own pair when the frame
        is a retrieval held under another (`StepRecord.manifest_ref`)."""
        capability = self.capabilities.get(step.tool)
        output = (
            next((o for o in capability.outputs if o.artifact is ArtifactKind.FRAME_REF), None)
            if capability
            else None
        )
        if output is None:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"no capability says how {step.tool.name!r} yields a frame",
                step_id=step.id,
            )
        try:
            frame = extract_frame_ref(output.extract, result.structured_content)
            if output.extract is not Extract.FRAME_REF_FROM_MANIFEST:
                return frame, None
            manifest_ref = manifest_frame_ref(result.structured_content or {})
        except ArtifactError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE, str(exc), step_id=step.id
            ) from exc
        return frame, manifest_ref if manifest_ref != frame else None

    async def _check_evidence(self, step: WorkflowStep, result: ToolCallResult) -> EvidenceRecord:
        requirement = self.requirements[self.evidence_bindings[step.id].requirement_id]
        # An evidence binding only ever names a retrieval, and only a dataset
        # requirement is ever carried out by one (`planner.proof.mismatches`
        # rejects any other pairing before a plan can be approved).
        assert isinstance(requirement, DatasetRequirement)
        try:
            # Membership compares every StatVar the trace lists, so a list the
            # result left out is read back first; a read that fails refuses.
            structured = await with_stat_var_list(result.structured_content or {})
            manifest = parse_manifest(structured)
            series_stats = await self.fetch_series_stats(manifest)
            evidence = evidence_from_manifest(manifest, series_stats=series_stats)
        except StatisticsUnreachableError as exc:
            raise self._refused(RefusalKind.TRANSPORT_FAILED, str(exc), step_id=step.id) from exc
        except (ArtifactError, ValueError, ResourceError, StatisticsUnusableError) as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE, str(exc), step_id=step.id
            ) from exc
        approved = self.approved_source[step.id]
        findings = check_evidence(requirement, evidence, approved=approved)
        failed = errors(findings)
        if failed:
            raise self._refused(
                RefusalKind.REQUIREMENT_UNMET,
                f"requirement {requirement.id!r} is not met by what {step.tool.name!r} returned",
                step_id=step.id,
                requirement_id=requirement.id,
                issues=tuple(findings),
            )
        if refs := unlabelled_series_refs(findings, limit=LABELLED_REFS):
            findings = with_series_labels(findings, await _series_labels(self.label_refs, refs))
        self.left_out[step.id] = {
            (entity.role, entity.ref): issue
            for entity, issue in left_out_entities(requirement, evidence, approved).items()
        }
        return EvidenceRecord(
            requirement_id=requirement.id,
            step_id=step.id,
            frame_ref=self.frames[step.id],
            spec_id=evidence.spec_id,
            stat_vars=evidence.stat_vars,
            window=evidence.window,
            gaps=tuple(f for f in findings if f.severity is not Severity.INFO),
            citation=manifest.citation.text,
        )

    def _refused(
        self,
        kind: RefusalKind,
        detail: str,
        *,
        step_id: str | None = None,
        requirement_id: str | None = None,
        payload: ToolFailure | None = None,
        issues: tuple[EvidenceIssue, ...] = (),
    ) -> _Stopped:
        return _Stopped(
            Refused(
                revision=self.revision,
                run_id=self.run_id,
                refusal=kind,
                detail=detail,
                step_id=step_id,
                requirement_id=requirement_id,
                payload=payload,
                issues=issues,
                completed=self._completed_so_far(),
                evidence=self._evidence_so_far(),
            )
        )
