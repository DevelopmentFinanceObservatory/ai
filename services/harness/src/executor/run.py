"""Run an approved plan's workflow, step by step, and check what comes back.

`execute_approved_plan` is the one public entry point: it moves a plan
`APPROVED` to `EXECUTING`, runs the workflow, and leaves it `COMPLETED` or
`FAILED`. A plan that is not `APPROVED` is refused (`NOT_APPROVED`) without
being touched; one whose content no longer hashes to what was approved is
refused (`STALE_REVISION`) by `start_execution`. `_run_workflow` is the step
loop underneath it, private because running a plan that has not gone through
`start_execution` is a caller error, not an outcome a caller should have to
handle.

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
capped at `EVIDENCE_ALLOWANCE_S` (`default_fetch_series_stats`); a fetch that
outruns it degrades the requirement to `STATISTICS_UNAVAILABLE` rather than
spending the rest of the step's own budget. A step that runs past that
budget stops as `TRANSPORT_FAILED`, the same refusal a call that ran past its
own deadline gets. A tool with no pinned timeout runs at the surface's
default per call and has no statistics fetch, so each of its attempts is
already bounded and no step budget applies.

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

A rolled start before the first year its dataset holds is refused
`ROLLING_WINDOW_OUTSIDE_DATASET` before the step's call, with the first year
read off the retrieval's own evidence. `resolve_anchor` finding no qualifying
year, or periods that are no calendar years, stops the run with the tool's
code and a message for the user. A `rolling_anchor_unresolved` of any other
detail says the run's own populations, roots, key or periods do not fit
(`executor.classify.is_anchor_fault`) and stops it as an `internal_error`.

A keyed `quotient_of` leaves out each numerator key it finds no denominator
for, and its result declares how many (`dropped_key_count`). The step's
`StepRecord` carries that count, so an answer resting on the step can say
its population lost members.

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
  stops with a `FAMILY_NOT_SUMMED` payload, which ends the turn.
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
import json
import logging
import random
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
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
)
from compiler.frame_columns import (
    FRAME_PARAMETER,
    FrameColumns,
    UnboundColumnError,
    bound_arguments,
    bound_populations,
    bound_role_keys,
    bound_stat_var_count,
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
from data_mcp.deadline import CallDeadline
from data_mcp.manifest import Manifest
from data_mcp.series_statistics import (
    SERIES_STATISTICS_ARROW_SCHEMA,
    SeriesStatisticsSchemaError,
    fetch_series_statistics_table,
    find_statistics_artifact,
)
from data_mcp.stat_var_list import with_stat_var_list
from fastmcp.exceptions import ResourceError
from planner.plan import DatasetRequirement, Plan, PlanStatus, plan_revision
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
    declared_dropped_key_count,
    declared_shape,
    extract_frame_ref,
    manifest_frame_ref,
    parse_manifest,
    retrieved_frame_columns,
)
from executor.classify import (
    BACKOFF_SECONDS,
    FAMILY_NOT_SUMMED,
    INTERNAL_ERROR,
    MALFORMED_REFUSAL,
    MAX_ATTEMPTS_PER_STEP,
    MAX_RETRY_AFTER_SECONDS,
    ROLLING_ANCHOR_UNRESOLVED,
    ROLLING_WINDOW_OUTSIDE_DATASET,
    Disposition,
    Jitter,
    ToolFailure,
    candidates_of,
    classify_refusal,
    delay_before_retry,
    is_anchor_fault,
    jittered_retry_delay,
    parse_tool_failure,
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
# something `default_fetch_series_stats` can catch.
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

Sleep = Callable[[float], Awaitable[None]]

# `evidence_from_manifest`'s `series_stats` argument, resolved for one manifest. Injectable
# so a caller whose tool surface is itself fully virtualised -- `harness.scripted`'s
# deterministic mode states "no upstream" as its whole point -- can answer this from its own
# canned data instead of `fetch_series_statistics_table`'s real signed-URL fetch, the same way it
# already supplies its own `surface`/`executor`/`resolver` in place of the real ones.
FetchSeriesStats = Callable[[Manifest], Awaitable[pa.Table | None]]

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
    `series_count` (dfo-apis mints an all-years definition that way) is
    fetched with no row count to check. `series_count == 0` answers an empty
    table directly, with no fetch either -- `_series_issues` treats `None`
    and no rows alike as
    `STATISTICS_UNAVAILABLE`, so there is nothing a fetch could add for
    a mint that resolved to no series. Otherwise this fetches every row,
    with no cap: unlike the MCP digest's own `_STATISTICS_SERIES_CAP`,
    which only bounds what a model reads, `_series_issues` is a bulk
    coverage check that reads every series a retrieval carries, as Arrow
    columns rather than one model a series. A fetch or read failure is not
    raised: it degrades to `None`, which
    `_series_issues` already reports as `STATISTICS_UNAVAILABLE` --
    failing the requirement closed rather than passing it unverified,
    `TimeoutError` included. The fetch runs under its own deadline --
    `current_step_deadline`'s remaining seconds, capped at
    `EVIDENCE_ALLOWANCE_S` and reduced by `_EVIDENCE_DEADLINE_MARGIN_S` -- so
    a `get_data` call that used most of the step's own budget still degrades
    the requirement alone rather than failing the whole step
    `transport_failed` once the outer budget catches up.
    """
    if manifest.series_count == 0:
        return SERIES_STATISTICS_ARROW_SCHEMA.empty_table()
    if manifest.definition is None:
        return None
    artifact = find_statistics_artifact(manifest)
    if artifact is None:
        if manifest.series_count is None:
            return None
        logger.warning(
            "executor: manifest for %s/%s names series_count=%d but carries no statistics artifact",
            manifest.definition.spec_id,
            manifest.content_version,
            manifest.series_count,
        )
        return None
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
            expected_rows=manifest.series_count,
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
        logger.warning(
            "executor: could not fetch or decode the statistics artifact for %s/%s: %s",
            manifest.definition.spec_id,
            manifest.content_version,
            exc,
        )
        return None


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


async def execute_approved_plan(
    plan: Plan,
    *,
    surface: ToolSurface,
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    on_progress: StepProgressCallback | None = None,
) -> tuple[Plan, RunOutcome]:
    """Run an approved plan and record the run in its status.

    The only public way to run a plan: a plan that is not `APPROVED` is
    refused (`NOT_APPROVED`) without being touched, and one whose content no
    longer hashes to `approved_revision` is refused (`STALE_REVISION`) by
    `Plan.start_execution`.

    `sleep` waits between retries and `jitter` draws each wait
    (`executor.classify.jittered_retry_delay`); both injectable so tests
    neither wait nor draw at random. `max_concurrent_steps` bounds how many
    of a wave's steps may have their tool call in flight at once (`_Run`'s
    semaphore). `fetch_series_stats`
    resolves a dataset step's manifest into the rows `evidence_from_manifest`
    checks coverage against; the real fetch by default, injectable for a
    caller whose `surface` is itself fully virtualised. `on_progress` hears
    every progress report a step's tool call sends, and a report naming the
    next attempt each time a step is retried, tagged with the step's id.

    Returns:
        The plan as the run left it -- `COMPLETED` after an accepted run,
        `FAILED` otherwise, or unchanged when it could not start -- and the
        outcome.
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
    outcome = await _run_workflow(
        running,
        surface=surface,
        capabilities=capabilities,
        sleep=sleep,
        jitter=jitter,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
        on_progress=on_progress,
    )
    finished = running.complete() if isinstance(outcome, Accepted) else running.fail()
    return finished, outcome


async def _run_workflow(
    plan: Plan,
    *,
    surface: ToolSurface,
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.uniform,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    on_progress: StepProgressCallback | None = None,
) -> RunOutcome:
    """Run `plan`'s workflow against `surface` and return what came of it.

    Private: reachable only through `execute_approved_plan`, once
    `start_execution` has moved the plan to `EXECUTING`. Calling this on a
    plan in any other status is a caller error.

    Args:
        plan: A plan `EXECUTING` its compiled workflow.
        surface: The servers to call, the same the workflow was compiled against.
        capabilities: The mapping table; the module default in production.
        sleep: Waits between retries. Injectable so tests need not wait.
        jitter: Draws each retry's wait between its bounds. Injectable so
            tests can seed it.
        max_concurrent_steps: How many of a wave's steps may have their tool
            call in flight at once.
        fetch_series_stats: Resolves a dataset step's manifest into evidence
            rows; the real fetch by default (`execute_approved_plan`'s own
            docstring).
        on_progress: Hears each step's progress reports, tagged with its id.

    Returns:
        `Accepted` when every step ran and every requirement is met,
        `MethodChoiceNeeded` when a tool refused with a choice for the user,
        `RediscoveryNeeded` when a tool said only a refresh could fix its
        refusal, `Refused` otherwise. A refusal's own `also_stopped` names
        the rest of its wave's stops, wave order, when there were any.

    Raises:
        ValueError: If `plan` is not `EXECUTING`.
    """
    if plan.status is not PlanStatus.EXECUTING:
        raise ValueError(
            f"this plan is {plan.status.value}, not executing; only execute_approved_plan "
            "may run a plan, and only once it has started execution"
        )
    assert plan.workflow is not None, "an EXECUTING plan always carries a workflow"
    run = _Run(
        plan,
        plan.workflow,
        surface,
        capabilities,
        sleep,
        jitter,
        max_concurrent_steps,
        fetch_series_stats,
        on_progress,
    )
    try:
        return await run.execute()
    except _Stopped as stopped:
        if not stopped.siblings:
            return stopped.outcome
        return stopped.outcome.model_copy(update={"also_stopped": stopped.siblings})


class _Run:
    """One run's working state."""

    def __init__(
        self,
        plan: Plan,
        workflow: Workflow,
        surface: ToolSurface,
        capabilities: Sequence[ToolCapability],
        sleep: Sleep,
        jitter: Jitter,
        max_concurrent_steps: int = 4,
        fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
        on_progress: StepProgressCallback | None = None,
    ) -> None:
        self.plan = plan
        self.workflow = workflow
        self.surface = surface
        self.sleep = sleep
        self.jitter = jitter
        self.fetch_series_stats = fetch_series_stats
        self.on_progress = on_progress
        # Bounds how many steps of one wave may have their tool call in
        # flight at once, acquired in `_attempt` around `_run_step`; a wave
        # wider than the bound queues at the point of call rather than
        # opening every call in the wave at once.
        self.concurrency = asyncio.Semaphore(max_concurrent_steps)
        self.revision = plan_revision(plan)
        self.capabilities = {c.tool: c for c in capabilities}
        self.requirements = {r.id: r for r in plan.evidence_requirements}
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
        self.frames: dict[str, FrameRef] = {}
        self.row_counts: dict[str, int | None] = {}
        self.stat_var_counts: dict[str, int | None] = {}
        self.dropped_key_counts: dict[str, int] = {}
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

    async def execute(self) -> Accepted:
        schemas = await self._surface_as_compiled()
        for wave in _waves(self.workflow):
            for group in self.anchor_groups:
                rolling = next((step for step in wave if step.id in group.steps), None)
                if rolling is None or rolling.id in self.anchored:
                    continue
                # Every retrieval of the group is in a wave before this one
                # (`_waves`), and the wave's steps start only once this returns.
                try:
                    await self._resolve_anchor(group, rolling, schemas[_RESOLVE_ANCHOR.name])
                except _Stopped as stopped:
                    raise _Stopped(self._restamped(stopped.outcome)) from stopped
            # Each step runs as its own task; `asyncio.gather` copies the
            # calling context per task, so `current_step_id`, set inside
            # `_run_step`, never leaks between concurrent siblings.
            # `return_exceptions=True` lets every step in the wave run to
            # completion even when one of them raises something `_attempt`
            # did not catch: an exception here is only ever a defect
            # `_Stopped` was never meant to carry (`_Stopped` itself is
            # always returned, never raised, out of `_attempt`), and letting
            # it end the wave at once would abandon that wave's other
            # in-flight steps unawaited rather than let them finish. A
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
                # `_Stopped` is an `Exception` subclass but is always
                # returned, never raised, out of `_attempt`, so it is read
                # off first, before the general `BaseException` check below
                # that re-raises anything else.
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
        answers, refused = self._answers(self.workflow.terminal_steps)
        if refused is not None:
            raise _Stopped(self._restamped(refused.outcome))
        return Accepted(
            revision=self.revision,
            evidence=self._evidence_so_far(),
            steps=self._completed_so_far(),
            results=answers,
            anchors=self._anchors_so_far(),
        )

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

    async def _surface_as_compiled(self) -> dict[str, ToolSchema]:
        """Re-read the surface and refuse to run against tools that have changed.

        Each step's tool is compared with what the step was compiled against.
        A workflow with a rolling step also calls `resolve_anchor`, which no
        step names, so that tool is compared with its row of the capability
        table, under the first rolling step's id.
        """
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
    ) -> None:
        """Resolve the newest period the rolling steps of `group` stand on, once.

        Sends `resolve_anchor` the populations the group's steps carry, with
        every column spelled the way the retrieved frames spell it
        (`bound_populations`, the value `_arguments` writes on each of them),
        and the frame of each retrieval of the group. `step` is the group's
        first step about to run: the call reports and is refused under its id.

        Writes `self.anchored` for every step of the group.
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
            self.anchored.update(dict.fromkeys(group.steps, held))
            return
        self._check_fit(step, _RESOLVE_ANCHOR, arguments, schema)
        token = current_step_id.set(step.id)
        try:
            result, _ = await self._call(step.id, _RESOLVE_ANCHOR, arguments)
        finally:
            current_step_id.reset(token)
        if result.is_error:
            raise self._stopped_by_anchor_refusal(step, result)
        try:
            anchor = AnchorResult.model_validate(result.structured_content)
        except ValidationError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"{_RESOLVE_ANCHOR.name!r} answered with no newest period the run can read: {exc}",
                step_id=step.id,
            ) from exc
        anchored = _Anchored(result=anchor, columns=roots)
        self._resolutions[sent] = anchored
        self.anchored.update(dict.fromkeys(group.steps, anchored))

    def _stopped_by_anchor_refusal(self, step: WorkflowStep, result: ToolCallResult) -> _Stopped:
        """The stop for a `resolve_anchor` call the tool refused.

        A refusal that states a fact of the data keeps its code and carries a
        message and a `fix.remedy` written for the user, who reads them in the
        turn's closing text.
        """
        failure = parse_tool_failure(result.structured_content or {})
        if is_anchor_fault(failure):
            return self._anchor_fault(step, failure)
        message = _ANCHOR_REFUSAL_MESSAGES.get(failure.code)
        if message is not None:
            failure = failure.model_copy(
                update={"message": message, "fix": failure.fix | {"remedy": _NAMED_YEARS_REMEDY}}
            )
        return self._step_refused(step.id, _RESOLVE_ANCHOR, result, failure)

    def _anchor_fault(self, step: WorkflowStep, failure: ToolFailure) -> _Stopped:
        """The stop for a refusal saying the run's own newest period is not in order.

        The populations, root frames, key and periods a rolling call carries
        are all the run's own, so the user is told of an internal error and
        the refusal is logged under the trace id the message quotes.
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
                code=INTERNAL_ERROR,
                message=internal_error_message(
                    subject="Working out the latest period", trace_id=trace_id
                ),
                retry_mode="do_not_retry",
                fix={"refusal": failure.model_dump(mode="json")},
            ),
        )

    def _check_rolled_start(self, step: WorkflowStep, bounds: Sequence[RollingBound]) -> None:
        """Refuse `ROLLING_WINDOW_OUTSIDE_DATASET` when `step`'s rolled start is before its data.

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
                    code=ROLLING_WINDOW_OUTSIDE_DATASET,
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
        `self.role_keys[step.id]` and
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
                try:
                    token = current_step_id.set(step.id)
                    try:
                        result, attempts = await self._call(step.id, step.tool, arguments)
                    finally:
                        current_step_id.reset(token)
                    if result.is_error:
                        raise self._stopped_by_refusal(step, result)
                    frame, manifest_ref = self._frame(step, result)
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
                        dropped_key_count=self.dropped_key_counts[step.id],
                    )
                    evidence = (
                        await self._check_evidence(step, result)
                        if step.id in self.evidence_bindings
                        else None
                    )
                    return record, evidence
                finally:
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
        the result declares it left out (`declared_dropped_key_count`).

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
        self.dropped_key_counts[step.id] = declared_dropped_key_count(result.structured_content)
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

        The refusal carries a `FAMILY_NOT_SUMMED` payload, which
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
                    code=FAMILY_NOT_SUMMED,
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
        compiled = {*step.expected_role_keys, *(entity.role for entity in source.entities)}
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

        Every call hands `on_progress`, bound to the step's id, to the
        surface. Each answer is reported as it arrives, so the step's line
        stops showing the server's last wait message once the call returns.
        Each retry reports the attempt it is about to make before its delay
        starts.
        """
        on_progress = partial(self.on_progress, step_id) if self.on_progress is not None else None
        for attempt in range(MAX_ATTEMPTS_PER_STEP):
            last = attempt + 1 == MAX_ATTEMPTS_PER_STEP
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
        reason = result.text if failure.code == MALFORMED_REFUSAL else failure.message
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
        except (ArtifactError, ValueError, ResourceError) as exc:
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
