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
- `_answer_frame` refuses `EMPTY_ANSWER` when a terminal step's own
  capability says an empty result is not an answer (`empty_is_an_answer`,
  `True` on `filter` alone) and the call declared zero rows, and refuses
  `STEP_RESULT_UNUSABLE` when the tool declares an output schema but the
  call carried no row count -- the schema promises the field, so its absence
  is a broken contract, not an unknown.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from functools import partial

import httpx2
import pyarrow as pa
from common.materialisation import FrameRef
from compiler.arguments import literal_argument_errors
from compiler.capabilities import CAPABILITIES, ToolCapability
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
    bound_role_keys,
    bound_stat_var_count,
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
from fastmcp.exceptions import ResourceError
from planner.plan import DatasetRequirement, Plan, PlanStatus, plan_revision
from planner.workflow import ArtifactKind, EvidenceBinding, SelectedSource, Workflow, WorkflowStep
from pydantic import JsonValue, ValidationError

from executor.artifacts import (
    ArtifactError,
    declared_shape,
    extract_frame_ref,
    parse_manifest,
    retrieved_frame_columns,
)
from executor.classify import (
    BACKOFF_SECONDS,
    FAMILY_NOT_SUMMED,
    MALFORMED_REFUSAL,
    MAX_ATTEMPTS_PER_STEP,
    MAX_RETRY_AFTER_SECONDS,
    Disposition,
    Jitter,
    ToolFailure,
    candidates_of,
    classify_refusal,
    delay_before_retry,
    jittered_retry_delay,
    parse_tool_failure,
)
from executor.outcomes import (
    Accepted,
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
# harness layer. `harness.graph`'s `fetch` node sets it too, for the one
# call it makes outside any plan.
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


def _waves(workflow: Workflow) -> list[tuple[WorkflowStep, ...]]:
    """Group `workflow`'s steps into the dependency levels their bindings state.

    Level 0 holds every step with no frame binding; level `n` holds every
    step whose bindings all name a step in a level below `n`. Steps that
    share a level carry no dependency between them and are safe to run
    concurrently; within a level, steps keep the order `workflow.steps`
    gives them.
    """
    steps_by_id = {step.id: step for step in workflow.steps}
    level_of: dict[str, int] = {}

    def level(step_id: str) -> int:
        if step_id in level_of:
            return level_of[step_id]
        step = steps_by_id.get(step_id)
        assert step is not None, (
            f"a binding names step {step_id!r}, which the workflow does not hold"
        )
        computed = (
            0
            if not step.bindings
            else 1 + max(level(binding.from_step) for binding in step.bindings.values())
        )
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
        # The source each evidence-bearing step is checked against: bound by
        # `EvidenceBinding.spec_id`, the source the step was compiled to
        # retrieve -- never looked up by whatever `spec_id` the tool answers
        # with, which is what `evidence.spec_id` could be instead.
        self.approved_source: dict[str, SelectedSource] = {
            b.step_id: self.sources[b.spec_id] for b in workflow.evidence_bindings
        }
        self.steps_by_id: dict[str, WorkflowStep] = {s.id: s for s in workflow.steps}
        self.frames: dict[str, FrameRef] = {}
        self.row_counts: dict[str, int | None] = {}
        self.stat_var_counts: dict[str, int | None] = {}
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

    async def execute(self) -> Accepted:
        schemas = await self._surface_as_compiled()
        for wave in _waves(self.workflow):
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
                # finished in this wave and in every wave before it --
                # re-stamp the outcome, and every sibling stop, with that
                # rather than whatever each saw at the moment it stopped.
                completed = self._completed_so_far()
                raise _Stopped(
                    stopped.outcome.model_copy(update={"completed": completed}),
                    siblings=tuple(
                        sibling.model_copy(update={"completed": completed})
                        for sibling in other_stops
                    ),
                )
        return Accepted(
            revision=self.revision,
            evidence=self._evidence_so_far(),
            steps=self._completed_so_far(),
            results={
                step_id: self._answer_frame(step_id) for step_id in self.workflow.terminal_steps
            },
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

    def _answer_frame(self, step_id: str) -> FrameRef:
        """The frame a terminal step produced, refusing when it produced none
        or when its own declared row count says it should not stand as an
        answer.
        """
        frame = self.frames.get(step_id)
        if frame is None:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"terminal step {step_id!r} produced no frame to answer with",
                step_id=step_id,
            )
        step = self.steps_by_id[step_id]
        row_count = self.row_counts.get(step_id)
        if step.output_schema_hash is not None and row_count is None:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE,
                f"{step.tool.name!r} declares an output schema promising a row count, but "
                f"terminal step {step_id!r}'s result carried none",
                step_id=step_id,
            )
        if row_count == 0:
            capability = self.capabilities.get(step.tool)
            if capability is None or not capability.empty_is_an_answer:
                raise self._refused(
                    RefusalKind.EMPTY_ANSWER,
                    f"terminal step {step_id!r} matched no rows, and {step.tool.name!r} is not "
                    "one whose empty result is itself an answer",
                    step_id=step_id,
                )
        return frame

    # -- before the first call ------------------------------------------------------

    async def _surface_as_compiled(self) -> dict[str, ToolSchema]:
        """Re-read the surface and refuse to run against tools that have changed."""
        try:
            schemas = {schema.name: schema for schema in await self.surface.list_tools()}
        except ToolSurfaceUnavailableError as exc:
            raise self._refused(RefusalKind.TRANSPORT_FAILED, str(exc)) from exc
        for step in self.workflow.steps:
            schema = schemas.get(step.tool.name)
            if schema is None or schema.server != step.tool.server:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"{step.tool.server} no longer serves {step.tool.name!r}",
                    step_id=step.id,
                )
            try:
                input_hash = schema.input_schema_hash
                output_hash = schema.output_schema_hash
            except UnhashableSchemaError as exc:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"{step.tool.name!r} now declares a schema that cannot be compared with the "
                    f"one the plan was compiled against: {exc}",
                    step_id=step.id,
                ) from exc
            if input_hash != step.input_schema_hash:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"{step.tool.name!r} takes different arguments than the plan was compiled "
                    "against; the plan needs compiling and approving again",
                    step_id=step.id,
                )
            if output_hash != step.output_schema_hash:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"{step.tool.name!r} declares a different output schema than the plan was "
                    "compiled against; the plan needs compiling and approving again",
                    step_id=step.id,
                )
            if schema.version != step.implementation_version:
                raise self._refused(
                    RefusalKind.SURFACE_CHANGED,
                    f"{step.tool.name!r} is at version {schema.version!r}, not the "
                    f"{step.implementation_version!r} the plan was compiled against",
                    step_id=step.id,
                )
        return schemas

    # -- one step -----------------------------------------------------------------------

    async def _run_step(
        self, step: WorkflowStep, schema: ToolSchema
    ) -> tuple[StepRecord, EvidenceRecord | None]:
        """Call `step`'s tool and return what it produced.

        Writes `self.frames[step.id]`, `self.row_counts[step.id]`,
        `self.stat_var_counts[step.id]`, `self.role_keys[step.id]` and
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
                        result, attempts = await self._call(step, arguments)
                    finally:
                        current_step_id.reset(token)
                    if result.is_error:
                        raise self._stopped_by_refusal(step, result)
                    frame = self._frame(step, result)
                    self._learn_columns(step, result)
                    self._check_declared_shape(step, result, arguments, expected_role_keys)
                    self.frames[step.id] = frame
                    record = StepRecord(
                        step_id=step.id,
                        tool=step.tool,
                        attempts=attempts,
                        frame_ref=frame,
                        digest=result.text,
                        row_count=self.row_counts[step.id],
                        role_keys=self.role_keys[step.id],
                        stat_var_count=self.stat_var_counts[step.id],
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
        declared row count, role keys and StatVar count for `_answer_frame`,
        the steps reading this one and the step's `StepRecord`.

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
        (`compiler.frame_columns.bound_arguments`)."""
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
        try:
            arguments = bound_arguments(
                step.tool,
                arguments,
                frames={
                    parameter: self.columns.get(binding.from_step)
                    for parameter, binding in step.bindings.items()
                },
            )
        except UnboundColumnError as exc:
            raise self._column_unbound(step, exc) from exc
        problems = literal_argument_errors(arguments=arguments, schema=schema.parameters)
        if problems:
            raise self._refused(
                RefusalKind.BINDING_INVALID,
                f"step {step.id!r}'s arguments do not fit {step.tool.name!r}: {problems!r}",
                step_id=step.id,
            )
        return arguments

    async def _call(
        self, step: WorkflowStep, arguments: Mapping[str, JsonValue]
    ) -> tuple[ToolCallResult, int]:
        """Call the step's tool, retrying only what the tool or the transport says can be retried.

        Every call hands `on_progress`, bound to this step's id, to the
        surface. Each answer is reported as it arrives, so the step's line
        stops showing the server's last wait message once the call returns.
        Each retry reports the attempt it is about to make before its delay
        starts.
        """
        on_progress = partial(self.on_progress, step.id) if self.on_progress is not None else None
        for attempt in range(MAX_ATTEMPTS_PER_STEP):
            last = attempt + 1 == MAX_ATTEMPTS_PER_STEP
            try:
                result = await self.surface.call_tool(
                    tool=step.tool, arguments=arguments, on_progress=on_progress
                )
            except ToolSurfaceUnavailableError as exc:
                capability = self.capabilities.get(step.tool)
                final = last or (
                    capability is not None and not capability.retry_on_transport_failure
                )
                if final:
                    raise self._refused(
                        RefusalKind.TRANSPORT_FAILED, str(exc), step_id=step.id
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
                )
            )
        # A rediscovery on a step bound to no evidence would name nothing to
        # rediscover, so it is a plain stop like any other refusal. The
        # detail reaches the planner's replan prompt. A refusal's text ends in
        # its JSON envelope (`common.refusal.refusal_text`), so the detail
        # takes the structured `message`. A result with no refusal payload,
        # such as a tool that raised, has only its text, which carries no
        # envelope.
        reason = result.text if failure.code == MALFORMED_REFUSAL else failure.message
        return self._refused(
            RefusalKind.STEP_REFUSED,
            f"{step.tool.name!r} refused: {reason or failure.code}",
            step_id=step.id,
            payload=failure,
        )

    def _frame(self, step: WorkflowStep, result: ToolCallResult) -> FrameRef:
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
            return extract_frame_ref(output.extract, result.structured_content)
        except ArtifactError as exc:
            raise self._refused(
                RefusalKind.STEP_RESULT_UNUSABLE, str(exc), step_id=step.id
            ) from exc

    async def _check_evidence(self, step: WorkflowStep, result: ToolCallResult) -> EvidenceRecord:
        requirement = self.requirements[self.evidence_bindings[step.id].requirement_id]
        # An evidence binding only ever names a retrieval, and only a dataset
        # requirement is ever carried out by one (`planner.proof.mismatches`
        # rejects any other pairing before a plan can be approved).
        assert isinstance(requirement, DatasetRequirement)
        try:
            manifest = parse_manifest(result.structured_content or {})
            series_stats = await self.fetch_series_stats(manifest)
            evidence = evidence_from_manifest(manifest, series_stats=series_stats)
        except (ArtifactError, ValueError) as exc:
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
            )
        )
