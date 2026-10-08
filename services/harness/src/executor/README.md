# executor

The executor runs an approved plan's compiled workflow and checks what comes
back. It makes no choice a plan did not state, and it stops with a typed
outcome the moment something is not as approved.

`executor` ships inside `harness` as a sibling top-level module
(`services/harness/src/executor/`), on harness's dependencies, tests, and CI.
Its tests live in `services/harness/tests/executor/`.

## What it does

| Module         | Role                                                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `run.py`       | `execute_approved_plan`, the one public entry point, and the private step loop underneath it: retries, checks and transitions |
| `artifacts.py` | Reads the frame reference a step was compiled to produce out of the tool's structured result                                  |
| `classify.py`  | Decides what a tool's refusal calls for: retry, a choice for the planner, rediscovery, or a stop                              |
| `outcomes.py`  | `Accepted`, `MethodChoiceNeeded`, `Refused`, `RediscoveryNeeded`, and the session-state slot that holds the latest            |

`execute_approved_plan` is a library function for the harness to call once a
plan is `APPROVED`; it moves the plan to `EXECUTING`, runs it, and returns the
plan `COMPLETED` or `FAILED` beside the outcome. Its intended caller is the DFO
Studio graph's execute node, once the studio is adapted to this workflow in a
separate change; no production code calls it yet. The caller stores what comes
back: the finished plan under `planner:plan` with `save_plan`, and the outcome
under `executor:run` with `save_outcome`. Nothing wires the executor into the
planner's turn: the planner and the executor never speak directly, and a run can
take longer than a model turn should.

## The properties that define done

**What runs is what was approved.** `execute_approved_plan` is the one entry
point: a plan that is not `APPROVED` is refused (`not_approved`) untouched,
and one that is starts through `Plan.start_execution`, which refuses a plan
whose content no longer hashes to its `approved_revision` (`stale_revision`).
The step loop underneath, `_run_workflow`, is private -- reachable only once
a plan is `EXECUTING` -- so nothing can run a plan the approval gate never
saw. Before the first call it re-reads the tool surface and refuses to run
(`surface_changed`) if any step's tool has changed since the plan was
compiled: a different argument schema, a different output schema, or a
different declared version.

**Independent steps run together; dependent steps run in binding order.** The
step loop groups the workflow's steps into waves from `WorkflowStep.bindings`.
A step with no unmet frame binding joins the first wave it can, and a step
that binds another step's frame waits for the wave that step is in. Steps in
one wave run concurrently. A stop native to one of them lets the rest of the
wave finish before the run ends, so what it reports as completed names every
step that finished, in the plan's own step order, not the order they happened
to finish in. A stop in one step therefore waits for the rest of its wave. The
failure path can take as long as the slowest sibling's own retry budget, not
just the step that stopped. When more than one step in a wave stops, the
outcome the run raises is the first in the wave's own step order, and its
`also_stopped` names the rest, so a replan can act on every stale source a
wave surfaced at once.

**A wave's concurrency is bounded.** `Settings.executor_max_concurrent_steps`
(default 4) caps how many of one wave's steps may have their tool call in
flight against the MCP servers at once; a wave wider than the bound queues
at the point of call rather than opening every call in the wave together.

**Every argument is checked before every call.** A step's bindings are resolved
from the frames earlier steps produced, each checked as a `FrameRef`, and the
full arguments are validated against the tool's live schema. A step that cannot
be given valid arguments is `binding_invalid`, a compiler fault the run reports
rather than hides.

**Role columns are spelled the way the frame spells them.** The compiler names
a role column by the source's entity role, a flow axis such as `origin` on a
flow dataset; the frame `get_data` registers names it by declared property
(`financingProvider`) and may carry more role columns (`stat_var_*`).
`get_data`'s result reports the frame's `role_keys` and each coverage
dimension's axis, and before each step downstream of a retrieval runs, the
executor rewrites every column the step's arguments name and the role keys it
must declare (`compiler.frame_columns`). A filter must declare every role
column of its input; every other verb the columns its arguments name. A role
the frame carries no column for is `column_unbound`, before the call.

**A step's StatVar count is bounded by the frame it reads.** A family
retrieval holds one StatVar per member the source has data for, which only the
minted frame states. `get_data` and every analysis verb declare their frame's
`stat_var_count`, and a non-empty result must declare at least one StatVar and
at most its verb's bound (`compiler.frame_columns.bound_stat_var_count`):

| Verb                               | Input holding several StatVars                                      | Most StatVars declared |
| ---------------------------------- | ------------------------------------------------------------------- | ---------------------- |
| `filter`                           | runs                                                                | its input's count      |
| `compare`                          | refuses `role_ambiguous` unless each slice keeps one row per period | its left input's count |
| `aggregate` grouping by `stat_var` | runs                                                                | its input's count      |
| `aggregate` otherwise              | refuses `stat_var_unresolved`                                       | 1                      |
| `growth`, `rank`, `share_of`       | refuses `stat_var_unresolved`                                       | 1                      |
| `quotient_of`                      | refuses `stat_var_unresolved`                                       | 1                      |
| `sum_of`                           | runs                                                                | 1                      |
| `join`                             | refuses `stat_var_unresolved`                                       | 2                      |

A count outside the bound is `shape_mismatch`. A `get_data` asked to sum its
family that declares more than one StatVar is one too, and it carries a
`family_not_summed` payload with `do_not_retry`, so the turn closes on a message
and remedy written for the user and spends no replan. A verb that refuses a
several-StatVar input stops the run with its own refusal. An input whose count
is unknown (a `get_data` call that did not read the frame) leaves a `filter`,
`compare` or grouped `aggregate` with the lower bound alone.

**A rolling step's years are written from the data the run retrieved.** A
`growth`, `rank` or `filter` step that states a period against the newest data
carries the tool's `rolling` argument beside the years the plan read from
Search (`compiler.rolling.is_rolling`). The rolling steps that some step reads
together form a group (`compiler.rolling.anchor_groups`), and a group stands on
one newest period, taken over the retrievals its steps rest on. A retrieval no
step of the group reads has no part in it. Every step of a group runs after
every retrieval of the group. Once those are in, the run calls analysis-mcp's
`resolve_anchor` once for the group, with the populations its steps carry,
spelled the way the retrieved frames spell their columns, and the frames of its
retrievals. The answer is the newest year at least half of the series of each
population hold, and the key of its stored resolution. Each step of the group
is then sent that year moved by its offsets, with the key. The outcome carries
each group's answer in `anchors`: the key, the year, and the series that hold
no value at it. The call takes the path a step's call takes, under the id of
the group's first step to run, and `resolve_anchor` is part of the surface
check, compared with its row of the capability table. A plan with no rolling
step never calls it.

| Stop                                                                                    | Reported as                                                                               |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `resolve_anchor` finds no year half of the series hold (`rolling_anchor_unresolved`)    | `step_refused` with the tool's code, a message for the user and a `fix.remedy`            |
| `resolve_anchor` finds periods that are no calendar years (`rolling_grain_unsupported`) | the same                                                                                  |
| A rolled start is before the first year its retrieval's evidence records                | `rolling_window_outside_dataset`, before the step's call, with a payload of the same code |
| Any other `rolling_anchor_unresolved`, from `resolve_anchor` or a rolling verb          | `step_refused` with an `internal_error` payload; the refusal is logged and kept in `fix`  |

Each of these carries a payload that classifies as a stop, so the turn closes
on it and spends no replan.

**Results are read as the values they were declared to be.** A step's frame
reference is read the way its capability says (`Extract`), never from digest
text; a result that does not carry one is `step_result_unusable`.

**Retrieved evidence is checked against the source it was bound to, not
whatever a tool answers with.** Each retrieval named in the workflow's
`evidence_bindings` is checked against the one `SelectedSource` that binding's
`spec_id` names -- the source the step was compiled to retrieve -- never one
looked up by whatever `spec_id` the tool's own answer happens to carry. The
manifest `get_data` returned is read into `RetrievalEvidence` and
`check_evidence` runs it against that bound source: a manifest under another
spec_id or another identity is `source_drift`, the required entities have to
be present, the period covered, each series one observation per year, rows
returned, and its statistics present -- missing per-series statistics, `None`
or a table with no rows alike, always refuse, since a check that cannot run is
not a passed one. Per-series figures live in the manifest's own statistics
artifact: `FetchSeriesStats` (`executor.run.default_fetch_series_stats` by
default) resolves the manifest's `series_count` and statistics-artifact
reference into the Arrow table `evidence_from_manifest` reads column by
column, through `data_mcp.series_statistics.fetch_series_statistics_table`; at
25,000 series that table holds about 8 MiB where one pydantic model a series
held about 180 MiB. A caller whose tool surface allows no real
upstream call, such as `harness.scripted`'s deterministic mode, injects its
own. An error stops the run as `requirement_unmet` with every finding
attached. Under `coverage.on_missing = report_gaps`, missing years are recorded
on the evidence instead of stopping it.

**No in-place refresh: what a tool says can be retried is retried, and what
needs rediscovering goes back to the planner.** A transport failure and a
`retry_after_delay` refusal are retried up to three attempts, waiting the
delay the refusal names or a doubling backoff, capped at thirty seconds -- a
delay that is negative, not finite, or not a number at all is ignored rather
than honoured. Each wait is drawn uniformly between that delay and twice it,
still capped at thirty seconds, so calls refused together retry at different
moments. The Studio graph's direct calls (`harness.mcp_client.McpExecutor`)
retry a `retry_after_delay` refusal on the same policy. A queued mint that
failed on a moved source revision arrives as one (`upstream_error` naming
`upstream_version_changed` as its `upstream_code`), so it is retried in place.
Neither a bare retry fixes: a synchronous `upstream_version_changed` says
`refresh_then_retry` on the wire, and `definition_not_held` says
`correct_and_retry`, but the approved `spec_id` has already left the store, so
retrying reaches the same absence. `open_role_unavailable` is
`correct_and_retry` too, yet Search refuses a re-resolved identity that keeps
an open role the StatVars it elects declare no role on, so the same
identity refuses again and the source must be searched for afresh. All three
come straight back as `RediscoveryNeeded`, naming the requirement and source
that need discovering again -- a new discovery under a new plan revision,
which only the translator can do. A rediscovery for a step carrying no evidence binding names nothing to
rediscover, so it is `step_refused` instead.

**A retrieval step runs under one absolute budget.** A step whose tool
capability pins `call_timeout_s` (`get_data`, among the workflow's tools) runs
every attempt and every retry delay under one deadline, `step_budget_s`: three
calls at the full timeout, the longest delay before each retry, and a
30-second evidence allowance on top, 630 seconds for `get_data`. Each call's
timeout is clipped to what remains. The evidence check's own statistics fetch
runs under a deadline of its own -- what's left of the step budget once the
check starts, capped at that 30-second allowance -- so a fetch that outruns
it degrades the requirement to `statistics_unavailable` instead of spending
the rest of the step's budget. A step that runs past its own budget is
`transport_failed`.

A `correct_and_retry` refusal is never corrected here either, and
`classify.py` places every code both servers can send. Fourteen whose `fix` is a
set of alternatives -- which unit, which facet, which entity, which window,
which denominator -- are `MethodChoiceNeeded`, with the candidates attached
for the planner to settle with the user. Nine more are stops **because the
compiler already rules them out**: it writes every column list from the
source's own roles, validates each argument against the live schema, checks
the capability implements the kind, reports a join with no common key itself,
and writes the predicates. One of those arriving is a compiler bug, not a
choice for the user. `quotient_of_side_unit_unsupported` is a stop of the other
kind. Its `fix` names the side and why it has no unit to compose, and dividing
something else is a different plan. Every remaining refusal is `step_refused`. A stop keeps
the refusal's own `fix` on `Refused.payload` either way -- what it withholds
is an action, not the alternatives.

**A run is accepted only when everything holds.** `Accepted` carries an
`EvidenceRecord` per requirement -- the frame, the source, the measures and
window found, the citation, and any gaps the requirement accepted -- and a
`StepRecord` per step, so an answer can say what it rests on.

## What waits on the analysis service

Analysis verbs return only `{spec_id, content_version}` as structured content,
so a derived result's unit, window and coverage cannot be checked here yet; the
verbs' own typed refusals are the guard. When analysis-mcp adds a structured
result summary, `check_evidence` gains a second adapter and `evidence_bindings`
can name terminal steps too.
