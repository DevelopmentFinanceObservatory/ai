# `execute` in the background on DBOS

The approved plan's run can be a DBOS 3.4.0 workflow that outlives the client's stream and a killed process, and the next request can read its outcome without running a finished step again. All 12 tests pass, three runs in a row, 45 to 50 s each. Every control fails the way its line predicts. Two findings decide whether this ships. Studio's own run ledger refuses every write once the stream that started the run has ended, and the next turn's carry-on runs the whole plan again outside DBOS.

The node is dev's `execute_node` with two names in `harness.graph.execute` swapped and nothing else changed (`durable_execute.py`). `execute_approved_plan` becomes `run_in_background`, which starts, finds or resumes the workflow `execute_plan` under the id `execute:{turn session id}:{plan revision}` and relays its progress into dev's `_ReportRelay`. `_stop` becomes `stop`, which cancels the workflow on Studio's Stop and leaves it running on a disconnect. The workflow calls dev's own `execute_approved_plan` against the node's ledger, publisher and surface. Each tool call is one DBOS step (`execute_tool_call`), the finest unit the executor allows: `DurableSurface` wraps the bare surface under Studio's `_RecordingSurface`, and a retried call is one step per attempt. Ledger writes, statistics fetches, label lookups and publication are workflow code and run again on replay.

Every test drives real Studio routes on dev's root workflow with dev's scripted orchestrator, dev's `ScriptedPlannerModel`, the canned two-step plan (`get_data` then `aggregate`) and `SlowSurface`, a `ScriptedSurface` that logs, sends two progress reports per call, and can be slowed or held per tool.

## Results

| Line                 | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Control, and how it fails                                                                                                                                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| X1 the client leaves | The client closes the stream once `aggregate` has reported. The run finishes `completed` with no client, and each tool runs once. The next turn's `seed` carries the run on from its recorded outcome, describes its frames through `plan_done`'s `HandBack`, and the orchestrator's request opens with "Calculations an earlier turn left unfinished were carried on" and the run `completed`. The run is then recorded delivered.                                                                               | Dev's node: `the run did not finish with no client: ('failed', [..., 'aggregate start', 'aggregate cancelled'])`. The next turn hands nothing back.                                                                                                                                                |
| X2 a kill mid-run    | Process 1 (executor `p1`) is killed with SIGKILL inside `aggregate`. Process 2 (executor `p2`) posts the same approval again, which is Studio's retry for a resume whose process died. The re-entered node finds the workflow `PENDING` under `p1`, resumes it, and the run finishes from its last completed step. Tool calls across both processes: `p1 get_data`, `p1 aggregate` (killed), `p2 aggregate`. The `plan_done` node hands back the frames, the run and the plan record, and the stream ends `done`. | Dev's node through the same two processes: `a finished step ran again: ['p1 get_data start', 'p1 aggregate start', 'p2 get_data start', 'p2 aggregate start']`.                                                                                                                                    |
| X3 live progress     | Every progress report reached the client 0.000 to 0.002 s after the surface sent it, in both arms. The `activity.step` frames (status, label, tool, detail) equal dev's, frame for frame.                                                                                                                                                                                                                                                                                                                         | `BUFFERED`, where progress rides in the workflow's result: no surface report reaches the client at all, because dev's relay keeps only the latest report per step and the executor's own "Result received; checking it" overwrites them at the end.                                                |
| X4 stop              | Stop while `get_data` is held. The `/cancel` route answers 204, the stream ends `cancelled`, `get_data` is cancelled mid-call, `aggregate` never starts, the workflow is `CANCELLED`, the ledger records the run `failed` `cancelled` as dev does, and the next turn reaches its plan card.                                                                                                                                                                                                                       | `LOCAL_CANCEL`, which cancels the local task only: `workflow left PENDING`. Relaunching DBOS as the same executor then recovers it and runs `get_data` and `aggregate` with nobody listening. Since the stopped run is failed, the recovered run is admitted as a second ledger run and completes. |
| X5 records           | Ledger books (runs, keys, submissions, retrievals, plans), published frames and the `plan_done` hand-back equal dev's once times and run ids are removed. The run is `completed` with 2 attempts, 1 published frame, and hand-back keys `frames`, `plan_record`, `result`, `run`, `step_summaries` and `totals`.                                                                                                                                                                                                  | Setting one attempt number to 2 in the background arm's copy is reported as the single difference `.ledger.<scope>.runs[0].attempts[0].attempt: 2 != 1`.                                                                                                                                           |

A step's `complete` frame reached the client about 2.0 s after `get_data` ended in both arms. Dev's translator drains the recorder on the node's next non-progress event, so this lag belongs to dev and is not held to the 1 s line.

### What reruns after the kill (X2)

- The `aggregate` call in flight at the kill runs again, from the top.
- `get_data` returns its recorded step result. The recorder in process 2 still writes an Activity record for it, so the resumed stream shows `get_data` running and complete with no call made.
- Ledger writes run again. The run gains `resumes: 1` and two more attempts, `retrieve` 2 and `totals` 2, although `get_data` was never called a second time.
- The executor's admission, statistics fetch (scripted here) and publication run in process 2 as workflow code.

### Probe: X1 on Studio's ledger

With `StudioRunLedger`, the ledger `harness.server` wires, both steps finish after the disconnect. The next ledger write then raises `SegmentLost`, because every write is fenced on the conversation's live segment and the stream released it. The workflow returns `ledger_unavailable` with the run left `running`. The next turn's carry-on (`harness.graph.resume.run_resumer`) then drives the run itself, outside DBOS, and calls `get_data` and `aggregate` again. The run ends `completed` and delivered, at the cost of the whole plan twice. X1 passes on the in-memory ledger dev's tests use and fails on the production one until the ledger has a fence the run owns.

### A stale frame in the next turn

In X1, the next turn's stream opens with `activity.step complete aggregate`. The background run's last call is recorded under the conversation after the old stream drained the recorder, and the next turn's translator drains it.

## A request while the run is going on another instance

The node handles two cases. The next turn reaches the run through `seed`'s carry-on, and `executor.run._carry_out` waits only for a run driven in its own process (`_IN_FLIGHT`). On another instance it finds the run `running` and drives it again in parallel. A request that re-enters `execute` sees the workflow `PENDING` under another executor and resumes it. It cannot tell a live owner from a dead one, so it takes over a live run: DBOS's `owner_xid` fence refuses the old owner's next step write, but the step in flight there finishes first. Progress is process-local either way. A resumed workflow goes on DBOS's queue, which any instance may claim, so its reports can land where nobody listens.

A production version needs three things, none built here:

- An owner check before takeover. DBOS Conductor, or a heartbeat row per executor, lets the node resume only workflows whose executor is dead and otherwise poll `get_workflow_status` until the run ends.
- Progress other instances can read. `DBOS.write_stream_async` and `read_stream_async` exist in 3.4.0 and cost one database row per report. Postgres `LISTEN`/`NOTIFY` keeps reports out of the workflow's record.
- Resumed runs on the instance that waits. A DBOS queue per instance, with the node enqueuing the takeover on its own queue, keeps progress and the stream together.

## What this relies on

- Harness, private: the module globals `harness.graph.workflow.execute_node`, `harness.graph.execute.execute_approved_plan` and `harness.graph.execute._stop`, swapped by `install`; `ConversationState.cancel_reason` read through `StudioStore.current` to tell Stop from a disconnect; the contextvars `current_conversation_id` and `current_conversation_key`, set inside the workflow because a resumed workflow runs with none; and `InMemoryRunLedger._books` and `_Book`, which `FileRunLedger` and the X5 snapshot read.
- Harness behaviour: Studio's retry of the same answer after a crash (`resuming_segment`) re-enters `approve` and then `execute` with the same turn session, so the workflow id repeats. A disconnect closes the turn, so after one, only `seed`'s carry-on reaches the run.
- ADK: `ctx.session.id` as the turn's id. Nothing private.
- DBOS: an inline call under `SetWorkflowID` runs in the caller's task, so Stop can cancel the call in flight. `start_workflow_async` shields its task and could not. A cancelled inline workflow stays `PENDING` until `cancel_workflow_async`. `resume_workflow_async` takes over a `PENDING` workflow of another executor, the same undocumented use the planner prototype in `prototype-node-stream/dbos/` made. A resumed workflow runs on DBOS's own event-loop thread, so the recorder, ledger and publisher are called from that thread; the file ledger tolerated it, and `StudioRunLedger` over async SQLAlchemy was not tried there. `DBOS.workflow_id` tells `DurableSurface` whether it runs inside a workflow.
- Pydantic AI: nothing.

## What a production version would change

- `harness/graph/execute.py`, about 80 lines: the workflow, start-or-find-or-resume, a cross-thread progress listener, and the stop policy. Studio should tell the node it was stopped, about 10 lines in `studio.py`, so the node need not read `cancel_reason`.
- A durable tool surface, about 30 lines, beside `compiler.surface`.
- `harness/run_ledger.py` and `studio_state.py`, about 60 to 100 lines: a fence the run owns, such as a lease per run renewed by the workflow, so writes after the stream ends land. Without it X1 fails in production, as the probe shows.
- `harness/graph/resume.py`, about 30 lines: carry a run on by awaiting or resuming its workflow, so the next turn never drives the run a second time.
- `harness/studio_translate.py`, about 10 lines: drop a background run's Activity records once the stream that started it has ended.
- `executor/run.py`, about 15 lines, or an accepted cost: replayed calls record attempts as if made.
- Wave order. DBOS numbers steps in the order they start, and every tool call shares the step name `execute_tool_call`. `_Run._run_wave` gathers a wave under a semaphore of `max_concurrent_steps`, so in a wide wave the order calls start in follows the order earlier calls finish. A replay could then hand one step's recorded result to another, and DBOS's name check would not notice. Steps need a deterministic start order, or results looked up by plan step id. The canned plan has one step per wave, so this was read from the code and not run.
- Deploy: a shared Postgres system database, one executor id per instance, and a new `application_version` whenever the executor's step order changes.

## Not covered

- Plans with several steps per wave, retries, publication faults (`pending` runs) and replans after a background run.
- A kill between the workflow's end and the node's output, or during publication.
- Several instances. The hazards above come from reading the source.
- A next request that arrives while the background run is still going. The X1 test waits for the run to end first.
- A served check against uvicorn, as against the in-process transports.

## Files

| File                            | Purpose                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `durable_execute.py`            | The workflow, the tool step and `DurableSurface`, `run_in_background`, `stop`, `background_execute_node`, `install`, `launch` |
| `fakes.py`                      | `SlowSurface`, `FileRunLedger`, `RecordingPublisher`, `ledger_snapshot`                                                       |
| `studio_app.py`                 | The Studio app on either arm                                                                                                  |
| `checks.py`                     | X1, X3, X4 and X5, each raising `AssertionError` when its line fails, and the two observations                                |
| `test_background.py`            | The in-process tests                                                                                                          |
| `kill_child.py`, `test_kill.py` | X2: one Studio request per child process over a shared SQLite session database, ledger file and DBOS system database          |

The streaming transport and timed reader come from `prototype-node-stream/studio/wiring.py`, and `conftest.py` is that directory's.

## Commands

From this directory:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest . -q -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest . -q -s -p no:cacheprovider
```

The second prints each check's numbers and each control's failure. Lint, from the worktree root:

```
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-background-execute
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-background-execute
```
