# DBOS durability for the Pydantic AI planner inside the ADK node

Stage A's three-node ADK 2.13.0 workflow, with a `planner` node that runs each planner segment as a DBOS 3.4.0 workflow through Pydantic AI 2.64.0's `DBOSDurability` capability. A segment runs from the brief, or from the stored `Conversation` plus the user's answer, to the next `DeferredToolRequests` or the final text. The pause still lives in ADK exactly as in stage A: the conversation goes into session state on the interrupt event, and no DBOS workflow stays open across a pause. DBOS's system database is a SQLite file beside the ADK session file. The planner's script and tools are stage A's, reused from `../minimal`. No model calls.

## Result

DBOS removes the one repeated model call stage A showed after a kill, and keeps live progress, the lossless pause, stop and failure. All 15 tests pass (6 checks, 4 controls, 5 measurements), three runs in a row, about 70 s each.

| Line                            | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Control, and how it fails                                                                                                                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1 a kill repeats no model call | three processes as in stage A's N4; no model call runs twice; plan hash `fe46f480...e1883` and the planner history equal the bare-agent reference                                                                                                                                                                                                                                                                                                      | stage A's `Mode.LIVE` node through the same driver: `model called again: ['model call after 5 messages']`                                                                                                                                                               |
| D2 live progress                | first progress reaches the `run_async` consumer 0.04 s after the turn starts while `propose_plan` waits on the client's signal; 40 progress events in order, all before the interrupt, read by a client sleeping 5 ms per event                                                                                                                                                                                                                        | `NO_HANDLER`: no durability handler, progress buffered into the workflow's output and relayed after it returns: `TimeoutError` at 2 s                                                                                                                                   |
| D3 re-entry finds the workflow  | process 3, executor `p3`, finds workflow `<invocation_id>:1` PENDING under `p2`, resumes it, and it continues from its last recorded step: the model call after 5 messages ran once, in process 2; the call after 7 ran once, in process 3. With all three processes as executor `local`, launch recovery re-enqueues the workflow first, the node only awaits it (`awaited in status ENQUEUED`), and every model call still runs once per workflow id | `RANDOM_ID`: a fresh workflow id per entry starts a new segment from the stored conversation: `model called again: ['model call after 5 messages']`                                                                                                                     |
| D4 stop                         | cancelling the consumer cancels the planner (`propose_plan call=1 cancelled`), the node then cancels the workflow (status `CANCELLED`); no task left, nothing unretrieved; the next turn runs to the ask with progress `(1,1) (1,2) (1,3)`                                                                                                                                                                                                             | `LOCAL_CANCEL`: cancels the local task only: `workflow left PENDING`. On the next launch DBOS recovers it and runs it to `SUCCESS` with nobody listening: `propose_plan` call 1 again, handler steps for `(1,2)` and `(1,3)`, `model call after 3 messages`, `ask_user` |
| D5 planner-side failure         | `propose_plan` raising gives the error event `PlannerToolError` and `run_async` raises `PlannerToolError: propose_plan call 1 failed`, identical to stage A's live node, within 5 s; the workflow is `ERROR`, stays `ERROR` across a relaunch and logs nothing more                                                                                                                                                                                    | `QUEUE_ONLY`: waits on the progress queue alone and never sees the failed task: `TimeoutError` at 5 s                                                                                                                                                                   |

### What a Stop button gets

- A model call in flight when the consumer is cancelled does not complete. The node cancels the task running the workflow inline, the cancellation reaches the model step's coroutine, and `slow model end` is never logged.
- `DBOSClient.cancel_workflow` from a second process stops a segment at its next step start. Inside `propose_plan`, which runs a handler step per progress event 25 ms apart, the node's stream ended 0.05 to 0.08 s after the cancel committed. Inside a 6 s model call it ended 5.3 s after, when the call finished: DBOS does not interrupt a running step by default, and the call's result is then refused (`no longer owned by this execution`). The consumer sees an error event and the exception `DBOSAwaitedWorkflowCancelledError`.

## What repeats after a kill, against stage A's N4

Process 1 starts the turn and pauses at the ask. Process 2 answers, runs `propose_plan` call 2 to its first progress event and parks; the parent kills it there. Process 3 sends the same answer again under another executor id, then approves.

| Stage A, `Mode.LIVE`                                          | DBOS node                                                                                                                            |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| the model call after 5 messages runs twice                    | runs once; process 3 replays its recorded step                                                                                       |
| the body of `propose_plan` call 2: `start` twice, `end` once  | the same: a plain function tool runs in workflow code, so the body reruns from the top                                               |
| progress `(2,1)` reaches the client twice and is stored twice | reaches the client once (process 2) and is stored once; process 3 sends `(2,2) (2,3)`. Stored: `(1,1) (1,2) (1,3) (2,1) (2,2) (2,3)` |
| no handler                                                    | the handler for `(2,1)` ran once: its step was recorded in process 2, so the rerun body's `emit` replays it instead of running it    |
| two function responses to `planner_ask:ask-1` in the session  | the same; the ADK side is unchanged                                                                                                  |
| nothing from before the pause repeats                         | the same                                                                                                                             |

The handler result is recorded only after the handler returns. In every run the step committed before the kill, because the kill follows the client reading the event, which follows the session append. A kill between the handler's hand-off and its commit would send that progress event a second time.

## Answering the brief's question on `ctx.emit`

Events a plain function tool emits with `ctx.emit` do reach `DBOSDurability(event_stream_handler=...)`. Each takes a handler step at the moment of emission and the handler runs without blocking the tool, which is why D2 holds while `propose_plan` waits on the client. The handler also receives the tool call and result events, each in its own step.

## The workflow id

`f"{invocation_id}:{segment}"`, with `segment` 0 for the run from the brief and the stored pause's segment plus one after each answer. ADK keeps the invocation id across a resume, so a node re-entered after a crash computes the same id and asks DBOS for the same workflow. A new turn gets a new invocation and so new ids, which is what lets the next turn run after a stop. The segment number is stored with the conversation in session state.

## What the node relies on in DBOS and Pydantic AI

- Calling the decorated workflow under `SetWorkflowID` runs it inline in the caller's task. `DBOS.start_workflow_async` instead wraps its task in `asyncio.shield` (`dbos/_core.py`, `start_workflow_async`), so cancelling the node would not cancel the planner. The node runs the inline call in a task of its own and cancels it on exit.
- A cancelled inline workflow is left `PENDING`: DBOS logs "Asyncio task cancelled" and records nothing. The node must call `DBOS.cancel_workflow_async`, or the next launch reruns the segment, as the D4 control shows.
- Starting a workflow whose id is already `PENDING` from another process does not run it. `_insert_workflow_status` (`dbos/_sys_db.py`) sees another `creator_xid` and returns a polling handle, which waits forever on a dead owner. The node therefore checks the status first and calls `DBOS.resume_workflow_async` when the owner is another executor. `resume_workflow` is public, but taking over a `PENDING` workflow from a dead executor is not a documented recipe: it relies on `resume_workflows` re-enqueuing any row not `SUCCESS` or `ERROR`, and on the `owner_xid` fence refusing a stale owner's later step writes. The node cannot tell a dead owner from a live one; a live one would finish its in-flight step, including a model call, before the fence refuses it.
- A resumed or recovered workflow runs on DBOS's own event loop thread, not the request's loop (`execute_dequeued_workflow`). The sink that carries progress to the node is thread-safe and buffers what arrives before the node attaches. The node reads the result through a polling handle at 0.1 s.
- Launch recovery re-enqueues rather than executes (`_recovery.py`, `_recover_workflow`), so a node that finds its workflow `ENQUEUED` or `PENDING` under its own executor only awaits it, and the workflow runs once.
- Pydantic AI reserves handler steps with the private `dbos._context.snapshot_step_context` (a TODO in `_durability.py` points at dbos-transact-py issue 915).
- The model's name is part of the recorded history: the reused script must keep stage A's `model_name`, or the history check fails on `model_id`.

## What constrains a deploy

- Step order. In these runs a segment records `planner__model.request_stream`, then one `planner__event_stream_handler` step per tool event in emission order. Changing what a tool emits, or adding a durable call to a tool, changes that order, and an in-flight workflow then fails recovery with `DBOSUnexpectedStepError`. Ship such a change as a new `application_version`.
- `application_version` is set explicitly (`planner-1`). Launch recovery picks only the executor's `PENDING` workflows of its own version. Whether a resumed workflow of an older version is dequeued by a newer process was not tested.
- Names. The workflow name `planner_segment` and the agent names `planner` and `planner_buffered` are in every recorded step name. Renaming either strands in-flight workflows.
- Executor ids. Each instance needs its own. With a shared id such as the default `local`, a new instance's launch recovery claims every `PENDING` segment of that id, including ones a live instance is still running. With unique ids, a dead instance's segments are picked up only when the user's next request re-enters the node, or by DBOS Conductor.
- Progress is process-local. `resume_workflow` puts the segment on DBOS's internal queue, and with several instances any of them may claim it. The node would then get the result through polling but none of the progress, which would sit in the other instance's sink. A deploy needs the resumed segment to run where the node waits, for example on a queue only that instance listens to; this was not built.
- SQLite is per machine. Cloud Run needs a shared Postgres system database.
- Size. Each segment's input carries the whole conversation JSON and its output carries it again, besides the copy in ADK session state. DBOS's guidance is to stay under about 2 MB per field.

## Files

| File                                    | Purpose                                                                                                                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `durable_planner.py`                    | `DurableDeps` (JSON log lines with pid and workflow id, optional model delay), the two durable agents, the progress handler and sinks, the `planner_segment` workflow, `launch` |
| `durable_graph.py`                      | The `planner` node in `DMode.DURABLE` and its four control modes, `open_segment`, `build_runner`                                                                                |
| `durable_checks.py`                     | The checks and measurements, each taking the `mode` its control uses                                                                                                            |
| `durable_driver.py`, `durable_procs.py` | One turn's steps as a child process with its own DBOS executor, and the runner that kills it on a chosen line; the cross-process cancel                                         |
| `test_durable.py`                       | The 15 tests                                                                                                                                                                    |

## Commands

From this directory:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest . -q -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest . -q -s -p no:cacheprovider -k "kill or cancel or abandoned or tool_failure"
```

The second prints what repeats, the stop timings and the failure. Lint, from the worktree root:

```
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-node-stream/dbos
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-node-stream/dbos
```

## Limits of this evidence

- One process per DBOS system database file at a time, and one instance. The multi-instance hazards above are read from DBOS's source, not run.
- The kill lands inside `propose_plan` after a progress event. A kill inside the model step, between the workflow's end and the ADK pause, or inside `approve` was not run. The status check in `open_segment` returns a finished segment's recorded output, which should cover the second case.
- DBOS's `preemptible` step option, which would let a cancel abort an in-flight model call, is not in Pydantic AI's `StepConfig` and was not tried.
