# The Pydantic AI planner inside an ADK workflow node: results

A Pydantic AI planner can run inside ADK's `planner` workflow node and stream its progress live. ADK keeps owning the turn. This unblocks live planning progress, a committed feature that ADK blocks today. A third stage adds DBOS checkpoints within each planner segment, so a crash repeats no model call. The prototype ran on dev at 190c7a9c, with google-adk 2.13.0, pydantic-ai 2.64.0 and a scripted planner. Its code is in `prototype-node-stream/`.

## Why dev cannot stream planning progress

Dev's `planner_node` (`harness/graph/planning.py`) awaits the ADK planner directly. The planner's ask pauses the run by raising `NodeInterruptedError` out of `ctx.run_node`, and that exception must not be caught inside a task. So the node queues every progress report until planning ends, and drops the queue when the planner asks the user something.

A Pydantic AI pause is a return value (`DeferredToolRequests`), not an exception. So the node can iterate `agent.run_stream_events` and yield each progress report as its own ADK event while the planner runs in Pydantic AI's background task. When the planner asks, the node turns the deferred call into an ADK `RequestInput` interrupt. The node imports nothing private from ADK.

## Stage A: a minimal ADK 2.13 workflow (`minimal/`)

The minimal workflow has three nodes, run by a real `Runner` over `DatabaseSessionService` on SQLite. Each check has a control built to fail. All 12 tests pass, and every control fails the way its check predicts.

| Check                           | Result                                                                                                                                                        | Control                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| N1 live progress                | First event reaches the consumer 0.06 s after the turn starts, while `propose_plan` is still waiting                                                          | Dev's queue-then-relay times out at 2 s                    |
| N2 a pause loses nothing        | 40 events arrive in order, all before the interrupt                                                                                                           | Dev's node drops them all                                  |
| N3 resume continues the planner | After the answer, no model call, tool body or event from before the pause repeats; the plan and planner history equal an unpaused run's                       | Restarting from the brief calls the model again            |
| N4 survives SIGKILL             | A process killed inside `propose_plan` repeats one model call, that tool body and one progress event; nothing from before the pause repeats; the plan matches | Keeping the history in process memory repeats earlier work |
| N5 stop                         | Cancelling the consumer cancels the planner; no task is left; the next turn runs                                                                              | A planner the node never cancels leaves a task running     |
| N6 a planner tool fails         | The error surfaces as it does for an awaited planner                                                                                                          | A node that waits on the queue alone hangs                 |

The planner's `Conversation` lives in ADK state under `planner:conversation`. A state write made just before the interrupt is saved with the interrupt event itself, so the history and the pause are stored in one write.

## Stage B: the real Studio turn (`studio/`)

This stage swaps the same node into dev's real root workflow, in test code only, and drives Studio in process over a streaming ASGI transport with dev's scripted orchestrator. All 11 tests pass, and no harness source changed.

- **S1 live progress.** The first progress frame reaches the SSE client after 0.09 to 0.20 s, while `propose_plan` is still waiting. Dev's node with a stub planner sends its first frame at about 3.1 s, after planning ends.
- **S2 the ask.** The planner's ask reaches Studio as an `elicitation.request`, after all 10 progress frames. Answering through `/interrupt` resumes the stored planner run with no repeated model call, and the plan card follows. Posting the same answer again returns 404, since Studio has already dropped the pending question, and the session holds one response.
- **S3 isolation.** No orchestrator model request contains the planner's text, tool names, call ids or interrupt id. This check has no failing control. The prototype wrote planner text into the session three ways, and ADK's context rules kept all three out of the orchestrator's requests, so isolation rests on those rules and on the node keeping planner messages out of shared state.
- **S4 stop.** Both Studio's `/cancel` and a client disconnect end the stream and cancel the planner. No task is left, and the next turn reaches the plan card.
- **S5 records.** The plan, its revision, the approval state, the plan records, the approvals and the stored turns all equal dev's, once times and run ids are removed. The only difference is a `planner:conversation` key, which holds `None` at the end. The `approve` and `execute` nodes need no change.

## DBOS checkpoints inside the node (`dbos/`)

This stage runs each planner segment, from the brief or an answer to the next pause, as a DBOS 3.4.0 workflow through Pydantic AI's `DBOSDurability`, inside stage A's node. The pause still lives in ADK. The workflow id is `{invocation_id}:{segment}`, so a node re-entered after a crash asks DBOS for the same workflow. All 15 tests pass, and every control fails as predicted.

- D1, a kill. Stage A's N4 kill repeats no model call. The `propose_plan` body still reruns from the top, since a plain function tool runs in workflow code. Progress `(2,1)` reached the client once, against twice in stage A, because its handler step was recorded before the kill; a kill between the handler and its commit would still send it twice.
- D2, live progress. The first event arrives in 0.04 s, and 40 events arrive in order before the interrupt. `ctx.emit` from a plain tool reaches the `event_stream_handler`, each event in its own step.
- D3, re-entry. Calling the workflow again with its id does not resume a segment a dead executor left `PENDING`: DBOS waits on that owner forever. The node checks the status and calls `DBOS.resume_workflow_async`, a public function used here in an undocumented way. It cannot tell a dead owner from a live one.
- D4, stop. Cancelling the consumer cancels the model call in flight and leaves the workflow `CANCELLED`. The node must cancel the DBOS workflow itself, or the next launch reruns the segment with no one listening. A cancel from another process takes effect at the next step: 0.05 to 0.08 s inside the tool loop, 5.3 s inside a 6 s model call.
- D5, failure. A tool failure surfaces as in stage A within 5 s, and the workflow ends `ERROR` without retrying.

Two hazards were read from DBOS's source and not run. A resumed segment goes on DBOS's queue, which any instance may claim, so with several instances its progress can land on an instance the client is not connected to. And each instance needs its own executor id, since with a shared one a new instance's launch recovery takes over segments a live instance is still running. A production version also needs a shared Postgres system database, and a new `application_version` whenever a change alters a segment's step order.

## What a production version needs

- **`planning.py`.** About 40 lines go: the await, the progress queue and the `NodeInterruptedError` handling. About 70 replace them: the relay loop, the ask's interrupt, and the stored conversation keyed on the invocation. A pick's replan runs under `switch._unless_stopped` as a task, and that wrapper must stop the stream it drives (about 20 lines).
- **`studio_translate.py`, about 50 lines.**
  - Render the planner node's progress events. Dev sends no frame for them today; the prototype used the running `activity.step` shape `execute` uses.
  - Rebuild the "Checking the plan against the data" and "Reworking the draft" lines from progress events. The Pydantic AI planner makes no ADK events for the translator to read.
  - `_planner_question` reads options from the top level of the ask's arguments, but a `RequestInput` puts them under `payload`. So the prototype's ask showed with no options, free text only.
- **The planner itself.** `PlannerModel` returning a dict after an await no longer fits: the node needs an agent and dependencies it can stream and resume. Porting the planner's tools to Pydantic AI is most of the cost. PR #129 makes the readers framework-neutral, which helps.
- **Duplicate answers.** In stage A, after the kill, ADK accepted the same answer twice and stored both. Studio's route refused the duplicate in stage B, but a production node should still refuse a second answer to one ask itself.

## Not covered

- Two asks in one planner run, and an ask during an amendment or a pick's replan.
- A crash between the answer and the planner's first event.
- A served check against uvicorn instead of the test transport.
- How much a slow client slows the planner. The planner runs about one event ahead of the client because both sides apply back-pressure.
- A live Gemini run. Every model here was scripted; the paired Gemini run in `prototype/gemini/` covers the model side.
- Several instances sharing one DBOS system database, and a kill inside the model step or between a segment's end and the ADK pause.
