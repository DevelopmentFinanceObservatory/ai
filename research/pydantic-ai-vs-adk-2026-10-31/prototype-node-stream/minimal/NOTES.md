# A Pydantic AI planner inside an ADK 2.13 workflow node, streaming live

A three-node ADK 2.13.0 `Workflow` (`orchestrate` -> `planner` -> `approve`) run by a real `Runner` over `DatabaseSessionService` on SQLite. The `planner` node (`rerun_on_resume=True`) runs a Pydantic AI 2.64.0 agent on a scripted `FunctionModel`. No harness imports, no model calls.

## Result

The hypothesis holds on all six pass lines. All 12 tests pass (6 checks, 6 controls), and each control fails the way its check predicts. The suite passed four runs in a row on the final code, 29 to 35 s each; the N4 subprocess test takes about 10 s, and the N1 and N6 controls take 2 s and 5 s because they wait out their timeouts.

| Line                    | Result                                                                                                                                                                                                                                                                              | Control, and how it fails                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| N1 live progress        | first progress event reaches the `run_async` consumer 0.06 s after the turn starts, while `propose_plan` is parked on the client's signal (its `end` line not yet logged)                                                                                                           | `AFTER_AWAIT`, dev's queue-then-relay: `wait_for` raises `TimeoutError` at 2.0 s                                                                                                 |
| N2 pause loses nothing  | 40 progress events, in order, all before the ask's interrupt, read by a client sleeping 5 ms per event                                                                                                                                                                              | `AFTER_AWAIT` drops queued reports on a pause: `got progress []`                                                                                                                 |
| N3 resume continues     | model calls before the pause `after 1, after 3 messages`, after it `after 5, after 7`, equal to the unpaused reference; `propose_plan` call 1, call 2 and `ask_user` bodies ran once each; 6 progress events, none twice; plan and planner history equal the bare-agent reference   | `RESTART` reruns from the brief: `model called again: [after 1, after 3, after 5, after 7]`                                                                                      |
| N4 survives SIGKILL     | the third process completes with plan hash `fe46f480...e1883` and the reference history (see below)                                                                                                                                                                                 | `PROCESS_MEMORY` keeps the paused `Conversation` in process memory: process 2 cannot find it, restarts from the brief, and the check fails with `repeated from before the pause` |
| N5 stop                 | cancelling the consumer task cancels the planner (`propose_plan call=1 cancelled` logged); no task left beyond the baseline; no exception reaches the loop's exception handler; a new turn on the same session runs to the ask with progress `(1,1) (1,2) (1,3)` and no error event | `ORPHAN` runs the planner in a task the node never cancels: `tasks left running: [Task-N]`                                                                                       |
| N6 planner-side failure | `propose_plan` raising gives an error event `PlannerToolError` and `run_async` raises `PlannerToolError` to the consumer, identical to the awaited planner (`AFTER_AWAIT`), within 5 s                                                                                              | `QUEUE_ONLY` waits on the progress queue and never on the planner task: the stream hangs and `wait_for` raises `TimeoutError` at 5 s                                             |

### N4 detail

Process 1 starts the turn and ends at the ask's interrupt. Process 2 answers it, runs `propose_plan` call 2 to its first progress event, and parks; the parent reads that line and sends SIGKILL (return code -9). Process 3 sends the same answer again, then approves.

What repeats, from the shared log file and each process's stdout:

- one model call, the one after 5 messages, which produced `propose_plan` call 2;
- the body of `propose_plan` call 2 (`start` twice, `end` once);
- progress event `(2, 1)`, which the client receives twice and the session stores twice (stored progress events: `(1,1) (1,2) (1,3) (2,1) (2,1) (2,2) (2,3)`).

Nothing from before the pause repeats. Process 3's second answer to `planner_ask:ask-1` is accepted: the session then holds two function responses to that one interrupt and ADK resumes the same invocation. So the fallback point is the pause, as in the loop prototype: a tool body interrupted by the kill runs again from the top, and progress is at-least-once with nothing on the event to deduplicate by except `(call, done)`.

## Where the `Conversation` lives

In ADK session state, under `planner:conversation`, as `{"invocation_id", "conversation": <ConversationTypeAdapter JSON>}`, written as the node pauses and cleared when the planner finishes.

The write survives the pause. A state write in a node goes to `ctx.actions.state_delta` and is moved onto the next non-partial event the node yields (`_node_runner.py`, `_flush_deltas`, called from `_enqueue_event`). The node writes the key and then yields the ask, so the delta travels on the interrupt event itself, which the `Runner` appends to the session before the consumer sees it. N3 asserts this: the interrupt event carries the delta, and a second `DatabaseSessionService` over the same file reads the key. N4 shows it across processes.

State wins over a table of our own because it commits in the same session append as the interrupt. A pause can then never be visible without its conversation, or the reverse, and no second store or transaction is needed. The `invocation_id` stamp makes a stored conversation resumable only by the invocation that paused it; an abandoned pause on a reused session is ignored and the planner starts from the brief. The cost is size: the whole conversation is rewritten into the state delta at each pause.

## What the node relies on in ADK 2.13

The node imports nothing private. It yields a public `RequestInput`, which `BaseNode` turns into the interrupt event (`_base_node.py`, line 211). The test client builds its resume message with `create_request_input_response` from the private `google.adk.workflow.utils._workflow_hitl_utils`, as `harness/studio.py` does.

Behaviour it depends on, none of it documented as a contract:

- `rerun_on_resume=True` reruns the node with `ctx.resume_inputs` keyed by interrupt id; the value is the response dict as sent (`{"answer": ...}`), and resumed inputs are rehydrated from the session's events (`_node_runner.py`, `_create_child_context`).
- `ctx.invocation_id` is the same on resume, since the `Runner` finds the paused invocation from the function response (`runners.py`, `_resolve_invocation_id_from_fr`).
- A non-partial event blocks the node in `InvocationContext._enqueue_event` until the `Runner` has appended it and the consumer has taken it (`runners.py`, `processed_signal.set()` after the `yield`). Together with Pydantic AI's zero-buffer stream in `run_stream_events`, this keeps the planner about one event ahead of the client. That is what makes N1 and N2 hold, and it also means a slow client slows the planner. The prototype does not measure that throttling; the next step is to time `propose_plan` against a client with a fixed per-event delay.
- Early close of the `run_async` stream cancels the root task (`runners.py`, `_cleanup_root_task`), which cancels the node, which leaves `run_stream_events`' `async with`, which cancels and drains the planner task (`pydantic_ai/agent/abstract.py`, `AgentRunEvents.aclose`).
- A node exception becomes an error event plus a re-raise to the consumer (`_node_runner.py`, `run`; `runners.py`, "Root node ... failed").

## ADK rules the design bends

ADK warns against running work that calls `ctx.run_node` in a detached task: an interrupt there cannot be resumed (`_workflow.py`, "Await ctx.run_node() directly"). The design does not touch that rule, because the planner task makes no ADK call at all. It reads only `Deps`, and its pause is a `DeferredToolRequests` return value that the node turns into its own interrupt. The task is Pydantic AI's own (`run_stream_events` starts it), and the node never creates one. What the design gives up is ADK owning the planner's pause: ADK sees an ordinary function node that yields a `RequestInput`, and the planner's resumability rests on the stored `Conversation` and the `invocation_id` check above.

## Files

| File                    | Purpose                                                                                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `planner_agent.py`      | The scripted planner: `propose_plan` (progress via `ctx.emit`, waits on a gate with a timeout, can raise) and `ask_user` (`CallDeferred`). Model calls and tool bodies are logged. |
| `graph.py`              | The workflow, the `planner` node in `Mode.LIVE` and the five control modes, and `build_runner`.                                                                                    |
| `turn.py`               | Sessions, resume messages and event helpers for a client.                                                                                                                          |
| `checks.py`             | The six checks, each taking the `mode` its control uses, and the bare-agent reference run.                                                                                         |
| `driver.py`, `procs.py` | One turn's steps as a child process with JSON lines on stdout, and the runner that kills it on a chosen line.                                                                      |
| `test_node.py`          | The 12 tests.                                                                                                                                                                      |

## Commands

From the worktree root:

```
cd research/pai-node-stream/minimal
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness pytest . -q -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness pytest . -q -s -p no:cacheprovider -k "n3_resume or n4_survives or n6_tool"
```

The second prints the N3 model calls, the N4 repeats and the N6 failure. Lint and types, from the worktree root:

```
uv run --package harness ruff format research/pai-node-stream/minimal
uv run --package harness ruff check research/pai-node-stream/minimal
uv run --package harness ty check research/pai-node-stream/minimal --extra-search-path research/pai-node-stream/minimal --extra-search-path <pydantic-ai 2.64.0 site-packages>
```

## Limits of this evidence

- The script is deterministic by construction, so history equality shows the store and resume add no difference and says nothing about a live model.
- The kill lands inside `propose_plan` after a progress event. A kill between the answer's append and the planner's first event, or inside `approve`, was not run.
- N5 cancels the consumer task. The Studio path closes the stream with `aclose()` from `_unwind`; both reach `_cleanup_root_task`, but only the cancel was run.
- One ask per turn. Two asks in one planner run would need `_answer_for` to take several calls; the interrupt id already carries the tool call id.
