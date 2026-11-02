# Our own turn loop against ADK's `Workflow`, on five cases that have hurt us

Two runtimes over the same stub nodes, with no model calls. The own loop (`loop.py`, `store.py`) is an enum of states, one async function per state, and a cursor row in Postgres that every write moves by compare-and-swap. The ADK runtime (`adk_flow.py`) is a google-adk 2.13.0 `Workflow` with the shape of dev's root workflow on `DatabaseSessionService`, run with and without dev's guards. Both stores are one throwaway `postgres:18-alpine` container bound to 127.0.0.1, started and removed by `conftest.py`. Postgres was chosen over SQLite because Docker is available, dev runs on Postgres, and it gives row locks and `LISTEN/NOTIFY` for the cancel. Every request runs in its own child process (`own_driver.py`, `adk_driver.py`), so nothing survives in memory between a pause and its resume.

## Result

The stored cursor holds on all five pass lines, and each control fails the line it removes. All 20 tests pass, in two full runs of 5.6 and 7.3 minutes; the 40 ADK race processes take about 3.3 of them.

| Case                                              | Own loop                                                                                                                                                                                                                        | Own-loop control, and how it fails                                                                                                                                             | ADK, no guards                                                                                                                                                            | ADK, dev's guards                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| T1 deploy between pause and resume                | resumed under v2 runs `approve, execute, plan_done, orchestrate`; `vet` does not run; the next turn runs `seed, orchestrate, planner, vet, approve`                                                                             | cursor ignored, replay from `seed`: resumed turn runs `seed, orchestrate, planner, vet, approve, ...`; fails `vet ran in the resumed turn`                                     | resumed turn runs `seed, vet, approve, ...`: `vet` runs live mid-turn, and `seed` runs again too                                                                          | passes, but only with a guard written for `vet` (see below)                                                                          |
| T2 repeated interrupt                             | pauses `(approve, 1)`, `(approve, 2)`; round-1 answer reaches round 1, round-2 answer reaches round 2; the late round-1 answer gets `conflict`                                                                                  | answers matched on interrupt id alone: the late round-1 answer approves round 2's plan, and the real round-2 answer then gets `conflict`; fails `late round-1 answer accepted` | the late round-1 answer resumes round 2's pause and approves round 2's plan                                                                                               | passes: the ledger's pending round refuses the late answer                                                                           |
| T3 race, 20 repeats                               | 20/20: one `done`, one `conflict`, no node or step twice                                                                                                                                                                        | no compare-and-swap: 20/20 fail, both `done`, and `approve`, `execute`, every step, `plan_done` and `orchestrate` ran twice                                                    | 20/20 pass: the loser's first append raises `StaleSessionError`                                                                                                           | 20/20 pass                                                                                                                           |
| T4 crash in `execute`, then the same answer twice | killed after step 1 finished with step 2 in flight; the first repeat takes the lapsed lease and runs `execute` from step 2 (steps 2, 3, 4), then `plan_done`, `orchestrate`; the second repeat gets `conflict` and runs nothing | checkpoints only at pauses: the repeat reruns `approve` and steps 0 to 4; fails `finished steps reran: [0, 1]`                                                                 | both repeats accepted; replay finds `execute` with events but no output, ends its branch and reports the turn done: steps 2 to 4 and `plan_done` never run, with no error | both repeats refused, since the ledger already consumed the answer; nothing in the turn continues the run                            |
| T5 cancel from another instance                   | A stops 0.005 s after the cancel write began; no step starts after it; process C resumes from step 2 and finishes, no finished step rerun                                                                                       | in-process flag only: A runs all 40 steps; fails `did not stop`                                                                                                                | never stops                                                                                                                                                               | stops 2.2 to 2.8 s after the write over four runs (dev's 5 s heartbeat, so up to 5 s); the turn cannot be resumed by answering again |

What reruns after a crash in the own loop is exactly the step in flight at the kill (step 2) and the `execute` node's entry. Everything checkpointed stays done. The latency in T5 is measured from just before process B opens its transaction, so it bounds the time from the commit. Process A hears the commit through `NOTIFY`, so A usually stops before B's own commit call has returned.

### What the ADK runs show

- A node that records no output reruns on every resume, deploy or not. Dev's `seed` returns nothing when it has nothing to hand over, so without `_turn_ran_before` it reran in every resumed segment here (T1, T4).
- A new node does run live mid-turn on a resumed session (T1, `vet`). Dev's `_turn_ran_before` does not transfer to it: it asks whether any other node ran in the invocation, which is always true for `vet` in a fresh turn too, so copied verbatim it would skip `vet` for good. The guard that works here asks whether `approve`, a node downstream of `vet`, already ran. Each new node needs its own reasoning about where it sits. This case was reasoned from the code, not run.
- The interrupt id repeats, and ADK delivers whatever answers it to the open pause (T2). The defence is outside ADK: the ledger's pending round, which dev implements as `plan_revision_stale` and the revision check in `approve`.
- Without `awaiting_approve`, round 2 still raised its pause in this stub: `ctx.resume_inputs` did not carry round 1's answer into the second `approve` run of the same invocation. The case that made dev add the flag was not reproduced here.
- ADK's session service is itself a compare-and-swap store: an append from a session object read before another writer's append raises `StaleSessionError` (`database_session_service.py`, the `_storage_update_marker` check under a row lock). That is why T3 passes on ADK with no guard of ours. It does not refuse the same answer arriving after the first one landed (T4), because the second request reads the session after it moved.
- A crash inside a node without `rerun_on_resume` loses the rest of the turn without an error (T4). Dev's `execute` is such a node; dev recovers its runs on the next turn through `seed`'s `run_resumer`, which this stub does not have.

## The own loop

- `store.py`: the `turns` cursor row (state, round, pending interrupt id, version, lease owner and expiry, cancel flag, the node data), `agent_records` (one JSON record per agent, written in the same transaction as the cursor move), and `answers` keyed `(turn, interrupt id, round)`. Every write is `update ... set version = version + 1 where version = $read and <guard>`; a zero-row update is `Conflict`, the 409. Claiming an answer checks the pause is open at that round. A repeated answer to a pause already answered takes the lease over only when the lease has lapsed and the turn is not done, so a crashed turn continues from its checkpoint and a second repeat is refused. Cancel sets the flag and sends `NOTIFY` in one transaction.
- `loop.py`: `drive` dispatches on the cursor's state and moves the cursor after every node, so every node boundary is a checkpoint; `execute` also checkpoints each finished step. Edges are a dict per code version, so v2's `vet` exists only in the table a new turn walks. `run_turn` adds the lease heartbeat (every TTL/3, TTL 2 s) and a `LISTEN` connection that cancels the running node on a notice for its turn, with the cancel flag read once after `LISTEN` starts and on each heartbeat as a fallback. A stop releases the lease where the cursor stands, so the turn is resumable.

### Size

Counting code lines without docstrings, comments or blanks: `store.py` 225 and the runtime part of `loop.py` about 205 (dispatch, heartbeat, cancel; the stub nodes are another 75), so about 430 lines. What they replace in dev: ADK's replay-based resume, the `rerun_on_resume` flags on four nodes, `seed`'s `_turn_ran_before` (17 lines) and the guard every future node would need, `approve`'s `awaiting_approve` flag, and the lease, checkpoint, renew and cancel half of `studio_state.py` that bends the ADK ledger session into a compare-and-swap store (about 700 lines by the footprint report, much of it merging and reconciling deltas that a cursor row with typed columns does not need). The loop does not replace `ConversationState`'s wire bookkeeping (frame `seq`, composition manifest, digests), which stays ours either way.

### Coverage of `footprint-runtime.md` items 1 to 19

Covered: 1 (resume on another process from Postgres without rerunning finished nodes), 2 (answer paired with its pause, by interrupt id and round), 3 (repeated interrupt ids, without a node flag), 4 (a deploy-added node is safe by construction), 6 (node state persists, as cursor data and agent records written with the cursor move), 17 (lease with TTL, heartbeat, checkpoints and cross-instance cancel on compare-and-swap writes), 18 (a lost race is a `Conflict`, never corruption). Partly covered: 11 (`execute` checkpoints per step and stops on cancel, but relays progress to stdout, not to a client stream) and 13 (a cross-instance stop leaves the turn resumable; a client disconnect was not run).

Left out: 5 (a pause inside a child agent run; `prototype-node-stream` showed a Pydantic AI planner's `Conversation` stored beside its pause, which would live in `agent_records`), 7 (reading dev's persisted ADK keys and session ids), 8, 9 and 10 (model context, hand-backs and sub-agent branches belong to the agent layer), 12, 14 and 15 (planner progress, partial text, translator frames), 16 (tool errors) and 19 (startup retries on `prepare_tables`).

## What a production version would change

- Lease the conversation, not the turn: dev's segment keeps a new turn from starting while another instance streams the last one. About 40 lines.
- One `LISTEN` connection per instance that routes notices to its running turns, in place of one connection per turn, and a check that the Cloud SQL path allows `LISTEN` (it does not survive transaction-mode pooling). About 40 lines.
- Refuse loudly a cursor whose state a release removed or renamed, with a mapping table for renames. About 20 lines.
- Store each agent's Pydantic AI `Conversation` in `agent_records` and resume a pause inside the planner from it, as `prototype-node-stream` does. About 100 lines.
- Emit Studio frames from `drive` and node events instead of translating ADK events. Size depends on the frame vocabulary; the translator it replaces is about 900 lines.
- Answer idempotency by a client answer id rather than "same pause, lapsed lease", so a retry while the first attempt still runs reads its outcome instead of a `Conflict`. About 30 lines.
- A migration from dev's ADK ledger and turn sessions. About 200 lines.

## Internals relied on

- ADK, private: `google.adk.workflow.utils._workflow_hitl_utils.create_request_input_response` to build answers, as `harness/studio.py` does.
- ADK, undocumented behaviour observed: replay from START reruns nodes with no recorded output; a node with events but no output on replay ends its branch with the log line from `workflow/_graph.py` ("has conditional/DEFAULT edges but none were matched"); `DatabaseSessionService.append_event` raises `StaleSessionError` on a stale storage marker, under `SELECT ... FOR UPDATE` on Postgres; a function response to an already answered interrupt is accepted and appended.
- Postgres: `READ COMMITTED` re-evaluates an `UPDATE`'s `WHERE` after a concurrent writer commits, which is what makes the version check a compare-and-swap; `pg_notify` is delivered at commit.
- Pydantic AI and DBOS: none.

## Not covered

- Planner pauses inside a child agent, live model calls, client streams and frames, and the `switch` node.
- A kill between an answer's claim and the first node, or inside `approve`; the kill here lands inside `execute` after a checkpoint.
- Two instances of the heartbeat racing a takeover at the TTL edge, and clock skew between instances (lease expiry uses the database's `now()`, so instance clocks do not matter).
- T3 forces both claimants to read before either writes. A second answer that reads after the first landed is T4's duplicate, which the own loop refuses by the pause's state and ADK accepts.

## Files

| File                      | Purpose                                                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `store.py`                | Cursor, agent records and answers in Postgres; compare-and-swap writes, lease, cancel                          |
| `loop.py`                 | States, per-version edges, stub nodes, `drive`, `run_turn` with heartbeat and `LISTEN`                         |
| `own_driver.py`           | One own-loop request per process: `start`, `answer`, `resume`, `cancel`, and the control flags                 |
| `adk_flow.py`             | The ADK workflow with v1/v2 and dev's guards on or off                                                         |
| `adk_driver.py`           | One ADK request per process, with dev's ledger claim, pending rounds and 5 s cancel heartbeat under `--guards` |
| `runlog.py`               | The shared run log every node writes to                                                                        |
| `procs.py`, `conftest.py` | Child processes; the Postgres container                                                                        |
| `test_turn_loop.py`       | The 20 tests: 5 own-loop checks, 5 controls, 10 ADK runs                                                       |

## Commands

From the worktree root, with Docker running:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-turn-loop -q -s -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-turn-loop -q -s -p no:cacheprovider -k "own"
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-turn-loop
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-turn-loop
```

`-s` prints each case's outcome on both runtimes. The second command runs only the own loop and its controls.
