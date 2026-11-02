# Three harness prototypes beyond the planner

The planner prototypes left three questions about the rest of the harness. Each was built on dev at 190c7a9c (google-adk 2.13.0, pydantic-ai 2.64.0, dbos 3.4.0) with scripted models, and each check has a control built to fail it. The code and full notes are in the three directories named below.

## The orchestrator on Pydantic AI inside the `orchestrate` node (`prototype-orchestrator-node/`)

The orchestrator runs on Pydantic AI inside the real `orchestrate` node of dev's workflow, behind the real Studio routes, while ADK still owns the turn. All 16 tests pass. With the planner prototypes, this shows that both model-backed agents can move to Pydantic AI and leave ADK only the workflow.

- O1, a full turn. A turn through plan, approval, execute and `plan_done` streams the reply as `message.delta` frames and then `message.complete`, and its frames and stored records equal dev's once ids and times are removed. The orchestrator's `Conversation` lives in state under `orchestrator:conversation` between node entries.
- O2, the clarify pause. The answer resumes the stored conversation, and no model call repeats.
- O3, the hand-back. `plan_done`'s hand-back arrives as the one result of the open `plan` call, a deferred tool, equal to what dev's `_call_result` builds.
- O4, a tool failure. A raising tool answers the model with dev's error, and the turn goes on.
- O5, isolation. Neither agent's requests hold the other's internals. Session state is the only path between them; an unbranched orchestrator event does not reach dev's ADK planner.
- O6, wire parity. The instruction, nine of ten declarations, their order and the thinking config are identical. Eight differences remain, all on the Pydantic AI side and each explained: a trailing newline, an empty parameter schema for `explain_dfo`, one `Tool` per declaration, an explicit AUTO tool config, digests and question in one content, no `automatic_continuation`, `response_modalities` set to `TEXT`, and headers.

The turn budget carries over only in part. A wrapper model enforces the call cap with dev's closing sentence, but a cut mid-stream ends the Pydantic AI turn on an error frame where dev ends on the sentence; a custom `StreamedResponse` or a catch in the node would fix it. The node also reads ADK's private `ctx._invocation_context` and `FunctionTool._get_declaration`.

A production version changes about 500 lines across `orchestrate.py`, `orchestrator/agent.py`, `turn_budget.py` and `prefix_cache.py`, and leaves `studio_translate.py` alone. `orchestrator_tools.py` (2,795 lines) stays behind an adapter until its tools leave `ToolContext` (46 sites), as the planner's tools are leaving.

## Our own turn loop against ADK's `Workflow` (`prototype-turn-loop/`)

The same stub nodes, with dev's shape and loop-back edges, ran under a minimal ADK 2.13 `Workflow` and under a loop of our own: plain async functions dispatched on a cursor row (state, round, pending interrupt id, version) in Postgres 18, with compare-and-swap writes and `LISTEN/NOTIFY` for cancel. Every resume ran in a fresh process. All 20 tests pass.

| Case                                                | Own loop                                                        | ADK without dev's guards                                                                              | ADK with dev's guards                                               |
| --------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| T1, a deploy adds a node before `approve`           | the resumed turn skips it; the next turn runs it                | `seed` and the new node run live mid-turn                                                             | passes only with a guard written for the new node                   |
| T2, `approve` raised again after an amendment       | each answer reaches its round; a late round-1 answer is refused | the late answer approves round 2                                                                      | passes                                                              |
| T3, two answers race, 20 times                      | one wins every time                                             | one wins every time                                                                                   | one wins every time                                                 |
| T4, a kill in `execute`, then the same answer twice | only the step in flight reruns; the second answer is refused    | both answers accepted; the turn reports done with three steps and `plan_done` never run, and no error | both refused; nothing continues the turn                            |
| T5, cancel from another instance                    | stops 0.005 s after the write and stays resumable               | never stops                                                                                           | stops after 2.2 to 2.8 s, the heartbeat; not resumable by answering |

Each own-loop line has a control that fails it: replaying from `seed`, matching answers on interrupt id alone, no compare-and-swap, checkpoints only at pauses, and an in-process cancel flag.

The research's central claim holds on these cases. A stored cursor needs no guard per node, where ADK runs new nodes mid-turn without one, and a crashed turn either loses three steps silently or, with dev's guards, is left with nothing to continue it. The own loop is about 430 lines: 225 for the store and about 205 for dispatch, heartbeat and cancel. It replaces ADK's replay, the `rerun_on_resume` flags, `_turn_ran_before`, `awaiting_approve` and the lease, checkpoint and cancel half of `studio_state.py`. Of the 19 runtime duties in `footprint-runtime.md`, it covers 1 to 4, 6, 17 and 18, part of 11 and 13, and leaves the rest. A production version adds about 430 more lines: a conversation lease (40), one `LISTEN` connection per instance (40), refusing a cursor whose state a release removed (20), each agent's stored conversation (100), answer idempotency by a client answer id (30) and a migration from dev's ledger (200). Studio frames would come from the loop, in place of the 900-line translator.

## `execute` in the background on DBOS (`prototype-background-execute/`)

`execute`'s compiled run becomes a DBOS workflow, with each tool call a step, inside dev's node, with dev's planner and scripted orchestrator behind the real Studio routes. All 12 tests pass.

- X1, the client leaves. The run finishes with no client, each tool runs once, and the next turn picks up the outcome through `seed`'s carry-on and hands it to the orchestrator. Dev's node cancels the run.
- X2, a kill mid-run. A second process under another executor id finishes the run, and only the tool in flight reruns. Dev reruns every step.
- X3, live progress. Each report reaches the client within 2 ms, in dev's frames.
- X4, stop. The workflow ends `CANCELLED`, and no step starts after the stop. Cancelling only the local task leaves the workflow `PENDING`, and the next launch runs it to the end with nobody listening.
- X5, records. Ledger records, published frames and the hand-back equal dev's.

X1 passes on the in-memory ledger dev's tests use and fails on the production one. `StudioRunLedger` fences each write on the live stream segment, so the background run's next write after a disconnect raises `SegmentLost`, and the next turn reruns every step outside DBOS. Each run needs a lease of its own, about 60 to 100 lines. Two more defects need fixing first. Every tool call shares one step name, and a wide wave starts its calls in the order earlier calls finish, so a replay could hand one step's recorded result to another; steps need a deterministic start order or a lookup by plan step id. That was read from the code and not run, since the scripted plan has one step per wave. And a background run's last activity frame leaks into the next turn. With several instances, a request may drive the run a second time or take over a live one, and progress stays on the instance running it, as in `prototype-node-stream/dbos/`. A production version is about 250 lines.

## What the three show

The mixed design reaches the whole agent layer: both agents can run on Pydantic AI inside ADK's nodes with Studio's translator unchanged, at about 500 lines for the orchestrator beside the planner's port. The turn loop is where a full move pays: the cursor removed every guard ADK needs and handled the crashed turn ADK either drops or strands, in about 430 lines plus about 430 for production. Background execution on DBOS works under either framework and is worth building on its own merits, once each run has its own lease and steps start in a fixed order.
