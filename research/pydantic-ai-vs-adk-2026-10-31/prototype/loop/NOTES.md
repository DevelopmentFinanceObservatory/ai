# A1: a turn loop we own, over Pydantic AI agents

The Studio turn (`orchestrate`, planner, ask, approve, execute) written as an explicit state machine over two Pydantic AI 2.64.0 agents on scripted `FunctionModel`s, with a SQLite store. No harness imports, no network, no model calls.

## Result

All five tests pass, and each fails on its broken variant. The kill observation is recorded below. The suite is 11 tests (5 checks, 5 broken-variant tests, 1 extra), 13 to 18 s per run. Before the amend test was added, the full suite passed three times in a row and the eight in-process tests seven more times.

| Test                             | Pass line                                                                                           | Result                                                       | Broken variant, and how it fails                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1 live progress                 | first `activity.step` frame reaches the client within 2 s while `propose_plan` is parked on a gate  | passes; the whole test takes 0.08 s                          | `buffer_planner_events=True` holds planner frames until the run ends; `wait_for` raises `TimeoutError` after 2.0 s (test takes 2.00 s, no hang)            |
| L2 pause loses nothing           | 40 progress frames in order before `elicitation.request`; `frames` table equals what the client got | passes                                                       | `drain_before_pause=False` ends the stream when the machine task finishes and leaves queued frames behind; the assertion `got progress frames [...]` fails |
| L3 survives SIGKILL              | orchestrator and planner histories and plan hash equal across a killed run and an unpaused run      | passes, plan hash `1c0c59ca...868e77`                        | the killed process loses its planner `Conversation` write (`--drop-agent planner`); the stored planner history is stale and `planner differs` fails        |
| L4 one resumer wins              | two resumes of one approval pause: one runs, one is refused                                         | passes; exactly one `done` frame                             | stores with `unconditional=True` ignore the version; both resumes run and `outcomes` shows two lists                                                       |
| L5 deploy-safe resume            | a state added to the opening path does not run on resume                                            | passes (seed ran 0 times on resume, 1 time on a fresh start) | `ReplayingLoop` re-enters the opening path on resume, like ADK; `the new opening state ran on resume` fails                                                |
| Amend and stale approval (extra) | amend gives revision 2 and a new hash; approving revision 1 is refused                              | passes                                                       | mutation: the revision check in `_claim` replaced by `if False:`; `DID NOT RAISE RefusedError` (mutation reverted)                                         |

### L3 detail

Three runs of the same scenario:

1. Process 1 starts the turn and ends at `elicitation.request`.
1. Process 2 answers the ask and runs until it prints `plan.proposed`. The parent reads that line and sends SIGKILL (the driver was run with `--linger 30`, so the process is alive; return code -9).
1. Process 3 approves revision 1 and prints `done`.

The comparison targets are the bare agents run in memory with the answers resolved inline (`reference.py`, no store, no serialisation), and the loop run in one process with no kill. Both equal the killed run, byte for byte, for the orchestrator and the planner.

Fields that differ before normalisation, after dropping `timestamp`: `run_id` only (the test records `{'run_id'}`). Every run gets a fresh UUID. I replace each `run_id` by the order in which it first appears, so messages that share a run still share a value and the run structure is compared; only the UUID text is hidden. `conversation_id` does not differ because the loop and the reference both construct `Conversation(conversation_id=f"{cid}:orchestrator")` and `...:planner`; with the library's default (a fresh uuid7 per `Conversation`) it would differ too, and the same ordinal rule would apply. Tool call ids are set by the scripts as a function of the history (`plan-1`, `ask-1`, `propose-2`), so they match.

### Observation: SIGKILL while `propose_plan` runs

`python observe_kill.py` kills the process after the first progress frame of a `propose_plan` call, then runs `recover`, then answers and approves to `done`. Output, abridged:

|                                             | kill in `propose_plan` call 1 (planner just started)                                 | kill in call 2 (after the user answered the ask)                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Checkpoint the turn falls back to           | cursor `PLANNING` (planner input `brief`), version 1; no planner conversation stored | cursor `PLANNING` (planner input `answer`, claimed by the resume), version 3; planner conversation stored as paused at the ask |
| What recovery re-runs                       | the whole planner run from the brief                                                 | the planner run from its paused conversation with the answer                                                                   |
| Progress frames repeat                      | yes: the client saw 1, then 1, 2, 3                                                  | yes: the client saw 1, 2, 3, then 1, then 1, 2, 3                                                                              |
| Planner text deltas repeat                  | yes ("Drafting a plan." sent again)                                                  | the call-2 text comes only once (nothing before the kill)                                                                      |
| Tool bodies that ran twice                  | `propose_plan call=1` started 2 times, ended 1 time                                  | `propose_plan call=2` started 2 times, ended 1 time                                                                            |
| Tool bodies that did not run twice          | `plan`, `ask_user`, `propose_plan call=2`: once each                                 | `plan`, `ask_user`, `propose_plan call=1`: once each                                                                           |
| Finished turn equals the unkilled reference | yes (both histories)                                                                 | yes (both histories)                                                                                                           |

So the fallback is the last checkpoint, which is the start of the interrupted run, not a point inside it. A tool body interrupted by the kill runs again from the top, so a tool with a side effect would need its own idempotency key. Frames are at-least-once: the client sees the progress frames again and has no id to deduplicate on. The stored histories do not show the kill at all.

### L5 against ADK

Under the loop the cursor holds the state, so a resume starts at `AWAIT_APPROVAL`'s successor and never touches the opening path. In the harness, ADK resumes a paused turn by replaying the workflow from START and runs live any node the turn's events hold no run of, so a node added to the opening path runs mid-turn on every turn that paused before the deploy. That is why `seed` needs the `_turn_ran_before` guard (`services/harness/src/harness/graph/seed.py`, and `resume.py`). The loop needs no such guard for this case; the test's `ReplayingLoop` is what the guard exists to defend against. This test covers a new opening state. It says nothing about a changed or removed state: a cursor holding a state name that a later deploy deleted would fail at the `states[...]` lookup.

## What was built

| File                                                       | Purpose                                                                                                                                                                                                                                                                                                                                                             | Lines            |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `store.py`                                                 | SQLite: `turns` (cursor plus version, compare-and-swap), `conversations` (`ConversationTypeAdapter` JSON), `frames` (append-only). One transaction writes the cursor and the changed conversations.                                                                                                                                                                 | 154              |
| `agents.py`                                                | Orchestrator (`plan` tool raises `CallDeferred`) and planner (`ask_user` raises `CallDeferred`; `propose_plan` emits `PlanProgress` custom events with short awaits, optionally parks on a gate, returns `{"outcome", "revision", "plan_hash"}`). Both scripts are pure functions of the message history; revision and call number are derived from `ctx.messages`. | 286              |
| `loop.py`                                                  | The state machine, `start_turn`, `resume`, `recover`, the claim logic and frame streaming.                                                                                                                                                                                                                                                                          | 440              |
| `driver.py`                                                | CLI for one step against a store file; JSON lines on stdout, flushed per frame.                                                                                                                                                                                                                                                                                     | 84               |
| `reference.py`                                             | The inline, unpaused reference for L3.                                                                                                                                                                                                                                                                                                                              | 64               |
| `procs.py`, `checks.py`, `test_loop.py`, `observe_kill.py` | Child-process runner, the checks and their broken variants, the pytest file, the observation script.                                                                                                                                                                                                                                                                | 55, 337, 65, 141 |

The loop needed 440 lines in `loop.py` (359 without blank lines, comments and docstrings), 154 in `store.py` and 286 in `agents.py` (mostly the two scripts).

How it works:

- States: `ORCHESTRATE`, `PLANNING`, `AWAIT_ASK`, `AWAIT_APPROVAL`, `EXECUTE`, `PLAN_DONE`, `DONE`. The cursor holds the state, the orchestrator's pending `plan` call id, the planner input (`brief`, `answer` or `amend`), the plan revision and hash, the interrupt id, a pause counter and the execute result.
- Each state function returns an `Outcome` (next cursor, conversations that changed, an optional terminal frame). The loop checkpoints after every state: cursor and conversations in one transaction, guarded by `version`. The terminal frame (`elicitation.request`, `plan.proposed`, `done`) is sent after the checkpoint.
- The machine runs in an `asyncio` task and puts frames on a queue; the generator forwards them while the task runs, drains the queue after the task ends, then sends the terminal frame. Each frame is written to `frames` just before it is yielded.
- A resume reads the cursor, validates the interrupt id and the answer, and claims the pause with a compare-and-swap that moves the cursor to the running state with the answer stored (`PLANNING` or `EXECUTE`). The claim is the lock for L4. A refused resume raises `RefusedError`. Approval of a revision other than the stored one is refused before the claim.
- Interrupt ids are `f"{conversation_id}:{n}"`, so they are the same in every process.

## Commands

All from `<worktree>/research/pai-prototype/loop`.

```
uv run --no-project --with 'pydantic-ai-slim==2.64.0' --with pytest --with pytest-asyncio python -m pytest -q
PYDANTIC_AI_NO_BANNER=1 uv run --no-project --with 'pydantic-ai-slim==2.64.0' python observe_kill.py
uv run --no-project --with 'pydantic-ai-slim==2.64.0' python driver.py --db /tmp/t.db start --cid c1 --text "ODA to Ghana"
uv run --no-project --with 'pydantic-ai-slim==2.64.0' python driver.py --db /tmp/t.db resume --cid c1 --interrupt c1:1 --answer '"Ghana only"'
uv run --no-project --with 'pydantic-ai-slim==2.64.0' python driver.py --db /tmp/t.db resume --cid c1 --interrupt c1:2 --answer '{"kind":"approve","revision":1}'
```

`driver.py` options: `--progress K`, `--delay S`, `--hold-at N` (block `propose_plan` call N after its first progress event), `--drop-agent NAME` (broken store), `--tool-log FILE` (one line per tool body entered), `--linger S`. A step that the loop refuses prints `{"type": "refused", ...}` and exits 3.

Checks run: `ruff format`, `ruff check` (repo config, clean) and `ty check` (clean, with `--extra-search-path` pointing at the pydantic-ai site-packages).

## Deviations from the brief

- Resume claims before it runs, and the loop checkpoints after every state, not only at a pause. L4 needs the claim, and it makes the kill fall back to the start of the interrupted run. This added `recover(conversation_id)`, which runs a turn left in a running state (not in the brief). Two simultaneous `recover` calls are not guarded.
- `Loop` has three seams that exist for tests: `opening_state()` and `on_resume()` (L5 variants) and `after_load` (L4). The two broken-behaviour flags `buffer_planner_events` and `drain_before_pause` are fields on `Loop`, as the brief suggested.
- L4 needs `after_load`, an awaitable between the resumer's read of the cursor and its claim, held by an `asyncio.Barrier(2)`. Without it, one event loop runs the first resumer's read and claim with no await between them, so the second always reads an already-claimed cursor and is refused by the interrupt check, never by the compare-and-swap, and the unconditional-store variant would pass. The barrier stands for the real window between a read and a write across instances. Each resumer has its own SQLite connection.
- L2 uses a client that sleeps 5 ms per frame, with 40 progress events at zero delay. With a client that reads as fast as the planner writes, the queue is empty when the task ends and the broken variant loses nothing. A slow client is what gives the drain its purpose.
- L3's broken variant drops the planner write in the killed process only (process 2), so the stale write is the one at the approval pause. Dropping it in every process makes process 2 fail on a missing conversation, which proves less.
- L3's reference is bare agents in memory plus a one-process run of the loop. Both are compared with the killed run.
- `output_type=str | DeferredToolRequests` instead of the documented list `[str, DeferredToolRequests]`: the library accepts both; `ty` rejects the list form with `no-matching-overload` and cascades. The behaviour is the same.
- `agent.run` uses a `FunctionModel`'s non-streaming `function`, which these models do not define. The loop and the reference both go through `run_stream_events`.
- One extra test (amend and stale approval) covers loop behaviour the brief lists but does not test.
- `gate` is an `asyncio.Event` as briefed, plus `gate_call` to pick which `propose_plan` call parks on it.

## Private imports and monkeypatches

None. All imports are from public modules: `pydantic_ai`, `pydantic_ai.messages`, `pydantic_ai.models.function`, `pydantic_ai.run`. No monkeypatching. Nothing from the harness or elsewhere in the repo was imported or edited.

## Limits of this evidence

- The scripted models make the planner deterministic by construction. L3's byte equality shows that the store, `ConversationTypeAdapter` round trip and the resume logic add no difference; it says nothing about a live model's variability.
- The kill is always after a checkpoint (L3) or inside a run (observation). A kill between the claim and the first checkpoint, or inside `EXECUTE` or `PLAN_DONE`, was not run; by the same design they fall back to the claim and to the state's start respectively.
- Frames are written to the table just before they are yielded, so a kill between the two can leave the table one frame ahead of the client. Not tested.
- A planner run that is cancelled mid-run (client disconnect) leaves the cursor at the last checkpoint, as a kill does; only the kill was run.
