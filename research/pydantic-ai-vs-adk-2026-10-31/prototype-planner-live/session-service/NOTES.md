# The planner node on Postgres sessions

## Verdict

On ADK's `DatabaseSessionService` over a real Postgres, the Pydantic AI planner node matches dev's ADK planner node on all five pass lines. A planned turn writes 26 event rows on each arm, within 0.4% on event bytes and 1.9% on stored bytes, with identical session state. A turn paused on an ask resumes in a second process that shares only the database, a kill mid-planning leaves the same wreckage and recovers the same way on both arms, a second turn starts with a clean planner record, and an ask paused under one planner is answered under the other in both orders. Each line has a control that fails it.

Two facts about dev's recovery apply to both arms:

- A turn killed mid-planning is not resumed. The conversation holds a 45 second lease on its stream (`harness.studio_state.SEGMENT_TTL_S`), so the next `/turn` is refused with 409 for up to 45 seconds. After that the message starts a new turn in a new ADK session, and the dead turn is closed as `abandoned`. The planner reads its skill again and proposes again.
- A turn killed after the user answered an ask is resumed in place. The same `/interrupt` answer, sent after the lease lapses, resumes the turn's own session. Nothing before the answer runs again; the dead run's `propose_plan` call stays in the session without a response and the retry adds a second one.

One difference in what is stored: the node writes each tool call's progress report as the tool ends, authored by the planner (`dfo_planner`), where dev's node writes the same event once the seam returns, authored by the graph (`dfo_analyst_graph`). The totals are equal; the planner's own rows are 9 against 6 on the one-ask turn, and the writes inside the planner's seam call are 9 INSERTs and 9 UPDATEs against 6 and 6.

16 tests pass, about 105 s a run, the container start included.

## Production wiring

`harness.session_store.build_session_service` returns `DatabaseSessionService(db_url=...)` when `DATABASE_URL` is set, or one over an async engine on the Cloud SQL connector. `harness.server` builds a `StudioStore(session_service, app_name=AGENT_NAME)` and a `StudioRunLedger(store)` over it. `rig.build` wires the same three pieces over the Postgres URL of the test database, with the real root workflow behind the real Studio routes.

Each conversation owns more than one ADK session:

- the turn session `{conversation_id}#t{n}`, app `dfo_analyst_graph`, one per turn, holding the orchestrator's and the planner's events;
- the ledger session `{conversation_id}`, app `dfo_analyst_graph.ledger`, holding `studio:*` keys: the lease (`studio:segment`), pending interrupts, `studio:turn:NNNN` records, approvals and the composition.

The Pydantic AI node and dev's node write the same turn session through the same service. The node's events are ordinary ADK `Event` rows (`event_data` is `JSONB`).

## Pass lines

Each line runs on the `adk` arm (dev's node) and the `pai` arm. A process of its own is a pytest child (`segment_child.py`) that gets the database URL through the environment and shares nothing else with its parent.

### 1. Rows, bytes and writes of a planned turn

The turn is a skill read, an ask, the answer and a proposal: two requests (`/turn`, `/interrupt`), one script (`skill-ask`).

|                                                | dev's node | Pydantic AI node | difference                |
| ---------------------------------------------- | ---------- | ---------------- | ------------------------- |
| event rows, turn session                       | 26         | 26               | 0%                        |
| of which authored by the planner               | 6          | 9                | +3 (the progress reports) |
| event rows, ledger session                     | 6          | 6                | 0%                        |
| event bytes, JSON text                         | 204,252    | 203,559          | -0.34%                    |
| event bytes, stored                            | 70,314     | 69,016           | -1.85%                    |
| state bytes, JSON text                         | 113,934    | 113,934          | 0%                        |
| state bytes, stored                            | 28,406     | 28,406           | 0%                        |
| INSERT into `events`, turn session             | 26         | 26               | 0%                        |
| UPDATE of `sessions`, turn session             | 27         | 27               | 0%                        |
| INSERT / UPDATE, ledger session                | 6 / 7      | 6 / 7            | 0%                        |
| INSERT / UPDATE inside the planner's seam call | 6 / 6      | 9 / 9            | +3 / +3                   |

The byte figures move by a few hundred bytes between runs (each event carries minted ids); the state size does not.

A second turn with four planner responses (three skill reads, then a proposal) gives 26 rows on both arms, event bytes -0.54%, stored bytes -2.14%. Each tool response costs both arms one progress row, so the equality holds as runs grow.

How it is measured. The rows come from SQL run through `asyncpg`, apart from ADK: `count(*)` per session of `events`; event bytes as `sum(octet_length(event_data::text))` (the JSONB rendered as text) and `sum(pg_column_size(event_data))` (what Postgres stores, compressed where it compressed); state bytes as `octet_length(state::text)` and `pg_column_size(state)` of the `sessions` row; authors as `event_data->>'author'`; progress reports as `event_data->'custom_metadata'->'progress'`. The writes are the INSERT, UPDATE and DELETE statements the service's engine ran, counted by a SQLAlchemy `before_cursor_execute` listener (`rig.WriteLog`) and told apart by the `app_name` among the bound parameters. The count of INSERTs into `events` (26) equals the rows. The seam call is bracketed by wrapping `PlannerSeam.__call__`, which both arms run (the Pydantic AI seam calls it).

Control: a node that writes each function-call event twice (`pai_node.call_events` doubled). Its rows and inserts leave the 5% band. Tolerance: 5% on every metric above except the planner-authored and in-window counts, which differ by design.

### 2. Resume after an ask from a fresh process

Process 1 runs the turn to the ask (two planner requests, the skill read and the ask) and exits on `elicitation.request`. Process 2, started afterwards, posts the answer to the stored interrupt id and runs one planner request to `plan.proposed`. On both arms the frames, the key set of the turn's state and the plan equal an uninterrupted run on the in-memory service. The processes' pids differ from each other and from the test's.

Control: process 2 pointed at another empty database in the same server. The route answers 404 (no such conversation) and the frames differ.

### 3. A kill mid-planning

The process is killed with SIGKILL (return code -9) while the first `propose_plan` is inside its translation. At that moment the turn session holds the planner rows `call get_skill`, `response get_skill`, `call propose_plan` (the call row is committed before the tool runs, on both arms) besides the orchestrator's four.

First turn, killed. Equal on both arms in every field:

| step                                                                             | result                                                                                                                       |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `/turn` sent at once                                                             | 409 `transient_unavailable`, `retry_after_seconds` 43 to 44 (the rest of the 45 s lease); no planner request, no translation |
| `/turn` after the lease (the second process's `StudioStore` clock is 60 s ahead) | 200, a new turn `#t2`, 2 planner requests (skill read, proposal), 1 translation, ends on `plan.proposed` version 1           |
| session `#t1` afterwards                                                         | unchanged: the proposal call has no response                                                                                 |
| ledger                                                                           | turn 1 recorded with `error: abandoned`                                                                                      |

So the turn restarts and cannot continue. Tools run twice in the sense that the skill read runs in both sessions and `propose_plan` starts in both, the first never returning. The user sees a dead stream, then 409 with a retry delay, then a complete new planning run.

Resumed run, killed. The turn asked, was answered, and the second process died in the resumed run's translation. The answer sent again at once gets 409; after the lease it gets 200 and `plan.proposed`. Everything stays in session `#t1`: one skill read, one ask, no new turn, and the planner rows end with two `propose_plan` calls and one response. The user sees the plan of the original turn. `propose_plan` is translated twice (the dead attempt and the retry); the skill read and the ask run once.

The Pydantic AI node differs from dev in none of these results. The property it needs is that each call event is stored before its tool runs, which it does, since `FeedingRunner.events` yields the call when the model response arrives.

Controls: the same sequences with the first process left to finish. The next `/turn` is then 200 and not 409 (turn), and the repeated answer is 404 (resume), and the checks that read the wreckage (the unanswered proposal call, the 409, the second proposal call) fail.

### 4. A second turn of the same conversation

One app, one database, five requests: turn one asks, is answered and proposes (version 1), is approved (`plan.approved`, then an `error` frame, because `execute` has no tool surface in these tests); turn two proposes (version 1) and is amended with "only 2020" (version 2).

- Turn two has a session of its own, `conv_01test0000#t2`, so its planner record holds its own steps only. The readers `_session_steps`, `elected_spec_ids`, `combination_picks`, `answers_since_referral` and `prices_chosen` over it equal dev's, and its steps hold neither turn one's question nor its answer. Turn one's steps do.
- Plan versions run `[1]`, `[1]`, `[2]`: the plan counts from one in turn two.
- The amendment's words reach the amendment's request. All four planner requests equal dev's apart from the listed wire differences (`parity.WIRE`).
- What carries across turns: the conversation ledger session (`studio:turn:0000` record, approvals, composition), and two keys the second turn's session is created with, `conversation_digest` and `orchestrator:prior_ledger` (one earlier search reading). Turn one's answer reaches turn two's first request only through the conversation digest. Both arms carry the same keys: 37 state keys in turn one, 39 in turn two.
- A second turn cannot amend turn one's plan: starting a turn clears the pending plan interrupt (`ConversationState.begin_turn`). An amendment in turn two amends turn two's own plan, which is what the test drives.

Control: a planner record built from both turns' events, as one session over the conversation would hold them, fails the clean check.

### 5. Cross-path on Postgres

The turn proposes, asks whether two flows add up, and is answered. The ask pauses in one process under one planner; the answer is posted in a new process under the other. Both orders (dev's then Pydantic AI, Pydantic AI then dev's) land the plan, with frames, key set and plan equal to an uninterrupted dev run on the in-memory service.

Controls: a Pydantic AI node that reads none of the earlier steps (`_earlier` replaced) answers the ask but loses the combination pick, and no plan lands. A Pydantic AI first half that stores no events (`events=False`) leaves dev's second half nothing to read, and no plan lands.

## Internals relied on

ADK 2.13.0: `DatabaseSessionService` (`db_engine`, `prepare_tables`, `list_sessions`, `get_session`, `close`), its `events` and `sessions` tables, `event_data` as JSONB with `author`, `custom_metadata` and `content.parts[*].function_call` / `function_response`, and `update_time` as the revision marker.

Harness: `StudioStore(clock=)`, `StudioRunLedger`, `SEGMENT_TTL_S`, the ledger's `studio:turn:NNNN` and `studio:segment` keys, `ConversationState.begin_turn`, `harness.graph.state.AGENT_NAME` and `ORCHESTRATOR_PRIOR_LEDGER_KEY`, `PlannerSeam.__call__` (wrapped for the in-window counts), `planner.agent._session_steps` and `AGENT_NAME`, `tests.orchestrator_fakes.orchestrator_app`.

From `node/` and `agent/`: `studio_arms.StudioResolver`, `orchestrator_script`, `TASKS`; `pai_node.pai_planner`, `PaiPlannerSeam._earlier` and `call_events` (replaced in controls), `scenes`, `streaming.AUTH`, `parity.WIRE`, `scripted`.

Postgres: `postgres:18-alpine` in Docker on `127.0.0.1` and a free port. The password is generated per test session, passed to Docker and to the children through the environment, and written nowhere. The container is `--rm`, named `pai-sessions-<hex>`, and removed at the end of the session.

Driver: `asyncpg`, already in the harness's dependencies. No `--with` beyond `pydantic-ai-slim[google]==2.66.0` is needed.

## Not covered

- The Cloud SQL connector path (`async_creator`, IAM authentication, pool sizes). The URL path is used, with ADK's own engine defaults.
- Two live instances. The lease is taken over by a second process whose clock is 60 s ahead; no first process is alive to be fenced (`SegmentLost`).
- A kill at another point: during a model stream, between a response and its tool, during a state write. The kill lands in the translation of `propose_plan`.
- Server-side counters (`pg_stat_*`). The write counts are the statements the client sent.
- A turn that executes. The approval in line 4 ends on an `error` frame because `execute` has no tool surface.
- Live Gemini; every model answer is scripted.

## Files

- `pg.py`: the throwaway Postgres.
- `rig.py`: the Studio app over a given session service, the write counter, the SQL inspection.
- `segment_child.py`: one process's share of a conversation.
- `test_session_service.py`: the 16 tests.
- `conftest.py`, `pytest.ini`: the harness suite's fixtures and path, and the `server` and `url` fixtures.

## Commands

From the worktree root:

```
cd research/pai-planner-live/session-service
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
```

Add `-s` to print the `REPORT` lines behind the tables.

```
uv run --package harness ruff format research/pai-planner-live/session-service
uv run --package harness ruff check research/pai-planner-live/session-service
```

Docker must be running; the first run pulls `postgres:18-alpine`.
