# DFO's planner over the plan MCP tools

Verdict: no arm that puts the planner's tools behind plan-mcp passes the decision rule as fixed, but none of the failures comes from MCP. The transport adds 1 to 2 ms per call in process and 5 to 11 ms over loopback HTTP, under 0.2% of a 14 s entry. Each request the model reads is 410 to 1,053 tokens smaller than today's. The criteria that fail move with the number of rounds Gemini takes, and three repetitions cannot separate that from the arm: every B/A interval below holds 1.0. Moving the planner onto plan-mcp is not worth it for cost or latency, and it breaks the pause and resume path, the turn budget and the graph's hold on the plan record (below).

## Decision rule, fixed before any run

Arm B passes only if, against arm A on the same tasks: mean input tokens per planner entry, and mean output plus thinking tokens, are each no more than 2% higher; median wall time from the brief to the first accepted plan is no more than 5% higher and its 90th percentile no more than 10% higher; and the accepted and right counts are no lower. Right is the planner-entry eval's own rule (`summarise.reduce_run`): the task's kinds and years, or, on a task whose right ending is a refusal and an ask, that ending.

## Arms

- A: dev's planner entry, `entry.run_entry` as `evals/planner_entry/run.py` runs it, on `gemini-3.8-flash` with the prefix cache, replaying each task's recorded searches and skill reads.
- B-inproc and B-http: the same entry with the planner seam swapped (`mcp_planner.McpPlannerSeam`). Its agent, model, thinking level, callbacks, prefix cache, turn budget and skill tools are dev's. The seam calls `start_plan` with the orchestrator's brief, asks and searches, and runs the planner on the first message plan-mcp renders. The model's `propose_plan` and `record_user_answer` are plan-mcp's, reached through `compiler.surface.McpSurface` over FastMCP's memory transport or over Streamable HTTP to uvicorn on 127.0.0.1 in the same event loop, which is how the harness reaches its own mounted data-mcp.
- B-served (B'): B-inproc with `propose_plan` declared as dev declares it, bounds and defaults included, in place of plan-mcp's lean listing.

The client is `McpSurface`, not ADK's `McpToolset`, for two reasons. It is the harness's existing client for data-mcp and analysis-mcp, so it brings the held connections, the caller header and the Langfuse session meta. It also lets the tool hand the model the answer's `structuredContent` as a dict, as dev's tools do. `McpToolset` hands the model the whole `CallToolResult`, with the answer twice (text block and `structuredContent`), which doubles every result: 979 characters become 2,088 for the `combination_unasked` refusal, and 482 become 1,086 for an `invalid_plan`.

Every change to what the model reads, against dev:

- The ask tool is `record_user_answer`, as plan-mcp names it, with plan-mcp's description and parameters less `plan_session` and `answer`, which the harness fills. Calling it pauses the turn as dev's `adk_request_input` does. On resume the seam records the question and the user's answer through `record_user_answer`, `by: harness` (`HarnessPlanMcp`), before the planner runs again.
- The instruction names `record_user_answer` for `adk_request_input` (`outward.guidance_text`) and drops the sentence on `reject_plan`, which plan-mcp does not serve. `reject_plan` is gone from the tools.
- `propose_plan` is declared as plan-mcp lists it less `plan_session`: `outward.lean` removes bounds, defaults and `additionalProperties`. B-served keeps dev's declaration.
- Answers pass through `outward.outward`: refusals name `record_user_answer`, and `invalid_plan` gains `expected`.

Both arms stop at the first plan `propose_plan` records and translate nothing. In B, plan-mcp's translation is a stop marker (`ab._stop_translate`) the harness side ends the entry on, as dev's eval does through `entry.Probe.translate`. A B entry's wall time includes `start_plan`, which does the work dev's seam does in process before the first model call.

## Results

Run 1 (`runs/live/`): A, B-inproc and B-http, ten sample tasks, three repetitions, 90 entries one at a time, the three arms back to back in an order that rotates per task. Run 2 (`runs/live-served/`): A and B-served the same way, 60 entries, 30 minutes later. No entry errored or missed its recording. B/A intervals are 95% bootstrap intervals over tasks.

| Criterion                         | Limit | A, run 1           | B-inproc                        | B-http                          | A, run 2           | B-served                        |
| --------------------------------- | ----- | ------------------ | ------------------------------- | ------------------------------- | ------------------ | ------------------------------- |
| Mean input tokens per entry       | 1.02  | 45,740 (sd 20,773) | 42,265, 0.924 [0.79, 1.07] pass | 36,020, 0.787 [0.68, 0.90] pass | 45,562 (sd 15,384) | 48,560, 1.066 [0.96, 1.18] fail |
| Mean output + thinking tokens     | 1.02  | 645 (sd 611)       | 663, 1.029 [0.79, 1.30] fail    | 575, 0.891 [0.68, 1.27] pass    | 649 (sd 471)       | 735, 1.133 [0.96, 1.26] fail    |
| Median wall s to accepted plan    | 1.05  | 14.2               | 13.4, 0.946 [0.77, 1.35] pass   | 14.1, 0.989 [0.73, 1.51] pass   | 12.2               | 9.8, 0.808 [0.66, 1.31] pass    |
| p90 wall s to accepted plan       | 1.10  | 36.1               | 37.2, 1.030 [0.52, 1.43] pass   | 34.4, 0.953 [0.64, 1.61] pass   | 27.6               | 30.8, 1.118 [0.46, 3.23] fail   |
| Accepted                          | >= A  | 29/30              | 29/30 pass                      | 27/30 fail                      | 29/30              | 29/30 pass                      |
| Right                             | >= A  | 23/30              | 21/30 fail                      | 20/30 fail                      | 21/30              | 21/30 pass                      |
| Model calls per entry             |       | 2.67               | 2.67                            | 2.37                            | 2.67               | 2.87                            |
| Entries closed by the turn budget |       | 0                  | 1                               | 3                               | 0                  | 0                               |

Per task: right, mean input tokens and median wall time.

| Task                                                | A                  | B-inproc           | B-http             | A, run 2           | B-served           |
| --------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| `series-gdp-ghana`                                  | 3/3, 32.9k, 12.9 s | 3/3, 30.8k, 11.4 s | 3/3, 30.8k, 14.1 s | 3/3, 32.9k, 5.9 s  | 3/3, 32.1k, 7.3 s  |
| `series-oda-germany-kenya`                          | 3/3, 32.9k, 7.5 s  | 3/3, 30.8k, 10.7 s | 3/3, 30.8k, 7.5 s  | 3/3, 32.9k, 5.9 s  | 3/3, 32.1k, 5.7 s  |
| `growth-total-oda-ghana`                            | 3/3, 38.9k, 14.2 s | 0/3, 25.5k, 9.8 s  | 0/3, 25.8k, 90.1 s | 0/3, 38.9k, 11.7 s | 0/3, 38.0k, 7.4 s  |
| `aggregate-total-oda-kenya`                         | 3/3, 38.8k, 11.6 s | 3/3, 47.8k, 10.1 s | 2/3, 26.1k, 32.7 s | 3/3, 44.8k, 10.2 s | 3/3, 43.9k, 14.6 s |
| `rank-top-donors-malawi`                            | 3/3, 51.0k, 15.6 s | 3/3, 47.9k, 21.3 s | 3/3, 42.2k, 10.9 s | 3/3, 51.0k, 13.8 s | 3/3, 49.8k, 12.1 s |
| `join-gdp-debt-service-ghana`                       | 2/3, 33.6k, 7.4 s  | 3/3, 37.3k, 8.3 s  | 3/3, 25.9k, 9.8 s  | 3/3, 33.6k, 7.9 s  | 3/3, 32.7k, 6.0 s  |
| `sum-oda-oof-ethiopia`                              | 0/3, 51.6k, 11.3 s | 0/3, 43.0k, 12.5 s | 0/3, 48.9k, 18.2 s | 0/3, 52.1k, 12.2 s | 0/3, 44.7k, 9.8 s  |
| `share-two-donors-of-kenya-oda`                     | 3/3, 70.5k, 27.9 s | 3/3, 78.5k, 29.2 s | 3/3, 54.7k, 13.6 s | 3/3, 63.8k, 20.9 s | 3/3, 87.9k, 37.5 s |
| `quotient-oda-per-person-ethiopia`                  | 3/3, 33.1k, 13.0 s | 3/3, 36.7k, 14.5 s | 3/3, 36.7k, 20.3 s | 3/3, 39.0k, 8.3 s  | 3/3, 43.9k, 30.8 s |
| `aggregate-health-population-oda-germany-by-sector` | 0/3, 74.0k, 37.1 s | 0/3, 44.5k, 30.7 s | 0/3, 38.1k, 38.5 s | 0/3, 66.6k, 28.0 s | 0/3, 80.6k, 24.1 s |

`runs/live/report.md` and `runs/live-served/report.md` hold every run's ending, model calls and wall time.

### Reading the failures

- Right, run 1. The gap is `growth-total-oda-ghana`: A drew `dataset`, `aggregate` and `growth` three times, and every B plan drew `dataset` and `growth` with no `aggregate`. Thirty minutes later A drew `dataset` and `growth` three times out of three too, and so did B-served. The arm does not decide it; Gemini's choice on that task moved during the session. Every other task scores the same in every arm, except `join-gdp-debt-service-ghana` (A 2/3 in run 1).
- Accepted, B-http. Four B entries in run 1 ran out the turn's 90 s of model time, three of them on `growth-total-oda-ghana`, each on a model call that never answered: the second call in three, the first in one. A had no closure in 60 entries, B-served none in 30, B-inproc one. A stall inside a model call does not depend on the transport, which by then had already answered. Three of six lean-declaration entries on `growth-total-oda-ghana` stalling is still the one result that could point at the lean declaration, and B-served's three clean entries on that task are too few to rule it out.
- Tokens. A request in B reads less than a request in A. The first request of `series-gdp-ghana` is 15,826 tokens in A, 14,773 in B (lean declaration, smaller ask, no `reject_plan`) and 15,416 in B-served. Mean prompt tokens per model call: A 17,153 and 17,086, B-inproc 16,050, B-http 15,891, B-served 16,940. An entry's total follows its number of calls, and that varies by task and repetition: `share-two-donors-of-kenya-oda` took 3, 5 and 3 calls in A and 6, 4 and 5 in B-served, which alone moves B-served's mean by about 2,700 tokens.
- p90 wall time, B-served. Two slow entries, 79 s with 3 model calls and 68 s with 5; the interval runs from 0.46 to 3.23.

What would close each, and its cost:

- Per-request tokens need nothing; they already fall. Per-entry means at a 2% bound need many more entries: at the spread seen here, the interval on the ratio is about 25 points wide at 30 entries per arm. A paired criterion on tokens per model call, or ten repetitions, would separate an arm effect of a few per cent.
- Accepted and right: B-served, the served declaration over MCP, matched A on both (29 and 21 of 30). It costs 643 more tokens per request than the lean listing.

## What the model sees

Characters as compact JSON of each declaration (`measure.declarations`), and of the answer to the same call (`measure.results`).

| Declaration                                     | A      | B                        |
| ----------------------------------------------- | ------ | ------------------------ |
| `propose_plan`                                  | 35,719 | 32,177 (B-served 35,719) |
| ask (`adk_request_input`, `record_user_answer`) | 2,007  | 653                      |
| `reject_plan`                                   | 215    | none                     |
| `get_skill`, `get_skill_file`                   | 1,689  | 1,689                    |
| Instruction                                     | 20,138 | 20,055                   |
| Total                                           | 59,768 | 54,574                   |

| Answer                                                                                    | A           | B                    | B through `McpToolset` |
| ----------------------------------------------------------------------------------------- | ----------- | -------------------- | ---------------------- |
| `propose_plan`, `combination_unasked` (`sum-oda-oof-ethiopia` reference)                  | 977         | 979                  | 2,088                  |
| `propose_plan`, `invalid_plan` (`growth-total-oda-ghana`, `period.start` "twenty twelve") | 380         | 482, adds `expected` | 1,086                  |
| `get_skill`                                                                               | as recorded | same object          |                        |

The 5,194 characters B drops are the 1,053 tokens its first request saves. The answers differ only where `outward` changes them.

## Time per call

Medians over the run's calls. Overhead is the client's time less plan-mcp's own time on the call (`HarnessPlanMcp.server_times`).

| Arm      | Tool                   | Calls | Call to response | MCP overhead |
| -------- | ---------------------- | ----- | ---------------- | ------------ |
| A        | `propose_plan`         | 46    | 9.7 ms           |              |
| B-inproc | `propose_plan`         | 47    | 12.1 ms          | 2.1 ms       |
| B-inproc | `start_plan` (harness) | 30    |                  | 1.0 ms       |
| B-http   | `propose_plan`         | 38    | 19.0 ms          | 11.0 ms      |
| B-http   | `start_plan` (harness) | 30    |                  | 5.1 ms       |

An entry makes one `start_plan` and one to three `propose_plan` calls, so MCP costs it 3 to 7 ms in process and 16 to 38 ms over HTTP, against median entries of 10 to 14 s.

`run_plan` against dev's `execute` and `plan_done` nodes on the canned plan (`scripted.PROPOSAL` on `harness.scripted`'s surface, 30 runs each, interleaved, `measure.run_plan_latency`): dev's nodes in process 17.95 ms median (p90 19.2), `run_plan` in process 21.45 ms (p90 23.1), `run_plan` over HTTP 25.02 ms (p90 25.9). `run_plan` runs the same two nodes, so the difference is the MCP call, the approval record and `run_data`.

## Code that would go and come

Nothing leaves the codebase: plan-mcp imports the planner's tools, the seam and dev's `execute` node and runs them behind MCP. What leaves is the harness process's own path to them. Spans at dev 190c7a9c (`measure.code_lines`):

- `planner/agent.py`: `build_planner_tools` 227 lines (the `propose_plan` closures, `propose_plan_reading`, `propose_plan_translating`, `ask_user`, `reject_plan`), `ToolContextSession` 64, `_execution_blocked` 54, `declare_ready` 24, `_AskTool` 21, `_approval_summary` 20.
- `harness/planner_seam.py`: `PlannerSeam.__call__` 235, `_persist_durable` 71, `revise` and `_revised` 91, `_reading_pin_sources` 49, `store_searched_pin_readings` 35, `_readings_ledger` 15. `revise` has no plan-mcp tool yet, so it stays until one exists.
- `harness/graph/execute.py`, 659 lines, if `run_plan` replaced the graph's `execute` node; plan-mcp's `execution.py` runs that same node.

What comes: plan-mcp itself (`plan_tools.py` 630, `server.py` 524, `outward.py` 268, `execution.py` 108, `store.py` 109, about 1,640 lines), Postgres sessions (about 80 more, per `../NOTES.md`), and the harness side of this prototype (`mcp_planner.py`, 378 lines with docstrings and the timing). The served seam would be about 200 lines. The harness would carry more code, not less.

## What breaks or gets harder

- Pause and resume. Dev recognises a planner pause by the name `adk_request_input`: `harness.studio_translate` reads it, the Studio's interrupt route answers it under that name (`create_request_input_response`), and so does the eval's `run_entry`, which `ab.run_b` patches for B. Under plan-mcp's name every one of those changes. The seam's own asks, a referral the user decides and a turn budget that closes with candidates (`_ask_response`), emit `adk_request_input` calls, which B's agent has no tool for. A resume costs one more MCP call before the planner runs. Dev's ask tool refuses a malformed ask before the turn pauses; under B, plan-mcp checks it only when the answer is recorded, so the user sees it first.
- The plan record. Dev's graph holds the plan in its own session: the approval card, `execute`, `plan_done`, `revise` and the next turn's amendment read it there. Over MCP the harness gets the version, revision and card, not the plan. Approval and execution have to move to `run_plan`, and `revise` (a card switch with no model call) needs a plan-mcp tool of its own.
- The turn budget. Dev's translation inside `propose_plan` runs within what is left of the request's wall-clock bound (`discovery_time_left`), and a cut `resolve_definition` is kept for the next proposal in the entry (`CutCalls`). plan-mcp opens a fresh `TurnBudget` and a fresh `CutCalls` on every call (`PlanSession.request_scope`, `PlanningService.propose`), so a proposal can spend a full budget past the turn's own. Fixing it means sending the turn's deadline in `_meta` and keeping cut calls on the plan session.
- Isolation and state. A plan session is bound to the caller `McpSurface` sends, which is none over the memory transport. A turn that pauses across requests needs the plan session to outlive the process, so plan-mcp's sessions move to Postgres beside the harness's own, and two stores now hold one turn.
- `ToolId.server` admits only `data-mcp` and `analysis-mcp`; `PlanClient` builds its ids with `model_construct`, and a served version adds `plan-mcp` to the literal.
- Tracing. Not measured. `McpSurface` sends the turn's Langfuse session only over loopback HTTP (`session_call_meta`), so in process plan-mcp's spans would sit outside the turn's trace unless the mount carries it.

## Tests

`test_b_matches_a.py`, offline with validate.py's scripted model, both transports, 46 tests in 22 s:

- On each of the ten tasks, the reference plan proposed through B, with the lean and the served declaration, records the same plan revision as A, and the planner's first request carries the same text.
- On `sum-oda-oof-ethiopia`, B proposes, is refused `combination_unasked`, asks, pauses, resumes with the user's answer, and the second proposal lands; plan-mcp holds one answer, `by: harness`, the user's pick. Control: with the resume recording nothing, the second proposal is refused `combination_unasked` again and no plan lands.
- Control: a session opened without the orchestrator's searches shows the planner another first request.

## What this relies on

- Dev's eval: `entry.run_entry`, whose module-level `build_seam`, `REQUEST_INPUT_FUNCTION_CALL_NAME` and `create_request_input_response` `ab.run_b` swaps for a B entry; `entry.Probe`, `entry.entry_record`, `summarise.reduce_run`, `validate.ScriptedLlm`, `run.revision`.
- Harness, private: `PlannerSeam._propose`, `_refused`, `_agent` (its tools and instruction are replaced after construction); `compiler.surface.McpSurface` and `ToolId.model_construct`.
- plan-mcp: `PlanMcp.start`, `propose` and `record_user_answer` (overridden in `HarnessPlanMcp`), `PlanningService`, `outward`.
- ADK: a `BaseTool` with `is_long_running` set pauses the turn when it returns `None`; `Event.timestamp` for tool times.

## Not covered

- Amendments, replans, plan-card switches and later turns: the seam drafts a turn's first plan only.
- Translation and the run inside an entry: both arms stop at the recorded plan, as dev's eval does.
- Concurrent entries: one plan-mcp serves every B entry from one recording at a time (`ab.Current`).
- Langfuse traces for B.

## Commands

From `research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/harness-ab`, with `P=.:..:../live:../../../../evals/planner_entry:../../../../services/harness`. No server or proxy is needed: both arms replay the eval's recordings.

```
PYTHONPATH=$P PYDANTIC_AI_NO_BANNER=1 uv run --env-file /Users/jorge/repos/dfo-ai/.env --package harness python run_ab.py --reps 3 --out runs/live
PYTHONPATH=$P PYDANTIC_AI_NO_BANNER=1 uv run --env-file /Users/jorge/repos/dfo-ai/.env --package harness python run_ab.py --arms A,B-served --reps 3 --out runs/live-served
PYTHONPATH=$P uv run --package harness python report.py runs/live
PYTHONPATH=$P PYDANTIC_AI_NO_BANNER=1 uv run --env-file /Users/jorge/repos/dfo-ai/.env --package harness python measure.py
```

The tests and lint, from the worktree root:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/harness-ab -q -s -p no:cacheprovider
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/harness-ab
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/harness-ab
```

A run of 90 entries takes about 25 minutes one at a time. `runs/smoke/` is a three-entry check of the rig.
