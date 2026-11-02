# Plan-and-run MCP tools for outside models

The planner's code runs without its model behind four MCP tools, and an outside model can drive it. All 109 tests pass in 44 s. Every reference plan of the 70 planner-entry tasks goes through `propose_plan` over a FastMCP client and lands as the same plan, with the same content hash and translation, that dev's planner entry records. Recorded and mutated refused drafts answer with the same `outcome`, `code` and `action`. An accepted plan runs on dev's own `execute` and `plan_done` nodes and hands back what dev's orchestrator receives, run id aside. Sessions keyed on the id `start_plan` returns stay apart. An outside model reads about 63,600 characters before its first proposal, against 61,500 for DFO's planner today.

## Design

The server is `FastMCP("plan-mcp")` (`server.py`) over `PlanningService` (`plan_tools.py`). It imports harness code and changes none.

- `start_plan(brief, asks, searches)` opens a plan session. The client sends back each `search` call it made, the user's question and the `parts` it sent. The server runs those searches again through the orchestrator's own `search` tool (`harness.orchestrator_tools`), so the pin sources come from Search's answers and never from blocks or spec ids a client asserts. It hands `brief` and `asks` to the orchestrator's `plan` tool, runs the graph's real `route` and `plan_start` nodes on the state, and then runs `PlannerSeam.__call__` itself, stopped where it would call the model (`_EntrySeam._propose`). The seam writes the pin sources and every slot an entry resets, and renders the planner's first message. The answer carries the session id, the guidance and that first message. The cost of this choice is one repeat Search call per part. Its strength is that a client cannot widen what the plan may pin. The alternative of passing spec ids would need the server to fetch and verify each block and still lacks the readings (warnings, windows, settled measures) the planner's checks read.
- `propose_plan` is the served planner's own tool object, taken from the seam's agent and called through ADK's `FunctionTool.run_async`, with the planner's compacted declaration and `plan_session` added. The tool context is a namespace over the session's state and events. The call and its response are recorded as ADK events, which the planner's readers (`_combination_picks`, `_answers_since_referral`, `_user_spoke_since_recorded` and the rest) walk exactly as on dev. Arguments reach the tool raw, as ADK passes them. An accepted answer gains `card`: the approval card's title, summary, steps and asks from `harness.plan_display`.
- `record_user_answer` serves every refusal whose `action` names `adk_request_input`. The client asks its user in its own conversation, then sends the question, the options or candidates it offered and the answer. The planner's own ask tool checks the arguments, and the server records the `adk_request_input` call and answer events the readers pair. The action is left unchanged, so P2 compares it byte for byte. The guidance tells the client how to read that name.
- `run_plan(plan_session, version, wait_s=30)` runs a version once it was declared ready and while it is the plan on record. It stamps the approval now and runs dev's `execute` and `plan_done` nodes under an ADK `Runner` on a copy of the session's state (`execution.py`). It answers `AgentHandBack.response()`, the result of the orchestrator's `plan` call on dev. A run still going after `wait_s` answers `running` with its id, and the same call again waits on the same run, so a 45 s client or `/mcp` call limit needs no second tool.

Sessions live in memory, keyed on `"ps_" + token_urlsafe(16)` and bound to the caller in `x-dfo-caller`. FastMCP 4's `ctx.session_id` changes on every request, over HTTP and in memory alike (P4's control), so it cannot key a session.

The guidance reaches the model in `start_plan`'s answer: a 691-character preamble mapping the planner's tool names onto these, then the served planner instruction (20,138 characters) unchanged. Every client calls `start_plan` before proposing, so the guidance arrives once per session without depending on host support for prompts or resources. It is also served as the MCP prompt `plan_guidance`.

## Results

| Line | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Control, and how it fails                                                                                                                                                                                                                                                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1   | 70 of 70 tasks: plan on record with the same `plan_revision` and the same `model_dump`, the same translation dict, status `pending_approval`, the period reduction and kinds `validate.py` holds the task to, and the planner's first message contained in dev's first request. The three reference payloads marked `combination_unasked` return dev's refusal dict whole, then pass after `record_user_answer` records the combine answer. `validate.py` itself passes 70 of 70 on this tree.                                                                                                                                                                                                                                      | `growth-total-oda-ghana` with `period.start` set to `"twenty twelve"`: `not accepted: invalid_input/invalid_plan`.                                                                                                                                                                                                                                  |
| P2   | 21 refused drafts answer equal to dev's first `propose_plan` answer, whole dicts: the 4 recorded refused drafts (`combination_unasked`, action `adk_request_input` with the two combination options), 16 mutated drafts over 4 tasks (`asks_unaccounted` with action `propose_plan`, and `invalid_plan` with none, three ways), and the DRC `plan_v3`. The DRC `first_attempt` and `plan_v1`, refused when recorded, are now admitted on both paths with the same translation verdict (`discovery_unavailable`, no recording).                                                                                                                                                                                                      | A refusal with its code changed to `asks_unaccounted` is reported as `.code: 'asks_unaccounted' != 'combination_unasked'`.                                                                                                                                                                                                                          |
| P3   | Dev's canned plan, proposed through `propose_plan` and translated against `harness.scripted_plan`'s schemas, runs on `ScriptedSurface`. The hand-back (`result`, `frames`, `step_summaries`, `totals`, `asks`, `run`, `plan_record`) equals the one dev's root workflow gives its orchestrator for the same plan through `approve`, `execute` and `plan_done`, with only `run_id` removed. `run_plan` refuses an unknown session (`plan_session_unknown`), a version never declared ready (`plan_version_not_ready`, before any proposal and for version 2 after one), and version 1 once version 2 is accepted (`plan_version_superseded`). With `wait_s=0` it answers `running`, and the next call returns the same run finished. | With the version check off, `run_plan` version 1 runs the superseded plan: `finished`, plan record `period 1960-2023` while version 2 states 1990.                                                                                                                                                                                                  |
| P4   | Two sessions over two datasets, interleaved step by step on one in-memory client and on two HTTP clients with different callers: each holds only its own pin source, plan, version, events and run, and neither planner message names the other's dataset. Session a cannot run b's version 2. The second HTTP client gets `plan_session_unknown` for a's id on `propose_plan` and `run_plan`, and a's plan stays at version 1.                                                                                                                                                                                                                                                                                                     | Keyed on `ctx.session_id`: the call after `start_plan` finds no session, in memory and over HTTP, because the id changes per request. With a session id stable per connection (patched, as transports that hold one behave), one client's two sessions share one key: a's proposal lands in b's session and a's `run_plan` version 2 runs b's plan. |
| P5   | See below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | A schema with a bad name, a recursive `$defs`, `minLength` and `default` is flagged under each rule it breaks, and a minimal strict schema under none.                                                                                                                                                                                              |

## P5: what an outside model reads

Characters as compact JSON for schemas, tokens at four characters each. `measurements.json` holds the full numbers.

| Read before the first proposal                               | plan-mcp                                                                                                | DFO's planner today                                                                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Tool declarations                                            | 40,972 (4 tools: `propose_plan` 35,167, `start_plan` 3,598, `record_user_answer` 1,419, `run_plan` 788) | 39,630 (5 tools: `propose_plan` 35,719, `adk_request_input` 2,007, `get_skill_file` 1,045, `get_skill` 644, `reject_plan` 215) |
| Guidance                                                     | 20,829                                                                                                  | 20,138 instruction                                                                                                             |
| Planner's first message, median of 70 (min 1,223, max 3,484) | 1,755                                                                                                   | 1,755                                                                                                                          |
| Total                                                        | 63,556, about 15,900 tokens                                                                             | 61,523, about 15,400 tokens                                                                                                    |

The outside model also reads the public `search` and `get_skill` declarations, which it holds anyway, and its own search results. `propose_plan` is 86% of the declarations; its schema is the planner's own, unchanged.

Both vendors' non-strict tool modes accept all four declarations: names match `^[a-zA-Z0-9_-]{1,64}$`, each schema is a `type: object`, and none is recursive. Strict mode rejects three of four on Anthropic and all four on OpenAI, so plan-mcp's tools must be declared non-strict.

| Tool                 | Anthropic `strict: true`                                                                                                                  | OpenAI `strict: true`                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `propose_plan`       | 106 faults: objects without `additionalProperties: false`; `minimum`, `maximum`, `exclusiveMaximum`, `minLength`, `maxLength`, `maxItems` | 102 faults: `additionalProperties`, optional properties, `default`, `minLength`, `maxLength` |
| `start_plan`         | 4: `additionalProperties`                                                                                                                 | 8: `additionalProperties`, optional properties, `default`                                    |
| `record_user_answer` | 3: `additionalProperties`                                                                                                                 | 8: the same three                                                                            |
| `run_plan`           | none                                                                                                                                      | 2: optional properties, `default`                                                            |

The rules are the documented ones `schema_rules.py` lists, checked statically; no vendor API was called. `propose_plan` stays inside OpenAI's strict size limits (5,000 properties, 10 levels, 1,000 enum values), so a strict variant is a rewrite of keywords, not of shape. Gemini's rule that nothing sits beside a `$ref` already holds, because `planner.declaration.compact_schema` inlines those.

## What this relies on

- Harness, private: `PlannerSeam._propose` (overridden to stop the entry) and `PlannerSeam._agent`'s tools and instruction; ADK's `FunctionTool._get_declaration` for the two schemas; `harness.scripted_plan`'s `_scripted_source`, `_SCRIPTED_CAPABILITIES`, `_SCHEMAS_BY_NAME` and step ids (P3 only); `entry._no_anchor_read` (tests only).
- Harness, duck-typed: the planner's tools, the translator and `declare_ready` read a tool context's `state`, `session.events`, `session.id`, `actions`, `function_call_id` and `tool_confirmation`, and nothing else on the paths these tests drive. `route`, `plan_start` and `PlannerSeam.__call__` read `state`, `session.state` and `resume_inputs`. The orchestrator's `search` and `plan` tools are plain functions, called as `evals/planner_entry/entry.py` calls them.
- Harness behaviour: the round cap (`MAX_PLAN_ROUNDS`) lives in `submission.service`, so it binds an outside model by construction. The seam's own corrections, stall watchdog and closing refusals do not run; a client that stops proposing simply stops.
- ADK: `FunctionTool.run_async` for argument preprocessing, `Event`, `Runner` over a two-node `Workflow`, and `Event.actions.route`. Nothing else.
- FastMCP 4.0.10: a `Tool` subclass with its own `parameters` and `run`, which validates nothing, as ADK does not; `get_http_headers`; `ctx.session_id`.

## What a production version would change

- A shared session store, about 150 lines: sessions in Postgres keyed on the returned id and the caller, so the next call can land on another instance. State is JSON already; events need serialising, or the readers need a store of their own.
- Durable runs, about 100 lines: `run_plan` starts a run that outlives the instance and finds it again by session and version. `prototype-background-execute/` is the pattern, with the ledger lease it found missing. The 45 s `_CallLimit` on `/mcp` then only bounds each wait.
- Approval, about 40 lines: `run_plan` trusts the client's word that the user approved. A host that supports MCP elicitation could ask the user directly; otherwise an approval token shown in the card and echoed back at least proves the card was shown.
- Names and codes, about 40 lines: a `Proposer` for the ask so refusal actions name `record_user_answer`, and registration in dfo-spec-kit of the six codes this server writes: `plan_session_unknown`, `search_failed`, `hand_off_refused`, `plan_version_not_ready`, `plan_version_superseded` and `run_failed`.
- Mounting on `/mcp`, about 30 lines in `harness/combined_mcp.py`, under its middleware: budgets, rate limits, request logs, refusal shapes.
- The full approval card, about 40 lines: the decisions and requirements `harness.studio_translate` renders.
- A cap on reruns of `start_plan`, whose searches cost Search calls on our bill.

## Not covered

- Plans that read data. P3's run is the scripted plan; a P1 plan would need recorded `get_data` and analysis answers for its own spec ids.
- Referral asks with `candidates`, prices asks, amendments and plan-card switches. Only the combination ask was driven.
- Alternative payloads of the 70 tasks (validate.py also runs them).
- Several instances, a killed instance mid-run, and real client hosts.
- Live client models, which the next stage measures.

## For the live stage

Start a harness from this worktree against deployed dev data as `~/.claude/dfo-ai-e2e/README.md` describes, on :8080, then plan-mcp beside it, from this directory, under the same `.env`:

```
cd /Users/jorge/repos/dfo-ai-worktrees/pai-research/research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools
PYTHONPATH=.:../../../services/harness uv run --env-file <the harness .env> --package harness python serve.py --port 8091
```

`serve.py` reaches data-mcp and analysis-mcp at `Settings.data_mcp_url` and `analysis_mcp_url` (the :8080 mounts by default) with the harness's bearer token, builds production's translation, and runs approved plans with `McpExecutor` and an in-memory ledger. It was built and listed offline; it has not served a live call. A client then connects to `http://127.0.0.1:8091/mcp` beside the harness's public `/mcp`, which serves `search` and `get_skill`, and sends `x-dfo-caller`.

A fair sample by plan size and difficulty, 10 of the 70:

- Easy, one dataset requirement: `series-gdp-ghana`, `series-oda-germany-kenya`.
- Medium, two or three requirements: `growth-total-oda-ghana`, `aggregate-total-oda-kenya` (across years), `rank-top-donors-malawi`, `join-gdp-debt-service-ghana`.
- Hard, four or five requirements, an ask, or several searches: `sum-oda-oof-ethiopia` (refused `combination_unasked`, then an ask), `share-two-donors-of-kenya-oda` (across years), `quotient-oda-per-person-ethiopia`, `aggregate-health-population-oda-germany-by-sector` (four searches, four asks).

## Files

| File                                           | Purpose                                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `plan_tools.py`                                | `PlanningService` and `PlanSession`: opening, proposing, answering, starting runs |
| `execution.py`                                 | `dev_run_graph`: dev's `execute` and `plan_done` nodes under a `Runner`           |
| `server.py`                                    | `PlanMcp`: the four tools, the guidance and the session keying                    |
| `scripted.py`                                  | The canned plan as a proposal, its scripted translation and `FixedPlanner`        |
| `schema_rules.py`                              | The vendor schema rules P5 checks                                                 |
| `checks.py`                                    | Builders and the comparisons each test asserts on                                 |
| `serve.py`                                     | The server wired to a running harness                                             |
| `test_p1_reference.py` to `test_p5_reading.py` | One file per pass line                                                            |
| `measurements.json`                            | P5's numbers, written by its test                                                 |

## Commands

From the worktree root:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools -q -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools -q -s -p no:cacheprovider -k "control or p2 or p5"
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools
```

The second prints each control's failure, each refusal compared, and the P5 measurements.
