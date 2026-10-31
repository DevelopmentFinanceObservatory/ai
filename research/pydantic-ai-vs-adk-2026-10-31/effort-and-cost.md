# Pydantic AI or Google ADK: effort and cost of a move

Checked on 2026-10-31 against google-adk 2.13.0 (we pin 2.10.0) and pydantic-ai 2.64.0.

## Recommendation

Stay on ADK and upgrade to 2.13.0 now. Moving to Pydantic AI would remove most of our ADK workarounds, but Pydantic AI has no durable graph: `pydantic_graph.persistence` was removed in V2. ADK's `Workflow` is the part that pauses a turn and resumes it on another Cloud Run instance, so leaving means writing that turn loop ourselves and rewriting 10,000 to 12,000 lines of harness source and a third of the harness tests. The strongest case for moving is that an explicit loop of our own would be correct by construction where ADK's replay model needs a guard on every new node. That case is sound, but Pydantic AI does not supply the loop, and the measured cost of staying is about three ADK-driven commits a month. A five-day spike that moves only the planner would test the case for moving at low cost; whether to run it is the open decision.

## What ADK does for us today

- One `Workflow` of nine nodes (`harness/graph/workflow.py`) run by one `App` and `Runner` (`harness/studio.py:550`). A paused turn resumes in a later HTTP request by replaying the graph from START over events stored in Postgres.
- Pauses for plan approval and clarification, built on `RequestInput` and the private `_workflow_hitl_utils` module (six source imports).
- `DatabaseSessionService` on Postgres, which also serves as the compare-and-swap store behind the Studio ledger's leases, checkpoints and cross-instance cancel (`studio_state.py`, about 700 lines).
- The tool loop, callbacks and streaming events for the planner and orchestrator, the only two model-backed agents. The translator is plain code.
- OpenTelemetry spans that our Langfuse mapping, cache dashboards and session viewer read by name.

No MCP tool reaches a model through ADK. Both agents call our FastMCP servers through our own `fastmcp.Client` wrappers.

Coupling: 30 of 94 harness source files import `google.adk`, as do 35 of 101 test files, eight scripted `BaseLlm` fakes, four eval files and 18 `adk eval` cases. Details: `footprint-runtime.md`, `footprint-model-tools.md`.

## What a switch would fix, and what it would not

| Problem                                                                                      | Root cause     | On Pydantic AI                                                               |
| -------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------- |
| Tool arguments arrive unvalidated; a raise ended the turn (fixed in #387, #389)              | ADK            | Validated against the signature, with a retry prompt to the model            |
| Events with no branch reach every agent's prompt (`orchestrate.py:44-67`, `_check_branches`) | ADK            | Gone: each agent keeps its own message history                               |
| Resume replays from START, so new nodes need `_turn_ran_before` and `rerun_on_resume`        | ADK            | Gone only because we would write our own loop                                |
| `include_contents` pins, `_persist_durable`, one output per node                             | ADK            | Gone with our own loop                                                       |
| Private ADK imports that move between releases                                               | ADK            | No private imports needed for the same features (unverified until a spike)   |
| `ContextCacheConfig` keys on tools, so we built `PrefixCache`                                | ADK and Gemini | Pydantic AI manages no caches; `PrefixCache` moves with us                   |
| A description beside `$ref` is dropped                                                       | Gemini         | Stays: Pydantic AI sends `$ref` as is, so `compact_schema` must still inline |
| Explicit cache forbids tools and system instruction on the request, and caps implicit hits   | Gemini         | Stays: `google_cached_content` strips the same three fields                  |
| Thought signatures, `MINIMAL` thinking on flash, text-only `count_tokens`                    | Gemini         | Stays                                                                        |
| Tool results over 1 MiB dropped; FastMCP 4 session id per request                            | MCP            | Stays                                                                        |
| Turn budget and first-chunk resend (`BoundedGemini`)                                         | Ours           | Stays, re-hosted on a wrapper model                                          |

## What a switch would cost

- Removed: about 600 lines of workarounds.
- Rewritten: the run loop in `studio.py`, the event-reading core of `studio_translate.py` (about 900 lines), most of `planner_seam.py` (1,875), about 500 lines of `planner/agent.py` that read the user's answers from session events, the 93 `tool_context` sites, `PrefixCache` and `BoundedGemini`.
- Built new: a durable turn loop with a persisted cursor and one stored `Conversation` per agent, reproducing resume, cancel and lease behaviour (items 1-5, 11-13 and 17-18 in `footprint-runtime.md`).
- Data: the `dfo_analyst_graph.ledger` sessions migrate to our own table. Turns paused at cutover are dropped.
- Observability: Langfuse mapping, cache dashboards and the out-of-repo session viewer, all of which key on ADK span names.
- Tests: 35 test files, eight fakes, `test_graph.py` (2,708 lines) and the 18 `adk eval` cases.
- Unchanged: about 40,000 lines of business logic in `compiler/`, `executor/`, `planner/plan.py`, `mcp_client.py` and the wire code.

Pydantic AI brings its own risks. It shipped 111 releases in the last six months and reshaped graphs, retries, durability and MCP classes in V2, which is four months old. It has 1,229 open issues. The message history we would persist has no schema version, and the stability promise exempts new message parts and OpenTelemetry attributes. It defaults Gemini to `VALIDATED` function calling, which is untested on a schema the size of `propose_plan`. Its Langfuse mapping is documented only by Langfuse. Details: `currency-pydantic-ai.md`.

## What upgrading ADK buys

2.13.0 is three minor releases ahead. It fixes lost responses from parallel tool calls (#7428), leaks between single-turn runs (#7227, #7488) and several replay and pause bugs. Two things break: `tests/orchestrator_fakes.py:30` imports `flows/llm_flows/_fencing.py`, which 2.11.0 deleted, and the default telemetry schema moved to OTel GenAI 1.44, which drops the `call_llm` span. Setting `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` keeps the old spans until at least 2027-03-31. The session schema is unchanged. Estimate: a day plus a Langfuse trace check.

None of our ADK defects is fixed upstream. Unvalidated arguments (#7574), resume replaying nodes without output (#7666) and the tool-keyed cache fingerprint (#7241) are open. Branch leakage has no issue, so Google may not regard it as a defect. Details: `currency-adk.md`.

ADK is well resourced: 605 commits in October, a median of nine days to close an issue, and current Java, Go and TypeScript ports. Google has deprecated the old `SequentialAgent` family in favour of `Workflow`, which is the API we use.

## The measured cost of staying

Of 287 harness, eval and telemetry commits since history begins on 2026-07-17, 19 were ADK-driven: 7 workarounds, 2 version bumps and 10 mixed. By month that is 3 in August, 9 in September and 7 in October. Counting workarounds and bumps alone, it is three a month throughout. The share of commits fell from 18% in September to 3% in October, but only because total commits rose sixfold; the absolute count has not fallen. The three version bumps (2.6, 2.9.1, 2.10.0) needed no repair commit. Two defects plausibly reached users: tool failures ending the turn (#387, #389) and a cancelled turn hanging in session teardown (#260, partly SQLAlchemy and anyio). Details: `adk-cost-history.md`.

The consult set its threshold before the count: under one ADK commit a month and falling means a migration buys nothing; two or more a month justifies a spike. The count clears that line, narrowly, on a steady rather than rising rate.

## The spike, if run

Put a Pydantic AI planner behind the existing `PlannerModel` seam (`graph/seams.py:56`) inside the current ADK `planner` node. Its `ask_user` becomes a deferred tool that the node turns into a graph interrupt, and the planner's `Conversation` lives in its own table. The planner carries every hard part: pause, reads of the user's answers, the large declaration, thoughts, cache and budget. Measure the ADK planner on the same cases first. Five working days, and every line must pass:

- No planner_entry or scripted Studio eval case falls from pass to fail in a majority of three runs.
- Cached share of prompt tokens on the second and later calls is within 5 points of the ADK planner.
- The `propose_plan` declaration reaches Gemini identical to `compact_schema` output, replayed calls keep their thought signatures, and p50 time to first chunk is within 10%.
- A pause, process kill and resume in a fresh process gives the same plan as an unpaused run, and a `Conversation` stored under 2.64.0 loads under the newest release at the end of the week.
- Net removal of workaround lines on the planner path, with no private imports and no patches to Pydantic AI internals.
- One Langfuse generation per model call, with usage and cached tokens priced once.

A failed line ends the spike and we stay. A pass justifies moving the orchestrator the same way and then replacing `Workflow` with our own loop.

## Sources in this folder

- `footprint-runtime.md`: workflow, Runner, sessions, persistence and pause/resume.
- `footprint-model-tools.md`: agents, model overrides, tools, caching, tracing and evals.
- `currency-adk.md`: ADK releases since 2.10.0, upstream status of our defects, investment.
- `currency-pydantic-ai.md`: Pydantic AI 2.64.0 against eleven requirements.
- `consult-verdict.md`: independent review of the four reports and the code.
- `adk-cost-history.md`: commit-by-commit classification of ADK-driven work.
