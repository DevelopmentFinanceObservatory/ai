# Google ADK 2.13.0 as an architecture for a multi-instance, human-in-the-loop data-analysis agent

Read at tag v2.13.0 (clone at `/private/tmp/claude-501/-Users-jorge--claude-spares-spare-vOcnuX/65792ff6-bdbb-4dff-9c17-171a37d2a0ca/scratchpad/adk-src`). Paths are under `src/google/adk/` unless they start with `docs/`, `CHANGELOG.md` or `contributing/`. Nothing was executed. "Unconfirmed" marks a claim read from code or an issue but not tested. Our own usage is in `footprint-runtime.md` and `currency-adk.md` in this directory and is not repeated.

Summary: ADK is a session-event log with a graph scheduler on top. Everything durable is an `Event` row; resume is replay of those rows. That makes pause, resume and cross-instance hand-off cheap to build and hard to make exact. Its limits are replay cost proportional to session length, session-level optimistic locking with no merge, and a context model whose branch rule favours leaking over hiding.

## 1. Execution model

A run is one `Runner.run_async` call, one `invocation_id` (`runners.py:1129`). Any root, whether a `Workflow` or a chat-mode `LlmAgent`, goes through `run_node_async` (`workflow/_node_runner_utils.py:55-383`). That function loads the session, appends the user event, starts the root node as a background task, and drains an in-process event queue. Each non-partial event is appended to the session service before it is yielded (`runners.py:798-860`).

Control flow has three forms. A `Workflow` takes a declarative edge list: chains, nested-tuple fan-out, dictionary routing on `Event.route`, conditional cycles, and nesting (`docs/guides/workflow/workflow/index.md`). An `LlmAgent` runs a model-and-tool loop capped by `RunConfig.max_llm_calls` (500; `agents/run_config.py:271`). Plain code runs as a function node, and `ctx.run_node()` schedules children dynamically with no edge (`docs/guides/workflow/dynamic_nodes/index.md`). Agents compose as tools, as transfer targets, or as nodes.

State lives in three places: `Session.state`, the event log, and per-key scopes by prefix (`app:`, `user:`, `temp:`, none). A write through `ctx.state` becomes `actions.state_delta` on the node's next event and is applied in `append_event` (`docs/guides/sessions/state/index.md`). Nothing is durable until it rides on a non-partial event.

Judgement. The single-log design means one mechanism (append an event) serves history, state, resume and tracing, which keeps the model small. The costs are that a node has exactly one `output` and that anything useful (progress, side channels) must ride on an event field. The log is also the only source of truth, so a state write that never gets an event is lost on a crash. Growth in users is cheap, since sessions are independent rows. Growth in session length is the dimension that hurts (sections 3 and 4).

## 2. Context engineering

A model request is built by an ordered list of request processors: basic, auth, confirmation, instructions, identity, compaction, interactions, contents, cache, planning, code execution, output schema (`flows/llm_flows/single_flow.py:36-80`). `_get_contents` (`flows/llm_flows/context/_contents.py:473-670`) applies rewinds, drops other branches' events, folds compaction summaries, rewrites other agents' replies as quoted user text, and repairs call and response pairing.

Developer control sits at five levers. `include_contents` is `default` or `none` (`agents/llm_agent.py:437`); an agent used as a workflow node defaults to `single_turn`, which implies `none` (`:410-419`, `docs/guides/agents/llm_agent/single_turn.md`). `before_model_callback` and plugins can rewrite the whole `LlmRequest`; `ContextFilterPlugin` keeps the last N invocations (`plugins/context_filter_plugin.py`). `RunConfig.model_input_context` adds unpersisted context for one run. `static_instruction` moves the dynamic `instruction` into user content so the static part can be cached (`llm_agent.py:325-395`). A custom `BaseLlm` sees the final request.

Branch semantics are the weak point. `_is_event_belongs_to_branch` returns true whenever either side has an empty branch (`_contents.py:768-781`), so an unbranched event reaches every agent. Isolation is therefore opt-in per event, by `branch`, `isolation_scope` (`events/event.py:134`) or sub-branch. Task and single-turn agents get an isolation scope; parallel workers get `parent@1/worker@N` sub-branches (`docs/guides/workflow/parallel_worker/index.md`).

Compaction is an LLM summary appended as an event with `actions.compaction`; raw events stay stored. `EventsCompactionConfig` offers a sliding window (interval and overlap) or a token threshold with an event-retention count (`apps/_configs.py:52-100`). The window runs after an invocation (`runners.py:929-960`); the token trigger also runs before a request (`flows/llm_flows/context/_compaction.py`). Summaries are isolated by `isolation_scope` as of 2.13.0 (`CHANGELOG.md`).

Caching: `ContextCacheConfig` on the `App` enables Gemini explicit caches (`apps/app.py:86`). Caching starts on the second turn and needs 2,048 tokens (Gemini 2.5) or 4,096 (Gemini 3) (`agents/context_cache_config.py`). The fingerprint covers system instruction, tools, tool config and the first N contents (`models/gemini_context_cache_manager.py:310-345`), so any tool-list change invalidates the cache. Our earlier finding that an explicit cache blocks implicit caching is Gemini API behaviour and stays.

Judgement. Contents assembly is a fixed pipeline you can intercept but not replace as a unit. Per-agent control is good where agents are single-turn or task-scoped and weak where history is shared, because the default is to show. A design that needs each model to see a curated slice must stamp every event with a branch and test for leaks. Growth in agents multiplies that stamping burden; growth in history is handled by compaction, which costs an extra model call and loses detail.

## 3. Durability and failure

Pause. A node yields `RequestInput`, which becomes a long-running `adk_request_input` function call whose `id` is the `interrupt_id` (`workflow/utils/_workflow_hitl_utils.py:105`, `docs/guides/events/request_input/index.md`). Tool-level pauses use `LongRunningFunctionTool` and `ToolConfirmation` (`tools/tool_confirmation.py:25-40`). The run ends with the node in `WAITING`.

Resume. The caller sends a new `run_async` whose message holds only `function_response` parts. `_extract_resume_inputs` pulls them out and `_resolve_invocation_id_from_fr` maps them back to the paused invocation (`runners.py:613-690`, `_node_runner_utils.py:85-120`). The root node then runs again from the top. `check_interception` fast-forwards any node with a recorded outcome, reruns nodes flagged `rerun_on_resume` or `wait_for_output`, and keeps unresolved nodes waiting (`workflow/utils/_replay_interceptor.py:57-206`). `ReplayManager` indexes every event in the session across all invocations, extending the index incrementally and rebuilding it after a rewind or compaction (`workflow/utils/_replay_manager.py:54-120`).

Crash mid-turn. `ResumabilityConfig(is_resumable=True)` is still `@experimental` and its docstring promises "best-effort" resume with at-least-once tool calls and no in-memory state (`apps/_configs.py:29-45`). In the Workflow path it adds only checkpoint events (`agent_state`, `end_of_agent`) (`workflow/_workflow.py:716-760`). For `LlmAgent` loops it lets `decide_step_resume` replay a persisted call that never produced a response, and keep a branch paused (`flows/llm_flows/core/_resume.py:320-440`). The legacy agent path refuses to run without a `new_message` unless resumable (`runners.py:1276-1283`). Nothing in ADK detects a crash, owns a work queue, or re-dispatches; the caller must hold the `invocation_id` and call `run_async(invocation_id=...)`. A Workflow root with no message appears to accept this (`_node_runner_utils.py:75-110`); unconfirmed by test.

Idempotency. ADK has no idempotency keys. Completed nodes are not re-executed, because their outputs come from the log. A tool call whose event persisted but whose response did not can run twice under `is_resumable`. A `rerun_on_resume` node re-executes its body, so side effects before its `ctx.run_node` calls repeat. Partial streaming events are never persisted (`runners.py:854`), so an in-flight model call is simply lost.

Resume cost. `_get_or_create_session` calls `get_session` with `RunConfig.get_session_config` (`runners.py:966-1007`). With no config, `DatabaseSessionService.get_session` selects every event for the session (`sessions/database_session_service.py:730-800`). `num_recent_events=0` skips events entirely, which a ledger-style caller can use (`:753`). The replay index is then O(events in session) per resume and per turn. Truncating the window with `num_recent_events` or `after_timestamp` would hide the paused invocation's events from replay; unconfirmed, but it follows from the index being built over `ic.session.events`. 2.12.0 and 2.13.0 report 40% faster first request and reduced event loading (`CHANGELOG.md`, not re-measured).

Judgement. Replay-from-log is simple and survives instance changes: any instance with the session row can resume. It is exact only for deterministic nodes; a node added after a pause runs live on resume, which is the defect behind our `_turn_ran_before` guard. There is no snapshot, so resume cost rises with session length; one session per turn, as we do, caps it, at the price of cross-turn context being rebuilt by hand.

## 4. Concurrency and scale

Parallel tool calls. All calls from one model response run together under `asyncio.gather`, with sibling cancellation on the first error; 2.13.0 keeps finished siblings' results (`flows/llm_flows/tools/_batch_executor.py:225-290`). Synchronous tools run on a thread pool of 4 by default (`agents/run_config.py:55-80`). I found no concurrency cap for async tools; unconfirmed.

Workflow fan-out and join. Nested tuples fan out; `JoinNode` waits for all predecessors and emits a dict keyed by node name; a graph ending in several producing branches without a join raises `WorkflowConfigurationError` (`docs/guides/workflow/join_node/index.md`). `Workflow.max_concurrency` bounds graph-scheduled nodes (`workflow/_workflow.py:160`). `parallel_worker=True` maps one node over a list with `max_parallel_workers`, fail-fast, order-preserving; `LlmAgent` workers cannot be throttled (`docs/guides/workflow/parallel_worker/index.md`).

Many instances, one store. `DatabaseSessionService` takes a per-process `asyncio.Lock` per session, then on Postgres and MySQL `SELECT ... FOR UPDATE` on the session row, then compares a revision marker and raises `StaleSessionError` if storage moved since this process loaded the session (`sessions/database_session_service.py:500-529, 907-1021`). The lock is held per event append, not per invocation. Two instances can therefore run one session at once; the slower one fails at its next append, mid-turn, with no merge or retry. SQLite and the MongoDB service use the same optimistic pattern (`sessions/sqlite_session_service.py:473-487`; `integrations/mongodb/_session_service.py`, revision field; MongoDB service is experimental). A turn lease, as we built, has to come from outside. `app:` and `user:` deltas lock their own rows, so a hot app-scoped key serialises every session in the app.

Storage. Each non-partial event is one row holding the full event JSON. Compaction never deletes rows (`runners.py` compaction docstring), and I found no prune or TTL API beyond `delete_session`; unconfirmed. Read cost per turn is the full event set unless windowed.

Long-running and background work. `LongRunningFunctionTool` returns a pending marker and the answer arrives later as a `function_response` with the same id (`tools/long_running_tool.py`). Generator tools stream `Event`s on an isolated sub-branch (`docs/guides/tools/function_tool/index.md`). There is no worker, queue or detached task that outlives the request; work belongs to the process holding the stream. `abort_signal` is an in-process `asyncio.Event`; `adk api_server` wires it to client disconnect (`docs/guides/runners/runner/abort.md`). Cross-instance cancel is yours to build. RFC #7642 (supervised task handles, queued steering) was filed 2026-10-21 by an outside contributor with a PR (#7643); it is not a maintainer commitment.

Judgement. Scale-out works for many independent sessions and fails ungracefully for contested ones. The store gives correctness (a stale writer cannot corrupt) without liveness (the loser's turn is lost). Long sessions, hot `app:` keys and wide agent fan-out are where it strains first.

## 5. Typing and contracts

State: any node accepts `state_schema`, a Pydantic model that rejects undeclared writes (`workflow/_base_node.py:139`); keys are otherwise untyped `dict[str, Any]` with prefix scoping. Node I/O: `input_schema` and `output_schema` validate node data (`_base_node.py:116-159`); `Event.output` is `Any` (`events/event.py:113`).

Tools: `FunctionTool` derives the declaration from signature and docstring and excludes `tool_context` (`docs/guides/tools/function_tool/index.md`). The docs say arguments are validated and failures return an error dict to the model. The code validates only when `FUNCTION_TOOL_ARG_VALIDATION` is on, and it is experimental and off (`tools/function_tool.py:285-296`, `features/_feature_registry.py:190-192`). Code decides; the docs are wrong for the default. An exception in a tool with no `on_tool_error_callback` ends the run (issue #7574, per `currency-adk.md`).

Dependency injection: none. `Context` or `ToolContext` is injected by parameter name or annotation. Services (session, artifact, memory, credential) are bound once on the `Runner`. Other node parameters bind from `ctx.state` by name and fail at bind time if absent (`docs/guides/workflow/parallel_worker/index.md`). Per-run dependencies travel as `state_delta`, `RunConfig.custom_metadata`, or closures. A misnamed parameter becomes a runtime failure.

Judgement. Typing is Pydantic at the edges and `Any` in the middle. That suits a graph whose nodes are written once; it gives a dependency-heavy data agent no typed per-run dependency object, so state keys become the contract, enforced by convention and tests.

## 6. First-class composition

- Agent as tool: `AgentTool` (`tools/agent_tool.py:115`), with `single_turn` and `task` agents exposed to a parent as tools. Task agents converse with the user and end through `finish_task` (`docs/guides/agents/llm_agent/task.md`).
- Transfer: `AutoFlow` adds `transfer_to_agent` among chat-mode sub-agents (`flows/llm_flows/auto_flow.py:44`).
- Workflow graph: the primary orchestration path. `SequentialAgent`, `LoopAgent` and `ParallelAgent` carry `@deprecated` in favour of it, with the note "Workflow cannot yet be used as an LlmAgent sub-agent" (`agents/loop_agent.py:59-62`, `sequential_agent.py:88-92`, `parallel_agent.py:172-175`). A workflow can be a tool through `NodeTool` (`tools/_node_tool.py`).
- A2A: `to_a2a` and `RemoteA2aAgent` (`a2a/utils/agent_to_a2a.py:80`), under an experimental decorator.
- MCP server: `to_mcp_server` exposes one agent as one MCP tool, one session per connection, in-memory services by default (`tools/mcp_tool/_agent_to_mcp.py:169`, experimental).
- Large tool sets: `BaseToolset.get_tools(readonly_context)` can vary the list per request (`tools/base_toolset.py:91`). There is no tool search or deferred loading at 2.13.0; progressive disclosure is open as #7241, and a changing list breaks the cache fingerprint.
- Skills: `SkillToolset` and `skills/` (README marks it experimental).
- Plugins: `BasePlugin` hooks cover run, user message, event, agent, model, tool and error callbacks (`plugins/base_plugin.py`), registered on the `App`.

Judgement. Composition is broad, and the Workflow-plus-agents shape is where Google is investing. Gaps appear at seams: an agent tree cannot host a Workflow, A2A and MCP exposure are experimental, and a large dynamic tool list fights the cache.

## 7. Human-in-the-loop shapes

- Approval and clarification: `RequestInput` from any node or generator tool, with optional `response_schema` that the client validates (`docs/guides/events/request_input/index.md`). Several interrupts can be open at once, each answered by id.
- Tool confirmation: `FunctionTool(require_confirmation=bool|predicate)` and `require_confirmation` on function nodes (`workflow/_function_node.py:267`). The response is `ToolConfirmation{confirmed, hint, payload}`; `payload` can carry edited input, so editing a proposal is a convention on top of confirmation, not a typed operation.
- Editing a proposal: a node pauses, the client returns edited data, the node reruns with `ctx.resume_inputs`. Interrupt ids repeat across rounds unless scoped by `invocation_id` and `run_id`, which the docs warn about.
- Steering a running turn: no API in text mode. Live mode has `LiveRequestQueue` (`docs/guides/live/live_request_queue`). Queued steering exists only in RFC #7642.
- Cancellation: `abort_signal` seals pending tool calls with synthetic events and emits `INVOCATION_ABORTED` (`runners.py:908-924`); `rewind_async` undoes a turn (`runners.py:1399`).

Judgement. Pause and resume across processes is the strongest ADK feature for this use. The approval model is generic (a function call and a function response), so structured plan review, amendment loops and multiple outstanding questions are application code. Resume rough edges still open (#7666, #7609, #7650, #7685, per `currency-adk.md`) sit in this area.

## 8. UI and streaming

The stream is `Event` objects: content parts, function calls and responses, `actions` (state delta, route, transfer, `render_ui_widgets`), `node_info.path`, `branch`, `output`, `custom_metadata`. `StreamingMode.SSE` yields partial chunks and one aggregated final event; partial events are never persisted (`agents/_streaming_mode.py:47`). `adk api_server` serves `/run` and `/run_sse` (`cli/api_server.py:2180`). A2A has its own streaming through the executor. AG-UI is not in the repository; the issues show it as third-party middleware (#3949, #4004). Generative UI exists as `UiWidget` with provider `mcp`, an MCP Apps iframe by resource URI (`events/ui_widget.py:25`, `events/event_actions.py:201`).

Judgement. The raw event stream carries everything a UI needs, and also exposes every internal event. A product UI needs a translation layer, as we built, because node paths, interrupts and branches are framework vocabulary. Streaming and persistence are decoupled cleanly. Multi-client fan-out of one run is not provided.

## 9. Models

Gemini is the native path. Thinking comes through `generate_content_config.thinking_config` or `BuiltInPlanner`. Explicit caching is `ContextCacheConfig` (section 2). Grounding comes through `google_search`, Maps, Vertex AI Search and URL context tools, with grounding attributes on spans. Code execution comes through `BuiltInCodeExecutor`, plus container, GKE and Agent Engine executors (`code_executors/`). Live and bidi sit behind `run_live` and Gemini Live connections. `VALIDATED` function calling has no ADK-specific support; `tool_config` is not blocked by the `generate_content_config` validator (`agents/llm_agent.py:1290-1320`), so it should pass through to genai; unconfirmed. Anything unexposed needs a `BaseLlm` subclass or `before_model_callback`.

Non-Gemini: `LiteLlm`, `AnthropicLlm`, Apigee, Gemma and a fallback-model wrapper (`models/`). Both main adapters are in heavy repair (`currency-adk.md`, section 7); I did not test them.

Judgement. Gemini feature access is wide and early. ADK's own abstraction is thinner than the genai SDK where features are new, so expect to subclass the model. Provider portability is real but second-class.

## 10. Observability and evals

Spans: an `invocation` span, `invoke_workflow` for workflow roots, `invoke_agent <name>`, `execute_tool <name>` and a model-call span (`telemetry/_instrumentation.py:418, 880, 921`). 2.13.0 defaults to OpenTelemetry GenAI semconv 1.44 and drops `call_llm`; `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` restores the old shape until at least 2027-03-31 (`CHANGELOG.md`). Content capture is configured per destination: legacy spans on, experimental spans off (`docs/guides/telemetry/telemetry_config/index.md`). Metrics cover tokens, tool calls and compactions per workflow. Plugins add BigQuery analytics and logging.

Evals: `AgentEvaluator` replays recorded conversations and scores them through `LocalEvalService` and a metric registry. Prebuilt metrics are tool-trajectory match, ROUGE-based response match, LLM-judged final-response match, rubric-based tool-use quality, hallucination, safety, multi-turn task success and trajectory quality, and tool-call count (`evaluation/eval_metrics.py:59-92`). User simulation exists. The design suits reference-trajectory regression tests; it assumes the agent runs through `Runner`.

Judgement. Spans nest well and follow a standard. The schema flip shows telemetry is a moving contract. The eval framework is oriented to conversation replay, so evals of a graph with pauses need a custom harness.

## 11. Design trajectory

Evidence: release notes for 2.11.0 to 2.13.0 (`CHANGELOG.md`, `currency-adk.md`). The direction is one runtime, `Workflow`, with agents as nodes. `SequentialAgent`, `LoopAgent` and `ParallelAgent` are deprecated. Task and single-turn agent modes replace transfer-heavy trees. Resume machinery is being unified (`decide_step_resume`) and `abort_signal`, node confirmation and 12 promoted features arrived in three minors. Storage is widening (MongoDB sessions, Vertex-managed sessions, SQLite memory) and Google deploys ADK to its managed agent runtime (secondary sources; unconfirmed in a primary note). Ports in Java, Go and TypeScript reached GA. There is no roadmap page; open RFCs cover streaming lifecycle (#7455), tool-result envelopes (#7754), context-boundary provenance (#7103) and Alembic migrations (#4387).

Implication. A team building on ADK gets steady improvement in the Workflow and HITL core and should expect churn: a minor release every 10 to 14 days, one flagged break in 2.13.0, and unflagged private-module moves. The durable-execution gap (no worker, no cross-instance liveness, linear replay cost) has no stated plan; the closest item is an outside RFC. A team that needs those must build them around the session store, as we have. Staying on ADK is a bet that Google's managed runtime, not the library, closes that gap.
