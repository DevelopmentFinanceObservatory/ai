# Pydantic AI 2.64.0 as an architecture for a multi-instance, human-in-the-loop analysis agent on Gemini

Sources: the repo at tag v2.64.0 (`docs/` and `pydantic_ai_slim/pydantic_ai/`; paths below are relative to those, with `PAI` for `pydantic_ai_slim/pydantic_ai/`), GitHub issues and release notes. Nothing was run against Gemini. "Unconfirmed" marks claims not read in source or docs. Features already surveyed in `currency-pydantic-ai.md` are cited, not repeated.

Headline: Pydantic AI is a stateless, in-process agent loop with typed hooks (capabilities). It owns no state between runs. Anything that must survive a request is a value you store (`Conversation`) or a journal an engine keeps (Temporal, DBOS, pgtask). Pause-for-human is cheap and fits our needs. Concurrency control across instances (one streamer per conversation, cross-instance cancel) is entirely ours to build.

## 1. Execution model

**Mechanism.** A run is a `pydantic_graph` graph of three node types: `UserPromptNode` (`PAI/_agent_graph.py:1031`), `ModelRequestNode` (`:2210`) and `CallToolsNode` (`:3721`), ending in `SetFinalResult` (`:4250`). `Agent.iter()` (`PAI/agent/__init__.py:1481`) yields these nodes; `run`, `run_stream` and `run_stream_events` drive the same loop. Run state is `GraphAgentState` (`:733`): the message list, usage, `run_step`, `run_id`, retry counters. A run holds no database handle; it takes `message_history=` or `conversation=` and returns messages.

Control flow comes in four layers:

- The agent loop decides everything inside a run (model picks tools until it emits output).
- Capabilities hook the loop (`before_model_request`, tool-execute hooks, `handle_deferred_tool_calls`; `docs/hooks.md`, `docs/capabilities/overview.md`).
- Plain Python orchestrates agents: delegation inside a tool, or application code calling agent A then B (`docs/multi-agent-applications.md`).
- `pydantic_graph.GraphBuilder` offers typed steps, decisions, broadcast and spread forks, and joins with reducers (`docs/graph/builder/`).
  A durable engine wraps any of these.

Composition is by call: an agent is a global stateless object; a delegate runs in its own run, with usage passed as `usage=ctx.usage`. State sits in three places. Message history is the conversation. `RunContext.deps` (`PAI/_run_context.py:218`) is per-run application state, a typed dataclass you build per request (DB handles, caller identity, frame store). `Conversation` (`PAI/conversation.py:48`) bundles messages, running usage, `conversation_id` and pending `deferred_tool_requests`. The graph builder's `state_type` is mutable in-memory state shared by parallel tasks (`docs/graph/builder/parallel.md:400`).

**Judgement.** Easy: a flat agent plus tools, with every other behaviour added as a hook. Our nine-node orchestrate/plan/approve/execute workflow maps to application code (an ordinary async function or a state machine of ours) calling several agents, with `Conversation` records between steps. Hard: pydantic-graph is not a durable workflow runtime. `pydantic_graph.persistence` was removed in V2 (`docs/graph/builder/index.md:325-328`), so a graph cannot pause in one process and resume in another. The graph builder suits short in-process fan-out, not a pausable plan-execute pipeline. As conversations and agents grow nothing in the framework degrades, because nothing is retained; the cost moves to the store you write.

## 2. Context engineering

**Mechanism.** `ModelRequestNode._prepare_request` (`:2842`) appends the new request to history, selects the model, resolves tools for this step (`tool_manager.for_run_step`), computes instructions, builds `model_request_parameters`, then `_apply_before_model_request` (`:~3000-3030`) runs the capability chain, where `ProcessHistory` and any `Hooks(before_model_request=...)` can rewrite messages, settings, tools and model. The request is: tools, then system prompt/instructions, then messages (`docs/capabilities/caching.md:9`). The developer controls this at every step through `RunContext` (it exposes `messages`, `deps`, `run_step`, `context_window_used`). Instructions can be dynamic, split into parts, or append-on-change (`docs/agent.md:1802`) to keep a cache prefix stable.

Each agent sees only its own history. A `SubAgent` delegate sees just the task string (`docs/harness/subagents.md`), so isolation between orchestrator and planner is by construction, not by branch filtering. History processors replace the history (`docs/message-history.md:1052-1090`): sliding window, LLM summary of old messages, or compaction at a context-fill threshold (`:1238`). Provider-native compaction exists for OpenAI and Anthropic only (`docs/capabilities/compaction.md`); on Gemini you write or import a processor. A request-side `CompactionPart` stores your summary (sent as user text in `<conversation_summary>`). Harness `Compaction` ships ready strategies, and `ConversationSearch` lets the model fetch dropped turns.

Gemini caching: implicit only by default (the `Caching` capability does nothing for Gemini, `caching.md:32`). Explicit `google_cached_content` is hand-built and strips tools, system and tool_config from the request (`PAI/models/google.py:1225-1288`). A `tool_choice` name list or per-step tool filtering changes the tools prefix and loses implicit hits. A cache-health monitor emits `pydantic_ai.cache.collapse` span events (`docs/logfire.md:282`).

**Judgement.** Strongest area. Every request is one function of history, tools and instructions, with a typed hook at the seam, so the "each agent sees exactly this" contract is testable with `FunctionModel`. Cost: history is re-sent in full every request (the maintainers' own Interactions-API note says "stateless (full history every request)", #6192), so context cost scales with conversation length unless a processor trims it. Mutating history early in a conversation busts the Gemini prefix; the docs suggest scheduling maintenance into cache-cold windows (`message-history.md:1287`).

## 3. Durability and failure

**Mechanism.**

- Pause/resume: `requires_approval=True`, `ApprovalRequired` or `CallDeferred` ends the run with `DeferredToolRequests` as output. Store `result.conversation` (JSON via `ConversationTypeAdapter`), later call `requests.build_results(approvals=...)` and `agent.run(conversation=..., deferred_tool_results=...)` (`docs/deferred-tools.md:492-545`; `PAI/_deferred.py:30,47`). No live process is held. Resume is a fresh run, on any instance that can read the store.
- Crash mid-turn without an engine: the run is lost; continue from the last `StepPersistence` snapshot (`docs/production.md:109`).
- `StepPersistence` (harness 0.64, Alpha): snapshot at every settled `CallToolsNode`, an append-only event trail, and a tool-effect ledger keyed `(run_id, tool_call_id)` with `started/completed/failed`. A `started` record with no end is "unknown after crash"; you decide whether to redo it. Stores: memory, file, SQLite, Mongo, Postgres (`docs/harness/step-persistence.md`). Resume is `continue_run` then `Agent.run(message_history=...)`; it is not exactly-once.
- Durable engines (Temporal, DBOS, Restate, pgtask, others) journal model requests and wrapped tool calls; a restart replays completed steps from the journal and retries the one in flight (`docs/durable_execution/overview.md`, `docs/harness/pgtask.md`). pgtask runs on Postgres alone (Alpha).
- Idempotency: the docs say plainly that an in-flight step re-runs from its start, so make side-effecting tools safe to repeat with an idempotency key (`production.md:111`). `annotate_tool_effect` records the key in the ledger. Engines return the journaled result for completed calls; DBOS does not wrap plain function tools unless you decorate them.
- Resume cost: a deferred-tool resume reads the stored `Conversation` (one JSON read, linear in history) and sends the full history to the model. Temporal puts message history into activity payloads recorded in workflow history, capped at 2 MB per payload by default (`temporal.md:221,322-371`); the fix is external storage (public preview) or a claim-check codec. `StepPersistence` snapshots grow with steps: nothing is pruned by default (`max_snapshots_per_run` opt-in); Postgres and Mongo stores keep each message once by hash, with snapshots as ordered key lists (`step-persistence.md` "Bounding snapshot growth").

**Judgement.** The split is clean: pausing needs a column, crash recovery needs an engine, and they do not mix. For our approval flow, `Conversation` in Postgres replaces the ledger-session workaround. Gaps: the framework gives no lease, compare-and-swap or "one writer per conversation" rule; two instances resuming the same paused `Conversation` both succeed unless our store guards it (unconfirmed in docs; none found). Crash mid-turn is not covered unless we adopt Temporal, DBOS or pgtask, each with its own determinism and payload rules. Replay safety of non-journaled tools is the developer's job. Resume cost grows linearly with history in every path.

## 4. Concurrency and scale

**Mechanism.** Parallel tool calls in one response run as asyncio tasks in emission order; `sequential=True` marks a barrier (`PAI/_tool_execution.py:926-933`, `docs/tools-advanced.md:806-812`); `parallel_execution_mode('sequential')` serialises all. `tool_calls_limit`, `UsageLimits` and `Agent(max_concurrency=ConcurrencyLimit(max_running, max_queued))` bound load (`docs/agent.md:1036,1398`). Parallel sub-agents are `asyncio.gather` over `agent.run` calls (usage shared via `usage=ctx.usage`), the graph builder's `broadcast`/`spread` with `Join` reducers (`parallel.md`, `joins.md`), or the harness `DynamicWorkflow`, where the model writes a Monty-sandboxed script that fans out sub-agents (`docs/harness/dynamic-workflow.md`). Under Temporal and DBOS, parallel branches replay in recorded order; order-dependent reducers need index keys (`parallel.md:505`). DBOS requires deterministic step start order and breaks on interleaved concurrent durable calls (`dbos.md:258`).

Multi-instance: the docs state that an agent is a Python object and "any instance can continue any conversation" because state lives in the store or the engine (`production.md:303`). Background work that outlives a request is a durable-engine workflow, a queue worker, or harness `BackgroundTools` (delivers a result as a follow-up user message within a run, not beyond it) (`docs/harness/background-tools.md`). Storage: whole-`Conversation` rewrite per turn is O(history); the docs recommend an append-only message table of `new_messages()` (`persistence.md`).

**Judgement.** In-run parallelism is solid and simple. Scale-out of stateless runs is trivial. What is missing is coordination: no distributed lock, no cross-instance cancel (a `CancellationToken` is process-local; passing it to DBOS raises `UserError`, `dbos.md:237`), no heartbeat. We would keep our ledger lease. A `Conversation` that carries thousands of messages makes each request and each write heavier; processors and a separate message table mitigate this. Usage of a delegate is lost under Temporal (#6886 per the earlier survey).

## 5. Typing and contracts

**Mechanism.** `Agent[DepsT, OutputT]` is generic; `deps_type` flows to `RunContext[DepsT]` in tools, instructions and hooks; pyright/mypy check it (`docs/agent.md:1506`). Tool arguments are validated against the function signature's JSON schema; failures return to the model as a retry that counts against `retries` (`docs/retries.md`). `args_validator` adds business checks before approval. Outputs: `output_type` accepts a Pydantic model, union, function, or `DeferredToolRequests`, with `output_validator` for semantic checks (`docs/output.md`). Modes: `ToolOutput` (default), `NativeOutput` (JSON-schema response format), `PromptedOutput` (`output.md:321-450`). On Gemini 3 native output combines with function tools; earlier Gemini cannot (`output.md:375`). Function declarations use `parametersJsonSchema`; `VALIDATED` calling is the default where supported (`PAI/models/google.py:1060-1070`). MCP tool arguments are not validated client-side.

**Judgement.** Better typed than a raw loop: deps, args, outputs and validators are one Pydantic story with static checking. Hard: model arguments are validated, but a raise in the body of a tool is a run error unless you use `ModelRetry` or `ToolFailed`; the contract is explicit rather than automatic. Gemini 3 constraints on mixing native output with tools are model-dependent, so the output mode is a per-model decision.

## 6. First-class composition patterns

- Delegation (agent-as-tool): documented hand-written pattern and harness `SubAgents` (`delegate_task(agent_name, task)`, shared deps and usage, per-delegate budgets).
- Handoff: output functions or application code (`multi-agent-applications.md`); no framework handoff primitive.
- Graphs: `pydantic_graph` builder, typed, in-process only (section 1).
- A2A: moved out to the `fasta2a` package in V2 (`docs/migration.md:31`).
- MCP server: not first-class. The docs show an agent called inside a FastMCP tool, plus MCP sampling to route model calls through the client (`docs/mcp/server.md`). MCP client side is rich (per-run headers/auth, MCP Apps metadata, filtering).
- Large tool sets: `ToolSearch` with `defer_loading=True` tools (keywords, BM25, regex or custom; native search on Anthropic/OpenAI, local `search_tools` elsewhere including Gemini) (`docs/capabilities/tool-search.md`); `on-demand` capabilities hide tools until loaded.
- Skills: harness `Skills` (SKILL.md libraries, catalog in instructions, `load_capability` tool).
- Capabilities: the extension primitive; 80+ in harness (`docs/harness/index.md`).

**Judgement.** Easy: many small behaviours as hooks, each testable alone. Hard: multi-agent workflows with approval gates are not a primitive; you compose them from application code plus `Conversation`. A deferred pause inside a delegate's run, surfacing to the parent's approver, is not documented (unconfirmed; DBOS issue #11377 on nested agents is open). Capabilities multiply surface area that sits in Alpha 0.x packages.

## 7. Human-in-the-loop shapes

- Approval: `requires_approval` per tool, or `ApprovalRequired` raised conditionally in the body; answers are `ToolApproved(override_args=...)` or `ToolDenied(message=...)` (`PAI/_deferred.py:106,116`). `override_args` is the edit-before-run path for a proposal that is a tool call's arguments.
- Confirmation inline: `HandleDeferredToolCalls` resolves in the same run (a blocking approver, useful for scripted tests); `ApprovalReviewer` has a second model approve low-risk calls.
- Clarification: harness `AskUser` takes an in-process `answerer`; with `answerer=None` every `ask_user_question` becomes a deferred call, the run ends, and the answer arrives in `DeferredToolResults.calls` (`docs/harness/ask-user.md:142-152`). Questions are multiple-choice, 1-10 questions, 2-6 options each.
- Editing a plan: model a plan as a deferred call (`submit_plan(plan)` with approval); the human answer can be approved, denied with a message, or approved with edited args. Free-form revision comes back as `ToolDenied(message)` for the model to act on. The framework has no "plan" object; harness `Planning` is a task-list tool.
- Steering a running turn: `RunContext.enqueue` or `AgentRun.enqueue` injects content mid-run, `'asap'` or `'when_idle'` (`docs/message-history.md:926-1004`).
- Cancellation: `CancellationToken` to a run raises `RunCancelled` with the history so far (`docs/agent.md:751-1010`); a delegate cancelling itself surfaces as a failed tool return.

**Judgement.** The best-fitting part. Approve and answer survive hours or days with no process held, and AG-UI maps approval to its interrupt lifecycle with `editedArgs` (`docs/ui/ag-ui.md:321-333`). Limits: the pause points are tool calls, so any gate not expressible as a tool call (a node in our workflow) must be one; clarification and approval share one resume mechanism, so interrupt ids are tool call ids (stable and unique per call, unlike our reused `approve` id). Cancellation is per process.

## 8. UI and streaming

**Mechanism.** `run_stream_events()` and `event_stream_handler` yield `AgentStreamEvent`s: part start/delta/end for text, thinking and tool calls; `FunctionToolCallEvent`/`ResultEvent`; `OutputToolCallEvent`; `EnqueuedMessagesEvent`; custom events from `await ctx.emit(...)` (`PAI/_run_context.py:859`, `docs/agent.md:300-450`). `UIAdapter` converts to protocol events; `AGUIAdapter` and `VercelAIAdapter` ship, served by `dispatch_request` from Starlette/FastAPI (`docs/ui/overview.md`). AG-UI carries shared state (`StateDeps`), frontend tools, interrupts and an `mcp-apps` activity; Vercel carries a `data-mcp-apps` chunk. `manage_message_history='server'` keeps the history on the server (`ui/overview.md:186`). Generative UI is by tool results and MCP Apps (metadata passed through, not rendered). Custom protocols: subclass `UIAdapter`.

**Judgement.** A frame translator like ours would be a `UIAdapter` subclass or a transform over `run_stream_events`. Events are a flat typed stream, with no node/branch structure, so a "which workflow step" frame is ours to add via custom events. Event types may grow in minors (version policy). Under Temporal and DBOS, streams are buffered unless a live handler is set. Open gaps: thought signatures dropped in UI dump/load (#11961).

## 9. Models

**Mechanism.** `GoogleModel` over google-genai for Gemini API and Vertex: thinking (`Thinking` or `google_thinking_config`), thought signatures round-tripped and swapped across transports, `VALIDATED` default, safety settings, logprobs, Model Armor, batch API, long decode, video input, image generation (`docs/models/google.md`). Native tools: `WebSearchTool` (Google grounding), `CodeExecutionTool`, `WebFetchTool` (url_context), `FileSearchTool`; Gemini 3 combines native with function tools (`docs/native-tools.md:35`). Live via `GoogleRealtimeModel` (`docs/realtime/gemini.md`). Other providers: roughly two dozen (`docs/models/`), with `ModelFallback`, `SelectModel`, `FallbackModel` and per-step model selection; fallback loses the prompt cache (`model-fallback.md`).

**Judgement.** Gemini coverage is deeper than most frameworks, and the abstraction does not hide provider settings. Weak spots: caching is manual; grounding/code-execution tool state lives as `NativeToolCall/ReturnPart` in history; Interactions API migration will change the request path (section 11).

## 10. Observability and evals

**Mechanism.** `Agent.instrument_all()` emits OpenTelemetry spans for the run, each model request and each tool call, following GenAI semantic conventions v1.37, with default format version 5 and aggregated-usage attributes on run spans (`docs/logfire.md:174-332`); it works with any OTLP backend. Cache health is an attribute and span event. `pydantic-evals` provides `Dataset`/`Case`/`Evaluator`/`ReportEvaluator`, an experiment per `Dataset.evaluate`, `HasMatchingSpan` over the OTel span tree, and `OnlineEvaluation` (sampled evaluators on live runs, with re-running from stored data) (`docs/evals/core-concepts.md`, `online-evaluation.md`). `TestModel` and `FunctionModel` script a model offline.

**Judgement.** Spans are by-product of the loop, so every run is traceable without custom code, and span-based evaluators judge trajectory (which tools, which order). Hard: attribute names and message formats change in minors; mapping to Langfuse is unconfirmed. Evals are code-first; no hosted harness, so our case files need a thin loader.

## 11. Design trajectory

Evidence: V2.0.0 on 2026-06-23 and about three minors a week since, with 2.64.0 on 2026-10-30. The compatibility block in the 2.64.0 notes still lists model fallback, per-attempt hooks, server-state endpoint binding and durable-run evaluator dispatch. Direction, in order of evidence:

1. Capabilities are the extension primitive; core keeps what needs model integration, and Harness (80+ capabilities, 0.64, Alpha) takes the rest. Harness API "may change between minor releases".
1. Durability by composition, not by owning a workflow runtime: `pydantic_graph.persistence` was removed; Temporal, DBOS, Restate, pgtask and others are first-class. Open work: `StepPersistence` as a branchable event-sourced session tree (#9001), a Redis store (#10597), side-effecting-tool guidance for DBOS (#11377), ephemeral side turns (#10618).
1. Conversation as the stored unit (`Conversation`, v2.55.0, closing #530's persistence ask).
1. Gemini: the maintainers' stance in #6192 is to migrate wholesale to the Interactions API once Vertex supports it (it does not yet). Listed blockers: server-minted tool-call ids against approval resume, thought signatures required at step level, and loss of safety settings, finish_reason and `count_tokens`. A history recorded before that migration must replay after it.

Implication: build the workflow and the lease in our own code, store `Conversation` objects, pin the version, and keep replay tests on stored histories across upgrades. Unconfirmed: any commitment to V3 or to a durable graph.

## Summary table for side-by-side comparison

| Question                   | Pydantic AI answer                                                     |
| -------------------------- | ---------------------------------------------------------------------- |
| Resume across instances    | Yes, via stored `Conversation` and deferred tools; no built-in lease   |
| Crash mid-turn             | Engine (Temporal, DBOS, pgtask) or `StepPersistence` snapshots (Alpha) |
| Workflow graph that pauses | No; pydantic-graph is in-process only                                  |
| Context control            | Full, per request, via hooks; history sent whole each request          |
| Gemini cache               | Implicit only; explicit is manual and strips tools                     |
| Stability                  | High churn, 0.x harness, wide exemptions in version policy             |
