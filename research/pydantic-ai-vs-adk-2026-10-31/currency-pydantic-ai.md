# Pydantic AI currency check, 2026-10-31

**Latest release: `pydantic-ai` 2.64.0, published 2026-10-31 00:22 UTC** (PyPI JSON: https://pypi.org/pypi/pydantic-ai/json; GitHub release v2.64.0, tagged 2026-10-30: https://github.com/pydantic/pydantic-ai/releases/tag/v2.64.0). Python requirement `>=3.11`. `pydantic-ai-slim`, `pydantic-graph` and `pydantic-evals` ship in lockstep at 2.64.0, all classified "Production/Stable". `pydantic-ai-harness` (extra capabilities, including `StepPersistence`) is 0.64.0, "Alpha" (https://pypi.org/pypi/pydantic-ai-harness/json).

Method: I read the source and docs at tag v2.64.0 (cloned; paths below are `https://github.com/pydantic/pydantic-ai/blob/v2.64.0/<path>`), PyPI, GitHub issues, and release notes. "Landed in" versions come from `git log -S` plus the first tag containing the commit. I ran nothing against live Gemini, so behaviour claims are from source and docs, not tests. I did not fetch ai.pydantic.dev pages; the repo `docs/` folder is their source.

Headline: V2.0.0 shipped 2026-06-23, and 2.64.0 is 64 minors later. Gemini support, MCP, deferred tools, durable execution and UI streams are all present. Two items bear directly on this stack: pydantic-graph can no longer persist or resume a graph (removed in V2), and the MCP client sends the model `structuredContent` by default, not the text digest.

## 1. Gemini via google-genai (developer API and Vertex): supported

- `GoogleModel` with `GoogleProvider` (Gemini API) or `GoogleCloudProvider` (Vertex, prefix `google-cloud:`); `google-genai>=2.30.0` (`pydantic_ai_slim/pyproject.toml`; `docs/models/google.md`). Present since v0.2.5 (Google GenAI provider, #1751).
- Thinking: provider-agnostic `Thinking(effort=...)` capability, or native `GoogleModelSettings.google_thinking_config` (`thinking_level`, `thinking_budget`, `include_thoughts`). Per-model level tables, with Vertex and Gemini API differences handled (`profiles/google.py`, `docs/models/google.md#configure-thinking`).
- Thought signatures: stored on each response part as `provider_details['thought_signature']` (base64) and sent back in the original part (`models/google.py` ~L1855-1861, L2061-2150). The signature lives in `ModelResponse`, so it survives if you persist messages with `ModelMessagesTypeAdapter`/`Conversation`. Gemini API and Vertex reject each other's signatures. Pydantic AI tags responses with `canonical_system` (`google` vs `google-cloud`) and substitutes the documented `skip_thought_signature_validator` when history crosses transports (landed v1.39.1, #3882; `docs/models/google.md` L76). Open bug: AG-UI/Vercel dump-and-load drops the signature (#11961). That affects only a UI-adapter round trip, not the native store.
- Function-calling modes: `tool_choice` setting (v1.93.0, #3611) maps `auto` to AUTO, `required` to ANY, `none` to NONE, and a name list to `allowed_function_names` (ANY) (`models/google.py` L1035-1075). **`VALIDATED` is now the default** where the model profile supports it and no tool sets `strict=False` (v2.22.0, #6353). A name list under `auto` removes the other tools from the request, which the source comments say "breaks caching".
- Tool schemas: sent as `parametersJsonSchema` (raw JSON Schema) and `response_json_schema` for return schemas (`_function_declaration_from_tool`, L2517). `$defs`/`$ref` are **kept, not inlined**, on the normal path. The transformer only drops `$schema`, `title`, `discriminator`, `examples`, `exclusiveMin/Max`, converts `const` to single-value `enum`, and folds `format` into the description (`profiles/google.py` L461-512). The inlining transformer (`GoogleOpenAPISchemaTransformer`, `prefer_inlined_defs=True`, rejects recursive refs) is used only for the Live/realtime path. MCP tool schemas are passed through unchanged, so FastMCP schemas reach Gemini with only those edits.
- Related open bugs: tool returns containing a `$ref` key (#11901), invalid `MINIMAL` thinking level on 3.x Pro (#9623).

## 2. Gemini context caching: partial

- Explicit: `GoogleModelSettings.google_cached_content='cachedContents/...'` (v1.0.2, #2832). Pydantic AI does **not** create or manage caches; you call `client.caches.create(...)` yourself (`docs/models/google.md` L468-509).
- When set, it strips `system_instruction`, `tools` and `tool_config` from the request, as Gemini requires, and emits a `UserWarning` (`models/google.py` L1225-1288). So explicit caching and dynamic per-turn tools are incompatible; the cache must own the tool list.
- Implicit: the docs say Gemini caches implicitly and the unified `Caching()`/`cache` setting adds nothing for Google (`docs/capabilities/caching.md` table). Implicit hits depend on a stable prefix. Pydantic AI orders tools, system, then messages, and offers `on_change='append'` instructions to keep the prefix stable (`docs/capabilities/caching.md`). Per-step tool filtering or a `tool_choice` name list changes the tools prefix and loses hits.
- Usage: `cached_content_token_count` maps to `usage.cache_read_tokens` and `details['cached_content_tokens']`; `thoughts_tokens`, `tool_use_prompt_tokens`, and per-modality prompt/cache/candidate details are also kept (`models/google.py` L2530-2700; `usage.py`). A streamed-through-gateway gap is tracked in #5205 (closed reference in source). Cache-health monitoring exists (`_cache_health.py`, harness `warn-on-cache-busts`).

## 3. MCP client: partial

- `MCPToolset` wraps the FastMCP client (v1.97.0, #5325; older `MCPServer*` deprecated). Streamable HTTP is the default for a URL (SSE only for `/sse` paths). Install pulls FastMCP 3 or 4 (`fastmcp-slim[client]>=3.3.0,<5`) (`docs/mcp/client.md`).
- Headers: static `headers=`; **`headers` and `auth` can be functions evaluated per run**, each run opens its own session (`docs/mcp/client.md#per-user-authentication`). Custom `http_client` (httpx2) for TLS and timeouts.
- Results: `_map_mcp_call_tool_result` **returns `structuredContent` to the model when all content blocks are text**, otherwise the text blocks (`mcp.py` L2190-2206). With FastMCP servers that return a large `structuredContent` plus a text digest, the model gets the large object, not the digest. To send the digest, wrap with `process_tool_call` or a `WrapperToolset` and return a `ToolReturn(return_value=digest, metadata=structured)`; `ToolReturn.metadata` is "accessible by the application but not sent to the LLM" (`messages.py` L1217-1252). The raw result with `structuredContent` and `_meta` is kept only for MCP Apps tools (`metadata['mcp_app']`). `ToolDefinition.metadata` carries tool `meta` and `annotations`; `return_schema` is the output schema.
- Request `_meta`: `process_tool_call(ctx, call_tool, name, args)` and `call_tool(..., metadata={...})` send request-level `_meta`.
- Filtering per turn: `FilteredToolset`, `PrepareTools` capability (v1.71.0), `DynamicToolset`; see section 4.
- Sampling: supported via `sampling_model=` / `agent.set_mcp_sampling_model()`. Elicitation: `elicitation_handler=`. **With FastMCP 4's "modern protocol mode" neither works**; `MCPToolset` warns, and legacy mode keeps the FastMCP 3 behaviour (`docs/mcp/client.md` L19-26). Our servers are FastMCP 4, so check which mode they use if we need either.
- Errors: `tool_error_behavior` `'retry'` (default, raises `ModelRetry`), `'failed'`, `'error'`. MCP tool arguments are not validated client-side (`TOOL_SCHEMA_VALIDATOR` is a pass-through, `mcp.py` L661); the server's validation error is what the model sees.
- Durable engines connect the MCP server inside each activity; each toolset needs a stable `id` (`docs/durable_execution/temporal.md` L207).

## 4. Tool validation, retries, dynamic tools, approval: supported

- Arguments are validated against the tool's schema; failures and `ModelRetry` come back as a `ToolReturnPart(outcome='retried')` and count against `retries` (default 1, per tool, reset on success); exhaustion raises `UnexpectedModelBehavior` (`docs/retries.md#tool-retries`). `args_validator` (v1.63.0, #4016) adds business validation before approval. `ToolFailed` gives a non-retry failure. Open: `before_tool_validate` rewrites do not reach history (#12029).
- Dynamic tool sets per turn: `PrepareTools`/`prepare` per step, `DynamicToolset`, `FilteredToolset`, `ToolReturn(tools=[...])` for deferred loading, tool search (`docs/toolsets.md#preparing-tool-definitions`, `docs/capabilities/tool-search.md`).
- Approval: `requires_approval=True` or `raise ApprovalRequired` (v1.0.0, #2581). The run ends with `DeferredToolRequests` as output; resume with `deferred_tool_results=`. **The pause survives a later HTTP request**: store `result.conversation` (includes `deferred_tool_requests`), reload it, call `build_results(approvals=...)` and `agent.run(conversation=..., deferred_tool_results=...)` (`docs/deferred-tools.md#pausing-a-conversation`; `Conversation` landed v2.55.0, #8337). `HandleDeferredToolCalls` resolves inline instead.

## 5. Multi-agent and graphs: partial (delegation supported; graph persistence missing)

- Delegation: an agent calls another inside a tool, passing `usage=ctx.usage`; programmatic hand-off; harness `subagents` (`docs/multi-agent-applications.md`).
- `pydantic-graph` builder API (`GraphBuilder`, steps, decisions, joins, parallel forks) left beta in V2 (introduced v1.5.0 as `pydantic_graph.beta`).
- **`pydantic_graph.persistence` was removed in V2.** Graph state is not snapshotted, because of parallel-execution consistency; there is no replacement in pydantic-graph (`docs/changelog.md` L136; `docs/graph/builder/index.md` L325-328). So a graph run cannot pause in one process and resume in another. The supported substitutes are agent-level: `Conversation` + deferred tools, harness `StepPersistence` (snapshots of message history at settled steps; memory, file, SQLite, MongoDB, PostgreSQL stores; harness is Alpha) (`docs/harness/step-persistence.md`), or a durable engine.

## 6. Durable execution: supported (maturity varies)

- Eight integrations: Temporal, DBOS, Prefect, Restate, AWS Lambda, Absurd, pgtask, Airflow, plus a backend builder (`docs/durable_execution/overview.md`). The capability form (`TemporalDurability`, `DBOSDurability`, `PrefectDurability`) landed v2.14.0 (#4977), replacing the `TemporalAgent`-style wrappers. Temporal, DBOS, Prefect, Restate and Lambda are co-maintained with the vendors.
- Guarantee: model requests, tool calls the engine wraps, and MCP calls are journaled; a restarted run replays completed steps and retries the one in flight. Not wrapped by default on DBOS: plain function tools and event handlers.
- Caveats from the docs: Temporal and DBOS stream events buffered (deltas not recorded; live delivery needs `event_stream_handler=` or Temporal Workflow Streams); `RunContext` inside activities is a limited copy and delegate usage is lost (#6886); payload size caps on Temporal; `agent.run()` outside a workflow is non-durable, so serving behind HTTP means starting a workflow (`docs/durable_execution/temporal.md` L78, L223-241, L385-395).
- Not needed for pause-for-approval (section 4); it matters only for surviving a crash mid-turn. Postgres-backed options: DBOS, pgtask (harness, Alpha).

## 7. Message history: supported

- `ModelMessagesTypeAdapter` and `ConversationTypeAdapter` (v2.55.0) dump to JSON; `Conversation` also carries usage, `conversation_id`, deferred requests (`docs/persistence.md`, `docs/message-history.md` L488-582). Store it in a Postgres `jsonb` column.
- Stability: version policy says new message parts, event types and optional fields may be added in minors, so "use `ModelMessagesTypeAdapter` to (de)serialize" and code defensively (`docs/version-policy.md`). No schema version number or migration tool is documented (unconfirmed beyond docs). V2 changed shapes: `RetryPromptPart` deprecated, loaded from old JSON as `ToolReturnPart(outcome='retried')` or `RetryFeedbackPart` (`docs/changelog.md` L12-20). Old retries re-render the new way, which costs one prompt-cache miss.
- Non-JSON tool return values (datetime, UUID, bytes) reload as strings (`docs/message-history.md` L560-575).
- Trimming: `history_processors` (v0.7.0), `ProcessHistory` capability, compaction (`docs/capabilities/compaction.md`), summarise-old-messages patterns.

## 8. Streaming and UI protocols: supported

- `Agent.run_stream_events()` (v1.0.15), `agent.iter()`, `event_stream_handler`, `ProcessEventStream` capability (v1.87.0). Custom progress events via `CustomEvent` and `await ctx.emit(...)` inside async tools; they reach AG-UI and Vercel streams (`docs/agent.md#custom-events`).
- `AGUIAdapter` and `VercelAIAdapter` (v1.9.0, #2923), served from Starlette/FastAPI with `dispatch_request()`; MCP Apps results forwarded to the frontend (`docs/ui/overview.md`). `UIAdapter` subclasses can add another protocol. Events may gain new types in minors.
- Open gaps: `canonical_system` is not carried through UI streams (#11801), and the thought-signature loss above (#11961).

## 9. Observability: supported (Langfuse path is documented by Langfuse, not by Pydantic AI)

- OpenTelemetry instrumentation via `Agent.instrument_all()` / `InstrumentationSettings`; follows GenAI semantic conventions v1.37.0; default format `version=5`, `version=6` available, 2-4 deprecated (`docs/logfire.md` L174-349; `models/instrumented.py`). Span attributes may change in minors (`docs/version-policy.md`).
- Works without Logfire: set an OTLP exporter or `OTEL_EXPORTER_OTLP_ENDPOINT`, call `Agent.instrument_all()` (`docs/logfire.md#otel-without-logfire`). Options `include_content`, `include_binary_content`.
- Langfuse: an integration page exists on Langfuse's site using `Agent.instrument_all()` and the Langfuse SDK; it names no Pydantic AI version and notes some attributes land in metadata (https://langfuse.com/integrations/frameworks/pydantic-ai). The Pydantic AI docs do not mention Langfuse (grep of `docs/`: no hits). Whether Langfuse maps every v5/v6 attribute (tool spans, cached tokens) is **unconfirmed**; test with our collector.

## 10. Testing and evals: supported

- `TestModel` (v0.0.6a4) and `FunctionModel` for scripted fakes, `Agent.override(model=, deps=, toolsets=)`, and `ALLOW_MODEL_REQUESTS=False` (`docs/testing.md`). `FunctionModel` lets a script return any `ModelResponse`, including tool calls, which fits scripted offline evals.
- `pydantic-evals` 2.64.0 (first release 0.0.47 on 2025-03-31), "Production/Stable" on PyPI: datasets, evaluators, trajectory evaluators, online evaluation (`docs/evals.md`). The docs call evals "an emerging practice" and the version policy says Evals span attributes may change.

## 11. Project health

- Cadence: 111 PyPI releases from 2026-05-01 to 2026-10-31 (23, 8, 24, 24, 18, 14 per month, May to Oct), including V1 maintenance (1.107.x) and 2.0.0b1-b7. Since 2.0.0 (2026-06-23): 2.1 to 2.64, roughly three a week.

- Stability: V1 on 2025-09-05 with a no-breaking-changes promise; V2.0.0 on 2026-06-23; "no intentional breaking changes in minor releases", deprecations removed only in the next major, no sooner than 3 months after 2.0 (so V3 is possible from late September 2026; none announced that I found). V1 gets security fixes for at least 6 months after V2 (until about 2026-12-23). Exemptions are broad: new message parts, stream events and optional fields, OTel attributes, bug fixes, and anything under a `beta` module (`docs/version-policy.md`). The 2.64.0 notes list a large "Compatibility Notes" block (https://github.com/pydantic/pydantic-ai/releases/tag/v2.64.0).

- Issues: 1,229 open issues and 486 open PRs on 2026-10-31 (GitHub search API). Churn is high: 826 issues opened and 573 closed since 2026-10-01.

- Five Gemini issues most relevant (all open at time of check):

  1. #11961 AG-UI dump/load drops Gemini thought signatures: https://github.com/pydantic/pydantic-ai/issues/11961
  1. #11901 tool returns with a `$ref` key are read by Gemini as a part reference: https://github.com/pydantic/pydantic-ai/issues/11901
  1. #11902 `http_client` default 5 s read timeout becomes Vertex's server deadline: https://github.com/pydantic/pydantic-ai/issues/11902
  1. #9623 invalid `MINIMAL` thinking_level for gemini-3.x Pro, Vertex regional prefix not stripped: https://github.com/pydantic/pydantic-ai/issues/9623
  1. #6192 plan to migrate `GoogleModel` to the Interactions API (blocked on Vertex parity), which would change the request path: https://github.com/pydantic/pydantic-ai/issues/6192

  Also: #6751 Vertex location per model (https://github.com/pydantic/pydantic-ai/issues/6751), #12092 Vertex streamed tool-call arguments (https://github.com/pydantic/pydantic-ai/issues/12092).

- Backing: Pydantic Services Inc.; Series A $12.5M in October 2024 led by Sequoia, with Partech and Irregular Expression; about $17M disclosed in total (https://techcrunch.com/2024/10/01/sequoia-backs-pydantic-to-expand-beyond-its-open-source-data-validation-framework). I found no later round (search only; unconfirmed). Revenue comes from Logfire and a paid Gateway; enterprise support is documented (`docs/enterprise-support.md`).

## Strengths for this stack

- Gemini treatment is detailed and current: per-model thinking levels, cross-transport signature handling, VALIDATED by default, raw JSON Schema declarations with `$defs` kept, usage details including cache and thought tokens.
- Approval pause and resume across HTTP requests is a first-class path: one serialisable `Conversation` holds history and pending requests, and `build_results` checks answers against requests.
- `ToolReturn(return_value, metadata)` separates the model-facing digest from application data, which matches digest-plus-`structuredContent` servers once wired.
- Per-run MCP session and per-run `headers`/`auth` functions give per-caller identity without sharing sessions.
- Capabilities and hooks (`PrepareTools`, `ProcessEventStream`, `HandleDeferredToolCalls`, `ModelFallback`) cover per-turn tool sets, streaming and retries without a custom runner.
- `FunctionModel` plus `Agent.override` suit scripted fake-model evals.
- Eight durable engines, if crash recovery mid-turn becomes a requirement.
- Stream adapters for AG-UI and Vercel, plus `ctx.emit` for tool progress.

## Risks for this stack

- No graph pause, persist or resume: removed from pydantic-graph in V2. A workflow graph that must survive a process hop needs `Conversation`-level pauses, harness `StepPersistence` (Alpha), or a durable engine.
- MCP defaults send the full `structuredContent` to the model; the digest needs explicit wrapping. FastMCP 4 modern mode drops sampling and elicitation, and the toolset is FastMCP-client based (httpx2), a different dependency stack from the MCP SDK.
- Explicit Gemini caching is hand-built and strips tools, system and tool_config; dynamic per-turn tools and `tool_choice` name lists cost implicit-cache hits. Neither is automatic.
- Churn: about three releases a week, 1,229 open issues, wide exemptions from the stability promise (message parts, OTel attributes, events, beta modules), and V1-to-V2 reshaped retries, graphs, durability wrappers and MCP classes within four months. Pin the version and budget for upgrade work.
- Stored history has no schema version; the policy relies on the type adapter tolerating additions. Old retries and non-JSON tool values reload differently. Test replay of stored conversations on each upgrade.
- Langfuse mapping of Pydantic AI spans is documented only on Langfuse's side; semconv version moves between minors.
- Durable engines add constraints: stable ids, serialisable deps, buffered streaming, limited `RunContext`, lost delegate usage.
- Open Gemini bugs (above) and the planned Interactions API migration could alter request shape under us.
