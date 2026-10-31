# ADK footprint: models, agents and tools

Scope: `services/harness/src/{planner,translator,orchestrator}`, `harness/{prefix_cache,turn_budget,planner_seam,orchestrator_tools}.py`, `evals/`. Paths are relative to `services/harness/src` unless they start with `evals/`, `tests/` or `packages/`. `orchestrator/agent.py` is read from `HEAD` (the working copy has conflict markers at lines 224-361). `google-adk[db]==2.10.0` is pinned in `services/harness/pyproject.toml:17`.

Three findings change the picture.

- Only two LLM agents exist, the planner and the orchestrator (`planner/agent.py:1455`, `orchestrator/agent.py:365` at HEAD). The translator has no model. It runs as plain code inside `propose_plan` (`translator/mechanical.py:1-15`; `translator/tool.py:764-770` unwraps `FunctionTool.func`).
- No MCP tool reaches a model through ADK. Both agents expose hand-written Python functions that call MCP through our own `fastmcp.Client` wrappers (`compiler/surface.py:31,264-275`; `harness/mcp_client.py`). The README says so: `services/harness/README.md:197`. ADK's `McpToolset` is not used.
- The deepest ADK coupling is the session object, not the model: tools read `tool_context.state` and scan `tool_context.session.events`.

## 1. Concepts used

| Concept                                                              | Files:lines                                                                                                         | Depth                                                                                                 | Behaviour we rely on                                                                                                                                                     | Our lines bound                                           |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `LlmAgent` (planner)                                                 | `planner/agent.py:1455-1492`                                                                                        | public API, thin use                                                                                  | instruction, tools, `include_contents="default"`, `planner=BuiltInPlanner`, `after_model_callback`; run as a workflow child                                              | ~40                                                       |
| `LlmAgent` (orchestrator)                                            | `orchestrator/agent.py` HEAD `365-387`                                                                              | public API, thin use                                                                                  | same, plus `before_agent_callback`, `before_model_callback`, `on_tool_error_callback`, `generate_content_config` thinking                                                | ~40                                                       |
| Callbacks on the planner (mutated after build)                       | `harness/planner_seam.py:437-451`, `802-920`                                                                        | public API, thin use, but reads `agent.canonical_after_model_callbacks` (`:446`)                      | after-tool, tool-error, before-model synthetic response, after-model; first truthy return stops the list                                                                 | ~180                                                      |
| `BaseLlm` / `Gemini` subclass: `BoundedGemini`                       | `harness/turn_budget.py:154-262`                                                                                    | subclass/override of `generate_content_async`                                                         | wraps the response stream: 90 s model time, 30 calls, 600 s wall, 12 s first-chunk resend; emits a synthetic final `LlmResponse`                                         | ~150                                                      |
| `Gemini` subclass: `PrefixCachedGemini` and `PrefixCache`            | `harness/prefix_cache.py:113-292`                                                                                   | subclass/override; reads `self.api_client`                                                            | one CachedContent per prefix hash; strips `system_instruction`, `tools` and `tool_config` off the request; resend on 403/404                                             | 303                                                       |
| `LLMRegistry.new_llm`                                                | `harness/prefix_cache.py:97-105`                                                                                    | public API, thin use                                                                                  | turns a model string into `Gemini`, or leaves non-Gemini names for ADK                                                                                                   | ~10                                                       |
| `LlmRequest` edits                                                   | `orchestrator/agent.py` HEAD `291-308`; `harness/prefix_cache.py:214-226`                                           | public API, thin use                                                                                  | insert a user content at `contents[0]`; `model_copy` the config                                                                                                          | ~30                                                       |
| `LlmResponse` built by hand                                          | `harness/turn_budget.py:265-284`; `planner_seam.py:1239-1246,1427-1446`                                             | public API, thin use                                                                                  | synthetic text, or a synthetic `adk_request_input` function call with a skip-signature                                                                                   | ~60                                                       |
| `FunctionTool`                                                       | `planner/skills.py:93`; `planner/agent.py:1409`; `orchestrator/agent.py` HEAD `364`; `translator/discovery.py:2495` | public API, thin use                                                                                  | declaration from signature and docstring; result dict becomes the function response                                                                                      | ~30                                                       |
| `FunctionTool` subclass `PlanTool`                                   | `planner/declaration.py:235-272`                                                                                    | subclass/override of private `_get_declaration`, plus `is_feature_enabled(JSON_SCHEMA_FOR_FUNC_DECL)` | per-argument descriptions on the JSON schema ADK builds                                                                                                                  | ~40 of 272 (rest is schema compaction, framework-neutral) |
| `LongRunningFunctionTool`                                            | `planner/agent.py:1397-1407`                                                                                        | public API, with a private constant                                                                   | `ask_user` renamed `adk_request_input` so the run pauses; returns `None`                                                                                                 | ~70                                                       |
| `REQUEST_INPUT_FUNCTION_CALL_NAME`                                   | `planner/agent.py:59`; `planner_seam.py:109`; `evals/studio/checks.py:1332`                                         | private import (`google.adk.workflow.utils._workflow_hitl_utils`)                                     | the pause call name, shared with the graph's `ask` node                                                                                                                  | ~15 sites                                                 |
| `ToolContext` state, `.session.events`, `.actions`, `.invocation_id` | `planner/agent.py:499-957`; `translator/{discovery,mechanical,tool}.py`; `orchestrator_tools.py` (93 mentions)      | public API, deep                                                                                      | state is the only store; the planner reads what the user answered off event history; `actions.skip_summarization` (`planner/agent.py:577`, `orchestrator_tools.py:1063`) | ~500 in `planner/agent.py` alone; ~200 mentions elsewhere |
| `on_tool_error_callback`                                             | `orchestrator/tool_failures.py:1-61`; `planner_seam.py:818-855`                                                     | public API, thin use                                                                                  | a raised tool error becomes a function response                                                                                                                          | ~100                                                      |
| `_validated_arguments`                                               | `harness/orchestrator_tools.py:852-915`, applied `:2047-2064`                                                       | workaround of an ADK defect                                                                           | pydantic validation of every model argument, error returned to the model                                                                                                 | ~65 plus hints and budget (`:832-850`)                    |
| `BuiltInPlanner` and thinking config                                 | `planner/agent.py:1487`; `planner/settings.py:45-47`                                                                | public API, thin use                                                                                  | `thinking_level=LOW`, `include_thoughts=True`                                                                                                                            | ~5                                                        |
| Thought stripping `drop_thought_parts`                               | `planner/settings.py:50-85`                                                                                         | workaround of an ADK defect                                                                           | keep thought parts out of session events and replay                                                                                                                      | ~36                                                       |
| Branch isolation                                                     | `planner_seam.py:187-204,1304-1342`                                                                                 | workaround (`override_branch`, then a post-run check)                                                 | planner events stay off the orchestrator's branch                                                                                                                        | ~60                                                       |
| `deep_merge_dicts` assumption                                        | `translator/discovery.py:782-811` (comment); `tests/translator/test_discovery_concurrency.py:494`                   | private import (tests only)                                                                           | parallel tool state deltas merge dict keys, replace string leaves                                                                                                        | ~30 (key layout)                                          |
| `_fencing` constants                                                 | `tests/orchestrator_fakes.py:30`; mirrored as a literal in `evals/studio/checks.py:1328-1333`                       | private import (tests and evals)                                                                      | ADK quotes another agent's turns as fenced user text                                                                                                                     | ~20                                                       |
| `_feature_registry.temporary_feature_override`                       | `tests/planner/test_agent.py:39-40,5496`                                                                            | private import (one test)                                                                             | proves `PlanTool` refuses to build with the flag off                                                                                                                     | ~6                                                        |
| `google.adk.evaluation`                                              | `evals/agents/harness_translator_v1/outcome_metric.py:99-102`; `justfile:327-336`                                   | public API, thin use                                                                                  | `adk eval` CLI, `Invocation`, `EvalMetric`, `get_all_tool_responses`                                                                                                     | ~400                                                      |
| Telemetry hooks on ADK span shapes                                   | `packages/common/src/common/telemetry.py:67-72,316-387`                                                             | depends on span names                                                                                 | see section 5                                                                                                                                                            | ~70                                                       |

## 2. Behaviour a replacement must reproduce

1. A model call carries one cached static prefix (instruction, tools, tool config) shared across sessions. The request sent against a cache carries none of those three fields (`prefix_cache.py:214-226`). A changed declaration is a new key.
1. A cache that is refused (400), slow to create (over 10 s), gone (403/404) or failing never fails the turn. A gone cache is recreated once and the request resent if no chunk was delivered (`prefix_cache.py:20-34,262-292`).
1. Every model call in one HTTP request shares one budget. A spent or cut bound makes no further request, and the model's final text is that bound's sentence tagged `dfo.turn_budget.closed_by` (`turn_budget.py:171-284`).
1. The first-chunk deadline resends a streamed call once per request, and only before any chunk has reached the caller (`turn_budget.py:171-250`).
1. A user stop cancels the call without retry, and the model-call span ends in error (`turn_budget.py:199-203`).
1. Transient 408, 429 and 5xx get three attempts at 1 s to 2 s (`orchestrator/agent.py` HEAD `311-332`).
1. A callback can answer a model request itself. The planner substitutes a terminal text, or a pause call with a skip thought-signature, and the run ends or suspends with no model turn (`planner_seam.py:893-920`).
1. A tool that raises answers the model with its name, the exception class and a `raised` code, never the message. An unknown tool name falls through to the framework's own "tool not found" reply (`tool_failures.py:26-61`; `planner_seam.py:818-855`).
1. Every orchestrator argument is validated against its annotations before the body runs. Failures come back as `{error, invalid[{loc,type}]}` and count against `BUDGETS["invalid_arguments"]` (`orchestrator_tools.py:852-915`).
1. The planner's `propose_plan` validates its own list arguments, because a failed list item leaves raw dicts (`planner/README.md:315-320`).
1. The `propose_plan` declaration carries a description on every argument, no `$ref` with sibling keys, and `Identifier` and `decision_ids` stated once (`planner/declaration.py:1-45,235-272`). A test fails when any `$ref` has siblings (`tests/test_function_declarations.py`).
1. The planner's thought summaries reach the trace but not later requests or the user stream. A part with a `thought_signature` stays (`planner/settings.py:50-85`).
1. The planner sees its whole branch history, and the orchestrator sees none of it (`planner/agent.py:1471-1484`; `planner_seam.py:187-204`).
1. The orchestrator's first content each request holds today's date and the digests, so its instruction stays constant (`orchestrator/agent.py` HEAD `291-308`).
1. A tool can end a turn's summarisation (`skip_summarization`) and pause for the user. The user's answer returns later, possibly on a new invocation, as that call's own response with the same id (`planner/agent.py:645-700`).
1. Tools read the user's answers from persisted event history. Replacing events with another store means rebuilding `_elected_spec_ids`, `_answers_since_referral`, `_combination_picks` and `_user_spoke_since_recorded` (`planner/agent.py:645-957`).
1. Concurrent tool calls in one turn lose no state. Discovery keys records by id so a merge unions them (`translator/discovery.py:782-811`).
1. The tool-failure and budget closing messages are final model text that Studio streams as a partial chunk, then a whole response (`turn_budget.py:265-284`).

## 3. Where ADK has cost us

ADK:

- `FunctionTool` does not validate model arguments. The validation flag is experimental, process-wide and off (memory `adk-tool-args-unvalidated.md`); the planner README says it falls back to raw dicts for a whole list (`planner/README.md:315-320`). Workaround: `_validated_arguments` (`orchestrator_tools.py:852`), `propose_plan`'s own `model_validate`. The memory note that no agent sets `on_tool_error_callback` is stale: both do (`orchestrator/agent.py` HEAD `371`; `planner_seam.py:438`).
- A workflow-node `LlmAgent` defaults to `single_turn` and `include_contents='none'`, which drops the brief. Pinned to `"default"` in both agents (`planner/agent.py:1471-1484`; `orchestrator/agent.py` HEAD `372-377`).
- An event with no branch belongs to every branch, so the orchestrator would read planner events as fenced text. Workaround: fixed `override_branch` plus `_check_branches` (`planner_seam.py:187-204,1304`).
- `BuiltInPlanner` does not strip thought parts, so `drop_thought_parts` runs as an after-model callback (`planner/settings.py:68-70`).
- ADK's Gemini makes one attempt unless told otherwise, so `_retrying` passes `HttpRetryOptions` (`orchestrator/agent.py` HEAD `318-323`).
- No model-call budget exists in ADK (inferred: nothing in ADK is used for it). `BoundedGemini` wraps the response generator.
- Explicit caching is our own `PrefixCache` instead of ADK's `ContextCacheConfig` (`prefix_cache.py:1-34`; memory `gemini-explicit-cache-blocks-implicit.md`). The reason recorded is that each Studio turn is its own ADK session, so session-scoped caches never carry across turns (memory `session-trace-viewer.md`). Inferred that this is the cause.
- A long-running tool that pauses writes nothing durable, since no event carries its state delta (`planner/agent.py:676-695`). History is read back from the call event's args.
- Tool declarations come from the signature. An optional list becomes `anyOf [array, null]`, so defaults are `= []` (`planner/agent.py:1269-1272`). Argument descriptions need a `FunctionTool` subclass and a private method (`declaration.py:1-12,248-272`).
- Parallel tool state deltas merge with `deep_merge_dicts`, so state is keyed by id, never a JSON string (`translator/discovery.py:782-811`).
- `LlmAgent.model` and callbacks are mutated after construction (`planner_seam.py:434-451`) because the seam owns the per-call state (inferred: no per-run hook).

Gemini:

- A `description` beside `$ref` never reaches the model; `compact_schema` inlines (`planner/declaration.py:29-36,170-200`; memory `gemini-drops-ref-sibling-descriptions.md`).
- AUTO mode lets the model send kinds the declaration lacks, so the Python validator alone gates (memory `gemini-declaration-does-not-gate.md`).
- A request replaying a function call needs a `thought_signature`; the placeholder is `skip_thought_signature_validator` (`planner_seam.py:257`).
- Explicit cache and `system_instruction` or `tools` on one request is a 400 (`prefix_cache.py:8-11`). A prefix under the minimum is a 400 (`:20-26`). An expired cache answers 403 (`:73-75`).
- An explicit cache caps the cached share at its prefix (memory `gemini-explicit-cache-blocks-implicit.md`).
- Thinking: `MINIMAL` is rejected on flash (`planner/settings.py:34-39`). About 94% of thinking falls on the first call (memory `gemini-thinking-on-first-call.md`).
- `count_tokens` refuses tools and system instruction on the developer API (memory `gemini-count-tokens-text-only.md`).

MCP:

- A tool result over 1 MiB is dropped silently by the client (memory `mcp-client-sse-1mib-cap.md`).
- With fastmcp 4 `ctx.session_id` is new per call, so identity rides the `x-dfo-caller` header (memory `fastmcp4-session-id-per-request.md`).
- MCP tool observations carry no input or output in Langfuse (memory `deployed-harness-langfuse-wiring.md`).

A framework switch removes only the ADK items. The Gemini and MCP items stay with whichever client sends the request, though a new framework changes where each workaround attaches.

## 4. What ADK gives us that we would rebuild

- The tool loop: declaration from signature, call dispatch, parallel execution, function-response events.
- `ToolContext.state` with deltas and the `temp:` prefix (`orchestrator/agent.py` HEAD `27-30`), and `session.events` as durable history.
- Content assembly: branch filtering, rewriting other agents' events as fenced text, `include_contents`.
- Pause and resume: long-running tools, function-call ids (`populate_client_function_call_id`, `planner_seam.py:1431`), and the user's answer as a function response.
- Callback chains (before and after model, tool, agent) with first-truthy-stops semantics.
- Streaming events (partial then final) that `harness.studio_translate` reads.
- Span emission with usage, response and invocation attributes (section 5).
- The `adk eval` CLI, rubric metrics and `EvalSet` format for 18 cases (section 6).
- A session service interface (`DatabaseSessionService`, `session_store.py`); another agent covers it.

## 5. Tracing dependencies on ADK span shapes

Tracing is both. ADK emits the spans under instrumentation scope `gcp.vertex.agent` and our `common.telemetry` adds a `turn` root, events and markers, and bends the ADK ones into shape.

- `_should_export_span` keeps ADK's attribute-less `invocation` span, which Langfuse's default filter drops, otherwise `invoke_workflow` spans show as roots (`telemetry.py:316-337`). Test: `packages/common/tests/test_telemetry.py:340-346`.
- `_mask_otel_spans` retypes every `call_llm` span in that scope as a plain span with `usage_details="{}"`, since ADK's `call_llm` and its inner `generate_content <model>` both carry usage and Langfuse would price twice (`telemetry.py:339-387`; memory `langfuse-otel-ingestion-rules.md`).
- `BoundedGemini` sets `langfuse.observation.completion_start_time`, `dfo.model.first_chunk_s` and `dfo.model.chunk_gaps_s` on `trace.get_current_span()`, which it takes to be ADK's `generate_content` span (`turn_budget.py:29-33,293-300`; inferred, since the span is not created in our code).
- Session grouping falls back to ADK's `gen_ai.conversation.id` when `session_id` is over 200 characters (`telemetry.py:484`).
- Trace tests build ADK-named spans: `tests/telemetry/test_studio_trace_export.py:107,178,253`; `packages/common/tests/test_telemetry_export.py:163,332,430`.
- `docs/architecture.md:197` says prompts are on ADK's `call_llm` spans.
- The session viewer (`~/.claude/dfo-ai-session-trace/extract.py`, outside the repo) depends on `invoke_agent <name>` span names and `gen_ai.agent.name` (`:256`), `call_llm` (`:298`), a `generate_content` child (`:172`), `gcp.vertex.agent.event_id` (`:290`), `gcp.vertex.agent.invocation_id` (`:478`) and `gen_ai.usage.*` (`:408`). It also restores the system instruction and tools that ADK strips from a logged request under an explicit cache (memory `session-trace-viewer.md`).
- Langfuse derives cached-token usage from `gcp.vertex.agent.llm_response` JSON (memory `langfuse-otel-ingestion-rules.md`). Cache-hit dashboards depend on it.
- The OTel backend and `turn` span are ours and framework-neutral. Server-side MCP spans use `turn_call_meta`, also ours.

A replacement must emit one generation per model call, with prompt, response and usage, and rebuild the session viewer's thread and cache logic.

## 6. Tests and evals on ADK types

Harness tests: 35 of 101 `test_*.py` files import `google.adk` (119 `.py` files under `tests/`). 22 import `google.genai`. Among the 35: `Runner` 12, `InMemorySessionService` 16, `google.adk.events` 12, `LlmResponse` 7, `LlmRequest` 6, `FunctionTool` 6, `google.adk.workflow` 6, `BaseLlm` 4, `ToolContext` 4, `LlmAgent` 3, `_fencing` 3, `_feature_registry` 1, `_batch_executor` 1. Six are under `tests/planner` and `tests/translator`.

Evals: five files mention `google.adk`.

- `evals/planner_entry/entry.py` and `validate.py`: `Runner`, `Workflow`, `RunConfig(StreamingMode.SSE)`, `BaseLlm`.
- `evals/studio/orchestrator_scripted.py`: `BaseLlm`, `Context`, `BaseTool`, `InMemorySessionService`.
- `evals/agents/harness_translator_v1/outcome_metric.py`: `google.adk.evaluation` (`Invocation`, `EvalMetric`, `EvaluationResult`, `PerInvocationResult`, `get_all_tool_responses`).
- `evals/studio/checks.py`: comment only.

`evals/agents/*/agent.py` expose a built `LlmAgent` as `root_agent` for `adk eval`. Three evalsets, 18 cases in all (12 translator, 4 planner, 2 pair). Metrics in `test_config.json`: `tool_trajectory_avg_score` (`in_order`), `rubric_based_tool_use_quality_v1`, `rubric_based_final_response_quality_v1` with a Gemini judge, and one custom `workflow_outcome` metric. The studio eval runs the HTTP surface and does not use ADK evaluation.

ADK-shaped fakes, eight `BaseLlm` subclasses that yield `LlmResponse(content=types.Content(...))` and record `LlmRequest`:

- `tests/conftest.py:237` `ScriptedLlm`
- `tests/orchestrator_fakes.py:65` `ScriptLlm` (also imports `_fencing`)
- `tests/context_probe.py:158` `RecordingLlm`
- `tests/test_planner_seam.py:139` `ScriptedPlannerLlm`
- `evals/planner_entry/validate.py:76` `ScriptedLlm` (calls `bounded`)
- `evals/studio/orchestrator_scripted.py:117,137,169` `OrchestratorLlm`, `PlannerLlm`, `ScriptedPlannerLlm`

Ten test files use a scripted or recording LLM. Tests also call `FunctionTool.func` directly against fake tool contexts (`tests/planner/test_agent.py:1-6`), which keeps tool logic portable. Both `test_prefix_cache.py` and `test_turn_budget.py` build `LlmRequest` and `LlmResponse` directly.
