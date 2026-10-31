# Pydantic AI patterns the prototype does not use

This note lists patterns in Pydantic AI's own documentation, source and release notes that the planned architecture (an owned turn loop over two Pydantic AI agents) should use and the prototype does not. It was written against 2.64.0, which is the latest release: PyPI reports 2.64.0 as current, and GitHub's newest release is v2.64.0 (30 October 2026). The only later GitHub entry is a `clai2-bleeding` prerelease rebuilt from `main`. The release notes of v2.62.0 to v2.64.0 add nothing that changes the sections below. The relevant items are `first_event_timeout` (#11582), `run_stream()` returning a `wrap_run` or `on_run_error` recovery result (#11571), and the UI adapters' `manage_message_history` (#11431).

Paths below are relative to the v2.64.0 checkout at `/private/tmp/claude-501/-Users-jorge--claude-spares-spare-vOcnuX/65792ff6-bdbb-4dff-9c17-171a37d2a0ca/scratchpad/pai-src`. `docs/` is the source of the pages on pydantic.dev/docs/ai, and `SLIM` stands for `pydantic_ai_slim/pydantic_ai`. Nothing here was run against Gemini. Every claim comes from reading the source and docs, so each section carries a confidence and the checks that would raise it.

## Three patterns that matter most

1. Run the planner seam and the turn budget through Pydantic AI's public hooks and settings. The prototype patches the google-genai client instead. The same pass shows that two of the three "no equivalent" ADK behaviours already have one, and that the turn budget has a trap in `conversation=` (section 5).
1. Build the explicit Gemini cache prefix from our own canonical declarations. Pydantic AI strips the instruction, tools and tool config from any request that names a cache, so the prefix is ours to define. Doing so lets ADK and Pydantic AI share one cache and takes the largest part of the prompt out of reach of release-to-release changes.
1. Checkpoint the model response before its tools run. A kill inside `propose_plan` then re-runs that one call, not the whole planner run with a fresh model call and a fresh tool call id.

Corrections to the cost list that fall out of this reading:

- Cost 4: calling an unknown tool is already answered with the list of available tools, and an empty response is already re-prompted. The prototype saw an exception because the retry budget defaults to 1. Only the mid-stream cut needs loop code.
- Cost 3: after section 2, ADK and Pydantic AI differ only on requests sent without a cache.
- Cost 1: section 3 narrows the re-run. It does not remove it, and no durable engine does either (section 8).

## 1. Use the public request pipeline for the seam and the turn budget

What it is. Pydantic AI exposes the points the prototype reaches by monkey-patching `client.aio.models.generate_content_stream` (`PlannerClient`, `_Pending`, `_RAW_CHUNK` in `pai_planner.py`):

- `before_model_request` or `wrap_model_request` can raise `SkipModelRequest(response)` to answer a request without calling the model (`docs/hooks.md:180`, `SLIM/exceptions.py:208-222`).
- `get_model_settings` can return a callable that sets model settings per step from the `RunContext` (`docs/capabilities/custom.md:370-398`).
- `ModelSettings` has `request_timeout`, a clock over a whole request including SDK retries and the stream read, and `first_event_timeout`, a deadline on the first streamed output (`docs/timeouts.md:13-24`, `:43-115`). Whatever has streamed when a deadline passes stays in history as a response with `state='interrupted'`.
- A run takes `timeout=` and raises `RunTimedOut` carrying `all_messages()` (`docs/timeouts.md:117-145`). `UsageLimits(request_limit=...)` stops a run before the request that would exceed it, and a documented capability tells the model to wrap up and removes function tools on the last allowed request (`docs/agent.md:1153-1218`).
- `on_tool_execute_error` can return a result in place of an exception, and `ToolFailed` reports a failed result without spending the retry budget (`docs/capabilities/custom.md:1080-1090`, `docs/hooks.md:557-631`).
- `RetryModelRequest`, raised from `on_model_request_error` or `after_model_request`, resends the same step without telling the model (`SLIM/exceptions.py:225-240`).
- `strict=False` on any tool keeps the whole Gemini request on `AUTO` (`docs/tools-advanced.md:226-236`, `SLIM/models/google.py:1059-1070`).

How the turn budget in `harness/turn_budget.py` maps onto them:

| Today (ADK seam)                                                   | Public equivalent                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A call that finds a bound spent answers with that bound's sentence | `before_model_request` raises `SkipModelRequest` with a one-part text response                                                                                                                                                                                                       |
| The planner seam's synthetic answer (`_before_planner_model`)      | The same hook                                                                                                                                                                                                                                                                        |
| 30 model calls per request, shared by both agents                  | `UsageLimits(request_limit=30)` on one `RunUsage` per turn, plus the wrap-up capability so the last request produces text                                                                                                                                                            |
| 600 s wall clock from the request start                            | `timeout=` set to the time left when each run starts; `RunTimedOut`                                                                                                                                                                                                                  |
| 12 s first chunk, one resend for the first miss in a turn          | `first_event_timeout=12`, and a per-turn capability that raises `RetryModelRequest` once                                                                                                                                                                                             |
| 90 s of model time                                                 | A settings callable that sets `request_timeout` to the model time left, with the spend added in `after_model_request`                                                                                                                                                                |
| A raising tool answers with its name, class and a `raised` code    | One `on_tool_execute_error` hook, replacing the try/except in `_tool_body`                                                                                                                                                                                                           |
| Unknown tool answered with the tool list                           | Already done: `SLIM/tool_manager.py:807-825` raises a retry that lists the available tools. Raise the budget with `retries={'tools': N}` per run (`docs/retries.md:554`, `docs/tools-advanced.md:626-644`)                                                                           |
| Empty model response                                               | Default is a re-prompt up to the output budget, then `UnexpectedModelBehavior` (`docs/models/overview.md:741-747`). Either raise `retries={'output': N}`, add `None` to `output_type` (`docs/output.md:874`), or resend silently with `RetryModelRequest` from `after_model_request` |
| `AUTO` function calling                                            | `strict=False` on the tools, or one `prepare_tools` hook that sets it, in place of the `google_supports_strict_tool_definition` profile key                                                                                                                                          |

The mid-stream cut is the one behaviour with no hook. When `RunTimedOut`, `ModelRequestTimeout` or `UsageLimitExceeded` ends a run, the loop takes the messages (`exc.all_messages()`, or `capture_run_messages` for the usage case, `docs/agent.md:2266-2305`), drops the trailing `ModelResponse` whose `state` is `'interrupted'`, and appends the bound's sentence as the final response (`docs/agent.md:973-1000`, `docs/message-history.md:379-424`). That answers cost 4's "keeps only a closing sentence" in one function the loop owns.

What it changes in the prototype. `PlannerClient`, `_Pending` and the chunk plumbing go. So does the private `PlannerSeam` callback wiring, which becomes one capability with `before_model_request`, `after_model_request` and `on_tool_execute_error`. The seam then runs under `FunctionModel` with no HTTP, which is cheaper than the tap-based tests. The prototype's remaining private names shrink to one (`GoogleModel._build_content_and_config`, removed by section 2).

Moves. Cost 2 (fewer private names to break across three releases a week), cost 4 (two gaps closed, one reduced to a loop function), and the owned, testable turn.

Confidence. High that each hook and setting exists as described. Medium that `on_model_request_error` receives `first_event_timeout` on a streamed request (`docs/timeouts.md:71` speaks only of non-streamed requests), so test that before relying on one resend. Medium that the loop-level cut matches ADK's stored history; ADK also sends the partial closing sentence to Studio, so compare the frames.

## 2. Own the explicit-cache prefix

What it is. With `google_cached_content` set, Pydantic AI sends `system_instruction=None`, `tools=None` and `tool_config=None` and puts only the cache name and contents on the request (`SLIM/models/google.py:1225-1229`, `:1268-1288`; `docs/models/google.md:485`; `docs/capabilities/caching.md:140-144`). The cache resource alone decides what Gemini sees for those three fields. Pydantic AI still needs its own `ToolDefinition`s to parse calls.

The prototype builds the cache key and the cache contents from Pydantic AI's own built config (`uncached_config`, `prefix` and `_cached` in `pai_planner.py:463-513`). `prefix_key` hashes the model, instruction, tools and tool config (`harness/prefix_cache.py:77-90`). Pydantic AI's shape differs from ADK's in grouping (one `Tool` per declaration) and in `reject_plan`'s empty schema, so the two frameworks create two caches for one prompt. That is cost 3.

Change. Build the prefix once from canonical declarations: the instruction text, every declaration inside one `Tool(function_declarations=[...])` as ADK declares them, and `tool_config=None`. During the migration these come from `PlannerParts.declarations`, which the prototype already reads off the ADK agent. After it, a small function turns each `ToolDefinition` into a `FunctionDeclaration` and a golden test pins the bytes. Hash that prefix with the existing `prefix_key`, create the cache from it, and hand Pydantic AI the name.

Effects:

- ADK and Pydantic AI requests hit the same `CachedContent` (cost 3, for the 85% of prompt tokens that arrive through the cache).
- The bytes that carry most of the prompt no longer move when a Pydantic AI release changes how it renders declarations (cost 2).
- `AUTO` follows from the absent tool config on cached calls.
- `GoogleModel._build_content_and_config`, the prototype's one private call, is no longer needed.
- `reject_plan` is declared as ADK declares it.

Two hazards need guards. First, Pydantic AI drops dynamic instructions and per-turn tool filters silently apart from a `UserWarning`, which `pai_planner.py:90` ignores globally. Replace that filter with an assertion that the instruction and tools Pydantic AI would have sent equal the canonical prefix, failing in tests and logging once in production. Second, for the same reason `PrepareTools` filtering and a `tool_choice` name list cannot work under the cache. Keep the planner's tool list static.

Requests sent without a cache (a create refused under the 1,024-token minimum, the 60 s cooldown, a cache gone) still carry Pydantic AI's own shape, so the `strict=False` pin and the parity tests stay for that path.

Moves. Costs 2 and 3, and the benefit of an owned, testable turn.

Confidence. Medium to high. It follows from reading Pydantic AI's stripping logic and the harness cache key, and no Pydantic AI document recommends it. A paired live run should show the same cache name from both arms with unchanged acceptance and first-chunk latency.

## 3. Checkpoint the model response before its tools run

What it is. Three documented behaviours combine:

- A history that ends on a `ModelResponse` with tool calls runs those calls when the next run has no new prompt, and the calls keep their original ids (`docs/message-history.md:398`).
- Harness `StepPersistence(capture_frontier=True)` saves exactly that frontier before tool execution, marked `interrupted` (`docs/harness/step-persistence.md:631-641`, `:163-171`).
- Its tool-effect ledger records `started` and `completed`, so a call with no end reads as unknown after a crash (`docs/harness/step-persistence.md:20-23`, `:289-308`).

The prototype checkpoints once per agent run, at its end, so a kill inside `propose_plan` falls back to the start of the planner run (`prototype-results.md:21`). The model is asked again, can propose a different plan, the call gets a new id, and its progress frames go out twice.

Change. A short per-run capability (`for_run` returns a fresh instance, `docs/capabilities/custom.md:1203-1243`) writes a frontier row from `after_model_request` into our store, inside the same compare-and-swap transaction as the cursor. `before_tool_execute` and `after_tool_execute` write the ledger rows. `recover()` then runs the agent with `message_history` set to the frontier and no prompt. Pending calls execute with their own ids. Custom events are stamped with `tool_call_id` and `tool_name` automatically (`docs/agent.md:320`), so `append_frame` can ignore a frame key it already holds and the client receives one `resumed` marker. A call whose ledger row is `started` and which is not idempotent forks to a fresh planner run instead.

To keep each frontier write small, store messages as rows appended per run with `new_messages()` and keep `usage`, `conversation_id` and `deferred_tool_requests` in columns beside them, as `docs/persistence.md:49` and `docs/message-history.md:709` describe. The format stays `ConversationTypeAdapter` and `ModelMessagesTypeAdapter`, so stored records keep their current compatibility record.

Borrow the idea and leave the package. `StepPersistence` is Alpha (harness 0.64.0), its snapshots hold messages only and lose usage and deferred requests (`docs/persistence.md:87`), and we already own the cursor and the compare-and-swap.

Moves. Cost 1: the re-run shrinks to the one interrupted call and the repeated progress frames disappear. Stable stored records stay stable.

Confidence. Medium. The mechanism is documented, but the saving is unmeasured. Extend `observe_kill.py` to kill inside `propose_plan` and compare model calls and frames with the current loop. Confirm what `ctx.messages` holds at `after_model_request` before relying on it.

## 4. Guard the wire and the upgrade

What it is. Four cheap controls for cost 2:

- Run the suite with `-W error::pydantic_ai.exceptions.PydanticAIDeprecationWarning`. Deprecations are removed only in the next major, and V3 is allowed from 23 September 2026 (`docs/version-policy.md:9`). The warning class is named at `docs/hooks.md:173`.
- Alert on the prompt-cache span attributes Pydantic AI records on every model-request span: `pydantic_ai.cache.hit_ratio`, `.collapsed`, `.collapse_reason`, and the `pydantic_ai.cache.collapse` event (`docs/logfire.md:267-306`). Google publishes no retention window, so a drop is classed `unknown` and does not raise the event. Alert on `collapsed=true` instead. Use `WarnOnCacheBusts` from the Harness only in CI, because it is Alpha (`docs/harness/warn-on-cache-busts.md:50`).
- Snapshot what the model would be sent. `FunctionModel` hands the script an `AgentInfo` with `function_tools`, `instructions`, `model_settings` and `model_request_parameters` (`SLIM/models/function.py:262-293`). An inline-snapshot of those (`docs/testing.md:15-18`) fails when a release changes a declaration, with no HTTP. Keep the tap run (`gemini/tap.py`) as the live check on each bump.
- Pin the exact version and run `stability/matrix.py` on every bump, as already done.

Pydantic AI checks every recorded provider conversation in its own suite for wire-level prefix stability (`docs/capabilities/caching.md:187`). That protects its conversations, not our declarations.

Moves. Cost 2.

Confidence. High that each control exists; the value is modest and the work small.

## 5. Count the turn budget on a fresh `RunUsage`

What it is. `conversation=` carries the conversation's running usage, so a `UsageLimits` budget covers the whole conversation and not each run (`docs/persistence.md:45`, `docs/message-history.md:654-660`). It also cannot be combined with `usage=`, `message_history=` or `conversation_id=` (`docs/message-history.md:700-702`). The prototype passes `conversation=` to every run (`loop.py:277-279`). A per-turn `request_limit=30` would therefore count every earlier turn of the conversation.

Change. Load the stored `Conversation`, then run with `message_history=conv.messages`, `conversation_id=conv.conversation_id`, `deferred_tool_results=` built from `conv.deferred_tool_requests`, and one `usage=RunUsage()` created per turn and shared by the orchestrator and planner runs, as ADK's `TurnBudget` is shared. Keep the cumulative figure in our own column for spend reporting.

Moves. The owned turn (a budget that means what it says).

Confidence. Medium to high. The doc statements are explicit; confirm by running a second turn under `request_limit=2` in a test.

## 6. Build the Studio translator on `UIEventStream`, and pin event names

What it is. `UIEventStream` is a public generic base that holds the per-run state a translator otherwise hand-writes (message id, the open part, pending tool calls). It has one overridable `handle_*` method per native event, plus `on_error`, `on_cancelled` and an abstract `encode_event` (`SLIM/ui/_event_stream.py:100-211`, `:428-990`). It runs on its own with no request or run input (`docs/ui/overview.md:132-149`). Custom events arrive through `handle_custom_event` and their payload through `CustomEvent.to_payload()`, and `ui=False` keeps one server-side (`docs/agent.md:431-447`).

Change. Make the Studio translator a `UIEventStream` subclass that yields today's frames. It replaces the `match` in `_run_agent` (`loop.py:281-298`) and the ADK translator's node-path logic. New native events in a later release then reach an overridable method with a default no-op, not a silent fall-through. Pin custom event names: `class PlanProgress(CustomEvent, name='plan_progress')`, because the name is the wire identifier in any serialised event and a rename is a compatibility break (`docs/agent.md:421-425`). The prototype's `PlanProgress` (`agents.py:48`) leaves the name derived from the class.

Moves. Cost 6, and the owned, testable turn.

Confidence. Medium. Not prototyped. The frozen frame contract rules out the AG-UI and Vercel adapters themselves; only the base class is useful. New event types may be added in minors (`docs/version-policy.md:18`), which is the reason for the default no-ops.

## 7. Langfuse: pin the conventions version and set the first-chunk time

What it is. Pydantic AI's default OpenTelemetry format version may change in a minor release (`docs/version-policy.md:19`), and `InstrumentationSettings(version=5)` pins it (`docs/logfire.md:343-347`). `Agent.instrument_all(settings)` with an ordinary tracer provider needs no Logfire (`docs/logfire.md:222-259`). Langfuse's own page documents this path only: `Agent.instrument_all()` plus the Langfuse SDK, with `propagate_attributes` for session, user and tags. It says nothing about the semantic-convention version, tool spans or cached tokens. The ADK path sets `langfuse.observation.completion_start_time` and `dfo.model.chunk_gaps_s` (`harness/turn_budget.py:60-63`). Pydantic AI records `gen_ai.client.operation.time_to_first_chunk` instead (`docs/logfire.md:316`). Capabilities may add attributes to the spans Pydantic AI creates through the OpenTelemetry API (`docs/capabilities/instrumentation.md:24`).

Change. Pin `version=5`, keep `use_aggregated_usage_attribute_names` at its default so agent spans do not double count (`docs/logfire.md:323-341`), and add one hook that sets the two Langfuse attributes on the model-request span.

Moves. Cost 5.

Confidence. Medium. Whether Langfuse maps every v5 attribute is unverified. Compare one trace from each arm for names, usage, cached tokens, tool spans and session id before cutover.

## 8. Durable engines fit beside the loop, for detached work only

The answer is beside, and the docs give reasons beyond the README's. An engine journals the model requests and tool calls of one agent run. A paused turn already ends that run with `DeferredToolRequests` (`docs/deferred-tools.md:21`), so the pause lives in our cursor, which survives deploys.

Specific limits for the interactive turn:

- DBOS and pgtask consume the model stream inside the durable unit and hand the response back whole, so model text does not stream live (`docs/durable_execution/dbos.md:217`, `docs/harness/pgtask.md:130-131`). Plain function tools under DBOS run in workflow code, so `ctx.emit` from them stays live (`dbos.md:221`, `:233`).
- DBOS pickles workflow inputs, so the `Deps` object must serialise (`dbos.md:187`), and a `CancellationToken` cannot be passed (`dbos.md:237`). Temporal copies dependencies and gives activities a limited `RunContext` (`docs/durable_execution/temporal.md:219-245`).
- Changing the order of recorded steps across a deploy raises `DBOSUnexpectedStepError` on recovery (`dbos.md:225`). That is the deploy hazard the cursor design avoids.
- Every engine re-runs the tool that was in flight (`docs/durable_execution/overview.md:9`, `docs/harness/pgtask.md:124-126`), so none improves on section 3 for cost 1.

For work that leaves the user's stream, DBOS is an in-process library on Postgres, is co-maintained with Pydantic, and needs no new service (`docs/durable_execution/overview.md:27`). pgtask is a smaller Postgres queue but sits in Harness 0.x. Only one engine may be attached to an agent (`overview.md:29-30`).

Confidence. High on "beside"; medium on DBOS over pgtask, which needs a trial on one detached job.

## 9. Test with the library's own helpers and its migration checklist

What it is.

- `FunctionModel(stream_function=...)` can yield `DeltaToolCall` and `DeltaThinkingPart` with a `signature` (`SLIM/models/function.py:296-314`, `docs/agent.md:342-359`). A script can therefore emit signed and unsigned thinking and exercise `drop_unsigned_thoughts` without Gemini.
- `models.ALLOW_MODEL_REQUESTS = False` in `conftest.py` stops any test from reaching a real model (`docs/testing.md:21`).
- The checkout ships a migration guide from ADK to Pydantic AI at `SLIM/.agents/skills/migrating-google-adk-to-pydantic-ai/`. It asks for a contract ledger with one row per caller-visible behaviour, marked preserved, changed, not applicable, unverified or blocked, and says no traffic moves while a row is unverified (`references/VERIFICATION-AND-CUTOVER.md:5-19`, `:50-58`). It lists a kill at each effect boundary under restart tests and says "A normal second run is not a restart test" (`:41`, `:54`). It warns that ADK resume can run tools more than once, and that approval needs correlation, duplicate-response and restart checks (`references/SEMANTIC-GAPS.md:17`, `:35`). It recommends plain async Python over `pydantic_graph` for simple deterministic control flow (`SKILL.md:22`), which matches the README.

Change. Turn `prototype-results.md` into the ledger and add rows for what is still open: the planner's tools on our own records, live progress through the real translator, and a live-Gemini `Conversation` round trip across releases.

Moves. Cheap scripted tests and the owned turn. Confidence high.

## Checked and rejected

- AG-UI and Vercel AI adapters: the Studio frame contract is frozen and the adapters emit their own protocols; only the `UIEventStream` base is used (section 6).
- Temporal: needs its own service, copies dependencies, limits `RunContext` in activities, and buffers streaming (section 8).
- `HandleDeferredToolCalls` for the user pauses: it resolves inside one process, and our pauses last days across deploys (`docs/deferred-tools.md:20-21`).
- `StepPersistence` as the store: Alpha, messages only, no usage or deferred requests in snapshots (section 3).
- `pydantic_graph` and graph persistence: removed in V2, and the turn has no fan-out between nodes (README).
- `Caching()` capability: no effect on Google (`docs/capabilities/caching.md:31-33`).
- `Thinking(effort=...)`: the right portability knob once a second provider arrives, but today byte parity with ADK's `thinking_config` matters more (`docs/models/google.md:334-378`).
- `ModelFallback` and `FallbackModel`: one provider today. Their `canonical_system` handling already keeps Gemini signatures from crossing transports (`docs/message-history.md:827-852`), so stored `Conversation`s will move providers safely later.
- Tool search and per-turn `PrepareTools` filtering: they change the tool list, and the explicit cache owns it (section 2).
- Moving `drop_unsigned_thoughts` into `after_model_request`: `ProcessHistory` works and persists its result (`docs/hooks.md:175`), so there is no measured gain.
- Harness `AskUser`: its pause with `answerer=None` is our pattern, but our question schema and frame are fixed.
- Logfire and the Pydantic AI Gateway: Langfuse over OpenTelemetry is already the destination.
- Record and replay of provider traffic: none ships in the library; the tap and `FunctionModel` cover it.

## Checks that would change a verdict

- Whether `on_model_request_error` receives a streamed `first_event_timeout` (section 1).
- Model calls and duplicate frames after a kill inside `propose_plan`, with and without a frontier checkpoint (section 3).
- One Langfuse trace per arm compared field by field (section 7).
