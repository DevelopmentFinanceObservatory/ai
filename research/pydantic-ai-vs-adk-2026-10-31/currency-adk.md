# ADK: state, trajectory and what an upgrade does for dfo-ai

Latest version: google-adk 2.13.0, released 2026-10-23 on PyPI (https://pypi.org/project/google-adk/; changelog dated 2026-10-22, https://github.com/google/adk-python/releases/tag/v2.13.0). We run 2.10.0 (2026-09-25). Checked 2026-10-31 against PyPI, the GitHub repo (source read at tags v2.10.0 and v2.13.0 and at main, 8cd3b12c, 178 commits past 2.13.0), and https://adk.dev. "Installed" means `/Users/jorge/repos/dfo-ai/.venv/lib/python3.13/site-packages/google/adk` (2.10.0). Paths in `src/google/adk/...` are read at v2.13.0 unless stated.

Verdict in one line: an upgrade fixes the resume, parallel-tool and cache-config rough edges Google has been fixing, leaves the four design-level defects in our notes untouched, and costs a small import repair plus a telemetry decision. Staying on ADK does not remove the structural problems; leaving does not remove ADK's pace of change as a risk if we stay.

## 1. Versions and cadence

Three releases follow 2.10.0 (compare links: https://github.com/google/adk-python/compare/v2.10.0...v2.13.0).

- 2.11.0 (2026-10-02): `abort_signal` on `Runner`, `Workflow` and nodes; tool nodes pause for confirmation through `RequestInput`; `ModelConsultTool`; SQLite memory service; opt-in MCP SDK 2.x path. No breaking change flagged; the maintainers later removed two false-positive flags (2.12.0 notes, commit d8dc299). Unflagged but breaking for us: the `flows.llm_flows._fencing` shim was deleted (commit 2e5b2f62, first tag v2.11.0).
- 2.12.0 (2026-10-13): 40% faster time to first model request in long sessions; `after_tool_callback` screening; `require_confirmation` on workflow nodes; Python 3.15; move to the `agentplatform` SDK; many resume fixes (`decide_step_resume` unified, tool sub-branch events kept out of the calling agent's context, `rerun_on_resume` nodes wait for all interrupts). Nothing flagged.
- 2.13.0 (2026-10-22): one flagged break, telemetry. Default is now OpenTelemetry GenAI semconv 1.44 and the `call_llm` span is gone; `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` restores the old shape, "supported at least until 31st of March 2027" (commit 7838488). Also: 12 experimental features promoted to stable (see 2), `ClassifierNode` and a TypeSafe Jev classifier, `MongoDbSessionService` (experimental), up to 8x faster turns for string-model agents, and floors raised to `google-genai>=2.30` (2.10 asked `>=2.19`) and `litellm<1.106.3` (`pyproject.toml` at both tags). Our earlier study ran on google-genai 2.24.0, so the upgrade moves genai too.

Cadence: 37 PyPI releases since 2026-05-01 (https://pypi.org/pypi/google-adk/json). 2.0.0 went GA on 2026-05-19, then a minor every 10 to 14 days (2.1.0 on 05-23 to 2.13.0 on 10-23) with patch releases between. The 1.x line ended at 1.39.1 on 2026-08-27 with no stated end-of-life date (https://adk.dev/2.0/). Breaking-flagged releases since 2.0: 2.2.0, 2.7.0, 2.13.0; I read each release body for the word BREAKING. Most breaks reach us through private modules, which no release note covers.

## 2. Stability of Workflow, HITL and the private modules we import

- Workflow/graph API: GA with ADK 2.0 on 2026-05-19; the 2.0 page marks nothing experimental (https://adk.dev/2.0/). No `FeatureName` gates it (`features/_feature_registry.py`). The engine still churns: replay and HITL bugs were fixed in 2.6.0 (#6497), 2.10.0 (#7027), 2.12.0 (#7286) and 2.13.0 (#7502, #5617). Two more fixes, #7618 and #7695, sit on main unreleased.
- `google.adk.workflow.utils._workflow_hitl_utils`: still there, still underscore-private, no public replacement. The docs name its helpers (`docs/guides/events/request_input/index.md:9`) but export none. Public pieces are `google.adk.events.RequestInput` and `ctx.resume_inputs`. Between 2.10.0 and 2.13.0 the module was rewritten (226 changed lines):
  - `REQUEST_INPUT_FUNCTION_CALL_NAME` moved to `utils/_function_call_names.py`. `from ..._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME` still works only because the module imports the name. We do that in `services/harness/src/planner/agent.py:59`, `planner_seam.py:109`, `studio_translate.py:61`, `submission/service.py:49` and four more files.
  - `REQUEST_CREDENTIAL_FUNCTION_CALL_NAME`, `has_request_input_function_call`, `has_auth_request_function_call` and `get_request_input_interrupt_ids` are gone. We import none of them.
  - `create_request_input_event` and `create_request_input_response` survive with unchanged signatures (`workflow/utils/_workflow_hitl_utils.py:105,130`). New helpers add tool-confirmation ids (`wf_tool_confirmation:<node_path>`).
- `flows/llm_flows/tools/_batch_executor.py`: exists; `deep_merge_dicts` is unchanged (line 52). The file was refactored (285 lines changed), including the 2.13.0 fix that keeps finished parallel tool results when a sibling raises (7d3658ef, #7428).
- `flows/llm_flows/_fencing.py`: deleted in 2.11.0. Constants and `_present_other_agent_message` live in `flows/llm_flows/context/_fencing.py` (lines 41, 45, 255). `services/harness/tests/orchestrator_fakes.py:30` imports the old path and fails on upgrade. Several test docstrings and `evals/studio/checks.py:1328` cite the old path in prose.
- `features/_feature_registry.py`: present; `FeatureName`, `is_feature_enabled`, `override_feature_enabled` are exported from `google.adk.features`, and `temporary_feature_override` is at line 445 (our `tests/planner/test_agent.py:40`). 2.13.0 promoted `JSON_SCHEMA_FOR_FUNC_DECL` to stable (line 208), which `planner/declaration.py:45` reads; `FUNCTION_TOOL_ARG_VALIDATION` stays experimental and off (line 190). Promotion also stripped the `@experimental` decorators from `ContextCacheConfig` and `TOOL_CONFIRMATION`.

Pattern: ADK reorganises private paths without notice, and two of the four modules we import changed shape in three minors. Expect another repair on each upgrade.

## 3. Our six defects

1. Tool args unvalidated, a raise ends the turn. Unchanged at 2.13.0. `FunctionTool._preprocess_args_with_validation` returns early unless `FUNCTION_TOOL_ARG_VALIDATION` is on, and the flag is experimental and off (`tools/function_tool.py:285-296`). An exception with no `on_tool_error_callback` is re-raised (`flows/llm_flows/tools/_caller.py:582-598`). Open issue: https://github.com/google/adk-python/issues/7574 (opened 2026-10-16; Go recovers by returning the error to the model, Python aborts). Related and closed in 2.13.0: https://github.com/google/adk-python/issues/7428 (parallel siblings lost their responses). Only `NodeTool` gained an error dict for rejected args (d0c8454e). Open: https://github.com/google/adk-python/issues/7280 (schema validation skipped for non-BaseModel types). Our `_validated_arguments` workaround stays needed.
1. Unbranched events reach every agent. Unchanged. `_is_event_belongs_to_branch` still returns True when either branch is empty (`flows/llm_flows/context/_contents.py:776`; installed `_contents.py:671`), and `_fencing._present_other_agent_message` still quotes foreign events. 2.12.0 and 2.13.0 only filter tool sub-branch events and single-turn node events. I found no issue about graph-node interrupts; unconfirmed whether Google considers it a defect. Our fix (give interrupts a branch) still applies.
1. Resume replays from START. Unchanged by design: `check_interception` replays only nodes with a recorded run (`workflow/utils/_replay_interceptor.py:57`). The docs promise "completed nodes do not run a second time" (`docs/guides/workflow/workflow/index.md:10`), but open issue https://github.com/google/adk-python/issues/7666 (node with no output reruns after `RequestInput`) and https://github.com/google/adk-python/issues/7609 show the gap. Our `_turn_ran_before` guard stays.
1. single_turn run boundary. Unchanged. `_SingleTurnAgentTool` still runs the sub-agent in the parent invocation and separates runs by sub-branch built from the function call id (`tools/agent_tool.py:501-513`). Fixed in 2.11.0 to 2.13.0: #7227 (node inputs leaking into the root request), #7488 (parallel single-turn runs anchoring on a sibling's output). Our branch-based scoping matches what Google now does.
1. Explicit cache blocks implicit. Unchanged. `gemini_context_cache_manager.py` has no edits since 2.10.0; the fingerprint still hashes tools, so any tool change drops the explicit cache. Open requests: https://github.com/google/adk-python/issues/7241 (cache-stable progressive tool disclosure, 2026-09-22, one comment) and https://github.com/google/adk-python/issues/7536 (app-scoped static caching). Our `PrefixCache` replacement stays; ADK does not offer an alternative. The implicit-versus-explicit behaviour is Gemini API behaviour, not ADK's.
1. Persisted `app_name`. ADK keys sessions on `(app_name, user_id, session_id)`; nothing in 2.11 to 2.13 renames or migrates keys. 2.13.0 adds nothing that conflicts. Never rename it, on any framework that stores sessions this way.

## 4. DatabaseSessionService schema

No DDL changes from 2.10.0 to 2.13.0. `schemas/v1.py` (JSON events, introduced 2025-12-15 in 1.22.0) is still `LATEST_SCHEMA_VERSION` (`sessions/migration/_schema_check_utils.py:42`). The one model diff drops `onupdate=func.now()` on `update_time` (aec7c0ad), an ORM-side default with no DDL. The `id DESC` events index arrived in 2.10.0 (a00a2a97), which we run; whether `create_all` adds it to a table that already exists is unconfirmed. Behaviour fixes: events purged before the session row on delete, equal-timestamp events keep append order (#7286). A v0 pickle to v1 migration exists (`sessions/migration/`), but there is no Alembic; the RFC is open (https://github.com/google/adk-python/issues/4387). A docs commit now treats new fields in storage services as breaking (f03015c), which promises notice for future schema moves. The new `sqlalchemy[asyncio]` extra requires `greenlet`.

## 5. Google's investment

- Contributors: 413 distinct commit authors and 2,504 commits on main since 2026-05-01, 605 in October alone; 1,828 from google.com addresses (git log at 8cd3b12c). Top authors: George Weale 649, Shangjie Chen 309, Kathy Wu 114.
- Responsiveness: of 148 issues opened 2026-09-15 to 2026-10-15, 112 are closed (104 completed, 8 not planned), median 9.1 days to close, 75th percentile 18.5 (GitHub API). 414 open issues and PRs on 2026-10-31.
- Ports, latest releases: Java 1.13.0 (10-21), Go 2.6.0 and 1.9.0 (10-13, 10-15), TypeScript 2.4.0 (10-28). Go GA on 2026-06-30 and TypeScript GA on 2026-08-21 (https://adk.dev/2.0/). The docs list Python, TypeScript, Java, Go and Kotlin.
- Platform: Vertex AI became Gemini Enterprise Agent Platform in April 2026, and Agent Engine became Agent Runtime (secondary sources, https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime/sdk-migration; unconfirmed in a primary release note). ADK 2.12 migrated to the `agentplatform` SDK and 2.12 added Agent Runtime deploy options. ADK is the Python SDK Google deploys there. adk.dev has no roadmap page; a community-call mention is the only reference.
- Consolidation signals: `SequentialAgent`, `ParallelAgent` and `LoopAgent` are marked deprecated in the README (2.13.0, c9f53a6) in favour of Workflow. Agents are nodes; the old `_run_async_impl` override is silently ignored. I found no sign of ADK itself being deprecated.

## 6. Features we may not use

- `abort_signal` (2.11.0): graceful cancel, and `/run_sse` cancels on client disconnect. Guide: `docs/guides/runners/runner/abort.md`.
- `require_confirmation` on nodes and tool-node pauses (2.11, 2.12): a framework route to the approval pause we built by hand in `graph.approve`. Open issues show rough edges (#7650, #7685, #7737).
- `after_tool_callback` and `on_tool_error_callback`: the supported place to turn tool exceptions into error dicts. Both agents already set `on_tool_error_callback` (`orchestrator/agent.py` at HEAD line 371, `harness/planner_seam.py:438`); the planner also sets an after-tool callback.
- `ReflectAndRetryToolPlugin`, `ToolCallIntegrityPlugin` (`docs/guides/plugins/`).
- `ResumabilityConfig` (still experimental) and `EventsCompactionConfig` (experimental); `ContextCacheConfig` is stable now but cannot cope with changing tool lists.
- Eval: `docs/guides/evaluation/` with efficiency evaluators and Dev UI runs. Our evals are custom.
- A2A: `docs/guides/a2a/`, with HITL resume fixes in 2.11 and 2.12.
- Telemetry: `TelemetryConfig` captures prompt and response text on spans by default. The 2.13 schema change moves content to `gen_ai.*` attributes and log events, so the Langfuse mapping in our OTLP notes needs a re-check. Pinning `otel_semconv_1_36` buys until at least 2027-03-31.
- MCP: SDK 2.x path is opt-in and experimental (`_MCP_MODERN_PROTOCOL`, default off).
- Jev: ADK 2.13.0 ships `TypesafeClassifier`/`ClassifierNode` (https://github.com/google/adk-python/releases/tag/v2.13.0), relevant given we already use Jev.

## 7. Non-Gemini models

ADK ships `LiteLlm` (`models/lite_llm.py`, 4,694 lines), a native `AnthropicLlm` (`models/anthropic_llm.py`, 1,661 lines), Apigee, Gemma and a fallback-model wrapper (`models/_fallback_model.py`). Both non-Gemini adapters are under heavy repair: 34 commits to `lite_llm.py` and 28 to `anthropic_llm.py` since 2026-08-01 (git log), including October fixes for truncated Claude tool calls, `tool_config` mode, output schema without tools, thinking config, and cache-token counts. Anthropic prompt caching is implemented (`models/_prompt_cache.py`). Issues still open: https://github.com/google/adk-python/issues/7480 (history Claude signed changes when single_turn turns are added) and #6880 (custom `BaseLlm` request mapping). I did not test either adapter against our tool set; unconfirmed. Net: usable as a second provider, but expect bugs in exactly our shape of use, single-turn sub-agents plus tools plus thinking.

## Reasons to stay

- The upgrade is bounded: three minors, one flagged break (telemetry), one deleted private path used in a single test file, and a rewritten HITL module that still exports the three names we use. Estimate under a day, plus a Langfuse trace check.
- Resume and parallel-tool bugs we may hit (#7286, #7428, #7502, #5617, #7227, #7488) are fixed in 2.12.0 and 2.13.0, and #7618 and #7695 are on main.
- Investment is large and rising: 605 commits in October, 9-day median to close an issue, four language ports at GA, Google's own agent platform deploys it.
- Workflow, `RequestInput`, `ContextCacheConfig`, `JSON_SCHEMA_FOR_FUNC_DECL` and `TOOL_CONFIRMATION` are now stable or GA, and `abort_signal` and `require_confirmation` cover pieces we hand-built.
- `app_name` and the stored sessions need no migration. A framework switch makes that a data migration.
- Strongest objection to leaving: the cost of a rewrite is certain, while the benefit is unverified here.

## Reasons to leave

- The four design defects stay: raw tool args and aborting exceptions (#7574 open), unbranched graph events quoted into every agent, replay that reruns unrecorded nodes (#7666 open), and a cache fingerprint that tracks tools (#7241 open). Each is on our workaround list and none is scheduled.
- Resume correctness is still moving. Six replay or HITL fixes landed in 2.6 to 2.13, two sit unreleased, and several are open (#7666, #7609, #7685, #7650). A human-in-the-loop product depends on that area.
- The surface we import is private and rewritten without notice; 2.10 to 2.13 alone removed one module and four functions we could have used.
- Pace: a minor every 10 to 14 days, a telemetry schema flip in a minor release, and dependency bounds that move (`google-genai>=2.30`, `litellm<1.106.3`, `opentelemetry-api<=1.44`).
- Non-Gemini support exists but is mid-repair, and its open bugs overlap our single-turn pattern.
- Strongest argument against this list: every item is fixable inside ADK with our own wrappers, as we have done, and a new framework brings its own set. Whether Pydantic AI's resume and event model avoids these four is the other study's question.
