# A Pydantic AI planner over plan-mcp

Verdict: all five pass lines hold. Pydantic AI's stock MCP client gives its model the same tools and the same answers as `harness-ab`'s arm B, character for character, and the planner's pause and resume work through a deferred tool and a serialised history. The adapter is 79 code lines against 270 for `harness-ab/mcp_planner.py`, but it is not a like-for-like count (see "Harness-side lines"). One premise of the question was out of date: Pydantic AI 2.64 answers a tool error with a retried `ToolReturnPart`, not a `RetryPromptPart`. The one real cost of the stock client is its error handling: plan-mcp's server faults reach the model as retries that use up the tool's retry budget unless `tool_error_behavior="failed"` is set.

Tests: 44 pass in 22 s, twice in a row. `ruff format` and `ruff check` are clean.

## What was built

`pai_plan_mcp.py` is the harness-side code. It uses Pydantic AI's own pieces and nothing else in the tool path:

- `MCPToolset` is the MCP client, over FastMCP's in-memory transport or a URL.
- `process_tool_call` fills `plan_session` from the run's deps and adds the request `_meta` that `meta()` returns.
- `PreparedToolset` (`.prepared`) is the model's view. One prepare function drops `start_plan` and `run_plan`, hides `plan_session` from `propose_plan`, and declares `record_user_answer` less `plan_session` and `answer` with `kind="external"`. An external tool is a deferred tool: the run returns `DeferredToolRequests` and never calls it. I chose the prepared toolset over a filtered one because the schemas need editing as well as the tool list, and over `ExternalToolset` because the ask's declaration comes from plan-mcp's own listing, so it cannot drift.
- `open_plan` and `record_answers` call `start_plan` and `record_user_answer` through the same client, outside the model's view. `record_answers` sends the user's answer with the ask's arguments from the deferred call.

`rig.py` runs plan-mcp (`harness-ab`'s `HarnessPlanMcp`, which records an answer `by: harness` and times its own work) in process and over loopback Streamable HTTP, with `harness-ab`'s arm B beside it. Models are scripted with `FunctionModel`; no live calls.

## Pass lines

1. Declarations. The model is offered `propose_plan` (32,177 characters as compact JSON) and `record_user_answer` (653), the same dictionaries `harness-ab`'s seam declares, field by field (name, description, parameters schema), plus the same instruction. `start_plan`, `run_plan` and `reject_plan` are absent, and the ask has no `plan_session` or `answer`. B's skill tools are ADK tools that this prototype leaves out of the comparison. Control: the same agent over an unprepared `MCPToolset` is offered all four tools, `propose_plan` at 32,270 characters and `record_user_answer` at 995; the comparison reports both. In process and over HTTP.

1. Answers. `combination_unasked` (979 characters) and `invalid_plan` (482) are a successful `ToolReturnPart` whose content equals B's answer as a value and as compact JSON. Control: with every answer sent `isError`, the part is `outcome="retried"` and the model reads `{"error":"<the answer as a JSON string>"}`, 1,045 and 540 characters; the comparison fails. In process and over HTTP.

1. Error path (`test_errors.py`).

   - plan-mcp never sets `isError`. FastMCP does when a tool body raises or the tool name is unknown. 236 malformed `propose_plan` calls (every top-level field and every field of each evidence requirement set to `None`, `{}`, `5`, `"x"` and similar, or removed) gave 0 `isError` in process and over HTTP. Every one was an answer (`invalid_plan`, a refusal).
   - A tool body that raises: `record_user_answer` with an `answer` that is not an object, and `run_plan` with a non-numeric `version` or `wait_s`. Of these only the first is reachable here, from the harness; the model cannot send `answer` (hidden) or call `run_plan` (not in its view). A fault injected into `propose` (a store or service failure) is the model-facing case.
   - Unknown tool: the client rejects a name outside the model's view before any call. The model reads `Unknown tool name: 'run_plan'. Available tools: ...` as a retried return (`retry.cause == "unknown_tool"`), and the server sees nothing.
   - What the model reads: `{"error":"Error calling tool 'propose_plan': service fault"}`, in a `ToolReturnPart` with `outcome="retried"` (`RetryInfo(cause="model_retry", attempt=1, max_retries=1)`).
   - Retries: a retried return uses the tool's retry budget (the agent's, 1 by default). The second consecutive fault ends the run with `UnexpectedModelBehavior: exceeded max retries`. With `tool_error_behavior="failed"` the return is `outcome="failed"`, uses no budget, and three faults in a row do not end the run.
   - A call past `read_timeout` is a retried return (`Request 'tools/call' timed out`). Over HTTP the same setting is also the transport's read timeout, and closing the held connection afterwards raises `httpx2.ReadTimeout`.
   - An unreachable server raises `RuntimeError` out of the run before any model call.
   - A harness-side call (`open_plan`, `record_answers`) that hits `isError` raises `ModelRetry` into harness code, where no model reads it.
   - Real case end to end: the injected `propose` fault, through the real transport, to the model and the retry count.

1. Pause and resume. On `sum-oda-oof-ethiopia` the scripted model proposes, is refused `combination_unasked`, and calls the ask; the run returns `DeferredToolRequests`. The history (`ModelMessagesTypeAdapter`) and the requests go through JSON, a new agent over a new `MCPToolset` connection records the answer, and `DeferredToolResults` resumes the run. plan-mcp holds one answer, `by: harness`, equal to the user's. The model's first request after the resume carries that answer as the ask's successful return. The second proposal lands the plan whose `plan_revision` equals the one B lands (923550a4...). Control: the same resume without `record_answers` is refused `combination_unasked` again and lands no plan. In process and over HTTP.

1. Overhead of `propose_plan` (client time less server time), 33 calls each after 3 warm-up, medians and p90, the two clients alternating on one machine against the same server:

   | Client                             | In process        | Over HTTP         |
   | ---------------------------------- | ----------------- | ----------------- |
   | Pydantic AI `MCPToolset`           | 0.89 ms, p90 1.07 | 4.58 ms, p90 4.74 |
   | B's `PlanClient` over `McpSurface` | 0.76 ms, p90 0.82 | 4.22 ms, p90 4.33 |
   | `harness-ab/NOTES.md`, live runs   | 2.1 ms            | 11.0 ms           |

   Control: 20 ms of delay in the server's transport layer, outside `server_times`, raises both clients' median overhead by 22 ms in process (0.74 to 22.72) and 22.7 ms over HTTP (4.55 to 27.27). The Pydantic AI client costs about 0.1 to 0.4 ms more per call than B's.

## Also

Harness-side lines. `pai_plan_mcp.py` has 79 code lines (132 with docstrings and blanks) and reuses 16 more from `harness-ab` (`without`, `user_answer`, `mcp_instruction`). `mcp_planner.py` has 270 code lines (378 in all), of which 41 are the server-side `HarnessPlanMcp`, which both arms need. The rest of B's count is what the Pydantic AI side does not need: tool wrapper classes (39), a client with call timing (29) and the seam's session bookkeeping and ADK event scan for the asks (about 100). The Pydantic AI adapter does not include the seam's budget, prefix-cache and callback wiring or the skill tools; `prototype/gemini/pai_planner.py` carries those for the planner already.

What Pydantic AI's client does that the ADK path did not:

- Retries. Tool errors become retried or failed returns that the model reads (above). `McpSurface` returns `is_error` for the caller to read and raises `ToolSurfaceUnavailableError` for a transport fault.
- Tool list. `cache_tools` is on: in process one `tools/list` serves two runs while the connection is held open, and each run lists again when it is not held. Over HTTP the FastMCP client lists again before each call (3 lists for 2 held runs, 4 for 2 unheld), in B too.
- Capabilities. It declares none: no elicitation, sampling or roots. Both transports negotiate protocol 2026-07-28 (stateless), where a handler it did not declare is never called.
- `_meta`. The FastMCP client sends `traceparent`, the protocol version, client info and a progress token on every call, in both arms. The Langfuse session key is sent only when `process_tool_call` passes it (`meta=`, here `org.one.dfo/langfuse-session`); the server saw it in process and over HTTP. `McpSurface` sends it only to loopback URLs, so a production `meta` would carry that guard.
- Caller header. A static `headers=` dict sends `x-dfo-caller` over HTTP; the server read `caller-1`. A `headers` function of the run's deps gives each run its own connection and refuses use outside a run (`UserError`), so `open_plan` would need `for_run`. `CallerClients` in the harness holds one client per caller instead.
- Arguments. The client validates nothing against the tool's schema (`TOOL_SCHEMA_VALIDATOR` accepts anything), as `McpSurface` does not; `period.start: "twenty twelve"` reaches the server and comes back `invalid_plan`.

## Internals relied on

Public Pydantic AI 2.64: `MCPToolset` (`process_tool_call`, `direct_call_tool`, `tool_error_behavior`, `max_retries`, `read_timeout`, `headers`), `.prepared`, `ToolDefinition.kind="external"`, `DeferredToolRequests`, `DeferredToolResults`, `ModelMessagesTypeAdapter`, `ToolReturnPart.outcome` and `.retry`, `FunctionModel`.

Not public or not Pydantic AI's: `MCPToolset.__aenter__` and `__aexit__` called directly to hold a connection across runs (`test_client_behaviour.py`); `harness-ab`'s `HarnessPlanMcp`, `Current`, `run_b`, `PlanClient`, `McpPlannerSeam` and `without`, `user_answer`, `mcp_instruction`; the seam's `_agent.tools` and ADK's `_get_declaration` for B's declarations; `server._result`, patched for the `isError` control; `fastmcp_request_ctx` to read the wire `_meta`; `HarnessPlanMcp.propose`, patched to inject a fault or a delay.

## What a production version would change

About 120 code lines in the adapter: the loopback guard on `meta` (2 lines, from `compiler.caller_clients.session_call_meta`), a held connection per turn (`async with` around the run and the resume, 5), persisting the history, the requests and the plan session between the pause and the resume (15), `tool_error_behavior="failed"` with a `UsageLimits` bound (3), and the skill tools. I would set `tool_error_behavior="failed"`: plan-mcp's `isError` results are server faults the model cannot correct, and a retry spends a turn on one. The objection is that a `failed` return lets the model repeat the call without limit, so the run needs a `UsageLimits` cap, which a retry budget gives for free.

The harness's own pause handling changes too, as in `harness-ab/NOTES.md`: the Studio route recognises the ask by the name `adk_request_input`, and under plan-mcp it is `record_user_answer`.

## Not covered

- A live model. The request Pydantic AI builds from the 32 KB `propose_plan` schema (Gemini's schema conversion, `$ref` handling) is not exercised; pass line 1 compares what the agent offers, not what a provider receives.
- The planner's turn budget, prefix cache, callbacks and skill tools; `revise`; `run_plan` and the approval card.
- A held connection that drops mid-run; only a refused connection and a timeout were tried.
- Concurrent runs on one `MCPToolset`.

## Rerun

```
cd /Users/jorge/repos/dfo-ai-worktrees/pai-research
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness \
  pytest research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/harness-pai -q -p no:cacheprovider
```

Add `-s` to print the declaration sizes, the answers, the error text the model reads and the overhead figures. Over `test_overhead.py` alone the run takes about 12 s.
