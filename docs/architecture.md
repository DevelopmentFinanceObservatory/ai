# Architecture

## What this repo is

dfo-ai is the AI layer of the Development Finance Observatory
(DFO). It's a uv workspace monorepo whose packages deploy together as
**one** Cloud Run service at launch.

**`services/harness`** runs the agent. It's built on Google ADK and
FastAPI, and powers the site's chat-and-canvas workspace (DFO Studio) over
an SSE event stream: the frozen wire contract in
`docs/studio-contract-v1.md`. A data turn runs one agent that calls its
own tools (`services/harness/src/orchestrator/README.md`) inside an ADK
workflow that owns routing, approval and execution; card edits, canvas
mutations, and cancellation are direct studio routes (see
`services/harness/README.md`). It mounts data-mcp, analysis-mcp, and
presentation-mcp as FastMCP ASGI sub-apps inside its own process,
reachable over streamable HTTP at in-container loopback paths
(`/mcp/data`, `/mcp/analysis`, `/mcp/presentation`).

**`services/harness/src/planner`** is the Planner Agent: an ADK agent that
drafts a methodological plan for a question, recording each decision it makes,
the alternatives it considered, and the evidence an acceptable answer rests on.
It holds the plan at the user's approval, ships inside `harness` as a sibling
top-level module, and knows nothing about execution. The DFO Studio runs the
modules described below, adapted for its workflow by
`harness.planner_seam.PlannerSeam`.

The plan-approve-execute workflow spans four top-level modules beside it,
`planner`, `translator`, `compiler` and `executor`. The planner settles
methodology with the user and records each choice as a `Decision` and what
evidence the answer rests on as `EvidenceRequirement`s, naming no tool. The
translator, code rather than an agent, finds the data each requirement needs
through a bounded, read-only discovery of data-mcp's `search`, and builds the
typed operation each requirement states.
The compiler, code rather than an agent, maps those operations to tools through
a hand-maintained capability table pinned to the live MCP schemas, writes every
tool argument, and checks the result against the plan before anyone approves
it. The planner states every calculation the answer needs as a typed
requirement (an aggregate, a growth, a comparison, and so on), alongside the
dataset requirements that feed them, and the planner's tools answer with a
typed result carrying a stable code, rather than an unstructured error string,
so a caller can branch on what stopped a call without parsing its message.
`harness.agents` attaches the translator to the planner's `propose_plan`: an
accepted proposal runs translation on the plan it recorded, the plan passes
through session state, and the proposal's answer carries a `translation`
verdict, `{"outcome": "ok"}` for a compiled plan or the referral translation
stopped on.
Nothing runs until the user approves. A `propose_plan` whose translation
compiles declares the plan ready for the user's answer in code
(`planner.agent.declare_ready`); it approves nothing and executes nothing.
The gate is the workflow's approval interrupt, raised by its `approve` node and answered through `POST /api/studio/interrupt`, keyed
on the plan's revision rather than its version integer, so what runs is
exactly what was approved. The executor, also code, takes the
approved plan from there: `execute_approved_plan` refuses a plan whose content
no longer hashes to its approved revision or whose tools' schemas have changed,
runs the workflow step by step with each step's frame references bound from the
steps before it, re-runs the evidence checks on the manifests `get_data`
returns, and ends the plan `COMPLETED` once an `Accepted` run completes, with
each derived result published with its replay recipe, or `FAILED` on a
`Refused` outcome, a `MethodChoiceNeeded` that hands a choice back to the
planner, or a run whose publication failed. An `Accepted` run whose
publication is still unfinished leaves the plan `EXECUTING` until a later turn
completes it.
See `services/harness/src/planner/README.md`,
`services/harness/src/translator/README.md`,
`services/harness/src/compiler/README.md` and
`services/harness/src/executor/README.md`.

**`services/data-mcp`** is a FastMCP server over the Data seam of the
`apis` repo: the Search API (served at `/api/search/*`) today, the
Datasets API (`/api/datasets/*`) next, both reading from the `dcp`
graph. Only agent-facing tool definitions belong here; the search and
dataset logic, ranking, and data access stay behind those HTTP APIs. This
wrapper exists so the harness gets a stable MCP-shaped interface without
duplicating that logic.

**`services/analysis-mcp`** is the Analysis Toolkit. It runs deterministic
computation over data the harness has already retrieved. Rather than asking
the model to compute a number directly, the model emits an op-spec against a
closed verb set (`aggregate`, `compare`, `growth`, `rank`, `share_of`,
`quotient_of`, `sum_of`, `join`, `filter`). Deterministic code executes that spec and returns a
frame reference, a summary, and a trace (instead of raw bulk data back into the model's
context). The public tool list serves `sum_of` as `compare`'s sum form and `quotient_of` as
`share_of`'s quotient form.

**`packages/common`** holds the small pieces of shared internals every
service needs: a settings base class, a logging setup, a telemetry
bootstrap, and the bearer-token verifier the three MCP mounts share
(`common.auth`). It's the analogue of a `_shared` package in a typical
monorepo. Nothing in it is deployed on its own.

Spec models, spec-id hashing, and the manifest and component-spec schemas
live in the `dfo-spec-kit` package, not this repo. Materialisation lives in
`packages/common/src/common/materialisation`, shared by data-mcp and
analysis-mcp.

## Architecture decisions

**Deployment topology.** Platform-wide at launch there are three Cloud Run
services: DFO DCP (the knowledge graph engine), the DFO website, and DFO `ai`
(this repo). This repo's workspace services (initially harness, data-mcp,
analysis-mcp, presentation-mcp) deploy together as that single `ai` Cloud
Run service.

Reasons for consolidating:

- **Min-instance cost.** One warm instance covers every workload instead of
  a separately-billed warm instance for each.
- **Cold starts.** Agent tool calls (harness → data-mcp / analysis-mcp /
  presentation-mcp) happen inside a single request. A cold-starting MCP
  service on the critical path would show up as latency in every chat turn
  that uses a tool.
- **Loopback instead of network hops.** Mounted in-process, the MCP servers
  are reachable over loopback HTTP rather than a real network call, cutting
  a hop out of every tool call.

Each service stays its own uv workspace package with its own dependencies and
tests, and the harness reaches them through `DATA_MCP_URL` /
`ANALYSIS_MCP_URL` / `PRESENTATION_MCP_URL` env vars that default to the
mounted loopback paths but can point anywhere.

GitHub Actions builds one image (`ai`) and updates the Cloud Run dev
service on every merge to `dev` that touches its inputs. Production deploys on
every push to `main`. The `infra` repo owns the Cloud Run service
_definition_ (service account, scaling, secrets). This repo's CI only swaps
the image on the service `infra` already provisioned.

**Harness-to-MCP traffic.** The harness talks to `data-mcp`,
`analysis-mcp`, and `presentation-mcp` over streamable HTTP, configured via
environment variables (`DATA_MCP_URL`, `ANALYSIS_MCP_URL`,
`PRESENTATION_MCP_URL`). At launch those default to the in-container
loopback paths of the mounted apps.
This traffic never transits the public website's proxy, loopback or
not.
The per-server mounts refuse any client off loopback with 403, apart from
their `/healthz` routes and presentation-mcp's theme stylesheet, so outside
MCP clients reach data-mcp, analysis-mcp and presentation-mcp only through the
combined server at `/mcp/`.

**MCP mount auth.** Each mounted FastMCP server requires a bearer token
(`FastMCP(auth=...)`, `common.auth`). fastmcp's auth wrapper covers only the
streamable-HTTP MCP route, so `custom_route` health probes like
`/mcp/data/healthz` are unaffected. Setting `MCP_API_TOKEN` makes that
token the one required. Left unset (the default), the harness mints one at
startup and hands it to its own `data-mcp` / `analysis-mcp` /
`presentation-mcp` clients, so self-calls work and every other caller is
refused. That default holds only as long as every mount shares the
harness's process, since nothing outside it can ever know a token minted
inside it. Pointing `DATA_MCP_URL` / `ANALYSIS_MCP_URL` /
`PRESENTATION_MCP_URL` off loopback with `MCP_API_TOKEN` still unset
refuses at startup rather than serving a split deployment that 401s on
every call. With `MCP_API_TOKEN` set, pointing those URLs off loopback
over anything but `https` refuses too: the harness attaches that shared
bearer to every request, and plain HTTP would put it on the wire in the
open. `MCP_AUTH_DISABLED=true` turns the gate off, for a raw client
with no bearer token to present. It logs at error level and is never
appropriate on a deployment reachable beyond one machine. The gate is a
capability check: whoever holds the shared token gets in, and reads carry
no per-user ownership restriction (`common.materialisation.store`'s own
docstring).

**Structured tool results.** An MCP tool's response splits `content` (the
digest the tool renders for a model) from the envelope for downstream code,
which `search` carries in `_meta` and the other tools in `structuredContent`
(see "What a tool result carries"). The harness reads every envelope in code
and drops it at that point, so a model's boundary sits at a function's
return value.
`harness.mcp_client.McpResolver` and `McpExecutor` read a result's
`structuredContent` to decide an outcome and return the workflow's own
`frame_refs` and one-line `summary`, or a typed `refusal` merging the
refusal's `code`, `retry_mode` and `fix` into the workflow's grammar. A
successful `search` puts its upstream envelope in the result's `_meta`
(`data_mcp.search_result.SEARCH_ENVELOPE_META_KEY`) and a small answer in
`structuredContent`, so a host that hands `structuredContent` to a model
shows it a few hundred bytes; the harness reads the envelope from `_meta`.
`translator.discovery` reads a `search` envelope
through `compiler.sources.outcome_from_envelope` and answers with
`compact_source`'s own field set, and keeps one `definition` string per
source out of a `describe` answer. `executor.run` keeps each server's digest
text as its step's `StepRecord.digest`, which the workflow's `plan_done`
node hands to the agent as `step_summaries`. The planner's model reaches
these Python functions through `FunctionTool`s
(`harness.agents.create_planner_with_translator`). ADK's `McpTool` plays no
part, so no `CallToolResult` dump is built for them and an
`after_tool_callback` editing one would have nothing to edit. Where a tool
does answer with a digest, a refusal's machine-readable detail rides the
same call. data-mcp and analysis-mcp put the actionable recovery grammar in
`structuredContent`, and the harness's own mounted consumers read it there.

**Telemetry.** The harness records agent usage through its own Langfuse and
OpenTelemetry traces. The website proxy's analytics cover other usage, so
proxy-level metrics do not dilute the agent traces.

A deployed service with the Langfuse variables set sends Langfuse Cloud (EU
region) the text of each Studio question and the assistant's replies; the
orchestrator and planner prompts, which include the canvas and conversation
digests (ADK's `call_llm` spans carry them); tool arguments and results at digest
level; and refusal codes with the refusal fields recorded on the trace events
below. Everything above is sent unmasked. `LANGFUSE_SAMPLE_RATE` and a
Langfuse `mask` function limit it.

One studio turn is one trace. `ConversationState.begin_turn` opens a `turn`
span whose trace id is derived from the turn's wire `trace_id`, and
`close_turn` ends it, so it stays open across a pause for approval, including
when the client disconnects at the pause (a disconnect while the turn runs ends
it as cancelled). The `/turn` stream and the `/interrupt` stream that resumes
the turn are separate requests, and each runs inside that span
(`ConversationState.in_turn_span`), so the
resumed segment's ADK, MCP and `record_event` spans nest under the same `turn`
span. `close_turn` gives the `turn` span the user's message as its input and
the closing text as its output, which Langfuse shows as the trace's input and
output. A turn closed with an error code (`cancelled`, `abandoned`, and the
like) ends at level `ERROR` with that code as its status message.

Every studio stream calls `common.telemetry.flush_telemetry` before its
response ends, so queued spans export while the request still has CPU.
`POST /api/studio/cancel` calls it when it ends a paused turn's span, since no
stream runs then.

**Evals.** Evaluation runs on ADK's own eval framework. `evals/` holds one
folder per eval set (the agent wiring, the cases, and the criteria they're
scored against) and isn't run in CI. Each service's `tests/` directory holds
the actual CI gates. This split exists because an eval set drives the real
agent through the real model, so it needs a live model credential, and the
rubric-scored cases need a judge model on top of that.

## What a tool result carries

A tool result has three places for information, and each has its own reader.

`content` is the text a model reads. `structuredContent` is a small structured answer a model can use, because some hosts, ChatGPT among them, put it in model context. data-mcp's `search` returns `data_mcp.digest.render_answer` there: the envelope's status and, per part, a label, a status, the `spec_id`s a model can pass to `get_data`, and warnings, beside the ids of the skills the digest lists. presentation-mcp's app tools return the view's layout and words there, with no data value.

The result's `_meta` carries machinery for code and views, and hosts keep it from the model.

- `org.one.dfo/search_envelope` (`data_mcp.search_result.SEARCH_ENVELOPE_META_KEY`) holds the full upstream resolve envelope from `search`. The harness's `McpResolver` and translator discovery read it from the data-mcp mount. The combined server at `/mcp/` drops it, with `structuredContent`, before a result reaches an outside client. It passes an MCP App tool's result whole, found by the `_meta.ui.resourceUri` the tool declares in `tools/list`, so `write_component` and `describe_component` keep their view in `structuredContent` and `_meta`.
- `org.one.dfo/resolved_payload` (`presentation_mcp.platform_contract.RESOLVED_PAYLOAD_META_KEY`) holds the platform's resolved revision from `write_component` and `describe_component`.
- `org.one.dfo/view_state` (`presentation_mcp.platform_contract.VIEW_STATE_META_KEY`) holds the values the view binds. The renderer page's script merges it into the view's state.

A view carries what it needs in its own result and makes no follow-up call, so it draws in hosts that open a new MCP session per request and on any revisit. Hosts keep the `ui://` address they first saw, so the server answers any `ui://presentation-mcp/renderer-{version}.html` with the current page.

## Conventions

Python `>=3.13`, pinned in `.python-version`. The workspace uses uv's
`uv_build` backend throughout. Lint and format are ruff, configured once at
the workspace root (`target-version = "py313"`, `line-length = 100`, rules
`E`, `W`, `F`, `I`, `N`, `UP`, `B`, `C4`, `SIM`, `PL`, `RUF`, `ERA`, `PERF`,
`ANN`). Types are checked
with `ty` (Astral). Tests are pytest with pytest-asyncio in `auto` mode.

Every service follows a src layout, `services/<name>/src/<pkg>/`, with tests
under `services/<name>/tests/`. Package names are `harness`, `data_mcp`,
`analysis_mcp`, `presentation_mcp`, and `common`. None carry a `dfo_` prefix, since this repo is
expected to move to a DFO GitHub org and be renamed `ai` later, and internal
names should stay org-agnostic until then.

Config uses `pydantic-settings`, subclassing `common.settings.BaseServiceSettings` and
driven entirely by environment variables. Each service has one `Settings`
class for the service itself; harness carries a second,
`planner.settings.AgentSettings`, so the planner pins its own model without
a caller reading a model string having to import the service's MCP URLs and
session backend. Recognised names include `DFO_APIS_URL`, `DFO_SITE_ORIGIN`
(required on Cloud Run), `DATA_MCP_URL`, `ANALYSIS_MCP_URL`,
`PRESENTATION_MCP_URL`, `PRESENTATION_PUBLIC_URL` (required), `MODEL_NAME`,
`PLANNER_MODEL_NAME`, `DATABASE_URL` and the
`SESSION_STORE_*` family (optional, for a durable harness session store;
unset falls back to in-memory sessions, see `services/harness/README.md`),
the `LANGFUSE_*` family, and `PORT`. Each
service exposes a console script (`harness-server`, `data-mcp`,
`analysis-mcp`, `presentation-mcp`) and a liveness check equivalent to
`/healthz`.

Style favours functions over classes where reasonable, keyword-only
arguments in public APIs, and no speculative abstractions. Build what the
current service needs, not what a hypothetical future one might.

## Failure vocabulary

Six failure concepts run through the planner/translator/compiler/executor
stack. Three answer to their own `ToolError`: `harness.wire.ToolError` on
the SSE contract, `planner.tool_results.ToolProblem` for the planner's and
translator's own tools, and the MCP refusal payload `executor.classify`
sorts, which is `executor.classify.ToolFailure`, a typed model. Each of the
other three carries a wider enum than any single failure name could:
`compiler.compile.CompileIssueCode` has 38 members, `planner.records.EvidenceIssueCode`
has 27, and `planner.tool_results.ProblemCode` has 26.

| Concept              | What it means                                                                                                                 | Live vocabulary                                                                                                                                                                                                                                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TranslationRefusal` | the plan cannot be expressed with the supported operations or the data on hand                                                | `planner.tool_results.ProblemCode`, carried on a `planner.tool_results.ToolProblem` the translator's own tools answer with                                                                                                                                                                                             |
| `CompilationError`   | the proposal has a broken reference, a cycle, an unsupported operation, an incompatible binding, or a stale tool contract     | `compiler.compile.CompileIssueCode`, carried on a `compiler.compile.CompileIssue`                                                                                                                                                                                                                                      |
| `ApprovalMismatch`   | execution did not load exactly the approved workflow revision                                                                 | `executor.outcomes.RefusalKind.NOT_APPROVED`, `.STALE_REVISION` and `.SURFACE_CHANGED`, carried on `executor.outcomes.Refused`                                                                                                                                                                                         |
| `ToolError`          | a concrete MCP call failed or returned `isError`                                                                              | `executor.classify.ToolFailure`, parsed by `executor.classify.parse_tool_failure` from `data_mcp.tool_errors.ToolErrorCode` and `analysis_mcp.tool_errors.ToolErrorCode`, and sorted by `executor.classify.classify_refusal`; the SSE-facing shape a client reads is the separate `harness.wire.ErrorCode`/`ToolError` |
| `EvidenceNotMet`     | the tool call succeeded but its result failed an evidence requirement                                                         | `planner.records.EvidenceIssueCode`, checked by `compiler.evidence.check_evidence` and carried on `executor.outcomes.Refused` as `RefusalKind.REQUIREMENT_UNMET` with its `issues`                                                                                                                                     |
| `PlanRevisionNeeded` | resolving the failure would change the approved method, the selected source, or the tool arguments outside an approved policy | `executor.outcomes.MethodChoiceNeeded` and `executor.outcomes.RediscoveryNeeded`, the two `RunOutcome` variants that hand a refusal back for a new plan revision instead of stopping the run                                                                                                                           |
