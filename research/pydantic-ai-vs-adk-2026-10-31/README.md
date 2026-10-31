# Pydantic AI or Google ADK: which architecture fits the harness

Judged on design alone, against google-adk 2.13.0 and pydantic-ai 2.64.0 as of 2026-10-31. The cost of moving is in `effort-and-cost.md`.

## Recommendation

On architecture, Pydantic AI agents under a turn loop we own fit the harness better than ADK's `Workflow`. Add a durable engine (DBOS, Temporal or a Postgres task queue) only for work that leaves the user's stream: long mints, turns that outlive the client, replay of shared cards. Keep the interactive turn out of the engine.

The choice is really two decisions stacked. The first is who owns the turn: ADK's replay engine or an explicit state machine of ours. Most of the gain comes from owning the turn. The second is which agent loop runs under it: Pydantic AI or ADK's `LlmAgent`. Pydantic AI wins there on context control, typing, streaming, tool search and model breadth. It leaves the turn to us, so the two decisions go together.

The turn needs no graph library. `harness/graph/workflow.py` wires nine nodes with four edges that loop back (`ask` and `plan_done` to `orchestrate`, `approve` and `execute` to `planner`), two user pauses (`ask` and `approve`) and no fan-out between nodes; the only parallel work runs inside `execute`. An enum of states, one async function per state and a loop that dispatches on the stored state express that directly, and resume becomes a ledger read followed by one call. A graph library would pay for itself only if separate agents ran in parallel and each could pause on its own, such as one approval per variant in a comparative plan. Nothing committed asks for that, and `asyncio.gather` inside one state covers the common case. So `pydantic_graph` dropping persistence in V2, because snapshots of parallel steps cannot be made consistent, costs us nothing.

## What the harness has to do

The product rests on three demands, drawn from 16 built and planned capabilities (`use-cases.md`):

1. Durable human pauses. Approval, amendment, clarification and cancel all pause a turn that resumes on another instance, sometimes days and several deploys later, with the approved plan's bytes intact.
1. Context control. Models see digests while rows stay out of band. Each agent gets a curated view, long conversations need compaction and recall, and a catalogue of skills and capabilities keeps growing.
1. Live streaming. The wire contract is frozen, progress must stream from inside long steps such as planning (median 59 s, worst 659 s), and the same components serve the site and MCP Apps hosts.

Next come parallel work inside a turn, background data work that outlives a request, outside agents calling the workflow through MCP, and possibly a second model provider.

## How each framework works

ADK is an append-only session event log with a graph scheduler on top. Every durable fact is an event row; a model's prompt is assembled from that shared log by filter rules; a pause is a function call in the log; and resume reruns the workflow from START, fast-forwarding nodes that have a recorded outcome. One mechanism serves history, state, resume and tracing (`architecture-adk.md`).

Pydantic AI is a stateless agent loop with typed hooks. A run takes a message history and typed dependencies and returns messages. Nothing persists unless we store it. A pause for a human ends the run with the pending tool calls as its output; a later run on any instance picks up from the stored `Conversation`. Every model request passes one hook where the developer can rewrite messages, tools, instructions and model (`architecture-pydantic-ai.md`).

## The three properties that decide it

Where a pause lives. ADK's replay is exact only when the code between pause and resume is unchanged and deterministic. Our pauses span deploys, so a node added in a release runs live in the middle of a resumed turn; that is why `seed` carries `_turn_ran_before`, and every new node needs the same guard. A stored cursor holding the plan revision, the interrupt id and each agent's history is a lookup, correct by construction. Durable engines replay too (DBOS raises `DBOSUnexpectedStepError` when step order changes across a deploy), which is why they suit minutes-long background work and not approval pauses that last days.

Who owns model context. In ADK an event with no branch reaches every agent (`_contents.py:776`), so isolation is opt-in, stamped per event and checked by tests. A plan of over 100,000 characters once reached prompts this way. In Pydantic AI each agent owns its message list, and one hook assembles each request. Since the product depends on curated views, the default matters more with every agent, skill and capability added.

Whether a pause is a value or an exception. In ADK a child agent's pause raises `NodeInterruptedError`, a `BaseException`, up through the parent node, and `run_node` must be awaited directly and never wrapped in a task (`agents/context.py:453-456`). That rules out a producer task under a streaming consumer, which is what live progress from planning and parallel pausable work both need. The planner-progress stream is a committed plan that is blocked on this today. In Pydantic AI a pause is the run's return value, so the planner can run in a task while the loop streams.

## Capability by capability

| Capability                          | ADK `Workflow`                                                              | Pydantic AI under our loop                                              |
| ----------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Durable human pauses                | Native, but replay hazards across deploys                                   | A cursor and stored records; deploy-safe                                |
| Per-agent context                   | Shared log, shown by default                                                | Separate histories, one request hook                                    |
| Live streaming to the UI            | About 900 lines translate node paths and branches into frames               | The loop emits our frames; `ctx.emit` from tools                        |
| Progress from inside planning       | Blocked: a pause is an exception that cannot cross a task                   | A pause is a value; planner runs in a task                              |
| Steering a running turn             | No text-mode API; cancel is in-process                                      | `enqueue` injects at the next request; cross-instance signal is ours    |
| Parallel agents                     | Sub-branches; state deltas merge by dict key                                | `asyncio.gather` over separate runs, with concurrency limits            |
| Direct execution beside the planner | Shared services travel as state keys and closures                           | One typed `deps` object shared by both agents                           |
| Long conversations                  | Compaction appends summaries; rows never pruned; resume reads the whole log | History processors replace history; recall reads our records            |
| Growing tool catalogue              | No tool search; a changing tool list drops the cache                        | `ToolSearch` with deferred loading                                      |
| Background data work                | No worker; work dies with the stream                                        | Not provided; an engine or queue does it (the recommended add-on)       |
| Outside agents through MCP          | Agent-as-MCP and A2A experimental                                           | A turn is a function a FastMCP tool can call; A2A is a separate package |
| Second model provider               | Gemini-first; LiteLLM and Anthropic adapters under repair                   | About two dozen providers, per-step model choice                        |
| Gemini-specific depth               | Wide and early                                                              | Wide; explicit caching stays ours on either                             |

Neither framework gives cross-instance coordination. We already run our own lease, checkpoint and cancel on a session table, and that stays.

## Where ADK is stronger

ADK's single log is a smaller idea than ours would be: one append gives history, state, resume and traces, and pause and resume across processes work today with no code of ours. Google's investment is heavy and aimed at `Workflow`, with a managed runtime behind it, and Gemini features arrive there early. If Google ships queued steering (RFC #7642), live node streaming (RFC #7455) and a branch default that hides rather than shows, the case above narrows. None of those is committed: the steering RFC is from an outside contributor, and no source shows a change to the branch default or to replay.

## The strongest argument against the recommendation

Owning the turn loop means owning a runtime for good, and Pydantic AI changes fast underneath it: about three releases a week, its harness package still 0.x, and stored message history with no schema version. The runtime half of this objection is weak, because we already own the lease, checkpoint, cancel and frame translation on ADK; replay is what ADK adds on top, and replay causes the defects. The churn half holds. The answer is to keep our own records (plans, approvals, digests, turn records) canonical, treat each agent's `Conversation` as derived state we can rebuild, pin the version, and replay stored histories on every upgrade. So far the format has held: records from ten releases load and resume under every later one. If that discipline fails, the same loop runs on plain `google-genai` and keeps every property above except provider breadth.

## What the prototype showed

A planner prototype tested each claim above (`prototype-results.md`). Every pass line held, with one condition: the planner must keep Gemini's function calling in `AUTO` mode, as ADK does, and not in `VALIDATED`, Pydantic AI's default.

- Live progress and pauses. Under our own loop, a progress event from inside `propose_plan` reached the client within 2 s while the tool was still running, and a pause mid-planning lost no event.
- Surviving a kill. A turn paused for approval, its process killed with SIGKILL, finished in a fresh process with orchestrator and planner histories identical to an unpaused run once timestamps and run ids were normalised.
- Deploy-safe resume. A step added to the start of a turn did not run when an older paused turn resumed, and two concurrent resumes of one pause let exactly one through.
- Stored histories. A paused `Conversation` written by each release from 2.55.0 to 2.64.0 loads and resumes under every later release, and under every earlier one. Releases only added fields. That holds for the record shapes tested; tool retries (2.60), compaction parts and tool-availability deltas (2.62) are stored in shapes older releases cannot read, so rollback past those releases is one-way for histories that hold them.
- The planner on Gemini. Over 558 paired entries on the 62 `planner_entry` tasks, the Pydantic AI planner in `AUTO` mode accepted 127 of 177 first proposals against ADK's 129, kept the cached share at 85.3% against 84.7%, ran first chunks at 1.04 times ADK's p50, and lost no thought signature. The `propose_plan` declaration reached Gemini byte for byte as ADK sends it. `VALIDATED` mode slowed the first chunk by 20% and each entry by 3 s, and one refusal task stopped ending as it should.

Two reviews of Pydantic AI's docs and of production reports, together with dev's upgrade to ADK 2.13, shrink four costs and add four risks (`patterns-and-risks.md`). Pydantic AI already answers an unknown tool with the tool list and re-prompts an empty response. A cache prefix built from the canonical declarations should let both frameworks share one Gemini cache. Public hooks replace the prototype's patch of the google-genai client. The new risks each need a rule in the loop: an effect ledger, because a pause resumes at least once; no `enqueue` where a turn can pause; an envelope with a schema version around each stored history; and a fresh usage counter per turn. ADK 2.13 showed the same kind of wire drift on its side: a line it appends to the instruction took the orchestrator's first call from 0 to as many as 703 reasoning tokens.

A second prototype put a Pydantic AI planner inside ADK's own `planner` workflow node (`prototype-node-stream.md`). It streamed progress live: the first frame reached Studio's SSE client in 0.09 to 0.20 s while `propose_plan` was still running, against about 3.1 s, after planning, for dev's node. It also lost no frame when the planner asked the user, resumed from the stored history after the answer, survived SIGKILL, stopped cleanly and kept the turn's stored records equal to dev's. So the planner can move first, inside the ADK turn, and unblock live planning progress before the turn loop changes. ADK accepted the same answer twice after a kill, so a production node needs its own duplicate guard.

Still open: the planner's tools on Pydantic AI (the readers are framework-neutral in PR #129), the translator lines that read the ADK planner's own events, a second prototype round against ADK 2.13 (`patterns-and-risks.md`), and whether turns must outlive the client. On that last point, if more than 5% of turns on deployed dev run past 60 s or end in a disconnect, the detached half is needed now.

## Decisions for you

1. Whether to adopt the architecture: an owned turn loop with Pydantic AI agents in `AUTO` function-calling mode. The prototype passed every line; the next steps are the planner's readers on our own record and a second prototype round against ADK 2.13.
1. Engine scope: detached work only (recommended), the whole turn (crash recovery mid-planning, at the cost of live streaming), or none.
1. Whether a planned turn should keep running after the client disconnects, with the stream as a subscriber. Recommended, given a 659 s worst turn against Cloudflare's 100 s cut and ChatGPT's 60 s cap; Stop then becomes an explicit action.

## Sources in this folder

- `use-cases.md`: current shape, 16 capabilities and their demands.
- `architecture-adk.md`, `architecture-pydantic-ai.md`: the same eleven questions answered from source.
- `consult-architecture.md`: independent fit assessment across four architectures.
- `footprint-runtime.md`, `footprint-model-tools.md`: how the harness uses ADK today.
- `currency-adk.md`, `currency-pydantic-ai.md`: releases, open issues and project health.
- `effort-and-cost.md`, `consult-verdict.md`, `adk-cost-history.md`: what a move would cost, and what staying has cost.
- `prototype-results.md` and `prototype/`: the planner prototype, its code and its results. The code ran against dev at afbf84ff (ADK 2.10) with `uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness`; the paired Gemini run's records and wire tap are gzipped in `prototype/gemini/runs/main/`, so unzip them before rerunning `analyse.py` or `summarise.py`.
- `patterns-and-risks.md`, `reviews/`: Pydantic AI patterns and production reports, and what they change.
- `tools-off-toolcontext.md`: the scope for the planner's readers on a neutral record.
- `prototype-node-stream.md` and `prototype-node-stream/`: the Pydantic AI planner inside an ADK 2.13 workflow node, on a minimal workflow and inside the real Studio turn.
