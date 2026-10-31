# Harness use cases and what they demand of the framework

## 1. Current shape

A Studio turn is one ADK `Workflow` of nine nodes (`harness/graph/workflow.py`): `seed`, `orchestrate`, `route`, then `ask`, or `plan_start`, `planner`, `approve`, `execute`, `plan_done`. Two model agents exist. The orchestrator calls nine tools itself (search, fetch, make_charts, ask_user, plan, get_skill, get_skill_file, explain_dfo, recall). The planner drafts a method with its own tools. A translator and compiler (plain code) turn the draft into typed, schema-checked tool calls.

Every calculation goes through `plan`. The plan reaches the user as `plan.proposed`, and the stream ends. The user approves, amends or rejects through `POST /api/studio/interrupt`, which opens a new stream. Approval is keyed on the plan's content revision, and the executor refuses a plan whose hash or tool schemas changed. The executor runs the steps, re-checks evidence, and hands frames back to the orchestrator.

The UI gets twelve SSE frame types (`message.delta`, `plan.proposed`, `activity.step`, `component.created`, `composition.updated`, `elicitation.request`, `done` and others). Models see digests only; rows stay in a shared frame store. State lives in Postgres through ADK's session service plus a per-conversation ledger with a 45 s segment lease. Three MCP servers mount in-process, and a combined `/mcp/` endpoint serves outside clients.

## 2. Forward capabilities

### A. Plan, approve, amend, execute

What: the user approves the method before data is touched, and can amend it. Amendment mints a new plan version.
Status: built. Evidence: `plan.versioned` and `plan.proposed` frames, revision-keyed approval, fixtures in `services/harness/tests/wire_fixtures/studio/`.
Demand: human-in-the-loop (approval, edit, clarification) with an exact-bytes guarantee that what runs is what was approved; durability, because a pause can last minutes and resume on another instance.

### B. Clarifying questions mid-turn

What: the agent asks a question, the turn pauses, the answer returns in the option's words.
Status: built (`ask_user`, `elicitation.request`).
Demand: human-in-the-loop clarification as a pause that is not an error; durability across the stream break.

### C. Steering a run in flight

What: amend a plan during execution; stop a turn. A missing-data failure pauses the run, consults the user, and mints a plan version before resuming.
Status: cancel is built (`/api/studio/cancel`, cross-instance). Executor hand-back to the planner on `MethodChoiceNeeded` is built. Free amendment during execution is committed in the planner concept, not built.
Demand: human-in-the-loop mid-run steering; durability (resume without repeating side effects); concurrency (cancel must reach a running segment on another instance).

### D. Durable conversations across instances

What: any instance continues any conversation; a crashed turn resumes; an interrupt answer is retryable.
Status: built. Evidence: ledger checkpoint before every pause or done frame, lease renewal every 5 s, retry rules in the studio contract. A shared state database was planned to end approval 404s.
Demand: durability (crash mid-turn, replay, exactly-once publish); scale (state keyed on principal and conversation, two pooled connections per instance against ten reserved).

### E. Streaming progress during planning

What: show steps while the planner and translator run, instead of a burst after 89.7 s.
Status: committed plan, not built. ADK's node runner has no streaming form, so a producer and consumer bridge is planned that still propagates a user pause.
Demand: UI protocol (streaming from inside a long node); concurrency (producer and consumer inside one step); long-running work.

### F. Parallel work inside a turn

What: parallel searches, fetches and charts for multi-part questions; one requirement per variant in comparative plans; the depth-hint call concurrent with the converse call; concurrent chunked mints.
Status: partly built and partly committed. A measured planner turn spent 9 to 12 sequential model calls of 1 to 5 s each, and the design names parallel calls as a cost lever.
Demand: concurrency (parallel tools, fan-out and join) with approval and event ordering kept deterministic.

### G. Direct execution beside the planner

What: the orchestrator runs fully specified calculations through one shared submission protocol and compact capability catalogue; the planner handles analytical design. Both agents share validation and execution.
Status: proposal (received design under `research/planner-schema-discovery-2026-10-11/claude/direct-execution-proposal.md`), to be tested at 10, 30 and 60 capabilities.
Demand: multi-agent composition over shared services; context control (catalogue growth, preloading, retrieval of capability cards as ordinary tool results).

### H. Skills and growing domain knowledge

What: skills load on demand from search results; sectoral skills and community contributions follow launch.
Status: built (`get_skill`, `get_skill_file`, `dfo-faq`); expansion is committed. ADK's `SkillToolset` was rejected because its registry returns every file and its catalogue is fixed at build time.
Demand: context control (disclosure by pull, stable tool prefix for caching); multi-agent composition (planner and orchestrator read the same text by the same names).

### I. Context control over long conversations

What: bounded model context with recall of older turns. Today: a canvas digest of the newest 16 blocks, a conversation digest of the newest 10 turns, a `recall` tool, a shared prefix cache, and a rule that no tool returns more than a few hundred tokens to the model. Compaction that keeps decisions, approvals and references, and collapses failed calls, is designed.
Status: digests, recall and caching built; compaction is a design intent.
Demand: context control (per-agent views, replace-not-append state, compaction, large results kept out of band); per-agent message isolation, because events without a branch currently reach every agent's prompt.

### J. Long-running data work

What: large aid queries run as background operations. The mint door returns 202 and an operation id, a Cloud Tasks worker runs it at 2 vCPU, and the client polls. The executor retries within a budget (three attempts per step, waits capped at 30 s, 180 s `get_data` timeout).
Status: committed plan for the worker; the executor's retry handling of `operation_still_processing` is built.
Demand: long-running or background work (minutes, survives restarts); durability and idempotent retry (operation ids derive from request identity).

### K. Live refresh and replay of shared cards

What: a share redraws a card on current data by replaying its stored recipe, with no model or session. A manifest lets any completed conversation replay with no model in the loop.
Status: refresh route built (`harness/live_refresh.py`); manifest replay is an ideal-launch requirement with recipe publication built.
Demand: durability (deterministic replay, exactly-once publish); background work outside any turn; observability (what ran is provable).

### L. Outside agents and MCP clients

What: Claude, ChatGPT, Gemini and Cursor users call DFO's eleven tools through `/mcp/`, and presentation tools render as MCP Apps. The workflow as a tool (`plan` and `run` behind MCP) was deferred until usage argues for it. Nothing inside the harness uses A2A.
Status: tools built, pilot not started, workflow-as-tool deferred (idea), presentation on the combined endpoint waits on the platform edge.
Demand: multi-agent composition (external agents through MCP, possibly A2A later); UI protocol (MCP Apps, `structuredContent` handling varies by client); scale (rate limit 5 calls/s with burst 20 per instance).

### M. Generative UI and the Edit Card loop

What: the model emits a small chart intent; the platform assembles and validates it. Edits restate the intent partially. Card edits bypass the workflow with one constrained model call.
Status: built (`harness/studio_edit.py`, `harness/generative.py`); the catalogue grows after launch (maps, layout tree).
Demand: UI protocol (typed component events, MCP Apps parity); model breadth (a cheap constrained model for edits); context control (the grammar stays in tool definitions, errors carry the fix).

### N. Model breadth and per-agent tuning

What: Gemini today with a pinned model and a LOW thinking level on planner and orchestrator. Design says provider-swappable. A judging model for Studio replies is wanted later. Cost and latency turned on thinking level (reasoning tokens ran 34 times visible ones).
Status: Gemini built; non-Gemini is stated intent (idea).
Demand: model breadth (per-agent model and thinking settings, schema quirks such as dropped `$ref` descriptions, caching on the provider side).

### O. Observability and evals

What: one Langfuse trace per turn, continuous across a pause and a resume on a different request; ADK eval sets; scripted fakes; a judge for rubric cases; a corpus gate (90% composition quality proposed).
Status: built.
Demand: observability across pauses and instances; evals that drive the real agent deterministically (scripted model fakes).

### P. Identity, access and abuse limits

What: anonymous and signed-in users on one code path; sign-in adopts anonymous sessions; the edge adds per-caller limits.
Status: committed. Studio auth is a placeholder shared secret; the edge design is not built.
Demand: scale (per-principal isolation, concurrency caps, expiry of anonymous sessions); durability.

## 3. Scale figures

- No user, turn-per-day or concurrent-session target is stated anywhere. Do not assume one.
- Search API p95 under 4 s at launch (launch scope).
- Simple Studio questions took 9 to 12 s; planned questions had a median 59 s in the planner and a worst turn of 659 s (a 54-turn run on deployed dev). Targets set for the fix: planner p50 under 20 s, then under 5 s on shaped questions; aid fetches under 5 s; a fully specified bilateral ask at or under 15 s.
- Runtime: one vCPU, up to two Cloud Run instances, no affinity; a mixed wave of six turns moved first byte from 2 to 6 s up to 12 to 21 s. Two CPUs and a minimum of one instance are proposed.
- Session store: two pooled Postgres connections per instance; ten reserved; up to four instances during a rollout.
- Segment lease 45 s, renewed every 5 s. Live refresh: 45 s deadline, hard stop at 53 s. Combined MCP call cap 45 s; ChatGPT and Codex stop at 60 s; Cloudflare cuts a silent stream at 100 s; SSE ping every 15 s. Sync mint timeout 120 s; async API budget 300 s.
- Context: canvas digest 16 blocks; conversation digest 10 turns; model tool results a few hundred tokens; 1 MiB MCP SSE cap; 32 MiB Cloud Run HTTP/1 request cap; 24 series per component.
- Data: 13,090 CRS StatVars (code index 228 KB); member cap 700; about 62 of 500 candidates carry data for a given country and year. Frame store retention 30 days from last use; pinned items 90 days.
- Outside MCP: 5 calls/s, burst 20, one bucket per instance for all callers.

## 4. The three demands that matter most

1. Durability with human pauses. Every headline feature (approval gate, amendment, clarification, cancel, retryable interrupt) is a pause that resumes on a different instance with the approved bytes intact, and the plan is a record that must replay.
1. Context control. The product rests on models seeing digests, with rows out of band, on per-agent views, recall, compaction and a growing skill and capability catalogue; several current workarounds exist because agents shared one event stream.
1. UI protocol and streaming. The wire contract is frozen, progress must stream from inside long steps, and the same components must serve the site and MCP Apps hosts.

Concurrency and long-running work come next: they are the clearest gap between the current design and the committed plans, but no document sets a user count that forces them.
