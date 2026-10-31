# Pydantic AI or ADK for the harness

## Verdict: REVISE. Stay on ADK, upgrade to 2.13.0 now, and decide on migration only after two measurements

Upgrade this week whatever else is decided: three minors, one deleted private path used in one test file (`tests/orchestrator_fakes.py:30`), and a telemetry flip held off with `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` until at least 2027-03-31. Do not migrate now, and drop the partial option as framed. Count the forward cost of ADK from git history first; run a five-day planner spike only if that count says there is a payoff.

## The facts that decide it

1. ADK's `Workflow` is our orchestration, so there is none of our own to keep. The nine-node graph (`harness/graph/workflow.py:140-158`), the replay engine that resumes a paused turn on another instance, and the pause itself (`RequestInput`, `NodeInterruptedError`, `ctx.resume_inputs`) all live in ADK. Pydantic AI offers no replacement: `pydantic_graph.persistence` is gone in V2, and its pause is agent-level (`Conversation` plus deferred tools). Leaving means writing a durable turn loop, a persisted cursor plus one stored `Conversation` per agent, on the compare-and-swap ledger we already run in `studio_state.py`. The other partial reading, a Pydantic AI model behind an ADK `LlmAgent`, buys nothing: our coupling is the session and event model (93 `tool_context` sites in eight files; four planner tools read `tool_context.session.events`, `planner/agent.py:705-927`), not the model call.

1. Most of our ADK workarounds sit in that orchestration layer, and an explicit cursor removes them by construction. Branch stamping and `_check_branches`, `_turn_ran_before`, `rerun_on_resume`, `_persist_durable`, the empty-string output (`graph/orchestrate.py:71-76`), the queued and dropped planner progress (`graph/planning.py:111-137`) and the `include_contents` pins exist because ADK shares one session across agents and replays from START. With one message history per agent and a pause that is a return value (`DeferredToolRequests`) instead of a `BaseException` raised through nested awaits, none is needed. Pydantic AI also validates tool arguments, which retires `_validated_arguments` (`orchestrator_tools.py:852-915`). A raised exception in a tool body still needs our `tool_failures.py` wrapper on either framework.

1. The model-layer work moves; it does not go away. `PrefixCache` (303 lines) and the turn budget (312) stay ours, since Pydantic AI creates no caches and has no budget. Our `$ref` inlining in `planner/declaration.py` must still run, since `GoogleModel` keeps `$defs`/`$ref`. Pydantic AI changes faster than ADK (about three releases a week, 1,229 open issues, graph persistence removed four months ago), and its message history, the thing we would persist, has no schema version and is exempt from the stability promise for new parts. We would trade a private-import repair per ADK upgrade for a stored-history replay check per Pydantic AI upgrade. The MCP `structuredContent` risk does not apply: no `McpToolset` exists in `services/harness/src`.

## The strongest argument against staying

ADK's model works against our design, and each new node or agent pays again. Most docstrings in `graph/` and `planner_seam.py` explain ADK semantics. The HITL engine took six replay fixes between 2.6 and 2.13, with #7666 and #7609 open, in the one area a human-in-the-loop product cannot get wrong. A nine-node state machine with an explicit cursor is small and correct by construction.

It holds as a design argument and fails as a reason to act today. The workarounds are built and tested, so what matters is the forward rate of new ADK-caused work, which nobody has measured. And Pydantic AI does not supply the cursor; we write it either way. Its value over our own loop on plain `google-genai` is the tool loop, argument validation, streaming events, `FunctionModel` test fakes and other providers. If those do not matter, our own loop with no agent framework is the cheaper structural option.

## What a migration costs

`services/harness/src` holds 61,237 lines in 94 files; 30 import `google.adk`.

Goes away, about 600 lines: ADK edge objects in `graph/workflow.py`, branch isolation (about 150 lines), `_turn_ran_before`, `_persist_durable`, `_validated_arguments` and the planner's list self-validation, `session_store.py` (162), ten private HITL imports, and ADK span bending in `packages/common/src/common/telemetry.py`.

Rewritten:

- The run loop in `studio.py` (1,437) and the event-reading core of `studio_translate.py` (about 900 of 2,108).
- `Context` and `Event` plumbing in the node bodies and `state.py` (about 2,500 lines, mostly business logic that stays).
- Most of `planner_seam.py` (1,875) and about 500 lines of `planner/agent.py`, including the four session-event readers, which must read the agent's message history instead.
- The 93 `tool_context` sites, moving to `RunContext.deps`, and `orchestrator/agent.py`.
- `PrefixCache` and `BoundedGemini`, re-hosted on a wrapper model.
- The ledger's compare-and-swap on our own table, with a data migration of the `dfo_analyst_graph.ledger` sessions; paused turns are dropped at cutover.
- Langfuse mapping, cache dashboards and the out-of-repo session viewer.
- 35 test files, eight scripted `BaseLlm` fakes, `test_graph.py` (2,708 lines) and the 18 `adk eval` cases.

Survives unchanged: `compiler/`, `executor/`, `planner/plan.py`, `mcp_client.py`, `combined_mcp.py`, `wire.py`, `sentences.py` and the rest of the business logic, about 40,000 lines.

That is 10,000 to 12,000 source lines and a third of the tests. The risk sits in the new turn loop, which must reproduce behaviours 1-5, 11-13 and 17-18 in `footprint-runtime.md` (resume, cancel, lease).

## Missing evidence and how to get it

The forward ADK cost. Count commits to `services/harness` since ADK 2.0 GA (2026-05-19) whose change is an ADK workaround or upgrade repair: diffs touching branches, replay guards, `rerun_on_resume`, `include_contents`, `_validated_arguments` or the private imports. Under one a month and falling, migration buys nothing. Two or more a month, or any production resume defect traced to ADK, justifies the spike.

Pydantic AI on our hardest agent. Put a Pydantic AI planner behind the existing `PlannerModel` seam (`graph/seams.py:56`), inside the current ADK `planner` node. Its `ask_user` becomes a deferred tool; the node turns `DeferredToolRequests` into a node interrupt, as `approve` does, and stores the planner's `Conversation` in its own table. The planner carries every hard part: pause, session-event reads, declaration, thoughts, cache and budget. Timebox five working days, measure an ADK baseline on the same cases first, and require every line:

- Outcomes: on the planner_entry and scripted Studio eval cases, three runs each, no case falls from passing to failing in a majority of runs.
- Cache: cached share of prompt tokens on the planner's second and later calls within 5 points of the ADK planner; requests against the cache carry no system instruction, tools or tool config.
- Wire: the `propose_plan` declaration reaches Gemini identical to `compact_schema` output; replayed calls keep their thought signatures; p50 first-chunk latency within 10%. Pydantic AI defaults Gemini to `VALIDATED` calling, which may refuse or slow a schema this large.
- Resume: pause, kill the process, resume in a fresh one; the plan equals an unpaused scripted run's, and a `Conversation` stored under 2.64.0 loads under the newest release at the spike's end.
- Code: net removal of workaround lines on the planner path, zero private imports, no patches of Pydantic AI internals.
- Traces: one Langfuse generation per model call, usage and cached tokens priced once.

Any failure ends the spike and we stay. A pass justifies a plan that moves the orchestrator the same way and then replaces `Workflow` with our own loop. Two frameworks in one service is acceptable only as that plan's first step, with a date for the second.

## Decisions for the user

1. Is a non-Gemini production model for the planner or orchestrator a goal in the next six months? Provider breadth is Pydantic AI's clearest gain; ADK's LiteLLM and Anthropic adapters are mid-repair. Recommendation: no, unless a named model or customer requires it, in which case run the spike now.
1. When to spend the spike week: now, after the ADK-cost count, or not at all. Recommendation: after the count, which takes an hour and decides whether the week buys anything.
1. If the spike passes, the target: Pydantic AI agents under our own loop, or our own loop on plain `google-genai`. Recommendation: Pydantic AI, because validation and `FunctionModel` replace code we would otherwise write, and the spike will have measured it.
