# ADK workaround and upgrade cost in dfo-ai

## Population

The brief's command (`git log --since=2026-05-19 origin/dev -- services/harness evals packages/common`) returns 106 commits, but 9 of them are bulk "Sync" or "Snapshot" commits (2026-10-05 to 2026-10-14) that squash the whole earlier history into 100 to 400 files each, so month by month they say nothing. The old remote `one` keeps that history at full granularity, back to the first commit on 2026-07-17. This study therefore uses the union of two sets over the same paths:

- `one/dev`: 194 non-merge commits, 2026-07-17 to 2026-10-14.
- `origin/dev`: 93 non-merge commits that have no twin in `one/dev` (matched on subject with the PR number stripped), 2026-10-12 to 2026-10-31. The 9 snapshot commits, 2 merge commits and the re-authoring chores that only repin dependencies are not double counted.

Total classified: 287. Of these 19 are ADK-classified (7 ADK-WORKAROUND, 2 ADK-UPGRADE, 10 ADK-PARTIAL) and 268 are OTHER. The repo has no history before 2026-07-17, so May and June 2026 cannot be measured, and the ADK 2.0 GA date (2026-05-19) predates every commit.

## Method and limits

- Every commit's message and numstat were read. Message bodies almost never mention ADK (14 mentions across 287 commits, none describing a defect), so classification rests on diff hunks: lines that add or remove `google.adk` imports, `rerun_on_resume`, `include_contents`, `state_delta`, `.branch`, `*_callback`, `ToolContext`, `NodeInterruptedError`, `RequestInput`, session-service and Runner calls, plus comments that cite ADK behaviour.
- 111 commits had three or more ADK-pattern lines in non-test Python (or an ADK import change). I read the hunks of every one that could plausibly qualify and classified 19. The other 176 had fewer than three such lines and are OTHER by that screen.
- ADK-PARTIAL needs about a fifth of the diff to be ADK accommodation. Ordinary use of ADK as the framework (building `LlmAgent`s, `Workflow` nodes, `ctx.state` reads, `adk eval` sets, `adk web`) is OTHER, because any framework needs equivalent wiring.
- Borderline commits left as OTHER: `9f96ccb7` (deletes the ADK `Workflow` graph engine, 27,527 Python lines; the message gives no ADK motive), `670248bf` (splits the ADK workflow module into a package; organisational), `73637e94` (resets stale state keys for reused ADK sessions; small), `3dc8f835` (telemetry filter for ADK's `invocation` span; under a fifth), `3e55ddcf` (comments on ADK event ordering; tiny), `6a475479` (Gemini transient-error retry; Gemini API behaviour), `291e2775` (Gemini ignores descriptions beside `$ref`; Gemini behaviour), `5e278aa9` and `69f5bee3` (MCP retry budget shaped by ADK's 300 s tool-call ceiling; comments only), and the ADK-eval commits `9898cd94`, `444d8085`, `62e1f695`, `1ffe0bac`.
- "Python lines changed" counts every added and removed `.py` line in the whole commit, so a mixed commit such as `85793a9b` or `96f686f3` inflates its month. "ADK-facing lines" counts the added and removed lines (source and tests) that match the ADK patterns above; it is an upper bound on workaround cost, because it also counts plain use of ADK APIs.

## Month by month

| Month   | Total commits | ADK-WORKAROUND | ADK-UPGRADE | ADK-PARTIAL | Python lines changed in ADK-classified commits | ADK-facing lines inside them | ADK-classified share of commits |
| ------- | ------------- | -------------- | ----------- | ----------- | ---------------------------------------------- | ---------------------------- | ------------------------------- |
| 2026-07 | 1             | 0              | 0           | 0           | 0                                              | 0                            | 0.0%                            |
| 2026-08 | 34            | 2              | 1           | 0           | 596                                            | 74                           | 8.8%                            |
| 2026-09 | 49            | 2              | 1           | 6           | 43,568                                         | 2,231                        | 18.4%                           |
| 2026-10 | 203           | 3              | 0           | 4           | 5,685                                          | 254                          | 3.4%                            |
| Total   | 287           | 7              | 2           | 10          | 49,849                                         | 2,559                        | 6.6%                            |

Per 1,000 changed Python lines, ADK-pattern lines across all commits are: July 37, August 8, September 29, October 1 to 14: 18, October 15 to 31: 14 (diff-wide, including plain ADK use).

Within October the ADK-classified commits fall from 6 of 119 (5.0%) in the first half to 1 of 84 (1.2%) in the second half (`aa08c49b`, a session-row migration).

## ADK-classified commits

- `1a6795d3` 2026-08-05 evals/README: the adk eval CLI needs --config_file_path explicitly | ADK-WORKAROUND | `adk eval` ignores the eval set's test_config.json unless `--config_file_path` is passed and scores against ADK's default criteria, failing every real run; the commit documents the flag (9 doc lines).
- `2a9788e0` 2026-08-05 harness: bump google-adk to match planner's >=2.6 pin | ADK-UPGRADE | raises the `google-adk` floor to >=2.6 to match the planner pin (14/2 lines in pyproject.toml); nothing needed repair.
- `253551b5` 2026-08-08 feat(harness): strip structured content from tool results (#46) | ADK-WORKAROUND | new `harness/tool_results.py` (104 lines) exists because ADK's `McpTool` dumps the whole structured content into the model's function response and plugin callbacks bypass the agent-level `after_tool_callback`; about a fifth of the commit's code.
- `0cdfed06` 2026-09-16 fix(harness): serialise discovery and bound every MCP call (#206) | ADK-PARTIAL | adds a per-session `discovery_lock` because ADK dispatches one turn's function calls with `asyncio.gather`, so two discovery calls raced on session state; the rest (MCP call bounds) is ordinary.
- `cd255bf3` 2026-09-16 test(harness): add the harness-translator-v1, harness-planner-v2 and harness-pair-v1 eval sets (#181) | ADK-PARTIAL | eval sets for `adk eval` and `adk web`: a custom outcome metric (363 lines, about 860 test lines), a discovery stub and sys.path shims, because ADK's evaluator cannot assert on session state or find sibling packages; roughly a third of the commit.
- `85793a9b` 2026-09-19 harness: run the planner and translator in one session, and let the planner ask the user (#221) | ADK-PARTIAL | runs planner and translator in one session via `ctx.run_node`: `rerun_on_resume=True` on the planner node, `NodeInterruptedError` handling, `include_contents` pinning, `state_delta` carried across resume, and the 2.6 to 2.9.1 pin move; about 270 ADK-facing source lines.
- `3d97b31c` 2026-09-20 harness: make the translator's closing turn ignore what the model retyped (#226) | ADK-WORKAROUND | sets the translator's `after_tool_callback` as a list ahead of `canonical_after_tool_callbacks` and uses `SetModelResponseTool` so ADK's closing turn ignores what the model retyped; the whole 456-line diff is callback-chain handling.
- `96f686f3` 2026-09-21 harness: bound what the planner and translator models see, and test it (#231) | ADK-PARTIAL | stamps the planner run with a branch so ADK's `_get_contents` drops the translator's sub-branch events (24,683 of 56,407 history characters had leaked as fenced user text), plus `context_probe`, `test_context_budget` (1,877 lines) and baselines that measure what ADK sends; over a quarter of the commit.
- `37df23dc` 2026-09-23 fix(harness): persist studio conversations through the ADK session service (#246) | ADK-PARTIAL | persists conversations through ADK `DatabaseSessionService`: `session_store.py` (162 lines) handles the asyncpg-only engine, Cloud SQL connector, pool settings and the fact that ADK `close()` does not dispose an engine it did not build; adds `google-adk[db]`.
- `555e45f8` 2026-09-23 fix(harness): plan at LOW thinking and bound a planner entry (#244) | ADK-PARTIAL | composes `BuiltInPlanner` thinking with callback lists because ADK's callback chain stops at the first non-None result and a resumed entry must read its draw back from session state; about 90 ADK-facing source lines of 3,598.
- `e2b12abb` 2026-09-29 feat: serve data and analysis to outside MCP clients through one server at /mcp/ (#261) | ADK-UPGRADE | moves `google-adk[db]` 2.9.1 to 2.10.0 and pins `mcp>=2.2.0` because only ADK 2.10.0 admits mcp 2.x; it lands inside the /mcp/ server feature, so the commit total overstates the cost (a few pin lines plus lock churn).
- `e1176853` 2026-09-30 fix(harness): stop a cancelled Studio turn hanging its segment teardown (#260) | ADK-WORKAROUND | a cancelled Studio turn hung in teardown because Starlette's repeated cancellation hit the ADK `DatabaseSessionService` rollback and connection close; fix is a shielded `_unwind` and `asyncio.timeout` (298 lines, 50 test lines). Part of the cause is SQLAlchemy and anyio, so attribution is partial.
- `ac8ef3c3` 2026-10-03 fix(common,harness): price model calls once, group turns by conversation (#314) | ADK-WORKAROUND | `common/telemetry.py` exports ADK's `call_llm` spans as plain spans with empty usage because Langfuse derived cost from the `gcp.vertex.agent.llm_response` JSON and priced each model call twice (ADK span shape).
- `cdc7a5c7` 2026-10-05 perf(harness,data-mcp): share prompt caches and cut wasted planner rounds (#331) | ADK-PARTIAL | `prefix_cache.py` (292 lines, 453 test lines) subclasses ADK's `Gemini`, `LlmRequest` and model registry to share implicit prompt caches, and drops the `ContextCacheConfig` path that blocked them; about 30% of the commit.
- `7fc12134` 2026-10-09 fix(harness): answer a raising agent tool instead of failing the turn (#389) | ADK-WORKAROUND | adds `on_tool_error_callback` and `orchestrator/tool_failures.py` because an exception raised in a tool body escapes to ADK and ends the turn (title: 'instead of failing the turn').
- `299f5fbc` 2026-10-09 fix(harness): return malformed orchestrator tool arguments as errors (#387) | ADK-WORKAROUND | orchestrator tools validate in `_validated_arguments` and return `{"error": ...}` because ADK hands a function tool the model's parsed JSON unvalidated and a `ValidationError` ends the turn (the docstring says so).
- `4290483c` 2026-10-09 fix(harness): say where a misplaced argument goes and report it as data (#388) | ADK-PARTIAL | extends `_validated_arguments`, the shim for ADK's unvalidated arguments, with per-argument hints (139 code lines, all inside that layer).
- `b091bd04` 2026-10-12 feat(harness): public planner seam hooks for the planner entry eval (#425) | ADK-PARTIAL | replaces the planner entry eval's own ADK `Workflow`, `Runner` and callback-list patching with public seam hooks in `evals/planner_entry/entry.py`; 40 ADK-facing source lines of 606.
- `aa08c49b` 2026-10-29 feat(harness): migrate stored bound refs to the list form the reader takes (#107) | ADK-PARTIAL | migrates stored bound refs inside ADK session rows: reads `DatabaseSessionService` and `StorageSession` v1 and preserves `update_time` because `append_event` raises on a stale one; about a third of 665 lines.

## Production or user-visible defects traced to ADK

No commit message names a production incident caused by ADK. Traced from the diffs and docstrings of the fixing commits, five defects qualify, and the first two are the only ones with evidence of reaching users:

1. `7fc12134` and `299f5fbc` (2026-10-09): a tool that raised, or a model call with malformed arguments, ended the whole Studio turn, because ADK passes arguments unvalidated and surfaces a tool exception as a failed graph node. User-visible as a failed turn; fixed with `on_tool_error_callback` and `_validated_arguments`.
1. `e1176853` (2026-09-30): a cancelled Studio turn hung in segment teardown, stranding its lease; the cleanup that was re-cancelled is ADK's session-service rollback and connection close. Partly SQLAlchemy and anyio.
1. `ac8ef3c3` (2026-10-03): model calls were priced twice in Langfuse because of the shape of ADK's `call_llm` spans. Cost reporting only.
1. `96f686f3` (2026-09-21): the planner's prompt carried the translator's calls as fenced user text, 24,683 of 56,407 history characters on the reference trace, until the planner run was stamped with a branch. Found by measurement, with no user report.
1. `253551b5` (2026-08-08): ADK's `McpTool` put the full structured content of every tool result into the model's context. Caught in development.

Two ADK-related hazards are recorded in memory but have no fixing commit in this population: a node added to the workflow runs live on turns paused before the deploy (resume replays from START), and graph-node events with no branch reach every agent. Neither shows up as a commit.

## Conclusion

Over the 106 days with data, 6.6% of commits (19 of 287) are ADK workarounds, upgrades or mixed commits, and the share peaked in September (18.4%, nine commits, during the planner, translator and session-persistence build-out) and fell to 3.4% in October (1.2% in the last 17 days). Upgrades cost almost nothing (three pin moves, 2.6, 2.9.1 and 2.10.0, with no repair commit after any of them), so the ADK cost is design friction in resume, branch, callback and session handling, which is falling rather than rising; it does not by itself justify a migration spike, although the 27,527-line removal of the `Workflow` graph engine (`9f96ccb7`) and the 6,271-line module split (`670248bf`) show the largest ADK-coupled code has already been rebuilt once.
