# Pydantic AI in production: what teams report outside its docs

Checked on 2026-10-31 against pydantic-ai 2.64.0. This extends `currency-pydantic-ai.md` and `use-cases.md` and does not repeat them.

## What the evidence looks like

First-person production write-ups are scarce. The one independent post I found (Codify) does not say where history lives or how pauses survive a deploy, and the vendor case studies do not either. Nearly everything useful sits in the pydantic/pydantic-ai issue tracker and in Langfuse's tracker, where users, maintainers and triage bots describe real failures. Many issues are filed by maintainers on behalf of unnamed users, and the `pydanty` triage bot has commented on several; I say so where the only voice is a bot. No source gives scale figures for a team that owns its loop against a team under a durable engine. I looked on Hacker News, Reddit and GitHub Discussions and found nothing usable.

Strength labels: "pattern" means three or more independent issues; "several" means two; "single" means one report or one post.

## Findings that should change the design

### 1. A pause is at-least-once, so add an effect ledger and do not steer with `enqueue` at approval pauses

Evidence: pattern.

- A crash after an approved tool runs but before the new history is saved runs the tool again. A reporter reproduced a payment running twice on 2.59.0, and found that `ctx.tool_call_id` as an idempotency key helped only where the payment provider honoured keys. The triage bot found no maintainer statement on whether at-least-once is intended. https://github.com/pydantic/pydantic-ai/issues/10926 (open, 2026-10-21)
- Contributors in that thread advise a key scoped to the application operation, `(operation_id, tool_name, tool_call_id)`, with `operation_id` persisted before the first attempt, and warn that a bare tool call id can wrongly deduplicate across operations. Same issue.
- A proposal for a persistent approval inbox lists the same failure family: duplicate deliveries of one approval ran the side effect twice in a sandbox repro, two competing decisions both completed, and an approval given for one set of arguments could be applied to changed ones. It names no production deployment. https://github.com/pydantic/pydantic-ai/issues/11847 (open, 2026-10-29)
- Resuming a harness `Shell` conversation after a crash launches the command a second time, because a resumed conversation is a new run with a new `run_id` and the job name hashes `run_id`. https://github.com/pydantic/pydantic-ai/issues/11951 (open, 2026-10-30)
- A message queued with `ctx.enqueue` in the same step as an approval-required tool turns the `DeferredToolRequests` pause into an ordinary reply, so the approval is skipped without an error. Still `p:4-low`; a maintainer asked for a higher priority because approvals are skipped silently. Fix PR #11956 is open. https://github.com/pydantic/pydantic-ai/issues/7537 (open)
- Deferred results that resolve nothing record an empty `ModelRequest` in stored history instead of raising. https://github.com/pydantic/pydantic-ai/issues/12132 (open, 2026-10-31)

What we would do differently. The prototype's compare-and-swap (L4) covers two resumers racing. It does not cover a crash between a side effect and the history write. Write one ledger row per side-effecting step, keyed on our own operation id and step id, in the same transaction that claims the pause, and have the executor read it before it acts. The prototype already found that a SIGKILL inside `propose_plan` replays that call; the ledger is what makes the same crash safe for `execute`. Drop `enqueue` as the steering mechanism in the README's capability table for any state that can end in a deferred pause, or guard it so nothing is queued in a step that can raise `ApprovalRequired`. Do not rely on `tool_call_id` alone as the idempotency key.

### 2. Stored history has no format version, and rollback is not safe once a release adds new part kinds

Evidence: pattern, plus one first-party upgrade report.

- The proposal for a core message store says old data is guaranteed to deserialize, but nothing says which version wrote an envelope. One project stamps the installed pydantic-ai version; another hard-fails on its own constant. A contributor running Postgres in production stores no version and relies on `ModelMessagesTypeAdapter` compatibility. `format_version` is written and nothing reads it. The core store and its Postgres backend are paused drafts. https://github.com/pydantic/pydantic-ai/issues/9786 (open, updated 2026-10-26)
- Since 2.60.0 tool retries are stored as `ToolReturnPart(outcome='retried')` or `RetryFeedbackPart`. Code that tests `isinstance(part, RetryPromptPart)` matches nothing and reports failed tool calls as successes, for new runs and for old histories. The Logfire team hit it upgrading 2.59 to 2.63, with 42 test failures across 8 subsystems, and noted the bump could not be reverted because stored histories gain parts that 2.59 cannot load. The maintainer rejected a runtime shim and chose documentation. https://github.com/pydantic/pydantic-ai/issues/11374 (open, `breaking change`)
- The 2.60.0 to 2.62.0 notes add new stored shapes: `ToolAvailabilityDeltaPart` for every mid-run tool change, a `CompactionPart` holding the `SummarizingCompaction` summary, and tool returns whose JSON form is a string (`datetime`, `UUID`) sent to the model unquoted. https://github.com/pydantic/pydantic-ai/releases
- Reload bugs for request-side parts surfaced in 2026: native tool returns, tool-search returns and `InstructionPart` do not survive a JSON round trip. Two are closed; `InstructionPart` (#8143) is open. https://github.com/pydantic/pydantic-ai/issues/7048, https://github.com/pydantic/pydantic-ai/issues/8143
- Version policy gives no migration tool. Pydantic's production page says the same: it names no stored-`Conversation` migration guidance. https://pydantic.dev/docs/ai/production/

What we would do differently. `prototype-results.md` says all 45 rollback pairs load. That holds for a record built from signatures, one tool call and return, usage and a pending deferred call. It does not cover a record with a retry, a compaction part or a tool-availability delta, which are the shapes that make 2.60 and 2.62 one-way. Extend `prototype/stability/` with those shapes and with a live Gemini response, and state in the plan that rollback past 2.60 is unsupported for stored planner histories. Wrap each stored `Conversation` in an envelope of our own: our schema version, the pydantic-ai version that wrote it, and the model id. Read the envelope on load and refuse or rebuild from our canonical records when it is older than a floor we set. Never branch on part classes with `isinstance`; branch on `outcome` or on a helper, as the maintainers advise.

### 3. Durable engines tie deploys to Pydantic AI's internal step order; keep them around our own non-agent work

Evidence: pattern. No post-mortem from a named production team; the reports come from users, maintainers and bots.

- A reporter says that upgrading across 2.54.0 changes hook order, so Temporal workflows started on 2.53.0 fail replay with `[TMPRL1100]` and rolling back breaks them too. The triage bot says this does not reproduce on released versions, and the reporter's repro says it does; the issue is open and `p:4-low`. https://github.com/pydantic/pydantic-ai/issues/11767 (open, 2026-10-28)
- DBOS step ordering broke recovery when `ctx.emit()` raced a store read in the harness `Planning` tool. Fixed in 2.62 (#11157); a sibling tool with two durable calls after an emit still fails on recovery, and the advice is `parallel_execution_mode='sequential'`. https://github.com/pydantic/pydantic-ai/issues/10300 (closed, residual gap reproduced by a maintainer)
- The 2.62.0 release notes themselves say that starting `GraphBuilder` broadcast branches in `.to()` order "can break replay of durable workflows". https://github.com/pydantic/pydantic-ai/releases
- Core code calls `asyncio.wait` on Temporal's workflow loop in three places, so completion order may differ between workers. No incident has been shown; PR #9559 and #11128 are in flight. https://github.com/pydantic/pydantic-ai/issues/11182
- Prefect: default `run_id` depends on call order, so concurrent branches swap ids on retry (probe on `TestModel`, not production); and a retry reruns a durable capability operation because the cache key includes measured values. https://github.com/pydantic/pydantic-ai/issues/10085, https://github.com/pydantic/pydantic-ai/issues/11575 (both `p:2-high`, open)
- Temporal payloads: an oversized activity argument, for example binary content in model history, fails the workflow task and retries forever, with no error, until the execution timeout. `p:4-low`; workaround is external storage or a claim-check codec. https://github.com/pydantic/pydantic-ai/issues/7164
- DBOS rules from Pydantic's own page: the agent `name` and each `MCPToolset` `id` cannot change after deploy; adding a workspace or changing steps means in-flight workflows cannot recover and must drain or ship as a new application version; events are buffered, not live; keep payloads under about 2 MB. https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/
- Pydantic's production page separates the two jobs the same way the README does: durable execution for crashes mid-run, deferred tools for waiting on a person, because "pausing for a human needs no live process". https://pydantic.dev/docs/ai/production/
- Temporal's own HITL pattern is a Signal into a waiting workflow with a timeout, and it says nothing about versioning in-flight agent workflows. https://temporal.io/blog/the-heros-journey-to-ai-durability-with-temporal

What we would do differently. The README already keeps the interactive turn out of an engine; the tracker supports that. It also argues for narrowing the add-on. Wrap the detached work that is not a Pydantic AI run (mints, chunked fetches, chart builds, replay of shared cards) in the engine or queue, and leave `agent.run` outside it. That way an upgrade of pydantic-ai cannot change the step order of a stored workflow. If a later requirement forces a Pydantic AI run inside an engine, pin an application version, drain in-flight runs before an upgrade, and treat every minor as a replay-compat event.

### 4. The Interactions API route is a threat to stored thought signatures and to explicit caching

Evidence: single, but it is the maintainers' own tracking issue with live probes.

- `GoogleModel` stores signatures per part; Interactions puts them on the function-call step or on a separate `thought` step. Dropping the `thought` step returned 400 in tool loops on gemini-3.8-flash, and histories recorded before a migration need conversion. https://github.com/pydantic/pydantic-ai/issues/6192
- Explicit caching returned 400 on Interactions; it works on `generateContent` and hit nearly all tokens in the maintainers' test. Implicit warm reads were similar across routes (19/24 and 21/24, 17/24 stateful). Same issue.
- Server-minted tool call ids on Interactions may break resumed tool approvals. Same issue.
- On 2026-10-31 a maintainer comment moved from "migrate wholesale, not maintain a dual path" to a separate stateless route with a history converter. The issue records no final decision.

What we would do differently. Keep explicit caching and the stored signatures on `GoogleModel`'s `generateContent` path and do not adopt the Interactions route. Add a stored-history test that replays a signed planner history through each pydantic-ai bump with the network tapped, and watch #6192 for a change to the default route.

### 5. Gemini defaults move in minors, and error handling around them is unsettled

Evidence: several. Each point is a single report; together they show a pattern of default and error-path drift.

- `VALIDATED` became the default in a minor (PR #6353, merged 2026-07-30). One tool with `strict=False` moves the whole request back to `AUTO`. The motive was an ops agent on gemini-2.5-flash with about 28 tools on pydantic-ai 1.62.0 that called invented tool names in about 3% of model cycles, each aborting the turn. Our prototype measured the opposite effect on gemini-3.8-flash: `VALIDATED` cost 0.6 s on the first chunk and one refusal task. https://github.com/pydantic/pydantic-ai/pull/6353, https://github.com/pydantic/pydantic-ai/issues/5366
- Review of that PR found that cassette replay can hide a change to the outbound request, so the author added matchers on `function_calling_mode` and `function_declaration_count`, and a later change moved Gemini tool-config assertions to the httpx boundary (#7174). https://github.com/pydantic/pydantic-ai/pull/6353
- A response that ends with `finish_reason='error'` is treated as success on Mistral, Bedrock and OpenRouter. A maintainer warns that raising for Gemini would break the recovery a plain agent gets today from an empty `MALFORMED_FUNCTION_CALL` response. Open. https://github.com/pydantic/pydantic-ai/issues/12059
- Gemini 429 `retry_after` is always empty because the delay sits in the body as `RetryInfo`. https://github.com/pydantic/pydantic-ai/issues/12063
- `TOO_MANY_TOOL_CALLS` and two image finish reasons map to `None`. https://github.com/pydantic/pydantic-ai/issues/10898
- On Vertex, a passed-in httpx client with the default 5 s read timeout is sent as the server deadline, so generations longer than about 5 s can 504 and trigger fallback. A fix is planned. The docs advise a read timeout of at least 20 minutes for long-decode models. https://github.com/pydantic/pydantic-ai/issues/11902, https://pydantic.dev/docs/ai/models/google/
- Tool call arguments always arrive whole when streaming on Gemini; progressive deltas exist only on Vertex and are not enabled. https://github.com/pydantic/pydantic-ai/issues/12092
- Explicit caching through `google_cached_content` failed outright on Vertex until 2026-06-02 (#5671), and the reporter noted that no one else had exercised the path. It strips `system_instruction`, `tools` and `tool_config` and warns. https://github.com/pydantic/pydantic-ai/issues/5671, https://pydantic.dev/docs/ai/models/google/

What we would do differently. Set the function-calling mode explicitly on every agent and pin it with a wire assertion at the httpx boundary, so a default flip in a minor fails a test and does not change plan acceptance. Handle the unknown-tool case in the loop, where ADK answers with the tool list and Pydantic AI raises after one retry: treat it as a recoverable state with its own counter. Parse `RetryInfo` ourselves if we back off on 429. If a passed-in HTTP client is used, give it an explicit long timeout. Treat explicit caching as a path with one user, ours.

### 6. Langfuse mapping is where Pydantic AI on Gemini loses information today

Evidence: several, from Langfuse's tracker plus one from Pydantic's.

- Langfuse sums `gen_ai.usage.details.*` keys (`text_prompt_tokens`, `thoughts_tokens`) into Total Usage on top of input and output, roughly doubling it. Reported with Gemini models on pydantic-ai 2.32.0, and again on 2.54.0 with OpenAI reasoning tokens. Closed "not planned" by the stale bot on 2026-10-09 with no maintainer reply; a user asked to reopen it on 2026-10-26. https://github.com/langfuse/langfuse/issues/16489
- Cache-read tokens from Pydantic AI 2.0 spans were dropped and input double-counted in a Langfuse server before the fix; the assignee says platform v4.31.0 or later resolves it. The report was on Bedrock, not Gemini, so the Gemini path needs its own check. https://github.com/langfuse/langfuse/issues/18057
- Langfuse's own page says some attributes land in observation metadata and does not say which. It names no Pydantic AI version and does not mention tool spans or cached usage. https://langfuse.com/integrations/frameworks/pydantic-ai
- Every model-request span carries the full history in `gen_ai.input.messages`, so a long run emits O(n squared) bytes; one customer measured 21% of CPU in OTLP encoding. `include_content=False` is the only lever; a delta option is `p:4-low`. https://github.com/pydantic/pydantic-ai/issues/11921

What we would do differently. Before the move, run one cached-prefix planner turn through the real exporter into the target Langfuse project and compare input, output, cached and thinking tokens and cost with what the Gemini response reports. Since we own the loop, create one parent span per turn from our own turn id and pass it across pauses, as ADK does today; Pydantic AI's run spans will not join a turn that resumes in another process. Decide the `include_content` setting by span size on a 54-turn planner run, the worst case in `use-cases.md`.

### 7. Upgrade cost is real for first-party teams; test at the wire and read the compatibility block each time

Evidence: several.

- The Logfire team's 2.59 to 2.63 bump broke 42 tests (finding 2). Pydantic runs a regression-detector bot; it filed the 2.32.0 regression where sync capability hooks and sync tools stopped propagating `contextvars`, because they now run in a thread pool. https://github.com/pydantic/pydantic-ai/issues/7594
- A Slack sample repo took Dependabot's 1.107.0 to 2.21.0 bump in six days; the only change needed was `MCPServerStreamableHTTP` to `MCPToolset`, and a human posted a message through it as the check. https://github.com/slack-samples/bolt-python-starter-agent/pull/126
- Defaults changed in minors: `end_strategy` to `'graceful'` and the bare `openai:` prefix to the Responses API in 2.0.0b1; Gemini `VALIDATED` in 2.22.0; instruction and compaction records in 2.62; ID replay limited to the creating endpoint in 2.64.0. https://pydantic.dev/docs/ai/project/changelog/, https://github.com/pydantic/pydantic-ai/releases
- 2.63.0 sends signed and encrypted history only to providers with the same `canonical_system`; single-provider users see byte-identical requests, mixed-provider histories do not. https://github.com/pydantic/pydantic-ai/issues/11463

What we would do differently. Pin an exact version and bump on purpose. For each bump, run the tapped-google-genai wire diff the prototype already has over the 62 planner tasks, compare request bytes and not only outcomes, and read the release notes' compatibility lines for every release crossed. Pydantic AI's own suite pairs VCR cassettes with httpx-boundary matchers for the reason in finding 5, which is the same mechanism.

## Migrations from ADK, LangGraph and the OpenAI Agents SDK

I found no first-person account of a team moving from ADK to Pydantic AI, and none from LangGraph or the OpenAI Agents SDK in 2026. Every source below is vendor-authored or a comparison.

- Pydantic's own migration skill for ADK says what does not carry over: `message_history` does not replace ADK session state, memory, artifacts or event records; ADK resumability replays recorded results and can run tools more than once, so "choose a durable execution design explicitly and prove restart plus idempotency for side effects"; hook ordering and skip rules differ; keep `google-adk` installed while anything still imports it. Vendor, 2026. https://raw.githubusercontent.com/pydantic/pydantic-ai/69ea1e57201ed59e7d30fbe4a7590d193acecce0/pydantic_ai_slim/pydantic_ai/.agents/skills/migrating-google-adk-to-pydantic-ai/SKILL.md
- Overjoy moved from LangChain and LangSmith to Pydantic AI and Logfire (2026-04-29): one dashboard and minutes to debug where it had taken half a day, a 20x usage spike caught from a silently repeated call. It lists nothing lost. Vendor case study. https://pydantic.dev/case-studies/overjoy.md
- MindsDB left LangChain (2025-01-27, an earlier release; the CEO called Pydantic AI "young"): "10x better" is undefined, and the open-source product took a breaking change. Vendor case study. https://pydantic.dev/case-studies/mindsdb.md
- ADK's tracker uses Pydantic AI as the model for two opt-ins: MCP output schemas default off, and MCP server instructions behind a flag. https://github.com/google/adk-python/issues/7690, https://github.com/google/adk-python/issues/7696
- A LangGraph write-up (2026-04) says checkpoints are not validated on load, so a new state schema meets old in-flight checkpoints with an error at runtime. It does not mention Pydantic AI; it describes the schema-drift risk our own envelope guards against. https://amtocsoft.blogspot.com/2026/04/langgraph-in-production-state-machine.html

What we would do differently: nothing beyond findings 1 and 2. What we lose by leaving ADK is what the skill lists, and the prototype already rebuilds it (store, cursor, frames).

## Teams that run Pydantic AI at scale

- Codify (2026-09-03): more than 12,000 active users; compaction in three tiers (sliding window, summarising, clearing tool results); tool calls and returns must stay paired when history is sliced, or providers error; thinking level names differ by provider and by model. It says nothing about history storage, durable execution or Gemini behaviour beyond naming `gemini-3.8-flash` in examples, and it names no version. Single post. https://codify.ch/post/a-year-of-pydantic-ai-in-production
- Pydantic's production page says to store the whole `Conversation` after every continuation, serialise concurrent updates with a version column or a lock, and treat a conversation id as a key and not a credential. It confirms our design; Pydantic is the author, so this is not independent. https://pydantic.dev/docs/ai/production/

## Risks we had not listed

- The paused-pause replay in finding 1: a crash between a tool's side effect and the history write has no guard in Pydantic AI, and the maintainers have not said whether they intend one.
- Stored histories are one-way past 2.60 and 2.62 (finding 2). The README's "records from ten releases load under every later one" is true for the record shapes tested and does not extend to rollback.
- `RetryPromptPart` checks that match nothing. Search the planner and orchestrator code for `isinstance` on message parts before the move.
- Default `run_id` format changed in 2.55.0 and may change again (#10085); core refuses to reuse a `run_id` already in `message_history` (#11951). Pass our own ids and keep them in the envelope.
- Sync hooks and tools run in a thread pool since 2.32.0, so `contextvars` set in them do not propagate (#7594). Matters if principal or trace context rides a context variable.
- Thought summaries are replayed to Gemini with the signatures; a Google contact told the maintainer the API drops them, so replay cost is not inflated (closed 2026-01-13). https://github.com/pydantic/pydantic-ai/issues/3947 This is the one item that lowers a worry.
- Google SDK pin: Pydantic AI requires `google-genai>=2.30.0`; the harness lock holds 2.24.0 (`prototype-results.md`). The first upgrade changes a library under ADK too if we keep both during a migration.
- `thinking='minimal'` maps to a 128-token budget that 2.5 flash-lite rejects with a 400 (#11366); other model families name the lowest level differently (Codify). Fixed per model in a table we do not control; test any model swap on its own. https://github.com/pydantic/pydantic-ai/issues/11366
- Harness is 0.x; its Postgres step store arrived in October 2026 (#8991, closed by #9899). Do not make it the system of record. https://github.com/pydantic/pydantic-ai/issues/8991
- `before_tool_validate` argument repairs do not reach stored history, so later requests replay the malformed originals (#12029). Relevant only if we adopt `RepairToolArguments`. https://github.com/pydantic/pydantic-ai/issues/12029

## Sources checked that said nothing useful

- dataaspirant, jahanzaib.ai, baeseokjae.github.io, pynions, freecodecamp tutorials: introductions, no production evidence.
- Medium (alexandrdzhumurat) multi-user post: returned 403; only the search snippet (FastAPI race conditions at 5 to 10 users) was seen.
- zenml.io durable execution page: a Temporal overview naming OpenAI, Replit and Lovable, with no Pydantic AI detail or incident.
- dailyaiworld.com Temporal approval-chain posts: promotional figures with no method.
- Tiger Data Slack bot post (2025-12-17, plushcap summary): no history, durable or scale content.
- pydantic.dev/case-studies index: sixteen entries, none mentions durable execution, an owned loop, Gemini or history storage; Dosu (697K runs) and Boosted.ai (50,000 workflows) are Logfire marketing figures seen only in a search snippet.
- Temporal "hero's journey" post: Signal pattern only, no versioning advice.
- Restate's Pydantic AI integration post: reached only through a mirror (plushcap); original not confirmed.
- pydantic-ai-temporal-example: a demo Slack bot; its pruning advice contradicts its own code.
- GitHub Discussions on pydantic/pydantic-ai: the search returned zero items, so Discussions went unchecked.
- Hacker News and Reddit: no threads returned.
- pydantic.dev/docs/ai/overview/version-policy: 404 at that path; the policy was read through `currency-pydantic-ai.md` only.
