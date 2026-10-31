# What the pattern reviews and the ADK 2.13 upgrade change

Two reviews looked for Pydantic AI patterns the prototype should use. One read Pydantic AI's docs, source and release notes at 2.64.0, still the latest release (`reviews/pai-docs-patterns.md`). The other read production reports and the issue trackers (`reviews/pai-in-the-wild.md`). The production evidence is thin: one first-person post from a team running Pydantic AI at scale, and the rest from the pydantic-ai and Langfuse trackers. Dev also moved from ADK 2.10 to 2.13 after the prototype ran.

The net effect is that four of the migration's costs shrink and four new risks join the list. None of the new risks is a reason to stay on ADK. Each one needs a design rule in the loop we would own, and each rule can be tested in a second prototype round.

## Costs that shrink

- **Cache sharing.** With `google_cached_content` set, Pydantic AI sends no instruction, tools or tool config, so the cache resource alone decides what Gemini sees (`models/google.py:1225-1229`). The prototype built its cache prefix from Pydantic AI's own request shape (`prefix()` in `prototype/gemini/pai_planner.py`), which is why the two frameworks got different cache keys. Built from the canonical declarations, as ADK declares them, one cache should serve both frameworks. This also fixes the bytes behind 85% of prompt tokens against changes in how Pydantic AI renders declarations, and it removes the prototype's one private call. This is unmeasured.

- **ADK behaviours with no equivalent.**

  - A call to an unknown tool is already answered with the list of available tools (`tool_manager.py`, `_resolve_tool`). The prototype raised an error only because the default retry budget is 1.
  - An empty response is already re-prompted.
  - A stream cut by the turn budget is the only one left. It needs one loop function: drop the interrupted response and append the closing sentence.

- **The seam on public hooks.** The prototype patched the google-genai client to apply the turn budget and the seam's synthetic answers. These public hooks and settings replace that patch:

  - `SkipModelRequest`;
  - `first_event_timeout` and `request_timeout`;
  - a run `timeout=`;
  - `UsageLimits` with the documented wrap-up capability;
  - `on_tool_execute_error`;
  - `RetryModelRequest`;
  - `strict=False` on tools, which keeps `AUTO`.

  With the cache change, no private Pydantic AI names remain.

- **google-genai.** Dev now runs 2.34, which Pydantic AI requires.

## The same risk on both sides

Both frameworks change what the model sees from one release to the next.

- Pydantic AI made `VALIDATED` the Gemini default in a minor release (pydantic-ai#6353, merged 2026-07-30).
- ADK 2.13 appends "You are an agent. Your internal name is …" to a chat-mode agent's instruction, with no setting to turn it off. On gemini-3.8-flash that line alone took the orchestrator's first call from 0 to between 98 and 703 reasoning tokens, and from about 2.5 s to 7 to 12 s a call. Dev strips it in a before-model callback (`_without_adk_identity`, `orchestrator/agent.py`).

A standing check on what reaches Gemini is therefore needed whichever framework we run, so it is not a migration cost. For Pydantic AI the cheap guards are:

- pin the exact version;
- set the function-calling mode explicitly;
- treat `PydanticAIDeprecationWarning` as an error in tests;
- snapshot the `AgentInfo` that `FunctionModel` receives;
- run the tap on each version bump.

## New risks, and the rule each needs

- **A pause resumes at least once.** A crash after a tool's side effect and before the history write runs the tool again (pydantic-ai#10926, open). Pydantic AI's own ADK migration guide says ADK resume can do the same.
  - Write an effect ledger, one row per side-effecting step, keyed on our own operation id, in the transaction that claims the pause.
  - Checkpoint each model response before its tools run (`after_model_request`). A kill inside `propose_plan` then reruns that one call with its original id, not the whole planner run.
- **A queued message skips an approval.** A message queued with `ctx.enqueue` in the step that pauses for approval turns the pause into an ordinary reply, with no error (pydantic-ai#7537, open). Keep `enqueue` out of every state that can pause.
- **Stored history carries no format version, and rollback is one-way for some shapes.** Tool retries (2.60), compaction parts and tool-availability deltas (2.62) are stored in shapes older releases cannot read. The Logfire team could not roll back from 2.63 to 2.59 (pydantic-ai#11374).
  - The prototype's 45 rollback pairs cover only the shapes it built.
  - Wrap each stored `Conversation` in an envelope of our own: a schema version, the Pydantic AI version that wrote it, and the model id.
  - Extend the stability matrix to those shapes and to a live Gemini response.
- **The turn budget would count the whole conversation.** `conversation=` carries cumulative usage, so a per-turn `request_limit` would count every earlier turn. Run each turn on a fresh `RunUsage` shared by both agents, as `TurnBudget` is shared today.
- **Tracing.** On Gemini, Langfuse doubles total usage from Pydantic AI spans (langfuse#16489, closed by the stale bot), and every span carries the full history (pydantic-ai#11921). Dev's new `harness.langfuse_export` loads Langfuse observations into BigQuery, and dev's spans follow ADK's schema v2.
  - Pin `InstrumentationSettings(version=5)`.
  - Compare one trace from each framework, field by field, through that export.
- **Gemini's Interactions API.** `GoogleModel` may move to it (pydantic-ai#6192, open), which threatens stored thought signatures and explicit caching. Stay on `generateContent` and watch that issue at each bump.

## Durable engines

Durable engines (DBOS, Temporal, Prefect) do not fit the interactive turn:

- DBOS and pgtask buffer the model stream;
- every engine reruns the tool that was in flight;
- DBOS fails on recovery when a deploy changes the order of recorded steps (`DBOSUnexpectedStepError`).

They fit beside the loop, for work that leaves the user's stream, such as mints and fetches. DBOS, a library on Postgres with no new service, is the candidate to trial there.

## The Studio translator

Build the Studio translator as a subclass of `UIEventStream`, the public base Pydantic AI's own UI adapters use, emitting today's frames. Give every custom event a fixed name (`name='plan_progress'`), since the name is the wire identifier.

## What the ADK 2.13 upgrade showed

The upgrade from 2.10 to 2.13 took one change of 52 files, 2,163 lines added and 995 removed. `planner_seam.py` alone took 508 changed lines and `test_planner_seam.py` 664. Since 2.11, ADK hides a user-authored event from agents running at another node of the workflow. So the orchestrator now runs in chat mode, and the planner gets its full brief on every request. This is the upgrade cost on the ADK side, measured on our code. The Pydantic AI side has no equal figure yet: its stored records held across ten releases, and its wire changes are the risk named above.

The upgrade also moved the baseline. The prototype's Gemini comparison ran against ADK 2.10. Its parity and paired run need repeating against dev at 190c7a9c.

## A second prototype round

Each item has a pass line:

1. **Parity and the paired Gemini run against dev on ADK 2.13**, with the cache prefix built from canonical declarations. Pass: both frameworks use one cache name, and acceptance and first-chunk latency hold.
1. **The seam and turn budget on public hooks**, with the client patch removed. Pass: the planner-entry results hold. One extra check: whether `on_model_request_error` receives a streamed `first_event_timeout`.
1. **A per-response checkpoint and the effect ledger.** Pass: after a kill inside `propose_plan`, the planner makes one extra model call at most and sends no duplicate frames.
1. **The stability matrix with the new shapes and a live Gemini response, plus the envelope.** Pass: forward loads hold, and the one-way rollback boundaries are stated.
1. **One Langfuse trace per framework through `harness.langfuse_export`.** Pass: names, usage, cached tokens, tool spans and session id match, or each difference is explained.
1. **A second turn under `request_limit=2`.** Pass: the budget counts only that turn.
