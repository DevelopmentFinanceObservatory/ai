# Planner prototype: results

The prototype tested the architecture the README recommends, Pydantic AI agents under a turn loop we own, against the pass lines the README set. Everything ran on dev at afbf84ff with pydantic-ai-slim 2.64.0. The code, run notes and summaries are in `prototype/`.

## Verdict

The architecture passed every line, with one condition: the planner must run Gemini's function calling in `AUTO` mode, as ADK does, and not in `VALIDATED`, Pydantic AI's default. In `AUTO` mode the Pydantic AI planner matched the ADK planner on plan acceptance, cache share, first-chunk latency and thought signatures. `VALIDATED` added 0.6 s to the first chunk and 3 s to each planner entry, and one refusal task stopped ending as it should.

## The turn loop

`prototype/loop/` is the Studio turn as our own state machine over two Pydantic AI agents, an orchestrator and a planner, on scripted models with a SQLite store. Eleven tests pass. Each pass check was also run against a deliberately broken loop and failed there.

| Line                     | Test                                                                                                                      | Result                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1 live progress         | The planner's `propose_plan` emits a progress event through `ctx.emit`, then waits until the client has received it       | The frame arrives within 2 s while the tool is still running; a loop that buffers events until the run ends times out                                                     |
| L2 a pause loses nothing | The planner emits progress, is refused, asks the user and pauses                                                          | Every progress frame reaches the client in order before the question, and the stored frame log equals what the client got                                                 |
| L3 survives SIGKILL      | Pause on the planner's question; resume in a second process; SIGKILL it at the approval pause; approve in a third process | Orchestrator and planner histories equal an unpaused run's byte for byte, with timestamps removed and run ids replaced by their order of first appearance; same plan hash |
| L4 one resumer wins      | Two resumes of one approval, both reading the pause before either claims it                                               | Exactly one runs; the compare-and-swap refuses the other                                                                                                                  |
| L5 deploy-safe resume    | Pause under one loop; resume under a loop with a new opening step                                                         | The new step does not run and the turn completes                                                                                                                          |

A SIGKILL in the middle of `propose_plan` falls back to the last checkpoint, the start of that planner run. The interrupted call starts again and its progress frames are sent twice; no other tool body runs twice, and the finished turn equals the reference. The loop is 440 lines, the store 154.

## Stored conversations across releases

`prototype/stability/` wrote a paused planner `Conversation` (thought signatures, a tool call and return, usage, a pending deferred call) under each of the ten releases from 2.55.0 to 2.64.0 and read each one under every release. All 55 forward pairs and all 45 rollback pairs load, and every pause resumes with its signatures reaching the model. Each release only added fields: `history_digest` and two usage fields in 2.58.0, `retry` in 2.60.0, `citations` in 2.61.0, `canonical_system` in 2.63.0 and `model_id` in 2.64.0. A rollback drops the newer fields; the only populated value lost is `model_id`, which readers before 2.64.0 drop. The records were built on a scripted model, so fields only a live Gemini response fills went through the round trip empty.

## The planner on Gemini

`prototype/gemini/` runs the `evals/planner_entry` tasks with the planner's model loop on Pydantic AI. The production seam, brief, tools, state, prefix cache, turn budget and retries stay as they are; only the agent loop changes. This measures Pydantic AI on Gemini with our declaration. Moving the planner's tools off ADK's `ToolContext` remains untested.

Each of 62 tasks ran three times in each of three arms, 558 entries in all, with the arms interleaved in a seeded random order, four in flight, data-mcp answered from recordings and no errors. A tap on google-genai recorded every request and response the same way for both frameworks. Both frameworks ran on google-genai 2.34.0, which Pydantic AI requires, against 2.24.0 in the harness lock.

| Measure                                                         | ADK            | Pydantic AI, `AUTO`    | Pydantic AI, `VALIDATED` |
| --------------------------------------------------------------- | -------------- | ---------------------- | ------------------------ |
| First proposal accepted                                         | 129/177 (73%)  | 127/177 (72%)          | 125/177 (71%)            |
| Paired difference [95%]                                         |                | -1.1% [-6.8%, +5.1%]   | -2.3% [-9.6%, +5.1%]     |
| Kind pass, tasks                                                | 56/59          | 58/59                  | 57/59                    |
| Refusal tasks ending right                                      | 6/9            | 8/9                    | 6/9                      |
| Tasks falling from passing                                      |                | 0                      | 1                        |
| Cached share of prompt tokens, later calls                      | 84.7%          | 85.3%                  | 85.1%                    |
| First chunk p50, all calls                                      | 3.30 s         | 3.42 s (1.04×)         | 3.95 s (1.20×)           |
| First chunk p50, first call                                     | 2.09 s         | 2.16 s                 | 2.66 s                   |
| Wall time per entry, mean and p90                               | 11.4 s, 19.7 s | 11.1 s, 19.6 s         | 14.5 s, 24.1 s           |
| Model calls per entry                                           | 2.56           | 2.40                   | 2.48                     |
| Thinking tokens per entry                                       | 99             | 254 (+156 [-69, +434]) | 426 (+335 [+95, +644])   |
| Entries closed by the turn budget                               | 2              | 0                      | 2                        |
| Thought signatures lost on replay                               | 0 of 547       | 0 of 389               | 0 of 449                 |
| Cached requests also carrying instruction, tools or tool config | 0 of 485       | 0 of 452               | 0 of 468                 |

The task that fell under `VALIDATED` is `sum-oda-fdi-ethiopia`, a refusal task that ended correctly in 3 of 3 baseline runs and 1 of 3. The first call carried 61 thinking tokens on average under `VALIDATED` against 21 under ADK and 20 under `AUTO`, which accounts for the slower first chunk.

The `propose_plan` declaration and every other declaration reach Gemini byte for byte as ADK sends them, except `reject_plan`, which Pydantic AI sends with an empty parameters schema. The system instruction and the first request's contents are identical on all 62 tasks. Two Pydantic AI defaults had to be switched off to get there: it merges consecutive user requests into one content, which the prototype stops by giving the user's message its own instructions note, and it prepends a 309-character sentence about mid-conversation system messages to the system instruction, which `supports_inline_system_prompts=True` removes. Differences that no public setting closes: each declaration goes in its own `Tool`, a function-calling config is always sent, `response_modalities` is always sent, the headers differ, the system instruction goes as a `Content`, and tool call ids are replayed. None changed a measured outcome. The prototype built its cache prefix from Pydantic AI's request shape, so the two frameworks used different caches. A request that names a cache carries no instruction, tools or tool config, so a prefix built from the canonical declarations should let both share one cache; that is untested (`patterns-and-risks.md`).

Three ADK behaviours have no Pydantic AI equivalent in the port. A call to a tool the planner lacks gets a retry prompt and then raises `UnexpectedModelBehavior`, where ADK answers with the list of tools. A turn-budget cut mid-stream keeps function calls already streamed, where ADK keeps only the closing sentence. An empty model response was not checked. None of the three occurred in the 372 Pydantic AI entries.

## Still open

- The planner's tools read the user's answers from ADK session events and write ADK session state, about 500 lines in `planner/agent.py`. The prototype kept them behind a stand-in `ToolContext`.
- Live progress was tested on scripted models. The real translator inside `propose_plan` has not emitted through `ctx.emit` to the Studio stream.
- Whether turns must outlive the client: the share of turns on deployed dev that run past 60 s or end in a disconnect is still unmeasured.
- Conversation records holding fields only a live Gemini response fills have not been round-tripped across releases.
