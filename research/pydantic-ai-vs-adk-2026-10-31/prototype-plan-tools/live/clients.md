# Outside client models on the plan tools

The planner's code works as MCP tools an outside model drives, and the client pays for every model call. On the second version of the tools, every client landed an accepted plan in every conversation and ran it: 50 of 50 conversations with MCP elicitation and 10 of 10 without, against 29 of 30 entries for our own planner. Claude Sonnet's plans were right in 9 of 9 conversations on tasks with a current reference, up from 6 of 9 on the first version, in 1.5 proposals instead of 2.7. No client needed `get_data` to read a run's figures, and every approval in a conversation with elicitation came from the user.

## Results

Each client drove `start_plan`, `propose_plan`, `record_user_answer` and `run_plan` on plan-mcp beside the public `/mcp` tools, on the ten sample tasks in `tasks.py`. A simulated user (`simuser.py`) answered from the task's reference and approved a plan only when it answered the question. Dev's planner, run three times per task through the planner-entry eval, is the baseline. `results.md` has the first version's per-task tables and `results-v2.md` the second's.

| Client                        | Tools, how it ran                            | Runs | Plan accepted | Plan right | Right, without `sum-oda-oof-ethiopia` | Mean proposals |
| ----------------------------- | -------------------------------------------- | ---- | ------------- | ---------- | ------------------------------------- | -------------- |
| Our planner, Gemini 3.8 Flash | planner-entry eval                           | 30   | 29            | 24         | 24 of 27                              | 1.6            |
| Gemini 3.8 Flash              | first version, Pydantic AI                   | 20   | 20            | 15         | 15 of 18                              | 1.6            |
| Gemini 3.8 Flash              | second version, Pydantic AI, elicitation     | 20   | 20            | 15         | 15 of 18                              | 1.6            |
| Gemini 3.8 Flash              | second version, Pydantic AI, no elicitation  | 10   | 10            | 6          | 6 of 9                                | 1.5            |
| gpt-6-luna                    | first version, Pydantic AI, non-strict tools | 10   | 10            | 8          | 8 of 9                                | 3.1            |
| gpt-6-luna                    | second version, Pydantic AI, elicitation     | 10   | 10            | 7          | 7 of 9                                | 2.1            |
| Claude Sonnet                 | first version, Claude Code through the CLIs  | 10   | 10            | 6          | 6 of 9                                | 2.7            |
| Claude Sonnet                 | second version, Claude Code, elicitation     | 10   | 10            | 9          | 9 of 9                                | 1.5            |
| Claude Haiku                  | first version, Claude Code through the CLIs  | 10   | 8             | 7          | 7 of 9                                | 1.8            |
| Claude Haiku                  | second version, Claude Code, elicitation     | 10   | 10            | 7          | 7 of 9                                | 2.4            |

"Right" applies the eval's own kind and year rules to the plan the client landed. `sum-oda-oof-ethiopia` is wrong in every arm, because live search now binds the official-donor totals its reference does not expect. Gemini and gpt-6-luna miss `quotient-oda-per-person-ethiopia` in both versions for another reason: each pins a published per-capita ODA series and plans one dataset, where the task's reference accepts only a computed quotient. Claude ran one conversation per task, so a difference of one or two tasks between rows is within noise.

## What the second version changed

- Searches reach the plan session. Clients passed the spec_ids their searches returned to `start_plan`. Proposals met `pinned_source_not_found` in 6 of 60 conversations, against 11 of 50 on the first version, and `source_not_offered` in 1, against 3. Fifteen conversations met `spec_not_held` once, on a spec_id that search had minted and no `get_data` had yet registered: dfo-apis' registry learns a definition only at its first fetch, and the rig's plan-mcp ran in its own process with its own definition store. Every one recovered by passing the search itself as `searches`. `serve.install_shared_stores` now installs the harness's definition store when `DEFINITION_STORE_BUCKET` is set, and the same never-fetched spec_id then opens a session. Mounted on `/mcp`, plan-mcp shares the harness's store.
- `run_plan` returns the data. Calls to `get_data` per conversation fell from 0.7 to 0.3 for Gemini, from 1.0 to none for gpt-6-luna, and from 0.4 to none for each Claude model; Claude's calls to `describe_frame` fell to none. Sonnet's conversations took 6 tool calls on average, down from 9.
- The user answers and approves directly. With elicitation, every approval was the user's: 20 of 20 for Gemini, 10 of 10 for gpt-6-luna, 10 for Sonnet and 11 for Haiku. The answers the planner asked for went to the user too, 4 in Gemini's conversations and 3 in gpt-6-luna's. Without elicitation, Gemini approved its own 10 plans and recorded one answer on its own word, which plan-mcp records as `client`.
- Refusals name the shape they expect. A call missing a required argument of `start_plan`, `record_user_answer` or `run_plan` is refused `invalid_arguments` with the argument's schema; before, plan-mcp answered a bare `KeyError`.

## Cost

The client pays for every model call; we pay for validation, translation, compilation and data.

- A Gemini conversation used about 508,000 input tokens with elicitation, 87% from cache, and 18,600 output and thinking tokens, against 539,000 and 19,200 on the first version.
- A gpt-6-luna conversation used about 439,000 input tokens, 92% from cache, and 7,000 output tokens, against 513,000 and 7,200.
- A Claude tester used about 128,000 tokens (Sonnet) and 142,000 (Haiku) per conversation, as Claude Code reports a subagent's use, which includes its own instructions. Sonnet's rose from 109,000 because its second-version prompt lets it read the 54 KB schema output its shell saves to a file.
- Our planner's entry uses about 46,000 input tokens and 650 output tokens, and stops at the first accepted plan.

A client's cost follows its number of model requests, each carrying the `propose_plan` schema and the planner's guidance. Returning rows removed fetches, but not those requests.

## Rig faults met on the rerun

- The simulated user's elicitation reply gave `approve` a default of `false`, so the small model that plays the user turned 13 approvals into declines by leaving the field out. Its reply shapes now require every answer field; on six right and six wrong cards it approved and declined all twelve as its script says. The ten affected conversations ran again; their first logs are in `runs-void-simuser-default/`.
- The harness and plan-mcp ran as two processes, each with in-memory stores, so `run_plan` read no rows of a fetched frame and `start_plan` held no unregistered spec_id. A local GCS emulator now gives both one bucket (`NOTES.md`, "Starting the servers").
- With ten Claude testers searching at once, every Haiku conversation's first `search` ran past the harness's 45 s limit and succeeded when repeated.
