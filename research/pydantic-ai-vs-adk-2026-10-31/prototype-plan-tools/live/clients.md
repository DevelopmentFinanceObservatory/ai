# Outside client models on the plan tools

The planner's code works as MCP tools an outside model drives, and the client pays for every model call. On ten tasks against deployed dev data, Gemini 3.8 Flash landed an accepted plan as often as our own planner, and Claude Sonnet, Claude Haiku and gpt-6-luna came close. The clients' main obstacles were in our tools: the plan session does not see the client's own searches, and a run returns a summary instead of each year's rows.

## Results

Each client drove `start_plan`, `propose_plan`, `record_user_answer` and `run_plan` on plan-mcp beside the public `/mcp` tools, on the ten sample tasks in `tasks.py`. A simulated user (`simuser.py`) answered the client's questions from the task's reference and approved a plan only when it answered the question. Dev's planner, run three times per task through the planner-entry eval, is the baseline. `results.md` has the per-task tables.

| Client                        | How it ran                                                  | Runs | Plan accepted | Plan right | Right, without `sum-oda-oof-ethiopia` | Mean proposals |
| ----------------------------- | ----------------------------------------------------------- | ---- | ------------- | ---------- | ------------------------------------- | -------------- |
| Our planner, Gemini 3.8 Flash | planner-entry eval                                          | 30   | 29            | 24         | 24 of 27                              | 1.6            |
| Gemini 3.8 Flash              | Pydantic AI agent, both MCP servers as toolsets             | 20   | 20            | 15         | 15 of 18                              | 1.6            |
| Claude Sonnet                 | Claude Code agent through the `dfo` and `user` CLIs         | 10   | 10            | 6          | 6 of 9                                | 2.7            |
| Claude Haiku                  | Claude Code agent through the `dfo` and `user` CLIs         | 10   | 8             | 7          | 7 of 9                                | 1.8            |
| gpt-6-luna                    | Pydantic AI agent through the local relay, non-strict tools | 10   | 10            | 8          | 8 of 9                                | 3.1            |

"Right" applies the eval's own kind and year rules to the plan the client landed. `sum-oda-oof-ethiopia` is wrong in every arm, because live search now binds the official-donor totals its reference does not expect, so the second column of rates leaves it out. Claude ran one conversation per task, so a difference of one or two tasks between clients is within noise.

Haiku first ran under a tester prompt that forbade reading files. Claude Code saves a long command output, here the 54 KB `propose_plan` schema, to a file, so three of Haiku's first six runs stopped without seeing the schema. The table counts ten reruns under a prompt that lets the tester read that saved output, which is what a native MCP client receives. Sonnet ran under the first prompt and landed every plan regardless.

gpt-6-luna ran twice as a native client through the local relay (`client.py --model luna`). The first time, the relay sent tool schemas to OpenAI's Responses API without `strict: false`, and the Responses API makes function tools strict unless told otherwise. Luna then filled every optional field with an empty value: asked to look up Ghana with a tool whose `dcid` and `target_year` are optional, it sent `dcid: ""` and `target_year: 0` three times out of three. Every `search` was refused `request_invalid`, and none of the ten runs (`luna-*`) landed a plan. With the relay sending `strict: false`, the same probe sent only `{"name": "Ghana"}`, and Luna landed a plan on all ten tasks (`luna-nonstrict-*`, the row above). Its mean of 3.1 proposals includes ten on `sum-oda-oof-ethiopia`. The public tools should still treat an empty string or zero in an optional field as absent, since strict-mode clients send them.

## Cost

The client pays for every model call; we pay for validation, translation, compilation and data.

- A Gemini conversation used about 539,000 input tokens, 87 per cent from cache, and 19,000 output and thinking tokens, covering search, planning, the run and the reply.
- A Claude tester used about 109,000 tokens (Sonnet) and 140,000 (Haiku) per conversation, as Claude Code reports a subagent's use, which includes its own instructions.
- A gpt-6-luna conversation used about 513,000 input tokens, 90 per cent from cache, and 7,000 output and thinking tokens.
- Our planner's entry uses about 46,000 input tokens and 650 output tokens, and stops at the first accepted plan.

## What the clients needed that the tools lack

1. The plan session does not know what the client's searches found. `start_plan` answered "no search has run" after the client had searched, so pinning the dataset the search returned failed with `pinned_source_not_found` or `source_not_offered`. This cost Sonnet and Haiku extra rounds and stopped two Haiku runs. `start_plan` should take the spec ids the client found, or `search` should return a handle the plan session accepts.
1. `run_plan` returns frame references and a summary (first, latest, minimum, maximum), not each year's values, and `describe_frame` and `export_data` refused those references for lacking a `content_version`. Five runs answered with endpoints only or rebuilt the table through `get_data` and `write_component`. `run_plan` should return the rows of a small result frame, and frame references the public tools accept.
1. `record_user_answer` and `run_plan` take the client's word. One Gemini run recorded an answer the user never gave, and Luna recorded seven answers over four user turns on `sum-oda-oof-ethiopia`; plan-mcp accepted them all, and approval rests on the client as well. An answer carrying only an `id` did not clear `combination_unasked`, which cost Sonnet two rounds.
1. The `propose_plan` schema is about 35,000 characters, and refusals name only the fields they check. Smaller models spent rounds on `invalid_plan` before learning that an `answered` entry needs `ask` and `outputs`.
1. One plan card said "constant 2015 US dollars" for a source whose unit states no base year.
1. A translation referral answers `outcome: ok` with no card, which the scorer counts as `referral_pending`.

None of these depends on the agent framework.
