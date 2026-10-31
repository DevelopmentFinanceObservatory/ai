# B1: the planner entry on Pydantic AI

The served planner entry runs on a Pydantic AI `Agent` over `GoogleModel` (pydantic-ai-slim 2.64.0, google-genai 2.34.0, google-adk 2.10.0 in one env). Offline parity (pass line W1) passes: across all 62 tasks, every one of the 78 task plans ends the same way through both entries, and the first request matches ADK's on system instruction, contents and session state. Every remaining wire difference is Pydantic AI's own and is listed below with its source line. Live smoke ran 2 tasks, 1 repetition, in both modes; every record reads cleanly through `summarise.py`.

## Files

- `pai_planner.py`: the agent. It holds the instruction and tool declarations read off the ADK agent the seam builds (`PlannerParts`), one Pydantic AI tool per ADK tool running the ADK tool object (`planner_tools`), the ADK session events rebuilt from Pydantic AI messages for the readers (`adk_events`), the thought filter (`drop_unsigned_thoughts`), the client wrapper that applies the turn budget and after-model callbacks (`PlannerClient`), and the model wrapper that applies the before-model callback and the prefix cache (`PlannerModel`).
- `pai_entry.py`: the entry (`run_pai_entry`) and the record (`pai_record`, the interface B2's driver calls). It also has an optional `rep` keyword, default 1.
- `parity.py`: the offline parity check. It writes `parity.json`.
- `smoke.py`: the live smoke. It writes `smoke/records.jsonl` and `smoke/summary.md`.

## Commands

From the worktree root:

```
uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness python research/pai-prototype/gemini/parity.py
uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness python research/pai-prototype/gemini/parity.py --only <ids> --break <name> --out <file>
uv run --env-file .env --with 'pydantic-ai-slim[google]==2.64.0' --package harness python research/pai-prototype/gemini/smoke.py
```

`parity.py` exits 1 unless every difference is explained. It takes about 35 s. The live runs need `--env-file .env`, as `just eval-planner-entry` does, since google-genai reads `GOOGLE_API_KEY` from the environment.

## How the entry is built

`run_pai_entry` follows `entry.run_entry` step for step. `entry.orchestrator_state` seeds the state from the recorded searches and hand-off. The graph's own node functions then run over a plain state dict: `route`, `plan_start` and `planner_node(...)`'s `draft_plan`, with the planner wrapped as the eval wraps it (`handing_back_planner_failures(probe.watch(seam))`). The real `PlannerSeam.__call__` therefore seeds the state, renders the brief, sets the tracker and budgets, and turns the outcome into a refusal or a plan. Only `ctx.run_node` is replaced (`PaiNodeContext.run_node`), and it runs `Agent.iter` with every model request streamed through `node.stream`. Each request gets a fresh `TurnBudget`. An ask the tool takes ends the run with `DeferredToolRequests`, which raises `PausedForUser`, a `BaseException`, as ADK's `NodeInterruptedError` is. A resume answers the ask through `DeferredToolResults` and re-runs only the planner node, with `ctx.resume_inputs` set. The record comes from `entry.entry_record` over ADK `Event`s rebuilt from the messages. `pai_record`'s experiment fingerprint is `run.experiment` with the mode, the Pydantic AI version and the bytes of `pai_planner.py` and `pai_entry.py` folded in.

## Parity table

Reproduced:

| Production behaviour                                                                                                                  | How                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The brief, state seeding, round reset, `fresh` flag, pin sources, asks                                                                | `PlannerSeam.__call__` itself runs                                                                                                                           |
| `route`, `plan_start`, `planner` nodes                                                                                                | the production node functions run over the state dict                                                                                                        |
| Instruction                                                                                                                           | `seam._agent.instruction`, sent as Pydantic AI `instructions`                                                                                                |
| Declarations                                                                                                                          | `tool._get_declaration()` of each ADK tool, given to `Tool.from_schema` verbatim, with the Gemini JSON schema transformer off                                |
| Tool argument checks, `ask_user` validation, `propose_plan` with first-year reads and the eval's translate                            | the ADK tool's own `run_async` runs, with a `ToolContext` stand-in                                                                                           |
| `_on_planner_tool_error`, `_after_planner_tool` (progress, terminal refusals, stalled referrals, the user-decided ask, `_MAX_EVENTS`) | called in the tool body where ADK's tool caller calls them; `PlannerSeamError` propagates as in ADK                                                          |
| Correction rounds and the round budget                                                                                                | `submission.service.submit` and the seam's tracker, both production code                                                                                     |
| `plan_not_ready`, `gave_up`, a turn-budget refusal                                                                                    | the seam's `_propose`, `probe.watch` and `handing_back_planner_failures`                                                                                     |
| Before-model synthetic responses (stop text, user-decided ask)                                                                        | `PlannerModel` asks `seam._before_planner_model` first; the client answers with that turn and nothing reaches the model or the budget                        |
| After-model: usage log, budget-closed ask or refusal                                                                                  | `PlannerClient` calls the probe's and the seam's after-model callbacks on the whole answer, and on a closing response                                        |
| The four event readers (`_elected_spec_ids`, `_answers_since_referral`, `_combination_picks`, `_user_spoke_since_recorded`)           | the production functions run over `adk_events(ctx.messages)`                                                                                                 |
| User message and brief as two user contents                                                                                           | the user's message carries its own `instructions` note (`QUESTION_INSTRUCTIONS`), which stops Pydantic AI merging the two requests                           |
| Thinking LOW with summaries; summary text never replayed, signed parts kept                                                           | `google_thinking_config`; `ProcessHistory(drop_unsigned_thoughts)`                                                                                           |
| Turn budget: 30 calls, 600 s wall, 90 s model time, 12 s first-chunk resend                                                           | `harness.turn_budget.bounded` wraps every `generate_content_stream` call, as `BoundedGemini` does                                                            |
| Explicit cache, lookup outside the budget clock, resend once on 403 or 404 before any chunk                                           | `PrefixCache.cache_name` in `PlannerModel`, sent with `google_cached_content`; a `ModelHTTPError` 403/404 at stream open forgets, looks up again and resends |
| HTTP retries                                                                                                                          | client built with `_retrying(...).retry_options` (3 attempts, 1 s to 2 s)                                                                                    |
| Streaming                                                                                                                             | every model request streams                                                                                                                                  |
| `skip_summarization` ends the run with no further model call                                                                          | `PaiRunner.run` stops before the next model request. The eval never reaches it, since its translate step stops first, so this path is untested               |

Not reproduced:

| Gap                                                                                                                                                                                                                                 | Tasks it could affect                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| A call to a tool name the planner lacks. ADK answers a tool-not-found response listing the tools. Pydantic AI answers a retry prompt and, past its one retry, raises `UnexpectedModelBehavior`, which the entry records as an error | any task where the live model guesses a tool name; none did in parity or smoke |
| A budget cut mid-stream. ADK keeps only the closing sentence in history. Pydantic AI keeps the chunks already streamed, so a function call delivered before the cut would run                                                       | runs with `budget_closed` set; none in smoke                                   |
| A response with no parts. Not checked whether Pydantic AI retries where ADK ends the turn                                                                                                                                           | unknown                                                                        |
| `_check_branches`                                                                                                                                                                                                                   | holds trivially; Pydantic AI has no branches                                   |

## Wire differences that are Pydantic AI's own

`parity.py` checks each against its exact form (`PAI_BEHAVIOUR`, `explain`), so a new difference shows as unexplained. Source lines are in `pydantic_ai_slim/pydantic_ai/models/google.py` at 2.64.0.

| Difference                                                                                                         | Source                           | Can a public setting close it?                                                                            |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Each function declaration goes in a `Tool` of its own; ADK sends one `Tool` holding all five                       | 1094                             | no                                                                                                        |
| `reject_plan` goes with `parameters_json_schema: {"type": "object", "properties": {}}`; ADK declares no parameters | 2523                             | no                                                                                                        |
| A function calling config is always sent: `AUTO` in like-adk mode, `VALIDATED` in validated mode; ADK sends none   | 1053-1074                        | the mode only; like-adk turns off `google_supports_strict_tool_definition` so the uncached mode is `AUTO` |
| `response_modalities: ["TEXT"]`                                                                                    | 1250                             | no                                                                                                        |
| Its own `Content-Type` and `User-Agent` headers; ADK sends its tracking headers                                    | 1256                             | no                                                                                                        |
| The system instruction goes as a `Content`; ADK's goes as a string                                                 | 1510                             | no                                                                                                        |
| Replayed function calls and responses carry Pydantic AI's tool call id; ADK strips its client ids                  | 2141, 1557                       | no                                                                                                        |
| The prefix key differs, so the ADK and PAI arms never share a cache                                                | follows from 1094, 1510 and 2523 | no                                                                                                        |

The coordinator's three items, as found:

1. One user content with two parts. Pydantic AI merges consecutive requests (`pydantic_ai/messages.py`, `_merge_consecutive_messages`, line 5356). Closed: it leaves apart two requests whose `instructions` both are set and differ, and the agent's instruction is what it sends, so the user's message request carries a note of its own. Contents now match ADK's on every task. B2's tap ran before this change.
1. The extra 309-character sentence on `<system>` tags. `Model.prepare_messages` adds `SYSTEM_VOICE_ANNOUNCEMENT` (`pydantic_ai/_framing.py:76`, added at `models/__init__.py:1866-1876`) to every request with instructions when the profile's `supports_inline_system_prompts` is false. Closed with the profile setting `supports_inline_system_prompts=True`. The same flag also stops the escaping of `<system>` in user prompts and tool returns (`models/__init__.py:1777`). The planner sends no system prompt part, so nothing else changes.
1. One `Tool` per declaration. `google.py:1094`, with no setting. Recorded above.

Pydantic AI also strips trailing whitespace from instructions: a break that appended a space made no difference until it appended `" X"`. ADK's instruction ends in no whitespace, so this changes nothing here.

## Thought replay

By default Pydantic AI replays a thought summary as a `{"text": ..., "thought": true}` part on every later request (`google.py:2078-2085`). The signature from a streamed part sits on that part's `provider_details`. `drop_unsigned_thoughts` removes every `ThinkingPart` that has no signature, as `drop_thought_parts` does. In parity's thought check, both arms' second request carries no summary text and keeps the signature on the function call. With the filter off (`--break thoughts`), Pydantic AI's request carries the summary.

## Results

Parity, all 62 tasks (`parity.json`):

- outcomes: 78 of 78 plans end the same in both entries (stop reason, plan and kinds, tool calls, each proposal's outcome, code, kinds and arguments, model call count, budget, misses, error), and none raises a `validate.py` fault through the PAI entry. That includes the ask and resume of every plan with a first-proposal refusal.
- first request: system instruction, contents, declaration order and session state at the call match on all 62 tasks. Declarations match on four tools; `reject_plan` differs as listed. Other differences are the ones in the table.
- prefix: like-adk prefix content matches once each side's wire form is normalised, except for `reject_plan`. The prefix keys differ on all 62 tasks.
- thoughts: equal.

Each check fails on a broken variant (`--break`, on `aggregate-average-oda-kenya` and `sum-net-flows-aid-debt-ghana`; exit 1 for every one):

| Break         | What it changes on the PAI side                 | Check that fails                                            |
| ------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| `instruction` | appends `" X"` to the instruction               | `first_request.system_instruction`, `prefix.prefix_content` |
| `declaration` | appends a space to `propose_plan`'s description | `first_request.declarations`, `prefix.prefix_content`       |
| `contents`    | drops the user message's instructions note      | `first_request.contents`                                    |
| `state`       | drops one seeded state key                      | `first_request.state`                                       |
| `outcome`     | the readers see no events                       | outcomes: 1 of 2 plans differ (the refusal task's resume)   |
| `thoughts`    | thought filter off                              | thoughts                                                    |
| `prefix`      | caches an `AUTO` tool config in like-adk mode   | `prefix.prefix_content`                                     |

Live smoke (`smoke/`, final code, one shared `PrefixCache`):

| Arm           | Task                         | Ended                                  | Calls | Cached tokens per call | Wall s |
| ------------- | ---------------------------- | -------------------------------------- | ----- | ---------------------- | ------ |
| pai-like-adk  | aggregate-average-oda-kenya  | accepted                               | 3     | 13,717                 | 11.2   |
| pai-validated | aggregate-average-oda-kenya  | accepted                               | 3     | 13,717                 | 12.9   |
| pai-like-adk  | sum-net-flows-aid-debt-ghana | asked_user after `combination_unasked` | 4     | 13,717                 | 13.5   |
| pai-validated | sum-net-flows-aid-debt-ghana | asked_user after `invalid_plan`        | 5     | 13,717                 | 17.1   |

Every call read the cache, signed function calls replayed without a 400, and no budget bound closed a call. An earlier smoke run of the same code before lint changes had one like-adk call think 6,777 tokens and take 51.8 s. One repetition says nothing about the arms; B2 runs the measurement.

## Deviations from the brief

- The scripted model in `parity.py` is a scripted google-genai client, not a `FunctionModel`. Both arms then run their real model class, and parity covers `GoogleModel` parsing, the budget wrapper and the cache path, which a `FunctionModel` would skip.
- The first request is captured at `client.aio.models.generate_content_stream` on both arms, not through the entry's `edits` hook. That is the request each framework actually sends, after ADK's own preprocessing.
- The state comparison drops `observed_at`, the wall-clock time the replayed search stamps on each reading. It differs between any two runs.
- `PlannerModel` builds the uncached request with `GoogleModel._build_content_and_config` to get the prefix, then sends with `google_cached_content`. Pydantic AI warns on every cached request that it stripped the prefix; `pai_planner` filters that warning.
- In like-adk mode, a request that goes uncached (the cache create failed) states `AUTO`, which is what ADK's absent tool config means, not ADK's absent field.
- `run_pai_entry` takes `client`, `answers` and `keep_thoughts`, and `pai_record` takes `rep`, all optional, beside the fixed interface.

## Private names and monkeypatches

- `PlannerSeam._agent`, `_before_planner_model`, `_after_planner_model`, `_after_planner_tool`, `_on_planner_tool_error` (`harness.planner_seam`).
- `FunctionTool._get_declaration` (ADK), through `PlannerParts.of`.
- `orchestrator.agent._retrying`, for the retry options.
- `GoogleModel._build_content_and_config` (Pydantic AI), in `PlannerModel.uncached_config`.
- `validate._faults` and `validate.SCRIPTED_USAGE`, in `parity.py`.
- Monkeypatch, `pai_planner.PlannerClient`: replaces `client.aio.models.generate_content_stream` on its own client instance.
- Monkeypatches, `parity.py` only, restored after each use: `entry.build_seam` (points ADK's model at the scripted client through `Gemini.client`), `entry.Probe.before_model` (records the state at the first call), the scripted client's `generate_content_stream` and `caches.create`, and the `--break` patches.
- `parity.py` imports `google.adk.workflow.utils._workflow_hitl_utils` for the ask's name, as `validate.py` does.
