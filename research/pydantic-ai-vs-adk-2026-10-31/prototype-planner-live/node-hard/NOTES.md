# The planner node under budget cuts, refusals, restarts and tracing

## Verdict

`node/` plus `agent/` match dev's ADK planner on four of the nine hard cases as they stand: a first-chunk stall, a gone prefix cache, an ask beside a proposal, and a restart after an ask. They differ on the other five, and each difference is closed here in a subclass:

| Case                                              | As `node/` and `agent/` ship                                                                                                    | Closed by                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| A budget cut after a function call was streamed   | The tool runs and writes its state; on dev it does not run                                                                      | `HoldingClient`                                           |
| An unknown tool name, a blocked or empty response | Different text for the model; a second unknown call, an empty stream or a blocked response fails the turn with an `error` frame | `hardened_agent`, `HardRunner(swallow)`                   |
| A turn that passes between planners               | A turn that asked twice, with the first answer under ADK, loses that answer when Pydantic AI resumes it                         | `HardSeam(record_from="events")`                          |
| A Stop mid-planning                               | Tool writes made before the Stop are lost; dev keeps them                                                                       | `HardRunner(deltas)`                                      |
| Tracing                                           | No generation, tool or agent span                                                                                               | `Instrumentation`, `conversation_id`, usage on each chunk |

With the fixes, 56 tests pass (`test_hard.py` 44, `test_regression.py` 12) twice in a row, about 28 s a run. Each case has a control, the same check on an arm with one fix off, that fails it. `test_regression.py` runs `node/`'s own pass lines again on the fixed planner and they hold.

Nothing in `agent/` or `node/` was edited. "Changes wanted upstream" lists what to move into them.

## What the harness does

`hard.py` runs the real Studio turn on either arm as `node/` does, with two changes: the planner's model is a `chunked.Chunked` client, and the app's `Settings` carry the request bounds the case needs (`turn_model_time_s`, `model_first_chunk_s`, `turn_model_calls_max`). `Chunked` answers each call from a `Call`: chunks with gaps, a wait that never ends before the first chunk or after the last, a refusal with a status code, a finish reason, an empty stream. Both arms talk to it at `client.aio.models.generate_content_stream`.

`assert_ends_alike` compares what a turn leaves: the frames (`node/`'s `assert_same_frames`), the number of requests the planner sent, the last request the orchestrator received (what the planner's end told it), and the stored session state, apart from `planner:conversation` and the call ids each framework mints.

## Pass lines

### 1. A budget cut after a function call was streamed

Dev: ADK's `Gemini` aggregates a streamed call and hands the flow a function call only once the stream ends. `bounded` wraps that aggregate, so a cut during a quiet stream drops the function call and the session keeps the closing sentence alone. No tool runs; the orchestrator is told "This request used all the time it allows the model to work before it finished, so it stopped here."

As shipped: `PlannerClient._stream` passes each chunk to Pydantic AI as it arrives. Pydantic AI holds the call, appends the closing sentence as a text part and runs the tool. The turn shows two more activity steps, and `translator:discovery`, `planner:pin_verdicts` and `translator:referral` appear in the stored state.

Fixed: `HoldingClient._stream` collects the chunks `bounded` yields and gives Pydantic AI the whole call, or the closing response alone when the budget cut it. Same frames, one request, same sentence, no tool state, stored state equal.

Test `test_a_cut_after_a_function_call_ends_as_dev_does_and_runs_no_tool`. Control `hold=False`: the tool state is present on the Pydantic AI arm and absent on dev's, so `assert_ends_alike` fails.

Cost: Pydantic AI sees no chunk of a call until `bounded` has finished with it. Every chunk still passes through `bounded` as it arrives, so the first-chunk deadline, the gaps and the model time are metered as before.

### 2. A first-chunk stall

Both arms send the same request twice and end on the plan. The second stall of a request is not resent: both arms wait under the model-time bound and end on "stopped here" after two requests. No change was needed, since `PlannerClient` runs `bounded` around each call and `bounded` owns the resend.

Tests `test_a_first_chunk_stall_is_resent_once`, `test_a_second_stall_is_not_resent_and_the_budget_ends_the_turn`. Control: `bounded` marks `retry_spent` before it runs, so the first stall is not resent. The Pydantic AI arm sends one request against dev's two.

### 3. A gone prefix cache

Dev's `PrefixCachedGemini` and `PlannerModel` both create a cache, send the request against it, get 403 (or 404), forget the cache, create another and resend once. Both arms record two cache creates and two requests naming `cachedContents/scripted-1` then `scripted-2`, and end on the plan. A second refusal ends both arms on an `error` frame with two requests. No change was needed. The caches hold one instruction text on both arms and are keyed apart, because the two arms declare `reject_plan` differently on the wire (`agent/NOTES.md`).

The resend works because `GoogleModel._process_streamed_response` reads the first chunk when the request stream opens and maps google-genai's `APIError` to `ModelHTTPError`, which `PlannerModel.request_stream` catches.

Tests `test_a_gone_cache_is_forgotten_looked_up_again_and_the_request_resent_once` (403 and 404), `test_a_cache_gone_twice_ends_the_turn_as_dev_does`. Control: an empty `_CACHE_GONE_CODES` leaves the first 403 uncaught; the Pydantic AI arm ends on an `error` frame after one request, dev's on the plan.

### 4. An unknown tool name, and output Pydantic AI refuses

What the model reads for a call to `no_such_tool`:

- Dev: `{"error": "Invoking `no_such_tool()` failed as no tool with that name is available. The tools you can call are: get_skill, get_skill_file, ask_user_question, propose_plan, reject_plan. You could retry, but it is IMPORTANT that you only call a tool from that list."}` (ADK's `build_tool_not_found_response`, in declaration order).
- Pydantic AI as shipped: `{"error": "Unknown tool name: 'no_such_tool'. Available tools: 'ask_user_question', 'get_skill', 'get_skill_file', 'propose_plan', 'reject_plan'"}` (sorted by name).
- Fixed: a history processor, `adk_tool_not_found`, gives the retried return of a name that is no tool ADK's text. The stored `planner:conversation` keeps Pydantic AI's.

How the turn ends when the model keeps calling it, with the call cap at 4:

- Dev: four requests, then the cap's sentence ("This request took as many model steps as it allows before it finished, so it stopped here"); the orchestrator answers.
- Pydantic AI as shipped: Pydantic AI charges each unknown name against `retries` (1) and raises `UnexpectedModelBehavior` on the second call; the turn ends on an `error` frame (`capability_unavailable`) after two requests.
- Fixed: `retries=1000`, so the call cap ends the run as on dev.

Output Pydantic AI refuses with `UnexpectedModelBehavior` (an empty stream; a response the provider blocked, finish reason `SAFETY`): dev ends the run as on any response with no function call, and the seam refuses with `plan_not_ready` ("No plan was made: the planner stopped before the plan was ready"). Pydantic AI as shipped fails the turn with an `error` frame. Fixed: `HardRunner` catches `UnexpectedModelBehavior` and ends the run with no plan; the seam's own refusal follows. A malformed function call and a response cut at the token limit already end as dev's do.

Tests `test_an_unknown_tool_name_is_answered_as_dev_answers_it`, `test_a_model_that_keeps_calling_an_unknown_tool_is_ended_by_the_call_cap`, `test_a_response_pydantic_ai_refuses_ends_the_run_as_a_response_with_no_call_does` (two inputs), `test_a_response_pydantic_ai_accepts_ends_as_dev_does_with_no_fix` (two inputs). Controls: `unknown_tools=False` gives the model Pydantic AI's text and ends the run after two requests with a different refusal; `swallow=False` fails the turn on both refused inputs.

### 5. An ask and a `propose_plan` in one response

Five shapes: ask then a proposal that writes nothing, a proposal that writes nothing then an ask, a skill read beside a proposal, an accepted proposal then an ask, an ask then an accepted proposal. On each, both arms end on the same frame with the same requests and the same state, and the tool bodies start and end in the same order, each with the same keys written. An accepted proposal beside an ask overlaps: the ask starts, writes `planner:pending_ask` and ends inside the proposal's translation, on both arms. The plan card wins on both. No change was needed.

Tests `test_an_ask_beside_a_proposal_ends_as_dev_does_in_the_same_order_of_writes`, `test_the_two_tools_of_one_response_overlap_as_they_do_on_dev`. Control: tools marked `sequential` run the proposal to its end before the ask starts; the order differs from dev's.

### 6. Resume from fresh objects after an ask

A turn with two asks, each answered on a new app (new `Runner`, `StudioStore`, seam, planner and orchestrator) over the same session service, ends as one app runs it: the same frames, every planner request equal, the same stored state, on both arms. The record survives on the Pydantic AI arm because it is in session state.

Test `test_a_turn_resumed_on_fresh_apps_ends_as_one_app_runs_it`. Control: a record kept on the seam object, as `agent/`'s `PaiSession` keeps it, is gone on the new app; the first answer is lost and the last proposal is refused.

### 7. A pause under one planner, answered under the other

The turn: refused proposal, ask 1 (combination), answer, ask 2 (years), answer, proposal that needs the combination answer. Each segment runs on its own app over one session service. As shipped:

| Planner of each segment (first ask, second ask, last run) | Result                                               |
| --------------------------------------------------------- | ---------------------------------------------------- |
| dev's, dev's, Pydantic AI                                 | No plan: the last run does not know the first answer |
| Pydantic AI, Pydantic AI, dev's                           | Same plan as dev                                     |
| Pydantic AI, dev's, Pydantic AI                           | No plan: the stored record is stale                  |
| dev's, Pydantic AI, dev's                                 | Same plan as dev                                     |

A single ask answered on the other path lands a plan on every pair, because the answer step alone carries the pick.

Fixed: the Pydantic AI path reads its earlier steps from the planner events in the session (`record_from_events`, over `planner.agent._session_steps`), which both planners append. All four orders land dev's plan with equal frames, requests and state.

What each path must read of the other's record:

- Rollback to ADK reads the planner events the node yields: the ask's call and its accepted return under the call's id, the answer, and every tool's response. The node must keep yielding all of them (`feed_tools=None`); `node/`'s control shows what a revision reads without the ask's call. It also reads the state keys the tools wrote (`planner:pending_ask`, skill pages, `translator:*`), which `mirror_changes` and the response deltas make durable.
- Deploy to Pydantic AI reads ADK's planner events the same way. The node's own `planner:conversation` is no source of truth: it is absent for a turn ADK started and stale for a turn ADK continued.

Tests `test_a_turn_that_passes_between_planners_lands_the_plan_dev_lands` (four orders). Control `record_from="stored"`, `node/`'s record: the first and third orders above fail, with no plan and three requests.

### 8. Stop mid-planning

A skill page is read, then the Stop arrives while `propose_plan` translates. Stored session, by arm:

| Arm                    | `planner:skills_read`, `planner:skill_page:oda-dac2a::0` | `propose_plan`'s writes |
| ---------------------- | -------------------------------------------------------- | ----------------------- |
| Dev                    | kept (the response event carried them)                   | lost                    |
| Pydantic AI as shipped | lost (copied into `ctx.state` when the seam call ends)   | lost                    |
| Pydantic AI, fixed     | kept                                                     | lost                    |

Fixed: `HardRunner` puts every key the tools of a response changed in that response event's `state_delta`, so ADK stores them as it appends the event. The stored state after a Stop is then equal to dev's.

The next turn behaves the same with or without the fix: each turn has a session of its own (`{conversation}#t{n}`), so nothing reads the stopped turn's session. Frames and every planner request of the next turn are equal on both arms. The lost writes matter only to a reader of the stopped session itself.

Tests `test_a_stop_keeps_the_tool_writes_dev_keeps`, `test_the_turn_after_a_stop_is_the_same_on_both_arms`. Control `deltas=False`: the skill keys are absent after the Stop.

### 9. Tracing

The planner's spans for a turn that reads a skill page and proposes, in an in-memory exporter (the harness's global provider, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`), and on the wire through the real Langfuse exporter against a stand-in server.

As shipped the Pydantic AI arm opens none of them. `bounded`'s `dfo.model.*` attributes land on the `invoke_node dfo_planner` span the node opens, where each call overwrites the last.

| Span                                         | Dev (scope `gcp.vertex.agent`)                                                             | Pydantic AI, fixed (scope `pydantic-ai`)                        |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Agent run                                    | `invoke_agent dfo_planner`                                                                 | `invoke_agent dfo_planner`                                      |
| Model call                                   | `generate_content gemini-3.8-flash`, two                                                   | `chat gemini-3.8-flash`, two; `gen_ai.operation.name` is `chat` |
| Tool call                                    | `execute_tool get_skill`, `execute_tool propose_plan`                                      | the same names                                                  |
| Model, agent, session                        | `gen_ai.request.model`, `gen_ai.agent.name`, `gen_ai.conversation.id` = the ADK session id | equal (with `conversation_id=` on the run)                      |
| Usage                                        | `gen_ai.usage.input_tokens`, `output_tokens`, `cache_read.input_tokens`                    | equal                                                           |
| `dfo.model.*`                                | `first_chunk_s`, `thoughts_token_count` on the model call span                             | equal (with usage on each chunk)                                |
| Langfuse `session.id`, `langfuse.trace.name` | the conversation id, `turn`                                                                | equal: the turn's context reaches Pydantic AI's spans           |

Not matched, pinned exactly in the tests:

- Names and provider: `chat` for `generate_content`; `gen_ai.provider.name` is `google`, not `gcp.gemini`. The span carries `gen_ai.*` attributes, and the wire test shows the exporter sends it.
- Missing on Pydantic AI: `gcp.vertex.agent.event_id` and `invocation_id`; `gen_ai.usage.reasoning.output_tokens` (Pydantic AI writes `gen_ai.usage.details.thoughts_tokens`); on tool spans `gen_ai.tool.description` and `gen_ai.tool.type`; on the run `gen_ai.agent.description`.
- Extra on Pydantic AI: `model_request_parameters`, `operation.cost`, `pydantic_ai.cache.*`, `gen_ai.client.operation.time_to_first_chunk`, `logfire.*`, `gen_ai.aggregated_usage.*` and `pydantic_ai.all_messages` on the run.

What it takes to match closer: rename `chat` to `generate_content` and set the operation and provider attributes in a span processor that updates the name on start (Pydantic AI has no setting for either that this run found), and add the missing tool attributes the same way.

Tests `test_the_planner_opens_the_spans_dev_opens_apart_from_the_pinned_differences`, `test_the_langfuse_session_and_trace_name_reach_the_planners_spans_on_both_arms`. Controls: no `Instrumentation` (no planner span, in memory or on the wire); no `conversation_id` (`gen_ai.conversation.id` is a UUID); a client that drops each chunk's usage (`dfo.model.thoughts_token_count` is absent).

## Changes wanted upstream

In `research/pai-planner-live/agent/pai_planner.py`:

- `PlannerClient._stream`: collect the chunks `bounded` yields and yield them after it returns, or the closing response alone when `CLOSED_BY_KEY` was seen (`HoldingClient._stream`).
- `PlannerClient._raw`: pass `usage_metadata=chunk.usage_metadata` to the `LlmResponse` it yields.
- `planner_agent`: `retries=1000`; a second `ProcessHistory` that gives a retried return for a name that is no tool ADK's text (`adk_tool_not_found`); `Instrumentation()` among the capabilities.

In `research/pai-planner-live/agent/pai_entry.py`:

- `PaiRunner.run`: pass `conversation_id=session.id` to `Agent.iter`; catch `UnexpectedModelBehavior` and end the run with no plan.

In `research/pai-planner-live/node/pai_node.py`:

- `FeedingRunner.events`: the same two changes; and in `response_events`, take the keys the tools of the response changed (a diff of `deps.state` against the last response) into `EventActions.state_delta`.
- `PaiPlannerSeam._record`: read the record from `ctx.session.events` through `record_from_events`, always; stop writing `planner:conversation` and drop `mirror_changes` for tool writes (the deltas cover them; the seam's own end-of-call writes still need `ctx.state`).
- `pai_planner`: build the agent through the changed `planner_agent`.

## Internals relied on

Harness: `harness.turn_budget.bounded`, `CLOSED_BY_KEY`, `current_turn_budget`; `harness.prefix_cache.PrefixCache` and `PlannerSeam(prefix_cache=...)`; `PlannerSeam._after_planner_model`; `planner.agent._session_steps` and `load_pending_ask`; `tests.orchestrator_fakes`; `tests.telemetry.conftest.fake_langfuse` and `common.telemetry`. The workspace `conftest.py` provides the in-memory exporter.

ADK 2.13.0: `google.adk.flows.llm_flows.tools._error_handler.build_tool_not_found_response`; that `Gemini` aggregates a streamed call and yields a function call only at the end of the stream.

Pydantic AI 2.66.0: `Agent(retries=)`, `Agent.iter(conversation_id=)`, `capabilities.Instrumentation`, `ProcessHistory`, `Tool.sequential`, `UnexpectedModelBehavior`; that `GoogleModel._process_streamed_response` reads the first chunk and maps `APIError` to `ModelHTTPError`; that an unknown tool name is a `ModelRetry` charged against `retries`, and that the retried return is a `RetryPromptPart` or a `ToolReturnPart` with `retry` set.

From `agent/` and `node/`: `PlannerClient`, `_RAW_CHUNK`, `_as_chunk`, `_NO_REQUEST_LIMIT`, `PlannerModel`, `FeedingRunner`, `PaiPlannerSeam`, `response_events`, `call_events`, `pai_call.PaiCall._in_flight`, `studio_arms.build_arm` (its `sc`, `orchestrator_app`, `pai_planner`, `PlannerSeam` and `InMemorySessionService` names are patched), and `test_node`'s turn helpers.

## A production version

- The five fixes are an estimated 150 lines against `agent/` and `node/`: `HoldingClient` folds into `PlannerClient` (about 25 lines changed), the tool-name processor and `retries` are 20, the runner's three changes 30, the event record 40, the instrumentation line 1. They replace `planner:conversation` (76 KB of a 386 KB session state for a one-ask turn) and the `mirror_changes` copy.
- The per-response state diff deep-copies the session state at each model response. That was not measured. A production version would take the changed keys from the tool bodies' own writes (a recording wrapper on `call.state`) and skip the copy.
- The Pydantic AI path depends on a private ADK function for the record (`_session_steps`) and for the tool-not-found text. A translator that reads neutral planner events would give both a neutral source.
- The span names differ from dev's. A dashboard or alert keyed on `generate_content` needs a second name, or the span processor above.

## Not covered

- A budget cut after streamed text and before a function call: ADK emits the text as partial events, which the node does not; Studio shows none of the planner's text, so no frame differs here, but no test reads them.
- A cut whose closing response the seam turns into an ask (`_after_planner_model` with a referral that names sources). The hold path calls the same callbacks, but no test reaches it.
- Concurrent budget cuts and Stops in one turn; a cut during a resumed run.
- A planner response with several function calls where one is an unknown name.
- The cost of the per-response state diff, and live Gemini behaviour for any case: every input is scripted.
- Langfuse's rendering of the `chat` spans.
- Roll-forward after a rollback inside one turn beyond the four orders above.

## Files

- `chunked.py`: the scripted client with stalls, cuts, refusals, finish reasons.
- `hard.py`: `bounds`, `hard_arm`, `phases` (a turn in segments on separate apps), `trace_tools`, `session_of`.
- `hardened.py`: `HoldingClient`, `adk_tool_not_found`, `hardened_agent`, `HardRunner`, `record_from_events`, `HardSeam`, `hard_planner`. Each fix has a switch (`hold`, `unknown_tools`, `swallow`, `deltas`, `record_from`, `conversation`, `instrument`).
- `test_hard.py`: the nine cases and their controls.
- `test_regression.py`: `node/`'s pass lines on the fixed planner.

## Commands

From the worktree root:

```
cd research/pai-planner-live/node-hard
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
```

```
uv run --package harness ruff format research/pai-planner-live/node-hard
uv run --package harness ruff check research/pai-planner-live/node-hard
```
