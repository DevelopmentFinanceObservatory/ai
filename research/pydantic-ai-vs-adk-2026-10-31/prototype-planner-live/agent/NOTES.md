# The planner on Pydantic AI, under the real seam

The planner runs as a Pydantic AI `Agent` (pydantic-ai-slim 2.66.0, google-genai 2.34.0, google-adk 2.13.0) over `GoogleModel`, in `AUTO` function-calling mode. Its tools call the neutral bodies in `planner/tools.py` and `planner/skills.py` with a `PaiCall` and an end-run signal. No ADK tool object runs, no `ToolContext` stand-in reaches a tool body, and no ADK event is built from messages for a tool to read. The seam's own tool callbacks are typed on ADK's `BaseTool` and `ToolContext`, so they receive the ADK tool object for its name and a small stand-in (see Private names). `PlannerSeam.__call__` runs unchanged; only `ctx.run_node` runs the Pydantic AI agent instead.

## Files

- `pai_call.py`: `PaiCall`, the `PlannerCall` over the steps of earlier runs and the live run's messages. `RecordingState` is the session state a tool body writes through, noting each key written. `user_message` and `answer_message` build the two requests an entry's record holds besides the run's own, and `steps_of` reads a record's steps.
- `pai_planner.py`: the agent. `PlannerParts` reads the instruction and each declaration off the seam's ADK agent. `planner_tools` declares one Pydantic AI tool per declaration, and `run_tool` maps each to its neutral body. `PlannerModel` and `PlannerClient` apply the seam's before-model callback, the prefix cache, the turn budget and the after-model callbacks. `drop_unsigned_thoughts` is the thought filter and `adk_tool_not_found` answers a call to a tool the planner lacks. `PaiSession` and `PlannerDeps` hold what a run reads and writes.
- `pai_entry.py`: `run_pai_entry`, the equivalent of `entry.run_entry`. `PaiRunner` runs `Agent.iter` and stops it before the next model request when a tool ended the run, or when Pydantic AI refuses a response. `PaiNodeContext` is the `ctx` the graph's planner node and the seam read.
- `scripted.py`: the scripted google-genai client, the ADK arm, and the request comparison.
- `test_parity.py`: 18 tests.

## How an entry runs

`run_pai_entry` seeds the state from the recording, runs the graph's `route`, `plan_start` and `planner` node functions over a plain state dict, under a fresh `TurnBudget` and the conversation's recorded identity, as `entry.run_entry` does.

A run starts from the user's message and the brief the seam leaves in state, and sees nothing of an earlier run, as ADK's `single_turn` run reads only its own node path. `PaiSession.record` holds every message of every run in order, with the user's answer joined before the run it resumes. A tool call's `PaiCall` gives the steps of that record (`steps_of`), or the steps `PaiSession.earlier` holds when a log of earlier runs exists (the node gives the ADK session's planner events), then the steps of the live run's own messages. So `elected_spec_ids`, `answers_since_referral`, `prices_chosen` and `_ask_beside_ask` see the session as ADK's events show it. Tool bodies write session state through `RecordingState`, which notes each key written in `PlannerDeps.written`; `PlannerDeps.drain_writes` returns those keys and values (the node puts them on a response event's `state_delta`).

An accepted ask or a plan declared ready calls `PlannerDeps.end_run`. `PaiRunner.run` sees the flag at the next `ModelRequestNode`, keeps that node's request (it carries the tool returns) and stops. The seam returns the ask; the entry reads it off state, appends the answer, sets `ctx.resume_inputs` to the answer's `result` (what ADK resumes a node with) and runs the node again in a request of its own.

## What passes

`pytest .` runs 18 tests in about 3 s and passes twice in a row. Each runs one script through both arms and compares how the entry ended (`entry.entry_record`'s stop reason, plan, tool calls, each proposal, each tool response, model call count, budget, misses, error) and every request each arm sent.

| Scenario                                                                                   | Requests per arm | Result                                                                                                                                                |
| ------------------------------------------------------------------------------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aggregate-average-oda-kenya`, one proposal                                                | 1                | accepted, same plan; system instruction, contents, five declarations equal                                                                            |
| `sum-net-flows-aid-debt-ghana`: refused proposal, ask, answer, proposal                    | 3                | accepted; no model call between the ask and the answer; the third request is the user's message and a brief with the answer, no call of the first run |
| a refused ask (`recommended_spec_id` naming no candidate), then a proposal                 | 2                | the same `invalid_arguments` problem returned as the ask's response; the run goes on                                                                  |
| two asks in one response                                                                   | 1                | first accepted (`asked_user`), second refused `invalid_arguments`                                                                                     |
| a plan declared ready (production translation over the recording, `summary` in the answer) | 1                | the run ends with no further model call                                                                                                               |
| `get_skill` and `get_skill_file`, then a proposal                                          | 3                | same page text; both pages on record in `planner:skills_read`                                                                                         |
| a turn with an unsigned thought summary and a signed call                                  | 2                | the summary is not replayed and the signature is kept, as on the ADK arm                                                                              |
| the cached prefix (`PrefixCache`)                                                          | 1                | one cache each, same instruction and declarations, no tool config; the request carries the cache name and no prefix                                   |

Every request differs from ADK's only by what `GoogleModel` writes for the same content (`WIRE` in the tests, asserted exactly, so a new difference and a closed one both fail):

| Difference                                                                                                                       | Source                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Each declaration is a `Tool` of its own; ADK sends one `Tool` holding all five                                                   | `pydantic_ai/models/google.py:1097`            |
| `reject_plan` carries `parameters_json_schema: {"type": "object", "properties": {}}`; ADK declares no parameters                 | `google.py:2534`                               |
| A tool config `AUTO` is sent; ADK sends none                                                                                     | `google.py:1062-1077`                          |
| `response_modalities: ["TEXT"]`                                                                                                  | `google.py:1304`                               |
| `automatic_continuation` is absent; ADK sets it `True`, which is google-genai's default (`should_enable_automatic_continuation`) | `GoogleModel` has no setting for it            |
| `Content-Type` and `User-Agent` headers of its own; ADK sends its tracking headers                                               | `google.py` `http_options`                     |
| Replayed function calls and responses carry Pydantic AI's tool call id; ADK sends none                                           | normalised out of `contents` in the comparison |
| A replayed function call with no signature gets the `skip_thought_signature_validator` placeholder; ADK sends none               | the scripted turns are signed, as Gemini's are |

Two differences the tests found and the code closes: a tool result that is no dict (`get_skill`'s text) goes to the model as `{"return_value": ...}` under Pydantic AI and `{"result": ...}` under ADK, so `_body` wraps it as ADK does; and a resume value is the answer's `result`, not the whole response, so the brief of the resumed run shows the pick.

## Behaviour at the edges

Each is proved against dev's ADK planner in `../node-hard/`, which runs the node on a scripted client that stalls, goes quiet, refuses and ends.

- A call the budget cuts after its function call was streamed runs no tool. `PlannerClient._stream` passes every chunk through `bounded` as it arrives and hands Pydantic AI the chunks once `bounded` returns, or the closing response alone when the budget cut the call, as ADK's `Gemini` hands its consumer a function call only when the stream ends.
- Usage rides on each chunk (`PlannerClient._raw`), so `bounded` stamps `dfo.model.thoughts_token_count` on the call's span.
- A call to a tool the planner lacks is answered with ADK's text (`tool_not_found`, written out in `pai_planner.py`, in a history processor) and does not end the run until the turn's model-call cap does: `retries` is `RETRIES` (1000), above the cap, since Pydantic AI charges each such call against it.
- A response Pydantic AI refuses (`UnexpectedModelBehavior`: an empty stream, a response the provider blocked) ends the run with no plan, as ADK ends a run on a response with no function call. `PaiRunner.run` catches it and the seam refuses with `plan_not_ready`.
- `Instrumentation()` gives the planner agent, generation and tool spans, and `Agent.iter(conversation_id=session.id)` sets `gen_ai.conversation.id` to the session's id.

## Controls

Each control runs the same comparison on a Pydantic AI arm broken in one place and expects it to fail. Output of the comparison on the broken arm:

| Control                                                          | How it fails                                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| instruction ends in `" X"`                                       | request 0 differs beyond `WIRE` in `system_instruction`                              |
| `propose_plan`'s description ends in a space                     | request 0 differs beyond `WIRE` in `declarations`                                    |
| the user's message loses its `instructions` note                 | request 0 differs beyond `WIRE` in `contents` (the two user contents merge into one) |
| `PlannerDeps.end_run` does nothing, on the ask script            | outcome differs; the arms send 3 and 5 requests; request 2 differs in `contents`     |
| `PlannerDeps.end_run` does nothing, on the declared-ready script | the arms send 1 and 2 requests                                                       |
| `ask_user` accepts every ask                                     | outcome differs; the arms send 2 and 1 requests                                      |
| `PaiCall.steps` returns nothing                                  | outcome differs: both asks of one response are accepted                              |
| `read_skill_page` answers a fixed string                         | outcome differs; requests 1 and 2 differ in `contents`                               |
| thought filter off (`keep_thoughts=True`)                        | the second request carries the summary text                                          |
| the cached prefix carries an `AUTO` tool config                  | the prefix differs in `tool_config` as well as in `reject_plan`'s parameters         |

## Private names and monkeypatches

Harness: `PlannerSeam._agent` (instruction and tool list), `_before_planner_model`, `_after_planner_model`, `_after_planner_tool`, `_on_planner_tool_error`; `orchestrator.agent._retrying` (retry options); `entry._no_anchor_read` and `entry.build_seam` (eval). The seam's tool callbacks are typed on ADK's `BaseTool` and `ToolContext`. `_body` gives them the ADK tool object for its name and a `SimpleNamespace` with `session.id` and `invocation_id`, which `_on_planner_tool_error` reads for its log line. A production port needs neutral callbacks.

`ToolWiring` repeats what `PlannerSeam.__init__` passes to `create_planner_with_translator`: `question_fixed=True` (the declarations' descriptions depend on it, so the first-request test covers it) and `reads_user_words=False` (no test covers it).

ADK: `BaseTool._get_declaration`, through `PlannerParts.of`.

Pydantic AI: `GoogleModel._build_content_and_config` and `GoogleModel.prepare_request` in `PlannerModel.uncached_config`; the profile keys `json_schema_transformer`, `supports_inline_system_prompts` and `google_supports_strict_tool_definition`; `Model.request_stream` overridden through `WrapperModel`. `supports_inline_system_prompts=True` keeps the `<system>` tag sentence (`_framing.SYSTEM_VOICE_ANNOUNCEMENT`) out of the instruction; it also stops the escaping of `<system>` in user prompts and tool returns.

Pydantic AI merges consecutive requests (`messages.py`, `_merge_consecutive_messages`) unless both carry differing `instructions`. The user's message carries `QUESTION_INSTRUCTIONS` for that reason, so contents match ADK's two user contents.

Monkeypatches: `PlannerClient` replaces `generate_content_stream` on its own client instance. The tests replace `entry.build_seam` (to point ADK's model at the scripted client), `entry.Probe.translate` (declared-ready tests only) and the broken variants, all restored by `monkeypatch`.

`warnings.filterwarnings` hides Pydantic AI's warning that `google_cached_content` strips the prefix; `pytest.ini` repeats it.

## Parity beyond the tests

`../parity/` runs every planner_entry task through both arms and settles the argument and empty-response gaps (`../parity/NOTES.md`). `_body` converts arguments and answers a missing mandatory argument through ADK's own tool object (`convert_arguments`, `missing_arguments`); the agent's output type is `str | None` so a response with no parts ends the run as it does on ADK (`OUTPUT_TYPE`).

## Not reproduced or not exercised

- The budget's bounds, the first-chunk resend, the 403/404 prefix-cache resend, an unknown tool name and a refused response. `test_parity.py` reaches none of them; `../node-hard/` does.
- `_check_branches`: Pydantic AI has no branches, and `ctx.session.events` is empty.
- A pick's replan, an amendment, `revise`, Stop and the progress relay: the entry runs `__call__` for a first draft and its resume only.
- Absent from `agent/`: the `validated` tool-config mode, `pai_record`, the experiment fingerprint and the live smoke.

## Commands

From the worktree root:

```
cd research/pai-planner-live/agent
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
```

```
uv run --package harness ruff format research/pai-planner-live/agent
uv run --package harness ruff check research/pai-planner-live/agent
```
