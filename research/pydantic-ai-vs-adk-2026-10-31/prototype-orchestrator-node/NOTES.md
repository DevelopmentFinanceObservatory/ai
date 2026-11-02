# The Pydantic AI orchestrator inside ADK's `orchestrate` node

The orchestrator can run on Pydantic AI 2.64.0 inside the real `orchestrate` node, with ADK 2.13.0 still owning the turn, and Studio sees the same frames and stores the same records. The node is `orchestrator_node.pai_orchestrate_node`. It runs a Pydantic AI `Agent` with dev's instruction, digests and tools. It swaps in for dev's node in test code only, through `create_root_workflow`, and nothing under `services/` changes. All 16 tests pass, two runs in a row, about 14 s each. No model is called.

Each pass line is a function that returns the ways a run fails it. The test asserts it returns nothing, and the control asserts it returns the failure the control was built to cause.

## Results

| Line                 | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Control, and how it failed                                                                                                                                                                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| O1 a full turn       | Question, `plan`, dev's scripted planner, approval, `execute`, `plan_done`, then the reply. The reply's three chunks arrive as three `message.delta` frames and one `message.complete`. Both segments' frames (5 and 15) equal dev's once ids and times are removed. Every `orchestrator*` state key, the plan, `awaiting_approve`, the approvals and the stored turns are equal too. The one extra state key is `orchestrator:conversation`, which holds `None` at the end. | `Variant.BUFFERED` sends no partial event. Its reply arrives as one `message.delta`, so the check reports "reply not streamed chunk by chunk", and the segment has 13 frames against dev's 15.                                                                                  |
| O2 the clarify pause | `ask_user` reaches Studio as an `elicitation.request` whose frames equal dev's. The answer to `/interrupt` resumes the stored conversation. The model log reads "after 1 messages", then "after 3 messages", so no call repeats. The `ask_user` result equals dev's hand-back, and the resumed frames equal dev's.                                                                                                                                                           | `Variant.RESTART` starts again from the user's message. The log reads "after 1 messages" twice, and the call result is missing (`ask_user result [] != {...}`).                                                                                                                 |
| O3 the hand-back     | After `plan_done`, the next request ends with one `ToolReturnPart` for `plan` and no user message. Its content equals the `{"result": text, **body}` that dev's `_call_result` gives the ADK model: the frames, the step summaries and the plan record.                                                                                                                                                                                                                      | `Variant.HAND_BACK_AS_MESSAGE` answers the call with its own record and sends the hand-back as a user message. The check reports that the plan result differs (it has `handed_off: 'plan'` and no `frames` or `plan_record`) and that the hand-back came with one user message. |
| O4 a tool failure    | `search` raises `RuntimeError`. The model gets `search failed inside DFO (RuntimeError)...` with code `raised` and `do_not_retry`, and the URL in the exception is withheld. One ERROR record is logged with the traceback. The turn ends `done`, and its frames equal dev's.                                                                                                                                                                                                | `answer_failures=False` gives no failure handling. The run fails, and the stream ends on an `error` frame (`transient_unavailable`).                                                                                                                                            |
| O5 isolation         | The planner is dev's ADK `PlannerSeam` on a recording model. No planner request holds the orchestrator's sentinel (in its thought and its text), its instruction's first line, `make_charts` or its call ids. No orchestrator request holds the planner's sentinel or any of its tool names. The positive checks hold: planner requests carry the brief and every planner tool name, and orchestrator requests carry their own sentinel.                                     | `Variant.LEAK_STATE` writes the sentinel into the pending hand-off's brief, which is the one channel from this node to the planner. It is found in a planner request.                                                                                                           |
| O6 wire parity       | The first Gemini request of each arm is captured offline, under `StreamingMode.SSE`, with digests in state. The system instruction text, 9 of 10 declarations, the declaration order and the thinking config (`LOW`, `include_thoughts`) are byte for byte equal. ADK's identity line is stripped. Every other difference is listed below and explained.                                                                                                                     | Appending a sentence to the `search` description gives an unexplained `declarations` difference at `search`.                                                                                                                                                                    |

A probe beside O5, `Variant.LEAK_EVENT`, writes the orchestrator's own text into the session as an unbranched model event. It lands in the session and reaches no planner request. Dev's planner runs as a `single_turn` node agent, and ADK's `_filter_single_turn_node_events` shows such an agent only events on its own node path and its ancestors. So events from `orchestrate` cannot reach it, branched or not.

### O6 differences

| Field                     | ADK                                         | Pydantic AI                                       | Why                                                                                                                                                                 |
| ------------------------- | ------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| system instruction        | `INSTRUCTION`, a string                     | `INSTRUCTION` without its final `\n`, a `Content` | `_instructions.py:256` strips each instruction string; the Google model sends the instruction as a `Content`                                                        |
| `explain_dfo` declaration | no `parameters_json_schema`                 | an empty object schema                            | `google.py:2519-2523` always sends `parameters_json_schema`                                                                                                         |
| tools grouping            | one `Tool` with 10 declarations             | 10 `Tool`s of one declaration each                | the Google model's `_get_tools`                                                                                                                                     |
| tool config               | none (Gemini reads AUTO)                    | `function_calling_config: AUTO`                   | always sent with function tools; AUTO with strict tool definitions off                                                                                              |
| contents                  | digests and question as two user `Content`s | one user `Content` with two parts                 | `_merge_consecutive_messages` merges adjacent requests and one request maps to one `Content`; dev's `prepend_digests` inserts the digests as a content of their own |
| `automatic_continuation`  | `true`                                      | absent                                            | `GoogleModelSettings` has no field for it, so the genai default (true) applies                                                                                      |
| `response_modalities`     | absent                                      | `["TEXT"]`                                        | always sent                                                                                                                                                         |
| headers                   | ADK's `user-agent`, `x-goog-api-client`     | `User-Agent: pydantic-ai/2.64.0`, `Content-Type`  | each framework's own                                                                                                                                                |

The instruction's missing newline and the tool grouping both change `harness.prefix_cache.prefix_key`, so the two frameworks cannot share one cache prefix, as the planner's parity run found.

## Design

### Where the conversation lives

The orchestrator's `Conversation` lives in session state under `orchestrator:conversation`. The `orchestrator:` prefix keeps it past `GraphState` validation. It is written on the node's own output event, the one that ends the entry, so it is saved before `route` sends the turn to `ask` or the planner. It is cleared to `None` when the agent answers.

Each Studio turn has an ADK session of its own (`{conversation}#t{n}`). So the session is the turn, and no invocation key is needed; stage A's planner needed one. Rebuilding Pydantic AI messages from ADK events at each entry was the alternative. That would repeat the lossy event-to-message mapping the planner prototype had to build (`prototype/gemini/pai_planner.adk_events`), with nothing gained.

### How a hand-off and its hand-back work

`plan` and `ask_user` still record their hand-off through dev's `_pend`. When a tool's result holds `HANDED_OFF`, the tool body raises `CallDeferred(metadata={"record": result})`. The run then ends with `DeferredToolRequests`, the way `skip_summarization` ends dev's agent with no further model call.

When `ask` or `plan_done` hands back, the node loads the conversation and finds the latest deferred call to the hand-back's tool. It continues with `DeferredToolResults(calls={id: hand_back.response()})`. With no such call it raises the same `RuntimeError` dev's `_call_result` raises. A second hand-off in one turn is refused by `_pend` as on dev, so it returns normally and never defers.

On the resumed run, Pydantic AI first re-presents the model response the last entry ended on, as a `CallToolsNode` holding `tool_call_results`. The node sends no model event for it. It stores the hand-back as dev's node does: a function response authored `user` on the agent's branch.

### How text and thoughts reach Studio

The node relays `Agent.iter` as the ADK events dev's `LlmAgent` makes, so `studio_translate` needs no change:

- a partial event for each text chunk and each thought chunk, the thought marked `thought=True`;
- then the call's whole content, with thought, text and function calls under Pydantic AI's call ids;
- then one event holding every tool result of that call, with the state each tool wrote merged into its `state_delta`, as ADK merges parallel responses.

All of them are authored `dfo_orchestrator` (through `ctx.event_author`) on the agent's branch. The output event's empty branch clears it for later nodes. The translator drops thought chunks on both arms, so Studio shows none, as on dev.

`Agent.iter` runs the agent in the node's own task, unlike `run_stream_events`. So a stop cancels it directly, and each `CallToolsNode` hands over the finished `ModelResponse`.

### The tools

Each ADK `FunctionTool` is declared to Pydantic AI with the JSON schema ADK declares (`Tool.from_schema`). Each call runs through `FunctionTool.run_async`. It gets a fresh ADK `Context` with the call's id, the same object ADK's `_create_tool_context` builds. ADK's argument preprocessing, `_validated_arguments` and every state write therefore behave as on dev. A tool that raises goes to the ADK agent's own `on_tool_error_callback` (`answering_tool_failures`).

### Stop mid-reply (measured)

Studio's `/cancel` was posted while the model stalled after its first chunk. The stream sent that chunk as a `message.delta` and then ended on the `cancelled` error frame, with no `message.complete`. The model stream was cancelled and no task outlived the stream. No conversation was stored, since state is written only when an entry ends, and the next turn on the conversation ran to its reply. Dev's agent was not run under the same stop.

### The turn budget (measured)

`budget.BoundedModel` is a `WrapperModel` that reads `current_turn_budget` around each streamed request. It carries over the call cap and the bounds checked before a call. With `turn_model_calls_max=2` and a script of three tool calls and a reply, both arms made two model calls and ended on the same sentence ("This request took as many model steps as it allows..."), with equal frames.

It does not carry over the cut mid-stream. With `turn_model_time_s=0.3` and a model stalled for 2 s after its first chunk, dev ended `done` on "DFO is " followed by the bound's sentence. The Pydantic AI turn sent "DFO is " and ended on an `error` frame, because `asyncio.timeout` raises out of the stream. The first-chunk resend and the span attributes are not carried over either. Closing on the sentence needs a `StreamedResponse` of our own that relays the wrapped response's events and parts, or a catch in the node that answers with the sentence. Neither was built.

## Internals relied on

| Internal                                                                                                                                                           | Kind                                            | Why                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| ADK `Context(ctx._invocation_context, function_call_id=...)`                                                                                                       | private attribute                               | each tool call's context, as `_caller._create_tool_context` builds it                           |
| ADK `FunctionTool._get_declaration`                                                                                                                                | private                                         | the declarations Pydantic AI is given                                                           |
| ADK `LlmAgent.canonical_on_tool_error_callbacks`, `FunctionTool.run_async`                                                                                         | public                                          | the failure answer and the tool call                                                            |
| ADK `_NodeRunner._enrich_event`: author from `ctx.event_author`, `branch=""` clears the branch                                                                     | behaviour, read from `workflow/_node_runner.py` | authoring the agent's events and clearing its branch                                            |
| ADK `_filter_single_turn_node_events`                                                                                                                              | behaviour                                       | why no `orchestrate` event reaches dev's planner (O5)                                           |
| `harness.turn_events._turn_question`; `orchestrator.agent._DIGESTS`, `_utc_today`, `_retrying`                                                                     | private harness                                 | the first prompt, as `prepend_digests` builds it; the ADK arm of O6                             |
| Pydantic AI `Agent.iter`, `node.stream`, `CallToolsNode.model_response`, `Conversation`, `CallDeferred`, `DeferredToolResults`, `Tool.from_schema`, `WrapperModel` | public                                          | the run, its events, the pause and the budget                                                   |
| Pydantic AI `CallToolsNode.tool_call_results`                                                                                                                      | public field, undocumented                      | telling the resumed node from a fresh one                                                       |
| `harness.graph.workflow.orchestrate_node`, `tests.orchestrator_fakes.ScriptLlm`                                                                                    | module attributes, monkeypatched                | the node swap and dev's streaming script model                                                  |
| `PlannerSeam._agent`                                                                                                                                               | private                                         | putting the recording model under dev's planner (O5), as `tests/test_orchestrator_path.py` does |

## What a production version would change

Sizes are rough, from dev at 190c7a9c.

- `harness/graph/orchestrate.py` (476 lines). `orchestrate_node`, `open_call` and `_call_result` (about 120 lines with docstrings) become the node, the entry and the event relay, about 220 lines without this prototype's variants. `AgentHandBack`, `route`, `ask` and the answer helpers stay.
- `orchestrator/agent.py` (447 lines). `create_orchestrator_agent`, `_without_adk_identity`, `prepend_digests`, `_stamp_today` and `_retrying` (about 150 lines) become the Pydantic AI agent's builder and the first prompt, about 80. `INSTRUCTION` stays.
- `harness/orchestrator_tools.py` (2,795 lines). Unchanged behind the adapter. To drop it, the tools move off `ToolContext` onto a record of their own, as the planner's tools are moving. They read only `state` (37 sites), `invocation_id` (8) and `actions.skip_summarization` (1, in `_pend`, which becomes the deferral).
- `orchestrator/tool_failures.py` (67 lines). Its signature names ADK's `BaseTool` and `ToolContext`; about 10 lines change once the tools leave ADK.
- `harness/graph/workflow.py`. `_fetch_tool` reads the `fetch` `FunctionTool` off the `LlmAgent` for `seed`; it reads it from the new builder instead, under 10 lines.
- `harness/turn_budget.py`. `BoundedModel` is about 50 lines. The cut mid-stream with its closing sentence, the first-chunk resend and the span attributes are about 80 more.
- `harness/prefix_cache.py`. `PrefixCache.model_for` builds an ADK model. The Pydantic AI side needs the `google_cached_content` path the planner prototype built (`prototype/gemini/pai_planner.PlannerModel`), about 80 lines.
- `harness/studio_translate.py`. No change, while the node keeps sending ADK's event shapes.
- Tests. `ScriptLlm` becomes a `FunctionModel` script. 10 test files drive `orchestrator_runner` or `orchestrator_app`, and 68 lines read `model.requests` as ADK `LlmRequest`s; they would read Pydantic AI messages.

## Not covered

- A seed hand-back (`answers` None): `first_prompt` puts it after the question, as `prepend_digests` does, but no test runs it.
- Two calls in one model response that mix a hand-off with another tool, such as `search` with `ask_user`. By the code, the search result and the record go in one event, as on dev.
- A made-up tool name. Pydantic AI answers with its own retry prompt, where ADK lists the tools it has. Not measured.
- A second pause in one turn, an amendment, a switch, and a planner that pauses on its own ask.
- A crash between entries. The conversation is in session state like dev's history, so a resumed turn reads it, but no kill was run.
- A live Gemini run, the prefix cache and replay of thought signatures on `GoogleModel`. The planner's parity run covers the last two for the planner.
- Dev's agent under the same stop mid-reply.

## Files

| File                        | Purpose                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `orchestrator_node.py`      | The node, the agent built from dev's `LlmAgent`, the event relay, and the controls (`Variant`)   |
| `script.py`                 | One script played by dev's ADK model (`StreamingScriptLlm`) and by a Pydantic AI `FunctionModel` |
| `wiring.py`                 | The Studio app for each arm, and the record comparison                                           |
| `budget.py`                 | `BoundedModel`, the turn budget on a Pydantic AI model                                           |
| `test_orchestrator_node.py` | O1 to O5, their controls, the O5 probe and the stop                                              |
| `test_wire_parity.py`       | O6 and its control                                                                               |
| `test_budget.py`            | The turn budget on both arms                                                                     |
| `conftest.py`, `pytest.ini` | The harness suite's fixtures and paths                                                           |

## Commands

From the worktree root:

```
cd research/pydantic-ai-vs-adk-2026-10-31/prototype-orchestrator-node
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,dbos]==2.64.0' --package harness pytest . -q -s -p no:cacheprovider
```

Lint, from the worktree root:

```
uv run --package harness ruff format research/pydantic-ai-vs-adk-2026-10-31/prototype-orchestrator-node
uv run --package harness ruff check research/pydantic-ai-vs-adk-2026-10-31/prototype-orchestrator-node
```
