# Stage A's planner node inside real Studio turns

Each test runs the real root workflow (`create_root_workflow`) behind the real Studio routes (`orchestrator_app` in `services/harness/tests/orchestrator_fakes.py`), with the scripted orchestrator model, `ScriptedExecutor` and, where a plan runs, `ScriptedSurface`. Only the `planner` node changes: stage A's relay (`minimal/graph.py`, `_relay_live`, imported unchanged) drives a Pydantic AI 2.64.0 planner on a `FunctionModel`. No `services/` file is edited and no model is called.

## Result

All 11 tests pass, two runs in a row, about 15 s each.

| Check                  | Result                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1 live progress       | The first progress frame reaches the SSE client 0.09 to 0.20 s after the request, while `propose_plan` waits on the client's signal; its gate never times out. The control, dev's `planner_node` with a seam stub reporting through `on_progress`, sends its first progress frame at 3.08 to 3.12 s: after the stub's 3 s gate timed out and `propose_plan` ended.                                                                                |
| S2 ask                 | The ask arrives as an `elicitation.request` with one question, after all 10 progress frames of the first proposal. Answering it through `/api/studio/interrupt` resumes the stored run: the model log shows calls after 1, 3, 5 and 7 messages, each once, and the resumed stream sends the second proposal's 10 frames and then `plan.proposed`.                                                                                                 |
| S3 isolation           | No orchestrator request in a full ask, answer, approve and run turn contains the planner's sentinel, its tool names, its call ids or the ask's interrupt id, and none carries an interrupt or quotes another agent. A positive control shows the same requests hold the brief and the hand-back.                                                                                                                                                  |
| S4 stop and disconnect | `POST /cancel` mid-`propose_plan` ends the stream with the `cancelled` error frame. Closing the response mid-`propose_plan` ends the app's response. In both, the planner logs `propose_plan call=1 cancelled`, no task outlives the stream, and the next turn on the conversation runs to `plan.proposed`.                                                                                                                                       |
| S5 records             | Against dev's node with the stub seam, the stored records are equal once times and per-run ids are removed: the plan in session state, `awaiting_approve`, the turn's plan records, the plan card and its revision, the `plan.approved` frame, the approval records and the stored turns. The one difference is an extra session state key, `planner:conversation`, holding `None` at the end of the turn. `approve` and `execute` are unchanged. |

### Stage A's open points

- Stop path. Studio's Stop and a disconnect both reach the planner as a `CancelledError` at its `await`. By the code, `_sse_stream` cancels the task running the next `anext` (`_unwind`), and the cancellation unwinds `_locked`, `_checkpointed` and `_frames`, whose `aclosing(runner.run_async(...))` then closes the ADK stream. So the planner sees a cancellation first and the `aclose` follows; the tests observe the cancellation, not which of the two reached ADK's `_cleanup_root_task`.
- Answering twice. Posting the same answer again after the resumed stream reached `plan.proposed` gets `404 not_found` ("no pending interrupt"): `_checkpointed` drops the pending question when the resumed segment checkpoints before its pause frame. The session holds one function response to `planner_ask:ask-1`. A second post while the first resume is still streaming would meet the live segment and get a 409 (`SegmentBusy`, by the route's code; not run). A retry after the resuming instance died, with the same answer, is let through by design and would append a second response, the case stage A saw.

## Harness seams reached

| Seam                                                                                   | Kind                                          | Why                                                                                                        |
| -------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `harness.graph.workflow.planner_node`                                                  | module attribute, replaced with `monkeypatch` | puts the Pydantic AI node in `planner_entry` without editing `create_root_workflow`                        |
| `harness.studio_translate._Translator._plan_path_frames`                               | private method, wrapped                       | dev's translator sends no frame for the planner node's `custom_metadata["progress"]`                       |
| `_Translator._open_status`, `_status`, `_conversation.next_seq`                        | private                                       | the wrapper puts progress on the planner's status line                                                     |
| `harness.studio_translate._CHECKING`, `_node_name`                                     | private                                       | the status label and the node test                                                                         |
| `harness.graph.state.interrupt_event`, `PLANNER_STARTED_KEY`, `ORCHESTRATOR_BRIEF_KEY` | public                                        | the ask's interrupt on dev's interrupt branch, the start event, the brief                                  |
| `planner.plan.Plan`, `plan_revision`; `harness.scripted_plan.scripted_plan`            | public                                        | the drafted plan and its revision, as dev's node writes them                                               |
| `tests/orchestrator_fakes.py`, `tests/conftest.py`, the root `conftest.py`             | test modules                                  | the app, the client, the stream reader, and the suite's autouse isolation, loaded by path in `conftest.py` |
| ADK `_contents` internals                                                              | read only, in diagnosis                       | see S3 below                                                                                               |

`studio_client` (`httpx2.ASGITransport`) collects the whole response before it returns, so it can neither time a frame nor disconnect. S1 and the disconnect half of S4 use `wiring.StreamingASGITransport`, which hands each body chunk over as the app sends it and sends `http.disconnect` when the client closes the stream. It passes no ASGI `spec_version`, so Starlette's `StreamingResponse` listens for the disconnect, the path uvicorn takes.

## What a production version would change

Rough sizes, from this prototype and dev's files at 190c7a9c.

- `harness/graph/planning.py` (573 lines). `planner_node`'s await of the seam, its progress queue and the `NodeInterruptedError` handling (about 40 lines, 466 to 503) become the relay loop, the ask's interrupt, and the stored conversation keyed on `invocation_id` (about 70 lines in `node.py`). The amendment, refusal, unreadable-plan and pick branches stay, but a pick's replan runs the seam under `switch._unless_stopped` as a task; with the relay that wrapper has to stop the stream it drives, about 20 lines. The docstring's direct-await argument (about 30 lines) is rewritten.
- `harness/graph/workflow.py` (218 lines). Under 10 lines: `rerun_on_resume=True` stays, now because the node reruns on the ask's answer, and the comment beside it changes.
- `harness/studio_translate.py` (2,716 lines). Three changes, about 50 lines:
  - render the planner node's `progress` events, about 15 lines, as `wiring.show_planner_progress` does;
  - replace the two branches that read the ADK planner's own events (`event.author == PLANNER_AGENT_NAME`: `_planner_call_status` on a `propose_plan` call, `_proposal_answered` on its response with `PLAN_STATE_KEY` from the state delta), about 25 lines. The Pydantic AI planner makes no ADK events, so the "Checking the plan against the data" and "Reworking the draft" lines, and the accepted plan's data phrase, have to come from progress events the node yields with a kind and an outcome;
  - `_planner_question` reads `options` and `candidates` from the top level of the `adk_request_input` args, where the ADK planner's own call puts them. A `RequestInput` puts them under `payload`, so Studio showed the question with no options and free text only. About 5 lines.
- The seam (`harness/graph/seams.py` `PlannerModel`, 137 lines; `harness/planner_seam.py`, 2,547 lines; `planner/agent.py`, 2,822 lines). `PlannerModel.__call__` returning a dict after an await no longer fits: the node needs an agent and deps it can stream (`run_stream_events`) and resume from a `Conversation`. `on_progress` and the `ctx` hand-through (for `ctx.run_node`) go. `PlannerSeam`'s ADK hosting, its `after_tool_callback` progress and its event cap are replaced by Pydantic AI tools that `ctx.emit` progress and a `UsageLimits` or turn cap. The port of the planner's tools themselves is outside this prototype and is the bulk of the cost.

## What did not work, or is not covered

- S3 has no failing control. I tried to make the node's content reach the orchestrator, writing the planner's text into the session three ways (`node.Leak`): a model-role event on the empty branch, one on the orchestrator's branch, and a `user`-authored event on that branch, as `orchestrate` writes a hand-back. All three land in the session and none reaches a request. ADK's `_should_include_event_in_context` admits the empty-branch event, so a later stage of `_get_contents` drops it; disabling `_rearrange_events_for_latest_function_response` did not bring it back, and the stage that drops it was not pinned. The `user`-authored event is dropped by the rule that hides a user event from another node path. So S3 rests on ADK's context selection for a chat-mode agent run by `ctx.run_node`, and on the node putting nothing planner-internal in `ctx.state` or its `output`. The probes are kept as tests; the control is the positive one.
- Two asks in one planner run, an ask during an amendment or a pick's replan, a planner-side failure, and a crash between the answer and the planner's first event were not run.
- The progress frame is the prototype's choice of shape: a running `activity.step` on the planner's status line with `detail` "check n of m". Dev sends no frame for these reports today, so S1's "shape `studio_translate` emits" means the running-step shape the translator uses for `execute`'s step progress.
- The streaming transport is the test's own. A served check against uvicorn was not run.

## Files

| File                        | Purpose                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `studio_planner.py`         | The scripted planner: `propose_plan` (progress, a gate, the canned plan) and `ask_user` (`CallDeferred`). |
| `node.py`                   | The `planner` node for the real workflow, the S3 probes, and `DevStubSeam` for the controls.              |
| `wiring.py`                 | The node swap, the translator's progress frame, and the streaming client.                                 |
| `test_studio.py`            | S1 to S5 and the controls.                                                                                |
| `conftest.py`, `pytest.ini` | The harness suite's fixtures and paths.                                                                   |

## Commands

From the worktree root:

```
cd research/pai-node-stream/studio
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness pytest . -q -s -p no:cacheprovider
```

Lint and types, from the worktree root:

```
uv run --package harness ruff format research/pai-node-stream/studio
uv run --package harness ruff check research/pai-node-stream/studio
uv run --package harness ty check research/pai-node-stream/studio --extra-search-path research/pai-node-stream/studio --extra-search-path research/pai-node-stream/minimal --extra-search-path services/harness --extra-search-path <pydantic-ai 2.64.0 site-packages>
```
