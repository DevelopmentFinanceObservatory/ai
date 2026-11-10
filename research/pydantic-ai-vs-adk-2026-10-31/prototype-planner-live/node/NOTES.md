# The planner on Pydantic AI in the planner node of a Studio turn

## Verdict

The real planner runs on Pydantic AI inside the real Studio turn and matches dev's ADK planner on every pass line this directory tests: frames, requests, ask and resume, amendment, execute replan, pick, Stop and stored records. In the stored-records comparison no turn executes, so the run ledger and the plan records are empty on both arms and their equality says nothing (see "What a stub stands in for"). 24 tests pass, about 13 s a run. Each line has a control, an arm broken in one place, that fails the same check. `../node-hard/` holds the budget, refusal, restart and tracing cases, which run on this node.

The node is dev's own `planner_node`; only the seam it awaits differs. The Pydantic AI agent runs where `ctx.run_node` ran, as a child node named and branched like ADK's planner. The child yields events as its loop proceeds, and ADK streams a child's events live, so the node needs no relay.

A Pydantic AI tool that writes a dict and nothing else differs from an ADK tool in three places, each closed in `pai_node.py`:

1. Dev's translator opens the "Checking the plan against the data" line from the planner's own function-call event. Without that event the frames lose two lines.
1. `PlannerSeam._persist_durable` carries a fixed list of state keys. ADK carries every other key a tool writes on the tool's response event. Without them a resumed run reads no skill page text (`KeyError`) and the stored session lacks the translator's state.
1. `PlannerSeam.revise`, which the `switch` node calls, reads the user's answers from the session's events. With no planner events, the ask's arguments appear one level down in the interrupt's call, and `combination_picks` reads nothing.

## Design

`pai_node.py` holds the code.

- `PaiPlannerSeam` is the `PlannerModel` the node awaits. `__call__` runs `PlannerSeam.__call__` unchanged over a `PaiCtx`, a proxy of the real `Context` whose `run_node` runs the agent. `revise` is the real seam's.
- `PaiCtx.run_node` runs a function node named `dfo_planner` with `ctx.run_node`, under the seam's own `run_id` and branch. The node sets `ctx.event_author` as `LlmAgent` does, so its events are authored by the planner.
- `FeedingRunner.events` is the agent loop (`agent/pai_entry.PaiRunner.run`) as an async generator. Before the tools of a response run it yields ADK's function-call event; after them it yields the function-response event, whose `state_delta` holds every key the response's tools wrote (`PlannerDeps.drain_writes`, from the `RecordingState` the tool bodies write through), the plan the call recorded under `PLAN_STATE_KEY` among them, which the translator reads. ADK stores the keys when it appends the event, so a tool's writes are durable when the tool ends and survive a Stop. It then yields each progress report the seam's tool callback queued. A run ends before the next model request when a tool ended it, or when Pydantic AI refuses a response, as in `agent/`.
- `PaiPlannerSeam._earlier` gives the steps of the turn so far: `planner.agent._session_steps` over the session's events, read once when the seam call starts. A call that resumes an ask the events hold no answer to yet adds the answer from `ctx.resume_inputs`, as ADK's interrupt route adds its response to the session. `PaiCall` reads those steps and then the live run's own, so no message record is stored: the session's planner events are the only record, which ADK's planner and this one both append. A run starts from the turn's message (`user_text`, the first user-authored event) and the brief.

The progress reports go to the child, so dev's queue stays empty and the node's own late relay finds nothing. Dev's translator draws no frame from a report (`_plan_path_frames`: they "add no frame"); the test for line 1 patches it to draw one, on both arms.

## Pass lines

Each test runs one script on the `adk` arm (dev's node) and the `pai` arm and compares them. The orchestrator's model is scripted to a planner-entry task's recorded `search` and `plan` calls, the planner's model is a scripted google-genai client, and the translation and surface answer from the task's recording.

| Line                                      | Result                                                                                                                                                                                                                                                                                                                                                                | Test                                                                                      | Control and how it fails                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Progress frames while the planner runs | First progress frame 0.04 to 0.07 s on Pydantic AI; on dev 3.1 s, the 3 s the held translation waits plus the time to finish. First "checking" frame 0.04 to 0.07 s against 0.05 to 0.11 s. The progress frame of the skill read releases `propose_plan`, which is still translating.                                                                                 | `test_progress_frames_reach_the_client_while_the_planner_runs`                            | Dev's arm is the control. `test_control_progress_reports_held_until_the_draft_is_done...` sends the reports to dev's queue (`events=False`): the gate times out and the frame arrives at 3.1 s.                                                                                                                                                 |
| 2. Frames of a planned turn equal dev's   | Frame names and fields equal after dropping `seq`, ids, durations and times. The requests equal dev's apart from the four wire fields in `agent/NOTES.md`. No explained difference remains.                                                                                                                                                                           | `test_the_frames_of_a_planned_turn_equal_devs`                                            | A planner yielding no events lacks the two "Checking the plan against the data" frames.                                                                                                                                                                                                                                                         |
| 3. Ask, pause, answer, resume, plan       | Same frames, three requests each, the question's options and labels reach Studio, the resumed run's first request holds the user's message and a brief with the answer and no call. The same answer posted twice answers 404 on both arms.                                                                                                                            | `test_an_ask_pauses_through_devs_interrupt...`, `test_the_answer_of_an_ask_is_taken_once` | A planner that reads no earlier step loses the user's pick of the combination, so the resumed proposal is refused, the turn ends with no plan card and the arm sends four requests against three.                                                                                                                                               |
| 4. Amendment; ask during an amendment     | Version 1 then 2 with equal frames and requests; the amendment's words are in the second request. The ask pauses the amendment and its answer drafts version 2.                                                                                                                                                                                                       | `test_an_amendment_drafts_version_two...`, `test_an_ask_during_an_amendment...`           | An amendment dropped before the seam: the second request lacks "only 2020". A resume hidden from the seam: the third request lacks the answer.                                                                                                                                                                                                  |
| 5. Refusal replan from `execute`          | `execute` routes `replan`; the planner node replans from the refusal and drafts version 2 with frames and requests equal to dev's; the refusal's code is in the second request.                                                                                                                                                                                       | `test_a_refusal_replan_from_execute...`                                                   | The refusal dropped before the seam: the second request differs.                                                                                                                                                                                                                                                                                |
| 6. A pick's replan; a pick after an ask   | A pick of five donors replans to version 2 with the pick stated on the card, equal to dev's, with and without an ask in the run before it. The steps a `revise` reads (`_session_steps` over the session's events) and the readers over them (`elected_spec_ids`, `combination_picks`, `answers_since_referral`, `prices_chosen`) equal dev's after a run that asked. | `test_a_pick_replans_the_plan_as_dev_does`, `test_a_revision_after_a_run_that_asked...`   | The pick dropped before the seam: the frames or requests differ from dev's. Only `propose_plan` events in the session: `combination_picks` answers `({}, False)` against dev's one pick.                                                                                                                                                        |
| 7. Stop mid-planning                      | Cancel answers 204, the stream ends on an `error` frame `cancelled`, no task is left running, and the next turn on the same app plans. Equal to dev's.                                                                                                                                                                                                                | `test_stop_mid_planning_ends_the_turn...`                                                 | The agent loop run in a task nothing cancels leaves tasks after the Stop.                                                                                                                                                                                                                                                                       |
| 8. Stored records                         | A turn that reads a skill page, asks, is answered, proposes and is approved leaves the same session state (keys and values, apart from the per-run values below), plan, approvals and turns. The Pydantic AI path adds no key.                                                                                                                                        | `test_the_records_a_turn_leaves_equal_devs`                                               | With the response events carrying only the plan, the stored session loses `planner:pin_verdicts`, `planner:windowed_pins`, `translator:discovery`, `translator:discovery_lock_token`, `translator:referral` and `translator:resolved*`. A skill page read before the ask raises `KeyError` on the resume and the turn ends on an `error` frame. |

The one-ask turn (a skill read, an ask, the answer, a proposal) stores 114,030 bytes of session state (JSON) on both arms, and a session of 325,123 bytes with its 26 events against dev's 326,094.

Per-run values the records comparison drops (`comparable`): `plan_version_id`, `supersedes` and `interrupt_id`, each a `_wire_id` (`harness.studio_state`, a fresh uuid per run; `supersedes` holds the previous `plan_version_id`), the translation's lock token (compared as present or absent), and the `id` of each record in `translator:discovery`, which also keys it. Every other `id`, a requirement's or a step's, is compared. Requests are compared against `parity.WIRE` exactly, so a node-level difference in declarations, thinking config or cache fields fails as it does in `parity/`.

## What a stub stands in for

- Line 5: `execute` is wrapped so its first run yields the `replan` route with a refusal built from `RediscoveryNeeded` through `harness.graph.execute._planner_refusal`; later runs are the real node. The recorded plans have no data surface to run on, so no step fails of itself. The stopped source is none the plan pins, because `propose_plan` refuses a replan that keeps a source the executor refused.
- Line 8: the approval runs `execute` against no tool surface, which fails the turn after `plan.approved`. The run ledger and plan records are empty on both arms, and the test asserts they are equal.

## Internals relied on

Harness: `PlannerSeam._agent`, `PlannerSeam.__call__` calling `ctx.run_node(self._agent, None, run_id=..., override_branch=...)` and reading `ctx.session.state`, `ctx.session.events`, `ctx.state` and `ctx.resume_inputs`; `planner.agent.AGENT_NAME`, `BRIEF_STATE_KEY`, `load_pending_ask`, `_session_steps` (the seam reads it for the earlier steps, as `AdkCall` does); `harness.studio_translate._plan_path_frames` (the author check at lines 1406 to 1415), `_node_name`, `_open_status`, `_status` and `_WORKING_ON_PLAN` (the progress patch); `harness.graph.execute._planner_refusal` and `workflow.execute_node` (the stub); `tests.orchestrator_fakes`.

ADK 2.13.0: `Context.run_node` runs a function node built with `google.adk.workflow.node`; `Context.event_author` (the setter, as `agents/llm_agent.py:692` uses it) decides an event's author, since `_node_runner._enrich_event` writes `ctx.event_author or node name`; every non-partial event a node yields is appended to the session (`runners.py`), so the planner's events are session content; `InMemorySessionService.get_session` returns a copy, so a write to the live dict outlasts the invocation only through a `state_delta`.

Pydantic AI 2.66.0: `Agent.iter`, `Agent.is_call_tools_node`, `Agent.is_model_request_node` (a `CallToolsNode` reaches the loop before its tools run), `UnexpectedModelBehavior`.

From `agent/`: `PaiRunner`, `PlannerDeps`, `PaiSession`, `planner_agent`, `PlannerClient`, `PlannerModel`, `CallHooks`, `ToolWiring`, `PlannerParts`, `google_model`, `genai_client`, `pai_call.user_message`, and the private `pai_entry._NO_REQUEST_LIMIT`.

## Duplication with `agent/`

`FeedingRunner.events` repeats the loop of `PaiRunner.run` (about 20 lines), with the yields added. `PaiRunner.run` could be an async generator that yields at the call-tools node and the next model-request node, with `run` consuming it. `_NO_REQUEST_LIMIT` is imported with its underscore.

## A production version

Dev's `planning.py` (584 lines), `planner_seam.py` (2,680) and `studio_translate.py` are unchanged. The node's code is `PaiPlannerSeam`, `PaiCtx`, `user_text`, `_earlier`, the agent loop as a generator and the two event builders. It reads and writes the session only as ADK's own planner does: the planner events it yields, and the `state_delta` on each response event, from a write-recording wrapper on the state the tool bodies receive. No deep copy of the state is made.

Open choices:

- The seam's `ctx.run_node` and `_check_branches` stay, so ADK's branch rule and the planner events in the session hold as today. A port that dropped the child node would also drop the planner's frames and `revise`'s reading of asks, unless the translator and `AdkCall` took neutral events.
- The node writes ADK-shaped events and reads them back through `_session_steps`, a private function. A test that every key a tool body writes survives a resume is the control of line 8.

## Not covered

- The `switch` node through Studio. A scope choice needs a resolved definition with scope choices and a `resolve_definition` surface, which no recorded task offers. The test reads what `revise` reads, the steps, and not `revise`'s answer.
- A turn that executes: the recorded plans have no data surface, so the ledger holds no run.
- The budget cut mid-stream, the first-chunk resend, the prefix cache resend, an unknown tool, an empty response, `UnexpectedModelBehavior`, an ask and a `propose_plan` in one response, a fresh `Runner` over the stored session, a pause under one path answered under the other, the writes a Stop keeps, and the planner's spans: `../node-hard/`.
- A second turn of a conversation: each turn has its own ADK session here, so the steps start from the turn's message.
- Live Gemini.

## Files

- `pai_node.py`: the node's planner.
- `studio_arms.py`: one Studio turn on either planner, over a planner-entry task's recording.
- `scenes.py`: frames and the differences between two runs.
- `streaming.py`: a streaming in-process client, copied from `pai-node-stream/studio/wiring.py`, for frame times and disconnects.
- `test_node.py`: the 24 tests.

## Commands

From the worktree root:

```
cd research/pai-planner-live/node
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
```

```
uv run --package harness ruff format research/pai-planner-live/node
uv run --package harness ruff check research/pai-planner-live/node
```
