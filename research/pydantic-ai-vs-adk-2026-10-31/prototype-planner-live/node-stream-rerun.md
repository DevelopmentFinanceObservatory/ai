# The Pydantic AI planner node on current dev

## Verdict

The design survives. The relay, the stored conversation, live progress, Stop and disconnect, and the stored records all work unchanged on dev at 245c0a44 (google-adk 2.13.0) with pydantic-ai 2.66.0. `minimal/` passes untouched on 2.64.0 and 2.66.0. `studio/` failed on its ask path, because dev's ask changed shape (#138). The repair is about 100 lines in `studio/` and removes prototype code instead of adding it: the node now yields dev's own `planner_ask_interrupt`, and Studio shows the ask's options and candidates with no translator change.

After the repair `studio/` has 20 tests (11 repaired, 9 new), all passing on 2.66.0 twice in a row, and on 2.64.0 once. `minimal/` has 12, all passing.

## Baseline

Both prototypes unchanged, against dev 245c0a44. `studio/pytest.ini` put dev's `services/harness` on the path one directory too high for this checkout, so every run needed that path corrected first (`../../../services/harness`); the rows below are with that one line fixed.

| Prototype  | pydantic-ai | Passed | Failed | Errored | Cause                        |
| ---------- | ----------- | ------ | ------ | ------- | ---------------------------- |
| `minimal/` | 2.64.0      | 12     | 0      | 0       | none                         |
| `minimal/` | 2.66.0      | 12     | 0      | 0       | none                         |
| `studio/`  | 2.64.0      | 5      | 6      | 0       | all six: dev #138 (ec1e076e) |
| `studio/`  | 2.66.0      | 5      | 6      | 0       | the same six                 |

The six failures are the S2 test and the five S3 tests, because each drives a turn through the planner's ask. The translator's `_planner_question` now reads the question from the interrupt's `payload` and expects each fixed option as `{"id", "label"}` (#138 moved the ask's arguments into `payload`). The prototype's payload was `{"options": ["Ghana only", ...]}`, so the translator raised `TypeError: string indices must be integers` and the turn ended before any `elicitation.request` frame. The S1, S4 and S5 tests pass unchanged. No Pydantic AI change between 2.64.0 and 2.66.0 touched the prototype.

## What changed in `studio/`

| File                          | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytest.ini`                  | path to `services/harness`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `studio_planner.py`           | `ask_user` (a `CallDeferred` tool) is replaced by `ask_user_question`, an output tool: `output_type=[str, ToolOutput(AskedQuestion, name="ask_user_question")]`. An accepted ask ends the run with an `AskedQuestion` carrying dev's argument names (`message`, `candidates`, `options` as `{"id","label"}`, `recommended_spec_id`, `rationale`). The scripted model reads the user's answer from the latest answer prompt; `StudioDeps` gains `ask` (a count), `amend_asks`, `double_ask` and `candidates`.          |
| `node.py`                     | The inner `while True` loop, `_answer_for`, `_answer_text` and `ASK_PREFIX` are gone. An `AskedQuestion` result stores the conversation with its ask id and yields `planner_ask_interrupt({"call_id", "args"})`. The resume finds the answer in `ctx.resume_inputs[ask_id]` and continues the stored conversation with the answer as its next prompt. The node reads an amendment from `node_input`, drafts under it, and writes the next version. Options `ask_shape`, `amendments` and `resume` build the controls. |
| `pai_call.py` (new, 88 lines) | `PaiCall`, a `planner.call.PlannerCall` over a Pydantic AI run's messages.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `wiring.py`                   | `use_pai_planner` passes node options through.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `test_studio.py`              | S2 and S3 follow the new ask; S6 to S8 added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

`minimal/` is imported unchanged (`_relay_live`, `_Entry`, `plan_of`, `PlanProgress`).

An accepted ask now ends the Pydantic AI run the way dev ends the ADK run (`skip_summarization`): through the framework's own end-of-run mechanism, a final-result tool, with no model call spent after the ask. The interrupt id is the ask's tool call id, as dev's `planner_ask_interrupt` sets it, and the stored key `planner:conversation` gains an `ask_id` field.

### Pass lines

Each line has a test and a control. A control is a test that builds the broken variant and asserts the failure it shows.

| Line                             | Result                                                                                                                                                                                                                                                                                                                 | Control, and how it fails                                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| S1 live progress                 | First progress frame at 0.045 to 0.053 s (six runs) while `propose_plan` waits on the client's signal; old figure 0.09 to 0.20 s.                                                                                                                                                                                      | Dev's node with a seam stub: first frame at 3.02 to 3.03 s, after the stub's 3 s gate times out.                  |
| S2 ask and resume                | The ask arrives as one `elicitation.request` after all 10 progress frames. Answering it resumes the stored run: model calls after 1, 3, 5 and 7 messages, once each; the session holds one response to `ask-1`; the same answer posted again gets 404.                                                                 | No new control; S6 and S7 cover the changed parts.                                                                |
| S3 isolation                     | No orchestrator request holds the planner's sentinel, tool names (`propose_plan`, `ask_user_question`), call ids or an interrupt. The positive control shows the same requests hold the brief and the hand-back. The three `Leak` probes still land in the session and miss every request.                             | As before: the positive control, and no failing control (see below).                                              |
| S4 stop and disconnect           | Both end in `propose_plan call=1 cancelled`, leave no task behind, and the next turn runs to `plan.proposed`.                                                                                                                                                                                                          | As before.                                                                                                        |
| S5 records                       | Equal to dev's node with the stub seam, apart from the extra state key `planner:conversation`, `None` at the end.                                                                                                                                                                                                      | As before.                                                                                                        |
| S6 options reach Studio          | The question holds the candidates, then the fixed options, as `{"id","label"}`; `recommended` is set on the recommended candidate; `rationale` shows; free text is off when fixed options exist. The old NOTES reported options and candidates were lost (about 5 translator lines); with dev's interrupt they arrive. | Arguments nested one level down (`payload["args"]`): the question has `options == []` and `allow_free_text` true. |
| S7a two asks in one run          | Ask, answer, ask again, answer, plan: two distinct interrupt ids, responses to `ask-1` and `ask-2`, and no model call or tool body repeated.                                                                                                                                                                           | A node that restarts from the brief on an answer asks `ask-1` again; the second answer never arrives.             |
| S7b ask during an amendment      | After `approve`'s amend, the amendment leg ends on an `elicitation.request`; the answer yields `plan.proposed` at version 2 with a new revision.                                                                                                                                                                       | A node that drops the amendment never asks, and the leg ends on `plan.proposed`.                                  |
| S8 dev's readers on this planner | `CallSession(PaiCall(messages)).elected_spec_ids()` returns the spec the user picked (`{"spec_ghana"}`).                                                                                                                                                                                                               | Before the answer it returns the empty set.                                                                       |
| S8 second ask in one response    | Pydantic AI processes the first ask and skips the second; `PaiCall` yields one `call` step (`ask-1`), as `_session_steps` drops a refused ask.                                                                                                                                                                         | With the skip text unrecognised, the steps hold `ask-1` and `ask-1b`.                                             |

Two behaviours to know:

- With fixed options on the question, dev's interrupt route refuses a free-text answer (`allow_free_text` is false), so the tests answer with `{"kind": "option", "option_id": ...}`. The old prototype offered no options, so a text answer passed.
- A second ask in one response never reaches Studio in this design. Pydantic AI skips it, and there is no tool body to refuse it. Dev refuses it in the tool (`_ask_beside_ask`); the result for the user is the same, one question.

## What #129 buys

The prototype never adapted ADK's `ToolContext`: its tools take Pydantic AI's `RunContext` and read the planner's own messages. So no adapter code is deleted. What it replaced was the prototype's private reading of "what was asked and answered", and the need to write one.

- Removed from `node.py`: `_answer_for`, `_answer_text`, `ASK_PREFIX`, the ask loop and the hand-built `RequestInput`. About 45 lines before; about 12 after (`_entry`, `_ask_id`, the store and one `planner_ask_interrupt` call).
- Removed from `studio_planner.py`: the `_returns`-based call count in `propose_plan` (2 lines, now a read of `PaiCall(...).steps()`), and the deferred `ask_user` tool (6 lines).
- Added: `pai_call.py`, 88 lines (about 50 of code) that make a Pydantic AI history a `PlannerCall`. This is what dev's tools need from this planner: `elected_spec_ids`, `answers_since_referral`, `prices_chosen`, `combination_picks` and `CallSession` all take it with no change. `planner/agent.py`'s `AdkCall` and `_session_steps` (about 120 lines) become the ADK half of a pair.
- `planner/call.py` imports no ADK in its own source, and `PaiCall` imports nothing from ADK either, so the neutral record does the job it was built for. `tests/planner/test_framework_free.py` still guards it.

## What a production version would change

Line counts are against dev's files now.

- `harness/graph/planning.py` (584 lines). The seam await, progress queue and drain (about 40 lines, 223 to 262) become the relay loop and the stored conversation, about 50 lines in the node. The `asked` branch (lines 266 to 281) is kept: it yields `planner_ask_interrupt`, now fed from the run's `AskedQuestion`, so `PENDING_ASK_STATE_KEY` and the `{"asked": ...}` hand-off through the seam are not needed. A pick's replan runs under `switch._unless_stopped` as a task; with the relay that wrapper has to stop the stream it drives, about 20 lines. The docstring's direct-await argument (about 30 lines) is rewritten.
- `harness/graph/workflow.py` (218 lines). About 10 lines: `rerun_on_resume=True` stays, now because the node reruns on the ask's answer.
- `harness/studio_translate.py` (2,722 lines). About 40 lines, down from the 50 in the old estimate: the ask path needs none. What remains is rendering the planner node's `progress` events (about 15 lines, as `wiring.show_planner_progress`), and replacing the branch that reads the ADK planner's own events (`event.author == PLANNER_AGENT_NAME`, lines 1406 to 1415, with `_planner_call_status` and `_proposal_answered`) with progress events carrying a kind and an outcome, about 25 lines.
- The seam (`harness/graph/seams.py`, 140 lines; `harness/planner_seam.py`; `planner/agent.py`). `PlannerModel.__call__` returning a dict after an await no longer fits: the node needs an agent and deps it can stream and resume from a `Conversation`. `on_progress` and the `ctx` hand-through go. `ask_user_question` becomes an output tool whose validator holds `_ask_check`; the planner's other tools take `PaiCall` where they took `AdkCall`. Porting the planner's tools is the bulk of the cost and was not run here.
- Amendments and answers: the node drafts an amendment under `node_input["answer"]["amendment"]` and the prior plan from `readable_plan`, as dev's node does; production also needs the refusal, unreadable-plan and pick branches, which this prototype leaves out.

## Not covered

- S3 has no failing control. Three probes write the planner's text into the session; none reaches a request. The rule that drops them is ADK's context selection, and the stage that drops the empty-branch event was not pinned.
- A pick's replan that asks, a planner-side failure inside Studio, and a crash between the answer and the planner's first event were not run.
- An ask refused by output validation (a retry prompt) is read by `PaiCall` but no test drives it.
- The scripted model sends two asks in one response only at the `PaiCall` level (S8); Studio is not driven through it.
- Answering twice while the first resume still streams (409 by the route's code) was not run.
- The streaming transport is the test's own; no served uvicorn check.
- `ty` was not run on this directory.
- Progress is the prototype's frame shape, a running `activity.step` with `detail` "check n of m". Dev sends no frame for these reports.

## Commands

From the worktree root:

```
uv sync --all-packages
cd research/pai-node-stream/studio
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -s -p no:cacheprovider -k "s1 or s2 or s6"
cd ../minimal
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
```

Use `==2.64.0` for the older version. Lint, from the worktree root:

```
uv run --package harness ruff format research/pai-node-stream
uv run --package harness ruff check research/pai-node-stream
```
