# The real planner on Pydantic AI inside the ADK planner node

Dev's planner runs as a Pydantic AI agent inside the `planner` workflow node of the real Studio turn, on live Gemini and on a Postgres session service. Over 840 paired live entries no "do not port" threshold is breached on its point estimate, and the runs do not rule out slower planning: the interval on median wall time reaches +32.8%. It ran against dev at 245c0a44 (google-adk 2.13.0) with pydantic-ai-slim 2.66.0. Nothing here ships; `s0.diff` is the one change to `services/`.

## What was built

| Directory                            | What it holds                                                                                                 | Result                                                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `S0.md`, `s0.diff`                   | The planner's tool bodies moved onto the neutral `PlannerCall`, with thin ADK wrappers and `response_calls()` | 10 files, 271 lines added, 1,620 removed, two new modules; declarations byte-equal; 2,662 planner, seam and context-budget tests pass |
| `port-map.md`                        | Every ADK touch point left on the planner path                                                                | Read from code; sizes are estimates                                                                                                   |
| `agent/`                             | The planner as a Pydantic AI `Agent` under dev's `PlannerSeam.__call__`, calling the moved tool bodies        | 18 tests                                                                                                                              |
| `parity/`                            | Both arms over every `planner_entry` task with a scripted google-genai client                                 | 70 tasks, 79 plans, 149 cases and 230 requests per arm; 11 controls                                                                   |
| `node/`                              | That agent inside the real Studio turn                                                                        | 24 tests against dev's node on the same scripts                                                                                       |
| `node-hard/`                         | Budget, cache, error, resume, Stop and tracing cases                                                          | 57 tests                                                                                                                              |
| `session-service/`                   | The same turns on Postgres 18 through ADK's `DatabaseSessionService`                                          | 16 tests, both arms                                                                                                                   |
| `private-names.md`, `private-names/` | Every private ADK, Pydantic AI and harness name on the path, with pins                                        | 14 pins, passing on 2.64.0 and 2.66.0                                                                                                 |
| `live-entry/`                        | `planner_entry` on live Gemini, arms paired and interleaved, two runs                                         | 840 entries                                                                                                                           |
| `live-turns/`                        | Full Studio turns on live Gemini and dev data services                                                        | 40 turns, and 10 reruns of one question                                                                                               |

## What held

- Offline, every case ends the same way on both arms, and every request matches ADK's apart from six wire differences that the checks pin exactly. 74 of the 230 requests follow an answered ask.
- Inside the Studio turn, frames, requests and stored records equal dev's for a planned turn, an ask and its resume, an amendment, an ask during an amendment, a refusal replan, a pick's replan and Stop. The ledger comparison is between empty ledgers, since no scripted turn executes.
- On Postgres, a turn with a skill read, an ask, its answer and a proposal writes 26 event rows on both arms, 203,559 bytes of events against dev's 204,252, and identical state. A resume from a fresh process that shares only the database equals an uninterrupted run. A SIGKILL mid-planning is handled identically on both arms: the next request is refused until the 45 s lease lapses, then a new turn starts; a turn killed after an answered ask resumes in place. A second turn of a conversation starts with a clean planner record. An ask paused under one planner and answered under the other lands the plan in both orders.
- Live, two runs of 420 entries, the second on the build with every fix below. Pooled paired differences, Pydantic AI less ADK, with 95% intervals: first proposals accepted +2.4 points (-2.9 to +7.9); kind pass -0.5 points (-4.5 to +3.6); cached share +0.1 points (-0.5 to +0.8). Time outside model calls is 0.02 s at the median against ADK's 0.06 s, so the framework adds no time.
- Live Studio turns: 40 of 40 ran to the end with approval and `execute`, and no failure class traced to the planner node. Each question ran once per arm, so the counts check for failures and do not compare the arms.

## The thresholds

Fixed before the first live run.

| Threshold                                 | First run | Second run | Pooled                   |
| ----------------------------------------- | --------- | ---------- | ------------------------ |
| A failure class seen only on Pydantic AI  | none      | none       | none                     |
| Cached share down more than 2 points      | -0.1      | +0.4       | +0.1 (-0.5 to +0.8)      |
| Median planner wall time up more than 10% | +9.6%     | +2.3%      | +6.9% (-13.5% to +32.8%) |

The third is not breached on any point estimate and not excluded by any interval. The model-time budget closed 32 Pydantic AI entries against 25 over both runs (sign test p = 0.38).

## What drives planner time on both arms

About a third of entries hold a model call that reasons for 2,000 tokens or more: 144 of 420 on Pydantic AI and 123 of 420 on ADK (+5.0 points, -2.9 to +13.1). Those entries take 59 to 62 s at the median against 10 to 11 s for the rest. The reasoning is triggered by the descriptions of the planner's skill tools, on dev's planner as much as this one, and it accounts for the budget closures and for the width of every wall-time interval above. The second run's gap on this share (+8.6 points, -1.4 to +19.5) is the one figure that leans against the port; it should be measured again once the planner's skill tool carries a short description, which removes most such calls.

## What differed from dev and how it was closed

Each is the behaviour of `agent/` and `node/`, tested in `node-hard/` against dev's arm with a control.

- A budget cut mid-stream ran a tool call already delivered; ADK aggregates the stream, so no tool runs. Chunks are held until the budget wrapper returns.
- A second unknown tool name, an empty stream or a blocked response failed the turn; dev ends the run `plan_not_ready`. The tool-not-found text also differed. The text is written out, the retry count is high and `UnexpectedModelBehavior` is caught.
- An ask paused under one planner and answered under the other failed for two-ask turns in two of four orders. The tool readers take earlier steps from the session's planner events.
- A Stop lost tool state writes made before it. Writes are recorded and sent as a state delta per response.
- No generation, tool or agent spans. Pydantic AI's `Instrumentation` and a conversation id give them; the generation span is named `chat <model>` where dev's is `generate_content <model>`.
- A response with no parts retried; ADK ends the run. ADK's conversion of typed list arguments and its answer to a missing argument are reproduced through two private methods of the ADK tool object.
- The ask check moved in `s0.diff` first read the call record's steps, which cannot tell an earlier unanswered ask from a second ask in one response. `PlannerCall.response_calls()` gives the calls of the response holding the call.

## The design this points to

`PlannerSeam.__call__` stays. Where it called `ctx.run_node` on ADK's `LlmAgent`, it runs a function node named `dfo_planner`, branched as ADK's planner is, which drives the Pydantic AI loop and yields one ADK event per model response and tool result. The translator and the `switch` node's `revise` read what they read today. The session's planner events are the one stored record on both paths: the tool readers take `_session_steps` over them plus the live run's own steps, and no Pydantic AI message history is stored. Each planner run starts from the user's message and the brief, as dev's `single_turn` run does, so thought signatures and the wire differences do not cross a resume.

Writing ADK-shaped events ties the node to `LlmAgent`'s event shape for as long as rollback to the ADK planner must work. Once that path is deleted the shape is ours, and the coupling to ADK's log sits in two places, the event builders and `_session_steps`.

Dev already shows planner progress live: ADK streams the planner's own events while it runs and the translator draws its status lines from them. The port changes nothing the user sees there.

## Proposed production slices

Slices 1 to 5 ship on ADK alone with no behaviour change.

1. `pydantic-ai-slim` in the harness lock beside ADK's google-genai pin, with image size and import time checked. A lock conflict blocks everything after it.
1. Tool bodies on `PlannerCall`, with `response_calls()` (`s0.diff`).
1. Declarations and instruction as data that both paths read, with ADK's argument conversion and missing-argument answer as code of ours (a `TypeAdapter` over the annotations and `inspect.signature`). Gates: equality with what ADK generates today, and no private ADK name on the path.
1. The seam's four callbacks as neutral hooks. Its tool callbacks are typed on ADK's `BaseTool` and `ToolContext`.
1. The record contract: `_session_steps` public, the event builders added and inert, a test that the readers read the same from both, and a test that every state key a tool body writes is stored. `_persist_durable`'s key list omits `translator:*`, `planner:alignments` and the skill page texts. This follows slice 4, which touches the same seam file.
1. The Pydantic AI runner as the `dfo_planner` function node behind a flag at `harness/wiring.py`, with the adapters over `bounded` and `PrefixCache`, the hold of streamed chunks, recorded state writes, the error endings, and the pins in `private-names/`. Gates: `parity/`, the cross-path resume tests, and `node/`'s frames-and-readers tests kept for as long as the flag exists.
1. Tracing, and the eval and scripted consumers (`evals/planner_entry`, `evals/agents`, `harness/scripted.py`), beside slice 6.
1. A live gate with the three thresholds, the long-reasoning share, budget closures and kind pass watched, and the model's response logged when a run ends with no tool call. Then the default flips and the ADK planner path is deleted.

Slices 2 and 3 can run together after slice 1. A translator that reads neutral planner events is needed only when the turn itself leaves ADK.

## Private names

ADK: three private names, all removable (`_get_declaration` has a public route through `process_llm_request`; `_preprocess_args` and `_get_mandatory_args` become code of ours). Pydantic AI: 15 rows, six with no public alternative: `GoogleModel._build_content_and_config` (removable by moving the prefix lookup into the client), the no-merge of consecutive requests whose `instructions` differ, the unknown-tool text and its retry accounting, the ending of an empty response when the output type allows `None`, the layout of `RunContext.messages`, and the client call sitting inside the `chat` span. Six of the 14 pins were shown failing under a perturbed library; the other eight were not. Nothing on the path changed between 2.64.0 and 2.66.0.

## Not covered

- Acceptance parity. The pooled interval is about 5 points, the size of the eval's noise.
- `revise` through the `switch` node in a Studio turn; no recorded task offers a scope choice. The test reads the steps `revise` reads.
- The Cloud SQL connector path, and a lease takeover by a live second instance (the test uses a clock offset).
- One live turn ended its first planner run with no plan after two refused proposals and a model response with no tool call. Ten reruns of the question did not reproduce it on either arm, and dev's node ends that response the same way. The response's content was not stored.
- The orchestrator on Pydantic AI on live Gemini.

## Rerunning

The run records (`*.jsonl.gz`) and `s0.diff.gz` are gzipped; unzip them before rerunning an analysis script or applying the diff. The `RESULTS.md` files in `live-entry/` and `live-turns/` are their scripts' output after Markdown formatting, so each script's `--check` reports formatting differences only. `session-service/` needs Docker.

Each directory's `NOTES.md` or `RESULTS.md` has its commands. From the worktree root, with `E` for `PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness`:

```
cd research/pai-planner-live/<agent|node|node-hard|live-entry|private-names> && E pytest . -q -p no:cacheprovider
E python research/pai-planner-live/parity/parity.py
```
