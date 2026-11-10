# The real planner on Pydantic AI inside the ADK planner node

Dev's planner runs as a Pydantic AI agent inside the `planner` workflow node of the real Studio turn, on live Gemini. No "do not port" threshold is breached on its point estimate, and the run does not rule out slower planning: the interval on median wall time reaches +44.8%. It ran against dev at 245c0a44 (google-adk 2.13.0) with pydantic-ai-slim 2.66.0. Nothing here ships. `s0.diff` is the one change to `services/`.

## What was built

| Directory          | What it holds                                                                                          | Result                                                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `S0.md`, `s0.diff` | The planner's tool bodies moved onto the neutral `PlannerCall`, with thin ADK wrappers                 | 10 files, 271 lines added, 1,620 removed, two new modules (`planner/tools.py`, `planner/source_reads.py`); declarations byte-equal; 2,662 planner, seam and context-budget tests pass |
| `port-map.md`      | Every ADK touch point left on the planner path                                                         | Read from code; sizes are estimates                                                                                                                                                   |
| `agent/`           | The planner as a Pydantic AI `Agent` under dev's `PlannerSeam.__call__`, calling the moved tool bodies | 18 tests                                                                                                                                                                              |
| `parity/`          | Both arms over every `planner_entry` task with a scripted google-genai client                          | 70 tasks, 79 plans, 149 cases and 230 requests per arm; 11 controls                                                                                                                   |
| `node/`            | That agent inside the real Studio turn                                                                 | 24 tests against dev's node on the same scripts                                                                                                                                       |
| `node-hard/`       | Budget, cache, error, resume, Stop and tracing cases, with the fixes they needed                       | 57 tests                                                                                                                                                                              |
| `live-entry/`      | `planner_entry` on live Gemini, arms paired and interleaved                                            | 420 entries                                                                                                                                                                           |
| `live-turns/`      | Full Studio turns on live Gemini and dev data services                                                 | 40 turns, and 10 reruns of one question                                                                                                                                               |

## What held

- Offline, every case ends the same way on both arms, and every request matches ADK's apart from six wire differences that the checks pin exactly. 74 of the 230 requests follow an answered ask.
- Inside the Studio turn, frames, requests and stored records equal dev's for a planned turn, an ask and its resume, an amendment, an ask during an amendment, a refusal replan, a pick's replan and Stop. The ledger comparison is between empty ledgers, since no scripted turn executes.
- Live, 420 entries: first proposals accepted 59.5% against ADK's 60.0% (difference -0.5 points, interval -6.7 to +5.7); cached share 88.6% against 88.7%; first chunk 3.7 s on both. Wall time less model stream time is 0.02 s at the median against ADK's 0.06 s, and stream time per completed call is 4.25 s against 4.18 s, so the framework adds no time.
- Live Studio turns: 40 of 40 ran to the end, and no failure class traced to the planner node. Each question ran once per arm, so the stage counts (a plan executed in 17 of 20 turns against dev's 16) check for failures and do not compare the arms.
- Planner progress reports are available 4.0 s after the planner starts, at the median, against 24.4 s on dev, where they arrive with the proposal. Dev's translator draws no frame from them today, so the figure is taken with the translator patched to draw them. The gap comes from how the node routes reports, and ADK's own planner events already stream live, so the same frames may be reachable on ADK.

## The thresholds

Fixed before the live run.

- A failure class seen only on Pydantic AI: none.
- Cached share down more than 2 points: no (-0.1, interval -1.0 to +0.7).
- Median planner wall time up more than 10%: not breached on the point estimate (+9.6%), not excluded by the interval (-17.9% to +44.8%).

The model-time budget closed 17 Pydantic AI entries against 10 (sign test p = 0.17), and kind pass is 3.8 points lower (interval -9.0 to +1.0). Neither is significant, and a production gate should watch both. The live entry run used `agent/` without the fixes below, so the hardened build has no live measurement at entry level. None of the 17 closed entries holds a proposal after the cut.

## What differed and how it was closed

Each fix is proved in `node-hard/hardened.py` against dev's arm, with a control.

- A budget cut mid-stream ran a tool call already delivered. ADK aggregates the stream, so no tool runs. Fixed by holding chunks until the budget wrapper returns.
- A second unknown tool name, an empty stream or a blocked response failed the turn; dev ends the run `plan_not_ready`. The tool-not-found text also differed. Fixed with ADK's text, a high retry count and a catch of `UnexpectedModelBehavior`.
- An ask paused under one planner and answered under the other failed for two-ask turns in two of four orders. Fixed by rebuilding the planner's record from the session's planner events on both paths.
- A Stop lost tool state writes made before it. Fixed with a state delta per response.
- No generation, tool or agent spans. Fixed with Pydantic AI's `Instrumentation` and a conversation id. The generation span is named `chat <model>` where dev's is `generate_content <model>`.
- A response with no parts retried; ADK ends the run. Fixed in `agent/`. ADK's conversion of typed list arguments and its answer to a missing argument are reproduced through two private methods of the ADK tool object.
- The ask check moved in `s0.diff` first read the call record's steps, which cannot tell an earlier unanswered ask from a second ask in one response. `PlannerCall.response_calls()` gives the calls of the response holding the call.

## The design this points to

`PlannerSeam.__call__` stays. Where it called `ctx.run_node` on ADK's `LlmAgent`, it runs a function node named `dfo_planner`, branched as ADK's planner is, which drives the Pydantic AI loop and yields one ADK event per model response and tool result. The translator and the `switch` node's `revise` read what they read today. The session's planner events are the one stored record on both paths, so a turn can pause under one planner and resume under the other.

Each planner run starts from the user's message and the brief, as dev's `single_turn` run does, so the model reads no rebuilt history and thought signatures and the wire differences do not cross a resume. Only the tool readers read the record. The prototype rebuilds it as Pydantic AI messages and reads steps back out of them; production should give the readers `_session_steps` over the session events plus the live run's own steps, with no message record.

Writing ADK-shaped events ties the node to `LlmAgent`'s event shape for as long as rollback to the ADK planner must work. Once that path is deleted the shape is ours, and the coupling to ADK's log sits in two places, the event builders and `_session_steps`.

## Proposed production slices

Slices 1 to 5 ship on ADK alone with no behaviour change.

1. `pydantic-ai-slim` in the harness lock beside ADK's google-genai pin, with image size and import time checked. A lock conflict blocks everything after it.
1. Tool bodies on `PlannerCall`, with `response_calls()` (`s0.diff`).
1. Declarations and instruction as data that both paths read, with ADK's argument conversion and missing-argument answer as code of ours. Gates: equality with what ADK generates today, and no private ADK name on the path.
1. The seam's four callbacks as neutral hooks. Its tool callbacks are typed on ADK's `BaseTool` and `ToolContext`.
1. The record contract: `_session_steps` public, the event builders added and inert, a test that the readers read the same from both, and a test that every state key a tool body writes survives a resume. `_persist_durable`'s key list omits `translator:*`, `planner:alignments` and the skill page texts. This follows slice 4, which touches the same seam file.
1. The Pydantic AI runner as the `dfo_planner` function node behind a flag at `harness/wiring.py`. It carries the Pydantic AI adapters over the existing `bounded` and `PrefixCache` with the hold of streamed chunks, tool state writes recorded on `call.state` and sent as a delta per response, the error endings with the tool-not-found text as our own, and no private ADK name. Gates: `parity/`, the cross-path resume tests, and `node/`'s frames-and-readers tests kept for as long as the flag exists.
1. Tracing, and the eval and scripted consumers (`evals/planner_entry`, `evals/agents`, `harness/scripted.py`), beside slice 6.
1. A live gate with the three thresholds, budget closures and kind pass watched, and the model's response logged when a run ends with no tool call. Then the default flips and the ADK planner path is deleted.

Slices 2 and 3 can run together after slice 1; slice 3 needs slice 2's module layout only for imports. Live progress frames in the translator are a separate product change. If dev's own planner events can drive them, they ship on ADK and do not depend on the port.

The flip needs no translator on neutral planner events. That is needed only when the turn itself leaves ADK.

## Before slice 6

- On the production session service: events and stored bytes per turn with the per-response events and deltas, against dev; a second turn of one conversation; a kill mid-planning with a resume on a new instance.
- The 70 tasks at 3 repetitions again, on the hardened build, pooled with this run. That narrows kind pass to about 4.5 points and shows whether 17 against 10 budget closures is a difference.
- An inventory of private names relied on in ADK and Pydantic AI, with a pinned test for each Pydantic AI one kept. `ctx.event_author`'s setter needs confirming as public.

## Not covered

- Acceptance parity. The live interval is about 6 points against the eval's noise of about 5.
- `revise` through the `switch` node in a Studio turn. No recorded task offers a scope choice. The test reads the steps `revise` reads.
- A second turn of one conversation, several instances, and a kill mid-planning on this node.
- One live turn ended its first planner run with no plan after two refused proposals and a model response with no tool call. Ten reruns of the question did not reproduce it on either arm, and dev's node ends that response the same way. The response's content was not stored.
- The per-response state delta deep-copies session state; its cost was not measured.

## A finding outside the port

On both arms the 90 s model-time budget closed 5% to 8% of live entries. In the live turns every cut had one signature: the planner's second model call, after it reads a skill, thinks for about 10,000 tokens until the budget cancels it at 71 to 84 s.

## Rerunning

The run records (`*.jsonl.gz`) and `s0.diff.gz` are gzipped; unzip them before rerunning an analysis script or applying the diff. The `RESULTS.md` files in `live-entry/` and `live-turns/` are their scripts' output after Markdown formatting, so each script's `--check` reports formatting differences only.

Each directory's `NOTES.md` or `RESULTS.md` has its commands. From the worktree root, with `E` for `PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness`:

```
cd research/pai-planner-live/<agent|node|node-hard|live-entry> && E pytest . -q -p no:cacheprovider
E python research/pai-planner-live/parity/parity.py
```
