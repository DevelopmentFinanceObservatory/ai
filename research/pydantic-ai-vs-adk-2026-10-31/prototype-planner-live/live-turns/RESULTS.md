# Live Studio turns on dev's ADK planner and the hardened Pydantic AI planner

## Verdict

The hardened Pydantic AI planner node ran 20 full Studio turns on live Gemini and the dev data services and all 40 turns reached `done`. One turn is open: question 18 on the Pydantic AI arm ended its first planner run with no plan and recovered on a second run, and its cause is not in the record. Pydantic AI answered 18 of 20 questions with data and dev's ADK planner 17 of 20; the rest of each arm ended in the same budget cut. Plans are not compared: two live runs of one planner also differ, and the runs here show it.

The Pydantic AI arm showed one clear difference in timing. Its first planner progress frame arrives a median 4.0 s after the planner starts, against 24.4 s on dev's arm, where the reports are held until the draft is done. Planner time, total working time and token use sit within the spread that one run per question and arm leaves.

Each turn ran once per arm, the arm that goes first alternating by question. Questions are numbered from 0 in `questions.txt`. A turn is the real harness (`harness.server.create_app`) served in process, with live Gemini and the dev data services through `gcloud run services proxy`. The `pai` arm replaces only the planner that `build_planner` returns with `hardened.hard_planner`; the orchestrator, translator, `execute` node and reply are dev's code on both arms. The driver answers every question the harness asks with the recommended option (else the first) and approves every proposed plan. Both arms draw the planner's progress reports as running status frames (`show_planner_progress` in `driver.py`), since dev's translator draws none; without that the first progress frame would be the same frame on both arms.

## What the figures say

- Stages. Both arms ran the planner on 19 of 20 turns (question 0 is answered by the orchestrator alone). The arms differ in how often the orchestrator asked a question before the planner started (7 on dev's arm, 3 on Pydantic AI's). Every ask came before the planner's first frame, so it is the orchestrator's, which is dev's code on both arms. A pair of earlier turns on question 1 (`runs/smoke`) shows the same swing within one arm: dev's arm asked nothing the first time and one question the second.
- Budget cut. Five turns ended with "This request used all the work allowed for one question": three on dev's arm (questions 12, 15, 17) and two on Pydantic AI's (questions 2, 6). All five show one signature: the planner's second model call, made after a `get_skill` call, thinks for 9 900 to 11 600 tokens until the turn's model-time budget cancels it at 71 to 84 s. The cut is the same on both arms, and the planner's thinking is the cause, not the node.
- Failed steps other than the planner's. Every other failed step is a chart-making, card-removal or search step in the orchestrator's own tool calls against the presentation and data services, and every turn that had one still ended with an answer. Four turns on dev's arm have one (questions 9, 10, 12, 14) and five on Pydantic AI's (1, 8, 10, 15, 19). The failing steps run after the plan, on results the plan shaped, and the plans differ, so this run cannot attribute the gap to the planner node.
- Time. The progress frame figure is the node's design: Pydantic AI's node yields the planner's reports as they happen, dev's node relays them after the draft. Planner time from the planner's first status frame to the proposal has medians of 21.2 s (Pydantic AI) and 24.4 s (dev), with both arms' ranges ending near 86 s, which are the long thinking calls. Total working time is 82.9 s and 74.8 s. With 20 turns per arm and 1 turn per question, the differences in planner time and working time are inside the run-to-run spread.
- Tokens. Dev's arm made 46 planner calls and Pydantic AI's 48; the planner's prompt tokens are 771 000 and 835 000, its thinking tokens 60 000 and 49 000. The orchestrator, which is dev's code on both arms, made 95 and 117 calls. Four calls could not be attributed to an agent and are listed as `other`.

## Failure classes seen only on the Pydantic AI arm

All five are one turn each, and each turn ended with an answer.

- Question 1, "That chart would draw more series than one chart can hold" (twice, with "The card could not be drawn as asked" once). The plan's two totals held 510 and 59 rows, where dev's plan for the same question had 10 and 1, all drawn from one 713-series fetch. The orchestrator asked for a chart the presentation service refused for its series count, a limit downstream of the plan. A later try drew a table of the 24 largest of 53 series.
- Question 10, "No card matched the one that was to be taken off". The orchestrator removed a card it had drawn, after two refused cards ("The card could not be drawn as asked"). The plan ranked the top donor in a 5-row result, where dev's arm planned a one-row result per year and drew five stat cards. The failing step is the orchestrator's card tidy-up. The turn ended with the cards drawn.
- Question 15, "The data behind this card does not hold a value it was asked to show" and "The data does not cover what was asked for". Both refusals came from the orchestrator's first two chart drafts, and the third drew. Dev's arm on this question ended in a budget cut, so there is no plan to set beside it.
- Question 18, "No plan could be made: A plan for what was asked could not be completed". The planner called `propose_plan` three times, the checking line closed without its usual detail after the first two, and the run ended without a plan. The harness then started the planner again and the second run proposed a plan, so the user saw an answer. The cause is not in the record: no ERROR was logged and the seam's refusal text is not kept. No turn on dev's arm shows it, and one occurrence is not enough to say whether it belongs to the node.

## Question 18 rerun

Question 18 was rerun five times on each arm (`q18.py`, records in `runs/q18.jsonl`), with the arms alternating. The driver wraps `PlannerSeam._propose`, `_track_tool_call` and `_on_planner_tool_error`, which both arms share, and records each `propose_plan` answer as the model read it, the run's tool errors, the tracker's final state and how the seam ended the run.

Neither arm's first planner run ended with no plan from the planner itself: 0 of 5 on dev's arm and 0 of 5 on Pydantic AI's. The one first run on either arm that returned no plan was a Pydantic AI run that the turn's model-time budget closed after 85.9 s with no tool call, the long-thinking pattern of the budget cuts above. No run ended `plan_not_ready` or `budget_exhausted`, and the proposals per run stayed at one to three, against a cap of four.

The only refusal code the model saw on either arm is `invalid_input / invalid_plan`, the plan's own validity check, which runs before translation: 5 answers on dev's arm and 3 on Pydantic AI's, in 3 turns on each. The most frequent message is the same on both arms: "states among='named', but axis None carries no narrowing filter; a named rank needs a visible FilterRequirement" (dev's arm twice, Pydantic AI's three times), and the model corrected it on the next call every time. No translation referral, stall or user-decided ask arose.

The original Pydantic AI turn cannot be replayed, but its stored frames fix the shape. It made two `propose_plan` calls (progress frames at 23.0 s and 30.6 s, the checking line closing without detail each time, as for a refused proposal), then a third model call of 10.9 s with no tool call, and the run ended 7 ms later with the sentence for `plan_not_ready` ("A plan for what was asked could not be completed"). Two proposals is below the four-round cap, so the round budget was not spent, and a stall stop would come before the third model call, not after it. The run ended because a model response made no function call, which dev's node also ends as `plan_not_ready`. Why that response made none (text, an empty or a blocked response) is not stored. Both arms share the ending, and one occurrence in 20 q18 turns is not evidence that the Pydantic AI node causes it.

<!-- q18 -->

Turns recorded: 5 on adk, 5 on pai.

First planner run of each turn, by how the seam ended it

| ending                                | adk | pai |
| ------------------------------------- | --- | --- |
| budget closed (turn_model_time_spent) | 0   | 1   |
| plan                                  | 5   | 4   |

Every planner run

| ending                                | adk | pai |
| ------------------------------------- | --- | --- |
| budget closed (turn_model_time_spent) | 0   | 1   |
| plan                                  | 5   | 4   |

`propose_plan` answers as the model read them

| outcome / code               | adk | pai |
| ---------------------------- | --- | --- |
| invalid_input / invalid_plan | 5   | 3   |
| ok                           | 5   | 4   |

| measure                                           | adk              | pai              |
| ------------------------------------------------- | ---------------- | ---------------- |
| turns with at least one refused proposal          | 3                | 3                |
| correction rounds per planner run, median (range) | 1.0 (0.0 to 2.0) | 1.0 (0.0 to 1.0) |

<!-- /q18 -->

## Tables

Recomputed from `runs/full/records.jsonl` by `analysis.py`; `python analysis.py --check` compares this block with the records. Time is in seconds from the start of the turn's stream. Planner start is the first "Working on a plan" or "Revising the plan" frame. Planner time runs from there to `plan.proposed`, or to the planner's last closing frame when no plan was proposed. A failure class is the first sentence of a failed step's detail, or the planner's own failed step.

<!-- tables -->

Turns recorded: 20 on adk, 20 on pai.

### Stages reached

| stage                            | adk | pai |
| -------------------------------- | --- | --- |
| planner ran                      | 19  | 19  |
| asked a question                 | 7   | 3   |
| asked before the planner started | 7   | 3   |
| asked after the planner started  | 0   | 0   |
| plan proposed                    | 16  | 17  |
| plan executed                    | 16  | 17  |
| answered with data               | 17  | 18  |

### Final outcome

| outcome            | adk | pai |
| ------------------ | --- | --- |
| answered with data | 17  | 18  |
| could not answer   | 0   | 0   |
| budget cut         | 3   | 2   |
| error              | 0   | 0   |

### Failure classes (turns affected)

| class                                                                                      | adk | pai | example |
| ------------------------------------------------------------------------------------------ | --- | --- | ------- |
| The card could not be drawn as asked, and asking for it again may work                     | 1   | 4   | adk q14 |
| planner budget cut                                                                         | 3   | 2   | adk q12 |
| This step could not be completed                                                           | 2   | 0   | adk q9  |
| A service this step needs did not complete the request, and trying again in a moment may w | 1   | 0   | adk q10 |
| No card matched the one that was to be taken off, and asking for it again may work         | 0   | 1   | pai q10 |
| That chart would draw more series than one chart can hold, so ask for fewer of them        | 0   | 1   | pai q1  |
| The data behind this card does not hold a value it was asked to show                       | 0   | 1   | pai q15 |
| The data does not cover what was asked for, such as the period or the breakdown            | 0   | 1   | pai q15 |
| no plan made                                                                               | 0   | 1   | pai q18 |

Classes seen only on pai: q10: No card matched the one that was to be taken off, and asking for it again may work; q15: The data behind this card does not hold a value it was asked to show; q15: The data does not cover what was asked for, such as the period or the breakdown; q18: no plan made; q1: That chart would draw more series than one chart can hold, so ask for fewer of them

### Time, median (range)

| measure                                      | adk                        | pai                        |
| -------------------------------------------- | -------------------------- | -------------------------- |
| planner start to first progress frame (s)    | 24.4 (8.6 to 88.0), n=19   | 4.0 (3.4 to 9.7), n=19     |
| planner time (s)                             | 24.4 (8.6 to 88.0), n=19   | 21.2 (8.3 to 86.6), n=19   |
| planner time, turns that proposed a plan (s) | 20.6 (8.6 to 86.2), n=16   | 18.9 (8.3 to 86.6), n=17   |
| total working time (s)                       | 74.8 (29.5 to 144.8), n=20 | 82.9 (20.8 to 121.4), n=20 |

### Model calls and tokens, summed over all turns

| agent        | measure         | adk     | pai     |
| ------------ | --------------- | ------- | ------- |
| orchestrator | calls           | 95      | 117     |
| orchestrator | prompt_tokens   | 1254175 | 1671808 |
| orchestrator | cached_tokens   | 1090425 | 1360100 |
| orchestrator | output_tokens   | 10735   | 12456   |
| orchestrator | thinking_tokens | 3072    | 7833    |
| planner      | calls           | 46      | 48      |
| planner      | prompt_tokens   | 770874  | 835113  |
| planner      | cached_tokens   | 691560  | 738048  |
| planner      | output_tokens   | 12757   | 13577   |
| planner      | thinking_tokens | 60443   | 49442   |
| other        | calls           | 3       | 1       |
| other        | prompt_tokens   | 44221   | 15905   |
| other        | cached_tokens   | 38818   | 11725   |
| other        | output_tokens   | 710     | 157     |
| other        | thinking_tokens | 0       | 0       |

### Every turn

| question | arm | outcome            | working s | first progress s | planner s | failure classes                                                                                                                                             |
| -------- | --- | ------------------ | --------- | ---------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| q0       | adk | answered with data | 38.168    |                  |           |                                                                                                                                                             |
| q0       | pai | answered with data | 20.841    |                  |           |                                                                                                                                                             |
| q1       | adk | answered with data | 54.936    | 13.249           | 13.256    |                                                                                                                                                             |
| q1       | pai | answered with data | 82.968    | 5.364            | 39.533    | That chart would draw more series than one chart can hold, so ask for fewer of them; The card could not be drawn as asked, and asking for it again may work |
| q2       | adk | answered with data | 72.175    | 30.993           | 31.004    |                                                                                                                                                             |
| q2       | pai | budget cut         | 93.002    | 3.385            | 78.891    | planner budget cut                                                                                                                                          |
| q3       | adk | answered with data | 29.524    | 8.551            | 8.559     |                                                                                                                                                             |
| q3       | pai | answered with data | 40.555    | 3.84             | 12.172    |                                                                                                                                                             |
| q4       | adk | answered with data | 130.413   | 20.311           | 20.318    |                                                                                                                                                             |
| q4       | pai | answered with data | 56.192    | 4.031            | 12.494    |                                                                                                                                                             |
| q5       | adk | answered with data | 87.121    | 66.188           | 66.198    |                                                                                                                                                             |
| q5       | pai | answered with data | 33.192    | 6.888            | 12.25     |                                                                                                                                                             |
| q6       | adk | answered with data | 110.388   | 9.749            | 9.756     |                                                                                                                                                             |
| q6       | pai | budget cut         | 107.774   | 9.734            | 84.559    | planner budget cut                                                                                                                                          |
| q7       | adk | answered with data | 39.737    | 11.994           | 12.002    |                                                                                                                                                             |
| q7       | pai | answered with data | 82.891    | 3.424            | 23.729    |                                                                                                                                                             |
| q8       | adk | answered with data | 108.539   | 76.789           | 76.796    |                                                                                                                                                             |
| q8       | pai | answered with data | 48.252    | 4.046            | 21.2      | The card could not be drawn as asked, and asking for it again may work                                                                                      |
| q9       | adk | answered with data | 62.194    | 25.054           | 25.062    | This step could not be completed                                                                                                                            |
| q9       | pai | answered with data | 113.021   | 4.342            | 14.431    |                                                                                                                                                             |
| q10      | adk | answered with data | 119.014   | 86.158           | 86.17     | A service this step needs did not complete the request, and trying again in a moment may w                                                                  |
| q10      | pai | answered with data | 94.78     | 5.416            | 46.49     | No card matched the one that was to be taken off, and asking for it again may work; The card could not be drawn as asked, and asking for it again may work  |
| q11      | adk | answered with data | 53.425    | 14.408           | 14.414    |                                                                                                                                                             |
| q11      | pai | answered with data | 56.484    | 3.642            | 13.569    |                                                                                                                                                             |
| q12      | adk | budget cut         | 114.785   | 76.223           | 76.224    | This step could not be completed; planner budget cut                                                                                                        |
| q12      | pai | answered with data | 85.233    | 4.446            | 16.03     |                                                                                                                                                             |
| q13      | adk | answered with data | 52.771    | 20.778           | 20.786    |                                                                                                                                                             |
| q13      | pai | answered with data | 30.729    | 3.621            | 9.819     |                                                                                                                                                             |
| q14      | adk | answered with data | 77.423    | 49.471           | 49.479    | The card could not be drawn as asked, and asking for it again may work                                                                                      |
| q14      | pai | answered with data | 121.439   | 4.481            | 86.646    |                                                                                                                                                             |
| q15      | adk | budget cut         | 103.674   | 78.243           | 78.243    | planner budget cut                                                                                                                                          |
| q15      | pai | answered with data | 88.562    | 3.569            | 18.901    | The data behind this card does not hold a value it was asked to show; The data does not cover what was asked for, such as the period or the breakdown       |
| q16      | adk | answered with data | 144.795   | 24.421           | 24.429    |                                                                                                                                                             |
| q16      | pai | answered with data | 103.2     | 3.643            | 28.497    |                                                                                                                                                             |
| q17      | adk | budget cut         | 105.154   | 88.001           | 88.001    | planner budget cut                                                                                                                                          |
| q17      | pai | answered with data | 35.592    | 3.651            | 8.307     |                                                                                                                                                             |
| q18      | adk | answered with data | 45.616    | 11.807           | 11.823    |                                                                                                                                                             |
| q18      | pai | answered with data | 94.823    | 7.884            | 66.438    | no plan made                                                                                                                                                |
| q19      | adk | answered with data | 35.384    | 13.306           | 13.314    |                                                                                                                                                             |
| q19      | pai | answered with data | 81.472    | 4.824            | 55.105    | The card could not be drawn as asked, and asking for it again may work                                                                                      |

<!-- /tables -->
