# Planner entry eval: baseline

30 runs over 10 tasks on gemini-3.8-flash; baseline at 1568eadb. Rates count runs; tasks with a first-proposal refusal are scored only on whether they ended in that refusal and an ask. Runs that errored or missed their recording are left out of every figure; attempts count every entry started, retries included. Unasked units count dataset requirements stating a unit in first proposals of tasks whose question and brief name none. Tokens are means over runs whose every model call finished, input including cached.

## Per arm

| Arm      | Runs scored (attempts) | Errored | Missed the recording | First proposal accepted | Kind pass, runs | Kind pass, tasks | Refusal tasks ending right | Unasked units | Model calls | Input tokens (cached) | Output tokens | Thinking tokens | Wall s: mean, p90, max | Closed by the turn budget |
| -------- | ---------------------- | ------- | -------------------- | ----------------------- | --------------- | ---------------- | -------------------------- | ------------- | ----------- | --------------------- | ------------- | --------------- | ---------------------- | ------------------------- |
| baseline | 30 (30)                | 0       | 0                    | 14/27 (52%)             | 24/27 (89%)     | 8/9 (89%)        | 0/3 (0%)                   | 0 of 41       | 2.67        | 45,680 (41,093)       | 654           | 0               | 10.3, 16.1, 39.2       | 0                         |

## How entries ended

| Arm      | accepted | asked_user |
| -------- | -------- | ---------- |
| baseline | 29       | 1          |

## Why first proposals were refused

| Arm      | Codes                                                             |
| -------- | ----------------------------------------------------------------- |
| baseline | invalid_plan rule 8, invalid_plan shape 3, measure_fixed_by_pin 2 |

## Years

What accepted plans do with the years of their data. A plan does as its task expects when its period reduction is the task's `period_reduction`, read with the asks the task marks `across_years`; a stored plan that no longer loads counts as a miss, and the second column counts them. The third column is over the accepted plans read of tasks that mark no ask `across_years`; `propose_plan` refuses a period reduction there, so only a run of a revision without the rule counts any. The rule applied to a task when, in any arm, it refused a proposal or accepted a period reduction for asks marked across years; rounds are `propose_plan` calls up to the accepted one.

| Arm      | Years as the task expects | Accepted plans that no longer load | No ask across years: accepted plans holding a period reduction | First proposals refused `period_reduction_unstated` | Rounds to acceptance where the rule applied |
| -------- | ------------------------- | ---------------------------------- | -------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| baseline | 29/29 (100%)              | 0/29 (0%)                          | 0/21 (0%)                                                      | 0/30 (0%)                                           | 2.25 over 8 of 9 runs on 3 tasks            |

## Per task

| Task                                              | Kind        | baseline: first ok, kind pass, years |
| ------------------------------------------------- | ----------- | ------------------------------------ |
| series-gdp-ghana                                  | dataset     | 3/3, 3/3, 3/3                        |
| series-oda-germany-kenya                          | dataset     | 3/3, 3/3, 3/3                        |
| aggregate-health-population-oda-germany-by-sector | aggregate   | 0/3, 0/3, 2/2                        |
| aggregate-total-oda-kenya                         | aggregate   | 2/3, 3/3, 3/3                        |
| growth-total-oda-ghana                            | growth      | 1/3, 3/3, 3/3                        |
| sum-oda-oof-ethiopia                              | sum_of      | ending 0/3                           |
| rank-top-donors-malawi                            | rank        | 0/3, 3/3, 3/3                        |
| share-two-donors-of-kenya-oda                     | share_of    | 0/3, 3/3, 3/3                        |
| quotient-oda-per-person-ethiopia                  | quotient_of | 2/3, 3/3, 3/3                        |
| join-gdp-debt-service-ghana                       | join        | 3/3, 3/3, 3/3                        |
