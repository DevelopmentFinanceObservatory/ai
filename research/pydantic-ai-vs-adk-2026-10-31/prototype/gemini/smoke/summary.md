# Planner entry eval: pai-like-adk against pai-validated

4 runs over 2 tasks on gemini-3.8-flash; pai-validated at afbf84ff; pai-like-adk at afbf84ff. Rates count runs; tasks with a first-proposal refusal are scored only on whether they ended in that refusal and an ask. Runs that errored or missed their recording are left out of every figure; attempts count every entry started, retries included. Unasked units count dataset requirements stating a unit in first proposals of tasks whose question and brief name none. Tokens are means over runs whose every model call finished, input including cached.

## Per arm

| Arm           | Runs scored (attempts) | Errored | Missed the recording | First proposal accepted | Kind pass, runs | Kind pass, tasks | Refusal tasks ending right | Unasked units | Model calls | Input tokens (cached) | Output tokens | Thinking tokens | Wall s: mean, p90, max | Closed by the turn budget |
| ------------- | ---------------------- | ------- | -------------------- | ----------------------- | --------------- | ---------------- | -------------------------- | ------------- | ----------- | --------------------- | ------------- | --------------- | ---------------------- | ------------------------- |
| pai-validated | 2 (2)                  | 0       | 0                    | 0/1 (0%)                | 1/1 (100%)      | 1/1 (100%)       | 0/1 (0%)                   | 0 of 1        | 4.00        | 64,210 (54,868)       | 822           | 0               | 15.0, 16.7, 17.1       | 0                         |
| pai-like-adk  | 2 (2)                  | 0       | 0                    | 0/1 (0%)                | 1/1 (100%)      | 1/1 (100%)       | 1/1 (100%)                 | 0 of 1        | 3.50        | 55,131 (48,010)       | 744           | 0               | 12.3, 13.3, 13.5       | 0                         |

## Paired against the baseline

Per task, the arm's rate over its repetitions minus the baseline's.

| Outcome                 | Tasks | Better / worse / same | Mean difference [95%] | Sign test p |
| ----------------------- | ----- | --------------------- | --------------------- | ----------- |
| First proposal accepted | 1     | 0 / 0 / 1             | +0.0% [+0.0%, +0.0%]  | 1.00        |
| Kind pass               | 1     | 0 / 0 / 1             | +0.0% [+0.0%, +0.0%]  | 1.00        |

| Cost per run    | Tasks | Mean difference [95%]       |
| --------------- | ----- | --------------------------- |
| Model calls     | 2     | -0.5 [-1.0, +0.0]           |
| Input tokens    | 2     | -9,079.0 [-18,201.0, +43.0] |
| Output tokens   | 2     | -78.0 [-230.0, +74.0]       |
| Thinking tokens | 2     | +0.0 [+0.0, +0.0]           |
| Wall s          | 2     | -2.7 [-3.6, -1.8]           |

## Switch triggers

| Trigger                                                        | Fired | Numbers                       |
| -------------------------------------------------------------- | ----- | ----------------------------- |
| First-proposal acceptance down 5% or more, interval below zero | no    | +0.0% [+0.0%, +0.0%]          |
| Kind pass down 2 or more tasks                                 | no    | 1 against 1 of 1 shared tasks |
| p90 wall time up 10 s or more                                  | no    | 13.3 s against 16.7 s         |
| More runs closed by the turn budget than the baseline          | no    | 0 of 2 runs against 0 of 2    |

## How entries ended

| Arm           | accepted | asked_user |
| ------------- | -------- | ---------- |
| pai-validated | 1        | 1          |
| pai-like-adk  | 1        | 1          |

## Why first proposals were refused

| Arm           | Codes               |
| ------------- | ------------------- |
| pai-validated | invalid_plan rule 1 |
| pai-like-adk  | invalid_plan rule 1 |

## Per task

| Task                         | Kind      | pai-validated: first ok, kind pass | pai-like-adk: first ok, kind pass |
| ---------------------------- | --------- | ---------------------------------- | --------------------------------- |
| aggregate-average-oda-kenya  | aggregate | 0/1, 1/1                           | 0/1, 1/1                          |
| sum-net-flows-aid-debt-ghana | sum_of    | ending 0/1                         | ending 1/1                        |
