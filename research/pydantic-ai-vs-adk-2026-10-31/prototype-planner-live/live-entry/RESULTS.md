# The planner on Pydantic AI against the planner on ADK, live entries

No threshold is breached on the hardened run's point estimates; the interval does not exclude (c). No threshold is breached on the pool's point estimates; the interval does not exclude (c). Over 420 pooled entries per arm, the Pydantic AI entry accepted 59.8% of first proposals against 57.4%, sent 88.5% of its input from the cache against 88.4%, and took a median 17.2 s against 16.1 s (+6.9%).

A threshold is met when the condition that would stop a port does not occur, and its interval excludes it. The first run used the agent before the edge-case fixes; the hardened run used it after them. The pool pairs entries by run, task and repetition.

## Side by side

| Paired difference [95% interval]                                    | First run                              | Hardened run                            | Pooled                                 |
| ------------------------------------------------------------------- | -------------------------------------- | --------------------------------------- | -------------------------------------- |
| First proposals accepted                                            | -0.5 points [-6.7 points, +5.7 points] | +5.2 points [-1.9 points, +12.4 points] | +2.4 points [-2.9 points, +7.9 points] |
| Kind pass                                                           | -3.8 points [-9.0 points, +1.0 points] | +2.9 points [-2.9 points, +8.6 points]  | -0.5 points [-4.5 points, +3.6 points] |
| Cached share of input                                               | -0.1 points [-1.0 points, +0.7 points] | +0.4 points [-0.6 points, +1.4 points]  | +0.1 points [-0.5 points, +0.8 points] |
| Planner wall time, median                                           | +1.6 s [-3.4 s, +6.3 s] (+9.6%)        | +0.4 s [-3.7 s, +5.1 s] (+2.3%)         | +1.1 s [-2.5 s, +4.7 s] (+6.9%)        |
| Planner wall time, p90                                              | +12.3 s [-3.7 s, +24.6 s] (+17.3%)     | -6.0 s [-23.2 s, +26.1 s] (-8.1%)       | +7.8 s [-10.9 s, +23.0 s] (+10.6%)     |
| Model-time budget closures, Pydantic AI only / ADK only (sign test) | 13 / 6 (p = 0.17)                      | 14 / 14 (p = 1.00)                      | 27 / 20 (p = 0.38)                     |
| Model-time budget closures, entries per arm (ADK, Pydantic AI)      | 10, 17                                 | 15, 15                                  | 25, 32                                 |

## Long reasoning

| Set          | Arm         | Entries with a call of 2,000+ thinking tokens | Median wall, with one | Median wall, without |
| ------------ | ----------- | --------------------------------------------- | --------------------- | -------------------- |
| First run    | ADK         | 63 of 210 (30.0%)                             | 59.0 s                | 11.5 s               |
| First run    | Pydantic AI | 66 of 210 (31.4%)                             | 65.8 s                | 11.0 s               |
| Hardened run | ADK         | 60 of 210 (28.6%)                             | 59.0 s                | 11.0 s               |
| Hardened run | Pydantic AI | 78 of 210 (37.1%)                             | 57.7 s                | 9.5 s                |
| Pooled       | ADK         | 123 of 420 (29.3%)                            | 59.0 s                | 11.3 s               |
| Pooled       | Pydantic AI | 144 of 420 (34.3%)                            | 62.3 s                | 10.5 s               |

First run: the share differs by +1.4 points [-7.6 points, +10.5 points], so the arms do not differ detectably on it. Hardened run: the share differs by +8.6 points [-1.4 points, +19.5 points], so the arms do not differ detectably on it. Pooled: the share differs by +5.0 points [-2.9 points, +13.1 points], so the arms do not differ detectably on it.

## First run

Records: `runs/main`.

### Thresholds

(a) Failure classes seen only on the Pydantic AI arm: **met**. Every failure class on the Pydantic AI arm also occurs on the ADK arm.

(b) Cached share down more than 2 points: **met**. Pydantic AI 88.6%, ADK 88.7%, difference -0.1 points [-1.0 points, +0.7 points].

(c) Median planner wall time up more than 10%: **not breached on the point estimate; not excluded by the interval**. Pydantic AI 17.7 s, ADK 16.2 s, ratio +9.6% [-17.9%, +44.8%].

### First-proposal acceptance is a sanity check

The eval's A/A run put two identical arms about 5 points apart in first-proposal acceptance. The paired difference here is -0.5 points [-6.7 points, +5.7 points], an interval half-width of 6.2 points. Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about ADK's rate. It is a sanity check and not a parity claim. Kind pass is 3.8 points lower on the Pydantic AI arm, with an interval ([-9.0 points, +1.0 points]) that includes zero.

### Results per arm

|                                               | ADK    | Pydantic AI | Difference [95% interval]               | Relative               |
| --------------------------------------------- | ------ | ----------- | --------------------------------------- | ---------------------- |
| First proposals accepted                      | 60.0%  | 59.5%       | -0.5 points [-6.7 points, +5.7 points]  |                        |
| Kind pass                                     | 88.6%  | 84.8%       | -3.8 points [-9.0 points, +1.0 points]  |                        |
| Model calls per entry                         | 2.63   | 2.65        | +0.02 [-0.09, +0.12]                    | +0.9% [-3.1%, +4.9%]   |
| Requests sent per entry                       | 2.73   | 2.74        | +0.00 [-0.12, +0.13]                    | +0.2% [-4.5%, +4.9%]   |
| Input tokens per entry                        | 45,532 | 46,039      | +507 [-1,698, +2,508]                   | +1.1% [-3.6%, +5.7%]   |
| Cached share of input                         | 88.7%  | 88.6%       | -0.1 points [-1.0 points, +0.7 points]  |                        |
| Output tokens per entry                       | 559    | 543         | -16 [-69, +36]                          | -2.8% [-11.8%, +6.5%]  |
| First chunk, median                           | 3.7 s  | 3.7 s       | +0.1 s [-0.2 s, +0.3 s]                 | +1.5% [-5.3%, +7.8%]   |
| First chunk, p90                              | 7.6 s  | 7.3 s       | -0.3 s [-2.2 s, +0.9 s]                 | -4.0% [-24.2%, +13.2%] |
| Planner wall time, median                     | 16.2 s | 17.7 s      | +1.6 s [-3.4 s, +6.3 s]                 | +9.6% [-17.9%, +44.8%] |
| Planner wall time, p90                        | 70.7 s | 83.0 s      | +12.3 s [-3.7 s, +24.6 s]               | +17.3% [-4.7%, +40.3%] |
| Entries with a call of 2,000+ thinking tokens | 30.0%  | 31.4%       | +1.4 points [-7.6 points, +10.5 points] |                        |

Each arm ran 210 entries: 70 tasks at 3 entries per task, on `gemini-3.8-flash` at `245c0a44+dirty.4b3918c9`. Every figure counts every entry; a failed entry counts as a first proposal not accepted and a kind pass missed. The interval resamples tasks, keeping a task's entries together. Model calls are the calls the entry started, as the eval counts them; requests sent add the resends the planner makes after a first-chunk stall. Tokens come from the calls that returned usage. Cached share is cached input over input, pooled over all calls. A call cancelled before its first chunk counts in first-chunk time as the seconds it waited.

### Endings and failure classes

| Class                                   | ADK entries | Pydantic AI entries | Example                                                  |
| --------------------------------------- | ----------- | ------------------- | -------------------------------------------------------- |
| `call cancelled before its first chunk` | 22          | 18                  | aggregate-average-oda-kenya rep 1, CancelledError        |
| `call cancelled mid-stream`             | 10          | 17                  | sum-then-quotient-germany-tanzania rep 1, CancelledError |
| `ended: turn_model_time_spent`          | 10          | 17                  | sum-then-quotient-germany-tanzania rep 1                 |

Entries that ended in an accepted plan: ADK 184, Pydantic AI 180. Entries that ended in a question to the user: ADK 16, Pydantic AI 13. Neither is a failure.

The turn's model-time budget closed 10 ADK entries and 17 Pydantic AI entries. Within a pair, Pydantic AI was closed alone in 13 and ADK alone in 6 (exact sign test p = 0.17).

### What each arm ran

Both arms ran dev at `245c0a44` with the tool-body move (`planner/tools.py` and `planner/source_reads.py`, which hold the tool bodies both frameworks call), so the ADK arm is dev plus that move. The Pydantic AI arm ran the agent without the held-chunk fix proved in `node-hard/`, so a budget cut mid-stream could run a tool call the model had already delivered. Of the 17 closed Pydantic AI entries, 0 hold a proposal and 1 record more tool calls than responses that completed before the cut, which a delivered call of the cut response would produce. That entry's responses wrote 21 output tokens per call, against a median 21 in the other closed entries, so it holds two calls in one response and none from the cut.

### Rate limits

No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK 0, Pydantic AI 0 (0 and 0 retries). Calls that drew a 429 and were repeated inside the planner's own retry layer: ADK 0, Pydantic AI 0. Entries ending in an error: ADK 0, Pydantic AI 0.

## Hardened run

Records: `runs/hardened`.

### Thresholds

(a) Failure classes seen only on the Pydantic AI arm: **met**. Every failure class on the Pydantic AI arm also occurs on the ADK arm. Classes seen only on ADK: `ended: plan_not_ready`.

(b) Cached share down more than 2 points: **met**. Pydantic AI 88.5%, ADK 88.1%, difference +0.4 points [-0.6 points, +1.4 points].

(c) Median planner wall time up more than 10%: **not breached on the point estimate; not excluded by the interval**. Pydantic AI 16.3 s, ADK 16.0 s, ratio +2.3% [-20.3%, +35.8%].

### First-proposal acceptance is a sanity check

The eval's A/A run put two identical arms about 5 points apart in first-proposal acceptance. The paired difference here is +5.2 points [-1.9 points, +12.4 points], an interval half-width of 7.1 points. Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about ADK's rate. It is a sanity check and not a parity claim. Kind pass is 2.9 points higher on the Pydantic AI arm, with an interval ([-2.9 points, +8.6 points]) that includes zero.

### Results per arm

|                                               | ADK    | Pydantic AI | Difference [95% interval]               | Relative               |
| --------------------------------------------- | ------ | ----------- | --------------------------------------- | ---------------------- |
| First proposals accepted                      | 54.8%  | 60.0%       | +5.2 points [-1.9 points, +12.4 points] |                        |
| Kind pass                                     | 83.3%  | 86.2%       | +2.9 points [-2.9 points, +8.6 points]  |                        |
| Model calls per entry                         | 2.78   | 2.63        | -0.14 [-0.30, +0.00]                    | -5.1% [-10.3%, +0.2%]  |
| Requests sent per entry                       | 2.81   | 2.67        | -0.14 [-0.30, +0.01]                    | -5.1% [-10.5%, +0.5%]  |
| Input tokens per entry                        | 48,422 | 45,764      | -2,658 [-5,612, +234]                   | -5.5% [-11.2%, +0.5%]  |
| Cached share of input                         | 88.1%  | 88.5%       | +0.4 points [-0.6 points, +1.4 points]  |                        |
| Output tokens per entry                       | 606    | 553         | -53 [-137, +29]                         | -8.8% [-21.3%, +5.1%]  |
| First chunk, median                           | 3.4 s  | 3.1 s       | -0.3 s [-0.6 s, -0.0 s]                 | -8.0% [-15.8%, -0.9%]  |
| First chunk, p90                              | 6.4 s  | 5.9 s       | -0.5 s [-1.5 s, +0.2 s]                 | -8.5% [-20.7%, +4.0%]  |
| Planner wall time, median                     | 16.0 s | 16.3 s      | +0.4 s [-3.7 s, +5.1 s]                 | +2.3% [-20.3%, +35.8%] |
| Planner wall time, p90                        | 74.2 s | 68.2 s      | -6.0 s [-23.2 s, +26.1 s]               | -8.1% [-26.1%, +41.2%] |
| Entries with a call of 2,000+ thinking tokens | 28.6%  | 37.1%       | +8.6 points [-1.4 points, +19.5 points] |                        |

Each arm ran 210 entries: 70 tasks at 3 entries per task, on `gemini-3.8-flash` at `245c0a44+dirty.54f02264`. Every figure counts every entry; a failed entry counts as a first proposal not accepted and a kind pass missed. The interval resamples tasks, keeping a task's entries together. Model calls are the calls the entry started, as the eval counts them; requests sent add the resends the planner makes after a first-chunk stall. Tokens come from the calls that returned usage. Cached share is cached input over input, pooled over all calls. A call cancelled before its first chunk counts in first-chunk time as the seconds it waited.

### Endings and failure classes

| Class                                   | ADK entries | Pydantic AI entries | Example                                             |
| --------------------------------------- | ----------- | ------------------- | --------------------------------------------------- |
| `call cancelled before its first chunk` | 8           | 8                   | aggregate-peak-year-oda-kenya rep 1, CancelledError |
| `call cancelled mid-stream`             | 15          | 15                  | series-oda-by-sector-kenya rep 1, CancelledError    |
| `ended: plan_not_ready`                 | 1           | 0                   | filter-donors-above-200m-kenya rep 1                |
| `ended: turn_model_time_spent`          | 15          | 15                  | series-oda-by-sector-kenya rep 1                    |

Entries that ended in an accepted plan: ADK 178, Pydantic AI 179. Entries that ended in a question to the user: ADK 16, Pydantic AI 16. Neither is a failure.

The turn's model-time budget closed 15 ADK entries and 15 Pydantic AI entries. Within a pair, Pydantic AI was closed alone in 14 and ADK alone in 14 (exact sign test p = 1.00).

### Rate limits

No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK 0, Pydantic AI 0 (0 and 0 retries). Calls that drew a 429 and were repeated inside the planner's own retry layer: ADK 0, Pydantic AI 0. Entries ending in an error: ADK 0, Pydantic AI 0.

## Pooled

Records: `runs/hardened`, `runs/main`.

### Thresholds

(a) Failure classes seen only on the Pydantic AI arm: **met**. Every failure class on the Pydantic AI arm also occurs on the ADK arm. Classes seen only on ADK: `ended: plan_not_ready`.

(b) Cached share down more than 2 points: **met**. Pydantic AI 88.5%, ADK 88.4%, difference +0.1 points [-0.5 points, +0.8 points].

(c) Median planner wall time up more than 10%: **not breached on the point estimate; not excluded by the interval**. Pydantic AI 17.2 s, ADK 16.1 s, ratio +6.9% [-13.5%, +32.8%].

### First-proposal acceptance is a sanity check

The eval's A/A run put two identical arms about 5 points apart in first-proposal acceptance. The paired difference here is +2.4 points [-2.9 points, +7.9 points], an interval half-width of 5.4 points. Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about ADK's rate. It is a sanity check and not a parity claim. Kind pass is 0.5 points lower on the Pydantic AI arm, with an interval ([-4.5 points, +3.6 points]) that includes zero.

### Results per arm

|                                               | ADK    | Pydantic AI | Difference [95% interval]               | Relative                |
| --------------------------------------------- | ------ | ----------- | --------------------------------------- | ----------------------- |
| First proposals accepted                      | 57.4%  | 59.8%       | +2.4 points [-2.9 points, +7.9 points]  |                         |
| Kind pass                                     | 86.0%  | 85.5%       | -0.5 points [-4.5 points, +3.6 points]  |                         |
| Model calls per entry                         | 2.70   | 2.64        | -0.06 [-0.15, +0.02]                    | -2.2% [-5.4%, +0.9%]    |
| Requests sent per entry                       | 2.77   | 2.70        | -0.07 [-0.17, +0.03]                    | -2.5% [-6.0%, +1.1%]    |
| Input tokens per entry                        | 46,977 | 45,902      | -1,075 [-2,801, +550]                   | -2.3% [-5.8%, +1.2%]    |
| Cached share of input                         | 88.4%  | 88.5%       | +0.1 points [-0.5 points, +0.8 points]  |                         |
| Output tokens per entry                       | 582    | 548         | -35 [-82, +11]                          | -5.9% [-13.7%, +1.9%]   |
| First chunk, median                           | 3.5 s  | 3.4 s       | -0.1 s [-0.3 s, +0.1 s]                 | -2.9% [-8.3%, +1.8%]    |
| First chunk, p90                              | 6.9 s  | 6.5 s       | -0.3 s [-1.3 s, +0.2 s]                 | -5.0% [-16.2%, +3.4%]   |
| Planner wall time, median                     | 16.1 s | 17.2 s      | +1.1 s [-2.5 s, +4.7 s]                 | +6.9% [-13.5%, +32.8%]  |
| Planner wall time, p90                        | 73.7 s | 81.5 s      | +7.8 s [-10.9 s, +23.0 s]               | +10.6% [-13.3%, +35.6%] |
| Entries with a call of 2,000+ thinking tokens | 29.3%  | 34.3%       | +5.0 points [-2.9 points, +13.1 points] |                         |

Each arm ran 420 entries: 70 tasks at 6 entries per task, on `gemini-3.8-flash` at `245c0a44+dirty.4b3918c9 and 245c0a44+dirty.54f02264`. Every figure counts every entry; a failed entry counts as a first proposal not accepted and a kind pass missed. The interval resamples tasks, keeping a task's entries together. Model calls are the calls the entry started, as the eval counts them; requests sent add the resends the planner makes after a first-chunk stall. Tokens come from the calls that returned usage. Cached share is cached input over input, pooled over all calls. A call cancelled before its first chunk counts in first-chunk time as the seconds it waited.

### Endings and failure classes

| Class                                   | ADK entries | Pydantic AI entries | Example                                                  |
| --------------------------------------- | ----------- | ------------------- | -------------------------------------------------------- |
| `call cancelled before its first chunk` | 30          | 26                  | aggregate-average-oda-kenya rep 1, CancelledError        |
| `call cancelled mid-stream`             | 25          | 32                  | sum-then-quotient-germany-tanzania rep 1, CancelledError |
| `ended: plan_not_ready`                 | 1           | 0                   | filter-donors-above-200m-kenya rep 1                     |
| `ended: turn_model_time_spent`          | 25          | 32                  | sum-then-quotient-germany-tanzania rep 1                 |

Entries that ended in an accepted plan: ADK 362, Pydantic AI 359. Entries that ended in a question to the user: ADK 32, Pydantic AI 29. Neither is a failure.

The turn's model-time budget closed 25 ADK entries and 32 Pydantic AI entries. Within a pair, Pydantic AI was closed alone in 27 and ADK alone in 20 (exact sign test p = 0.38).

### Rate limits

No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK 0, Pydantic AI 0 (0 and 0 retries). Calls that drew a 429 and were repeated inside the planner's own retry layer: ADK 0, Pydantic AI 0. Entries ending in an error: ADK 0, Pydantic AI 0.
