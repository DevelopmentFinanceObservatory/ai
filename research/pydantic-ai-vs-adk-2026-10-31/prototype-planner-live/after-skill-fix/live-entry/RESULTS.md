# The planner on Pydantic AI against the planner on ADK, with the one-tool skill fix

With the fix in place the arms do not differ detectably in how often they reason long: 1.0% of ADK entries and 0.5% of Pydantic AI entries hold a call of 2,000 or more thinking tokens (-0.5 points [-2.4 points, +1.0 points]). Threshold (a), failure classes seen only on Pydantic AI: breached. (b), cached share down more than 2 points: met. (c), median wall time up more than 10%: met.

## Side by side

| ADK / Pydantic AI; paired difference [95% interval] | Fixed dev (210 per arm)                                         | Earlier, pooled (420 per arm)                                                    |
| --------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Entries with a call of 2,000+ thinking tokens       | 1.0% / 0.5%; -0.5 points [-2.4 points, +1.0 points]             | 29.3% (123 of 420) / 34.3% (144 of 420); +5.0 points [-2.9 points, +13.1 points] |
| First proposals accepted                            | 49.0% / 51.0%; +1.9 points [-4.8 points, +8.1 points]           | 57.4% / 59.8%; +2.4 points [-2.9 points, +7.9 points]                            |
| Final acceptance                                    | 91.9% / 91.4%; -0.5 points [-1.4 points, 0.0 points]            | not reported                                                                     |
| Kind pass                                           | 89.0% / 89.5%; +0.5 points [-2.4 points, +3.3 points]           | 86.0% / 85.5%; -0.5 points [-4.5 points, +3.6 points]                            |
| Cached share of input                               | 88.7% / 88.8%; +0.1 points [-0.3 points, +0.5 points]           | 88.4% / 88.5%; +0.1 points [-0.5 points, +0.8 points]                            |
| Planner wall time, median                           | 9.3 s / 8.1 s; -1.2 s [-2.2 s, +0.3 s] (-12.9%)                 | 16.1 s / 17.2 s; +1.1 s [-2.5 s, +4.7 s] (+6.9%)                                 |
| Planner wall time, p90                              | 23.7 s / 20.0 s; -3.7 s [-8.4 s, +0.3 s] (-15.6%)               | 73.7 s / 81.5 s; +7.8 s [-10.9 s, +23.0 s] (+10.6%)                              |
| Model calls per entry                               | 2.85 / 2.74; -0.10 [-0.23, +0.02]                               | 2.70 / 2.64; -0.06 [-0.15, +0.02]                                                |
| Entries that read a skill                           | 86.2% / 85.7%; -0.5 points [-2.9 points, +1.4 points]           | not reported                                                                     |
| Budget closures                                     | 0 ADK only, 1 Pydantic AI only (p = 1.00); entries closed 0 / 1 | 20 ADK only, 27 Pydantic AI only (p = 0.38); entries closed 25 / 32              |

## Long reasoning

ADK 2 of 210, Pydantic AI 1 of 210; the share differs by -0.5 points [-2.4 points, +1.0 points], so the arms do not differ detectably on it.

- adk rank-sectors-kenya rep 1: call 2 of 3 (3,940 tokens), call 3 of 3 (5,563 tokens)
- adk aggregate-combined-oda-kenya-uganda rep 3: call 2 of 3 (3,840 tokens)
- pai aggregate-peak-year-oda-kenya rep 3: call 2 of 2 (11,235 tokens)

### Thresholds

(a) Failure classes seen only on the Pydantic AI arm: **breached**. `ended: turn_model_time_spent`.

(b) Cached share down more than 2 points: **met**. Pydantic AI 88.8%, ADK 88.7%, difference +0.1 points [-0.3 points, +0.5 points].

(c) Median planner wall time up more than 10%: **met**. Pydantic AI 8.1 s, ADK 9.3 s, ratio -12.9% [-22.1%, +4.0%].

### First-proposal acceptance is a sanity check

The eval's A/A run put two identical arms about 5 points apart in first-proposal acceptance. The paired difference here is +1.9 points [-4.8 points, +8.1 points], an interval half-width of 6.4 points. Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about ADK's rate. It is a sanity check and not a parity claim. Kind pass is 0.5 points higher on the Pydantic AI arm, with an interval ([-2.4 points, +3.3 points]) that includes zero.

### Results per arm

|                                               | ADK    | Pydantic AI | Difference [95% interval]              | Relative               |
| --------------------------------------------- | ------ | ----------- | -------------------------------------- | ---------------------- |
| First proposals accepted                      | 49.0%  | 51.0%       | +1.9 points [-4.8 points, +8.1 points] |                        |
| Final acceptance                              | 91.9%  | 91.4%       | -0.5 points [-1.4 points, 0.0 points]  |                        |
| Kind pass                                     | 89.0%  | 89.5%       | +0.5 points [-2.4 points, +3.3 points] |                        |
| Entries that read a skill                     | 86.2%  | 85.7%       | -0.5 points [-2.9 points, +1.4 points] |                        |
| Model calls per entry                         | 2.85   | 2.74        | -0.10 [-0.23, +0.02]                   | -3.7% [-8.0%, +0.7%]   |
| Requests sent per entry                       | 2.90   | 2.78        | -0.12 [-0.26, +0.01]                   | -4.3% [-8.7%, +0.3%]   |
| Input tokens per entry                        | 48,587 | 46,675      | -1,912 [-4,229, +395]                  | -3.9% [-8.5%, +0.8%]   |
| Cached share of input                         | 88.7%  | 88.8%       | +0.1 points [-0.3 points, +0.5 points] |                        |
| Output tokens per entry                       | 692    | 677         | -14 [-70, +39]                         | -2.1% [-9.8%, +5.8%]   |
| First chunk, median                           | 3.4 s  | 3.5 s       | +0.1 s [-0.2 s, +0.2 s]                | +1.7% [-5.9%, +6.7%]   |
| First chunk, p90                              | 6.0 s  | 5.6 s       | -0.4 s [-0.9 s, +0.3 s]                | -7.1% [-14.4%, +5.0%]  |
| Planner wall time, median                     | 9.3 s  | 8.1 s       | -1.2 s [-2.2 s, +0.3 s]                | -12.9% [-22.1%, +4.0%] |
| Planner wall time, p90                        | 23.7 s | 20.0 s      | -3.7 s [-8.4 s, +0.3 s]                | -15.6% [-32.2%, +1.5%] |
| Entries with a call of 2,000+ thinking tokens | 1.0%   | 0.5%        | -0.5 points [-2.4 points, +1.0 points] |                        |

Each arm ran 210 entries: 70 tasks at 3 entries per task, on `gemini-3.8-flash` at `5f37d16d+dirty.6d0c9605`. Every figure counts every entry; a failed entry counts as a first proposal not accepted and a kind pass missed. The interval resamples tasks, keeping a task's entries together. Model calls are the calls the entry started, as the eval counts them; requests sent add the resends the planner makes after a first-chunk stall. Tokens come from the calls that returned usage. Cached share is cached input over input, pooled over all calls. A call cancelled before its first chunk counts in first-chunk time as the seconds it waited.

### Endings and failure classes

| Class                                   | ADK entries | Pydantic AI entries | Example                                             |
| --------------------------------------- | ----------- | ------------------- | --------------------------------------------------- |
| `call cancelled before its first chunk` | 12          | 8                   | series-debt-stock-kenya-ghana rep 1, CancelledError |
| `ended: turn_model_time_spent`          | 0           | 1                   | rank-top-donors-malawi-latest-year-and-series rep 1 |

Entries that ended in an accepted plan: ADK 193, Pydantic AI 192. Entries that ended in a question to the user: ADK 17, Pydantic AI 17. Neither is a failure.

The turn's model-time budget closed 0 ADK entries and 1 Pydantic AI entries. Within a pair, Pydantic AI was closed alone in 1 and ADK alone in 0 (exact sign test p = 1.00).

### Rate limits

No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK 0, Pydantic AI 0 (0 and 0 retries). Calls that drew a 429 and were repeated inside the planner's own retry layer: ADK 0, Pydantic AI 0. Entries ending in an error: ADK 0, Pydantic AI 0.
