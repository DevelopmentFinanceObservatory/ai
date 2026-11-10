# The planner on Pydantic AI against the planner on ADK, live entries

No threshold is breached on its point estimate; the interval does not exclude (c). Over 210 entries on the same tasks as ADK's 210, the Pydantic AI entry accepted 59.5% of first proposals against 60.0%, sent 88.6% of its input from the cache against 88.7%, and took a median 17.7 s against 16.2 s (+9.6%).

The run does not exclude a breach of (c), whose interval reaches +44.8%. The turn's model-time budget closed 17 Pydantic AI entries against 10 ADK entries, a gap the sign test cannot separate from chance (p = 0.17); each closure ends an entry at the budget's bound and lifts the upper percentiles of wall time.

A threshold is met when the condition that would stop a port does not occur, and its interval excludes it.

## Thresholds

(a) Failure classes seen only on the Pydantic AI arm: **met**. Every failure class on the Pydantic AI arm also occurs on the ADK arm.

(b) Cached share down more than 2 points: **met**. Pydantic AI 88.6%, ADK 88.7%, difference -0.1 points [-1.0 points, +0.7 points].

(c) Median planner wall time up more than 10%: **not breached on the point estimate; not excluded by the interval**. Pydantic AI 17.7 s, ADK 16.2 s, ratio +9.6% [-17.9%, +44.8%].

## First-proposal acceptance is a sanity check

The eval's A/A run put two identical arms about 5 points apart in first-proposal acceptance. This run's paired difference is -0.5 points [-6.7 points, +5.7 points], an interval half-width of 6.2 points. Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about ADK's rate. It is a sanity check and not a parity claim. Kind pass is 3.8 points lower on the Pydantic AI arm, with an interval ([-9.0 points, +1.0 points]) that includes zero.

## Results per arm

|                           | ADK    | Pydantic AI | Difference [95% interval]              | Relative               |
| ------------------------- | ------ | ----------- | -------------------------------------- | ---------------------- |
| First proposals accepted  | 60.0%  | 59.5%       | -0.5 points [-6.7 points, +5.7 points] |                        |
| Kind pass                 | 88.6%  | 84.8%       | -3.8 points [-9.0 points, +1.0 points] |                        |
| Model calls per entry     | 2.63   | 2.65        | +0.02 [-0.09, +0.12]                   | +0.9% [-3.1%, +4.9%]   |
| Requests sent per entry   | 2.73   | 2.74        | +0.00 [-0.12, +0.13]                   | +0.2% [-4.5%, +4.9%]   |
| Input tokens per entry    | 45,532 | 46,039      | +507 [-1,698, +2,508]                  | +1.1% [-3.6%, +5.7%]   |
| Cached share of input     | 88.7%  | 88.6%       | -0.1 points [-1.0 points, +0.7 points] |                        |
| Output tokens per entry   | 559    | 543         | -16 [-69, +36]                         | -2.8% [-11.8%, +6.5%]  |
| First chunk, median       | 3.7 s  | 3.7 s       | +0.1 s [-0.2 s, +0.3 s]                | +1.5% [-5.3%, +7.8%]   |
| First chunk, p90          | 7.6 s  | 7.3 s       | -0.3 s [-2.2 s, +0.9 s]                | -4.0% [-24.2%, +13.2%] |
| Planner wall time, median | 16.2 s | 17.7 s      | +1.6 s [-3.4 s, +6.3 s]                | +9.6% [-17.9%, +44.8%] |
| Planner wall time, p90    | 70.7 s | 83.0 s      | +12.3 s [-3.7 s, +24.6 s]              | +17.3% [-4.7%, +40.3%] |

Each arm ran 210 entries: 70 tasks at 3 repetitions, on `gemini-3.8-flash` at `245c0a44+dirty.4b3918c9`. Every figure counts every entry; a failed entry counts as a first proposal not accepted and a kind pass missed. The interval resamples tasks, keeping a task's repetitions together. Model calls are the calls the entry started, as the eval counts them; requests sent add the resends the planner makes after a first-chunk stall. Tokens come from the calls that returned usage. Cached share is cached input over input, pooled over all calls. A call cancelled before its first chunk counts in first-chunk time as the seconds it waited.

## Endings and failure classes

| Class                                   | ADK entries | Pydantic AI entries | Example                                                  |
| --------------------------------------- | ----------- | ------------------- | -------------------------------------------------------- |
| `call cancelled before its first chunk` | 22          | 18                  | aggregate-average-oda-kenya rep 1, CancelledError        |
| `call cancelled mid-stream`             | 10          | 17                  | sum-then-quotient-germany-tanzania rep 1, CancelledError |
| `ended: turn_model_time_spent`          | 10          | 17                  | sum-then-quotient-germany-tanzania rep 1                 |

Entries that ended in an accepted plan: ADK 184, Pydantic AI 180. Entries that ended in a question to the user: ADK 16, Pydantic AI 13. Neither is a failure.

The turn's model-time budget closed 10 ADK entries and 17 Pydantic AI entries. Within a pair, Pydantic AI was closed alone in 13 and ADK alone in 6 (exact sign test p = 0.17).

## What each arm ran

Both arms ran dev at `245c0a44` with the tool-body move (`planner/tools.py` and `planner/source_reads.py`, which hold the tool bodies both frameworks call), so the ADK arm is dev plus that move. The Pydantic AI arm ran the agent without the held-chunk fix proved in `node-hard/`, so a budget cut mid-stream could run a tool call the model had already delivered. Of the 17 closed Pydantic AI entries, 0 hold a proposal and 1 record more tool calls than responses that completed before the cut, which a delivered call of the cut response would produce. That entry's responses wrote 21 output tokens per call, against a median 21 in the other closed entries, so it holds two calls in one response and none from the cut.

## Rate limits

No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK 0, Pydantic AI 0 (0 and 0 retries). Calls that drew a 429 and were repeated inside the planner's own retry layer: ADK 0, Pydantic AI 0. Entries ending in an error: ADK 0, Pydantic AI 0.
