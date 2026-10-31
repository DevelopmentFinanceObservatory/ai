# B2: wire tap, paired driver and analysis

## What was built

- `tap.py` patches `AsyncModels.generate_content`, `AsyncModels.generate_content_stream` and `AsyncCaches.create` at class level and appends one JSON line per call to a tap file. `tap.installed(path)` is a context manager that installs and restores the patch; `tap.labelled(arm=..., task=..., rep=..., entry_id=...)` sets the labels a call reads when it starts, through a `ContextVar`, so tasks an entry spawns inherit them.
  - A generate record holds: model, `cached_content`, whether `system_instruction`, `tools` and `tool_config` are set, the full `contents` as plain JSON (bytes as urlsafe base64, `None` fields dropped, so a model and a dict normalise alike), each request function call (name, id, a digest of its arguments, whether it carries a thought signature and a digest of it, the content index), the count of request parts with `thought=True`, the same for the response (function calls, thought parts), seconds to first and last chunk for a stream, final usage (prompt, cached, candidates, thoughts), finish reason, and the error class and code.
  - A cache-create record holds: model, ttl, canonical JSON and sha256 of `system_instruction`, `tools` and `tool_config`, the cache name returned, and the error.
  - Bookkeeping failures go into the record's `tap_errors` and never reach the caller. `analyse.py` counts them.
- `driver.py` runs `baseline` through `entry.run_entry` and `entry.entry_record` as `run.py` does, and the PAI arms through `pai_entry.pai_record(task, recording, arm=..., mode=..., prefix_cache=..., run_timeout_s=...)`. Arms share one `PrefixCache`, run at a total concurrency of 4, and write `records.jsonl` (the layout `run.py` writes) and `tap.jsonl` into `--out`.
- `analyse.py` reads `records.jsonl` and `tap.jsonl` and writes `RESULTS.md` into the run directory, plus `summary-<arm>.md` (the planner entry eval's own summary of each PAI arm against the baseline).
- `test_tap.py`: 21 tests over the tap, the checks and the driver's own logic.

## Commands

All from the worktree root. Run directory used below: `OUT=research/pai-prototype/gemini/runs/main`.

Tests (no network):

```
cd /Users/jorge/repos/dfo-ai-worktrees/pai-planner-proto && uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness pytest research/pai-prototype/gemini/test_tap.py -p no:cacheprovider -q
```

The full run, in 8 chunks of at most 8 tasks (62 tasks, sorted by id), 3 repetitions and 3 arms, so 72 entries per chunk. Run each chunk as its own foreground command with a tool timeout of 540000 ms. The live smoke measured 5 to 17 s per entry, so a chunk should take 3 to 5 minutes. `--stop-after-s 360` starts no new entry after six minutes, so a slow chunk ends cleanly and resumes.

```
cd /Users/jorge/repos/dfo-ai-worktrees/pai-planner-proto && uv run --env-file .env --with 'pydantic-ai-slim[google]==2.64.0' --package harness python research/pai-prototype/gemini/driver.py --out research/pai-prototype/gemini/runs/main --chunk 1/8 --stop-after-s 360
```

Repeat with `--chunk 2/8` through `--chunk 8/8`. Rerunning any chunk with the same `--out` skips its finished entries and retries errored ones. Exit code 1 means some entry still ends in an error. A chunk that prints "N runs not started" needs the same command again.

Analysis, after the last chunk:

```
cd /Users/jorge/repos/dfo-ai-worktrees/pai-planner-proto && uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness python research/pai-prototype/gemini/analyse.py --out research/pai-prototype/gemini/runs/main
```

This writes `runs/main/RESULTS.md` and quotes `parity.json` from this directory when B1 has written it (`--parity` names another file). Analysis can run on a partial run.

Other driver flags: `--arms baseline,pai,pai-validated` (default all), `--only <task ids>`, `--reps` (default 3), `--concurrency` (default 4), `--run-timeout` (300 s), `--passes` (2, each retrying errors), `--seed` (arm order), `--chunk K/N`, `--stop-after-s`.

## Results

### Tap transparency

`test_tap.py` runs google-genai's real classes against an `httpx.MockTransport`, so the patched methods, the SSE parser and the response models are the library's own. One scenario (a plain call, a two-chunk stream, a cache create, a 400) runs untapped and tapped and the returned data, every chunk and the raised error match exactly. A second test with fake classes checks that `config` and `contents` reach the original as the same objects, that a consumer that stops early closes the original stream (and the record says `GeneratorExit`), that a response the tap cannot read still reaches the caller, and that a `TimeoutError` from a create propagates.

Each check was run against a deliberately broken variant. I mutated the file, ran the suite, then restored it:

| Mutation                                        | Test that failed                                                         |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| tap forwards other `contents`                   | `test_arguments_pass_through_untouched_and_errors_and_closing_propagate` |
| tap drops the last chunk of a real stream       | `test_tap_returns_exactly_what_the_untapped_library_returns`             |
| tap re-yields one chunk fewer (in-test variant) | `test_the_comparison_catches_a_tap_that_drops_a_chunk`                   |
| W3 stops counting lost signatures               | `test_w3_matches_replays_to_their_originals_and_flags_a_lost_signature`  |
| W2 ignores `tools` (and `tool_config`)          | `test_w2_counts_a_cached_request...[tools]`                              |
| W4 counts first calls                           | `test_w4_reads_only_second_and_later_calls...`                           |
| parity never reports a difference               | `test_parity_names_where_first_requests_differ`                          |

W5, the verdict functions and the retry selection (`chosen_entries`) have tests with a passing and a failing fixture.

### Live smoke, baseline only

`driver.py --arms baseline --only series-gdp-ghana,growth-total-oda-ghana --reps 1` on `gemini-3.8-flash`: both entries accepted, 2 model calls each, 7.3 s and 9.6 s. The tap file held 4 generate calls (all streamed, all against the prefix cache, none carrying system instruction, tools or tool config) and 1 cache create shared by both entries, and `tap_errors` was empty. Each second call replayed the `get_skill` call with its thought signature. `summarise.py` read the records (`summary.md` written) and `analyse.py` read the tap file. Baseline numbers from that run: W4 89.7% cached on later calls, W5 first-chunk p50 3.2 s.

### Live smoke with the PAI arms

B1's `pai_entry.py` appeared during the work, so I ran all three arms on the same two tasks, one repetition. All six entries were accepted. W2, W3, W4 and O1 passed for both PAI arms (small numbers: 4 cached requests and 2 signed originals per arm). Two findings, both from the tap:

- Contents parity fails on both tasks. The baseline's first request is two user contents (the question, then the seam's brief); both PAI arms send one user content with the two as two parts. The text is the same. `analyse.py` calls this `regrouped` and still scores it a fail, since the pass line is identical canonical JSON. Whether the grouping matters to the model is for the coordinator to judge.
- The cache prefixes differ, so each arm made its own cache (three creates). The PAI system instruction is a user-role content with two parts, the first a 309-character note on mid-conversation system messages that the baseline's lacks, then the planner instruction (16,959 characters, the baseline's string). The baseline declares its five functions in one `Tool`; the PAI arms declare each in its own. `pai-validated` also sets `tool_config` to `VALIDATED`. The canonical JSON of each is in `tap.jsonl` for B1 to diff.

### Full run

Not run. The brief asked for the smoke only; the commands above run it.

## Deviations and interpretations

- W4: "within 5 points of baseline" is read as not more than 5 points below the baseline's share. A PAI arm with a higher share passes. The signed gap is printed.
- W2 and W3 report `inconclusive` when they could not fail: W2 when no request carried a cache, W3 when no replayed call had a signed original. W4 and W5 report it when the baseline has no figure.
- O1 has no pass rule in the brief. It passes when no switch trigger fires in `summarise.build_report` and no task falls. A task falls when the baseline passes it in a majority of runs (kind selection, or the expected ending for a refusal task, over runs the summary counts) and the arm does not.
- Contents parity compares the first generate call of each entry. A task whose runs disagree among themselves in one arm is `unstable`; a task with no run in one arm is `missing`. Both count against the line. `regrouped` is an addition to the brief.
- W3 matches a replayed function call to its original by name and a digest of its arguments, in order, not by id, because ADK and Pydantic AI treat ids differently. Calls with no original in the entry (for example one the seam inserted) are counted as `unmatched`. It also reports signatures `changed` and `added`, which are not part of the pass rule.
- The driver stamps `arm`, `rep`, `model`, `revision` and `experiment` onto every record, overriding what B1's `pai_record` computed, so the resume check uses one fingerprint per arm (`run.experiment`) across all three. `pai_record` also takes a `rep` argument that the brief's call omits; the driver does not pass it. A PAI record missing a key `summarise.py` reads is turned into an error record that names the key.
- The arms of one (task, repetition) start in a seeded shuffle (seed from `--seed`, the task and the repetition), so a resume repeats the order. Entries go repetition by repetition, then task, as `run.py` does.
- Flags beyond the brief: `--chunk`, `--stop-after-s`, `--seed`, `--passes`, `--run-timeout`. Records carry an extra `entry_id` that links them to the tap's calls; a retried error's earlier calls stay in the tap file and `analyse.py` ignores them.
- A cache-create record carries the labels of the entry that triggered the create, since the cache is shared by all arms.
- Seconds to first chunk run from the call to `generate_content_stream` to the first chunk, so they include client-side setup before the request goes out. The same holds for both frameworks.
- Under `uv run --with 'pydantic-ai-slim[google]==2.64.0'`, the run resolves google-genai 2.34.0 (the harness lock has 2.24.0). Both frameworks use that one library in the run. The tap was exercised on 2.34.0 only.
- W5 is noisy at small n. With a stand-in `pai_entry` that ran the ADK entry under the PAI arm names (3 identical arms, 2 tasks, 2 repetitions, 8 streams each), the same code gave first-chunk p50 ratios of 0.82 and 1.14 against the baseline, and the 1.10 line failed one of them. A pass or fail at this size says little; the full run has 186 entries per arm, at two or more streams each.

## Private imports and monkeypatches

- Monkeypatches: `AsyncModels.generate_content`, `AsyncModels.generate_content_stream`, `AsyncCaches.create` (google-genai), restored on exit of `tap.installed`.
- `driver.py` imports the eval's own modules by putting `evals/planner_entry` on `sys.path`: `run` (`revision`, `experiment`), `entry` (`run_entry`, `entry_record`, `inputs_digest`, `load_recording`, `load_tasks`, `quiet_logs`, `RECORD_SCHEMA`, `Task`), `summarise` (`latest`, `read_records`), plus `harness.prefix_cache.PrefixCache`, `harness.settings.Settings` and `planner.settings.AgentSettings`.
- `analyse.py` imports `entry` (`Task`, `load_tasks`) and `summarise` (`build_report`, `latest`, `left_out_reason`, `percentile`, `read_records`, `reduce_run`, `table`).
- `pai_entry.pai_record` (B1), imported when a PAI arm is selected. If `pai_entry` is absent the driver raises `NotImplementedError` before it starts any entry.
- `ty check` reports only unresolved imports for `entry`, `run`, `summarise` and `pai_entry`, which resolve through `sys.path` at run time. ruff format and check are clean.
