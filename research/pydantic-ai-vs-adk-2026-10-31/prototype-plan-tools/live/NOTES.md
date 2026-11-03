# plan-mcp against live data, with outside clients

Gemini 3.8 Flash on Pydantic AI, given only DFO's public `/mcp` and plan-mcp, landed an accepted plan in 20 of 20 conversations, in 1.6 proposals on average, and every accepted plan ran to `finished`. 15 of 20 landed plans are right by the planner-entry eval's own rules (kinds and years). DFO's planner, run through the planner-entry eval on the same 10 tasks three times each, accepted 29 of 30 entries, also in 1.6 proposals, and 24 of 30 are right. The client's conversations cost far more: a mean 539,000 input tokens (87% read from Gemini's cache) and 19,200 output and thinking tokens over 12.2 model requests, against 45,700 and 650 for a planner entry. The two are not like for like: the client's figure covers search, skills, planning, the run and the reply with its charts, where a planner entry stops at the first accepted plan.

The results below are the first version of the tools. The second version (`../NOTES.md`) closes the gaps those runs found: `start_plan` takes the spec_ids the client's searches returned, `run_plan` returns rows and frame pairs the public tools read, strict-mode clients are not refused, the simulated user answers and approves through MCP elicitation, refusals name the shape they expect, and sessions persist in SQLite. The rig below drives the second version, and every client ran again against it: `clients.md` compares the two versions and `results-v2.md` holds the second version's tables.

## Results

`results.md` holds the tables below and `results.jsonl` one row per run. Rounds are `propose_plan` calls up to and including the first accepted one. Right counts runs whose landed plan uses the task's kinds (`entry.Task.selects_kinds`) and does what the task expects with its years (`entry.period_reductions`); the brackets give each count alone. Tokens are means per run.

Gemini client, 20 runs (2 per task):

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance                                                                                       | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `series-gdp-ghana`                                  | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 374,648      | 7,188             |
| `series-oda-germany-kenya`                          | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 401,700      | 9,845             |
| `growth-total-oda-ghana`                            | 2/2      | 4 4      | `referral_pending`, `translation:pinned_source_not_found`, `translation:requirement_unmeetable`                       | 2/2 (2, 2)           | 2/2          | 10         | 563,306      | 27,460            |
| `aggregate-total-oda-kenya`                         | 2/2      | 3 5      | `referral_pending`, `source_not_offered`, `translation:pinned_source_not_found`, `translation:requirement_unmeetable` | 2/2 (2, 2)           | 2/2          | 14         | 829,578      | 33,263            |
| `rank-top-donors-malawi`                            | 2/2      | 2 1      | `invalid_plan`                                                                                                        | 2/2 (2, 2)           | 2/2          | 8          | 484,736      | 24,071            |
| `join-gdp-debt-service-ghana`                       | 2/2      | 1 1      | none                                                                                                                  | 1/2 (1, 2)           | 2/2          | 9          | 469,257      | 14,408            |
| `sum-oda-oof-ethiopia`                              | 2/2      | 1 1      | none                                                                                                                  | 0/2 (0, 2)           | 2/2          | 10         | 499,102      | 20,068            |
| `share-two-donors-of-kenya-oda`                     | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 427,016      | 24,157            |
| `quotient-oda-per-person-ethiopia`                  | 2/2      | 1 1      | none                                                                                                                  | 0/2 (0, 2)           | 2/2          | 9          | 449,297      | 10,280            |
| `aggregate-health-population-oda-germany-by-sector` | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 17         | 891,424      | 21,132            |
| all                                                 | 20/20    | mean 1.6 |                                                                                                                       | 15/20 (15, 20)       | 20/20        | 10         | 539,006      | 19,187            |

DFO's planner, 30 entries (3 per task), recorded searches, live model:

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance        | Right (kinds, years) | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | -------------------------------------- | -------------------- | ---------- | ------------ | ----------------- |
| `series-gdp-ghana`                                  | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | 2          | 32,883       | 226               |
| `series-oda-germany-kenya`                          | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | 2          | 32,949       | 210               |
| `growth-total-oda-ghana`                            | 3/3      | 2 1 2    | `invalid_plan`, `measure_fixed_by_pin` | 3/3 (3, 3)           | 3          | 44,767       | 561               |
| `aggregate-total-oda-kenya`                         | 3/3      | 1 2 1    | `measure_fixed_by_pin`                 | 3/3 (3, 3)           | 2          | 38,981       | 328               |
| `rank-top-donors-malawi`                            | 3/3      | 2 2 2    | `invalid_plan`                         | 3/3 (3, 3)           | 3          | 50,995       | 925               |
| `join-gdp-debt-service-ghana`                       | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | 2          | 33,555       | 351               |
| `sum-oda-oof-ethiopia`                              | 3/3      | 1 1 1    | none                                   | 0/3 (0, 3)           | 3          | 45,797       | 533               |
| `share-two-donors-of-kenya-oda`                     | 3/3      | 2 2 2    | `invalid_plan`                         | 3/3 (3, 3)           | 3          | 51,271       | 933               |
| `quotient-oda-per-person-ethiopia`                  | 3/3      | 1 2 1    | `invalid_plan`                         | 3/3 (3, 3)           | 2          | 39,043       | 535               |
| `aggregate-health-population-oda-germany-by-sector` | 2/3      | - 3 5    | `combination_unasked`, `invalid_plan`  | 0/3 (0, 2)           | 5          | 86,559       | 1,933             |
| all                                                 | 29/30    | mean 1.6 |                                        | 24/30 (24, 29)       | 3          | 45,680       | 654               |

The planner's own summary is `baseline/run/summary.md`: first proposal accepted 14 of 27, kind pass 24 of 27, refusal task ending right 0 of 3, mean wall time 10.3 s.

What the wrong plans did:

- `sum-oda-oof-ethiopia`, both arms, every run: two `dataset` requirements and a `sum_of`, with no `aggregate` and no `combination_unasked` refusal. Search now binds the published totals from official donors on DAC2a and DAC2b, so the sum needs no aggregate over donors and the two OECD tables add without asking. The plan answers the question; the task's reference, which expects the refusal and an aggregate, is out of date. The eval's own summary scores it 0 of 3 too.
- `quotient-oda-per-person-ethiopia`, Gemini, both runs: one `dataset` requirement on DAC2a's published per-capita series, where the task wants ODA divided by population. The figures answer the question; the kind rule counts it wrong.
- `join-gdp-debt-service-ghana`, Gemini run 1: two `dataset` requirements and no `join`. The client built the table in its reply from the two frames.
- `aggregate-health-population-oda-germany-by-sector`, planner: a `comparison` or `sum_of` where the task allows only `dataset` and `aggregate`. Gemini answered all four asks with `dataset` and `aggregate` in one proposal both times.

Every Gemini run called `run_plan` only after the user had replied to the card. In one run (`gemini-growth-total-oda-ghana-r2`) the client called `record_user_answer` with an answer it wrote itself ("Sum across official and private donors"), before the user had said anything after the question. plan-mcp recorded it and the next proposal was accepted. The server cannot tell an answer the user gave from one the client made up.

The smoke run (`runs/smoke-series-gdp-ghana.jsonl`, printed in `logs/smoke.txt`) sends the task's search, `start_plan`, the reference plan and `run_plan`: accepted at version 1, run `finished`, 13 rows of Ghana's GDP.

## The rig

- `harness-server` on :8080 from this worktree, against dev's dfo-apis through `gcloud run services proxy` on :8083. It serves the public `/mcp` (`search`, `get_skill`, the data, analysis and presentation tools).
- `serve_live.py` on :8091: plan-mcp from `serve.build`, behind the harness's bearer gate, appending each accepted plan to `runs/plans.jsonl`. Sessions persist in `plan-sessions.sqlite3` beside it (`--sessions`), so a restart keeps every open session. It installs the stores data-mcp's lifespan would (`serve.install_shared_stores`). With `DEFINITION_STORE_BUCKET` naming the harness's bucket, `start_plan` holds a spec_id a client's `search` minted and no `get_data` has registered yet; without it such an id is refused `spec_not_held`, since dfo-apis' registry learns a definition only at its first fetch. With `FRAME_STORE_BUCKET` naming the harness's bucket, `run_plan` reads a fetched frame's rows; derived frames are read from dfo-apis either way, and a frame it cannot read comes back with its pair and no rows.
- `dfo` (`dfo_cli.py`): one tool call per process on either server, printing a model's view (text blocks, and `structuredContent` only on an error result, as `~/.claude/dfo-ai-e2e/mcp` does), with the run's own `x-dfo-caller`, appended to `runs/<run>.jsonl`. It refuses a call past 40 tool calls or 12 proposals. Each call declares MCP elicitation, and its handler (`rig.simulated_user`) puts plan-mcp's question or plan card to the run's simulated user and logs the exchange as `say` lines with `via: elicitation`; `--no-elicit` makes the call as a host without elicitation. It now runs under `uv run` with Pydantic AI and the `.env` key, as `user` does, since the simulated user is a model call.
- `users/<task>.json` (`make_users.py`): the simulated user's script for each task, derived from the task file alone. The user opens with `question`; what they mean is `brief` and `asks`; a decision the reference records as `resolved_by: user` is answered with its `chosen` option (only `sum-oda-oof-ethiopia` has one: "add them"); one the planner settled by default gets "no preference"; a plan is approved only if it answers every ask.
- `simuser.py`: the user on `gemini-3.5-flash-lite`, holding the script. `reply` answers the conversation; `elicit` answers a form plan-mcp sends: a pick among the choices it lists, words of their own, or a card to approve or turn down by the same rule. A check before the first runs: shown a plan in current dollars that sums the years for `series-gdp-ghana`, it refused and named both faults; asked "add or show separately" for `sum-oda-oof-ethiopia`, it said "add them". `elicit` has not run against the live model.
- `user` (`user_cli.py`): the simulated user for a Claude-side tester; `TESTER.md` is the tester's prompt, with an `<ELICIT>` slot for the run without elicitation.
- `client.py`: the Pydantic AI client on `gemini-3.8-flash`, or `gpt-6-luna` through the local relay with `--model luna`. Both servers are `MCPToolset`s; every call goes through `rig.call_tool`, so the model reads what `dfo` prints and each call is logged. Both toolsets and every call carry the same elicitation handler, Pydantic AI's `elicitation_handler` on the toolset and FastMCP's on the call's own client; `--no-elicit` drops both. `not_strict` declares each tool whose `_meta` says `strict: false` non-strict to the model, so the relay sends `strict: false` without being told. Its instruction is the first version's, unchanged, so the two batches differ only in the tools.
- `score.py`: the scorer, reusing `evals/planner_entry/entry.py` for kinds and years. A run's landed plan is read from `runs/plans.jsonl`, since an amendment's arguments carry only what changed. It also reports how each answer and approval was given: `record_user_answer` answers `by` the user or the client, and each version run `approved_by` the user or the client, with the user's refusals. A first-version run counts as the client's throughout. A `referred` answer counts as `translation:<kind>`.
- `baseline/run/`: `evals/planner_entry/run.py baseline --only <the 10> --reps 3` at 1568eadb, whose harness is dev at 190c7a9c.

## Starting the servers

From `/Users/jorge/repos/dfo-ai-worktrees/pai-research`, with `R=research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools`. A local GCS emulator gives the harness and plan-mcp one definition store and one frame store, as one process would have; without it each process keeps its own in-memory stores (see `serve_live.py` below).

```
docker run -d --rm --name pai-fake-gcs -p 4443:4443 fsouza/fake-gcs-server:latest -scheme http -port 4443 -public-host localhost:4443
curl -s -X POST -H 'Content-Type: application/json' -d '{"name":"pai-frames"}' 'http://localhost:4443/storage/v1/b?project=local'
export STORAGE_EMULATOR_HOST=http://localhost:4443 FRAME_STORE_BUCKET=pai-frames DEFINITION_STORE_BUCKET=pai-frames GOOGLE_CLOUD_PROJECT=local

perl -MPOSIX -e 'setsid; exec @ARGV' gcloud run services proxy dfo-dev-apis --region northamerica-northeast1 \
  --project dev-fin-observatory --port 8083 > $R/live/logs/apis-proxy.log 2>&1 < /dev/null &

PORT=8080 DFO_APIS_URL=http://localhost:8083 SEED_FIXTURE_FRAMES=true MCP_API_TOKEN=local-dev \
  PRESENTATION_RELEASE_REQUIRED=false PRESENTATION_PUBLIC_URL=http://localhost:8080/mcp/presentation \
  perl -MPOSIX -e 'setsid; exec @ARGV' uv run --env-file /Users/jorge/repos/dfo-ai/.env --package harness \
  harness-server > $R/live/logs/harness.log 2>&1 < /dev/null &

cd $R && DFO_APIS_URL=http://localhost:8083 MCP_API_TOKEN=local-dev PYTHONPATH=.:live:../../../services/harness \
  perl -MPOSIX -e 'setsid; exec @ARGV' uv run --env-file /Users/jorge/repos/dfo-ai/.env --package harness \
  python live/serve_live.py --port 8091 > live/logs/plan-mcp.log 2>&1 < /dev/null &
```

Stop them by port: `lsof -tiTCP:8080 -sTCP:LISTEN | xargs kill`, and the same for 8091 and 8083. Stop the emulator with `docker stop pai-fake-gcs`. Delete `live/plan-sessions.sqlite3` to start with no sessions.

Then, from `$R/live`, a smoke run (public `search`, `start_plan` with the spec_ids it returned, the reference plan, `run_plan` approved on the client's word) and the two Pydantic AI batches, with and without elicitation:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,anthropic,mcp]==2.64.0' \
  --env-file /Users/jorge/repos/dfo-ai/.env --package harness python smoke.py series-gdp-ghana
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,anthropic,mcp]==2.64.0' \
  --env-file /Users/jorge/repos/dfo-ai/.env --package harness python client.py --reps 2 --concurrency 4 --prefix gemini2
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,anthropic,mcp]==2.64.0' \
  --env-file /Users/jorge/repos/dfo-ai/.env --package harness python client.py --reps 2 --concurrency 4 \
  --prefix gemini2-noelicit --no-elicit
```

`--model luna --prefix luna2` runs gpt-6-luna the same way, through the relay on :8787. `client.py` skips runs whose log already ended cleanly, so the same command resumes a batch. Score from the worktree root, writing to a new file so the first version's `results.md` stays:

```
uv run --package harness python $R/live/score.py --runs 'gemini2-r*' --runs 'gemini2-noelicit-*' \
  --baseline $R/live/baseline/run/records.jsonl --out $R/live/results-v2.md
```

The first 20 Gemini runs took 130 s each on average (229 s at most), about 11 minutes at four in flight. The baseline took under 10 minutes at the eval's default of four.

## Problems the first runs met

- Dev's dfo-apis answers 403 to a request without an identity token, so `DFO_APIS_URL` from `.env` (the Cloud Run URL) failed every search, and presentation-mcp could not fetch its release. Both servers take `DFO_APIS_URL=http://localhost:8083`, the gcloud proxy. plan-mcp needs it too: `execute` reads series statistics in its own process, and `serve.held_identity` reads the registry.
- `serve.py` never resolves the harness's auth policy, so the harness clients sent no bearer and every call to the data-mcp mount answered 401. `serve_live.py` runs `apply_gate` before `serve.build`, as `harness.server.create_app` does, and gates plan-mcp with the same token.
- `uv run --env-file` lets a variable already set win over `.env`, so a client process took `.env`'s token and got 401. `rig.py` reads its own `RIG_MCP_TOKEN`, default `local-dev`.
- A translation referral answered `outcome: ok` with no card, and in 4 runs Gemini proposed again without asking the user and was refused `referral_pending`. It now answers `referred`.
- `referral_pending` named `adk_request_input`; refusals now name `record_user_answer`.
- `run_plan`'s hand-back named frames by plan-mcp's short ids, which the public tools refused; clients fetched the data again with `get_data` (13 calls after a run). `run_plan` now returns each frame's pair and, when small, its rows.
- A connect timeout to the simulated user's model ended the first batch's task group. `simuser.reply` and `simuser.elicit` retry three times, and a failure ends only its own run.

## What a production version would change

- Fewer tokens per request. Each Gemini request carried about 43,500 input tokens, most of them the tool declarations and the guidance in `start_plan`'s answer, re-sent on every request. The second version cuts what is read before the first proposal from 63,556 characters to 58,452; `propose_plan`'s descriptions (about 17,000 characters) and the planner's instruction (20,141) are the rest.
- Tasks that match today's search: `sum-oda-oof-ethiopia` no longer draws `combination_unasked`, so the eval scores both arms wrong for a plan that answers the question.
- The rest is in `../NOTES.md`.

## Not covered

- The second version against live clients: the rig is ready and has not run.
- Periods beyond what `entry.period_reductions` reads, units, and the reply's figures: right means kinds and years only, and no figure in a reply was checked against the data.
- The planner baseline replays recorded searches; the client searched live, so the two arms planned over different readings.
- Two repetitions per task; the spread between the two runs of a task (3 and 5 rounds on `aggregate-total-oda-kenya`) says more repetitions are needed before reading a difference of a round.
