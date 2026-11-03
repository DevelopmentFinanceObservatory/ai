# plan-mcp against live data, with an outside Gemini client

Gemini 3.8 Flash on Pydantic AI, given only DFO's public `/mcp` and plan-mcp, landed an accepted plan in 20 of 20 conversations, in 1.6 proposals on average, and every accepted plan ran to `finished`. 15 of 20 landed plans are right by the planner-entry eval's own rules (kinds and years). DFO's planner, run through the planner-entry eval on the same 10 tasks three times each, accepted 29 of 30 entries, also in 1.6 proposals, and 24 of 30 are right. The client's conversations cost far more: a mean 539,000 input tokens (87% read from Gemini's cache) and 19,200 output and thinking tokens over 12.2 model requests, against 45,700 and 650 for a planner entry. The two are not like for like: the client's figure covers search, skills, planning, the run and the reply with its charts, where a planner entry stops at the first accepted plan.

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
- `serve_live.py` on :8091: plan-mcp from `serve.build`, behind the harness's bearer gate, appending each accepted plan to `runs/plans.jsonl`.
- `dfo` (`dfo_cli.py`): one tool call per process on either server, printing a model's view (text blocks, and `structuredContent` only on an error result, as `~/.claude/dfo-ai-e2e/mcp` does), with the run's own `x-dfo-caller`, appended to `runs/<run>.jsonl`. It refuses a call past 40 tool calls or 12 proposals.
- `users/<task>.json` (`make_users.py`): the simulated user's script for each task, derived from the task file alone. The user opens with `question`; what they mean is `brief` and `asks`; a decision the reference records as `resolved_by: user` is answered with its `chosen` option (only `sum-oda-oof-ethiopia` has one: "add them"); one the planner settled by default gets "no preference"; a plan is approved only if it answers every ask.
- `simuser.py`: the user on `gemini-3.5-flash-lite`, holding the script. A check before the runs: shown a plan in current dollars that sums the years for `series-gdp-ghana`, it refused and named both faults; asked "add or show separately" for `sum-oda-oof-ethiopia`, it said "add them".
- `user` (`user_cli.py`): the simulated user for a Claude-side tester; `TESTER.md` is the tester's prompt.
- `client.py`: the Pydantic AI client on `gemini-3.8-flash`, the model `PLANNER_MODEL_NAME` names and the newest flash the key serves. Both servers are `MCPToolset`s; every call goes through `rig.call_tool`, so the model reads what `dfo` prints and each call is logged. Its instruction says to search, open, propose, show the card and run once approved.
- `score.py`: the scorer, reusing `evals/planner_entry/entry.py` for kinds and years. A Gemini or tester run's landed plan is read from `runs/plans.jsonl`, since an amendment's arguments carry only what changed (`gemini-rank-top-donors-malawi-r1`'s version 2 sends only its `rank`).
- `baseline/run/`: `evals/planner_entry/run.py baseline --only <the 10> --reps 3` at 1568eadb, whose harness is dev at 190c7a9c.

## Servers left running

| Server         | Port | pids (uv, Python)                | Log                   |
| -------------- | ---- | -------------------------------- | --------------------- |
| dfo-apis proxy | 8083 | 83682 (gcloud), 83700 (listener) | `logs/apis-proxy.log` |
| harness-server | 8080 | 83808, 83812                     | `logs/harness.log`    |
| plan-mcp       | 8091 | 88033, 88035                     | `logs/plan-mcp.log`   |

Stop them with `kill 83808 83812 88033 88035 83682 83700`, or by port: `lsof -tiTCP:8080 -sTCP:LISTEN | xargs kill`, and the same for 8091 and 8083. plan sessions live in plan-mcp's memory, so a restart of plan-mcp ends every open session; `runs/plans.jsonl` survives it.

Restart, from `/Users/jorge/repos/dfo-ai-worktrees/pai-research`, with `R=research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools`:

```
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

Then, from `$R/live`:

```
../../../../.venv/bin/python smoke.py series-gdp-ghana
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google,mcp]==2.64.0' --env-file /Users/jorge/repos/dfo-ai/.env \
  --package harness python client.py --reps 2 --concurrency 4 --prefix gemini
```

`client.py` skips runs whose log already ended cleanly, so the same command resumes a batch. Score from the worktree root:

```
uv run --package harness python $R/live/score.py --runs 'smoke-*' --runs 'gemini-*' \
  --baseline $R/live/baseline/run/records.jsonl --out $R/live/results.md
```

The 20 Gemini runs took 130 s each on average (229 s at most), about 11 minutes at four in flight. The baseline took under 10 minutes at the eval's default of four.

## Problems met

- Dev's dfo-apis answers 403 to a request without an identity token, so `DFO_APIS_URL` from `.env` (the Cloud Run URL) failed every search, and presentation-mcp could not fetch its release. Both servers take `DFO_APIS_URL=http://localhost:8083`, the gcloud proxy. plan-mcp needs it too: `execute` reads series statistics in its own process through `data_mcp.series_statistics`, which failed `step_result_unusable` against the Cloud Run URL.
- `serve.py` never resolves the harness's auth policy, so `common.auth.bearer_token()` is `None`, the harness clients send no bearer, and every call to the data-mcp mount answered 401. `serve_live.py` runs `apply_gate` before `serve.build` builds those clients, as `harness.server.create_app` does, and gates plan-mcp with the same token.
- `uv run --env-file` lets a variable already set win over `.env`, so the servers took `MCP_API_TOKEN=local-dev` while a client process took `.env`'s token and got 401. `rig.py` reads its own `RIG_MCP_TOKEN`, default `local-dev`.
- `propose_plan` answers `outcome: ok`, `status: draft` when translation hands a requirement back (`requirement_unmeetable`, `pinned_source_not_found`), with no `summary` and no `card`. In 4 runs Gemini proposed again without asking the user and was refused `referral_pending`, which costs a round. The scorer counts such an answer as a refusal, `translation:<kind>`.
- `referral_pending` carries `action: {"call": "adk_request_input"}` with no candidates. One `record_user_answer` was refused `invalid_arguments` because the client sent a `recommended_spec_id` with no candidates; it retried without one.
- `run_plan`'s hand-back names frames by plan-mcp's short ids (`fde783e`). The public tools cannot read them: 4 `describe_frame` calls were refused `frame_ref_incomplete` and 2 `request_invalid`, and clients fetched the data again with `get_data` (13 calls after a run) to chart it.
- A connect timeout to the simulated user's model ended the first batch's task group. `simuser.reply` now retries three times, and a failure ends only its own run.

## What a production version would change

- Approval and answers the server can trust. `run_plan` takes the client's word that the user approved, and `record_user_answer` the client's word for what the user said; one run in 20 wrote its own answer. MCP elicitation where the host supports it, or a token shown in the card and echoed back.
- One accepted shape. A translation referral should answer `refused` with the referral as its detail, so `outcome: ok` always means a ready plan with a card.
- Frames the public tools can read. The hand-back should carry `spec_id` and `content_version`, so a client charts the run's frames without fetching them again.
- Fewer tokens per request. Each Gemini request carried about 43,500 input tokens, most of them the tool declarations (`propose_plan` alone is 35,167 characters) and the guidance in `start_plan`'s answer, re-sent on every request. A client host with a prompt cache pays mostly for cached tokens (87% here); one without pays it all.
- Tasks that match today's search: `sum-oda-oof-ethiopia` no longer draws `combination_unasked`, so the eval scores both arms wrong for a plan that answers the question.

## Not covered

- Claude-side clients: `TESTER.md` is written, and no tester has run.
- Periods beyond what `entry.period_reductions` reads, units, and the reply's figures: right means kinds and years only, and no figure in a reply was checked against the data.
- The planner baseline replays recorded searches; the client searched live, so the two arms planned over different readings.
- One client model and two repetitions per task; the spread between the two runs of a task (3 and 5 rounds on `aggregate-total-oda-kenya`) says more repetitions are needed before reading a difference of a round.
