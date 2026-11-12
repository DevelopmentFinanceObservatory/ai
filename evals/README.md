# evals/

Evals for this repo's agents. Runs that drive a real model need a live
credential and **never run in CI**. CI's gates live in each service's own
`tests/` directory and use no model.

`evals/agents/harness_translator_v1/` holds the translation fixtures: the cases
in `harness-translator-v1.evalset.json`, the canned discovery stub, and the
`workflow_outcome` metric. No model runs them. `services/harness/tests/test_eval_outcomes.py`
drives each case's expected outcome through the real translator functions in CI,
and `test_eval_outcome_metric.py` tests the metric.

`evals/studio/` (harness-studio-orchestrator-v1) evals the studio's HTTP
surface with its own runner. Its deterministic contract layer runs in CI, its
judged layer never does. See `evals/studio/README.md`.

`evals/planner_entry/` (harness-planner-entry-v1) has its own runner too. It runs
the planner one entry at a time over its task set, with data-mcp answering from
recordings, and pairs an arm against the planner as served, locally or through a
baseline stored in Langfuse. Only its validation runs in CI. It is the
planner's eval. See `evals/planner_entry/README.md`.

`evals/rolling_anchor/` counts how seeded rolling plans end over synthetic
series, as admitted and as translation restates them. It needs no model, and CI
runs a share of it. See `evals/rolling_anchor/README.md`.

## Run it

For a live turn rather than an eval run, start `just studio-live` in one
terminal and send a question with `just ask "<question>"` from another.
The mounts forward every search, describe and presentation call to the
apis platform at `DFO_APIS_URL`, `http://localhost:8083` by default. Run
`just apis-proxy` in a third terminal to serve the dev deployment there, or
start a local apis on that port. `just studio-live` refuses to start when
nothing answers. `just dev` starts anyway, logs a failed
`/api/presentation/release` at startup, and every data call fails.
