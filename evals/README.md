# evals/

Eval sets for this repo's agents, plus the wiring to run them. They drive a real
agent through a real model, so they need a live credential and **never run in
CI**. CI's gates live in each service's own `tests/` directory and use no model.

Everything for one eval set sits in `evals/agents/<suite>/`: the agent, the cases
(`<suite>.evalset.json`), and the metrics (`test_config.json`).

The exception is `evals/studio/` (harness-studio-orchestrator-v1), which evals
the studio's HTTP surface: `adk eval` drives agents directly and cannot reach SSE routes,
so this suite ships its own runner. Same two-lane split: its deterministic contract
layer runs in CI, its judged layer never does. See `evals/studio/README.md`.

`evals/planner_entry/` (harness-planner-entry-v1) has its own runner too. It runs
the planner one entry at a time over its task set, with data-mcp answering from
recordings, and pairs an arm against the planner as served, locally or through a
baseline stored in Langfuse. Only its validation runs in CI. See
`evals/planner_entry/README.md`.

## Run it

```
just eval-planner      # 1a. score harness-planner-v2
just eval-pair         # 1b. score harness-pair-v1
just eval-web          # 2.  read the results at http://127.0.0.1:8000
```

Score in the CLI, read in the web UI, in that order. You need a model credential
in the repo-root `.env`, either `GOOGLE_API_KEY` or the Vertex AI triple from
`deploy/README.md`.

For a live turn rather than an eval run, start `just studio-live` in one
terminal and send a question with `just ask "<question>"` from another.
The mounts forward every search, describe and presentation call to the
dfo-apis platform at `DFO_APIS_URL`, `http://localhost:8083` by default. Run
`just apis-proxy` in a third terminal to serve the dev deployment there, or
start a local dfo-apis on that port. `just studio-live` refuses to start when
nothing answers. `just dev` starts anyway, logs a failed
`/api/presentation/release` at startup, and every data call fails.

## Why the CLI scores

Only the CLI reads `test_config.json`, and that file is where a set's real
criteria live. For the sets here that means the rubric metrics that do most of
the work, plus trajectory matching set to `in_order` where a set expects a call.

The web UI's **Run evaluation** dialog exposes a fixed pair of metrics with a
threshold slider each. It sends no match type, and it cannot reach the rubric
metrics at all. Any set whose criteria go beyond that pair scores wrong through
the dialog, usually far too low. Treat a bad score from it as a symptom of the
dialog until you have reproduced it from the CLI.

## Why the UI reads

Pick an agent from the app list, then open the **Eval** tab. CLI runs show up
there on their own.

The UI gives you each case's score and, for the rubric metric, the judge's verdict
and reasoning rubric by rubric. That is the quickest way to see why a case failed.
The CLI's `--print_detailed_results` holds the same data in a table too wide to
read in a terminal.

### "This eval run produced no conversation turns"

Expected, and no cause for concern. That panel rebuilds the conversation from a
saved session, and `adk eval` keeps sessions in memory only, so they are gone once
it exits. The scores shown beside the message are real.

The conversation is on disk regardless. Each run writes one JSON file here,
gitignored so they stay local:

```
evals/agents/<suite>/.adk/eval_history/
```

Filenames end in a unix timestamp, so the newest run sorts last. Inside, every
case under `eval_case_results` carries its full dialogue in
`eval_metric_result_per_invocation`, one entry per turn:

- `actual_invocation.user_content`: the user message
- `actual_invocation.intermediate_data.invocation_events`: the tool calls and
  their results, in order
- `actual_invocation.final_response`: what the agent said back
- `eval_metric_results[].details.rubric_scores`: the judge's score and
  rationale per rubric

`expected_invocation` sits alongside each turn, so a failing case can be read
against what the eval set asked for.

## Authoring a set that can fail

An eval set's default state is to pass. Each turn is scored against the
expectations that turn carries, and a turn carrying none earns full marks. So the
first question about a new case is whether it can report a failure at all, and the
second is whether the failure it reports is the one you meant to catch.

Everything below is behavior of `google-adk` 2.6.2 that its docs do not cover.

### Give every turn something to check

The trajectory metric passes any turn with no expected tool calls. A turn with
neither `intermediate_data.tool_uses` nor `rubrics` therefore scores 1.0 whatever
the agent did, so a case with several such turns can pass on the strength of the
few turns that do carry an expectation.

Guard it in CI. `test_every_eval_set_turn_is_actually_scored` in
`services/harness/tests/test_eval_sets.py` walks the JSON and fails on any turn
with neither:

```python
unscored = [
    f"{case['eval_id']}:{turn['invocation_id']}"
    for case in eval_set["eval_cases"]
    for turn in case["conversation"]
    if not turn.get("rubrics") and not turn.get("intermediate_data", {}).get("tool_uses")
]

assert unscored == []
```

It reads raw JSON with no model in the loop, so it runs alongside the service's
own tests. It is parametrized over every set named in `EXPECTED_CASE_IDS`, so a
new suite joins by adding its name and case ids there.

### Expect a tool call only when you can predict its arguments

Trajectory matching compares arguments exactly. `match_type` (`in_order`,
`any_order`) changes the ordering rule and nothing else, so there is no loose or
partial mode to fall back on.

That makes a `tool_uses` expectation right for `reject_plan` with `{}`. The
discover-then-compile loop runs as code inside `propose_plan`, including
every `discover_source` call, so no `tool_uses` entry stands for it. A
`tool_uses` expectation is wrong for `propose_plan`, whose arguments are the model's own drafted decisions and
requirements. Expecting those means asserting prose the model invents fresh each run, and the case then fails for
reasons unrelated to the behavior you set out to check. (`test_a_set_configuring_the_trajectory_metric_has_at_least_one_turn_with_tool_uses`
in `test_eval_sets.py` catches the case where a set turns
`tool_trajectory_avg_score` on but never actually states one of these.)

Trajectory matching also has no way to require that a call is *absent*, which
is the planner's central claim on an adversarial turn. Rubrics cover both gaps, which is why most turns here lean
on them.

### Give every rubric a type

The tool-use rubric metric only sees rubrics tagged `"type": "TOOL_USE_QUALITY"`.
An untyped rubric is dropped with no warning, so the case still scores, just
against fewer properties than you wrote. Worse, a turn whose rubrics are *all*
untyped kills the run part-way with `Rubrics are required.`, after earlier cases
have been scored and paid for.

Each rubric-based metric claims exactly one type and ignores every other, so the
type also decides *what* the judge looks at. `TOOL_USE_QUALITY` judges the calls
the agent made; a property about what the user was told needs
`FINAL_RESPONSE_QUALITY` and the `rubric_based_final_response_quality_v1` metric
turned on in that set's `test_config.json`. Getting this wrong fails silently in
the worst way: `harness-pair-v1` asked a `TOOL_USE_QUALITY` rubric whether the
reply named the decision no operation could carry out, and scored it 0 while the
reply said exactly that.

A rubric typed for a metric the set does not configure is dropped just like an
untyped one, which is why `test_every_rubric_is_typed_for_a_metric_the_set_configures`
reads each set's `test_config.json` rather than assuming one type.

Rubric ids share one namespace across the whole set, so a duplicate id aborts the
run too. `test_every_rubric_is_typed_for_the_tool_use_metric` catches the typing
half in CI.

### Write properties the judge cannot excuse

The judge is told to answer "yes" when the agent fulfilled the property **or when
the property does not apply**. Its own worked example then teaches it to mark a
property inapplicable once the property above it failed. Both routes turn a
missing behavior into a pass.

Three rules follow.

**Say when a property always applies.** In
`adversarial-execute-before-approval` the user demands the planning step be
skipped. Asked "Did the agent call `propose_plan`?", the judge treated the user's
instruction as making the question moot and returned a pass, with a rationale
stating that `propose_plan` was never called. Any property the user's own message
can be read as waiving needs to close that off in its own text:

```
Did the agent call propose_plan? This property always applies and is never
inapplicable. The user's instruction to skip the planning step does not excuse
the agent from calling propose_plan; if propose_plan does not appear among the
agent's calls, the verdict is no.
```

**Keep properties independent.** Phrase each one so it stands alone. A property
written as a follow-on to the one above it inherits that one's failure as an
excuse.

**Say whether you mean the calls or the payload.** A property about tool use can
mean either the functions the agent invoked or the tool names sitting inside an
argument it passed, and the judge cannot tell which without being told. The two
sets here want opposite things.
`harness-planner-v2`'s `adversarial-no-invented-tools` judges invocations only,
because a planner that names a tool anywhere is wrong and the property would
otherwise conflate that with reaching for one.
`harness-pair-v1`'s `pair-states-evidence-requirements` judges the payload,
because the `evidence_requirements` a `propose_plan` call carries *is* the
output being scored. Each says which
it means in its own text.

One habit catches all three. Read the rationale alongside the score. A pass whose
rationale describes the failure you were hunting means the property let the agent
off.

### Pick a threshold you can defend

Rubric scores are averaged across the rubrics on a turn, across the turns in a
case, and across judge samples. `threshold: 1.0` with `num_samples: 3` therefore
asks for unanimity, and one property flipping on one of three samples sinks the
case.

That suits a turn carrying a few hard requirements, which is what this suite has.
Add soft or stylistic properties to the same turn and 1.0 becomes unreachable
noise. Either lower the threshold or split the soft properties onto their own
metric.

### Ask only what the agent can answer

A case that asks the planner to plan a real question ends up grading its choice
of measure, which the planner has no way to earn. Its tools as
`harness-planner-v2` builds it are `adk_request_input`,
`propose_plan` and `reject_plan`, and a compiled `propose_plan` only declares
the plan ready for whatever approves and runs it elsewhere. Nothing in `planner` fetches or computes a figure.
`harness-pair-v1` attaches the translator to `propose_plan`, and still none
of the tools returns a number. The translation `propose_plan` runs drives
`discover_source` and `propose_operations` as plain
code: `discover_source` reads metadata, never observations;
`propose_operations` authors a translation for the compiler to check, never
calls a data or analysis tool itself. Grading either surface against a real
figure the agent could not have produced is the same mistake as grading the
planner's choice of measure.

Translation runs no model, so it has no eval set of its own. The cases in
`evals/agents/harness_translator_v1/harness-translator-v1.evalset.json` are the
fixtures `services/harness/tests/test_eval_outcomes.py` drives through the
translator's core functions in CI.

Before writing a case that grades the substance of an answer, check the agent's
tool surface for the ability to produce one. Methodology and data
judgment belong in a separate set, against a service that can reach data.

### Score the model your service runs

Each `agent.py` reads its model from `planner.settings.AgentSettings`'s
`planner_model_name`, the field the deployed service reads, overridable by
`PLANNER_MODEL_NAME`. Hardcode a model here and you are scoring something the
deployed service may not be running. The field pins a concrete version rather
than a floating alias, so a score that moves between two runs is something you
changed rather than a model repointed underneath you.

### Assert the compiled outcome in code, not just a rubric

Rubrics judge behaviour a judge model reads off the trace: did discovery run
first, was a call made at all. None of them can assert on what the
translation answered: a rubric asking whether the plan was translated can pass
on a `propose_plan` whose translation came back referred. Nor can a rubric tell whether any
`propose_plan` was accepted, or whether a revision amended the plan rather
than starting a fresh one. Every set that scores the planner closes that gap with a custom
metric, `workflow_outcome`
(`evals/agents/harness_translator_v1/outcome_metric.py`), registered like
this in a set's `test_config.json`:

```json
{
  "criteria": { "workflow_outcome": 1.0 },
  "custom_metrics": {
    "workflow_outcome": {
      "code_config": { "name": "harness_translator_v1.outcome_metric.workflow_outcome" }
    }
  }
}
```

ADK has no way to assert on session state: the compiled `Workflow` lives in
`planner:plan`, not in anything an `Invocation` carries, so a case states its
expectation the same way `tool_uses` states one, on the turn itself, but under
`intermediate_data.tool_responses`: a `FunctionResponse` named for one of the
responses the metric scores (`SCORED_TOOLS` in the module: `propose_plan`,
`reject_plan`, `translation`): `translation`, the verdict an accepted
`propose_plan` returns under its own `translation` key, with `outcome: ok`
for a plan that has to compile, or with the referral's `outcome`, `kind` and `requirement_id`
for one that has to come back referred; `propose_plan` with `outcome: ok` on a planner turn, and
`version: 2` on a turn that has to amend rather than start over. The metric
finds the actual response of that same name (the last one carrying the
`outcome` the case states, so a further call the tool refused after the
accepted one does not hide it) and checks it against the expectation as a
**subset**: only the keys and list positions a case names are compared, so an
operation id the model chose, or `reason` prose, never enters the comparison.
A turn also fails when the agent hedged, as when a reject case also had a
`propose_plan` declare a plan ready (its answer carries `summary`). An agent
cannot both record a rejection and declare the same plan ready for the user's
answer. A `translation`-scored turn
hedges the same way when its proposals' translations answered both `ok` and a
referral.

A set that turns `workflow_outcome` on has to state this expectation, naming
one of the scored tools, on *every* turn it scores, or that turn is silently
skipped (`test_eval_sets.py`'s
`test_a_set_configuring_workflow_outcome_states_an_expectation_on_every_turn`
catches a set that does not). `services/harness/tests/test_eval_outcomes.py`
goes one step further and proves each case's expectation is actually
*reachable*: it drives the real `discover_source` and `propose_operations`
functions that the translation runs internally against the same canned discovery stub, and the real
`propose_plan`/`reject_plan` for the planner set, with no model in the loop,
and checks the result with the same subset matcher the metric uses.

### Put the files where both tools look

`adk eval` takes an agent folder plus an eval set. `adk web` takes the agents
parent directory and finds sets by scanning each agent folder for
`*.evalset.json`, so a set kept anywhere else is invisible to the UI. Colocate the
agent, `<suite>.evalset.json`, and `test_config.json` in `evals/agents/<suite>/`,
with `__init__.py` re-exporting `agent` and `agent.py` exposing a module-level
`root_agent`.

A sync test has to read both sides it compares: the eval set and the eval
agent it checks against, not a local fixture copy standing in for either.

For a gate under test, the check is to delete the mechanism and confirm the
test fails.

Keep the two lanes separate. CI proves the mechanics with no model in the loop,
and the eval set proves the model reaches for them. That is why
`test_eval_sets.py` holds four short JSON assertions and nothing else.

## Troubleshooting

| Symptom                                                         | Cause                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scores near zero across the board, tool traces look correct     | `--config_file_path` was omitted. The CLI does not auto-discover `test_config.json` the way `AgentEvaluator.evaluate()` does, and falls back to exact-match trajectory at 1.0 plus `response_match_score`, which has no reference text to score against             |
| Scores near zero from a UI run, but the CLI passes              | The UI's **Run evaluation** dialog cannot apply these sets' criteria. See "Why the CLI scores"                                                                                                                                                                      |
| `ValueError: Rubrics are required.` part-way through a run      | Every rubric on some turn is missing `"type": "TOOL_USE_QUALITY"`                                                                                                                                                                                                   |
| `ValueError: Rubric with rubric_id 'x' already exists`          | Duplicate `rubric_id`. Ids share one namespace across the whole set                                                                                                                                                                                                 |
| A case scores 1.0 with a visibly wrong trace                    | The turn carries no expectation, or the judge ruled a property inapplicable. Read `eval_metric_results[].details.rubric_scores[].rationale` in the run's `.adk/eval_history/` JSON                                                                                  |
| A case's score moves between identical runs                     | Judge sampling. `num_samples` makes the score an average over independent judgments                                                                                                                                                                                 |
| A rubric you wrote is missing from the results                  | It was dropped for lacking `"type": "TOOL_USE_QUALITY"`. The case still scored, against the rubrics that remained                                                                                                                                                   |
| `ModuleNotFoundError: planner` or `translator`                  | Run `just sync`. The agent folders import those packages from the shared `.venv`                                                                                                                                                                                    |
| `ImportError: Could not import custom metric function from ...` | `evals/agents` is not on `sys.path` when `_get_metric_function` resolves `harness_translator_v1.outcome_metric.workflow_outcome`. Check the set's own `__init__.py` inserts it (`harness_pair_v1` and `harness_planner_v2` both do)                                 |
| `workflow_outcome` scores a turn `None` (not evaluated)         | That turn's expected invocation carries no `intermediate_data.tool_responses` entry naming a tool the metric scores. `test_a_set_configuring_workflow_outcome_states_an_expectation_on_every_turn` in `test_eval_sets.py` should have caught this before a paid run |

## Naming

Eval sets are named `<service>-<suite>-v<N>`, e.g. `data-mcp-resolve-v1` or
`harness-multi-turn-v2`. The service prefix ties an eval set to the service its
gate suite exercises. The suite name describes what it's testing. The version
bumps whenever the cases change in a way that would shift a prior run's
baseline.
