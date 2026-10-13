# evals/studio — harness-studio-orchestrator-v1

The eval set for the studio surface. It scores two things at once:

- **Contract adherence** (mechanical, model-free): every SSE frame validates
  against `harness.wire`, and the stream obeys the invariants
  `docs/frontend-interface.md` promises the frontend — `seq` starts at 0 per
  turn and strictly increases across resumed segments, a turn ends on `done`
  or `error` and nothing follows, composition revisions strictly increase,
  every number in a `message.complete` traces to a fact frame. `checks.py`
  holds the invariants; each case's `expect` blocks add per-turn assertions
  (terminal frame, component/block counts, error codes, elicitation
  affordances, plan versions, 409s on stale mutations), and its `checks`
  read what each model was shown.
- **Agent behavior** (rubric-judged, live only): a judge model scores each
  case's `rubrics` against a digest of the recorded conversation
  (`test_config.json` sets the judge model, samples, and threshold).

This suite has its own runner. The behaviors under test live in the HTTP
surface, which `adk eval` cannot reach: the translator, the Edit Card
route, interrupt resumes, composition mutations, the canvas digest. The
runner drives the real routes instead and shares the repo's two-lane rule
(`evals/README.md`): the deterministic layer runs in CI, the model-judged
layer never does.

## Run it

```
just eval-studio-contract   # --mode scripted: in-process, deterministic, no key
just eval-studio            # --mode live: live agent + judge (needs GOOGLE_API_KEY
                            # and a running harness)
```

Live mode needs a running harness first. `just studio-live` starts one on
port 8080:

```
just studio-live
just eval-studio
```

`--case <id>` (repeatable) runs a subset; `--mode live` without `--judge`
runs only the contract layer against the live agent. Results land in
`.results/<mode>-<timestamp>.json` (gitignored), one record per case with
its violations and rubric scores.

## Scripted mode

Scripted mode (`orchestrator_scripted.py`) serves the studio routes over the real
workflow, the real orchestrator tools and the real `PlannerSeam`, and scripts what a
model or an upstream would answer. It is the same battery CI runs via
`services/harness/tests/test_studio_orchestrator_eval.py`. A case's `scenario`
(inline, or the name of an entry in the file's `scenarios`) holds:

- `orchestrator`: the orchestrator model's replies, one per model call across every
  turn, each `{"call": tool, "args": {...}}` or `{"text": ...}`. A script that runs out
  fails the case as a contract violation.
- `search`: the canned results `search` returns, in order, each a list of `parts`
  (`variable`, `status`, and for a definite part `spec_id`, `sentence`, `coverage`,
  `warnings`), run through the same digest the served resolver builds.
- `plan_spec_id`: the source the canned plan retrieves. The planner's model submits
  the plan `harness.scripted_plan` builds, so the seam renders its real first message
  and the approval, execution and hand-back nodes run as served; `propose_plan` and
  its translation are not exercised.
- `planner`: the planner model's replies instead, in the orchestrator script's shape.
  The planner then calls the real `propose_plan`, whose translation discovers
  through `translator_search`, the canned results of each translation `search`, in the
  `search` entries' shape, and compiles against the real servers' tool schemas. The
  translation's verdict comes back under `translation` in `propose_plan`'s answer.
- `get_data_refusals`: spec id to the refusal (`code`, `retry_mode`) a plan step's
  `get_data` returns for it.
- `direct`: set for a case whose orchestrator calls `submit_analysis`, the direct recipes
  (`harness.direct_recipes`). The tool surface then declares the real servers' tool
  schemas, which a recipe compiles against, and answers its `growth` step.

Every model request (system instruction and contents, as the model received them) and
every tool call, a scripted planner's own included, is recorded in order. A
case's `checks` list names properties from `checks.py`'s `CASE_CHECKS` that read that record:

- `approval_payload_absent_after_approval`: no request made after the plan's first
  step ran holds the approval pause (the plan and the user's answer).
- `no_relay_marker`, `planner_sees_no_relay_marker`: no request, or no planner
  request, holds ADK's fence around another agent's quoted turn, which is how a
  workflow node's output reaches a model when it is authored by the node.
- `planner_first_message_carries_search_evidence`: the planner's first request names
  every spec id, warning code and window year the turn's searches returned.
- `previous_final_message_in_view`: each turn's first orchestrator request shows the
  previous turn's closing message whole, so a later "yes" has its offer in view.
- `earlier_plan_in_view`: each turn's first orchestrator request names every source
  an earlier turn's plan retrieved, by spec id, so a question about that plan can be
  answered.
- `translation_stops_on_a_repeated_failure`: once a translation fails the way it
  failed before, nothing is translated again in that planner entry.
- `blocked_reading_is_searched_again`: a source whose search reading carries a
  blocking warning is searched again by the plan's translation before a plan step
  retrieves it.
- `every_ask_accounted_for`: each output the orchestrator listed as an ask when it
  called `plan` is on the plan card, answered by steps the card shows or declined with
  a reason, and comes back in the plan's hand-back with each answering step's status
  or its decline.
- `direct_run_completed`: a `submit_analysis` call answered `completed` with a run that
  completed and a frame to chart, and no plan card was proposed.
- `unverified_source_refused`: a `submit_analysis` call naming a spec_id no search
  settled on was answered `needs_fields` for that spec_id, and nothing was retrieved.
- `no_repeated_failed_step` reads frames only, so it also runs live: a failed plan
  step does not fail again with the same tool, label and outcome text.

A case that lists a request check and runs without the log, or runs no plan step
where the check needs one, fails rather than passing on nothing.

## Live mode

The live cases (`orchestrator-live-*`) hold the properties only a model's behaviour
settles, with rubrics for the judge. Among them are that a period outside the
data's coverage is a recorded decision, every requested output is planned, declined or put to the user, a
later "yes" resolves the previous turn's offer, a blocking search warning is not
reused as an answer, and an unchanged failed step is not repeated.

## Reading a run

- `PASS` / `FAIL` — the case's mechanical expectations, the stream
  invariants, and (when judged) its rubric threshold.
- `XFAIL` — the case failed and was expected to. It encodes a **known gap**.
  The marker text says which gap and when to retire it.
  An xfail marker excuses missed _expectations_ only — wire-contract
  violations (invalid frames, broken stream discipline) fail the case
  regardless, because the contract layer has no known gaps.
- `XPASS` — an xfail case passed. The gap closed; retire the marker so the
  behavior is held from now on.
- `JUDGE_ERROR` — the judge model still answered 503 or 429 after its retries, so
  the rubrics went unscored. The suite goes on to the next case but does not
  pass; rerun the case with `--case <id>`.

## Authoring cases

Cases live in `harness-studio-orchestrator-v1.cases.json`, each with `modes` of
`["scripted"]` or `["live"]`. A turn's `plan_decision` is
spent on the first plan card. One further card is approved (the harness
allows itself one execution-triggered replan per turn); a card after that
is declined, the turn ends there, and the miss is recorded as an
expectation (`plan.max_version` pins the bound explicitly). Every case must be able to
fail: at least one `expect` block, check, or rubric, and every `expect` key one
the runner checks (`TURN_EXPECT_KEYS` / `MUTATION_EXPECT_KEYS` in `run.py`) —
`test_every_case_can_fail` in `test_studio_orchestrator_eval.py` enforces this,
along with unique case and rubric ids, reasoned xfail markers, and no request
check on a live case. `message_contains` and
`message_excludes` each take a non-empty list of strings and match the turn's
`message.complete` text case-insensitively. Every `message_contains` string
must appear in it, and no `message_excludes` string may. `plan.operations_contain`
takes a list of strings, each of which the turn's highest-version plan card must
say, case-insensitively, of an ask it answers and does not decline. Only the
operations that ask lists in `answered_by` count, and of those only a retrieve's
sentence, which is Search's account of what the source holds, and a rank's
population after its colon. Every other sentence, the step labels included,
restates the planner's measure, which repeats the user's words whatever the plan
keeps. A turn's `seed_spec` is sent
as the request's seeds, the Search results a seeded turn starts from, and
`step_tools` takes a non-empty list of tool names, each of which some activity
step of the turn must have completed (`receive_seeds` is the seed intake); a
failed step does not count. Scripted cases use
them because their closing text is fixed. A live message varies, so live cases
judge wording with a rubric. Rubric text follows the lessons in
`evals/README.md`: state when a property always applies, keep properties
independent, and leave the judge no route to excuse a miss.

What the set deliberately does not cover: cancel and client-disconnect
(timing-dependent; proven in `services/harness/tests/test_studio.py` and
against real uvicorn), and heartbeat cadence (no `seq`, ignored by
clients).
