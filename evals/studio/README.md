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

### The reply check

Every turn's closing message is held to the figures its cards supplied
(`check_reply` in `checks.py`). The stream carries each card and no tool result,
so the runner rebuilds a card from its `component.created` frame as `make_charts`
reads it (`harness.reply_figures`). The check gives a case one of three verdicts:

- `fail`: some closing message breaks a rule below. The case fails.
- `undecided`: no rule is broken, and the check cannot pair some stated value
  with one series and one year. This fails nothing: not the case, not the
  contract, not the run's exit status.
- `pass`: neither.

Each case prints its verdict on a `reply check:` line, the rules it broke on a
`reply rules failed` line and each undecided sentence on a `reply undecided:`
line. The run ends with the count of each verdict. The results file holds the
verdict under `reply_verdict`, the broken rules under `reply_rules` and the
undecided sentences under `reply_undecided`. The rules:

- `figure_missing`: a figure the turn's `reply.states` lists is not stated.
- `value_outside_reply_figures`: the message states a value a card draws that is
  not one of its figures, or states a figure beside another series' name.
- `figure_without_period`: a figure sits beside a year that is not its period's,
  or is stated nowhere with its period.
- `series_outside_reply_figures`: the message names a series of a card of more
  than four series that it may state no figure of.
- `number_in_no_frame`: the grounding check of the contract layer, which reports
  it as a contract violation.

A number states a figure when it writes the value whole or rounded to three
significant figures. A series is recognised by its label. The check pairs a value
with a series and a year by sentence shape, and decides only these readings:

- One name beside the value, with no other value between them, is its series; so
  is the only label a card of one series has. One year beside it is its year.
- Several names and then a multiple of as many values pair in order ("Uganda and
  Kenya stood at 68.5 and 63.8"), or each name takes its values in turn ("Kenya and
  Uganda rose from 60.1 to 63.8 and from 58.0 to 68.5"), or every value is a figure
  of one of the names. A reading counts when every value fits it; when none fits
  and the names divide the values, the values are held to their order and fail.
  Several years and then values pair the same way, in order or all at the last.
- A sentence that names one series and states no value gives that name to the
  first value of the next sentence when that names none ("Uganda leads. It stood at
  68.5 years.").
- A sentence that names no year is read in the year of the last sentence before it
  that names one, as a list item is under the line that introduces its list. A year
  no card draws a point at dates no value.
- A number of two figures that rounds a figure of a series its sentence names is
  held to that series and year, and states no figure. One written after "above",
  "below", "over", "under", "at least", "more than" or "less than" is a bound.
- A four-digit number is an amount when a currency sign, a per cent sign or a
  series' unit sits beside it, or when it is a value a card draws at no year a card
  draws. It is a year when it is a year a card draws a point at and no drawn value,
  or when it follows "in", "for", "since", "from", "to" or "between" or sits in a
  range of years.

Everything else is undecided, reported under `pairing_undecided` with the sentence
quoted, the open number and why. Such a value never passes by default, and a
figure `reply.states` requires that it writes is not reported missing:

- a value with a name on each side ("68.5 years in Uganda and 63.8 years in Kenya"
  leaves 63.8 open), or after several names it does not pair with;
- a value with a year on each side ("56.1 years in 2000, rising to 63.6 years in
  2023" leaves 63.6 open);
- a value in a sentence that names no series, on a card of several;
- a value beside a label that is not its own, where the message spells its own
  label out nowhere ("the United States" for "United States of America"), or where
  the label sits past another value ("from $543 million (IDA) down to $27.0 million
  (Ireland)");
- values under several carried years they do not pair with in order;
- a two-figure number that rounds a figure of a series the sentence does not name,
  which may be a count;
- a four-digit number that no form, drawn value, drawn year or place decides.

Ordinary prose is often undecided. With no `reply` expectation, 53 of 132
recorded streams (40%) and 45 of 71 recorded case replies (63%) hold an undecided
sentence; with the figures each card returned, 42 of the 71 (59%). The verdicts
over the 132 streams are 75 pass, 13 fail and 44 undecided. A value with a name or
a year on each side is the commonest shape: every open value sits in a strictly
alternating sentence in 26 of the 53 streams and 35 of the 45 replies.

Two shapes still pass wrongly:

- A year carries across a paragraph break, so "In 2000 both were lower.", a blank
  line, then "Today Kenya stands at 56.1 years." passes with the 2000 value.
- A card's series count and the two counts of a cut card's title ground wherever a
  reply writes them, so "70 donors cut their aid" grounds on a card cut from 70.

Which figures a card may be stated with depends on the turn's `expect`:

- With a `reply` expectation, a card gives what `make_charts` gives a card none of
  whose series the model named: a lone series' latest value, the largest and
  smallest of a card whose series share one period, and nothing for a card whose
  every series a card on the canvas drew as the turn began, as the last
  composition before it holds. The figures `reply.states` lists are added,
  and any other value a card draws fails.
- With none, any point a card draws may be stated, and the check holds only that
  each value sits beside its own series and its own year. A point stated that way
  grounds at three figures; a number that matches no point of the series beside it
  fails the grounding check.

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
failed step does not count. `card_title_contains` takes a non-empty list of
strings, each of which some card the turn created must hold in its title,
case-insensitively, and `max_card_series` is the most series any card the turn
created may draw. `reply` takes `{"states": [...]}`, the figures the closing
message must state, each `{"series", "which", "period"}`: `which` is `first`,
`last`, `at` (with `period`, the point at that period), `largest` or `smallest`
(the marks of a card whose series share one period, which take no `series`).
`series` names a drawn series by its label or by a coordinate such as
`country/KEN`. Left out, any drawn series answers. An empty `states` list still
holds the reply to what its cards give unasked. Scripted cases use the text keys
because their closing text is fixed. A live message varies, so live cases
judge wording with a rubric. Rubric text follows the lessons in
`evals/README.md`: state when a property always applies, keep properties
independent, and leave the judge no route to excuse a miss.

What the set deliberately does not cover: cancel and client-disconnect
(timing-dependent; proven in `services/harness/tests/test_studio.py` and
against real uvicorn), and heartbeat cadence (no `seq`, ignored by
clients).
