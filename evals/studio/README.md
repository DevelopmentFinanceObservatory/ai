# evals/studio — harness-studio-v1

The eval set for the studio surface. It scores two things at once:

- **Contract adherence** (mechanical, model-free): every SSE frame validates
  against `harness.wire`, and the stream obeys the invariants
  `docs/frontend-interface.md` promises the frontend — `seq` starts at 0 per
  turn and strictly increases across resumed segments, a turn ends on `done`
  or `error` and nothing follows, composition revisions strictly increase,
  every number in a `message.complete` traces to a fact frame. `checks.py`
  holds the invariants; each case's `expect` blocks add per-turn assertions
  (terminal frame, component/block counts, error codes, elicitation
  affordances, plan versions, 409s on stale mutations).
- **Agent behavior** (rubric-judged, live only): interpretation correctness,
  plan legibility, narration honesty, amend handling, edit-vs-recreate
  correctness, follow-up resolution through the canvas digest, honest
  refusals. A judge model scores each case's `rubrics` against a digest of
  the recorded conversation (`test_config.json` sets the judge model,
  samples, and threshold).

This suite has its own runner: the behaviors under test live in the HTTP
surface, which `adk eval` cannot reach — the
translator, the Edit Card route, interrupt resumes, composition mutations,
the canvas digest. The runner drives the real routes instead and shares the
repo's two-lane rule (`evals/README.md`): the deterministic layer runs in
CI, the model-judged layer never does.

## Run it

```
just eval-studio-contract   # scripted, in-process, deterministic, no key
just eval-studio            # live agent + judge (needs GOOGLE_API_KEY,
                            # a running harness, and usually the stub)
```

Scripted mode builds the app in-process with the deterministic seams
(`harness.scripted`) — no server, no ports, no model, no upstream. It is
the same battery CI runs via
`services/harness/tests/test_studio_conformance.py`.

Live mode needs a running harness first:

```
SEED_FIXTURE_FRAMES=true PORT=8080 uv run --env-file .env harness-server
just eval-studio
```

`--case <id>` (repeatable) runs a subset; `--mode live` without `--judge`
runs only the contract layer against the live agent. Results land in
`.results/<mode>-<timestamp>.json` (gitignored), one record per case with
its violations and rubric scores.

## Reading a run

- `PASS` / `FAIL` — the case's mechanical expectations, the stream
  invariants, and (when judged) its rubric threshold.
- `XFAIL` — the case failed and was expected to. It encodes a **known gap**
  (over-resolution where an elicitation is due, thin analytical depth, a
  two-series card the platform refuses).
  The marker text says which gap and when to retire it.
  An xfail marker excuses missed *expectations* only — wire-contract
  violations (invalid frames, broken stream discipline) fail the case
  regardless, because the contract layer has no known gaps.
- `XPASS` — an xfail case passed. The gap closed; retire the marker so the
  behavior is held from now on.

Two live cases (`live-identity-bug-fails-safe`,
`live-rank-three-countries-fails-safe`) assert the harness fails SAFE on
the upstream identity-reproduction bug (`docs/upstream-asks.md` #4). When
the platform fixes that bug these cases fail — that is the signal to rewrite
them as real resolves.

The `live-route-*` cases hold the turn's route: whether it replies without
searching (`activity_steps: 0`, counting steps that ran a data tool), fetches directly (`no_plan: true` with at
least one card per requested output), or proposes a plan.

## Authoring cases

Cases live in `harness-studio-v1.cases.json`. A turn's `plan_decision` is
spent on the first plan card. One further card is approved (the harness
allows itself one execution-triggered replan per turn); a card after that
is declined, the turn ends there, and the miss is recorded as an
expectation (`plan.max_version` pins the bound explicitly). Every case must be able to
fail: at least one `expect` block or one rubric, and every `expect` key one
the runner checks (`TURN_EXPECT_KEYS` / `MUTATION_EXPECT_KEYS` in `run.py`) —
`test_studio_conformance.py` enforces this, along with unique case and
rubric ids and reasoned xfail markers. `message_contains` and
`message_excludes` each take a non-empty list of strings and match the turn's
`message.complete` text case-insensitively. Every `message_contains` string
must appear in it, and no `message_excludes` string may. A turn's `seed_spec` is sent
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
clients). Determinism (`repeat_identical`) is asserted scripted-only — the
live agent makes no such promise.
