# translator

The Translator Agent turns a plan the planner has drafted into the workflow the
user approves. It finds the data each evidence requirement needs, proposes the
operations that carry the plan's decisions out, and has the compiler build and
check the result. What cannot be carried out goes back to the planner as a typed
referral rather than being approximated.

`translator` ships inside `harness` as a sibling top-level module
(`services/harness/src/translator/`), on harness's dependencies, tests, and CI.
Its tests live in `services/harness/tests/translator/`.

## What it does

One tool, `translate`, built by `build_translator_tools()` and wired into an
`LlmAgent` by `create_translator_agent()`. The planner writes the plan to
session state (`planner.plan.save_plan`, keyed on `PLAN_STATE_KEY`), and
`translate` reads it back itself (`translator.discovery.current_draft`,
`planner.plan.load_plan`), so no copy of it crosses the `dfo_translator` tool
call. That call's `request` string is a free-text instruction -- what the
planner wants done, not the plan -- which ADK appends as the translator's
first user turn.

`translate` runs the whole discover-then-compile loop as code
(`translator.mechanical.run_translate`): `discover_source` for every dataset
requirement, then `propose_operations`, following each step's own next
action, the same order `INSTRUCTION` once walked one model turn at a time.
None of that costs a model call -- a clean, one-requirement plan compiles in
0 model calls inside the loop, against 4 translator-side model calls per
cycle before this loop existed. Calling it answers one of three shapes:

| Outcome          | Meaning                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `ok`             | The plan compiled; `submit_plan` reads the richer summary off the plan's own record. |
| `referred`       | Something could not be carried out; a typed `Referral` says what and why.            |
| `needs_decision` | The loop paused for the one judgement it cannot make on its own -- see below.        |

The translator keeps exactly one judgement: which of several candidate
sources best answers a requirement, once `discover_source` finds more than
one and none is pinned. `translate` stops there and answers
`needs_decision`/`candidate_ranking`, carrying the candidates (each with its
own `describe`d definition) and the decisions citing the requirement. The
model ranks them and calls `translate` again with `requirement_id`, `choice`
(the candidate's own spec_id) and `rationale`; that answer rides into the
`source_ambiguous` referral as a `recommendation` for the person who settles
the choice to read, never a pick `translate` makes on its own --
`discover_source`'s own answer already forbids that. A `no_operation_covers`
escalation (a compile issue naming `needs=("decision_id",)`, the one other
case `_compile_failure_action` cannot resolve alone) is settled the same way,
with `decision_id` in place of `choice`/`rationale`. A `no_data`/
`unsupported_ask` answer refers straight to the planner instead: no wording
a model could try reaches an entity, role or unit that never resolved, and
the planner already re-asks the user with the question in hand.

`describe` is not a tool either: `translate` calls it from code
(`translator.discovery.fetch_definitions`), immediately before compiling, to
fetch a genuinely `definite` source's own definition and carry it into the
approval readback (`planner.workflow.SelectedSource.definition`, rendered by
`harness.plan_display._sentence_dataset`) -- and again, per candidate, to
enrich a `candidate_ranking` judgement's own payload. Memoized per
requirement, keyed the way `selectable` itself is scoped
(`requirement_revision`), so a `COMPILE_FAILED` retry within one plan
revision reads what an earlier call already fetched rather than
re-describing the same source. A missing or failed fetch never fails the
compile: the readback renders what it renders without a definition.

None of this is confirmation-gated. Discovery reads metadata; compiling
writes a workflow that nothing runs before the user approves it. The gate
stays in the harness graph.

The agent reads the tool surface through a `ToolSurface` (`compiler.surface`),
the same object the compiler checks against, so what it discovers on and what a
workflow is compiled against are one surface. Built with `mode="single_turn"`,
ADK wraps it as one tool on the planner, `dfo_translator`; it reads the plan
from session state and returns one function response.

## The properties that define done

**Observations and computation are out of reach.** The translator's tool
wraps `search`; `get_data` and the analysis verbs are not reachable from it,
so no instruction has to forbid them. `describe` is reached only from code,
at compile time or to enrich a candidate, never by a model call.

**Discovery is bounded, per plan revision.** Every `discover_source` call
counts against `DISCOVERY_CALL_BUDGET` for the plan *revision* it serves, not
the plan's version number, which a session may share across an amendment,
and not the session as a whole, which may hold more than one question. The
call past the budget comes back as `discovery_budget_exhausted`,
pointing at `return_to_planner`. A new revision (an amendment, or a second
question) gets a fresh budget; what an earlier revision discovered stays known
in the log, but out of reach of a later revision's selection.

**The search is built from the requirement, not proposed by the model.**
`discover_source` takes a requirement id and nothing else. The variable,
entities and roles, time window and unit (currency and price basis, "constant
2015 USD") it searches with are all read off the requirement the plan already
states, so a model cannot supply a search that contradicts it. A `no_data` or
`unsupported` answer refers straight back to the planner rather than inviting
a reworded search: rewording is a second model judgement, and the planner is
where the requirement itself changes instead, with a `no_data` referral
carrying the reason and the nearest real sources search found. Whatever
phrase was sent is stamped on each source as `searched_as`, so the approved
workflow shows the words the source answered. An answer `search` reached by
dropping a phrase it did not recognise (a `role_text_not_recognised` or
`time_text_not_parsed` warning) is refused as `ask_not_understood`, pointing
at a referral of the same kind; its record is never selectable, pinned or not.
An answer that offers a dataset the turn dropped (a seed whose StatVars left
its dataset, over any years) is refused as `seed_dropped` and referred the
same way, since the user is told the plan leaves that dataset out.
Other warnings are advisory and ride out with the answer.

**A source is selected only from a definite answer, for its own requirement.**
The `DiscoveryLog` records every source a call returned and which answer
returned it, scoped by `(revision, requirement_id)`. The compiler is handed
only the sources `definite` answers named for each requirement's own retrieval,
so a retrieval naming any other `spec_id` (another requirement's source
included) is refused, whatever it looks like. A requirement whose
`source_spec_id` pins a source is selected from that pin instead, from among
what a `definite` or `candidates` answer recorded for it (pinning is how a
user settles a `source_ambiguous` referral), and never from a `no_data`
answer's nearest series; when the pin was never named, discovery says so with a
`pinned_source_not_found` action. A pin naming a dataset the user handed over
from Search, or its copy over the requirement's period, is honoured from the
seeded source with no `search` call. A copy is another identity, so its
source is Search's answer for it, read through data-mcp's
`resolve_definition` on the surface the translator calls
(`translator.discovery.resolve_definition`, `resolved_spec`): the sentence,
coverage and StatVars are the ones Search states for the requirement's
period, and data-mcp stores the copy's definition as it answers, so its
retrieval finds it wherever data-mcp runs. A copy Search does not resolve to
that one dataset is refused as `search_refused`, naming the dataset and the
period, and referred back as `no_data` when Search finds no data for the
period, with Search's reason code, or as `requirement_unmeetable` for any
other answer. One Search could not be asked about is refused as
`surface_unavailable` and referred back as `discovery_unavailable`.
`run_translate` and `propose_operations` take a requirement's
source through `translator.discovery.settled_sources`, which leaves out a
source that is a dataset the turn dropped (`planner.plan.names_dropped_seed`),
whenever its record was written. An unpinned requirement left without a source
is searched again, and an answer naming the dropped dataset is refused as
`seed_dropped`. A dataset the turn still holds as a seed is never read as
dropped, so a second seed over the same dataset and other years stays
usable. `translate` answers with each dataset described once
(`translator.discovery.described_sources`): a dataset several requirements
retrieve, under one spec_id or under a copy over each period, lists its other
retrievals as `also_retrieved_as`, each a spec_id, the phrase it was
searched as, the coverage Search states for its period, and its `has_data`
when that differs from the first retrieval's. Two sources are one dataset
when their identities are equal apart from the years
(`planner.plan.dataset_of`). The log keeps each requirement's own view of a
source, so a spec two requirements both resolve is recorded once for each,
each with the words it was searched under. `candidates`, `no_data` and
`unsupported` are referred to the planner: the choice of source is
methodological, and a requirement no data meets is one to amend. Each such
answer arrives with an `action` naming the referral kind to use, and describes
its sources without their `spec_id`s, so there is nothing for a retrieval to
name.

**A referral's candidates come from the log.** `return_to_planner` attaches the
candidate sources discovery recorded for the requirement in this revision; it
refuses a `source_ambiguous` referral for a requirement discovery never
answered with candidates. A referral names the requirement or decision it is
about, and both have to exist in the plan. Saved to session state under
`translator.referral.REFERRAL_STATE_KEY` rather than returned inline, so the
planner reads it on its own turn.

**Translation is compiled or it is nothing.** `propose_operations` runs
`compiler.compile_plan`. Only a workflow that compiles reaches the plan, through
`Plan.with_translation`, which is what moves a `DRAFT` to `PENDING_APPROVAL`.
Otherwise the plan is untouched and every issue comes back, with an `action`
when the issues mean the plan rather than the proposal has to change.

**Only a draft is translated.** A plan in any other status is refused, as is one
that states no evidence requirements.

**Every tool returns a typed result.** Nothing answers `{"error": "..."}`.
Success is `{"outcome": "ok", ...}`; a referral saved is `{"outcome": "referred", "referral": {...}}`; anything else is a `ToolProblem`
(`planner.tool_results`): a stable `outcome` (`invalid_input`, `state_missing`
or `refused`), a `code` from one shared vocabulary naming what went wrong, a
`message`, and, where the next step is known, an `action` (which tool to
call next and with which arguments) in place of a hint a caller had to parse
out of prose.
