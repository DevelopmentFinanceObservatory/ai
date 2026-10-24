# translator

The translator turns a plan the planner has drafted into the workflow the user
approves, as code that `propose_plan` runs on every plan it records. It finds the data
each evidence requirement needs, proposes the operations that carry the plan's
decisions out, and has the compiler build and check the result. What cannot be
carried out goes back to the planner as a typed referral rather than being
approximated.

`translator` ships inside `harness` as a sibling top-level module
(`services/harness/src/translator/`), on harness's dependencies, tests, and CI.
Its tests live in `services/harness/tests/translator/`.

## What it does

One function, built by `translator.tool.build_translate()` and attached by
`harness.agents.create_planner_with_translator`, which the planner's
`propose_plan` calls on every plan it records. `propose_plan` writes the plan
to session state (`planner.plan.save_plan`, keyed on `PLAN_STATE_KEY`), and
translation reads it back itself (`translator.discovery.current_draft`,
`planner.plan.load_plan`). The planner's model never calls it, so it spends
no generation on it.

Translation runs the whole discover-then-compile loop as code
(`translator.mechanical.run_translate`): `discover_source` for every dataset
requirement, then `propose_operations`, following each step's own next
action. None of that costs a model call. It answers one verdict from a
closed set, which `propose_plan` returns under `translation`:

| Outcome           | Meaning                                                                        |
| ----------------- | ------------------------------------------------------------------------------ |
| `ok`              | The plan compiled; `declare_ready` reads the summary off the plan's record.    |
| `choice_required` | Several sources answer a requirement; the referral carries the candidates.     |
| `invalid_input`   | A requirement the planner repairs or the user amends; the referral says which. |
| `unsupported`     | No operation or search serves what the requirement asks.                       |
| `transient`       | A data service did not answer just now; proposing the same plan again retries. |
| `internal`        | A fault no plan change repairs, logged under the referral's `reference`.       |

Every outcome but `ok` is the typed `Referral` itself, its fields at the top
of the answer. A plan that is not a translatable draft answers with the
`ToolProblem` that says so.

When `discover_source` finds more than one source for a requirement and none
is pinned, translation closes on a `source_ambiguous` referral (outcome
`choice_required`) carrying the candidates as the discovery log recorded
them. The planner records the choice as a source decision naming one of them,
or puts the candidates to the user, and the pin settles the requirement on
the next translation. A `no_data`/
`unsupported_ask` answer refers straight to the planner instead: no wording
a model could try reaches an entity, role or unit that never resolved, and
the planner already re-asks the user with the question in hand.

Every referral carries an `outcome` from the planner's own `Outcome`
vocabulary (`translator.referral.OUTCOME_OF`): `choice_required`,
`invalid_input` for input the planner can repair, `unsupported`, `transient`
and `internal`. `translator.tool.issue_resolution` maps every
`CompileIssueCode` to one resolution: discover the source again, refer the
requirement for repair (naming the `constraint` and the permitted `repair`),
refer it as unmeetable or unsupported, or close as an `internal_failure`,
logged under the `reference` the referral carries. The planner seam ends the
planner's entry on an internal failure. A translation that fails with the
same issues it failed with before in the session is referred again marked
`repeated`, and the seam stops the planner's run on it.

`describe` is not a tool of its own: translation calls it from code
(`translator.discovery.fetch_definitions`), immediately before compiling, to
fetch a genuinely `definite` source's own definition and carry it into the
approval readback (`planner.workflow.SelectedSource.definition`, rendered by
`harness.plan_display._sentence_dataset`). Memoized per
requirement, keyed the way `selectable` itself is scoped
(`requirement_revision`), so a `COMPILE_FAILED` retry within one plan
revision reads what an earlier call already fetched rather than
re-describing the same source. A missing or failed fetch never fails the
compile: the readback renders what it renders without a definition.

A requirement stating a breakdown reads its source's named codes up the code
hierarchy, through the graph's node door
(`translator.discovery.fetch_code_paths`), immediately before compiling. The
paths set the level the breakdown reads (`compiler.shapes.breakdown_level`):
health (120) and population (130) share social infrastructure (100) at level
0, so a breakdown of the two reads level 1, and health alone reads level 2,
its subsectors. A node door that does not answer ends the proposal as
`surface_unavailable`, a transient referral, and the next translate reads
again. A code the graph holds no path for leaves the breakdown without a
level, and compile refuses it as `source_unusable`.

None of this is confirmation-gated. Discovery reads metadata; compiling
writes a workflow that nothing runs before the user approves it. The gate
is the harness workflow's `approve` node. The one read of observations is a
rolling workflow's newest years, below, which shows the user nothing and
records nothing.

Translation reads the tool surface through a `ToolSurface` (`compiler.surface`),
the same object the compiler checks against, so what it discovers on and what a
workflow is compiled against are one surface.

## The properties that define done

**Observations and computation are out of a model's reach.** The translator's tool
wraps `search`; `get_data` and the analysis verbs are not reachable from it,
so no instruction has to forbid them. Code reads observations once, for the
newest years of a rolling workflow. `describe` is reached only from code,
at compile time or to enrich a candidate, never by a model call.

**Discovery is bounded, per plan revision.** Every `discover_source` call
counts against `DISCOVERY_CALL_BUDGET` for the plan _revision_ it serves, not
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
`pinned_source_not_found` action. When discovery read the pin's own source
and set it aside because it holds closed a side `across` asks for, a source
the fresh search finds that drops that source's scope (a group or named
members on a role it leaves open or binds in part) goes under the referral's
`withheld`, never under `candidates`, so the planner cannot pin it. The
referral's `repair` then names the change that keeps that scope, when the
pinned source held the side to one group or one entity ("name 'Africa' as
recipient in `entities` and drop 'recipient' from `across`"). That holds
only while every requirement reading the pinned one sums the side away. Any
other reader, such as a rank or a filter, reads the side per member, which
naming one member cannot answer: the pin gives way, and the referral offers
the sources found as `candidates`. The pin's source is read for
this check from the discovery log too, when a search the translator ran found
it. A pin `propose_plan` moved to a copy over the requirement's period, or a
pin whose identity a search in the conversation recorded over that period,
is honoured from Search's answer for that identity with no `search` call,
read through data-mcp's `resolve_definition` on the surface the translator
calls (`translator.discovery.resolve_definition`, `resolved_spec`): the
sentence, coverage and StatVars are the ones Search states for the
requirement's period, and data-mcp stores the copy's definition as it
answers, so its retrieval finds it wherever data-mcp runs. When Search does
not answer that identity with that one dataset, the translator searches as
for a pin with nothing on record. One Search could not be asked about is
refused as `surface_unavailable` and referred back as
`discovery_unavailable`. `propose_operations` answers with each dataset
described once (`translator.discovery.described_sources`): a dataset several requirements
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
answered with candidates. A `requirement_unmeetable` referral carries candidates
only when it is marked `source_mismatch`, meaning the source the requirement
states names none of the sources discovery found. Those sources are offered for
the user to take one in its place. A period or price basis the source cannot meet
offers none. A referral names the requirement or decision it is
about, and both have to exist in the plan. Saved to session state under
`translator.referral.REFERRAL_STATE_KEY` rather than returned inline, so the
planner reads it on its own turn.

**Translation is compiled or it is nothing.** `propose_operations` runs
`compiler.compile_plan`. Only a workflow that compiles reaches the plan, through
`Plan.with_translation`, which is what moves a `DRAFT` to `PENDING_APPROVAL`.
Otherwise the plan is untouched and every issue comes back, with an `action`
when the issues mean the plan rather than the proposal has to change.

**The years approved are the years the run reads.** A workflow with a rolling
step has each group's newest year read before it reaches the plan, through the
run's own path (`executor.run.read_anchors`): the retrievals the group rests on,
whole-history copies included, and `resolve_anchor` over the same populations.
The read takes no longer than the planner entry has left
(`planner.plan.discovery_time_left`).
A group whose newest year is the one its rolling years state, or later, keeps
its years, since the run reads them moved later (`executor.run`). So does every
group when the read runs out of time, which the trace records as
`translation.anchor_read_skipped`, and a group whose read meets a method choice
or a source to discover again, since the run meets those as it does for any
plan and refuses a newest year before the stated one. A group behind has its
rolling years, in the plan and in the translation, moved to the newest year
(`planner.plan.restated_to_anchors`, `restated_translation`), and the moved plan
is compiled again. The `ok` answer then lists each requirement moved under
`restated`, with the newest year and its years before and after
(`planner.plan.Restatement`), and the approval card reads the moved plan. A plan
rule or a compile that refuses the moved plan comes back as a
`rolling_anchor_behind` issue on the requirement refused, naming the year and
quoting the refusal, a repair for the planner. A read the run would stop on
answers `run_would_stop`, whose `detail` names the step the run would report
(`executor.outcomes.failed_step`): its id, the run's own `code`, its message and
its requirement. The referral carries the code as `stop_code` and goes by whose
stop it is (`executor.outcomes.stop_cause`): a transient stop as
`surface_unavailable`, the harness's own fault as `internal_failure`, and a fact
of the data as `requirement_unmeetable` on that requirement.

**Only a draft is translated.** A plan in any other status is refused, as is one
that states no evidence requirements.

**Every core function returns a typed result.** Nothing answers `{"error": "..."}`.
Success is `{"outcome": "ok", ...}`; a referral saved is `{"outcome": "referred", "referral": {...}}`, which translation answers the planner with as the referral alone; anything else is a `ToolProblem`
(`planner.tool_results`): a stable `outcome` (`invalid_input`, `state_missing`
or `refused`), a `code` from one shared vocabulary naming what went wrong, a
`message`, and, where the next step is known, an `action` (which tool to
call next and with which arguments) in place of a hint a caller had to parse
out of prose.
