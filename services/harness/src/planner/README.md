# planner

The Planner Agent drafts a methodological plan for a development finance question
and holds it at the user's approval. Nothing computes or retrieves until they say
yes.

It settles methodology and states what evidence the answer rests on, then stops.
Finding the data, proposing the operations and compiling the workflow are the
translator's and the compiler's, so no tool name, column or operation appears
here.

`planner` ships inside `harness` as a sibling top-level module
(`services/harness/src/planner/`), on harness's dependencies, tests, and CI. Its
tests live in `services/harness/tests/planner/`.

## What it does

Four tools, built by `build_planner_tools()` and wired into an `LlmAgent` by
`create_planner_agent()`:

| Tool                | Behaviour                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `adk_request_input` | Asks the user to choose among a referral's candidates or a refusal's options, suspending the turn until they answer |
| `propose_plan`      | Records a drafted plan as a `DRAFT`. Called again for every revision                                                |
| `reject_plan`       | Marks the current plan `REJECTED`, which blocks a declaration of readiness until a new version                      |

A `propose_plan` whose translation compiles declares the plan ready for the
user's answer in code (`declare_ready`), adds the approval summary to its
answer under `summary`, and ends the planner's run, so no model call follows.

The planner reaches the translator directly. `create_planner_agent` takes
`translator.tool`'s `build_translate` function as `translate`, which
`harness.agents` builds. The model cannot call it: every `propose_plan` that
records a plan runs it on that plan and answers with the verdict under
`translation`, beside its own `ok`, so the model spends no generation on
translation. The plan travels through session state. Omit it and
`create_planner_agent` returns the standalone planner, whose `propose_plan`
records the plan and stops; that is what the planner's own eval set scores
and why that set needs no reachable tool surface.

## Skills

`create_planner_agent` takes a `surface`, the harness's
`compiler.surface.ToolSurface`. Given one, the planner also carries
`get_skill` and `get_skill_file` from `planner.skills`, each calling data-mcp's
tool of the same name through that surface, and its instruction gains the
paragraph that names them. `harness.agents.create_planner_with_translator`
passes the same surface translation discovers on. Without a surface, as the
standalone eval agents build it, the planner has neither tool and the
instruction says nothing about skills.

The planner loads a skill when its description bears on the question. A
question about DFO itself starts with `get_skill`, whose `skill_id`
description names the one skill no search result carries, `dfo-faq`. Both tools answer with data-mcp's text, a refusal
included, so a deployment with no Skills API answers `capability_unavailable`
and the planner drafts without the skill.

## Typed results

Every tool in the table above answers with one of `planner.tool_results`' typed shapes
(`ok`, `problem`, `referred`) rather than an unstructured `{"error": ...}` a
caller has to parse back into a decision. A `problem` carries a stable `code`
from the shared `ProblemCode` vocabulary and, where the next step is known, an
`action` naming which tool to call next. The translator's core functions
answer the same way, from one vocabulary. `propose_plan`'s `translation` is
`ok` once the plan compiled, or the `Referral`'s own fields, with its
`outcome` from the closed set, read straight off the tool result. The
proposal's own `outcome` stays `ok` either way, since the version is recorded.
A transient referral is retried by proposing again with `amends_version` set to
the recorded version and nothing else changed: translation runs again on that
version, and no new version is recorded and no plan round is spent.

## The plan record

A `Plan` states the question, its `answer_mode` (`series`, the default, or
`computed`), every `Decision` the plan makes, the `EvidenceRequirement`s those
decisions fix, and, once translated, the `Translation` and `Workflow` that
carry them out (`planner.operations`, `planner.workflow`).

A `Decision` carries the alternatives considered, the choice, a one-line
rationale, and whether the user or the planner settled it. Its `slot` names it,
and a plan's slots are unique.

An `EvidenceRequirement` is one of a typed family. A `DatasetRequirement` states
what data an acceptable retrieved result rests on, in fields code can check a
retrieved frame against: the `measure`, the `entities` and optional `source` in
the words of the question; a closed `period` of calendar years, an `annual`
frequency, an optional `unit` keyed the way the frame grammar states one
(`currency`, `prices`, `base_year`), and what to do about missing years
(`coverage.on_missing`), and an optional `breakdown` that splits the total by a
classification's codes at one level, such as sector groups. Every other
calculation the plan needs is a derived requirement stated in plan vocabulary:
`AggregateRequirement` reduces an input's rows by `group_by` (period, entity,
or a breakdown's codes alone); `GrowthRequirement` states a
window analysis-mcp computes `(value_end - value_start) / value_start` over;
`ComparisonRequirement` takes the difference between two requirements' results
(`minuend` minus `subtrahend`) at each common period and answers the question outright; `SumOfRequirement`
adds two single-series results at each period both report, leaving out a
period only one reports; `RankRequirement`
orders every entity, and keeps the top or bottom N when it states `limit`,
reducing an input that spans several years to one value each first; `ShareOfRequirement` divides one
requirement's result by another's, period by period, when the two sources declare one unit
or both declare none; `QuotientRequirement` divides one measure by another
at each period both report, in the unit the two compose, either as one figure
or per entity of `axis`, where `denominator_axis` names the same entities
on the denominator, by the role they state there or `'country'` where they
state none, so each divides by its own counterpart; `JoinRequirement` joins
two requirements' results on the keys they share; and `FilterRequirement` keeps
the rows that satisfy every one of a conjunction of predicates: a value or
year threshold, an `EntityIn` naming entities, or a `PlaceLevel` keeping one
level of place. A dataset
requirement may pin `source_spec_id` to the exact dataset definition
discovery is to use, after which discovery has to find that definition and
nothing else. A pin needs no decision, since the requirement states it. A pin
of a dataset the planner was shown, a series the orchestrator fetched
included, is honoured from its block or its identity on record with no
search. A referral names the candidates' `spec_id`s, since pinning one is
how a `source_ambiguous` referral is settled.
`propose_plan` refuses a `SumOfRequirement` adding two datasets unless
it cites a decision resolved by `user` (`combination_unasked`): whether two
datasets add up without counting a flow twice, as ODA loans count in both
aid and debt flows, is the user's to settle. Tables of a provider's flows,
the OECD and ONE aid tables whose provenance Search serves with account
`provider` (`compiler.sources.PROVIDER_ACCOUNT`), are the exception: their
skill says how they add up, so a sum of them needs no user decision. Its
sides read one dataset, or only those tables, when every dataset
requirement they read pins a source whose provenance states it: a
block the planner was shown, or a source the translator's discovery
recorded. The check reads the pins the planner wrote, before `propose_plan`
moves a pin to its copy over the requirement's period. A `user` decision a
`SumOfRequirement` cites needs the user's pick on a combination ask: the
`combination_unasked` refusal's action carries the ask's options
(`planner.agent.COMBINATION_OPTIONS`), Studio shows them with no free-text
box, and the pick comes back as the option's id. A combination ask asks about
the sum the `combination_unasked` refusal before it names in its `detail`:
the requirement id and the datasets its sides read. Each refusal binds the
first ask after it and no other, so a second ask with no new refusal before
it asks about nothing. A dataset requirement whose dataset no pin states
reads a dataset of its own, named `spec:<source_spec_id>` for a pin whose
dataset the turn cannot read and `requirement:<id>:<revision>` for an
unpinned requirement, where the revision is the
`translator.discovery.requirement_revision()` hash of what it asks for. A pick
over one unknown side settles nothing for a sum over another, nor for the same
requirement id asking for something else. A sum over datasets that
are not all tables of a provider's flows passes `combination_unasked` only when the
user's latest pick for that requirement over those datasets is `add them`
and the sum cites a `user` decision whose `chosen` is `add them`. Typed
text, a skip, an answer to another question, or a pick for another sum or
for the same sum over other datasets does not count. A Studio turn opens its
own session, so a sum the amended plan already stated unchanged, over the
same datasets and citing the same `user` decisions to add them, keeps its
sum without a new ask, unless the user has answered any combination ask in
the session, about this sum, another, or nothing, with anything but
`add them`.

While a `pinned_source_not_found` referral withholds the sources found in
place of a pin, because they drop the pinned source's scope, `propose_plan`
refuses the requirement unpinned with its `entities` and `across` unchanged
(`source_not_offered`): the translator's fresh search would settle on a
withheld source. The planner makes the referral's `repair` or asks the user.

A pin's identity carries the years of the search that found it.
`propose_plan` moves each pin whose identity is on record to that identity
over the requirement's own period (`_window_pins`,
`data_mcp.definition_writes.windowed_identity`), so the plan retrieves the
years it states, and records the copy beside the original. One decision
cannot fix a pin over two periods.

Each requirement cites the decisions that fixed it
in `decision_ids` and, if derived, names the requirements it reads: the
requirements form a DAG, and a comparison is terminal: nothing may read one.
Under `answer_mode="computed"`, every dataset requirement has to feed some
calculation; under `answer_mode="series"`, a dataset requirement nothing
reads is itself the answer. Once a plan states any requirement, every
decision has to be cited by one.

The planner chooses between the two dividing kinds by what the question asks
for, without reading a source's unit: `share_of` for a part of its whole, both
in one unit, and `quotient_of` for one measure per unit of another, such as a
per-head figure or "divide one by the other". An `axis` or a
`denominator_axis` names a role its side's entities state, or `'country'` for
places that state none.

A dataset requirement whose `across` states the place party, such as
`'country'`, names no place (`planner.plan.names_no_place`). When its source
leaves `observationAbout` open, the retrieval holds every place with data at
every level: the world, regions, World Bank groups, countries and their
subdivisions, or the countries alone when dfo-apis cannot list them all. A rank,
an aggregate that does not group by entity, a `share_of` and a `quotient_of`
over it are refused until a filter with a `PlaceLevel` predicate keeps one
level, named by its Data Commons type (`PlaceType`: `Country`, `Continent`,
`AdministrativeArea1` and the rest). That filter reads the requirement directly
or through other filters, and states one level. The rule lives in
`planner.shapes.rejects` (`FrameShape.mixed_levels` and `place_typed`). The plan
permits the filter, since it cannot know which axis the word covers, and the
compiler demands it from the selected source
(`compiler.evidence.opens_place_axis`); analysis-mcp refuses the same calls at
run time as `place_levels_mixed`. Where the word covers a flow axis instead, as
IDS holds its debtors on `destination`, the axis holds parties whose kinds keep
totals out of a sum or a rank, and the compiler refuses a level filter there as
`level_filter_on_flow_axis`.
The planner's instruction tells it to take the level the question names,
choose `Country` as its own default when the question names none and reads
as a comparison of countries, and ask the user when the question's word
covers several levels ("by region"). The card names the level
(`harness.plan_display`): "Keep countries", "Rank the top 10 countries".

Every part of a plan is a `StrictModel`: frozen, and closed to fields it does not
declare, which makes a revision a value. `plan_revision()` hashes one (canonical
JSON under SHA-256, leaving out `status` and `approved_revision`), `approve()`
records that hash on the plan, and `start_execution()` refuses a plan whose
content no longer hashes to what was approved.

`PlanStatus` runs `DRAFT` → `PENDING_APPROVAL` → `APPROVED` | `REJECTED`, then
`EXECUTING` → `COMPLETED` | `FAILED` for whatever runs an approved plan. The
planner writes `DRAFT`; `with_translation`, which only the translator calls,
writes `PENDING_APPROVAL`. Every transition returns a new `Plan` rather than
mutating one, and each refuses the statuses it does not apply to: a plan is
translated only as a draft, approved only while pending, rejected only before a
run starts, completed or failed only while running.

`planner.settings.AgentSettings` holds the model the planner runs on,
`PLANNER_MODEL_NAME`, pinned to a concrete version rather than a floating alias
so an eval score stays attributable to one model. It sits here rather than in `harness.settings`, which
configures the service around the agents, so that reading a model string costs a
caller none of that surface.

## v1 completion criteria

Eleven properties: the first three gate submission, the rest keep `Plan` state
consistent.

**Submission requires a compiled plan.** `declare_ready` declares nothing for
a plan that is still a `DRAFT` or carries no workflow, answering
`workflow_missing` under `submission` with `action.call == "propose_plan"`. A
draft states what to do rather than what to run, so declaring one ready would
put nothing real in front of the user. A pending referral is checked first and
refused the same way, as `referral_pending`, with the referral itself in
`detail`. Its `action.call` is `adk_request_input` for an `unsupported_ask` or
`requirement_unmeetable` referral the user has not answered, and
`propose_plan` for every other case.

**Rejection blocks submission.** A rejected plan answers `plan_rejected`, with
`action.call == "propose_plan"`. The planner's own tools stay callable, so
`propose_plan` with `amends_version` naming the rejected version amends the
lineage into a fresh `DRAFT` version, which is how a user revises after
declining.

**Submission is bound to the plan on record.** `declare_ready` reads the plan
the proposal just recorded and translated, so the summary's version and
revision are that plan's. It approves nothing and gates nothing itself.
Approving the revision it reports, and running what an approved plan compiles
to, is the harness workflow's business.

**The planner never writes `PENDING_APPROVAL`.** It writes `DRAFT` when it
proposes and `REJECTED` when the user declines. The status that says a plan is
ready to approve is the translator's alone, minted by `Plan.with_translation`,
which refuses anything but a `DRAFT`; `APPROVED` is the harness workflow's to
write, once the user answers. ADK has no per-key owner for session state, so
this is an invariant a test holds rather than the framework.

**An amendment mints a new version.** Calling `propose_plan` again with
`amends_version` naming the recorded version amends rather than resets:
`version` increments, `previous_version` points back, `status` returns to
`DRAFT`, and `approved_revision`, `translation` and `workflow` are cleared,
since the operations were translated from decisions this version may no longer
make. A successful `propose_plan` also clears any pending referral: a new
proposal supersedes whatever the translator referred back about the version it
replaces. For an `unsupported_ask` or `requirement_unmeetable` referral,
`propose_plan` refuses until the user has answered an `adk_request_input` call
made after the referral, since a patch replaces a requirement whole and would
otherwise drop what the referral names. Under `harness.planner_seam` that ask
is made in code for a referral the user decides that carries candidates (a
`requirement_unmeetable` marked `source_mismatch`, or a `source_name_ambiguous`),
led by the referral's sentence from `harness.sentences`, so no model turn
composes it. A `requirement_unmeetable` referral
marked `source_mismatch` offers the sources discovery found in place of the
source the requirement states. A requirement the next proposal pins to the one
the user picked, answering an ask raised after the referral, drops its stated
`source`, so the compiler accepts the source they chose. A pin the planner chose
keeps the stated source, and the compiler refuses it again. The amended plan
passes through the same validators as a first draft. `starts_new_plan` begins a
plan at version 1 instead. The wording of `question` plays no part in that
choice, so a reworded question cannot be read as a new plan.

**`question_fixed` holds the turn's question against a restart.**
`create_planner_agent(..., question_fixed=True)`. The harness's planner
seam builds the planner this way, since the orchestrator's `plan` brief fixes
a turn's `question` before the planner ever runs, and a restart would
silently swap the question out from under the plan the workflow holds. Under it,
`starts_new_plan` with a plan already recorded is refused with
`question_fixed` rather than beginning a fresh plan, records nothing, and is
how the model tells the harness the user's reply asks a different question:
the turn ends there for the harness to tell the user to send it as its own
message. An amendment whose own `question` states something other than the
one recorded is refused as `invalid_plan` instead, correctable within the same
round. Standalone and `harness_pair_v1`'s own multi-turn session leave
`question_fixed` at its default `False`, where a restart behaves as described
above.

Rounds are capped. Every proposal opens a fresh discovery budget, so one entry
into the planner may make `MAX_PLAN_ROUNDS` proposals, an amendment or a
restart alike (counted under `PLAN_ROUNDS_STATE_KEY`, beside the plan rather
than on it, since a count on the plan would enter the revision it counts), and
a proposal past that is `budget_exhausted` with
`detail.scope == "plan_rounds"`, no `action`, and nothing touched; the last
version that compiled stays approvable. Every entry into the planner (a new
message, an answer that amends, a replan after a refused run) starts the count
over while `Plan.version` keeps climbing, so a plan can pass version
`MAX_PLAN_ROUNDS` without exceeding the budget. A proposal that fails
validation spends nothing.

**A `Decision`'s `chosen` value must be one of its `options`, and `options` has
to hold at least two of them.** Anything else fails validation, so a plan cannot
state a choice it never considered, and a decision whose only option is what was
chosen records no decision at all. `propose_plan` re-validates both lists itself,
because ADK's argument preprocessing falls back to raw dicts for a whole list
when any one item fails, and skips conversion entirely for a value that isn't a
list. Either would otherwise build a `Plan` from unvalidated data. A plan states
at least one requirement, and the plan-level rules (unique slots and ids, every
citation naming a decision, every decision cited, and a rank that collapses its
input) come back from `propose_plan` as
`invalid_plan`, carrying the validator's own message and, for a malformed list,
the pydantic errors in
`detail`.

**A requirement cites decisions the plan makes, and every decision is cited.**
Duplicate slots, duplicate requirement ids, a `decision_ids` entry naming no
decision, and a decision no requirement cites all fail validation. A decision no
requirement rests on is a decision about nothing, or a requirement the planner
left out. The requirements themselves have to agree with each other, too: every
input a derived requirement names has to be a requirement the plan states, the
requirements form a DAG, and nothing reads a comparison. Under
`answer_mode="computed"`, every dataset requirement has to be read by some
calculation; under `answer_mode="series"` (the default), a dataset
requirement nothing reads is itself the answer, one series per entity. A
`RequirementSide.entity` and a growth window each have to fit the one dataset
requirement their input chain rests on, when that chain is single-input all
the way down, and a rank whose chain still spans several years has to state a
reduction, since an entity with more than one row has no single value to
rank.

**A decision names what it settles, and the requirement citing it states the value.**
The model writes a decision to `propose_plan` as a `DecisionDraft`, whose
`fixes` lists field names only (period, unit, entities, measure, or a
calculation field). `settle_decisions` reads each value off the requirements
citing the decision and leaves out a field none of them states or that they
state differently, so a value lives once, on the requirement, and nothing is
refused for it. `Plan` holds every requirement citing a decision to each value
its `fixes` states, `source_spec_id` aside (`fix_disagreements`). The plan
record says whether the planner assumed a period, and the answer says a summed
reading was the user's. A rank's `among='population'` and a `share_of`'s
`denominator_scope='pooled'` each cite a decision whose `fixes` states the
claim. Code writes `fixes.source_spec_id`.
`propose_plan` sets it on the measure decision it writes for a pin of a measure
a settled source offers (`_measure_decisions`) and moves it with the pin. A
decision the plan being amended records goes once a patch leaves no requirement
citing it (`_uncited_carried_decisions`); one the patch itself states has to be
cited. A patch moving a value a carried decision fixes drops a planner's
decision with its citations, unless another requirement citing it still rests
on it, which is refused as `shared_decision_contradicted`
(`_contradicted_carried_decisions`). Until the user says something after the
plan was recorded (an answer, an amendment, a message, or a new turn:
`planner.agent._user_spoke_since_recorded`), each of their decisions stands as
they made it (`_user_decision_refused`). A patch removing one or leaving no
requirement citing it is refused as `user_decision_dropped`, and one changing
its `chosen`, `resolved_by` or `fixes`, or citing it from a requirement that
states another value, as `user_decision_contradicted`. Both point at
`adk_request_input`. Once the user has spoken, their decisions go the way a
planner's do.

**A period can follow the newest data.** A growth, a rank and a period threshold
state their years as the plan read them, and a `rolling` field beside the years
says which of them are relative to the newest year the data holds
(`planner.records.RelativePeriod`, an offset of 0 or less from `latest`). Every
rule that compares years reads the stated years. A relative period needs
`period_end_open` on each dataset requirement it rests on, and `propose_plan`
pins such a requirement to its dataset over a range with no end year
(`_window_pins`, `planner.plan.period_window`), which dfo-apis ends at the
current year at every mint. The range starts at the requirement's own first
year when only ends roll. When a start rolls, the newest year the data holds
can sit behind the one the plan read, so the range starts at the first year
the dataset holds (`planner.plan.full_history`, read from a Search answer over
no window). Search answers a question that names years over those years, so
when no answer on record is over no window `propose_plan` reads the pinned
identity over no window itself, with one `resolve_definition` call through the
planner's tool surface (`read_full_history`), and holds the answer in session
state for the turn (`planner.plan.full_histories`). That window is read for
its first year and never retrieved. An unpinned requirement whose start is
fixed stays unpinned, and the translator searches for it from that year on.
`propose_plan` refuses with `detail.issue` `rolling_needs_open_period` when a
dataset is closed (`dataset_closed`), has no identity on record where it needs
one (`no_source_on_record`), or has no first year because the read did not
complete or stated none (`first_year_read_failed`), `rolling_years_disagree` when the
offsets and the years differ, and `rolling_frame_too_large` when Search counted
more rows in the whole history than data-mcp's `max_frame_rows`.

**A plan is a value.** No part of a plan can be assigned to after construction,
and none accepts a field it does not declare. `plan_revision()` is therefore
stable across a JSON round trip and across approval, and changes whenever any
reviewed content does.

**A saved plan keeps its revision when a model gains a field.** A field added to
any model a plan holds carries `HASHED_WHEN_SET` (`common.hashed_fields`) in its
annotation and declares a default. `revision_content()`, which `plan_revision()`
and `translator.discovery.requirement_revision()` hash, leaves such a field out
while it holds that default, so a plan saved without the field hashes as it did,
and includes it once it holds anything else.
`tests/fixtures/revision_base_fields.json` lists the fields hashed
unconditionally, and `tests/planner/test_revision_format.py` fails for a field
that is in neither place.
