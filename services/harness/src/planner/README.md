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
or a breakdown's codes alone), and an empty `group_by` reduces every row to one
figure over every year, party and code; `GrowthRequirement` states a
window analysis-mcp computes `(value_end - value_start) / value_start` over;
`ComparisonRequirement` takes the difference between two requirements' results
(`minuend` minus `subtrahend`) at each common period and answers the question outright; `SumOfRequirement`
adds two single-series results at each period both report, leaving out a
period only one reports; `RankRequirement`
orders every entity, and keeps the top or bottom N when it states `limit`,
reading one year when `window_start` and `window_end` state the same year and
reducing an input that spans several years to one value each first otherwise; `ShareOfRequirement` divides one
requirement's result by another's, period by period, when the two sources declare one unit
or both declare none; `QuotientRequirement` divides one measure by another
at each period both report, in the unit the two compose, either as one figure
or per entity of `axis`, where `denominator_axis` names the same entities
on the denominator, by the role they state there or `'country'` where they
state none, so each divides by its own counterpart; `JoinRequirement` joins
two requirements' results on the keys they share; and `FilterRequirement` keeps
the rows that satisfy every one of a conjunction of predicates: a value or
year threshold, an `EntityIn` naming entities, or a `PlaceLevel` keeping one
level of place, each stated once, since analysis-mcp's filter refuses a predicate
stated twice, an `EntityIn` naming the same entities in another order
included. Two `EntityIn`s whose names resolve to one entity, such as one
naming it under its role and one under none, write one predicate too, and the
compiler refuses that filter with `bad_args`, by analysis-mcp's own op-spec
check (`compiler.arguments.op_spec_errors`). An `EntityIn` keeps only names
the dataset requirements behind its input state, under its `role` when it
gives one (`planner.plan.unknown_names`, which the compiler reads for
`unknown_entity`). A breakdown's codes are no entities. The frame holds the
labels apis decodes for them, which no plan knows, so a filter keeps no
sector by name, and a requirement whose measure names the sector reads it. A
two-sided step reads a side naming no entity as one series on every role but
its key, so a side over an `across` party nobody named is refused with the
pooling remedy (`planner.shapes.open_on_whole_side`, which the compiler reads
for `entity_required`). A dataset
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

An entity's words can name more than one graph node, as "Congo" names both
the Republic of the Congo and the Democratic Republic of the Congo. Search
then answers an `entity_name_ambiguous` referral. Each reading in its
`ambiguous_entities` holds one entity's `(name, role)` and the candidate nodes,
each a `ref`, its graph `label` and a `pick`, so "aid from Congo to Congo" gives
two readings, one per role. A `pick` names the requirement and the reading as
well as the node (`planner.records.entity_pick`), as
`flows|donor|Congo|country/COD`. The planner
settles a reading itself when the question or brief decides it and asks the
user otherwise, one ask per reading offering its candidates by `pick`, and
states the node in the entity's `dcid`, keeping its name and role. While the
referral is pending, `adk_request_input` refuses an ask whose candidates are
not all the picks of one reading. The referral's
`repair` carries that instruction (`translator.referral.REPAIR_OF`), so the
planner reads it only when such a referral is pending, and the planner's
instruction does not name the kind. Two entities of one requirement never state
the same `dcid`. `propose_plan` refuses as
`invalid_plan` a `dcid` that is neither among the candidates a pending
`entity_name_ambiguous` referral offers for that entity nor unchanged from the
plan being amended (`submission.service._entity_dcid_unoffered`). It refuses
too a `dcid` other than the node the user last picked for that entity, and one
stated for an entity an ask offered that the user has not answered
(`submission.service._entity_dcid_unpicked`). A pick keeps binding when Search
refers the same requirement again, since the new referral offers the same pick.
The latest answer about a reading stands, so an answer in the user's own words
releases an earlier pick, and a message the user types or an amendment of the
approval card releases every pick. An ask the tool or ADK refused offered
nothing. An entity nobody asked about, or one
the user answered in their own words, the planner settles.

A row asked for its members cannot pin a group's own node, so Search offers
such a candidate, the UN's "Central Africa" beside the African Union's, by the
group's registered name. The candidate carries that name as `restate_as` and
its `pick` ends with it in place of the node. The planner states it by setting
the entity's `name` to `restate_as`, `members` true and no `dcid`, and both
checks count such an entity as that candidate's, keyed by the reading whose
candidate it restates. Two readings of one role that offer the same group
restate it alike, so an entity restating it passes when either reading admits
that group, even if the user picked another candidate for the other.

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

An ask the orchestrator marks as totalled across a party (`AskBreakdown.over`)
holds a `share_of` or `quotient_of` answering it to a whole spanning that
party; a share keyed by the party and pooled across it needs a pool of two or
more. A denominator naming, on that side, only entities its numerator names
divides the member by itself, as the United States' ODA to Kenya over a total
naming the United States and Kenya does, and the plan is refused with the way
out as `whole_is_its_member`: keep that party open in the denominator's
`across`, name the group the user named as the whole, or, when translation
finds no source holding every member, decline the ask with the code its answer
names (`planner.plan._check_wholes_span_their_totals`). Names are compared as
the plan writes them, and an ask with no `over` is not checked. The
orchestrator's `plan` call is refused until an ask names in `over` the side a
search's `share_of` operation runs over, or in `by` when the search also held
that side open (`harness.orchestrator_tools._breakdowns_missing`).

A dataset requirement whose `across` states the place party, such as
`'country'`, names no place. When its source
leaves `observationAbout` open, the retrieval holds every place with data at
every level: the world, regions, World Bank groups, countries and their
subdivisions, or the countries alone when apis cannot list them all. A rank,
an aggregate that does not group by entity, a `share_of` and a `quotient_of`
over it are refused until a filter with a `PlaceLevel` predicate keeps one
level, named by its Data Commons type (`PlaceType`: `Country`, `Continent`,
`AdministrativeArea1` and the rest). That filter reads the requirement directly
or through other filters, and states one level. The rule lives in
`planner.shapes.rejects` (`FrameShape.mixed_levels` and `place_typed`, each
read from the frame's roles), and `planner.shapes.level_filter` names the role
each `PlaceLevel` keeps one level of. The plan permits the filter, since it
cannot know which axis the word covers, and the compiler demands it from the
selected source (`compiler.evidence.opens_place_axis`); analysis-mcp refuses
the same calls at run time as `place_levels_mixed`. Where the word covers a flow
axis instead, as IDS holds its debtors on `destination`, the axis holds parties
served without their group totals. A `PlaceLevel` naming that `role` reads the
flow side's type column, and sets `strict` to keep a party's countries, so
analysis-mcp refuses the filter when a party on the role has no known type.
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

## Years a step reads

A frame holds some years today, and newer data moves them as `planner.shapes.PeriodDrift` says. Each first or last year is fixed, follows the newest data, or, for a last year, is open to each series' own newest. A year stated with `rolling` follows the anchor, the newest year at least half the series hold (`resolve_anchor`), moved by its offset. The run resolves one anchor per group of rolling years (`planner.plan.rolling_groups`), and every rolling year of one group counts from it. A dataset retrieved with `period_end_open` keeps every later year a series reports, so one year of it counts as several while newer data can add years; a fixed last year still ahead of the data stops them only once the data reaches it. A filter from a year that follows the newest data therefore keeps one year only beside a last year that follows too. A dataset that a rolling-start reader reads through steps that keep observation years (`planner.plan.rolls_from_its_start`, `planner.plan.passes_observation_years`) is read whole: its retrieval starts at the first year on record (`Plan.first_years`), and only those readers and the steps between read it (`planner.plan.whole_history_split`), so each holds the years from there unless its own window or threshold sets a later start. Where the first year on record is the dataset's stated start, nothing splits. Where a step outside those paths or an answering result reads a step on them, the code gives the readers their own copies of it (`Plan.rolling_copies`) and the originals keep the stated years. A rolling reader above a collapse is admitted only beside another rolling-start reader that already makes its dataset a whole-history read, so the rolled-start check (`executor.run._check_rolled_start`) always finds a whole-history root under it. The model and the card see the plan as written, with no copy and no first year (`planner.plan.authored`, `harness.plan_display.shown_plan`, `translator.tool.authored_answer`), and every refusal and referral the model reads names the requirements it wrote (`planner.plan.authored_text`). A keyed two-sided step returns rows only for the keys both sides hold in a year. A `share_of` or `quotient_of` leaves out a numerator key with no denominator group, or one sharing no year with its group, and counts it in the result's `dropped_key_count`, which the answer states. A `join` counts each side's entities that match in no year, the left side's in `dropped_key_count` and the right side's in `right_dropped_key_count`, and the answer says each had no figure on the other side for the same year, naming both measures. An entity both sides hold in years that never overlap counts on each side. A per-axis `share_of`, or a `quotient_of` stating `denominator_axis`, whose two sides hold no entity in common has no key to divide. The plan does not refuse it, because two names can stand for one entity and only the sources know which; the compiler does, by the members each side's source resolves after its filters (`keys_disjoint`). A dataset requirement's own entities narrow nothing, since its retrieval holds every member its source resolves. A rank `limit` keeps entities only the data decides, so a numerator ranked with a `limit` pairs with a denominator filtered to France and Germany; when the ranked donor is neither, no key is left and the step refuses, which is the true answer. Every later step keeps the years an `ne` removed (`PeriodDrift.excluded`), so a later bound or window ending on one ends on the year beyond it. Whether a year stays inside years that move is written once, in `planner.shapes._stays`, which `years_run_out`, `anchor_year_stays`, `growth_endpoint_leaves`, `span_end_leaves` and `_picked_year_leaves` read. It compares two years by how each moves and, where both are known, by their offsets on the anchor and their order today: a first year a fixed one holds today, with a bound on the anchor behind it, stays at or after a bound on the anchor at an offset no later than that bound's (`PeriodDrift.start_offset`).

The anchor a run reads is the newest year the plan states or a later one. Translation reads each group's anchor before approval, through the run's own retrievals and `resolve_anchor` (`executor.run.read_anchors`), and states the group's rolling years from it where it is earlier (`planner.plan.restated_to_anchors`). A restated plan the rules refuse goes back to the planner (`rolling_anchor_behind`). The run refuses an anchor earlier than the stated one (`rolling_anchor_behind_plan`), which only data changed after approval can give, and the user asks again. Every rule here holds of years that move only later. Each rule also assumes stated coverage: each series of a dataset holds every year of the period Search states for that dataset, and newer years reach every series together. Under those two conditions an admitted plan reads a year at every step, its two-sided steps match periods, and newer data leaves each step years to read. Stated coverage is the assumption only for a series that starts late, lags or has stopped. Three checks cover those series. `planner.shapes.anchor_misses` refuses an open side the anchor is not resolved over, since a dataset outside the group can lag it. `executor.run._check_rolled_start` compares a rolled start with the retrieved frame's own first year, which a named entity can push past the first year Search states. `planner.records.PeriodThreshold` refuses a `gt` at offset 0, which asks for the years after the anchor. The plan cannot see a series that lags the newest year, starts after the stated first year or has stopped, so there the run decides, and it answers, refuses or records what it left out. A run whose answering step is empty refuses `empty_answer` (`executor.run`), except a filter, whose empty result is an answer (`compiler.capabilities.ToolCapability.empty_is_an_answer`). Each retrieved series is held to its requirement's period (`compiler.evidence._series_issues`). A series that starts late, lags or has stopped is recorded as `window_short` or `coverage_gaps` in the run's evidence (`executor.outcomes.EvidenceRecord.gaps`), or refused as `requirement_unmet` under `coverage.on_missing="fail"`. analysis-mcp refuses two sides that match no period (`comparison_periods_disjoint`, `share_of_grain_mismatch`), leaves out of a growth each entity missing an endpoint and names it (`endpoint_missing`), and counts the keys a `share_of`, `quotient_of` or `join` leaves out in `dropped_key_count` and `right_dropped_key_count`, which the answer states. A retrieval holding no series for an entity the plan names and the approved source resolved, such as a donor whose series stopped before the period, leaves it out and records it as `entity_missing` in the run's evidence, and the answer names it, while another entity the plan names on the same column of the approved source remains (`compiler.evidence._left_out_severity`). With none remaining, for a name the approved source never resolved, or under `coverage.on_missing="fail"`, the executor refuses the run with `requirement_unmet`, carrying that finding. A step that narrows to left-out entities alone, a side narrowed to one or a filter keeping only them, refuses the run before its first derived step, and the anchor read at translation before it resolves a group, with `requirement_unmet` naming that step and carrying the finding as an error (`executor.run._Run._check_left_out_narrowings`). A total over a span labels every key with the span across all keys, so a key whose series starts late, lags or has stopped totals fewer years under the same label. The answer states each series that misses part of the period the model wrote, or misses years inside it, such as "Germany's series ends in 2021" (`harness.graph.answer._short_series_sentences`), naming up to three in one note and counting the rest. A period open to the newest data (`period_end_open`) ends there at the anchor of the answer's own rolling steps over its retrieval (`executor.outcomes.AnchorResult.steps` and `roots`), or else at the newest year the retrieval holds, and the evidence check holds each series of it to that newest year (`compiler.evidence._held_period`). Under rolling steps, the answer states only the series their anchor was taken over (`AnchorResult.populations`), so a sibling answer over the same retrieval changes neither its window nor the series it names. A series with an entity the approved source gives no label is counted without a name, and so is a series whose name another in the note shares, such as each StatVar's series of one entity. Series of several requirements are named with their measure.

`propose_plan` refuses:

- A step that turns more than one year into one figure per entity, unless every ask its result answers is marked `(across years)`: `period_reduction_unstated`, from `planner.plan.unstated_period_reductions`. Such a step is an `AggregateRequirement` whose `group_by` leaves out `period`, an empty one included, or a `RankRequirement` with a `reduction` and `per_period` unset whose window keeps more than one year. An aggregate that keeps `period` in its `group_by`, a rank stating `per_period`, and a rank whose `window_start` and `window_end` state the same year reduce no years. Every answering result computed from the step, the step included, answers at least one ask, and each ask it answers carries the mark. The orchestrator sets the mark on an ask from the user's words, the plan holds the marked asks' numbers as `asks_across_years`, and the planner reads the mark beside the ask in its message.
- A rank with no `reduction` whose window does not stay one year, with both ends fixed on it or both following the newest data at one offset: `invalid_plan`, from `planner.plan._rank_reduction_reason`.
- A first year that follows the newest data, in a filter, a rank window or a growth window, below a fixed last year, whichever step states either: a fixed `lte` or `eq` in the same filter, an earlier filter, or the other side of a two-sided verb. Each new year moves the first year toward that last year until no year is left. A rolling bound behind a later fixed first year passes it once the newest year has moved far enough, and is refused the same way: `rolling_years_run_out`, from `planner.shapes.years_run_out`.
- A growth window year that newer data takes out of its input. A growth reads each entity's values in exactly its window's first and last years, so a window year that follows the newest data needs an input whose last year no fixed bound caps, and a fixed window year needs an input whose first year does not follow the newest data: `rolling_years_run_out`, from `planner.shapes.growth_endpoint_leaves`.
- A growth window whose first or last year is one an `ne` removed, which leaves no entity both endpoints: `invalid_plan`, from `planner.shapes._rejects_growth_window`.
- Two sides of a two-sided verb that share no year, or only years an `ne` removed: `invalid_plan`, from `planner.shapes._rejects_paired_periods`.
- A two-sided step pairing a side whose years follow the newest year with a side open to each series' own newest that the newest year is not resolved over: `rolling_side_unanchored`, from `planner.shapes.anchor_misses`.
- Rolling years that stand for different newest years where a run reads one, or for a newest year later than the datasets they read hold: `rolling_years_disagree`, from `planner.plan._check_rolling_years_agree`.
- A year that follows the newest data over a dataset retrieved with a closed end: `rolling_needs_open_period`, from `planner.plan._check_rolling_reads_open_periods`.
- An `ne` that follows the newest data, or removes a year from a dataset retrieved with `period_end_open`. An `ne` names a fixed year over datasets of fixed years alone: `rolling_exclusion`, from `planner.plan._check_exclusions_fixed`.
- A `gt` with `rolling` at offset 0, which keeps only years after the newest, years most series lack: `invalid_plan`, from `planner.records.PeriodThreshold`.
- A span that newer data moves out of the window or side that reads it: `invalid_plan`, from `planner.shapes.span_end_leaves`. A step that collapses years labels its rows with one span: a total with every year it summed, across all its rows, a growth with its window, and a rank with the years its window kept. A rank window keeps a span row only while it holds the span's years, and the other side of a two-sided verb matches it only while it carries the same span. So a fixed window or a fixed span meets no span whose years newer data moves, and two spans that move match only where one step wrote both (`planner.shapes.FrameShape.span_of`) or both follow the newest data alike. A total of years open to the data ends at the newest year each dataset reports, so it pairs only with a span the same step wrote, such as a total of that total by period. A value filter keeps the rows whose values pass, so which years a total over them spans only the data decides (`planner.shapes.PeriodDrift.decided_by_data`), and that total too pairs only with a span the same step wrote. No change to such a pair computes the figure it states, so the refusal offers none. A rank window reads its stated years as a bound, since the label lies inside them.
- A max or min by entity, matched on each entity's own year against a series that misses a year the max or min can fall in, today or as newer data moves either side, from which a filter removed such a year alone, or whose rows a value filter kept, which can leave out any such year: `invalid_plan`, from `planner.shapes._rejects_paired_picked`. A max or min over a dataset read from its first year can fall in any year back to it, so the series has to be read from its first year too, or the max or min's input bounded from the series' first year, following the newest data where the series' first year does. A series and a max or min read from the same datasets may both stay open to the newest data.
- A year filter over a max or min by entity whose thresholds keep a year that newer data can move a picked year past. A picked year moves only later, and only while the max or min's input years move. So a fixed first year keeps every entity whose max or min falls at or after it, today and later, and a first year that follows the newest data keeps them only while the input's own first year follows it at the same offset or further ahead. A last year keeps them only while the input's last year sits at or before it today and stays there, unless the input's years do not move at all. The entity whose max or min moves past the year drops out: `invalid_plan`, from `planner.shapes._picked_year_leaves`. Filter the years before the max or min. A filter with no year threshold keeps every entity it keeps today.

Where the planner reads the user's own words, a step is also admitted when the `propose_plan` call names it in `across_years`. The standalone planner and `harness_pair_v1` read them in every entry, and `harness.planner_seam` builds its planner with `reads_user_words` false, so there only an entry the user's amendment at the approval card opened reads them (`planner.plan.amendment_entry`). Elsewhere a call naming a step is refused `invalid_plan`, unless the plan it amends already licenses that step as the call states it, which grants nothing new. A call naming a step that keeps its years is refused everywhere. Each named step becomes a `PeriodLicence` in `Plan.period_licences` (step id, kind and `reduction`), which `propose_plan` writes and the model cannot (`planner.plan.period_licences_held`). `Plan.amend` clears the field, and `propose_plan` carries each licence of the plan it amends onto the new one while that step keeps its id and kind and combines its years with the licensed `reduction`, whatever its years: a replan that narrows a licensed total or ranking to one year and a later one that restores its years both keep it. The licence lies unused while a licensed rank states no `reduction`, and admits the rank again once it states the licensed one. An aggregate grouping by `period`, a rank set `per_period`, a step stating another reduction, a step of another kind, or a removed step drops the licence, since each states another figure. A step renamed in an amendment entry holds none of its old id and takes a new one when the call names it, and a restart begins with no licences. The field is hashed once it holds a licence, so a plan holding none keeps its revision.

The `period_reduction_unstated` refusal's `detail.gaps` holds one entry per refused step, in the order `unstated_period_reductions` finds them. Each entry holds:

- `requirement_id`, the step.
- `output_id`, the result computed from it.
- `asks`, the numbers of the asks that result answers.
- `years`, the `start` and `end` of the years the step holds today, with `start` the first year on record where the step reads its dataset whole.
- `follows_newest`, whether newer data adds later years.
- `until`, the last year a fixed bound lets them reach, `null` when nothing stops them.
- `dataset_requirement_id`, the dataset requirement the years come from, the first of them for a step over several.
- `keep_years`, the change that keeps the step's years: `{"group_by": [...]}`, the aggregate's group with `period` added (`entity` and `period` where the group already names two columns), or `{"per_period": true}` for a rank. It is `null` where another result computed from the step answers an ask across years and the step stays as it is.
- `one_year`, the change that has the step read one year, which edits that step alone, and `together_with`, the other refused steps whose changes the plan has to make with it.

`one_year` is `null` where another result computed from the step answers an ask across years and the step stays as it is. It is also `null` where neither year below gives a plan that validates under the refused plan's first years on record (`planner.plan.one_year_followed`, the same change applied to the step alone), and the plan making the change on every such step at once does not pass either. Where that plan passes, as for the two sides of a quotient of two totals over the same years, where either side read alone in one year matches no row of the other's span, each of those steps is offered its year with `together_with` naming the others, and the plan following one alone is refused (`planner.plan._one_year_offered`). Each step is tried on its first year below, then each on its second. `together_with` is empty for a change admitted alone. A change that edits a window a decision the rank cites fixes gives no such plan, so the years a user chose stay as they chose them. Its `change` is a `submission.service.OneYear` value, `window` to set a rank's `window_start` and `window_end` on `year`, or `filter` to have an aggregate read its own input through a `filter` keeping `year` with op `eq`.

`year` is the last year the step holds today, `years.end`, except where it follows the newest data and the last year is open to each series' newest: there it is the newest year the step's other rolling years stand for (`planner.plan.rolling_groups`), moved by `offset`. A rolling year has the step read its dataset whole, from the first year on record, through copies of its own where another result needs the stated years (`planner.plan.whole_history_split`). So where no first year is on record for the dataset (an unpinned open dataset, a pin with no first year on record, or a whole history over `max_frame_rows`), where the rules refuse the rolling year, or where the step's other rolling years stand for several newest years, the change offers `years.end` fixed. `offset` is `null` for a fixed year, which the change states wherever a fixed bound stops the input's last year, and otherwise the `rolling` offset the change states: that of the input's last year where a bound sets it, one less than the bound's own for `lt`, or 0 where the last year is open to the data (`planner.plan.YearsHeld.one_year`).

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
source a pending referral offers in place of a pin counts as settled, so a
re-pin to it states its measure decision too. A
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

**A period can follow the newest data.** A growth, a rank and a period threshold state their years as the plan read them, and a `rolling` field beside the years says which of them are relative to the newest year the data holds (`planner.records.RelativePeriod`, an offset of 0 or less from `latest`). Every rule that compares years reads the stated years. A relative period needs `period_end_open` on each dataset requirement it rests on, and `propose_plan` pins such a requirement to its dataset over a range with no end year (`_window_pins`, `planner.plan.period_window`), which apis ends at the current year at every mint. The range starts at the requirement's own first year when only ends roll. When a start rolls, the newest year the data holds can sit behind the one the plan read, so the rolling-start readers read the dataset whole, from the first year the dataset holds (`planner.plan.full_history`, read from a Search answer over no window, kept in `Plan.first_years`), and every other step keeps the range from the requirement's own first year. Search answers a question that names years over those years, so when no answer on record is over no window `propose_plan` reads the pinned identity over no window itself, with one `resolve_definition` call through the planner's tool surface (`read_full_history`), and holds the answer in session state for the turn (`planner.plan.full_histories`). That window is read for its first year and never retrieved. An unpinned requirement whose start is fixed stays unpinned, and the translator searches for it from that year on. `propose_plan` refuses with `detail.issue` `rolling_needs_open_period` when a dataset is closed (`dataset_closed`), has no identity on record where it needs one (`no_source_on_record`), or has no first year because the read did not complete or stated none (`first_year_read_failed`), `rolling_years_disagree` when rolling years that one run counts from one newest year, in one step or in steps one result reads (`planner.plan.rolling_groups`), less their offsets, give different newest years or one after the last year of a dataset they rest on (its `period`, the years Search found it over), `rolling_years_run_out` when a first year that follows the newest data meets a fixed last year (`planner.shapes.years_run_out`), or newer data takes a growth window's first or last year out of its input (`planner.shapes.growth_endpoint_leaves`), `rolling_side_unanchored` when a two-sided step pairs a side whose years follow the newest year with a side open to each series' own newest year that the newest year is not read over (`planner.shapes.anchor_misses`), since the run reads the newest year over the datasets the rolling steps counting from it reach (`planner.plan.anchored_over`, `compiler.rolling.anchor_groups`), so the open side's dataset can lag it, and a rolling bound on a step reading both sides reads it over both, `rolling_exclusion` when a filter's `ne` follows the newest data or reads a dataset stating `period_end_open`, where newer data can make the removed year the one a step reads (the years before the newest are `lte` with `rolling` at offset -1), and `rolling_frame_too_large` when Search counted more rows in the whole history than data-mcp's `max_frame_rows`.

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
