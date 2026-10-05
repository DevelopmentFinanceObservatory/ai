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

| Tool                | Behaviour                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `adk_request_input` | Asks the user to choose among a referral's candidates, suspending the turn until they answer |
| `propose_plan`      | Records a drafted plan as a `DRAFT`. Called again for every revision                         |
| `reject_plan`       | Marks the current plan `REJECTED`, which blocks `submit_plan` until a new version            |
| `submit_plan`       | Declares the current plan's compiled workflow ready for the user's answer                    |

The planner reaches the translator directly. `create_planner_agent` takes
`translator.tool`'s `translate` function tool, which `harness.agents` builds,
alongside the four above and, when the planner is built with a surface, the
two skill tools below. The planner calls it with no arguments inside its own
turn, and the plan travels through session state. Omit it and
`create_planner_agent` returns the standalone planner, which is what the
planner's own eval set scores and why that set needs no reachable tool
surface.

## Skills

`create_planner_agent` takes a `surface`, the harness's
`compiler.surface.ToolSurface`. Given one, the planner also carries
`get_skill` and `get_skill_file` from `planner.skills`, each calling data-mcp's
tool of the same name through that surface, and its instruction gains the
paragraph that names them. `harness.agents.create_planner_with_translator`
passes the same surface `translate` discovers on. Without a surface, as the
standalone eval agents build it, the planner has neither tool and the
instruction says nothing about skills.

The resolve envelope's `skills` field names the skills that apply to its
results, by id and description, and the planner loads one when its
description bears on the question. A question about DFO itself starts with
`get_skill`, whose `skill_id` description names the one skill no search result
carries, `dfo-faq`. Both tools answer with data-mcp's text, a refusal
included, so a deployment with no Skills API answers `capability_unavailable`
and the planner drafts without the skill.

## Typed results

Every tool in the table above answers with one of `planner.tool_results`' typed shapes
(`ok`, `problem`, `referred`) rather than an unstructured `{"error": ...}` a
caller has to parse back into a decision. A `problem` carries a stable `code`
from the shared `ProblemCode` vocabulary and, where the next step is known, an
`action` naming which tool to call next. The translator's core functions
answer the same way, from one vocabulary. `translate` answers `ok` once the
plan compiled, or the `Referral`'s own fields, with its `outcome` from the
closed set, read straight off the tool result.

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
at each common period and answers the question outright; `SumOfRequirement`
adds two single-series results at each period both report, leaving out a
period only one reports; `RankRequirement`
orders every entity, and keeps the top or bottom N when it states `limit`,
reducing an input that spans several years to one value each first; `ShareOfRequirement` divides one
requirement's result by another's, period by period, when the two sources declare one unit
or both declare none; `JoinRequirement` joins
two requirements' results on the keys they share; and `FilterRequirement` keeps
the rows that satisfy every one of a conjunction of predicates: a value or
year threshold, an `EntityIn` naming entities, or a `PlaceLevel` keeping one
level of place. A dataset
requirement may pin `source_spec_id` to the exact dataset definition
discovery is to use when it found several, after which discovery has to find
that definition and nothing else; a requirement carrying the pin has to cite
a decision whose `fixes` names `source_spec_id`.
A referral still names the candidates' `spec_id`s, since pinning one is
how a `source_ambiguous` referral is settled: what holds the pin to a
decision is the plan's own rule, not withholding the id. Who resolved that
decision -- the user or `planner_default` -- is not this validator's to
check; `resolved_by=user` is model-self-reported, so `Plan`'s own
validators cannot verify it, having no session to read. `propose_plan`
checks it instead, against what the user chose: every spec_id the user
picked answering the planner's `adk_request_input` tool call (read from the
session's event history, and not merely one it offered), and every dataset
the user handed to Studio from Search (`planner.plan.seeded_sources`). A
`resolved_by=user` source pin naming anything else -- never offered, offered
but not chosen, or offered with no answer on record yet -- is refused there,
before a plan carrying it ever reaches this validator.

A turn that starts from Search carries seeds: the datasets the user saw,
each with the operations the question asked of it. The planner's brief lists
them with their open entity axes, and the plan gives each its own dataset
requirement pinned to the seed's spec_id. `propose_plan` moves that pin to
the seed's identity over the requirement's own period
(`_window_pins`, `data_mcp.definition_writes.windowed_identity`), so
the plan retrieves the years it states, and records the copy beside the seed.
One decision cannot fix a seed over two periods. When a seed's StatVars have
left its dataset, the turn drops it (`planner.plan.dropped_seeds`) and tells
the user the plan leaves it out; `propose_plan` then refuses any plan that
pins it or its copy, whoever the decision says resolved it, as
`seed_dropped`, and checks an amended plan whole, since a patch carries
forward what it does not name. `planner.plan.names_dropped_seed` is the one
rule for which dataset that is, asked here of a pin and by the translator of
a source; a seed the turn still holds is never read as dropped. Each requirement cites
the decisions that fixed it
in `decision_ids` and, if derived, names the requirements it reads: the
requirements form a DAG, and a comparison is terminal: nothing may read one.
Under `answer_mode="computed"`, every dataset requirement has to feed some
calculation; under `answer_mode="series"`, a dataset requirement nothing
reads is itself the answer. Once a plan states any requirement, every
decision has to be cited by one.

A dataset requirement whose `across` names the place party, such as
`'country'`, names no place (`planner.plan.names_no_place`). Its retrieval holds
every place with data at every level: the world, regions, World Bank groups,
countries and their subdivisions, or the countries alone when dfo-apis cannot
list them all. A rank, an aggregate that does not group by entity, and a
`share_of` over it are refused until a filter with a `PlaceLevel` predicate
keeps one level, named by its Data Commons type (`PlaceType`: `Country`,
`Continent`, `AdministrativeArea1` and the rest). That filter reads the
requirement directly or through other filters, and states one level. The
rule lives in `planner.shapes.rejects` (`FrameShape.mixed_levels` and
`place_typed`), so the plan and the compiled workflow are held to it alike,
and analysis-mcp refuses the same calls at run time as `place_levels_mixed`.
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

**Submission requires a compiled plan.** `submit_plan` refuses a plan that is
still a `DRAFT` or carries no workflow, as `workflow_missing` with
`action.call == "translate"`. A draft states what to do rather than what
to run, so declaring one ready would put nothing real in front of the user. A
pending referral is checked first and refused the same way, as
`referral_pending`, with the referral itself in `detail`. Its `action.call` is
`adk_request_input` for an `unsupported_ask` or `requirement_unmeetable`
referral the user has not answered, and `propose_plan` for every other case: a
referral has to be settled, with the user where the planner cannot choose, and
the plan re-proposed before anything compiled is submitted. A stale version (the
plan was amended since this one was reviewed) is `version_stale`, with the
current and given versions in `detail`.

**Rejection blocks submission.** After `reject_plan`, `submit_plan` answers
`plan_rejected`, with `action.call == "propose_plan"`. The planner's own tools
stay callable, so `propose_plan` with `amends_version` naming the rejected
version amends the lineage into a fresh `DRAFT` version, which is how a user
revises after declining.

**Submission is version-bound.** `submit_plan` takes the version being
submitted and refuses any value other than the current one, so a plan amended
after review cannot be declared ready on the strength of the old look at it.
`submit_plan` approves nothing and gates nothing itself: it reports the
version, revision and summary of whatever the current plan already is, and
calling it again with the same version reports the same thing again. Approving
the revision it reports, and running what an approved plan compiles to, is the
harness graph's business.

**The planner never writes `PENDING_APPROVAL`.** It writes `DRAFT` when it
proposes and `REJECTED` when the user declines. The status that says a plan is
ready to approve is the translator's alone, minted by `Plan.with_translation`,
which refuses anything but a `DRAFT`; `APPROVED` is the harness graph's to
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
otherwise drop what the referral names. The amended plan passes through the same validators as a first draft.
`starts_new_plan` begins a plan at version 1 instead. The wording of `question`
plays no part in that choice, so a reworded question cannot be read as a new
plan.

**`question_fixed` holds the turn's question against a restart.**
`create_planner_agent(..., question_fixed=True)` -- the harness graph's own
seam builds the planner this way, since the graph derives and narrates a turn's
`resolve_envelope`, `depth` and `question` before the planner ever runs, and a
restart would silently swap the question out from under that state. Under it,
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
citation naming a decision, every decision cited, a decision's `fixes` agreeing
with the requirements citing it, a pinned source citing a decision that fixes it,
and a rank that collapses its input) come back from `propose_plan` as
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

**A decision and the requirement citing it have to agree.** A decision's
`chosen` is prose, so citation on its own enforces reference rather than
relevance: a requirement could name a period, unit or measure that contradicts
the very decision it rests on and both would validate. `Decision.fixes` states
the same choice in the requirement's own vocabulary (period, unit, entities,
measure, source), and every field it sets is compared with the citing dataset
requirement's own. A field it leaves unset is not compared, so a decision says
only as much as it means to. The model writes a decision to `propose_plan` as a
`DecisionDraft`, whose `fixes` lists field names only.
`settle_decisions` reads each value off the requirements citing the decision,
and refuses a named field no citing requirement states (`fix_unstated`) or one
they state differently (`fix_disagrees`). The decision `propose_plan` writes
for a pin a search settled on is dropped once no requirement cites it.

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
