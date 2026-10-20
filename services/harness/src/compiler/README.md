# compiler

The compiler turns a plan's translation into the workflow that runs it, and
checks both against the plan's requirements and the live tool surface before
anyone approves the plan. It makes no judgement beyond the translator's choice
of operations, and everything it writes from that choice is checked.

`compiler` ships inside `harness` as a sibling top-level module
(`services/harness/src/compiler/`), on harness's dependencies, tests, and CI.
Its tests live in `services/harness/tests/compiler/`.

## What it does

| Module             | Role                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| `surface.py`       | Reads the live tool surface over MCP (`McpSurface`) and hashes each tool's argument schema             |
| `capabilities.py`  | The hand-maintained table of what each tool implements, takes and returns, pinned to a schema hash     |
| `sources.py`       | Reads a `search` envelope into the `SelectedSource`s a retrieval may name                              |
| `evidence.py`      | Compares a requirement with what is known about a retrieval; before approval, with the selected source |
| `arguments.py`     | Validates a step's literal arguments against the tool's own JSON Schema, holding bound ones out        |
| `compile.py`       | `compile_plan`: checks the translation, maps operations to tools, writes every argument, builds steps  |
| `frame_columns.py` | Binds a step's role columns to a retrieved frame's and bounds its StatVar count, for the executor      |

The records it consumes and produces live in `planner`: `Translation` and the
`Operation` union in `planner.operations`, `Workflow`, `WorkflowStep`,
`Binding` and `SelectedSource` in `planner.workflow`, and `mismatches` --
which proves an operation carries out the requirement it names -- in
`planner.proof`.

## The properties that define done

**The compiler writes every tool argument.** A `RetrieveOperation` names a
requirement and a `spec_id`, and the compiler writes `get_data`'s arguments. An
`AggregateOperation` names a reduction and what to group by in the plan's
vocabulary (`period`, `entity`), and the compiler writes the frame grammar's
period column and the role columns the requirement's own entities resolve
under -- not every role column the source happens to carry, since a source
often resolves a creditor alongside the debtor that was asked for. A `CompareOperation` names
entities by the requirement's names, and the compiler writes `{role_column: ref}`
entity arguments as literal args from what the source resolved. Only frame
references are `Binding`s. No proposal carries a tool name, a column name, a
dcid, or a frame reference.

**A retrieval names a source discovery recorded.** A retrieval's `spec_id` has
to be what discovery recorded for *its own* requirement -- naming a real
`spec_id` discovered for a different requirement is `source_not_discovered_ for_requirement` just the same as naming one nothing discovered. A requirement
whose `source_spec_id` pins one exact definition refuses any other with
`source_pin_violated`. The `SelectedSource` the workflow keeps is copied from
the search result, so the approved plan states what was known about its
source, and every non-blocking finding the compiler raised, so the record the
user approves carries its own caveats rather than leaving them in the turn
that produced them.

**One translation compiles to one workflow.** Step order, source order and
terminal order follow the requirement ids rather than the order the translator
happened to list its operations in, so two listings of the same operations
compile to the same bytes and therefore the same approved revision. Inputs
still precede the steps that read them.

**Coverage is a proof.** Every requirement is carried out by
exactly one operation (`requirement_unsatisfied`, `duplicate_satisfaction`,
`requirement_unknown` for an operation naming one the plan does not state),
and that operation repeats the requirement's calculation fields --
`planner.proof.mismatches` compares them field for field, so a `kind` that
does not correspond is `kind_mismatch` and any other disagreement (a
reduction, a window, a side's entity, an unresolved upstream reference) is
`operation_mismatch`, naming the fields. Beyond that: every operation feeds a
terminal one (`operation_unreachable`); every input names an operation whose
frame rests on one requirement, and no operation reads itself, directly or
through others (`dangling_input`, `input_is_a_comparison`, `cycle`). A
retrieved frame is a legal terminal when it answers a question that asks to
see a level or a trend: under `answer_mode="series"` whether or not a
calculation reads it, and under `answer_mode="computed"` only when one does
and an ask names it in `Plan.answered`. `Plan` already refuses an unread
dataset requirement under `computed`, so this module checks no answer mode of
its own. Any terminal may also be read by another operation, since an ask
answered by an interior result still gets that result. A comparison's result
answers the question and feeds nothing further.
Decisions are covered because the plan's requirements cite them.

**A requirement a source cannot meet is refused before approval.**
`check_evidence` runs on the selected source and compares it against the
requirement on seven counts: whether the source holds data; each entity by
name, and by role once a name resolves under more than one role column; the
requirement's period against the source's coverage window, under
`coverage.on_missing`; the currency and the price basis the source's unit
names; what the source's identity document binds, in entities and in the
window it was minted for; and the source the requirement pins by `spec_id` or
states in prose. No data, a missing or role-ambiguous entity, a currency or a
price basis that names something other than what was asked for, an entity the
definition does not bind, and a pinned or stated source that does not match
what was found are all errors, and the plan does not compile; a short coverage
window is an error only when `coverage.on_missing` is `"fail"`, and a warning
under `"report_gaps"`.

A stated role is checked by side of the flow. `creditor` and `debtor`,
`donor` and `recipient`, `lender` and `borrower` name a party whichever way
the money moves, so a requirement asking for one and a source resolving the
other is an error. A phrase the lexicon cannot place, and a role column it
cannot place, are warnings rather than silence -- the exact check needs
`search` to say which requested entity each resolved entity answers, which is
filed as issue #196.

A requirement that states `across` covers every member of each party it names,
one word per party, such as every donor of aid to Kenya (`['donor']`) or every
donor by every recipient (`['donor', 'recipient']`). `open_axes_for` matches
each word to one entity axis the source leaves open: `origin` for the giving
side, `destination` for the receiving side, and for a place noun such as
`country` the `observationAbout` axis when it is open over several members,
else the one flow axis open over several members that no other word covers, as
on a source keyed by debtor. The finding is `open_axis_unresolved`, an error,
when a word is narrower than a party (`bilateral donor`, `donor countries`),
when no open axis or more than one sits on a word's party, when a covered axis
holds fewer than two members or `search` did not count them, and when an open
axis no word covers holds more than one member or is uncounted, since each
member of the population would repeat once per member of it. At run time the
retrieved series have to hold at least two distinct entities under each
approved axis's column. A source recorded without its open entity axes is
discovered again for a requirement that states `across`
(`DiscoveryLog.selectable`).

What a source cannot confirm -- no entities resolved at all, no coverage
window, coverage stated as breadth rather than an exact window, no
recognisable currency, no price basis for a current-price requirement, no
provenance to check a stated source against -- comes back as a warning for the approval summary,
at the severity `coverage.on_missing` asks for. A source stating no identity
document at all is an error: nothing was written to the definition store for
it, so no retrieval could serve it and no drift check could protect it. No
operation restates a series in other prices, so a requirement in constant
prices is met only by a source already in constant prices of that base year,
stated by the unit the definition binds by request or by the unit id. A
retrieved series states its unit in those two forms alone, so `get_data`'s
manifest is checked against the same reading. A source in current prices, in
another base year, stating no basis or no base year, or stating constant
prices only in its label, is `source_unusable`.

**Each step is checked against the tool it will call.** Its literal arguments
validate against the live `inputSchema`, with bound arguments held out of
`required`; a bound argument names a parameter the tool declares; the kind of
value a binding takes is the kind its producer yields and its consumer expects.

**A tool the compiler builds calls for is pinned.** Each capability row carries
the hash of the argument schema it was written against, and -- once a server
declares one -- the hash of its output schema and its version too. A live
schema with a different pinned hash or version is `capability_schema_drift`;
a live schema declaring an output schema or a version the row pins as `None`
is `capability_unpinned`. Either way the plan does not compile, and `tests/compiler/test_capabilities.py` fails until the row is
re-read and re-pinned. Changing a tool the compiler targets therefore costs a
reviewed change here, which is the point.

**The six analysis verbs beyond aggregate, growth and compare.** `rank`
orders `RankRequirement`'s entities and writes the requirement's own role
columns as `rank_by`; `sum_of`, `share_of`, `quotient_of` and `join` are two-sided
like a comparison, writing
`{role_column: ref}` entity bindings the same way and, for `join`, the role
columns both upstream sources resolve in common as `join_keys` (else
`join_no_common_key_columns`); `filter` writes each `Predicate` as the frame
grammar's own `ordered_threshold` or `category_match` shape, resolving an
`EntityIn`'s names against the upstream requirement's entities and the
source, the same way a comparison's side does. A `PlaceLevel` becomes a
`place_level` predicate on `observationAbout_type`, and the retrieval that filter
reads, directly or through other filters, asks `get_data` for that column with
`place_types: true` (`_read_by_level_filter`). Every flow retrieval asks for
`without_party_totals: true`, so apis serves a frame holding each open side's parties
without the donors and recipients that total other parties, under a version of its own,
whichever verb reads it. Every kind gets `mismatches`'s
proof and its own operation-level checks (`entity_unresolved`,
`entity_ambiguous`, `entity_not_on_frame` for a name the frame it is read from
carries no column for, `keys_disjoint` for two keyed sides holding no entity in
common, `degenerate_sides`, `periods_disjoint`, and so on) exactly
as aggregate, growth and compare already did.

**`share_of` and `quotient_of` divide differently.** A `share_of` divides a part by
its whole, so two sides in different units, or one unit against none, are
`share_units_differ`, which the translator refers back to the planner for repair
with `quotient_of` among the permitted repairs. A `quotient_of` takes any pair of
units. It pins each side's own unit (`numerator_unit`, `denominator_unit`) when its
source declares one, and analysis-mcp states the result in the unit the two
compose. A quotient's numerator is keyed by `axis` and its denominator by
`denominator_axis`, each resolved on that side's own frame, and `denominator_by`
maps the numerator's key column to the denominator's. A numerator that still holds
several members with no `axis` is `quotient_axis_required`, and a key its frame
carries no column for is `quotient_key_unknown`; both are repairs the planner makes.

**`compare` and `sum_of` pair two sides in one unit.** analysis-mcp sets two
frames against each other only when their `unit` cells are equal. A
requirement may ask for current prices or for constant prices of one base year,
and a side whose requirement pins its source is always served in that source's
own unit. When the pinned source serves constant prices, and either states no
base year by request or unit id or the other side is pinned too, no
requirement change states both sides in one unit. The pair is then `frame_units_irreconcilable`:
it names the pinned side's requirement, the translator refers it as
`requirement_unmeetable` so the planner asks the user, and it records a
`no_operation_covers` answer so a plan that then shows the two series apart may
decline the pairing. Any other pair with a side in constant prices is
`frame_price_bases_differ`, repaired by stating one unit on both sides, in
current prices or in constant prices of one base year;
`frame_unit_forms_differ` and `frame_units_differ` cover the rest.

**Every issue is reported in one pass.** `compile_plan` never stops at the
first problem: a failed attempt returns the whole list, each issue naming the
operation, requirement or tool it is about, and a `Workflow` only when the list
is empty.
