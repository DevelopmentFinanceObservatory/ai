# Studio wire contract v1

> Frontend developers: read [`frontend-interface.md`](frontend-interface.md)
> first. It is the self-contained integration guide. This document records
> the contract's design decisions and its numbered deltas.

The interface between the DFO harness (this repo) and the website's Studio
experience (chat + canvas). This document and the pydantic models in
`services/harness/src/harness/wire.py` are the contract's source of truth:
the models are what the server actually validates and emits, this document
explains the protocol around them, and `services/harness/tests/test_wire.py`
pins the two together against the fixtures in
`services/harness/tests/wire_fixtures/studio/`.

v1 records its design decisions as numbered deltas (1–35; 16 is folded
into 11). The deltas are marked where they apply and listed in the delta
index at the end.

The contract is frozen at `schema_version: 1`. Agent internals (planning,
tool selection, models) will keep moving underneath it; a
breaking structural change bumps the version and ships a migration note.

## Endpoints

| Method | Route                     | Returns                                                                |
| ------ | ------------------------- | ---------------------------------------------------------------------- |
| POST   | `/api/studio/turn`        | SSE stream of frames                                                   |
| POST   | `/api/studio/interrupt`   | a NEW SSE stream resuming the paused turn (delta 1)                    |
| POST   | `/api/studio/resume`      | a NEW SSE stream carrying on a running turn its client left (delta 26) |
| POST   | `/api/studio/cancel`      | 204, or 200 for a Stop naming another turn (deltas 14, 32)             |
| POST   | `/api/studio/composition` | the new composition manifest, as a JSON body                           |
| GET    | `/api/studio/state`       | the conversation as stored, as a JSON body; takes no lease (delta 26)  |
| GET    | `/api/studio/places`      | the places a card's role can change to, as a JSON body (delta 35)      |

The Studio's other routes (`/api/sources`, `/api/datasets/{spec_id}`,
gallery, published views) are website/platform reads served outside the
harness.

## Auth and conversation identity (delta 5)

Every studio request carries:

- `Authorization: Bearer <token>`: required; a request without it is 401.
  In v1 the harness treats the token as opaque: a presence check, plus a
  shared-secret comparison when the `STUDIO_API_TOKEN` setting is present.
  **This is placeholder access control.** A server with the setting absent
  accepts any non-empty token and logs a warning at startup. The
  verification check is one function in `harness/auth.py`, where full JWT
  validation plugs in later without touching routes.
- `conversation_id` in the body: client-minted, stable for one
  conversation. `session_id`, its earlier name, is an unknown member and
  is refused.

Server-side conversation state is keyed on (principal derived from the
token, `conversation_id`), so conversations are isolated per principal.
Each conversation also gets its own presentation-platform session (the
platform leases canvas components to the `x-dfo-session` that created
them, verified live): the harness holds one platform-facing client per
conversation, so a conversation's components group under one platform
session. The header value is a per-conversation transport id that the
server maps to the `conversation_id`.

That keyed state (the turn index, pending interrupts, plan version
lineage, the composition manifest, turn records, and plan approvals)
persists in the harness's own Postgres tables, one conversation row and its
per-item rows. Any harness instance can load it and continue: a plan
approval, an elicitation answer, and a follow-up turn all land correctly
whichever instance receives them. A conversation streams at most one
segment at a time, on any instance: a durable segment lease on the
conversation's row is claimed by `/turn` and by `/interrupt`'s
resume, expires 45 s after it is last renewed, and is renewed by its
owner every 5 s; a second stream on a live segment gets the 409
`TRANSIENT_UNAVAILABLE` below with `retry_after_seconds` set to the
lease's remaining time. The owning instance saves the full conversation
state into its row, and clears the lease, in one write before any frame
that says "paused" (`plan.proposed`, `elicitation.request`) or "done"
leaves the wire. A write that fails ends the stream with a stream-fatal
`transient_unavailable` error in that frame's place, so any success
frame the client sees already matches the stored conversation. Every
stream ends on exactly one ending frame, which is its last. The
database is a deployment setting (`DATABASE_URL` or the `SESSION_STORE_*`
family, see `services/harness/README.md`). The harness refuses to start
without it. A stalled owner that lost its segment cannot write the
conversation or its turn, whose writes are fenced on the lease, but a
presentation-platform write already in flight can land as an orphaned
component. One piece of state stays bound to the instance that produced
it regardless of the database: editing a component requires the
instance that created it, because the presentation platform leases a
component to the MCP session that created it (above), and that session,
and the frame lineage behind it, exist only in that instance's process.

A `composition.mutate` body also carries `conversation_id` (delta 5; the
`request-05` fixture carries it).

## Requests: five kinds, discriminated on `kind`

Models: `TurnRequest`, `InterruptResponse`, `CompositionMutation`,
`CancelRequest`, `ResumeRequest` (`STUDIO_REQUEST_ADAPTER` parses the
union). `GET /api/studio/state` takes query parameters and no body.

- **`turn.request`** starts or continues a turn. `message` is the user's
  prose. `surface` ∈ `studio | download | embed` and is **optional with
  default `"studio"`** (delta 6: doc 1 required it, doc 2's fixtures omit
  it; resolved in favour of the lenient parse). It sets what a turn may
  do: on `download` the agent fetches one dataset and plans nothing.
  `component_ref` scopes the turn to one card ("Ask follow
  up" / agent-driven "Edit Card"). `seed_spec` is the Search handoff
  (delta 18, see "Seeds from Search" below). `auto_add`
  (optional, default `true`; delta 11): when `false`, components the turn
  creates still stream as `component.created` but are NOT appended to the
  composition; the client adds them later with an `add` op (the Auto-add
  toggle and the "✓ Added" chip). `after_turn` (optional; delta 28) names
  the turn the request follows.
- **`interrupt.response`** resumes a paused turn: `interrupt_id` echoed
  verbatim from the pausing frame, `decision` ∈
  `approve | amend | reject | answer | skip`. `payload` carries the
  amendment text of `amend`. An elicitation's `answer` carries `answers`,
  one per question id of the frame (delta 21). `skip` (delta 12) applies
  to an elicitation whose frame sets `skippable`: it dismisses the whole
  pause and answers no question. When the orchestrator asked, the turn
  closes. When the planner asked, the skip counts as the user declining to
  choose, and the planner plans on without a pick of theirs.
- **`turn.cancel`** → `POST /api/studio/cancel` (delta 14) aborts the conversation's in-flight segment (the Stop button), whichever instance owns it: the request records a durable Stop request on the conversation's row, and the owning instance's own heartbeat observes it and ends the stream within one heartbeat interval (5 s). Idempotent; returns 204, or 200 for a Stop naming another turn than the latest (delta 32). The aborted stream, if still open, ends with a stream-fatal `error` frame whose code is `cancelled` (a studio-side addition to the error grammar); no `done` follows. A Stop that lands after the turn's pause was stored but before the stream ended ends the stream on `cancelled` alone, with no `plan.proposed` or `elicitation.request` before it. A Stop also ends a calculation the turn is running, so no later turn carries it on. Client disconnect (closing the SSE response) stops the running segment when the server observes it (the response's ASGI call ends and the stream's owner cancels the work) but leaves the turn and its calculations unfinished: the next `turn.request` carries an unfinished calculation on and lists its results, a retried `interrupt.response` with the same answer carries the turn on where it stopped, and `turn.resume` carries the turn on with no answer (delta 26). To end the work, send `cancel`. A `cancel` may name the turn it stops in `turn_index` (delta 32).
- **`turn.resume`** → `POST /api/studio/resume` (delta 26) carries on the
  conversation's open turn `turn_index`, see "Resuming a turn and reading
  the conversation" below.
- **`composition.mutate`** applies canvas operations
  (`add | reorder | remove | pin | unpin | set_layout_hints | set_card_meta`) and returns the new manifest. `add` (delta 11) appends a
  block for a component this conversation already streamed (the server
  rebuilds revision/type/title from what it sent). `set_card_meta`
  (delta 13) writes presentation-only card fields the platform intent
  cannot hold (`meta: {subtitle, description, x_label, y_label, style}`),
  owned by the harness manifest, rendered by the client (the Edit Card
  panel's manual fields). `expected_revision` is optimistic concurrency: a stale value is
  refused with an `edit_conflict` tool error and **nothing applies**;
  re-render from the current manifest and retry. Removing a block does not
  delete the component; it stays addressable.

`/turn` reads at most 6,356,992 bytes of body, `/composition` 1,048,576 bytes, and `/cancel` and `/resume` 4,096 bytes each. A longer body is refused with 413 `resource_exhausted` before it is parsed. A `remove` or `pin` operation takes about 60 bytes, so a `/composition` body holds about 17,000 of them.

## The SSE stream

Frames are SSE `data:` lines, one JSON object per frame, discriminated on
`event` (`STUDIO_FRAME_ADAPTER`). Thirteen frame types; at least one fixture
each in `wire_fixtures/studio/` (elicitation has three: the plain
question, the skippable free-text question, and two questions in one
pause; `plan.proposed` has a second showing a failed switch).

| `event`               | What it is                                                                                                                                                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `message.delta`       | append to the streaming assistant message; text is GitHub-flavored Markdown (delta 15)                                                                                                                                                          |
| `message.complete`    | final text for the turn's message. A turn its cards answer sends the text streamed beside its calls, the notes and the basis line, and none when it has none of these.                                                                          |
| `plan.proposed`       | the plan awaiting approval; **pauses and ends the stream** (delta 1)                                                                                                                                                                            |
| `plan.versioned`      | a new plan replaces the one `supersedes` names; a fresh `plan.proposed` follows. `version` can be lower than the superseded one (see below)                                                                                                     |
| `plan.approved`       | the plan the user approved; sent by the segment whose `approve` node takes the approval, and again as the first frame of an answer retry or `turn.resume` that carries on a turn which already took it; a page keys it by `revision` (delta 22) |
| `activity.step`       | one execution step, the only progress channel; `step_id` is the call's position within the stream (see `docs/frontend-interface.md`)                                                                                                            |
| `component.created`   | a new card: opaque `payload` + `fact_summary` + inline `provenance`                                                                                                                                                                             |
| `component.revised`   | new revision under the same `component_id`                                                                                                                                                                                                      |
| `composition.updated` | the authoritative canvas manifest                                                                                                                                                                                                               |
| `elicitation.request` | one to three clarifying questions; **pauses and ends the stream** (delta 1)                                                                                                                                                                     |
| `suggestions`         | follow-up chips (may be empty in v1; delta 8)                                                                                                                                                                                                   |
| `error`               | stream-fatal tool error (delta 4)                                                                                                                                                                                                               |
| `done`                | turn complete: `trace_id` + final composition or null, and `new_question` (delta 27)                                                                                                                                                            |

`component.created.payload` / `component.revised.payload` are the platform's resolved revision, passed through verbatim as the envelope `{schema_version, catalog_version, component_id, revision, provenance, component}` that dfo-spec-kit's `envelope.schema.json` defines. `payload.component` is the resolved component for its type, points included, and `payload.provenance` is the platform's provenance for that revision, authoritative for the card's back. The member is opaque to this contract, which does not restate its shape.

`fact_summary.unit` is the Presentation contract's unit slot, and this contract does not restate its shape either: a unit name string of 1 to 200 characters, or the platform's unit object forwarded with every member it states (an empty object included). The key is absent, or null, when the platform states no unit; the harness reads a null as absent. The harness refuses an empty or longer string and any other JSON type, and does not read the object's members, so a member the platform adds reaches the website unchanged.

### Interrupt resume protocol (delta 1)

`plan.proposed` and `elicitation.request` are **terminal for their HTTP
response**: the run is genuinely paused (nothing is computed; the plan's data may already be loading),
the frame is the last one on that stream, and the stream closes. To resume,
POST `interrupt.response` to `/api/studio/interrupt`; the response is a
**new SSE stream** continuing the same turn. A new stream per resume is
deliberate: human approval can take minutes, and idle SSE connections
through the website proxy and Cloud Run are fragile.

Amendment flow: `decision: "amend"` yields `plan.versioned` followed by a
fresh `plan.proposed` (a new pause, new `interrupt_id`) on the new stream.
The new `version` is higher than the one it supersedes. An `amend` whose
text asks a different question ends the turn instead (delta 27, below).

A plan the harness cannot read when the client approves or amends it is
planned again from the question. The new stream sends a `message.delta`
and `message.complete` saying so, then `plan.versioned` with
`version: 1`, a new `plan_version_id` and `supersedes` naming the card
it replaces, then a fresh `plan.proposed`. That `version` can be lower
than the one it supersedes, so a client replaces its card from the
`plan.proposed` that follows. When no plan can be drafted, the turn
closes with `message.complete` and `done`.

An `/interrupt` answer stays retryable after a crash: the answer is
recorded durably before the resumed stream starts, so a client that gets
no response (the instance resuming it died before finishing) can retry
the identical `interrupt.response` body. Retrying while that resumed
segment is still live answers 409 `transient_unavailable` with
`retry_after_seconds` set to the lease's remaining time; retrying after
that segment's lease has expired resumes the turn again, on a fresh
stream whose `seq` starts in a block of numbers past every one the dead
stream could have sent, so it never repeats a `seq` the dead stream sent.
When the turn had already paused again on a later question before the
resumed stream was lost, the retry's stream sends that question
(`plan.proposed` or `elicitation.request`, with the later pause's own
`interrupt_id`) and runs nothing again, and the client answers it as it
answers any pause.
Retrying with a DIFFERENT decision, payload or answers than the one that actually resumed answers 409
`edit_conflict`. Once the turn has paused again on a later question, a
different answer finds no pending interrupt and answers 404 `not_found`. The approval record for a plan interrupt is appended
once per `interrupt_id`, on its first answer, however many times the
resume itself retries.

### `seq` (delta 2)

`seq` increases **per turn**, across resumed segments. Each segment of a
turn reserves a block of 1,048,576 numbers (2^20) when it is claimed (a
new turn numbers from 0), and a resumed segment, whether it follows a
pause or a dead stream, starts in the block after every one reserved
before. The first frame after a resume therefore carries a number past
the paused segment's last, with a gap between them. Gaps are legal, and
`seq` always orders frames. Use it for ordering only.

### Heartbeats (delta 3)

During long tool calls (data mints run up to ~125 s) the stream carries SSE
comment lines (`: ka`) roughly every 5 seconds. Standard SSE clients
ignore comments; proxies see traffic. Heartbeats carry no `seq`.

### Error fatality (delta 4)

Two failure channels, by scope:

- A **step-scoped** failure arrives as `activity.step` with
  `status: "failed"`, and its `detail` is one sentence for the user, chosen by the failure's code (the tool error stays in the trace). The run
  continues where the plan allows, and `done` still arrives. A failed step
  can be followed by a `complete` frame on the same `step_id` when a later
  try succeeds.
- A **top-level `error` frame is stream-fatal**: `done` does not follow it.
  Branch on `error.retry`: `correct_and_retry` means a fixed call can work
  and `fix` says how; `refresh_then_retry` means your state is stale;
  `retry_after_delay` says when; `do_not_retry` means don't offer one.

The error grammar (`ToolError` × four retry modes) is a closed WIRE
vocabulary: the platform refusal codes the studio can surface, plus the
studio-side `cancelled` and `capability_unavailable`. The platform's own
grammar is a superset that keeps growing; codes the wire enum does not
know are surfaced as `transient_unavailable` with the real reason in
`detail`, and a step-scoped refusal shows as a failed `activity.step`.

`/interrupt` and `/composition` on a `conversation_id` the store does not
hold answer 404 `not_found`, `do_not_retry`, `pointer: "/conversation_id"`.
So does `/turn` when its body carries `component_ref` (there is no card to
edit or follow up on). A plain `/turn` on an unknown `conversation_id`
carries no such signal on the wire, and the server starts a new
conversation. `/cancel` on an unknown or idle conversation is a no-op
204, and a second `/turn` while one is already claimed on that
conversation answers 409 `transient_unavailable` with
`retry_after_seconds` set to the live segment's remaining lease.

### `interrupt_id` is opaque (delta 7)

Wire interrupt ids are opaque per-interrupt strings. Echo them verbatim.
The fixtures' literal `"approve"` is a sample value; the server mints a
fresh id per pause.

## Card edits, follow-ups, and canvas questions (deltas 11-13, v1.1)

Three turn shapes beyond the plain data question:

- **Edit a card**: `turn.request` with `component_ref: {component_id, revision, intent: "edit"}`. No search, no plan. The stream is:
  `activity.step` (read the card) → `activity.step` (apply/rebuild) → then
  EITHER `component.revised` (title/selection edits: same component_id,
  revision bumps, `changed` lists the patched fields) OR `component.created`
  (type/window edits: the platform cannot revise those, so a NEW component
  takes the replaced block's slot; the replaced component stays readable at
  its prior revisions) → `composition.updated` (only if the card was on the
  canvas) → `message.complete` → `suggestions` → `done`. A `component.created`
  that took a block's slot names the card it replaced in `replaces`; it is
  sent only after the canvas write that swaps it in commits. An impossible
  edit (e.g. selecting data the card's frame doesn't hold) closes with an
  honest `message.complete` and no component frame.
- **Card follow-up**: `component_ref.intent: "follow_up"` scopes the turn:
  the referenced card's digest slice travels with the question.
- **Canvas questions**: "what's on my canvas?" is answered from the
  conversation's canvas digest without any data search (message frames,
  `suggestions`, `done`). An empty canvas answers "nothing yet".

Cross-turn context: every model call of the turn's agent opens with
two digests, the canvas digest (block titles, types, fact
summaries, never rows; bounded to the newest 16 blocks), which is what
lets "now do the same for Uganda" resolve, and the conversation digest
(the newest 10 turn records: user message, assistant close, resulting
cards; byte-capped), which is what lets "the country I mentioned" resolve
when the earlier turn made no card. Context past either bound is served
on demand by the agent's `recall` tool (three calls per turn, a
mechanical store lookup: turn records, readings and fact summaries, never
rows).
Nothing about this appears on the wire: no new frames, no new request
fields.

## Composition state (delta 9)

The harness owns a minimal composition manifest in conversation state:
blocks (component_id, revision, type, title, pinned, optional `meta`
(delta 13)), server-computed `counts`, `revision`, and `layout_hints`. The front end holds no layout
state of its own; re-render from every `composition.updated` frame and
from `/api/studio/composition` responses. `expected_revision` mismatches
refuse with `edit_conflict`. `/composition` can land while a turn streams
on the same conversation, on the same or another instance: each change
(an edit from `/composition`, a card the turn adds, swaps, revises or
removes) applies under the conversation's row lock to the canvas as
stored, so both writers' edits survive.
The harness tells the model which cards are on the canvas and which are
pinned, and no card's place on it: the blocks keep the order cards joined
the canvas and any `reorder` ops, which is not the order Studio's board
shows.

## Plan detail (delta 17, v1.2)

`plan.proposed` gains three optional fields, and `PlanStep` gains one.
Every addition is additive: `steps` and `version` keep the shape they had
in v1, so a client built against the earlier contract keeps working
unchanged.

- `PlanStep.args`: the tool arguments backing the step, for a client that
  wants to show more than the label.
- `PlanProposed.revision`: a string identifying the plan's approval key,
  distinct from the integer `version`.
- `PlanProposed.requirements`: the evidence requirements behind the plan,
  each `{id, kind, sentence, decision_ids}`.
- `PlanProposed.operations`: the plan's computed steps, each `{id, kind, sentence, requirement_id?}`. An operation that carries out no single
  requirement omits `requirement_id`.

`sentence` on both `PlanRequirement` and `PlanOperation` is composed
server-side (`harness/plan_display.py`), never assembled by the client, so
the plan card and the model's own narration describe the same plan in the
same words.

## Seeds from Search (delta 18, v1.3)

`TurnRequest.seed_spec` is a list of 1 to `MAX_SEEDS` (6) `Seed`s, each
`{spec, operations}`, or `null`. The member was first reserved as one
`{spec_id, content_version}` reference, which the server parsed and did
not act on. That shape still parses and still seeds nothing, since a
reference carries no Spec block to start from: the turn runs from the
question, and the `seed` node reports the reference in a failed step,
`seed_reference_unused`. The change adds a shape and removes none, so
`schema_version` stays 1.

- `spec` is a definite part's Search API Spec block (`SeedSpec`), capped
  at `MAX_SEED_SPEC_BYTES` as posted. The harness keeps its `spec_id`,
  its `definition`, and `resolved.entity_groups`, the sets a role's
  places were asked for as, which it posts again beside the identity when
  it resolves the seed. It keeps a set only on a role the identity binds
  and with a name of 1 to 120 characters, drops any other set, and drops
  every other member as it reads the seed. A client may leave `resolved`
  out.
- `operations` are the question's operations on it (`SeedOperation`):
  `{verb, phrase, over, target_year}`, with no other member, at most
  `MAX_SEED_OPERATIONS` (8) of them. `target_year` is the year a forecast
  projects to, 1000 to 2200, set on a `forecast` alone.

A seed is browser input, so every member the harness keeps is bounded
(`harness.wire`'s `MAX_SEED_*`, each stating the Search API limit it
comes from). The harness reads nothing else of it. Every label,
sentence, coverage count, headline, slot and summability flag a seeded
turn uses comes from the Search API's own answer for the seed's
identity, which is held to the same bounds (`SpecBlock`). A seed or an
answer past a bound is refused whole.

The wire holds the items unparsed, and the route refuses a body past
`MAX_TURN_REQUEST_BYTES` before parsing it. A seeded request cannot
carry an `edit` `component_ref`. The workflow's `seed` node, the first of
every turn, reads each item once as a `Seed` (`harness.seeds.take_seeds`),
so one the harness cannot use fails its own `seed-<n>` step
(`seed_invalid`, `identity_mismatch`, or `seed_duplicate` for a `spec_id`
an earlier usable seed names) and the others go on.

The node then resolves every usable seed's identity again, all at once:
one `resolve_definition` call per seed to the data-mcp at
`DATA_MCP_URL`, which posts the identity alone to the Search API's
`dataset_resubmit` door and stores the definition Search answers with,
so the `get_data` that retrieves the seed reads it wherever data-mcp
runs. The turn waits for the slowest answer. A seed data-mcp or Search
could not be reached for fails its step (`service_unavailable`); when
that leaves no seed, the `seeds` step fails too. A seed whose identity
Search refuses, no longer resolves to a definite part, or resolves to
another `spec_id`, or whose answer the harness cannot read, is dropped:
its step fails as `seed_dropped`, with the reason and a request to
search for it again. The `seeds` step is recorded on every turn that
posts a list of seeds, counting the seeds kept, none included.

The seeds kept become definite parts of one search, read from Search's
answer, so the turn's agent holds them as it holds any search result,
and the planner reads them as search readings in this turn and later ones.
The agent's first message of
the turn lists them with the operations the user asked of each, and
names each dropped seed as `dataset <n> from Search` with Search's
reason, for the agent to tell the user. The agent fetches or plans with a
kept seed's `spec_id` with no `search` call. With no seed kept, the turn
goes on from the question alone. The node retrieves nothing.

Fixture: `request-11-turn-seeded.json`.

## A question that names no place (delta 19, v1.4)

A dataset that names no place covers every place with data, at every
level: the world, regions, World Bank groups, countries and their
subdivisions. apis falls back to the countries when the full list is
over 5,000 places or unavailable, and says so in a `coverage_below_scope`
warning. A plan that ranks, totals, averages or shares such a dataset
keeps one level first, so "top 10 by GDP" never ranks World beside the
United States, and a total never sums a country with its own provinces.

The card shows the level as a step of its own, with no new wire shape:

- The retrieval's `PlanStep.label` and requirement `sentence` read "every
  place with data". Its `args` carry `place_types: true` when a level
  filter reads it.
- The filter step's label reads "Keep countries" (or "Keep continents",
  "Keep first-level subdivisions", and so on), and its sentence "Keeps
  rows where the place is a country". Its `args` carry one
  `place_level` predicate on `observationAbout_type`.
- A rank over it names the level: "Rank the top 10 countries by GDP".
- The decision that chose the level is a row of `decisions` like any
  other, resolved by the planner when the question names no level and
  reads as a comparison of countries, and by the user when the planner
  asks.

A place sits at each level its Data Commons types name, except that a
place carrying a Eurostat NUTS type is not a country: NUTS divides an EU
or EFTA state, so the place's figures sit inside that state's. Åland
(`nuts/FI2`), which Data Commons types both `Country` and
`EurostatNUTS1`, is kept by "Keep first-level subdivisions" and left out
of "Keep countries". A territory typed `Country` and
`AdministrativeArea1` with no NUTS type, such as Hong Kong, Guam or
Greenland, is a country, since its sovereign state's figures leave it
out.

A dataset holds every place exactly when its source leaves the place
axis, `observationAbout`, open and not known to hold a single place: a
counted axis with more than one member, or an axis the source did not
count. The compiler
refuses a requirement over such a source unless the requirement states
the place party in `across`, so the card's "every place with data", the
level step and the `place_types` argument all follow from `across`.

A question that names its places ("the states of India", "the world")
retrieves those places and shows no level step. `schema_version` stays 1.

## Requested outputs on the plan card (delta 20, v1.5)

`plan.proposed` gains an optional `asks`: each output the user asked for,
in the order the agent listed them when it called the
planner, as `{ask, answered_by, declined?, not_computed?}`. `answered_by`
holds the `PlanStep` ids whose results answer the ask; `declined` says in
words why the plan does not compute it. A declined ask has an empty
`answered_by`, unless the plan answers it in part: such an ask carries
both, and `not_computed` names the part the plan does not compute, in the
user's words. `not_computed` is optional and appears only on an ask
answered in part. An ask the user removed by amending the plan is
declined with the reason "you removed it from the plan", on every later
version of the plan until the user's own amendment asks for it again.
Every ask is answered, declined, or answered in part exactly once. A
plan given no asks omits the field. `schema_version` stays 1.

## Several questions in one pause (delta 21, v1.6)

`elicitation.request` carries `questions`, one to three, in place of the
top-level `question`, `input`, `options`, `allow_free_text` and
`rationale`. Each question is `{id, question, input, options?, allow_free_text, skippable, rationale?}`. Question ids (`q1`, `q2`, …)
are minted by the harness; option ids are scoped to their question, so
two questions may both offer `o1`. The frame keeps a top-level
`skippable`: the whole pause may be dismissed with `decision: "skip"`.

| Ask                                | Frame `skippable` | Question `skippable` | Question `allow_free_text`  |
| ---------------------------------- | ----------------- | -------------------- | --------------------------- |
| The agent's ask, one question      | `true`            | `false`              | `true`                      |
| The agent's ask, several questions | `false`           | `true`               | `true`                      |
| The planner's ask (always one)     | `true`            | `false`              | `true` unless fixed choices |

The planner's recommended option keeps `recommended: true`, and its
`rationale` sits on the question.

The reply is one `interrupt.response` with `decision: "answer"` and
`answers`, keyed by question id, naming every question of the frame
exactly once:

```json
{
  "q1": { "kind": "option", "option_id": "spec_4d5e6f7a33333333" },
  "q2": { "kind": "skipped" }
}
```

An answer is `{"kind": "option", "option_id"}`, one of that question's
offered ids; `{"kind": "text", "text"}`, non-empty; or
`{"kind": "skipped"}`, only on a question marked `skippable`, which
leaves that choice to the agent and still resumes the turn. The route
answers 422 `binding_invalid` for a missing or extra question id, an
option id the question did not offer, a `skipped` answer on a question
that is not skippable, `payload` on an elicitation answer, `answers` on
a plan interrupt, an `answer` with neither `answers` nor `payload`
("decision 'answer' requires answers, one per question id"), a `skip`
that carries `payload` or `answers`, and `skip` on a frame that is not
skippable.

An elicitation opened before this delta and still paused takes its one
answer as `payload` (an offered option id, or free text). Any other
elicitation refuses `payload`. `schema_version` stays 1.

`/interrupt` reads at most 641,536 bytes of body and refuses more with 413 `resource_exhausted` before parsing. `payload` and a `text` answer take at most 16,000 characters each. A longer one gets a 422 `string_too_long` on the field. `/resume` carries a stored answer whole, whatever its length.

## Plan summary, answer step, failed switch and approval (delta 22, v1.7)

Every addition is optional, so a client built against an earlier contract
ignores it and an older harness omits it.

- `PlanStep.gives_answer`: true on the one step whose result answers the
  user's question; at most one step of a plan carries it. `label` stays
  the short step label and `description` the detail shown when the step is
  opened.
- `PlanProposed.summary`: one plain sentence saying what is counted, for
  whom, when and from which source.
- `PlanProposed.switch_failed`: `{choice_id, option_id, code, reason}`,
  set on the plan re-offered after a switch that did not apply.
  `choice_id` and `option_id` name the option the user picked, `code` is
  the refusal's code, and `reason` says in a reader's words why it failed.
- `plan.approved`: `{plan_version_id, version, revision?, title, summary?, steps[], choices[]}`. Each step is `{label, gives_answer?}`;
  each choice is `{question, chosen, changed_from?}`, where `question` is
  the choice's short name (for example "Basis"), `chosen` the option label
  and `changed_from` the label it held before the user switched it. The
  harness sends it once, on the `approve` answer to
  `/api/studio/interrupt`, before the first frame of the resumed run. The
  client attaches it to the message that stream produces and keeps it in
  the chat snapshot.

`schema_version` stays 1.

## Stable choice ids, a stopped switch and caveats (delta 23, v1.8)

- `PlanDecision.choice_id` is `"<requirement id>:<axis>"`: the first
  requirement pinning a source the row stands for, and the axis the
  choice settles. A switch re-pins requirements and keeps their ids, so a
  row keeps its `choice_id` on every plan of the conversation, and a switch
  naming it resolves on any of them. `moved[].choice_id` and
  `switch_failed.choice_id` use the same ids. A row whose sources no
  requirement pins keeps Search's own id.
- `PlanSwitchFailed.retryable`: whether the same switch can succeed if
  asked again. The harness sets it from the failure: false when it will
  repeat, such as an option Search cannot serve; true when Search or the
  calculation service could not answer in time, and for a stopped switch.
- A `/cancel` while a switch runs stops the switch. The
  stream ends on `plan.proposed` re-offering the unchanged plan under a
  fresh `interrupt_id`, with `switch_failed.code` `switch_cancelled`,
  `retryable` true and the `reason` "You stopped this change, so the plan
  is as it was.", and the turn stays paused on it. The user asked for
  this outcome, so the client shows that reason as a quiet note on the
  row, with no failure lead and no error styling. A second `/cancel`
  before then, or 10 seconds with no re-offer, ends the stream with the
  `cancelled` error frame. A switch
  that fails before it can re-offer the plan ends the turn as any failed
  stream does.
- `plan.proposed.caveats` and `plan.approved.caveats`: the plan's
  decisions that qualify the answer, such as a share whose two sides count
  different providers or a total derived from its part's dataset, each one
  reader sentence; empty when there are none. The card carries them, so
  the user reads them before Approve, and `plan.approved` repeats the
  approved card's sentences.
- `plan.approved.choices[].changed_from` is the label the choice held on
  the card the turn's first switch of it was made from, set when the
  choice holds another label, whether the user switched it or a
  switch moved it, across several switches.

`schema_version` stays 1.

## Approved choices carry their id (delta 24, v1.9)

- `plan.approved.choices[].choice_id` is the id the plan card's
  `PlanDecision.choice_id` carries for the same row:
  `"<requirement id>:<axis>"` for a scope choice and `"decision/<slot>"`
  for a planner decision. A client tells the two apart by the
  `decision/` prefix. The field is present from v1.9. A `plan.approved`
  a client stored earlier, such as one in a chat snapshot, has no
  `choice_id` on its choices, and the client reads it without one.

`schema_version` stays 1.

## A new plan in the same turn (delta 25, v1.10)

- `plan.proposed.new_plan` is `true` on a plan card that starts a new plan
  after an earlier plan in the same turn, and absent otherwise. A first draft made after an
  earlier plan card in the turn starts a new plan: its card has a `plan_version_id` of its own,
  no `plan.versioned` precedes it, and its `version` counts from 1. A client treats it as a new
  plan and drops the choice labels it kept from the earlier plan. A client
  that ignores the member reads the card as before. The member is optional
  and absent from a `plan.proposed` stored earlier.
- A plan drafted over a plan the harness could not read restarts too: its
  `version` counts from 1 and its `plan_version_id` is new. Its first card
  carries no `new_plan`, because it stands for the same question. The card it
  replaces is superseded, as for any revision, and a client keeps the choice
  labels it holds.

`schema_version` stays 1.

## Resuming a turn and reading the conversation (delta 26, v1.11)

A turn makes progress only while a client holds its stream. When the client leaves, the server stops the running segment and clears the lease, and the turn stays open at the node it was running. A page that comes back reads the conversation's state and carries the turn on.

### The turn index on every stream

Every stream, from `/turn`, `/interrupt` and `/resume`, names the index of the turn it carries in the `x-dfo-turn-index` response header, a decimal integer from 1. It is the `turn.index` a state read gives that turn and the `turn_index` of its entry in `closed_since`, so a page keys each prompt and reply on the server's index and counts nothing itself. A refused request carries no header. The header leaves every frame as it was, so a client that ignores it reads the stream as before.

### `GET /api/studio/state`

`GET /api/studio/state?conversation_id=<id>&since=<n>` answers `ConversationRead` as a JSON body. It takes no lock and no lease, so it answers while a turn streams on any instance, and it changes nothing. An unknown `conversation_id` answers 404 `not_found`. Members with no value are left out, as frames leave them out.

| Member         | What it holds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `turn`         | the latest turn: `index`, `status` (`running`, `paused`, `done`, `failed`, `cancelled`, `abandoned`), `trace_id`, `user_message` (as `closed_since` words it, so a page can show the prompt of a turn it never held), `approved`, the `plan.approved` frame of each plan the open turn approved, one per `revision`, absent when it approved none, `card_change` on a turn that asks a change of a card in place (delta 35), `lease_live`, and `retry_after_seconds`, the lease's remaining time, while `lease_live` is true; absent before the first turn                                                                                                                                                                                                                                                                                                                                                                |
| `closed_since` | the closed turns after `since`, oldest first, at most 3: `turn_index`, `user_message`, `reply` (as `message.complete` sent it, empty when the turn closed with none), `component_ids`, `cards` (each id's stored card: `component_id`, `type`, `title`, `revision`, `fact_summary`, `unit_change` and `replaces` as `component.created` gives them (delta 32), `plan_version_id` and `plan_step_id` as its `created_from` gives them (delta 34), and `card_changes` as `component.created` gives it (delta 35); no `payload` or `provenance`, which the component id reads from the presentation platform), `approved`, the `plan.approved` frame of each plan the turn approved, one per `revision`, `error`, the code a turn that did not end cleanly closed on, `new_question` on a turn ended by one (delta 27), and `card_change` on a turn that asked a change of a card in place (delta 35); empty without `since` |
| `closed_total` | how many closed turns the conversation holds, which is the index of the last one                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `pause`        | when `turn.status` is `paused`: the `interrupt_id` to answer, its `kind` (`plan` or `elicitation`), and `frames`, the frames the pause was sent as, its `plan.proposed` or `elicitation.request` last, each with the `seq` it went out with                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `manifest`     | the composition manifest; absent when the conversation has no canvas                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

`user_message` is the turn's record. It holds the message, with a `(then: ...)` line for each amendment, answer or switch on the turn's cards.

Each read returns at most 3 closed turns, since each carries its cards. A page that lacks turns reads again with `since` set to the last `turn_index` it got, until it holds `closed_total` turns.

A closed turn's `cards` are the cards at their current revision. A card a later turn revised shows that revision's `type`, `title`, `revision` and `fact_summary`. The canvas shows current cards, so a page drawing a closed turn's cards draws what the canvas holds.

### Where a turn stands

The server derives one standing for the latest turn from its stored record, its stored cursor, the lease, and the pause's stored entry. The read reports the standing as `turn.status`, and `/resume` acts on the same standing, so every status the read gives has one page action and one `/resume` answer.

| Standing                                                                               | `turn.status`       | `lease_live` | `pause` | Page action                                                                                                                               | `/resume` answers                                      |
| -------------------------------------------------------------------------------------- | ------------------- | ------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| a stream holds the turn, on any instance                                               | `running`           | `true`       | absent  | read again after `retry_after_seconds`, until the turn pauses, closes or its lease clears                                                 | 409 `transient_unavailable` with `retry_after_seconds` |
| running, and no stream holds it                                                        | `running`           | `false`      | absent  | send `turn.resume`                                                                                                                        | 200, a new stream running the turn on                  |
| paused on an answer `/interrupt` recorded and the turn has not yet taken               | `running`           | `false`      | absent  | send `turn.resume`                                                                                                                        | 200, a new stream carrying that answer on              |
| paused on a question whose entry the server never stored, or stored without its frames | `running`           | `false`      | absent  | send `turn.resume`                                                                                                                        | 200, a new stream sending the pause again              |
| paused on a question that awaits its answer                                            | `paused`            | `false`      | present | show `pause` and answer it with `interrupt.response`                                                                                      | 409 `edit_conflict`                                    |
| closed                                                                                 | the record's status | `false`      | absent  | append the turns in `closed_since` the page lacks; send a closed turn's `new_question` as the next turn when the page holds none after it | 409 `edit_conflict`                                    |

A live lease holds the turn whatever its record and cursor say, so a turn whose segment is still writing its pause or its close reads `running` until the write lands. A closed turn's status is the one its record states, since the record is what the stream told the client: `done` with no `error`, `cancelled` or `abandoned` for those codes, and `failed` for any other `error`, `state_not_saved` included. A closed status always comes with the turn's record, which a read with a lower `since` returns in `closed_since`.

The server writes a turn's record and closes its cursor in one transaction. The closing write that commits a record closes the cursor with the status the record states. A turn recorded with its cursor open reads closed by the record, and the next request that loads the conversation closes the cursor.

One failure stays ambiguous. A turn whose work failed ends its stream on the work's `error` frame. When the closing write that stores its record fails too, the store holds the turn running with no lease, which is what an instance that died mid-turn leaves. The read then reports `running` with `lease_live: false`, and `turn.resume` runs the turn on, although the page already showed its error. The store cannot tell the two apart, so the page follows the read. It resumes the turn and shows the stream the resume opens.

### `POST /api/studio/resume`

`POST /api/studio/resume` takes `{kind: "turn.resume", schema_version: 1, conversation_id, turn_index}` and answers a NEW SSE stream continuing the turn. The segment runs the node the turn stopped in again from its start, so the stream carries no `activity.step` lines for nodes before it. A `plan.approved` the turn has not yet sent goes out once, as on an `interrupt.response` retry. The stream opens with the `plan.approved` of each plan the turn already approved, under fresh `seq`s, since the client that left may have read none of them. A page keys them by `revision` and ignores one it holds. Frame numbers continue past every one the turn's earlier segments reserved (delta 2). The stream ends as any segment does: `done`, a pause, or an `error` frame.

The route reads the standing before it claims the conversation and again after, since a segment whose lease lapsed can move the turn until the claim commits. Besides the answers in the table above:

| Case                                               | Answer                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| `turn_index` is not the conversation's latest turn | 409 `edit_conflict`, `refresh_then_retry`, `pointer: "/turn_index"`     |
| a Stop is live for the turn                        | the turn closes `cancelled`, and the resume answers 409 `edit_conflict` |
| unknown `conversation_id`                          | 404 `not_found`                                                         |

Every 409 `edit_conflict` from `/resume` carries `refresh_then_retry` and `pointer: "/turn_index"`. Read the state again and follow what it shows. A resume whose client also leaves leaves the turn open in the same way, for the next resume.

## A new question on the plan card (delta 27, v1.12)

Users type new questions into the plan card's revise box. The client sends them as `amend`, and the planner decides whether the text changes the plan or asks something else (`propose_plan`'s `starts_new_plan`, with or without `amends_version`). For a new question, the turn ends with its plan set aside unapproved. The new stream sends a `message.complete` saying the plan was set aside for a new question, which a client without this delta still shows, then `done` with `new_question` and no `plan.versioned` or new card. `new_question` is the user's latest words as sent. That is the `amend` text or, when the planner asked a question about the amendment and the answer was the new question, the answer's text. The turn closes `done`, its pending interrupt is cleared, so a later answer to it gets 404, and the conversation's next turn sees that the plan was set aside for a new question.

When the user's words ask for a new question without stating it, as "I want to ask something else" does, the planner names no message (`new_question_in` `none`). The turn closes `done` the same way, with the plan set aside, but `message.complete` reads "The plan was set aside. What would you like to ask?" and `done` carries no `new_question`. The client closes the card and waits for the user's next message.

The client closes the card ("Taken as a new question") and sends `new_question` as the next `turn.request`, which is metered, opens its own trace, and plans or answers the question like any other. A state read's closed turn carries the same `new_question`, for a page that lost the stream before `done`. A client that ignores the field shows the message and loses the question. The delta adds one optional member, `done.new_question`, and keeps the existing frames, request kinds and turn statuses.

## A turn that names the turn it follows (delta 28, v1.13)

A page that sends a closed turn's `new_question` as the next turn sends it automatically, and two pages open on one conversation can both send it. `turn.request` gains an optional `after_turn`, the index of the turn the request follows, as a state read's `turn.index` or the stream's `x-dfo-turn-index` header gives it (0 before the first turn). The server compares it with the conversation's latest turn index under the row lock every turn's claim takes, in the claim's transaction, so of two requests naming the same turn at most one begins a turn. The other is answered 409:

```json
{
  "error": "edit_conflict",
  "retry": "refresh_then_retry",
  "constraint_id": "turn_moved",
  "pointer": "/after_turn",
  "detail": "the request follows turn 3, and the conversation's latest turn is 4",
  "fix": "read the conversation's state and show the turns it holds"
}
```

`constraint_id: "turn_moved"` tells this refusal apart from a busy conversation, which answers 409 `transient_unavailable` with `retry_after_seconds`. A busy refusal comes first while another request's stream holds the conversation; once that stream ends, the retry answers `turn_moved`. On `turn_moved` the page reads the state, shows the turns it lacks, and does not send the question again. The check begins no turn, meters nothing and writes nothing. A request without `after_turn` is admitted as before.

## Count chips and count rows on the plan card (delta 29, v1.14)

- Answers offer count chips, `apply_choice` chips with `choice_id`
  `"<spec_id>:count"` (`docs/frontend-interface.md` 4.1.2).
- The row stating how a pinned source's flow is counted ("How the flow is
  counted.", with options such as "gross disbursements" and "net
  disbursements") carries `choice_id` `"<requirement id>:count"`,
  `active_option_id` and `options`, as any switchable row does. Sources
  counted the same way, such as a share's numerator and denominator, share
  one row, which offers only the counts every one of them offers. The
  requirement is the first pinning a source of the row, so the id survives
  every switch of the count. Each option's `option_id` is derived from its
  label, as a planner decision's is (`"option_"` and the first 12 hex
  digits of the label's SHA-256), and its `status` is `available`, since
  Search offered each count beside every source of the row.
- A client switches it as it switches a scope row: `interrupt.response`
  with `decision: "switch"` and `{choice_id, option_id}`. The harness
  re-pins every requirement on every source of the row to that source's
  other count, a dataset Search already named, in one revision, and offers
  the revised plan with no `moved` entries; the row then holds the chosen
  count as the user's choice. A
  switch to the count the row already holds is refused as any such switch
  is, and the plan is re-offered with `switch_failed`.
- `plan.approved.choices[]` carries the row under the same `choice_id`.
  A client tells it from a scope row by the `:count` ending, where it
  wants to word it apart.

`schema_version` stays 1.

## A unit question, a card's unit change and several answers (delta 30, v1.15)

A unit is a currency, a price basis, and for constant prices a base year. Three shapes carry it.

```
UnitCurrency  {code, name, prices[], base_year_min?, base_year_max?}
UnitChoice    {currency, prices, base_year?}
UnitOffer     {currencies[], presets[], start}
UnitChange    {spec_id, offer}
ApplyUnit     {component_id, spec_id, turn, unit}
```

- `prices` is `"current"` or `"constant"`. `UnitCurrency.prices` lists the bases the currency serves (at least one); `name` is the noun to show ("US dollars"). `base_year_min` and `base_year_max` are both set or both absent, and only when `constant` is among `prices`.
- `UnitChoice.base_year` is set when `prices` is `"constant"` and absent otherwise.
- `UnitOffer.currencies` holds every offered currency (at least one). `presets` lists the codes shown before "Change to any currency". `start` is the unit the chips open on.

### The unit question

`elicitation.request.questions[].input` gains `"unit"`. Such a question carries `unit`, a `UnitOffer`, and no `options`; `allow_free_text` is `false`. `unit` is absent on every other input. It may be `skippable`.

```json
{
  "id": "q1",
  "question": "Which unit should Kenya's debt service be shown in?",
  "input": "unit",
  "skippable": true,
  "unit": {
    "currencies": [
      {
        "code": "USD",
        "name": "US dollars",
        "prices": ["current", "constant"],
        "base_year_min": 1970,
        "base_year_max": 2024
      },
      { "code": "KES", "name": "Kenyan shillings", "prices": ["current"] }
    ],
    "presets": ["USD", "KES"],
    "start": { "currency": "USD", "prices": "constant", "base_year": 2023 }
  }
}
```

The answer is `{"kind": "unit", "currency", "prices", "base_year?"}`, with the same base-year rule as `UnitChoice`, or `{"kind": "skipped"}`. A unit outside the offer is refused when the answer is read, not by the wire models. A skip, of the question or of a lone unit question's whole pause, goes on with the unit the offer starts on (`offer.start`), as the agent reads it: the reading is fetched in that unit.

### Several answers to one question

A `multi_select` question has `options` set and `allow_free_text` is `true`. Its answer is `{"kind": "options", "option_ids": [...]}` with at least one id and no repeats; `{"kind": "text"}` and `{"kind": "skipped"}` stay valid. A single `option` answer to a `multi_select` question, and an `options` answer to any other, is refused with 422. Each id must be one the question offered. The agent asks one when several answers can hold together, and offers labels only, so the answer elects no reading. It reads "The user chose: Health; Water", and the turn's record keeps the same labels.

### Question forms a client declares

`turn.request` and `interrupt.response` gain `client_features`, an optional list of at most 16 strings: `"unit_question"` when the client renders a question whose `input` is `unit`, and `"multi_select"` when it renders one whose `input` is `multi_select`. The harness ignores a value it does not know, and refuses a longer list with 422. The conversation stores the list the latest request carrying the field declared (`ConversationState.client_features`); a request without it leaves the list as it was, empty on a new conversation, and every question the turn asks, before and after a resume and on any instance, takes the forms it names (`harness.studio_translate._asked_question`).

Without `"unit_question"`, a unit question is sent as `input: "single_select"` with no `unit` and `allow_free_text: false`. Its options are both price bases of the offer's start currency, or, when that currency serves one basis, of the first offered currency that serves both, the start's basis first, each id naming its unit (`harness.unit_offer.whole_units`). A constant unit takes the start's base year when the currency's range holds it, else the currency's latest.

```json
{
  "id": "q1",
  "question": "Which unit should Kenya's debt service be shown in?",
  "input": "single_select",
  "options": [
    { "id": "USD-constant-2023", "label": "Constant 2023 US dollars" },
    { "id": "USD-current", "label": "Current US dollars" }
  ],
  "skippable": true
}
```

An `option` answer to it resumes as the unit its option names, as a `unit` answer does, and a `text` answer is refused with 422. Without `"multi_select"`, a multi-select question is sent as `input: "single_select"` with the same options and free text, and an `option` answer resumes as a one-item `options` answer. A question keeps the form it was sent in (`_PendingQuestion.input`, with its own form as `native_input`), so its answer is checked against that form whatever the answering request declares.

The harness deploys before the website: a harness without the field refuses it with 422, and a page opened before the website deployed sends none and gets the `single_select` forms on a conversation no newer page declared features for. The website sends both features on every turn and interrupt request.

### A card's unit change

`component.created` gains an optional `unit_change`, a `UnitChange`: the `spec_id` of the one reading the card draws from, which is the card's `provenance.inputs[0].spec_id`, and the offer its unit can move within. `offer.start` is the card's current unit. It is set when the card draws the frame of one `get_data` retrieval, with no `scope`, of a spec Search converts; a card on a derived frame, or on a retrieval whose rendering the harness cannot map back to `get_data`'s arguments, carries none. `component.revised` does not carry it.

A client moves the card with `turn.request` carrying `apply_unit`, an `ApplyUnit`: the `component_id`, the `spec_id` from `unit_change`, the `turn` that drew the card (from 1), and the new `unit`. Like `apply_choice`, it comes alone: a request with `apply_unit` and `message`, `seed_spec`, `component_ref` or `apply_choice` fails validation. A request carries `message`, `apply_choice` or `apply_unit`.

The route answers 422 `constraint_violation` before any stream starts when the change cannot apply. `constraint_id` is `scope_choice_stale` (pointer `/apply_unit`) when the card is no longer on the canvas or its `provenance.inputs[0].spec_id` is not `spec_id`, and `unit_not_offered` (pointer `/apply_unit/unit`) for a unit outside the card's offer or the unit it already shows. `detail` is copy for the user. `turn` is not checked beyond its range, so a card from any earlier turn, a pinned one included, can change unit, also on a reopened conversation.

Otherwise the change runs in code, with no model call, as the Edit Card route does (`harness.studio_edit.unit_events`). Search reads the card's spec in the new unit, `get_data` fetches the new spec with the arguments that shaped the card's frame, and the card is written again from its stored intent on the new frame. Its title names the new unit where it named the old, and a title the client overrides moves with the card. The stream sends `activity.step` frames, a `component.created` for the new card with a fresh `unit_change` whose `offer.start` is the new unit, a `composition.updated` whose manifest holds the new card in the old card's slot, `suggestions` and `done`. `component.created` names the old card in `replaces` and is sent only after the canvas write that swaps the new card in commits. The new block keeps the old block's `pinned` flag and `meta`, the words of `meta`'s `subtitle`, `description`, `x_label` and `y_label` that named the old unit naming the new one (`harness.card_unit.retitled`). A `row` entry of `layout_hints` that named the old card names the new one; a `comparison` entry, which aligns its members' units, drops it, and goes when fewer than two members remain. A card that leaves the canvas while its new unit is read ends the stream on a failed `activity.step` and a `message.complete` saying "This card is no longer on the canvas, so nothing changed.", with no `component.created`. The stream sends no `message.complete` when the change lands: the redrawn card is the answer. The turn's record keeps the sentence "The card now shows constant 2023 US dollars." for the next turn's model, and a state read gives the turn's `reply` as empty, as the stream sent it. A step that fails ends the stream on its failed step and a `message.complete` that begins "The card is unchanged.". The change counts as a turn. Its record reads "Changed Aid to Kenya to constant 2023 US dollars (spec\_…).", and the new reading joins the turn's readings, so the next turn's model reads the card in the new unit under its new spec_id. A state read gives the turn's `user_message` as "Change Aid to Kenya to constant 2023 US dollars.", with no spec_id.

`schema_version` stays 1.

## A unit row on the plan card (delta 31, v1.16)

After the user answers a unit question, the plan card states the unit in one row for the parts the question covered that are served in the unit chosen. That row can be switched to another unit. A part Search could not read in that unit has a row of its own stating the unit it is served in, with no `choice_id`, and a switch of the unit row leaves it as it is.

- `PlanDecision` gains an optional `unit`, a `UnitOffer`, set only on a row that also carries `choice_id`, `active_option_id` and `options`. The row's `choice_id` is `"decision/user_unit_<lead part>"`, the lead part being the first spec_id the question covered. `unit.currencies` lists every unit all the series the plan reads for those parts convert into, a count or scope switch of the card included, and `unit.start` is the unit the plan states for them now.
- The row's `options` are whole units, so a client that renders no unit picker switches between them: both price bases of the start's currency, the start first and active; or, when that currency serves one basis, the start and then both bases of the first offered currency that serves both. Each `option_id` names its unit as a unit question's whole units do (`USD-current`, `USD-constant-2023`), and each `status` is `available`.
- `interrupt.response` gains an optional `unit`, a `UnitChoice`, allowed only with `decision: "switch"`. `choice_id`, `option_id` and `plan_revision` stay required; `option_id` names the unit `unit` carries in the same form. The harness reads `unit` when present, else the unit `option_id` names among the row's options, and refuses a unit outside the row's offer. A switch re-reads in that unit every series the plan reads for the parts the row covers and offers the revised plan; one series Search cannot re-read leaves every part as it was, and the plan comes back with `switch_failed`.
- A row stating a unit without `choice_id` cannot be switched.

```json
{
  "kind": "interrupt.response",
  "decision": "switch",
  "choice_id": "decision/user_unit_spec_5c1e…",
  "option_id": "USD-constant-2015",
  "plan_revision": "rev_01j9ze2k7d",
  "unit": { "currency": "USD", "prices": "constant", "base_year": 2015 }
}
```

`schema_version` stays 1.

## A Stop that names its turn, and a stored card's unit and slot (delta 32, v1.17)

### A Stop that names its turn

`turn.cancel` gains an optional `turn_index`, the turn the Stop is for. It is the index the stream's `x-dfo-turn-index` header named, from 1 to 2147483647. A page leaves it out until a stream's header names its turn, and drops the index it held when it sends a `/turn`, which starts a new turn. A Stop sent before the new header arrives then reaches the new turn once that turn's claim has landed. One that lands earlier, on another server instance, can answer 204 without reaching the new turn, so a page that sent a Stop in that window sends it again with the new `turn_index` once the header arrives.

- When the conversation's latest turn is `turn_index`, the route acts on the Stop and answers 204. On a live lease, the Stop is recorded at once naming that turn, so a turn begun after it under the same lease runs on. When that lease was claimed by a `/turn` for the next turn, the save that begins the next turn closes the named turn `cancelled` when it has a cursor, and the next turn runs. An Edit Card or unit-change turn left open has no cursor and closes `abandoned`. A lease on that turn that goes live while the Stop is recorded, from an `/interrupt` or `/resume` of another tab, takes the Stop.
- When the latest turn is another, the route answers 200 with a `CancelSkipped` JSON body, `{"code": "turn_moved", "turn_index": <the latest turn>}`, `turn_index` being 0 before the first turn, and the conversation stays as it was. The turn the page stopped has already ended, and a later turn, begun from another tab say, runs on. A `turn_index` above the latest turn gets the same answer. A page sends only an index a header named. `/turn` answers a stale `after_turn` in another shape, a 409 `edit_conflict` whose `constraint_id` is `turn_moved`.
- Without `turn_index`, the Stop is for whatever turn the conversation runs, as before.

A release from before this delta refuses `turn_index` with a 422, as it refuses any member it does not name. A page that gets a 422 for a Stop carrying `turn_index` sends the Stop again without it.

```json
{
  "kind": "turn.cancel",
  "schema_version": 1,
  "conversation_id": "conv_01J9ZC3AB4YT9Q6WGX0S6H2K9E",
  "turn_index": 2
}
```

### A stored card's unit and slot

A state read's stored card (`closed_since[].cards[]`, `StoredCard`) gains `unit_change` and `replaces`, with the names and shapes `component.created` gives them, so a page applies one rule to a card from a stream and one from a state read.

- `unit_change`, a `UnitChange`, is the card's offer now, as `apply_unit` reads it from what the conversation keeps with the card. It is set while the card is on the canvas, and its `offer.start` is the unit the card shows. It can differ from the offer the card's `component.created` sent when a later turn kept a newer Search block for the same spec.
- `replaces` names the card whose canvas slot this card took, as for a unit change or an Edit Card rebuild. It comes on the closed turn that swapped the card in, recorded when the swap committed. A later turn that revised the card lists it with no `replaces`, and a card an earlier release stored has none.

Each is left out when it has no value. `schema_version` stays 1.

## Where a typed question came from (delta 33, v1.18)

`turn.request` gains an optional `prompt_source`, set when the page filled in the question for the user: `"example"` for a Studio landing example, `"availability"` for the button asking what data is available, and `"search_example"` for a Search example handed over to the Studio. A question the user typed carries none. The harness drops any other value, as it does an absent one, and runs the turn. The turn's `studio.turn` log line names it as `prompt_source`, and its Langfuse trace holds it under the metadata key `prompt_source`; a resume or an answer to the turn's pause keeps it.

A release from before this delta refuses `prompt_source` with a 422, as it refuses any member it does not name, so the harness deploys before a page that sends it. `schema_version` stays 1.

```json
{
  "kind": "turn.request",
  "schema_version": 1,
  "conversation_id": "conv_01J9ZC3AB4YT9Q6WGX0S6H2K9E",
  "message": "How much aid did Kenya receive in 2023?",
  "prompt_source": "example"
}
```

## Plan steps on cards, activity lines and choices (delta 34, v1.19)

Each frame that concerns a plan step names it by the step's own id, `PlanStep.id`, so a page lists a plan's cards under it and joins its steps, Activity lines and choices without matching labels.

- `component.created.created_from` gains `plan_version_id` and `plan_step_id` on a card drawn from a plan step's result: the plan's `plan_version_id`, as `plan.proposed` and `plan.approved` carry it, and the step's `id`. The harness keeps, for the turn, the plan and step of each final frame a plan run hands back, and names them on a card the agent draws on that frame. A card drawn on any other frame, such as one the agent fetched itself, names neither. A card that takes another card's slot (`replaces`: a unit change, an Edit Card rebuild) names the plan of the card it replaced. A card drawn on the frame of a run an earlier turn left unfinished, which a later turn carries on, names no plan. `step_id` is the `activity.step` line that wrote the card.
- `activity.step` gains `plan_version_id` and `plan_step_id` on a line that runs a plan step: the `plan_version_id` of the card whose plan runs, and the step's `id`, both or neither. A line of the agent's own work, such as a status line or a chart it draws, has neither.
- A state read's stored card (`closed_since[].cards[]`, `StoredCard`) gains `plan_version_id` and `plan_step_id`, as the card's `created_from` named them, so a page that missed the stream links the card to the turn's `approved` plan.
- `plan.approved.steps[]` gains `id`, the step's `PlanStep.id` on the approved card, so an approved step and its Activity lines share one id.
- `plan.proposed.decisions[]` and `plan.approved.choices[]` gain `step_ids`, the ids of the plan steps the decision applies to: the steps carrying out a requirement that cites it and every step reading their results. A decision no requirement cites applies to the whole plan and names every step.

A replan or a second plan in the turn can reuse step ids, so a page keys a plan step by the plan's `plan_version_id` and the step id together. A `plan.approved` a client stored earlier, such as one in a chat snapshot, has no `id` on its steps and no `step_ids` on its choices, and the client reads it without them. Each member is left out when it has no value. `schema_version` stays 1.

## A card's years and places, and what a plan decision settles (delta 35, v1.20)

A card drawn from one retrieval can change its years and its places with no model call, through the flow a unit change runs (delta 30), and the changes compose: each change starts from the card's arguments, years and places as the last change left them.

- `component.created` and a state read's stored card gain `card_changes`, `{spec_id, window?, places?}`, set while the card is on the canvas and offers at least one change. `spec_id` is the spec a change names. `window` is `{start_year, end_year, first_year, last_year, missing_years?, single?}`: the years the card draws, and the first and last years the spec holds over the places the card shows, with the years inside them that hold no figures (Search's `years_with_data`, else the window its answer serves, with none missing). The spec's Search block states the years of its own places. A card whose places a change moved off them offers the years one Search read states for the new places (a `resolve_definition` of the block's identity with those places, made once per change that needs it), and no `window` when that read states none or fails; a card the model drew over other places than its block's offers no `window` until a change of places reads them. `start_year` equals `end_year` on a card that shows one year, including one asked for its latest year. `single: true` marks a card that shows one value per series, a key number or a bar chart of one year: `start_year` and `end_year` both name the one year it draws, read from its drawn payload (a key number's `period`, a bar chart's `category_axis.period`), so the picker starts on that year. Its years change one at a time, and a change names one year as both `start_year` and `end_year`. Such a card whose drawn year cannot be read offers no `window`. Other cards leave it out. It is set when the spec holds more than one year. `places` lists one entry per role whose places the card binds, `{role, multiple, chosen[]}`, each place `{id, label}`: the places the card draws, the ones its chart's `select` names for the role where it names any of the frame's, else every place the frame holds, and whether the role takes more than one. Each place carries a name, and a role showing a place with no name (one the card's retrieval resolved without naming) is left out. `multiple` is fixed by the card's encoding, whatever number of places it shows: a role takes several where each place draws its own series, on a table, on a line chart whose `select` lists no other dimension, and on a bar chart whose listed dimension is the role's (one bar per place). It takes one on a key number, on a line or bar chart listing another dimension, on a line or bar chart where another place role already draws several series, and on a bar chart of one series over time, which a second place would turn into one bar per place at one period. It is set only where the deployment reads Search's places door, that is, where `STUDIO_PLACE_OPTIONS` is on (the default) and apis serves the `options` door. When the door answers 404 or outside its contract and Search's capabilities document lists no live `options` door, cards offer no places for 60 s, and `/places` answers `capability_unavailable` during that time. A card cut to its largest members at one period ("10 largest of 70 in 2023"), as `make_charts` recorded it, has no `window` and no `places`, since another year or other places have other largest members. A card drawn on a derived frame, such as a share or a ranking, has no `card_changes`.
- `GET /api/studio/places?conversation_id=…&component_id=…&role=…` answers the places the role can change to, read from Search when asked for the identity the card reads now (its block's, with the places and window its scope names, so the other roles' places and the card's years filter the list, as they do for the check of a change of places), `{component_id, role, options[], truncated}`: at most 300 places `{id, label}` in label order (case and accents ignored, ties by `id`), `truncated` when Search listed more. It is read with no lease, as `/state` is. It refuses an unknown conversation 404 `not_found`, a card off the canvas or offering no change 422 `scope_choice_stale` (pointer `/component_id`), a role the card does not offer 422 `change_not_offered` (pointer `/role`), Search's door failing, taking more than 2 s or answering outside its contract 503 `transient_unavailable` with `constraint_id` `place_options_unavailable` and `retry_after_seconds`, and a deployment with the door off 503 `capability_unavailable`. A role's places are kept for 60 s, so a change applied from an open picker reads none again.
- `turn.request` gains `apply_change: {component_id, spec_id, change}`, which comes alone, as `apply_unit` does. `change` is `{kind: "window", start_year, end_year}`, with `end_year` equal to or after `start_year` (one year when equal), or `{kind: "places", role, entities[]}`, the place ids for that role, 1 to 50. A places change replaces that role's places and keeps the others.
- The server refuses a change it cannot apply before the stream starts, `pointer` `/apply_change`: 422 `constraint_violation` with `scope_choice_stale` when the card has left the canvas or no longer reads `spec_id`, `change_not_offered` for years outside `first_year` to `last_year`, the years the card already draws, a range on a card whose `window` is `single`, a role it does not offer, two places for a role whose `multiple` is false, the places the card already shows (in any order), or a place Search does not list for the role, and `change_no_data` for years that all fall in `missing_years`; and 503 `transient_unavailable`, `place_options_unavailable`, when Search's list for a places change cannot be read, which the server checks every place against, or 503 `capability_unavailable` when apis serves no places door, as `/places` answers, including for the 60 s the offers stay stopped after the server learnt it.
- The stream is a unit change's: `activity.step` frames, a `component.created` for the redrawn card, with `replaces`, the old card's plan, a fresh `unit_change` and `card_changes`, then `composition.updated`, `suggestions` and `done`. A retrieval that holds no figures for the new years or places ends on a failed `activity.step` and a `message.complete`, and the card keeps its slot and its offers; neither frame carries a code for it. A card-change turn that ends with reply text was refused or failed, and the text is the reason shown to the reader. After a change of years the title keeps the subject it named before its period and ends on the new period ("2010 to 2024", or one year for a one-year period). After a change of places the title names the new places where it named the old ones: the old places' words as a list ("Kenya and Uganda") replaced whole, else, for a change swapping places one for one, each place left that the title names once replaced by the one that took its place ("Health aid to Kenya, 2015 to 2020" becomes "Health aid to Uganda, 2015 to 2020"). Where neither holds, where the title already names a place the change brings, or where a place left is still named after it, the title is built from the block: its concept, else its measure, then the card's places, then the period the title named. A built title is at most 120 characters, its period always whole and last. Its places give way before its subject: they are listed as far as they fit, closing on "and N others", then counted ("2 places"), then left out. A title never names a place by its ref; a place with no name leaves the places out. A year inside the served unit's words ("constant 2023 US dollars") is no period, and stays. A change of places rewrites only the changed role's `select` entry, so another role naming the same place keeps it. Every change of years or places fetches with the facets the card was drawn with bound (`scope.facets`: `provenance`, `measurementMethod` and `unit`, the ones every drawn series' coordinates share, and each the spec's identity binds by ref), so the data API elects none again; a spec whose identity binds a facet by request, as constant prices do, binds none on the fetch. A unit change binds the source and method on the identity it asks Search for. A redraw whose series name another facet on a bound axis is refused, the card unchanged, with "The card is unchanged. The data for that choice is counted another way.", and a bound facet with no data in the new scope ends as a retrieval with no figures. The model's `subtitle` and `description` are dropped, and an axis label is kept only when it names no four-digit year outside the unit's words and no place the card showed, so the chart's own labels show there. A unit change names the new unit in the title and `meta` where they named the old, in the unit offer's own words. The turn's record names the `get_data` arguments, scope included, that draw the new card, and its canvas entry keeps them, so the next turn's model fetches what the card shows. A `comparison` layout hint keeps the card. A state read gives the turn's `user_message` as "Change Aid to Kenya to 2010 to 2024.".
- A state read's `closed_since[]` turns and its open `turn` gain `card_change`, `{component_id, kind, role?}`, on a turn that asked a change of a card in place, whether or not the change landed: the card the `apply_unit` or `apply_change` request named (the one the change replaced), `kind` `unit`, `window` or `places`, and `role` for a change of places. A page rebuilding its transcript shows such a turn as a change on that card; the turn's `reply` or `error` says how it ended. Other turns have none.
- `plan.proposed.decisions[]` and `plan.approved.choices[]` gain `kind`, what the decision settles, by the first rule that holds: `unit` for a unit question's row, `count` for the row of how pinned sources' flows are counted, the scope choice's axis (`sector`, `region`, `measure_basis`, `provider`), `unit` for a decision fixing the unit or its prices, `window` for one fixing the period or its first or last year, `places` for one fixing the places. It is left out where none holds.

Each member is left out when it has no value. `schema_version` stays 1.

```json
{
  "kind": "turn.request",
  "schema_version": 1,
  "conversation_id": "sess_01j9zc0000",
  "apply_change": {
    "component_id": "cmp_01j9zc3aqw",
    "spec_id": "spec_8f3a1c5e9d2b4a7c6e0f1a2b3c4d5e6f",
    "change": { "kind": "window", "start_year": 2010, "end_year": 2024 }
  }
}
```

## Sample fixture values (delta 10)

Fixture values that are samples:

- The fixtures' `interrupt_id: "approve"` literal is a sample value (see delta 7).
- Forward-dated catalogue versions in fixtures (e.g. `2026.10.0`) are
  sample values.
- Fixture tool names (e.g. `datasets.get`) are samples; `tool` in
  `activity.step` is a display string.
- The `component.created` fixture's `fact_summary.series[].change`
  arithmetic is a sample value.
- The `component.created` fixture's `payload` holds only `schema_version` and the component's `type` and `title`. A served `payload` is the full resolved revision.

## Known gaps (upstream, named)

- **Some searches refuse** with `capability_unavailable` ("identity does
  not reproduce spec_id"), an upstream/spec-kit identity gap on specific
  spec ids (not arity-dependent), reported to ONE. The search step fails
  with the refusal's own detail and the agent answers without that data.

## Reference implementation and fixtures

- Models (enforced): `services/harness/src/harness/wire.py`
- Server: `services/harness/src/harness/studio.py` (routes, runner, SSE),
  `studio_state.py`, `studio_translate.py`, and `studio_edit.py`, behind
  `POST /api/studio/*` on the harness app
- Static fixtures (one per frame/request):
  `services/harness/tests/wire_fixtures/studio/`
- To run a conversation yourself:
  `just studio-live` (it runs the harness on the local Postgres `just pg-up`
  starts, unless `.env` sets `DATABASE_URL`) then
  a local (untracked) `test_scripts/studio_client.py` on the dev machine;
  any SSE client following the resume protocol above reproduces them

## Delta index

1. Interrupts close the stream; `/api/studio/interrupt` answers with a new
   stream continuing the turn.
1. `seq` is monotonic per turn across segments; gaps legal.
1. SSE comment heartbeats (`: ka`) ~every 5 s during long calls.
1. `activity.step status=failed` is recoverable; a top-level `error` frame
   is stream-fatal and `done` does not follow.
1. `Authorization: Bearer <token>` + client-minted `conversation_id` on
   every request;
   `composition.mutate` gains `conversation_id`; conversation state keyed
   per principal; each conversation maps to its own platform
   `x-dfo-session` server-side.
1. `surface` optional, default `"studio"`.
1. Wire `interrupt_id`s are opaque; never hardcode fixture literals.
1. `suggestions` may be empty in v1.
1. The harness owns the minimal composition manifest;
   `expected_revision` → `edit_conflict` on mismatch.
1. Sample fixture values named (see above).
1. `turn.request.auto_add` (default `true`) decouples component creation
   from composition membership; `component.created` on an auto_add=false
   turn does not imply a composition block; `composition.mutate` gains the
   `add` op, and `composition_id` becomes optional: omitted targets the
   current composition, and an all-`add` mutate on an empty canvas
   bootstraps one.
1. Elicitations may set `allow_free_text` and `skippable`;
   `decision: "skip"` dismisses the whole pause; an answer may be free
   text when `allow_free_text`, and reaches the agent or the planner in
   the user's words. Delta 21 moves both flags onto each question and
   keeps a whole-pause `skippable` on the frame.
1. `CompositionBlock.meta` (`set_card_meta` op) carries presentation-only
   card fields (subtitle/description/axis labels/style) the platform
   intent cannot hold; harness-owned, never sent to the platform.
1. `POST /api/studio/cancel` aborts the in-flight segment, reaching one
   running on any instance within one heartbeat interval; the stream ends
   with error code `cancelled`. Client disconnect stops the stream and
   leaves the turn for the next request to carry on.
1. `message.delta` / `message.complete` text is GitHub-flavored Markdown;
   render it, don't escape it.
1. (folded into delta 11.)
1. `plan.proposed` gains `revision`, `requirements`, and `operations`, and
   `PlanStep` gains `args`, all optional; `steps` and `version` are
   unchanged.
1. `turn.request.seed_spec` is a list of 1 to 6 `Seed`s
   (`{spec, operations}`) handed over from Search; the reserved
   single-reference shape still parses and seeds nothing. The harness
   keeps each seed's `spec_id`, `definition` and
   operations, and reads every other fact from Search's answer for the
   identity.
1. A plan that ranks, totals or shares a dataset naming no place keeps
   one level of place first, and the card shows that step; no wire shape
   changes.
1. `plan.proposed` gains `asks`, each `{ask, answered_by, declined?, not_computed?}`: the outputs the user asked for, each with the steps
   that answer it, the reason it is declined, or both with the part not
   computed; optional.
1. `elicitation.request` carries one to three `questions`, each with its
   own options, free-text and skip flags; the frame keeps a whole-pause
   `skippable`. `interrupt.response` answers them with `answers` keyed by
   question id.
1. `PlanStep` gains `gives_answer`; `plan.proposed` gains `summary` and
   `switch_failed`; a new `plan.approved` frame carries the approved plan
   on the approve answer, once the run has accepted it. All optional.
1. `choice_id` is `"<requirement id>:<axis>"`, stable across switches;
   `switch_failed` gains `retryable`; a `/cancel` during a switch
   re-offers the unchanged plan on a live pause under `switch_cancelled`,
   with a reason the client shows as a quiet note;
   `plan.proposed` and `plan.approved` gain `caveats`, and `changed_from`
   names the label before the turn's first switch.
1. `plan.approved.choices[]` gains `choice_id`, the id the plan card's
   choice carries, so a client tells a planner decision
   (`"decision/<slot>"`) from a scope choice; present from v1.9, absent
   from a `plan.approved` stored earlier.
1. `plan.proposed` gains an optional `new_plan`, `true` on a plan card that
   starts a new plan after an earlier plan in the same turn, so a first draft made after an
   earlier plan card in the turn reads as a new plan with its own `plan_version_id`.
1. `POST /api/studio/resume` (`turn.resume`) carries on a running turn whose client left, on a new stream; `GET /api/studio/state` reads the conversation as stored (turn status and lease, closed turns after `since` with `closed_total`, the open pause and its frames, the manifest) with no lease; every stream names its turn in the `x-dfo-turn-index` header.
1. `done` gains an optional `new_question`: an `amend` whose text asks a different question ends the turn, its plan set aside, and the client sends the text as the next `turn.request`. An `amend` that asks for a new question without stating it ends the turn the same way with no `new_question`, and the reply asks the user for their question.
1. `turn.request` gains an optional `after_turn`, the turn it follows; a request whose `after_turn` is not the conversation's latest turn is refused 409 `edit_conflict` with `constraint_id: "turn_moved"` and begins no turn.
1. `suggestions` gains count chips: `apply_choice` chips whose `choice_id`
   is `"<spec_id>:count"` and whose `option_id` is the spec_id of another
   way of counting the part's flow that Search offered, such as net
   disbursements beside gross. They take the shape and request of a scope
   chip and run first (`docs/frontend-interface.md` 4.1.2). The plan card's
   row stating how a pinned source is counted gains `choice_id`
   `"<requirement id>:count"`, so a switch can name it (see "Count chips and
   count rows on the plan card").
1. `elicitation.request` questions gain `input: "unit"` with a `unit` offer, and `multi_select` is emitted; answers gain `{kind: "unit", ...}` and `{kind: "options", option_ids}`. `turn.request` and `interrupt.response` gain `client_features`; a client that declares no `unit_question` or `multi_select` gets those questions as `single_select` (see "Question forms a client declares"). `component.created` gains `unit_change` and `replaces`; `turn.request` gains `apply_unit`, which comes alone (see "A unit question, a card's unit change and several answers").
1. `plan.proposed.decisions[]` gains `unit` on the row stating a unit question's whole unit, whose `options` are whole units; `interrupt.response` gains `unit` on a `switch` (see "A unit row on the plan card").
1. `turn.cancel` gains an optional `turn_index`; a Stop naming another turn than the latest is answered 200 `{"code": "turn_moved", "turn_index"}` and leaves the conversation as it was. A state read's stored card gains `unit_change` and `replaces` (see "A Stop that names its turn, and a stored card's unit and slot").
1. `turn.request` gains an optional `prompt_source` (`"example"`, `"availability"` or `"search_example"`), which the turn's `studio.turn` line and Langfuse trace metadata carry; any other value is dropped (see "Where a typed question came from").
1. `component.created.created_from` gains `plan_version_id` and `plan_step_id`, `activity.step` and a state read's stored card gain `plan_version_id` and `plan_step_id`, `plan.approved.steps[]` gains `id`, and `plan.proposed.decisions[]` and `plan.approved.choices[]` gain `step_ids`, all optional (see "Plan steps on cards, activity lines and choices").
1. `component.created` and a state read's stored card gain `card_changes`, the years a card can change to and the roles whose places can change; `GET /api/studio/places` reads a role's places on demand; a state read's turns gain `card_change`; `turn.request` gains `apply_change`, which comes alone; `plan.proposed.decisions[]` and `plan.approved.choices[]` gain `kind`, all optional (see "A card's years and places, and what a plan decision settles").
