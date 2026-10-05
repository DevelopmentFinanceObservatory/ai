# Studio wire contract v1

> Frontend developers: read [`frontend-interface.md`](frontend-interface.md)
> first. It is the self-contained integration guide. This document records
> the contract's design decisions and its deltas against the original
> drafts.

The interface between the DFO harness (this repo) and the website's Studio
experience (chat + canvas). This document and the pydantic models in
`services/harness/src/harness/wire.py` are the contract's source of truth:
the models are what the server actually validates and emits, this document
explains the protocol around them, and `services/harness/tests/test_wire.py`
pins the two together against the fixtures in
`services/harness/tests/wire_fixtures/studio/`.

v1 implements the contract drafts of 2026-08-04 (`dfowebsitecontractv1`
appendix A3 and `dfofrontendpayloads`, "Studio" sections) with numbered
deltas (1–19; 16 is folded into 11) where the drafts were ambiguous,
silent, or stale. The deltas are marked where they apply; every unmarked
shape is the drafts' shape.

The contract is frozen at `schema_version: 1`. Agent internals (planning,
depth policy, tool selection, models) will keep moving underneath it; a
breaking structural change bumps the version and ships a migration note.

## Endpoints

| Method | Route                     | Returns                                                     |
| ------ | ------------------------- | ----------------------------------------------------------- |
| POST   | `/api/studio/turn`        | SSE stream of frames                                        |
| POST   | `/api/studio/interrupt`   | a NEW SSE stream resuming the paused turn (delta 1)         |
| POST   | `/api/studio/cancel`      | 204; aborts the conversation's in-flight segment (delta 14) |
| POST   | `/api/studio/composition` | the new composition manifest, as a JSON body                |

The drafts' other routes (`/api/sources`, `/api/datasets/{spec_id}`,
gallery, published views) are website/platform reads served outside the
harness.

## Auth and conversation identity (delta 5)

Every studio request carries:

- `Authorization: Bearer <token>`: required; a request without it is 401.
  In v1 the harness treats the token as opaque: a presence check, plus a
  shared-secret comparison when the `STUDIO_API_TOKEN` setting is present.
  **This is placeholder access control.** A server with the setting absent
  accepts any non-empty token and logs a warning at startup. The
  verification seam is one function in `harness/auth.py`, where full JWT
  validation plugs in later without touching routes.
- `conversation_id` in the body: client-minted, stable for one
  conversation. `session_id`, the drafts' name for it, parses as a
  deprecated alias in v1 (the fixtures still use it). Sending both
  spellings in one body is refused.

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
persists in an ADK session, through the same session service the runner
itself uses. Any harness instance can load it and continue: a plan
approval, an elicitation answer, and a follow-up turn all land correctly
whichever instance receives them. A conversation streams at most one
segment at a time, on any instance: a durable segment lease in the
ledger (`studio:segment`) is claimed by `/turn` and by `/interrupt`'s
resume, expires 45 s after it is last renewed, and is renewed by its
owner every 5 s; a second stream on a live segment gets the 409
`TRANSIENT_UNAVAILABLE` below with `retry_after_seconds` set to the
lease's remaining time. The owning instance checkpoints the full
conversation state into the ledger before any frame that says "paused"
(`plan.proposed`, `elicitation.request`) or "done" leaves the wire; a
checkpoint that fails ends the stream with a stream-fatal
`transient_unavailable` error in that frame's place, so any success
frame the client sees already matches the ledger. Persistence
itself is a deployment setting (`DATABASE_URL` or the `SESSION_STORE_*`
family, see `services/harness/README.md`); unset, the harness falls back
to an in-memory session service, and a restart or a request landing on a
different instance loses everything. What a stalled owner writes after
losing its segment stays unfenced: its per-turn ADK session writes race
the new owner's (ADK's own compare-and-swap fails one of the two), and a
presentation-platform write already in flight can land as an orphaned
component. One piece of state stays bound to the instance that produced
it regardless of the session backend: editing a component requires the
instance that created it, because the presentation platform leases a
component to the MCP session that created it (above), and that session,
and the frame lineage behind it, exist only in that instance's process.

A `composition.mutate` body also carries `conversation_id` (delta 5; the
`request-05` fixture carries it).

## Requests: four kinds, discriminated on `kind`

Models: `TurnRequest`, `InterruptResponse`, `CompositionMutation`,
`CancelRequest` (`STUDIO_REQUEST_ADAPTER` parses the union).

- **`turn.request`** starts or continues a turn. `message` is the user's
  prose. `surface` ∈ `studio | download | embed` and is **optional with
  default `"studio"`** (delta 6: doc 1 required it, doc 2's fixtures omit
  it; resolved in favour of the lenient parse). It feeds the harness's
  depth policy per turn (`download` earns no dialogue). `component_ref` scopes the turn to one card ("Ask follow
  up" / agent-driven "Edit Card"). `seed_spec` is the Search handoff
  (delta 18, see "Seeds from Search" below). `auto_add`
  (optional, default `true`; delta 11): when `false`, components the turn
  creates still stream as `component.created` but are NOT appended to the
  composition; the client adds them later with an `add` op (the Auto-add
  toggle and the "✓ Added" chip).
- **`interrupt.response`** resumes a paused turn: `interrupt_id` echoed
  verbatim from the pausing frame, `decision` ∈
  `approve | amend | reject | answer | skip`, `payload` carrying the
  amendment text (`amend`) or the answer (`answer`, an option id, or free
  text when the elicitation set `allow_free_text`). `skip` (delta 12)
  applies to elicitations that set `skippable`: the run proceeds on its
  default reading (the top-ranked candidate).
- **`turn.cancel`** → `POST /api/studio/cancel` (delta 14) aborts the
  conversation's in-flight segment (the Stop button), whichever instance
  owns it: the request sets the segment's durable cancel flag in the
  ledger, and the owning instance's own heartbeat observes it and ends
  the stream within one heartbeat interval (5 s). Idempotent; returns
  204\. The aborted stream, if still open, ends with a stream-fatal `error`
  frame whose code is `cancelled` (a studio-side addition to the error
  grammar); no `done` follows. Client disconnect (closing the SSE
  response) also aborts the running segment when the server observes it
  (the ASGI server cancels the response task), but a dropped connection
  is not reliably observable through proxies, so `cancel` is the
  dependable path.
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

## The SSE stream

Frames are SSE `data:` lines, one JSON object per frame, discriminated on
`event` (`STUDIO_FRAME_ADAPTER`). Twelve frame types; at least one fixture
each in `wire_fixtures/studio/` (elicitation has two: the plain and the
skippable/free-text shapes).

| `event`               | What it is                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `message.delta`       | append to the streaming assistant message; text is GitHub-flavored Markdown (delta 15)                                               |
| `message.complete`    | final text for the turn's message                                                                                                    |
| `plan.proposed`       | the plan awaiting approval; **pauses and ends the stream** (delta 1)                                                                 |
| `plan.versioned`      | a new plan version after an amendment; a fresh `plan.proposed` follows                                                               |
| `activity.step`       | one execution step, the only progress channel; `step_id` is the call's position within the stream (see `docs/frontend-interface.md`) |
| `component.created`   | a new card: opaque `payload` + `fact_summary` + inline `provenance`                                                                  |
| `component.revised`   | new revision under the same `component_id`                                                                                           |
| `composition.updated` | the authoritative canvas manifest                                                                                                    |
| `elicitation.request` | a clarifying question; **pauses and ends the stream** (delta 1)                                                                      |
| `suggestions`         | follow-up chips (may be empty in v1; delta 8)                                                                                        |
| `error`               | stream-fatal tool error (delta 4)                                                                                                    |
| `done`                | turn complete: `trace_id` + final composition or null                                                                                |

`component.created.payload` / `component.revised.payload` are the platform's resolved revision, passed through verbatim as the envelope `{schema_version, catalog_version, component_id, revision, provenance, component}` that dfo-spec-kit's `envelope.schema.json` defines. `payload.component` is the resolved component for its type, points included, and `payload.provenance` is the platform's provenance for that revision, authoritative for the card's back. The member is opaque to this contract, which does not restate its shape.

### Interrupt resume protocol (delta 1)

`plan.proposed` and `elicitation.request` are **terminal for their HTTP
response**: the run is genuinely paused (nothing is retrieved or computed),
the frame is the last one on that stream, and the stream closes. To resume,
POST `interrupt.response` to `/api/studio/interrupt`; the response is a
**new SSE stream** continuing the same turn. A new stream per resume is
deliberate: human approval can take minutes, and idle SSE connections
through the website proxy and Cloud Run are fragile.

Amendment flow: `decision: "amend"` yields `plan.versioned` followed by a
fresh `plan.proposed` (a new pause, new `interrupt_id`) on the new stream.

An `/interrupt` answer stays retryable after a crash: the answer is
recorded durably before the resumed stream starts, so a client that gets
no response (the instance resuming it died before finishing) can retry
the identical `interrupt.response` body. Retrying while that resumed
segment is still live answers 409 `transient_unavailable` with
`retry_after_seconds` set to the lease's remaining time; retrying after
that segment's lease has expired resumes the turn again, on a fresh
stream whose `seq` jumps 1000 past the saved value so it never repeats a
`seq` the dead stream already sent. Retrying with a DIFFERENT decision
or payload than the one that actually resumed answers 409
`edit_conflict`. The approval record for a plan interrupt is appended
once per `interrupt_id`, on its first answer, however many times the
resume itself retries.

### `seq` (delta 2)

`seq` is monotonic **per turn**, across resumed segments: the first frame
after a resume continues counting where the paused segment stopped. Gaps
are legal; order is not. Use it to order frames, never as an index.

### Heartbeats (delta 3)

During long tool calls (data mints run up to ~125 s) the stream carries SSE
comment lines (`: ka`) roughly every 15 seconds. Standard SSE clients
ignore comments; proxies see traffic. Heartbeats carry no `seq`.

### Error fatality (delta 4)

Two failure channels, by scope:

- A **step-scoped** failure arrives as `activity.step` with
  `status: "failed"` and the tool error's detail in `detail`; the run
  continues where the plan allows, and `done` still arrives.
- A **top-level `error` frame is stream-fatal**: `done` does not follow it.
  Branch on `error.retry`: `correct_and_retry` means a fixed call can work
  and `fix` says how; `refresh_then_retry` means your state is stale;
  `retry_after_delay` says when; `do_not_retry` means don't offer one.

The error grammar (`ToolError` × four retry modes) is a closed WIRE
vocabulary: the platform refusal codes the studio can surface, plus the
studio-side `cancelled` and `capability_unavailable`. The platform's own
grammar is a superset that keeps growing; codes the wire enum does not
know are surfaced as `transient_unavailable` with the real reason in
`detail`, and step-scoped refusals travel as prose in
`activity.step.detail`.

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
  canvas) → `message.complete` → `suggestions` → `done`. An impossible
  edit (e.g. selecting data the card's frame doesn't hold) closes with an
  honest `message.complete` and no component frame.
- **Card follow-up**: `component_ref.intent: "follow_up"` scopes the turn:
  the referenced card's digest slice travels with the question.
- **Canvas questions**: "what's on my canvas?" is answered from the
  conversation's canvas digest without any data search (message frames,
  `suggestions`, `done`). An empty canvas answers "nothing yet".

Cross-turn context: each turn seeds two digests into the opening and
planning model calls, the canvas digest (block titles, types, fact
summaries, never rows; bounded to the newest 16 blocks), which is what
lets "now do the same for Uganda" resolve, and the conversation digest
(the newest 10 turn records: user message, assistant close, resulting
cards; byte-capped), which is what lets "the country I mentioned" resolve
when the earlier turn made no card. Context past either bound is served
on demand by a bounded server-side recall step (one round per turn,
mechanical store lookup: digests and fact summaries, never rows).
Nothing about this appears on the wire: no new frames, no new request
fields.

## Composition state (delta 9)

The harness owns a minimal composition manifest in conversation state:
blocks (component_id, revision, type, title, pinned, optional `meta`
(delta 13)), server-computed `counts`, `revision`, and `layout_hints`. The front end holds no layout
state of its own; re-render from every `composition.updated` frame and
from `/api/studio/composition` responses. `expected_revision` mismatches
refuse with `edit_conflict`. `/composition` can land while a turn streams
on the same conversation, on the same or another instance: the mutation
rebuilds the manifest from the ledger's latest value plus that turn's own
not-yet-saved edits before applying the operation, and the turn's own
checkpoint replays its edits on top of whatever `/composition` landed in
the meantime, so both writers' edits survive.

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
`{spec, operations}`, or `null`. The drafts reserved the member as one
`{spec_id, content_version}` reference, which the server parsed and did
not act on. That shape still parses and still seeds nothing, since a
reference carries no Spec block to start from: the turn runs from the
question, and the `seed` node reports the reference in a failed step,
`seed_reference_unused`. The change adds a shape and removes none, so
`schema_version` stays 1.

- `spec` is a definite part's Search API Spec block (`SeedSpec`), capped
  at `MAX_SEED_SPEC_BYTES` as posted. The harness keeps its `spec_id`,
  its `definition` and the dcids in `resolved.stat_vars.items[].ref`,
  and drops every other member as it reads the seed. A client may send
  `resolved` holding `stat_vars.items[].ref` alone.
- `operations` are the question's operations on it (`SeedOperation`):
  `{verb, phrase, over, target_year}`, with no other member, at most
  `MAX_SEED_OPERATIONS` (8) of them. `target_year` is the year a forecast
  projects to, 1000 to 2200, set on a `forecast` alone.

A seed is browser input, so every member the harness keeps is bounded
(`harness.wire`'s `MAX_SEED_*`, each stating the Search API limit it
comes from). The harness reads nothing else of it. Every label,
sentence, coverage count, headline, slot and summability flag a seeded
turn uses comes from the Search API's own answer for the seed's
identity, which is held to the same bounds (`SpecBlock`), as is what one
seed puts into a tool result (`MAX_SEED_SOURCE_CHARS`) and into session
state (`MAX_SEED_RECORD_CHARS`). A seed or an answer past a bound is
refused whole.

The wire holds the items unparsed, and the route refuses a body past
`MAX_TURN_REQUEST_BYTES` before parsing it. A seeded request cannot
carry an `edit` `component_ref`. The graph's `seed` node reads each item
once as a `Seed` (`harness.seeds.take_seeds`), so one the harness
cannot use fails its own `seed-<n>` step (`seed_invalid`,
`identity_mismatch`, or `seed_duplicate` for a `spec_id` an earlier
usable seed names) and the others go on.

The node then resolves every usable seed's identity again, all at once:
one `resolve_definition` call per seed to the data-mcp at
`DATA_MCP_URL`, which posts the identity as its own base, with the
StatVar dcids, to the Search API's `dataset_resubmit` door and stores
the definition Search answers with, so the `get_data` that retrieves
the seed reads it wherever data-mcp runs. The turn waits for the
slowest answer. A seed data-mcp or Search could not be reached for
fails its step (`service_unavailable`); when that leaves no seed, the
receive step fails too. A seed whose identity Search refuses, no longer
resolves to a definite part, or resolves to another `spec_id` is
dropped: its step fails as `seed_dropped`, with the reason and a request
to search for it again, and the turn plans from the others. It leaves
the turn as a stale seed does (below): no retrieval in the turn takes
that dataset again, and the turn's next words name it, as
`dataset <n> from Search` with Search's reason, in a `message.delta`
before the plan card, in the `question` of an `elicitation.request`, or
in the closing text when no plan is drafted. The node retrieves nothing.
It plans from the kept seeds in place of `probe`, and a turn with no
kept seed falls back to the question alone, whose search `probe` reads
without any reading of a dropped dataset: a `definite` reading of one,
or candidates naming only such datasets, close the turn with text
naming the drop. The receive step is recorded on every turn that posts a
list of seeds, counting the seeds kept, none included.

A plan pins each seed over the years it retrieves. A fetch with no plan
retrieves the year Search showed when the question named none. Every
`get_data` of a seed posts the StatVars Search's answer lists in
`resolved.stat_vars.items` as `resolved_stat_var_dcids` when that list is
the dataset's membership, which holds when `resolved.shape` states no
`member_digest` or the seed mints as its family's root. A family with a
member digest posts none, since the mint elects its members again.

On the `download` surface the first kept seed is retrieved, every
other kept seed fails its step as `one_dataset_per_download`, and the
retrieved seed's operations are reported by a failed
`seed-<n>-operations` step, `no_operations_on_download`.

A seed whose StatVars have left its dataset fails its retrieval. A turn
with another usable seed drops it and offers a new plan from the
others: the dropped seed's name arrives in a `message.delta` frame
before the new `plan.proposed`, and the plan cannot retrieve that
dataset again in the turn, pinned or through a search
(`planner.plan.names_dropped_seed`). The turn stops, closing with a
remedy asking for a fresh search, when it has one usable seed, when
every usable seed is stale in the same run, when another step of that
run stopped on a failure a new plan cannot fix, when the turn already
replanned once, and on the `download` surface. A replan that drafts no
plan, or that fails, closes with text naming the dropped seed and the
seeds left unplanned. The drop builds the resolve envelope and the depth
verdict again from the seeds left, so the replanned card's `depth` and
`depth_notice` describe those.

Fixture: `request-11-turn-seeded.json`.

## A question that names no place (delta 19, v1.4)

A dataset that names no place covers every place with data, at every
level: the world, regions, World Bank groups, countries and their
subdivisions. dfo-apis falls back to the countries when the full list is
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
in the order the orchestrator engine listed them when it called the
planner, as `{ask, answered_by, declined?, not_computed?}`. `answered_by`
holds the `PlanStep` ids whose results answer the ask; `declined` says in
words why the plan does not compute it. A declined ask has an empty
`answered_by`, unless the plan answers it in part: such an ask carries
both, and `not_computed` names the part the plan does not compute, in the
user's words. `not_computed` is optional and appears only on an ask
answered in part. Every ask is answered, declined, or answered in part
exactly once. A plan given no asks, and every plan on the classic engine,
omits the field. `schema_version` stays 1.

## Corrections to the drafts (delta 10)

Fixture values that are samples, corrected here:

- The fixtures' `interrupt_id: "approve"` literal is a sample value (see delta 7).
- Forward-dated catalogue versions in fixtures (e.g. `2026.10.0`) are
  sample values.
- Fixture tool names (e.g. `datasets.get`) are samples; `tool` in
  `activity.step` is a display string.
- The `component.created` fixture's `fact_summary.series[].change`
  arithmetic is a sample value.
- The `component.created` fixture's `payload` holds only `schema_version` and the component's `type` and `title`. A served `payload` is the full resolved revision.

## Known gaps (upstream, named)

- **`fact_summary.indicators` may be empty**: the platform serves no
  indicator definition text. `attribution` counts still reflect the data.
- **Some questions refuse at resolve** with `capability_unavailable`
  ("identity does not reproduce spec_id"), an upstream/spec-kit identity
  gap on specific spec ids (not arity-dependent), reported to ONE. The
  stream fails safe: a stream-fatal `error` frame carrying that code and
  the refusal's own detail.

## Reference implementation and fixtures

- Models (enforced): `services/harness/src/harness/wire.py`
- Server: `services/harness/src/harness/studio.py` (routes, runner, SSE),
  `studio_state.py`, `studio_translate.py`, and `studio_edit.py`, behind
  `POST /api/studio/*` on the harness app
- Static fixtures (one per frame/request):
  `services/harness/tests/wire_fixtures/studio/`
- Recorded LIVE transcripts (real model, real platform, real WDI data):
  `docs/transcripts/`: four recorded live streams, every frame
  contract-validated: `studio_transcript.jsonl` (direct-fetch turn),
  `studio_transcript_plan.jsonl` (plan → approve → component),
  `studio_transcript_followup.jsonl` (three turns on one conversation:
  fetch, a digest-resolved follow-up, a canvas question), and
  `studio_transcript_edit.jsonl` (two Edit Card turns: a title revise →
  `component.revised`, then a type change → replacement
  `component.created`), recorded from a scripted SSE client
- To run a conversation yourself:
  `uv run --env-file .env harness-server` then
  a local (untracked) `test_scripts/studio_client.py` on the dev machine;
  any SSE client following the resume protocol above reproduces them

## Delta index

1. Interrupts close the stream; `/api/studio/interrupt` answers with a new
   stream continuing the turn.
1. `seq` is monotonic per turn across segments; gaps legal.
1. SSE comment heartbeats (`: ka`) ~every 15 s during long calls.
1. `activity.step status=failed` is recoverable; a top-level `error` frame
   is stream-fatal and `done` does not follow.
1. `Authorization: Bearer <token>` + client-minted `conversation_id` on
   every request (`session_id` accepted as deprecated alias);
   `composition.mutate` gains `conversation_id`; conversation state keyed
   per principal; each conversation maps to its own platform
   `x-dfo-session` server-side.
1. `surface` optional, default `"studio"`.
1. Wire `interrupt_id`s are opaque; never hardcode fixture literals.
1. `suggestions` may be empty in v1.
1. The harness owns the minimal composition manifest;
   `expected_revision` → `edit_conflict` on mismatch.
1. Stale draft artefacts corrected (see above).
1. `turn.request.auto_add` (default `true`) decouples component creation
   from composition membership; `component.created` on an auto_add=false
   turn does not imply a composition block; `composition.mutate` gains the
   `add` op, and `composition_id` becomes optional: omitted targets the
   current composition, and an all-`add` mutate on an empty canvas
   bootstraps one.
1. Elicitations may set `allow_free_text`, `skippable`, and a `progress`
   counter (`{completed, total}`, the `StepProgress` shape);
   `decision: "skip"` proceeds on the default reading;
   an `answer` payload may be free text when `allow_free_text`. v1 sets
   `allow_free_text` on every elicitation, so a payload outside the option
   ids is taken as free text and refines the resolve.
1. `CompositionBlock.meta` (`set_card_meta` op) carries presentation-only
   card fields (subtitle/description/axis labels/style) the platform
   intent cannot hold; harness-owned, never sent to the platform.
1. `POST /api/studio/cancel` aborts the in-flight segment, reaching one
   running on any instance within one heartbeat interval; the stream ends
   with error code `cancelled`. Client disconnect aborts too.
1. `message.delta` / `message.complete` text is GitHub-flavored Markdown;
   render it, don't escape it.
1. (folded into delta 11.)
1. `plan.proposed` gains `revision`, `requirements`, and `operations`, and
   `PlanStep` gains `args`, all optional; `steps` and `version` are
   unchanged.
1. `turn.request.seed_spec` is a list of 1 to 6 `Seed`s
   (`{spec, operations}`) handed over from Search; the reserved
   single-reference shape still parses and seeds nothing. The harness
   keeps each seed's `spec_id`, `definition`, StatVar dcids and
   operations, and reads every other fact from Search's answer for the
   identity.
1. A plan that ranks, totals or shares a dataset naming no place keeps
   one level of place first, and the card shows that step; no wire shape
   changes.
1. `plan.proposed` gains `asks`, each `{ask, answered_by, declined?, not_computed?}`: the outputs the user asked for, each with the steps
   that answer it, the reason it is declined, or both with the part not
   computed; optional.
