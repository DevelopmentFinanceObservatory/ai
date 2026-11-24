# DFO Studio: the agent interface for the frontend

This document tells you how to connect the Studio frontend to the agent
service. It is the one document you need. The source of truth for every
shape is `services/harness/src/harness/wire.py`, and
`services/harness/tests/wire_fixtures/studio/` holds one example of each
frame and request. This document agrees with both.

## Start here

1. Get the repo and install: `uv sync --all-packages` (Python 3.13, uv).

1. Put a `GOOGLE_API_KEY` in `.env` (`just env`) and start a local
   server against the real model and platform: `just studio-live`. Every
   turn costs model spend. The server stores conversations in Postgres:
   the recipe starts a local one in Docker unless `.env` sets
   `DATABASE_URL`.

1. Send a turn and watch the stream:

   ```sh
   curl -N -X POST http://localhost:8080/api/studio/turn \
     -H 'Authorization: Bearer dev' -H 'Content-Type: application/json' \
     -d '{"kind":"turn.request","schema_version":1,
          "conversation_id":"conv_1","message":"life expectancy in Kenya"}'
   ```

   You get `data:` lines: steps, a card, the message, `done`. Render them
   per section 5.

1. Build each UI state against the fixtures in
   `services/harness/tests/wire_fixtures/studio/` and the prompt gallery
   (section 12).

1. When your UI renders all of that, change the base URL to the dev
   deployment. Nothing else changes.

Everything below is the reference for what you built in steps 3 and 4.

The words in this document have one meaning each. A **conversation** is one
chat with its one canvas. A **turn** is one user message and the agent's
response to it. A **frame** is one JSON object the server sends on the
stream. A **component** is one card the agent made. The **composition** is
the list of blocks on the canvas.

## 1. Connection and identity

- Send all `/api/studio/*` requests as HTTP POST with a JSON body.
- Send the header `Authorization: Bearer <token>` with every
  `/api/studio/*` request. A request without it gets HTTP 401.
  `GET /healthz` needs no header and no body. Access control is a
  placeholder for now: when the server has a shared secret configured, the
  token must be that secret; until then any non-empty token is accepted.
  Send the token from your server-side proxy, never from the browser.
- Make a new `conversation_id` on the client for each conversation. Send it
  in every request body. The server keys all state to your token and this
  id together. On a deployment with a database configured, that state is
  durable, so a plan approval, an elicitation answer, or a follow-up turn
  reaches whichever server instance handles it.
- Send `schema_version: 1` in every request body.
- Unknown fields are errors. The server refuses a request with a field this
  document does not name.
- The production server sets no CORS headers; same-origin access is your
  arrangement to make. A live local server answers preflights for the
  origins listed in `STUDIO_CORS_ORIGINS` (comma-separated).

## 2. Endpoints

| Method | Path                      | Response                                         |
| ------ | ------------------------- | ------------------------------------------------ |
| POST   | `/api/studio/turn`        | SSE stream of frames                             |
| POST   | `/api/studio/interrupt`   | SSE stream of frames (continues a paused turn)   |
| POST   | `/api/studio/composition` | JSON: the composition manifest after the request |
| POST   | `/api/studio/cancel`      | HTTP 204, no body                                |
| POST   | `/api/studio/resume`      | SSE stream of frames (continues a running turn)  |
| GET    | `/api/studio/state`       | JSON: the conversation as stored                 |
| GET    | `/healthz`                | JSON liveness                                    |

## 3. The stream

The server answers `/turn`, `/interrupt` and `/resume` with Server-Sent Events. Each stream's response carries the `x-dfo-turn-index` header, the index of the turn it carries: the `turn.index` a state read gives that turn (§4.6). Key each prompt and reply on it.

- Each frame is one `data: <json>` line. Parse the JSON. Read the `event`
  field to know the frame type.
- Lines that start with `:` are keep-alive comments. The server sends
  `: ka` about every 5 seconds when a step is slow. Ignore these lines.
- `seq` increases within one turn. It does not reset when a paused turn
  continues on a new stream. Gaps are permitted. A new turn starts at 0.
- A stream ends in one of three ways. (1) A `done` frame: the turn is
  complete. (2) A `plan.proposed` or `elicitation.request` frame: the turn
  is paused. Answer on `/interrupt`. (3) An `error` frame: the turn failed.
  No `done` follows an `error`. The server saves the conversation's state
  durably before it sends a `done`, `plan.proposed`, or
  `elicitation.request` frame; if that save fails, the stream ends with a
  stream-fatal `error` frame (`transient_unavailable`) in that frame's
  place, so any pause or completion frame you see is one the server has
  already durably recorded.
- One request streams per conversation at a time, across every server
  instance: a durable lease in the conversation's ledger, renewed every 5
  seconds and good for 45, enforces it. A second `/turn` or `/interrupt`
  during a stream gets HTTP 409 with `retry_after_seconds` set to the
  lease's remaining time. `/cancel`, `/composition` and `GET /state` are
  not blocked by a stream. To stop a stream, use `/cancel` or close the
  connection;
  `/cancel` reaches a stream running on another instance within about 5
  seconds.

## 4. Requests

### 4.1 `turn.request` → POST `/api/studio/turn`

| Field             | Type                                    | Notes                                                                           |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| `kind`            | `"turn.request"`                        | required                                                                        |
| `schema_version`  | `1`                                     | required                                                                        |
| `conversation_id` | string                                  | required                                                                        |
| `message`         | string                                  | the user's text; required unless `apply_choice` or `apply_unit` is set          |
| `surface`         | `"studio"` \| `"download"` \| `"embed"` | optional; default `"studio"`                                                    |
| `component_ref`   | object \| null                          | optional; see section 8                                                         |
| `seed_spec`       | list of `Seed` \| null                  | optional; 1 to 6 Search results to start from; see 4.1.1                        |
| `auto_add`        | boolean                                 | optional; default `true`; see section 9                                         |
| `apply_choice`    | object \| null                          | optional; a clicked suggestion chip; see 4.1.2                                  |
| `apply_unit`      | object \| null                          | optional; a card's unit change; see 4.1.4                                       |
| `after_turn`      | integer \| null                         | optional; the turn this request follows; see 4.1.3                              |
| `client_features` | list of string                          | optional; the question forms the page renders; absent keeps the list; see 4.1.5 |

#### 4.1.1 Seeds from Search

A "... in Studio" action on the Search page sends the question with one
seed for each definite part of the result the user saw. `seed_spec` is a
list of one to six seeds, or `null`, or absent. A seed is
`{spec, operations}`:

- `spec` is the part's Spec block as the Search API returned it. The
  harness keeps `spec_id`, `definition` (`{schema_version, spec_id, identity}`)
  and `resolved.entity_groups`, the sets the part's places were asked for
  as ("African countries"), and ignores every other member, so a client
  may leave `resolved` out.
  `definition.spec_id` must equal `spec_id`. The block may run to 1,048,576 bytes of compact
  JSON; a CRS block is about 400 KB.
- `operations` lists what the question asked of that data that Search
  does not compute, as the Search API's `Operation`:
  `{verb, phrase, over, target_year}`. `verb` is one of `rank`,
  `aggregate`, `growth`, `share_of`, `explain`, `forecast` or `report`.
  `phrase` is the question's own words for it. `over` is `place`,
  `origin`, `destination`, `category`, `year` or null. `target_year` is
  the year a forecast projects to, a whole number between 1000 and 2200,
  set on a `forecast` alone, or null. The list holds at most eight
  operations and may be empty, a `phrase` runs to 200 characters, and an
  operation with any other member is refused.

`seed_spec` also accepts one `{spec_id, content_version}` object, the
shape this member was reserved with before it carried seeds. A reference
carries no Spec block, so it seeds nothing: the turn runs from the
question, as it did while the member was reserved, and the "Receive the
datasets from Search" step shows `failed` with code
`seed_reference_unused`.

Some requests are refused whole. A body past 6,356,992 bytes gets a 413
`resource_exhausted` before it is parsed. A `seed_spec` gets a 422 when
it is an empty list, holds more than six items, holds an item that is
not a JSON object, or is a list that comes with an `edit`
`component_ref`. A `client_features` list of more than 16 values
gets a 422.

Each seed is checked inside the turn, so one bad seed fails alone, with
a failed step `seed-<n>` (`n` is its position in the list, from 0) that
states a code and the reason, and the other seeds go on:

- `identity_mismatch`: the identity does not hash to the `spec_id`.
- `seed_duplicate`: an earlier usable seed names the same `spec_id`. The
  first stands, whatever the repeat's block says.
- `seed_invalid`: anything else. The identity must fit in 65,536 bytes
  of compact JSON, each member the harness keeps must have the type the
  Search API gives it, and each is within its bound.
- `seed_dropped`: the Search API refuses the seed's identity, no longer
  resolves it to a definite part, or resolves it to another `spec_id`.
  The step states Search's reason and asks the user to search again.
- `service_unavailable`: data-mcp, the Search API or the definition
  store could not be reached for the seed.

The bounds are what the Search API can serve. They hold for the members
a seed carries and for Search's answer for its identity, which supplies
the rest. A seed or an answer past one is refused whole; nothing is cut
to fit.

| Member of `spec` or of Search's answer                       | Bound                     |
| ------------------------------------------------------------ | ------------------------- |
| `resolved.rendered_sentence`                                 | 2,000 characters          |
| `resolved.entities`                                          | 1,024 items               |
| the roles of `definition.identity.entities`                  | 16                        |
| `resolved.slots`, `coverage.dimensions`, `role_axis_facts`   | 64 items each             |
| any other string the harness reads (a dcid, a role, a label) | 256 characters            |
| any count or year the harness reads                          | a whole number, 12 digits |

**What a seeded turn shows.** The harness resolves each usable seed's
identity again through the Search API, all seeds at once, stores the
definition Search answers with in data-mcp's definition store, and
retrieves nothing yet. Every label, sentence, count and headline the turn
shows comes from that answer, never from the block the client posted. The
Activity feed shows one "Receive the datasets from Search" step counting
the seeds kept. A seed dropped or not reached shows its own failed
`seed-<n>` step. The agent then starts from the kept seeds as definite
search results: it fetches or plans with them without searching, and the
turn goes on as any data turn does. When no seed is kept, the turn
continues from the question alone, as a turn without seeds does.

**Planning.** The agent hands the operations a seed carries to the
planner. The planner pins each seed and chooses the years to retrieve,
and the plan retrieves the seed over those years. One retrieval returns
at most data-mcp's `MAX_FRAME_ROWS` rows, so the planner asks for the
fewest years the question needs: one year for a rank across places or
donors, and the range a growth runs over. Education aid to Kenya is 242
rows in 2024 and 3,780 over every year. A retrieval past the ceiling
shows a failed step, and the planner replans with a narrower window and
the seed still pinned. An operation over `place`, `origin` or
`destination` runs across every member of the axis the seed leaves open,
and a rank that asks for a top or bottom N keeps that many rows. A
component draws at most 24 series.

**The download surface.** A `download` turn retrieves one dataset and
computes nothing: the agent fetches once and does not plan.

**A dropped seed.** The turn drops a seed when Search's answer for its
identity is not the dataset the user handed over: Search refuses the
identity, resolves it to something other than one dataset, resolves it
to another `spec_id`, or answers with a block the harness cannot read.
The turn checks each seed once, as it receives the seeds and before it
retrieves anything. The seed's `seed-<n>` step fails with code
`seed_dropped` and carries the reason. The agent is told the seed's
name, `dataset <n> from Search` (its position in `seed_spec`, from 1),
and Search's reason, and its reply tells the user and suggests searching
for the dataset again.

#### 4.1.2 Chips

After an answer whose figures came from a fetch, `suggestions.items` can
hold up to four chips of kind `apply_choice`:
`{kind: "apply_choice", label, choice_id, option_id, turn, spec_id}`. Each moves one
choice of the data the answer used to another option. A scope chip moves
one scope choice (the sector's codes, a region's members, the provider
scope or the measure basis) to an option Search can serve, such as
"health, CRS 120 alone" or "including imputed multilateral". Its
`choice_id` is `"<spec_id>:<axis>"` and its `option_id` the option's id. A
count chip switches a part to another way of counting its flow that
Search offered beside the count it chose, such as "net disbursements" on
an answer counted in gross disbursements. Its `choice_id` is
`"<spec_id>:count"`, with the answer's `spec_id`, and its `option_id` is
the spec_id of the other count. Both kinds have the same shape and are
sent the same way, so a client renders and sends a count chip as it does
a scope chip, and tells them apart, where it wants to word them apart, by
the `:count` ending of `choice_id`. `label` is the option's label, quoted
after the part's words when the answer used several parts. A client
builds the chip's action text from it. Chips run count first, then
sector, region, provider and measure basis. An option Search cannot serve
has no chip, and an answer from a plan has none: its card carries the
switch. `turn` is the number of the conversation turn whose answer offers
the chip, counted from 1, and `spec_id` names the stored data the chip
re-counts. Together they bind the chip to its answer.

A click sends a `turn.request` with
`apply_choice: {choice_id, option_id, turn, spec_id}` copied from the
chip and no `message`, `seed_spec` or `component_ref`;
one beside any of them gets a 422. A scope chip's turn re-counts the
stored data with that option through the Search API. A count chip's turn
reads the other count Search already named, with nothing re-counted.
Either turn retrieves the data and answers from it with the usual basis
line, without searching or planning. The Activity feed shows a "Count
your choice" step and the retrieval. When Search cannot serve the option,
the step fails with the note a plan card's switch shows, the agent says
so, and the earlier answer stands.

A chip that no longer applies gets a 422 `constraint_violation` before
any stream starts, with `pointer: "/apply_choice"` and a `detail` to
show:

- `scope_choice_stale`: a later turn has answered, so the chip's answer is
  no longer the latest; a switch on a later plan card has re-counted the
  choice's data, so its newest data is no longer `spec_id`; a later
  retrieval read one of the part's other counts, so the part
  reads a count other than that of `spec_id`; no stored answer holds the choice; or the
  choice is already on that option. A client shows chips under the latest
  answer only.
- `switch_option_unavailable`: the choice does not offer the option, or
  Search cannot serve it. For a count chip, the part offers no other count
  with that spec_id.

#### 4.1.3 `after_turn`

Set `after_turn` to the `turn.index` the page last read (0 before the first turn) when the request must follow that turn, as when the page sends a closed turn's `new_question` itself. Two pages open on one conversation can both send it, and the server begins at most one turn: a request whose `after_turn` is not the conversation's latest turn gets 409 `{error: "edit_conflict", retry: "refresh_then_retry", constraint_id: "turn_moved", pointer: "/after_turn", detail, fix}` and begins nothing. On `turn_moved`, read the state, show the turns the page lacks and do not send the question again. A busy conversation still answers 409 `transient_unavailable` with `retry_after_seconds`; wait and send again, and the retry answers `turn_moved` if the other page's turn began.

#### 4.1.4 `apply_unit`

A card whose `component.created` frame carries `unit_change` can change its unit. `unit_change` is `{spec_id, offer}`; `offer` is `{currencies[], presets[], start}`, where each currency is `{code, name, prices[], base_year_min?, base_year_max?}` and `start` is the card's current unit `{currency, prices, base_year?}`. `prices` is `"current"` or `"constant"`; `base_year` is set exactly when `prices` is `"constant"`.

Send `turn.request` with `apply_unit: {component_id, spec_id, turn, unit}`: the card, the `spec_id` from `unit_change`, the turn that drew the card, and the new unit `{currency, prices, base_year?}`. It carries no `message`, `seed_spec`, `component_ref` or `apply_choice`; one beside any of them gets a 422.

The server refuses a change it cannot apply with 422 `constraint_violation` before the stream starts, its `pointer` starting `/apply_unit` and its `detail` written for the user: `scope_choice_stale` when the card has left the canvas or no longer reads `spec_id`, `unit_not_offered` for a unit outside the offer or the unit the card already shows. `turn` decides nothing, so a card from any earlier turn can change unit.

Otherwise the stream runs with no model call: `activity.step` frames, a `component.created` for the redrawn card, with a new `component_id`, `replaces` naming the old card, and a fresh `unit_change` that starts on the new unit, a `composition.updated` whose manifest holds it in the old card's slot, then `suggestions` and `done`, with no `message.complete`: the redrawn card is the reply, so show no assistant text for the turn. Swap the old card for the new one where it stands: the widget keeps its turn, pin, position, size and note under the new `component_id`, and the change does not advance the turn the canvas hides earlier cards by. Take the title from the new card, which names the new unit where the old title named the old one; a title the page overrides on its side moves with the card. The new block keeps the old block's `pinned` and `meta`, with the words of `meta`'s `subtitle`, `description`, `x_label` and `y_label` that named the old unit naming the new one. A `row` entry of `layout_hints` names the new card; a `comparison` entry, which aligns its members' units, drops it, and goes when fewer than two members remain. A step that fails ends on a failed `activity.step` and a `message.complete` that begins "The card is unchanged.", and the card stays as it was. A card removed while the change runs ends on a failed `activity.step` and a `message.complete` saying "This card is no longer on the canvas, so nothing changed.", with no `component.created`. The change counts as a turn. A state read gives its `reply` as empty and its `user_message` as "Change Aid to Kenya to constant 2023 US dollars.".

#### 4.1.5 `client_features`

`client_features` lists the question forms the page renders: `"unit_question"` for a question whose `input` is `unit`, and `"multi_select"` for one whose `input` is `multi_select`. The server ignores a value it does not know. Send both on every `turn.request` and `interrupt.response`. The conversation keeps the list the latest request carrying the field declared, and every question the turn asks, before and after a resume and on any server instance, takes the forms it names. A request without the field leaves the list as it was; a new conversation starts with none.

A page that declares no `"unit_question"` gets a unit question as a `single_select` with no `unit` and no free text. Its options are both price bases of the offer's start currency, or, when that currency serves one basis, of the first offered currency that serves both, the start's basis first, such as `USD-constant-2023` "Constant 2023 US dollars" and `USD-current` "Current US dollars". An `option` answer moves the reading to that unit, as a `unit` answer does. A page that declares no `"multi_select"` gets a multi-select question as a `single_select` with the same options and free text, and an `option` answer counts as one tick. A question keeps the form it was sent in, and its answer is checked against that form whatever the answering request declares.

The harness deploys before the website. A harness that predates the field refuses it with 422, and a page opened before the website deployed sends none and gets the `single_select` forms on a conversation no newer page declared features for.

### 4.2 `interrupt.response` → POST `/api/studio/interrupt`

| Field             | Type                                                                           | Notes                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `kind`            | `"interrupt.response"`                                                         | required                                                                                                                    |
| `schema_version`  | `1`                                                                            | required                                                                                                                    |
| `conversation_id` | string                                                                         | required                                                                                                                    |
| `interrupt_id`    | string                                                                         | required; copy it from the pausing frame; it is opaque and single-use                                                       |
| `decision`        | `"approve"` \| `"amend"` \| `"reject"` \| `"switch"` \| `"answer"` \| `"skip"` | required                                                                                                                    |
| `payload`         | string \| null                                                                 | required for `amend`; for `answer` only on an elicitation paused before delta 21; forbidden otherwise                       |
| `answers`         | object \| null                                                                 | required for `answer` (except as above): question id → answer; forbidden otherwise                                          |
| `choice_id`       | string \| null                                                                 | required for `switch`: the decision's `choice_id`; forbidden otherwise                                                      |
| `option_id`       | string \| null                                                                 | required for `switch`: one of that decision's `options[].option_id`; forbidden otherwise                                    |
| `plan_revision`   | string \| null                                                                 | required for `switch`, optional for `approve`: the `revision` of the `plan.proposed` the user answered; forbidden otherwise |
| `client_features` | list of string                                                                 | optional; the question forms the page renders, as in 4.1.5; absent keeps the list                                           |

Each answer in `answers` is one of:

- `{"kind": "option", "option_id": "<id>"}`: an option the question offered.
- `{"kind": "options", "option_ids": ["<id>", ...]}`: one or more options a
  `multi_select` question offered, each once.
- `{"kind": "unit", "currency": "EUR", "prices": "constant", "base_year": 2023}`:
  the unit a `unit` question asked for; `base_year` is set exactly when
  `prices` is `"constant"`.
- `{"kind": "text", "text": "<the user's words>"}`: non-empty; offer it when
  the question sets `allow_free_text: true`.
- `{"kind": "skipped"}`: the user leaves this question to the agent. Only on
  a question that sets `skippable: true`; the turn still resumes as answered.

A body past 641,536 bytes gets a 413 `resource_exhausted` before it is parsed. An amendment (`payload`) or a `text` answer of more than 16,000 characters gets a 422 whose `detail` names the field with type `string_too_long`, and a `client_features` list of more than 16 values gets one with type `too_long`. Cap the composer at 16,000 characters when it amends a plan or answers a question.

```json
{
  "kind": "interrupt.response",
  "schema_version": 1,
  "conversation_id": "conv_01J9ZC3AB4YT9Q6WGX0S6H2K9E",
  "interrupt_id": "elc_7a1c0e5b42d94f18",
  "decision": "answer",
  "answers": {
    "q1": { "kind": "option", "option_id": "spec_4d5e6f7a33333333" },
    "q2": { "kind": "skipped" }
  }
}
```

Rules the server enforces with HTTP 422:

- `approve`, `amend`, `reject`, `switch` apply only to a `plan.proposed` pause.
- `switch` carries no `payload`. Its `option_id` must be one the decision
  offers as `available` or `partial`, other than its `active_option_id`; any
  other is refused with `constraint_id: "switch_option_unavailable"`. On a
  row carrying `unit`, a switch may name any unit the offer holds, with
  `unit` and the `option_id` naming it.
- Send `plan_revision` with every `approve`, set to the `revision` of the
  card the user approved. An `approve` without one approves whatever plan
  is paused, so an approval sent from an out-of-date card cannot be told
  apart.
- A `switch` or `approve` whose `plan_revision` is not the paused plan's is
  refused with HTTP 409 and `constraint_id: "plan_revision_stale"`. So is
  one naming an `interrupt_id` a later card replaced, when its
  `plan_revision` differs from that card's. The body is a ToolError plus
  `current_plan`: the `plan.proposed` frame the stream sent for the plan
  awaiting an answer, `seq` included. Replace the answered card with it,
  show `detail`, and let the user answer again; do not feed the frame to
  your `seq` ordering. `current_plan` is absent only for a pause stored
  before cards were kept. Nothing is approved or switched.
- `switch_option_unavailable` and `plan_revision_stale` both carry a
  `detail` to show.
- `answer`, `skip` apply only to an `elicitation.request` pause.
- `answers` names every question id of the frame exactly once, and an
  `option_id` must be one that question offered.
- `skip` is permitted when the frame sets `skippable: true`. It dismisses
  the whole pause and answers no question. When the orchestrator asked, the
  turn closes. When the planner asked, the skip counts as the user declining
  to choose, and the planner plans on without a pick of theirs.
- An elicitation that paused before the frame carried `questions` takes
  its one answer as `payload` (an option `id` or free text) instead of
  `answers`.

### 4.3 `composition.mutate` → POST `/api/studio/composition`

| Field               | Type                   | Notes                                            |
| ------------------- | ---------------------- | ------------------------------------------------ |
| `kind`              | `"composition.mutate"` | required                                         |
| `schema_version`    | `1`                    | required                                         |
| `conversation_id`   | string                 | required                                         |
| `composition_id`    | string \| null         | optional; omit to target the current composition |
| `expected_revision` | integer \| null        | optional; see below                              |
| `operations`        | array                  | required; at least one operation                 |

Each operation carries `op` plus only the fields for that op:

| `op`               | Fields                            | Effect                                                                                                                                                                        |
| ------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add`              | `component_id`                    | Appends a block for a component this conversation streamed. When no composition exists and `composition_id` is omitted, the server creates one that contains the added block. |
| `reorder`          | `order` (array of component ids)  | Puts the named blocks first, in the given order.                                                                                                                              |
| `remove`           | `component_id`                    | Removes the block. The component stays readable.                                                                                                                              |
| `pin` / `unpin`    | `component_id`                    | Sets the block's `pinned` flag.                                                                                                                                               |
| `set_layout_hints` | `layout_hints` (array \| null)    | Replaces the layout hints. Null clears them. Each hint is `{type: "row" \| "comparison", members: [component ids, 2 or more]}`.                                               |
| `set_card_meta`    | `component_id`, `meta` (required) | Sets client-owned card fields; see section 9.                                                                                                                                 |

Concurrency and atomicity:

- Send `expected_revision` to prevent lost updates. A stale value gets HTTP
  409 with the `edit_conflict` error, and nothing applies.
- Operations are atomic. If one operation is invalid, the server applies
  none of them.
- A request that changes nothing returns the current manifest and does not
  make a new revision.
- On success, the response body is the full `CompositionManifest` (new,
  or unchanged for a no-op). On failure, it is a ToolError (section 7).
- `/composition` applies while a turn streams on the same conversation,
  whether on this instance or another: it applies against the
  conversation's latest saved manifest, and a turn streaming at the same
  time reconciles its own composition changes on top of whatever
  `/composition` landed, so a canvas edit and a turn's own card both
  survive when they collide.

### 4.4 `turn.cancel` → POST `/api/studio/cancel`

Fields: `kind: "turn.cancel"`, `schema_version: 1`, `conversation_id`.
The server stops the conversation's active stream and returns HTTP 204.
The active stream may be running on a different server instance than the
one that answers this request; the server finds it there too and stops
it within about 5 seconds. The stopped stream ends with an `error` frame
whose code is `cancelled`. It is the stream's only ending frame, so a Stop
that lands as the turn pauses ends the stream on `cancelled` with no
`plan.proposed` or `elicitation.request` before it. A stream running a
plan-card `switch` is the exception: `/cancel` stops the switch, and the stream ends on a
`plan.proposed` re-offering the unchanged plan with a fresh
`interrupt_id` and `switch_failed.code` `switch_cancelled`. Keep reading
the stream after the call. A second `/cancel`, or 10 seconds with no
re-offered plan, ends it with the `cancelled` error frame. The call is
safe when nothing streams. A
closed connection stops the stream too and leaves the turn open where it
stopped, for `/resume` to carry on (§4.5). Proxies can hide a closed
connection. Use `/cancel` for the Stop button.

### 4.5 `turn.resume` → POST `/api/studio/resume`

Fields: `kind: "turn.resume"`, `schema_version: 1`, `conversation_id`, `turn_index` (the `turn.index` a state read returned, §4.6).

The server carries on the turn on a new stream, which you read as any other. Send it when the state read shows `status: "running"` and `lease_live: false`. The turn's last stream lost its client, a reload say, and the turn makes no progress until a client resumes it. The turn runs the step it stopped in again from its start, so the new stream carries no `activity.step` lines for earlier steps. A `plan.approved` the turn never sent goes out once. The stream opens with a `plan.approved` for each plan the turn already approved, since the page may have read none of them, so key them by `revision` and ignore one the page holds. The stream ends with `done`, a pause or an `error` frame.

| Case                                                                                            | Answer                                                                                    |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| the turn is open, running, and no stream holds it                                               | a new SSE stream                                                                          |
| the turn is paused on an answer recorded and not yet taken                                      | a new SSE stream carrying that answer on                                                  |
| the turn is paused on a question whose entry was never stored, or was stored without its frames | a new SSE stream sending the pause again                                                  |
| `turn_index` is not the latest turn, or the turn is closed                                      | 409 `edit_conflict`, `pointer: "/turn_index"`. Read state again                           |
| the turn is paused on a question with no answer recorded                                        | 409 `edit_conflict`. Answer it on `/interrupt`                                            |
| a stream holds the turn                                                                         | 409 `transient_unavailable` with `retry_after_seconds`                                    |
| a Stop is live for the turn                                                                     | the turn closes `cancelled`, and the resume answers 409 `edit_conflict`. Read state again |
| unknown `conversation_id`                                                                       | 404 `not_found`                                                                           |

### 4.6 GET `/api/studio/state`

Query: `conversation_id`, and `since`, the number of the last turn the page holds (0 for none). The answer is JSON. The call takes no lease, so it answers while a turn streams, and it changes nothing. An unknown `conversation_id` answers 404 `not_found`. A member with no value is left out.

- `turn`: `{index, status, trace_id, user_message, approved?, lease_live, retry_after_seconds?}`, absent before the first turn. `user_message` is the turn's prompt. Show it above a resumed reply when the page lacks the prompt. `approved` holds the `plan.approved` frame of each plan the open turn approved, one per `revision`. Attach any the page lacks to the turn's message. `status` is `running`, `paused`, `done`, `failed`, `cancelled` or `abandoned`. `retry_after_seconds` is set while `lease_live` is true.
- `closed_since`: the closed turns after `since`, oldest first, at most 3, each `{turn_index, user_message, reply, component_ids, cards, approved?, error?, new_question?}`. `approved` holds the `plan.approved` frame of each plan the turn approved, one per `revision`. `cards` holds each card the turn made, in `component_ids` order, at its current revision, `{component_id, type, title, revision, fact_summary?}`, so a card a later turn revised shows that revision. It holds no `payload` or `provenance`. Read the resolved revision from the presentation platform by `component_id`. `reply` is the text `message.complete` sent, empty when the turn closed with none. `error` is the code of a turn that did not end cleanly. `new_question` is set on a turn ended by a new question (§6), which the page sends as the next turn when it holds none after it.
- `closed_total`: how many closed turns the conversation holds, the index of the last one. While the page holds fewer, read again with `since` set to the last `turn_index` returned.
- `pause`: on a paused turn, `{interrupt_id, kind, frames}`. `kind` is `plan` or `elicitation`. `frames` are the frames the pause was sent as, its `plan.proposed` or `elicitation.request` last. Feed them to the same handlers as live frames.
- `manifest`: the composition manifest, absent when there is no canvas.

On page load, after drawing what the page kept, read the state, page through `closed_since` to `closed_total`, and act on `turn`:

| `turn.status`     | `lease_live` | What the page does                                                                                                                                                                         |
| ----------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `running`         | `false`      | send `turn.resume` with `turn.index`                                                                                                                                                       |
| `running`         | `true`       | another stream holds the turn. Read the state again after `retry_after_seconds`, until it changes                                                                                          |
| `paused`          | `false`      | show `pause` and answer it on `/interrupt`                                                                                                                                                 |
| any closed status | `false`      | append the turns of `closed_since` the page does not hold, and send a closed turn's `new_question` as the next turn, with `after_turn` set to that turn, when the page holds none after it |

A turn reads `running` while a stream still writes its pause or its close, and also when its answer was recorded and not yet taken or its pause's entry was never stored or holds no frames. `turn.resume` carries those on or sends the pause again. A closed turn's status is the one its record states, `failed` for any `error` code but `cancelled` and `abandoned`.

A turn whose stream ended on an `error` frame can still read `running` with `lease_live: false`. Its error was sent and the write that stores its record failed, which the server cannot tell from an instance that died mid-turn. Follow the read and send `turn.resume`. The new stream runs the turn on.

## 5. Frames

Every frame has `event` and `seq`. The other fields per type:

| `event`               | Fields                                                                                                                                                                                                             | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `message.delta`       | `text`                                                                                                                                                                                                             | Append this text to the streaming assistant message.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `message.complete`    | `text`                                                                                                                                                                                                             | The full final message for the turn. Replaces the deltas. A turn whose model drew cards and closed with no text sends none: the cards are the reply, and the turn's `reply` is empty.                                                                                                                                                                                                                                                                                                                                    |
| `plan.proposed`       | `interrupt_id`, `plan_version_id`, `version`, `title`, `steps[]`, `actions[]`, `decisions[]?`, `revision?`, `requirements[]?`, `operations[]?`, `moved[]?`, `summary?`, `switch_failed?`, `caveats[]`, `new_plan?` | The plan card. **Pauses the turn and ends the stream.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `plan.versioned`      | `plan_version_id`, `version`, `supersedes`, `changed[]?`                                                                                                                                                           | A new plan replaces the card `supersedes` names; a new `plan.proposed` follows, except that a new plan in the same turn (`plan.proposed.new_plan`) has no `plan.versioned` before it. `version` rises after an amend and is 1, which can be lower than the superseded version, when the plan was drafted again.                                                                                                                                                                                                          |
| `plan.approved`       | `plan_version_id`, `version`, `revision?`, `title`, `summary?`, `steps[]`, `choices[]`, `caveats[]`                                                                                                                | The plan the user approved. Sent once the run has accepted the `approve` answer. A retried answer whose approval the run already took, and a `turn.resume`, send it again as their first frame. Key it by `revision`, ignore one the page holds, and attach it to the message that stream produces.                                                                                                                                                                                                                      |
| `activity.step`       | `step_id`, `label`, `status`, `detail?`, `tool?`, `duration_ms?`, `produced?`, `progress?`                                                                                                                         | One execution step for the Activity feed. `step_id` is `s` plus the `seq` of the step's first frame, unique across the turn's segments; match a step's later frames by it, not by position. On the stream after an approval the plan steps run first; each plan step's final frame arrives in plan order, though steps that run at the same time may send their `running` frames in another order. Each plan step's `label` is that step's `label` on the plan card. A plan-step reference on this frame is a follow-up. |
| `component.created`   | `component_id`, `revision`, `type`, `catalog_version`, `min_catalog_version`, `payload`, `fact_summary`, `provenance`, `created_from?`, `unit_change?`, `replaces?`                                                | A new card exists; `unit_change` (4.1.4) when its unit can change; `replaces` names the card whose slot it took.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `component.revised`   | `component_id`, `revision`, `changed[]`, `fact_summary`, `payload?`                                                                                                                                                | An existing card has a new revision. When `payload` is absent, keep your cached payload and apply the `changed` fields from the new `fact_summary`/frame.                                                                                                                                                                                                                                                                                                                                                                |
| `composition.updated` | `manifest`                                                                                                                                                                                                         | The canvas changed. Re-render from this manifest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `elicitation.request` | `interrupt_id`, `questions[]` (1 to 3), `skippable`                                                                                                                                                                | One to three questions for the user. **Pauses the turn and ends the stream.**                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `suggestions`         | `items[]`                                                                                                                                                                                                          | Follow-up chips: `apply_choice` chips (4.1.2), or none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `error`               | `error` (a ToolError object)                                                                                                                                                                                       | The turn failed. **No `done` follows.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `done`                | `trace_id`, `composition?`, `new_question?`                                                                                                                                                                        | The turn is complete. `composition` is the conversation's current manifest, or absent when the conversation has no canvas. `new_question` is set when the user's `amend` text asked a different question. See §6.                                                                                                                                                                                                                                                                                                        |

Notes on specific fields:

- `message.*` text is GitHub-flavored Markdown. Render it with a
  sanitizing Markdown renderer; never inject it as raw HTML.
- `plan.proposed.steps[]`: `{id, label, description?, args?}`. Show
  `label`; show `description` when the user expands the step. `args`
  holds the tool arguments backing the step, for a client that wants to
  show more than the label.
- `plan.proposed.decisions[]`: `{id, question, chosen, reason, resolved_by, alternatives?, choice_id?, active_option_id?, options?}`, the choices the agent made for the user. A decision stating how Search counted a source carries the scope choice's `choice_id`, `"<requirement id>:<axis>"`, which stays the same on every plan of the conversation, switches included, its `active_option_id` and every option as `{option_id, label, status, reason?, years?}`, where `status` is `available`, `partial` or `unavailable`, `reason` says why a partial or unavailable option is limited, and `years` are the years a partial option covers. The decision stating how pinned sources' flows are counted ("How the flow is counted.") carries `choice_id` `"<requirement id>:count"`, its `active_option_id` and each count every source of the row offers as an `available` option whose `option_id` is derived from its label. Sources counted the same way share the row, and a switch of it moves every requirement on each of them to that count. A decision without `choice_id` is a planner decision. The row stating the unit a unit question's answer holds its parts to carries `choice_id` `"decision/user_unit_<spec_id>"` and `unit`, an offer shaped as in 4.1.4 whose `start` is the unit the parts are in now; its `options` are whole units such as `USD-constant-2023` "Constant 2023 US dollars", the start active. Show the unit picker on that row when the page renders unit questions, else its options. A switch of it sends `interrupt.response` with `decision: "switch"`, the row's `choice_id`, `plan_revision`, `unit` (the unit picked) and `option_id` naming that unit in the same form (`<currency>-current` or `<currency>-constant-<base year>`).
- `plan.proposed.revision` is a string identifying the plan's approval
  key, distinct from the integer `version`. It sits beside `version`: a
  client that shows a version badge keeps using `version`.
- `plan.proposed.actions[]` lists `switch` only when some decision has a `choice_id` and an option other than its `active_option_id` that is `available` or `partial`. Offer those options as the switch; show an `unavailable` one disabled, with its `reason`.
- `plan.proposed.moved[]`: `{choice_id, from_label, to_label, reason}`, set on the plan a switch offers. Each entry is another scope choice the switch moved, with the option it held, the one it holds now, and a sentence saying why.
- `plan.proposed.summary`: one plain sentence saying what is counted, for whom, when and from which source. `plan.proposed.steps[].gives_answer` is true on the one step whose result answers the question.
- `plan.proposed.switch_failed`: `{choice_id, option_id, code, reason, retryable}`, set on the plan re-offered after a switch that did not apply. Show `reason` on the row of the choice `choice_id` names. Offer the same switch again only when `retryable` is true. A switch the user stopped carries `code` `switch_cancelled`, `retryable` true and the reason "You stopped this change, so the plan is as it was." Show that reason as a quiet note on the row, with no "Could not switch" lead and no error styling.
- `plan.proposed.new_plan`: `true` on a plan card that starts a new plan after an earlier plan in the same turn; absent otherwise. A second plan call in one turn sends it on a card with a `plan_version_id` of its own: treat the card as a new plan, and drop the choice labels kept from the earlier one.
- `plan.proposed.caveats[]`: one sentence each for the plan's decisions that qualify the answer, such as a share whose two sides count different providers or a total derived from its part's dataset; empty when there are none. Show them on the card and in the full plan before Approve. `plan.approved.caveats[]` carries the same sentences once the plan is approved.
- `plan.approved.steps[]`: `{label, gives_answer?}`. `plan.approved.choices[]`: `{choice_id, question, chosen, changed_from?}`, each choice's id (the one the plan card's choice carries; `decision/<slot>` for a planner decision; absent from a `plan.approved` stored in a chat snapshot before it existed), short name, the option label it holds, and, when a switch of the turn changed it, the label it held on the card the first switch was made from. `plan.approved.caveats[]`: one sentence each for the plan's decisions that qualify the answer, such as a share whose two sides count different providers, as the approved card stated them; show them with the approved plan.
- `plan.proposed.requirements[]`: `{id, kind, sentence, decision_ids}`. It
  is the evidence behind the plan. `plan.proposed.operations[]`: `{id, kind, sentence, requirement_id?}`. It is the plan's computed steps.
  `sentence` on both is composed server-side and rendered as given, so
  the plan card and the model's own narration describe the same plan in
  the same words.
- `revision`, `requirements`, `operations`, and `steps[].args` are
  optional. A client that ignores them renders the same plan card it
  always has.
- `activity.step.status` is one of `pending`, `running`, `complete`,
  `failed`, `skipped`. A `failed` step is recoverable; the turn continues. A
  failed step can be followed by a `complete` frame on the same `step_id`
  when a later try succeeds, with `detail` "This worked on a later try."
  `detail` holds a one-line result on success in plain words: a plan's
  fetch step carries the retrieval's summary (observations, years,
  source, series), as a fetch does; a plan's analysis step and a card's
  step (labelled with the card's kind and title) carry none. A tool
  call's failed step, a plan step's, an agent tool's or an edit's, carries
  none: a refusal's code and message are written for the agent and stay in
  the trace. A failed "Receive the datasets from Search" step says in plain
  words that the datasets the user brought could not be used, a "Making N
  charts" step keeps its `detail` (below), and a step still running when a
  turn is stopped or fails closes `failed` with a plain reason such as
  "Stopped before this step finished". `produced` (`{component_id, revision}`)
  links a step to its card. In a data turn, the agent's reply says why a
  card it asked for was not drawn. An edit turn whose write is refused
  closes with "The edit was refused: <message>", quoting the refusal's own
  message.
- An `activity.step` with no `tool` is a status line: what is happening
  now, in the present tense. It opens `running` and later gets a
  `complete` frame with `duration_ms` on the same `step_id`. One status
  line runs at a time, and one runs whenever an agent is working and no
  tool step is running. Show the label of the
  most recently opened step that is still `running`; a `complete` or
  `failed` frame updates the Activity record only. Fold status lines in
  the Activity record.
- The planner's status lines name its work: "Working on a plan" (from the
  hand-off to the planner until its first draft) or "Revising the plan"
  (from the hand-off of an amendment), "Checking the plan against the
  data" (from each proposal the planner makes until the proposal is
  answered), "Reworking the draft" (a draft after a refused proposal;
  `detail` gives the reason in plain words when there is one), and, once
  a plan is approved, "Starting the plan" until its first step's frame. The checking line's label is the
  same for every plan. Its `complete` frame carries a `detail` naming the
  data of the plan the proposal recorded, such as "health spending per
  person for Kenya and Uganda and 1 more" (the first dataset, then the
  count of the others), and no `detail` when that plan has no dataset.
  The Activity feed keeps a line's first label and shows a later frame's
  `detail` as its summary.
- A plan step's line opens when the step starts. Each `running` frame of
  a plan step carries the whole running `detail`: a progress message when there is one (dataset preparation, a retry) then `step i of n`, as in "Preparing the dataset (queued, 11s) · step 5 of 10", else `step i of n` then the step's description from the plan card (without its closing full stop), joined with " · ". `i` is the step's row on the plan card and `n` the number of rows. The message that says only that the call returned counts as none, so the description returns. A step the card does not show (a rolling copy folded into another row) carries its progress message alone. A step whose line is still `running` when the plan's run hands back, with nothing recorded for it, gets a `failed` frame then, whose `detail` says why in plain words.
- Every running `detail` the harness composes for these lines is at most 110 characters: a plan step's, the asks on "Working on a plan", the charts on "Making N charts" and "Going over the charts", the words on "Searching for data", and the step label on "Reading the plan's results". A longer one is cut at a word and ends in "…"; a first word longer than the limit is cut inside the word. The chat's status line shows about 43 characters of it, so each detail leads with its point. The limits a reading line states (below) are summaries of results and can run longer.
- While a skill read runs, the open status line gets a `running` frame
  whose `detail` is "reading a method note", for the agent's reads and the
  planner's alike. When the read is answered the line gets a `running`
  frame with its own `detail` back, an empty string when it opened with
  none.
- The orchestrator agent's status lines are "Reading your question",
  "Reading your answer", and, after each batch of tool results, a line
  naming what the model is reading: "Reading what the search found",
  "Looking at <measure>, <first>–<last>" after one fetch ("Looking at
  <measure>" when the retrieval states no years; "Looking at the data"
  after several fetches, or after one that failed or whose measure no
  search named), "Going over the charts" (`detail`: what the running "Making N charts" step
  carried, below, taken from a call that drew its charts; no `detail` when every call failed), "Reading the earlier turn",
  "Going over your canvas" after the agent looked at the canvas (its tool
  step reads "Looking at your canvas"), "Going over the notes on DFO", "Working out the next step" after a hand-off
  to the planner or a question to the user that was refused, and
  "Reading the plan's results" from the moment an approved plan finishes
  running, with the plan's label for the step that gives the answer as `detail`, absent when the plan gave that step no label or the step was refused or failed. A batch that mixes tools is named by its latest stage (charts,
  then fetch, then search, then recall). A successful plan hand-off opens
  "Working on a plan", whose `detail` is what the plan is for: the first
  output the agent's asks name, then `· +k more` for the others. A successful ask hands over to the question, and
  neither opens a reading line. A batch of background-note reads keeps the
  line before it running. `detail` on these lines carries the limits the
  results state, in plain words: for a search, each found measure's years, the parts with more than one
  reading and the parts with no data; for a fetch, the retrieval's
  summary (observations, years, series), with each measure named when
  there are several. The line completes when the next thing happens: a
  tool call, answer text, a hand-off, or the end of the stream. A plan
  hand-back that carries no plan shows a `failed` "No plan could be
  made", then "Reading why no plan was made"; the closing message
  then says what could not be calculated. No model thought text is sent.
- The agent shows one step per tool call: "Searching for data" (detail: the user's words when the agent searched with `text`, otherwise the measures it asked for, then, on the `complete` frame, what it found, named by the reading each measure resolved to, and what it did not find; on the `running` frame alone the detail is led by `again: ` when the search sends the user's words for one part whose result earlier on the same stream carried `retry_with_text`),
  "Fetching <measure>" (detail: the retrieval's summary), "Making N
  charts" (detail while running: the first chart title, then `· +k more`
  for the other charts, led by `again: ` when it redraws charts a
  refusal asked to correct earlier on the same stream; a redraw after the turn paused for a question to the user shows no `again: `), and one step per card written ("Line chart: <title>"). The
  calls a fetch makes on its own behalf (retrieving and checking the
  data), skill reads and the hand-off to the planner are not shown when
  they succeed; each still shows as a `failed` step when it fails. A
  "Making N charts" step is `failed` when any chart it asked for was not
  drawn, with `detail` naming how many and the titles of the refused
  ones; the cards it did draw
  still arrive.
- `component.created.payload` is the platform's resolved revision, `{schema_version, catalog_version, component_id, revision, provenance, component}`, as dfo-spec-kit's envelope schema defines it. `payload.component` holds everything the card draws, points included (see "Point data" below), and its title at `payload.component.title.text` (a note or a source card has none). `payload.provenance` is the platform's provenance for this revision and is authoritative for the card's back. Take the headline values and attribution from `fact_summary`.

**Point data.** The points come in `payload.component`: `series[].points[]` on a line or bar chart, each `{period, value}`, or `{period, status}` where no value is published, `rows[]` on a table, and `value` with `period` on a stat card. Plotting a card needs no second request.

A client that wants the raw frame behind a card takes its reference from `payload.provenance.inputs[0]` = `{spec_id, content_version}` (a note has none) and fetches it from the datasets door:

1. Send `GET /api/datasets/{spec_id}/{content_version}` to `apis`.
   How your app reaches `apis` (direct, through your proxy, or another
   way) is your decision; the reference pages already talk to the same
   service.
1. Read `artifacts[0].url` from the response. It is a root-relative link
   that does not expire. Resolve it against the `apis` origin.
1. Fetch that link and follow its 302 redirect to the file. The redirect
   target is a signed URL that lasts five minutes, so follow it at once.
   The body is a Parquet file. Each row is one point:
   `{stat_var, observationAbout, time_period, value, unit, measurementMethod, provenance, status}`.

The frame can hold more rows than the card draws. `payload.component` holds exactly the drawn points. The platform mints the file, and every row carries its provenance.

Known limit: the datasets door resolves `spec_` frames only. A card whose
frame `spec_id` starts with `ops_` rests on an analysis result (a
filter, join, growth rate, per-capita figure, or any other analysis step).
The frame is published to the platform, but
`GET /api/datasets/{spec_id}/{content_version}` answers 404 `not_found` for
it, so its raw frame is not served. The card's own points are in `payload.component` either way.

- `fact_summary` holds `component_id`, `revision`, `series[]` (`{id, label, point_count, coordinates?, first?, first_period?, last?, last_period?, change?}`; `first_period` and `last_period` are the periods of the first and last drawn points that hold a value; each is absent when its point carries no period, and both are absent for a stat card or table, which draw no series), `attribution` (`{label, source_count, indicator_count, source_detail_url?}`), `window?`, `unit?` (the unit slot of the card's payload, repeated as the platform states it, with its shape below),
  `series_truncated?`, `countries?`, `defaults?` (`[{pointer, detail}]`: the choices the platform made where the card's intent left one open, each naming the intent field it filled by JSON Pointer; absent means it made none). By design, the numbers in the message
  text come from fact summaries; the models are constrained to them. A chart's `window` is the one the
  platform states, or else the periods its axis draws: a bar chart of
  entities at one period has that period as both `start` and `end`. A
  table's and a stat card's `window` is null, and `series[].change` is
  null on every card. A table draws without either field.
- `fact_summary.unit` is the unit slot of the card's payload, in the payload's shape, so one reader serves both: a stat card's `unit`, a chart's first series' `unit`, and a table's row `unit` when every row has the same one. It is absent when the platform states none, as for a table whose rows differ in unit, a note, and a source card. Otherwise it is one of two shapes:
  - a string of 1 to 200 characters, the unit's name, or its reference where the graph names none: `"US Dollar"`.
  - the platform's unit object, forwarded with every member it states and read by no check here: `{"currency": "USD", "prices": "constant", "base_year": 2023}`. The Presentation contract owns the object's members, so a reader takes the ones it knows and ignores the rest, and the object may be empty.
- `provenance` holds `inputs[]`, `sources[]`, `freshness`, `warnings[]`,
  `generated_at`, `citation`, `execution_ref?`, `refreshed_from?`. It is
  complete on the frame; no second fetch is needed to show source
  information. A card built from one retrieval lists its own frame as the
  one input. A card built from several (a join, a compare, a share of two
  datasets) lists its frame first, then each retrieval it rests on.
  `sources[]` then holds one entry per distinct source, `citation` joins
  the distinct citations with "; ", `freshness.last_refreshed` is the
  oldest input's date, `freshness.data_through` is the earliest input's end
  year, and each warning carries the `spec_id` of the retrieval it came
  from. `last_refreshed` is `""` when any input's creation date is missing
  or unreadable, and `data_through` is `""` when any input's end year is
  missing or not a whole number. A card built from one retrieval with no
  creation date gets today's date (`YYYY-MM-DD`) as `last_refreshed`, and
  its `data_through` is `""` when that retrieval has no end year. A card
  whose retrievals cannot all be identified is withheld.
- `elicitation.request.questions[]` holds one to three questions, each
  `{id, question, input, options[]?, allow_free_text, skippable, rationale?}`.
  Answer each under its `id`. Option ids are scoped to their question, so
  two questions may both offer `o1`. One question comes with a frame-level
  `skippable: true` and question-level `skippable: false`; several come
  with the frame-level flag off and each question skippable on its own.
  An option with `recommended: true` is the planner's pick, and the
  question's `rationale` says why.
- A question's `input` is `single_select`, `multi_select`, `unit`, or `text`.
  A `single_select` answer is one option or one text. A `multi_select`
  question has `options` and `allow_free_text: true`; its answer is
  `{"kind": "options", "option_ids"}`, or text; a single `option` answer to
  it is refused with 422. The agent asks one when several answers can hold
  together, with labels only. A `unit` question has no
  `options` and no free text, and carries `unit`, an offer
  `{currencies[], presets[], start}` shaped as in 4.1.4; its answer is
  `{"kind": "unit", ...}`. Any question that is `skippable` takes
  `{"kind": "skipped"}`; skipping a unit question goes on with the unit the
  offer starts on (`start`). A page that does not declare
  `"unit_question"` or `"multi_select"` in `client_features` gets those
  questions as `single_select` (4.1.5).

## 6. Pauses and resumes

A turn can pause more than once. The agent pauses to ask at most twice a
turn, one to three questions each time, then goes on with what it has. Plan reviews
recur after each amendment, with no fixed count. Follow this procedure in
a loop:

1. Read frames until the stream ends.
1. If the last frame is `plan.proposed`: show the plan. Send
   `interrupt.response` with `approve`, `amend` (payload = the user's
   change request), or `reject`.
1. If the last frame is `elicitation.request`: show its questions. Send
   one `answer` with `answers` for every question, or `skip` when the
   frame is `skippable`.
1. The `/interrupt` response is a new stream that continues the same turn.
   Read it the same way.
1. After `amend`, the new stream sends `plan.versioned` and then a new
   `plan.proposed` with a new `interrupt_id`. Respond again.
1. When the harness cannot read the plan you approved or amended, it
   plans the question again. The new stream sends a `message.delta` and
   `message.complete` saying so, then a `plan.versioned` with `version`
   1 whose `supersedes` names the card shown, then a new
   `plan.proposed`. Replace the card from that `plan.proposed`, and
   accept a `version` lower than the one it supersedes. When no plan can
   be drafted, the stream sends `message.complete` and `done`.
1. After `switch`, the new stream sends `plan.versioned` and then a new
   `plan.proposed` with a new `interrupt_id` and `revision`, the switched
   decision now `resolved_by: "user"`, and `moved` listing any other choice
   that moved with it. A switch approves nothing: answer the new card. When
   the switch cannot apply, the stream sends the same plan again under a
   new `interrupt_id`, with `switch_failed` saying why on the row. A
   `/cancel` while the switch runs ends the stream the same way, under
   `switch_cancelled`.
1. `reject` ends the turn. The stream sends `message.complete` and `done`.
   Nothing was computed.
1. When the planner reads the `amend` text as a new question, the turn ends with the plan set aside unapproved. The stream sends a `message.complete` saying so, then `done` with `new_question`, the user's latest words as sent (the `amend` text, or the text answering a question the planner asked about it), and no new card. Close the card as "Taken as a new question" and send `new_question` as a new `turn.request` with `after_turn` set to the closed turn's index (4.1.3). An answer to the closed card's `interrupt_id` gets 404. When the user's words ask for a new question without stating it, `message.complete` reads "The plan was set aside. What would you like to ask?" and `done` carries no `new_question`: close the card the same way and wait for the user's next message.

A new `turn.request` on a paused conversation abandons the pause.

Retry an `interrupt.response` you are unsure landed: the server records
the answer before it starts resuming, so retrying the identical body is
safe. While the resumed stream from your first attempt is still running,
on any instance, the retry answers 409 `transient_unavailable` with
`retry_after_seconds`; once that stream is gone, the retry resumes the
turn again on a fresh stream, whose `seq` starts in a block of 2^20 numbers
after every block the dead stream reserved, so above anything it sent.
Retrying with a different
decision, payload or answers than the one that actually resumed answers 409
`edit_conflict`.

## 7. Errors

The `error` field of an `error` frame, and the body of every non-200
response FROM THE STUDIO ROUTES' OWN CHECKS (404, 409, and the 422s in
section 4), is a ToolError: `{error, retry, detail, constraint_id?, pointer?, fix?, retry_after_seconds?}`; the 409 `plan_revision_stale` body adds `current_plan` (section 4.2). Two framework responses have a
different shape: 401 is `{"detail": "..."}`, and a request body that fails
schema validation gets a 422 with `{"detail": [...]}` (FastAPI's format).

- `error` codes: `schema_invalid`, `binding_invalid`,
  `constraint_violation`, `unsupported_component_type`,
  `unsupported_version`, `edit_conflict`, `not_found`, `resource_exhausted`,
  `transient_unavailable`, `rate_limited`, `cancelled`,
  `capability_unavailable`.
- `retry` is one of `correct_and_retry`, `refresh_then_retry`,
  `retry_after_delay`, `do_not_retry`. Obey it. Offer a retry button only
  for `retry_after_delay`.
- A step-scoped failure is an `activity.step` with `status: "failed"`, and
  the turn continues. A top-level `error` frame ends the turn.
- `detail` is text you can show. The `done` frame carries the turn's
  `trace_id`; error `detail` text names the trace id when the failure is
  server-side. Show it so users can report it.
- `not_found` with `pointer: "/conversation_id"` is what `/interrupt` and
  `/composition` answer for a `conversation_id` the server has never seen,
  and what `/turn` answers when its body carries `component_ref` for one
  (a card to edit or follow up on needs a conversation the server already
  holds). A plain `/turn` with no `component_ref` starts a new
  conversation when the `conversation_id` is one the server has not seen.
  `/cancel` on an unknown or idle conversation is a no-op 204.
- A second `/turn` sent while one is already streaming on the same
  conversation, on any instance, answers 409 `transient_unavailable` with
  `retry_after_seconds` set to the remaining lease. Wait for the first to
  finish, or send `/cancel`.
- A `/turn` whose `after_turn` is not the conversation's latest turn
  answers 409 `edit_conflict` with `constraint_id: "turn_moved"` and
  `pointer: "/after_turn"` (4.1.3).
- A running stream can also end with a stream-fatal `error` frame carrying
  `transient_unavailable` when the instance serving it stops renewing its
  claim (a crash, or a slow write) and another instance has since taken
  the conversation over. The turn continues there; the taking-over
  instance's own stream, if you have one open, is what finishes it.

## 8. Card edits, follow-ups, and canvas questions

Set `component_ref` on a `turn.request` to scope the turn to one card:
`{component_id, revision, intent}` with `intent` = `"edit"` or
`"follow_up"`.

**Edit** (`intent: "edit"`, message = the user's edit instruction, or the
Edit Card panel's "Describe your edits" text):

- The turn does no search and makes no plan.
- For a title or data-selection change, the stream sends two
  `activity.step` frames, `component.revised`, `composition.updated` (only
  when the card is on the canvas), `message.complete`, `suggestions`,
  `done`.
- For a type or window change, the platform cannot revise the card. The
  stream sends `component.created` for a replacement card, with `replaces`
  naming the card whose slot it took when that card was on the canvas. The
  new block takes the replaced block's slot, pin and `meta` on the canvas;
  map the old card to the new one in place. The replaced component stays
  readable at its earlier revisions.
- An impossible edit ends with an honest `message.complete` and no
  component frame.
- To implement quick type-switch icons, send the same edit request with a
  fixed message, for example "turn it into a bar chart".

**Follow-up** (`intent: "follow_up"`): the agent gets the card's summary as
context for a normal data turn, and is told the question is about that
card. The agent reads the card's latest revision. A `component_id` this conversation never held answers 404
`not_found` with `pointer: "/component_ref/component_id"`; a card inherited
by a copy of someone else's conversation is one of these.

**Canvas questions**: a message like "what is on my canvas?" is answered
from the canvas state without a data search. An empty canvas gets the
answer "nothing yet".

**Cross-turn memory**: the agent remembers two things. It remembers the
canvas (card titles and fact summaries, for the last 16 blocks in canvas order): "now
the same for Uganda" works after a Kenya card. It also remembers the
recent conversation (the newest 10 exchanges: what you asked, what it
answered, which cards resulted): "show life expectancy for the country I
mentioned" resolves to the country you actually named earlier, even when
that earlier turn made no card. For references beyond those windows the
agent can look up older turns and full card details on its own. All of
this is server-side; the client sends only the normal `turn.request`.

## 9. The canvas

The server owns the composition. Hold no layout state on the client.
Re-render from every `composition.updated` frame and from every
`/composition` response.

The manifest: `{composition_id, revision, actor, blocks[], counts, layout_hints?}`. Each block: `{component_id, revision, type, title, pinned?, meta?}`. `counts` is `{all, stats, charts, tables, notes}`, the numbers for the
filter chips. `actor` says who made the revision: `model` or `user`.

**Auto-add**: with `auto_add: true` (default), each created card joins the
canvas at once. With `auto_add: false`, cards stream as `component.created`
but do not join the canvas. Add one later with the `add` operation. Use
this for the Auto-add toggle and the "Added" chip.

**Removal by the agent**: when the user asks in chat to remove cards or
clear the canvas, the agent takes the blocks off, and a
`composition.updated` frame carries the manifest without them. The removed
components stay readable. A model turn's manifest can therefore shrink or
be empty.

**Card meta** (`set_card_meta`): `meta` = `{subtitle?, description?, x_label?, y_label?, style?}`. The server stores these fields and returns
them on the block. The server never interprets them, and the platform never
sees them. `style` is a free object for client-defined settings, for
example `{"show_area_fill": true}`.

## 10. Component types

`type` is one of: `stat_card`, `line_chart`, `bar_chart`, `scatter_chart`,
`table`, `note_user`, `note_narration`, `source_card`. This set is closed
in v1; a new type arrives only with a new contract version. Render an
unknown type as an unsupported-block state so that day is safe.
`catalog_version` names the platform catalog release the payload comes
from (`min_catalog_version` currently equals it).

## 11. Known limits in this version

- `suggestions.items` holds only `apply_choice` chips (4.1.2).
- Some searches refuse with `capability_unavailable` ("identity does not
  reproduce spec_id"). This is a known upstream data gap. The search step
  shows `failed` and the agent answers without that data; a retry will not
  help.
- Cross-turn memory is bounded: the newest 16 canvas blocks and the
  newest 10 exchanges (older context is looked up on demand, which can
  miss). Very long conversations keep working from that window.
- On a deployment running more than one server instance, editing or
  following up on a card is reliable only when the request lands on the
  same instance that created the card. The conversation's other state
  (plan approvals, elicitation answers, plain data turns) works from any
  instance.

## 12. Prompt gallery and test material

Each row is a real prompt and the behavior it causes. Every frame
validates against the models in `services/harness/src/harness/wire.py`,
and `services/harness/tests/wire_fixtures/studio/` holds one static
example of each frame and request.

| Prompt or request                                                                                  | Behavior                                                                       |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| "life expectancy in Nigeria"                                                                       | Direct data turn: steps → card → message → done                                |
| "give me a comprehensive overview of health spending in Kenya"                                     | Plan pause → approve on `/interrupt` → card                                    |
| Same plan, `decision: "amend"`, payload "focus on the change since 2015"                           | `plan.versioned` → new `plan.proposed` → approve → card                        |
| Same plan, `decision: "reject"`                                                                    | `message.complete` → `done`; nothing computed                                  |
| Turn 1 "life expectancy in Kenya"; turn 2 "now the same for Uganda"; turn 3 "what's on my canvas?" | Canvas memory across turns; a canvas question with no data search              |
| `component_ref` edit: "rename it to …", then "turn it into a bar chart"                            | `component.revised`, then a replacement `component.created`                    |
| "average annual rainfall in Middle Earth"                                                          | No data: honest `message.complete` → `done`, no card                           |
| Any turn, then POST `/cancel`                                                                      | Stream ends with the `cancelled` error frame                                   |
| A plan-card `switch`, then POST `/cancel` while it runs                                            | The same plan re-offered, `switch_failed.code` `switch_cancelled`              |
| "quickly, life expectancy in Japan" with `auto_add: false`, then an `add` operation                | Cards stream without joining the canvas; the `add` response holds the manifest |
| An ambiguous question that resolves to several readings                                            | `elicitation.request` pause; answer, free text, or skip                        |

A real turn takes 15 to 60 seconds and several model calls. Card meta,
pin, reorder, remove, edit conflict and malformed requests are plain HTTP
request → response cycles; build them from sections 4, 7, and 9.
