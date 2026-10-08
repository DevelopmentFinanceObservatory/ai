# DFO Studio: the agent interface for the frontend

This document tells you how to connect the Studio frontend to the agent
service. It is the one document you need. The source of truth for every
shape is `services/harness/src/harness/wire.py`. Recorded example streams
are in `docs/transcripts/`. This document agrees with both.

## Start here

1. Get the repo and install: `uv sync --all-packages` (Python 3.13, uv).

1. Start the deterministic test server (no keys, no network):
   `just studio-scripted`

1. Send a turn and watch the stream:

   ```sh
   curl -N -X POST http://localhost:8080/api/studio/turn \
     -H 'Authorization: Bearer dev' -H 'Content-Type: application/json' \
     -d '{"kind":"turn.request","schema_version":1,
          "conversation_id":"conv_1","message":"life expectancy in Kenya"}'
   ```

   You get `data:` lines: steps, a card, the message, `done`. Render them
   per section 5.

1. Build each UI state against the recordings in `docs/transcripts/` and
   the prompt gallery (section 12). The scripted server reproduces every
   state on demand, including pauses and errors.

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
  origins listed in `STUDIO_CORS_ORIGINS` (comma-separated). The scripted
  test server (below) allows all origins
  so a local page can talk to it directly.

## 2. Endpoints

| Method | Path                      | Response                                         |
| ------ | ------------------------- | ------------------------------------------------ |
| POST   | `/api/studio/turn`        | SSE stream of frames                             |
| POST   | `/api/studio/interrupt`   | SSE stream of frames (continues a paused turn)   |
| POST   | `/api/studio/composition` | JSON: the composition manifest after the request |
| POST   | `/api/studio/cancel`      | HTTP 204, no body                                |
| GET    | `/healthz`                | JSON liveness                                    |

## 3. The stream

The server answers `/turn` and `/interrupt` with Server-Sent Events.

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
  lease's remaining time. `/cancel` and `/composition` are not blocked by
  a stream. To stop a stream, use `/cancel` or close the connection;
  `/cancel` reaches a stream running on another instance within about 5
  seconds.

## 4. Requests

### 4.1 `turn.request` → POST `/api/studio/turn`

| Field             | Type                                    | Notes                                                           |
| ----------------- | --------------------------------------- | --------------------------------------------------------------- |
| `kind`            | `"turn.request"`                        | required                                                        |
| `schema_version`  | `1`                                     | required                                                        |
| `conversation_id` | string                                  | required (`session_id` is a deprecated alias; do not send both) |
| `message`         | string                                  | required; the user's text                                       |
| `surface`         | `"studio"` \| `"download"` \| `"embed"` | optional; default `"studio"`                                    |
| `component_ref`   | object \| null                          | optional; see section 8                                         |
| `seed_spec`       | list of `Seed` \| null                  | optional; 1 to 6 Search results to start from; see 4.1.1        |
| `auto_add`        | boolean                                 | optional; default `true`; see section 9                         |

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
`component_ref`.

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

| Member of `spec` or of Search's answer                             | Bound                     |
| ------------------------------------------------------------------ | ------------------------- |
| `resolved.rendered_sentence`                                       | 2,000 characters          |
| `resolved.entities`                                                | 1,024 items               |
| the roles of `definition.identity.entities`                        | 16                        |
| `resolved.slots`, `coverage.dimensions`, `role_axis_facts`         | 64 items each             |
| any other string the harness reads (a dcid, a role, a label)       | 256 characters            |
| any count or year the harness reads                                | a whole number, 12 digits |
| the sentence, measure and entities together, as a model reads them | 65,536 characters         |
| the record the turn keeps of the seed, identity included           | 131,072 characters        |

**What a seeded turn shows before approval.** The harness resolves each
usable seed's identity again through the Search API, all seeds at once,
stores the definition Search answers with in data-mcp's definition
store, and retrieves nothing yet. Every label, sentence, count and
headline the turn shows comes from that answer, never from the block
the client posted. The Activity feed shows one "Receive the datasets
from Search" step counting the seeds received. A seed dropped or not
reached shows its own failed `seed-<n>` step. No retrieval in the turn
takes a dropped seed's dataset (below), and the turn's next words name
it as `dataset <n> from Search` (its position in `seed_spec`, from 1)
with Search's reason. Those words
are a `message.delta` before the plan card, the `question` of an
`elicitation.request` the turn pauses on, or the closing text when no
plan is drafted. The turn then plans from the kept seeds without
searching, and the plan card follows as on any planned turn. When no
seed is kept, the turn continues from the question alone, as a turn
without seeds does, except that a reading of a dropped dataset is left
out of the question's search: the turn asks between the other readings,
or closes saying the question resolves to a dataset it left out.

**Planning.** A turn whose seeds carry an operation always plans. A
turn with one seed and no operation takes the same depth policy as a
searched answer: a seed that names one series goes straight to
`get_data`, and one that names a family, such as aid to Kenya from
every donor, is planned. The planner pins each seed and chooses the
years to retrieve, and the plan retrieves the seed over those years.
One retrieval returns at most data-mcp's `MAX_FRAME_ROWS` rows, so the
planner asks for the fewest years the question needs: one year for a
rank across places or donors, and the range a growth runs over.
Education aid to Kenya is 242 rows in 2024 and 3,780 over every year. A
retrieval past the ceiling shows a failed step, and the planner replans
with a narrower window and the seed still pinned. An operation over
`place`, `origin` or `destination` runs across every member of the axis
the seed leaves open, and a rank that asks for a top or bottom N keeps
that many rows. A component draws at most 24 series. When a turn goes
straight to `get_data` with a seed whose question named no year, it
retrieves the year Search showed.

**The download surface.** A `download` turn retrieves one dataset and
computes nothing. It retrieves the first usable seed. Every other
usable seed shows a failed `seed-<n>` step with code
`one_dataset_per_download`. When the retrieved seed carries operations,
a failed `seed-<n>-operations` step with code
`no_operations_on_download` names each one, and the data is still
retrieved. A question asking for a per-head figure that no series
holds closes with a refusal, since the division is a computation.

**A dropped seed.** The turn drops a seed when Search's answer for its
identity is not the dataset the user handed over: Search refuses the
identity, resolves it to something other than one dataset, resolves it
to another `spec_id`, or answers with a block the harness cannot read.
The turn checks each seed once, as it receives the seeds and before it
plans or retrieves anything. The seed's `seed-<n>` step fails with code
`seed_dropped` and carries the reason.

On a planned turn the dropped seed's name and the reason arrive in a
`message.delta` frame, and its `message.complete`, before
`plan.proposed`, with a request to search for the dataset again. The
plan card has no field for the drop. The card's `depth` and
`depth_notice` describe the kept seeds.

For the rest of the turn no plan retrieves the dropped dataset, pinned
or through a new search, over the seed's years or any others. A kept
seed over the same dataset and other years stays usable.

A dropped seed stops the turn in two cases:

- the planner drafts no plan from the kept seeds. The closing text names
  the dropped seed, the seeds that were not planned, and the planner's
  reason, and the turn draws no card;
- no seed is kept and the question's own search resolves only to a
  dropped dataset. The closing text names the dropped seed and says the
  question resolves to a dataset the turn left out.

### 4.2 `interrupt.response` → POST `/api/studio/interrupt`

| Field             | Type                                                             | Notes                                                                 |
| ----------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| `kind`            | `"interrupt.response"`                                           | required                                                              |
| `schema_version`  | `1`                                                              | required                                                              |
| `conversation_id` | string                                                           | required                                                              |
| `interrupt_id`    | string                                                           | required; copy it from the pausing frame; it is opaque and single-use |
| `decision`        | `"approve"` \| `"amend"` \| `"reject"` \| `"answer"` \| `"skip"` | required                                                              |
| `payload`         | string \| null                                                   | required for `amend` and `answer`; forbidden for the others           |

Rules the server enforces with HTTP 422:

- `approve`, `amend`, `reject` apply only to a `plan.proposed` pause.
- `answer`, `skip` apply only to an `elicitation.request` pause.
- For `answer`, send an option `id`, or free text when the elicitation set
  `allow_free_text: true`.
- `skip` is permitted when the elicitation set `skippable: true`. The agent
  then continues with its default reading.

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
whose code is `cancelled`. The call is safe when nothing streams. A
closed connection also stops the turn, but proxies can hide a closed
connection. Use `/cancel` for the Stop button.

## 5. Frames

Every frame has `event` and `seq`. The other fields per type:

| `event`               | Fields                                                                                                                                                                    | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `message.delta`       | `text`                                                                                                                                                                    | Append this text to the streaming assistant message.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `message.complete`    | `text`                                                                                                                                                                    | The full final message for the turn. Replaces the deltas.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `plan.proposed`       | `interrupt_id`, `plan_version_id`, `version`, `title`, `steps[]`, `actions[]`, `depth?`, `depth_notice?`, `decisions[]?`, `revision?`, `requirements[]?`, `operations[]?` | The plan card. **Pauses the turn and ends the stream.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `plan.versioned`      | `plan_version_id`, `version`, `supersedes`, `changed[]?`                                                                                                                  | A new plan replaces the card `supersedes` names; a new `plan.proposed` follows. `version` rises after an amend and is 1, which can be lower than the superseded version, when the plan was drafted again.                                                                                                                                                                                                                                                                                                                |
| `activity.step`       | `step_id`, `label`, `status`, `detail?`, `tool?`, `duration_ms?`, `produced?`, `progress?`                                                                                | One execution step for the Activity feed. `step_id` is `s` plus the `seq` of the step's first frame, unique across the turn's segments; match a step's later frames by it, not by position. On the stream after an approval the plan steps run first; each plan step's final frame arrives in plan order, though steps that run at the same time may send their `running` frames in another order. Each plan step's `label` is that step's `label` on the plan card. A plan-step reference on this frame is a follow-up. |
| `component.created`   | `component_id`, `revision`, `type`, `catalog_version`, `min_catalog_version`, `payload`, `fact_summary`, `provenance`, `created_from?`                                    | A new card exists.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `component.revised`   | `component_id`, `revision`, `changed[]`, `fact_summary`, `payload?`                                                                                                       | An existing card has a new revision. When `payload` is absent, keep your cached payload and apply the `changed` fields from the new `fact_summary`/frame.                                                                                                                                                                                                                                                                                                                                                                |
| `composition.updated` | `manifest`                                                                                                                                                                | The canvas changed. Re-render from this manifest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `elicitation.request` | `interrupt_id`, `question`, `input`, `options[]?`, `allow_free_text`, `skippable`, `progress?`                                                                            | A question for the user. **Pauses the turn and ends the stream.**                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `suggestions`         | `items[]`                                                                                                                                                                 | Follow-up chips. The list is empty in v1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `error`               | `error` (a ToolError object)                                                                                                                                              | The turn failed. **No `done` follows.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `done`                | `trace_id`, `composition?`                                                                                                                                                | The turn is complete. `composition` is the conversation's current manifest, or absent when the conversation has no canvas.                                                                                                                                                                                                                                                                                                                                                                                               |

Notes on specific fields:

- `message.*` text is GitHub-flavored Markdown. Render it with a
  sanitizing Markdown renderer; never inject it as raw HTML.
- `plan.proposed.steps[]`: `{id, label, description?, args?}`. Show
  `label`; show `description` when the user expands the step. `args`
  holds the tool arguments backing the step, for a client that wants to
  show more than the label.
- `plan.proposed.depth` says how much dialogue the plan may still ask the
  user for before it runs: `shallow` allows at most one clarifying question,
  `standard` up to two, and `deep` up to three with no silent defaulting on
  an unresolved choice. A client may ignore `depth`; it only explains why
  the turns that follow may ask more or fewer questions.
- `plan.proposed.depth_notice` is one plain sentence explaining why a plan
  is being proposed for this question (a compound ask, a methodological
  choice, a degraded resolution, a wide identity, or the user's own request
  for a thorough treatment). Show it beside the plan card as the reason the
  card appeared; it is absent when the server has none to give.
- `plan.proposed.decisions[]`: `{id, question, chosen, reason, resolved_by, alternatives?}`, the choices the agent made for the user.
- `plan.proposed.revision` is a string identifying the plan's approval
  key, distinct from the integer `version`. It sits beside `version`: a
  client that shows a version badge keeps using `version`.
- `plan.proposed.requirements[]`: `{id, kind, sentence, decision_ids}`. It
  is the evidence behind the plan. `plan.proposed.operations[]`: `{id, kind, sentence, requirement_id?}`. It is the plan's computed steps.
  `sentence` on both is composed server-side and rendered as given, so
  the plan card and the model's own narration describe the same plan in
  the same words.
- `revision`, `requirements`, `operations`, and `steps[].args` are
  optional. A client that ignores them renders the same plan card it
  always has.
- `activity.step.status` is one of `pending`, `running`, `complete`,
  `failed`, `skipped`. A `failed` step is recoverable; the turn continues.
  `detail` holds a one-line result on success; a card's step (labelled
  with the card's kind and title) carries none. On a failed step it is
  `<code>: <message>` from the refusal, or the message alone when the
  failure has no code, cut at 400 characters, for a reader diagnosing
  the turn. `produced` (`{component_id, revision}`)
  links a step to its card. In a data turn, the closing message states the
  reason a card was refused in plain words. It names the platform only when
  a platform service refused the write, and it says so when the assistant
  withheld the card. Any other refusal reads "the card could not be added".
  An edit turn whose write is refused closes with "The edit was refused:
  <message>", quoting the refusal's own message.
- The step with `tool: "search"` is the data search. Its `detail` reads
  `resolved: <status>`, then `N parts: <variable>: <status>; ...` when
  the question named several measures, then the search's own
  diagnostics. A measure asked per head with no such series is searched
  again as two, the measure and its denominator, and `detail` says so:
  `searched again as N parts in place of "<variable>", ...`, two parts
  for each measure named, followed by each
  part's status when the second search did not resolve both and the
  first result stands. `searched again with the unit in the measure: "<variable>"` precedes it when the per-head phrase was read as a unit.
- An `activity.step` with no `tool` is a status line: what is happening
  now, in the present tense. It opens `running` and later gets a
  `complete` frame with `duration_ms` on the same `step_id`. One status
  line runs at a time, and on the orchestrator engine one runs whenever
  an agent is working and no tool step is running. Show the label of the
  most recently opened step that is still `running`; a `complete` or
  `failed` frame updates the Activity record only. Fold status lines in
  the Activity record.
- The planner's status lines name its work: "Working on a plan" (from the
  hand-off to the planner until its first draft) or "Revising the plan"
  (from the hand-off of an amendment), "Checking the plan against the
  data" (from each proposal the planner makes until the proposal is
  answered), "Reworking the draft" (a draft after a refused proposal;
  `detail` gives the reason in plain words when there is one), "Getting
  the plan ready for you", and, once a plan is approved, "Starting the
  plan" until its first step's frame. The checking line's label is the
  same for every plan. Its `complete` frame carries a `detail` naming the
  data of the plan the proposal recorded, such as "health spending per
  person for Kenya and Uganda and 1 more" (the first dataset, then the
  count of the others), and no `detail` when that plan has no dataset.
  The Activity feed keeps a line's first label and shows a later frame's
  `detail` as its summary.
- The orchestrator agent's status lines are "Reading your question",
  "Reading your answer", and, after each batch of tool results, a line
  naming what the model is reading: "Reading what the search found",
  "Looking at <measure>, <first>–<last>" after one fetch ("Looking at
  <measure>" when the retrieval states no years; "Looking at the data"
  after several fetches, or after one that failed or whose measure no
  search named), "Going over the charts", "Reading the earlier turn",
  "Going over the notes on DFO", "Working out the next step" after a hand-off
  to the planner or a question to the user that was refused, and
  "Reading the plan's results" from the moment an approved plan finishes
  running. A batch that mixes tools is named by its latest stage (charts,
  then fetch, then search, then recall). A successful plan hand-off opens
  "Working on a plan", a successful ask hands over to the question, and
  neither opens a reading line. A batch of background-note reads keeps the
  line before it running. `detail` on these lines carries the limits the
  results state, in plain words: for a search, each found measure's years, the parts with more than one
  reading and the parts with no data; for a fetch, the retrieval's
  summary (observations, years, series), with each measure named when
  there are several. The line completes when the next thing happens: a
  tool call, answer text, a hand-off, or the end of the stream. A plan
  hand-back that carries no plan shows a `failed` "The planner could not
  make a plan", then "Reading the planner's answer"; the closing message
  then says what could not be calculated. No model thought text is sent.
- The orchestrator engine shows one step per agent tool call: "Searching
  for data" (detail: the query, then what it found and did not find),
  "Fetching <measure>" (detail: the retrieval's summary), "Making N
  charts", and one step per card written ("Line chart: <title>"). The
  calls a fetch makes on its own behalf (retrieving and checking the
  data), skill reads and the hand-off to the planner are not shown when
  they succeed; each still shows as a `failed` step when it fails. A
  "Making N charts" step is `failed` when any chart it asked for was not
  drawn, with `detail` naming how many and why; the cards it did draw
  still arrive.
- `component.created.payload` is the platform's resolved revision, `{schema_version, catalog_version, component_id, revision, provenance, component}`, as dfo-spec-kit's envelope schema defines it. `payload.component` holds everything the card draws, points included (see "Point data" below), and its title at `payload.component.title.text` (a note or a source card has none). `payload.provenance` is the platform's provenance for this revision and is authoritative for the card's back. Take the headline values and attribution from `fact_summary`.

**Point data.** The points come in `payload.component`: `series[].points[]` on a line or bar chart, each `{period, value}`, or `{period, status}` where no value is published, `rows[]` on a table, and `value` with `period` on a stat card. Plotting a card needs no second request.

A client that wants the raw frame behind a card takes its reference from `payload.provenance.inputs[0]` = `{spec_id, content_version}` (a note has none) and fetches it from the datasets door:

1. Send `GET /api/datasets/{spec_id}/{content_version}` to `dfo-apis`.
   How your app reaches `dfo-apis` (direct, through your proxy, or another
   way) is your decision; the reference pages already talk to the same
   service.
1. Read `artifacts[0].url` from the response. It is a short-lived signed
   URL. Do not store it; request a fresh one each time.
1. Fetch that URL. The body is a Parquet file. Each row is one point:
   `{stat_var, observationAbout, time_period, value, unit, measurementMethod, provenance, status}`.

The frame can hold more rows than the card draws. `payload.component` holds exactly the drawn points. The platform mints the file, and every row carries its provenance.

Known limit: the datasets door resolves `spec_` frames only. A card whose
frame `spec_id` starts with `ops_` rests on an analysis result (a
filter, join, growth rate, per-capita figure, or any other analysis step).
The frame is published to the platform, but
`GET /api/datasets/{spec_id}/{content_version}` answers 404 `not_found` for
it, so its raw frame is not served. The card's own points are in `payload.component` either way.

- `fact_summary` holds `component_id`, `revision`, `series[]` (`{id, label, point_count, coordinates?, first?, first_period?, last?, last_period?, change?}`; `first_period` and `last_period` are the periods of the first and last drawn points that hold a value; each is absent when its point carries no period, and both are absent for a stat card or table, which draw no series), `indicators[]`
  (`{id, name, description}`), `attribution` (`{label, source_count, indicator_count, source_detail_url?}`), `window?`, `unit?` (the unit slot of the card's payload, repeated as the platform states it, with its shape below),
  `series_truncated?`, `countries?`, `defaults?` (`[{pointer, detail}]`: the choices the platform made where the card's intent left one open, each naming the intent field it filled by JSON Pointer; absent means it made none). By design, the numbers in the message
  text come from fact summaries; the models are constrained to them, and
  the recorded transcripts hold to it. A chart's `window` is the one the
  platform states, or else the periods its axis draws: a bar chart of
  entities at one period has that period as both `start` and `end`. A
  table's and a stat card's `window` is null, and `series[].change` is
  null on every card. A table draws without either field.
- `fact_summary.unit` is the unit slot of the card's payload, in the payload's shape, so one reader serves both: a stat card's `unit`, a chart's first series' `unit`, and a table's row `unit` when every row has the same one. It is absent when the platform states none, as for a table whose rows differ in unit, a note, and a source card. Otherwise it is one of three shapes, and each object is closed:
  - a string of 1 to 200 characters, the unit's name, or its reference where the graph names none: `"US Dollar"`.
  - a structured unit, `{currency?, magnitude?, prices?, base_year?}` with at least one member: `{"currency": "USD", "prices": "constant", "base_year": 2023}`. `currency` is a three-letter ISO 4217 code, `magnitude` is `units`, `thousands`, `millions`, `billions`, or `trillions`, `prices` is `current` or `constant`, and `base_year` is the year of constant prices.
  - a composed unit, `{numerator, denominator}`, the unit of one measure divided by another, read "numerator per denominator": `{"numerator": "US Dollar", "denominator": {"stat_var": "dcid:Count_Person", "label": "Total population"}}`. Each side is a string or a structured unit as above, or `{stat_var, label}` for a measure that declares no unit, where `stat_var` is the `dcid:`-prefixed StatVar the measure counts and `label` is the name to print for it. A side is never null and never a composed unit.
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
- `elicitation.request.input` is `single_select`, `multi_select`, or
  `text`. In v1 the server emits only `single_select`, and the `answer`
  payload is one string. Treat `multi_select` as reserved. `progress` is
  `{completed, total}` for an "N of M" counter.

## 6. Pauses and resumes

A turn can pause more than once. Clarifying questions stop after two
rounds (the agent then continues with its default reading). Plan reviews
recur after each amendment, with no fixed count. Follow this procedure in
a loop:

1. Read frames until the stream ends.
1. If the last frame is `plan.proposed`: show the plan. Send
   `interrupt.response` with `approve`, `amend` (payload = the user's
   change request), or `reject`.
1. If the last frame is `elicitation.request`: show the question. Send
   `answer` (payload = option id or free text) or `skip`.
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
1. `reject` ends the turn. The stream sends `message.complete` and `done`.
   Nothing was computed.

A new `turn.request` on a paused conversation abandons the pause.

Retry an `interrupt.response` you are unsure landed: the server records
the answer before it starts resuming, so retrying the identical body is
safe. While the resumed stream from your first attempt is still running,
on any instance, the retry answers 409 `transient_unavailable` with
`retry_after_seconds`; once that stream is gone, the retry resumes the
turn again on a fresh stream, with `seq` restarting 1000 higher than
whatever the dead stream already sent. Retrying with a different
decision or payload than the one that actually resumed answers 409
`edit_conflict`.

## 7. Errors

The `error` field of an `error` frame, and the body of every non-200
response FROM THE STUDIO ROUTES' OWN CHECKS (404, 409, and the 422s in
section 4), is a ToolError: `{error, retry, detail, constraint_id?, pointer?, fix?, retry_after_seconds?}`. Two framework responses have a
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
  stream sends `component.created` for a replacement card. The new block
  takes the replaced block's slot on the canvas. The replaced component
  stays readable at its earlier revisions.
- An impossible edit ends with an honest `message.complete` and no
  component frame.
- To implement quick type-switch icons, send the same edit request with a
  fixed message, for example "turn it into a bar chart".

**Follow-up** (`intent: "follow_up"`): the agent gets the card's summary as
context for a normal data turn. On the orchestrator engine the agent is also
told the question is about that card. The agent reads the card's latest
revision. A `component_id` this conversation never held answers 404
`not_found` with `pointer: "/component_ref/component_id"`; a card inherited
by a copy of someone else's conversation is one of these.

**Canvas questions**: a message like "what is on my canvas?" is answered
from the canvas state without a data search. An empty canvas gets the
answer "nothing yet".

**Cross-turn memory**: the agent remembers two things. It remembers the
canvas (card titles and fact summaries, for the newest 16 blocks): "now
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

- `suggestions.items` is always empty.
- `indicators[].description` is a technical property string (for example
  `mp=lifeExpectancy,pt=Person`) or empty. Prose definitions
  wait on the upstream platform (`docs/upstream-asks.md`).
- Some questions fail at resolve with `capability_unavailable` ("identity
  does not reproduce spec_id"). This is a known upstream data gap. Show the
  error; a retry will not help.
- Cross-turn memory is bounded: the newest 16 canvas blocks and the
  newest 10 exchanges (older context is looked up on demand, which can
  miss). Very long conversations keep working from that window.
- On a deployment running more than one server instance, editing or
  following up on a card is reliable only when the request lands on the
  same instance that created the card. The conversation's other state
  (plan approvals, elicitation answers, plain data turns) works from any
  instance.

## 12. Prompt gallery and test material

Each row is a real prompt, the behavior it causes, and the recorded stream
that shows it. Every recording came from a live run against real data; all
frames validate against the models in
`services/harness/src/harness/wire.py`.

| Prompt or request                                                                                  | Behavior                                                                                                | Recording                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "life expectancy in Nigeria"                                                                       | Direct data turn: steps → card → message → done                                                         | `studio_transcript.jsonl`                                                                                                                                                                                                                              |
| "give me a comprehensive overview of health spending in Kenya"                                     | Plan pause → approve on `/interrupt` → card                                                             | `studio_transcript_plan.jsonl`                                                                                                                                                                                                                         |
| Same plan, `decision: "amend"`, payload "focus on the change since 2015"                           | `plan.versioned` → new `plan.proposed` → approve → card                                                 | `studio_transcript_amend.jsonl`                                                                                                                                                                                                                        |
| Same plan, `decision: "reject"`                                                                    | `message.complete` → `done`; nothing computed                                                           | `studio_transcript_reject.jsonl`                                                                                                                                                                                                                       |
| Turn 1 "life expectancy in Kenya"; turn 2 "now the same for Uganda"; turn 3 "what's on my canvas?" | Canvas memory across turns; a canvas question with no data search                                       | `studio_transcript_followup.jsonl`                                                                                                                                                                                                                     |
| `component_ref` edit: "rename it to …", then "turn it into a bar chart"                            | `component.revised`, then a replacement `component.created`                                             | `studio_transcript_edit.jsonl`                                                                                                                                                                                                                         |
| "average annual rainfall in Middle Earth"                                                          | No data: honest `message.complete` → `done`, no card                                                    | `studio_transcript_nodata.jsonl`                                                                                                                                                                                                                       |
| "population of Ghana"                                                                              | Stream-fatal `error` (`capability_unavailable`, a known upstream data gap)                              | `studio_transcript_error.jsonl`                                                                                                                                                                                                                        |
| Any turn, then POST `/cancel`                                                                      | Stream ends with the `cancelled` error frame                                                            | `studio_transcript_cancel.jsonl`                                                                                                                                                                                                                       |
| "quickly, life expectancy in Japan" with `auto_add: false`, then an `add` operation                | Cards stream without joining the canvas; the `add` response manifest is `composition_add_response.json` | `studio_transcript_autoadd.jsonl`                                                                                                                                                                                                                      |
| An ambiguous question that resolves to several readings                                            | `elicitation.request` pause; answer, free text, or skip                                                 | No live recording (current data rarely returns candidates). The frame shape is the fixture `services/harness/tests/wire_fixtures/studio/event-13-elicitation-skippable.json`; the request shapes are `request-07-interrupt-skip.json` and section 4.2. |

**Deterministic local server.** Clone the repo, run `uv sync --all-packages`, then `just studio-scripted`. This serves the full
interface on `http://localhost:8080` with no model, no upstream service,
and no credentials: fixed data, and identical requests produce identical
streams (only `duration_ms` varies), so end-to-end tests can snapshot
them. Any non-empty bearer token is accepted. The prompt picks the path:

| Prompt contains | Path                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `comprehensive` | plan pause (approve / amend / reject)                                            |
| `ambiguous`     | elicitation pause (answer / free text / skip), a state live data rarely produces |
| `nothing`       | no-data close                                                                    |
| `fail`          | stream-fatal `error` frame                                                       |
| anything else   | direct fetch → one card → done                                                   |

Card edits work too: a type word ("bar chart") recreates, any other edit
message renames.

Against the real agent, phrasing steers the same way: "comprehensive",
"thorough", or "detailed analysis" leads to the plan path; "quickly" leads
to a direct fetch. A real turn takes 15 to 60 seconds and several model
calls; a scripted turn is instant.

Each `.jsonl` file is one recorded conversation, one frame per line, in
order. A file can hold several turns and stream segments: `seq` restarts
at 0 at each turn, and a paused stream continues in the same file after
its `/interrupt`. States with no recording (card meta, pin, reorder,
remove, edit conflict, malformed requests) are plain HTTP request →
response cycles; build them from sections 4, 7, and 9.
