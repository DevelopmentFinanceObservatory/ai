"""The generative nodes' real model calls (the configured model, JSON-constrained).

One class per seam in `harness.graph.GenerativeNodes` EXCEPT the planner,
each a single constrained-decoding call on google.genai's async client --
async because the graph's nodes run on the server's event loop
(`harness.graph`'s own docstring). The model generates; the graph's code
decides: every output here is parsed and re-validated by the node that
consumes it (the platform's own intent schema for compose), so a malformed
generation fails loudly at the seam.

The planner seam is the exception and lives in `harness.planner_seam`: it
adapts the planner-translator pair in `src/planner` and `src/translator`,
each an `LlmAgent` with its own tools and several model calls per plan.
`harness.wiring.build_generative_nodes` assembles every seam.

Context discipline: prompts carry the question, the digest-sized resolve
envelope, describe_frame views, and fact summaries, never rows, never the
raw search envelope.

Where a freeform mapping is needed (tool args, intent select), the schema
asks for a JSON STRING and the caller parses it: Gemini's constrained
decoding requires objects to declare their properties, and tool args are
open by design.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from compiler.surface import ToolSurface
from google.genai import Client, types

from harness.dfo_faq import DFO_FAQ_NOT_READ, DfoFaqUnavailable, read_dfo_faq
from harness.usage import log_usage

logger = logging.getLogger(__name__)

_JSON_TEMPERATURE = 0.2


def _schema(**properties: types.Schema) -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties=dict(properties),
        required=list(properties.keys()),
    )


_STR = types.Schema(type=types.Type.STRING)
_OPT_STR = types.Schema(type=types.Type.STRING, nullable=True)


def _array(items: types.Schema) -> types.Schema:
    return types.Schema(type=types.Type.ARRAY, items=items)


def _content(**fields: Any) -> str:
    """The model-call input, as `_schema` is to outputs: named fields, one
    JSON document."""
    return json.dumps(fields)


class _ModelCall:
    """One JSON-constrained generation. The genai client is built on first
    use: the server constructs keyless (tests, CI), and a missing key then
    fails the first turn loudly."""

    def __init__(self, *, model_name: str, client: Client | None = None) -> None:
        self.model_name = model_name
        self._client = client

    def _resolved_client(self) -> Client:
        if self._client is None:
            self._client = Client()
        return self._client

    async def _generate(
        self, *, instruction: str, content: str, schema: types.Schema
    ) -> dict[str, Any]:
        response = await self._resolved_client().aio.models.generate_content(
            model=self.model_name,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_schema=schema,
                response_mime_type="application/json",
                temperature=_JSON_TEMPERATURE,
            ),
        )
        log_usage(type(self).__name__, response.usage_metadata)
        return json.loads(response.text or "{}")


_CONVERSE_INSTRUCTION = """\
You open a turn for a development-finance data assistant. Given the user's
question, produce:
- text: one short sentence acknowledging what you are looking into. No
  greetings, no promises of specific numbers you have not seen. For about
  "reply" (below), text is instead the whole answer.
- search_parts: the question decomposed for the data search, ONE entry per
  measured variable, with every entity the user asked about for that
  variable in that entry's entities list. Two entities on one variable are
  one entry with two entities, not two entries -- "life expectancy in
  Nigeria and Guatemala" is [{"variable": "life expectancy", "entities":
  [{"text": "Nigeria"}, {"text": "Guatemala"}]}]. Two entries are for two
  different variables ("life expectancy and GDP per capita in Nigeria").
  Use the user's own words for variable, entities, and unit; do not translate to
  canonical names. role and time are the exceptions -- see below. Include
  only fields the user actually expressed: variable (required, what is
  measured), entities (who/where, each with its text and, when the user
  stated a relation, a role), time (their time phrase), unit (their unit
  phrase).
- about: "canvas" ONLY when the question asks about the canvas/cards
  themselves ("what's on my canvas", "describe that chart") rather than for
  data; "reply" when the message asks for no data at all -- a greeting,
  thanks, small talk, or a question about the Development Finance
  Observatory itself or what you can ask; otherwise "data". For "reply",
  write no search_parts, and make text the complete answer in two to four
  plain sentences. Never state a figure in a reply. When about is "reply"
  and the question is about DFO itself, set about_dfo and leave the
  description out of text.
- about_dfo: true when about is "reply" because the message asks about the
  Development Finance Observatory itself or what you can do; it is answered
  from DFO's own published description. A question about what data exists
  on a topic stays about "data". Otherwise false.

Two of those fields are matched downstream against a fixed grammar, not
read as free text -- normalize into it when you can, rather than passing
through a phrase it will not parse:
- role: state which side of the flow the user's word names, not the word
  itself. A word for where something originates (e.g. donor, provider,
  creditor, exporter) means "from"; a word for where it is directed (e.g.
  recipient, debtor, importer) means "to". Write "from" or "to".
- time: a bare year ("2020"), a range ("2015 to 2020" or "2015-2020"), an
  open start ("since 2015"), or "latest"/"current"/"now". Normalize a
  relative or colloquial time phrase into this grammar when you can; when
  you cannot, pass the user's phrase through as they said it rather than
  omitting it: an omitted time resolves across every period with data,
  which is rarely what a user who named a period meant.

When a canvas digest is supplied (the user's existing cards, each naming
the variable and entities it shows), use it to resolve references to prior
work: "do the same for Uganda" after a Kenya life-expectancy card means
life expectancy in Uganda -- write search_parts with the CONCRETE variable
and entities, never the words "same" or "that".

One wording asks to CHANGE a card that is already on the canvas rather than
to draw a new one: "add Guatemala to the chart", "show both on the same
chart", "also include Benin there". For that, set `revises` to the
`component_id` of the card being changed, and write the search part for the
SAME variable naming that card's EXISTING entities together with the new
ones -- the changed card then shows every entity in the part. Leave
`revises` null for a new, separate chart: "now show me Guatemala" after a
Nigeria card is a new card about Guatemala alone, and repeating one of the
existing card's entities is what tells the two apart.

When a conversation digest is supplied (recent exchanges: what the user
said, what was answered, which cards resulted), use it the same way: "the
country I mentioned" resolves to the country actually named earlier.

- recall: normally null. Set it ONLY when the question clearly references
  something in this conversation that neither digest contains -- an
  exchange older than the digest window ({"scope": "conversation"}) or a
  card's full details beyond its one-line summary ({"scope": "canvas"}) --
  with a short query naming what you need. Never set it when the digests
  already answer the reference, and never to search for new data.
When a recall_result is supplied, it is the answer to that request: use it
and produce the final output (recall may not be requested again).
"""

_CONVERSE_SCHEMA = _schema(
    about=types.Schema(type=types.Type.STRING, enum=["data", "canvas", "reply"]),
    about_dfo=types.Schema(type=types.Type.BOOLEAN),
    text=_STR,
    # The canvas card a "change that card" wording targets, when there is
    # one -- the node validates it against the digest and drops it otherwise.
    revises=_OPT_STR,
    search_parts=_array(
        types.Schema(
            type=types.Type.OBJECT,
            properties={
                "variable": _STR,
                "entities": types.Schema(
                    type=types.Type.ARRAY,
                    nullable=True,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"text": _STR, "role": _OPT_STR},
                        required=["text"],
                    ),
                ),
                "time": _OPT_STR,
                "unit": _OPT_STR,
            },
            required=["variable"],
        )
    ),
    recall=types.Schema(
        type=types.Type.OBJECT,
        nullable=True,
        properties={
            "scope": types.Schema(type=types.Type.STRING, enum=["conversation", "canvas"]),
            "query": _STR,
        },
        required=["scope", "query"],
    ),
)

# The second draft of a reply about DFO itself, from the dfo-faq skill.
_DFO_REPLY_INSTRUCTION = """\
You answer a question about the Development Finance Observatory (DFO), the
data assistant's own platform. You are given the question, DFO's own
description of itself as dfo_faq, and, when there are any, digests of the
conversation and the user's canvas. Answer from dfo_faq alone, in two to four
plain sentences. State only the figures dfo_faq itself gives, and add
nothing dfo_faq does not say.
"""

_DFO_REPLY_SCHEMA = _schema(text=_STR)


def _role_free(entities: list[dict[str, Any]]) -> bool:
    """True when no entity in `entities` names a role. The converse schema
    declares `role` an optional string, so Gemini may emit `"role": null`
    for a role-free entity -- `entity.get("role")` reads that the same as
    the key's plain absence, where `"role" not in entity` would not."""
    return all(not entity.get("role") for entity in entities)


def _dedup_entities_by_text(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`entities` with repeated `text` values dropped, first occurrence kept."""
    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for entity in entities:
        text = entity.get("text")
        if text in seen:
            continue
        seen.add(text)
        deduped.append(dict(entity))
    return deduped


def _merge_same_variable_parts(
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold parts that ask the same question of different entities into one.

    Two parts merge when their keys match and every key but `entities` is
    equal -- the same variable, time, unit and roles, asked of two entity
    lists -- and both the held part's and the incoming part's entities are
    role-free. The merged part carries the entities in first-seen order
    without repeats: a repeat already in the held part's own entities, or
    inside one incoming part's own list, folds away the same as a repeat
    across parts. A part whose entities carry a role, or that would merge
    into one that does, stays separate: the role marks each entity's side of
    a flow, and merging would swap sides.
    """
    merged: list[dict[str, Any]] = []
    for part in parts:
        entities = part.get("entities") or []
        if entities and _role_free(entities):
            for held in merged:
                held_entities = held.get("entities") or []
                if (
                    _role_free(held_entities)
                    and set(held.keys()) == set(part.keys())
                    and all(held.get(key) == part.get(key) for key in part if key != "entities")
                ):
                    held["entities"] = _dedup_entities_by_text(held_entities + entities)
                    break
            else:
                merged.append({**part, "entities": _dedup_entities_by_text(entities)})
        else:
            merged.append(dict(part))
    return merged


class GeminiConverse(_ModelCall):
    """Converse's draft. A reply about DFO itself is drafted a second time
    with dfo-faq (`harness.dfo_faq`) read through `surface`, so only a
    question about DFO reads the skill and the reply quotes its current text.
    A deployment that cannot serve it gets the sentence that says so, with
    no second draft."""

    def __init__(
        self,
        *,
        model_name: str,
        client: Client | None = None,
        surface: ToolSurface | None = None,
    ) -> None:
        super().__init__(model_name=model_name, client=client)
        self._surface = surface

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"question": question}
        if canvas_digest:
            fields["canvas"] = canvas_digest
        if conversation_digest:
            fields["history"] = conversation_digest
        if recall_result:
            fields["recall_result"] = recall_result
        content = _content(**fields) if len(fields) > 1 else question
        drafted = await self._generate(
            instruction=_CONVERSE_INSTRUCTION, content=content, schema=_CONVERSE_SCHEMA
        )
        if drafted.get("about") == "reply" and drafted.get("about_dfo"):
            return await self._dfo_reply(fields)
        parts = [
            {key: value for key, value in part.items() if value not in (None, [], "")}
            for part in drafted.get("search_parts") or []
        ]
        recall = drafted.get("recall")
        search_parts = _merge_same_variable_parts([p for p in parts if p.get("variable")])
        revises = drafted.get("revises")
        return {
            "text": drafted.get("text", ""),
            "search_parts": search_parts,
            "about": drafted.get("about") or "data",
            # The canvas card a "change that card" wording targets, taken
            # verbatim from the model's own component_id: the graph checks
            # it against this conversation's own canvas, which this seam
            # cannot see.
            "revises": revises if isinstance(revises, str) and revises else None,
            # Suppressed once answered: the node's round cap is structural,
            # but a well-formed reply never re-asks either.
            "recall": recall if isinstance(recall, dict) and not recall_result else None,
        }

    async def _dfo_reply(self, fields: dict[str, Any]) -> dict[str, Any]:
        """A reply about DFO itself, drafted from dfo-faq, or the sentence
        saying why it could not be read."""
        faq = await read_dfo_faq(self._surface)
        if isinstance(faq, DfoFaqUnavailable):
            text = faq.reply
        else:
            drafted = await self._generate(
                instruction=_DFO_REPLY_INSTRUCTION,
                content=_content(**fields, dfo_faq=faq),
                schema=_DFO_REPLY_SCHEMA,
            )
            text = str(drafted.get("text") or "")
            if not text:
                logger.warning(
                    "converse drafted an empty reply from dfo-faq: question=%r",
                    fields["question"],
                )
                text = DFO_FAQ_NOT_READ
        return {"text": text, "search_parts": [], "about": "reply", "revises": None, "recall": None}


_COMPOSE_INSTRUCTION = """\
You choose canvas components to present data frames. For each frame you are
given its reference index, the plan step index and tool that produced it,
and whether that step is a final step (is_final_step). A final step's frame
answers part of the question; an earlier step's frame (a fetch feeding an
aggregate or filter step) is only that step's raw input.

A frame's view is either present (its dimensions, under "view") or missing
(an "unavailable_reason" string in its place, because describe_frame
refused). Never draw a component from a different frame's view to cover for
a missing one, and never claim a frame's data when its view is missing.

Produce intents:
- frame_index: which frame (index into the given list). Only pick a frame
  whose view is present.
- type: line_chart for a series over time (several entities over time is a
  line_chart whose selected entity dimension is the array -- one line per
  entity), bar_chart for a single period read across entities, stat_card
  for a single headline value, table for detail.
- select_json: a JSON object literal (as a string) mapping dimension name to
  value, choosing what the component shows -- REQUIRED. Values must come
  from the frame view; never invent dimension values. The view's
  series_count says how many series the frame holds before any selection;
  1 means every dimension is already pinned. The platform's binding rules,
  by type:
  - stat_card must resolve to exactly one series: pin EVERY dimension the
    view marks plural to one value, except the facet columns unit,
    measurementMethod and provenance. Those need no pin: the platform
    refuses only when one series holds several of their values. Pin a facet
    only to choose one of its listed values.
    A stat_card select carries no array -- every value is a bare string.
  - line_chart and bar_chart may use an array for at most one dimension
    (line_chart up to 8 values, bar_chart up to 24). table may use arrays
    on more than one dimension; their cross-product is at most 24 series.
    Every dimension not given an array takes a single string value.
- title: a short human title stating what the component shows (<= 120 chars).

Every final step's frame gets at least one component, so every part of the
question is on the canvas. Add another component from the same frame when it
shows something the first does not: a stat_card for the latest value beside
a line_chart of the trend, or a table beside a chart when the exact figures
matter. Never add one that repeats what another already shows. If a final
step's frame has no view, produce no component for it -- do not substitute
an earlier step's frame in its place. The closing message names the gap from
the same facts, so leaving it out here is not losing it.
"""

_COMPOSE_SCHEMA = _schema(
    intents=_array(
        types.Schema(
            type=types.Type.OBJECT,
            properties={
                "frame_index": types.Schema(type=types.Type.INTEGER),
                "type": types.Schema(
                    type=types.Type.STRING,
                    enum=["line_chart", "bar_chart", "stat_card", "table"],
                ),
                "select_json": _STR,
                "title": _STR,
            },
            required=["frame_index", "type", "select_json", "title"],
        )
    ),
)


class GeminiCompose(_ModelCall):
    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frames: list[dict[str, Any]] = []
        for index, view in enumerate(frame_views):
            provenance = frame_provenance[index] if index < len(frame_provenance) else {}
            frame: dict[str, Any] = {
                "frame_index": index,
                "step": provenance.get("step"),
                "tool": provenance.get("tool"),
                "is_final_step": bool(provenance.get("is_final_step")),
            }
            if view.get("available"):
                frame["view"] = view.get("view") or {}
            else:
                frame["unavailable_reason"] = view.get("reason") or "view unavailable"
            frames.append(frame)
        content = _content(question=question, frames=frames)
        drafted = await self._generate(
            instruction=_COMPOSE_INSTRUCTION, content=content, schema=_COMPOSE_SCHEMA
        )
        intents: list[dict[str, Any]] = []
        for entry in drafted.get("intents", []):
            index = int(entry.get("frame_index", 0))
            if not 0 <= index < len(frame_refs):
                continue
            # select_json is model-drafted free text: a malformed string
            # raises from json.loads, and a well-formed but non-object
            # value (a JSON array or scalar) parses fine but is not
            # something `select.items()` (harness.graph._select_binding_ok)
            # can walk. Either shape skips the intent rather than taking
            # down the whole compose node.
            try:
                select = json.loads(entry["select_json"])
            except json.JSONDecodeError:
                logger.warning(
                    "compose seam skipped an intent whose select_json did not parse: "
                    "type=%r title=%r select_json=%r",
                    entry.get("type"),
                    entry.get("title"),
                    entry.get("select_json"),
                )
                continue
            if not isinstance(select, dict):
                logger.warning(
                    "compose seam skipped an intent whose select_json was not a JSON "
                    "object: type=%r title=%r select_json=%r",
                    entry.get("type"),
                    entry.get("title"),
                    entry.get("select_json"),
                )
                continue
            intents.append(
                {
                    "type": entry["type"],
                    "frame_ref": frame_refs[index],
                    "select": select,
                    "title": entry["title"][:120],
                }
            )
        return {"intents": intents}


_NARRATE_INSTRUCTION = """\
You write the closing message of a data-assistant turn, in GitHub-flavored
Markdown. You are given the question and the bounded facts: component fact
summaries (series labels, point counts, first/last values, windows), frame
views, per-step result summaries, frame references, refused_writes,
revises_requested, and components_replaced. Answer the question FROM THESE
FACTS ONLY -- never a number that is not in the facts, no speculation about
causes. Compare the measure, unit, places, direction and period in the facts
with the question. When the facts answer a different or narrower question,
state the specific difference in the first sentence ("The available data
shows Zambia's debt service as a share of exports, not payment amounts";
"The available data covers infant mortality only"), then describe what
they show. Describe the data provided, not what a search failed to find
or what other data might exist. Use plain names and units, never raw
identifiers or codes. If their meaning is unclear, say so rather than
guess. For a current value, give the latest year in the facts without a
missing-data caveat. For an analytical question, distinguish what the
facts show from what they cannot establish; an unproven conclusion is
not missing data. When the facts match the question, add no mismatch
caveat. A frame view's counts (its series, entities or providers) describe
the data's shape, not the answer: never state them. Write a figure about as
precisely as the chart labels it: two or three significant figures, a scale
word from millions up ("7.6 million", "$1.1 billion"), no trailing ".0".
A simple lookup gets two or three plain sentences; a multi-step or
multi-series answer may use short bold lead-ins or a compact bullet list
(like an analyst's note), still bounded to the facts given. Mention the
canvas only when components_created is at least 1. When components_replaced
is true, the question asked to change a card already on the canvas and it
did: say the existing chart was updated to show what it now shows ("updated
the chart to show Nigeria and Guatemala"), never that a chart was added.
When revises_requested is true and components_replaced is false: with
components_created at least 1, the card the question named was gone by the
time the new one landed, so describe the new card as added, the same as any
other new component, never as an update; with components_created 0 and a
non-empty refused_writes, say the change did not go through and the chart
still shows what it showed before, never that a component was declined as
if it were new. Otherwise, when components_created is 0, distinguish two
cases: an empty refused_writes means no component was ever drafted, so say
the data could not be presented and never point to a visual; a non-empty
refused_writes means a component was drafted and not added -- say so
plainly, using only refused_writes's own type/title/reason. Attribute the
missing card exactly as the reason does: say the platform declined only
when the reason names the platform; when the reason says the card was
withheld or the assistant could not build it, say that and never say the
platform or the canvas declined it; when the reason names nobody, name
nobody. Invent nothing
beyond the reason, and offer asking again only when the reason itself says
so. If the facts are thin, say what was prepared rather than inventing
detail. When failed_step is set, a plan step could not run: say what that
step was doing and why it could not, in plain words without its id or the
counts or codes its failure message carries, and do not claim the result it
would have produced. Each stat_var_totals entry says the figures are a total
over several indicators: state it word for word, and never describe such a
figure as one indicator's value. Never internal jargon (cosine scores,
spec ids, content versions, status codes, "no_observations") even when a
fact contains it -- describe an unavailable frame or a refused write in
plain words instead.
"""

_NARRATE_SCHEMA = _schema(text=_STR)


class GeminiNarrate(_ModelCall):
    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
        content = _content(question=question, facts=facts)
        drafted = await self._generate(
            instruction=_NARRATE_INSTRUCTION, content=content, schema=_NARRATE_SCHEMA
        )
        return {"text": drafted.get("text", "")}


_REFUSAL_INSTRUCTION = """\
A data question could not be answered. You are given the question, the
resolve status, and the resolver's diagnostic messages. Write the short
closing chat message: one or two plain sentences saying what could not be
done, in the user's own terms, and inviting a different phrasing. When the
diagnostics name concrete alternative readings of the question, offer them
in plain words ("did you mean military spending, or government debt?").
Use ONLY what the diagnostics say -- no other facts, no numbers unless the
user needs them, and never internal jargon (cosine scores, spec ids,
status codes, "no_observations"). Do not promise background work or claim
anything was added to the canvas.
"""

_REFUSAL_SCHEMA = _schema(text=_STR)


class GeminiRefusal(_ModelCall):
    async def __call__(self, *, question: str, status: str, warnings: list[str]) -> dict[str, Any]:
        content = _content(question=question, status=status, diagnostics=warnings)
        drafted = await self._generate(
            instruction=_REFUSAL_INSTRUCTION, content=content, schema=_REFUSAL_SCHEMA
        )
        return {"text": drafted.get("text", "")}


_EDIT_INSTRUCTION = """\
You apply a user's edit to one canvas card. You are given the card's stored
intent (type, title, select, frame_ref, window), a view of its frame's
dimensions, and the user's instruction. The platform's rules, which are
hard constraints:
- A REVISION may change ONLY title and/or select. Emit patch_json for
  title/selection edits: a JSON object literal (as a string) with only the
  keys you change, e.g. {"title": "..."} or {"select": {...}}.
- type, window, and frame_ref CANNOT be revised. For those edits emit
  recreate_json instead: the FULL new intent as a JSON object literal
  (type, title, select, optional window) -- keep every field you are not
  changing identical to the stored intent. Never include frame_ref; the
  caller pins it.
- window is an object of period labels at the frame's grain, e.g.
  {"start": "2000", "end": "2023"}. Either side may be omitted and runs to
  the edge of the frame's coverage: "since 2000" is {"start": "2000"}.
- A title that names something you change (a period, an entity, a chart
  type) changes with it: "... (1960-2023)" shown since 2000 becomes
  "... (2000-2023)", with an open side read from the frame's coverage.
- select values must come from the frame view; never invent dimension
  values. If the user asks for data the frame does not hold, emit neither
  field and say so in text.
Emit exactly one of patch_json / recreate_json (the other null), plus
text: one plain sentence stating what you changed (or why you could not).
"""

_EDIT_SCHEMA = _schema(
    text=_STR,
    patch_json=_OPT_STR,
    recreate_json=_OPT_STR,
)

_PATCH_KEYS = frozenset({"title", "select"})
_RECREATE_KEYS = frozenset({"type", "title", "select", "window"})


class GeminiEdit(_ModelCall):
    """The Edit Card model call: stored intent + frame view + instruction ->
    a revise patch or a recreate intent. The studio edit route validates and
    executes the result; see `harness.studio`."""

    async def __call__(
        self,
        *,
        message: str,
        stored_intent: dict[str, Any],
        frame_view: dict[str, Any],
    ) -> dict[str, Any]:
        content = _content(instruction=message, stored_intent=stored_intent, frame_view=frame_view)
        drafted = await self._generate(
            instruction=_EDIT_INSTRUCTION, content=content, schema=_EDIT_SCHEMA
        )
        out: dict[str, Any] = {"text": drafted.get("text", "")}
        for field, legal_keys in (("patch", _PATCH_KEYS), ("recreate", _RECREATE_KEYS)):
            raw = drafted.get(f"{field}_json")
            parsed = json.loads(raw) if raw else None
            if parsed is not None:
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"edit {field} must be a JSON object, got {type(parsed).__name__}"
                    )
                if not set(parsed) <= legal_keys:
                    raise ValueError(
                        f"edit {field} carries illegal keys: {sorted(set(parsed) - legal_keys)}"
                    )
            out[field] = parsed
        if out["patch"] and out["recreate"]:
            raise ValueError("edit drafted BOTH a patch and a recreate; exactly one is legal")
        return out


__all__ = [
    "GeminiCompose",
    "GeminiConverse",
    "GeminiEdit",
    "GeminiNarrate",
    "GeminiRefusal",
]
