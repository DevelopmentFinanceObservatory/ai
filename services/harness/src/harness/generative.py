"""The Edit Card's model call (the configured model, JSON-constrained).

`GeminiEdit` is a single constrained-decoding call on google.genai's async
client: stored intent, frame view and instruction in, a revise patch or a
recreate intent out. The model generates; `harness.studio_edit` validates and
executes what it returns, so a malformed generation fails loudly there.

Where a freeform mapping is needed (an intent's select), the schema asks for
a JSON STRING and the caller parses it: Gemini's constrained decoding requires
objects to declare their properties.
"""

from __future__ import annotations

import json
from typing import Any

from google.genai import Client, types

from harness.usage import log_usage

_JSON_TEMPERATURE = 0.2


def _schema(**properties: types.Schema) -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties=dict(properties),
        required=list(properties.keys()),
    )


_STR = types.Schema(type=types.Type.STRING)
_OPT_STR = types.Schema(type=types.Type.STRING, nullable=True)


def _content(**fields: Any) -> str:
    """The model-call input, as `_schema` is to outputs: named fields, one
    JSON document."""
    return json.dumps(fields)


class _ModelCall:
    """One JSON-constrained generation. The genai client is built on first
    use: the server constructs keyless (tests, CI), and a missing key then
    fails the first turn loudly.

    `timeout_s` bounds the call, genai's own retries included, as the HTTP
    timeout of its request. The server passes the request's model-time budget
    (`harness.settings.Settings.turn_model_time_s`).
    """

    def __init__(self, *, model_name: str, timeout_s: float, client: Client | None = None) -> None:
        self.model_name = model_name
        self._timeout_ms = round(timeout_s * 1000)
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
                http_options=types.HttpOptions(timeout=self._timeout_ms),
            ),
        )
        log_usage(type(self).__name__, response.usage_metadata)
        return json.loads(response.text or "{}")


_EDIT_INSTRUCTION = """\
You apply a user's edit to one canvas card. You are given the card's stored
intent (type, title, select, frame_ref, window), a view of its frame's
dimensions (with rolling: true when the frame's period is counted back from
the latest data), and the user's instruction. The platform's rules are hard
constraints:
- A revision may change ONLY title and select, together or alone. For title
  or selection edits, emit patch_json, a JSON object literal (as a string)
  with only the keys you change, e.g. {"title": "..."} or {"select": {...}}.
- type, window, change, and frame_ref CANNOT be revised. For those edits
  emit recreate_json instead, the FULL new intent as a JSON object literal
  (type, title, select, optional window, optional change). Keep every field
  you are not changing identical to the stored intent. Do not include
  frame_ref, because the caller pins it.
- window is an object of period labels at the frame's grain, e.g. {"start":
  "2000", "end": "2023"}. A window of labels may leave out either side,
  which then runs to the edge of the frame's coverage. "Since 2000" is
  {"start": "2000"}.
- A period the user counts back from the latest data is a relative period,
  {"anchor": "latest", "offset": N}, where 0 is the latest period and -1 the
  one before it. "The last three years" is {"start": {"anchor": "latest",
  "offset": -2}, "end": {"anchor": "latest", "offset": 0}}. When either side
  is relative, state both sides. Years the user names stay labels. "2015 to
  2020" is {"start": "2015", "end": "2020"}.
- Use change on a stat_card only. It compares two periods of the card's
  series as {"basis": [from, to]}, each a label or a relative period.
  "Compare with the previous year" is {"basis": [{"anchor": "latest",
  "offset": -1}, {"anchor": "latest", "offset": 0}]}. "Compare 2015 with
  2020" is {"basis": ["2015", "2020"]}.
- A title that names something you change (a period, an entity, a chart
  type) changes with it. "... (1960-2023)" shown since 2000 becomes "...
  (2000-2023)", with an open side read from the frame's coverage.
- A card whose window or change holds a relative period, or whose frame_view
  says rolling: true, moves with the data. Its title names no year and no
  year range ("... over the last three years", not "... (2021-2023)").
- select values must come from the frame view. Do not invent dimension
  values. If the user asks for data the frame does not hold, emit neither
  field and say so in text.
Emit exactly one of patch_json and recreate_json, with the other null. Also
emit text, one plain sentence stating what you changed (or why you could
not).
"""

_EDIT_SCHEMA = _schema(
    text=_STR,
    patch_json=_OPT_STR,
    recreate_json=_OPT_STR,
)

_PATCH_KEYS = frozenset({"title", "select"})
_RECREATE_KEYS = frozenset({"type", "title", "select", "window", "change"})


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


__all__ = ["GeminiEdit"]
