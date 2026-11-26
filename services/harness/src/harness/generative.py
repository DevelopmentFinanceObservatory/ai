"""The Edit Card's model call (the configured model, JSON-constrained).

`GeminiEdit` is a single constrained-decoding call on google.genai's async
client: stored intent, frame view and instruction in, a revise patch or a
recreate intent out. The model generates; `harness.studio_edit` validates and
executes what it returns, so a malformed generation fails loudly there.

Where a freeform mapping is needed (an intent's select), the schema asks for
a JSON STRING and the caller parses it: Gemini's constrained decoding requires
objects to declare their properties.

Each call runs at its `thinking_level` (`EDIT_THINKING_LEVEL`) and inside a
`chat <model>` span that carries the attributes Pydantic AI puts on the
orchestrator's and the planner's model calls (`gen_ai.*`, under the agent
name `EDIT_AGENT_NAME`), the model surface (`harness.gemini_wire.surface_attributes`)
and the usage Langfuse reads from a GenAI span (`harness.gemini_wire.record_usage`).
"""

from __future__ import annotations

import json
from typing import Any

from google.genai import Client, types
from opentelemetry import trace

from harness.conversation import note_model_usage
from harness.gemini_wire import record_usage, surface_attributes
from harness.settings import DEFAULT_EDIT_THINKING_LEVEL, ThinkingLevelName
from harness.usage import log_usage

_JSON_TEMPERATURE = 0.2

# The agent name the Edit Card call's span carries, as `dfo_orchestrator` and
# `dfo_planner` name theirs.
EDIT_AGENT_NAME = "dfo_edit"

_tracer = trace.get_tracer(__name__)


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
    (`harness.settings.Settings.turn_model_time_s`). `thinking_level` is the
    call's reasoning effort (`harness.settings.Settings.edit_thinking_level`).
    """

    def __init__(
        self,
        *,
        model_name: str,
        timeout_s: float,
        thinking_level: ThinkingLevelName = DEFAULT_EDIT_THINKING_LEVEL,
        client: Client | None = None,
    ) -> None:
        self.model_name = model_name
        self._timeout_ms = round(timeout_s * 1000)
        self._thinking_level = thinking_level
        self._client = client

    def _resolved_client(self) -> Client:
        if self._client is None:
            self._client = Client()
        return self._client

    async def _generate(
        self, *, instruction: str, content: str, schema: types.Schema
    ) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=instruction,
            response_schema=schema,
            response_mime_type="application/json",
            temperature=_JSON_TEMPERATURE,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(self._thinking_level)
            ),
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )
        with _tracer.start_as_current_span(
            f"chat {self.model_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.agent.name": EDIT_AGENT_NAME,
                "gen_ai.provider.name": "google",
                "gen_ai.system": "google",
                "gen_ai.request.model": self.model_name,
                **surface_attributes(config),
            },
        ) as span:
            try:
                response = await self._resolved_client().aio.models.generate_content(
                    model=self.model_name, contents=content, config=config
                )
            except Exception as exc:
                span.set_attribute("dfo.model.error", type(exc).__name__)
                raise
            _record_response(span, response, model_name=self.model_name)
        note_model_usage(response.usage_metadata)
        log_usage(type(self).__name__, response.usage_metadata)
        return json.loads(response.text or "{}")


def _record_response(
    span: trace.Span, response: types.GenerateContentResponse, *, model_name: str
) -> None:
    """Put on `span` the response model, finish reason and token usage of
    `response`, under the names Pydantic AI and `harness.gemini_wire` give
    them on the orchestrator's and the planner's model calls."""
    if not span.is_recording():
        return
    span.set_attribute("gen_ai.response.model", response.model_version or model_name)
    candidates = response.candidates or []
    reason = candidates[0].finish_reason if candidates else None
    span.set_attribute("dfo.model.finish_reason", reason.name if reason else "none")
    if reason:
        span.set_attribute("gen_ai.response.finish_reasons", (reason.name.lower(),))
    usage = response.usage_metadata
    if usage is None:
        return
    prompt = (usage.prompt_token_count or 0) + (usage.tool_use_prompt_token_count or 0)
    span.set_attribute("gen_ai.usage.input_tokens", prompt)
    span.set_attribute(
        "gen_ai.usage.output_tokens",
        (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
    )
    if usage.cached_content_token_count is not None:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cached_content_token_count)
    record_usage(span, usage)


_EDIT_INSTRUCTION = """\
You apply a user's edit to one canvas card. You are given the card's stored
intent (type, title, select, frame_ref, window, order), a view of its frame's
dimensions (with rolling: true when the frame's period is counted back from
the latest data), and the user's instruction. The platform's rules are hard
constraints:
- A revision may change ONLY title, select and order. For those edits emit
  patch_json, a JSON object literal (as a string) with only the keys you
  change, e.g. {"title": "..."} or {"select": {...}}.
- type, window, change, and frame_ref CANNOT be revised. For those edits
  emit recreate_json instead, the FULL new intent as a JSON object literal
  (type, title, select, optional window, optional change, optional order).
  Keep every field you are not changing identical to the stored intent. Do
  not include frame_ref, because the caller pins it.
- order, on a line_chart, bar_chart or table, draws its members ranked by
  value: {"by": "value", "direction": "descending"} for largest first,
  "ascending" for smallest first; {"order": null} in a patch removes it, and
  a recreate without order draws the card unordered.
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

_PATCH_KEYS = frozenset({"title", "select", "order"})
_RECREATE_KEYS = frozenset({"type", "title", "select", "window", "change", "order"})


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
