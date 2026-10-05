"""Classify the user's stated depth preference with a small model.

"Answer quickly", "give me a comprehensive overview", "go deep on X" are
depth signals the harness should honor -- but keyword matching is a path to
misery ("don't rush this" contains "rush"; "a quick note before my real
question: explain X fully" contains "quick"). A small model reads the
question and emits one flag; `harness.depth.decide_depth` folds that flag
into its verdict deterministically. The model flags, the pure function
decides.

`DepthHintExtractor` is the seam `harness.graph` depends on, so tests inject a
scripted extractor and stay model-free (the repo's CI/eval split: pytest
proves mechanics with no model in the loop). `GeminiDepthHintExtractor` is the
real implementation. Its failures raise -- a broken classifier surfacing as
"no preference" would silently flatten every depth hint, so the caller
decides what an outage means.
"""

from __future__ import annotations

import json
from typing import Protocol, cast

from google.genai import Client, types

from harness.depth import DepthHint
from harness.usage import log_usage

_CLASSIFY_INSTRUCTION = (
    "Classify whether the user's request expresses a preference about how "
    "deep or fast the answer should be. Respond with exactly one label:\n"
    '- "quick": they want a fast, brief, or minimal answer.\n'
    '- "thorough": they want a deep, comprehensive, or detailed treatment.\n'
    '- "none": no stated preference about depth or speed.\n'
    "Judge intent, not keywords: 'don't rush this' is not quick; 'a quick "
    "note before my question: explain X fully' is thorough."
)

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.STRING,
    enum=["quick", "thorough", "none"],
)


class DepthHintExtractor(Protocol):
    """Reads the user's question, returns their stated depth preference."""

    def __call__(self, *, question: str) -> DepthHint | None: ...


class GeminiDepthHintExtractor:
    """The real extractor: one constrained-decoding call to a small model."""

    def __init__(self, *, model_name: str, client: Client | None = None) -> None:
        self.model_name = model_name
        # Built on first call: the server constructs this keyless
        # (tests, CI), and `Client()` refuses to construct without a key.
        self._client = client

    def __call__(self, *, question: str) -> DepthHint | None:
        if self._client is None:
            self._client = Client()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=_CLASSIFY_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        log_usage(type(self).__name__, response.usage_metadata)
        label = json.loads(response.text or '""')
        if label == "none":
            return None
        if label in ("quick", "thorough"):
            return cast("DepthHint", label)
        raise ValueError(f"hint classifier returned {label!r}, not one of the schema's labels")


__all__ = ["DepthHintExtractor", "GeminiDepthHintExtractor"]
