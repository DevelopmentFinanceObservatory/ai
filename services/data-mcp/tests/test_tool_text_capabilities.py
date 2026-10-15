"""The tool text data-mcp publishes, checked against what the Graph API's resolve door answers.

`fixtures/graph_api_capabilities.json` is the body of apis' `GET /api/graph/capabilities`:
`kinds` is the closed kind list, `resolve_kinds` the kinds a resolve request may name. The
kinds `get_data` resolves are the ones a request naming none admits, since
`data_mcp.resolution` names none: `resolve_request.kinds.default` in
`fixtures/graph_api_schema.json`. Every other kind is one a scope entity never resolves to.

`_KIND_WORDS` gives the words tool text uses for each kind. A claim is a sentence with the verb
"resolve" in it. The checks:

- a negated claim (never, not, cannot) names no resolvable kind, and an "only" claim names
  every resolvable kind, in every tool's description and input schema;
- `get_data`'s text names every resolvable kind in a positive claim and no other kind there.

A kind the snapshot adds fails `test_every_kind_has_words` until it has words, and then the
coverage check until `get_data` names it. The checks cannot catch a claim made without the
verb ("World Bank data is out of reach"), a kind named by a word missing from `_KIND_WORDS`,
or a claim about one entity of a kind that resolves ("Kosovo does not resolve") unless that
entity's name is listed as an exemplar. They read words, so a negation in one clause of a
sentence counts against every kind the sentence names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from data_mcp.server import mcp
from fastmcp import Client

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture

_FIXTURES = Path(__file__).parent / "fixtures"
_CAPABILITIES = json.loads((_FIXTURES / "graph_api_capabilities.json").read_text())
_SCHEMA = json.loads((_FIXTURES / "graph_api_schema.json").read_text())

_RESOLVABLE = frozenset(_SCHEMA["schemas"]["resolve_request"]["properties"]["kinds"]["default"])
_UNRESOLVABLE = frozenset(_CAPABILITIES["kinds"]) - _RESOLVABLE

# The nouns tool text names each kind by, singular and plural, and the named entities it
# uses as examples of one. `other` is the kind of a node nothing else classes; text never
# names it.
_KIND_WORDS: dict[str, tuple[str, ...]] = {
    "country": ("country", "countries"),
    "continent": ("continent", "continents"),
    "subregion": ("subregion", "subregions"),
    "continental_union": ("continental union", "continental unions"),
    "place_other": ("place below a country", "places below a country"),
    "aggregate": ("aggregate", "aggregates"),
    "organisation": (
        "organisation",
        "organisations",
        "organization",
        "organizations",
        "World Bank",
    ),
    "group": ("group", "groups", "G7"),
    "stat_var": ("StatVar", "StatVars", "statistical variable", "statistical variables"),
    "code_system": ("code system", "code systems"),
    "code": ("code", "codes"),
    "unit": ("unit", "units"),
    "source": ("source", "sources"),
    "dataset": ("dataset", "datasets"),
    "provenance": ("provenance", "provenances"),
    "other": (),
}

_CLAIM = re.compile(r"\bresolves?\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:never|not|cannot|can't|won't|doesn't|don't)\b", re.IGNORECASE)
_ONLY = re.compile(r"\bonly\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")


def _kind_pattern(kind: str) -> re.Pattern[str]:
    words = "|".join(re.escape(word) for word in _KIND_WORDS[kind])
    return re.compile(rf"\b(?:{words})\b", re.IGNORECASE)


def _claims(text: str) -> list[str]:
    """Each sentence of `text` that says something resolves, whitespace collapsed."""
    sentences = _SENTENCE_END.split(" ".join(text.replace("`", "").split()))
    return [sentence for sentence in sentences if _CLAIM.search(sentence)]


def _named_kinds(sentence: str) -> set[str]:
    return {
        kind for kind in _KIND_WORDS if _KIND_WORDS[kind] and _kind_pattern(kind).search(sentence)
    }


def _restriction_errors(text: str) -> list[str]:
    """Claims in `text` that a resolvable kind does not resolve."""
    errors = []
    for sentence in _claims(text):
        named = _named_kinds(sentence)
        if _NEGATION.search(sentence) and (denied := named & _RESOLVABLE):
            errors.append(f"denies {sorted(denied)}: {sentence!r}")
        elif _ONLY.search(sentence) and named and (left_out := _RESOLVABLE - named):
            errors.append(f"'only' leaves out {sorted(left_out)}: {sentence!r}")
    return errors


def _scope_errors(text: str) -> list[str]:
    """`_restriction_errors`, plus a positive claim for a kind that never resolves and any
    resolvable kind no positive claim names."""
    errors = _restriction_errors(text)
    claimed: set[str] = set()
    for sentence in _claims(text):
        if _NEGATION.search(sentence):
            continue
        named = _named_kinds(sentence)
        if wrong := named & _UNRESOLVABLE:
            errors.append(f"claims {sorted(wrong)} resolve: {sentence!r}")
        claimed |= named
    if missing := _RESOLVABLE - claimed:
        errors.append(f"names no claim that {sorted(missing)} resolve")
    return errors


def _schema_descriptions(schema: Any) -> list[str]:
    """Every `description` in a JSON Schema, depth first."""
    if isinstance(schema, dict):
        own = [schema["description"]] if isinstance(schema.get("description"), str) else []
        return own + [d for value in schema.values() for d in _schema_descriptions(value)]
    if isinstance(schema, list):
        return [d for value in schema for d in _schema_descriptions(value)]
    return []


async def _published_texts() -> dict[str, str]:
    """Each tool's description and input schema descriptions, as `list_tools` publishes them."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {
        tool.name: "\n\n".join([tool.description or "", *_schema_descriptions(tool.input_schema)])
        for tool in tools
    }


def test_the_snapshots_agree_on_the_resolvable_kinds() -> None:
    assert _RESOLVABLE
    assert _RESOLVABLE <= set(_CAPABILITIES["resolve_kinds"]) <= set(_CAPABILITIES["kinds"])


def test_every_kind_has_words() -> None:
    assert set(_KIND_WORDS) == set(_CAPABILITIES["kinds"])
    assert all(_KIND_WORDS[kind] for kind in _RESOLVABLE)


async def test_no_tool_text_denies_a_kind_resolve_answers() -> None:
    texts = await _published_texts()

    assert {name: _restriction_errors(text) for name, text in texts.items()} == {
        name: [] for name in texts
    }


async def test_get_data_claims_exactly_the_kinds_resolve_answers() -> None:
    texts = await _published_texts()

    assert _scope_errors(texts["get_data"]) == []


@pytest.mark.parametrize(
    "text",
    [
        # The `get_data` text before organisations and groups resolved.
        "Only countries, subregions, and continents resolve. Organisations such as the "
        "World Bank can never resolve here.",
        "Countries, continents and organisations resolve. Groups such as the G7 do not resolve.",
        "Countries, subregions, continents, continental unions, places below a country, "
        "organisations, DAC aggregates, membership groups and units resolve.",
        "Countries and continents resolve.",
    ],
)
def test_a_wrong_claim_is_caught(text: str) -> None:
    assert _scope_errors(text)
