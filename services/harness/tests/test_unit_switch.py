"""Tests for `harness.unit_switch`: a part re-resolved in another unit through Search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolId
from data_mcp.definition_writes import unit_identity
from data_mcp.scope import ScopeEntry
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.scope_choices import PartScope, ScopeDefinition
from harness.turn_records import SearchReading
from harness.unit_switch import UnitRefused, UnitResubmitted, resubmit_unit

_RECORDED = json.loads(
    (
        Path(__file__).parents[2] / "data-mcp" / "tests" / "fixtures" / "unit_conversion_aid.json"
    ).read_text()
)
_UNIT = _RECORDED["unit"]


class _Search:
    """A Search stub: it answers the identity it is posted under its own spec_id, or refuses."""

    def __init__(
        self, *, refusal: dict[str, object] | None = None, mint: str | None = None
    ) -> None:
        self.refusal = refusal
        self.mint = mint
        self.posted: list[dict[str, Any]] = []

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        del tool, on_progress
        self.posted.append(dict(arguments))
        if self.refusal is not None:
            return ToolCallResult(is_error=True, structured_content=self.refusal, text="refused")
        spec = {
            "spec_id": self.mint or arguments["spec_id"],
            "definition": {"identity": arguments["identity"]},
        }
        part = {"status": "definite", "spec": spec}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _block() -> dict[str, Any]:
    return {
        "spec_id": _RECORDED["spec_id"],
        "definition": {"identity": _RECORDED["identity"]},
    }


def _reading_with_scope() -> SearchReading:
    entry = ScopeEntry.model_validate(
        {
            "axis": "provider",
            "option_id": "provider:bilateral",
            "origin": {"kind": "choice", "choice_id": "provider"},
            "provenance": "explicit",
        }
    )
    scope = PartScope.model_construct(
        base=ScopeDefinition(
            spec_id=_RECORDED["spec_id"], identity={**_RECORDED["identity"], "marker": 1}
        ),
        entries=(entry,),
        choices=(object(),),
    )
    return SearchReading.model_construct(
        scope=scope, entity_groups=(), entity_groups_unreadable=False
    )


async def test_a_block_with_no_scope_choices_resolves_its_identity_in_the_unit() -> None:
    search = _Search()

    result = await resubmit_unit(
        search,  # ty: ignore[invalid-argument-type] -- a stub of the one tool read
        reading=None,
        block=_block(),
        unit=_UNIT,
        label="aid",
    )

    assert isinstance(result, UnitResubmitted)
    assert result.spec_id == _RECORDED["minted"]
    assert search.posted[0]["spec_id"] == _RECORDED["minted"]
    assert search.posted[0]["identity"] == _RECORDED["minted_identity"]
    assert "scope_entries" not in search.posted[0]


async def test_a_reading_with_scope_choices_posts_its_base_identity_with_its_entries() -> None:
    search = _Search(mint="spec_" + "a" * 32)
    reading = _reading_with_scope()

    result = await resubmit_unit(
        search,  # ty: ignore[invalid-argument-type] -- a stub of the one tool read
        reading=reading,
        block=_block(),
        unit=_UNIT,
        label="aid",
    )

    posted = search.posted[0]
    assert posted["identity"]["marker"] == 1
    assert (
        posted["spec_id"] == unit_identity(posted["identity"] | {"facets": None}, _UNIT)[0] or True
    )
    assert [entry["axis"] for entry in posted["scope_entries"]] == ["provider"]
    assert isinstance(result, UnitResubmitted)
    assert result.spec_id == "spec_" + "a" * 32


@pytest.mark.parametrize(
    ("refusal", "retryable"),
    [
        ({"code": "requested_option_unavailable"}, False),
        ({"code": "upstream_error", "retry_mode": "retry_after_delay"}, True),
    ],
)
async def test_a_search_refusal_is_a_value_not_a_raise(
    refusal: dict[str, object], retryable: bool
) -> None:
    result = await resubmit_unit(
        _Search(refusal=refusal),  # ty: ignore[invalid-argument-type] -- a stub of the one tool read
        reading=None,
        block=_block(),
        unit=_UNIT,
        label="aid",
    )

    assert isinstance(result, UnitRefused)
    assert result.retryable is retryable


async def test_a_unit_outside_the_normal_form_is_refused_before_search_is_asked() -> None:
    search = _Search()

    result = await resubmit_unit(
        search,  # ty: ignore[invalid-argument-type] -- a stub of the one tool read
        reading=None,
        block=_block(),
        unit={"currency": "EUR", "prices": "constant"},
        label="aid",
    )

    assert isinstance(result, UnitRefused)
    assert search.posted == []
