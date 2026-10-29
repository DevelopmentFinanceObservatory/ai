"""Tests for the tool-facing refusal grammar."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest
from data_mcp.errors import UpstreamError, UpstreamSurfaceUnavailable, UpstreamTransportError
from data_mcp.tool_errors import (
    _FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE,
    _RETRY_MODE_BY_CODE,
    _UPSTREAM_CEILING_KINDS,
    _UPSTREAM_CODE_MAP,
    ENTITY_CANDIDATE_CAP,
    EntityCandidate,
    FacetAlternative,
    FacetIssue,
    RetryMode,
    ToolErrorCode,
    ToolRefusal,
    from_upstream_error,
    to_tool_result,
)
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from .test_client import _client, _counting_handler, _problem_body


# 1. Table completeness.
def test_retry_mode_table_is_complete() -> None:
    """Every `ToolErrorCode` has exactly one `_RETRY_MODE_BY_CODE` entry, no stray keys."""
    assert set(_RETRY_MODE_BY_CODE) == set(ToolErrorCode)


# 2. The two same-`retryable`-different-mode pairs, proven end to end through
# two real parsed problem bodies each, not just table lookups.
async def test_same_upstream_retryable_flag_lands_on_different_modes() -> None:
    """Upstream's `retryable` flag alone can't pick a mode; the locked table can."""
    facet_exc = await _exc_from(422, code="facet_not_available", retryable=False)
    capability_exc = await _exc_from(501, code="capability_unavailable", retryable=False)
    # The shared premise: both parsed exceptions really do carry `retryable=False`.
    assert facet_exc.retryable is False
    assert capability_exc.retryable is False
    facet_refusal = from_upstream_error(facet_exc)
    capability_refusal = from_upstream_error(capability_exc)
    assert facet_refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert capability_refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert facet_refusal.retry_mode != capability_refusal.retry_mode

    version_exc = await _exc_from(503, code="upstream_version_changed", retryable=True)
    upstream_exc = await _exc_from(503, code="upstream_error", retryable=True)
    assert version_exc.retryable is True
    assert upstream_exc.retryable is True
    version_refusal = from_upstream_error(version_exc)
    upstream_refusal = from_upstream_error(upstream_exc)
    assert version_refusal.retry_mode == RetryMode.REFRESH_THEN_RETRY
    assert upstream_refusal.retry_mode == RetryMode.RETRY_AFTER_DELAY
    assert version_refusal.retry_mode != upstream_refusal.retry_mode


# 3. `ToolResult` round trip through a real MCP call.
async def test_tool_result_round_trip_via_call_tool() -> None:
    scratch = FastMCP(name="scratch")

    @scratch.tool
    async def refuse() -> ToolResult:
        return to_tool_result(
            ToolRefusal.entity_not_found(role="country", query="Freedonia", near_matches=["France"])
        )

    async with Client(scratch) as client:
        result = await client.call_tool("refuse", {}, raise_on_error=False)

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "entity_not_found"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"] == {
        "role": "country",
        "near_matches": ["France"],
        "had_near_matches": True,
        "not_held": False,
    }


# 4. Contrast: a raised `ToolError` carries no structured content.
async def test_raised_tool_error_carries_no_structured_content() -> None:
    scratch = FastMCP(name="scratch")

    @scratch.tool
    async def boom() -> None:
        raise ToolError("boom")

    async with Client(scratch) as client:
        result = await client.call_tool("boom", {}, raise_on_error=False)

    assert result.is_error is True
    assert result.structured_content is None


# 5. One concrete, non-vacuous message assertion per retry mode.
def test_retry_after_delay_message_names_the_wait() -> None:
    refusal = ToolRefusal.service_unavailable(service="dfo-apis", retry_after_s=1.0)
    assert "Retry after 1.0s" in refusal.message


def test_correct_and_retry_message_names_the_alternatives() -> None:
    refusal = ToolRefusal.facet_not_available(
        issues=[
            FacetIssue(
                axis="quarter",
                alternatives=[
                    FacetAlternative(ref="year", label="Year"),
                    FacetAlternative(ref="month", label="Month"),
                ],
                truncated=False,
            )
        ]
    )
    assert "quarter" in refusal.message
    assert "pick one of the listed alternatives" in refusal.message
    assert refusal.fix["issues"][0]["alternatives"] == [
        {"ref": "year", "label": "Year"},
        {"ref": "month", "label": "Month"},
    ]


def test_refresh_then_retry_message_says_to_run_the_request_again() -> None:
    refusal = ToolRefusal.upstream_version_changed(capability_version="v2")
    assert refusal.message == (
        "The source data refreshed while it was being retrieved, so the frame could not "
        "be built from one consistent version. Run the request again."
    )


def test_do_not_retry_message_says_check_capabilities_not_retry() -> None:
    refusal = ToolRefusal.capability_unavailable(detail="graph is not deployed")
    assert "retry" not in refusal.message
    assert "capabilities" in refusal.message


def test_upstream_response_unusable_carries_no_upstream_code_and_does_not_retry() -> None:
    # No `UpstreamError` backs this factory -- the call succeeded, only its
    # body couldn't be used -- so nothing sets `upstream_code`, unlike
    # `upstream_error` itself.
    refusal = ToolRefusal.upstream_response_unusable(detail="not valid JSON")
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.upstream_code is None
    assert refusal.retry_after_s is None
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert "not valid JSON" in refusal.message
    assert "marked this non-retryable" not in refusal.message


# 6. `from_upstream_error` mapping for the six upstream-exact codes plus the
# explicit `upstream_error` entry, each exercised through a real `UpstreamClient`
# call so the test hits the real parse path, not a hand-built `UpstreamError`.
async def _exc_from(status: int, **problem_kwargs: object) -> UpstreamError:
    _, handler = _counting_handler(httpx2.Response(status, content=_problem_body(**problem_kwargs)))
    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")
    return exc_info.value


async def _refusal_from(status: int, **problem_kwargs: object) -> ToolRefusal:
    return from_upstream_error(await _exc_from(status, **problem_kwargs))


async def _refusal_from_body(problem: dict[str, Any]) -> ToolRefusal:
    """The refusal an apis service response carrying exactly `problem` parses into."""
    response = httpx2.Response(problem["status"], content=json.dumps(problem).encode())
    _, handler = _counting_handler(response)
    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/operations/op_1", route="/api/datasets")
    return from_upstream_error(exc_info.value)


# apis' own bodies, rendered through the datasets code catalogue's `Envelope`.
# Each `sync` is the mint door's body for one refusal (`classify_definition_exception`).
# Its `poll` is the operation poll's body once the worker stored that refusal
# (`classify_meta_exception` over `OperationFailed(refusal=...)`): `operation_failed`,
# the refusal's code in `cause.code`, and the refusal's own detail, pointer, issues and
# ceiling fields. `legacy_over_ceiling` is the poll's body for a row with no stored
# refusal, the only shape an upstream without `datasets.operation.refusal` serves.
_FAILED_OPERATION_PROBLEMS: dict[str, dict[str, dict[str, Any]]] = json.loads(
    (Path(__file__).parent / "fixtures" / "failed_operation_problems.json").read_text()
)


async def test_facet_not_available_maps_through_a_real_parse() -> None:
    refusal = await _refusal_from(
        422,
        code="facet_not_available",
        retryable=False,
        issues=[
            {
                "axis": "unit",
                "alternatives": [{"ref": "USD", "label": "US Dollar"}],
                "truncated": False,
            }
        ],
    )
    assert refusal.code == ToolErrorCode.FACET_NOT_AVAILABLE
    assert refusal.fix == {
        "issues": [
            {
                "axis": "unit",
                "alternatives": [{"ref": "USD", "label": "US Dollar"}],
                "truncated": False,
                "code": None,
                "detail": None,
            }
        ],
        "detail": "because",
    }


async def test_facet_not_available_aggregates_multiple_axes() -> None:
    # apis evaluates all facet axes and aggregates every failing one into
    # a single `issues` list before raising (`test_facet_goldens.py`
    # `test_mixed_failure_aggregates_into_one_error_prioritising_undefined`).
    refusal = await _refusal_from(
        422,
        code="facet_not_available",
        retryable=False,
        issues=[
            {
                "axis": "unit",
                "alternatives": [{"ref": "USD", "label": "US Dollar"}],
                "truncated": False,
            },
            {
                "axis": "year",
                "alternatives": [{"ref": "2020", "label": "2020"}],
                "truncated": True,
            },
        ],
    )
    assert refusal.code == ToolErrorCode.FACET_NOT_AVAILABLE
    assert len(refusal.fix["issues"]) == 2
    assert [i["axis"] for i in refusal.fix["issues"]] == ["unit", "year"]
    assert "unit" in refusal.message
    assert "year" in refusal.message


async def test_facet_not_available_maps_the_search_surfaces_pointer_shape() -> None:
    # Search's `facet_not_available` sets only a top-level
    # `pointer`/`alternatives` pair, never `issues`
    # (`search/engine/region/facet_refusals.py`).
    refusal = await _refusal_from(
        422,
        code="facet_not_available",
        retryable=False,
        pointer="/input/facets/unit",
        alternatives=[{"ref": "USD", "label": "US Dollar"}],
    )
    assert refusal.code == ToolErrorCode.FACET_NOT_AVAILABLE
    assert len(refusal.fix["issues"]) == 1
    issue = refusal.fix["issues"][0]
    assert issue["axis"] == "unit"
    assert issue["alternatives"] == [{"ref": "USD", "label": "US Dollar"}]


async def test_facet_not_available_keeps_the_axis_when_no_alternatives_exist() -> None:
    # `search/engine/region/facet_refusals.py` sets `pointer`
    # unconditionally but collapses `alternatives` to `None` (omitted from
    # the wire) when the failing axis has zero live values.
    refusal = await _refusal_from(
        422, code="facet_not_available", retryable=False, pointer="/input/facets/unit"
    )
    assert refusal.code == ToolErrorCode.FACET_NOT_AVAILABLE
    assert refusal.fix["issues"] == [
        {"axis": "unit", "alternatives": [], "truncated": False, "code": None, "detail": None}
    ]
    # Search refuses an unmatched unit with no dataset or family in play, so the
    # message states the axis and the missing alternatives and nothing else.
    assert refusal.message == (
        "Facet unavailable on ['unit'], and no alternative values were listed."
    )


async def test_default_facet_undefined_maps_with_the_narrowed_contract() -> None:
    refusal = await _refusal_from(
        422, code="default_facet_undefined", retryable=False, issues=[{"axis": "provenance"}]
    )
    assert refusal.code == ToolErrorCode.DEFAULT_FACET_UNDEFINED
    # No `alternatives` reach `fix` — `DefaultFacetUndefinedError` takes no
    # `alternatives` parameter at all (`src/materialise/errors.py`).
    assert refusal.fix == {
        "issues": [
            {
                "axis": "provenance",
                "alternatives": [],
                "truncated": False,
                "code": None,
                "detail": None,
            }
        ],
        "detail": "because",
        "retryable": False,
    }


async def test_default_facet_undefined_aggregates_multiple_axes() -> None:
    refusal = await _refusal_from(
        422,
        code="default_facet_undefined",
        retryable=False,
        issues=[{"axis": "provenance"}, {"axis": "unit"}],
    )
    assert refusal.code == ToolErrorCode.DEFAULT_FACET_UNDEFINED
    assert [i["axis"] for i in refusal.fix["issues"]] == ["provenance", "unit"]


async def test_default_facet_undefined_relays_the_apis_refusal_without_a_cause_of_its_own() -> None:
    """The API's detail, axes and retryable ride in `fix` as sent; the message
    says only that the platform refused, with no cause or advice of data-mcp's own."""
    refusal = await _refusal_from(
        422,
        code="default_facet_undefined",
        retryable=False,
        detail="one or more facet axes could not be resolved",
        issues=[{"axis": "provenance"}, {"axis": "measurementMethod"}, {"axis": "unit"}],
    )
    assert [i["axis"] for i in refusal.fix["issues"]] == [
        "provenance",
        "measurementMethod",
        "unit",
    ]
    assert refusal.fix["detail"] == "one or more facet axes could not be resolved"
    assert refusal.fix["retryable"] is False
    assert refusal.trace_id == "trace-1"
    assert refusal.message == "The platform refused the requested data."


async def test_each_facet_issue_keeps_its_own_code_and_detail() -> None:
    """The platform sends each issue's own code and detail. Both facet refusals keep
    them, so a multi-axis refusal says why on each axis."""
    issues = [
        {
            "pointer": "/facets/unit",
            "axis": "unit",
            "alternatives": [{"ref": "USDollar", "label": "US Dollar"}],
            "truncated": False,
            "code": "facet_not_available",
            "detail": "facet axis 'unit' bound to 'Euro' has no observations",
        },
        {
            "pointer": "/facets/provenance",
            "axis": "provenance",
            "code": "default_facet_undefined",
            "detail": (
                "facet axis 'provenance' is unbound, no default policy is declared, "
                "and no probed candidate has any observations"
            ),
        },
    ]
    for code in ("default_facet_undefined", "facet_not_available"):
        refusal = await _refusal_from(422, code=code, retryable=False, issues=issues)
        assert [(i["axis"], i["code"], i["detail"]) for i in refusal.fix["issues"]] == [
            (i["axis"], i["code"], i["detail"]) for i in issues
        ]


async def test_default_facet_undefined_asks_for_a_change_only_when_an_issue_lists_values() -> None:
    """`retryable: false` means the same request fails again. A bound unit that emptied
    the pool lists the units that work, so the refusal asks for that change; with no
    listed values it stops."""
    with_values = await _refusal_from(
        422,
        code="default_facet_undefined",
        retryable=False,
        issues=[{"axis": "unit", "alternatives": [{"ref": "USDollar", "label": "US Dollar"}]}],
    )
    assert with_values.retry_mode is RetryMode.CORRECT_AND_RETRY
    without = await _refusal_from(
        422, code="default_facet_undefined", retryable=False, issues=[{"axis": "unit"}]
    )
    assert without.retry_mode is RetryMode.DO_NOT_RETRY


@pytest.mark.parametrize(
    ("retryable", "mode"),
    # `retryable` says whether the same request might succeed (apis
    # `dfo_platform/problem.py`), so `true` waits and repeats; it never asks for a change.
    [(False, RetryMode.DO_NOT_RETRY), (True, RetryMode.RETRY_AFTER_DELAY)],
)
async def test_default_facet_undefined_takes_its_retry_mode_from_the_api(
    retryable: bool, mode: RetryMode
) -> None:
    refusal = await _refusal_from(
        422, code="default_facet_undefined", retryable=retryable, issues=[{"axis": "unit"}]
    )
    assert refusal.retry_mode is mode
    assert refusal.fix["retryable"] is retryable


async def test_over_ceiling_carries_detail_verbatim() -> None:
    refusal = await _refusal_from(422, code="over_ceiling", retryable=False, ceiling_kind="rows")
    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix == {"detail": "because", "ceiling_kind": "rows"}
    assert "None" not in refusal.message
    assert "because" in refusal.message


def test_over_ceiling_message_and_fix_shape() -> None:
    refusal = ToolRefusal.over_ceiling(detail="5000 rows, 200 countries", ceiling_kind="rows")
    assert refusal.fix == {"detail": "5000 rows, 200 countries", "ceiling_kind": "rows"}
    assert "5000 rows, 200 countries" in refusal.message


@pytest.mark.parametrize(
    "detail",
    [
        "spec 'x' enumerates 5000 member(s) consistent with its own five-tuple and slots, "
        "exceeding member_cap=2000",
        "materialised coverage of 5000 rows exceeds the ceiling",
    ],
)
async def test_over_ceiling_with_no_ceiling_kind_names_no_kind_whatever_its_detail_says(
    detail: str,
) -> None:
    refusal = await _refusal_from(
        422, code="over_ceiling", retryable=False, detail=detail, count=5000, cap=2000
    )
    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix == {"detail": detail}
    assert refusal.message == f"Over the coverage ceiling: {detail}."


def test_over_ceiling_inserts_a_full_stop_between_detail_and_advice_when_missing() -> None:
    refusal = ToolRefusal.over_ceiling(
        detail="spec 'x' enumerates 5000 member(s) exceeding member_cap=2000",
        ceiling_kind="member_cardinality",
    )
    assert "member_cap=2000. This spec_id's definition has more member series" in refusal.message


def test_over_ceiling_does_not_double_the_full_stop_when_detail_already_ends_with_one() -> None:
    refusal = ToolRefusal.over_ceiling(
        detail="materialised coverage of 5000 rows exceeds the ceiling.",
        ceiling_kind="rows",
    )
    assert "ceiling.." not in refusal.message
    assert "ceiling. Narrow the scope and retry." in refusal.message


async def test_over_ceiling_reads_ceiling_kind_off_the_wire_extension() -> None:
    refusal = await _refusal_from(
        422,
        code="over_ceiling",
        retryable=False,
        detail="because",
        ceiling_kind="member_cardinality",
    )
    assert refusal.fix["ceiling_kind"] == "member_cardinality"
    assert "This spec_id's definition has more member series than one request can serve" in (
        refusal.message
    )


async def test_a_candidate_window_ceiling_states_its_remedy_and_asks_for_no_stat_vars() -> None:
    """A candidate window is a family too broad to enumerate unaided. The search
    digest names no StatVar, so the message asks for none and states the remedy."""
    refusal = await _refusal_from(
        422,
        code="over_ceiling",
        retryable=False,
        detail=(
            "spec 'x' enumerates against FiveTuple(...) past the 500-candidate reverse-arc "
            "window this deployment reads: this read returned 500 candidates of the 500 "
            "that window allows, and the member count past it is unknown, not zero."
        ),
        ceiling_kind="candidate_window",
        count=500,
        cap=500,
    )

    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix["ceiling_kind"] == "candidate_window"
    assert (refusal.fix["count"], refusal.fix["cap"]) == (500, 500)
    assert "too broad a family of series" in refusal.fix["remedy"]
    assert "has more member series than one request can serve" not in refusal.message
    assert "resolved_stat_var_dcids" not in refusal.message
    assert refusal.fix["remedy"] in refusal.message


# Every `ceiling_kind` apis' Datasets surface raises (each `OverCeilingError`
# raise site under `src/datasets_api` and `materialise`, and `mint_errors.py`'s
# SDMX cap classifier for a mint's `rows` and `bytes` caps), with the kind each
# reads as here.
_UPSTREAM_CEILING_KIND_PINS = {
    "rows": "rows",
    "sync_rows": "rows",
    "operation_rows": "rows",
    "bytes": "bytes",
    "sync_requests": "requests",
    "operation_requests": "requests",
    "member_cardinality": "member_cardinality",
    "candidate_window": "candidate_window",
}


def test_every_upstream_ceiling_kind_is_mapped_and_pinned() -> None:
    assert _UPSTREAM_CEILING_KINDS == _UPSTREAM_CEILING_KIND_PINS


@pytest.mark.parametrize("problem", ["sync", "poll"])
@pytest.mark.parametrize(("wire_kind", "kind"), sorted(_UPSTREAM_CEILING_KIND_PINS.items()))
async def test_an_upstream_ceiling_kind_reads_as_its_mapped_kind(
    wire_kind: str, kind: str, problem: str
) -> None:
    body = {**_FAILED_OPERATION_PROBLEMS["rows"][problem], "ceiling_kind": wire_kind}

    refusal = await _refusal_from_body(body)

    assert refusal.fix["ceiling_kind"] == kind
    assert (refusal.fix["count"], refusal.fix["cap"]) == (120000, 100000)
    narrows = kind in {"rows", "bytes", "requests"}
    assert refusal.message.endswith("Narrow the scope and retry.") is narrows
    assert ("remedy" in refusal.fix) is not narrows


@pytest.mark.parametrize("problem", ["sync", "poll"])
@pytest.mark.parametrize("wire_kind", ["values", "some_future_kind", "ROWS"])
async def test_an_unmapped_ceiling_kind_names_no_kind_and_no_count(
    wire_kind: str, problem: str
) -> None:
    """`values` is the SDMX availability value cap, which no Datasets mint path
    reaches, so it is unmapped like any kind a newer upstream adds."""
    body = {**_FAILED_OPERATION_PROBLEMS["rows"][problem], "ceiling_kind": wire_kind}

    refusal = await _refusal_from_body(body)

    assert refusal.fix == {"detail": body["detail"]}
    assert "Narrow the scope" not in refusal.message


async def test_over_ceiling_carries_count_and_cap_as_siblings_of_ceiling_kind() -> None:
    refusal = await _refusal_from(
        422,
        code="over_ceiling",
        retryable=False,
        ceiling_kind="rows",
        count=5000,
        cap=2000,
    )
    assert refusal.fix["count"] == 5000
    assert refusal.fix["cap"] == 2000


async def test_over_ceiling_omits_count_and_cap_when_the_wire_carries_neither() -> None:
    # The issue's own captured refusal: the enumeration couldn't page fast
    # enough to produce a count, so the fix omits the keys rather than
    # rendering an unknown count as zero.
    refusal = await _refusal_from(
        422, code="over_ceiling", retryable=False, ceiling_kind="member_cardinality"
    )
    assert "count" not in refusal.fix
    assert "cap" not in refusal.fix


async def test_capability_unavailable_maps_with_upstreams_detail() -> None:
    refusal = await _refusal_from(501, code="capability_unavailable", retryable=False)
    assert refusal.code == ToolErrorCode.CAPABILITY_UNAVAILABLE
    assert refusal.fix == {"detail": "because"}


async def test_spec_not_exportable_maps_with_upstreams_detail_and_does_not_retry() -> None:
    # `SpecNotExportable` (`datasets_api/doors/mint_errors.py`, 409): the identity
    # enumerates no member consistent with its five-tuple and slots.
    refusal = await _refusal_from(409, code="spec_not_exportable", retryable=False)
    assert refusal.code == ToolErrorCode.SPEC_NOT_EXPORTABLE
    assert refusal.fix == {"detail": "because"}
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


# The 422 `datasets_api/doors/mint_errors.py::_classify_rollup_not_summable` sends for a
# family whose gross and net members share every decoded code (`groupby.py`).
_ROLLUP_REMEDY = "Ask about one of the series it combines instead of the total."

_ROLLUP_DETAIL = (
    "groupby axis 'stat_var' refused: StatVar(s) Amount_Gross, Amount_Net decode to the same "
    "values across ('crsPurpose',) but are distinct measures; collapsing stat_var would merge "
    "them with no discriminating column left"
)


async def test_rollup_not_summable_maps_with_its_axis_and_detail_and_does_not_retry() -> None:
    refusal = await _refusal_from(
        422,
        code="rollup_not_summable",
        retryable=False,
        detail=_ROLLUP_DETAIL,
        pointer="/options/groupby",
        issues=[{"pointer": "/options/groupby", "axis": "stat_var"}],
    )
    assert refusal.code == ToolErrorCode.ROLLUP_NOT_SUMMABLE
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert refusal.fix == {
        "axes": ["stat_var"],
        "detail": _ROLLUP_DETAIL,
        "remedy": _ROLLUP_REMEDY,
    }
    # Written for the user: why there is no total, and what to ask in its place.
    # The upstream detail stays in `fix`.
    assert refusal.message == (
        "The series in this dataset can't be added into one total: a total could count "
        f"the same amount twice or merge different measures. {_ROLLUP_REMEDY} Repeating "
        'the call refuses the same way. A get_data call with stat_vars "separate" returns '
        "the series one by one."
    )
    assert refusal.trace_id == "trace-1"


async def test_rollup_not_summable_across_sources_offers_a_breakdown_by_source() -> None:
    refusal = await _refusal_from(
        422,
        code="rollup_not_summable",
        retryable=False,
        issues=[{"pointer": "/options/groupby", "axis": "provenance"}],
    )
    assert refusal.fix["axes"] == ["provenance"]
    assert refusal.fix["remedy"] == (
        "Ask about one of the series it combines instead of the total, "
        "or for the figures broken down by source."
    )


@pytest.mark.parametrize(
    "code",
    [
        "rollup_not_summable",
        "default_facet_undefined",
        "facet_not_available",
        "invalid_role_ref",
        "definition_not_reproducible",
    ],
)
@pytest.mark.parametrize(
    "issues",
    [[], ["stat_var"], [None, 3], "stat_var", {"axis": "stat_var"}],
    ids=["empty", "strings", "scalars", "not-a-list", "an-object"],
)
async def test_an_issues_member_that_holds_no_object_still_reaches_its_refusal(
    code: str, issues: object
) -> None:
    """apis sends `issues[]` as a list of objects. A body that breaks that
    shape still refuses with its own code, with the fields an issue would have
    set left empty."""
    refusal = await _refusal_from(422, code=code, retryable=False, issues=issues)
    assert refusal.code == _UPSTREAM_CODE_MAP[code]
    assert refusal.fix.get("axes", []) == []
    assert refusal.fix.get("issues", []) == []
    assert "role" not in refusal.fix
    assert "spec_id" not in refusal.fix


@pytest.mark.parametrize(
    "code", ["rollup_not_summable", "default_facet_undefined", "facet_not_available"]
)
@pytest.mark.parametrize("axis", [3, ["stat_var"], {"name": "stat_var"}])
async def test_an_issue_whose_axis_is_no_string_names_no_axis(code: str, axis: object) -> None:
    refusal = await _refusal_from(422, code=code, retryable=False, issues=[{"axis": axis}])
    assert refusal.code == _UPSTREAM_CODE_MAP[code]
    assert refusal.fix.get("axes", []) == []
    assert all(issue["axis"] is None for issue in refusal.fix.get("issues", []))
    assert str(axis) not in refusal.message


async def test_the_first_issue_is_the_first_object_whatever_precedes_it() -> None:
    refusal = await _refusal_from(
        422,
        code="invalid_role_ref",
        retryable=False,
        issues=["x", {"pointer": "/scope/entities/donor", "axis": "donor"}],
    )
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.fix["pointer"] == "/scope/entities/donor"
    assert refusal.fix["role"] == "donor"


async def test_a_malformed_issue_is_skipped_and_the_objects_beside_it_are_read() -> None:
    refusal = await _refusal_from(
        422,
        code="default_facet_undefined",
        retryable=False,
        issues=["unit", {"axis": "provenance"}, {"axis": None}],
    )
    assert [i["axis"] for i in refusal.fix["issues"]] == ["provenance", None]


async def test_facet_alternatives_that_are_not_objects_are_skipped() -> None:
    refusal = await _refusal_from(
        422,
        code="facet_not_available",
        retryable=False,
        issues=[{"axis": "unit", "alternatives": ["USD", {"ref": "EUR", "label": "Euro"}]}],
    )
    assert refusal.fix["issues"] == [
        {
            "axis": "unit",
            "alternatives": [{"ref": "EUR", "label": "Euro"}],
            "truncated": False,
            "code": None,
            "detail": None,
        }
    ]


async def test_a_failed_operation_caused_by_rollup_not_summable_keeps_the_typed_refusal() -> None:
    """A summed mint apis ran as an operation reports its refusal as `cause.code`
    alone, with no `issues` and a `detail` that names the operation."""
    refusal = await _refusal_from(
        500,
        code="operation_failed",
        retryable=False,
        detail="operation 'op_1' did not complete successfully: rollup_not_summable",
        cause={"code": "rollup_not_summable"},
    )
    assert refusal.code == ToolErrorCode.ROLLUP_NOT_SUMMABLE
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert refusal.upstream_code is None
    assert refusal.fix == {"axes": [], "detail": None, "remedy": _ROLLUP_REMEDY}
    assert refusal.message == (
        "The series in this dataset can't be added into one total: a total could count "
        f"the same amount twice or merge different measures. {_ROLLUP_REMEDY} Repeating "
        'the call refuses the same way. A get_data call with stat_vars "separate" returns '
        "the series one by one."
    )


async def test_upstream_version_changed_maps_to_refresh_then_retry() -> None:
    refusal = await _refusal_from(503, code="upstream_version_changed", retryable=True)
    assert refusal.code == ToolErrorCode.UPSTREAM_VERSION_CHANGED
    assert refusal.retry_mode == RetryMode.REFRESH_THEN_RETRY


async def test_open_role_unavailable_maps_to_a_fresh_search() -> None:
    refusal = await _refusal_from(
        422, code="open_role_unavailable", retryable=False, pointer="/input/identity/open_roles"
    )
    assert refusal.code == ToolErrorCode.OPEN_ROLE_UNAVAILABLE
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.upstream_code is None
    assert refusal.fix == {
        "pointer": "/input/identity/open_roles",
        "detail": "because",
        "next_step": "search",
    }


async def test_service_unavailable_maps_through_a_real_parse() -> None:
    # apis carries no `store`/`service` extension key at any raise site
    # for this code, so the mapped path uses a fixed label rather than
    # reading one off the wire; most raise sites do name what's backed up
    # in `detail`, which the mapped path threads through.
    refusal = await _refusal_from(
        503, code="service_unavailable", retryable=True, detail="registry write pool exhausted"
    )
    assert refusal.code == ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.fix == {
        "service": "a dfo-apis dependency",
        "detail": "registry write pool exhausted",
    }
    assert "registry write pool exhausted" in refusal.message
    # `upstream_code` stays unset, distinguishing this from the
    # `upstream_error` fallback.
    assert refusal.upstream_code is None


async def test_request_invalid_maps_and_surfaces_the_pointer() -> None:
    refusal = await _refusal_from(
        400, code="request_invalid", retryable=False, pointer="/input/parts/0/variable_text"
    )
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {"pointer": "/input/parts/0/variable_text", "detail": "because"}
    assert "/input/parts/0/variable_text" in refusal.message


async def test_request_invalid_without_a_pointer_still_names_the_detail() -> None:
    refusal = await _refusal_from(400, code="request_invalid", retryable=False)
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.fix == {"pointer": None, "detail": "because"}
    assert "because" in refusal.message


async def test_invalid_role_ref_maps_to_request_invalid_with_the_role() -> None:
    # `mint.py`'s `InvalidRoleRef` branch: no top-level `pointer`, one
    # `issues[]` entry naming the offending scope role in both `pointer` and
    # `axis`.
    refusal = await _refusal_from(
        422,
        code="invalid_role_ref",
        retryable=False,
        issues=[
            {
                "pointer": "/input/scope/entities/recipient",
                "member": "recipient",
                "axis": "recipient",
            }
        ],
    )
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {
        "pointer": "/input/scope/entities/recipient",
        "detail": "because",
        "role": "recipient",
    }
    assert "recipient" in refusal.message


async def test_role_incompatible_maps_to_request_invalid_with_the_role() -> None:
    # Same `issues[]` shape as `invalid_role_ref` (`mint.py`'s
    # `RoleIncompatible` branch), a different upstream code for a different
    # failure reason but the identical wire shape and fix.
    refusal = await _refusal_from(
        422,
        code="role_incompatible",
        retryable=False,
        issues=[{"pointer": "/input/scope/entities/donor", "member": "donor", "axis": "donor"}],
    )
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {
        "pointer": "/input/scope/entities/donor",
        "detail": "because",
        "role": "donor",
    }


async def test_definition_not_reproducible_maps_to_request_invalid_with_the_spec_id() -> None:
    # `mint_errors.py`'s `DefinitionNotReproducible` branch: no top-level
    # `pointer`, one issue naming the spec_id as `member`, and one more per
    # undeclared role naming it as `axis`.
    refusal = await _refusal_from(
        422,
        code="definition_not_reproducible",
        retryable=False,
        detail="spec 'spec-abc' binds entity role(s) recipient that 'dcid:exemplar' no "
        "longer declares",
        issues=[{"member": "spec-abc"}, {"member": "spec-abc", "axis": "recipient"}],
    )
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {
        "spec_id": "spec-abc",
        "upstream_code": "definition_not_reproducible",
        "role_bound": True,
        "roles": ["recipient"],
        "next_tool": "search",
    }
    assert refusal.message == (
        "The stored definition could not be rebuilt for this mint because it binds the "
        "entity role(s) recipient, which the source no longer declares. Run search again "
        "without the role phrase and use the spec_id it returns. Or call get_data again "
        "with scope.entities, leaving out recipient."
    )


@pytest.mark.parametrize("problem", ["sync", "poll"])
async def test_definition_not_reproducible_names_the_undeclared_role_however_the_mint_ran(
    problem: str,
) -> None:
    body = _FAILED_OPERATION_PROBLEMS["definition_not_reproducible"][problem]

    refusal = await _refusal_from_body(body)

    assert refusal.fix["roles"] == ["recipient"]
    assert refusal.message.endswith("with scope.entities, leaving out recipient.")


async def test_definition_not_reproducible_names_each_role_once() -> None:
    refusal = await _refusal_from(
        422,
        code="definition_not_reproducible",
        retryable=False,
        issues=[
            {"member": "spec-a"},
            {"member": "spec-a", "axis": "recipient"},
            {"member": "spec-a", "axis": "donor"},
            {"member": "spec-b", "axis": "recipient"},
        ],
    )
    assert refusal.fix["roles"] == ["recipient", "donor"]
    assert refusal.message.endswith("leaving out recipient, donor.")


async def test_definition_not_reproducible_with_no_role_offers_only_search() -> None:
    refusal = await _refusal_from(
        422, code="definition_not_reproducible", retryable=False, issues=[{"member": "spec-abc"}]
    )
    assert "roles" not in refusal.fix
    assert "scope.entities" not in refusal.message
    assert refusal.message.endswith("use the spec_id it returns.")


async def test_definition_not_reproducible_omits_spec_id_when_no_issue_is_carried() -> None:
    refusal = await _refusal_from(422, code="definition_not_reproducible", retryable=False)
    assert refusal.code == ToolErrorCode.REQUEST_INVALID
    assert refusal.fix == {
        "upstream_code": "definition_not_reproducible",
        "role_bound": True,
        "next_tool": "search",
    }


async def test_definition_not_reproducible_message_differs_from_upstream_errors() -> None:
    definition_refusal = await _refusal_from(
        422, code="definition_not_reproducible", retryable=False, issues=[{"member": "spec-abc"}]
    )
    graph_read_refusal = await _refusal_from(502, code="upstream_error", retryable=True)

    assert definition_refusal.message != graph_read_refusal.message
    assert definition_refusal.code == ToolErrorCode.REQUEST_INVALID
    assert graph_read_refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert "Retry after" not in definition_refusal.message
    assert "Run search again without the role phrase" in definition_refusal.message
    assert "Run search again without the role phrase" not in graph_read_refusal.message


def test_request_invalid_with_role_names_it_in_the_fix_and_message() -> None:
    refusal = ToolRefusal.request_invalid(
        pointer="/input/scope/entities/country", detail="not an accepted role", role="country"
    )
    assert refusal.fix == {
        "pointer": "/input/scope/entities/country",
        "detail": "not an accepted role",
        "role": "country",
    }
    assert "country" in refusal.message


async def test_parquet_size_bound_maps_to_over_ceiling() -> None:
    # `materialise/formats.py`'s `ParquetSizeBoundError`: `detail` only, no
    # structured fields beyond `ceiling_kind`, exactly like `over_ceiling` itself.
    refusal = await _refusal_from(422, code="parquet_size_bound", retryable=False)
    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {"detail": "because", "ceiling_kind": "bytes"}


async def test_explicit_upstream_error_code_maps_to_itself() -> None:
    refusal = await _refusal_from(503, code="upstream_error", retryable=True)
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.upstream_code == "upstream_error"


# 7. An unmapped upstream code surfaces its own code rather than being swallowed.
async def test_unmapped_upstream_code_surfaces_verbatim() -> None:
    refusal = await _refusal_from(422, code="something_never_seen", retryable=True)
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.upstream_code == "something_never_seen"


async def test_unmapped_non_retryable_code_does_not_claim_a_numeric_wait() -> None:
    # e.g. apis' real `resolution_ambiguous`, `retryable: false` per
    # `src/dfo_platform/codes.py`, so a repeated call won't succeed.
    refusal = await _refusal_from(422, code="resolution_ambiguous", retryable=False)
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.retry_after_s is None
    assert "Retry after" not in refusal.message
    assert "non-retryable" in refusal.message
    # `_RETRY_MODE_BY_CODE` alone would say `retry_after_delay` here, which
    # contradicts the message above -- no wait exists to claim.
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


async def test_member_not_resolvable_from_the_graph_lists_the_unresolved_members_in_fix() -> None:
    """Inconsistent arcs for a walked candidate."""
    refusal = await _refusal_from(
        422,
        code="member_not_resolvable",
        retryable=False,
        issues=[{"member": "dcid:Count_Person_Odd"}],
    )

    assert refusal.code == ToolErrorCode.MEMBER_NOT_RESOLVABLE
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert refusal.fix["members"] == ["dcid:Count_Person_Odd"]
    assert "indicators the platform cannot confirm" in refusal.fix["remedy"]
    assert "indicators the platform cannot confirm (1 the graph" in refusal.message
    assert "Count_Person_Odd" not in refusal.message
    assert "StatVar" not in refusal.message


async def test_member_not_resolvable_naming_the_spec_id_names_no_stat_var() -> None:
    """The identity as a whole did not resolve, so the issue's member is the spec_id."""
    spec_id = "spec_567bf6828e3cc23c7d70e121cc36f2f9"
    refusal = await _refusal_from(
        422,
        code="member_not_resolvable",
        retryable=False,
        issues=[{"member": spec_id, "pointer": "/input/spec_id"}],
    )

    assert refusal.code == ToolErrorCode.MEMBER_NOT_RESOLVABLE
    assert refusal.fix["members"] == [spec_id]
    assert "indicators the platform cannot confirm, so" in refusal.message
    assert spec_id not in refusal.message


async def test_a_failed_operation_caused_by_member_not_resolvable_names_no_stat_var() -> None:
    """A failed operation's body carries its cause's code and none of its issues."""
    refusal = from_upstream_error(
        await _exc_from(
            500,
            code="operation_failed",
            retryable=False,
            cause={"code": "member_not_resolvable"},
        )
    )

    assert refusal.code == ToolErrorCode.MEMBER_NOT_RESOLVABLE
    assert refusal.fix["members"] == []
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


async def test_failed_operation_is_relayed_under_its_cause_code() -> None:
    exc = await _exc_from(
        500, code="operation_failed", retryable=False, cause={"code": "entities_unbound"}
    )
    assert exc.code == "operation_failed"
    refusal = from_upstream_error(exc)
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.upstream_code == "entities_unbound"
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


async def test_failed_operation_without_a_cause_keeps_its_own_code() -> None:
    refusal = await _refusal_from(500, code="operation_failed", retryable=False)
    assert refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert refusal.upstream_code == "operation_failed"


@pytest.mark.parametrize(
    "cause_code",
    sorted(
        code
        for code in _UPSTREAM_CODE_MAP
        if code != "upstream_error" and code not in _FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE
    ),
)
async def test_failed_operation_reaches_its_causes_typed_factory(cause_code: str) -> None:
    """A deferred mint fails with `operation_failed`, naming the refusal that
    failed it in `cause.code`, so every mapped cause but the ones retried in
    place refuses with the code and retry mode of the synchronous mint."""
    failed = await _refusal_from(
        500, code="operation_failed", retryable=False, cause={"code": cause_code}
    )
    synchronous = await _refusal_from(422, code=cause_code, retryable=False)
    assert failed.code == _UPSTREAM_CODE_MAP[cause_code]
    assert failed.code != ToolErrorCode.UPSTREAM_ERROR
    assert failed.upstream_code is None
    assert (failed.code, failed.retry_mode) == (synchronous.code, synchronous.retry_mode)


def test_upstream_version_changed_is_the_one_failed_operation_cause_retried_in_place() -> None:
    assert sorted(_FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE) == ["upstream_version_changed"]


@pytest.mark.parametrize("retryable", [True, False])
async def test_a_failed_operation_caused_by_upstream_version_changed_is_retried_in_place(
    retryable: bool,
) -> None:
    """Submitting the same mint again fixes a source revision that moved
    mid-materialisation, so the failed operation says `retry_after_delay`
    with a wait, where the synchronous refusal says `refresh_then_retry`."""
    detail = "operation 'op_1' did not complete successfully: upstream_version_changed"
    failed = await _refusal_from(
        500,
        code="operation_failed",
        retryable=retryable,
        detail=detail,
        cause={"code": "upstream_version_changed"},
    )
    synchronous = await _refusal_from(503, code="upstream_version_changed", retryable=True)

    assert failed.code == ToolErrorCode.UPSTREAM_ERROR
    assert failed.upstream_code == "upstream_version_changed"
    assert failed.retry_mode == RetryMode.RETRY_AFTER_DELAY
    assert failed.retry_after_s == 1.0
    assert failed.fix == {"detail": detail}
    assert failed.message == (
        f"Upstream call failed (upstream_version_changed): {detail}. Retry after 1.0s."
    )
    assert synchronous.code == ToolErrorCode.UPSTREAM_VERSION_CHANGED
    assert synchronous.retry_mode == RetryMode.REFRESH_THEN_RETRY


@pytest.mark.parametrize(
    ("cause_code", "code", "fix", "message"),
    [
        (
            "request_invalid",
            ToolErrorCode.REQUEST_INVALID,
            {"pointer": None},
            "The request was invalid: {detail}. Correct the argument and retry.",
        ),
        (
            "invalid_role_ref",
            ToolErrorCode.REQUEST_INVALID,
            {"pointer": None},
            "The request was invalid: {detail}. Correct the argument and retry.",
        ),
        (
            "role_incompatible",
            ToolErrorCode.REQUEST_INVALID,
            {"pointer": None},
            "The request was invalid: {detail}. Correct the argument and retry.",
        ),
        (
            "capability_unavailable",
            ToolErrorCode.CAPABILITY_UNAVAILABLE,
            {},
            "This capability isn't live: {detail}. Check capabilities before calling this "
            "tool again.",
        ),
        (
            "service_unavailable",
            ToolErrorCode.SERVICE_UNAVAILABLE,
            {"service": "a dfo-apis dependency"},
            "a dfo-apis dependency is unreachable right now: {detail}. Retry after 1.0s.",
        ),
        (
            "spec_not_exportable",
            ToolErrorCode.SPEC_NOT_EXPORTABLE,
            {},
            "This spec can't be exported: {detail}. This is a dead end for this definition "
            "-- search again for a narrower or different ask.",
        ),
    ],
)
async def test_a_failed_operations_refusal_carries_the_operations_own_detail(
    cause_code: str, code: ToolErrorCode, fix: dict[str, object], message: str
) -> None:
    """The cause carries no detail of its own, so `fix.detail` and the message
    hold the poll route's sentence about the operation."""
    detail = f"operation 'op_1' did not complete successfully: {cause_code}"
    refusal = await _refusal_from(
        500, code="operation_failed", retryable=False, detail=detail, cause={"code": cause_code}
    )
    assert refusal.code == code
    assert refusal.upstream_code is None
    assert refusal.fix == fix | {"detail": detail}
    assert refusal.message == message.format(detail=detail)


@pytest.mark.parametrize(
    ("cause_code", "ceiling_kind"), [("over_ceiling", None), ("parquet_size_bound", "bytes")]
)
async def test_a_failed_operations_ceiling_names_only_the_kind_its_code_fixes(
    cause_code: str, ceiling_kind: str | None
) -> None:
    """The cause carries no `ceiling_kind`, so only `parquet_size_bound`'s own
    code says which ceiling it is. An unnamed one leaves `fix` without a kind
    and its message without advice to narrow the scope."""
    refusal = await _refusal_from(
        500, code="operation_failed", retryable=False, cause={"code": cause_code}
    )
    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix.get("ceiling_kind") == ceiling_kind
    assert ("Narrow the scope" in refusal.message) is (ceiling_kind is not None)


@pytest.mark.parametrize(
    "name",
    [
        "rows",
        "operation_rows",
        "definition_not_reproducible",
        "rollup_not_summable",
        "facet_not_available",
    ],
)
async def test_a_failed_operation_carrying_its_stored_refusal_refuses_as_the_inline_mint(
    name: str,
) -> None:
    """The poll carries the stored refusal's own detail, issues and ceiling fields, so
    the refusal matches the one the same failure gets inline, field for field."""
    problems = _FAILED_OPERATION_PROBLEMS[name]

    failed = await _refusal_from_body(problems["poll"])
    inline = await _refusal_from_body(problems["sync"])

    assert failed == inline


@pytest.mark.parametrize(
    ("name", "count", "cap"), [("rows", 120000, 100000), ("operation_rows", 2500000, 2000000)]
)
async def test_a_failed_operation_over_a_row_ceiling_offers_to_narrow_the_scope(
    name: str, count: int, cap: int
) -> None:
    """`operation_rows` is the worker's own admission ceiling, which the inline path
    reads as a row ceiling too."""
    poll = _FAILED_OPERATION_PROBLEMS[name]["poll"]

    refusal = await _refusal_from_body(poll)

    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert refusal.fix == {
        "detail": poll["detail"],
        "ceiling_kind": "rows",
        "count": count,
        "cap": cap,
    }
    assert refusal.message.endswith("Narrow the scope and retry.")


async def test_a_failed_operation_with_no_stored_refusal_names_no_ceiling_kind() -> None:
    poll = _FAILED_OPERATION_PROBLEMS["legacy_over_ceiling"]["poll"]

    refusal = await _refusal_from_body(poll)

    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix == {"detail": poll["detail"]}
    assert "Narrow the scope" not in refusal.message


@pytest.mark.parametrize("problem", ["sync", "poll"])
@pytest.mark.parametrize("ceiling_kind", [None, 7, ["rows"], {"kind": "rows"}])
async def test_a_ceiling_kind_that_is_not_a_string_names_no_kind(
    ceiling_kind: object, problem: str
) -> None:
    body = {**_FAILED_OPERATION_PROBLEMS["rows"][problem], "ceiling_kind": ceiling_kind}

    refusal = await _refusal_from_body(body)

    assert "ceiling_kind" not in refusal.fix
    assert "Narrow the scope" not in refusal.message


async def test_a_failed_operation_with_a_count_and_no_ceiling_kind_names_no_kind() -> None:
    poll = dict(_FAILED_OPERATION_PROBLEMS["rows"]["poll"])
    del poll["ceiling_kind"]

    refusal = await _refusal_from_body(poll)

    assert "ceiling_kind" not in refusal.fix
    assert "Narrow the scope" not in refusal.message


@pytest.mark.parametrize("problem", ["sync", "poll"])
async def test_a_count_or_cap_that_is_not_an_integer_is_left_out(problem: str) -> None:
    body = {**_FAILED_OPERATION_PROBLEMS["rows"][problem], "count": "120000", "cap": True}

    refusal = await _refusal_from_body(body)

    assert refusal.fix["ceiling_kind"] == "rows"
    assert not refusal.fix.keys() & {"count", "cap"}


@pytest.mark.parametrize(
    ("cause_code", "message"),
    [
        (
            "facet_not_available",
            "Facet unavailable on ['the requested axis'], and no alternative values were listed.",
        ),
        (
            "definition_not_reproducible",
            "The stored definition could not be rebuilt for this mint because it binds an "
            "entity role the source no longer declares. Run search again without the role "
            "phrase and use the spec_id it returns.",
        ),
    ],
)
async def test_a_failed_operations_message_points_only_at_what_its_body_carries(
    cause_code: str, message: str
) -> None:
    """A failed operation's body holds `cause.code` and the poll route's own
    detail, so its message names no listed alternative and no role."""
    refusal = await _refusal_from(
        500,
        code="operation_failed",
        retryable=False,
        detail=f"operation 'op_1' did not complete successfully: {cause_code}",
        cause={"code": cause_code},
    )
    assert refusal.message == message


@pytest.mark.parametrize(
    ("code", "cause_code"),
    [("recipe_not_declared", "recipe_undeclared"), ("rate_coverage_gap", "currency_unmapped")],
)
async def test_a_problem_with_its_own_cause_keeps_its_code(code: str, cause_code: str) -> None:
    exc = await _exc_from(422, code=code, retryable=False, cause={"code": cause_code})
    assert exc.code == code
    assert exc.extensions["cause"] == {"code": cause_code}
    refusal = from_upstream_error(exc)
    assert refusal.upstream_code == code


async def test_a_mapped_problem_with_a_cause_still_reaches_its_typed_factory() -> None:
    refusal = await _refusal_from(
        422,
        code="over_ceiling",
        retryable=False,
        ceiling_kind="member_cardinality",
        cause={"code": "recipe_undeclared"},
    )
    assert refusal.code == ToolErrorCode.OVER_CEILING
    assert refusal.fix["ceiling_kind"] == "member_cardinality"


# 8. `UpstreamTransportError` and `UpstreamSurfaceUnavailable` routing, built
# directly with no network.
def test_transport_error_routes_to_service_unavailable() -> None:
    exc = UpstreamTransportError(detail="connection reset")
    exc.elapsed_s = 2.4

    refusal = from_upstream_error(exc)

    assert refusal.code == ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode == RetryMode.RETRY_AFTER_DELAY
    assert refusal.fix == {"service": "dfo-apis", "detail": "connection reset"}
    assert refusal.retry_after_s == 2.4


def test_surface_unavailable_routes_to_capability_unavailable() -> None:
    exc = UpstreamSurfaceUnavailable(
        prefix="/api/datasets", probe_path="/api/datasets/capabilities"
    )

    refusal = from_upstream_error(exc)

    assert refusal.code == ToolErrorCode.CAPABILITY_UNAVAILABLE
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert refusal.fix == {"detail": exc.detail}


# 9. `entity_not_found`/`entity_ambiguous` local factories.
def test_entity_not_found_distinguishes_near_miss_from_total_whiff() -> None:
    near_miss = ToolRefusal.entity_not_found(
        role="country", query="Fredonia", near_matches=["France"]
    )
    total_whiff = ToolRefusal.entity_not_found(role="country", query="Qwerty", near_matches=[])

    assert near_miss.fix["had_near_matches"] is True
    assert "Nearest" in near_miss.message
    assert total_whiff.fix["had_near_matches"] is False
    assert "nothing close" in total_whiff.message


def test_entity_not_found_distinguishes_not_held_from_no_match_by_field() -> None:
    """`not_held` must be readable as a field, not inferred from message prose."""
    no_match = ToolRefusal.entity_not_found(role="country", query="Qwerty", near_matches=[])
    not_held = ToolRefusal.entity_not_found(
        role="country", query="Andorra", near_matches=[], not_held=True
    )
    memberless = ToolRefusal.entity_not_found(
        role="country", query="Micronesia", near_matches=[], memberless_kind="subregion"
    )

    assert no_match.fix["not_held"] is False
    assert "entity_type" not in no_match.fix
    assert not_held.fix["not_held"] is True
    assert "entity_type" not in not_held.fix
    assert memberless.fix["not_held"] is True
    assert memberless.fix["entity_type"] == "subregion"


def test_entity_ambiguous_truncates_to_the_candidate_cap() -> None:
    candidates = [
        EntityCandidate(dcid=f"dcid/{i}", label=f"Country {i}", type="Country")
        for i in range(ENTITY_CANDIDATE_CAP + 3)
    ]

    refusal = ToolRefusal.entity_ambiguous(role="country", query="Georgia", candidates=candidates)

    assert len(refusal.fix["candidates"]) == ENTITY_CANDIDATE_CAP
    # The real, pre-cap count and a machine-readable partial-list flag both
    # survive the cap -- a caller can tell "5 shown" from "5 of 9 total".
    assert refusal.fix["candidate_count"] == ENTITY_CANDIDATE_CAP + 3
    assert refusal.fix["truncated"] is True


def test_entity_ambiguous_names_the_real_pre_cap_count_in_the_message() -> None:
    candidates = [
        EntityCandidate(dcid=f"dcid/{i}", label=f"Country {i}", type="Country") for i in range(9)
    ]

    refusal = ToolRefusal.entity_ambiguous(role="recipient", query="X", candidates=candidates)

    assert "5 of 9" in refusal.message
    assert refusal.fix["candidate_count"] == 9


def test_entity_ambiguous_names_each_candidate_as_the_marked_dcid_get_data_takes() -> None:
    """A client that shows the model only `content` (see `to_tool_result`) still needs
    the exact value to pass back, so the message carries each `fix` candidate's `entity`."""
    candidates = [
        EntityCandidate(dcid="country/GEO", label="Georgia (country)", type="Country"),
        EntityCandidate(dcid="GroupOfSeven", label="Group of Seven", type="group"),
    ]

    refusal = ToolRefusal.entity_ambiguous(role="country", query="G", candidates=candidates)

    entities = [c["entity"] for c in refusal.fix["candidates"]]
    assert entities == ["dcid:country/GEO", "dcid:GroupOfSeven"]
    assert [c["dcid"] for c in refusal.fix["candidates"]] == ["country/GEO", "GroupOfSeven"]
    assert all(f"({entity})" in refusal.message for entity in entities)


def test_entity_ambiguous_stays_untruncated_under_the_cap() -> None:
    candidates = [EntityCandidate(dcid="dcid/1", label="Georgia", type="Country")]

    refusal = ToolRefusal.entity_ambiguous(role="recipient", query="Georgia", candidates=candidates)

    assert refusal.fix["truncated"] is False
    assert refusal.fix["candidate_count"] == 1
    assert "5 of" not in refusal.message


# 9b. `to_tool_result`'s `preamble` keyword.
def test_preamble_is_prepended_to_content_and_leaves_structured_content_identical() -> None:
    refusal = ToolRefusal.entity_not_found(role="country", query="Freedonia", near_matches=[])

    plain = to_tool_result(refusal)
    with_preamble = to_tool_result(refusal, preamble="Described 3 of 4 dcids.")

    plain_text = plain.content[0].text  # type: ignore[union-attr]
    with_preamble_text = with_preamble.content[0].text  # type: ignore[union-attr]
    assert with_preamble_text == f"Described 3 of 4 dcids.\n\n{plain_text}"
    assert with_preamble.structured_content == plain.structured_content


def test_no_preamble_leaves_content_unchanged() -> None:
    refusal = ToolRefusal.entity_not_found(role="country", query="Freedonia", near_matches=[])

    plain = to_tool_result(refusal)
    explicit_none = to_tool_result(refusal, preamble=None)
    assert plain.content[0].text == explicit_none.content[0].text  # type: ignore[union-attr]


async def test_an_upstream_trace_id_is_logged_and_left_out_of_the_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trace_id = "0af7651916cd43dd8448eb211c80319c"
    problem = json.loads(_problem_body(code="over_ceiling", retryable=False, trace_id=trace_id))
    refusal = await _refusal_from_body({"status": 422, **problem})
    assert refusal.trace_id == trace_id

    with caplog.at_level(logging.WARNING, logger="data_mcp.tool_errors"):
        result = to_tool_result(refusal)

    assert result.structured_content is not None
    assert result.structured_content["code"] == "over_ceiling"
    assert "trace_id" not in result.structured_content
    assert trace_id not in json.dumps(result.structured_content) + result.content[0].text  # type: ignore[union-attr]
    [record] = [r for r in caplog.records if r.name == "data_mcp.tool_errors"]
    assert trace_id in record.getMessage()


# 9c. `node_not_held`.
def test_node_not_held_names_every_dcid_in_the_message_and_retries_correct_and_retry() -> None:
    refusal = ToolRefusal.node_not_held(dcids=["country/ZZZ", "country/YYY"])

    assert refusal.code == ToolErrorCode.NODE_NOT_HELD
    assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert "country/ZZZ" in refusal.message
    assert "country/YYY" in refusal.message
    assert refusal.fix == {"dcids": ["country/ZZZ", "country/YYY"]}


# 10. `service_unavailable` vs. `definition_not_held`: same "store" family,
# different retry modes — one reports the store down, the other reports the
# id unregistered while the store is up.
def test_service_unavailable_and_definition_not_held_contrast_retry_modes() -> None:
    store_down = ToolRefusal.service_unavailable(service="a dfo-apis dependency", retry_after_s=3.0)
    id_not_registered = ToolRefusal.definition_not_held(definition_id="def-123")

    assert store_down.retry_mode == RetryMode.RETRY_AFTER_DELAY
    assert id_not_registered.retry_mode == RetryMode.CORRECT_AND_RETRY


# 11. Exact `fix` key set per factory.
def test_each_factory_produces_its_exact_fix_key_set() -> None:
    cases: list[tuple[ToolRefusal, set[str]]] = [
        (
            ToolRefusal.facet_not_available(
                issues=[FacetIssue(axis="quarter", alternatives=[], truncated=False)]
            ),
            {"issues", "detail"},
        ),
        (
            ToolRefusal.default_facet_undefined(issues=[], detail=None, retryable=False),
            {"issues", "detail", "retryable"},
        ),
        (
            ToolRefusal.over_ceiling(detail="5000 rows, 200 countries", ceiling_kind="rows"),
            {"detail", "ceiling_kind"},
        ),
        (ToolRefusal.capability_unavailable(detail="not live"), {"detail"}),
        (
            ToolRefusal.capability_unavailable(detail="not live", option="breakdown"),
            {"detail", "option"},
        ),
        (
            ToolRefusal.upstream_version_changed(capability_version="v2"),
            {"capability_version"},
        ),
        (
            ToolRefusal.entity_not_found(role="country", query="x", near_matches=[]),
            {"role", "near_matches", "had_near_matches", "not_held"},
        ),
        (
            ToolRefusal.entity_not_found(role="country", query="x", near_matches=[], not_held=True),
            {"role", "near_matches", "had_near_matches", "not_held"},
        ),
        (
            ToolRefusal.entity_not_found(
                role="country", query="x", near_matches=[], memberless_kind="subregion"
            ),
            {"role", "near_matches", "had_near_matches", "not_held", "entity_type"},
        ),
        (
            ToolRefusal.entity_ambiguous(
                role="country",
                query="x",
                candidates=[EntityCandidate(dcid="dcid/1", label="A", type="Country")],
            ),
            {"role", "candidates", "candidate_count", "truncated"},
        ),
        (
            ToolRefusal.upstream_error(upstream_code="x", detail="y", retry_after_s=1.0),
            {"detail"},
        ),
        (
            ToolRefusal.service_unavailable(service="dfo-apis", retry_after_s=1.0),
            {"service"},
        ),
        (
            ToolRefusal.service_unavailable(
                service="dfo-apis", retry_after_s=1.0, detail="registry write pool exhausted"
            ),
            {"service", "detail"},
        ),
        (
            ToolRefusal.definition_not_held(definition_id="def-1"),
            {"definition_id", "next_step"},
        ),
        (
            ToolRefusal.request_invalid(pointer="/input/parts/0", detail="empty variable_text"),
            {"pointer", "detail"},
        ),
        (
            ToolRefusal.request_invalid(
                pointer="/input/scope/entities/country", detail="bad role", role="country"
            ),
            {"pointer", "detail", "role"},
        ),
        (
            ToolRefusal.node_not_held(dcids=["country/ZZZ"]),
            {"dcids"},
        ),
        (
            ToolRefusal.spec_not_exportable(detail="no member survives"),
            {"detail"},
        ),
        (
            ToolRefusal.rollup_not_summable(axes=["stat_var"], detail="gross and net"),
            {"axes", "detail", "remedy"},
        ),
        (
            ToolRefusal.over_ceiling(detail="over the ceiling", ceiling_kind=None),
            {"detail"},
        ),
        (
            ToolRefusal.open_role_unavailable(pointer=None, detail=None),
            {"pointer", "detail", "next_step"},
        ),
    ]
    for refusal, expected_keys in cases:
        assert set(refusal.fix) == expected_keys


def test_mint_unanswered_is_a_retryable_service_unavailable_refusal() -> None:
    """A mint POST with no answer may have run inline, with no operation for a repeat
    call to rejoin, or never have reached apis, so the refusal promises neither."""
    refusal = ToolRefusal.mint_unanswered(
        spec_id="spec_" + "a" * 32, elapsed_s=104.26, retry_after_s=5.0, waits_out_sync_mint=True
    )

    assert refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refusal.retry_after_s == 5.0
    assert refusal.fix == {"service": "dfo-apis", "spec_id": "spec_" + "a" * 32, "elapsed_s": 104.3}
    assert "same spec_id and scope is safe" in refusal.message
    assert "otherwise the request runs again" in refusal.message


def test_mint_unanswered_under_a_short_mint_cap_promises_nothing_of_a_retry() -> None:
    """A mint cap under apis' own synchronous mint timeout can cut the same inline
    mint on every repeat, so the message says the same request is refused until the
    mint ends, that a later call runs it again, and names a narrower scope."""
    refusal = ToolRefusal.mint_unanswered(
        spec_id="spec_" + "a" * 32, elapsed_s=34.9, retry_after_s=85.1, waits_out_sync_mint=False
    )

    assert refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refusal.retry_after_s == 85.1
    assert refusal.fix == {"service": "dfo-apis", "spec_id": "spec_" + "a" * 32, "elapsed_s": 34.9}
    assert "may still be running it upstream for up to 85s more" in refusal.message
    assert "refuses the same spec_id and scope until then" in refusal.message
    assert "or one that reaches another instance, runs the request again" in refusal.message
    assert "runs the request again from the start" in refusal.message
    assert "narrower scope" in refusal.message
    assert "is safe" not in refusal.message
    assert refusal.message.endswith("Retry after 85.1s.")


def test_mint_joined_unanswered_is_retryable_and_names_the_wait() -> None:
    """The mint belongs to another call and this instance does not refuse a repeat, so
    the refusal carries the joiner's own wait, not a ledger hold."""
    refusal = ToolRefusal.mint_joined_unanswered(
        spec_id="spec_" + "a" * 32, waited_s=31.96, retry_after_s=5.0
    )

    assert refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refusal.retry_after_s == 5.0
    assert refusal.fix == {"service": "dfo-apis", "spec_id": "spec_" + "a" * 32, "waited_s": 32.0}


def test_mint_still_running_names_the_mint_and_the_wait() -> None:
    refusal = ToolRefusal.mint_still_running(
        spec_id="spec_" + "a" * 32, started_s_ago=40.04, retry_after_s=80.0
    )

    assert refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refusal.retry_after_s == 80.0
    assert refusal.fix == {
        "service": "dfo-apis",
        "spec_id": "spec_" + "a" * 32,
        "started_s_ago": 40.0,
    }
    assert "started a mint 40s ago" in refusal.message
    assert "or one that reaches another instance, runs the request again" in refusal.message
    assert "narrower scope" in refusal.message
    assert refusal.message.endswith("Retry after 80.0s.")


def test_mint_slots_full_names_the_limit_and_the_wait() -> None:
    refusal = ToolRefusal.mint_slots_full(limit=1, retry_after_s=0.5)

    assert refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refusal.retry_after_s == 0.5
    assert refusal.fix == {"service": "dfo-apis", "limit": 1}
    assert "as it allows at once (1)" in refusal.message
    assert refusal.message.endswith("Retry after 0.5s.")


def test_operation_still_processing_words_each_last_reported_status() -> None:
    def refusal(*, last_status: str, last_poll_error: str | None) -> ToolRefusal:
        return ToolRefusal.operation_still_processing(
            operation_id="op_1",
            last_status=last_status,
            last_poll_error=last_poll_error,
            elapsed_s=40.0,
            retry_after_s=2.0,
        )

    seen = refusal(last_status="running", last_poll_error=None)
    unconfirmed = refusal(last_status="pending", last_poll_error="service_unavailable")
    unfetched = refusal(last_status="failed", last_poll_error=None)

    assert "still being prepared upstream" in seen.message
    assert "last_poll_error" not in seen.fix
    assert "could not confirm the status" in unconfirmed.message
    assert "last reported it pending" in unconfirmed.message
    assert "still being prepared" not in unconfirmed.message
    assert unconfirmed.fix["last_poll_error"] == "service_unavailable"
    assert "dfo-apis reported operation 'op_1' failed, but this call could not fetch" in (
        unfetched.message
    )
    assert "may still be running" not in unfetched.message
    for each in (seen, unconfirmed, unfetched):
        assert each.code is ToolErrorCode.OPERATION_STILL_PROCESSING
        assert "same spec_id and scope" in each.message


# 12. The refusal envelope rendered into `content` itself, for a client that shows the
# model only `content` and never reads `structured_content`.
def test_to_tool_result_appends_the_envelope_as_one_line_of_json_after_a_blank_line() -> None:
    refusal = ToolRefusal.service_unavailable(service="dfo-apis", retry_after_s=3.0)

    result = to_tool_result(refusal)
    text = result.content[0].text  # type: ignore[union-attr]
    message, _, envelope_line = text.partition("\n\n")

    assert message == refusal.message
    assert "\n" not in envelope_line
    envelope = json.loads(envelope_line)
    assert envelope == {
        "code": "service_unavailable",
        "retry_mode": "retry_after_delay",
        "fix": refusal.fix,
        "retry_after_s": 3.0,
    }


def test_envelope_omits_retry_after_s_when_unset() -> None:
    refusal = ToolRefusal.capability_unavailable(detail="graph is not deployed")

    result = to_tool_result(refusal)
    _, _, envelope_line = result.content[0].text.partition("\n\n")  # type: ignore[union-attr]

    assert "retry_after_s" not in json.loads(envelope_line)


def test_preamble_still_precedes_the_message_with_the_envelope_appended() -> None:
    refusal = ToolRefusal.entity_not_found(role="country", query="Freedonia", near_matches=[])

    result = to_tool_result(refusal, preamble="Described 3 of 4 dcids.")
    text = result.content[0].text  # type: ignore[union-attr]

    assert text.startswith("Described 3 of 4 dcids.\n\n")
    _, message, envelope_line = text.split("\n\n")
    assert message == refusal.message
    assert json.loads(envelope_line)["code"] == "entity_not_found"


# 13. A pin on the longest refusal text this server can produce, derived from the one
# multiplicity cap `tool_errors.py` declares (`ENTITY_CANDIDATE_CAP`). `EntityCandidate`'s
# own fields (dcid/label/type) carry no length cap in this module, so the ceiling
# below assumes a worst-case field length rather than deriving one from an enforced bound,
# documented here rather than left implicit, so a future growth in either the candidate cap
# or this assumption updates the ceiling deliberately instead of the test silently going stale.
_WORST_CASE_FIELD_CHARS = 100


def test_entity_ambiguous_text_stays_under_a_byte_ceiling_derived_from_its_candidate_cap() -> None:
    candidates = [
        EntityCandidate(
            dcid="d" * _WORST_CASE_FIELD_CHARS,
            label="l" * _WORST_CASE_FIELD_CHARS,
            type="t" * _WORST_CASE_FIELD_CHARS,
        )
        for _ in range(ENTITY_CANDIDATE_CAP)
    ]
    refusal = ToolRefusal.entity_ambiguous(
        role="r" * _WORST_CASE_FIELD_CHARS,
        query="q" * _WORST_CASE_FIELD_CHARS,
        candidates=candidates,
    )

    text = to_tool_result(refusal).content[0].text  # type: ignore[union-attr]

    # Each shown candidate contributes its label and dcid to the message (2 fields) and
    # its full dataclass (3 fields) plus its marked `entity` (a dcid again) to
    # `fix["candidates"]`'s JSON, 6 field-lengths per candidate, plus a 100-byte allowance per candidate for scaffolding (JSON keys,
    # quotes, braces, the message's " (...)," punctuation). `role`/`query` (2 more
    # field-lengths) and a fixed allowance for the rest of the message and envelope
    # (codes, keys, the surrounding sentences) round it out.
    ceiling = (
        ENTITY_CANDIDATE_CAP * (_WORST_CASE_FIELD_CHARS * 6 + 100)
        + _WORST_CASE_FIELD_CHARS * 2
        + 500
    )
    assert len(text.encode("utf-8")) <= ceiling
