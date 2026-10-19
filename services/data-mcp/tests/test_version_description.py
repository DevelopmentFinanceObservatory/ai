"""Tests for `data_mcp.version_description` and the description lines of the `get_data` digest.

The description bodies are dfo-spec-kit's accepting description vectors, copied
verbatim into `fixtures/descriptions/`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest
from common.materialisation import FrameRef
from data_mcp import get_data_tool
from data_mcp.frame_materialisation import MaterialisationOutcome
from data_mcp.get_data_digest import render_get_data_digest, summed_family
from data_mcp.manifest import Manifest, SeriesStatisticsModel, strip_manifest
from data_mcp.version_description import VersionDescription

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _problem_body, text_of
from .test_get_data_digest import _frame_ref, _manifest_payload, _series
from .test_get_data_tool import (
    Handler,
    _call_get_data,
    _manifest_response,
    _stored_definition,
    _StubStore,
    _summed_manifest_response,
    mint_pair,
)

# The root conftest.py chdirs every test into a tmp_path, so this must be absolute.
_VECTORS = Path(__file__).resolve().parent / "fixtures" / "descriptions"

_MANIFEST_CITATION_LINE = "Source: OECD DAC2a (OECD DAC2a)."


def _vector(name: str) -> dict[str, Any]:
    return json.loads((_VECTORS / f"{name}.json").read_text())["input"]


def _render(description: VersionDescription) -> str:
    manifest = strip_manifest(_manifest_payload())
    return render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        description=description,
    )


def _summed(vector: dict[str, Any]) -> dict[str, Any]:
    """`vector` as a summed version's description: one `sum_variables` step, the shape
    dfo-spec-kit's `steps-mint-choices` vector gives it, with the composer's sentence."""
    return {
        **vector,
        "steps": [
            {
                "id": "m1",
                "op": "sum_variables",
                "policy": "sum_family_members",
                "summed": {
                    "count": 2,
                    "variables": [
                        {"ref": "dfo/ODA_Grants", "label": "ODA grants"},
                        {"ref": "dfo/ODA_Loans", "label": "ODA loans"},
                    ],
                },
                "text": "Summed 2 variables: ODA grants and ODA loans.",
            }
        ],
    }


def _without_cite_text() -> dict[str, Any]:
    vector = _vector("description-worked-example")
    del vector["cite"]["text"]
    return vector


# Rendering


def test_the_worked_example_renders_its_citation_latest_period_and_unknowns() -> None:
    vector = _vector("description-worked-example")
    digest = _render(VersionDescription.model_validate(vector))

    sections = digest.split("\n\n")
    assert (
        "Source: OECD Creditor Reporting System (CRS) via ONE DFO, retrieved 2026-09-28; "
        "latest period 2024."
    ) in sections
    for source in vector["sources"]:
        assert digest.count(source["label"]) == 1
    assert (
        "Not known: measure definition, source release, source downloaded at mint, "
        "source pipeline, estimated values, provisional values."
    ) in sections
    assert _MANIFEST_CITATION_LINE not in sections
    assert not [s for s in sections if s.startswith(("Steps:", "Caveats:"))]


def test_a_versions_steps_render_as_their_sentences() -> None:
    vector = _vector("description-source-pipeline-steps")
    digest = _render(VersionDescription.model_validate(vector))

    step_text = " ".join(step["text"] for step in vector["steps"])
    assert f"Steps: {step_text}" in digest.split("\n\n")


def test_a_versions_caveats_render_as_their_sentences() -> None:
    vector = _vector("description-computed-quotient")
    digest = _render(VersionDescription.model_validate(vector))

    sections = digest.split("\n\n")
    assert (
        "Source: ONE DFO, computed 2026-10-11 from Gross ODA disbursements and "
        "Gross national income; latest period 2024."
    ) in sections
    assert f"Caveats: {vector['caveats'][0]['text']}" in sections
    assert not [s for s in sections if s.startswith("Not known:")]


def test_caveats_past_the_cap_are_counted() -> None:
    vector = _vector("description-computed-quotient")
    vector["caveats"] = [{**vector["caveats"][0], "text": f"Caveat {n}."} for n in range(7)]
    digest = _render(VersionDescription.model_validate(vector))

    assert (
        "Caveats: Caveat 0. Caveat 1. Caveat 2. Caveat 3. Caveat 4. 2 more caveats not shown."
    ) in digest.split("\n\n")


def test_a_summed_versions_family_is_named_by_its_descriptions_title() -> None:
    spec_id = "spec_" + "a" * 32
    payload = _manifest_payload(
        spec_id=spec_id, dimensions=[{"label": "recipient", "count": 1, "axis": "destination"}]
    )
    payload["series_count"] = 1
    manifest = strip_manifest(payload)
    series_stats = [
        SeriesStatisticsModel.model_validate(
            _series(
                stat_var=f"dcid:{spec_id}", entities=[{"role": "recipient", "ref": "country/NGA"}]
            )
        )
    ]
    description = VersionDescription.model_validate(_summed(_vector("description-worked-example")))
    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        series_stats=series_stats,
        description=description,
        summed=summed_family(
            manifest, frame_summed=None, series_stats=series_stats, description=description
        ),
    )

    assert digest.splitlines()[0].startswith(
        "Retrieved Gross disbursements to Nigeria for health for "
    )
    assert digest.splitlines()[0].endswith(
        "with its StatVars summed into one total series per recipient."
    )
    assert "\n\nSummed total of Gross disbursements to Nigeria for health · " in digest
    assert "Steps: Summed 2 variables: ODA grants and ODA loans." in digest.split("\n\n")


# The read, through `get_data`


@pytest.fixture(autouse=True)
def _stub_materialise_and_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifests here carry `content_version="cv_1"`, which the real
    `materialise_and_register` refuses, so a no-op success stands in for it."""

    async def _stub(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", _stub)


def _mint_and_description_handler(
    description: httpx2.Response | type[httpx2.TransportError],
    paths: list[str] | None = None,
) -> tuple[Handler, str]:
    """A handler answering the mint, the Datasets capabilities probe, and the
    description route with `description`, or raising it when it is an exception
    type; and the minted `spec_id`."""
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        if request.url.path == "/api/datasets/capabilities":
            return httpx2.Response(200, json={})
        if request.url.path.endswith("/description"):
            if paths is not None:
                paths.append(request.url.path)
            if isinstance(description, httpx2.Response):
                return description
            raise description("read timed out", request=request)
        raise AssertionError(f"unexpected call to {request.url.path}")

    return handler, stored.spec_id


@pytest.mark.parametrize(
    ("description", "reason"),
    [
        (
            httpx2.Response(
                404,
                content=_problem_body(code="not_found", retryable=False),
                headers={"content-type": "application/problem+json"},
            ),
            "status=404 code=not_found",
        ),
        (
            httpx2.Response(
                410,
                content=_problem_body(code="revoked", retryable=False),
                headers={"content-type": "application/problem+json"},
            ),
            "status=410 code=revoked",
        ),
        (
            httpx2.Response(
                503,
                content=_problem_body(code="upstream_unavailable", retryable=True),
                headers={"content-type": "application/problem+json"},
            ),
            "status=503 code=upstream_unavailable",
        ),
        (httpx2.ReadTimeout, "no response"),
        (httpx2.Response(200, json=_without_cite_text()), "did not parse"),
    ],
    ids=["404", "410", "503", "timeout", "no-cite-text"],
)
async def test_a_failed_description_read_renders_todays_digest_and_logs_once(
    description: httpx2.Response | type[httpx2.TransportError],
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler, spec_id = _mint_and_description_handler(description)

    with caplog.at_level(logging.WARNING, logger="data_mcp.version_description"):
        result = await _call_get_data(
            handler, _StubStore(_stored_definition()), {"spec_id": spec_id}
        )

    assert result.is_error is False
    sections = text_of(result).split("\n\n")
    assert _MANIFEST_CITATION_LINE in sections
    assert not [s for s in sections if s.startswith("Not known:")]
    warnings = [r for r in caplog.records if r.name == "data_mcp.version_description"]
    assert len(warnings) == 1
    assert reason in warnings[0].getMessage()


async def test_the_read_is_keyed_by_dfo_apis_pair_not_the_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame minted with `options.columns` is held under a pair of its own; the
    description is apis' version's, read under the manifest's pair."""

    async def materialise(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        frame_ref = FrameRef.model_construct(
            spec_id=manifest.definition.spec_id, content_version="cv_frame_columns"
        )
        return MaterialisationOutcome(status="registered", attempts=0, frame_ref=frame_ref)

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)
    paths: list[str] = []
    handler, spec_id = _mint_and_description_handler(
        httpx2.Response(200, json=_vector("description-worked-example")), paths
    )

    result = await _call_get_data(handler, _StubStore(_stored_definition()), {"spec_id": spec_id})

    assert result.is_error is False
    assert paths == [f"/api/datasets/{spec_id}/cv_1/description"]
    assert "; latest period 2024." in text_of(result)


def _gapped_title(vector: dict[str, Any]) -> dict[str, Any]:
    return {**vector, "gaps": [*vector["gaps"], "title"]}


_NOT_FOUND = httpx2.Response(
    404,
    content=_problem_body(code="not_found", retryable=False),
    headers={"content-type": "application/problem+json"},
)


_UNAVAILABLE = httpx2.Response(
    503,
    content=_problem_body(code="service_unavailable", retryable=True),
    headers={"content-type": "application/problem+json"},
)


@pytest.mark.parametrize(
    ("frame_summed", "warning_count", "description", "count", "label"),
    [
        (
            True,
            None,
            httpx2.Response(200, json=_summed(_vector("description-worked-example"))),
            2,
            "Gross disbursements to Nigeria for health",
        ),
        (
            True,
            None,
            httpx2.Response(
                200, json=_gapped_title(_summed(_vector("description-worked-example")))
            ),
            2,
            None,
        ),
        (True, None, _UNAVAILABLE, 2, None),
        (True, 2, _NOT_FOUND, 2, None),
        (False, None, httpx2.Response(200, json=_vector("description-worked-example")), None, None),
    ],
    ids=[
        "described",
        "title-gap",
        "description-unavailable",
        "stored-before-descriptions",
        "frame-not-summed",
    ],
)
async def test_a_summed_result_reports_the_family_its_frame_summed(
    monkeypatch: pytest.MonkeyPatch,
    frame_summed: bool,
    warning_count: int | None,
    description: httpx2.Response,
    count: int | None,
    label: str | None,
) -> None:
    """A frame holding the family ref `dcid:<spec_id>` summed its family, whether or not
    the description read succeeds or the version predates descriptions. The count is the
    trace's StatVars with rows; only the name comes from the description's title, and
    without one the digest names the family by its cited sources, never by its ref."""
    stored = _stored_definition()

    async def materialise(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered",
            attempts=0,
            frame_ref=mint_pair(manifest),
            family_summed=frame_summed,
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_summed_manifest_response(
                    spec_id=stored.spec_id,
                    content_version="cv_1",
                    contributing_member_count=warning_count,
                    stat_var_label="French education grants",
                ),
            )
        if request.url.path == "/api/datasets/capabilities":
            return httpx2.Response(200, json={})
        if request.url.path.endswith("/description"):
            return description
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "stat_vars": "sum"}
    )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["summed_stat_var_count"] == count
    assert result.structured_content["summed_stat_var_label"] == label
    acknowledgement = text_of(result).splitlines()[0]
    assert acknowledgement.endswith("summed into one total series per entity.") is frame_summed
    assert f"dcid:{stored.spec_id}" not in acknowledgement
    if frame_summed and label is None:
        assert acknowledgement.startswith("Retrieved the OECD DAC2a family ")
