"""Tests for `data_mcp.manifest`: URL stripping, the `extra="ignore"`
policy, and the shapes `strip_manifest` accepts.

`_DEFINITION_MANIFEST`/`_COLLECTION_MANIFEST` were captured by driving
dfo-apis' own offline `TestClient` (`tests/datasets_api/conftest.py
::make_client`) through `POST /api/datasets`, the same way
`tests/datasets_api/test_mint_spec_id.py` and `test_mint_collection_id.py`
do it, against a `FakeArtifactStore` whose `signed_get_url` returns
`fake://artifact/<tier>/<digest>` -- pasted as JSON so a real production
shape backs every assertion here, including ones a hand-built fixture would
happily satisfy while describing something production never emits. Each
capture's inline `statistics` array is edited by hand, replaced by
`series_count` plus a `rendering: {"statistics": "series"}` (or, per member,
`{"statistics": "series", "member": <spec_id>}`) artifact entry -- the shape
dfo-apis uses for per-series figures stored in their own Parquet artifact;
nothing else in either capture is touched by hand.
"""

from __future__ import annotations

import copy
import json

import pytest
from data_mcp.manifest import (
    ARTIFACT_URI_TEMPLATE,
    Manifest,
    render_validation_errors,
    strip_manifest,
)
from data_mcp.series_statistics import find_statistics_artifact
from pydantic import ValidationError

_DEFINITION_MANIFEST = json.loads("""
{
  "artifacts": [
    {
      "artifact_digest": "sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "bytes": 167,
      "created_at": "2026-08-11T20:00:17.015350+00:00",
      "media_type": "text/csv",
      "rendering": {"bundle": false, "format": "csv"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "url_expires_at": "2026-08-11T20:15:17.015350+00:00"
    },
    {
      "artifact_digest": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
      "bytes": 512,
      "created_at": "2026-08-11T20:00:17.015350+00:00",
      "media_type": "application/vnd.apache.parquet",
      "rendering": {"statistics": "series"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:9999999999999999999999999999999999999999999999999999999999999999",
      "url_expires_at": "2026-08-11T20:15:17.015350+00:00"
    }
  ],
  "citation": {
    "sources": [{"label": "OECD DAC2a", "ref": "OECD_DAC2A"}],
    "text": "OECD DAC2a"
  },
  "content_version": "cv_6b2c86ce87fdd1f7ab7b7bbca9cbf066",
  "coverage": {
    "dimensions": [{"count": 1, "label": "recipient"}, {"count": 1, "label": "years"}],
    "rows": 1,
    "window": {"end_year": 2015, "kind": "all", "origin": "default", "start_year": 2015}
  },
  "definition": {
    "identity": {
      "entities": {"recipient": ["country/KEN"]},
      "facets": {
        "measurementMethod": {"ref": "GrantEquivalent", "state": "value"},
        "provenance": {"ref": "OECD_DAC2A", "state": "value"},
        "unit": {"ref": "USD", "state": "value"}
      },
      "identity_version": "spec1",
      "requested_window": {"kind": "all"},
      "shape": {"id": "MalariaGrantDisbursements", "kind": "stat_var"},
      "slots": []
    },
    "schema_version": 1,
    "spec_id": "spec_2dffbc390c1c97d9992ee162ce690b19"
  },
  "schema_version": 1,
  "series_count": 1,
  "trace": {
    "applied_window": {"end_year": 2015, "kind": "all", "origin": "default", "start_year": 2015},
    "capability_version": "dset_bbe643884591f64b",
    "facets": [
      {
        "axis": "provenance", "basis": "bound", "default_policy": null,
        "default_policy_version": null, "dispatch_facet_id": "facet1",
        "ref": "OECD_DAC2A", "selection_reason": null
      }
    ],
    "frame_grammar_version": "1.0.0",
    "graph_revision": "unavailable:dfo-dcp-revision-endpoint-absent",
    "materialiser_build": "datasets-materialiser-dev",
    "normalisations": [],
    "resolved_stat_vars": [{"ref": "dcid:MalariaGrantDisbursements", "roles": ["recipient"]}],
    "source_revisions": []
  },
  "warnings": [
    {
      "code": "processing_level_undeclared",
      "message": "A source StatVar lacks a processing-level declaration; the citation credits the source and omits the processing qualifier.",
      "severity": "info"
    }
  ]
}
""")

_COLLECTION_MANIFEST = json.loads("""
{
  "artifacts": [
    {
      "artifact_digest": "sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "bytes": 167,
      "created_at": "2026-08-11T20:00:17.023125+00:00",
      "media_type": "text/csv",
      "rendering": {"bundle": false, "format": "csv"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "url_expires_at": "2026-08-11T20:15:17.023125+00:00"
    },
    {
      "artifact_digest": "sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "bytes": 167,
      "created_at": "2026-08-11T20:00:17.023144+00:00",
      "media_type": "text/csv",
      "rendering": {"bundle": false, "format": "csv"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7",
      "url_expires_at": "2026-08-11T20:15:17.023144+00:00"
    },
    {
      "artifact_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "bytes": 512,
      "created_at": "2026-08-11T20:00:17.023125+00:00",
      "media_type": "application/vnd.apache.parquet",
      "rendering": {"statistics": "series", "member": "spec_2dffbc390c1c97d9992ee162ce690b19"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "url_expires_at": "2026-08-11T20:15:17.023125+00:00"
    },
    {
      "artifact_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "bytes": 512,
      "created_at": "2026-08-11T20:00:17.023144+00:00",
      "media_type": "application/vnd.apache.parquet",
      "rendering": {"statistics": "series", "member": "spec_75bacda207b42d5632b1502efc37fa2b"},
      "tier": "working",
      "url": "fake://artifact/working/sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "url_expires_at": "2026-08-11T20:15:17.023144+00:00"
    }
  ],
  "citation": {
    "sources": [{"label": "OECD DAC2a", "ref": "OECD_DAC2A"}],
    "text": "OECD DAC2a"
  },
  "collection": {
    "collection_id": "col_a33c5a92b6f9672086df93c6131e2965",
    "members": [
      {
        "content_version": "cv_6b2c86ce87fdd1f7ab7b7bbca9cbf066",
        "coverage": {
          "dimensions": [{"count": 1, "label": "recipient"}, {"count": 1, "label": "years"}],
          "rows": 1,
          "window": {"end_year": 2015, "kind": "all", "origin": "default", "start_year": 2015}
        },
        "definition": {
          "identity": {
            "entities": {"recipient": ["country/KEN"]},
            "facets": {
              "measurementMethod": {"ref": "GrantEquivalent", "state": "value"},
              "provenance": {"ref": "OECD_DAC2A", "state": "value"},
              "unit": {"ref": "USD", "state": "value"}
            },
            "identity_version": "spec1",
            "requested_window": {"kind": "all"},
            "shape": {"id": "MalariaGrantDisbursements", "kind": "stat_var"},
            "slots": []
          },
          "schema_version": 1,
          "spec_id": "spec_2dffbc390c1c97d9992ee162ce690b19"
        },
        "series_count": 1
      },
      {
        "content_version": "cv_6b2c86ce87fdd1f7ab7b7bbca9cbf066",
        "coverage": {
          "dimensions": [{"count": 1, "label": "recipient"}, {"count": 1, "label": "years"}],
          "rows": 1,
          "window": {"end_year": 2015, "kind": "all", "origin": "default", "start_year": 2015}
        },
        "definition": {
          "identity": {
            "entities": {"recipient": ["country/UGA"]},
            "facets": {
              "measurementMethod": {"ref": "GrantEquivalent", "state": "value"},
              "provenance": {"ref": "OECD_DAC2A", "state": "value"},
              "unit": {"ref": "USD", "state": "value"}
            },
            "identity_version": "spec1",
            "requested_window": {"kind": "all"},
            "shape": {"id": "MalariaGrantDisbursements", "kind": "stat_var"},
            "slots": []
          },
          "schema_version": 1,
          "spec_id": "spec_75bacda207b42d5632b1502efc37fa2b"
        },
        "series_count": 1
      }
    ]
  },
  "content_version": "cv_a198ad21968212c0b720aaaf7f6d43cd",
  "coverage": {
    "dimensions": [{"count": 1, "label": "recipient"}, {"count": 1, "label": "years"}],
    "rows": 2,
    "window": {"end_year": 2015, "kind": "range", "origin": "query", "start_year": 2015}
  },
  "schema_version": 1,
  "trace": {
    "applied_window": {"end_year": 2015, "kind": "all", "origin": "default", "start_year": 2015},
    "capability_version": "dset_bbe643884591f64b",
    "facets": [],
    "frame_grammar_version": "1.0.0",
    "graph_revision": "unavailable:dfo-dcp-revision-endpoint-absent",
    "materialiser_build": "datasets-materialiser-dev",
    "normalisations": [],
    "resolved_stat_vars": [
      {"ref": "dcid:MalariaGrantDisbursements", "roles": ["recipient"]},
      {"ref": "dcid:MalariaGrantDisbursements", "roles": ["recipient"]}
    ],
    "source_revisions": []
  },
  "warnings": []
}
""")


def _signed_urls(payload: dict) -> set[str]:
    """Every `url`/`url_expires_at` string anywhere in `payload`'s artifact entries."""
    urls: set[str] = set()

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("url", "url_expires_at") and isinstance(value, str):
                    urls.add(value)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return urls


# no signed URL survives anywhere in the output


def test_no_signed_url_survives_stripping_a_definition_manifest() -> None:
    original_urls = _signed_urls(_DEFINITION_MANIFEST)
    assert original_urls, "fixture must actually carry signed URLs for this test to mean anything"

    stripped = strip_manifest(_DEFINITION_MANIFEST)
    dumped = json.dumps(stripped.model_dump(mode="json"))

    assert '"url"' not in dumped
    assert '"url_expires_at"' not in dumped
    for url in original_urls:
        assert url not in dumped


def test_no_signed_url_survives_stripping_a_collection_manifest() -> None:
    original_urls = _signed_urls(_COLLECTION_MANIFEST)
    assert original_urls

    stripped = strip_manifest(_COLLECTION_MANIFEST)
    dumped = json.dumps(stripped.model_dump(mode="json"))

    assert '"url"' not in dumped
    assert '"url_expires_at"' not in dumped
    for url in original_urls:
        assert url not in dumped


def test_a_nested_artifact_list_a_future_refactor_might_add_would_still_be_caught() -> None:
    """Guards a future refactor that gives `CollectionMemberModel` its own
    `artifacts` field (upstream's `Manifest` currently only carries
    `artifacts` at the top level) and copies upstream's URL-bearing entry
    wholesale instead of this module's `ArtifactEntry`. Today, an
    `artifacts` key nested under a member is unrecognised and `extra`
    (`"ignore"`) drops it, so the planted URL below already fails to
    surface -- exactly the property a real nested-artifacts field must keep
    satisfying.
    """
    payload = copy.deepcopy(_COLLECTION_MANIFEST)
    payload["collection"]["members"][0]["artifacts"] = [
        {
            "artifact_digest": "sha256:" + "1" * 64,
            "rendering": {"bundle": False, "format": "csv"},
            "bytes": 1,
            "media_type": "text/csv",
            "created_at": "2026-08-11T20:00:17.000000+00:00",
            "tier": "working",
            "url": "https://signed.example/should-not-leak",
            "url_expires_at": "2026-08-11T20:15:17.000000+00:00",
        }
    ]

    stripped = strip_manifest(payload)
    dumped = json.dumps(stripped.model_dump(mode="json"))

    assert "https://signed.example/should-not-leak" not in dumped


# artifact_digest survives


def test_artifact_digest_survives_stripping() -> None:
    stripped = strip_manifest(_DEFINITION_MANIFEST)

    assert [a.artifact_digest for a in stripped.artifacts] == [
        a["artifact_digest"] for a in _DEFINITION_MANIFEST["artifacts"]
    ]


# statistics is optional


def test_series_count_absent_validates_as_none() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    del payload["series_count"]

    stripped = strip_manifest(payload)

    assert stripped.series_count is None


def test_series_count_present_on_a_definition_manifest_survives() -> None:
    stripped = strip_manifest(_DEFINITION_MANIFEST)

    assert stripped.series_count == 1
    artifact = find_statistics_artifact(stripped)
    assert artifact is not None
    assert artifact.rendering == {"statistics": "series"}


def test_a_coverage_dimension_carries_its_flow_axis_when_dfo_apis_states_one() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    payload["coverage"]["dimensions"] = [
        {"label": "financingProvider", "count": 1, "axis": "origin"},
        {"label": "financingRecipient", "count": 3, "axis": "destination"},
        {"label": "years", "count": 10},
    ]

    stripped = strip_manifest(payload)

    assert [(d.label, d.axis) for d in stripped.coverage.dimensions] == [
        ("financingProvider", "origin"),
        ("financingRecipient", "destination"),
        ("years", None),
    ]


def test_a_coverage_dimension_from_a_dfo_apis_without_axes_still_parses() -> None:
    stripped = strip_manifest(_DEFINITION_MANIFEST)

    assert stripped.coverage.dimensions
    assert all(dimension.axis is None for dimension in stripped.coverage.dimensions)


def test_top_level_series_count_on_a_collection_manifest_refuses() -> None:
    payload = copy.deepcopy(_COLLECTION_MANIFEST)
    payload["series_count"] = _DEFINITION_MANIFEST["series_count"]

    with pytest.raises(ValidationError):
        strip_manifest(payload)


# collection manifest shape


def test_collection_manifest_validates() -> None:
    stripped = strip_manifest(_COLLECTION_MANIFEST)

    assert stripped.collection is not None
    assert stripped.definition is None
    assert stripped.collection.collection_id == "col_a33c5a92b6f9672086df93c6131e2965"
    assert len(stripped.collection.members) == 2
    assert stripped.collection.members[0].series_count == 1
    member_spec_id = stripped.collection.members[0].definition.spec_id
    member_rendering = {"statistics": "series", "member": member_spec_id}
    assert any(a.rendering == member_rendering for a in stripped.artifacts)


# definition/collection exclusivity


def test_neither_definition_nor_collection_refuses() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    del payload["definition"]

    with pytest.raises(ValidationError):
        strip_manifest(payload)


def test_both_definition_and_collection_refuses() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    payload["collection"] = _COLLECTION_MANIFEST["collection"]

    with pytest.raises(ValidationError):
        strip_manifest(payload)


# extra="ignore": an unrecognised field is dropped, never forwarded, never fatal


def test_an_unrecognised_top_level_field_is_dropped_not_forwarded_and_not_fatal() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    payload["a_field_this_service_has_never_seen"] = "some future upstream addition"

    stripped = strip_manifest(payload)

    assert "a_field_this_service_has_never_seen" not in json.dumps(stripped.model_dump(mode="json"))


def test_an_unrecognised_artifact_field_is_dropped_not_forwarded_and_not_fatal() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    payload["artifacts"][0]["a_future_artifact_field"] = "not reviewed yet"

    stripped = strip_manifest(payload)

    assert "a_future_artifact_field" not in json.dumps(stripped.model_dump(mode="json"))


# a payload that doesn't validate raises a clear error


def test_a_missing_required_field_raises_validation_error() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    del payload["content_version"]

    with pytest.raises(ValidationError):
        strip_manifest(payload)


def test_strip_manifest_returns_a_manifest_instance() -> None:
    assert isinstance(strip_manifest(_DEFINITION_MANIFEST), Manifest)


# resource_uri: derivation, the collection plural, no signed URL, and overwrite-on-input


def test_resource_uri_derived_from_definition_spec_id_content_version_and_digest() -> None:
    stripped = strip_manifest(_DEFINITION_MANIFEST)

    assert len(stripped.artifacts) == 2
    assert stripped.artifacts[0].resource_uri == (
        "dfo://artifact/spec_2dffbc390c1c97d9992ee162ce690b19"
        "/cv_6b2c86ce87fdd1f7ab7b7bbca9cbf066"
        "/sha256:8d82b3452a2f7ac2c03c9d611548ee8d63cb90f37a8e98978d325372547822a7"
    )


def test_collection_manifest_yields_one_resource_uri_per_artifact() -> None:
    """A collection manifest is the only place `get_data` can't reach
    (it refuses `col_` ids), so the plural is only exercisable here. Three
    members, three distinct digests, three distinct `resource_uri`s, all
    keyed on `collection.collection_id` rather than any one member's own id.
    """
    payload = copy.deepcopy(_COLLECTION_MANIFEST)
    member_template = payload["collection"]["members"][0]
    payload["collection"]["members"] = [copy.deepcopy(member_template) for _ in range(3)]
    payload["artifacts"] = [
        {**copy.deepcopy(payload["artifacts"][0]), "artifact_digest": f"sha256:{str(i) * 64}"}
        for i in range(1, 4)
    ]

    stripped = strip_manifest(payload)

    resource_uris = [a.resource_uri for a in stripped.artifacts]
    assert len(resource_uris) == 3
    assert len(set(resource_uris)) == 3
    for i, artifact in enumerate(stripped.artifacts, start=1):
        assert artifact.resource_uri == (
            f"dfo://artifact/col_a33c5a92b6f9672086df93c6131e2965"
            f"/cv_a198ad21968212c0b720aaaf7f6d43cd"
            f"/sha256:{str(i) * 64}"
        )


def test_resource_uri_is_never_a_signed_url() -> None:
    payload = copy.deepcopy(_DEFINITION_MANIFEST)
    payload["artifacts"][0]["url"] = "https://signed.example/should-not-leak-into-resource-uri"
    payload["artifacts"][0]["url_expires_at"] = "2026-08-11T20:15:17.015350+00:00"

    stripped = strip_manifest(payload)
    dumped = json.dumps(stripped.model_dump(mode="json"))

    assert "https://signed.example/should-not-leak-into-resource-uri" not in dumped
    for artifact in stripped.artifacts:
        assert artifact.resource_uri is not None
        assert artifact.resource_uri.startswith("dfo://artifact/")


def test_a_resource_uri_already_on_the_input_is_overwritten_not_trusted() -> None:
    """A round-tripped, already-stripped document carries its own `resource_uri` from a
    prior validation. Re-validating it must derive the value fresh rather than accept
    whatever the input claims -- otherwise a stored or replayed document could carry a
    stale or forged URI forward.
    """
    once_stripped = strip_manifest(_DEFINITION_MANIFEST)
    payload = once_stripped.model_dump(mode="json")
    payload["artifacts"][0]["resource_uri"] = (
        "dfo://artifact/not_the_real/subject/sha256:" + "0" * 64
    )

    twice_stripped = strip_manifest(payload)

    assert twice_stripped.artifacts[0].resource_uri == once_stripped.artifacts[0].resource_uri
    assert twice_stripped.artifacts[0].resource_uri != (
        "dfo://artifact/not_the_real/subject/sha256:" + "0" * 64
    )


def test_artifact_uri_template_constant_is_the_dfo_artifact_grammar() -> None:
    assert (
        ARTIFACT_URI_TEMPLATE == "dfo://artifact/{subject_id}/{content_version}/{artifact_digest}"
    )


# render_validation_errors: no raw input value ever rendered


def test_render_validation_errors_suppresses_the_raw_input_value() -> None:
    malformed = {"url": "https://signed.example/should-not-leak-into-a-refusal-message"}

    with pytest.raises(ValidationError) as exc_info:
        strip_manifest({**copy.deepcopy(_DEFINITION_MANIFEST), "artifacts": [malformed]})

    rendered = render_validation_errors(exc_info.value)

    assert "https://signed.example/should-not-leak-into-a-refusal-message" not in rendered
    assert "artifact_digest" in rendered


def test_render_validation_errors_caps_and_counts_omitted_errors() -> None:
    # `ArtifactEntry` has 6 required fields; an empty dict fails all of them,
    # comfortably past `_MAX_RENDERED_VALIDATION_ERRORS` (5).
    with pytest.raises(ValidationError) as exc_info:
        strip_manifest({**copy.deepcopy(_DEFINITION_MANIFEST), "artifacts": [{}]})

    rendered = render_validation_errors(exc_info.value)

    assert "... and" in rendered
    assert "more" in rendered
