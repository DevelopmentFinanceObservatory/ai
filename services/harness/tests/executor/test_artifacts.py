"""Tests for `executor.artifacts`: reading a frame reference out of a tool result."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from compiler.capabilities import Extract
from executor.artifacts import (
    ArtifactError,
    declared_shape,
    extract_frame_ref,
    parse_manifest,
)

from .conftest import CONTENT_VERSION, RESULT_FRAME, SPEC_KENYA

ManifestFactory = Callable[..., dict[str, Any]]


def test_a_verbs_pair_is_read_as_a_frame_reference() -> None:
    frame = extract_frame_ref(Extract.FRAME_REF_PAIR, RESULT_FRAME)

    assert (frame.spec_id, frame.content_version) == (
        RESULT_FRAME["spec_id"],
        RESULT_FRAME["content_version"],
    )


def test_a_retrievals_frame_reference_is_read_beside_its_manifest(
    manifest: ManifestFactory,
) -> None:
    """A frame minted with extra columns is held under a `content_version` of its
    own, so the manifest's own pair is not the frame's reference."""
    typed = {"spec_id": SPEC_KENYA, "content_version": "cv_" + "9" * 32}

    frame = extract_frame_ref(Extract.FRAME_REF_FROM_MANIFEST, manifest(frame_ref=typed))

    assert (frame.spec_id, frame.content_version) == (SPEC_KENYA, typed["content_version"])
    assert manifest()["content_version"] == CONTENT_VERSION


def test_a_manifest_stating_no_frame_reference_is_refused(manifest: ManifestFactory) -> None:
    result = {key: value for key, value in manifest().items() if key != "frame_ref"}

    with pytest.raises(ArtifactError, match="valid frame reference"):
        extract_frame_ref(Extract.FRAME_REF_FROM_MANIFEST, result)


@pytest.mark.parametrize(
    ("extract", "content", "expected"),
    [
        (Extract.FRAME_REF_PAIR, None, "no structured content"),
        (Extract.FRAME_REF_PAIR, {"spec_id": RESULT_FRAME["spec_id"]}, "valid frame reference"),
        (
            Extract.FRAME_REF_PAIR,
            {"spec_id": "nope", "content_version": "x"},
            "valid frame reference",
        ),
        (Extract.FRAME_REF_FROM_MANIFEST, {"schema_version": 1}, "not a manifest"),
        (
            Extract.SPEC_IDS_FROM_ENVELOPE,
            {"status": "definite"},
            "does not yield a frame reference",
        ),
    ],
    ids=["missing", "half-a-pair", "malformed-pair", "not-a-manifest", "wrong-extractor"],
)
def test_a_result_that_carries_no_frame_reference_is_refused(
    extract: Extract, content: dict[str, Any] | None, expected: str
) -> None:
    with pytest.raises(ArtifactError, match=expected):
        extract_frame_ref(extract, content)


def test_parse_manifest_reads_the_series_count(manifest: ManifestFactory) -> None:
    parsed = parse_manifest(manifest())

    assert parsed.series_count == 1


def test_declared_shape_reads_a_verb_results_own_wire_fields() -> None:
    content = {"rows": 3, "role_keys": ["reporter"], "stat_var_count": 1}

    assert declared_shape(content) == (3, ("reporter",), 1)


def test_declared_shape_is_all_none_for_a_manifest() -> None:
    # `get_data`'s structured result is a `Manifest`, which carries none of
    # the three fields a `VerbResult` declares.
    assert declared_shape({"schema_version": 1, "coverage": {"rows": 10}}) == (None, None, None)


def test_declared_shape_is_all_none_for_no_content() -> None:
    assert declared_shape(None) == (None, None, None)


def test_declared_shape_reads_each_field_independently() -> None:
    # A malformed or partial payload reads whatever fields it does carry;
    # the caller decides what a missing field means.
    assert declared_shape({"rows": 0}) == (0, None, None)
    assert declared_shape({"role_keys": ["donor", "sector"]}) == (None, ("donor", "sector"), None)
    assert declared_shape({"stat_var_count": 2}) == (None, None, 2)


@pytest.mark.parametrize(
    "role_keys",
    [
        ["donor", "value"],  # a reserved column name
        ["caf\u00e9", "cafe\u0301"],  # one key after NFC  # codespell:ignore
    ],
)
def test_declared_shape_refuses_role_keys_the_frame_grammar_rejects(role_keys: list[str]) -> None:
    with pytest.raises(ArtifactError, match="not valid role columns"):
        declared_shape({"rows": 3, "role_keys": role_keys, "stat_var_count": 1})
