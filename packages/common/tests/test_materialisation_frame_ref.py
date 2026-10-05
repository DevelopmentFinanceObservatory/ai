"""`FrameRef`: accepts both pair families, rejects malformed or extra fields."""

from __future__ import annotations

import pytest
from common.materialisation import FrameRef
from pydantic import ValidationError


def test_accepts_a_datasets_family_pair() -> None:
    ref = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)

    assert ref.spec_id == "spec_" + "a" * 32
    assert ref.content_version == "cv_" + "b" * 32


def test_accepts_a_toolkit_family_pair() -> None:
    ref = FrameRef(spec_id="ops_" + "a" * 32, content_version="ocv_" + "b" * 32)

    assert ref.spec_id == "ops_" + "a" * 32
    assert ref.content_version == "ocv_" + "b" * 32


def test_rejects_a_malformed_spec_id() -> None:
    with pytest.raises(ValidationError):
        FrameRef(spec_id="not-a-real-id", content_version="cv_" + "b" * 32)


def test_rejects_a_malformed_content_version() -> None:
    with pytest.raises(ValidationError):
        FrameRef(spec_id="spec_" + "a" * 32, content_version="not-a-real-cv")


def test_rejects_an_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        FrameRef(
            spec_id="spec_" + "a" * 32,
            content_version="cv_" + "b" * 32,
            extra_field="nope",
        )
