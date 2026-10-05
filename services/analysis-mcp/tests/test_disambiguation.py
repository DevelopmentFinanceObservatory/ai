"""The disambiguation payload: a candidate set, distinct from a refusal."""

from __future__ import annotations

import pytest
from analysis_mcp.disambiguation import DisambiguationCandidate, DisambiguationPayload
from pydantic import ValidationError


def test_payload_round_trips() -> None:
    payload = DisambiguationPayload(
        candidates=(
            DisambiguationCandidate(
                id="a", label="Candidate A", detail={"reason": "closest match"}
            ),
            DisambiguationCandidate(id="b", label="Candidate B"),
        )
    )
    dumped = payload.model_dump(mode="json")
    assert DisambiguationPayload.model_validate(dumped) == payload


def test_candidate_detail_defaults_to_empty_mapping() -> None:
    candidate = DisambiguationCandidate(id="a", label="Candidate A")
    assert candidate.detail == {}


def test_empty_candidate_set_is_refused() -> None:
    with pytest.raises(ValidationError):
        DisambiguationPayload(candidates=())


def test_unknown_field_refused() -> None:
    with pytest.raises(ValidationError):
        DisambiguationCandidate(id="a", label="A", extra_field="nope")  # type: ignore[call-arg]
