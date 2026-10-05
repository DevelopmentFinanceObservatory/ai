"""Replays every case in `engine_cases.jsonl` through each engine
`harness.IMPLEMENTATIONS` lists for its verb, and requires the serialised
outcome to match the recorded one exactly: every float bit, row, exclusion,
warning and refusal payload.

Outcomes are compared as JSON text, so `-0.0` and `0.0`, or `1` and `1.0`,
count as different.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from dfo_frame_grammar import frame_sha256
from differential.cases import frame_from_spec, generate_cases
from differential.harness import (
    IMPLEMENTATIONS,
    implementation_label,
    load_fixture,
    run_case,
)

RECORDS = load_fixture()

_REPLAYS = [
    pytest.param(record, engine, id=f"{record['id']}[{implementation_label(engine)}]")
    for record in RECORDS
    for engine in IMPLEMENTATIONS[record["verb"]]
]


def _pretty(outcome: Any) -> str:
    return json.dumps(outcome, indent=1)


@pytest.mark.parametrize(("record", "engine"), _REPLAYS)
def test_engine_reproduces_recorded_outcome(record: dict[str, Any], engine: Any) -> None:
    assert _pretty(run_case(record, engine)) == _pretty(record["outcome"])


def test_input_frames_rebuild_to_recorded_hashes() -> None:
    """Every stored input frame still builds to the frame it was recorded
    from, so a replay failure points at the engine, not the frame grammar."""
    mismatched = [
        record["id"]
        for record in RECORDS
        if [frame_sha256(frame_from_spec(spec)) for spec in record["frames"]]
        != record["input_frame_sha256"]
    ]
    assert mismatched == []


def test_generator_reproduces_fixture_inputs() -> None:
    """`generate_cases()` still yields the fixture's inputs, so recording the
    fixture again changes outcomes only."""
    generated = generate_cases()
    recorded = [
        {key: record[key] for key in ("id", "verb", "frames", "params")} for record in RECORDS
    ]
    assert json.dumps(generated) == json.dumps(recorded)


def test_fixture_covers_every_verb_and_outcome_kind() -> None:
    kinds = Counter((record["verb"], next(iter(record["outcome"]))) for record in RECORDS)
    for verb in IMPLEMENTATIONS:
        assert kinds[verb, "result"] >= 10, verb
        assert kinds[verb, "refusal"] >= 5, verb
