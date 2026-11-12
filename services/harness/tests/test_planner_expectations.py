"""The stored-expectation helpers check every value they are given."""

from __future__ import annotations

import json

import pytest

from . import planner_expectations
from .planner_expectations import expect, shared


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """An empty expectations directory with one shared value, `{"temperature": 0}`
    as `shared/config.json`."""
    monkeypatch.setattr(planner_expectations, "PLANNER_RUNS", tmp_path)
    monkeypatch.setattr(planner_expectations, "_SHARED", tmp_path / "shared")
    monkeypatch.setattr(planner_expectations, "_WRITING", False)
    monkeypatch.setattr(planner_expectations, "_PRODUCED", {})
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "config.json").write_text(json.dumps({"temperature": 0}))
    return tmp_path


@pytest.mark.parametrize(
    "values",
    [
        [{"temperature": 999}, {"temperature": 0}],
        [{"temperature": 0}, {"temperature": 999}],
    ],
    ids=["wrong-then-right", "right-then-wrong"],
)
def test_one_wrong_value_among_those_sharing_a_name_fails(stored, values):
    (stored / "run.json").write_text(json.dumps([{"shared": "config"}, {"shared": "config"}]))

    with pytest.raises(pytest.fail.Exception):
        expect("run", [shared(value, "config") for value in values])


def test_values_equal_to_the_stored_one_share_its_name(stored):
    (stored / "run.json").write_text(json.dumps([{"shared": "config"}, {"shared": "config"}]))

    expect("run", [shared({"temperature": 0}, "config") for _ in range(2)])
