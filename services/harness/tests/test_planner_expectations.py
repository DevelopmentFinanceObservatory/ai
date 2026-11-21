"""The stored-expectation helpers check every value they are given."""

from __future__ import annotations

import json
import re
from pathlib import Path

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


@pytest.mark.parametrize(
    ("state", "rewritten"),
    [({"b": 1, "a": 2}, False), ({"b": 1, "a": 3}, True)],
    ids=["same-value-other-key-order", "changed-value"],
)
def test_a_write_run_keeps_the_unread_keys_and_rewrites_only_a_changed_value(
    stored, monkeypatch, state, rewritten
):
    monkeypatch.setattr(planner_expectations, "_WRITING", True)
    monkeypatch.setattr(planner_expectations, "_CHANGED", [])
    path = stored / "run.json"
    text = json.dumps({"events": ["kept"], "state": {"a": 2, "b": 1}}, indent=2) + "\n"
    path.write_text(text)

    if rewritten:
        with pytest.raises(pytest.fail.Exception, match=re.escape("wrote run.json")):
            expect("run", {"state": state}, unread=("events",))
    else:
        expect("run", {"state": state}, unread=("events",))

    assert json.loads(path.read_text()) == {"events": ["kept"], "state": state}
    assert (path.read_text() != text) is rewritten


def test_the_write_recipe_names_every_module_that_stores_an_expectation():
    tests = Path(__file__).parent
    storing = {
        str(module.relative_to(tests.parents[2]))
        for module in tests.rglob("test_*.py")
        if module != Path(__file__) and "expectation_name" in module.read_text()
    }
    recipe = planner_expectations.__doc__ or ""

    assert storing, "no module stores an expectation"
    assert sorted(module for module in storing if module not in recipe) == []
