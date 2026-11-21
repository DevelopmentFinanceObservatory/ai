"""Unit tests for `harness_translator_v1.outcome_metric.matches_subset`.

Model-free: this pins the matcher's own rules (keys the expectation leaves
out are not compared, lists match by position and length, `sources` in any
order) independently of whether the eval set's cases are internally
consistent. `test_eval_outcomes.py` proves each case's expected outcome is
reachable.

`evals/agents` is not on `sys.path` for a normal `pytest` run, so this
module inserts it, as `test_eval_outcomes.py` does, to import
`harness_translator_v1` as a top-level package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = REPO_ROOT / "evals" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from harness_translator_v1.outcome_metric import matches_subset  # noqa: E402

DEBT_SPEC = "spec_" + "d" * 32
OTHER_SPEC = "spec_" + "e" * 32


def test_matches_subset_ignores_keys_and_extra_fields_the_expectation_does_not_name() -> None:
    expected = {"outcome": "ok", "steps": [{"tool": "data-mcp.get_data"}]}
    actual = {
        "outcome": "ok",
        "version": 1,
        "steps": [{"id": "op-1", "tool": "data-mcp.get_data"}],
    }

    assert matches_subset(expected, actual)


def test_matches_subset_rejects_a_list_of_a_different_length() -> None:
    expected = {"steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}]}
    actual = {"steps": [{"tool": "data-mcp.get_data"}]}

    assert not matches_subset(expected, actual)


def test_matches_subset_rejects_a_differing_value() -> None:
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}]}, {"sources": [{"spec_id": OTHER_SPEC}]}
    )


def test_matches_subset_compares_sources_in_any_order_but_steps_in_order() -> None:
    # The compiler emits sources in a canonical order of its own; a case
    # states which were selected, not how the compiler sorts them. Steps
    # are the tool sequence under test, so their order still counts.
    expected = {
        "steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}],
        "sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}],
    }
    sources_swapped = {
        "steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}],
        "sources": [{"spec_id": OTHER_SPEC}, {"spec_id": DEBT_SPEC}],
    }
    steps_swapped = {
        "steps": [{"tool": "analysis-mcp.aggregate"}, {"tool": "data-mcp.get_data"}],
        "sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}],
    }

    assert matches_subset(expected, sources_swapped)
    assert not matches_subset(expected, steps_swapped)


def test_matches_subset_still_counts_sources() -> None:
    # Unordered is not "a subset of": every actual source has to be one the
    # case named, and each expected one has to match a distinct actual one.
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}]},
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}]},
    )
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": DEBT_SPEC}]},
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}]},
    )
