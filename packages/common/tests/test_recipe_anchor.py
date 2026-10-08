"""Tests for `common.recipe.recipe_anchor` over hand-written
trees. `services/analysis-mcp/tests/test_rolling_tool.py` builds a rolling recipe
from real verb calls."""

from __future__ import annotations

from typing import Any

import pytest
from common.materialisation import AnchorRecord, FrameRef, InMemoryMaterialisationStore
from common.recipe import RecipeAnchorError, recipe_anchor

_SPEC = "spec_" + "a" * 32
_ROOT = {"spec_id": _SPEC, "content_version": "cv_" + "1" * 32}
_POPULATIONS: list[dict[str, Any]] = [{"spec_id": _SPEC, "bindings": {}}]
_OTHER = "spec_" + "b" * 32
_OTHER_ROOT = {"spec_id": _OTHER, "content_version": "cv_" + "1" * 32}
_WIDER: list[dict[str, Any]] = [*_POPULATIONS, {"spec_id": _OTHER, "bindings": {}}]


def _latest(offset: int) -> dict[str, Any]:
    return {"anchor": "latest", "offset": offset, "rule": "half1"}


def _frame(digit: str) -> dict[str, str]:
    return {"spec_id": "ops_" + digit * 32, "content_version": "ocv_" + digit * 32}


def _growth(
    *, start: str, end: str, populations: list[dict[str, Any]], below: dict[str, Any]
) -> dict[str, Any]:
    return {
        "frame": _frame("2"),
        "op_spec": {
            "verb": "growth",
            "parameters": {
                "window_start": start,
                "window_end": end,
                "rolling": {
                    "window_start": _latest(-2),
                    "window_end": _latest(0),
                    "population": populations,
                },
            },
        },
        "inputs": {"frame_ref": below},
    }


def _filter(*, since: str, populations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frame": _frame("1"),
        "op_spec": {
            "verb": "filter",
            "parameters": {
                "predicates": [
                    {
                        "kind": "ordered_threshold",
                        "column": "time_period",
                        "op": "gte",
                        "value": since,
                        "rolling": _latest(-3),
                    }
                ],
                "rolling": {"population": populations},
            },
        },
        "inputs": {"frame_ref": {"frame": _ROOT, "root": 0}},
    }


def _tree(
    *,
    since: str = "2020",
    start: str = "2021",
    end: str = "2023",
    populations: list[dict[str, Any]] = _POPULATIONS,
    filter_populations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _growth(
        start=start,
        end=end,
        populations=populations,
        below=_filter(since=since, populations=filter_populations or populations),
    )


def _beside(rolling: dict[str, Any], fixed_root: dict[str, str]) -> dict[str, Any]:
    """`rolling` joined with the root frame `fixed_root`, which no rolling step reads."""
    return {
        "frame": _frame("2"),
        "op_spec": {"verb": "join", "parameters": {"right_frame_ref": fixed_root}},
        "inputs": {"frame_ref": rolling, "right_frame_ref": {"frame": fixed_root, "root": 1}},
    }


async def _store_with_record(*, roots: list[dict[str, str]]) -> InMemoryMaterialisationStore:
    store = InMemoryMaterialisationStore()
    await store.put_anchor_record(
        AnchorRecord(
            populations=_POPULATIONS,
            roots=roots,
            rule="half1",
            period="2023",
            left_out=[{"root": 0, "coordinates": {"reporter": "dcid:USA"}}],
        )
    )
    return store


async def test_a_rolling_tree_takes_its_anchor_from_the_record_under_its_key() -> None:
    store = await _store_with_record(roots=[_ROOT])

    assert await recipe_anchor(_tree(), store) == {
        "period": "2023",
        "rule": "half1",
        "left_out": [{"root": 0, "coordinates": {"reporter": "dcid:USA"}}],
    }


async def test_the_record_is_keyed_on_the_roots_under_the_rolling_steps_alone() -> None:
    # join(rolling growth on one dataset, another dataset): the latest period was
    # taken over the first root alone, and the recipe finds that record.
    store = await _store_with_record(roots=[_ROOT])

    anchor = await recipe_anchor(_beside(_tree(), _OTHER_ROOT), store)

    assert anchor is not None
    assert anchor["period"] == "2023"


async def test_a_tree_with_no_rolling_op_spec_has_no_anchor_and_reads_no_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryMaterialisationStore()
    monkeypatch.setattr(store, "get_anchor_record", None)
    fixed = {
        "frame": _frame("3"),
        "op_spec": {"verb": "aggregate", "parameters": {"aggregation": "sum"}},
        "inputs": {"frame_ref": {"frame": _ROOT, "root": 0}},
    }

    assert await recipe_anchor(fixed, store) is None


@pytest.mark.parametrize(
    ("tree", "record_roots", "reason"),
    [
        pytest.param(_tree(), None, "no_anchor_record", id="no-record"),
        pytest.param(
            _beside(_tree(), _OTHER_ROOT),
            [_ROOT, _OTHER_ROOT],
            "no_anchor_record",
            id="record-over-a-root-no-rolling-step-reads",
        ),
        pytest.param(
            _beside(_tree(populations=_WIDER), _OTHER_ROOT),
            [_ROOT, _OTHER_ROOT],
            "populations_outside_reach",
            id="populations-over-a-dataset-no-rolling-step-reads",
        ),
        pytest.param(
            _tree(filter_populations=[{"spec_id": _SPEC, "bindings": {"reporter": ["dcid:USA"]}}]),
            [_ROOT],
            "populations_differ",
            id="populations-differ",
        ),
        pytest.param(
            _tree(since="2019", start="2020", end="2022"),
            [_ROOT],
            "period_differs",
            id="literals-from-another-period",
        ),
    ],
)
async def test_a_rolling_tree_without_a_matching_record_is_an_error_and_never_null(
    tree: dict[str, Any], record_roots: list[dict[str, str]] | None, reason: str
) -> None:
    store = (
        InMemoryMaterialisationStore()
        if record_roots is None
        else await _store_with_record(roots=record_roots)
    )

    with pytest.raises(RecipeAnchorError) as raised:
        await recipe_anchor(tree, store)

    assert raised.value.reason == reason
    assert raised.value.ref == FrameRef(**_frame("2"))
