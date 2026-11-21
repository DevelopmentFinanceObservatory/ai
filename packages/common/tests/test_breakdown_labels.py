"""`common.breakdown_labels`: which breakdown cells are read as codes, what label each
gets, and how many node reads one call makes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from common.breakdown_labels import (
    MAX_LABELLED_CELLS,
    NO_LABELS,
    NODE_BATCH_DCIDS,
    breakdown_axis,
    label_cells,
)

_PEACE = "civilian peace-building, conflict prevention and resolution"


def _code(dcid: str, label: str, *, axis: str = "crsPurpose") -> dict[str, Any]:
    """A code node as `POST /api/graph/nodes` answers it, its system named after its label."""
    return {
        "ref": dcid,
        "kind": "code",
        "label": f"{label} (CRS Purpose)",
        "code": {
            "property": axis,
            "code_system": {"ref": "CRSPurposeCodeEnum", "label": "CRS Purpose"},
        },
    }


class _Graph:
    """Answers each batch with the held node of each dcid, `not_held` for the rest,
    keeping every batch it was asked."""

    def __init__(self, *nodes: Mapping[str, Any]) -> None:
        self.held = {node["ref"]: node for node in nodes}
        self.batches: list[list[str]] = []

    async def read(self, dcids: list[str]) -> Sequence[Mapping[str, Any]]:
        self.batches.append(dcids)
        return [self.held.get(dcid, {"ref": dcid, "status": "not_held"}) for dcid in dcids]


async def test_a_code_dcid_cell_reads_as_its_label_and_a_label_cell_stays() -> None:
    graph = _Graph(_code("CRSPurposeCode15220", _PEACE))
    column = "stat_var_crsPurpose"

    labels = await label_cells(
        [(column, "CRSPurposeCode15220"), (column, "basic education"), (column, "agriculture")],
        read_nodes=graph.read,
    )

    assert labels.text(column, "CRSPurposeCode15220") == f"{_PEACE} (CRSPurposeCode15220)"
    assert labels.label(column, "CRSPurposeCode15220") == _PEACE
    assert labels.text(column, "basic education") == "basic education"
    assert labels.text(column, "agriculture") == "agriculture"
    # Free text is no dcid, so it is never sent; a dcid-shaped label is, and stays.
    assert graph.batches == [["CRSPurposeCode15220", "agriculture"]]


async def test_a_cell_keeps_its_value_unless_the_node_is_a_code_of_its_own_axis() -> None:
    graph = _Graph(
        _code("CRSTypeOfFinance421", "standard loan", axis="crsTypeOfFinance"),
        {"ref": "country/KEN", "kind": "country", "label": "Kenya"},
    )

    labels = await label_cells(
        [
            ("stat_var_crsPurpose", "CRSTypeOfFinance421"),
            ("stat_var_crsPurpose", "country/KEN"),
            ("stat_var_crsTypeOfFinance", "dcid:CRSTypeOfFinance421"),
            ("recipient", "country/KEN"),
        ],
        read_nodes=graph.read,
    )

    assert labels.labels == {
        ("stat_var_crsTypeOfFinance", "dcid:CRSTypeOfFinance421"): "standard loan"
    }
    # The role column is no breakdown, so its dcid is not read.
    assert graph.batches == [["CRSTypeOfFinance421", "country/KEN"]]


async def test_a_failed_read_labels_nothing() -> None:
    async def failed(_dcids: list[str]) -> Sequence[Mapping[str, Any]]:
        return []

    labels = await label_cells([("stat_var_crsPurpose", "CRSPurposeCode15220")], read_nodes=failed)

    assert labels == NO_LABELS


async def test_cells_are_read_in_batches_and_past_the_cap_keep_their_values() -> None:
    dcids = [f"CRSPurposeCode{n}" for n in range(MAX_LABELLED_CELLS + 10)]
    graph = _Graph(*(_code(dcid, f"purpose {dcid}") for dcid in dcids))

    labels = await label_cells(
        [("stat_var_crsPurpose", dcid) for dcid in dcids], read_nodes=graph.read
    )

    assert [len(batch) for batch in graph.batches] == [NODE_BATCH_DCIDS] * (
        MAX_LABELLED_CELLS // NODE_BATCH_DCIDS
    )
    assert len(labels.labels) == MAX_LABELLED_CELLS
    assert labels.text("stat_var_crsPurpose", dcids[-1]) == dcids[-1]


async def test_no_breakdown_cell_reads_nothing() -> None:
    graph = _Graph()

    labels = await label_cells(
        [
            ("stat_var_crsPurpose", "Humanitarian aid"),
            ("donor", "country/FRA"),
            ("stat_var_x", ".."),
        ],
        read_nodes=graph.read,
    )

    assert labels == NO_LABELS
    assert graph.batches == []


def test_breakdown_axis_names_the_axis_of_a_stat_var_column_only() -> None:
    assert breakdown_axis("stat_var_crsPurpose") == "crsPurpose"
    assert breakdown_axis("stat_var") is None
    assert breakdown_axis("stat_var_") is None
    assert breakdown_axis("recipient") is None
