"""Tests for `render_describe_digest`: pure rendering, no client, off the node fixtures."""

from __future__ import annotations

import pytest
from data_mcp.describe_digest import render_describe_digest
from data_mcp.graph_wire import (
    CountryNode,
    NodeRef,
    NotHeldEntry,
    OtherNode,
    SourceNode,
    StatVarNode,
)
from data_mcp.node_lookup import HeldNode, NodeOutcome

from .node_fixtures import NODE_FIXTURES, node_fixture


def _render(*nodes: HeldNode) -> str:
    outcomes: dict[str, NodeOutcome] = {node.ref: node for node in nodes}
    return render_describe_digest(dcids=list(outcomes), outcomes=outcomes)


# 1. One rendering test per kind asserting what its block carries.


def test_stat_var_block_carries_its_definition_roles_years_and_sources() -> None:
    node = node_fixture("node_stat_var")
    assert isinstance(node, StatVarNode)

    text = _render(node)

    assert text.splitlines()[:3] == [
        "GDP, current prices (GDP_Nominal)",
        "kind: stat_var",
        "Gross domestic product at current prices.",
    ]
    assert "Measures gross domestic product of Economic activity." in text
    assert "roles: observationAbout" in text
    assert "years held: 2000 to 2020" in text
    assert "sources (1 in all): provenance/IMFWorldEconomicOutlook" in text


def test_stat_var_without_a_description_renders_no_description_line() -> None:
    node = node_fixture("node_stat_var", description=None)

    text = _render(node)

    assert "None" not in text
    assert text.splitlines()[2] == "Measures gross domestic product of Economic activity."


def test_stat_var_without_availability_says_it_is_not_known() -> None:
    node = node_fixture("node_stat_var")
    assert isinstance(node, StatVarNode)
    node = node.model_copy(
        update={"stat_var": node.stat_var.model_copy(update={"availability": None})}
    )

    text = _render(node)

    assert "availability: not known" in text
    assert "years held" not in text


def test_stat_var_gaps_render_as_not_known() -> None:
    node = node_fixture("node_stat_var")

    text = _render(node)

    assert ("not known: stat_var.definition.stat_type.label, stat_var.definition.measure") in text


@pytest.mark.parametrize("fixture_name", NODE_FIXTURES)
def test_a_node_without_a_label_is_headed_by_its_ref(fixture_name: str) -> None:
    node = node_fixture(fixture_name, label=None)

    text = _render(node)

    assert text.startswith(f"{node.ref}\nkind: {node.kind}")
    assert "None" not in text


def test_source_block_carries_its_licence_freshness_and_url() -> None:
    node = node_fixture("node_source")
    assert isinstance(node, SourceNode)

    text = _render(node)

    assert "licence: CC BY-NC-SA 3.0 IGO (cc-by-nc-sa)" in text
    assert "freshness: release Yearly · refresh NoAutomaticRefresh" in text
    assert "url: https://www.who.int" in text
    assert "publisher: not recorded" in text


def test_source_null_release_frequency_renders_not_recorded() -> None:
    # The BIS-style asymmetry: `automaticRefreshType` without `sourceReleaseFrequency`.
    node = node_fixture("node_source")
    assert isinstance(node, SourceNode)
    freshness = node.source.freshness.model_copy(
        update={"release_frequency": None, "release_frequency_label": None}
    )
    node = node.model_copy(
        update={"source": node.source.model_copy(update={"freshness": freshness})}
    )

    text = _render(node)

    assert "freshness: release not recorded · refresh NoAutomaticRefresh" in text


def test_country_block_names_what_it_is_contained_in() -> None:
    text = _render(node_fixture("node_country"))

    assert "contained in: Africa" in text


def test_containment_entry_with_a_null_label_renders_its_ref() -> None:
    node = node_fixture("node_country")
    assert isinstance(node, CountryNode)
    place = node.country.model_copy(
        update={"contained_in": [NodeRef(ref="undata-geo/G00134000", label=None)]}
    )
    node = node.model_copy(update={"country": place})

    assert "contained in: undata-geo/G00134000" in _render(node)


def test_containment_is_capped_with_a_remainder_count() -> None:
    node = node_fixture("node_country")
    assert isinstance(node, CountryNode)
    place = node.country.model_copy(
        update={"contained_in": [NodeRef(ref=f"r{i}", label=f"R{i}") for i in range(11)]}
    )
    node = node.model_copy(update={"country": place})

    assert "contained in: R0, R1, R2, R3, R4, R5, R6, R7, and 3 more" in _render(node)


def test_group_and_region_blocks_carry_their_authority_and_member_count() -> None:
    group = _render(node_fixture("node_group"))
    continent = _render(node_fixture("node_continent"))

    assert "membership: resolvekit curated membership" in group
    assert "member countries: 7" in group
    assert "membership: resolvekit UN M49 containment" in continent
    assert "member countries: 23" in continent


# 1b. An `other` node whose types are all Enum-suffixed carries a line naming it a
# classification value; an ordinary one does not.


def test_classification_value_carries_the_flag_line() -> None:
    node = node_fixture("node_other", types=["GeographicalIndicatorEnum"])

    assert "classification value" in _render(node)


@pytest.mark.parametrize("types", [["Thing"], ["GeographicalIndicatorEnum", "Dataset"], []])
def test_other_node_without_only_enum_types_carries_no_classification_line(
    types: list[str],
) -> None:
    node = node_fixture("node_other", types=types)

    assert "classification value" not in _render(node)


# 2. An eight-id mixed-kind digest stays under a stated character ceiling.

_DIGEST_CHARACTER_CEILING = 6000


def test_eight_id_mixed_kind_digest_stays_under_the_character_ceiling() -> None:
    nodes = [node_fixture(name) for name in NODE_FIXTURES[:4]]
    wide = node_fixture("node_country")
    assert isinstance(wide, CountryNode)
    wide = wide.model_copy(
        update={
            "ref": "country/WIDE",
            "country": wide.country.model_copy(
                update={"contained_in": [NodeRef(ref=f"r{i}", label="x" * 40) for i in range(200)]}
            ),
        }
    )
    other = OtherNode.model_validate(
        {**node_fixture("node_other").model_dump(), "gaps": [f"gap.{i}" for i in range(200)]}
    )
    outcomes = {node.ref: node for node in [*nodes, wide, other]}
    dcids = list(outcomes) + list(outcomes)[:2]

    text = render_describe_digest(dcids=dcids, outcomes=outcomes)

    assert len(text) < _DIGEST_CHARACTER_CEILING


# 3. Unheld dcids appear by name in the digest text, once each.


def test_unheld_dcids_appear_by_name_once() -> None:
    held = node_fixture("node_country")
    unheld_dcid = "country/ZZZ"

    text = render_describe_digest(
        dcids=[held.ref, unheld_dcid, unheld_dcid],
        outcomes={held.ref: held, unheld_dcid: NotHeldEntry(ref=unheld_dcid, status="not_held")},
    )

    assert text.endswith(f"Not held by this deployment's graph: {unheld_dcid}.")
    assert held.ref in text
