"""Tests for `render_describe_digest`: pure rendering, no client, off the node fixtures."""

from __future__ import annotations

import pytest
from data_mcp.describe_digest import render_describe_digest
from data_mcp.node_lookup import (
    ContainedInPlace,
    EntityNode,
    Freshness,
    GenericNode,
    NodeShape,
    NotHeldNode,
    SourceNode,
    StatVarNode,
)
from pydantic import TypeAdapter

from .node_fixtures import load_node_fixture

_ADAPTER: TypeAdapter[NodeShape] = TypeAdapter(NodeShape)


def _node(
    fixture_name: str, **overrides: object
) -> StatVarNode | SourceNode | EntityNode | GenericNode:
    payload = {**load_node_fixture(fixture_name), **overrides}
    return _ADAPTER.validate_python(payload)


# 1. One rendering test per shape asserting every contract field appears.


def test_stat_var_block_carries_every_contract_field() -> None:
    node = _node("node_stat_var")

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.label in text
    assert node.dcid in text
    assert node.definition is not None
    assert node.definition in text
    for role in node.roles:
        assert role in text
    assert node.population_type in text


def test_stat_var_without_a_definition_renders_label_and_roles_only() -> None:
    node = _node("node_stat_var", definition=None)

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.label in text
    assert "None" not in text
    assert text.index(node.dcid) < text.index("roles:")


@pytest.mark.parametrize(
    "fixture_name", ["node_stat_var", "node_source", "node_entity", "node_generic"]
)
def test_a_node_without_a_label_is_headed_by_its_dcid(fixture_name: str) -> None:
    node = _node(fixture_name, label=None)

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert text.startswith(node.dcid)
    assert "None" not in text


def test_source_block_carries_every_contract_field() -> None:
    node = _node("node_source")

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.label in text
    assert node.dcid in text
    assert node.license in text
    assert node.license_type in text
    assert node.description in text
    assert node.url in text
    assert node.freshness.release_frequency in text
    assert node.freshness.refresh_schedule in text


def test_entity_block_carries_every_contract_field() -> None:
    node = _node("node_entity")

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.label in text
    assert node.dcid in text
    assert node.entity_type in text
    for member in node.contained_in:
        assert (member.label or member.dcid) in text


def test_generic_block_carries_every_contract_field() -> None:
    node = _node("node_generic")

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.label in text
    assert node.dcid in text
    for type_name in node.types:
        assert type_name in text
    for key, values in node.properties.items():
        assert key in text
        for value in values:
            assert value in text


# A generic node whose types are all Enum-suffixed carries a line naming it
# as a classification value; an ordinary generic node does not.


def test_classification_value_generic_node_carries_the_flag_line() -> None:
    node = GenericNode(
        shape="node",
        dcid="Japan",
        label="Japan",
        types=["GeographicalIndicatorEnum"],
        properties={},
    )

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "classification value" in text
    assert "search" in text


def test_ordinary_generic_node_carries_no_classification_line() -> None:
    node = _node("node_generic")
    assert node.is_classification_value is False

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "classification value" not in text


# 2. The StatVar gap line names all three unavailable fields.


def test_stat_var_gap_line_names_all_three_when_absent() -> None:
    node = _node("node_stat_var")
    assert node.facets is None
    assert node.processing_level is None
    assert node.sources is None

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "facet values" in text
    assert "processing level" in text
    assert "sources" in text


def test_stat_var_gap_line_absent_when_all_three_are_populated() -> None:
    node = _node(
        "node_stat_var", processing_level="raw", facets={"unit": ["USD"]}, sources=["dc/base/OECD"]
    )

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "does not serve them" not in text
    assert "processing level: raw" in text
    assert "facet axis unit: USD" in text
    assert "sources: dc/base/OECD" in text


def test_stat_var_partial_fields_renders_present_and_names_absent() -> None:
    node = _node("node_stat_var", processing_level="raw")
    assert node.facets is None
    assert node.sources is None

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "processing level: raw" in text
    assert "the facet values per axis" in text
    assert "its sources" in text
    assert "its processing level" not in text


def test_stat_var_sources_in_unrecognised_shape_names_the_field() -> None:
    node = _node("node_stat_var", sources={"unexpected": "mapping"})

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "sources: arrived in a shape this digest does not recognise" in text


def test_stat_var_facets_in_unrecognised_shape_names_the_field() -> None:
    node = _node("node_stat_var", facets=["not", "a", "mapping"])

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "facet values: arrived in a shape this digest does not recognise" in text


# 3. A source node with a null release_frequency renders `not recorded` for
# it and the real value for refresh_schedule -- the BIS-style asymmetry.


def test_source_null_release_frequency_renders_not_recorded() -> None:
    node = _node("node_source")
    node = node.model_copy(
        update={"freshness": Freshness(release_frequency=None, refresh_schedule="P3M")}
    )

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "release not recorded" in text
    assert "refresh P3M" in text


# 4. A source node with both licence fields and both freshness fields
# renders all four.


def test_source_with_all_four_fields_populated_renders_all_four() -> None:
    node = _node("node_source")

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert node.license in text
    assert node.license_type in text
    assert node.freshness.release_frequency in text
    assert node.freshness.refresh_schedule in text
    assert "not recorded" not in text


# 5. An entity's `contained_in` entry with a null label renders its dcid.


def test_entity_null_label_containment_entry_renders_its_dcid() -> None:
    node = EntityNode(
        shape="entity",
        dcid="country/KEN",
        label="Kenya",
        entity_type="geo.country",
        contained_in=[ContainedInPlace(dcid="undata-geo/G00134000", label=None)],
    )

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "undata-geo/G00134000" in text


# 6. A generic node with 40 properties renders at most the property cap plus
# a remainder count; a property with 30 values renders at most the value cap
# plus a remainder count.


def test_generic_node_caps_properties_and_values_with_remainder_counts() -> None:
    properties = {"wide_prop": [f"val{i}" for i in range(30)]}
    properties.update({f"prop{i}": [f"v{i}"] for i in range(40)})
    node = GenericNode(
        shape="node",
        dcid="dc/base/Wide",
        label="Wide node",
        types=["Thing"],
        properties=properties,
    )

    text = render_describe_digest(dcids=[node.dcid], outcomes={node.dcid: node})

    assert "more properties not shown" in text
    assert "val29" not in text
    assert "more" in text


# 7. An eight-id mixed-shape digest stays under a stated character ceiling.

_DIGEST_CHARACTER_CEILING = 6000


def test_eight_id_mixed_shape_digest_stays_under_the_character_ceiling() -> None:
    stat_var = _node("node_stat_var")
    source = _node("node_source")
    entity = _node("node_entity")
    generic = GenericNode(
        shape="node",
        dcid="dc/base/Wide",
        label="Wide node",
        types=["Thing"],
        properties={f"prop{i}": [f"v{i}"] for i in range(40)},
    )
    dcids = [stat_var.dcid, source.dcid, entity.dcid, generic.dcid] * 2
    outcomes = {
        stat_var.dcid: stat_var,
        source.dcid: source,
        entity.dcid: entity,
        generic.dcid: generic,
    }

    text = render_describe_digest(dcids=dcids, outcomes=outcomes)

    assert len(text) < _DIGEST_CHARACTER_CEILING


# 8. Unheld dcids appear by name in the digest text.


def test_unheld_dcids_appear_by_name() -> None:
    held = _node("node_entity")
    unheld_dcid = "country/ZZZ"

    text = render_describe_digest(
        dcids=[held.dcid, unheld_dcid],
        outcomes={held.dcid: held, unheld_dcid: NotHeldNode(dcid=unheld_dcid)},
    )

    assert unheld_dcid in text
    assert held.dcid in text
