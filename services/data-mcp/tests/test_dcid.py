"""Tests for the dcid syntax the graph's node routes accept."""

from __future__ import annotations

import pytest
from data_mcp.dcid import DCID_MAX_LENGTH, bare_dcid, has_dot_segment, is_dcid


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("country/KEN", True),
        ("sdg/DC_TOF_HLTHL", True),
        ("a" * DCID_MAX_LENGTH, True),
        ("a" * (DCID_MAX_LENGTH + 1), False),
        ("humanitarian aid", False),
        ("x/", False),
        ("oecd:DAC", False),
        ("dcid:country/KEN", False),
        ("", False),
    ],
)
def test_is_dcid_mirrors_the_nodes_route_pattern_and_length_cap(ref: str, expected: bool) -> None:
    assert is_dcid(ref) is expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [("..", True), (".", True), ("a/..", True), ("country/../x", True), ("a.b/c", False)],
)
def test_has_dot_segment_names_a_segment_of_exactly_dots(ref: str, expected: bool) -> None:
    assert has_dot_segment(ref) is expected


def test_bare_dcid_strips_the_marker_once() -> None:
    assert bare_dcid("dcid:country/KEN") == "country/KEN"
    assert bare_dcid("country/KEN") == "country/KEN"
