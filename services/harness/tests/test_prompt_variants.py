"""The process's surface picks the orchestrator and planner instructions.

The `default` set is pinned byte for byte by the recorded requests in
`fixtures/orchestrator_requests` and `planner/test_recorded_requests.py`.
"""

from __future__ import annotations

import pytest
from harness.surface import FLASH_SURFACE, HYDROGEN_SURFACE, set_studio_surface
from orchestrator.agent import DEFAULT_INSTRUCTION, HYDROGEN_INSTRUCTION
from planner.instruction import planner_instruction


@pytest.mark.parametrize("surface", [FLASH_SURFACE, HYDROGEN_SURFACE])
def test_the_planner_serves_the_instruction_set_of_the_process_s_surface(surface) -> None:
    set_studio_surface(surface)

    assert planner_instruction() == planner_instruction(surface.variant)


def test_each_instruction_set_is_its_own() -> None:
    assert planner_instruction("hydrogen") != planner_instruction("default")
    assert HYDROGEN_INSTRUCTION != DEFAULT_INSTRUCTION


@pytest.mark.parametrize(
    ("variant", "rule"),
    [
        ("default", "The bilateral share of US aid widens 'provider'"),
        ("default", "Leave it unset for a sector share"),
        ("hydrogen", "A donor's bilateral share of its aid widens 'provider'"),
        ("hydrogen", "Leave it unset for a sector share"),
        ("hydrogen", "the `axis` it widens"),
    ],
)
def test_both_planner_instructions_take_a_share_s_whole_from_search(variant, rule) -> None:
    assert rule in planner_instruction(variant)


@pytest.mark.parametrize("variant", ["default", "hydrogen"])
def test_no_planner_instruction_keeps_the_rules_search_replaced(variant) -> None:
    # Before Search derived a share's whole, a sector share opened 'sector' and
    # a provider share left `whole_of` unset.
    instruction = planner_instruction(variant)

    assert "opens 'sector'" not in instruction
    assert "a provider share" not in instruction
