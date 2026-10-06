"""Seeds built at the bounds `harness.wire` holds a seed to.

`test_context_budget.py` measures what such a seed puts into the planner's
brief, a tool result and session state, and `translator/test_mechanical.py`
what `translate` returns for a plan that retrieves one many times. Search's
answer for each identity is the block the seed carries (`read_seed`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from compiler.sources import source_from_spec_block
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import MAX_IDENTITY_BYTES
from harness.seeds import ReadSeed, TakenSeed, read_answer
from harness.wire import (
    MAX_OPERATION_PHRASE_CHARS,
    MAX_SEED_COUNT,
    MAX_SEED_ENTITIES,
    MAX_SEED_LIST_ITEMS,
    MAX_SEED_OPERATIONS,
    MAX_SEED_RECORD_CHARS,
    MAX_SEED_ROLES,
    MAX_SEED_SENTENCE_CHARS,
    MAX_SEED_SOURCE_CHARS,
    MAX_SEED_TEXT_CHARS,
    MAX_TARGET_YEAR,
    Seed,
)
from planner.plan import SeededSource
from translator.discovery import compact_source


def read_seed(index: int, seed: Mapping[str, Any]) -> ReadSeed:
    """`seed` as `harness.graph`'s `seed` node reads it when Search answers its
    identity with the block the seed carries.

    Raises:
        ValueError: As `harness.seeds.read_answer` does, for a block past a cap.
    """
    taken = TakenSeed(index=index, seed=Seed.model_validate(seed))
    return read_answer(taken, {"status": "definite", "spec": seed["spec"]})


def seed_text_at(tag: str, cap: int = MAX_SEED_TEXT_CHARS) -> str:
    """A string of exactly `cap` characters that starts with `tag`, so each one is distinct."""
    return (tag + "x" * cap)[:cap]


def bounded_seed(
    number: int,
    *,
    entities: int = 0,
    open_members: bool = False,
    identity_bytes: int = 0,
) -> dict[str, Any]:
    """One `harness.wire.Seed` whose identity hashes to its spec_id.

    Every string is at its bound, and the sentence starts with `number`, so
    two seeds describe two datasets. `entities` counts the resolved entities;
    `open_members` adds the members
    the planner's envelope states at their own bounds (coverage dimensions,
    open axes, roles, operations); `identity_bytes` pads the identity to that
    size.
    """
    identity: dict[str, Any] = {
        "identity_version": "spec1",
        "seed": number,
        "shape": {"kind": seed_text_at("k")},
        "requested_window": {
            "kind": "range",
            "start_year": MAX_SEED_COUNT - 1,
            "end_year": MAX_SEED_COUNT,
        },
        "padding": "",
    }
    resolved: dict[str, Any] = {
        "rendered_sentence": seed_text_at(f"s{number}", MAX_SEED_SENTENCE_CHARS),
        "confidence": seed_text_at("c"),
        "entities": [
            {
                "role": seed_text_at(f"e{index:04d}"),
                "ref": seed_text_at(f"r{index:04d}"),
                "label": seed_text_at("l"),
            }
            for index in range(entities)
        ],
        "facets": {
            "unit": {"ref": seed_text_at("u"), "label": seed_text_at("ul")},
            "provenance": {"ref": seed_text_at("p"), "label": seed_text_at("pl")},
        },
        "coverage": {"kind": "exact", "has_data": True, "window": {"start": 1000, "end": 9999}},
        "stat_vars": {
            "member_count": MAX_SEED_COUNT,
            "member_count_exact": True,
            "served_count": MAX_SEED_COUNT,
        },
        "shape": {"member_digest": seed_text_at("g"), "member_count": MAX_SEED_COUNT},
        "headline": {"year": MAX_SEED_COUNT, "year_named": False, "observations": MAX_SEED_COUNT},
    }
    operations: list[dict[str, Any]] = []
    if open_members:
        identity["entities"] = {
            seed_text_at(f"role{index:02d}"): ["x"] for index in range(MAX_SEED_ROLES)
        }
        resolved["coverage"]["dimensions"] = [
            {"label": seed_text_at("d"), "count": MAX_SEED_COUNT}
            for _ in range(MAX_SEED_LIST_ITEMS)
        ]
        resolved["slots"] = [
            {
                "key": {"property": {"dcid": seed_text_at(f"p{index:02d}")}},
                "binding": {"kind": "unbound"},
                "summable": True,
            }
            for index in range(MAX_SEED_LIST_ITEMS)
        ]
        resolved["role_axis_facts"] = [
            {"role": seed_text_at(f"a{index:02d}"), "member_count": MAX_SEED_COUNT}
            for index in range(MAX_SEED_LIST_ITEMS)
        ]
        resolved["scope"] = {"axis": seed_text_at("z"), "members": []}
        operations = [
            {
                "verb": "forecast",
                "phrase": seed_text_at("f", MAX_OPERATION_PHRASE_CHARS),
                "over": "destination",
                "target_year": MAX_TARGET_YEAR,
            }
        ] * MAX_SEED_OPERATIONS
    room = identity_bytes - len(json.dumps(identity, separators=(",", ":")).encode())
    identity["padding"] = "x" * max(room, 0)
    spec_id = spec_id_for(canonicalize(identity))
    return {
        "spec": {
            "spec_id": spec_id,
            "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            "resolved": resolved,
        },
        "operations": operations,
    }


def most_entities_under(cap: int, measure: Callable[[int], int]) -> int:
    """The largest entity count whose `measure` stays at or under `cap`."""
    low, high = 0, MAX_SEED_ENTITIES
    while low < high:
        middle = (low + high + 1) // 2
        if measure(middle) <= cap:
            low = middle
        else:
            high = middle - 1
    return low


def fullest_described_seed(number: int) -> tuple[dict[str, Any], int]:
    """A seed whose description is as long as `MAX_SEED_SOURCE_CHARS` allows: as many
    entities as fit. It lists no StatVars, which the description counts and never
    names, so its record stays under `MAX_SEED_RECORD_CHARS`. Returns it with that count."""

    def measure(entities: int) -> int:
        source = source_from_spec_block(bounded_seed(number, entities=entities)["spec"])
        return len(json.dumps(compact_source(source)))

    entities = most_entities_under(MAX_SEED_SOURCE_CHARS, measure)
    return bounded_seed(number, entities=entities), entities


def recorded_seed(number: int, *, entities: int) -> dict[str, Any]:
    """A seed with the identity at data-mcp's bound and the open members at theirs."""
    return bounded_seed(
        number, entities=entities, open_members=True, identity_bytes=MAX_IDENTITY_BYTES
    )


def most_recorded_entities(number: int) -> int:
    """How many entities such a seed holds before its record passes `MAX_SEED_RECORD_CHARS`."""

    def measure(entities: int) -> int:
        block = recorded_seed(number, entities=entities)["spec"]
        record = SeededSource(source=source_from_spec_block(block))
        return len(json.dumps(record.model_dump(mode="json")))

    return most_entities_under(MAX_SEED_RECORD_CHARS, measure)


def fullest_recorded_seed(number: int) -> dict[str, Any]:
    """A seed whose record is as long as `MAX_SEED_RECORD_CHARS` allows."""
    return recorded_seed(number, entities=most_recorded_entities(number))
