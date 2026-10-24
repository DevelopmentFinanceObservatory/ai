"""The anchor record: one resolution of the latest period of a recipe's data,
kept in the materialisation store as an object of its own.

A rolling verb call names its periods relative to the latest period of the data
and is sent the literal periods that stood for. The record is what those
literals are checked against. It is written once, by the call that resolved the
period (`analysis_mcp.rolling_tool.resolve_anchor`), and read by every rolling
verb call and by `common.recipe.recipe_anchor`.

A record holds what was resolved from beside the result:

- `populations`: the populations the period was taken over, in the spelling
  `common.rolling.canonical_populations` gives them. A population's `levels` is
  written only when it holds a column to a level; `strict_levels` records the
  held columns that must classify every entity. A pairing population
  (`AnchorPairing`) holds the populations of a keyed verb's two sides and the
  keys it pairs them on.
- `roots`: the `{spec_id, content_version}` pair of each root frame read, sorted.
- `rule`: the name of the rule that chose the period.
- `period`: the latest period.
- `left_out`: each series counted that holds no value at `period`, as the index
  of its population in `populations` and its `coordinates`, the series' value
  in each of its columns. A column the series holds no value in is absent.

The first three are the preimage of the key the record is stored under
(`common.rolling.anchor_key`), and a frame's bytes are a function of its pair,
so the whole record is a function of its key. `encode_anchor_record` is the
RFC 8785 form of the record, so two writers resolving the same key write the
same bytes, and the store keeps the first.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import rfc8785
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    TypeAdapter,
    model_serializer,
)

from common.materialisation.frame_ref import FrameRef
from common.rolling import ANCHOR_KEY_PATTERN, anchor_key, canonical_populations, canonical_roots


class AnchorRecordIntegrityError(Exception):
    """The store holds something under `anchor_key` that is no record of that
    key: it does not decode, or its populations, roots and rule hash to another
    key."""

    def __init__(self, anchor_key: str, *, reason: str) -> None:
        self.anchor_key = anchor_key
        self.reason = reason
        super().__init__(f"anchor record integrity check failed: {anchor_key}: {reason}")


class AnchorPopulation(BaseModel):
    """One population of a record: a root dataset, the bindings narrowing its
    series, and the level each place-type column is held to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str
    bindings: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    levels: dict[str, str] = Field(default_factory=dict)
    strict_levels: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _without_unset_levels(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped = handler(self)
        if not dumped["levels"]:
            del dumped["levels"]
        if not dumped["strict_levels"]:
            del dumped["strict_levels"]
        return dumped


class AnchorPairing(BaseModel):
    """One pairing population of a record: the populations of a keyed verb's
    two sides, left then right, and each left key column to the right column
    it pairs with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair: tuple[tuple[AnchorEntry, ...], tuple[AnchorEntry, ...]]
    keys: dict[str, str]


def _entry_kind(value: object) -> str:
    if isinstance(value, AnchorPairing) or (isinstance(value, dict) and "pair" in value):
        return "pair"
    return "root"


# One population of a record, flat or pairing.
type AnchorEntry = Annotated[
    Annotated[AnchorPopulation, Tag("root")] | Annotated[AnchorPairing, Tag("pair")],
    Discriminator(_entry_kind),
]
_ENTRIES: TypeAdapter[tuple[AnchorEntry, ...]] = TypeAdapter(tuple[AnchorEntry, ...])


class LeftOutSeries(BaseModel):
    """A series that holds no value at the latest period. `root` indexes the
    record's `populations`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: int = Field(ge=0)
    coordinates: dict[str, str]


def _canonical_populations(populations: tuple[AnchorEntry, ...]) -> tuple[AnchorEntry, ...]:
    canonical = canonical_populations(
        population.model_dump(mode="json") for population in populations
    )
    return _ENTRIES.validate_python(canonical)


def _canonical_roots(roots: tuple[FrameRef, ...]) -> tuple[FrameRef, ...]:
    return tuple(FrameRef(**root) for root in canonical_roots(root.model_dump() for root in roots))


def _canonical_left_out(left_out: tuple[LeftOutSeries, ...]) -> tuple[LeftOutSeries, ...]:
    """`left_out` by population, then by each series' RFC 8785 bytes, each once."""
    keyed = {(series.root, rfc8785.dumps(series.coordinates)): series for series in left_out}
    return tuple(keyed[key] for key in sorted(keyed))


class AnchorRecord(BaseModel):
    """One resolution of a latest period. See the module docstring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    populations: Annotated[
        tuple[AnchorEntry, ...], Field(min_length=1), AfterValidator(_canonical_populations)
    ]
    roots: Annotated[tuple[FrameRef, ...], Field(min_length=1), AfterValidator(_canonical_roots)]
    rule: str = Field(min_length=1)
    period: str = Field(min_length=1)
    left_out: Annotated[tuple[LeftOutSeries, ...], AfterValidator(_canonical_left_out)] = ()

    @property
    def anchor_key(self) -> str:
        """The key this record is stored under."""
        return anchor_key(
            populations=[population.model_dump(mode="json") for population in self.populations],
            roots=[root.model_dump() for root in self.roots],
            rule=self.rule,
        )


def encode_anchor_record(record: AnchorRecord) -> bytes:
    """The bytes a store writes for `record`: its RFC 8785 form."""
    return rfc8785.dumps(record.model_dump(mode="json"))


def decode_anchor_record(data: bytes) -> AnchorRecord:
    """The record `encode_anchor_record` wrote as `data`. A member a later
    deployment added beside the record's own is dropped.

    Raises:
        ValueError: When `data` is not JSON or does not fit the record's shape
            (`pydantic.ValidationError` is a `ValueError`).
    """
    stored: Any = json.loads(data)
    if not isinstance(stored, dict):
        raise ValueError("a stored anchor record is a JSON object")
    return AnchorRecord.model_validate(
        {name: value for name, value in stored.items() if name in AnchorRecord.model_fields}
    )


__all__ = [
    "ANCHOR_KEY_PATTERN",
    "AnchorEntry",
    "AnchorPairing",
    "AnchorPopulation",
    "AnchorRecord",
    "AnchorRecordIntegrityError",
    "LeftOutSeries",
    "decode_anchor_record",
    "encode_anchor_record",
]
