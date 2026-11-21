"""The `resolve_anchor` tool's signature and docstring, its result, and the checks on
its arguments that read no frame.

A rolling call states a period relative to the latest period of the data and
carries the literal period that stood for. `resolve_anchor` finds that latest
period once, over the populations of a question (`analysis_mcp.op_spec.Populations`)
and its root frames, and stores it as an anchor record
(`common.materialisation.AnchorRecord`) under `common.rolling.anchor_key`.
`verbs.resolve_anchor` registers the signature and docstring here, runs
`_too_many` and `_unmatched`, and asks dfo-apis' analysis runner for the period
(`runner_client.find_anchor`).

An expression of more than `MAX_DISTINCT_POPULATIONS` distinct populations, flat
and pairing, refuses `populations_too_many` before any call. A flat population
with no root frame of its `spec_id`, or a root frame matching no population,
refuses too.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Annotated, Any

from common.materialisation import FrameRef, LeftOutSeries
from common.rolling import ANCHOR_KEY_PATTERN
from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict, Field

from analysis_mcp.op_spec import PairingPopulation, Population, Populations, RootPopulation
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result

RESOLVE_ANCHOR_VERSION = "1"

# The most distinct populations, flat and pairing, one `resolve_anchor` call
# reads. Each is one reading of the admitted rows of its datasets, and a
# compiled plan states one for each root dataset binding and keyed verb, which
# its at most 24 requirements keep well under this.
MAX_DISTINCT_POPULATIONS = 64


class _AnchorRequest(BaseModel):
    """`resolve_anchor`'s arguments, the populations in their one spelling."""

    model_config = ConfigDict(extra="forbid")

    populations: Populations
    roots: tuple[FrameRef, ...] = Field(min_length=1)


class ResolvedAnchor(BaseModel):
    """What `resolve_anchor` returns as `structuredContent`.

    - `anchor_key`: the handle of the stored record, passed to each rolling
      `growth`, `rank` and `filter` call as `anchor_key`.
    - `period`: the latest period, which a rolling literal at offset 0 is.
    - `left_out`: each series counted that holds no value at `period`, as the
      index of its population in the canonical populations and its
      `coordinates`, the series' value in each of its columns.
    """

    model_config = ConfigDict(extra="forbid")

    anchor_key: str = Field(pattern=ANCHOR_KEY_PATTERN)
    period: str
    left_out: tuple[LeftOutSeries, ...] = ()


RESOLVE_ANCHOR_OUTPUT_SCHEMA: dict[str, Any] = ResolvedAnchor.model_json_schema(
    mode="serialization"
)


def _unresolved(detail: str, reason: str, **context: object) -> ToolResult:
    return to_tool_result(
        ToolRefusal.rolling_anchor_unresolved(detail=detail, reason=reason, context=context)
    )


def _flat(populations: Iterable[Population]) -> Iterator[RootPopulation]:
    """Every flat population of `populations`, those on each side of a pairing included."""
    for population in populations:
        if isinstance(population, PairingPopulation):
            for side in population.pair:
                yield from _flat(side)
        else:
            yield population


def _content(population: Population) -> str:
    """`population`'s identity: its JSON text, one for each content since a
    population is held in its one spelling."""
    return population.model_dump_json()


def _too_many(request: _AnchorRequest) -> ToolResult | None:
    """The refusal for populations of more than `MAX_DISTINCT_POPULATIONS`
    distinct entries, `None` otherwise. A repeated entry is walked once."""
    seen: set[str] = set()
    pending: list[Population] = list(request.populations)
    while pending:
        population = pending.pop()
        if (key := _content(population)) in seen:
            continue
        seen.add(key)
        if len(seen) > MAX_DISTINCT_POPULATIONS:
            return _unresolved(
                "populations_too_many",
                f"the populations hold more than {MAX_DISTINCT_POPULATIONS} distinct entries",
                ceiling=MAX_DISTINCT_POPULATIONS,
            )
        if isinstance(population, PairingPopulation):
            for side in population.pair:
                pending.extend(side)
    return None


def _unmatched(request: _AnchorRequest) -> ToolResult | None:
    """The refusal for a flat population or a root frame with no counterpart,
    `None` when each has one."""
    flat = list(_flat(request.populations))
    for population in flat:
        if all(root.spec_id != population.spec_id for root in request.roots):
            return _unresolved(
                "population_without_root",
                f"no root frame was given for the population over {population.spec_id}",
                spec_id=population.spec_id,
            )
    for root in request.roots:
        if all(root.spec_id != population.spec_id for population in flat):
            return _unresolved(
                "root_without_population",
                f"the root frame {root.spec_id} matches the spec_id of no population",
                spec_id=root.spec_id,
            )
    return None


async def resolve_anchor(
    populations: Annotated[
        Populations,
        Field(
            description=(
                "The populations the latest period is taken over: one {spec_id, bindings} "
                "for each root dataset of the question, bindings narrowing its series as "
                "{column: [values]}, and levels, when set, holding a place-type column to "
                "one level of place as {column: level}. A keyed share_of or "
                "join whose two sides both count is one {pair: [left populations, right "
                "populations], keys: {left column: right column}} instead. The same value "
                "goes on every rolling call as rolling.population."
            )
        ),
    ],
    roots: Annotated[
        tuple[FrameRef, ...],
        Field(
            min_length=1,
            description=(
                "The {spec_id, content_version} pair of every root frame of the question, "
                "as get_data returned it. Each population is counted over every frame "
                "given with its spec_id."
            ),
        ),
    ],
) -> ToolResult:
    """Find the latest period of the data a question's rolling periods follow.

    The latest period is the newest calendar year at least half of the series
    of every population hold. A series is one combination of a frame's columns
    other than `time_period`, `value` and `status`.

    Use it before a rolling `growth`, `rank` or `filter` call, to find the
    latest period the data holds.

    The result is stored, and its `anchor_key` goes on every rolling `growth`,
    `rank` and `filter` call of the question, whose literal periods are
    `period` moved back by each rolling offset. This tool computes no frame
    and returns no {spec_id, content_version} pair.

    Args:
        populations: The populations the latest period is taken over.
        roots: The pair of every root frame of the question.

    Returns:
        The `anchor_key`, the latest `period`, and `left_out`: each series
        counted that has no value at that period. On refusal, a typed error
        with a machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.resolve_anchor runs the tool")
