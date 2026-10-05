"""Entity resolution seam: human-typed names to graph dcids, via dfo-apis' Graph door.

`get_data`'s scope block carries entities as spoken text ("France", "Sub-Saharan Africa")
keyed by role. `resolve_entities` turns that into structured outcomes -- one of six wire
statuses, or a local validation failure -- without ever building a `ToolRefusal`. The
calling tool owns that translation (`data_mcp.tool_errors`), which keeps that module the
sole owner of the refusal grammar; this module returns plain data, never imports it, and
never catches `UpstreamError`.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2

from data_mcp.client import GRAPH_PREFIX
from data_mcp.client_holder import get_client

_RESOLVE_ROUTE = f"{GRAPH_PREFIX}/resolve"

# The door's own published limits on `POST /api/graph/resolve`'s `texts` list.
_MAX_TEXTS_PER_REQUEST = 25
_MAX_NAME_CHARS = 200

# The door constrains matching to exactly these three types. `NotHeld.is_region` relies on
# this being the complete set: anything other than an individual country is a region.
_REGION_ENTITY_TYPES = frozenset({"geo.subregion", "geo.continent"})


@dataclass(frozen=True, kw_only=True)
class ResolvedEntity:
    """Resolution status `resolved`: the name maps unambiguously to one graph entity."""

    dcid: str
    name: str
    entity_type: str


@dataclass(frozen=True, kw_only=True)
class RegionMember:
    """One member of a resolved region that this deployment's graph holds."""

    dcid: str
    name: str


@dataclass(frozen=True, kw_only=True)
class UnheldRegionMember:
    """One member resolvekit named for a region that this deployment's graph does not hold.

    Carries no `dcid` -- the wire's `members_not_held` entries never do, since there is
    nothing held to point one at. Carries no `entity_type` either -- nothing downstream
    reads it; a caller only ever names an unheld member by `name`.
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class ResolvedRegion:
    """Resolution status `resolved_region`: the name maps to a region with a held member.

    `members` is what a caller can actually use as scope entities. `members_not_held` can
    run past 40 entries for a large region (a continent's worth of unheld countries); it
    exists so a caller can tell a partial region apart from a complete one, not to be
    resolved further.
    """

    name: str
    entity_type: str
    members: list[RegionMember]
    members_not_held: list[UnheldRegionMember]


@dataclass(frozen=True, kw_only=True)
class AmbiguousCandidate:
    """One candidate graph entity offered for a name that matched more than one."""

    dcid: str
    name: str
    entity_type: str
    parent: str | None


@dataclass(frozen=True, kw_only=True)
class Ambiguous:
    """Resolution status `ambiguous`: more than one graph entity matches the name."""

    candidates: list[AmbiguousCandidate]
    truncated: bool


@dataclass(frozen=True, kw_only=True)
class NearMatch:
    """One near-miss the door offers alongside a `not_found` name.

    Carries only `name` -- nothing downstream reads a near match's `entity_type`, `dcid`,
    or `confidence`; the refusal grammar surfaces `name` alone as a "did you mean" hint.
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class NotFound:
    """Resolution status `not_found`: nothing in the graph matches the name.

    `near_matches` carries a distinction the wire status alone collapses: non-empty means
    the name is probably a misspelling of one of these; empty means resolvekit found nothing
    close at all, e.g. an organisation name like "World Health Organization" -- the door only
    ever matches {geo.country, geo.subregion, geo.continent}, and there is no near-miss to
    offer for a name outside that space.
    """

    near_matches: list[NearMatch]


@dataclass(frozen=True, kw_only=True)
class NotHeld:
    """Resolution status `not_held`: resolvekit resolved the name, but this deployment's
    graph does not hold it -- a fact distinct from "matched nothing".
    """

    name: str
    entity_type: str

    @property
    def is_region(self) -> bool:
        """Whether this is a region whose members are entirely unheld, not one unheld entity.

        The wire's `not_held` status carries only `name` and `entity_type`, collapsing two
        different facts onto one status: an individual entity resolvekit knows but the graph
        doesn't hold (`entity_type == "geo.country"`), and a region whose members are ALL
        unheld, which the door reports as `not_held` rather than `resolved_region` with an
        empty `members` list. The door's constraint to {geo.country, geo.subregion,
        geo.continent} is what makes `entity_type` alone enough to tell the two apart.
        """
        return self.entity_type in _REGION_ENTITY_TYPES


@dataclass(frozen=True, kw_only=True)
class OutOfScope:
    """Resolution status `out_of_scope`: the name resolves to a real entity that lies
    outside the door's `geo.country`/`geo.subregion`/`geo.continent` scope -- a
    continental union, an international organisation, or a Data Commons source, for
    example. `entity_type` is resolvekit's own tag for it. `dcid` is populated only when
    this deployment's graph holds the entity, and `None` when it does not.
    """

    name: str
    entity_type: str
    dcid: str | None


@dataclass(frozen=True, kw_only=True)
class InvalidName:
    """A name rejected by local validation before any request reached the door.

    Never sent upstream: sending it would burn a request only to eat a 400 that fails the
    whole batch (`texts` entries must be 1..200 characters after whitespace stripping).
    """

    name: str
    reason: str


ResolutionOutcome = (
    ResolvedEntity | ResolvedRegion | Ambiguous | NotFound | NotHeld | OutOfScope | InvalidName
)


def _outcome_from_result(result: Mapping[str, Any]) -> ResolutionOutcome:
    """Translate one wire result (one of six `status` variants) into a `ResolutionOutcome`."""
    status = result["status"]
    if status == "resolved":
        entity = result["entity"]
        return ResolvedEntity(
            dcid=entity["dcid"], name=entity["name"], entity_type=entity["entity_type"]
        )
    if status == "resolved_region":
        region = result["region"]
        return ResolvedRegion(
            name=region["name"],
            entity_type=region["entity_type"],
            members=[RegionMember(dcid=m["dcid"], name=m["name"]) for m in result["members"]],
            members_not_held=[
                UnheldRegionMember(name=m["name"]) for m in result["members_not_held"]
            ],
        )
    if status == "ambiguous":
        return Ambiguous(
            candidates=[
                AmbiguousCandidate(
                    dcid=c["dcid"],
                    name=c["name"],
                    entity_type=c["entity_type"],
                    parent=c["parent"],
                )
                for c in result["candidates"]
            ],
            truncated=result["truncated"],
        )
    if status == "not_found":
        return NotFound(near_matches=[NearMatch(name=m["name"]) for m in result["near_matches"]])
    if status == "not_held":
        return NotHeld(name=result["name"], entity_type=result["entity_type"])
    if status == "out_of_scope":
        return OutOfScope(
            name=result["name"], entity_type=result["entity_type"], dcid=result.get("dcid")
        )
    raise ValueError(f"unrecognised resolution status {status!r}")


def _invalid_reason(stripped: str) -> str | None:
    """Return why `stripped` fails local validation, or `None` if it's fine to send."""
    if not stripped:
        return "empty after stripping whitespace"
    if len(stripped) > _MAX_NAME_CHARS:
        return f"longer than {_MAX_NAME_CHARS} characters after stripping whitespace"
    return None


async def resolve_entities(
    scope: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, ResolutionOutcome]]:
    """Resolve every entity name in `scope` into a graph outcome, keyed the way it arrived.

    Args:
        scope: role -> the names supplied under that role, exactly as `get_data`'s scope
            block carries them, e.g. `{"donor": ["France"], "recipient": ["Togo",
            "Sub-Saharan Africa"]}`.

    Returns:
        The same role -> names shape back, each name mapped to its `ResolutionOutcome`.
        Role travels as the return value's outer key rather than as a field on the outcome
        itself: `get_data`'s refusal contract requires naming the role a failed name was
        supplied under, and that's already this shape, so the caller needs no bookkeeping
        of its own to reassociate an outcome with its role. A name repeated under two roles
        is resolved once against the door (see below) and appears twice in the return
        value, once per role, both entries equal.

    Raises:
        UpstreamError: dfo-apis' graph door failed, timed out, or is unmounted. Left to
            propagate uncaught -- the caller already owns a refusal grammar for
            `UpstreamError` (`data_mcp.tool_errors.from_upstream_error`) and re-wrapping it
            here would just shadow that.
    """
    unique_stripped: dict[str, None] = {}
    for names in scope.values():
        for name in names:
            stripped = name.strip()
            if _invalid_reason(stripped) is None:
                unique_stripped.setdefault(stripped, None)

    ordered_names = list(unique_stripped)
    outcomes_by_stripped: dict[str, ResolutionOutcome] = {}
    if ordered_names:
        client = get_client()
        chunks = list(itertools.batched(ordered_names, _MAX_TEXTS_PER_REQUEST, strict=False))
        # Concurrent dispatch bounds the total wait by one client retry budget rather than
        # their sum: two 25-name chunks awaited one after another can each burn the full
        # budget, and their sum plus the mint that follows can outrun the harness's MCP
        # read window. `return_exceptions` keeps every chunk's task awaited to completion
        # here, so a failure in one chunk can't orphan a task or produce a "never
        # retrieved" warning while its siblings finish.
        responses = await asyncio.gather(
            *(
                client.request(
                    "POST", _RESOLVE_ROUTE, route=_RESOLVE_ROUTE, json={"texts": list(chunk)}
                )
                for chunk in chunks
            ),
            return_exceptions=True,
        )
        # Raise the first chunk's failure, in chunk order, so a caller sees exactly one
        # exception no matter how many chunks failed. Rebuilding the list here (rather
        # than checking and re-iterating `responses`) is also what narrows each entry
        # from `Response | BaseException` down to `Response` for the zip below.
        successful_responses: list[httpx2.Response] = []
        for response in responses:
            if isinstance(response, BaseException):
                raise response
            successful_responses.append(response)
        for chunk, response in zip(chunks, successful_responses, strict=True):
            results = response.json()["results"]
            # `strict=True` turns a length mismatch into an immediate `ValueError` rather
            # than a silently truncated or misaligned zip -- the one failure mode that
            # would attribute one name's result to another.
            for stripped, result in zip(chunk, results, strict=True):
                outcomes_by_stripped[stripped] = _outcome_from_result(result)

    resolved: dict[str, dict[str, ResolutionOutcome]] = {}
    for role, names in scope.items():
        per_role: dict[str, ResolutionOutcome] = {}
        for name in names:
            stripped = name.strip()
            reason = _invalid_reason(stripped)
            per_role[name] = (
                InvalidName(name=name, reason=reason)
                if reason is not None
                else outcomes_by_stripped[stripped]
            )
        resolved[role] = per_role
    return resolved
