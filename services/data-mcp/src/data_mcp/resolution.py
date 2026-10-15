"""Entity resolution seam: human-typed names to graph dcids, via apis' Graph door.

`get_data`'s scope block carries entities as spoken text ("France", "Sub-Saharan Africa")
keyed by role. `resolve_entities` posts them to `POST /api/graph/resolve` with the group and
region members inline (`expand: ["members"]`), validates the answer through the generated
`data_mcp.graph_wire` models, and turns each of the four wire statuses (`resolved`,
`ambiguous`, `not_found`, `not_held`) or a local validation failure into a structured
outcome. Organisations, aggregates, places below a country and membership groups resolve
alongside countries, continents and subregions.

The calling tool owns the translation into refusals (`data_mcp.tool_errors`), which keeps
that module the sole owner of the refusal grammar; this module returns plain data, never
imports it, and never catches `UpstreamError`.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import httpx2

from data_mcp.client import GRAPH_PREFIX
from data_mcp.client_holder import get_client
from data_mcp.graph_wire import EntityRef, ResolveResponse, ResolveResult

_RESOLVE_ROUTE = f"{GRAPH_PREFIX}/resolve"

# The door's own published limits on `POST /api/graph/resolve`'s `texts` list.
_MAX_TEXTS_PER_REQUEST = 25
_MAX_NAME_CHARS = 200


@dataclass(frozen=True, kw_only=True)
class ResolvedEntity:
    """Resolution status `resolved`: the name maps unambiguously to one graph entity."""

    dcid: str
    name: str
    entity_type: str


@dataclass(frozen=True, kw_only=True)
class RegionMember:
    """One member of a resolved group or region, under the default `countries` projection."""

    dcid: str
    name: str


@dataclass(frozen=True, kw_only=True)
class ResolvedRegion:
    """Resolution status `resolved` for a group or a region the membership policy holds.

    `members` is what a caller can use as scope entities: the policy's countries that belong
    to the United Nations today.
    """

    name: str
    entity_type: str
    members: list[RegionMember]


@dataclass(frozen=True, kw_only=True)
class AmbiguousCandidate:
    """One candidate graph entity offered for a name that matched more than one."""

    dcid: str
    name: str
    entity_type: str


@dataclass(frozen=True, kw_only=True)
class Ambiguous:
    """Resolution status `ambiguous`: more than one graph entity matches the name."""

    candidates: list[AmbiguousCandidate]


@dataclass(frozen=True, kw_only=True)
class NearMatch:
    """One near-miss the door offers alongside a `not_found` name.

    Carries only `name`: the refusal grammar surfaces it alone as a "did you mean" hint, and
    the door marks near matches as hints for a person that a program should pick none of.
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class NotFound:
    """Resolution status `not_found`: no candidate of the requested kinds matches the name.

    `near_matches` carries a distinction the wire status alone collapses: non-empty means
    the name is probably a misspelling of one of these; empty means the door found nothing
    close at all.
    """

    near_matches: list[NearMatch]


@dataclass(frozen=True, kw_only=True)
class NotHeld:
    """Resolution status `not_held`: the resolver named a real entity, but this deployment's
    graph does not hold it -- a fact distinct from "matched nothing". The door names no kind
    for it.
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class InvalidName:
    """A name rejected by local validation before any request reached the door.

    Never sent upstream: sending it would burn a request only to eat a 400 that fails the
    whole batch (`texts` entries must be 1..200 characters after whitespace stripping).
    """

    name: str
    reason: str


ResolutionOutcome = ResolvedEntity | ResolvedRegion | Ambiguous | NotFound | NotHeld | InvalidName


def _outcome_from_result(result: ResolveResult) -> ResolutionOutcome:
    """Translate one wire result into a `ResolutionOutcome`.

    A group or a region resolves with its members inline, since the request expands them,
    and maps to `ResolvedRegion`. A region the membership policy does not hold carries no
    members and maps to `ResolvedEntity`, like any other entity.

    Raises:
        ValueError: the result lacks a field its status sets, or its inline members do not
            add up to its `member_count`.
    """
    match result.status:
        case "resolved":
            entity = _entity(result)
            if entity.kind is None:
                raise ValueError(f"resolved entity {entity.ref!r} carries no kind")
            if result.members is None:
                return ResolvedEntity(dcid=entity.ref, name=entity.label, entity_type=entity.kind)
            if isinstance(result.members, str) or len(result.members) != result.member_count:
                raise ValueError(f"{entity.ref!r} did not carry all its members inline")
            return ResolvedRegion(
                name=entity.label,
                entity_type=entity.kind,
                members=[RegionMember(dcid=m.ref, name=m.label or m.ref) for m in result.members],
            )
        case "ambiguous":
            return Ambiguous(
                candidates=[
                    AmbiguousCandidate(dcid=c.ref, name=c.label, entity_type=c.kind or "entity")
                    for c in _present(result.candidates, "candidates")
                ]
            )
        case "not_found":
            near_matches = _present(result.near_matches, "near_matches")
            return NotFound(near_matches=[NearMatch(name=m.label) for m in near_matches])
        case "not_held":
            return NotHeld(name=_entity(result).label)


def _entity(result: ResolveResult) -> EntityRef:
    return _present(result.entity, "entity")


def _present[T](value: T | None, field: str) -> T:
    if value is None:
        raise ValueError(f"a {field!r} result carries no {field}")
    return value


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
        UpstreamError: apis' graph door failed, timed out, or is unmounted. Left to
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
                    "POST",
                    _RESOLVE_ROUTE,
                    route=_RESOLVE_ROUTE,
                    json={"texts": list(chunk), "expand": ["members"]},
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
            results = ResolveResponse.model_validate(response.json()).results
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
