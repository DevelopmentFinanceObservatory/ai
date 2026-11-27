"""The places of the same kind a card's role can change to, read from Search.

A card offers a change of places for each role it binds
(`harness.card_unit.card_changes`), and the places a role can change to are
read only when asked: by `GET /api/studio/places`, when the page opens a
role's picker, and once more when a change of places is applied, to check
every place it names is listed and to take their labels. Nothing is read
when a card is drawn, and nothing is stored.

Search's options door lists the places of the type the role binds that hold
data in the years the card reads (`POST /api/search/options`,
`for: {role, scope: "peers"}`, read through
`common.live_apis.LiveApisClient.read_entity_options`, which asks for each
place's coverage when the identity's `requested_window` is a range; under
`all` or `latest` every listed place holds data in it). It reads no model and
answers from the posted identity: on a local Search over the dev catalogue a
first read of an identity took 0.25 to 1.3 s with coverage and 0.01 to 1.1 s
without, and a repeat, which Search answers from its cache, about 10 ms. `read_role_places` waits
at most `PLACE_OPTIONS_DEADLINE_S` and keeps at most `PLACE_OPTIONS_CAP`
places, sorted by label. `ApisPlaceOptions` keeps a role's places for
`PLACE_OPTIONS_TTL_S`, so a picker's read and the change applied from it
make one call.

Search lists the peers of a card's `observationAbout` places only when they
share one place type, a country or a continent, so a card offers a change of
that role only for such places (`peers_listable`). Search refuses any other
request it cannot answer for the posted identity with a 400 or a 422, which
`read_role_places` raises as `PlacesRefusedError`: asking again gets the same
answer.

Places are offered only where dfo-apis serves the door, learnt when it is
read, as the stated-scope reader learns its own door
(`planner.stated_scope.ApisStatedScopeReader`): a 404 or an answer outside
the contract makes `ApisPlaceOptions` read Search's capabilities document,
and a document listing no live `options` door stops the offers for
`PLACE_OPTIONS_TTL_S` (`ApisPlaceOptions.offering`), after which the next
read tries the door again. Nothing is read at startup, so an apis that
cannot be reached then costs nothing once it is back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from common.apis_auth import apis_auth_provider
from common.live_apis import RESPONSE_INVALID, LiveApisClient, LiveApisError
from common.place_type import PLACE_ROLE

logger = logging.getLogger(__name__)

# How long a places read waits for Search.
PLACE_OPTIONS_DEADLINE_S = 2.0
# The most places one role lists; a longer list is cut and marked `truncated`.
PLACE_OPTIONS_CAP = 300
# How long `ApisPlaceOptions` keeps a role's places.
PLACE_OPTIONS_TTL_S = 60.0
# The door Search's capabilities document lists when apis serves the options route.
OPTIONS_DOOR = "options"
# The status apis answers a route it does not serve with.
_NOT_SERVED_STATUS = 404
# The statuses Search refuses a request with that asking again cannot change:
# its deterministic input refusals, such as `unsupported_peers_anchor_type`.
_REFUSED_STATUSES = frozenset({400, 422})
# The place types whose peers Search lists for `observationAbout`: the places
# of that type under Earth.
PEER_PLACE_TYPES = frozenset({"Country", "Continent"})


class PlaceOptionsReader(Protocol):
    """Reads the places of the same kind as those role `role` of `identity`
    binds, each as `(ref, label)`, and whether Search capped the list."""

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]: ...


class PlacesUnavailableError(Exception):
    """Search's options door failed or missed `PLACE_OPTIONS_DEADLINE_S`."""


class PlacesNotServedError(PlacesUnavailableError):
    """apis answered the options route 404 or outside its contract and its
    capabilities document lists no live `options` door."""


class PlacesRefusedError(Exception):
    """Search's options door refused the identity and role it was asked for
    with a 400 or a 422, which it answers again for the same request."""


@dataclass(frozen=True)
class RolePlaces:
    """The places a role can change to: `(ref, label)` in label order, case
    and accents ignored, ties by ref, at most `PLACE_OPTIONS_CAP`, and
    whether more were listed (`truncated`)."""

    places: tuple[tuple[str, str], ...]
    truncated: bool


class ApisPlaceOptions:
    """The `PlaceOptionsReader` of dfo-apis at `base_url` (`DFO_APIS_URL`),
    with the Cloud Run ID token `common.apis_auth.apis_auth_provider` mints.
    Build it once for the process and close it at shutdown (`aclose`).

    Each answer is kept for `PLACE_OPTIONS_TTL_S` by the identity's canonical
    JSON and the role, read by `clock`. Reads of one key while its lookup runs
    wait on that lookup, so concurrent reads make one call; a lookup that
    fails is kept for no one and raises its failure in every read waiting on
    it. A door apis does not serve stops the offers for `PLACE_OPTIONS_TTL_S`
    (`offering`)."""

    def __init__(
        self,
        *,
        base_url: str,
        client: LiveApisClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or LiveApisClient(
            base_url=base_url,
            auth=apis_auth_provider(base_url=base_url),
            read_timeout_s=PLACE_OPTIONS_DEADLINE_S,
            max_attempts=1,
        )
        self._clock = clock
        # Until when the door counts as not served, on `clock`.
        self._unserved_until = float("-inf")
        self._kept: dict[
            tuple[str, str], tuple[float, tuple[tuple[tuple[str, str], ...], bool]]
        ] = {}
        # The lookup running for each key, which every read of the key awaits.
        self._running: dict[
            tuple[str, str], asyncio.Task[tuple[tuple[tuple[str, str], ...], bool]]
        ] = {}

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        key = (json.dumps(identity, sort_keys=True, default=str), role)
        kept = self._kept.get(key)
        if kept is not None and self._clock() < kept[0]:
            return kept[1]
        if not self.offering():
            raise PlacesNotServedError(role)
        running = self._running.get(key)
        if running is None:
            running = asyncio.ensure_future(self._lookup(key, identity=identity, role=role))
            self._running[key] = running
            running.add_done_callback(self._landed)
        # A read that gives up, at its deadline say, leaves the lookup to the
        # reads still waiting on it.
        return await asyncio.shield(running)

    @staticmethod
    def _landed(done: asyncio.Task[tuple[tuple[tuple[str, str], ...], bool]]) -> None:
        """Read the failure of `done`, a lookup that ended, so one no read
        waits on any longer is not reported as never retrieved."""
        if not done.cancelled():
            done.exception()

    async def _lookup(
        self, key: tuple[str, str], *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        """Read `role`'s places of `identity` from apis and keep them under
        `key`. The lookup leaves `_running` in the step that ends it, before
        any callback runs, so a read arriving after it ends, however it
        ended, starts a lookup of its own."""
        try:
            return await self._read(key, identity=identity, role=role)
        finally:
            if self._running.get(key) is asyncio.current_task():
                del self._running[key]

    async def _read(
        self, key: tuple[str, str], *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        now = self._clock()
        try:
            answer = await self._client.read_entity_options(identity=identity, role=role)
        except LiveApisError as exc:
            if (exc.status == _NOT_SERVED_STATUS or exc.code == RESPONSE_INVALID) and not (
                await self._door_listed()
            ):
                self._unserved_until = self._clock() + PLACE_OPTIONS_TTL_S
                logger.warning(
                    "studio place options not served: apis answered %s %s and lists no "
                    "live %s door; cards offer no places for %.0f s",
                    exc.status,
                    exc.code,
                    OPTIONS_DOOR,
                    PLACE_OPTIONS_TTL_S,
                )
                raise PlacesNotServedError(role) from exc
            raise
        self._kept = {k: v for k, v in self._kept.items() if now < v[0]}
        self._kept[key] = (now + PLACE_OPTIONS_TTL_S, answer)
        return answer

    def offering(self) -> bool:
        """Whether cards offer places: false for `PLACE_OPTIONS_TTL_S` after a
        read learnt apis serves no options door."""
        return self._clock() >= self._unserved_until

    async def _door_listed(self) -> bool:
        """Whether Search's capabilities document lists a live options door.
        A document that cannot be read raises a retryable failure, so the read
        ends on a retryable refusal and the offers stay. Any other failure
        reads as listed, so the read's own failure stands."""
        try:
            doors = await self._client.read_search_doors()
        except LiveApisError as exc:
            logger.error(
                "places door answered outside its contract and the search capabilities "
                "document could not be read: status=%s code=%s",
                exc.status,
                exc.code,
            )
            if exc.retryable:
                raise
            return True
        return OPTIONS_DOOR in doors

    async def aclose(self) -> None:
        await self._client.aclose()


async def read_role_places(
    reader: PlaceOptionsReader,
    *,
    identity: Mapping[str, Any],
    role: str,
    conversation_id: str = "",
    component_id: str = "",
) -> RolePlaces:
    """The places role `role` of `identity`, a card's Search block identity,
    can change to, read under `PLACE_OPTIONS_DEADLINE_S`. A failed read logs
    the conversation and the card it was read for.

    Raises:
        PlacesNotServedError: apis serves no options door (`ApisPlaceOptions`).
        PlacesRefusedError: the door refused the identity and role for good.
        PlacesUnavailableError: the door failed or missed the deadline.
    """
    started = time.monotonic()
    try:
        async with asyncio.timeout(PLACE_OPTIONS_DEADLINE_S):
            listed, truncated = await reader(identity=identity, role=role)
    except (LiveApisError, OSError, TimeoutError) as exc:
        if isinstance(exc, LiveApisError) and exc.status in _REFUSED_STATUSES:
            logger.warning(
                "places refused: conversation=%s component=%s role=%s status=%s code=%s",
                conversation_id,
                component_id,
                role,
                exc.status,
                exc.code,
            )
            raise PlacesRefusedError(role) from exc
        logger.warning(
            "places not read: conversation=%s component=%s role=%s: %r",
            conversation_id,
            component_id,
            role,
            exc,
        )
        raise PlacesUnavailableError(role) from exc
    logger.info(
        "places for role %s read in %.0f ms (%d listed)",
        role,
        (time.monotonic() - started) * 1000,
        len(listed),
    )
    ordered = sorted(listed, key=lambda place: (_label_key(place[1]), place[0]))
    return RolePlaces(
        places=tuple(ordered[:PLACE_OPTIONS_CAP]),
        truncated=truncated or len(listed) > PLACE_OPTIONS_CAP,
    )


def peers_listable(role: str, refs: Iterable[str], block: Mapping[str, Any]) -> bool:
    """Whether Search can list the places role `role` can change to from
    `refs`, the places the card reads for it, typed by the `entity_type` its
    Search block `block` states for each. Search lists the peers of
    `observationAbout` places only when they share one type in
    `PEER_PLACE_TYPES`. A place the block states no type for, one a scope or
    an earlier change named, counts as any type, so the offer stays and
    Search's refusal of a place it cannot list ends the change
    (`PlacesRefusedError`). Every other role is listed whatever its places."""
    if role != PLACE_ROLE:
        return True
    types = {
        str(entity["ref"]): str(entity["entity_type"])
        for entity in (block.get("resolved") or {}).get("entities") or ()
        if isinstance(entity, Mapping) and entity.get("ref") and entity.get("entity_type")
    }
    known = {types[ref] for ref in refs if ref in types}
    return len(known) <= 1 and known <= PEER_PLACE_TYPES


def _label_key(label: str) -> str:
    """`label` as a reader alphabetises it: case and accents ignored, so
    "Côte d'Ivoire" sorts among the Cs."""
    decomposed = unicodedata.normalize("NFKD", label)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def places_offered(reader: PlaceOptionsReader | None) -> bool:
    """Whether cards offer a change of places through `reader`: one is set,
    and, for `ApisPlaceOptions`, apis has not been learnt to serve no door."""
    if reader is None:
        return False
    return not isinstance(reader, ApisPlaceOptions) or reader.offering()
