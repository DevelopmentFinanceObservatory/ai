"""The places of the same kind a card's role can change to, read from Search.

A card offers a change of places for each role it binds
(`harness.card_unit.card_changes`), and the places a role can change to are
read only when asked: by `GET /api/studio/places`, when the page opens a
role's picker, and once more when a change of places is applied, to check
every place it names is listed and to take their labels. Nothing is read
when a card is drawn, and nothing is stored.

Search's options door lists the places of the type the role binds
(`POST /api/search/options`, `for: {role, scope: "peers"}`, read through
`common.live_apis.LiveApisClient.read_entity_options`). It reads no model and
answers from the posted identity, in about 0.3 to 0.6 s on the dev
deployment. `read_role_places` waits at most `PLACE_OPTIONS_DEADLINE_S` and
keeps at most `PLACE_OPTIONS_CAP` places.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from common.apis_auth import apis_auth_provider
from common.live_apis import LiveApisClient, LiveApisError

logger = logging.getLogger(__name__)

# How long a places read waits for Search.
PLACE_OPTIONS_DEADLINE_S = 2.0
# The most places one role lists; a longer list is cut and marked `truncated`.
PLACE_OPTIONS_CAP = 300


class PlaceOptionsReader(Protocol):
    """Reads the places of the same kind as those role `role` of `identity`
    binds, each as `(ref, label)`, and whether Search capped the list."""

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]: ...


class PlacesUnavailableError(Exception):
    """Search's options door failed or missed `PLACE_OPTIONS_DEADLINE_S`."""


@dataclass(frozen=True)
class RolePlaces:
    """The places a role can change to: `(ref, label)` in Search's order, at
    most `PLACE_OPTIONS_CAP`, and whether more were listed (`truncated`)."""

    places: tuple[tuple[str, str], ...]
    truncated: bool


class ApisPlaceOptions:
    """The `PlaceOptionsReader` of dfo-apis at `base_url` (`DFO_APIS_URL`),
    with the Cloud Run ID token `common.apis_auth.apis_auth_provider` mints.
    Build it once for the process and close it at shutdown (`aclose`)."""

    def __init__(self, *, base_url: str) -> None:
        self._client = LiveApisClient(
            base_url=base_url,
            auth=apis_auth_provider(base_url=base_url),
            read_timeout_s=PLACE_OPTIONS_DEADLINE_S,
            max_attempts=1,
        )

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        return await self._client.read_entity_options(identity=identity, role=role)

    async def aclose(self) -> None:
        await self._client.aclose()


async def read_role_places(
    reader: PlaceOptionsReader, *, identity: Mapping[str, Any], role: str
) -> RolePlaces:
    """The places role `role` of `identity`, a card's Search block identity,
    can change to, read under `PLACE_OPTIONS_DEADLINE_S`.

    Raises:
        PlacesUnavailableError: the door refused, failed or missed the deadline.
    """
    started = time.monotonic()
    try:
        async with asyncio.timeout(PLACE_OPTIONS_DEADLINE_S):
            listed, truncated = await reader(identity=identity, role=role)
    except (LiveApisError, OSError, TimeoutError) as exc:
        logger.info("places for role %s not read: %r", role, exc)
        raise PlacesUnavailableError(role) from exc
    logger.info(
        "places for role %s read in %.0f ms (%d listed)",
        role,
        (time.monotonic() - started) * 1000,
        len(listed),
    )
    return RolePlaces(
        places=tuple(listed[:PLACE_OPTIONS_CAP]),
        truncated=truncated or len(listed) > PLACE_OPTIONS_CAP,
    )
