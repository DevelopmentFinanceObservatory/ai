"""One explicit Gemini cache per static prompt prefix, shared by every session.

An agent's system instruction, tool declarations and tool config are the
same on every call it makes, across turns and conversations, and they are
most of its prompt. `PrefixCache` keeps one Gemini `CachedContent` per
distinct prefix, keyed by `prefix_key`. A model's stream looks the request's
prefix up (`lookup`) and sends the request against the cache it names
(`against_cache`): the request carries `cached_content` and none of the
three fields, which the API refuses alongside a cache (400
INVALID_ARGUMENT). The contents, thought signatures included, travel on the
request as before. `harness.gemini_wire.GeminiStream` is that stream.

`harness.server.create_app` builds one `PrefixCache` per process and hands it
to the model constructors when `STUDIO_PREFIX_CACHE` is on and the model's
surface allows an explicit cache (`harness.surface.surface_prefix_cache`);
`gemini-4-hydrogen` takes none. Each Cloud Run instance creates its own cache per
prefix, once per `CACHE_TTL_S`. A changed prompt or model is a new key, and
the old cache expires on its TTL.

The cache is an optimisation, so any failure to find or create one sends the
request uncached, and only cancellation propagates. A create the API refuses
with a 400 (a prefix under the model's minimum token count, measured at 1,024
for gemini-3.8-flash) is remembered for that key, since the same prefix is
refused every time. Any other create failure, whichever transport raised it,
or a create that takes longer than `CREATE_TIMEOUT_S`, sends that key's calls
uncached without a new create for `CREATE_COOLDOWN_S`, then is retried.
Concurrent calls on one key share a single create. A stream whose cache has
gone (Gemini answers 403 "CachedContent not found", `CACHE_GONE_CODES`) drops
the entry (`forget`), looks it up again and resends once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.genai import Client, errors, types

logger = logging.getLogger(__name__)

# How long a created cache lives. Storage is billed per hour, and one hour
# holds the planner's prefix for well under the price of one uncached call.
CACHE_TTL_S = 3600

# A cache with less than this left is replaced rather than handed out, so no
# call outlives it.
REFRESH_MARGIN_S = 300

# How long a create may run before the calls waiting on it go uncached. A
# create takes about 1.4 s, and every call on the prefix waits on it.
CREATE_TIMEOUT_S = 10.0

# How long a prefix's calls go uncached after a create fails or times out, so
# a struggling cache API costs one call the create bound, not every call.
CREATE_COOLDOWN_S = 60.0

# Gemini answers a request whose `cached_content` has expired or been deleted
# with 403 PERMISSION_DENIED, "CachedContent not found (or permission
# denied)". 404 covers the same on the Vertex backend.
CACHE_GONE_CODES = frozenset({403, 404})


def prefix_key(model: str, config: types.GenerateContentConfig) -> str | None:
    """The sha256 of `model` and `config`'s system instruction, tools and tool
    config, or None when the request carries neither a system instruction
    nor tools."""
    if config.system_instruction is None and not config.tools:
        return None
    payload = {
        "model": model,
        "system_instruction": config.system_instruction,
        "tools": config.tools,
        "tool_config": config.tool_config,
    }
    canonical = json.dumps(payload, sort_keys=True, default=_jsonable)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True)


@dataclass
class _Entry:
    name: str
    expires_at: float


class PrefixCache:
    """The process's cache names, one per prefix key.

    Concurrent calls on one prefix wait on one shared create, and calls on
    different prefixes never wait on each other.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        create_timeout_s: float = CREATE_TIMEOUT_S,
    ) -> None:
        self._clock = clock
        self._create_timeout_s = create_timeout_s
        self._entries: dict[str, _Entry] = {}
        self._creating: dict[str, asyncio.Task[_Entry | None]] = {}
        self._cooling_until: dict[str, float] = {}
        self._refused: set[str] = set()

    async def cache_name(
        self, key: str, *, client: Client, model: str, config: types.GenerateContentConfig
    ) -> str | None:
        """The live cache for `key`, created from `config`'s prefix when there
        is none or the current one has under `REFRESH_MARGIN_S` left. None
        when the cache cannot be had, and the request goes uncached."""
        if key in self._refused:
            return None
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at - self._clock() > REFRESH_MARGIN_S:
            return entry.name
        if self._clock() < self._cooling_until.get(key, float("-inf")):
            return None
        creating = self._creating.get(key)
        if creating is None:
            creating = asyncio.create_task(
                self._create(key, client=client, model=model, config=config)
            )
            self._creating[key] = creating
            creating.add_done_callback(lambda _: self._creating.pop(key, None))
        # A cancelled waiter leaves the create running for the others.
        entry = await asyncio.shield(creating)
        return entry.name if entry is not None else None

    async def _create(
        self, key: str, *, client: Client, model: str, config: types.GenerateContentConfig
    ) -> _Entry | None:
        started = self._clock()
        try:
            async with asyncio.timeout(self._create_timeout_s):
                created = await client.aio.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        system_instruction=config.system_instruction,
                        tools=config.tools,
                        tool_config=config.tool_config,
                        ttl=f"{CACHE_TTL_S}s",
                        display_name=f"prefix-{key[:16]}",
                    ),
                )
        except Exception as exc:
            # Either transport the SDK runs on (httpx or aiohttp) raises its
            # own errors past `errors.APIError`; any of them leaves the call
            # uncached. Cancellation is not an `Exception` and propagates.
            if isinstance(exc, errors.ClientError) and exc.code == 400:
                self._refused.add(key)
                logger.warning("prefix cache %s: create refused (400): %s", key[:16], exc)
            else:
                self._cooling_until[key] = self._clock() + CREATE_COOLDOWN_S
                logger.warning(
                    "prefix cache %s: create failed (%s): %s", key[:16], type(exc).__name__, exc
                )
            return None
        tokens = created.usage_metadata.total_token_count if created.usage_metadata else None
        logger.info("prefix cache %s: created %s (%s tokens)", key[:16], created.name, tokens)
        if created.name is None:
            self._cooling_until[key] = self._clock() + CREATE_COOLDOWN_S
            return None
        self._entries[key] = _Entry(name=created.name, expires_at=started + CACHE_TTL_S)
        return self._entries[key]

    async def lookup(
        self, *, client: Client, model: str, config: types.GenerateContentConfig
    ) -> tuple[str, str] | None:
        """The prefix key and cache name a request for `model` built as `config`
        is sent against, or None when it has no prefix, or finding or creating
        its cache fails in any way short of cancellation."""
        try:
            key = prefix_key(model, config)
            if key is None:
                return None
            name = await self.cache_name(key, client=client, model=model, config=config)
        except Exception as exc:
            logger.warning("prefix cache: lookup failed (%s): %s", type(exc).__name__, exc)
            return None
        return None if name is None else (key, name)

    def forget(self, key: str, name: str) -> None:
        """Drop `key`'s entry if it still names `name`, the cache a request
        found gone; a concurrent call may already have replaced it."""
        entry = self._entries.get(key)
        if entry is not None and entry.name == name:
            del self._entries[key]


def against_cache(config: types.GenerateContentConfig, name: str) -> types.GenerateContentConfig:
    """A copy of `config` sent against cache `name`: it names the cache and
    leaves out the system instruction, tools and tool config the cache holds."""
    return config.model_copy(
        update={
            "cached_content": name,
            "system_instruction": None,
            "tools": None,
            "tool_config": None,
        }
    )


__all__ = [
    "CACHE_GONE_CODES",
    "CACHE_TTL_S",
    "CREATE_COOLDOWN_S",
    "CREATE_TIMEOUT_S",
    "REFRESH_MARGIN_S",
    "PrefixCache",
    "against_cache",
    "prefix_key",
]
