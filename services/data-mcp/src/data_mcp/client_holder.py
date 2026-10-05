"""The process-wide upstream client shared by every data-mcp tool.

Built lazily on first use: the deployed topology mounts data-mcp's FastMCP
app in-process and never calls `data_mcp.server.main()`, so nothing would
otherwise wire a client. `set_client()` (tests, and `main()` for a
standalone data-mcp run) pre-empts the lazy default. `aclose_client()` is
called from `server.py`'s lifespan on shutdown.
"""

from __future__ import annotations

from data_mcp.client import UpstreamClient
from data_mcp.settings import Settings


class _ClientHolder:
    """Module-level holder for the upstream client tools call."""

    def __init__(self) -> None:
        self._client: UpstreamClient | None = None

    def set(self, client: UpstreamClient | None) -> None:
        self._client = client

    def get(self) -> UpstreamClient:
        if self._client is None:
            self._client = UpstreamClient(settings=Settings())
        return self._client

    async def aclose(self) -> None:
        # Checked directly rather than routed through `get()`, so shutdown
        # never lazily constructs a client that no call ever used. Cleared
        # after closing, so a later `get()` (a second server lifespan)
        # rebuilds instead of handing out the closed client.
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_holder = _ClientHolder()


def get_client() -> UpstreamClient:
    """Return the process-wide upstream client, building it lazily if needed."""
    return _holder.get()


def set_client(client: UpstreamClient | None) -> None:
    """Set (or clear) the upstream client tools call, pre-empting the lazy default."""
    _holder.set(client)


async def aclose_client() -> None:
    """Close the upstream client, if one was ever set.

    Called from `server.py`'s lifespan on shutdown.
    """
    await _holder.aclose()
