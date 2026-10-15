"""apis' canonical frames for the harness tests' fake mints.

data-mcp registers a default mint's frame from the version's canonical frame,
which the snapshot route lists, and only when that frame mints the version's
`content_version`. A fake mint here serves one frame file per version. The
version is named by the token of the frame data-mcp parses out of that file
(`version_of`), and `serve_canonical_frames` answers the snapshot lookup and
the canonical fetch with that frame in the canonical layout.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from common.materialisation import FrameRef, frame_to_parquet_bytes
from data_mcp import frame_materialisation
from dfo_frame_grammar import frame_sha256, mint_content_version

_LINK_PREFIX = "canonical:"

# Each version `version_of` named, with its canonical frame's bytes. A version
# names one frame, so the map is shared across tests.
_CANONICAL: dict[str, bytes] = {}


def version_of(frame_file: bytes) -> str:
    """The `content_version` of the version whose frame file is `frame_file`: the
    token of the frame data-mcp parses out of it."""
    frame = frame_materialisation._frame_from_parquet_bytes(frame_file)
    content_version = mint_content_version(frame_sha256=frame_sha256(frame))
    _CANONICAL[content_version] = frame_to_parquet_bytes(frame)
    return content_version


def serve_canonical_frames(
    monkeypatch: pytest.MonkeyPatch, fetch: Callable[..., Awaitable[bytes | None]]
) -> list[str]:
    """Install `fetch` as data-mcp's frame-file fetch, and answer the snapshot lookup
    and fetch of the canonical frame of every version `version_of` named. Returns the
    versions whose canonical frame is fetched, in order."""
    fetched: list[str] = []

    async def locate(ref: FrameRef, **_: Any) -> tuple[str, str]:
        data = _CANONICAL[ref.content_version]
        return f"sha256:{hashlib.sha256(data).hexdigest()}", _LINK_PREFIX + ref.content_version

    async def fetch_any(*, link: str | None, **kwargs: Any) -> bytes | None:
        if link is None:
            return await fetch(**kwargs)
        content_version = link.removeprefix(_LINK_PREFIX)
        fetched.append(content_version)
        return _CANONICAL[content_version]

    monkeypatch.setattr(frame_materialisation, "_locate_canonical_frame", locate)
    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fetch_any)
    return fetched
