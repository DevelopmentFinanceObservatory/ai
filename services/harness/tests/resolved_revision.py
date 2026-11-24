"""Resolved revisions for tests, as presentation-mcp carries them in a
`write_component` result's `_meta[RESOLVED_PAYLOAD_META_KEY]`.

`fixtures/resolved_revision/<member>.json` are dfo-spec-kit's catalogue
fixtures (`catalogue/fixtures/<member>/payload.json`), copied verbatim.
`resolved_revision` loads one and restamps the ids a test pins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FIXTURES = Path(__file__).parent / "fixtures" / "resolved_revision"


def resolved_revision(
    member: str,
    *,
    component_id: str | None = None,
    revision: int | None = None,
    frame_ref: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """The `member` fixture envelope, with `component_id`, `revision`, the
    one input frame and the title text replaced where given."""
    envelope = json.loads((_FIXTURES / f"{member}.json").read_text())
    if component_id is not None:
        envelope["component_id"] = component_id
    if revision is not None:
        envelope["revision"] = revision
    if frame_ref is not None:
        envelope["provenance"]["inputs"] = [frame_ref]
    if title is not None:
        envelope["component"]["title"]["text"] = title
    return envelope
