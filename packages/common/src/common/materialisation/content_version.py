"""Registry for the frame-grammar `content_version` deriver a durable store's
`verify_content_version_on_read` check calls, following the same
registration precedent as `trace_codec`: `analysis_mcp` owns frame-grammar's
content_version derivation and registers its own function here, at its own
import time, passing this module the function object -- `common` and
`data_mcp` never import `analysis_mcp`.

There is exactly one deriver, not a `trace_codec`-style registry keyed by
several kinds: only one process component (`analysis_mcp`) ever has
frame-grammar available to derive from, so the seam only needs to answer
"has it registered one yet," not "which one."
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfo_frame_grammar import Frame

ContentVersionDeriver = Callable[["Frame"], str]

_deriver: ContentVersionDeriver | None = None


class ContentVersionDeriverUnregistered(Exception):  # noqa: N818 -- named for the situation it names, not a generic "Error"
    """`verify_content_version_on_read` was requested but nothing in this process has
    registered a content_version deriver. Only `analysis_mcp.frame_ref_resolution`
    registers one, at its own import time -- a process that never imports
    `analysis_mcp` (a standalone data-mcp deployment, for instance) has nothing to
    check against, so the check is refused rather than silently skipped."""

    def __init__(self) -> None:
        super().__init__(
            "frame_store_verify_content_version is enabled but no content_version "
            "deriver is registered in this process -- only analysis_mcp registers one, "
            "at its own import time, so this setting has no effect unless analysis_mcp "
            "is part of this deployment"
        )


def register_content_version_deriver(deriver: ContentVersionDeriver) -> None:
    """Register the frame-grammar content_version deriver a verify check calls.
    Re-registering (a module re-import) replaces the prior registration."""
    global _deriver  # noqa: PLW0603 -- deliberate idempotent-registration flag
    _deriver = deriver


def get_content_version_deriver() -> ContentVersionDeriver | None:
    """The registered deriver, or `None` if nothing has registered one yet."""
    return _deriver
