"""Errors raised by the definition store."""

from __future__ import annotations


class DefinitionStoreUnavailable(Exception):  # noqa: N818 -- named for the refusal it maps to, not a generic "Error"
    """Raised when the definition store's backend cannot be reached.

    Distinct from `DefinitionStore.get` returning `None` for a genuine miss:
    this means the backend itself is unreachable, not that the requested id
    was never registered. The two map to different tool-facing refusals.
    """
