"""The platform contract: what this service needs from the assembler.

Interface and boundary types only, no implementation. Intents, patches, and
the describe replies are opaque mappings, forwarded without being read, with
two exceptions. `get_frame_view` reads `frame_ref`'s `spec_id` and
`content_version` to build the platform request, and `server.py` lifts
`resolved_payload` out of a `get_component` reply into the tool result's
`_meta`. Refusals are return values, not exceptions; exceptions mean
malfunction and are `server.py`'s concern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

# The tool result `_meta` key under which write_component and
# describe_component carry the platform's resolved revision to clients. MCP
# hosts keep `_meta` out of the model's context.
RESOLVED_PAYLOAD_META_KEY = "org.one.dfo/resolved_payload"

# The tool result `_meta` key under which an app client's write_component and
# describe_component result carries the values its view binds. The renderer
# page merges them into the view's state (`theme.renderer_html`).
VIEW_STATE_META_KEY = "org.one.dfo/view_state"

# The key under which a describe_frame view or a create's reply names the
# version its `spec_` frame_ref asked for and the newest version read in its
# place, when the platform no longer holds the one asked for and the call set
# `rebuild_lost_frames`.
FRAME_NOTE_KEY = "frame_note"


class Caller(StrEnum):
    """Who is calling, determined by transport, never by a client claim."""

    EXTERNAL = "external"
    FIRST_PARTY = "first_party"


class RetryMode(StrEnum):
    """How the calling model should react to a `Refusal`."""

    CORRECT_AND_RETRY = "correct_and_retry"
    REFRESH_THEN_RETRY = "refresh_then_retry"
    RETRY_AFTER_DELAY = "retry_after_delay"
    DO_NOT_RETRY = "do_not_retry"


@dataclass(frozen=True, kw_only=True)
class Refusal:
    """A structured refusal, built by the assembler.

    `error` stays an open string: the assembler owns the code vocabulary,
    and an enum here would turn every new refusal into a contract change.
    """

    error: str
    retry: RetryMode
    detail: str
    # How a corrected call can succeed, e.g. the legal values for a refused
    # binding. Absent when there is nothing to correct.
    fix: str | Mapping[str, Any] | None = None
    # A JSON Pointer naming the offending field. For a binding refusal this
    # points into the platform's internal normalized shape, not the intent
    # as submitted; it is passed through as received either way.
    pointer: str | None = None
    # Seconds to wait before repeating the call unchanged, on a
    # `retry_after_delay` refusal that names a wait.
    retry_after_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class NormalFormConflict:
    """A publish apis answered `recipe_requires_matching_normal_form_hash`: it
    holds the pair under a recipe with another `normal_form_hash` and kept that
    recipe. The frame is published, and apis draws a card of the sent recipe on
    it under no render. `detail` is apis' own."""

    detail: str


@dataclass(frozen=True, kw_only=True)
class WriteReply:
    """The reply to a successful create or revise.

    Every field but `resolved_payload` is model vocabulary.
    """

    component_id: str
    revision: int
    # What the component now shows. The shape is per component type.
    fact_summary: Mapping[str, Any]
    # Edits only: each edited field, as {"field": ..., "prior": ..., "new":
    # ...} (or just {"field": ...} when the value isn't disclosed). Empty on
    # a create.
    changed: tuple[Mapping[str, Any], ...] = ()
    # The platform's resolved revision, the stamped envelope `{schema_version,
    # catalog_version, component_id, revision, provenance, component}`,
    # verbatim. Not model vocabulary. `server.py` moves it into the tool
    # result's `_meta`. `None` when the platform's reply carried none.
    resolved_payload: Mapping[str, Any] | None = None
    # Creates only: the component's write token (`cmw_...`), which the
    # platform returns once and requires on every revise. Model vocabulary on
    # a create result alone. `server.py` keeps it out of `_meta` and out of
    # the Prefab shell. `None` on a revise.
    write_token: str | None = None
    # Creates only: the sentence naming the version the intent's `spec_`
    # frame_ref asked for and the newest version drawn in its place, when the
    # platform no longer held the one asked for. Model vocabulary, under
    # `FRAME_NOTE_KEY`. `None` otherwise.
    frame_note: str | None = None


class Assembler(Protocol):
    """The five calls this service makes on the platform."""

    async def create_component(
        self,
        *,
        intent: Mapping[str, Any],
        caller: Caller,
        rebuild_lost_frames: bool = False,
    ) -> WriteReply | Refusal:
        """Resolve a complete intent against its frame and mint revision 1.

        With `rebuild_lost_frames`, a `spec_` frame_ref the platform no longer
        holds is drawn at its newest version, and the reply names both
        versions under `FRAME_NOTE_KEY`.
        """
        ...

    async def revise_component(
        self,
        *,
        component_id: str,
        patch: Mapping[str, Any],
        write_token: str,
        caller: Caller,
    ) -> WriteReply | Refusal:
        """Merge a patch into the stored intent, re-resolve, mint the next revision.

        `write_token` is the token the component's create returned. The
        platform refuses a missing or wrong one with `write_refused`.
        """
        ...

    async def get_frame_view(
        self,
        *,
        frame_ref: Mapping[str, Any],
        dimension: str | None = None,
        query: str | None = None,
        rebuild_lost_frames: bool = False,
    ) -> Mapping[str, Any] | Refusal:
        """What the frame holds: dimensions, values, coverage.

        Reads `frame_ref["spec_id"]` and `frame_ref["content_version"]` to
        address the platform request. A `spec_` id with no content_version
        reads its newest version, which the view names. With
        `rebuild_lost_frames`, so does a `spec_` pair the platform no longer
        holds, and the view names both versions under `FRAME_NOTE_KEY`.
        `dimension` narrows the view to one dimension's values. `query`
        filters those values.
        """
        ...

    async def get_component(
        self,
        *,
        component_id: str,
        revision: int | None = None,
    ) -> Mapping[str, Any] | Refusal:
        """A stored intent and its facts, current or at an older revision.

        The platform's body also carries that revision's `resolved_payload`
        envelope, which `server.py` moves into the tool result's `_meta`.
        """
        ...

    async def release(self) -> Mapping[str, Any]:
        """The catalog release the assembler is running.

        Its `schema_version` is checked against this deployment's admission
        bound at startup, and its `intent_schemas_tolerant` is what the tool
        schemas are assembled from. Not model-facing: raises on failure
        rather than returning a `Refusal`, the one call on this Protocol
        that does.
        """
        ...
