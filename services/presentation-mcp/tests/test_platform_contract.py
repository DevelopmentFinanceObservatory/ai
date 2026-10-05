"""Contract tests: the `Assembler` protocol and the types that cross it.

The echo fake exists to pin the contract itself: assigning it to an
`Assembler`-typed name makes conformance a lint failure, and calling every
method pins the signatures at runtime.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest
from presentation_mcp.platform_contract import (
    RESOLVED_PAYLOAD_META_KEY,
    Assembler,
    Caller,
    Refusal,
    RetryMode,
    WriteReply,
)

RELEASE: dict[str, Any] = {"schema_version": 1, "catalog_version": "2026.8.0"}


class _EchoAssembler:
    async def create_component(
        self,
        *,
        intent: Mapping[str, Any],
        caller: Caller,
    ) -> WriteReply | Refusal:
        return WriteReply(
            component_id="cmp_1",
            revision=1,
            fact_summary={"intent": dict(intent), "caller": caller},
        )

    async def revise_component(
        self,
        *,
        component_id: str,
        patch: Mapping[str, Any],
        caller: Caller,
    ) -> WriteReply | Refusal:
        return Refusal(
            error="unknown_component",
            retry=RetryMode.DO_NOT_RETRY,
            detail=f"no component {component_id!r}",
        )

    async def get_frame_view(
        self,
        *,
        frame_ref: Mapping[str, Any],
        dimension: str | None = None,
        query: str | None = None,
    ) -> Mapping[str, Any] | Refusal:
        return {
            "frame_ref": dict(frame_ref),
            "dimension": dimension,
            "query": query,
        }

    async def get_component(
        self,
        *,
        component_id: str,
        revision: int | None = None,
    ) -> Mapping[str, Any] | Refusal:
        return {"component_id": component_id, "revision": revision}

    async def release(self) -> Mapping[str, Any]:
        return RELEASE


# The point of the fake: this assignment fails `just lint` if the fake and
# the protocol ever drift apart.
ASSEMBLER: Assembler = _EchoAssembler()


async def test_create_component_signature() -> None:
    reply = await ASSEMBLER.create_component(
        intent={"type": "line_chart"}, caller=Caller.FIRST_PARTY
    )

    assert isinstance(reply, WriteReply)
    assert reply.revision == 1


async def test_revise_component_signature() -> None:
    refusal = await ASSEMBLER.revise_component(
        component_id="cmp_missing", patch={"title": "New"}, caller=Caller.EXTERNAL
    )

    assert isinstance(refusal, Refusal)
    assert refusal.retry is RetryMode.DO_NOT_RETRY


async def test_get_frame_view_signature() -> None:
    view = await ASSEMBLER.get_frame_view(
        frame_ref={"spec_id": "spec_1", "content_version": "2026-01"},
        dimension="recipient",
    )

    assert not isinstance(view, Refusal)
    assert view["frame_ref"] == {"spec_id": "spec_1", "content_version": "2026-01"}
    assert view["dimension"] == "recipient"
    assert view["query"] is None


async def test_get_component_signature() -> None:
    component = await ASSEMBLER.get_component(component_id="cmp_1")

    assert not isinstance(component, Refusal)
    assert component["revision"] is None


async def test_release_signature() -> None:
    assert await ASSEMBLER.release() == RELEASE


def test_caller_values_match_the_wire() -> None:
    assert [caller.value for caller in Caller] == ["external", "first_party"]


def test_retry_mode_values_match_the_wire() -> None:
    assert [mode.value for mode in RetryMode] == [
        "correct_and_retry",
        "refresh_then_retry",
        "retry_after_delay",
        "do_not_retry",
    ]


def test_write_reply_changed_defaults_to_empty() -> None:
    reply = WriteReply(component_id="cmp_1", revision=1, fact_summary={"unit": "usd"})

    assert reply.changed == ()


def test_write_reply_resolved_payload_defaults_to_none() -> None:
    reply = WriteReply(component_id="cmp_1", revision=1, fact_summary={})

    assert reply.resolved_payload is None


def test_resolved_payload_meta_key_matches_the_wire() -> None:
    # The harness reads the envelope from `_meta` by this exact key.
    assert RESOLVED_PAYLOAD_META_KEY == "org.one.dfo/resolved_payload"


def test_refusal_fix_and_pointer_default_to_none() -> None:
    refusal = Refusal(
        error="binding_invalid",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail="recipient 'country/COD' is not in the frame",
    )

    assert refusal.fix is None
    assert refusal.pointer is None


def test_contract_types_are_frozen_and_keyword_only() -> None:
    for cls in (WriteReply, Refusal):
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, cls.__name__
        assert params.kw_only, cls.__name__


def test_replies_reject_mutation() -> None:
    reply = WriteReply(component_id="cmp_1", revision=1, fact_summary={})

    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.revision = 2  # type: ignore[invalid-assignment]
