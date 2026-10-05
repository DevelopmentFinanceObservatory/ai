"""Tests for `ApprovalRecord`: the typed record of one answer to one plan
revision. These tests cover the model's own contract -- construction,
validation, serialisation -- in isolation. `/api/studio/interrupt` writes an
`ApprovalRecord` at the HTTP boundary; that route is tested in
`tests/test_studio.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from harness.approval import ApprovalDecision, ApprovalRecord
from pydantic import ValidationError

REVISION = "sha256:" + "a" * 64
APPROVER = "u_" + "b" * 16
CONVERSATION = "conv_01test0000"
INTERRUPT_ID = "int_01test0000"


def _record(**overrides: object) -> ApprovalRecord:
    fields: dict[str, object] = {
        "revision": REVISION,
        "approver": APPROVER,
        "decision": ApprovalDecision.APPROVED,
        "at": datetime(2026, 9, 16, tzinfo=UTC),
        "conversation_id": CONVERSATION,
        "interrupt_id": INTERRUPT_ID,
        "auth_enforced": True,
    }
    fields.update(overrides)
    return ApprovalRecord.model_validate(fields)


def test_a_record_round_trips_through_json() -> None:
    record = _record()
    restored = ApprovalRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_auth_enforced_false_is_a_distinguishable_value_not_a_default() -> None:
    # `auth_enforced` has no default: a record made under an open token gate
    # must say so explicitly, never by omission.
    open_gate = _record(auth_enforced=False)
    closed_gate = _record(auth_enforced=True)
    assert open_gate.auth_enforced is False
    assert closed_gate.auth_enforced is True
    with pytest.raises(ValidationError):
        ApprovalRecord.model_validate(
            {
                "revision": REVISION,
                "approver": APPROVER,
                "decision": ApprovalDecision.APPROVED,
                "at": datetime(2026, 9, 16, tzinfo=UTC),
                "conversation_id": CONVERSATION,
                "interrupt_id": INTERRUPT_ID,
                # auth_enforced omitted entirely
            }
        )


def test_a_record_refuses_an_extra_field() -> None:
    with pytest.raises(ValidationError):
        _record(plan_content={"steps": []})


def test_every_decision_value_is_accepted() -> None:
    for decision in ApprovalDecision:
        assert _record(decision=decision).decision == decision


def test_the_field_set_carries_no_plan_content_beyond_revision_and_version() -> None:
    # Guard, not behaviour: pins the exact field set, so adding plan state
    # (steps, decisions, requirements) to `ApprovalRecord` fails this
    # assertion. The record identifies which revision was answered; it does
    # not carry what that revision contains.
    expected = {
        "revision",
        "approver",
        "decision",
        "at",
        "conversation_id",
        "interrupt_id",
        "plan_version",
        "auth_enforced",
    }
    assert set(ApprovalRecord.model_fields) == expected


def test_a_record_serialises_decision_as_its_string_value() -> None:
    record = _record(decision=ApprovalDecision.REJECTED)
    dumped = json.loads(record.model_dump_json())
    assert dumped["decision"] == "rejected"
