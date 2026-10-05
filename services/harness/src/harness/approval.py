"""The typed record of a user's answer to a plan revision.

`ApprovalRecord` is written at the HTTP boundary in `studio.py`'s
`/interrupt` route, for every decision a user makes (approve, amend or
reject), not only for approvals. It records the act, not its consequence:
whether the graph goes on to execute the revision named is the `approve`
node's business, decided one step later against the same revision.

The record persists with the rest of the conversation. `StudioStore`
(`studio_state.py`) mirrors every `ApprovalRecord` a conversation
accumulates into its ADK ledger session, so it survives a restart and
reaches whichever instance next loads the conversation, the same as
`ConversationState`'s other durable fields.

`approver` is not a person. It is `harness.auth.principal_for`'s return
value, `"u_" + sha256(token)[:16]`, a hash of the bearer token that
authenticated the request. Two requests bearing the same token produce the
same `approver`; a request under an unconfigured `STUDIO_API_TOKEN` accepts
any non-empty token, so in that mode `approver` identifies whichever token
the caller chose to send, not who they are. `auth_enforced` is what
distinguishes the two cases: it is read from `settings.studio_api_token is
not None` at the moment the record is written, so a record made while the
token gate was open is never mistaken for one made while it was closed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from planner.records import StrictModel


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    AMENDED = "amended"
    REJECTED = "rejected"


class ApprovalRecord(StrictModel):
    """One answer a user gave to one plan revision.

    A record naming a revision the graph then refuses as superseded (a
    stale plan card, answered after a newer one replaced it) is a correct
    record of what the user answered, not a bug to reconcile.
    """

    revision: str
    approver: str
    decision: ApprovalDecision
    at: datetime
    conversation_id: str
    interrupt_id: str
    plan_version: int | None = None
    auth_enforced: bool
