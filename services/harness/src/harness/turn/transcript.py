"""The turn's transcript: the agents' records, in the order they were written.

The stored transcript is the turn's `agent_records` rows
(`harness.turn.store.TurnStore.records`). A segment of the turn loads it once
and holds it as a `Transcript`; the nodes append to it as they run, and the
loop writes what they appended with the cursor at each node boundary
(`TurnStore`'s `records` argument), so entries appended after the last
boundary die with a crash.

The loop writes three kinds of entry under agent `loop`: the user's message
(`user_message`, written when the turn opens), each pause (`pause`) and each
accepted answer (`answer`). The planner writes its own run records under
agent `planner`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.turn.store import AgentRecord

USER = "user"
PAUSE = "pause"
ANSWER = "answer"


class Transcript:
    """One segment's view of the turn's transcript: the `stored` records it
    loaded, then everything appended since."""

    def __init__(self, stored: Sequence[AgentRecord] = ()) -> None:
        self._records = list(stored)
        self._written = len(self._records)

    @property
    def records(self) -> tuple[AgentRecord, ...]:
        """Every record, stored and unwritten, in order."""
        return tuple(self._records)

    def append(self, record: AgentRecord) -> None:
        """Add `record` at the end; the loop writes it at the next boundary."""
        self._records.append(record)

    def take(self) -> list[AgentRecord]:
        """The records appended since the last `take`, which the caller
        writes; they count as written from here on."""
        unwritten = self._records[self._written :]
        self._written = len(self._records)
        return unwritten


def user_message(text: str) -> AgentRecord:
    """The message the user opened the turn with."""
    return AgentRecord(agent="loop", kind=USER, body={"text": text})


def pause(
    *, kind: str, interrupt_id: str, round: int, payload: dict[str, Any], message: str
) -> AgentRecord:
    """A pause put to the user: its kind, wire id, round and question."""
    return AgentRecord(
        agent="loop",
        kind=PAUSE,
        body={
            "kind": kind,
            "interrupt_id": interrupt_id,
            "round": round,
            "payload": payload,
            "message": message,
        },
    )


def answer(*, interrupt_id: str, round: int, value: dict[str, Any]) -> AgentRecord:
    """The user's answer `value` to the pause `(interrupt_id, round)`."""
    return AgentRecord(
        agent="loop",
        kind=ANSWER,
        body={"interrupt_id": interrupt_id, "round": round, "answer": value},
    )
