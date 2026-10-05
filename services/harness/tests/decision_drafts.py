"""Canonical `Decision`s as a model writes them to `propose_plan`.

The fixtures build canonical `Decision`s, whose `fixes` carries values.
`propose_plan` takes a `DecisionDraft`, whose `fixes` names fields, and reads
each value back off the requirements citing it
(`planner.plan.settle_decisions`).
"""

from __future__ import annotations

from collections.abc import Iterable

from planner.plan import Decision, DecisionDraft


def drafts(decisions: Iterable[Decision | object]) -> list[object]:
    """Each `Decision` in `decisions` as its `DecisionDraft`; anything else as it is."""
    return [
        DecisionDraft.model_validate(
            {
                **decision.model_dump(exclude={"fixes"}),
                "fixes": None
                if decision.fixes is None
                else [field for field, value in decision.fixes if value is not None],
            }
        )
        if isinstance(decision, Decision)
        else decision
        for decision in decisions
    ]
