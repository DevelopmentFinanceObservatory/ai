"""The arms a run can set against the baseline.

An arm changes what the planner is shown and nothing else: each of its `edits` rewrites
each request the planner sends its model, after the seam's own before-model callbacks,
so the arm's planner runs the same entry as the baseline's. Add an arm here to try a
declaration or instruction change before writing it into the planner. A change already
in production code is measured across revisions instead: run `baseline` on the base
revision, then `baseline-replica` on the changed one against those records
(`run.py --baseline-records`).
"""

from __future__ import annotations

from dataclasses import dataclass

from entry import RequestEdit


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    edits: tuple[RequestEdit, ...] = ()


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm("baseline", "The planner as the harness serves it."),
        Arm(
            "baseline-replica",
            "The baseline under a second name, for an A/A run that shows the noise floor, "
            "or for a changed revision set against a stored baseline.",
        ),
    )
}
