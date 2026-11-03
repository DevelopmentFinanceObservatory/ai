"""The planner-entry tasks and the simulated user's scripts for them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASKS = HERE.parents[3] / "evals" / "planner_entry" / "tasks"
USERS = HERE / "users"

SAMPLE = (
    "series-gdp-ghana",
    "series-oda-germany-kenya",
    "growth-total-oda-ghana",
    "aggregate-total-oda-kenya",
    "rank-top-donors-malawi",
    "join-gdp-debt-service-ghana",
    "sum-oda-oof-ethiopia",
    "share-two-donors-of-kenya-oda",
    "quotient-oda-per-person-ethiopia",
    "aggregate-health-population-oda-germany-by-sector",
)


def load_task(task_id: str) -> dict[str, Any]:
    return json.loads((TASKS / f"{task_id}.json").read_text())


def load_user(task_id: str) -> dict[str, Any]:
    return json.loads((USERS / f"{task_id}.json").read_text())
