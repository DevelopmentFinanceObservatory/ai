"""Write `users/<task>.json`, what the simulated user says, for each sample task.

Everything comes from the task file. The user opens with the task's
`question`. What they mean, if asked, is the task's `brief` and `asks`. A
decision the reference plan records as `resolved_by: user` is answered with
its `chosen` option; one the planner settled by default
(`resolved_by: planner_default`) gets no preference, since the user stated
none. A plan is approved only if its card answers every ask.

    python make_users.py
"""

from __future__ import annotations

import json
from typing import Any

from tasks import SAMPLE, USERS, load_task


def ask_text(ask: Any) -> str:
    return ask if isinstance(ask, str) else ask["text"]


def user_script(task: dict[str, Any]) -> dict[str, Any]:
    asks = [ask_text(ask) for ask in task["asks"]]
    answers = []
    for decision in task["reference_payload"].get("decisions") or []:
        if decision.get("resolved_by") == "user":
            say = decision["chosen"]
        else:
            say = "No preference; choose what is usual and say what you chose."
        answers.append(
            {"if_asked": decision["description"], "options": decision["options"], "say": say}
        )
    answers.append(
        {
            "if_asked": "anything else the question does not settle (a source, a year, a unit, "
            "a definition)",
            "say": "No preference beyond what I asked; choose what answers my question and say "
            "what you chose.",
        }
    )
    return {
        "task": task["id"],
        "opening": task["question"],
        "meaning": {"brief": task["brief"], "asks": asks},
        "answers": answers,
        "approve": (
            "Approve a plan only if what it says it will produce answers the question: every "
            "one of the asks above, for the places, flows and years asked. If it misses or "
            "changes one, say which and do not approve."
        ),
    }


def main() -> None:
    USERS.mkdir(exist_ok=True)
    for task_id in SAMPLE:
        script = user_script(load_task(task_id))
        (USERS / f"{task_id}.json").write_text(json.dumps(script, indent=2) + "\n")
    print(f"wrote {len(SAMPLE)} user scripts to {USERS}")


if __name__ == "__main__":
    main()
