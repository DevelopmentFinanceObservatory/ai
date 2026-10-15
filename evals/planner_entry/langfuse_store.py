"""The eval's runs in Langfuse, so a baseline run on one machine pairs against an arm on another.

`push` copies a run directory's records into `PROJECT_ID`, the Langfuse project the
repo's `.env` keys name. The dataset `harness-planner-entry-v1` holds one item per task:
its question, brief and asks, and the digest of what a run of it shows the planner
(`entry.inputs_digest`). Each arm in the run directory is one experiment, named
`<run directory>/<arm> <revision> <experiment> <records>` (`experiment_name`), where
`<records>` begins the digest of the arm's records, so a directory changed after its
push goes up as a new experiment. Langfuse shows one item run per dataset item in an
experiment, so each task has one item run, whose output holds every record of the
task: each repetition, and each attempt that errored and was retried. Its scores are
the task's, over its runs as the summary counts them (`scores`).

`pull` reads an experiment's records back in their `records.jsonl` order, so
`summarise.build_report` pairs an arm against a pushed baseline exactly as against the
file. Every item run carries the number of tasks and records the arm pushed and the
digest of them all, so a pull refuses an experiment Langfuse shows only in part, as
after a push that failed. `run.py --baseline-langfuse` pairs against one, and
`pull --out` writes one to a file.

The local records stay the source of truth. Both directions refuse keys that are missing
or name another project; a push reads Langfuse back until every task and score it sent
shows, and a pull checks each task's records against the digest pushed with them.
Pushing a run directory again sends only the tasks and scores Langfuse lacks.

An item run's trace holds the task's records and nothing else. The eval exports no ADK
spans, so no planner entry trace exists to link to, and the client sends on a tracer
provider of its own, so a push from `run.py` exports none either.

    uv run --env-file .env --package harness python evals/planner_entry/langfuse_store.py \\
        push evals/planner_entry/.results/<run>
    uv run --env-file .env --package harness python evals/planner_entry/langfuse_store.py \\
        pull '<experiment name>' --out <file>.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from entry import RECORD_SCHEMA, Task, inputs_digest, load_recording, load_tasks
from langfuse import Evaluation, Langfuse
from langfuse.api import DatasetItem, NotFoundError
from opentelemetry.sdk.trace import TracerProvider
from planner.settings import AgentSettings
from summarise import (
    check_schema,
    latest,
    left_out_reason,
    read_records,
    reduce_run,
    write_records,
)

DATASET = "harness-planner-entry-v1"
# "DFO website" on Langfuse Cloud EU: the project the `dfo-dev-langfuse-*` keys in `.env`
# name, where the apis service's search evals keep their experiments too.
PROJECT_ID = "cmqpekjoj046iad0e8b3r8psp"
# How long a push waits for Langfuse to show everything it sent, and how often it looks.
VISIBLE_WITHIN_S = 180.0
LOOK_EVERY_S = 5.0
# Item runs a push sends at once.
CONCURRENCY = 8
_PAGE = 100
# The token counts a task is scored on, as `summarise.Run` names them.
TOKENS = ("input_tokens", "cached_tokens", "output_tokens", "thinking_tokens")


def connect() -> Langfuse:
    """A client for `PROJECT_ID` from the `LANGFUSE_*` keys, refused when a key is missing
    or names another project.

    It sends on a tracer provider of its own, so ADK spans in the same process stay unsent.
    """
    settings = AgentSettings()
    keys = {
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_BASE_URL": settings.langfuse_base_url,
    }
    if missing := [name for name, value in keys.items() if not value]:
        raise SystemExit(f"Langfuse needs {', '.join(missing)} in .env (CONTRIBUTING.md)")
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        tracer_provider=TracerProvider(),
    )
    check_project(client)
    return client


def check_project(client: Langfuse) -> None:
    """Refuse a client whose keys name any project but `PROJECT_ID`."""
    found = [project.id for project in client.api.projects.get().data]
    if found != [PROJECT_ID]:
        raise SystemExit(
            f"the Langfuse keys name project {', '.join(found) or 'none'}; the eval's runs "
            f"belong in {PROJECT_ID}"
        )


def digest(value: Any) -> str:
    """A digest of `value` as JSON, which what a pull reads back must match."""
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def experiment_name(run: str, records: Sequence[Mapping[str, Any]]) -> str:
    """`<run>/<arm> <revision> <experiment> <records>` for one arm's records from the run
    directory named `run`, refused unless they share one experiment (`run.experiment`).
    `<records>` is the start of their digest."""
    treatments = sorted({(r["arm"], r["revision"], r["experiment"]) for r in records})
    if len(treatments) != 1:
        raise SystemExit(
            f"{run} holds runs of {treatments[0][0]} under {len(treatments)} experiments; "
            "push each run directory on its own"
        )
    if "," in run:
        raise SystemExit(f"Langfuse reads the comma in {run} as a list of names; rename it")
    arm, revision, experiment = treatments[0]
    return f"{run}/{arm} {revision} {experiment} {digest(list(records))[:12]}"


def scores(records: Sequence[Mapping[str, Any]], task: Task) -> list[Evaluation]:
    """A task's scores over its runs as the summary counts them: one per repetition, the
    last that is not an error (`summarise.latest`), left out when it errored or missed its
    recording (`summarise.left_out_reason`).

    `runs_scored` and `runs_left_out` count them. The outcomes are the shares of scored
    runs whose first proposal was accepted and whose kinds pass, or on a task with a
    `first_proposal_refusal`, the share that ended in that refusal and an ask. The costs
    are means over the scored runs, and the tokens over those whose usage is known.
    """
    chosen = list(latest(records).values())
    runs = [reduce_run(record, task) for record in chosen if left_out_reason(record) is None]
    values: dict[str, float] = {"runs_scored": len(runs), "runs_left_out": len(chosen) - len(runs)}
    if runs:
        if task.first_proposal_refusal is not None:
            values["expected_ending"] = fmean(run.expected_ending for run in runs)
        else:
            values["first_proposal_accepted"] = fmean(run.first_ok for run in runs)
            values["kind_pass"] = fmean(run.kind_pass for run in runs)
        values["model_calls"] = fmean(run.calls for run in runs)
        values["wall_s"] = fmean(run.wall_s for run in runs)
        if known := [run for run in runs if run.usage_known]:
            values |= {name: fmean(getattr(run, name) for run in known) for name in TOKENS}
    return [
        Evaluation(name=name, value=float(value), data_type="NUMERIC")
        for name, value in values.items()
    ]


# --- The dataset ---------------------------------------------------------------------


def item_id(task_id: str) -> str:
    """A task's dataset item id. Langfuse wants an item id unique across a project's datasets."""
    return f"{DATASET}:{task_id}"


def task_item(task: Task) -> dict[str, Any]:
    """A task's dataset item: what the orchestrator handed the planner, and the digest of
    everything a run of the task shows it."""
    return {
        "input": {"question": task.question, "brief": task.brief, "asks": task.asks},
        "metadata": {
            "task": task.id,
            "kind": task.kind,
            "inputs_digest": inputs_digest(task, load_recording(task.id)),
        },
    }


def dataset_items(client: Langfuse, tasks: Iterable[Task]) -> dict[str, DatasetItem]:
    """Each task's item in `DATASET`, written where Langfuse lacks it or holds another."""
    try:
        held = {item.id: item for item in client.get_dataset(DATASET).items}
    except NotFoundError:
        client.create_dataset(
            name=DATASET, description="dfo-ai's planner entry eval: one item per task."
        )
        held = {}
    items = {}
    for task in tasks:
        wanted = task_item(task)
        item = held.get(item_id(task.id))
        if item is None or (item.input, item.metadata) != (wanted["input"], wanted["metadata"]):
            item = client.create_dataset_item(dataset_name=DATASET, id=item_id(task.id), **wanted)
        items[task.id] = item
    return items


# --- Reading an experiment back ------------------------------------------------------


@dataclass(frozen=True)
class Pushed:
    """One task's item run in Langfuse: its records, the `records.jsonl` line of each, the
    count of tasks and records its arm pushed and their digest, and its trace and score
    names."""

    lines: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    digest: str
    pushed_with: tuple[str, int, int]
    trace_id: str
    scores: frozenset[str]


def task_run(task_id: str, numbered: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The output of `task_id`'s item run: its records among `numbered`, by line."""
    lines = [line for line, record in numbered.items() if record["task"] == task_id]
    return {"task": task_id, "lines": lines, "records": [numbered[line] for line in lines]}


def pushed(client: Langfuse, name: str) -> dict[str, Pushed]:
    """The item runs Langfuse shows under the experiment `name`, by task, each refused
    unless its records match the digest pushed with them."""
    try:
        dataset = client.api.datasets.get(dataset_name=DATASET)
    except NotFoundError:
        return {}
    found: dict[str, Pushed] = {}
    cursor = None
    while True:
        page = client.api.experiments.list_items(
            from_start_time=dataset.created_at,
            experiment_name=name,
            dataset_id=dataset.id,
            fields="core,io,metadata,scores",
            limit=_PAGE,
            cursor=cursor,
        )
        for item in page.data:
            metadata = item.metadata or {}
            output = item.output
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    output = None
            if not isinstance(output, dict) or digest(output) != metadata.get("task_digest"):
                raise SystemExit(
                    f"item run {item.id} of {name} in Langfuse does not hold the records "
                    "pushed with it"
                )
            found[output["task"]] = Pushed(
                lines=tuple(output["lines"]),
                records=tuple(output["records"]),
                digest=digest(output),
                pushed_with=(
                    str(metadata["records_digest"]),
                    int(metadata["tasks"]),
                    int(metadata["records"]),
                ),
                trace_id=item.trace_id,
                scores=frozenset(score.name for score in item.scores or ()),
            )
        cursor = page.meta.cursor
        if not page.data or not cursor:
            return found


def pull(client: Langfuse, name: str) -> list[dict[str, Any]]:
    """The records pushed under the experiment `name`, in their `records.jsonl` order,
    refused unless Langfuse shows every task and record the arm pushed."""
    held = pushed(client, name)
    if not held:
        raise SystemExit(f"Langfuse holds no runs under {name!r} in {DATASET}")
    if len({task.pushed_with for task in held.values()}) != 1:
        raise SystemExit(f"the item runs of {name} in Langfuse were pushed with different runs")
    wanted, tasks, count = next(iter(held.values())).pushed_with
    by_line = {
        line: record
        for task in held.values()
        for line, record in zip(task.lines, task.records, strict=True)
    }
    records = [by_line[line] for line in sorted(by_line)]
    if len(held) < tasks or len(records) < count:
        raise SystemExit(
            f"Langfuse shows {len(held)} of the {tasks} tasks of {name}, {len(records)} of "
            f"its {count} runs; push the run directory again to send the rest"
        )
    if digest(records) != wanted:
        raise SystemExit(f"the runs of {name} in Langfuse do not match the runs pushed")
    return check_schema(records)


# --- Pushing a run -------------------------------------------------------------------


def push(
    client: Langfuse, run_dir: Path, *, tasks: Sequence[Task], wait_s: float = VISIBLE_WITHIN_S
) -> list[str]:
    """Push the records in `run_dir`, one experiment per arm, and return the experiments'
    names. Langfuse is read back until it shows every task and score, for up to `wait_s`."""
    records = read_records([run_dir / "records.jsonl"])
    by_id = {task.id: task for task in tasks}
    ran = sorted({r["task"] for r in records})
    if unknown := [task_id for task_id in ran if task_id not in by_id]:
        raise SystemExit(f"{run_dir} holds runs of tasks the set lacks: {', '.join(unknown)}")
    items = dataset_items(client, [by_id[task_id] for task_id in ran])
    names = []
    for arm in dict.fromkeys(r["arm"] for r in records):
        numbered = {line: r for line, r in enumerate(records) if r["arm"] == arm}
        name = experiment_name(run_dir.name, list(numbered.values()))
        _push_arm(client, name, numbered, items=items, tasks=by_id, wait_s=wait_s)
        names.append(name)
    return names


def _push_arm(
    client: Langfuse,
    name: str,
    numbered: Mapping[int, dict[str, Any]],
    *,
    items: Mapping[str, DatasetItem],
    tasks: Mapping[str, Task],
    wait_s: float,
) -> None:
    """Send what Langfuse lacks of one arm's records, `numbered` by line, one item run per
    task, then wait for it."""
    outputs = {
        task_id: task_run(task_id, numbered)
        for task_id in dict.fromkeys(r["task"] for r in numbered.values())
    }
    pushed_with = {
        "records_digest": digest(list(numbered.values())),
        "tasks": len(outputs),
        "records": len(numbered),
    }
    expected = {
        task_id: scores(output["records"], tasks[task_id]) for task_id, output in outputs.items()
    }
    held = pushed(client, name)
    if any(digest(outputs.get(task_id)) != task.digest for task_id, task in held.items()):
        raise SystemExit(
            f"Langfuse holds runs under {name} that the run directory lacks: they were edited "
            "there since the push"
        )
    pending = {task_id: output for task_id, output in outputs.items() if task_id not in held}
    if pending:
        first = next(iter(numbered.values()))
        result = client.run_experiment(
            name=DATASET,
            run_name=name,
            description=f"{first['arm']} on {first['model']} at {first['revision']}",
            data=[
                items[task_id].model_copy(
                    update={
                        "metadata": {
                            "task": task_id,
                            "task_digest": digest(output),
                            **pushed_with,
                        }
                    }
                )
                for task_id, output in pending.items()
            ],
            task=lambda *, item, **_: pending[item.metadata["task"]],
            evaluators=[lambda *, output, **_: scores(output["records"], tasks[output["task"]])],
            metadata={
                "arm": first["arm"],
                "model": first["model"],
                "revision": first["revision"],
                "experiment": first["experiment"],
                "record_schema": str(RECORD_SCHEMA),
            },
            max_concurrency=CONCURRENCY,
        )
        if len(result.item_results) != len(pending):
            raise SystemExit(
                f"{len(pending) - len(result.item_results)} of {len(pending)} item runs of "
                f"{name} failed to send; push the run directory again"
            )
    for task_id, task in held.items():
        for evaluation in expected[task_id]:
            if evaluation.name not in task.scores:
                client.create_score(  # ty: ignore[no-matching-overload]
                    trace_id=task.trace_id,
                    name=evaluation.name,
                    value=evaluation.value,
                    data_type=evaluation.data_type,
                )
    client.flush()
    print(f"{name}: sent {len(pending)} tasks, {len(held)} already in Langfuse", file=sys.stderr)
    _await_visible(client, name, expected, wait_s=wait_s)


def _await_visible(
    client: Langfuse, name: str, expected: Mapping[str, list[Evaluation]], *, wait_s: float
) -> None:
    """Return once Langfuse shows every task in `expected` with its scores, or refuse
    after `wait_s`."""
    deadline = time.monotonic() + wait_s
    while True:
        held = pushed(client, name)
        short = [
            task_id
            for task_id, evaluations in expected.items()
            if task_id not in held or not {e.name for e in evaluations} <= held[task_id].scores
        ]
        if not short:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"after {wait_s:.0f} s Langfuse shows {len(expected) - len(short)} of the "
                f"{len(expected)} tasks of {name} with all their scores; push the run "
                "directory again to send the rest"
            )
        time.sleep(min(LOOK_EVERY_S, max(deadline - time.monotonic(), 0.0)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    push_command = commands.add_parser("push", help="push a run directory's records")
    push_command.add_argument("run", type=Path, help="the run directory, holding records.jsonl")
    pull_command = commands.add_parser("pull", help="write a pushed experiment's records")
    pull_command.add_argument("name", help="the experiment's name, as push printed it")
    pull_command.add_argument("--out", type=Path, required=True, help="the JSONL file to write")
    args = parser.parse_args(argv)
    client = connect()
    if args.command == "push":
        for name in push(client, args.run, tasks=load_tasks()):
            print(name)
    else:
        write_records(args.out, pull(client, args.name))
        print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
