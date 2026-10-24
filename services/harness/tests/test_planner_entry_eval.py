"""CI's share of the planner entry eval (`evals/planner_entry`): its task set holds,
and its summary scores runs as its README says.

`validate.py` proposes each task's plans through the eval's own entry, the real
orchestrator tools, graph nodes and `PlannerSeam` over the task's recording, with a
scripted model in place of the planner's. So a change to the planner's rules, the
seam's brief or the orchestrator's search storage that leaves a reference plan
refused, a plan whose translation no longer compiles, or a recording that no longer
replays or lacks a first-year read or a call the translation makes, fails here rather
than in a paid run. No model, no network. The
full-set test takes about 20 s: it runs one scripted entry for each of the 68 tasks'
plans, and one more per task for the datasets its seam shows the planner.

`evals/` sits outside the workspace, so its modules are loaded by file path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import dataclasses
import datetime
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, get_args

import pytest
from compiler.surface import ToolSurfaceUnavailableError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from langfuse.api import DatasetItem, NotFoundError
from planner.plan import EvidenceRequirement
from pydantic import ValidationError
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "planner_entry"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EVAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


entry = _load("entry")
validate = _load("validate")
summarise = _load("summarise")
# `run.py` imports `arms` and `langfuse_store` by name, so they load first.
arms = _load("arms")
langfuse_store = _load("langfuse_store")
run = _load("run")
recorder = _load("record")


async def test_every_task_validates_and_the_set_covers_every_kind() -> None:
    entry.quiet_logs()

    faults = await validate.check_all(entry.load_tasks())

    assert faults == {}


def _search_reworded(task: Any) -> dict[str, Any]:
    searches = [search.model_dump() for search in task.searches]
    searches[0]["query"] += " (reworded)"
    return {"searches": searches}


def _ask_unanswered(task: Any) -> dict[str, Any]:
    return {"reference_payload": {**task.reference_payload, "answered": []}}


def _required_kind_unused(task: Any) -> dict[str, Any]:
    return {
        "required_kinds": [*task.required_kinds, "join"],
        "acceptable_kinds": [*task.acceptable_kinds, "join"],
    }


def _years_misdeclared(task: Any) -> dict[str, Any]:
    del task
    return {"period_reduction": "none"}


def _ask_unmarked(task: Any) -> dict[str, Any]:
    return {"asks": [ask.text for ask in task.asks]}


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        (_search_reworded, "RecordingMismatchError"),
        (_ask_unanswered, "not accepted"),
        (_required_kind_unused, "lacks required kinds ['join']"),
        (_years_misdeclared, "period_reduction is none, but the plan's is across_years"),
        (_ask_unmarked, "not accepted: invalid_input/period_reduction_unstated"),
    ],
)
async def test_a_broken_task_is_reported(change: Any, fault: str) -> None:
    entry.quiet_logs()
    task = next(task for task in entry.load_tasks() if task.id == "aggregate-total-oda-kenya")

    broken = entry.Task.model_validate({**task.model_dump(), **change(task)})

    faults = await validate.check_task(broken)

    assert any(fault in line for line in faults), faults


def _task(task_id: str) -> Any:
    return next(task for task in entry.load_tasks() if task.id == task_id)


async def test_a_plan_propose_plan_accepts_but_the_translation_refuses_is_reported() -> None:
    # GDP as a share of population: `propose_plan` admits it, and the compile refuses
    # dividing US dollars by a count with `share_of`. Both pins' blocks are exact, so
    # the translation needs no recorded call.
    entry.quiet_logs()
    task = _task("join-population-gdp-kenya")
    payload = copy.deepcopy(task.reference_payload)
    payload["evidence_requirements"][2] = {
        "kind": "share_of",
        "id": "table",
        "numerator": {"input_id": "gdp"},
        "denominator": {"input_id": "population"},
    }
    broken = entry.Task.model_validate({**task.model_dump(), "reference_payload": payload})

    faults = await validate.check_task(broken)

    assert any(
        line.startswith("reference: translation requirement_needs_repair: share_units_differ@table")
        for line in faults
    ), faults


async def test_a_translation_reads_only_the_search_recorded_under_the_arguments_it_sends_now() -> (
    None
):
    # `growth-gdp-kenya` pins nothing, so `discover_source` searches. Its answer is the
    # orchestrator's recorded search, filed under whatever arguments discovery sent.
    entry.quiet_logs()
    task = _task("growth-gdp-kenya")
    recording = entry.load_recording(task.id)
    reference = validate.task_payloads(task)[0]
    schemas = await validate.tool_schemas()

    async def translate(held: dict[str, Any], payload: Any) -> tuple[list[str], Any]:
        calls = entry.TranslationCalls(held, schemas=schemas)
        run = await validate.accepted_run(task, recording, payload, calls)
        return validate.translation_faults(run, calls), calls

    unrecorded, calls = await translate({}, reference)
    (asked,) = calls.misses
    answer = {
        "is_error": False,
        "structured_content": {},
        "text": "",
        "meta": {SEARCH_ENVELOPE_META_KEY: recording["searches"][0]["envelope"]},
    }
    held = {"calls": {asked: answer}}
    recorded, _ = await translate(held, reference)
    body = copy.deepcopy(dict(reference.body))
    body["evidence_requirements"][0]["measure"] = "gross domestic product"
    edited, calls = await translate(held, dataclasses.replace(reference, body=body))

    assert unrecorded == [f"no recording for {asked}"]
    assert asked.startswith("search ") and '"variable":"GDP"' in asked
    assert recorded == []
    assert len(edited) == 1 and '"variable":"gross domestic product"' in edited[0], edited
    assert calls.unasked() == [asked]


class _LiveClient:
    """A data-mcp client whose every call `answer` answers, keeping what it was asked."""

    def __init__(self, answer: Callable[[], Any]) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any], **_: Any) -> Any:
        self.asked.append(entry.surface_key(name, arguments))
        return await self.answer()


def _searched(envelope: dict[str, Any]) -> Callable[[], Any]:
    async def answer() -> Any:
        return SimpleNamespace(
            is_error=False,
            structured_content={},
            content=[],
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )

    return answer


async def _refused_for_now() -> Any:
    return SimpleNamespace(
        is_error=True,
        structured_content={"retry_mode": "retry_after_delay", "message": "busy"},
        content=[],
        meta=None,
    )


async def _cut() -> Any:
    raise ToolSurfaceUnavailableError("search timed out")


async def _broken() -> Any:
    raise RuntimeError("connection reset")


async def _stalled() -> Any:
    await asyncio.sleep(30)


@pytest.mark.parametrize(
    ("answer", "refusal"),
    [
        (_refused_for_now, "refused for now"),
        (_cut, "ToolSurfaceUnavailableError: search timed out"),
        (_broken, "RuntimeError: connection reset"),
        (_stalled, "ran past 0.5 s"),
    ],
)
async def test_a_live_translation_that_cannot_be_replayed_is_not_recorded(
    answer: Callable[[], Any], refusal: str
) -> None:
    entry.quiet_logs()
    task = _task("growth-gdp-kenya")
    recording = entry.load_recording(task.id)
    client = _LiveClient(answer)

    with pytest.raises(recorder.TranslationRecordError) as raised:
        await recorder._translation_calls(client, task, recording, wall_s=0.5)

    assert raised.value.args[0].startswith("reference: ")
    assert refusal in raised.value.args[0]
    if answer is not _stalled:
        assert client.asked[0] in raised.value.args[0]


async def test_a_live_translation_records_each_call_a_replay_reads() -> None:
    entry.quiet_logs()
    task = _task("growth-gdp-kenya")
    recording = entry.load_recording(task.id)
    client = _LiveClient(_searched(recording["searches"][0]["envelope"]))

    kept = await recorder._translation_calls(client, task, recording, wall_s=60.0)

    assert list(kept["calls"]) == client.asked
    assert kept["calls"][client.asked[0]]["meta"][SEARCH_ENVELOPE_META_KEY]["parts"]


def test_a_task_states_its_period_reduction_in_the_closed_vocabulary() -> None:
    task = next(task for task in entry.load_tasks() if task.id == "aggregate-total-oda-kenya")

    with pytest.raises(ValidationError) as refused:
        entry.Task.model_validate({**task.model_dump(), "period_reduction": "one_year"})
    with pytest.raises(ValidationError) as missing:
        entry.Task.model_validate(task.model_dump(exclude={"period_reduction"}))

    assert [error["type"] for error in refused.value.errors()] == ["enum"]
    assert [error["type"] for error in missing.value.errors()] == ["missing"]


def _tamper_answer(directory: Path, name: str) -> None:
    path = directory / "answers" / f"{name}.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**stored, "answer": {"edited": True}}), encoding="utf-8")


def _drop_answer(directory: Path, name: str) -> None:
    (directory / "answers" / f"{name}.json").unlink()


@pytest.mark.parametrize(
    ("change", "fault"),
    [(_tamper_answer, "does not match its content"), (_drop_answer, "is not in answers/")],
)
def test_a_recording_refuses_an_answer_it_cannot_vouch_for(
    tmp_path: Path, change: Any, fault: str
) -> None:
    task_id = "series-gdp-ghana"
    entry.save_recording(task_id, entry.load_recording(task_id), tmp_path)
    name = json.loads(entry.recording_path(task_id, tmp_path).read_text())["answers"][0]

    change(tmp_path, name)

    with pytest.raises(entry.RecordingMismatchError, match=fault):
        entry.load_recording(task_id, tmp_path)


def test_stray_recordings_name_files_no_task_uses(tmp_path: Path) -> None:
    task_id = "series-gdp-ghana"
    entry.save_recording(task_id, entry.load_recording(task_id), tmp_path)
    entry.save_recording(
        "removed-task", entry.load_recording("aggregate-total-oda-kenya"), tmp_path
    )

    stray = entry.stray_recordings({task_id}, tmp_path)

    kept = set(json.loads(entry.recording_path(task_id, tmp_path).read_text())["answers"])
    assert stray[0] == entry.recording_path("removed-task", tmp_path)
    assert stray[1:] and all(path.parent.name == "answers" for path in stray[1:])
    assert not kept & {path.stem for path in stray[1:]}


def test_a_translation_answer_is_stored_once_and_read_back(tmp_path: Path) -> None:
    call = 'search {"parts":[{"variable":"GDP"}]}'
    answer = {"is_error": False, "structured_content": {}, "text": "", "meta": {"envelope": 1}}
    translation = {"calls": {call: answer}, "resolved": {"Kenya": None}}
    task_ids = ("series-gdp-ghana", "aggregate-total-oda-kenya")
    for task_id in task_ids:
        recorded = {**entry.load_recording(task_id), "translation": translation}
        entry.save_recording(task_id, recorded, tmp_path)

    body = json.loads(entry.recording_path(task_ids[0], tmp_path).read_text())

    name = entry.answer_id(call, answer)
    assert body["translation"] == {"answers": [name], "resolved": {"Kenya": None}}
    assert (tmp_path / "answers" / f"{name}.json").exists()
    assert entry.load_recording(task_ids[0], tmp_path)["translation"] == translation
    assert entry.stray_recordings(set(task_ids), tmp_path) == []


class _StalledLlm(validate.ScriptedLlm):
    """A model whose every generation starts and never answers."""

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        self.requests.append(llm_request)
        await asyncio.Event().wait()
        yield


_SCRIPTED = (
    validate.SCRIPTED_USAGE.prompt_token_count,
    validate.SCRIPTED_USAGE.cached_content_token_count,
    validate.SCRIPTED_USAGE.candidates_token_count,
    validate.SCRIPTED_USAGE.thoughts_token_count,
)
_UNKNOWN = (None, None, None, None)


# The stalled case waits out a one-second run limit. Allowed one model call, the entry
# sends a refused proposal, and the turn budget answers the next call itself.
@pytest.mark.parametrize(
    ("calls_max", "model_type", "script", "stop", "closed", "usage"),
    [
        ("30", validate.ScriptedLlm, ["plan"], "accepted", None, [_SCRIPTED]),
        (
            "1",
            validate.ScriptedLlm,
            ["empty plan", "plan"],
            "turn_model_calls_spent",
            "turn_model_calls_spent",
            [_SCRIPTED],
        ),
        ("30", _StalledLlm, ["plan"], "timeout", None, [_UNKNOWN]),
    ],
)
async def test_a_record_counts_each_model_call_and_keeps_the_seams_stop_code(
    monkeypatch: pytest.MonkeyPatch,
    calls_max: str,
    model_type: type[Any],
    script: list[str],
    stop: str,
    closed: str | None,
    usage: list[tuple[int | None, ...]],
) -> None:
    monkeypatch.setenv("TURN_MODEL_CALLS_MAX", calls_max)
    entry.quiet_logs()
    task = next(task for task in entry.load_tasks() if task.id == "series-gdp-ghana")
    payloads = {
        "plan": dict(task.reference_payload),
        "empty plan": {**task.reference_payload, "evidence_requirements": []},
    }
    model = model_type(script=[[("propose_plan", payloads[name])] for name in script])

    run = await entry.run_entry(
        task, entry.load_recording(task.id), prefix_cache=None, model=model, timeout_s=1.0
    )
    record = entry.entry_record(
        task=task, run=run, arm="a", rep=1, model="m", revision="r", experiment="e"
    )

    assert (record["stop_reason"], record["budget_closed"]) == (stop, closed)
    assert len(model.requests) == len(usage)
    assert [
        (
            call["prompt_tokens"],
            call["cached_tokens"],
            call["output_tokens"],
            call["thinking_tokens"],
        )
        for call in record["calls"]
    ] == usage


_PLAIN = "aggregate-total-oda-kenya"
# Its first proposal is refused with `combination_unasked`.
_REFUSAL = "sum-net-flows-aid-debt-ghana"
_COMBINATION_ASK = (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
    {"message": "Add these?", "options": [dict(option) for option in COMBINATION_OPTIONS]},
)
_UNOFFERED_ASK = (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
    {"message": "Add these?", "options": [{"label": "Yes", "spec_id": "yes"}]},
)
_ADDED = [{"result": {"spec_id": COMBINE_OPTION}}]


# A script step names the task's reference payload ("plan") or one with no requirements
# ("empty plan"), or holds a call.
@pytest.mark.parametrize(
    ("task_id", "script", "answers", "stop", "first"),
    [
        # Two proposals in one response: the first records the plan, the second is rejected.
        (_PLAIN, [["plan", "empty plan"]], [], "accepted", "ok"),
        (_REFUSAL, [["plan"], [_COMBINATION_ASK]], [], "asked_user", "refused"),
        # The tool rejects an ask whose options no refusal offered.
        (_REFUSAL, [["plan"], [_UNOFFERED_ASK]], [], "no_plan", "refused"),
        # The user answers the ask, and the planner then ends its turns with no plan.
        (_REFUSAL, [["plan"], [_COMBINATION_ASK]], _ADDED, "plan_not_ready", "refused"),
    ],
)
async def test_a_record_credits_the_proposal_that_recorded_the_plan_and_an_ask_left_awaiting(
    task_id: str, script: list[list[Any]], answers: list[Any], stop: str, first: str
) -> None:
    entry.quiet_logs()
    task = next(task for task in entry.load_tasks() if task.id == task_id)
    payloads = {
        "plan": dict(task.reference_payload),
        "empty plan": {**task.reference_payload, "evidence_requirements": []},
    }
    model = validate.ScriptedLlm(
        script=[
            [("propose_plan", payloads[call]) if isinstance(call, str) else call for call in step]
            for step in script
        ]
    )

    run = await entry.run_entry(
        task,
        entry.load_recording(task.id),
        prefix_cache=None,
        model=model,
        timeout_s=60.0,
        answers=answers,
    )
    record = entry.entry_record(
        task=task, run=run, arm="a", rep=1, model="m", revision="r", experiment="e"
    )

    assert (record["stop_reason"], record["first_proposal"]["outcome"]) == (stop, first)


def _summing_years(payload: dict[str, Any]) -> dict[str, Any]:
    """`payload` with every `aggregate` leaving `period` out of its `group_by`."""
    requirements = [
        {**requirement, "group_by": ["entity"]}
        if requirement["kind"] == "aggregate"
        else requirement
        for requirement in payload["evidence_requirements"]
    ]
    return {**payload, "evidence_requirements": requirements}


_COVERAGE_BRIEF = "aggregate-total-education-oda-kenya-coverage-brief"


# Each task's reference plan, proposed after the same plan summing its years when
# `summed_first`. The cases give the asks the plan holds as marked, what the record
# says of the first proposal, and what it says of the accepted plan's years. A record
# stripped of the period-reduction fields and the plan's marks reads the same, from the
# task's marks.
@pytest.mark.parametrize(
    ("task_id", "summed_first", "marked", "first_code", "reductions", "reduction"),
    [
        (
            "aggregate-total-oda-kenya",
            False,
            [0],
            None,
            [{"step": "total", "start": 2015, "end": 2022, "refused": False}],
            "across_years",
        ),
        ("rank-top-donors-malawi", False, [], None, [], "none"),
        (_COVERAGE_BRIEF, True, [], "period_reduction_unstated", [], "none"),
    ],
)
async def test_a_record_says_what_the_accepted_plan_does_with_its_years(
    task_id: str,
    summed_first: bool,
    marked: list[int],
    first_code: str | None,
    reductions: list[dict[str, Any]],
    reduction: str,
) -> None:
    entry.quiet_logs()
    task = next(task for task in entry.load_tasks() if task.id == task_id)
    payloads = [_summing_years(task.reference_payload)] if summed_first else []
    payloads.append(dict(task.reference_payload))
    model = validate.ScriptedLlm(script=[[("propose_plan", payload)] for payload in payloads])

    run = await entry.run_entry(
        task, entry.load_recording(task.id), prefix_cache=None, model=model, timeout_s=60.0
    )
    record = entry.entry_record(
        task=task, run=run, arm="a", rep=1, model="m", revision="r", experiment="e"
    )

    assert record["stop_reason"] == "accepted"
    assert record["plan"]["asks_across_years"] == marked
    assert record["first_proposal"]["code"] == first_code
    assert record["period_reductions"] == reductions
    assert record["period_reduction"] == reduction
    assert entry.record_period_reduction(_before_the_mark(record), task) == reduction


def _before_the_mark(record: dict[str, Any]) -> dict[str, Any]:
    """`record` stripped of its reading of the plan's years and of the plan's ask marks."""
    kept = {k: v for k, v in record.items() if not k.startswith("period_reduction")}
    return {**kept, "plan": {k: v for k, v in record["plan"].items() if k != "asks_across_years"}}


def test_an_unmarked_ask_digests_as_it_did_before_asks_carried_the_mark() -> None:
    task = next(task for task in entry.load_tasks() if task.id == "rank-donor-each-year-kenya")
    marked = entry.Task.model_validate(
        {**task.model_dump(), "asks": [{**task.asks[0].model_dump(), "across_years": True}]}
    )
    recording = {"searches": [], "tools": {}}

    # The pinned digest is the one `inputs_digest` gives this task over this recording
    # when no ask is marked.
    assert entry.inputs_digest(task, recording) == (
        "dcda82f03fa9c004ffacfc14bc3fe4c66255aed890f6155ca84d79c395465afe"
    )
    assert entry.inputs_digest(marked, recording) != entry.inputs_digest(task, recording)


def _record(
    arm: str,
    task: str,
    rep: int,
    *,
    first: tuple[str, str | None] = ("ok", None),
    kinds: list[str],
    stop: str = "accepted",
    status: str = "ok",
    wall_s: float = 5.0,
    inputs: str = "shown",
    experiment: str = "e",
    misses: tuple[str, ...] = (),
    finished: bool = True,
    closed: str | None = None,
    years: str | None = None,
    plan: dict[str, Any] | None = None,
    proposals: int = 1,
) -> dict[str, Any]:
    """One run's record. `years` is the accepted plan's period reduction. A record given
    none holds no reading and is read from `plan`, the accepted plan it holds. The last
    of several `proposals` is accepted."""
    outcome, code = first
    tokens = (100, 50, 10, 0) if finished else (None, None, None, None)
    accepted = {"outcome": "ok", "code": None, "message": "", "kinds": kinds, "args": {}}
    kept = {} if years is None else {"period_reduction": years}
    return {
        **kept,
        "plan": plan,
        "arm": arm,
        "task": task,
        "rep": rep,
        "model": "m",
        "revision": "r",
        "experiment": experiment,
        "inputs": inputs,
        "status": status,
        "stop_reason": stop,
        "plan_kinds": kinds if stop == "accepted" else None,
        "proposals": [
            {"outcome": outcome, "code": code, "message": "", "kinds": kinds, "args": {}},
            *([accepted] * (proposals - 1)),
        ],
        "first_proposal": {"outcome": outcome, "code": code},
        "calls": [
            dict(
                zip(
                    ("prompt_tokens", "cached_tokens", "output_tokens", "thinking_tokens"),
                    tokens,
                    strict=True,
                ),
                latency_s=wall_s if finished else None,
            )
        ],
        "budget_closed": closed,
        "surface_misses": list(misses),
        "wall_s": wall_s,
    }


@pytest.mark.parametrize(
    ("kinds", "passes"),
    [
        (["dataset"], False),
        (["dataset", "share_of"], True),
        (["dataset", "quotient_of"], True),
        (["dataset", "comparison"], False),
    ],
)
def test_kind_selection_needs_one_of_a_tasks_calculations(kinds: list[str], passes: bool) -> None:
    task = next(t for t in entry.load_tasks() if t.id == "quotient-debt-service-per-gdp-ghana")

    run = summarise.reduce_run(_record("a", task.id, 1, kinds=kinds), task)

    assert run.kind_pass is passes


def test_the_summary_pairs_by_task_and_leaves_out_what_it_cannot_vouch_for() -> None:
    tasks = {task.id: task for task in entry.load_tasks()}
    total, growth, series = "aggregate-total-oda-kenya", "growth-gdp-kenya", "series-gdp-ghana"
    refusal, changed = "sum-net-flows-aid-debt-ghana", "rank-top-gdp-2022"
    refused = ("refused", "combination_unasked")
    records = []
    for rep in (1, 2, 3):
        records += [
            _record("baseline", total, rep, kinds=["aggregate", "dataset"]),
            _record("baseline", growth, rep, kinds=["dataset", "growth"]),
            _record("baseline", series, rep, kinds=["dataset"]),
            _record(
                "baseline",
                refusal,
                rep,
                kinds=["dataset", "sum_of"],
                first=refused,
                stop="asked_user",
            ),
            _record(
                "arm",
                total,
                rep,
                first=("invalid_input", "invalid_plan"),
                kinds=["aggregate", "dataset"],
            ),
            _record("arm", growth, rep, kinds=["dataset"]),
            _record("arm", series, rep, kinds=["dataset"], status="error"),
            _record(
                "arm", refusal, rep, kinds=["dataset", "sum_of"], first=refused, stop="asked_user"
            ),
            _record("baseline", changed, rep, kinds=["dataset", "rank"]),
            _record("arm", changed, rep, kinds=["dataset"], first=refused, inputs="re-recorded"),
        ]
    records += [
        _record(
            "arm",
            total,
            4,
            kinds=["aggregate", "dataset"],
            stop="turn_model_time_spent",
            wall_s=200.0,
            finished=False,
            closed="turn_model_time_spent",
        ),
        # A baseline run that missed its recording, and worse on every count, is left out.
        _record(
            "baseline",
            growth,
            4,
            kinds=["dataset"],
            first=refused,
            wall_s=300.0,
            misses=("resolve_definition {}",),
        ),
    ]

    _report, machine = summarise.build_report(
        records,
        [tasks[t] for t in (total, growth, series, refusal, changed)],
        arm="arm",
        baseline="baseline",
    )

    first = machine["paired"]["First proposal accepted"]
    fired = {row["trigger"].split(" ")[0]: row["fired"] for row in machine["triggers"]}
    assert (first["tasks"], first["better"], first["worse"], first["same"]) == (2, 0, 1, 1)
    assert machine["left_out"] == {
        "baseline": {"errored": {}, "missed the recording": {growth: 1}},
        "arm": {"errored": {series: 3}, "missed the recording": {}},
    }
    assert machine["attempts"] == {"baseline": 16, "arm": 16}
    assert machine["changed_inputs"] == [changed]
    assert machine["arms"]["arm"]["expected_ending"] == 3
    assert machine["arms"]["arm"]["usage_unknown"] == 1
    assert machine["budget_alarm"] == {"baseline": 0, "arm": 1}
    # Two shared tasks leave the first-proposal interval touching zero; the series task,
    # whose arm runs all errored, is no kind-pass task lost, and the task whose inputs
    # changed between the arms, worse in every arm run, pairs nothing.
    assert fired == {"First-proposal": False, "Kind": False, "p90": True, "More": True}


async def _unmarked_plan(task: entry.Task, payload: dict[str, Any]) -> dict[str, Any]:
    """`payload` as a plan stored without ask marks, accepted with the task's one ask
    marked so that `propose_plan` takes its years summed."""
    asked = entry.Task.model_validate(
        {**task.model_dump(), "asks": [{**task.asks[0].model_dump(), "across_years": True}]}
    )
    run = await entry.run_entry(
        asked,
        entry.load_recording(task.id),
        prefix_cache=None,
        model=validate.ScriptedLlm(script=[[("propose_plan", payload)]]),
        timeout_s=60.0,
    )
    assert run.probe.plan is not None
    return run.probe.plan.model_dump(mode="json", exclude={"asks_across_years"})


async def test_the_summary_holds_each_accepted_plan_to_its_tasks_years() -> None:
    entry.quiet_logs()
    tasks = {task.id: task for task in entry.load_tasks()}
    top, total, series = "rank-top-donors-malawi", "aggregate-total-oda-kenya", "series-gdp-ghana"
    summed, ranked = ["aggregate", "dataset"], ["dataset", "rank"]
    # The plans a baseline record keeps: the coverage brief's years summed, where its
    # task marks no ask, and the marked total's reference.
    brief = tasks[_COVERAGE_BRIEF]
    before = await _unmarked_plan(brief, _summing_years(brief.reference_payload))
    marked_total = await _unmarked_plan(tasks[total], dict(tasks[total].reference_payload))
    refused = ("invalid_input", "period_reduction_unstated")
    records = [
        # Refused for its years, then accepted keeping them.
        _record("arm", brief.id, 1, kinds=summed, first=refused, years="none", proposals=2),
        _record("arm", brief.id, 2, kinds=summed, years="none"),
        # A ranking of every year summed, which only a revision without the rule accepts.
        _record("arm", top, 1, kinds=ranked, years="unstated"),
        _record("arm", top, 2, kinds=ranked, years="none"),
        # A marked ask with one total, and a plan that kept the years the task sums.
        _record("arm", total, 1, kinds=summed, years="across_years"),
        _record("arm", total, 2, kinds=summed, years="none"),
        _record("arm", series, 1, kinds=["dataset"], years="none"),
        _record("arm", series, 2, kinds=["dataset"], first=("refused", None), stop="no_plan"),
        # Records that keep no reading, read from their plans with the asks their task
        # marks: one summing years its task keeps, one whose plan `Plan` refuses to
        # validate, which counts as a miss, and one summing the years its task sums.
        _record("baseline", brief.id, 1, kinds=summed, plan=before),
        _record("baseline", brief.id, 2, kinds=summed, plan={**before, "answered": 1}),
        _record("baseline", total, 1, kinds=summed, plan=marked_total),
    ]

    report, machine = summarise.build_report(
        records, [tasks[t] for t in (brief.id, top, total, series)], arm="arm", baseline="baseline"
    )

    arm, baseline = machine["arms"]["arm"], machine["arms"]["baseline"]
    assert (arm["years_expected"], arm["years_accepted"], arm["years_unloaded"]) == (5, 7, 0)
    assert (arm["unmarked_reducing"], arm["unmarked_accepted"]) == (1, 5)
    assert arm["period_first_refusals"] == 1
    assert (arm["rule_tasks"], arm["rule_accepted"], arm["rule_runs"]) == (2, 4, 4)
    assert arm["rule_mean_rounds"] == pytest.approx(5 / 4)
    assert (baseline["years_expected"], baseline["years_accepted"]) == (1, 3)
    assert baseline["years_unloaded"] == 1
    assert (baseline["unmarked_reducing"], baseline["unmarked_accepted"]) == (1, 1)
    assert (baseline["rule_accepted"], baseline["rule_runs"]) == (3, 3)
    rows = {line.split(" | ")[0].strip("| "): line.split(" | ")[-1] for line in report.splitlines()}
    assert rows[top] == "2/2, 2/2, 1/2 |"
    assert rows[series] == "1/2, 2/2, 1/1 |"


def test_the_summary_refuses_an_arm_run_under_two_experiments() -> None:
    task = next(t for t in entry.load_tasks() if t.id == "series-gdp-ghana")
    records = [
        _record("arm", task.id, 1, kinds=["dataset"]),
        _record("arm", task.id, 2, kinds=["dataset"], experiment="changed code"),
    ]

    with pytest.raises(ValueError, match="2 experiments"):
        summarise.build_report(records, [task], arm="arm", baseline="arm")


async def test_a_resume_under_another_experiment_is_refused(tmp_path: Path) -> None:
    task = next(t for t in entry.load_tasks() if t.id == "series-gdp-ghana")
    earlier = {
        **_record("baseline", task.id, 1, kinds=["dataset"], experiment="earlier"),
        "schema": entry.RECORD_SCHEMA,
    }
    (tmp_path / "records.jsonl").write_text(json.dumps(earlier) + "\n", encoding="utf-8")
    args = argparse.Namespace(out=tmp_path, run_timeout=300.0, concurrency=1, reps=1, passes=1)

    with pytest.raises(SystemExit, match="under another model, code, setting or arm"):
        await run.run_all(args, [task], ["baseline"])


def _requirement_kinds(annotation: Any) -> set[str]:
    fields = getattr(annotation, "model_fields", None)
    if fields is not None:
        return {fields["kind"].default}
    return {kind for member in get_args(annotation) for kind in _requirement_kinds(member)}


def test_the_eval_knows_every_requirement_kind_the_planner_declares() -> None:
    assert set(entry.KINDS) == _requirement_kinds(EvidenceRequirement)


# --- Langfuse ------------------------------------------------------------------------

_NOW = datetime.datetime(2026, 10, 12, tzinfo=datetime.UTC)


class _FakeLangfuse:
    """Langfuse as `langfuse_store` drives it, in memory, never on the network.

    `run_experiment` runs the task and evaluators on each item as the SDK does and keeps one
    item run per item: its output serialised, its metadata the item's and the experiment's
    with every value a string, and its scores. `api.experiments.list_items` reads item runs
    back newest first, two to a page, and shows at most one item run per dataset item in an
    experiment, as Langfuse does. The next push loses its first `lose_runs` item runs and
    the scores of the `lose_scores` after them, as a failed send would.
    """

    def __init__(self, *, project: str = langfuse_store.PROJECT_ID) -> None:
        self.datasets: dict[str, Any] = {}
        self.items: dict[str, DatasetItem] = {}
        self.runs: list[dict[str, Any]] = []
        self.lose_runs = 0
        self.lose_scores = 0
        self.api = SimpleNamespace(
            projects=SimpleNamespace(
                get=lambda: SimpleNamespace(data=[SimpleNamespace(id=project)])
            ),
            datasets=SimpleNamespace(get=self._dataset),
            experiments=SimpleNamespace(list_items=self._list_items),
        )

    def _dataset(self, *, dataset_name: str) -> Any:
        if dataset_name not in self.datasets:
            raise NotFoundError(body=dataset_name)
        return self.datasets[dataset_name]

    def get_dataset(self, name: str) -> Any:
        dataset = self._dataset(dataset_name=name)
        return SimpleNamespace(
            items=[item for item in self.items.values() if item.dataset_id == dataset.id]
        )

    def create_dataset(self, *, name: str, description: str) -> None:
        del description
        self.datasets[name] = SimpleNamespace(id=f"dataset-{name}", created_at=_NOW)

    def create_dataset_item(
        self, *, dataset_name: str, id: str, input: Any, metadata: Any
    ) -> DatasetItem:
        self.items[id] = DatasetItem(
            id=id,
            status="ACTIVE",
            input=input,
            expected_output=None,
            metadata=metadata,
            dataset_id=self._dataset(dataset_name=dataset_name).id,
            dataset_name=dataset_name,
            created_at=_NOW,
            updated_at=_NOW,
            media_references=[],
        )
        return self.items[id]

    def run_experiment(
        self, *, run_name: str, data: list[Any], task: Any, evaluators: list[Any], **kwargs: Any
    ) -> Any:
        results = []
        for item in data:
            output = task(item=item)
            evaluations = [
                evaluation
                for evaluator in evaluators
                for evaluation in evaluator(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
            ]
            results.append(output)
            if self.lose_runs:
                self.lose_runs -= 1
                continue
            if self.lose_scores:
                self.lose_scores -= 1
                evaluations = []
            metadata = {**item.metadata, **kwargs["metadata"]}
            self.runs.append(
                {
                    "experiment": run_name,
                    "dataset": item.dataset_id,
                    "item": item.id,
                    "trace_id": f"trace-{len(self.runs)}",
                    "output": json.dumps(output),
                    "metadata": {key: str(value) for key, value in metadata.items()},
                    "scores": {e.name: e.value for e in evaluations},
                }
            )
        return SimpleNamespace(item_results=results)

    def create_score(self, *, trace_id: str, name: str, value: Any, **_: Any) -> None:
        next(r for r in self.runs if r["trace_id"] == trace_id)["scores"][name] = value

    def flush(self) -> None:
        pass

    def _list_items(
        self, *, experiment_name: str, dataset_id: str, cursor: str | None, **_: Any
    ) -> Any:
        held: dict[str, dict[str, Any]] = {}
        for r in reversed(self.runs):
            if (r["experiment"], r["dataset"]) == (experiment_name, dataset_id):
                held.setdefault(r["item"], r)
        shown = list(held.values())
        start = int(cursor or 0)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id=r["trace_id"],
                    trace_id=r["trace_id"],
                    output=r["output"],
                    metadata=r["metadata"],
                    scores=[SimpleNamespace(name=name) for name in r["scores"]],
                )
                for r in shown[start : start + 2]
            ],
            meta=SimpleNamespace(cursor=str(start + 2) if start + 2 < len(shown) else None),
        )


_TOTAL, _GROWTH, _REFUSAL = (
    "aggregate-total-oda-kenya",
    "growth-gdp-kenya",
    "sum-net-flows-aid-debt-ghana",
)


def _stored(arm: str, task: str, rep: int, **fields: Any) -> dict[str, Any]:
    return {**_record(arm, task, rep, **fields), "schema": entry.RECORD_SCHEMA}


def _run_directory(tmp_path: Path) -> Path:
    """A baseline run over a task, a refusal task and a third, two repetitions each: one
    run retried after an error, one whose usage is unknown, one that missed its recording."""
    refused = ("refused", "combination_unasked")
    records = [
        _stored("baseline", _TOTAL, 1, kinds=[], status="error", stop="error"),
        _stored("baseline", _TOTAL, 1, kinds=["aggregate", "dataset"]),
        _stored("baseline", _TOTAL, 2, kinds=["aggregate", "dataset"], finished=False),
        _stored("baseline", _GROWTH, 1, kinds=["dataset", "growth"]),
        _stored("baseline", _GROWTH, 2, kinds=["dataset"], wall_s=12.0),
        _stored("baseline", _REFUSAL, 1, kinds=["dataset"], first=refused, stop="asked_user"),
        _stored("baseline", _REFUSAL, 2, kinds=["dataset"], misses=("get_skill dfo-faq",)),
    ]
    directory = tmp_path / "20261012T000000Z-baseline"
    directory.mkdir()
    summarise.write_records(directory / "records.jsonl", records)
    return directory


def _name(directory: Path) -> str:
    records = summarise.read_records([directory / "records.jsonl"])
    return langfuse_store.experiment_name(directory.name, records)


def test_a_baseline_pushed_to_langfuse_pairs_as_its_records_file_does(tmp_path: Path) -> None:
    tasks = entry.load_tasks()
    directory = _run_directory(tmp_path)
    fake = _FakeLangfuse()

    names = langfuse_store.push(fake, directory, tasks=tasks, wait_s=0)
    langfuse_store.push(fake, directory, tasks=tasks, wait_s=0)

    by_id = {task.id: task for task in tasks}
    digest = langfuse_store.digest(summarise.read_records([directory / "records.jsonl"]))
    assert names == [f"{directory.name}/baseline r e {digest[:12]}"]
    assert {item.id: item.metadata["inputs_digest"] for item in fake.items.values()} == {
        langfuse_store.item_id(t): entry.inputs_digest(by_id[t], entry.load_recording(t))
        for t in (_TOTAL, _GROWTH, _REFUSAL)
    }
    # One item run per task, the second push sending none again, scored over each
    # repetition's last run that is not an error. Outcomes leave out the run that missed
    # its recording, tokens the run whose usage is unknown, and a refusal task is scored
    # on its ending alone.
    tokens = {"input_tokens": 100, "cached_tokens": 50, "output_tokens": 10, "thinking_tokens": 0}
    assert {json.loads(run["output"])["task"]: run["scores"] for run in fake.runs} == {
        _TOTAL: {
            "runs_scored": 2,
            "runs_left_out": 0,
            "first_proposal_accepted": 1,
            "kind_pass": 1,
            "model_calls": 1,
            "wall_s": 5,
            **tokens,
        },
        _GROWTH: {
            "runs_scored": 2,
            "runs_left_out": 0,
            "first_proposal_accepted": 1,
            "kind_pass": 0.5,
            "model_calls": 1,
            "wall_s": 8.5,
            **tokens,
        },
        _REFUSAL: {
            "runs_scored": 1,
            "runs_left_out": 1,
            "expected_ending": 1,
            "model_calls": 1,
            "wall_s": 5,
            **tokens,
        },
    }

    replica = [
        _stored("baseline-replica", task, rep, kinds=kinds)
        for task, kinds in ((_TOTAL, ["aggregate", "dataset"]), (_GROWTH, ["dataset"]))
        for rep in (1, 2)
    ]

    def paired(**source: Any) -> tuple[str, dict[str, Any]]:
        args = argparse.Namespace(
            **{"baseline_records": None, "baseline_langfuse": None, **source},
            baseline="baseline",
            out=tmp_path,
        )
        baseline = run.stored_baseline(args, fake)
        return summarise.build_report(
            replica + baseline, tasks, arm="baseline-replica", baseline="baseline"
        )

    from_file = paired(baseline_records=directory / "records.jsonl")
    assert paired(baseline_langfuse=names[0]) == from_file
    assert from_file[1]["attempts"] == {"baseline": 7, "baseline-replica": 4}
    assert summarise.read_records([tmp_path / "baseline-records.jsonl"]) == summarise.read_records(
        [directory / "records.jsonl"]
    )


def test_a_push_langfuse_shows_only_in_part_fails_and_pushing_again_completes_it(
    tmp_path: Path,
) -> None:
    tasks = entry.load_tasks()
    directory = _run_directory(tmp_path)
    records = summarise.read_records([directory / "records.jsonl"])
    fake = _FakeLangfuse()
    fake.lose_runs, fake.lose_scores = 1, 1

    with pytest.raises(SystemExit, match="shows 1 of the 3 tasks"):
        langfuse_store.push(fake, directory, tasks=tasks, wait_s=0)
    (name,) = langfuse_store.push(fake, directory, tasks=tasks, wait_s=0)

    assert len(fake.runs) == 3
    assert all(run["scores"] for run in fake.runs)
    assert langfuse_store.pull(fake, name) == records

    # A directory changed after its push, as by a resume, goes up as a new experiment.
    summarise.write_records(directory / "records.jsonl", records[1:])
    (resumed,) = langfuse_store.push(fake, directory, tasks=tasks, wait_s=0)

    assert resumed != name
    assert langfuse_store.pull(fake, name) == records
    assert langfuse_store.pull(fake, resumed) == records[1:]


def _without_a_secret(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del fake, directory
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    langfuse_store.connect()


def _in_another_project(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del fake, directory, monkeypatch
    langfuse_store.check_project(_FakeLangfuse(project="cm-another"))


def _pushed(fake: _FakeLangfuse, directory: Path) -> str:
    (name,) = langfuse_store.push(fake, directory, tasks=entry.load_tasks(), wait_s=0)
    return name


def _edited_in_langfuse(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del monkeypatch
    name = _pushed(fake, directory)
    fake.runs[1]["output"] = fake.runs[1]["output"].replace('"wall_s": 5.0', '"wall_s": 4.0')
    langfuse_store.pull(fake, name)


def _shown_in_part(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del monkeypatch
    fake.lose_runs = 1
    with contextlib.suppress(SystemExit):
        _pushed(fake, directory)
    langfuse_store.pull(fake, _name(directory))


def _replace_a_task(fake: _FakeLangfuse, directory: Path) -> str:
    """Push `directory`, then replace a task's item run with records it never held, under
    their own digest."""
    name = _pushed(fake, directory)
    output = json.loads(fake.runs[1]["output"])
    output["records"][0]["wall_s"] = 4.0
    fake.runs[1]["output"] = json.dumps(output)
    fake.runs[1]["metadata"]["task_digest"] = langfuse_store.digest(output)
    return name


def _replaced_in_langfuse(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del monkeypatch
    langfuse_store.pull(fake, _replace_a_task(fake, directory))


def _replaced_then_pushed_again(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del monkeypatch
    _replace_a_task(fake, directory)
    _pushed(fake, directory)


def _never_pushed(fake: _FakeLangfuse, directory: Path, monkeypatch: Any) -> None:
    del monkeypatch
    _pushed(fake, directory)
    langfuse_store.pull(fake, _name(directory).replace("/baseline ", "/baseline-replica "))


@pytest.mark.parametrize(
    ("act", "refusal"),
    [
        (_without_a_secret, "needs LANGFUSE_SECRET_KEY"),
        (_in_another_project, "name project cm-another"),
        (_edited_in_langfuse, "does not hold the records pushed with it"),
        (_shown_in_part, "shows 2 of the 3 tasks"),
        (_replaced_in_langfuse, "do not match the runs pushed"),
        (_replaced_then_pushed_again, "that the run directory lacks"),
        (_never_pushed, "holds no runs"),
    ],
)
def test_langfuse_refuses_what_it_cannot_vouch_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    act: Callable[[_FakeLangfuse, Path, Any], None],
    refusal: str,
) -> None:
    with pytest.raises(SystemExit, match=refusal):
        act(_FakeLangfuse(), _run_directory(tmp_path), monkeypatch)
