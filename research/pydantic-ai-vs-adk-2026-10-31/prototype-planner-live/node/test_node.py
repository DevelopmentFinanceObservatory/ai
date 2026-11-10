"""The planner on Pydantic AI inside real Studio turns, against dev's ADK planner.

Each turn runs the real root workflow (`create_root_workflow`) behind the real Studio routes
with the orchestrator's model scripted to a planner-entry task's recorded `search` and `plan`
calls, the real `PlannerSeam`, and a scripted google-genai client answering the planner's
requests. The `adk` arm is dev's node as it stands; the `pai` arm differs only in the planner
the seam awaits (`pai_node.PaiPlannerSeam`). Each test runs one script on both arms and
compares them; each has a control, an arm broken in one place, which the same check must
refuse.
"""

from __future__ import annotations

import asyncio
import copy
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pai_node
import pytest
import scripted as sc
from entry import quiet_logs
from executor.classify import ToolFailure
from executor.outcomes import RediscoveryNeeded
from google.adk import Event
from google.adk.events import EventActions
from harness.graph import workflow as workflow_module
from harness.graph.execute import _planner_refusal
from harness.graph.state import AGENT_NAME as GRAPH_NAME
from harness.graph.state import ORCHESTRATOR_PLAN_RECORDS_KEY, readable_plan
from harness.studio_translate import _WORKING_ON_PLAN, _node_name, _Translator
from harness.turn_events import PLANNER_ASK
from harness.wire import ActivityStep, StepStatus
from parity import WIRE, shape
from planner.agent import AGENT_NAME, _session_steps
from planner.call import CallSession, ResponseCall, Step, prices_chosen
from planner.plan import PLAN_STATE_KEY
from planner.workflow import ToolId
from scenes import (
    INTERRUPT,
    TURN,
    differences,
    events,
    frame,
    interrupt_body,
    stable,
)
from streaming import AUTH, Timed, read_timed, streaming_client
from studio_arms import TASKS, Arm, Built, Gate, build_arm, orchestrator_script
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION
from tests.orchestrator_fakes import CONVERSATION, stream, studio_client, turn_body

quiet_logs()

PLAIN = TASKS["aggregate-average-oda-kenya"]
WITH_ASK = TASKS["sum-net-flows-aid-debt-ghana"]
ARMS: list[Arm] = ["adk", "pai"]


def propose(task: Any) -> tuple[str, dict[str, Any]]:
    return ("propose_plan", dict(task.reference_payload))


def ask(**arguments: Any) -> tuple[str, dict[str, Any]]:
    return (PLANNER_ASK, arguments)


ASK_SCRIPT = [
    [propose(WITH_ASK)],
    [ask(message="Do these add up?", options=[dict(o) for o in COMBINATION_OPTIONS])],
    [propose(WITH_ASK)],
]
ANSWER = {"q1": {"kind": "option", "option_id": COMBINE_OPTION}}
TEXT = {"q1": {"kind": "text", "text": "Since 2020"}}
# A patch of version 1 that restates nothing: ADK accepts it as version 2.
PATCH = ("propose_plan", {"amends_version": 1})
CANCEL = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}


def assert_same_requests(adk: Built, pai: Built) -> None:
    """Both planners sent the same requests, and each differs by exactly `parity.WIRE`, what
    `GoogleModel` writes differently from ADK's `Gemini` for the same request."""
    assert len(adk.client.calls) == len(pai.client.calls)
    for index, (a, b) in enumerate(zip(adk.client.calls, pai.client.calls, strict=True)):
        assert shape(sc.request_differences(a, b)) == WIRE, f"request {index}"


type Segments = list[list[dict[str, Any]]]


async def turn(
    arm: Arm,
    task: Any,
    steps: Any,
    *,
    pai: dict[str, Any] | None = None,
    after: tuple[tuple[str, Any], ...] = (),
    **build: Any,
) -> tuple[Built, Segments]:
    """Run `task` on `arm`: the turn, then each of `after` against the pause the last
    segment ended on. Each is `("answer", answers)`, `("amend", words)`, `("approve", {})` or
    `("switch", {"row": choice id, "label": the option's label})`.
    Returns the arm and every segment's frames."""
    built = await build_arm(arm, task, steps, pai=pai, **build)
    segments: Segments = []
    async with studio_client(built.app) as client:
        segments.append(await stream(client, TURN, turn_body(task.question)))
        for decision, payload in after:
            last = segments[-1]
            if decision == "answer":
                pause = frame(last, "elicitation.request")
                body = interrupt_body(pause["interrupt_id"], "answer", answers=payload)
            elif decision == "switch":
                pause = frame(last, "plan.proposed")
                row = next(d for d in pause["decisions"] if d.get("choice_id") == payload["row"])
                option = next(o for o in row["options"] if o["label"] == payload["label"])
                body = interrupt_body(
                    pause["interrupt_id"],
                    "switch",
                    choice_id=payload["row"],
                    option_id=option["option_id"],
                    plan_revision=pause["revision"],
                )
            else:
                pause = frame(last, "plan.proposed")
                body = interrupt_body(
                    pause["interrupt_id"], decision, **({"payload": payload} if payload else {})
                )
            segments.append(await stream(client, INTERRUPT, body))
    return built, segments


async def both(
    task: Any,
    steps: Any,
    *,
    pai: dict[str, Any] | None = None,
    after: tuple[tuple[str, Any], ...] = (),
    **build: Any,
) -> tuple[tuple[Built, Segments], tuple[Built, Segments]]:
    """One script on dev's arm and on the Pydantic AI arm."""
    adk = await turn("adk", task, steps, after=after, **build)
    paid = await turn("pai", task, steps, pai=pai, after=after, **build)
    return adk, paid


def assert_same_frames(adk: Segments, pai: Segments) -> None:
    assert [events(s) for s in adk] == [events(s) for s in pai]
    assert differences(stable(adk), stable(pai)) == []


# --- Pass line 2: the frames of a planned turn equal dev's ------------------------------


async def test_the_frames_of_a_planned_turn_equal_devs() -> None:
    (adk_built, adk), (pai_built, pai) = await both(PLAIN, [[propose(PLAIN)]])

    assert_same_frames(adk, pai)
    assert events(pai[0])[-1] == "plan.proposed"
    labels = [f["label"] for f in pai[0] if f["event"] == "activity.step"]
    assert "Checking the plan against the data" in labels
    assert_same_requests(adk_built, pai_built)


async def test_control_a_planner_that_yields_no_events_misses_the_checking_line() -> None:
    (_, adk), (_, pai) = await both(PLAIN, [[propose(PLAIN)]], pai={"events": False})

    with pytest.raises(AssertionError):
        assert_same_frames(adk, pai)
    shown = {f["label"] for s in (adk, pai) for f in s[0] if f["event"] == "activity.step"}
    missing = shown - {f["label"] for f in pai[0] if f["event"] == "activity.step"}
    assert missing == {"Checking the plan against the data"}


# --- Pass line 3: ask, pause, answer, resume, plan --------------------------------------


def assert_same_ask_turn(adk: tuple[Built, Segments], pai: tuple[Built, Segments]) -> None:
    assert_same_frames(adk[1], pai[1])
    assert events(pai[1][0])[-1] == "elicitation.request"
    assert events(pai[1][1])[-1] == "plan.proposed"
    assert len(adk[0].client.calls) == len(pai[0].client.calls) == 3


async def test_an_ask_pauses_through_devs_interrupt_and_the_answer_resumes_the_plan() -> None:
    adk, pai = await both(WITH_ASK, ASK_SCRIPT, after=(("answer", ANSWER),))

    assert_same_ask_turn(adk, pai)
    assert_same_requests(adk[0], pai[0])
    (question,) = frame(pai[1][0], "elicitation.request")["questions"]
    assert [o["id"] for o in question["options"]] == [o["id"] for o in COMBINATION_OPTIONS]
    assert [o["label"] for o in question["options"]] == [o["label"] for o in COMBINATION_OPTIONS]
    # The resumed run's first request is the user's message and a brief carrying the
    # answer, with no call of the run that asked.
    resumed = sc.contents_list(pai[0].client.calls[2]["contents"])
    assert [c["role"] for c in resumed] == ["user", "user"]
    assert not any("function_call" in p for c in resumed for p in c["parts"])


@pytest.mark.parametrize("arm", ["adk", "pai"])
async def test_the_answer_of_an_ask_is_taken_once(arm: Arm) -> None:
    built = await build_arm(arm, WITH_ASK, ASK_SCRIPT)
    async with studio_client(built.app) as client:
        first = await stream(client, TURN, turn_body(WITH_ASK.question))
        body = interrupt_body(
            frame(first, "elicitation.request")["interrupt_id"], "answer", answers=ANSWER
        )
        await stream(client, INTERRUPT, body)
        again = await client.post(INTERRUPT, json=body, headers=AUTH)

    assert again.status_code == 404


async def test_control_a_planner_that_reads_no_earlier_step_loses_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pai_node.PaiPlannerSeam, "_earlier", lambda self, ctx: [Step("message")])
    adk, pai = await both(WITH_ASK, ASK_SCRIPT, after=(("answer", ANSWER),))

    # The resumed proposal needs the user's pick of the combination, which no step shows.
    with pytest.raises(AssertionError):
        assert_same_ask_turn(adk, pai)
    assert "plan.proposed" not in events(pai[1][1])
    assert len(pai[0].client.calls) == 4


# --- Pass line 4: an amendment, and an ask during one ------------------------------------


def amend_after(*more: tuple[str, Any]) -> tuple[tuple[str, Any], ...]:
    return (("amend", "only 2020"), *more)


async def test_an_amendment_drafts_version_two_as_dev_does() -> None:
    adk, pai = await both(PLAIN, [[propose(PLAIN)], [PATCH]], after=amend_after())

    assert_same_frames(adk[1], pai[1])
    assert [frame(s, "plan.proposed")["version"] for s in pai[1]] == [1, 2]
    assert_same_requests(adk[0], pai[0])
    # The second request is the amendment's: its brief carries the user's words.
    assert "only 2020" in str(sc.contents_list(pai[0].client.calls[1]["contents"]))


async def test_control_an_amendment_that_never_reaches_the_planner_changes_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pai_node.PaiPlannerSeam.__call__

    async def dropped(self: Any, *, ctx: Any, **kwargs: Any) -> Any:
        return await original(self, ctx=ctx, **{**kwargs, "amendment": None})

    monkeypatch.setattr(pai_node.PaiPlannerSeam, "__call__", dropped)
    adk, pai = await both(PLAIN, [[propose(PLAIN)], [PATCH]], after=amend_after())

    with pytest.raises(AssertionError):
        assert_same_requests(adk[0], pai[0])
    assert "only 2020" not in str(sc.contents_list(pai[0].client.calls[1]["contents"]))


AMEND_ASK_SCRIPT = [[propose(PLAIN)], [ask(message="Which years do you mean?")], [PATCH]]


async def test_an_ask_during_an_amendment_pauses_and_its_answer_drafts_version_two() -> None:
    adk, pai = await both(PLAIN, AMEND_ASK_SCRIPT, after=amend_after(("answer", TEXT)))

    assert_same_frames(adk[1], pai[1])
    assert events(pai[1][1])[-1] == "elicitation.request"
    assert "plan.proposed" not in events(pai[1][1])
    assert [frame(s, "plan.proposed")["version"] for s in (pai[1][0], pai[1][2])] == [1, 2]
    assert_same_requests(adk[0], pai[0])


async def test_control_an_ask_during_an_amendment_that_hides_the_resume_loses_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pai_node.PaiCtx, "resume_inputs", {}, raising=False)
    adk, pai = await both(PLAIN, AMEND_ASK_SCRIPT, after=amend_after(("answer", TEXT)))

    # The seam sees a fresh entry, so the resumed run's brief lacks the user's answer.
    with pytest.raises(AssertionError):
        assert_same_requests(adk[0], pai[0])


# --- Pass line 1: progress frames reach the client while the planner runs ----------------

GET_SKILL = ("get_skill", {"skill_id": "oda-dac2a"})


def show_planner_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render each progress report of the planner as a running status frame, which dev's
    translator does not (`harness.studio_translate._plan_path_frames`: the seam's late
    reports "add no frame"). Both arms run with it, so a frame's time says only when the
    report reached the stream."""
    original = _Translator._plan_path_frames

    def plan_path_frames(self: _Translator, event: Any) -> list[Any]:
        progress = (event.custom_metadata or {}).get("progress")
        if _node_name(event) not in {"planner", AGENT_NAME} or not isinstance(progress, dict):
            return original(self, event)
        frames: list[Any] = self._open_status(_WORKING_ON_PLAN)
        assert self._status is not None
        step_id, label, _ = self._status
        frames.append(
            ActivityStep(
                event="activity.step",
                seq=self._conversation.next_seq(),
                step_id=step_id,
                label=label,
                status=StepStatus.RUNNING,
                detail=f"called {progress['tools'][0]}",
            )
        )
        return frames

    monkeypatch.setattr(_Translator, "_plan_path_frames", plan_path_frames)


def is_progress(frame_: dict[str, Any]) -> bool:
    return frame_["event"] == "activity.step" and str(frame_.get("detail", "")).startswith(
        "called "
    )


def is_checking(frame_: dict[str, Any]) -> bool:
    return frame_.get("label") == "Checking the plan against the data"


async def first_frames(arm: Arm, *, pai: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a turn whose planner reads a skill page and then proposes, over the streaming
    client. The translation inside `propose_plan` waits until the client has read a progress
    frame, or 3 s. Returns the seconds at which the first progress frame and the first
    "checking" frame arrived, whether `propose_plan` had already waited out the gate, and
    the frames."""
    gate = Gate(event=asyncio.Event(), timeout_s=3.0)
    built = await build_arm(arm, PLAIN, [[GET_SKILL], [propose(PLAIN)]], gate=gate, pai=pai)
    timed = Timed()
    seen: dict[str, Any] = {}

    def on_frame(parsed: dict[str, Any]) -> None:
        if is_progress(parsed) and "progress" not in seen:
            seen["progress"] = parsed
            gate.event.set()

    started = time.monotonic()
    async with (
        streaming_client(built.app) as client,
        client.stream("POST", TURN, json=turn_body(PLAIN.question), headers=AUTH) as response,
    ):
        assert response.status_code == 200
        await asyncio.wait_for(read_timed(response, timed, started, on_frame=on_frame), 15)
    checking = next((i for i, f in enumerate(timed.frames) if is_checking(f)), None)
    at = {
        "progress": timed.at[timed.frames.index(seen["progress"])],
        "checking": None if checking is None else timed.at[checking],
    }
    return {"at": at, "timed_out": gate.timed_out, "frames": timed.frames}


async def test_progress_frames_reach_the_client_while_the_planner_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show_planner_progress(monkeypatch)
    adk = await first_frames("adk")
    pai = await first_frames("pai")
    print(
        f"first progress frame: dev {adk['at']['progress']:.3f} s, pai {pai['at']['progress']:.3f} s"
    )
    print(
        f"first checking frame: dev {adk['at']['checking']:.3f} s, pai {pai['at']['checking']:.3f} s"
    )

    # The report of the skill read reaches the client before the proposal's translation
    # ends, and releases it. Dev's node holds the reports until its draft is done.
    assert not pai["timed_out"]
    assert pai["at"]["progress"] < 1.0
    assert adk["timed_out"]
    assert adk["at"]["progress"] >= 3.0
    # The "checking" line opens as the call is made on both arms: the translator reads the
    # planner's own call event, which the Pydantic AI arm yields as ADK does.
    assert pai["at"]["checking"] < 1.0
    assert adk["at"]["checking"] < 1.0
    assert [f["detail"] for f in pai["frames"] if is_progress(f)] == [
        "called get_skill",
        "called propose_plan",
    ]


async def test_control_progress_reports_held_until_the_draft_is_done_reach_the_client_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show_planner_progress(monkeypatch)
    held = await first_frames("pai", pai={"events": False})

    # The planner's reports go to dev's own queue, which the node relays after the draft.
    assert held["timed_out"]
    assert held["at"]["progress"] >= 3.0


# --- Pass line 7: Stop mid-planning -------------------------------------------------------


class OrphanRunner(pai_node.FeedingRunner):
    """The agent loop run in a task of its own that nothing cancels."""

    async def events(self, brief: str) -> AsyncIterator[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def run() -> None:
            async for event in pai_node.FeedingRunner.events(self, brief):
                queue.put_nowait(event)
            queue.put_nowait(None)

        self.orphan = asyncio.ensure_future(run())
        while (event := await queue.get()) is not None:
            yield event


def running() -> set[asyncio.Task[Any]]:
    return {t for t in asyncio.all_tasks() if not t.done()}


async def stop_mid_planning(arm: Arm, *, pai: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cancel a turn while `propose_plan` is in its translation. Returns what the stream
    ended on, the tasks left running, and what the next turn on the same app did."""
    before = running()
    gate = Gate(event=asyncio.Event(), timeout_s=5.0)
    built = await build_arm(
        arm,
        PLAIN,
        [[propose(PLAIN)], [propose(PLAIN)]],
        gate=gate,
        pai=pai,
        extra_orchestrator=orchestrator_script(PLAIN),
    )
    async with studio_client(built.app) as client:
        turn_ = asyncio.ensure_future(stream(client, TURN, turn_body(PLAIN.question)))
        while not gate.reached:
            await asyncio.sleep(0.01)
        cancel = await client.post("/api/studio/cancel", json=CANCEL, headers=AUTH)
        stopped = await asyncio.wait_for(turn_, 5)
        await asyncio.sleep(0.1)
        left = running() - before - {asyncio.current_task()}
        after = await stream(client, TURN, turn_body("And again?"))
    for task in left:
        task.cancel()
    await asyncio.gather(*left, return_exceptions=True)
    return {"cancel": cancel.status_code, "stopped": stopped, "left": left, "after": after}


def assert_stopped_cleanly(result: dict[str, Any]) -> None:
    assert result["cancel"] == 204
    assert events(result["stopped"])[-1] == "error"
    assert frame(result["stopped"], "error")["error"]["error"] == "cancelled"
    assert not result["left"], [t.get_coro() for t in result["left"]]
    assert events(result["after"])[-1] == "plan.proposed"


async def test_stop_mid_planning_ends_the_turn_as_dev_does_and_leaves_no_task() -> None:
    adk = await stop_mid_planning("adk")
    pai = await stop_mid_planning("pai")

    assert_stopped_cleanly(adk)
    assert_stopped_cleanly(pai)
    assert events(pai["stopped"]) == events(adk["stopped"])


async def test_control_a_planner_run_in_a_task_nothing_cancels_outlives_the_stop() -> None:
    orphaned = await stop_mid_planning("pai", pai={"runner": OrphanRunner})

    with pytest.raises(AssertionError):
        assert_stopped_cleanly(orphaned)
    assert orphaned["left"]


# --- Pass line 8: the records a turn leaves equal dev's ----------------------------------

SKILL_ASK_SCRIPT = [[GET_SKILL], [ask(message="Which years do you mean?")], [propose(PLAIN)]]
HEX_ID = re.compile(r"[0-9a-f]{32}")
DISCOVERY = "translator:discovery"
LOCK_TOKEN = "translator:discovery_lock_token"


def comparable(value: Any) -> Any:
    """`stable(value)` without the ids a run mints in the translation's discovery log, whose
    records are keyed by one each and carry it as `id`, and the lock token.

    `stable` drops `plan_version_id`, `supersedes` and `interrupt_id` at every depth: each is
    a `_wire_id` (`harness.studio_state`), a fresh uuid per run, and `supersedes` holds the
    previous `plan_version_id`. Every other `id`, a requirement's or a step's, is compared.
    """
    return _strip_run_ids(stable(value))


def _strip_run_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _discovery(v)
            if k == DISCOVERY
            else bool(v)
            if k == LOCK_TOKEN
            else _strip_run_ids(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_strip_run_ids(v) for v in value]
    return value


def _discovery(log: Any) -> Any:
    """The discovery log with its `records` (keyed by an id minted per run, which each also
    carries as `id`) as a sorted list of the records without it."""
    records = log.get("records") if isinstance(log, dict) else None
    if not (isinstance(records, dict) and all(HEX_ID.fullmatch(key) for key in records)):
        return _strip_run_ids(log)
    stripped = [
        _strip_run_ids({k: v for k, v in record.items() if k != "id"})
        for record in records.values()
    ]
    stripped.sort(key=lambda record: str(sorted(map(str, record))))
    return {**_strip_run_ids({k: v for k, v in log.items() if k != "records"}), "records": stripped}


async def stored(
    arm: Arm,
    task: Any,
    steps: Any,
    after: tuple[tuple[str, Any], ...],
    *,
    pai: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a turn through `after` and return what the session, the Studio store and the run
    ledger hold."""
    built, segments = await turn(arm, task, steps, pai=pai, after=after)
    sessions = built.sessions
    (principal,) = sessions.sessions[GRAPH_NAME]
    conversation = await built.app.state.studio.store.load(principal, CONVERSATION)
    (session,) = [
        s
        for by_user in sessions.sessions[GRAPH_NAME].values()
        for s in by_user.values()
        if s.id.startswith(CONVERSATION + "#")
    ]
    state = dict(session.state)
    return {
        "segments": segments,
        "state": state,
        "plan": state.get("plan"),
        "plan_records": state.get(ORCHESTRATOR_PLAN_RECORDS_KEY),
        "approvals": [a.model_dump(mode="json") for a in conversation.approvals],
        "turns": list(conversation.turns),
        "ledger": {
            scope: {id_: run.model_dump(mode="json") for id_, run in book.runs.items()}
            for scope, book in built.ledger._books.items()
        },
    }


def assert_same_records(adk: dict[str, Any], pai: dict[str, Any]) -> None:
    """The records equal dev's."""
    assert set(adk["state"]) == set(pai["state"])
    assert differences(comparable(adk["state"]), comparable(pai["state"])) == []
    for key in ("plan", "plan_records", "approvals", "turns", "ledger"):
        assert differences(comparable(adk[key]), comparable(pai[key])) == [], key


APPROVED = (("answer", TEXT), ("approve", {}))


async def test_the_records_a_turn_leaves_equal_devs() -> None:
    adk = await stored("adk", PLAIN, SKILL_ASK_SCRIPT, APPROVED)
    pai = await stored("pai", PLAIN, SKILL_ASK_SCRIPT, APPROVED)

    assert_same_records(adk, pai)
    assert events(pai["segments"][-1])[0] == "plan.approved"
    # A turn this small runs no plan step: the recorded plan has no surface to run on, so
    # the ledger and the plan records are empty on both arms.
    assert adk["ledger"] == pai["ledger"]


def plan_only(response_events: Any) -> Any:
    """`pai_node.response_events` whose events carry no written key but the plan."""

    def narrowed(*args: Any) -> list[Event]:
        events_ = response_events(*args)
        for event in events_:
            assert event.actions is not None
            delta = event.actions.state_delta
            event.actions = EventActions(
                state_delta={k: v for k, v in delta.items() if k == PLAN_STATE_KEY}
            )
        return events_

    return narrowed


async def test_control_with_only_the_seams_own_list_durable_keys_a_tool_writes_are_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk = await stored("adk", WITH_ASK, ASK_SCRIPT, (("answer", ANSWER),))
    monkeypatch.setattr(pai_node, "response_events", plan_only(pai_node.response_events))
    pai = await stored("pai", WITH_ASK, ASK_SCRIPT, (("answer", ANSWER),))

    with pytest.raises(AssertionError):
        assert_same_records(adk, pai)
    # The first proposal wrote keys the seam's own list lacks (`PlannerSeam._persist_durable`),
    # which ADK carries on the proposal's response event. The stored session misses them.
    lost = set(adk["state"]) - set(pai["state"])
    assert {"translator:referral", "translator:discovery", "planner:pin_verdicts"} <= lost


async def test_control_a_skill_page_read_before_an_ask_is_lost_on_the_resume_without_the_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk = await stored("adk", PLAIN, SKILL_ASK_SCRIPT, (("answer", TEXT),))
    monkeypatch.setattr(pai_node, "response_events", plan_only(pai_node.response_events))
    pai = await stored("pai", PLAIN, SKILL_ASK_SCRIPT, (("answer", TEXT),))

    # The brief of the resumed run lists the page, whose text lives under a key the seam does
    # not carry (`planner.skills.skill_page_key`), so the entry raises and the turn fails.
    assert events(adk["segments"][-1])[-1] == "plan.proposed"
    assert events(pai["segments"][-1])[-1] == "error"


# --- Pass line 6: a pick's replan, and a pick after a run that asked ---------------------

RANK = TASKS["rank-top-donors-malawi"]
PICK = ("switch", {"row": "decision/top_n", "label": "5"})


def pick_patch() -> tuple[str, dict[str, Any]]:
    """The patch a model drafts for the pick of five donors: the decision and the rank
    requirement that states it."""
    proposal = copy.deepcopy(dict(RANK.reference_payload))
    top_n = next(d for d in proposal["decisions"] if d["slot"] == "top_n")
    top_n.update(chosen="5", resolved_by="user", rationale="The user picked five.")
    rank = next(r for r in proposal["evidence_requirements"] if r["kind"] == "rank")
    rank["limit"] = 5
    return (
        "propose_plan",
        {"amends_version": 1, "decisions": [top_n], "evidence_requirements": [rank]},
    )


ASKED_FIRST = [[ask(message="Which period do you mean?")], [propose(RANK)], [pick_patch()]]
ASKED_AFTER = (("answer", TEXT), PICK)
NOT_ASKED = [[propose(RANK)], [pick_patch()]]


def assert_picked(adk: tuple[Built, Segments], pai: tuple[Built, Segments]) -> None:
    """Both arms replanned the pick the same way: the same frames and requests, and the
    card of the replan states the pick."""
    assert_same_frames(adk[1], pai[1])
    assert_same_requests(adk[0], pai[0])
    card = frame(pai[1][-1], "plan.proposed")
    assert card["version"] == 2
    assert {d["id"]: d["chosen"] for d in card["decisions"]}["top_n"] == "5"


@pytest.mark.parametrize(
    ("steps", "after"),
    [
        pytest.param(NOT_ASKED, (PICK,), id="no ask"),
        pytest.param(ASKED_FIRST, ASKED_AFTER, id="after an ask"),
    ],
)
async def test_a_pick_replans_the_plan_as_dev_does(steps: Any, after: Any) -> None:
    adk, pai = await both(RANK, steps, after=after)

    assert_picked(adk, pai)


async def test_control_a_pick_that_never_reaches_the_planner_is_not_replanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pai_node.PaiPlannerSeam.__call__

    async def dropped(self: Any, *, ctx: Any, **kwargs: Any) -> Any:
        return await original(self, ctx=ctx, **{**kwargs, "pick": None})

    monkeypatch.setattr(pai_node.PaiPlannerSeam, "__call__", dropped)
    adk, pai = await both(RANK, NOT_ASKED, after=(PICK,))

    with pytest.raises(AssertionError):
        assert_picked(adk, pai)


# `PlannerSeam.revise`, which the `switch` node calls for a scope choice, reads the steps of
# its call through `AdkCall`, that is `planner.agent._session_steps` over the session's events.


def session_of(built: Built) -> Any:
    (session,) = [
        s
        for by_user in built.sessions.sessions[GRAPH_NAME].values()
        for s in by_user.values()
        if s.id.startswith(CONVERSATION + "#")
    ]
    return session


@dataclass(frozen=True)
class SessionCall:
    """The `PlannerCall` `revise` reads, over a finished session's events."""

    steps_: tuple[Step, ...]
    state: dict[str, Any]
    call_id: str | None = None

    def steps(self) -> tuple[Step, ...]:
        return self.steps_

    def response_calls(self) -> tuple[ResponseCall, ...]:
        return ()


def revision_reads(built: Built) -> dict[str, Any]:
    """What a revision after this turn reads of the session: its steps, without the ids a
    framework mints for a call, and each reader's answer over them."""
    session = session_of(built)
    steps = tuple(_session_steps(session.events))
    reading = CallSession(SessionCall(steps, dict(session.state)))
    return {
        "steps": [(s.kind, s.name, s.args, dict(s.body)) for s in steps],
        "elected": reading.elected_spec_ids(),
        "combinations": reading.combination_picks(),
        "since_referral": reading.answers_since_referral(),
        "prices": prices_chosen(SessionCall(steps, dict(session.state))),
    }


async def test_a_revision_after_a_run_that_asked_reads_what_it_reads_on_dev() -> None:
    adk, pai = await both(WITH_ASK, ASK_SCRIPT, after=(("answer", ANSWER),))

    reads = revision_reads(pai[0])
    assert reads == revision_reads(adk[0])
    assert reads["combinations"][0], "the user's pick to add the two flows is read"


async def test_control_with_only_the_proposals_in_the_session_a_revision_misses_the_pick() -> None:
    adk, pai = await both(
        WITH_ASK,
        ASK_SCRIPT,
        pai={"feed_tools": frozenset({"propose_plan"})},
        after=(("answer", ANSWER),),
    )

    # The ask's own call is not in the session, so the interrupt's call stands in for it,
    # with the ask's arguments one level down, where the combination reader does not look.
    assert revision_reads(pai[0]) != revision_reads(adk[0])
    assert revision_reads(pai[0])["combinations"] == ({}, False)
    assert revision_reads(adk[0])["combinations"][0]


# --- Pass line 5: a refusal replan routed back from `execute` -----------------------------


# The source the stub's run stopped on. It is none the plan pins, so the replan may keep the
# plan's own sources (`propose_plan` refuses a plan that pins one the executor refused).
STOPPED_SOURCE = "spec_" + "a" * 32


def replan_from_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `execute` node refuses its first run with a retrieval whose source needs finding
    again, as a run does that `execute` replans (`harness.graph.execute._planner_refusal`),
    and runs as it is on every later entry. The recorded plans have no data surface to run
    on, so no step of one fails of itself."""
    original = workflow_module.execute_node

    def factory(**kwargs: Any) -> Any:
        real = original(**kwargs)

        async def execute(ctx: Any, node_input: Any) -> AsyncIterator[Any]:
            if ctx.state.get("replans"):
                async for event in real(ctx, node_input):
                    yield event
                return
            ctx.state["replans"] = 1
            plan = readable_plan(ctx, ctx.state["plan"])
            assert plan is not None and plan.workflow is not None
            stopped = RediscoveryNeeded(
                revision="rev1",
                step_id=plan.workflow.steps[0].id,
                tool=ToolId(server="data-mcp", name="get_data"),
                requirement_id=plan.evidence_requirements[0].id,
                spec_id=STOPPED_SOURCE,
                refusal=ToolFailure(
                    code="upstream_version_changed", message="", retry_mode="refresh_then_retry"
                ),
                completed=(),
            )
            refusal = _planner_refusal(stopped, plan=plan)
            yield Event(output={"refusal": refusal, "plan_version": plan.version}, route="replan")

        return execute

    monkeypatch.setattr(workflow_module, "execute_node", factory)


async def test_a_refusal_replan_from_execute_drafts_the_next_version_as_dev_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replan_from_execute(monkeypatch)
    adk, pai = await both(PLAIN, [[propose(PLAIN)], [PATCH]], after=(("approve", {}),))

    assert_same_frames(adk[1], pai[1])
    assert_same_requests(adk[0], pai[0])
    assert [frame(s, "plan.proposed")["version"] for s in (pai[1][0], pai[1][1])] == [1, 2]
    assert pai[1][1][0]["event"] == "plan.approved"
    # The second request is the replan's: its brief carries the stop `execute` reported.
    assert "upstream_version_changed" in str(sc.contents_list(pai[0].client.calls[1]["contents"]))


async def test_control_a_replan_that_never_reaches_the_planner_changes_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replan_from_execute(monkeypatch)
    original = pai_node.PaiPlannerSeam.__call__

    async def dropped(self: Any, *, ctx: Any, **kwargs: Any) -> Any:
        return await original(self, ctx=ctx, **{**kwargs, "refusal": None})

    monkeypatch.setattr(pai_node.PaiPlannerSeam, "__call__", dropped)
    adk, pai = await both(PLAIN, [[propose(PLAIN)], [PATCH]], after=(("approve", {}),))

    with pytest.raises(AssertionError):
        assert_same_requests(adk[0], pai[0])
