"""The planner node's hard cases, each run on dev's ADK planner and on the Pydantic AI
planner in the same Studio turn, and compared.

Both arms talk to a `chunked.Chunked` client that can stall, go quiet mid-stream and refuse
a request. The Pydantic AI arm is `hardened.hard_planner`, `node/`'s planner with the fixes
the cases called for. Every case has a control: the same check on an arm with one fix off,
which must fail it.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Sequence
from typing import Any

import hardened
import pai_call
import pai_planner
import pytest
import scripted as sc
import studio_arms
from chunked import NEVER, Call, function_call, say
from common.telemetry import flush_telemetry, init_telemetry
from google.genai import types
from hard import bounds, hard_arm, phases, session_of, trace_tools
from harness.planner_seam import PlannerSeam
from harness.prefix_cache import PrefixCache
from harness.turn_budget import bounded, current_turn_budget
from planner.agent import AGENT_NAME
from scenes import TURN, differences, events
from studio_arms import Gate, build_arm
from submission.service import COMBINATION_OPTIONS
from test_node import (
    ANSWER,
    CANCEL,
    GET_SKILL,
    PLAIN,
    TEXT,
    WITH_ASK,
    ask,
    assert_same_frames,
    comparable,
    turn,
)
from tests.orchestrator_fakes import request_text, stream, studio_client, turn_body

type Arm = tuple[studio_arms.Built, list[list[dict[str, Any]]]]

CONVERSATION_KEY = "planner:conversation"


def calls(*step: tuple[str, dict[str, Any]]) -> Call:
    """A model call answering with `step`'s function calls in one chunk."""
    return say(*[function_call(name, args) for name, args in step])


def normal(value: Any) -> Any:
    """`comparable(value)` with the ids each framework mints for a call made alike."""
    value = comparable(value)
    if isinstance(value, dict):
        return {
            k: ("<call id>" if k == "call_id" else normal(v))
            for k, v in value.items()
            if not k.startswith("translator:resolved_at:")
        }
    if isinstance(value, list):
        return [normal(v) for v in value]
    return value


def state_of(built: studio_arms.Built) -> dict[str, Any]:
    """The turn's stored session state, without the Pydantic AI path's own record."""
    return {k: v for k, v in session_of(built).state.items() if k != CONVERSATION_KEY}


def told(built: studio_arms.Built) -> str:
    """What the orchestrator read of the planner's end: its last request."""
    return request_text(built.orchestrator.requests[-1])


def assert_ends_alike(adk: Arm, pai: Arm) -> None:
    """The two arms ended the turn the same way: the frames, the number of requests the
    planner sent, what the orchestrator was told and the stored session state."""
    assert_same_frames(adk[1], pai[1])
    assert len(adk[0].client.calls) == len(pai[0].client.calls)
    assert told(adk[0]) == told(pai[0])
    assert differences(normal(state_of(adk[0])), normal(state_of(pai[0]))) == []


async def pair(
    monkeypatch: pytest.MonkeyPatch,
    programme: Sequence[Call],
    *,
    settings: dict[str, Any] | None = None,
    fixes: dict[str, Any] | None = None,
    task: Any = PLAIN,
    after: tuple[tuple[str, Any], ...] = (),
) -> tuple[Arm, Arm]:
    """One programme of model calls on dev's arm and on the hardened Pydantic AI arm, under
    `settings`' request bounds."""
    with bounds(monkeypatch, **(settings or {})), hard_arm(monkeypatch, **(fixes or {})):
        adk = await turn("adk", task, tuple(programme), after=after)
        pai = await turn("pai", task, tuple(programme), after=after)
    return adk, pai


# --- 1. A budget cut after a function call was streamed ---------------------------------

PROPOSE = ("propose_plan", dict(PLAIN.reference_payload))
# The tool's own writes: its translation runs inside `propose_plan`.
TOOL_RAN = {"translator:discovery", "planner:pin_verdicts", "translator:referral"}
CUT = (Call(chunks=[[function_call(*PROPOSE)]], hang_after=True),)
MODEL_TIME = {"turn_model_time_s": 0.5}


def assert_no_tool_ran(arm: Arm) -> None:
    assert not TOOL_RAN & set(state_of(arm[0]))
    assert len(arm[0].client.calls) == 1
    assert "stopped here" in told(arm[0])


async def test_a_cut_after_a_function_call_ends_as_dev_does_and_runs_no_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, CUT, settings=MODEL_TIME)

    assert_no_tool_ran(adk)
    assert_no_tool_ran(pai)
    assert_ends_alike(adk, pai)
    assert "plan.proposed" not in events(pai[1][0])
    # The stored record holds the closing sentence and no function call.
    record = json.loads(session_of(pai[0]).state[CONVERSATION_KEY])
    assert all(part["part_kind"] != "tool-call" for m in record for part in m.get("parts", []))


async def test_control_a_client_that_passes_chunks_on_runs_the_tool_the_cut_call_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, CUT, settings=MODEL_TIME, fixes={"hold": False})

    with pytest.raises(AssertionError):
        assert_ends_alike(adk, pai)
    # `agent/`'s `PlannerClient` hands Pydantic AI the function call as it arrives.
    assert set(state_of(pai[0])) >= TOOL_RAN
    assert not TOOL_RAN & set(state_of(adk[0]))
    assert "plan.proposed" not in events(adk[1][0])


# --- 2. A first-chunk stall is resent once -----------------------------------------------

STALL = Call(chunks=[[function_call(*PROPOSE)]], first_chunk_wait_s=NEVER)
FIRST_CHUNK = {"model_first_chunk_s": 0.2, "turn_model_time_s": 1.0}


def same_request_twice(arm: Arm) -> None:
    first, second = (sc.contents_list(c["contents"]) for c in arm[0].client.calls[:2])
    assert first == second


async def test_a_first_chunk_stall_is_resent_once(monkeypatch: pytest.MonkeyPatch) -> None:
    adk, pai = await pair(monkeypatch, (STALL, calls(PROPOSE)), settings=FIRST_CHUNK)

    assert_ends_alike(adk, pai)
    assert events(pai[1][0])[-1] == "plan.proposed"
    assert len(pai[0].client.calls) == 2
    same_request_twice(adk)
    same_request_twice(pai)


async def test_a_second_stall_is_not_resent_and_the_budget_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, (STALL, STALL, calls(PROPOSE)), settings=FIRST_CHUNK)

    assert_ends_alike(adk, pai)
    assert len(pai[0].client.calls) == 2
    assert "plan.proposed" not in events(pai[1][0])
    assert "stopped here" in told(pai[0])


async def test_control_a_stall_that_may_not_be_resent_ends_the_turn_on_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_resend(request: Any, *, stream: bool) -> Any:
        budget = current_turn_budget.get()
        assert budget is not None
        budget.retry_spent = True
        return bounded(request, stream=stream)

    monkeypatch.setattr(hardened, "bounded", no_resend)
    adk, pai = await pair(monkeypatch, (STALL, calls(PROPOSE)), settings=FIRST_CHUNK)

    with pytest.raises(AssertionError):
        assert_ends_alike(adk, pai)
    assert len(adk[0].client.calls) == 2
    assert len(pai[0].client.calls) == 1


# --- 3. A prefix cache gone at stream open is forgotten, looked up again, resent once ----


def instruction_of(config: Any) -> str:
    """The text of a cache config's system instruction, whichever form it is in."""
    found = sc.jsonable(config.system_instruction)
    if isinstance(found, dict):
        return "".join(part["text"] for part in found["parts"])
    return "".join(part if isinstance(part, str) else part["text"] for part in found)


def cached_names(built: studio_arms.Built) -> list[str | None]:
    return [sc.config_dict(call["config"]).get("cached_content") for call in built.client.calls]


async def cached_arm(
    monkeypatch: pytest.MonkeyPatch, arm: str, programme: Sequence[Call]
) -> tuple[Arm, PrefixCache]:
    """One arm with a prefix cache of its own (`planner.agent` on dev's arm, `PlannerModel` on
    the Pydantic AI arm) and a model that answers `programme`."""
    cache = PrefixCache()
    with monkeypatch.context() as scope:
        scope.setattr(
            studio_arms, "PlannerSeam", functools.partial(PlannerSeam, prefix_cache=cache)
        )
        with bounds(scope), hard_arm(scope):
            built = await turn(arm, PLAIN, tuple(programme), pai={"prefix_cache": cache})
    return built, cache


@pytest.mark.parametrize("status", [403, 404])
async def test_a_gone_cache_is_forgotten_looked_up_again_and_the_request_resent_once(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    programme = (Call(refuse=status), calls(PROPOSE))
    adk, _ = await cached_arm(monkeypatch, "adk", programme)
    pai, _ = await cached_arm(monkeypatch, "pai", programme)

    assert_ends_alike(adk, pai)
    assert events(pai[1][0])[-1] == "plan.proposed"
    for arm in (adk, pai):
        # Created, refused, created again, sent against the new cache.
        assert len(arm[0].client.creates) == 2
        assert cached_names(arm[0]) == ["cachedContents/scripted-1", "cachedContents/scripted-2"]
    # The prefixes differ only in the declarations' wire form (`agent/NOTES.md`), so the
    # caches are keyed apart; the instruction they hold is one text.
    adk_made, pai_made = (
        [instruction_of(create["config"]) for create in arm[0].client.creates] for arm in (adk, pai)
    )
    assert adk_made == pai_made


async def test_a_cache_gone_twice_ends_the_turn_as_dev_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programme = (Call(refuse=403), Call(refuse=403), calls(PROPOSE))
    adk, _ = await cached_arm(monkeypatch, "adk", programme)
    pai, _ = await cached_arm(monkeypatch, "pai", programme)

    assert len(adk[0].client.calls) == len(pai[0].client.calls) == 2
    assert events(adk[1][0])[-1] == events(pai[1][0])[-1] == "error"
    assert_same_frames(adk[1], pai[1])


async def test_control_a_cache_error_that_is_never_gone_is_not_resent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pai_planner, "_CACHE_GONE_CODES", frozenset())
    programme = (Call(refuse=403), calls(PROPOSE))
    adk, _ = await cached_arm(monkeypatch, "adk", programme)
    pai, _ = await cached_arm(monkeypatch, "pai", programme)

    assert events(adk[1][0])[-1] == "plan.proposed"
    assert events(pai[1][0])[-1] == "error"
    assert len(pai[0].client.calls) == 1


# --- 4. What the model reads, and how the turn ends, for output Pydantic AI refuses ------

UNKNOWN = calls(("no_such_tool", {"a": 1}))
FINISH = types.FinishReason


def reads(built: studio_arms.Built, index: int) -> list[dict[str, Any]]:
    """The function responses in request `index` of the planner's model."""
    return [
        part["function_response"]
        for content in sc.contents_list(built.client.calls[index]["contents"], ids=False)
        for part in content.get("parts", [])
        if "function_response" in part
    ]


async def test_an_unknown_tool_name_is_answered_as_dev_answers_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, (UNKNOWN, calls(PROPOSE)))

    assert_ends_alike(adk, pai)
    assert events(pai[1][0])[-1] == "plan.proposed"
    (response,) = reads(adk[0], 1)
    assert response["name"] == "no_such_tool"
    assert response["response"]["error"].startswith(
        "Invoking `no_such_tool()` failed as no tool with that name is available. The tools you"
        " can call are: get_skill, get_skill_file, ask_user_question, propose_plan, reject_plan."
    )
    assert reads(pai[0], 1) == reads(adk[0], 1)


async def test_control_pydantic_ais_own_answer_to_an_unknown_tool_name_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, (UNKNOWN, calls(PROPOSE)), fixes={"unknown_tools": False})

    # Both end on a plan; the model reads different words.
    assert events(pai[1][0])[-1] == events(adk[1][0])[-1] == "plan.proposed"
    (read,) = reads(pai[0], 1)
    assert read["response"]["error"] == (
        "Unknown tool name: 'no_such_tool'. Available tools: 'ask_user_question', 'get_skill',"
        " 'get_skill_file', 'propose_plan', 'reject_plan'"
    )
    assert reads(pai[0], 1) != reads(adk[0], 1)


CALL_CAP = {"turn_model_calls_max": 4}


async def test_a_model_that_keeps_calling_an_unknown_tool_is_ended_by_the_call_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(monkeypatch, (UNKNOWN,) * 6, settings=CALL_CAP)

    assert_ends_alike(adk, pai)
    assert len(pai[0].client.calls) == 4
    assert "as many model steps as it allows" in told(pai[0])


async def test_control_pydantic_ai_ends_a_run_after_two_calls_to_an_unknown_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, pai = await pair(
        monkeypatch, (UNKNOWN,) * 6, settings=CALL_CAP, fixes={"unknown_tools": False}
    )

    # `UnexpectedModelBehavior` ("exceeded max retries") ends the run on the second call,
    # with the refusal for a plan that was never made in place of the call cap's.
    assert len(pai[0].client.calls) == 2
    assert len(adk[0].client.calls) == 4
    assert told(pai[0]) != told(adk[0])
    with pytest.raises(AssertionError):
        assert_ends_alike(adk, pai)


NO_RESPONSE = {
    "an empty stream": Call(chunks=[]),
    "a response the provider's filter blocked": Call(chunks=[[]], bare=True, finish=FINISH.SAFETY),
}


@pytest.mark.parametrize("answer", NO_RESPONSE.values(), ids=NO_RESPONSE.keys())
async def test_a_response_pydantic_ai_refuses_ends_the_run_as_a_response_with_no_call_does(
    answer: Call, monkeypatch: pytest.MonkeyPatch
) -> None:
    adk, pai = await pair(monkeypatch, (answer,))

    assert_ends_alike(adk, pai)
    assert events(pai[1][0])[-1] == "done"
    assert len(pai[0].client.calls) == 1


ENDS_ALIKE_ALREADY = {
    "a malformed function call": Call(
        chunks=[[]], bare=True, finish=FINISH.MALFORMED_FUNCTION_CALL
    ),
    "a response cut at the token limit": Call(
        chunks=[[function_call("propose_plan", {})]], finish=FINISH.MAX_TOKENS
    ),
}


@pytest.mark.parametrize("answer", ENDS_ALIKE_ALREADY.values(), ids=ENDS_ALIKE_ALREADY.keys())
async def test_a_response_pydantic_ai_accepts_ends_as_dev_does_with_no_fix(
    answer: Call, monkeypatch: pytest.MonkeyPatch
) -> None:
    adk, pai = await pair(monkeypatch, (answer,), fixes={"swallow": False})

    assert_ends_alike(adk, pai)


@pytest.mark.parametrize("answer", NO_RESPONSE.values(), ids=NO_RESPONSE.keys())
async def test_control_an_unswallowed_unexpected_model_behavior_fails_the_turn(
    answer: Call, monkeypatch: pytest.MonkeyPatch
) -> None:
    adk, pai = await pair(monkeypatch, (answer,), fixes={"swallow": False})

    assert events(pai[1][0])[-1] == "error"
    assert events(adk[1][0])[-1] == "done"
    with pytest.raises(AssertionError):
        assert_ends_alike(adk, pai)


# --- 5. An ask and a `propose_plan` in one response --------------------------------------

COMBINE = ask(message="Do these add up?", options=[dict(o) for o in COMBINATION_OPTIONS])
ASK_YEARS = ask(message="Which years do you mean?")
PROPOSE_WITH = ("propose_plan", dict(WITH_ASK.reference_payload))
# Each case: the model's one response, its later calls, and the task it answers.
SIBLINGS = {
    "ask first": ([COMBINE, PROPOSE_WITH], [PROPOSE_WITH], WITH_ASK),
    "propose first": ([PROPOSE_WITH, COMBINE], [PROPOSE_WITH], WITH_ASK),
    "skill and propose": ([GET_SKILL, PROPOSE], [PROPOSE], PLAIN),
    "a proposal that is accepted, then an ask": ([PROPOSE, ASK_YEARS], [PROPOSE], PLAIN),
    "an ask, then a proposal that is accepted": ([ASK_YEARS, PROPOSE], [PROPOSE], PLAIN),
}


async def traced_pair(
    monkeypatch: pytest.MonkeyPatch, programme: Sequence[Call], task: Any
) -> tuple[Arm, Arm, list[str], list[str]]:
    logs: dict[str, list[str]] = {"adk": [], "pai": []}
    arms: dict[str, Arm] = {}
    for name in ("adk", "pai"):
        with monkeypatch.context() as scope:
            trace_tools(scope, logs[name])
            with bounds(scope), hard_arm(scope):
                arms[name] = await turn(name, task, tuple(programme))
    return arms["adk"], arms["pai"], logs["adk"], logs["pai"]


@pytest.mark.parametrize("case", SIBLINGS)
async def test_an_ask_beside_a_proposal_ends_as_dev_does_in_the_same_order_of_writes(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    together, later, task = SIBLINGS[case]
    adk, pai, adk_log, pai_log = await traced_pair(
        monkeypatch, (calls(*together), calls(*later)), task
    )

    assert_ends_alike(adk, pai)
    # Each body starts and ends in the same order on both arms, with the same keys written
    # between its start and its end (a concurrent body's writes show in the other's span).
    assert pai_log == adk_log
    assert any("wrote [" in line and "[]" not in line for line in adk_log), adk_log


async def test_the_two_tools_of_one_response_overlap_as_they_do_on_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    together, later, task = SIBLINGS["a proposal that is accepted, then an ask"]
    _, _, adk_log, pai_log = await traced_pair(monkeypatch, (calls(*together), calls(*later)), task)

    # The proposal awaits its translation, so the ask starts, writes and ends inside it.
    assert [line.split(" ")[0] for line in pai_log[:4]] == [
        "propose_and_translate:start",
        "ask_user:start",
        "ask_user:end",
        "propose_and_translate:end",
    ]
    assert pai_log == adk_log


async def test_control_tools_run_one_after_the_other_write_in_another_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def one_at_a_time(seam: Any, parts: Any, wiring: Any) -> list[Any]:
        tools = pai_planner.planner_tools(seam, parts, wiring)
        for tool in tools:
            tool.sequential = True
        return tools

    monkeypatch.setattr(hardened, "planner_tools", one_at_a_time)
    together, later, task = SIBLINGS["a proposal that is accepted, then an ask"]
    _, _, adk_log, pai_log = await traced_pair(monkeypatch, (calls(*together), calls(*later)), task)

    # The proposal runs to its end before the ask starts.
    assert pai_log[1].startswith("propose_and_translate:end")
    assert adk_log[1] == "ask_user:start"


# --- 6. Resume from a fresh Runner and fresh node objects after an ask --------------------

# Two asks in one turn: the proposal after the second needs what the first one's answer chose.
P1 = calls(("propose_plan", dict(WITH_ASK.reference_payload)))
COMBINE_ASK = calls(COMBINE)
YEARS_ASK = calls(ask(message="Since when?"))
TWO_ASKS = [P1, COMBINE_ASK, YEARS_ASK, P1]
ANSWERS = [ANSWER, TEXT]


async def two_asks_in_one_app(monkeypatch: pytest.MonkeyPatch, arm: str) -> Arm:
    with bounds(monkeypatch), hard_arm(monkeypatch):
        return await turn(
            arm, WITH_ASK, tuple(TWO_ASKS), after=(("answer", ANSWER), ("answer", TEXT))
        )


def assert_phases_equal_one_app(
    built: list[studio_arms.Built], segments: list[list[dict[str, Any]]], whole: Arm
) -> None:
    """A turn run in phases, each on its own app, ended as one app ran it: the same frames,
    the planner's requests in the same order, and the same stored session."""
    assert [events(s)[-1] for s in segments] == [
        "elicitation.request",
        "elicitation.request",
        "plan.proposed",
    ]
    assert_same_frames(segments, whole[1])
    sent = [c for b in built for c in b.client.calls]
    assert len(sent) == len(whole[0].client.calls)
    for index, (a, b) in enumerate(zip(sent, whole[0].client.calls, strict=True)):
        assert sc.contents_list(a["contents"], ids=False) == sc.contents_list(
            b["contents"], ids=False
        ), f"request {index}"
    assert differences(normal(state_of(built[-1])), normal(state_of(whole[0]))) == []


@pytest.mark.parametrize("arm", ["adk", "pai"])
async def test_a_turn_resumed_on_fresh_apps_ends_as_one_app_runs_it(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [(arm, TWO_ASKS[:2]), (arm, TWO_ASKS[2:3]), (arm, TWO_ASKS[3:])]
    built, segments = await phases(monkeypatch, WITH_ASK, plan, ANSWERS)
    whole = await two_asks_in_one_app(monkeypatch, arm)

    assert_phases_equal_one_app(built, segments, whole)


async def test_control_a_record_kept_on_the_node_object_does_not_survive_a_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def remembered(self: hardened.HardSeam, ctx: Any) -> list[Any]:
        record = (
            vars(self)
            .setdefault("memory", {})
            .setdefault(ctx.session.id, [pai_call.user_message(hardened.pai_node.user_text(ctx))])
        )
        asked = hardened.load_pending_ask(ctx.session.state)
        if ctx.resume_inputs and asked is not None and asked.call_id in ctx.resume_inputs:
            record.append(
                pai_call.answer_message(asked.call_id, {"result": ctx.resume_inputs[asked.call_id]})
            )
        return record

    monkeypatch.setattr(hardened.HardSeam, "_record", remembered)
    plan = [("pai", TWO_ASKS[:2]), ("pai", TWO_ASKS[2:3]), ("pai", TWO_ASKS[3:])]
    built, segments = await phases(monkeypatch, WITH_ASK, plan, ANSWERS)
    whole = await two_asks_in_one_app(monkeypatch, "pai")

    with pytest.raises(AssertionError):
        assert_phases_equal_one_app(built, segments, whole)
    # The first answer is not in the new app's record, so the last proposal is refused.
    assert events(segments[-1])[-1] != "plan.proposed"


# --- 7. A pause under one planner, answered under the other ---------------------------------

SWITCHES = {
    "dev's planner, then dev's, then Pydantic AI": ("adk", "adk", "pai"),
    "Pydantic AI, then Pydantic AI, then dev's": ("pai", "pai", "adk"),
    "Pydantic AI, then dev's, then Pydantic AI": ("pai", "adk", "pai"),
    "dev's, then Pydantic AI, then dev's": ("adk", "pai", "adk"),
}


@pytest.mark.parametrize("arms", SWITCHES.values(), ids=SWITCHES.keys())
async def test_a_turn_that_passes_between_planners_lands_the_plan_dev_lands(
    arms: tuple[str, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [(arms[0], TWO_ASKS[:2]), (arms[1], TWO_ASKS[2:3]), (arms[2], TWO_ASKS[3:])]
    built, segments = await phases(monkeypatch, WITH_ASK, plan, ANSWERS)
    whole = await two_asks_in_one_app(monkeypatch, "adk")

    assert_phases_equal_one_app(built, segments, whole)


@pytest.mark.parametrize(
    "arms",
    [("adk", "adk", "pai"), ("pai", "adk", "pai")],
    ids=["dev's, dev's, Pydantic AI", "Pydantic AI, dev's, Pydantic AI"],
)
async def test_control_a_record_read_only_from_the_nodes_own_key_loses_what_dev_recorded(
    arms: tuple[str, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [(arms[0], TWO_ASKS[:2]), (arms[1], TWO_ASKS[2:3]), (arms[2], TWO_ASKS[3:])]
    built, segments = await phases(
        monkeypatch, WITH_ASK, plan, ANSWERS, fixes={"record_from": "stored"}
    )

    # Under `node/`'s record the last run reads no first answer (none, or a stale record),
    # refuses the proposal and asks the model again; the turn ends with no plan.
    assert events(segments[-1])[-1] != "plan.proposed"
    assert len(built[-1].client.calls) == 3


# --- 8. Stop mid-planning: which tool writes the stored session keeps ----------------------

SKILL_THEN_PROPOSE = [calls(GET_SKILL), calls(PROPOSE), calls(PROPOSE)]
SKILL_KEYS = {"planner:skills_read", "planner:skill_page:oda-dac2a::0"}


async def stopped(monkeypatch: pytest.MonkeyPatch, arm: str, **fixes: Any) -> dict[str, Any]:
    """Cancel a turn whose planner has read a skill page and is in the translation inside
    `propose_plan`, then read the session the Stop left and run the next turn on the app."""
    gate = Gate(event=asyncio.Event(), timeout_s=5.0)
    with bounds(monkeypatch), hard_arm(monkeypatch, **fixes):
        built = await build_arm(
            arm,  # type: ignore[arg-type]
            PLAIN,
            tuple(SKILL_THEN_PROPOSE),
            gate=gate,
            extra_orchestrator=studio_arms.orchestrator_script(PLAIN),
        )
        async with studio_client(built.app) as client:
            running = asyncio.ensure_future(stream(client, TURN, turn_body(PLAIN.question)))
            while not gate.reached:
                await asyncio.sleep(0.01)
            cancel = await client.post(
                "/api/studio/cancel", json=CANCEL, headers={"Authorization": "Bearer test-token"}
            )
            ended = await asyncio.wait_for(running, 5)
            await asyncio.sleep(0.1)
            state = state_of(built)
            after = await stream(client, TURN, turn_body("And again?"))
    return {
        "cancel": cancel.status_code,
        "ended": ended,
        "state": state,
        "after": after,
        "built": built,
    }


def assert_stop_kept_what_dev_keeps(adk: dict[str, Any], pai: dict[str, Any]) -> None:
    assert adk["cancel"] == pai["cancel"] == 204
    assert events(pai["ended"]) == events(adk["ended"])
    assert set(adk["state"]) >= SKILL_KEYS
    assert set(pai["state"]) >= SKILL_KEYS
    assert differences(normal(adk["state"]), normal(pai["state"])) == []


async def test_a_stop_keeps_the_tool_writes_dev_keeps(monkeypatch: pytest.MonkeyPatch) -> None:
    adk = await stopped(monkeypatch, "adk")
    pai = await stopped(monkeypatch, "pai")

    assert_stop_kept_what_dev_keeps(adk, pai)
    # The proposal was cancelled inside its translation, so neither arm holds its writes.
    assert not TOOL_RAN & (set(adk["state"]) | set(pai["state"]))


async def test_the_turn_after_a_stop_is_the_same_on_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk = await stopped(monkeypatch, "adk")
    pai = await stopped(monkeypatch, "pai", deltas=False)

    # Each turn has a session of its own, so what a stopped turn left in its session is
    # not read by the next one, with the writes or without them.
    assert_same_frames([adk["after"]], [pai["after"]])
    assert events(pai["after"])[-1] == "plan.proposed"
    assert len(adk["built"].client.calls) == len(pai["built"].client.calls)
    for a, b in zip(adk["built"].client.calls, pai["built"].client.calls, strict=True):
        assert sc.contents_list(a["contents"], ids=False) == sc.contents_list(
            b["contents"], ids=False
        )


async def test_control_with_the_writes_made_when_the_seam_call_ends_a_stop_loses_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk = await stopped(monkeypatch, "adk")
    pai = await stopped(monkeypatch, "pai", deltas=False)

    with pytest.raises(AssertionError):
        assert_stop_kept_what_dev_keeps(adk, pai)
    assert not SKILL_KEYS & set(pai["state"])


# --- 9. Tracing ---------------------------------------------------------------------------

MODEL = "gemini-3.8-flash"
SKILL_THEN_PLAN = [calls(GET_SKILL), calls(PROPOSE)]


async def planner_spans(
    monkeypatch: pytest.MonkeyPatch, exporter: Any, arm: str, **fixes: Any
) -> dict[str, list[Any]]:
    """The spans the planner opened in a turn that reads a skill page and proposes, by the
    operation they record (`gen_ai.operation.name`), with the harness's spans left out."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")
    with bounds(monkeypatch), hard_arm(monkeypatch, **fixes):
        await turn(arm, PLAIN, tuple(SKILL_THEN_PLAN))  # type: ignore[arg-type]
    found: dict[str, list[Any]] = {}
    for span in exporter.get_finished_spans():
        attributes = span.attributes or {}
        if attributes.get("gen_ai.agent.name") == AGENT_NAME:
            found.setdefault(attributes["gen_ai.operation.name"], []).append(span)
    return found


def keys(span: Any) -> set[str]:
    return set(span.attributes or {})


SPAN_COUNTS = {"invoke_agent": 1, "execute_tool": 2}

# What a generation, a tool and an agent-run span carry on one arm and not the other. Pinned
# exactly, so a difference that closes and one that opens both fail.
GENERATION_MISSING = {
    "gcp.vertex.agent.event_id",
    "gcp.vertex.agent.invocation_id",
    "gen_ai.usage.reasoning.output_tokens",
}
GENERATION_EXTRA = {
    "gen_ai.agent.call.id",
    "gen_ai.client.operation.time_to_first_chunk",
    "gen_ai.response.model",
    "gen_ai.system",
    "gen_ai.usage.details.cache_read_tokens",
    "gen_ai.usage.details.cached_content_tokens",
    "gen_ai.usage.details.thoughts_tokens",
    "logfire.json_schema",
    "model_request_parameters",
    "operation.cost",
    "pydantic_ai.cache.established_tokens",
    "pydantic_ai.cache.hit_ratio",
    "server.address",
}
TOOL_MISSING = {"gcp.vertex.agent.event_id", "gen_ai.tool.description", "gen_ai.tool.type"}
TOOL_EXTRA = {
    "gen_ai.agent.call.id",
    "gen_ai.conversation.id",
    "logfire.json_schema",
    "logfire.msg",
}
RUN_MISSING = {"gen_ai.agent.description"}
RUN_EXTRA = {
    "agent_name",
    "gen_ai.agent.call.id",
    "gen_ai.aggregated_usage.cache_read.input_tokens",
    "gen_ai.aggregated_usage.details.cache_read_tokens",
    "gen_ai.aggregated_usage.details.cached_content_tokens",
    "gen_ai.aggregated_usage.details.thoughts_tokens",
    "gen_ai.aggregated_usage.input_tokens",
    "gen_ai.aggregated_usage.output_tokens",
    "gen_ai.system_instructions",
    "logfire.json_schema",
    "logfire.msg",
    "model_name",
    "pydantic_ai.all_messages",
    "pydantic_ai.new_message_index",
}
USAGE = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "dfo.model.thoughts_token_count",
)


def assert_spans_match(adk: dict[str, list[Any]], pai: dict[str, list[Any]]) -> None:
    """The planner's spans on the Pydantic AI arm are the ADK arm's, by the pinned
    differences: one agent run, a generation span per model call (named `chat` where ADK's
    is `generate_content`), a tool span per tool call, each under the agent run, with the
    model, the ADK session as conversation, the usage, the thought tokens and the first-chunk
    time `harness.turn_budget` records."""
    assert {k: len(v) for k, v in adk.items() if k != "generate_content"} == SPAN_COUNTS
    assert {k: len(v) for k, v in pai.items() if k != "chat"} == SPAN_COUNTS
    assert len(adk["generate_content"]) == len(pai["chat"]) == 2
    for kind, ours, theirs, missing, extra in (
        (
            "generate_content",
            adk["generate_content"],
            pai["chat"],
            GENERATION_MISSING,
            GENERATION_EXTRA,
        ),
        ("execute_tool", adk["execute_tool"], pai["execute_tool"], TOOL_MISSING, TOOL_EXTRA),
        ("invoke_agent", adk["invoke_agent"], pai["invoke_agent"], RUN_MISSING, RUN_EXTRA),
    ):
        for one, other in zip(ours, theirs, strict=True):
            assert keys(one) - keys(other) == missing, kind
            assert keys(other) - keys(one) == extra, kind
    for one, other in zip(adk["generate_content"], pai["chat"], strict=True):
        for key in ("gen_ai.request.model", "gen_ai.agent.name", "gen_ai.conversation.id", *USAGE):
            assert one.attributes[key] == other.attributes[key], key
        assert "dfo.model.first_chunk_s" in other.attributes
    assert [s.name for s in adk["execute_tool"]] == [s.name for s in pai["execute_tool"]]
    for spans, kinds in (
        (adk, ["generate_content", "execute_tool"]),
        (pai, ["chat", "execute_tool"]),
    ):
        (run,) = spans["invoke_agent"]
        for kind in kinds:
            assert {s.parent.span_id for s in spans[kind]} == {run.context.span_id}, kind


async def spans_of(
    monkeypatch: pytest.MonkeyPatch, exporter: Any, arm: str, **fixes: Any
) -> dict[str, list[Any]]:
    exporter.clear()
    return await planner_spans(monkeypatch, exporter, arm, **fixes)


async def test_the_planner_opens_the_spans_dev_opens_apart_from_the_pinned_differences(
    monkeypatch: pytest.MonkeyPatch, exported_spans: Any
) -> None:
    adk = await spans_of(monkeypatch, exported_spans, "adk")
    pai = await spans_of(monkeypatch, exported_spans, "pai")

    assert_spans_match(adk, pai)
    # The operation and the provider are named differently.
    assert adk["generate_content"][0].attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert pai["chat"][0].attributes["gen_ai.provider.name"] == "google"


async def test_control_a_planner_with_no_instrumentation_opens_no_generation_or_tool_span(
    monkeypatch: pytest.MonkeyPatch, exported_spans: Any
) -> None:
    adk = await spans_of(monkeypatch, exported_spans, "adk")
    pai = await spans_of(monkeypatch, exported_spans, "pai", instrument=False)

    assert not pai
    with pytest.raises(AssertionError):
        assert_spans_match(adk, pai)


async def test_control_a_run_with_no_conversation_id_carries_one_pydantic_ai_makes_up(
    monkeypatch: pytest.MonkeyPatch, exported_spans: Any
) -> None:
    adk = await spans_of(monkeypatch, exported_spans, "adk")
    pai = await spans_of(monkeypatch, exported_spans, "pai", conversation=False)

    with pytest.raises(AssertionError):
        assert_spans_match(adk, pai)
    made_up = pai["chat"][0].attributes["gen_ai.conversation.id"]
    assert made_up != adk["generate_content"][0].attributes["gen_ai.conversation.id"]
    assert "#" not in made_up


async def test_control_a_client_that_drops_the_usage_of_each_chunk_loses_the_thought_tokens(
    monkeypatch: pytest.MonkeyPatch, exported_spans: Any
) -> None:
    monkeypatch.setattr(hardened.HoldingClient, "_raw", hardened.PlannerClient._raw)
    adk = await spans_of(monkeypatch, exported_spans, "adk")
    pai = await spans_of(monkeypatch, exported_spans, "pai")

    with pytest.raises(AssertionError):
        assert_spans_match(adk, pai)
    assert "dfo.model.thoughts_token_count" in adk["generate_content"][0].attributes
    assert "dfo.model.thoughts_token_count" not in pai["chat"][0].attributes


def wire_attribute(span: Any, key: str) -> Any:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


async def exported_planner_spans(
    fake_langfuse: Any, monkeypatch: pytest.MonkeyPatch, arm: str, **fixes: Any
) -> list[tuple[str, Any]]:
    """What the real Langfuse exporter sends for a planned turn: each span of the planner's
    with the name of its instrumentation scope."""
    fake_langfuse.exports.clear()
    with bounds(monkeypatch), hard_arm(monkeypatch, **fixes):
        await turn(arm, PLAIN, tuple(SKILL_THEN_PLAN))  # type: ignore[arg-type]
    await flush_telemetry()
    return [
        (scope, span)
        for scope, span in fake_langfuse.scoped_spans()
        if wire_attribute(span, "gen_ai.agent.name") == AGENT_NAME
    ]


async def test_the_langfuse_session_and_trace_name_reach_the_planners_spans_on_both_arms(
    fake_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_telemetry(service_name="harness")
    adk = await exported_planner_spans(fake_langfuse, monkeypatch, "adk")
    pai = await exported_planner_spans(fake_langfuse, monkeypatch, "pai")

    assert sorted({scope for scope, _ in adk}) == ["gcp.vertex.agent"]
    assert sorted({scope for scope, _ in pai}) == ["pydantic-ai"]
    assert sorted(span.name for _, span in adk) == sorted(
        span.name.replace("chat ", "generate_content ") for _, span in pai
    )
    for _, span in [*adk, *pai]:
        assert wire_attribute(span, "session.id") == "conv_01test0000"
        assert wire_attribute(span, "langfuse.trace.name") == "turn"
    generations = [span for _, span in pai if span.name.startswith("chat ")]
    assert all(
        wire_attribute(span, "langfuse.observation.completion_start_time") for span in generations
    )


async def test_control_with_no_instrumentation_the_planners_spans_do_not_reach_langfuse(
    fake_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_telemetry(service_name="harness")
    pai = await exported_planner_spans(fake_langfuse, monkeypatch, "pai", instrument=False)

    assert pai == []
