"""The Pydantic AI planner ends where the ADK planner ends on the same model behaviour,
and its later requests carry what the ADK planner's carry.

Each such test runs one planner on a script (`planner_pairs`), on each planner in
turn, and checks what the run sent and left against a stored expectation
(`tests.planner_expectations`): the requests made, what the model reads, and the
conversation the run leaves. The Pydantic AI planner's result is the stored file;
the ADK planner's equals it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from google.genai import types
from harness.turn_events import PLANNER_ASK
from planner.pydantic_agent import tool_not_found
from planner.specs import PROPOSE_PLAN
from planner.tools import ASK_ACCEPTED

from ..planner_expectations import (
    STORING_RUNNER,
    expect,
    expectation_name,
    on_each_runner,
)
from ..planner_seam_runners import PlannerRunner as Runner
from ..scripted_genai import (
    Reply,
    blocked_reply,
    call_part,
    empty_reply,
    reply,
    text_part,
    thought_part,
)
from .planner_pairs import BRIEF, QUESTION, PairRun, adk_run, pydantic_run

_TOOLS = (PLANNER_ASK, PROPOSE_PLAN, "reject_plan")


async def _run(runner: Runner, script: list[Reply], **kwargs: Any) -> PairRun:
    return await (adk_run if runner == "adk" else pydantic_run)(script, **kwargs)


def _contents(request: Any) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json", exclude_none=True) for c in request.content_list]


def _last_content(request: Any) -> dict[str, Any]:
    return request.content_list[-1].model_dump(mode="json", exclude_none=True)


@on_each_runner
async def test_a_call_to_a_missing_tool_is_answered_the_same_on_both_planners(
    runner: Runner, request: pytest.FixtureRequest
) -> None:
    calls = 8
    script = [
        *(reply(call_part("lookup_data", {"q": n}, id=f"u{n}")) for n in range(calls)),
        reply(text_part("I cannot look that up.")),
    ]

    run = await _run(runner, script)

    # Every call is answered and the run goes on, with one request per call and then the last.
    assert len(run.requests) == calls + 1
    answer = tool_not_found("lookup_data", _TOOLS)
    assert ("answer", "lookup_data", "u0", answer) in run.conversations[0]
    assert _last_content(run.requests[1]) == {
        "role": "user",
        "parts": [{"function_response": {"id": "u0", "name": "lookup_data", "response": answer}}],
    }
    expect(
        expectation_name(request, runner=runner),
        {
            "conversations": run.conversations,
            "last_contents": [_last_content(each) for each in run.requests[1:]],
        },
        stores=runner == STORING_RUNNER,
    )


_ENDINGS = {
    "a response with no parts": empty_reply(),
    "a stream with no chunk": Reply(chunks=[]),
    "a blocked prompt": blocked_reply(),
}


async def test_the_warning_for_a_refused_response_carries_no_model_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "SENTINEL-thought-text-7f3a"
    filtered = reply(thought_part(sentinel), finish_reason=types.FinishReason.SAFETY)

    with caplog.at_level(logging.WARNING, logger="planner.pydantic_agent"):
        await pydantic_run([filtered])

    warnings = [r for r in caplog.records if r.name == "planner.pydantic_agent"]
    assert len(warnings) == 1
    assert "ContentFilterError" in warnings[0].getMessage()
    assert all(sentinel not in r.getMessage() for r in caplog.records)


@on_each_runner
@pytest.mark.parametrize("ending", _ENDINGS.values(), ids=_ENDINGS.keys())
async def test_a_run_ends_the_same_on_both_planners_with_what_it_had_said(
    ending: Reply, runner: Runner, request: pytest.FixtureRequest
) -> None:
    # A refused proposal first, so the run has something to keep when it ends.
    script = [reply(call_part(PROPOSE_PLAN, {}, id="p1")), ending, reply(text_part("never"))]

    run = await _run(runner, script)

    assert len(run.requests) == 2
    ((conversation,),) = [run.conversations]
    assert [turn[0] for turn in conversation] == ["part", "answer"]
    expect(
        expectation_name(request, runner=runner),
        run.conversations,
        stores=runner == STORING_RUNNER,
    )


@on_each_runner
async def test_the_request_after_a_tool_answer_is_the_same_on_both_planners(
    runner: Runner, request: pytest.FixtureRequest
) -> None:
    signed = thought_part("Weighing the sources.", signature=b"thought-signature")
    unsigned = thought_part("A summary of the weighing.", signature=None)
    call = call_part(PROPOSE_PLAN, {"question": QUESTION}, id="p1", signature=b"call-signature")
    script = [reply(signed, unsigned, call), reply(text_part("done"))]

    run = await _run(runner, script)

    assert len(run.requests) == 2
    contents = _contents(run.requests[1])
    model_turn = contents[2]
    assert model_turn["role"] == "model"
    assert [part.get("thought") for part in model_turn["parts"]] == [True, None]
    assert model_turn["parts"][1]["function_call"]["id"] == "p1"
    expect(
        expectation_name(request, runner=runner),
        {"contents": contents, "conversations": run.conversations},
        stores=runner == STORING_RUNNER,
    )


@on_each_runner
async def test_the_first_request_of_a_run_resumed_after_an_answered_ask_is_the_same_on_both_planners(
    runner: Runner, request: pytest.FixtureRequest
) -> None:
    ask = {"message": "Which donor do you mean?"}
    resumed = "Context for this question: the readings.\n\nThe user answered: France."
    script = [reply(call_part(PLANNER_ASK, ask, id="a1")), reply(text_part("done"))]

    run = await _run(runner, script, briefs=(BRIEF, resumed))

    assert len(run.requests) == 2
    contents = _contents(run.requests[1])
    assert contents == [
        {"role": "user", "parts": [{"text": QUESTION}]},
        {"role": "user", "parts": [{"text": resumed}]},
    ]
    assert ("answer", PLANNER_ASK, "a1", dict(ASK_ACCEPTED)) in run.conversations[0]
    expect(
        expectation_name(request, runner=runner),
        {"contents": contents, "conversations": run.conversations, "state": run.state},
        stores=runner == STORING_RUNNER,
    )
