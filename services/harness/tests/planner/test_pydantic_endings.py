"""The Pydantic AI planner ends where the ADK planner ends on the same model behaviour,
and its later requests carry what the ADK planner's carry.

Each test runs both planners on one script (`planner_pairs`) and asserts they
agree: the requests made, what the model reads, and the conversation each run
leaves.
"""

from __future__ import annotations

from typing import Any

import pytest
from harness.turn_events import PLANNER_ASK
from planner.pydantic_agent import tool_not_found
from planner.specs import PROPOSE_PLAN
from planner.tools import ASK_ACCEPTED

from ..scripted_genai import (
    Reply,
    blocked_reply,
    call_part,
    empty_reply,
    reply,
    text_part,
    thought_part,
)
from .planner_pairs import BRIEF, QUESTION, adk_run, pydantic_run

_TOOLS = (PLANNER_ASK, PROPOSE_PLAN, "reject_plan")


def _last_content(request: Any) -> dict[str, Any]:
    return request.content_list[-1].model_dump(mode="json", exclude_none=True)


async def test_a_call_to_a_missing_tool_is_answered_as_the_adk_planner_answers_it() -> None:
    calls = 8
    script = [
        *(reply(call_part("lookup_data", {"q": n}, id=f"u{n}")) for n in range(calls)),
        reply(text_part("I cannot look that up.")),
    ]

    adk = await adk_run(script)
    pydantic = await pydantic_run(script)

    # Every call is answered and the run goes on, with one request per call and then the last.
    assert len(adk.requests) == len(pydantic.requests) == calls + 1
    assert pydantic.conversations == adk.conversations
    answer = tool_not_found("lookup_data", _TOOLS)
    assert ("answer", "lookup_data", "u0", answer) in pydantic.conversations[0]
    for adk_request, pydantic_request in zip(adk.requests[1:], pydantic.requests[1:], strict=True):
        assert _last_content(pydantic_request) == _last_content(adk_request)
    assert _last_content(pydantic.requests[1]) == {
        "role": "user",
        "parts": [{"function_response": {"id": "u0", "name": "lookup_data", "response": answer}}],
    }


_ENDINGS = {
    "a response with no parts": empty_reply(),
    "a stream with no chunk": Reply(chunks=[]),
    "a blocked prompt": blocked_reply(),
}


@pytest.mark.parametrize("ending", _ENDINGS.values(), ids=_ENDINGS.keys())
async def test_a_run_ends_where_the_adk_planner_ends_with_what_it_had_said(ending: Reply) -> None:
    # A refused proposal first, so the run has something to keep when it ends.
    script = [reply(call_part(PROPOSE_PLAN, {}, id="p1")), ending, reply(text_part("never"))]

    adk = await adk_run(script)
    pydantic = await pydantic_run(script)

    assert len(adk.requests) == len(pydantic.requests) == 2
    assert pydantic.conversations == adk.conversations
    ((conversation,),) = [pydantic.conversations]
    assert [turn[0] for turn in conversation[:2]] == ["part", "answer"]
    blocked = ending.prompt_feedback is not None
    assert conversation[2:] == ([("blocked", "SAFETY")] if blocked else [])


async def test_the_request_after_a_tool_answer_equals_the_adk_planners() -> None:
    signed = thought_part("Weighing the sources.", signature=b"thought-signature")
    unsigned = thought_part("A summary of the weighing.", signature=None)
    call = call_part(PROPOSE_PLAN, {"question": QUESTION}, id="p1", signature=b"call-signature")
    script = [reply(signed, unsigned, call), reply(text_part("done"))]

    adk = await adk_run(script)
    pydantic = await pydantic_run(script)

    assert len(adk.requests) == len(pydantic.requests) == 2
    adk_contents = [
        c.model_dump(mode="json", exclude_none=True) for c in adk.requests[1].content_list
    ]
    pydantic_contents = [
        c.model_dump(mode="json", exclude_none=True) for c in pydantic.requests[1].content_list
    ]
    assert pydantic_contents == adk_contents
    model_turn = adk_contents[2]
    assert model_turn["role"] == "model"
    assert [part.get("thought") for part in model_turn["parts"]] == [True, None]
    assert model_turn["parts"][1]["function_call"]["id"] == "p1"
    assert pydantic.conversations == adk.conversations


async def test_the_first_request_of_a_run_resumed_after_an_answered_ask_equals_the_adk_planners() -> (
    None
):
    ask = {"message": "Which donor do you mean?"}
    resumed = "Context for this question: the readings.\n\nThe user answered: France."
    script = [reply(call_part(PLANNER_ASK, ask, id="a1")), reply(text_part("done"))]

    adk = await adk_run(script, briefs=(BRIEF, resumed))
    pydantic = await pydantic_run(script, briefs=(BRIEF, resumed))

    assert len(adk.requests) == len(pydantic.requests) == 2
    adk_contents = [
        c.model_dump(mode="json", exclude_none=True) for c in adk.requests[1].content_list
    ]
    pydantic_contents = [
        c.model_dump(mode="json", exclude_none=True) for c in pydantic.requests[1].content_list
    ]
    assert (
        pydantic_contents
        == adk_contents
        == [
            {"role": "user", "parts": [{"text": QUESTION}]},
            {"role": "user", "parts": [{"text": resumed}]},
        ]
    )
    assert pydantic.conversations == adk.conversations
    assert ("answer", PLANNER_ASK, "a1", dict(ASK_ACCEPTED)) in pydantic.conversations[0]
    assert pydantic.state == adk.state
