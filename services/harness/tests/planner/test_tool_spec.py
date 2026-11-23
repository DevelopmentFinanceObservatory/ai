"""Planner tool specs reach the model and their bodies as ADK's `FunctionTool` did.

`planner.tool_spec` builds each declaration and converts each call's
arguments itself. These tests hold both to what ADK 2.13's
`FunctionTool` made from an equivalent Python signature, recorded in
`fixtures/function_tool_oracle.json`. The declaration is compared as JSON in
the order the model receives it. The conversion is checked over the
`propose_plan` calls the `planner_entry` eval recorded, as sent and as a proto
`Struct` hands them back, plus hand cases for each rule.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harness.scripted import use_compiled_proposals
from planner.model import refuse_vertex_ai
from planner.plan import AskAnswer, Decline
from planner.specs import PROPOSE_PLAN, planner_tool_specs, skill_tool_spec
from planner.tool_spec import (
    Parameter,
    ToolSpec,
    convert_arguments,
    function_declaration,
    missing_arguments,
    run_tool,
)

# What ADK 2.13's `FunctionTool` made of each spec's equivalent Python
# signature: each declaration as the model receives it, by variant and tool
# name, and the arguments each call was converted to, as `repr`, or the refusal
# it was given.
_ORACLE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "function_tool_oracle.json").read_text(
        encoding="utf-8"
    )
)

_VARIANTS = list(itertools.product([False, True], repeat=2))


async def _translate(call: Any) -> dict[str, object]:
    raise AssertionError("no test here translates")


def _surface() -> Any:
    return SimpleNamespace()


def _specs(*, translate: bool, surface: bool) -> list[ToolSpec]:
    return [
        *([skill_tool_spec(_surface())] if surface else []),
        *planner_tool_specs(
            translate=_translate if translate else None,
            surface=_surface() if surface else None,
        ),
    ]


def _compiled_proposal_spec() -> ToolSpec:
    specs = use_compiled_proposals(_specs(translate=False, surface=False))
    (spec,) = [spec for spec in specs if spec.name == PROPOSE_PLAN]
    return spec


@pytest.mark.parametrize(("translate", "surface"), _VARIANTS)
def test_every_declaration_is_the_one_adk_builds_from_the_same_signature(
    translate: bool, surface: bool
) -> None:
    specs = _specs(translate=translate, surface=surface)

    oracle = _ORACLE["declarations"][f"translate={translate},surface={surface}"]
    assert [spec.name for spec in specs] == list(oracle)
    for spec in specs:
        built = function_declaration(spec).model_dump_json(exclude_none=True)
        assert built == oracle[spec.name], spec.name


def test_the_scripted_proposal_declares_what_adk_builds_from_the_same_signature() -> None:
    spec = _compiled_proposal_spec()

    built = function_declaration(spec).model_dump_json(exclude_none=True)
    assert built == _ORACLE["compiled_proposal"]


@pytest.mark.parametrize(
    ("environment", "model_name", "refused"),
    [
        pytest.param({}, "gemini-2.5-flash", False, id="developer-api"),
        pytest.param(
            {"GOOGLE_GENAI_USE_ENTERPRISE": "true"}, "gemini-2.5-flash", True, id="enterprise"
        ),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "1"}, "gemini-2.5-flash", True, id="legacy-vertexai"
        ),
        pytest.param(
            {"GOOGLE_GENAI_USE_ENTERPRISE": "false", "GOOGLE_GENAI_USE_VERTEXAI": "true"},
            "gemini-2.5-flash",
            False,
            id="enterprise-decides-over-vertexai",
        ),
    ],
)
def test_the_planner_runs_on_the_developer_api_and_refuses_vertex_ai(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], model_name: str, refused: bool
) -> None:
    """The declarations are the Gemini developer API's form, which Vertex AI rejects.
    The planner's google-genai client takes Vertex AI from the environment."""
    for name in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    if refused:
        with pytest.raises(RuntimeError):
            refuse_vertex_ai(model_name)
    else:
        refuse_vertex_ai(model_name)


# --- Conversion ----------------------------------------------------------------------


def _ours(spec: ToolSpec, raw: dict[str, Any]) -> tuple[str, Any]:
    refusal = missing_arguments(spec, convert_arguments(spec, raw))
    if refusal is not None:
        return "refused", refusal
    return "called", convert_arguments(spec, raw)


def _typed(value: Any) -> str:
    """`value` with every type spelled out.

    `1` and `1.0` differ, and so do a model and its dict.
    """
    return repr(value)


def _recorded_proposals() -> list[dict[str, Any]]:
    """The `propose_plan` calls the oracle holds: each `planner_entry` task's
    reference payload and its alternatives, with what `FunctionTool` made of
    each as sent and as a `Struct` hands it back."""
    return _ORACLE["proposals"]


def _as_struct(value: Any) -> Any:
    """`value` as a proto `Struct` hands it back: every number a float."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, dict):
        return {key: _as_struct(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_struct(item) for item in value]
    return value


def test_the_recorded_proposals_cover_the_task_set() -> None:
    # The task set holds at least 70 proposals. A lower count points at the glob
    # or the task shape.
    assert len(_recorded_proposals()) >= 70


async def test_every_recorded_proposal_converts_as_adk_converts_it() -> None:
    (propose,) = [spec for spec in planner_tool_specs() if spec.name == PROPOSE_PLAN]

    for recorded in _recorded_proposals():
        payload = recorded["payload"]
        for raw, adk in (
            (payload, recorded["as_sent"]),
            (_as_struct(payload), recorded["as_struct"]),
        ):
            kind, value = _ours(propose, raw)
            assert (kind, _typed(value)) == (adk["kind"], adk["typed"])


_HAND = ToolSpec(
    name="hand",
    description="A tool with one parameter of each kind ADK converts.",
    parameters=(
        Parameter("answer", AskAnswer),
        Parameter("maybe", AskAnswer | None, None),
        Parameter("answers", list[AskAnswer], []),
        Parameter("either", AskAnswer | Decline | None, None),
        Parameter("count", int, 0),
        Parameter("counts", list[int], []),
        Parameter("version", int | None, None),
        Parameter("label", str | None, None),
    ),
    body=lambda call, end_run, arguments: arguments,
)

_ANSWER = {"ask": 0, "outputs": ["a"]}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"answer": _ANSWER, "bogus": 1}, id="unknown-key-dropped"),
        pytest.param({"answer": _ANSWER}, id="model"),
        pytest.param({"answer": {"ask": "x"}}, id="model-that-fails-stays-a-dict"),
        pytest.param({"answer": _ANSWER, "maybe": _ANSWER}, id="optional-model"),
        pytest.param({"answer": _ANSWER, "maybe": None}, id="optional-model-none"),
        pytest.param(
            {"answer": _ANSWER, "answers": [_ANSWER, {"ask": "x"}, "text"]},
            id="list-of-models-keeps-a-failing-item-as-its-dict",
        ),
        pytest.param({"answer": _ANSWER, "either": _ANSWER}, id="union-of-models"),
        pytest.param({"answer": _ANSWER, "either": {"nothing": 1}}, id="union-that-fails"),
        pytest.param({"answer": _ANSWER, "count": 3.0, "counts": [1.0, 2.5, 3]}, id="floats"),
        pytest.param({"answer": _ANSWER, "count": 2.5, "version": 4.0}, id="optional-int"),
        pytest.param({"answer": _ANSWER, "label": 1.0}, id="unconverted"),
        pytest.param({"maybe": _ANSWER}, id="missing-mandatory"),
        pytest.param({}, id="nothing"),
    ],
)
async def test_a_hand_call_converts_as_adk_converts_it(
    request: pytest.FixtureRequest, raw: dict[str, Any]
) -> None:
    adk = _ORACLE["hand"][request.node.callspec.id]
    kind, value = _ours(_HAND, raw)

    assert (kind, _typed(value)) == (adk["kind"], adk["typed"])


async def test_a_call_missing_an_argument_runs_nothing_and_answers_as_adk_does() -> None:
    (ask,) = [spec for spec in planner_tool_specs() if spec.name == "ask_user_question"]
    ran: list[object] = []
    recording = ToolSpec(
        name=ask.name,
        description=ask.description,
        parameters=ask.parameters,
        body=lambda call, end_run, arguments: ran.append(arguments),
    )

    answer = run_tool(recording, {"options": []}, call=SimpleNamespace(), end_run=lambda: None)  # type: ignore[arg-type]

    assert ran == []
    adk = _ORACLE["missing_ask"]
    assert ("refused", answer) == (adk["kind"], adk["answer"])


def test_a_body_receives_every_omitted_parameter_at_its_default() -> None:
    received = run_tool(_HAND, {"answer": _ANSWER}, call=SimpleNamespace(), end_run=lambda: None)  # type: ignore[arg-type]

    assert isinstance(received, dict)
    assert list(received) == [parameter.name for parameter in _HAND.parameters]
    assert received["answers"] == []
    assert received["count"] == 0
    assert received["answer"] == AskAnswer.model_validate(_ANSWER)
