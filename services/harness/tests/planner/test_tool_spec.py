"""The planner's tool specs reach the model and their bodies as ADK's `FunctionTool` would.

`planner.tool_spec` builds each declaration and converts each call's
arguments itself. These tests hold both to what ADK's `FunctionTool` makes
from an equivalent Python signature: the declaration compared as JSON in the
order the model receives it, and the conversion over every `propose_plan`
call the `planner_entry` eval records plus hand cases for each rule.
A `SpecTool` serving a spec through an ADK run is checked as well. The request
carries the built declarations and the instruction a tool-less agent would,
and a refused call marks the tool's span as a `FunctionTool`'s refusal does.
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from harness.scripted import use_compiled_proposals
from planner.agent import AGENT_NAME, SpecTool, create_planner_agent, planner_instruction
from planner.declaration import compact_schema
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
from pydantic import Field

_TASKS = Path(__file__).parents[4] / "evals" / "planner_entry" / "tasks"

_VARIANTS = list(itertools.product([False, True], repeat=3))


async def _translate(call: Any) -> dict[str, object]:
    raise AssertionError("no test here translates")


def _surface() -> Any:
    return SimpleNamespace()


def _specs(*, translate: bool, surface: bool, question_fixed: bool) -> list[ToolSpec]:
    return [
        *([skill_tool_spec(_surface())] if surface else []),
        *planner_tool_specs(
            translate=_translate if translate else None,
            surface=_surface() if surface else None,
            question_fixed=question_fixed,
        ),
    ]


def _equivalent_function(spec: ToolSpec, recorded: list[dict[str, Any]]) -> Callable[..., Any]:
    """A function ADK's `FunctionTool` reads as `spec`: same name, docstring and signature.

    A non-compact spec's descriptions ride on `Annotated[..., Field(...)]`, as a
    hand-written tool's would. Calling it records the keyword arguments ADK
    passes in `recorded`.
    """

    def annotation(parameter: Parameter) -> Any:
        if parameter.description is None or spec.compact:
            return parameter.annotation
        return Annotated[parameter.annotation, Field(description=parameter.description)]

    def tool(**arguments: Any) -> dict[str, object]:
        recorded.append(arguments)
        return {"ran": True}

    tool.__name__ = tool.__qualname__ = spec.name
    tool.__doc__ = spec.description or None
    tool.__annotations__ = {p.name: annotation(p) for p in spec.parameters} | {
        "return": dict[str, object]
    }
    tool.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                p.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=p.default,
                annotation=annotation(p),
            )
            for p in spec.parameters
        ],
        return_annotation=dict[str, object],
    )
    return tool


def _adk_declaration(spec: ToolSpec) -> str:
    """ADK's declaration of `spec`'s equivalent function, as the model receives it.

    A compact spec is then given its descriptions over each parameter's
    schema and compacted (`compact_schema`).
    """
    declaration = FunctionTool(_equivalent_function(spec, []))._get_declaration()
    assert declaration is not None
    if spec.compact:
        schema = declaration.parameters_json_schema
        assert isinstance(schema, dict)
        descriptions = {p.name: p.description for p in spec.parameters}
        described = {
            name: {**argument, "description": descriptions[name]}
            for name, argument in schema["properties"].items()
        }
        declaration.parameters_json_schema = compact_schema(schema | {"properties": described})
    return declaration.model_dump_json(exclude_none=True)


def _compiled_proposal_spec() -> ToolSpec:
    agent = create_planner_agent(model_name="m")
    use_compiled_proposals(agent)
    (tool,) = [tool for tool in agent.tools if getattr(tool, "name", None) == PROPOSE_PLAN]
    assert isinstance(tool, SpecTool)
    return tool.spec


@pytest.mark.parametrize(("translate", "surface", "question_fixed"), _VARIANTS)
def test_every_declaration_is_the_one_adk_builds_from_the_same_signature(
    translate: bool, surface: bool, question_fixed: bool
) -> None:
    specs = _specs(translate=translate, surface=surface, question_fixed=question_fixed)

    for spec in specs:
        built = function_declaration(spec).model_dump_json(exclude_none=True)
        assert built == _adk_declaration(spec), spec.name


def test_the_scripted_proposal_declares_what_adk_builds_from_the_same_signature() -> None:
    spec = _compiled_proposal_spec()

    assert function_declaration(spec).model_dump_json(exclude_none=True) == _adk_declaration(spec)


@pytest.mark.parametrize(("translate", "surface", "question_fixed"), _VARIANTS)
async def test_an_adk_run_carries_the_built_declarations_and_the_planner_instruction(
    translate: bool,
    surface: bool,
    question_fixed: bool,
    scripted_llm: type[Any],
    run_to_completion: Any,
) -> None:
    """The first request of a planner run holds every spec's declaration, in
    order, in one `types.Tool`, every tool under its name, and the system
    instruction an agent with no tools and `planner_instruction` gets."""
    agent = create_planner_agent(
        model_name="m",
        translate=_translate if translate else None,
        surface=_surface() if surface else None,
        question_fixed=question_fixed,
    )
    agent.model = served = scripted_llm()
    instruction = planner_instruction(
        has_translator=translate, has_skills=surface, question_fixed=question_fixed
    )
    plain = LlmAgent(name=AGENT_NAME, model=scripted_llm(), instruction=instruction)
    reference = plain.model

    await run_to_completion(agent=agent, session_service=InMemorySessionService(), app_name="a")
    await run_to_completion(agent=plain, session_service=InMemorySessionService(), app_name="a")

    request = served.captured_requests[0]
    specs = _specs(translate=translate, surface=surface, question_fixed=question_fixed)
    (tool,) = request.config.tools
    assert [
        declaration.model_dump_json(exclude_none=True) for declaration in tool.function_declarations
    ] == [function_declaration(spec).model_dump_json(exclude_none=True) for spec in specs]
    assert list(request.tools_dict) == [spec.name for spec in specs]
    expected = reference.captured_requests[0].config.system_instruction
    assert request.config.system_instruction == expected
    assert expected.startswith(instruction)


def test_the_planner_builds_on_the_developer_api_and_refuses_vertex_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declarations are the Gemini developer API's form, which Vertex AI rejects."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
    create_planner_agent(model_name="m")

    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    with pytest.raises(RuntimeError):
        create_planner_agent(model_name="m")


# --- Conversion ----------------------------------------------------------------------


async def _adk_call(spec: ToolSpec, raw: dict[str, Any]) -> tuple[str, Any]:
    """What ADK's `FunctionTool.run_async` does with `raw` for `spec`'s equivalent
    function: the arguments it calls the function with, or its refusal."""
    recorded: list[dict[str, Any]] = []
    tool = FunctionTool(_equivalent_function(spec, recorded))
    answer = await tool.run_async(args=raw, tool_context=SimpleNamespace(function_call_id="c"))
    if recorded:
        return "called", recorded[0]
    return "refused", answer


def _ours(spec: ToolSpec, raw: dict[str, Any]) -> tuple[str, Any]:
    refusal = missing_arguments(spec, convert_arguments(spec, raw))
    if refusal is not None:
        return "refused", refusal
    return "called", convert_arguments(spec, raw)


def _typed(value: Any) -> str:
    """`value` with every type spelled out, so `1` and `1.0`, or a model and its dict, differ."""
    return repr(value)


def _recorded_proposals() -> list[dict[str, Any]]:
    """Every `propose_plan` call the `planner_entry` tasks record: each task's
    reference payload and its alternatives."""
    payloads: list[dict[str, Any]] = []
    for path in sorted(_TASKS.glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(task["reference_payload"])
        payloads += [alternative["payload"] for alternative in task.get("alternative_payloads", [])]
    return payloads


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
    # The task set holds at least 70 proposals. A lower count points at the glob or task shape.
    assert len(_recorded_proposals()) >= 70


@pytest.mark.parametrize("question_fixed", [False, True])
async def test_every_recorded_proposal_converts_as_adk_converts_it(question_fixed: bool) -> None:
    (propose,) = [
        spec
        for spec in planner_tool_specs(question_fixed=question_fixed)
        if spec.name == PROPOSE_PLAN
    ]

    for payload in _recorded_proposals():
        for raw in (payload, _as_struct(payload)):
            adk_kind, adk_value = await _adk_call(propose, raw)
            kind, value = _ours(propose, raw)
            assert (kind, _typed(value)) == (adk_kind, _typed(adk_value))


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
async def test_a_hand_call_converts_as_adk_converts_it(raw: dict[str, Any]) -> None:
    adk_kind, adk_value = await _adk_call(_HAND, raw)
    kind, value = _ours(_HAND, raw)

    assert (kind, _typed(value)) == (adk_kind, _typed(adk_value))


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
    assert ("refused", answer) == await _adk_call(ask, {"options": []})


def test_a_body_receives_every_omitted_parameter_at_its_default() -> None:
    received = run_tool(_HAND, {"answer": _ANSWER}, call=SimpleNamespace(), end_run=lambda: None)  # type: ignore[arg-type]

    assert isinstance(received, dict)
    assert list(received) == [parameter.name for parameter in _HAND.parameters]
    assert received["answers"] == []
    assert received["count"] == 0
    assert received["answer"] == AskAnswer.model_validate(_ANSWER)


# --- Telemetry -----------------------------------------------------------------------


async def _tool_span_failure(
    tool: Any, scripted_llm: type[Any], run_to_completion: Any, exported_spans: Any
) -> tuple[object, object, object]:
    """The `error.type`, status code and status description of `tool`'s span when the
    model calls it with no arguments."""
    agent = LlmAgent(name="a", model=scripted_llm(tool_name=tool.name, tool_args={}), tools=[tool])
    await run_to_completion(agent=agent, session_service=InMemorySessionService(), app_name="a")
    (span,) = [
        span
        for span in exported_spans.get_finished_spans()
        if span.name == f"execute_tool {tool.name}"
    ]
    exported_spans.clear()
    return (
        (span.attributes or {}).get("error.type"),
        span.status.status_code,
        (span.status.description),
    )


async def test_a_call_missing_an_argument_marks_its_span_as_adk_marks_a_function_tools(
    scripted_llm: type[Any], run_to_completion: Any, exported_spans: Any
) -> None:
    (ask,) = [spec for spec in planner_tool_specs() if spec.name == "ask_user_question"]

    served = await _tool_span_failure(
        SpecTool(ask), scripted_llm, run_to_completion, exported_spans
    )
    adk = await _tool_span_failure(
        FunctionTool(_equivalent_function(ask, [])), scripted_llm, run_to_completion, exported_spans
    )

    assert adk[0] == "TOOL_ERROR"
    assert served == adk
