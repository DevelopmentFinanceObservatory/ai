"""The planner on Pydantic AI against the planner on ADK, both under the real
`PlannerSeam`, both against a scripted google-genai client.

Each test runs the same task and script through `entry.run_entry` (ADK) and
`pai_entry.run_pai_entry` (Pydantic AI) and compares how the entry ended and every
request each arm sent. The requests must differ only by `WIRE`, the forms Pydantic AI's
`GoogleModel` gives the same content. Each test has a control: the Pydantic AI arm broken
in one place, which the same comparison must refuse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import entry
import pai_call
import pai_entry
import pai_planner
import pytest
import scripted as sc
import validate
from entry import Task, TranslationCalls, load_recording, load_tasks, quiet_logs
from google.genai import types
from harness.prefix_cache import PrefixCache
from harness.turn_events import PLANNER_ASK
from pai_call import PaiCall
from planner.plan import load_plan
from planner.tools import ASK_ACCEPTED
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION

quiet_logs()

TASKS = {task.id: task for task in load_tasks()}
PLAIN = TASKS["aggregate-average-oda-kenya"]
WITH_ASK = TASKS["sum-net-flows-aid-debt-ghana"]

# The user's reply to the planner's ask, as the studio's interrupt route sends it.
ANSWER = {"result": {"answers": {"q1": {"kind": "option", "spec_id": COMBINE_OPTION}}}}

# What Pydantic AI's `GoogleModel` sends differently from ADK's `Gemini` for the same
# request, by the field `sc.request_differences` reports it under. `other_config` is listed
# by path, since a header's value carries each library's version.
WIRE: dict[str, Any] = {
    "declarations": [
        {
            "path": "reject_plan.parameters_json_schema",
            "adk": "<absent>",
            "pai": {"type": "object", "properties": {}},
        }
    ],
    "tools_grouping": [{"path": "", "adk": [4], "pai": [1, 1, 1, 1]}],
    "tool_config": [
        {"path": "", "adk": None, "pai": {"function_calling_config": {"mode": "AUTO"}}}
    ],
    "other_config": [
        "automatic_continuation",
        "http_options.headers.Content-Type",
        "http_options.headers.User-Agent",
        "http_options.headers.user-agent",
        "http_options.headers.x-goog-api-client",
        "response_modalities",
    ],
}


def shape(differences: Mapping[str, Any]) -> dict[str, Any]:
    """`differences` with `other_config` reduced to the paths that differ."""
    shaped = dict(differences)
    if "other_config" in shaped:
        shaped["other_config"] = [found["path"] for found in shaped["other_config"]]
    return shaped


def propose(task: Task) -> tuple[str, dict[str, Any]]:
    return ("propose_plan", dict(task.reference_payload))


def ask(**arguments: Any) -> tuple[str, dict[str, Any]]:
    return (PLANNER_ASK, arguments)


class Arms:
    """One script run through both arms."""

    def __init__(
        self,
        task: Task,
        steps: Sequence[sc.Step],
        answers: Sequence[Mapping[str, Any]],
        *,
        translated: bool = False,
    ) -> None:
        self.task, self.steps, self.answers = task, steps, answers
        self.translated = translated
        self.adk_client, self.pai_client = sc.scripted(steps), sc.scripted(steps)

    async def translation(self) -> TranslationCalls | None:
        """Production's translation over the recording, one for each arm to read its misses."""
        if not self.translated:
            return None
        held = load_recording(self.task.id).get("translation") or {}
        return TranslationCalls(held, schemas=await validate.tool_schemas())

    async def run(self, **pai_options: Any) -> Arms:
        self.adk = await sc.run_adk(
            self.task, self.adk_client, answers=self.answers, translation=await self.translation()
        )
        self.pai = await sc.run_pai(
            self.task,
            self.pai_client,
            answers=self.answers,
            translation=await self.translation(),
            **pai_options,
        )
        return self


def assert_same_run(arms: Arms, *, requests: int) -> None:
    """The arms ended the same way and sent `requests` requests each, which differ only by
    `WIRE`."""
    assert arms.adk.error is None
    assert arms.pai.run.error is None
    assert sc.outcome(arms.pai.run) == sc.outcome(arms.adk)
    assert len(arms.adk_client.calls) == len(arms.pai_client.calls) == requests
    for index, (adk, pai) in enumerate(
        zip(arms.adk_client.calls, arms.pai_client.calls, strict=True)
    ):
        assert shape(sc.request_differences(adk, pai)) == WIRE, f"request {index}"


# --- One task, end to end ---------------------------------------------------------------


async def test_a_task_ends_the_same_way_with_the_same_first_request() -> None:
    arms = await Arms(PLAIN, [[propose(PLAIN)]], []).run()

    outcome = sc.outcome(arms.pai.run)
    assert outcome["stop_reason"] == "accepted"
    assert outcome["plan"] is not None
    assert_same_run(arms, requests=1)


def _append_to_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    of = pai_planner.PlannerParts.of

    def instruction_of(seam: Any) -> pai_planner.PlannerParts:
        parts = of(seam)
        return pai_planner.PlannerParts(parts.instruction + " X", parts.tools, parts.declarations)

    monkeypatch.setattr(pai_planner.PlannerParts, "of", staticmethod(instruction_of))


def _append_to_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    of = pai_planner.PlannerParts.of

    def declaration_of(seam: Any) -> pai_planner.PlannerParts:
        parts = of(seam)
        declarations = tuple(
            d.model_copy(update={"description": (d.description or "") + " "})
            if d.name == "propose_plan"
            else d
            for d in parts.declarations
        )
        return pai_planner.PlannerParts(parts.instruction, parts.tools, declarations)

    monkeypatch.setattr(pai_planner.PlannerParts, "of", staticmethod(declaration_of))


def _merge_question_and_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pai_call, "QUESTION_INSTRUCTIONS", None)
    monkeypatch.setattr(
        pai_call,
        "user_message",
        lambda text: pai_call.ModelRequest(
            parts=[pai_call.UserPromptPart(content=text)], metadata={pai_call.USER_MESSAGE: True}
        ),
    )
    monkeypatch.setattr(pai_entry, "user_message", pai_call.user_message)


@pytest.mark.parametrize(
    ("breaker", "field"),
    [
        (_append_to_instruction, "system_instruction"),
        (_append_to_declaration, "declarations"),
        (_merge_question_and_brief, "contents"),
    ],
)
async def test_control_the_first_request_check_fails_on_a_changed_request(
    monkeypatch: pytest.MonkeyPatch, breaker: Callable[[pytest.MonkeyPatch], None], field: str
) -> None:
    breaker(monkeypatch)
    arms = await Arms(PLAIN, [[propose(PLAIN)]], []).run()

    first = shape(sc.request_differences(arms.adk_client.calls[0], arms.pai_client.calls[0]))
    assert field in first
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=1)


# --- A plan declared ready ends the run ---------------------------------------------------


@pytest.fixture
def translating(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe's translate step answers the translation's verdict, as production's does,
    in place of the eval's stop at the first recorded plan."""

    async def translate(self: entry.Probe, call: Any) -> dict[str, object]:
        self.plan = load_plan(call.state)
        self.plan_call = call.call_id
        assert self.translator is not None
        self.translation = await self.translator(call)
        return self.translation

    monkeypatch.setattr(entry.Probe, "translate", translate)


async def test_a_plan_declared_ready_ends_the_run_with_no_further_model_call(
    translating: None,
) -> None:
    arms = await Arms(PLAIN, [[propose(PLAIN)]], [], translated=True).run()

    assert_same_run(arms, requests=1)
    answer = sc.outcome(arms.pai.run)["responses"][0]
    assert answer["translation"]["outcome"] == "ok"
    assert answer["summary"]["version"] == 1


async def test_control_a_run_that_is_not_ended_asks_the_model_again(
    translating: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pai_planner.PlannerDeps, "end_run", lambda self: None)
    arms = await Arms(PLAIN, [[propose(PLAIN)]], [], translated=True).run()

    assert (len(arms.adk_client.calls), len(arms.pai_client.calls)) == (1, 2)
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=1)


# --- An ask ends the run, and its answer resumes it ---------------------------------------

ASK_SCRIPT = [
    [propose(WITH_ASK)],
    [ask(message="Do these add up?", options=[dict(option) for option in COMBINATION_OPTIONS])],
    [propose(WITH_ASK)],
]


async def test_an_accepted_ask_ends_the_run_and_the_answer_resumes_it() -> None:
    arms = await Arms(WITH_ASK, ASK_SCRIPT, [ANSWER]).run()

    assert_same_run(arms, requests=3)
    pai_calls = arms.pai_client.calls
    # The ask is the second request's answer, and the third request is the resumed run's
    # first: the user's message and a brief that carries the answer, with no call of the
    # run that asked.
    resumed = sc.contents_list(pai_calls[2]["contents"])
    assert [content["role"] for content in resumed] == ["user", "user"]
    assert "They picked" in resumed[1]["parts"][0]["text"]
    assert not any("function_call" in part for c in resumed for part in c["parts"])
    outcome = sc.outcome(arms.pai.run)
    assert outcome["tool_calls"] == ["propose_plan", PLANNER_ASK, "propose_plan"]
    assert outcome["stop_reason"] == "accepted"


async def test_control_an_ask_that_does_not_end_the_run_makes_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pai_planner.PlannerDeps, "end_run", lambda self: None)
    arms = await Arms(WITH_ASK, ASK_SCRIPT, [ANSWER]).run()

    # The request after the ask answers inside the run that asked.
    assert "contents" in sc.request_differences(arms.adk_client.calls[2], arms.pai_client.calls[2])
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=3)


# --- A refused ask --------------------------------------------------------------------


REFUSED_SCRIPT = [
    [ask(message="Which one?", recommended_spec_id="a-candidate-never-offered")],
    [propose(PLAIN)],
]


async def test_a_refused_ask_answers_the_model_with_the_same_problem_and_the_run_goes_on() -> None:
    arms = await Arms(PLAIN, REFUSED_SCRIPT, []).run()

    assert_same_run(arms, requests=2)
    refusal = sc.outcome(arms.pai.run)["responses"][0]
    assert refusal["code"] == "invalid_arguments"
    assert refusal["outcome"] == "invalid_input"
    # The second request carries that problem back as the ask's response.
    returned = [
        part["function_response"]
        for content in sc.contents_list(arms.pai_client.calls[1]["contents"], ids=False)
        for part in content["parts"]
        if "function_response" in part
    ]
    assert [r["name"] for r in returned] == [PLANNER_ASK]
    assert returned[0]["response"] == refusal
    assert sc.outcome(arms.pai.run)["stop_reason"] == "accepted"


async def test_control_an_ask_that_accepts_everything_ends_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def accepting(call: Any, end_run: Callable[[], None], **arguments: Any) -> dict[str, object]:
        end_run()
        return dict(ASK_ACCEPTED)

    monkeypatch.setattr(pai_planner, "ask_user", accepting)
    arms = await Arms(PLAIN, REFUSED_SCRIPT, []).run()

    assert sc.outcome(arms.pai.run)["responses"][0] != sc.outcome(arms.adk)["responses"][0]
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=2)


# --- Two asks in one response -------------------------------------------------------------

TWO_ASKS = [
    [
        ask(message="First?", options=[dict(option) for option in COMBINATION_OPTIONS]),
        ask(message="Second?"),
    ]
]


async def test_the_second_ask_of_a_response_is_refused_and_the_first_pauses_the_entry() -> None:
    arms = await Arms(PLAIN, TWO_ASKS, []).run()

    assert_same_run(arms, requests=1)
    outcome = sc.outcome(arms.pai.run)
    assert outcome["stop_reason"] == "asked_user"
    accepted, refused = outcome["responses"]
    assert accepted["status"] == "asked"
    assert refused["code"] == "invalid_arguments"


async def test_control_a_call_that_sees_no_response_calls_accepts_both_asks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PaiCall, "response_calls", lambda self: ())
    arms = await Arms(PLAIN, TWO_ASKS, []).run()

    assert [r["status"] for r in sc.outcome(arms.pai.run)["responses"]] == ["asked", "asked"]
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=1)


# --- Skill pages ------------------------------------------------------------------------

SKILL_SCRIPT = [
    [("get_skill", {"skill_id": "oda-dac2a"})],
    [("get_skill", {"skill_id": "oda-dac2a", "path": "SKILL.md"})],
    [propose(PLAIN)],
]


async def test_skill_pages_read_the_same_text_and_the_brief_of_a_later_run_would_list_them() -> (
    None
):
    arms = await Arms(PLAIN, SKILL_SCRIPT, []).run()

    assert_same_run(arms, requests=3)
    responses = sc.outcome(arms.pai.run)["responses"]
    assert all(isinstance(r, dict) and r.get("result") for r in responses[:2])
    # Both pages are on record under the keys the brief of a resumed run reads them from.
    assert len(arms.pai.run.state["planner:skills_read"]) == 2


async def test_control_a_skill_tool_that_reads_nothing_answers_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nothing(*args: Any, **kwargs: Any) -> str:
        return "No page."

    monkeypatch.setattr(pai_planner, "read_skill_page", nothing)
    arms = await Arms(PLAIN, SKILL_SCRIPT, []).run()

    assert sc.outcome(arms.pai.run)["responses"][:2] != sc.outcome(arms.adk)["responses"][:2]
    with pytest.raises(AssertionError):
        assert_same_run(arms, requests=3)


# --- Thought summaries ----------------------------------------------------------------

THOUGHT = "A thought summary the model wrote, which no later request may carry."


def thought_script() -> list[list[types.Part]]:
    return [
        [
            types.Part(text=THOUGHT, thought=True),
            types.Part(
                function_call=types.FunctionCall(name="reject_plan", args={}),
                thought_signature=sc.SIGNATURE,
            ),
        ]
    ]


async def _second_requests(**pai_options: Any) -> tuple[list[Any], list[Any]]:
    adk, pai = sc.Scripted(thought_script()), sc.Scripted(thought_script())
    await sc.run_adk(PLAIN, adk)
    await sc.run_pai(PLAIN, pai, **pai_options)
    return (
        sc.contents_list(adk.calls[1]["contents"], ids=False),
        sc.contents_list(pai.calls[1]["contents"], ids=False),
    )


async def test_a_thought_summary_is_not_replayed_and_its_signature_is_kept() -> None:
    adk, pai = await _second_requests()

    assert pai == adk
    parts = [part for content in pai for part in content["parts"]]
    assert not any(part.get("text") == THOUGHT for part in parts)
    assert any("function_call" in part and "thought_signature" in part for part in parts), (
        "the signed call is replayed with its signature"
    )


async def test_control_with_the_filter_off_the_summary_is_replayed() -> None:
    adk, pai = await _second_requests(keep_thoughts=True)

    assert pai != adk
    assert any(part.get("text") == THOUGHT for content in pai for part in content["parts"])


# --- The prefix cache -------------------------------------------------------------------


def prefix_of(create: Mapping[str, Any]) -> dict[str, Any]:
    """A cache create's prefix, the system instruction as text and each declaration by name."""
    raw = sc.jsonable(create["config"])
    config = sc.config_dict(
        {key: raw.get(key) for key in ("system_instruction", "tools", "tool_config")}
    )
    return {
        "system_instruction": sc.instruction_text(config),
        "declarations": sc.declarations(config),
        "tool_config": config.get("tool_config"),
    }


async def _prefixes() -> tuple[sc.Scripted, sc.Scripted]:
    adk, pai = sc.scripted([[propose(PLAIN)]]), sc.scripted([[propose(PLAIN)]])
    await sc.run_adk(PLAIN, adk, prefix_cache=PrefixCache())
    await sc.run_pai(PLAIN, pai, prefix_cache=PrefixCache())
    return adk, pai


async def test_the_cached_prefix_holds_the_same_instruction_and_declarations() -> None:
    adk, pai = await _prefixes()

    assert len(adk.creates) == len(pai.creates) == 1
    assert sc.diff(prefix_of(adk.creates[0]), prefix_of(pai.creates[0])) == [
        {
            "path": "declarations.reject_plan.parameters_json_schema",
            "adk": "<absent>",
            "pai": {"type": "object", "properties": {}},
        }
    ]
    # The request carries the cache name and no prefix.
    for client in (adk, pai):
        request = sc.config_dict(client.calls[0]["config"])
        assert request["cached_content"] == "cachedContents/scripted-1"
        assert "system_instruction" not in request and "tools" not in request


async def test_control_a_prefix_with_a_tool_config_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pai_planner.PlannerModel.prefix

    def with_tool_config(self: Any, config: Any) -> types.GenerateContentConfig:
        built = original(self, config)
        built.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        )
        return built

    monkeypatch.setattr(pai_planner.PlannerModel, "prefix", with_tool_config)
    adk, pai = await _prefixes()

    assert [d["path"] for d in sc.diff(prefix_of(adk.creates[0]), prefix_of(pai.creates[0]))] == [
        "declarations.reject_plan.parameters_json_schema",
        "tool_config",
    ]
