"""`workflow_outcome`: a custom ADK eval metric asserting the compiled outcome in code.

The rubric metrics judge behaviour a judge model reads off the trace --
whether a call was made at all, whether a judgement was answered. None of
them can assert on what a tool actually answered: a rubric asking whether
the plan was translated can be satisfied by a `propose_plan` whose
translation came back referred. That gap is what this metric closes.

How a case states its expectation
----------------------------------
`adk eval` has no way to assert on session state (the compiled `Workflow`
lives in `planner:plan`, not in anything an `Invocation` carries), so this
metric works from the one thing every run does carry: the tool responses in
the trace. A case states what it expects by adding a `FunctionResponse` to a
turn's `intermediate_data.tool_responses` in the `.evalset.json` file, named
for one of `SCORED_TOOLS` --
`{"name": "translation", "response": {"outcome": "ok"}}` for a plan that has to
compile, `{"name": "translation", "response": {"outcome": "invalid_input",
"kind": ..., "requirement_id": ...}}` for one that has to come back referred,
`{"name": "propose_plan", "response": {"outcome": "ok", "version": 2}}` for a
planner turn that has to amend rather than start over, or `{"name":
"reject_plan", "response": {"outcome": "ok"}}` for one that has to record a
rejection. `translation` is no tool of the agent's: it is the verdict an
accepted `propose_plan` carries under its own `translation` key, `ok` once the
plan compiled, or the referral itself, whose `outcome` is one of
`translator.referral.OUTCOME_OF`'s. `_with_translations` lists each such
verdict as a response of its own. This metric finds the *actual*
response of that same name -- the last one whose `outcome` is the one the
case states, so a
later proposal that came back differently does not hide it (`_by_name`)
-- and checks it against the expectation as a **subset**: every key the case
names has to be present and equal on the actual response; keys the case
leaves out (an operation id the model chose, `reason` prose, `sentence`,
`searched_as`, ...) are not compared. A list is compared element-wise, in
order, and must be the same length -- `steps` is checked by `tool` name
alone, per step, in order, since the tool sequence is the thing under test.
`sources` alone is compared as an unordered collection: the compiler emits
its sources in a canonical order of its own, and a case states which
`spec_id`s were selected, not the order the compiler happens to put them
in.

A turn is also failed when the agent hedged: a reject case that also had a
`propose_plan` declare a plan ready (its answer carries `summary`,
`planner.agent.declare_ready`) -- an agent cannot both record a rejection and
declare the same plan ready for the user's answer. A `translation` case hedges
the same way -- a case expecting `ok` that also had a proposal's translation
answer a referral, or a referral case that also had one answer `ok` -- since `_by_name` picks whichever response carries the stated
outcome and would otherwise miss a second, contradicting proposal the model
also made on the same turn (`_hedged`). A plan-proposal case has nothing to hedge
against: declaring a plan ready approves and executes nothing itself.

What ADK actually populates in `intermediate_data`
----------------------------------------------------
Two different types implement `IntermediateDataType`
(`google/adk/evaluation/eval_case.py:76`): `IntermediateData`, with
`tool_uses`/`tool_responses` fields, and `InvocationEvents`, wrapping a raw
list of `InvocationEvent`s. A case's own JSON authors the *expected* side as
`IntermediateData` -- that is what `tool_uses`/`tool_responses` in a
`.evalset.json` turn deserialize into.

A live run's *actual* side is never that. `EvaluationGenerator.
convert_events_to_eval_invocations`
(`google/adk/evaluation/evaluation_generator.py:673-744`) builds every actual
`Invocation` with `intermediate_data=InvocationEvents(invocation_events=...)`
-- one `InvocationEvent(author=event.author, content=event.content)` per
event carrying a function call, function response, text, or inline data,
across the *whole* invocation, not filtered to the root agent. `IntermediateData.
tool_uses`/`tool_responses` are therefore always empty on an actual run; only
`invocation_events` is populated.

Both metric authors (this module and `TrajectoryEvaluator`) are spared having
to branch on which type they were handed: `get_all_tool_calls` and
`get_all_tool_responses` (`google/adk/evaluation/eval_case.py:191-234`) read
either representation uniformly, pulling function calls and responses out of
`invocation_events`' parts when given an `InvocationEvents`, or straight out
of the two list fields when given an `IntermediateData`. This module uses
`get_all_tool_responses` for exactly that reason, on both the actual and the
expected side -- so a case's own JSON stays the more readable
`IntermediateData` shape while the metric reads a live run's `InvocationEvents`
the same way.

Registered as a custom metric (`google/adk/evaluation/eval_config.py`):
declare it under `criteria` (`{"workflow_outcome": 1.0}`) and under
`custom_metrics` (`{"workflow_outcome": {"code_config": {"name":
"harness_translator_v1.outcome_metric.workflow_outcome"}}}`) in a set's
`test_config.json`. `_get_metric_function`
(`google/adk/evaluation/custom_metric_evaluator.py:31-42`) imports that dotted
path with `importlib.import_module`, which needs `evals/agents` on `sys.path`
to resolve `harness_translator_v1` as a top-level package -- the `__init__.py`
of every set that turns it on (`harness_pair_v1`, `harness_planner_v2`)
inserts it for exactly this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from google.adk.evaluation.eval_case import Invocation, get_all_tool_responses
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus, EvaluationResult, PerInvocationResult
from google.genai import types as genai_types

# The tools whose response a case may state an expectation on. A
# `tool_responses` entry naming anything else is not an expectation this
# metric scores -- `test_eval_sets.py` imports this to refuse such a turn.
# `translation` is the verdict an accepted proposal returns under its own
# `translation` key: `ok` or the referral translation stopped on. No tool of
# that name exists; `_with_translations` reads it off the proposal's response.
TRANSLATION = "translation"
SCORED_TOOLS = (
    "propose_plan",
    "reject_plan",
    TRANSLATION,
)

# The one list a case's expectation names whose order is the compiler's,
# not the case's: see `matches_subset`.
_UNORDERED_KEYS = frozenset({"sources"})


# The default threshold, used only when a `test_config.json` names this
# metric in `criteria` with a bare float rather than a `BaseCriterion`.
# `_CustomMetricEvaluator.evaluate_invocations` clears `eval_metric.threshold`
# before calling this function (`custom_metric_evaluator.py:59`), but leaves
# `eval_metric.criterion` -- a `BaseCriterion`, whose own `.threshold` field
# still carries the configured value -- untouched, so that is where this
# reads it from.
_DEFAULT_THRESHOLD = 1.0


def matches_subset(expected: object, actual: object) -> bool:
    """Whether `actual` agrees with `expected` on every key or item `expected` names.

    A dict matches when every one of `expected`'s keys is present in `actual`
    with a matching value -- a key `expected` leaves out is never checked. A
    list matches when `actual` is the same length and each pair matches by
    position; a shorter or longer `actual` fails, since a case names every
    step it expects, in order. The one exception is a list under a key in
    `_UNORDERED_KEYS` (`sources`), matched as a collection: same length, and
    every expected item matched to a distinct actual one, in any order --
    the compiler orders sources canonically and a case should not have to
    know how. Anything else matches by equality.

    `services/harness/tests/test_eval_outcomes.py` imports this to check the
    same real tool responses against the same eval-set expectations with no
    model in the loop, so a case's expected outcome and this metric's notion
    of "matches" never drift apart.
    """
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        actual_by_key = dict(actual)
        return all(
            key in actual_by_key and _matches_value(key, value, actual_by_key[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches_subset(e, a) for e, a in zip(expected, actual, strict=True))
        )
    return expected == actual


def _matches_value(key: object, expected: object, actual: object) -> bool:
    if key in _UNORDERED_KEYS and isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and _matches_unordered(expected, actual)
    return matches_subset(expected, actual)


def _matches_unordered(expected: Sequence[object], actual: Sequence[object]) -> bool:
    """Whether every item of `expected` matches a distinct item of `actual`, in any order.

    Tries each unmatched actual item for the first expected one and recurses
    on the rest, so an expectation loose enough to match several actual items
    still finds an assignment when one exists. The lists are a handful of
    sources long, so the search is cheap.
    """
    if not expected:
        return True
    first, rest = expected[0], expected[1:]
    return any(
        matches_subset(first, candidate)
        and _matches_unordered(rest, [*actual[:index], *actual[index + 1 :]])
        for index, candidate in enumerate(actual)
    )


def _outcome(response: genai_types.FunctionResponse) -> object:
    return response.response.get("outcome") if isinstance(response.response, dict) else None


def _by_name(
    responses: list[genai_types.FunctionResponse], expected: genai_types.FunctionResponse
) -> genai_types.FunctionResponse | None:
    """The actual response to grade against `expected`, or `None` if none shares its name.

    The last response of `expected`'s name whose `outcome` is the one
    `expected` states; failing that, the last of that name. Grading the last
    response alone scored zero for a translator that compiled correctly and
    then made one more proposal the tool refused -- a second proposal
    the planner turned down leaves the accepted workflow exactly where it
    was, so the refusal is not the turn's outcome. Preferring the response
    that carries the stated outcome only ever turns that zero into a one: a
    turn whose last response already carries the stated outcome grades the
    same response as before, and a turn with none of that outcome still
    grades the last one, which fails on `outcome` as it always did. It is
    not a loosened gate because the exclusivity check, `_hedged`, reads
    every response on the turn rather than the graded one: a mapping case
    that also referred, or a referral case that also had a proposal
    accepted, still scores zero whichever response this picks.
    """
    matching = [response for response in responses if response.name == expected.name]
    if not matching:
        return None
    stated = _outcome(expected)
    preferred = [response for response in matching if _outcome(response) == stated]
    return (preferred or matching)[-1]


def _with_translations(
    responses: list[genai_types.FunctionResponse],
) -> list[genai_types.FunctionResponse]:
    """`responses` as a case names them, each proposal's verdict following it as a `translation` response.

    A verdict is lifted from any response carrying one, as
    `harness.planner_seam` reads it; a proposal the planner refused ran no
    translation and adds nothing.
    """
    listed: list[genai_types.FunctionResponse] = []
    for response in responses:
        listed.append(response)
        answer = response.response
        verdict = answer.get("translation") if isinstance(answer, dict) else None
        if isinstance(verdict, dict):
            listed.append(genai_types.FunctionResponse(name=TRANSLATION, response=verdict))
    return listed


def _is_referral(response: genai_types.FunctionResponse) -> bool:
    """Whether `response` is a translation referral: the one answer naming a `kind`."""
    return (
        response.name == TRANSLATION
        and isinstance(response.response, dict)
        and "kind" in response.response
    )


def _answered_by(response: genai_types.FunctionResponse, name: str, *outcomes: str) -> bool:
    """Whether `response` is `name`'s own answer, carrying one of `outcomes`."""
    if response.name != name or not isinstance(response.response, dict):
        return False
    return response.response.get("outcome") in outcomes


def _declared_ready(response: genai_types.FunctionResponse) -> bool:
    """Whether `response` is a `propose_plan` answer that declared the plan ready.

    `planner.agent.declare_ready` adds `summary` to the answer of a proposal
    whose translation compiled and whose plan the gate let through.
    """
    return (
        response.name == "propose_plan"
        and isinstance(response.response, dict)
        and "summary" in response.response
    )


def _stated_expectation(expected: Invocation) -> genai_types.FunctionResponse | None:
    """The last response `expected` states for a tool in `SCORED_TOOLS`, if any."""
    responses = [
        response
        for response in get_all_tool_responses(expected.intermediate_data)
        if response.name in SCORED_TOOLS
    ]
    return responses[-1] if responses else None


def _hedged(
    expected: genai_types.FunctionResponse, actual_responses: list[genai_types.FunctionResponse]
) -> bool:
    """Whether `actual_responses` covers itself alongside the outcome `expected` wants.

    A case expecting the translation to answer `ok` is hedged by a translation
    that also answered a referral (`_is_referral`); a referral case is
    hedged by a translation that also answered `ok`. A translation that
    answered with a problem names no `kind`, wrote nothing, and hedges
    nothing. A reject case is hedged by a `propose_plan` that declared a
    plan ready (`_declared_ready`), and an agent cannot both do that and
    record a rejection on the same turn. A proposal whose readiness the gate
    refused -- blocked by the very rejection it is paired with, among other
    reasons -- declared nothing and hedges nothing. A plan-proposal case has
    no hedge to check: declaring a plan ready approves nothing and executes
    nothing itself.
    """
    match expected.name:
        case "translation" if _outcome(expected) == "ok":
            return any(_is_referral(r) for r in actual_responses)
        case "translation" if _is_referral(expected):
            return any(_answered_by(r, TRANSLATION, "ok") for r in actual_responses)
        case "reject_plan":
            return any(_declared_ready(r) for r in actual_responses)
        case _:
            return False


def _turn_score(actual: Invocation, expected: Invocation) -> float | None:
    """The score for one turn, or `None` when `expected` states no expectation.

    `None` signals "not applicable" rather than pass or fail -- a turn this
    metric was never meant to check, which `test_eval_sets.py`'s
    `test_a_set_configuring_workflow_outcome_states_an_expectation_on_every_turn`
    guards against ever actually occurring for a set that turns this metric
    on.
    """
    expected_response = _stated_expectation(expected)
    if expected_response is None:
        return None

    actual_responses = _with_translations(get_all_tool_responses(actual.intermediate_data))
    actual_response = _by_name(actual_responses, expected_response)
    matched = actual_response is not None and matches_subset(
        expected_response.response, actual_response.response
    )
    return 1.0 if matched and not _hedged(expected_response, actual_responses) else 0.0


def workflow_outcome(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: Any = None,
) -> EvaluationResult:
    """Score each turn on whether its actual response for the `SCORED_TOOLS` tool
    the case names matches the one the case's own `intermediate_data.tool_responses`
    expects, as a subset (`matches_subset`) -- see this module's docstring for
    what a case states and why ADK's rubric metrics cannot make this
    assertion themselves.

    Args:
        eval_metric: This metric's configuration; only `criterion.threshold`
            is read, since `threshold` itself arrives cleared.
        actual_invocations: The turns the agent actually produced.
        expected_invocations: The case's own turns, carrying the expectation
            in `intermediate_data.tool_responses`.
        conversation_scenario: Unused; this metric only scores a static
            conversation.

    Returns:
        `overall_score` 1.0 only when every scored turn matched, else 0.0 --
        this is a correctness gate, not an average to partially credit.
        `overall_eval_status` and each `PerInvocationResult.eval_status` are
        set from `eval_metric.criterion.threshold` (default 1.0). A turn
        whose expected invocation names no response for a tool in
        `SCORED_TOOLS` scores `None` (not evaluated) and is left out of
        `overall_score`.

    Raises:
        ValueError: If `expected_invocations` is `None` or a different length
            from `actual_invocations` -- this metric has nothing to check a
            turn's outcome against without it.
    """
    if expected_invocations is None:
        raise ValueError("workflow_outcome needs expected_invocations to compare against.")
    threshold = (
        eval_metric.criterion.threshold
        if eval_metric.criterion is not None and eval_metric.criterion.threshold is not None
        else _DEFAULT_THRESHOLD
    )

    per_invocation_results: list[PerInvocationResult] = []
    scored: list[float] = []
    for actual, expected in zip(actual_invocations, expected_invocations, strict=True):
        score = _turn_score(actual, expected)
        if score is not None:
            scored.append(score)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=(
                    EvalStatus.NOT_EVALUATED
                    if score is None
                    else EvalStatus.PASSED
                    if score >= threshold
                    else EvalStatus.FAILED
                ),
            )
        )

    overall_score = 1.0 if scored and all(s >= threshold for s in scored) else 0.0
    overall_status = (
        EvalStatus.NOT_EVALUATED
        if not scored
        else EvalStatus.PASSED
        if overall_score >= threshold
        else EvalStatus.FAILED
    )
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=overall_status,
        per_invocation_results=per_invocation_results,
    )
