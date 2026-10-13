"""`submit_analysis`: stage gating, contracts, the complete stage's outcomes, and attempts.

The tool is driven two ways. Through `planner.agent.build_planner_tools` with
`SchemaDelivery.COMPACT`, so a call runs the real `submission.service` chain
and a stub translation, which counts its calls. And through
`submission.tool.submit_analysis` with a stub `propose`, where a test needs
one particular service answer to see how it is labelled.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest
from google.adk.events import EventActions
from google.adk.sessions.state import State
from google.adk.tools.base_tool import BaseTool
from planner.agent import build_planner_tools
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    RankRequirement,
    UnsupportedOperation,
    load_plan,
    plan_revision,
    record_unsupported,
    save_plan,
)
from planner.records import RankAmong, RankDirection
from planner.settings import SchemaDelivery
from planner.tool_results import Outcome, ProblemCode, ok, problem
from submission.catalogue import catalogue_revision, select_contracts
from submission.tool import (
    ATTEMPT_STATE_PREFIX,
    MAX_ATTEMPTS,
    TOOL_NAME,
    SubmissionOutcome,
    load_attempts,
    submit_analysis,
)

from ..decision_drafts import drafts
from ..planner.test_agent import _translate as _compile


class _Session:
    def __init__(self) -> None:
        self.events: list[Any] = []


class _ToolContext:
    """What the planner's tools read of an ADK `ToolContext`."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.actions = EventActions()
        self.session = _Session()
        self.invocation_id = "turn-1"


class _Translate:
    """A translation that answers `verdict` and counts its calls.

    An `ok` verdict compiles the plan on record first, as the real translation
    does, so `planner.agent.declare_ready` finds it awaiting approval.
    """

    def __init__(self, verdict: dict[str, object] | None = None) -> None:
        self.calls = 0
        self.verdict = verdict or {"outcome": "ok"}

    async def __call__(self, tool_context: Any) -> dict[str, object]:
        self.calls += 1
        if self.verdict.get("outcome") == "ok":
            _compile(tool_context)
        return self.verdict


def _tool(
    *, translate: _Translate | None = None, schema_delivery: SchemaDelivery = SchemaDelivery.COMPACT
) -> BaseTool:
    (tool,) = [
        tool
        for tool in build_planner_tools(translate=translate, schema_delivery=schema_delivery)
        if tool.name in {"propose_plan", TOOL_NAME}
    ]
    return tool


def _plan_arguments(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
) -> dict[str, Any]:
    """A first draft's `propose_plan` arguments, as JSON a model sends."""
    return {
        "question": "How much debt service did Kenya pay each year?",
        "answer_mode": "computed",
        "decisions": [draft.model_dump(mode="json") for draft in drafts([decision()])],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }


async def _call(tool: BaseTool, ctx: _ToolContext, **args: Any) -> dict[str, Any]:
    return await tool.run_async(args=args, tool_context=ctx)  # type: ignore[arg-type]


async def _stub(answer: dict[str, object], **args: Any) -> dict[str, object]:
    """`submit_analysis` over `args`, whose `propose` answers `answer`."""

    async def propose(**_: Any) -> dict[str, object]:
        return answer

    return await submit_analysis({}, args, propose=propose)


# -- stages --------------------------------------------------------------------


async def test_an_intent_with_a_complete_graph_records_and_translates_nothing(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The intent stage never reaches the service, however complete the graph it states."""
    translate = _Translate()
    ctx = _ToolContext()

    answer = await _call(
        _tool(translate=translate),
        ctx,
        submission_stage="intent",
        catalogue_revision=catalogue_revision(),
        **_plan_arguments(decision, requirement, derived_requirement),
    )

    assert answer["outcome"] == SubmissionOutcome.REQUIREMENTS
    assert load_plan(ctx.state) is None
    assert translate.calls == 0
    assert answer["kinds"] == {"annual_debt_service": "dataset", "annual_totals": "aggregate"}
    selection = select_contracts(["dataset", "aggregate"])
    assert json.loads(answer["contracts"]) == selection.payload
    assert answer["contract_digest"] == selection.digest
    assert answer["catalogue_revision"] == catalogue_revision()
    assert answer["action"]["arguments"]["attempt_id"] == answer["attempt_id"]


@pytest.mark.parametrize("stage", [None, "draft", 1], ids=["missing", "unknown", "not-a-string"])
async def test_a_call_without_a_known_stage_is_refused_and_admits_nothing(
    stage: object, decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    translate = _Translate()
    ctx = _ToolContext()
    args = {
        "catalogue_revision": catalogue_revision(),
        **_plan_arguments(decision, requirement, derived_requirement),
    }
    if stage is not None:
        args["submission_stage"] = stage

    answer = await _call(_tool(translate=translate), ctx, **args)

    assert answer["code"] == ProblemCode.INVALID_ARGUMENTS
    assert "contracts" not in answer
    assert load_plan(ctx.state) is None
    assert translate.calls == 0
    (attempt,) = load_attempts(ctx.state)
    assert attempt.stage == (stage if isinstance(stage, str) else None)
    assert attempt.code == ProblemCode.INVALID_ARGUMENTS


async def test_an_intent_lists_unknown_kinds_and_returns_the_known_contracts() -> None:
    answer = await _stub(
        {},
        submission_stage="intent",
        evidence_requirements=[
            {"kind": "dataset", "id": "a"},
            {"kind": "median", "id": "b", "input_id": "a"},
        ],
    )

    assert answer["unknown_kinds"] == ["median"]
    assert set(json.loads(answer["contracts"])["contracts"]) == {"dataset"}
    assert answer["kinds"] == {"a": "dataset", "b": "median"}


@pytest.mark.parametrize(
    ("malformed", "issues"),
    [
        ({"amends_version": "the last one"}, 3),
        ({"attempt_id": "attempt-9", "contract_digest": "f" * 64}, 4),
    ],
    ids=["single-value", "unknown-receipts"],
)
async def test_a_malformed_intent_still_returns_the_known_contracts(
    malformed: dict[str, object], issues: int
) -> None:
    answer = await _stub(
        {},
        submission_stage="intent",
        **malformed,
        evidence_requirements=[
            {"kind": "growth", "id": "g", "input_id": "d"},
            {"kind": "dataset"},
            "a dataset",
            {"kind": "dataset", "id": "d"},
        ],
    )

    assert answer["outcome"] == SubmissionOutcome.REQUIREMENTS
    assert set(json.loads(answer["contracts"])["contracts"]) == {"dataset", "growth"}
    assert len(answer["issues"]) == issues


# -- the complete stage --------------------------------------------------------


@pytest.mark.parametrize("revision", [None, "0" * 64], ids=["missing", "stale"])
@pytest.mark.parametrize(
    "malformed",
    [{}, {"amends_version": "x"}, {"attempt_id": "attempt-9"}],
    ids=["well-formed", "single-value", "receipt"],
)
async def test_a_complete_call_on_another_revision_is_refused_before_any_check(
    revision: str | None,
    malformed: dict[str, object],
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
) -> None:
    translate = _Translate()
    ctx = _ToolContext()
    args = {
        "submission_stage": "complete",
        **_plan_arguments(decision, requirement, derived_requirement),
        **malformed,
    }
    if revision is not None:
        args["catalogue_revision"] = revision

    answer = await _call(_tool(translate=translate), ctx, **args)

    assert answer["code"] == ProblemCode.CATALOGUE_STALE
    assert answer["catalogue_revision"] == catalogue_revision()
    assert set(json.loads(answer["contracts"])["contracts"]) == {"dataset", "aggregate"}
    assert load_plan(ctx.state) is None
    assert translate.calls == 0


async def test_needs_fields_names_every_field_and_the_contracts_of_the_kinds_involved(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    args = _plan_arguments(decision, requirement, derived_requirement)
    dataset, aggregate = args["evidence_requirements"]
    del dataset["measure"]
    aggregate["reduction"] = "median"
    ctx = _ToolContext()

    answer = await _call(
        _tool(), ctx, submission_stage="complete", catalogue_revision=catalogue_revision(), **args
    )

    assert answer["outcome"] == SubmissionOutcome.NEEDS_FIELDS
    paths = {field["path"] for field in answer["fields"]}
    assert "evidence_requirements.0.dataset.measure" in paths
    assert "evidence_requirements.1.aggregate.reduction" in paths
    assert answer["kinds"] == {"annual_debt_service": "dataset", "annual_totals": "aggregate"}
    assert set(json.loads(answer["contracts"])["contracts"]) == {"dataset", "aggregate"}
    assert load_plan(ctx.state) is None


async def test_needs_fields_names_every_field_error_past_the_propose_plan_cap(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Field errors on decisions come first, so a cap would drop the requirements' and their contracts."""
    args = _plan_arguments(decision, requirement, derived_requirement)
    args["decisions"] = [
        {**args["decisions"][0], "slot": f"slot_{index}", "description": "d" * 600}
        for index in range(11)
    ]
    dataset, aggregate = args["evidence_requirements"]
    del dataset["measure"]
    aggregate["reduction"] = "median"

    answer = await _call(
        _tool(),
        _ToolContext(),
        submission_stage="complete",
        catalogue_revision=catalogue_revision(),
        **args,
    )

    assert answer["outcome"] == SubmissionOutcome.NEEDS_FIELDS
    paths = [field["path"] for field in answer["fields"]]
    assert sum(path.startswith("decisions.") for path in paths) == 11
    assert "evidence_requirements.0.dataset.measure" in paths
    assert "evidence_requirements.1.aggregate.reduction" in paths
    assert set(json.loads(answer["contracts"])["contracts"]) == {"dataset", "aggregate"}


async def test_a_compiled_plan_is_ready_for_review(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    translate = _Translate({"outcome": "ok"})
    ctx = _ToolContext()

    answer = await _call(
        _tool(translate=translate),
        ctx,
        submission_stage="complete",
        catalogue_revision=catalogue_revision(),
        **_plan_arguments(decision, requirement, derived_requirement),
    )

    assert answer["outcome"] == SubmissionOutcome.READY_FOR_REVIEW
    assert answer["translation"] == {"outcome": "ok"}
    assert answer["summary"]["version"] == 1
    assert ctx.actions.skip_summarization
    assert answer["version"] == 1
    assert translate.calls == 1
    assert load_attempts(ctx.state)[-1].plan_version == 1


@pytest.mark.parametrize(
    ("sent", "proposed"), [("3", 3), (3, 3), ("the last one", None)], ids=["text", "int", "words"]
)
async def test_a_complete_call_hands_the_service_its_checked_single_values(
    sent: object, proposed: int | None
) -> None:
    calls: list[dict[str, Any]] = []

    async def propose(**arguments: Any) -> dict[str, object]:
        calls.append(arguments)
        return ok(status="draft", version=4)

    answer = await submit_analysis(
        {},
        {
            "submission_stage": "complete",
            "catalogue_revision": catalogue_revision(),
            "answer_mode": "series",
            "amends_version": sent,
        },
        propose=propose,
    )

    if proposed is None:
        assert answer["outcome"] == SubmissionOutcome.NEEDS_FIELDS
        assert calls == []
        return
    assert calls == [{"answer_mode": "series", "amends_version": proposed}]


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (
            ok(status="draft", version=2, translation=ok(), summary={"version": 2}),
            "ready_for_review",
        ),
        (
            ok(
                status="draft",
                version=2,
                translation=ok(),
                submission=problem(
                    ProblemCode.TRANSITION_REFUSED, "not awaiting approval", outcome=Outcome.REFUSED
                ),
            ),
            "recorded",
        ),
        (ok(status="draft", version=2, translation={"outcome": "choice_required"}), "recorded"),
        (ok(status="draft", version=2), "recorded"),
        (
            problem(ProblemCode.REFERRAL_PENDING, "ask", outcome=Outcome.REFUSED),
            "needs_decision",
        ),
        (
            problem(ProblemCode.COMBINATION_UNASKED, "ask", outcome=Outcome.REFUSED),
            "needs_decision",
        ),
        (problem(ProblemCode.BUDGET_EXHAUSTED, "cap", outcome=Outcome.REFUSED), "refused"),
    ],
    ids=[
        "declared-ready",
        "gate-blocked",
        "referred",
        "untranslated",
        "referral-pending",
        "combination-unasked",
        "budget",
    ],
)
async def test_each_service_answer_gets_its_outcome_and_keeps_its_code(
    answer: dict[str, object], outcome: str
) -> None:
    labelled = await _stub(
        answer, submission_stage="complete", catalogue_revision=catalogue_revision()
    )

    assert labelled["outcome"] == outcome
    assert labelled.get("code") == answer.get("code")


@pytest.mark.parametrize(
    ("schema_delivery", "stated"),
    [
        (SchemaDelivery.EAGER, {}),
        (SchemaDelivery.COMPACT, {"submission_stage": "complete"}),
    ],
    ids=["propose_plan", "submit_analysis"],
)
async def test_a_refusal_action_calls_the_proposing_tool_with_what_every_call_states(
    schema_delivery: SchemaDelivery,
    stated: dict[str, str],
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
) -> None:
    operation = next(iter(UnsupportedOperation))
    ctx = _ToolContext()
    record_unsupported(ctx.state, operation)
    tool = _tool(schema_delivery=schema_delivery)
    revision = {"catalogue_revision": catalogue_revision()} if stated else {}

    answer = await _call(
        tool,
        ctx,
        **stated,
        **revision,
        **_plan_arguments(decision, requirement, derived_requirement),
    )

    assert answer["code"] == ProblemCode.DECLINE_MISSING
    assert answer["action"] == {
        "call": tool.name,
        "arguments": {**stated, **revision, "declined": [{"code": operation.value}]},
        "needs": ["declined.ask"],
    }


class _TranslateThenReject(_Translate):
    """An `ok` translation after which the plan on record is rejected, so
    `planner.agent.declare_ready`'s gate refuses it."""

    async def __call__(self, tool_context: Any) -> dict[str, object]:
        verdict = await super().__call__(tool_context)
        plan = load_plan(tool_context.state)
        assert plan is not None
        save_plan(tool_context.state, plan.reject())
        return verdict


@pytest.mark.parametrize(
    ("schema_delivery", "stated"),
    [
        (SchemaDelivery.EAGER, {}),
        (SchemaDelivery.COMPACT, {"submission_stage": "complete"}),
    ],
    ids=["propose_plan", "submit_analysis"],
)
async def test_a_gate_refusal_calls_the_proposing_tool_with_what_every_call_states(
    schema_delivery: SchemaDelivery,
    stated: dict[str, str],
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
) -> None:
    tool = _tool(translate=_TranslateThenReject(), schema_delivery=schema_delivery)
    revision = {"catalogue_revision": catalogue_revision()} if stated else {}

    answer = await _call(
        tool,
        _ToolContext(),
        **stated,
        **revision,
        **_plan_arguments(decision, requirement, derived_requirement),
    )

    assert "summary" not in answer
    if stated:
        assert answer["outcome"] == SubmissionOutcome.RECORDED
    blocked = answer["submission"]
    assert blocked["code"] == ProblemCode.PLAN_REJECTED
    arguments = {**stated, **revision}
    assert blocked["action"] == {
        "call": tool.name,
        **({"arguments": arguments} if arguments else {}),
    }


@pytest.mark.parametrize(
    ("schema_delivery", "stated"),
    [
        (SchemaDelivery.EAGER, {}),
        (SchemaDelivery.COMPACT, {"submission_stage": "complete"}),
    ],
    ids=["propose_plan", "submit_analysis"],
)
async def test_a_shared_decision_refusal_calls_the_proposing_tool_with_what_every_call_states(
    schema_delivery: SchemaDelivery, stated: dict[str, str]
) -> None:
    flows = DatasetRequirement(
        id="flows",
        decision_ids=["scope"],
        measure="aid received",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2024, end=2024),
    )
    ranked = RankRequirement(
        id="ranked",
        decision_ids=["scope"],
        input_id="flows",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
    )
    scope = {
        "slot": "scope",
        "description": "Which year, and among whom.",
        "options": ["2024 among all", "2023 among all"],
        "chosen": "2024 among all",
        "resolved_by": "planner_default",
        "rationale": "The latest year, among every recipient.",
        "fixes": ["period", "among"],
    }
    moved = flows.model_copy(update={"period": Period(start=2023, end=2023)})
    tool = _tool(schema_delivery=schema_delivery)
    revision = {"catalogue_revision": catalogue_revision()} if stated else {}
    ctx = _ToolContext()
    first = await _call(
        tool,
        ctx,
        **stated,
        **revision,
        question="q",
        answer_mode="computed",
        decisions=[scope],
        evidence_requirements=[r.model_dump(mode="json") for r in (flows, ranked)],
    )
    assert "version" in first, first

    refused = await _call(
        tool,
        ctx,
        **stated,
        **revision,
        evidence_requirements=[moved.model_dump(mode="json")],
        amends_version=first["version"],
    )

    assert refused["code"] == ProblemCode.SHARED_DECISION_CONTRADICTED
    arguments = {**stated, **revision}
    assert refused["action"] == {
        "call": tool.name,
        **({"arguments": arguments} if arguments else {}),
        "needs": ["decisions"],
    }


# -- attempts ------------------------------------------------------------------


async def test_every_call_appends_an_attempt_linked_to_the_one_it_corrects(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    tool = _tool(translate=_Translate())
    ctx = _ToolContext()
    arguments = _plan_arguments(decision, requirement, derived_requirement)

    intent = await _call(tool, ctx, submission_stage="intent", **arguments)
    stale = await _call(
        tool,
        ctx,
        submission_stage="complete",
        catalogue_revision="0" * 64,
        attempt_id=intent["attempt_id"],
        **arguments,
    )
    before = load_attempts(ctx.state)
    admitted = await _call(
        tool,
        ctx,
        submission_stage="complete",
        catalogue_revision=catalogue_revision(),
        attempt_id=stale["attempt_id"],
        contract_digest=intent["contract_digest"],
        **arguments,
    )

    assert load_attempts(ctx.state)[:2] == before
    first, second, third = load_attempts(ctx.state)
    assert [first.attempt_id, second.attempt_id, third.attempt_id] == [
        intent["attempt_id"],
        stale["attempt_id"],
        admitted["attempt_id"],
    ]
    assert [first.stage, second.stage, third.stage] == ["intent", "complete", "complete"]
    assert [first.outcome, second.outcome, third.outcome] == [
        "requirements",
        "invalid_input",
        "ready_for_review",
    ]
    assert [first.corrects, second.corrects, third.corrects] == [
        None,
        first.attempt_id,
        second.attempt_id,
    ]
    assert [first.plan_version, second.plan_version, third.plan_version] == [None, None, 1]
    assert first.contract_digest == intent["contract_digest"]
    assert len({first.request_digest, second.request_digest, third.request_digest}) == 3


async def test_a_session_keeps_the_latest_attempts_whole_and_numbers_on() -> None:
    state: dict[str, Any] = {}

    async def propose(**_: Any) -> dict[str, object]:
        return {}

    for _ in range(MAX_ATTEMPTS):
        await submit_analysis(state, {"submission_stage": "intent"}, propose=propose)
    kept = load_attempts(state)[2:]
    first = await submit_analysis(state, {"submission_stage": "intent"}, propose=propose)
    second = await submit_analysis(state, {"submission_stage": "intent"}, propose=propose)

    assert sum(key.startswith(ATTEMPT_STATE_PREFIX) for key in state) == MAX_ATTEMPTS
    assert load_attempts(state)[:-2] == kept
    assert [first["attempt_id"], second["attempt_id"]] == [
        f"attempt-{MAX_ATTEMPTS + 1}",
        f"attempt-{MAX_ATTEMPTS + 2}",
    ]
    assert [attempt.sequence for attempt in load_attempts(state)] == list(
        range(3, MAX_ATTEMPTS + 3)
    )


async def test_concurrent_calls_in_one_session_answer_distinct_attempts() -> None:
    """Parallel calls share the session's state, each through its own delta, as ADK runs them.

    ADK merges the deltas of parallel calls key by key, the later call's
    winning, so each call's attempt survives either merge order only if no
    two deltas write one key.
    """
    session: dict[str, Any] = {}
    deltas: list[dict[str, Any]] = [{}, {}]

    async def propose(**_: Any) -> dict[str, object]:
        await asyncio.sleep(0)
        return problem(ProblemCode.INVALID_PLAN, "refused", outcome=Outcome.INVALID_INPUT)

    args = {"submission_stage": "complete", "catalogue_revision": catalogue_revision()}
    answers = await asyncio.gather(
        *(submit_analysis(State(session, delta), args, propose=propose) for delta in deltas)
    )

    ids = sorted(answer["attempt_id"] for answer in answers)
    assert ids == ["attempt-1", "attempt-2"]
    assert [len(delta) for delta in deltas] == [1, 1]
    for order in (deltas, deltas[::-1]):
        stored: dict[str, Any] = {}
        for delta in order:
            stored.update(delta)
        assert [attempt.attempt_id for attempt in load_attempts(stored)] == ids


async def test_attempts_another_version_stored_are_read_or_skipped() -> None:
    """An unknown key is ignored, a missing optional field is `None`, an unreadable record is skipped."""
    state: dict[str, Any] = {
        f"{ATTEMPT_STATE_PREFIX}1": {
            "sequence": 1,
            "attempt_id": "attempt-1",
            "request_digest": "0" * 64,
            "catalogue_revision": catalogue_revision(),
            "outcome": "requirements",
            "written_by": "another version",
        },
        f"{ATTEMPT_STATE_PREFIX}2": {"sequence": 2, "attempt_id": "attempt-2"},
        f"{ATTEMPT_STATE_PREFIX}3": "attempt-3",
    }

    async def propose(**_: Any) -> dict[str, object]:
        return {}

    answer = await submit_analysis(
        state, {"submission_stage": "intent", "attempt_id": "attempt-1"}, propose=propose
    )

    assert answer["attempt_id"] == "attempt-2"
    assert "issues" not in answer
    first, second = load_attempts(state)
    assert (first.attempt_id, first.code, first.plan_version) == ("attempt-1", None, None)
    assert second.corrects == "attempt-1"


@pytest.mark.parametrize(
    "receipt",
    [{"attempt_id": "attempt-9"}, {"contract_digest": "f" * 64}],
    ids=["attempt", "digest"],
)
async def test_a_receipt_no_answer_gave_is_refused(
    receipt: dict[str, str], decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    translate = _Translate()
    ctx = _ToolContext()

    answer = await _call(
        _tool(translate=translate),
        ctx,
        submission_stage="complete",
        catalogue_revision=catalogue_revision(),
        **receipt,
        **_plan_arguments(decision, requirement, derived_requirement),
    )

    assert answer["code"] == ProblemCode.INVALID_ARGUMENTS
    assert load_plan(ctx.state) is None
    assert translate.calls == 0


# -- delivery parity -----------------------------------------------------------


async def test_both_deliveries_record_the_same_plan(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`propose_plan` and `submit_analysis`'s complete stage run one chain, so one plan."""
    arguments = _plan_arguments(decision, requirement, derived_requirement)
    eager, compact = _ToolContext(), _ToolContext()

    await _call(_tool(schema_delivery=SchemaDelivery.EAGER), eager, **arguments)
    await _call(
        _tool(),
        compact,
        submission_stage="complete",
        catalogue_revision=catalogue_revision(),
        **arguments,
    )

    eager_plan, compact_plan = load_plan(eager.state), load_plan(compact.state)
    assert eager_plan is not None and compact_plan is not None
    assert plan_revision(eager_plan) == plan_revision(compact_plan)
