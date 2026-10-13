"""Tests for `executor.run`: the step loop, retries, checks and transitions.

Every case runs an approved retrieve-then-aggregate plan against a scripted
surface carrying the real servers' schemas, so the executor's own surface
check passes and what is asserted is how it handles what the tools answer.
`execute_approved_plan` is the one public entry, so every test drives the run
through it; `_run` is a thin wrapper that returns just the outcome, for tests
that do not care about the finished plan's status.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx2
import pyarrow as pa
import pytest
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from common.materialisation import FrameRef
from compiler.capabilities import CAPABILITIES, ToolCapability
from compiler.compile import compile_plan
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
)
from data_mcp import stat_var_list
from data_mcp.client import UpstreamClient
from data_mcp.deadline import CallDeadline
from data_mcp.manifest import strip_manifest
from data_mcp.series_statistics import SERIES_STATISTICS_ARROW_SCHEMA
from data_mcp.settings import Settings as DataMcpSettings
from data_mcp.stat_var_list import STAT_VARS_OMITTED_KEY
from executor.classify import BACKOFF_SECONDS, MAX_RETRY_AFTER_SECONDS, retry_delay
from executor.completion import FramePublished, ResultPublisher
from executor.ledger import Attempt, InMemoryRunLedger, RunChange, RunRecord
from executor.outcomes import Accepted, MethodChoiceNeeded, RediscoveryNeeded, RefusalKind, Refused
from executor.run import (
    EVIDENCE_ALLOWANCE_S,
    MAX_ATTEMPTS_PER_STEP,
    RunAbandonedError,
    current_step_deadline,
    default_fetch_series_stats,
    execute_approved_plan,
    step_budget_s,
)
from planner.operations import (
    AggregateOperation,
    GroupBy,
    Reduction,
    RetrieveOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    Coverage,
    DatasetRequirement,
    Decision,
    Plan,
    PlanStatus,
    plan_revision,
)
from planner.workflow import SelectedSource, ToolId

from .conftest import RESULT_FRAME, SPEC_KENYA, SPEC_POPULATION, ScriptedSurface, ok, refusal

PlanFactory = Callable[..., Plan]
ManifestFactory = Callable[..., dict[str, Any]]
RequirementFactory = Callable[..., DatasetRequirement]


class _Sleeps:
    """Records the waits the executor asks for instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _floor(low: float, high: float) -> float:
    """A jitter that always draws its lower bound, so each wait is exactly `retry_delay`."""
    del high
    return low


def _surface(
    real_schemas: list[ToolSchema], manifest: dict[str, Any], **overrides: Any
) -> ScriptedSurface:
    responses: dict[str, Any] = {"get_data": [ok(manifest)], "aggregate": [ok(RESULT_FRAME)]}
    return ScriptedSurface(schemas=real_schemas, responses=responses | overrides)


async def _run(plan: Plan, *, surface: ScriptedSurface, sleep: _Sleeps, **kwargs: Any) -> Any:
    """Run `plan` through `execute_approved_plan` and return just the outcome.

    `**kwargs` forwards to `execute_approved_plan` -- `capabilities`, for a
    test that needs a capability row other than the production table's own.
    """
    _, outcome = await execute_approved_plan(plan, surface=surface, sleep=sleep, **kwargs)
    return outcome


async def test_a_run_binds_the_retrieved_frame_into_the_verb_and_is_accepted(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    surface = _surface(real_schemas, manifest())

    outcome = await _run(plan, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    assert outcome.revision == plan_revision(plan)
    (get_data_call, aggregate_call) = surface.calls
    assert get_data_call == (
        "get_data",
        {"spec_id": SPEC_KENYA, "place_types": True},
    )
    # The binding resolved to the frame get_data's manifest named.
    assert aggregate_call[1]["frame_ref"] == {
        "spec_id": SPEC_KENYA,
        "content_version": "cv_" + "1" * 32,
    }
    assert aggregate_call[1]["aggregation"] == "sum"
    assert [step.step_id for step in outcome.steps] == ["debt", "totals"]
    assert outcome.results["totals"].spec_id == RESULT_FRAME["spec_id"]
    # `get_data`'s structured result is a `Manifest`, which carries no `rows`
    # field of its own kind; `aggregate`'s does, read off `RESULT_FRAME`.
    assert outcome.steps[0].row_count is None
    assert outcome.steps[1].row_count == RESULT_FRAME["rows"]
    (evidence,) = outcome.evidence
    assert evidence.requirement_id == "annual_debt_service"
    assert evidence.frame_ref.spec_id == SPEC_KENYA
    assert evidence.stat_vars == ("DebtServicePaid",)
    assert evidence.citation.startswith("World Bank")
    # The source's unit states a currency and no price basis, so the run
    # records that as the one thing it could not confirm (dfo-ai#193).
    assert [gap.code.value for gap in evidence.gaps] == ["unit_price_basis_unverified"]


async def _published(frame: FrameRef) -> FramePublished:
    return FramePublished(replay_key="rpk_" + "e" * 64)


@pytest.mark.parametrize(
    ("publisher", "status"),
    [(_published, PlanStatus.COMPLETED), (None, PlanStatus.EXECUTING)],
    ids=["published", "publication_unfinished"],
)
async def test_execute_approved_plan_records_the_run_in_the_plans_status(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    publisher: ResultPublisher | None,
    status: PlanStatus,
) -> None:
    finished, outcome = await execute_approved_plan(
        approved_plan(),
        surface=_surface(real_schemas, manifest()),
        publisher=publisher,
        sleep=_Sleeps(),
    )

    assert isinstance(outcome, Accepted)
    assert finished.status is status


async def test_execute_approved_plan_refuses_a_plan_that_is_not_approved(
    approved_plan: PlanFactory, real_schemas: list[ToolSchema]
) -> None:
    pending = approved_plan().model_copy(
        update={"status": PlanStatus.PENDING_APPROVAL, "approved_revision": None}
    )
    surface = ScriptedSurface(schemas=real_schemas, responses={})

    finished, outcome = await execute_approved_plan(pending, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.NOT_APPROVED
    assert finished is pending
    assert surface.calls == []


async def test_a_plan_whose_content_changed_after_approval_does_not_run(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    tampered = approved_plan().model_copy(update={"question": "a question nobody approved"})
    surface = _surface(real_schemas, manifest())

    plan, outcome = await execute_approved_plan(tampered, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.STALE_REVISION
    assert plan.status is PlanStatus.APPROVED
    assert surface.calls == []


async def test_a_tool_whose_schema_changed_since_compiling_stops_the_run_before_any_call(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    narrowed = [
        schema.model_copy(update={"parameters": {"type": "object"}})
        if schema.name == "aggregate"
        else schema
        for schema in real_schemas
    ]
    surface = _surface(narrowed, manifest())

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == "totals"
    assert surface.calls == []


@pytest.mark.parametrize(
    "update",
    [
        {"output_schema": {"type": "object", "properties": {"x": {}}}},
        {"version": "2.0.0"},
    ],
    ids=["output_schema", "version"],
)
async def test_a_tools_output_schema_or_version_changing_since_compiling_stops_the_run(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    update: dict[str, Any],
) -> None:
    drifted = [
        schema.model_copy(update=update) if schema.name == "aggregate" else schema
        for schema in real_schemas
    ]
    surface = _surface(drifted, manifest())

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == "totals"
    assert surface.calls == []


async def test_a_tool_whose_schema_cannot_be_hashed_is_a_surface_changed_refusal(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `aggregate`'s live schema now embeds an integer RFC 8785 cannot encode.
    # The executor's own surface check refuses typed (`SURFACE_CHANGED`)
    # rather than letting `UnhashableSchemaError` escape uncaught.
    unhashable = [
        schema.model_copy(update={"parameters": {**schema.parameters, "maximum": 2**53}})
        if schema.name == "aggregate"
        else schema
        for schema in real_schemas
    ]
    surface = _surface(unhashable, manifest())

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == "totals"
    assert "cannot be compared" in outcome.detail
    assert surface.calls == []


async def test_a_transport_failure_on_get_data_is_not_retried(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `get_data`'s mint is not safely retryable
    # (services/data-mcp/src/data_mcp/get_data_tool.py:715-733): a transport
    # failure on it stops the run after one attempt, not
    # `MAX_ATTEMPTS_PER_STEP`.
    surface = _surface(real_schemas, manifest(), get_data=[ToolSurfaceUnavailableError("gone")])

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    assert len(surface.calls) == 1


async def test_a_transport_failure_on_a_retryable_tool_is_retried_and_the_run_goes_on(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `aggregate` has no capability row saying otherwise, so it keeps today's
    # retry behaviour; `get_data`'s own non-retryable case is
    # `test_a_transport_failure_on_get_data_is_not_retried`.
    sleeps = _Sleeps()
    surface = _surface(
        real_schemas,
        manifest(),
        aggregate=[ToolSurfaceUnavailableError("gone"), ok(RESULT_FRAME)],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=sleeps)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[1].attempts == 2
    assert len(sleeps.delays) == 1


async def test_a_transport_that_never_answers_stops_the_run_after_the_last_attempt(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(real_schemas, manifest(), aggregate=[ToolSurfaceUnavailableError("gone")])

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    # One get_data call, then `aggregate` retried up to the cap.
    assert len(surface.calls) == 1 + MAX_ATTEMPTS_PER_STEP


async def test_a_refusal_that_names_a_delay_is_retried_after_it(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # On `get_data`, deliberately: `retry_on_transport_failure` covers
    # transport failures only. A refusal carrying `retry_after_delay` means
    # the server answered and the mint did not happen, so that retry stays
    # even on a tool a transport failure would not retry
    # (`test_a_transport_failure_on_get_data_is_not_retried`).
    sleeps = _Sleeps()
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[
            refusal("service_unavailable", "retry_after_delay", retry_after_s=7),
            ok(manifest()),
        ],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=sleeps, jitter=_floor)

    assert isinstance(outcome, Accepted), outcome
    assert sleeps.delays == [7.0]


async def test_a_step_reports_each_answer_and_the_retry_it_is_about_to_make(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[
            refusal("service_unavailable", "retry_after_delay", retry_after_s=7),
            ok(manifest()),
        ],
    )
    heard: list[tuple[str, str | None]] = []

    _, outcome = await execute_approved_plan(
        approved_plan(),
        surface=surface,
        sleep=_Sleeps(),
        jitter=_floor,
        on_progress=lambda step_id, report: heard.append((step_id, report.message)),
    )

    assert isinstance(outcome, Accepted), outcome
    assert heard == [
        ("debt", "Result received; checking it"),
        ("debt", f"Retrying in 7s (attempt 2 of {MAX_ATTEMPTS_PER_STEP})"),
        ("debt", "Result received; checking it"),
        ("totals", "Result received; checking it"),
    ]


async def test_a_raising_progress_callback_never_fails_the_run(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[
            refusal("service_unavailable", "retry_after_delay", retry_after_s=7),
            ok(manifest()),
        ],
    )
    heard: list[str] = []

    def _raises(step_id: str, report: ProgressReport) -> None:
        heard.append(step_id)
        raise RuntimeError("relay closed")

    _, outcome = await execute_approved_plan(
        approved_plan(), surface=surface, sleep=_Sleeps(), on_progress=_raises
    )

    assert isinstance(outcome, Accepted), outcome
    assert heard == ["debt", "debt", "debt", "totals"]


class _ReportingSurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` sends two progress reports naming
    the `spec_id` it was called with, yielding to the event loop after each,
    so two concurrent retrievals interleave their reports."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data" and on_progress is not None:
            for tick in range(2):
                on_progress(ProgressReport(progress=tick, message=f"{arguments['spec_id']} {tick}"))
                await asyncio.sleep(0)
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def test_two_concurrent_steps_report_under_their_own_step_ids(
    two_source_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # "debt" and "pop" share wave 0, so their calls run at once and their
    # reports interleave on one callback.
    surface = _ReportingSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(RESULT_FRAME)]},
    )
    heard: list[tuple[str, str | None]] = []

    await execute_approved_plan(
        two_source_plan(),
        surface=surface,
        sleep=_Sleeps(),
        on_progress=lambda step_id, report: heard.append((step_id, report.message)),
    )

    assert heard == [
        ("debt", f"{SPEC_KENYA} 0"),
        ("pop", f"{SPEC_POPULATION} 0"),
        ("debt", f"{SPEC_KENYA} 1"),
        ("pop", f"{SPEC_POPULATION} 1"),
        ("debt", "Result received; checking it"),
        ("pop", "Result received; checking it"),
    ]


@pytest.mark.parametrize(
    ("named", "waited"),
    [(3600, MAX_RETRY_AFTER_SECONDS), (-5, BACKOFF_SECONDS), (float("inf"), BACKOFF_SECONDS)],
    ids=["an-hour", "negative", "infinite"],
)
async def test_a_delay_a_tool_names_is_capped_and_an_unusable_one_falls_back_to_the_backoff(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    named: float,
    waited: float,
) -> None:
    sleeps = _Sleeps()
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[
            refusal("service_unavailable", "retry_after_delay", retry_after_s=named),
            ok(manifest()),
        ],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=sleeps, jitter=_floor)

    assert isinstance(outcome, Accepted), outcome
    assert sleeps.delays == [waited]


class _RefusedOnceSurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` refuses each `spec_id`'s first call
    with `retry_after_delay` and 2s named, the way a full row budget refuses
    every call that reaches it, then answers that spec's own manifest."""

    def __init__(self, *, manifests: Mapping[str, dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.manifests = dict(manifests)
        self.refused: set[str] = set()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.calls.append((tool.name, dict(arguments)))
        spec_id = str(arguments["spec_id"])
        if spec_id not in self.refused:
            self.refused.add(spec_id)
            return refusal("service_unavailable", "retry_after_delay", retry_after_s=2)
        return ok(self.manifests[spec_id])


async def test_two_calls_refused_together_retry_at_different_moments(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    # Both retrievals share wave 0 and are refused with the same named 2s. A
    # fixed wait would send both retries at the same instant, into the same
    # full budget; the jitter draws each wait between 2s and 4s.
    debt, population = source(), population_source()
    surface = _RefusedOnceSurface(
        schemas=real_schemas,
        responses={"aggregate": [ok(RESULT_FRAME)]},
        manifests={
            debt.spec_id: manifest_for(debt),
            population.spec_id: manifest_for(population),
        },
    )
    sleeps = _Sleeps()

    outcome = await _run(
        two_source_plan(), surface=surface, sleep=sleeps, jitter=random.Random(7).uniform
    )

    assert isinstance(outcome, Accepted), outcome
    assert len(sleeps.delays) == 2
    assert all(2.0 <= delay <= 4.0 for delay in sleeps.delays)
    assert sleeps.delays[0] != sleeps.delays[1]


# The step budget: every attempt, retry delay and evidence fetch under one deadline


def test_the_step_budget_covers_every_attempt_at_its_timeout_and_the_longest_retry_delays() -> None:
    # The longest wait `retry_delay` can ask for before each retry: a named
    # delay far past the cap, or none at all and the backoff alone.
    longest_delays = sum(
        max(retry_delay({"retry_after_s": 10**9}, attempt), retry_delay({}, attempt))
        for attempt in range(MAX_ATTEMPTS_PER_STEP - 1)
    )
    (get_data,) = (c for c in CAPABILITIES if c.tool.name == "get_data")
    assert get_data.call_timeout_s is not None

    budget = step_budget_s(get_data.call_timeout_s, evidence=True)

    assert budget == (
        MAX_ATTEMPTS_PER_STEP * get_data.call_timeout_s + longest_delays + EVIDENCE_ALLOWANCE_S
    )
    assert budget == 630.0
    assert step_budget_s(get_data.call_timeout_s, evidence=False) == 600.0


@pytest.fixture
def short_budget(monkeypatch: pytest.MonkeyPatch) -> tuple[ToolCapability, ...]:
    """Capabilities whose `get_data` call times out at 50ms, with no retry delay
    counted and a 50ms evidence allowance: a 200ms step budget for the retrieval.
    """
    monkeypatch.setattr("executor.run.MAX_RETRY_AFTER_SECONDS", 0.0)
    monkeypatch.setattr("executor.run.BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr("executor.run.EVIDENCE_ALLOWANCE_S", 0.05)
    return tuple(
        c.model_copy(update={"call_timeout_s": 0.05}) if c.tool.name == "get_data" else c
        for c in CAPABILITIES
    )


async def test_a_retrieval_that_never_answers_stops_at_its_step_budget(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    short_budget: tuple[ToolCapability, ...],
) -> None:
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(RESULT_FRAME)]},
        delays={"get_data": 3600},
    )

    outcome = await asyncio.wait_for(
        _run(approved_plan(), surface=surface, sleep=_Sleeps(), capabilities=short_budget),
        timeout=5,
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    assert outcome.step_id == "debt"
    assert "0.2s budget" in outcome.detail
    assert [name for name, _ in surface.calls] == ["get_data"]


class _SlowAttempts(InMemoryRunLedger):
    """Takes 300ms to record each attempt, longer than `short_budget`'s step budget."""

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        if isinstance(change, Attempt):
            await asyncio.sleep(0.3)
        return await super().record(run_id, change)


async def test_a_slow_attempt_record_spends_none_of_the_step_budget(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    short_budget: tuple[ToolCapability, ...],
) -> None:
    outcome = await asyncio.wait_for(
        _run(
            approved_plan(),
            surface=_surface(real_schemas, manifest()),
            sleep=_Sleeps(),
            capabilities=short_budget,
            ledger=_SlowAttempts(),
        ),
        timeout=5,
    )

    assert isinstance(outcome, Accepted), outcome


async def test_a_statistics_fetch_that_never_answers_stops_at_its_step_budget(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    short_budget: tuple[ToolCapability, ...],
) -> None:
    async def never_answers(_manifest: Any) -> None:
        await asyncio.Event().wait()

    outcome = await asyncio.wait_for(
        _run(
            approved_plan(),
            surface=_surface(real_schemas, manifest()),
            sleep=_Sleeps(),
            capabilities=short_budget,
            fetch_series_stats=never_answers,
        ),
        timeout=5,
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    assert outcome.step_id == "debt"
    assert outcome.completed == ()


async def test_a_slow_statistics_fetch_degrades_to_statistics_unavailable_not_transport_failed(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    short_budget: tuple[ToolCapability, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`default_fetch_series_stats` gives `fetch_series_statistics_table` a deadline built off
    what is left of the step budget, capped at `EVIDENCE_ALLOWANCE_S`; a fetch that outlives
    it raises `TimeoutError` the way the real one does once its own internal
    `asyncio.timeout` expires, and `default_fetch_series_stats` catches that like any other
    fetch failure. Before F3 this degraded path did not exist: the fetch ran with no
    deadline of its own, so only the step's outer budget could ever cut it off, stopping the
    whole step `transport_failed` instead of the requirement alone.

    `_EVIDENCE_DEADLINE_MARGIN_S` is shrunk to fit `short_budget`'s millisecond scale --
    production's 1s margin would swallow the whole 50ms evidence allowance and refuse before
    the fetch ever starts, which would not tell `default_fetch_series_stats`'s own deadline
    apart from a deadline of zero.
    """
    monkeypatch.setattr("executor.run._EVIDENCE_DEADLINE_MARGIN_S", 0.01)

    async def slow_fetch_series_statistics(*, deadline: Any, **_kwargs: Any) -> None:
        async with asyncio.timeout(deadline.remaining() if deadline is not None else None):
            await asyncio.Event().wait()

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", slow_fetch_series_statistics)

    outcome = await asyncio.wait_for(
        _run(
            approved_plan(),
            surface=_surface(real_schemas, manifest()),
            sleep=_Sleeps(),
            capabilities=short_budget,
        ),
        timeout=5,
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert outcome.step_id == "debt"
    assert "statistics_unavailable" in [issue.code.value for issue in outcome.issues]


async def test_a_refresh_then_retry_refusal_sends_the_requirement_back_to_discovery(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `upstream_version_changed` reproduces identically on a bare retry, so
    # the executor does not retry it: the requirement and its source go back
    # to the planner as `RediscoveryNeeded`, and `get_data` is called once.
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[refusal("upstream_version_changed", "refresh_then_retry")],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, RediscoveryNeeded), outcome
    assert outcome.step_id == "debt"
    assert outcome.requirement_id == "annual_debt_service"
    assert outcome.spec_id == SPEC_KENYA
    assert len(surface.calls) == 1


async def test_a_definition_no_longer_held_sends_the_requirement_back_to_discovery(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `definition_not_held` is `correct_and_retry` on the wire, but the
    # approved spec_id has left the store: a bare retry reproduces it, and
    # only a new discovery under a new revision changes what gets retrieved.
    fix = {"definition_id": SPEC_KENYA, "next_step": "search"}
    surface = _surface(
        real_schemas,
        manifest(),
        get_data=[refusal("definition_not_held", "correct_and_retry", fix=fix)],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, RediscoveryNeeded), outcome
    assert (outcome.step_id, outcome.requirement_id) == ("debt", "annual_debt_service")
    assert outcome.spec_id == SPEC_KENYA
    assert outcome.refusal.fix == fix
    assert len(surface.calls) == 1


async def test_a_rediscovery_on_a_step_bound_to_no_evidence_is_a_plain_refusal(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # No analysis verb answers `refresh_then_retry` today; the executor
    # classifies by mode, so one that did would reach the aggregate step,
    # which no `EvidenceBinding` names. There is nothing to rediscover, so
    # the run stops on the step rather than instructing a rediscovery of
    # nothing in particular.
    surface = _surface(
        real_schemas,
        manifest(),
        aggregate=[refusal("upstream_version_changed", "refresh_then_retry")],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.step_id == "totals"
    assert outcome.payload is not None and outcome.payload.code == "upstream_version_changed"
    assert [step.step_id for step in outcome.completed] == ["debt"]


async def test_a_refused_steps_detail_carries_the_message_without_the_envelope_line(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """The detail reaches the planner's replan prompt, cut at 2,000
    characters, so the JSON envelope a refusal's text ends in stays out."""
    served = to_tool_result(ToolRefusal.internal_error(tool="aggregate", trace_id="0" * 32))
    text = served.content[0].text
    assert isinstance(text, str) and '"retry_mode"' in text
    surface = _surface(
        real_schemas,
        manifest(),
        aggregate=[
            ToolCallResult(is_error=True, structured_content=served.structured_content, text=text)
        ],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    message = ToolRefusal.internal_error(tool="aggregate", trace_id="0" * 32).message
    assert outcome.detail == f"'aggregate' refused: {message}"
    assert "{" not in outcome.detail


async def test_a_raised_steps_detail_keeps_the_text_it_raised_with(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """A tool that raised answers with text alone and no refusal payload."""
    surface = _surface(
        real_schemas,
        manifest(),
        aggregate=[ToolCallResult(is_error=True, text="Error calling tool 'aggregate': boom")],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused), outcome
    assert outcome.detail == "'aggregate' refused: Error calling tool 'aggregate': boom"


async def test_a_step_is_checked_against_the_source_it_was_bound_to_not_whatever_get_data_answers(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    population_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    # get_data answers the debt step (bound to Kenya's debt source) with the
    # population source's manifest: a different spec_id and identity, under
    # the very same tool. The step is checked against the source it was
    # compiled to retrieve, not whichever spec_id the manifest happens to
    # name, so this is still checked, and fails, as source drift.
    wrong = manifest_for(population_source())
    surface = ScriptedSurface(
        schemas=real_schemas, responses={"get_data": [ok(wrong)], "aggregate": [ok(RESULT_FRAME)]}
    )

    outcome = await _run(two_source_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert "source_drift" in [issue.code.value for issue in outcome.issues]


class _RacingSurface:
    """A `ToolSurface` whose `get_data` answers two concurrent retrievals apart by `spec_id`:
    the Kenya debt call raises after a short delay, and the population call
    succeeds only after a longer one. A wave that stops at the first exception
    rather than waiting for the rest of it would see the population call still
    short of its own delay when the run raises; one that awaits the whole wave
    sees it finish.
    """

    def __init__(self, schemas: list[ToolSchema], population_manifest: dict[str, Any]) -> None:
        self.schemas = list(schemas)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.population_manifest = population_manifest
        self.population_completed = False

    async def list_tools(self) -> list[ToolSchema]:
        return list(self.schemas)

    async def call_tool(
        self, *, tool: Any, arguments: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> Any:
        self.calls.append((tool.name, dict(arguments)))
        if tool.name == "aggregate":
            return ok(RESULT_FRAME)
        if arguments.get("spec_id") == SPEC_KENYA:
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")
        await asyncio.sleep(0.05)
        self.population_completed = True
        return ok(self.population_manifest)


async def test_an_unexpected_exception_in_one_step_waits_for_its_wave_before_propagating(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    population_source: Callable[..., Any],
    real_schemas: list[ToolSchema],
) -> None:
    # `two_source_plan`'s two retrievals, "debt" and "pop", share wave 0. An
    # exception `_attempt` never caught (anything but `_Stopped`) must still
    # let the wave's other step run to completion, not abandon it unawaited
    # the moment the first exception surfaces.
    # The run is recorded failed, and what is raised names it, the defect
    # its cause.
    surface = _RacingSurface(real_schemas, manifest_for(population_source()))
    ledger = InMemoryRunLedger()

    with pytest.raises(RunAbandonedError) as raised:
        await execute_approved_plan(
            two_source_plan(), surface=surface, ledger=ledger, sleep=_Sleeps()
        )

    assert {name for name, _ in surface.calls} == {"get_data"}
    assert surface.population_completed is True
    assert isinstance(raised.value.__cause__, RuntimeError)
    run = await ledger.get(raised.value.run_id)
    assert run is not None and run.failure is not None and run.failure.code == "internal_error"


async def test_a_retrieval_reminted_under_another_id_is_still_checked_against_its_source(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `get_data` may answer under a new spec_id; the approved source is found
    # by the id the step was compiled to call with, and its identity check
    # still runs.
    reminted = manifest()
    reminted["definition"]["spec_id"] = "spec_" + "e" * 32
    reminted["definition"]["identity"] = {"stat_var": "SomethingElse"}
    surface = _surface(real_schemas, reminted)

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert "source_drift" in [issue.code.value for issue in outcome.issues]


async def test_one_entity_short_of_the_period_is_not_hidden_by_anothers_reach(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    requirement: RequirementFactory,
) -> None:
    two = manifest()
    two["statistics"].append(
        {
            **two["statistics"][0],
            "entities": [{"role": "recipient", "ref": "country/GHA"}],
            "observations": 5,
            "first": {"value": 1.0, "period": "2015"},
            "latest": {"value": 1.0, "period": "2019"},
        }
    )
    strict = approved_plan(requirement=requirement(coverage=Coverage(on_missing="fail")))

    outcome = await _run(strict, surface=_surface(real_schemas, two), sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert "window_short" in [issue.code.value for issue in outcome.issues]


async def test_a_refusal_the_tool_says_not_to_retry_stops_the_run_at_once(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(
        real_schemas, manifest(), aggregate=[refusal("non_finite_result", "do_not_retry")]
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.step_id == "totals"
    assert outcome.payload is not None and outcome.payload.code == "non_finite_result"
    assert [step.step_id for step in outcome.completed] == ["debt"]
    assert len(surface.calls) == 2


async def test_a_refusal_carrying_a_choice_goes_back_to_the_planner_with_the_candidates(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    candidates = [{"id": "usd", "label": "US dollars"}, {"id": "eur", "label": "Euro"}]
    surface = _surface(
        real_schemas,
        manifest(),
        aggregate=[
            refusal("unit_heterogeneous", "correct_and_retry", fix={"candidates": candidates})
        ],
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, MethodChoiceNeeded)
    assert (outcome.step_id, outcome.code) == ("totals", "unit_heterogeneous")
    assert outcome.candidates == candidates
    assert [step.step_id for step in outcome.completed] == ["debt"]


@pytest.mark.parametrize(
    ("code", "fix"),
    [
        ("cross_frame_unit_mismatch", {"left_unit": "USD", "right_unit": "EUR"}),
        ("cross_frame_unit_unbound", {"unbound_side": "right", "bound_unit": "USD"}),
        ("cross_frame_facet_mismatch", {"axis": "provenance", "left": ["a"], "right": ["b"]}),
        ("coverage_insufficient", {"observed_periods": ["2015", "2016"], "required": 5}),
        ("share_of_zero_denominator", {"time_period": "2019"}),
    ],
    ids=["unit-mismatch", "unit-unbound", "facet-mismatch", "coverage", "zero-denominator"],
)
async def test_a_refusal_whose_fix_is_a_methodological_alternative_goes_back_to_the_planner(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    code: str,
    fix: dict[str, Any],
) -> None:
    # Each of these carries the alternatives the user could choose between;
    # picking one changes the plan, so the planner gets them, not a dead end.
    surface = _surface(
        real_schemas, manifest(), aggregate=[refusal(code, "correct_and_retry", fix=fix)]
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, MethodChoiceNeeded), outcome
    assert (outcome.step_id, outcome.code) == ("totals", code)
    assert outcome.candidates == fix
    assert [step.step_id for step in outcome.completed] == ["debt"]


@pytest.mark.parametrize(
    "code", ["role_ambiguous", "join_no_common_key_columns"], ids=["role", "join-keys"]
)
async def test_a_correction_the_compiler_already_rules_out_stops_the_run_with_the_fix_kept(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema], code: str
) -> None:
    # The compiler writes every column list from the source's own roles and
    # reports `join_no_common_key_columns` itself, so either reaching the
    # executor is a compiler bug, not a choice for the user. The tool's own
    # `fix` still rides along on the refusal, unacted on.
    fix = {"parameter": "group_by", "available": ["recipient", "creditor"]}
    surface = _surface(
        real_schemas, manifest(), aggregate=[refusal(code, "correct_and_retry", fix=fix)]
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.step_id == "totals"
    assert outcome.payload is not None and outcome.payload.fix == fix


async def test_a_terminal_step_that_produced_no_frame_is_refused_not_a_crash(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `Workflow` rejects a terminal id it does not hold, so this can only be
    # reached by a workflow that bypassed its own validation -- but the run
    # answers with a typed refusal naming the step, not a `KeyError`.
    approved = approved_plan()
    assert approved.workflow is not None
    workflow = approved.workflow.model_copy(
        update={"terminal_steps": (*approved.workflow.terminal_steps, "ghost")}
    )
    plan = approved.model_copy(
        update={
            "workflow": workflow,
            "status": PlanStatus.PENDING_APPROVAL,
            "approved_revision": None,
        }
    ).approve(at=datetime.now(UTC))
    surface = _surface(real_schemas, manifest())

    finished, outcome = await execute_approved_plan(plan, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_RESULT_UNUSABLE
    assert outcome.step_id == "ghost"
    assert [step.step_id for step in outcome.completed] == ["debt", "totals"]
    assert finished.status is PlanStatus.FAILED


async def test_a_result_without_the_frame_it_was_compiled_to_yield_stops_the_run(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(
        real_schemas, manifest(), aggregate=[ok({"spec_id": RESULT_FRAME["spec_id"]})]
    )

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.STEP_RESULT_UNUSABLE
    assert outcome.step_id == "totals"


async def test_a_retrieval_missing_a_required_entity_is_not_evidence(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    ghana_only = manifest()
    ghana_only["statistics"][0]["entities"] = [{"role": "recipient", "ref": "country/GHA"}]
    surface = _surface(real_schemas, ghana_only)

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert outcome.requirement_id == "annual_debt_service"
    assert "entity_missing" in [issue.code.value for issue in outcome.issues]
    # The verb never ran: evidence is checked before anything depends on it.
    assert [name for name, _ in surface.calls] == ["get_data"]


async def test_a_retrieval_whose_identity_drifted_since_approval_is_not_evidence(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    drifted = manifest()
    drifted["definition"]["identity"] = {"stat_var": "SomethingElse"}
    surface = _surface(real_schemas, drifted)

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert "source_drift" in [issue.code.value for issue in outcome.issues]


@pytest.mark.parametrize(
    ("snapshot", "refused_as"),
    [
        # The list read back serves two StatVars for a source approved as one:
        # membership drift.
        (
            httpx2.Response(
                200,
                json={
                    "trace": {
                        "resolved_stat_vars": [
                            {"ref": "dcid:DebtServicePaid", "roles": []},
                            {"ref": "dcid:Other", "roles": []},
                        ]
                    }
                },
            ),
            RefusalKind.REQUIREMENT_UNMET,
        ),
        # The list cannot be read back: the result is unusable as evidence.
        (httpx2.Response(404, json={"code": "not_found"}), RefusalKind.STEP_RESULT_UNUSABLE),
    ],
)
async def test_a_stat_var_list_the_result_left_out_is_read_back_for_the_membership_check(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
    snapshot: httpx2.Response,
    refused_as: RefusalKind,
) -> None:
    left_out = manifest()
    del left_out["trace"]["resolved_stat_vars"]
    # The count the result recorded: the read-back list holds as many.
    left_out[STAT_VARS_OMITTED_KEY] = 2
    upstream = UpstreamClient(
        settings=DataMcpSettings(upstream_max_attempts=1),
        transport=httpx2.MockTransport(lambda request: snapshot),
    )
    monkeypatch.setattr(stat_var_list, "get_client", lambda: upstream)

    outcome = await _run(approved_plan(), surface=_surface(real_schemas, left_out), sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is refused_as
    assert outcome.step_id == "debt"


async def test_missing_years_are_recorded_or_refused_as_the_requirement_says(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    requirement: RequirementFactory,
) -> None:
    sparse = manifest()
    sparse["statistics"][0]["observations"] = 7

    accepting = approved_plan(requirement=requirement(coverage=Coverage(on_missing="report_gaps")))
    strict = approved_plan(requirement=requirement(coverage=Coverage(on_missing="fail")))

    lenient = await _run(accepting, surface=_surface(real_schemas, sparse), sleep=_Sleeps())
    refused = await _run(strict, surface=_surface(real_schemas, sparse), sleep=_Sleeps())

    assert isinstance(lenient, Accepted), lenient
    assert "coverage_gaps" in [gap.code.value for gap in lenient.evidence[0].gaps]
    assert isinstance(refused, Refused)
    assert refused.refusal is RefusalKind.REQUIREMENT_UNMET


async def test_a_retrieval_returning_no_rows_is_not_evidence(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    empty = manifest(coverage={"rows": 0, "dimensions": []})
    surface = _surface(real_schemas, empty)

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert "rows_empty" in [issue.code.value for issue in outcome.issues]


async def test_execute_approved_plan_marks_a_refused_run_failed(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(
        real_schemas, manifest(), aggregate=[refusal("internal_error", "do_not_retry")]
    )

    finished, outcome = await execute_approved_plan(
        approved_plan(), surface=surface, sleep=_Sleeps()
    )

    assert isinstance(outcome, Refused)
    assert finished.status is PlanStatus.FAILED


async def test_an_unreadable_surface_stops_the_run_before_any_call(
    approved_plan: PlanFactory, real_schemas: list[ToolSchema]
) -> None:
    class Unreadable(ScriptedSurface):
        async def list_tools(self) -> list[ToolSchema]:
            raise ToolSurfaceUnavailableError("could not read the tool surface")

    surface = Unreadable(schemas=real_schemas, responses={})

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    assert surface.calls == []


# -- the shape check and the empty-result rule ----------------------------------------


@pytest.fixture
def three_step_plan(
    decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: Callable[..., AggregateRequirement],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Plan:
    """retrieve -> aggregate (non-terminal) -> aggregate (terminal): proves a
    non-terminal step's own zero-row result does not refuse the run on its
    own, since the empty-result rule applies to a terminal step alone.
    """
    dataset = requirement()
    totals = derived_requirement()
    totals_again = derived_requirement(id="annual_totals_again", input_id="annual_totals")
    draft = Plan(
        question="How much does Kenya pay to service its debt each year, summed twice over?",
        decisions=[decision()],
        evidence_requirements=[dataset, totals, totals_again],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC_KENYA),
            AggregateOperation(
                id="totals",
                requirement_id="annual_totals",
                input="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            AggregateOperation(
                id="totals2",
                requirement_id="annual_totals_again",
                input="totals",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals2"],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas={schema.name: schema for schema in real_schemas},
        discovered={"annual_debt_service": source()},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_a_zero_row_terminal_result_is_refused(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `aggregate`'s own capability row leaves `empty_is_an_answer` at its
    # `False` default, so a terminal aggregate matching nothing is refused,
    # not silently completed (`ac1024c`'s own behaviour, per `run.py:36-43`
    # at the time this rule did not exist).
    empty = {**RESULT_FRAME, "rows": 0, "stat_var_count": 0}
    surface = _surface(real_schemas, manifest(), aggregate=[ok(empty)])

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.EMPTY_ANSWER
    assert outcome.step_id == "totals"


async def test_a_zero_row_terminal_result_is_accepted_when_its_capability_says_so(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # Stands in for `filter`'s own `empty_is_an_answer=True` row on the
    # terminal step's tool, isolating the empty-result rule itself from a
    # full filter chain: a filter matching nothing is a real answer, and it
    # is the capability row alone that says so.
    empty = {**RESULT_FRAME, "rows": 0, "stat_var_count": 0}
    permissive = tuple(
        c.model_copy(update={"empty_is_an_answer": True}) if c.tool.name == "aggregate" else c
        for c in CAPABILITIES
    )
    surface = _surface(real_schemas, manifest(), aggregate=[ok(empty)])

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps(), capabilities=permissive)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.steps[-1].row_count == 0


async def test_a_non_terminal_zero_row_step_does_not_refuse_on_its_own(
    three_step_plan: Plan, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    zero = {**RESULT_FRAME, "rows": 0, "role_keys": [], "stat_var_count": 0}
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(zero), ok(RESULT_FRAME)]},
    )

    outcome = await _run(three_step_plan, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    assert [step.step_id for step in outcome.steps] == ["debt", "totals", "totals2"]
    assert outcome.steps[1].row_count == 0
    assert outcome.steps[2].row_count == RESULT_FRAME["rows"]


@pytest.mark.parametrize(
    ("declared", "recorded"),
    [
        pytest.param({"dropped_key_count": 2}, 2, id="two-keys"),
        pytest.param({}, 0, id="no-member"),
        pytest.param({"dropped_key_count": "2"}, 0, id="not-a-number"),
        pytest.param({"dropped_key_count": True}, 0, id="a-boolean"),
        pytest.param({"dropped_key_count": -1}, 0, id="negative"),
    ],
)
async def test_a_step_records_the_keys_its_result_declares_it_left_out(
    declared: dict[str, Any],
    recorded: int,
    two_input_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    gdp_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    # `get_data`'s manifest never carries the member, so both retrievals
    # record 0 whatever the quotient declares.
    plan = two_input_plan(kind="quotient_of")
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest_for(source())), ok(manifest_for(gdp_source()))],
            "quotient_of": [ok(RESULT_FRAME | declared)],
        },
    )

    outcome = await _run(plan, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    assert {step.step_id: step.dropped_key_count for step in outcome.steps} == {
        "debt": 0,
        "gdp": 0,
        "combined": recorded,
    }


async def test_a_declared_shape_differing_from_the_prediction_is_refused(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `totals`'s step carries `expected_role_keys=()`; a result declaring a
    # role column the plan was not compiled to expect is a mismatch between
    # what the tool actually did and what compiling predicted it would do.
    mismatched = {**RESULT_FRAME, "role_keys": ["reporter"]}
    surface = _surface(real_schemas, manifest(), aggregate=[ok(mismatched)])

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused)
    assert outcome.refusal is RefusalKind.SHAPE_MISMATCH
    assert outcome.step_id == "totals"


async def test_a_declared_measure_count_is_compared_only_once_there_are_rows_to_count(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    two_measures = {**RESULT_FRAME, "stat_var_count": 2}
    zero_row_two_measures = {**RESULT_FRAME, "rows": 0, "stat_var_count": 2}
    # The empty-result rule is a separate check from the shape check this
    # test is about, so `aggregate` is made permissive of an empty result
    # here to isolate the measure-count comparison on its own.
    permissive = tuple(
        c.model_copy(update={"empty_is_an_answer": True}) if c.tool.name == "aggregate" else c
        for c in CAPABILITIES
    )

    refused = await _run(
        approved_plan(),
        surface=_surface(real_schemas, manifest(), aggregate=[ok(two_measures)]),
        sleep=_Sleeps(),
        capabilities=permissive,
    )
    accepted = await _run(
        approved_plan(),
        surface=_surface(real_schemas, manifest(), aggregate=[ok(zero_row_two_measures)]),
        sleep=_Sleeps(),
        capabilities=permissive,
    )

    assert isinstance(refused, Refused)
    assert refused.refusal is RefusalKind.SHAPE_MISMATCH
    # With zero rows there is no stat_var to count, so the measure mismatch
    # is not compared and only the (matching) role keys are.
    assert isinstance(accepted, Accepted), accepted


# -- concurrent, independent steps ----------------------------------------------------


async def test_two_independent_retrievals_run_concurrently_and_land_in_workflow_order(
    two_source_plan: PlanFactory,
    manifest: ManifestFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    # `get_data` is delayed, so the two retrievals overlap only if the run
    # actually holds both in flight at once rather than awaiting one to
    # completion before starting the other -- a serial loop could never
    # push `max_concurrency` past 1.
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest_for(source())), ok(manifest_for(population_source()))],
            "aggregate": [ok(RESULT_FRAME), ok(RESULT_FRAME)],
        },
        delays={"get_data": 0.05},
    )

    outcome = await _run(two_source_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    assert surface.max_concurrency == 2
    # `workflow.steps` interleaves the two retrieve/aggregate pairs
    # (`debt`, `totals`, `pop`, `population_totals_op`); the wave that ran
    # `debt` and `pop` together does not disturb that order.
    assert [step.step_id for step in outcome.steps] == [
        "debt",
        "totals",
        "pop",
        "population_totals_op",
    ]
    (debt_evidence, population_evidence) = outcome.evidence
    assert debt_evidence.requirement_id == "annual_debt_service"
    assert population_evidence.requirement_id == "population"


async def test_an_aggregate_never_starts_before_its_own_retrieval_returns(
    three_step_plan: Plan, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # `three_step_plan` is a straight chain (retrieve -> aggregate ->
    # aggregate); delaying `get_data` proves the chain is honoured rather
    # than merely inferred from a passing outcome -- a step that started
    # early would overlap the delayed retrieval and push `max_concurrency`
    # past 1.
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(RESULT_FRAME)]},
        delays={"get_data": 0.05},
    )

    outcome = await _run(three_step_plan, surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    assert surface.max_concurrency == 1
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate", "aggregate"]


async def test_a_refusal_on_one_of_two_concurrent_retrievals_names_the_sibling_that_finished(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    # `debt`'s `get_data` call is delayed and then transport-fails; `pop`'s
    # own `get_data` call, in the same wave, succeeds well before the
    # delay elapses. The wave drains before the stop is raised, so the
    # finished sibling is still named on `completed`, deterministically,
    # across repeated runs.
    for _ in range(3):
        surface = ScriptedSurface(
            schemas=real_schemas,
            responses={
                "get_data": [
                    ToolSurfaceUnavailableError("debt's connection dropped"),
                    ok(manifest_for(population_source())),
                ],
                "aggregate": [ok(RESULT_FRAME)],
            },
            delays={"get_data": 0.05},
        )

        outcome = await _run(two_source_plan(), surface=surface, sleep=_Sleeps())

        assert isinstance(outcome, Refused), outcome
        assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
        assert outcome.step_id == "debt"
        assert [step.step_id for step in outcome.completed] == ["pop"]


async def test_two_refusals_in_one_wave_carry_the_second_as_a_sibling(
    two_source_plan: PlanFactory,
    real_schemas: list[ToolSchema],
) -> None:
    """`stopped = stopped or outcome` used to keep only the first `_Stopped` in
    wave order and drop `pop`'s own refusal; a wave that surfaces two stale
    sources should let one replan fix both. Both retrievals refuse with
    `upstream_version_changed`, one of the refusals the existing fixtures
    already produce (see `test_a_refresh_then_retry_refusal_sends_the_
    requirement_back_to_discovery`)."""
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [
                refusal("upstream_version_changed", "refresh_then_retry"),
                refusal("upstream_version_changed", "refresh_then_retry"),
            ]
        },
    )

    outcome = await _run(two_source_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, RediscoveryNeeded), outcome
    assert outcome.step_id == "debt"
    assert outcome.spec_id == SPEC_KENYA
    assert [step.step_id for step in outcome.completed] == []
    assert len(outcome.also_stopped) == 1
    sibling = outcome.also_stopped[0]
    assert isinstance(sibling, RediscoveryNeeded)
    assert sibling.step_id == "pop"
    assert sibling.spec_id == SPEC_POPULATION
    assert [step.step_id for step in sibling.completed] == []


@pytest.mark.parametrize(("rows", "answers"), [(1, ["totals"]), (0, [])])
async def test_a_stopped_run_answers_with_the_finished_terminal_steps_an_accepted_run_would(
    two_source_plan: PlanFactory,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
    rows: int,
    answers: list[str],
) -> None:
    """`totals` finishes in the wave where its sibling terminal step refuses.
    Zero rows from `aggregate`, whose empty result is not an answer, keeps it
    out of the stop's `answers`, as `EMPTY_ANSWER` would on an accepted run."""
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest_for(source())), ok(manifest_for(population_source()))],
            "aggregate": [
                ok({**RESULT_FRAME, "rows": rows, "stat_var_count": rows}),
                refusal("growth_gap_year", "do_not_retry"),
            ],
        },
    )

    outcome = await _run(two_source_plan(), surface=surface, sleep=_Sleeps())

    assert not isinstance(outcome, Accepted), outcome
    assert outcome.step_id == "population_totals_op"
    assert [step.step_id for step in outcome.completed] == ["debt", "totals", "pop"]
    assert list(outcome.answers) == answers


# -- wave concurrency is bounded -------------------------------------------------------

_SIX_REQ_IDS = tuple(f"req{n}" for n in range(6))
_SIX_SPEC_IDS = tuple(f"spec_{chr(ord('a') + n) * 32}" for n in range(6))


class _ConcurrencyProbeSurface:
    """Answers each `get_data` call by its own `spec_id` and every `aggregate`
    call with the same result, and records the in-flight call count at the
    moment each call is made (increment, yield to the loop, decrement) --
    the highest count observed is the most calls this surface ever held in
    flight at once, at the point of call.
    """

    def __init__(self, *, schemas: list[ToolSchema], manifests: dict[str, dict[str, Any]]) -> None:
        self.schemas = list(schemas)
        self.manifests = manifests
        self.peak_in_flight = 0
        self._in_flight = 0

    async def list_tools(self) -> list[ToolSchema]:
        return list(self.schemas)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: dict[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> Any:
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        await asyncio.sleep(0)
        try:
            if tool.name == "get_data":
                return ok(self.manifests[arguments["spec_id"]])
            return ok(RESULT_FRAME)
        finally:
            self._in_flight -= 1


@pytest.fixture
def six_step_plan(
    decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: Callable[..., AggregateRequirement],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Plan:
    """Six independent retrieve-then-aggregate pairs: every retrieve step is
    level 0, with no binding on any other, so all six share the first wave.
    """
    totals_ids = [f"totals{n}" for n in range(6)]
    requirements = [requirement(id=req_id) for req_id in _SIX_REQ_IDS]
    totals = [
        derived_requirement(id=totals_id, input_id=req_id)
        for req_id, totals_id in zip(_SIX_REQ_IDS, totals_ids, strict=True)
    ]
    sources = {
        req_id: source(spec_id=spec_id)
        for req_id, spec_id in zip(_SIX_REQ_IDS, _SIX_SPEC_IDS, strict=True)
    }
    draft = Plan(
        question="What does each of six independent sources total?",
        decisions=[decision()],
        evidence_requirements=[*requirements, *totals],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            *(
                RetrieveOperation(id=req_id, requirement_id=req_id, spec_id=spec_id)
                for req_id, spec_id in zip(_SIX_REQ_IDS, _SIX_SPEC_IDS, strict=True)
            ),
            *(
                AggregateOperation(
                    id=totals_id,
                    requirement_id=totals_id,
                    input=req_id,
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                )
                for req_id, totals_id in zip(_SIX_REQ_IDS, totals_ids, strict=True)
            ),
        ],
        terminal=totals_ids,
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas={schema.name: schema for schema in real_schemas},
        discovered=sources,
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_a_wave_wider_than_the_bound_queues_at_the_point_of_call(
    six_step_plan: Plan,
    manifest_for: Callable[..., dict[str, Any]],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    manifests = {spec_id: manifest_for(source(spec_id=spec_id)) for spec_id in _SIX_SPEC_IDS}
    surface = _ConcurrencyProbeSurface(schemas=real_schemas, manifests=manifests)

    outcome = await _run(six_step_plan, surface=surface, sleep=_Sleeps(), max_concurrent_steps=2)

    assert isinstance(outcome, Accepted), outcome
    assert len(outcome.steps) == 12
    assert surface.peak_in_flight == 2


# default_fetch_series_stats: the row-count check


async def test_default_fetch_series_stats_passes_the_manifests_series_count_and_continuation(
    monkeypatch: pytest.MonkeyPatch, manifest: ManifestFactory
) -> None:
    captured: dict[str, Any] = {}

    async def fake_fetch(**kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", fake_fetch)
    stripped = strip_manifest(manifest())
    stripped = stripped.model_copy(
        update={
            "trace": {
                **stripped.trace,
                "continuations": [
                    {
                        "axis": "measurementMethod",
                        "method": "Estimated",
                        "periods": ["2025", "2026"],
                    }
                ],
            }
        }
    )

    await default_fetch_series_stats(stripped)

    assert (captured["expected_series"], captured["continuing"]) == (
        stripped.series_count,
        "Estimated",
    )


async def test_default_fetch_series_stats_reads_the_artifact_when_series_count_is_missing(
    monkeypatch: pytest.MonkeyPatch, manifest: ManifestFactory
) -> None:
    """dfo-apis mints an all-years definition with the statistics artifact but no
    `series_count` (seen live 2026-09-28). The artifact is what the check reads, so it
    is fetched, with no row count to hold it to, instead of failing the requirement
    `STATISTICS_UNAVAILABLE`.
    """
    captured: dict[str, Any] = {}

    async def fake_fetch(**kwargs: Any) -> pa.Table:
        captured.update(kwargs)
        return SERIES_STATISTICS_ARROW_SCHEMA.empty_table()

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", fake_fetch)
    stripped = strip_manifest(manifest(series_count=None))

    result = await default_fetch_series_stats(stripped)
    assert result is not None
    assert result.num_rows == 0
    assert captured["expected_series"] is None


async def test_default_fetch_series_stats_answers_none_for_an_old_record(
    monkeypatch: pytest.MonkeyPatch, manifest: ManifestFactory
) -> None:
    """No `series_count` and no statistics artifact: a record minted before either
    existed, with nothing to fetch."""

    async def fake_fetch(**kwargs: Any) -> pa.Table:
        raise AssertionError("an old record has no artifact to fetch")

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", fake_fetch)
    stripped = strip_manifest(manifest(series_count=None, artifacts=[]))

    assert await default_fetch_series_stats(stripped) is None


async def test_a_manifest_with_statistics_but_no_series_count_is_accepted(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """The all-years mint end to end: the evidence check reads the artifact's rows and
    the run is accepted, instead of refusing `statistics_unavailable`."""
    surface = _surface(real_schemas, manifest(series_count=None))

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Accepted), outcome
    (evidence,) = outcome.evidence
    assert evidence.stat_vars == ("DebtServicePaid",)


async def test_a_manifest_with_neither_series_count_nor_statistics_still_fails_closed(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    surface = _surface(real_schemas, manifest(series_count=None, artifacts=[]))

    outcome = await _run(approved_plan(), surface=surface, sleep=_Sleeps())

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.REQUIREMENT_UNMET
    assert "statistics_unavailable" in [issue.code.value for issue in outcome.issues]


async def test_default_fetch_series_stats_bounds_the_deadline_by_the_smaller_of_the_two(
    monkeypatch: pytest.MonkeyPatch, manifest: ManifestFactory
) -> None:
    """The fetch's own deadline is `current_step_deadline`'s remaining seconds, capped at
    `EVIDENCE_ALLOWANCE_S`, less `_EVIDENCE_DEADLINE_MARGIN_S` -- whichever of the step
    deadline and the allowance is smaller, never the other one.
    """
    captured: dict[str, Any] = {}

    async def fake_fetch(*, deadline: Any, **kwargs: Any) -> list[Any]:
        captured["deadline"] = deadline
        return []

    monkeypatch.setattr("executor.run.fetch_series_statistics_table", fake_fetch)
    monkeypatch.setattr("executor.run.EVIDENCE_ALLOWANCE_S", 10.0)
    monkeypatch.setattr("executor.run._EVIDENCE_DEADLINE_MARGIN_S", 1.0)
    stripped = strip_manifest(manifest())

    # Plenty left on the step deadline: the allowance, minus the margin, wins.
    token = current_step_deadline.set(CallDeadline.after(100.0))
    try:
        await default_fetch_series_stats(stripped)
    finally:
        current_step_deadline.reset(token)
    assert 8.5 < captured["deadline"].remaining() <= 9.0

    # The step deadline is already the tighter bound: it wins instead.
    token = current_step_deadline.set(CallDeadline.after(3.0))
    try:
        await default_fetch_series_stats(stripped)
    finally:
        current_step_deadline.reset(token)
    assert 1.5 < captured["deadline"].remaining() <= 2.0
