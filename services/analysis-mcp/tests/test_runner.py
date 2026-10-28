"""The registered tools against dfo-apis' analysis runner: the tree a call sends,
the root a frame ref becomes, the caller it names, and the refusal each runner
answer maps to."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from analysis_mcp import verbs
from analysis_mcp.runner_client import RunnerClient, runner_holder
from analysis_mcp.summary import VerbSummary
from common.caller import supplied_caller
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MintRecord,
    set_store,
)
from runner_replay import REPLIES_PATH

_SPEC = "spec_" + "1" * 32
_CV = "cv_" + "2" * 32
_DATASET_CV = "cv_" + "3" * 32
_OPS = "ops_" + "4" * 32
_OCV = "ocv_" + "5" * 32
_FRAME = {"spec_id": _SPEC, "content_version": _CV}
_DERIVED = {"spec_id": _OPS, "content_version": _OCV}
_ADDRESS_CALLER = "ip:" + "a" * 32
_SUM = {"by": ["time_period"], "op": "sum"}
_BREAKDOWN = {"by": ["time_period", "crsPurpose"], "op": "sum", "levels": {"crsPurpose": 1}}

type Handler = Callable[[httpx2.Request], httpx2.Response]


async def _no_token() -> dict[str, str]:
    return {}


class _Runner:
    """A runner answering every request with `answer`, keeping each request."""

    def __init__(self, answer: httpx2.Response) -> None:
        self.answer = answer
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.answer

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def _install(answer: httpx2.Response, *, principal: str | None = None) -> _Runner:
    runner = _Runner(answer)
    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=5.0,
            caller_principal=principal,
            auth=_no_token,
            transport=httpx2.MockTransport(runner.handle),
        )
    )
    return runner


def _problem(status: int, code: str, *, fix: object = None, **extra: object) -> httpx2.Response:
    body = {"code": code, "detail": f"{code} detail", "retryable": False, "fix": fix, **extra}
    return httpx2.Response(status, json=body)


@pytest.fixture
def store() -> Iterator[InMemoryMaterialisationStore]:
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _recorded_ready() -> dict[str, Any]:
    """A 200 body dfo-apis' runner gave, from the recorded replies."""
    replies = json.loads(REPLIES_PATH.read_text())
    return next(
        reply["body"]
        for key, reply in replies.items()
        if reply["status"] == 200 and key.startswith("/api/datasets/analysis {")
    )


# The tree a call sends -----------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "inputs"),
    [
        (
            lambda: verbs.compare(frame_ref=_FRAME, right_frame_ref=_DERIVED),
            {"data": _DERIVED, "other": _FRAME},
        ),
        (lambda: verbs.compare(frame_ref=_FRAME), {"data": _FRAME}),
        (
            lambda: verbs.sum_of(frame_ref=_FRAME, right_frame_ref=_DERIVED),
            {"data": _FRAME, "other": _DERIVED},
        ),
        (
            lambda: verbs.share_of(frame_ref=_FRAME, denominator_frame_ref=_DERIVED),
            {"data": _FRAME, "other": _DERIVED},
        ),
        (
            lambda: verbs.join(frame_ref=_DERIVED, right_frame_ref=_FRAME),
            {"data": _DERIVED, "other": _FRAME},
        ),
    ],
    ids=["compare-right-is-data", "compare-one-frame", "sum_of", "share_of", "join"],
)
async def test_a_two_sided_call_fills_data_and_other_by_the_runners_slots(
    store: InMemoryMaterialisationStore, call: Callable[[], Any], inputs: dict[str, Any]
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))
    await call()

    rendered = {
        name: {key: node[key] for key in ("spec_id", "content_version")}
        for name, node in runner.body["tree"]["inputs"].items()
    }
    assert rendered == inputs
    assert runner.body["latest"] is False


async def test_a_rolling_call_sends_its_op_spec_and_no_anchor_key(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))
    rolling = {
        "window_start": {"anchor": "latest", "offset": -1, "rule": "half1"},
        "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
        "population": [{"spec_id": _SPEC, "bindings": {}}],
    }
    await verbs.growth(
        frame_ref=_FRAME,
        window_start="2021",
        window_end="2022",
        rolling=rolling,
        anchor_key="rak_" + "6" * 32,
    )

    tree = runner.body["tree"]
    assert tree["verb"] == "growth"
    assert tree["arguments"]["window_start"] == "2021"
    assert tree["arguments"]["rolling"]["window_end"] == rolling["window_end"]
    assert "anchor_key" not in tree["arguments"]
    assert "frame_ref" not in tree["arguments"]


# The root a frame ref becomes ------------------------------------------------------


def _record(options: dict[str, Any]) -> MintRecord:
    return MintRecord(
        frame=FrameRef(**_FRAME),
        dataset=FrameRef(spec_id=_SPEC, content_version=_DATASET_CV),
        options=options,
    )


@pytest.mark.parametrize(
    ("record", "root"),
    [
        (None, {**_FRAME, "rendering": {"without_party_totals": False, "columns": []}}),
        (
            _record(
                {
                    "format": "parquet",
                    "columns": ["observationAbout_type"],
                    "without_party_totals": True,
                }
            ),
            {
                "spec_id": _SPEC,
                "content_version": _DATASET_CV,
                "rendering": {"without_party_totals": True, "columns": ["observationAbout_type"]},
            },
        ),
        (
            _record({"format": "parquet", "groupby": _SUM}),
            {
                "spec_id": _SPEC,
                "content_version": _DATASET_CV,
                "rendering": {"without_party_totals": False, "columns": [], "groupby": _SUM},
            },
        ),
        (
            _record(
                {
                    "format": "parquet",
                    "columns": ["observationAbout_type"],
                    "groupby": _BREAKDOWN,
                }
            ),
            {
                "spec_id": _SPEC,
                "content_version": _DATASET_CV,
                "rendering": {
                    "without_party_totals": False,
                    "columns": ["observationAbout_type"],
                    "groupby": _BREAKDOWN,
                },
            },
        ),
    ],
    ids=["no-mint-record", "mint-record", "stat-vars-sum", "breakdown-with-columns"],
)
async def test_a_get_data_frame_is_sent_as_the_dataset_version_its_mint_record_names(
    store: InMemoryMaterialisationStore, record: MintRecord | None, root: dict[str, Any]
) -> None:
    if record is not None:
        await store.put_mint_record(record)
    runner = _install(_problem(422, "op_spec_invalid"))
    await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    assert runner.body["tree"]["inputs"] == {"data": root}


async def test_a_frame_fetched_with_an_option_the_runner_does_not_read_refuses_before_any_call(
    store: InMemoryMaterialisationStore,
) -> None:
    await store.put_mint_record(_record({"format": "parquet", "currency": "EUR"}))
    runner = _install(_problem(422, "op_spec_invalid"))
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    assert result.structured_content is not None
    assert result.structured_content["code"] == "rendering_not_recorded"
    assert result.structured_content["fix"]["options"] == ["currency"]
    assert runner.requests == []


# The reply -------------------------------------------------------------------------


async def test_a_ready_answer_replies_with_its_summarys_verb_result_and_digest(
    store: InMemoryMaterialisationStore,
) -> None:
    body = _recorded_ready()
    _install(httpx2.Response(200, json=body))
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    summary = VerbSummary.parse(body["summary"])
    assert result.is_error is False
    assert result.structured_content == summary.verb_result().model_dump(mode="json")
    text = result.content[0].text
    assert text.endswith(f"execution_identity: {summary.execution_identity.canonical_string()}")
    assert f"spec_id={summary.spec_id}, content_version={summary.content_version}" in text


# The caller ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("caller", "principal", "sent"),
    [
        (
            _ADDRESS_CALLER,
            "secret",
            {"x-dfo-caller-principal": "secret", "x-dfo-caller": _ADDRESS_CALLER},
        ),
        (_ADDRESS_CALLER, None, {}),
        ("conversation:" + "b" * 32, "secret", {}),
    ],
    ids=["website-client", "no-principal", "harness-conversation"],
)
async def test_a_website_caller_is_named_beside_the_principal(
    store: InMemoryMaterialisationStore,
    caller: str,
    principal: str | None,
    sent: dict[str, str],
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"), principal=principal)
    with supplied_caller(caller):
        await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    headers = runner.requests[-1].headers
    named = {
        name: headers[name]
        for name in ("x-dfo-caller-principal", "x-dfo-caller")
        if name in headers
    }
    assert named == sent


# The refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "code", "retry_mode", "fix"),
    [
        (
            _problem(422, "unit_heterogeneous", fix={"parameter": "unit", "units": ["a", "b"]}),
            "unit_heterogeneous",
            "correct_and_retry",
            {"parameter": "unit", "units": ["a", "b"]},
        ),
        (
            _problem(404, "snapshot_not_found", fix={**_FRAME, "latest": True}),
            "frame_not_held",
            "correct_and_retry",
            _FRAME,
        ),
        (
            _problem(410, "artifact_expired", fix={**_FRAME, "latest": True}),
            "frame_expired",
            "correct_and_retry",
            _FRAME,
        ),
        (
            _problem(422, "over_ceiling", fix={"ceiling_kind": "analysis_root_rows"}),
            "frame_too_large",
            "correct_and_retry",
            {"ceiling_kind": "analysis_root_rows"},
        ),
        (
            httpx2.Response(
                503,
                json={"code": "service_unavailable", "detail": "busy", "retryable": True},
                headers={"retry-after": "3"},
            ),
            "service_unavailable",
            "retry_after_delay",
            {"service": "dfo-apis"},
        ),
        (
            httpx2.Response(
                202, json={"status": "pending", "retry_after_s": 2.0, "operations": ["op_1"]}
            ),
            "service_unavailable",
            "retry_after_delay",
            {"service": "dfo-apis", "operations": ["op_1"]},
        ),
        (
            _problem(404, "spec_not_found"),
            "upstream_error",
            "do_not_retry",
            {"detail": "spec_not_found detail"},
        ),
        (
            httpx2.Response(504, json={"code": "timeout", "detail": "slow", "retryable": True}),
            "upstream_error",
            "retry_after_delay",
            {"detail": "slow"},
        ),
        (
            httpx2.Response(502, text="bad gateway"),
            "upstream_error",
            "retry_after_delay",
            {"detail": "dfo-apis answered 502 with no problem document"},
        ),
        (
            httpx2.Response(200, json={"status": "ready", "summary": {"verb": "aggregate"}}),
            "upstream_error",
            "do_not_retry",
            None,
        ),
    ],
    ids=[
        "engine-code",
        "root-not-stored",
        "root-expired",
        "over-ceiling",
        "busy",
        "pending",
        "unmapped-code",
        "retryable-unmapped-code",
        "no-problem-document",
        "summary-unreadable",
    ],
)
async def test_each_runner_answer_maps_to_its_refusal(
    store: InMemoryMaterialisationStore,
    answer: httpx2.Response,
    code: str,
    retry_mode: str,
    fix: dict[str, Any] | None,
) -> None:
    _install(answer)
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    structured = result.structured_content
    assert result.is_error is True
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (code, retry_mode)
    if fix is not None:
        assert structured["fix"] == fix


async def test_an_unreachable_runner_refuses_retryable_upstream_error(
    store: InMemoryMaterialisationStore,
) -> None:
    def refuse(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=5.0,
            caller_principal=None,
            auth=_no_token,
            transport=httpx2.MockTransport(refuse),
        )
    )
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == ("upstream_error", "retry_after_delay")
    assert structured["retry_after_s"] == 5.0


async def test_an_op_spec_that_does_not_validate_refuses_before_any_call(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))
    result = await verbs.aggregate(frame_ref="not-a-pair", aggregation="sum")

    assert result.structured_content is not None
    assert result.structured_content["code"] == "op_spec_invalid"
    assert runner.requests == []


# resolve_anchor --------------------------------------------------------------------


async def test_resolve_anchor_asks_the_anchor_route_and_stores_the_record(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(
        httpx2.Response(
            200,
            json={
                "period": "2022",
                "rule": "half1",
                "left_out": [{"root": 0, "coordinates": {"observationAbout": "country/KEN"}}],
                "roots": [_FRAME],
            },
        )
    )
    result = await verbs.resolve_anchor(
        populations=[{"spec_id": _SPEC, "bindings": {}}], roots=[_FRAME]
    )

    assert runner.requests[-1].url.path == "/api/datasets/analysis/anchor"
    assert runner.body == {
        "roots": [{**_FRAME, "rendering": {"without_party_totals": False, "columns": []}}],
        "populations": [{"spec_id": _SPEC, "bindings": {}}],
        "latest": False,
    }
    structured = result.structured_content
    assert structured is not None
    assert structured["period"] == "2022"
    assert structured["left_out"] == [
        {"root": 0, "coordinates": {"observationAbout": "country/KEN"}}
    ]
    record = await store.get_anchor_record(structured["anchor_key"])
    assert record is not None
    assert record.period == "2022"
