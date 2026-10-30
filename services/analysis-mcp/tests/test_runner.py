"""The registered tools against dfo-apis' analysis runner: the tree a call sends,
the root a frame ref becomes, the caller it names, and the refusal each runner
answer maps to."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import httpx2
import pytest
from analysis_mcp import verbs
from analysis_mcp.digest import render_summary_digest
from analysis_mcp.runner_client import (
    DEFAULT_RETRY_AFTER_S,
    RunnerClient,
    read_derived_manifest,
    runner_holder,
)
from analysis_mcp.summary import LIST_CAP, VerbSummary
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from common.caller import supplied_caller
from common.materialisation import (
    AnchorRecord,
    FrameIntegrityError,
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
_OTHER_SPEC = "spec_" + "7" * 32
_OTHER_FRAME = {"spec_id": _OTHER_SPEC, "content_version": "cv_" + "8" * 32}
_ADDRESS_CALLER = "ip:" + "a" * 32
_SUM = {"by": ["time_period"], "op": "sum"}
_BREAKDOWN = {"by": ["crsPurpose", "time_period"], "op": "sum", "levels": {"crsPurpose": 1}}

type Handler = Callable[[httpx2.Request], httpx2.Response]


async def _no_token() -> dict[str, str]:
    return {}


class _Runner:
    """A runner answering every request with `answer`, or with what `answer` gives
    for the request's body, keeping each request."""

    def __init__(
        self, answer: httpx2.Response | Callable[[dict[str, Any]], httpx2.Response]
    ) -> None:
        self.answer = answer
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if isinstance(self.answer, httpx2.Response):
            return self.answer
        # A read has no body: `answer` gets its path.
        return self.answer(
            json.loads(request.content) if request.content else {"get": request.url.path}
        )

    @property
    def body(self) -> dict[str, Any]:
        """The body of the last POST, past any manifest read after it."""
        posts = [request for request in self.requests if request.method == "POST"]
        return json.loads(posts[-1].content)


def _install(
    answer: httpx2.Response | Callable[[dict[str, Any]], httpx2.Response],
    *,
    principal: str | None = None,
) -> _Runner:
    runner = _Runner(answer)
    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=5.0,
            read_timeout_s=5.0,
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


def _recorded_ready(verb: str | None = None) -> dict[str, Any]:
    """A 200 body dfo-apis' runner gave, for a call of `verb` when given, from the
    recorded replies."""
    replies = json.loads(REPLIES_PATH.read_text())
    return next(
        reply["body"]
        for key, reply in replies.items()
        if reply["status"] == 200
        and key.startswith("/api/datasets/analysis {")
        and verb in (None, reply["body"]["summary"]["verb"])
    )


def _reading(body: dict[str, Any], *roots: dict[str, str]) -> dict[str, Any]:
    """The ready answer `body` with its ref naming `roots` as the roots it read."""
    members = [
        {**member, "roots": [{**root, "rendering": {"format": "parquet"}} for root in roots]}
        for member in body["ref"]["members"]
    ]
    return {**body, "ref": {**body["ref"], "members": members}}


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
    assert "anchor_roots" not in runner.body
    assert "rolling" not in runner.body


@pytest.mark.parametrize(
    ("right_frame_ref", "rolling"),
    [(_DERIVED, "tree"), (_OTHER_FRAME, None)],
    ids=["derived-input", "fetched-frames"],
)
async def test_a_call_following_its_tree_flags_only_a_tree_with_a_derived_input(
    store: InMemoryMaterialisationStore, right_frame_ref: dict[str, str], rolling: str | None
) -> None:
    # A tree of fetched frames sends no flag, so a dfo-apis that does not know it
    # still answers.
    runner = _install(_problem(422, "op_spec_invalid"))
    await verbs.compare(
        frame_ref=_FRAME, right_frame_ref=right_frame_ref, rolling_follows_tree=True
    )

    assert runner.body.get("rolling") == rolling


_ROLLING = {
    "window_start": {"anchor": "latest", "offset": -1, "rule": "half1"},
    "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
    "population": [{"spec_id": _SPEC, "bindings": {}}],
}
_POPULATIONS = [{"spec_id": _SPEC, "bindings": {}}, {"spec_id": _OTHER_SPEC, "bindings": {}}]
_OTHER_DATASET = {"spec_id": _OTHER_SPEC, "content_version": _DATASET_CV}


async def _anchor_over_both(store: InMemoryMaterialisationStore) -> AnchorRecord:
    """An anchor record over `_FRAME` and `_OTHER_FRAME`, the second a fetched frame
    of the dataset version `_OTHER_DATASET` read with an enrichment column."""
    await store.put_mint_record(
        MintRecord(
            frame=FrameRef(**_OTHER_FRAME),
            dataset=FrameRef(**_OTHER_DATASET),
            options={"format": "parquet", "columns": ["observationAbout_type"]},
        )
    )
    record = AnchorRecord(
        populations=_POPULATIONS,
        roots=(FrameRef(**_FRAME), FrameRef(**_OTHER_FRAME)),
        rule="half1",
        period="2022",
    )
    await store.put_anchor_record(record)
    return record


async def _rolling_growth(frame_ref: dict[str, str], anchor_key: str) -> Any:
    return await verbs.growth(
        frame_ref=frame_ref,
        window_start="2021",
        window_end="2022",
        rolling={**_ROLLING, "population": _POPULATIONS},
        anchor_key=anchor_key,
    )


@pytest.mark.parametrize("frame_ref", [_FRAME, _DERIVED], ids=["fetched-frame", "derived-input"])
async def test_a_rolling_call_sends_its_op_spec_and_its_anchor_records_roots(
    store: InMemoryMaterialisationStore, frame_ref: dict[str, str]
) -> None:
    # The record was resolved over a second root beside the call's own frame, as
    # over both sides of a pairing whose one side the call grows. The second was
    # fetched with columns, which its rendering carries.
    record = await _anchor_over_both(store)
    anchor_roots = [
        {**_FRAME, "rendering": {"without_party_totals": False, "columns": []}},
        {
            "spec_id": _OTHER_SPEC,
            "content_version": _DATASET_CV,
            "rendering": {"without_party_totals": False, "columns": ["observationAbout_type"]},
        },
    ]
    ready = _reading(_recorded_ready("growth"), _FRAME, _OTHER_DATASET)

    def answer(body: dict[str, Any]) -> httpx2.Response:
        sent = sorted(body.get("anchor_roots", ()), key=lambda root: root["spec_id"])
        if sent == anchor_roots:
            return httpx2.Response(200, json=ready)
        return _problem(422, "op_spec_invalid")

    runner = _install(answer)
    result = await _rolling_growth(frame_ref, record.anchor_key)

    assert result.is_error is False, result.content
    # The harness's step states its populations, which a derived input's own
    # overlay keeps: the request leaves the input's rolling years as stored, so no
    # input's manifest is read and none is named as computed again.
    assert "rolling" not in runner.body
    assert [request.method for request in runner.requests] == ["POST"]
    assert "Rolling: the latest year is" in result.content[0].text
    assert "computed its rolling inputs again" not in result.content[0].text
    tree = runner.body["tree"]
    assert tree["verb"] == "growth"
    assert tree["arguments"]["window_start"] == "2021"
    assert tree["arguments"]["rolling"]["window_end"] == _ROLLING["window_end"]
    assert "anchor_key" not in tree["arguments"]
    assert "frame_ref" not in tree["arguments"]


@pytest.mark.parametrize(
    "frame_ref",
    # A frame of other data, and a derived input the runner computes from it.
    [{"spec_id": "spec_" + "9" * 32, "content_version": "cv_" + "9" * 32}, _DERIVED],
    ids=["fetched-frame", "derived-input"],
)
async def test_a_rolling_call_that_read_data_outside_its_anchor_record_refuses(
    store: InMemoryMaterialisationStore, frame_ref: dict[str, str]
) -> None:
    # The runner takes the latest period over the anchor roots alone, so it
    # answers a call over other data with a window that ends at their period.
    record = await _anchor_over_both(store)
    outside = {"spec_id": "spec_" + "9" * 32, "content_version": "cv_" + "9" * 32}
    _install(
        httpx2.Response(
            200, json=_reading(_recorded_ready("growth"), outside, _FRAME, _OTHER_DATASET)
        )
    )

    result = await _rolling_growth(frame_ref, record.anchor_key)

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "rolling_anchor_unresolved"
    assert structured["fix"] == {
        "detail": "root_not_in_record",
        "anchor_key": record.anchor_key,
        "root": outside,
    }


async def test_a_rolling_call_over_other_populations_than_its_record_refuses_before_any_call(
    store: InMemoryMaterialisationStore,
) -> None:
    record = await _anchor_over_both(store)
    runner = _install(_problem(422, "op_spec_invalid"))

    result = await verbs.growth(
        frame_ref=_FRAME,
        window_start="2021",
        window_end="2022",
        rolling=_ROLLING,
        anchor_key=record.anchor_key,
    )

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "rolling_anchor_unresolved"
    assert structured["fix"]["detail"] == "populations_differ"
    assert runner.requests == []


@pytest.mark.parametrize("code", ["snapshot_not_found", "artifact_expired"])
async def test_a_lost_anchor_root_refuses_the_anchor_not_a_frame_the_call_never_passed(
    store: InMemoryMaterialisationStore, code: str
) -> None:
    record = await _anchor_over_both(store)
    _install(_problem(410, code, fix={**_OTHER_DATASET, "latest": True}))

    result = await _rolling_growth(_FRAME, record.anchor_key)

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (
        "rolling_anchor_unresolved",
        "correct_and_retry",
    )
    assert structured["fix"] == {
        "detail": "anchor_root_lost",
        "anchor_key": record.anchor_key,
        "root": _OTHER_DATASET,
    }
    assert "resolve_anchor" in structured["message"]


async def _anchor_over_a_derived_root(store: InMemoryMaterialisationStore) -> str:
    record = AnchorRecord(
        populations=[{"spec_id": _SPEC, "bindings": {}}],
        roots=(FrameRef(**_FRAME), FrameRef(**_DERIVED)),
        rule="half1",
        period="2022",
    )
    await store.put_anchor_record(record)
    return record.anchor_key


async def _no_record(store: InMemoryMaterialisationStore) -> str:
    return "rak_" + "6" * 32


@pytest.mark.parametrize(
    ("anchor", "detail"),
    [
        (_no_record, "anchor_record_missing"),
        (_anchor_over_a_derived_root, "anchor_record_unreadable"),
    ],
    ids=["no-record", "derived-root"],
)
async def test_a_rolling_call_whose_anchor_record_cannot_be_sent_refuses_before_any_call(
    store: InMemoryMaterialisationStore,
    anchor: Callable[[InMemoryMaterialisationStore], Awaitable[str]],
    detail: str,
) -> None:
    anchor_key = await anchor(store)
    runner = _install(_problem(422, "op_spec_invalid"))
    result = await verbs.growth(
        frame_ref=_FRAME,
        window_start="2021",
        window_end="2022",
        rolling=_ROLLING,
        anchor_key=anchor_key,
    )

    assert result.structured_content is not None
    assert result.structured_content["code"] == "rolling_anchor_unresolved"
    assert result.structured_content["fix"]["detail"] == detail
    assert runner.requests == []


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
        (
            # data-mcp's flow enrichment order: the type columns, then party_kind,
            # party_class and party_whole. dfo-apis keeps a root's columns sorted.
            _record(
                {
                    "format": "parquet",
                    "columns": [
                        "financingProvider_type",
                        "financingRecipient_type",
                        "party_kind",
                        "party_class",
                        "party_whole",
                        "party_kind",
                    ],
                    "without_party_totals": True,
                }
            ),
            {
                "spec_id": _SPEC,
                "content_version": _DATASET_CV,
                "rendering": {
                    "without_party_totals": True,
                    "columns": [
                        "financingProvider_type",
                        "financingRecipient_type",
                        "party_class",
                        "party_kind",
                        "party_whole",
                    ],
                },
            },
        ),
    ],
    ids=[
        "no-mint-record",
        "mint-record",
        "stat-vars-sum",
        "breakdown-with-columns",
        "flow-columns-sorted",
    ],
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


class _UnreadableMintRecords(InMemoryMaterialisationStore):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        raise self.error


@pytest.mark.parametrize(
    ("error", "code", "retry_mode"),
    [
        (OSError("store unreachable"), "service_unavailable", "retry_after_delay"),
        (
            FrameIntegrityError(FrameRef(**_FRAME), reason="sha256 mismatch"),
            "frame_integrity_failed",
            "do_not_retry",
        ),
    ],
    ids=["unreachable", "corrupt"],
)
async def test_a_failed_mint_record_read_refuses_before_any_call(
    error: Exception, code: str, retry_mode: str
) -> None:
    set_store(_UnreadableMintRecords(error))
    try:
        runner = _install(_problem(422, "op_spec_invalid"))
        result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")
    finally:
        set_store(None)

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (code, retry_mode)
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


async def test_a_summary_tolerates_fields_dfo_apis_adds_to_its_exclusions_and_time(
    store: InMemoryMaterialisationStore,
) -> None:
    body = _recorded_ready("rank")
    summary = body["summary"]
    assert summary["exclusions"], "the recorded rank summary carries an exclusion"
    added = {
        **summary,
        "exclusions": [{**exclusion, "series": []} for exclusion in summary["exclusions"]],
        "time": {**summary["time"], "basis": "observed"},
    }
    _install(httpx2.Response(200, json={**body, "summary": added}))

    result = await verbs.rank(frame_ref=_FRAME, rank_by=["donor"])

    assert result.is_error is False, result.content
    assert result.structured_content is not None
    assert result.structured_content["exclusions"] == summary["exclusions"]


@pytest.mark.parametrize(
    ("verb", "field"),
    [("growth", "window_start"), ("growth", "window_end"), ("aggregate", "aggregation")],
)
async def test_a_summary_without_a_parameter_its_digest_reads_refuses_naming_it(
    store: InMemoryMaterialisationStore, verb: str, field: str
) -> None:
    body = _recorded_ready(verb)
    summary = body["summary"]
    parameters = {key: value for key, value in summary["parameters"].items() if key != field}
    _install(httpx2.Response(200, json={**body, "summary": {**summary, "parameters": parameters}}))

    if verb == "growth":
        result = await verbs.growth(frame_ref=_FRAME, window_start="2017", window_end="2024")
    else:
        result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["upstream_code"]) == (
        "upstream_error",
        "response_invalid",
    )
    assert f"parameters.{field}" in structured["fix"]["detail"]


@pytest.mark.parametrize(
    ("verb", "entries", "count", "value"),
    [
        ("join", "matched_keys", "matched_key_count", "left_value"),
        ("filter", "kept_row_summaries", "kept_row_summary_count", "value"),
    ],
)
def test_a_summary_holds_only_the_entries_its_digest_shows(
    verb: str, entries: str, count: str, value: str
) -> None:
    summary = _recorded_ready(verb)["summary"]
    entry = summary["result"][entries][0]
    listed = [{**entry, value: float(index)} for index in range(LIST_CAP * 5)]

    parsed = VerbSummary.parse({**summary, "result": {**summary["result"], entries: listed}})

    held = getattr(parsed.result, entries)
    assert getattr(parsed.result, count) == LIST_CAP * 5
    # The largest values, in the order the runner listed them.
    assert [getattr(item, value) for item in held] == [
        float(index) for index in range(LIST_CAP * 4, LIST_CAP * 5)
    ]
    assert f"{LIST_CAP * 5} " in render_summary_digest(parsed)


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


@pytest.mark.parametrize("code", ["snapshot_not_found", "recipe_not_found", "artifact_expired"])
async def test_a_lost_root_behind_a_derived_input_refuses_that_input_unrecoverable(
    store: InMemoryMaterialisationStore, code: str
) -> None:
    # The runner computes the `ops_` input again from its stored recipe, whose
    # dataset root it no longer holds: the fix names a pair the call never sent.
    root = {"spec_id": _SPEC, "content_version": _DATASET_CV, "latest": True}
    _install(_problem(404, code, fix=root))
    result = await verbs.aggregate(frame_ref=_DERIVED, aggregation="sum")

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (
        "derived_input_unrecoverable",
        "do_not_retry",
    )
    assert structured["fix"] == {
        "frame_refs": [_DERIVED],
        "root": {"spec_id": _SPEC, "content_version": _DATASET_CV},
    }


@pytest.mark.parametrize(
    ("follows_tree", "code", "fix"),
    [
        (True, "rolling_reanchor_unavailable", {"frame_refs": [_DERIVED], "tool": "get_data"}),
        # A call that keeps each input's stored years refuses as the frame lost.
        (False, "frame_not_held", None),
    ],
    ids=["following-the-tree", "harness"],
)
async def test_a_derived_input_with_no_recipe_refuses_by_whether_it_follows_the_tree(
    store: InMemoryMaterialisationStore,
    follows_tree: bool,
    code: str,
    fix: dict[str, Any] | None,
) -> None:
    _install(_problem(404, "recipe_not_found", fix=_DERIVED))
    result = await verbs.compare(
        frame_ref=_FRAME, right_frame_ref=_DERIVED, rolling_follows_tree=follows_tree
    )

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (code, "correct_and_retry")
    if fix is not None:
        assert structured["fix"] == fix


async def test_an_input_whose_manifest_cannot_be_read_is_not_named_as_computed_again(
    store: InMemoryMaterialisationStore,
) -> None:
    ready = _recorded_ready("growth")
    assert ready["resolved_periods"]

    def answer(body: dict[str, Any]) -> httpx2.Response:
        return (
            _problem(404, "snapshot_not_found")
            if "get" in body
            else httpx2.Response(200, json=ready)
        )

    runner = _install(answer)
    result = await verbs.compare(
        frame_ref=_FRAME, right_frame_ref=_DERIVED, rolling_follows_tree=True
    )

    assert [request.method for request in runner.requests] == ["POST", "GET"]
    text = result.content[0].text
    assert "Rolling: the latest year is" in text
    assert "computed its rolling inputs again" not in text


@pytest.mark.parametrize(
    ("code", "lost"),
    [("snapshot_not_found", "frame_not_held"), ("artifact_expired", "frame_expired")],
)
async def test_both_sides_read_from_one_lost_dataset_version_are_named_in_one_refusal(
    store: InMemoryMaterialisationStore, code: str, lost: str
) -> None:
    # Two frames fetched from one dataset version with different columns: the
    # runner reads both as one root, so losing it loses both.
    second = {"spec_id": "spec_" + "6" * 32, "content_version": "cv_" + "6" * 32}
    await store.put_mint_record(_record({"format": "parquet"}))
    await store.put_mint_record(
        MintRecord(
            frame=FrameRef(**second),
            dataset=FrameRef(spec_id=_SPEC, content_version=_DATASET_CV),
            options={"format": "parquet", "columns": ["observationAbout_type"]},
        )
    )
    _install(_problem(410, code, fix={"spec_id": _SPEC, "content_version": _DATASET_CV}))

    result = await verbs.compare(frame_ref=_FRAME, right_frame_ref=second)

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == lost
    assert structured["fix"]["frame_refs"] == [second, _FRAME]


async def test_a_lost_root_names_only_the_derived_input_computed_from_it(
    store: InMemoryMaterialisationStore,
) -> None:
    other = {"spec_id": "ops_" + "a" * 32, "content_version": "ocv_" + "a" * 32}
    lost = {"spec_id": _SPEC, "content_version": _DATASET_CV}
    roots = {_OPS: [lost], other["spec_id"]: [_OTHER_DATASET]}
    reads: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            spec_id = request.url.path.split("/")[-2]
            reads.append(spec_id)
            return httpx2.Response(200, json={"roots": roots[spec_id]})
        return _problem(404, "snapshot_not_found", fix={**lost, "latest": True})

    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=5.0,
            read_timeout_s=5.0,
            caller_principal=None,
            auth=_no_token,
            transport=httpx2.MockTransport(handle),
        )
    )
    result = await verbs.share_of(frame_ref=_DERIVED, denominator_frame_ref=other)

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "derived_input_unrecoverable"
    assert structured["fix"]["frame_refs"] == [_DERIVED]
    assert sorted(reads) == sorted(roots)


@pytest.mark.parametrize(
    ("operations", "queued"), [(["op_1"], True), ([], False)], ids=["rebuilding", "in-flight"]
)
async def test_a_pending_answer_counts_its_rebuilds_only_when_it_names_some(
    store: InMemoryMaterialisationStore, operations: list[str], queued: bool
) -> None:
    body = {"status": "pending", "retry_after_s": 2.0, "operations": operations}
    _install(httpx2.Response(202, json=body))
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    assert ("rebuild(s) queued" in result.content[0].text) is queued


async def test_an_unreachable_runner_refuses_retryable_upstream_error(
    store: InMemoryMaterialisationStore,
) -> None:
    def refuse(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=5.0,
            read_timeout_s=5.0,
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


@pytest.mark.parametrize(
    ("call", "argument"),
    [
        (lambda: verbs.aggregate(frame_ref={"spec_id": _OPS}, aggregation="sum"), "frame_ref"),
        (
            lambda: verbs.aggregate(frame_ref={"spec_id": "x" * 100_000}, aggregation="sum"),
            "frame_ref",
        ),
        (
            lambda: verbs.compare(frame_ref=_FRAME, right_frame_ref={"spec_id": _OPS}),
            "right_frame_ref",
        ),
        (
            lambda: verbs.share_of(frame_ref=_FRAME, denominator_frame_ref={"spec_id": _OPS}),
            "denominator_frame_ref",
        ),
    ],
    ids=["frame_ref", "long-id", "right_frame_ref", "denominator_frame_ref"],
)
async def test_a_frame_argument_without_its_content_version_refuses_before_any_call(
    store: InMemoryMaterialisationStore, call: Callable[[], Any], argument: str
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))
    result = await call()

    assert result.structured_content is not None
    assert (result.structured_content["code"], result.structured_content["retry_mode"]) == (
        "frame_ref_incomplete",
        "correct_and_retry",
    )
    assert result.structured_content["fix"]["argument"] == argument
    # A caller's id is echoed capped, so no argument makes the refusal any size.
    assert len(_text(result)) < 1_000
    assert runner.requests == []


# Bare spec_ids, lost frames and place_level -----------------------------------------

_NEWER_CV = "cv_" + "9" * 32
_REPLAYED = {"spec_id": "ops_" + "7" * 32, "content_version": "ocv_" + "6" * 32}


def _text(result: Any) -> str:
    return result.content[0].text


def _ready_as(pair: dict[str, str]) -> dict[str, Any]:
    """A recorded aggregate answer whose ref names the stored frame `pair`."""
    body = _recorded_ready("aggregate")
    member = body["ref"]["members"][0]
    member = {
        **member,
        "spec": {**member["spec"], "spec_id": pair["spec_id"]},
        "content_version": pair["content_version"],
    }
    return {**body, "ref": {**body["ref"], "members": [member]}}


# The frame a Spec root was read as, as the result's stored recipe records it.
_READ_FRAME = {"spec_id": _SPEC, "content_version": "cv_" + "a" * 32}
_READ_RECORD = MintRecord(
    frame=FrameRef(**_READ_FRAME),
    dataset=FrameRef(spec_id=_SPEC, content_version=_NEWER_CV),
    options={
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "columns": ["observationAbout_type"],
    },
)


def _reading_spec(body: dict[str, Any]) -> httpx2.Response:
    """The runner's answer to a call that reads `_SPEC` at its newest version, and to
    the read of its result's manifest, whose recipe records the frame read."""
    if "get" in body:
        return httpx2.Response(
            200, json={"recipe": {"roots": [_READ_RECORD.model_dump(mode="json")]}}
        )
    read = {"spec_id": _SPEC, "content_version": _NEWER_CV}
    return httpx2.Response(200, json=_reading(_recorded_ready("aggregate"), read))


def _posts(runner: _Runner) -> list[dict[str, Any]]:
    return [json.loads(request.content) for request in runner.requests if request.content]


async def test_a_bare_spec_id_is_sent_as_its_spec_and_the_frame_it_read_is_recorded(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_reading_spec)
    result = await verbs.aggregate(frame_ref={"spec_id": _SPEC}, aggregation="sum")

    assert result.is_error is False, _text(result)
    assert _posts(runner)[-1]["tree"]["inputs"] == {"data": {"spec_id": _SPEC}}
    # The digest names the frame's pair, and the store holds its mint record, so a
    # later call naming the pair reads the frame this call read.
    assert _READ_FRAME["content_version"] in _text(result)
    assert await store.get_mint_record(FrameRef(**_READ_FRAME)) == _READ_RECORD


async def test_a_bare_spec_id_whose_read_cannot_be_recorded_names_no_pair(
    store: InMemoryMaterialisationStore,
) -> None:
    def answer(body: dict[str, Any]) -> httpx2.Response:
        return _problem(503, "service_unavailable") if "get" in body else _reading_spec(body)

    _install(answer)
    result = await verbs.aggregate(frame_ref={"spec_id": _SPEC}, aggregation="sum")

    assert result.is_error is False, _text(result)
    assert _READ_FRAME["content_version"] not in _text(result)
    assert await store.get_mint_record(FrameRef(**_READ_FRAME)) is None


@pytest.mark.parametrize(
    ("record", "fetched", "lost"),
    [
        (
            _record({"format": "parquet", "columns": ["observationAbout_type"]}),
            {
                "spec_id": _SPEC,
                "rendering": {"without_party_totals": False, "columns": ["observationAbout_type"]},
            },
            {"spec_id": _SPEC, "content_version": _DATASET_CV},
        ),
        (
            _record(
                {
                    "format": "parquet",
                    "without_party_totals": True,
                    "groupby": {"by": ["stat_var"]},
                }
            ),
            {
                "spec_id": _SPEC,
                "rendering": {
                    "without_party_totals": True,
                    "columns": [],
                    "groupby": {"by": ["stat_var"]},
                },
            },
            {"spec_id": _SPEC, "content_version": _DATASET_CV},
        ),
    ],
    ids=["columns", "groupby-and-cut"],
)
async def test_a_frame_the_runner_lost_is_fetched_again_at_its_newest_version(
    store: InMemoryMaterialisationStore,
    record: MintRecord,
    fetched: dict[str, Any],
    lost: dict[str, str],
) -> None:
    await store.put_mint_record(record)

    def answer(body: dict[str, Any]) -> httpx2.Response:
        if "tree" in body and "content_version" in body["tree"]["inputs"]["data"]:
            return _problem(404, "snapshot_not_found", fix=lost)
        return _reading_spec(body)

    runner = _install(answer)
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum", rebuild_lost_frames=True)

    assert result.is_error is False, _text(result)
    assert _posts(runner)[-1]["tree"]["inputs"] == {"data": fetched}
    # The frame's pair moved: the digest names the one it had and the one it has.
    assert _CV in _text(result)
    assert _READ_FRAME["content_version"] in _text(result)
    assert await store.get_mint_record(FrameRef(**_READ_FRAME)) == _READ_RECORD


async def test_a_lost_frame_no_record_states_refuses_with_no_rebuild(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_problem(404, "snapshot_not_found", fix=_FRAME))
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum", rebuild_lost_frames=True)

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (
        "rendering_not_recorded",
        "correct_and_retry",
    )
    assert structured["fix"]["tool"] == "get_data"
    assert structured["fix"]["options"] == []
    assert len(runner.requests) == 1


class _RecordUnreadableAfterFirstRead(InMemoryMaterialisationStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        self.reads += 1
        if self.reads > 1:
            raise OSError("store unreachable")
        return await super().get_mint_record(ref)


async def test_a_lost_frame_whose_record_read_fails_refuses_retryably() -> None:
    set_store(_RecordUnreadableAfterFirstRead())
    try:
        runner = _install(_problem(404, "snapshot_not_found", fix=_FRAME))
        result = await verbs.aggregate(
            frame_ref=_FRAME, aggregation="sum", rebuild_lost_frames=True
        )
    finally:
        set_store(None)

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_after_s"]) == (
        "service_unavailable",
        DEFAULT_RETRY_AFTER_S,
    )
    assert len(runner.requests) == 1


@pytest.mark.parametrize(
    "resent",
    [
        httpx2.Response(504, json={"code": "timeout", "detail": "slow", "retryable": True}),
        httpx2.Response(502, text="bad gateway"),
    ],
    ids=["timeout", "no-problem-document"],
)
async def test_a_transient_failure_of_the_call_sent_again_stays_retryable(
    store: InMemoryMaterialisationStore, resent: httpx2.Response
) -> None:
    await store.put_mint_record(_record({"format": "parquet"}))
    lost = {"spec_id": _SPEC, "content_version": _DATASET_CV}

    def answer(body: dict[str, Any]) -> httpx2.Response:
        if "content_version" in body["tree"]["inputs"]["data"]:
            return _problem(404, "snapshot_not_found", fix=lost)
        return resent

    _install(answer)
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum", rebuild_lost_frames=True)

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (
        "upstream_error",
        "retry_after_delay",
    )
    assert structured["retry_after_s"] is not None


async def test_a_derived_input_that_lost_its_root_is_computed_again_on_current_data(
    store: InMemoryMaterialisationStore,
) -> None:
    lost_root = {"spec_id": _SPEC, "content_version": _DATASET_CV}

    def answer(body: dict[str, Any]) -> httpx2.Response:
        if body["latest"]:
            return httpx2.Response(200, json=_ready_as(_REPLAYED))
        if body["tree"]["inputs"]["data"] == _DERIVED:
            return _problem(404, "snapshot_not_found", fix=lost_root)
        return httpx2.Response(200, json=_recorded_ready("aggregate"))

    runner = _install(answer)
    result = await verbs.aggregate(frame_ref=_DERIVED, aggregation="sum", rebuild_lost_frames=True)

    bodies = [json.loads(request.content) for request in runner.requests]
    assert result.is_error is False, _text(result)
    assert bodies[1] == {"tree": _DERIVED, "latest": True}
    assert bodies[2]["tree"]["inputs"] == {"data": _REPLAYED}
    assert bodies[2]["latest"] is False
    assert _OCV in _text(result)
    # A replay can mint a new `ops_` id: the digest names the pair a later call sends.
    assert f"{_REPLAYED['spec_id']}, {_REPLAYED['content_version']}" in _text(result)


@pytest.mark.parametrize(
    ("frame_ref", "lost", "code", "retry_mode"),
    [
        (
            _FRAME,
            {"spec_id": _SPEC, "content_version": _DATASET_CV},
            "frame_not_held",
            "correct_and_retry",
        ),
        (
            _DERIVED,
            {"spec_id": _SPEC, "content_version": _DATASET_CV},
            "derived_input_unrecoverable",
            "do_not_retry",
        ),
    ],
    ids=["fetched", "derived"],
)
async def test_a_frame_the_server_cannot_rebuild_refuses_as_the_first_call_did(
    store: InMemoryMaterialisationStore,
    frame_ref: dict[str, str],
    lost: dict[str, str],
    code: str,
    retry_mode: str,
) -> None:
    await store.put_mint_record(_record({"format": "parquet"}))
    runner = _install(_problem(404, "snapshot_not_found", fix=lost))
    unrebuilt = await verbs.aggregate(
        frame_ref=frame_ref, aggregation="sum", rebuild_lost_frames=True
    )

    structured = unrebuilt.structured_content
    assert structured is not None
    assert (structured["code"], structured["retry_mode"]) == (code, retry_mode)
    # The call, then one rebuild: the Spec fetched again, or the recipe replayed.
    assert len(runner.requests) == 2


def _losing_dataset_versions(body: dict[str, Any]) -> httpx2.Response:
    """The runner's answer when it holds no dataset version: like `resolve_roots`, it
    refuses on the first side read at one, and reads a Spec root at its newest."""
    if "get" in body:
        return _reading_spec(body)
    for node in body["tree"]["inputs"].values():
        if "content_version" in node:
            pair = {"spec_id": node["spec_id"], "content_version": node["content_version"]}
            return _problem(404, "snapshot_not_found", fix=pair)
    return _reading_spec(body)


async def test_a_call_whose_two_sides_were_both_lost_rebuilds_each_side(
    store: InMemoryMaterialisationStore,
) -> None:
    other = FrameRef(**_OTHER_FRAME)
    await store.put_mint_record(_record({"format": "parquet"}))
    await store.put_mint_record(MintRecord(frame=other, dataset=other, options={}))
    runner = _install(_losing_dataset_versions)
    result = await verbs.share_of(
        frame_ref=_FRAME, denominator_frame_ref=_OTHER_FRAME, rebuild_lost_frames=True
    )

    assert result.is_error is False, _text(result)
    rendering = {"without_party_totals": False, "columns": []}
    assert _posts(runner)[-1]["tree"]["inputs"] == {
        "data": {"spec_id": _SPEC, "rendering": rendering},
        "other": {"spec_id": _OTHER_SPEC, "rendering": rendering},
    }


async def test_a_lost_frame_on_a_call_that_does_not_rebuild_returns_the_runners_refusal(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_losing_dataset_versions)
    result = await verbs.aggregate(frame_ref=_FRAME, aggregation="sum")

    unchanged = to_tool_result(ToolRefusal.frames_lost(frame_refs=[_FRAME], expired=False))
    assert result.structured_content == unchanged.structured_content
    assert _text(result) == _text(unchanged)
    assert len(runner.requests) == 1


@pytest.mark.parametrize(
    ("replay", "code", "retry_after_s", "logged"),
    [
        (
            httpx2.Response(202, json={"status": "pending", "retry_after_s": 4.0}),
            "service_unavailable",
            4.0,
            False,
        ),
        (
            _problem(503, "service_unavailable", retryable=True),
            "service_unavailable",
            DEFAULT_RETRY_AFTER_S,
            False,
        ),
        (
            httpx2.Response(504, json={"code": "timeout", "detail": "slow", "retryable": True}),
            "upstream_error",
            DEFAULT_RETRY_AFTER_S,
            True,
        ),
        (
            _problem(500, "internal_error"),
            "derived_input_unrecoverable",
            None,
            True,
        ),
    ],
    ids=["pending", "busy", "timeout", "failed"],
)
async def test_a_derived_input_whose_replay_cannot_run_yet_refuses_with_its_delay(
    store: InMemoryMaterialisationStore,
    caplog: pytest.LogCaptureFixture,
    replay: httpx2.Response,
    code: str,
    retry_after_s: float | None,
    logged: bool,
) -> None:
    lost_root = {"spec_id": _SPEC, "content_version": _DATASET_CV}

    def answer(body: dict[str, Any]) -> httpx2.Response:
        return replay if body["latest"] else _problem(404, "snapshot_not_found", fix=lost_root)

    _install(answer)
    with caplog.at_level(logging.WARNING, logger="analysis_mcp.runner_client"):
        result = await verbs.aggregate(
            frame_ref=_DERIVED, aggregation="sum", rebuild_lost_frames=True
        )

    structured = result.structured_content
    assert structured is not None
    assert (structured["code"], structured.get("retry_after_s")) == (code, retry_after_s)
    assert any(_OPS in record.getMessage() for record in caplog.records) is logged


def _level_column(node: dict[str, Any]) -> str:
    """The column of the place_level predicate of the filter step `node`."""
    assert node["verb"] == "filter"
    (predicate,) = node["arguments"]["predicates"]
    assert (predicate["kind"], predicate["level"]) == ("place_level", "Country")
    return predicate["column"]


@pytest.mark.parametrize(
    ("call", "columns"),
    [
        (
            lambda: verbs.rank(
                frame_ref=_FRAME, rank_by=("observationAbout",), place_level="Country"
            ),
            {"data": "observationAbout_type"},
        ),
        (
            lambda: verbs.aggregate(frame_ref=_FRAME, aggregation="sum", place_level="Country"),
            {"data": "observationAbout_type"},
        ),
        (
            lambda: verbs.share_of(
                frame_ref=_FRAME,
                denominator_frame_ref=_OTHER_FRAME,
                numerator_by=["financingRecipient"],
                denominator_by={"financingRecipient": "observationAbout"},
                place_level="Country",
            ),
            {"data": "financingRecipient_type", "other": "observationAbout_type"},
        ),
        (
            lambda: verbs.share_of(
                frame_ref=_FRAME,
                denominator_frame_ref=_OTHER_FRAME,
                numerator_by=["financingRecipient"],
                place_level="Country",
            ),
            {"data": "financingRecipient_type", "other": "observationAbout_type"},
        ),
    ],
    ids=["rank", "aggregate-pooled-place", "share_of-each-side", "share_of-pooled-denominator"],
)
async def test_place_level_filters_each_input_on_the_type_of_the_role_it_keys_by(
    store: InMemoryMaterialisationStore, call: Callable[[], Any], columns: dict[str, str]
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))
    await call()

    inputs = runner.body["tree"]["inputs"]
    assert {name: _level_column(node) for name, node in inputs.items()} == columns
    assert inputs["data"]["inputs"]["data"]["spec_id"] == _SPEC


@pytest.mark.parametrize(
    ("fix", "place_level"),
    [
        ({"column": "observationAbout_type", "types": ["Country", "Place"]}, "Country"),
        (
            {"column": "observationAbout_type", "types": ["AdministrativeArea1", "Place"]},
            "AdministrativeArea1",
        ),
        ({"column": "observationAbout_type", "types": ["Place", "ProvisionalNode"]}, None),
        ({"column": "financingRecipient_party_kind", "types": []}, None),
    ],
    ids=["country", "another-level", "no-level", "no-kind-column"],
)
async def test_a_runner_place_levels_mixed_names_the_level_to_keep(
    store: InMemoryMaterialisationStore, fix: dict[str, Any], place_level: str | None
) -> None:
    _install(_problem(422, "place_levels_mixed", fix=fix))
    result = await verbs.rank(frame_ref=_FRAME, rank_by=("observationAbout",))

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "place_levels_mixed"
    assert structured["fix"].get("place_level") == place_level
    # The message is this service's, written from the fix: the engine's names calls
    # this service's tools do not take.
    assert "place_levels_mixed detail" not in structured["message"]


async def test_a_malformed_bare_spec_id_refuses_as_an_argument_error(
    store: InMemoryMaterialisationStore,
) -> None:
    runner = _install(_problem(422, "op_spec_invalid"))

    result = await verbs.growth(
        frame_ref={"spec_id": "spec_abc"}, window_start="2021", window_end="2022"
    )

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "op_spec_invalid"
    assert structured["retry_mode"] == "correct_and_retry"
    assert runner.requests == []


async def test_a_tool_without_place_level_is_told_to_filter(
    store: InMemoryMaterialisationStore,
) -> None:
    fix = {"column": "observationAbout_type", "types": ["Country", "Place"]}
    _install(_problem(422, "place_levels_mixed", fix=fix))

    result = await verbs.quotient_of(
        frame_ref=_FRAME, denominator_frame_ref=_OTHER_FRAME, numerator_unit="USD"
    )

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "place_levels_mixed"
    assert structured["fix"] == fix
    assert "place_level set to" not in structured["message"]
    assert "place_level predicate" in structured["message"]


async def test_a_party_refusal_keeps_its_own_advice_on_a_tool_without_place_level(
    store: InMemoryMaterialisationStore,
) -> None:
    fix = {
        "column": "financingRecipient_party_kind",
        "types": ["party", "total"],
        "whole_total": "DAC countries",
        "member_count": 2,
        "example_members": ["France"],
    }
    _install(_problem(422, "place_levels_mixed", fix=fix))

    result = await verbs.quotient_of(
        frame_ref=_FRAME, denominator_frame_ref=_OTHER_FRAME, numerator_unit="USD"
    )

    structured = result.structured_content
    assert structured is not None
    assert structured["fix"]["whole_total"] == "DAC countries"


async def test_a_call_with_no_anchor_key_is_not_told_to_resolve_one(
    store: InMemoryMaterialisationStore,
) -> None:
    fix = {"detail": "no_qualifying_period"}
    engine = (
        "The latest period of the data is not established: r. Call resolve_anchor with the "
        "populations and every root frame of the question, then retry with the anchor_key "
        "and the periods it gives."
    )
    _install(_problem(422, "rolling_anchor_unresolved", fix=fix, detail=engine))

    result = await verbs.growth(
        frame_ref=_FRAME,
        window_start={"anchor": "latest", "offset": -5},
        window_end={"anchor": "latest", "offset": 0},
    )

    structured = result.structured_content
    assert structured is not None
    assert structured["code"] == "rolling_anchor_unresolved"
    assert structured["fix"] == fix
    assert "resolve_anchor" not in structured["message"]
    assert "anchor_key" not in structured["message"]


# The merged forms ------------------------------------------------------------------

_ONE_FRAME = {"frame_ref": _FRAME}
_TWO_FRAMES = {"frame_ref": _FRAME, "right_frame_ref": _DERIVED}
_KEYED_SHARE = {
    "frame_ref": _FRAME,
    "denominator_frame_ref": _DERIVED,
    "numerator_by": ["recipient"],
    "denominator_by": {"recipient": "observationAbout"},
}


@pytest.mark.parametrize(
    ("merged", "own", "arguments"),
    [
        (
            functools.partial(verbs.compare, operation="sum"),
            verbs.sum_of,
            _TWO_FRAMES,
        ),
        (functools.partial(verbs.compare, operation="sum"), verbs.sum_of, _ONE_FRAME),
        (
            functools.partial(verbs.share_of, operation="quotient"),
            verbs.quotient_of,
            {"frame_ref": _FRAME, "denominator_frame_ref": _DERIVED},
        ),
        (
            functools.partial(verbs.share_of, operation="quotient"),
            verbs.quotient_of,
            _KEYED_SHARE,
        ),
        (verbs.compare, functools.partial(verbs.compare, operation="difference"), _TWO_FRAMES),
        (verbs.share_of, functools.partial(verbs.share_of, operation="share"), _KEYED_SHARE),
    ],
    ids=["sum", "sum-one-frame", "quotient", "quotient-keyed", "difference", "share"],
)
@pytest.mark.parametrize("answer", ["ready", "refused"])
async def test_a_merged_form_sends_and_replies_as_its_own_tool(
    store: InMemoryMaterialisationStore,
    merged: Callable[..., Any],
    own: Callable[..., Any],
    arguments: dict[str, Any],
    answer: str,
) -> None:
    """`compare(operation="sum")` is `sum_of` and `share_of(operation="quotient")` is
    `quotient_of`: the same tree to the runner, and the same reply or refusal."""
    response = (
        httpx2.Response(200, json=_recorded_ready())
        if answer == "ready"
        else _problem(422, "comparison_periods_disjoint", fix={"left_periods": ["2020"]})
    )
    replies = []
    bodies = []
    for tool in (merged, own):
        runner = _install(response)
        replies.append(await tool(**arguments))
        bodies.append(runner.body)

    assert bodies[0] == bodies[1]
    assert replies[0].structured_content == replies[1].structured_content
    assert replies[0].content == replies[1].content
    assert replies[0].is_error is replies[1].is_error is (answer == "refused")


_CONVERTED = {"base": "dcid:USDollar", "price_basis": "constant", "base_year": 2022}
_COMPOSED = {"numerator": "dcid:USDollar", "denominator": "dcid:Person"}


@pytest.mark.parametrize(
    ("units", "fix"),
    [
        (
            {"numerator": "dcid:CurrencyUSD", "denominator": "dcid:USDollar"},
            {"operation": "quotient"},
        ),
        (
            {"numerator": _CONVERTED, "denominator": "dcid:USDollar"},
            {"numerator": _CONVERTED, "denominator": "dcid:USDollar"},
        ),
        (
            {"numerator": _COMPOSED, "denominator": "dcid:USDollar"},
            {"numerator": _COMPOSED, "denominator": "dcid:USDollar"},
        ),
        (
            {"numerator": "Percent", "denominator": "dcid:USDollar"},
            {"numerator": "Percent", "denominator": "dcid:USDollar"},
        ),
    ],
    ids=["two-refs", "converted", "composed", "percent"],
)
async def test_a_share_across_two_unit_refs_refuses_naming_the_quotient_form(
    store: InMemoryMaterialisationStore, units: dict[str, Any], fix: dict[str, Any]
) -> None:
    _install(_problem(422, "cross_frame_unit_mismatch", fix=units))
    share = await verbs.share_of(frame_ref=_FRAME, denominator_frame_ref=_DERIVED)
    _install(_problem(422, "cross_frame_unit_mismatch", fix={"left": "a", "right": "b"}))
    difference = await verbs.compare(frame_ref=_FRAME, right_frame_ref=_DERIVED)

    assert share.structured_content is not None
    assert (share.structured_content["code"], share.structured_content["fix"]) == (
        "cross_frame_unit_mismatch",
        fix,
    )
    assert difference.structured_content is not None
    assert difference.structured_content["fix"] == {"left": "a", "right": "b"}


# Derived frame reads --------------------------------------------------------------


async def test_a_derived_frame_read_waits_the_read_timeout_not_the_compute_timeout() -> None:
    reads: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        reads.append(request)
        return httpx2.Response(200, json={"roots": []})

    runner_holder.set(
        RunnerClient(
            base_url="http://runner.test",
            timeout_s=130.0,
            read_timeout_s=7.0,
            caller_principal=None,
            auth=_no_token,
            transport=httpx2.MockTransport(handle),
        )
    )
    await read_derived_manifest(FrameRef(**_DERIVED))

    assert reads[0].extensions["timeout"]["read"] == 7.0


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
