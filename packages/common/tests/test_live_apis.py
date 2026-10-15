"""`common.live_apis`: each route's request and its parsing of answers and refusals."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx2
import pytest
from common.apis_auth import TOKEN_HEADER, ApisAuthError, AuthProvider
from common.live_apis import (
    FRAME_NOT_REFRESHABLE,
    PRINCIPAL_HEADER,
    PUBLISH_CONFLICT,
    RECIPE_NOT_FOUND,
    REFRESH_NOT_PERMITTED,
    RENDER_TARGET_MISMATCH,
    RENDERING_NOT_RECORDED,
    RESPONSE_INVALID,
    Current,
    DatasetVersion,
    FramePair,
    LiveApisClient,
    LiveApisError,
    LiveApisNotConfigured,
    LiveApisTransportError,
    Moved,
    Pending,
    RecipeWritten,
)

DATASETS_PRINCIPAL = "datasets-principal-7f3a"
LIVE_PRINCIPAL = "live-principal-91bc"
OPS = "ops_" + "a" * 16
OCV = "ocv_" + "b" * 16
SPEC = "spec_" + "c" * 16
CV = "cv_" + "d" * 16
FRAME = FramePair(spec_id=SPEC, content_version=CV)
RECIPE: dict[str, Any] = {
    "schema_version": 1,
    "normal_form_hash": "rnf_" + "0" * 32,
    "replay_key": "rpk_" + "1" * 32,
    "execution_identity": "engine|grammar|reduction",
    "tree": {"frame": OPS},
    "roots": [],
    "anchor": None,
}

Handler = Callable[[httpx2.Request], httpx2.Response]


class Calls:
    """The requests a handler saw, and an optional answer per request."""

    def __init__(self, *answers: httpx2.Response | Exception) -> None:
        self.requests: list[httpx2.Request] = []
        self._answers = list(answers)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_client(
    handler: Handler,
    *,
    datasets_principal: str | None = DATASETS_PRINCIPAL,
    live_refresh_principal: str | None = LIVE_PRINCIPAL,
    auth: AuthProvider | None = None,
    max_attempts: int = 1,
) -> LiveApisClient:
    async def no_sleep(_: float) -> None:
        return None

    return LiveApisClient(
        base_url="http://apis.test",
        datasets_principal=datasets_principal,
        live_refresh_principal=live_refresh_principal,
        auth=auth,
        transport=httpx2.MockTransport(handler),
        max_attempts=max_attempts,
        sleep=no_sleep,
    )


class CountingAuth:
    """An auth hook that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> Mapping[str, str]:
        self.calls += 1
        return {TOKEN_HEADER: f"Bearer token-{self.calls}"}


def ok(body: object, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=body)


def problem(
    status: int, code: str, *, constraint_id: str | None = None, retryable: bool = False
) -> httpx2.Response:
    body: dict[str, Any] = {"code": code, "retryable": retryable, "detail": "refused"}
    if constraint_id is not None:
        body["constraint_id"] = constraint_id
    return httpx2.Response(status, json=body, headers={"content-type": "application/problem+json"})


def body_of(request: httpx2.Request) -> Any:
    return json.loads(request.content)


# put_recipe


async def test_put_recipe_sends_the_recipe_with_the_datasets_principal() -> None:
    calls = Calls(ok({"spec_id": OPS, "content_version": OCV, "created": True}))

    async with make_client(calls) as client:
        written = await client.put_recipe(spec_id=OPS, content_version=OCV, recipe=RECIPE)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("PUT", f"/api/datasets/frames/{OPS}/{OCV}/recipe")
    assert request.headers[PRINCIPAL_HEADER] == DATASETS_PRINCIPAL
    assert body_of(request) == RECIPE
    assert written == RecipeWritten(frame=FramePair(OPS, OCV), created=True)


@pytest.mark.parametrize(
    ("constraint_id"),
    [
        "recipe_requires_matching_normal_form_hash",
        "recipe_requires_matching_replay_key",
        "replay_key_requires_one_frame",
    ],
)
async def test_put_recipe_conflict_carries_its_constraint(constraint_id: str) -> None:
    calls = Calls(problem(409, PUBLISH_CONFLICT, constraint_id=constraint_id))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.put_recipe(spec_id=OPS, content_version=OCV, recipe=RECIPE)

    assert (raised.value.status, raised.value.code) == (409, PUBLISH_CONFLICT)
    assert raised.value.constraint_id == constraint_id


async def test_put_recipe_for_an_unpublished_frame_raises_snapshot_not_found() -> None:
    calls = Calls(problem(404, "snapshot_not_found"))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.put_recipe(spec_id=OPS, content_version=OCV, recipe=RECIPE)

    assert raised.value.code == "snapshot_not_found"


async def test_put_recipe_answer_without_created_is_invalid() -> None:
    calls = Calls(ok({"spec_id": OPS, "content_version": OCV}))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.put_recipe(spec_id=OPS, content_version=OCV, recipe=RECIPE)

    assert raised.value.code == RESPONSE_INVALID


# read_recipe


async def test_read_recipe_returns_the_stored_body() -> None:
    calls = Calls(ok(RECIPE))

    async with make_client(calls) as client:
        recipe = await client.read_recipe(spec_id=OPS, content_version=OCV)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("GET", f"/api/datasets/frames/{OPS}/{OCV}/recipe")
    assert request.headers[PRINCIPAL_HEADER] == DATASETS_PRINCIPAL
    assert recipe == RECIPE


async def test_read_recipe_is_none_on_recipe_not_found() -> None:
    calls = Calls(problem(404, RECIPE_NOT_FOUND))

    async with make_client(calls) as client:
        assert await client.read_recipe(spec_id=OPS, content_version=OCV) is None


async def test_read_recipe_raises_other_refusals() -> None:
    calls = Calls(problem(403, "publish_not_permitted"))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.read_recipe(spec_id=OPS, content_version=OCV)

    assert (raised.value.status, raised.value.code) == (403, "publish_not_permitted")


# find_replay


async def test_find_replay_returns_the_frame() -> None:
    key = "rpk_" + "1" * 32
    calls = Calls(ok({"spec_id": OPS, "content_version": OCV}))

    async with make_client(calls) as client:
        found = await client.find_replay(replay_key=key)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("GET", f"/api/datasets/recipes/replays/{key}")
    assert request.headers[PRINCIPAL_HEADER] == DATASETS_PRINCIPAL
    assert found == FramePair(OPS, OCV)


async def test_find_replay_is_none_on_recipe_not_found() -> None:
    calls = Calls(problem(404, RECIPE_NOT_FOUND))

    async with make_client(calls) as client:
        assert await client.find_replay(replay_key="rpk_" + "1" * 32) is None


# list_versions

_GROUPBY = {"by": ["recipient"], "op": "sum"}


async def test_list_versions_parses_newest_first_with_base_versions() -> None:
    calls = Calls(
        ok(
            {
                "schema_version": 1,
                "spec_id": SPEC,
                "versions": [
                    {
                        "content_version": "cv_5",
                        "base_content_version": "cv_1",
                        "rendering": "groupby",
                        "groupby": _GROUPBY,
                        "without_party_totals": True,
                    },
                    {
                        "content_version": "cv_4",
                        "base_content_version": "cv_1",
                        "rendering": "without_party_totals",
                        "groupby": None,
                        "without_party_totals": True,
                    },
                    {
                        "content_version": "cv_3",
                        "base_content_version": "cv_1",
                        "rendering": "not_recorded",
                        "groupby": None,
                    },
                    {
                        "content_version": "cv_2",
                        "base_content_version": "cv_1",
                        "rendering": "groupby",
                        "groupby": _GROUPBY,
                    },
                    {
                        "content_version": "cv_1",
                        "base_content_version": None,
                        "rendering": "plain",
                        "groupby": None,
                    },
                ],
            }
        )
    )

    async with make_client(calls) as client:
        versions = await client.list_versions(spec_id=SPEC)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("GET", f"/api/datasets/{SPEC}/versions")
    assert request.headers[PRINCIPAL_HEADER] == DATASETS_PRINCIPAL
    assert versions == (
        DatasetVersion("cv_5", "cv_1", "groupby", _GROUPBY, without_party_totals=True),
        DatasetVersion("cv_4", "cv_1", "without_party_totals", None, without_party_totals=True),
        DatasetVersion("cv_3", "cv_1", "not_recorded", None),
        DatasetVersion("cv_2", "cv_1", "groupby", _GROUPBY),
        DatasetVersion("cv_1", None, "plain", None),
    )


@pytest.mark.parametrize(
    "version",
    [
        {"content_version": "cv_1", "base_content_version": None},
        {"content_version": "cv_1", "base_content_version": None, "rendering": "summed"},
        {"content_version": "cv_2", "base_content_version": "cv_1", "rendering": "groupby"},
        {
            "content_version": "cv_1",
            "base_content_version": None,
            "rendering": "plain",
            "groupby": {"by": ["recipient"], "op": "sum"},
        },
        {
            "content_version": "cv_2",
            "base_content_version": "cv_1",
            "rendering": "without_party_totals",
            "groupby": {"by": ["recipient"], "op": "sum"},
            "without_party_totals": True,
        },
        {
            "content_version": "cv_2",
            "base_content_version": "cv_1",
            "rendering": "without_party_totals",
            "groupby": None,
            "without_party_totals": "true",
        },
    ],
    ids=[
        "no_rendering",
        "unknown_rendering",
        "groupby_without_one",
        "plain_with_groupby",
        "cut_with_groupby",
        "flag_not_a_boolean",
    ],
)
async def test_list_versions_refuses_a_version_outside_the_contract(
    version: dict[str, Any],
) -> None:
    calls = Calls(ok({"schema_version": 1, "spec_id": SPEC, "versions": [version]}))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as refused:
            await client.list_versions(spec_id=SPEC)

    assert refused.value.code == RESPONSE_INVALID


async def test_list_versions_of_an_unstored_dataset_is_empty() -> None:
    calls = Calls(ok({"schema_version": 1, "spec_id": SPEC, "versions": []}))

    async with make_client(calls) as client:
        assert await client.list_versions(spec_id=SPEC) == ()


# refresh_frame


async def test_refresh_frame_sends_the_pair_with_the_live_principal() -> None:
    calls = Calls(ok({"status": "current", "frame": FRAME.as_json()}))

    async with make_client(calls) as client:
        await client.refresh_frame(spec_id=SPEC, content_version=CV)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("POST", "/api/datasets/frames/refresh")
    assert request.headers[PRINCIPAL_HEADER] == LIVE_PRINCIPAL
    assert body_of(request) == {"spec_id": SPEC, "content_version": CV}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"status": "current", "frame": FRAME.as_json()}, Current(FRAME)),
        ({"status": "moved", "frame": {"spec_id": SPEC, "content_version": "cv_new"}}, None),
        ({"status": "pending", "operation_id": "op_1", "retry_after_s": 6}, Pending("op_1", 6)),
    ],
)
async def test_refresh_frame_parses_each_status(answer: dict[str, Any], expected: object) -> None:
    calls = Calls(ok(answer))

    async with make_client(calls) as client:
        result = await client.refresh_frame(spec_id=SPEC, content_version=CV)

    if expected is None:
        assert result == Moved(FramePair(SPEC, "cv_new"))
    else:
        assert result == expected


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "stale", "frame": FRAME.as_json()},
        {"status": "moved"},
        {"status": "pending", "operation_id": "op_1"},
        {"status": "pending", "operation_id": "op_1", "retry_after_s": True},
        ["current"],
    ],
)
async def test_refresh_frame_answer_outside_the_contract_is_invalid(answer: object) -> None:
    calls = Calls(ok(answer))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.refresh_frame(spec_id=SPEC, content_version=CV)

    assert raised.value.code == RESPONSE_INVALID


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (403, REFRESH_NOT_PERMITTED),
        (422, FRAME_NOT_REFRESHABLE),
        (422, RENDERING_NOT_RECORDED),
        (404, "snapshot_not_found"),
        (404, "spec_not_found"),
    ],
)
async def test_refresh_frame_raises_each_registered_refusal(status: int, code: str) -> None:
    calls = Calls(problem(status, code))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.refresh_frame(spec_id=SPEC, content_version=CV)

    assert (raised.value.status, raised.value.code) == (status, code)


# render


async def test_render_returns_the_answer_untouched() -> None:
    answer = {
        "component_id": "cmp_1",
        "revision": 3,
        "payload": {"provenance": {"refreshed_from": 3}},
        "fact_summary": {},
        "provenance": {"inputs": [FRAME.as_json()], "refreshed_from": 3},
    }
    calls = Calls(ok(answer))

    async with make_client(calls) as client:
        rendered = await client.render(component_id="cmp_1", revision=3, frame=FRAME)

    (request,) = calls.requests
    assert (request.method, request.url.path) == (
        "POST",
        "/api/presentation/components/cmp_1/revisions/3/render",
    )
    assert request.headers[PRINCIPAL_HEADER] == LIVE_PRINCIPAL
    assert body_of(request) == {"frame": FRAME.as_json()}
    assert rendered == answer


@pytest.mark.parametrize(
    ("status", "code"), [(409, RENDER_TARGET_MISMATCH), (403, REFRESH_NOT_PERMITTED)]
)
async def test_render_raises_each_registered_refusal(status: int, code: str) -> None:
    calls = Calls(problem(status, code))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.render(component_id="cmp_1", revision=3, frame=FRAME)

    assert (raised.value.status, raised.value.code) == (status, code)


# stream_render_data


async def test_stream_render_data_yields_the_bytes_and_forwardable_headers() -> None:
    calls = Calls(
        httpx2.Response(
            200,
            content=b"a,b\n1,2\n",
            headers={
                "content-type": "text/csv",
                "content-disposition": 'attachment; filename="x.csv"',
            },
        )
    )

    async with (
        make_client(calls) as client,
        client.stream_render_data(
            component_id="cmp_1", revision=3, frame=FRAME, format="csv", bundle=False
        ) as stream,
    ):
        data = b"".join([chunk async for chunk in stream.chunks])

    (request,) = calls.requests
    assert (request.method, request.url.path) == (
        "GET",
        "/api/presentation/components/cmp_1/revisions/3/render/data",
    )
    assert dict(request.url.params) == {
        "spec_id": SPEC,
        "content_version": CV,
        "format": "csv",
        "bundle": "false",
    }
    assert request.headers[PRINCIPAL_HEADER] == LIVE_PRINCIPAL
    assert data == b"a,b\n1,2\n"
    assert stream.content_type == "text/csv"
    assert stream.content_disposition == 'attachment; filename="x.csv"'
    assert stream.content_length == len(data)


async def test_stream_render_data_omits_unset_options() -> None:
    calls = Calls(httpx2.Response(200, content=b"x"))

    async with (
        make_client(calls) as client,
        client.stream_render_data(component_id="cmp_1", revision=3, frame=FRAME),
    ):
        pass

    assert set(calls.requests[0].url.params) == {"spec_id", "content_version"}


async def test_stream_render_data_refusal_raises_on_entry() -> None:
    calls = Calls(problem(409, RENDER_TARGET_MISMATCH))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            async with client.stream_render_data(component_id="cmp_1", revision=3, frame=FRAME):
                pytest.fail("a refused stream must not open")

    assert (raised.value.status, raised.value.code) == (409, RENDER_TARGET_MISMATCH)


async def test_stream_render_data_transport_failure_is_unreachable() -> None:
    calls = Calls(httpx2.ConnectError("refused"))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisTransportError):
            async with client.stream_render_data(component_id="cmp_1", revision=3, frame=FRAME):
                pytest.fail("an unreachable stream must not open")


# read_component


async def test_read_component_needs_no_credential_and_returns_the_frame_and_body() -> None:
    answer = {
        "component_id": "cmp_1",
        "revision": 2,
        "stored_intent": {"type": "stat_card", "frame_ref": FRAME.as_json()},
        "resolved_payload": {},
    }
    calls = Calls(ok(answer))

    async with make_client(calls, datasets_principal=None, live_refresh_principal=None) as client:
        read = await client.read_component(component_id="cmp_1", revision=2)

    (request,) = calls.requests
    assert (request.method, request.url.path) == ("GET", "/api/presentation/components/cmp_1")
    assert dict(request.url.params) == {"revision": "2"}
    assert PRINCIPAL_HEADER not in request.headers
    assert read.frame_ref == FRAME
    assert read.body == answer


async def test_read_component_without_a_revision_sends_no_query_and_no_frame_is_none() -> None:
    calls = Calls(ok({"component_id": "cmp_1", "stored_intent": {"type": "note_narration"}}))

    async with make_client(calls) as client:
        read = await client.read_component(component_id="cmp_1")

    assert not calls.requests[0].url.params
    assert read.frame_ref is None


async def test_read_component_not_found_raises() -> None:
    calls = Calls(problem(404, "not_found"))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.read_component(component_id="cmp_1", revision=9)

    assert raised.value.code == "not_found"


# principals


@pytest.mark.parametrize("principal", [None, "", "  "])
async def test_recipe_routes_refuse_an_unset_datasets_principal_before_any_request(
    principal: str | None,
) -> None:
    calls = Calls(ok({}))

    async with make_client(calls, datasets_principal=principal) as client:
        for call in (
            client.put_recipe(spec_id=OPS, content_version=OCV, recipe=RECIPE),
            client.read_recipe(spec_id=OPS, content_version=OCV),
            client.find_replay(replay_key="rpk_" + "1" * 32),
            client.list_versions(spec_id=SPEC),
        ):
            with pytest.raises(LiveApisNotConfigured, match="DATASETS_FIRST_PARTY_PRINCIPAL"):
                await call

    assert calls.requests == []


@pytest.mark.parametrize("principal", [None, "", "  "])
async def test_live_routes_refuse_an_unset_live_principal_before_any_request(
    principal: str | None,
) -> None:
    calls = Calls(ok({}))
    auth = CountingAuth()

    async with make_client(calls, live_refresh_principal=principal, auth=auth) as client:
        for call in (
            client.refresh_frame(spec_id=SPEC, content_version=CV),
            client.render(component_id="cmp_1", revision=1, frame=FRAME),
        ):
            with pytest.raises(LiveApisNotConfigured, match="LIVE_REFRESH_PRINCIPAL"):
                await call
        with pytest.raises(LiveApisNotConfigured, match="LIVE_REFRESH_PRINCIPAL"):
            async with client.stream_render_data(component_id="cmp_1", revision=1, frame=FRAME):
                pass

    assert calls.requests == []
    assert auth.calls == 0


async def test_the_two_principals_are_not_interchangeable() -> None:
    calls = Calls(ok({"status": "current", "frame": FRAME.as_json()}))

    async with make_client(calls, live_refresh_principal=None) as client:
        with pytest.raises(LiveApisNotConfigured):
            await client.refresh_frame(spec_id=SPEC, content_version=CV)

    assert calls.requests == []


async def test_no_principal_value_reaches_an_error_message_or_repr() -> None:
    answers: list[tuple[httpx2.Response | Exception, Callable[[LiveApisClient], Any]]] = [
        (
            problem(403, REFRESH_NOT_PERMITTED),
            lambda c: c.refresh_frame(spec_id=SPEC, content_version=CV),
        ),
        (
            problem(403, "publish_not_permitted"),
            lambda c: c.read_recipe(spec_id=OPS, content_version=OCV),
        ),
        (
            httpx2.ConnectError("refused"),
            lambda c: c.render(component_id="cmp_1", revision=1, frame=FRAME),
        ),
        (httpx2.Response(502, content=b"bad gateway"), lambda c: c.list_versions(spec_id=SPEC)),
    ]
    for answer, call in answers:
        async with make_client(Calls(answer)) as client:
            with pytest.raises(LiveApisError) as raised:
                await call(client)
            texts = [
                str(raised.value),
                repr(raised.value),
                repr(client),
                *map(str, raised.value.args),
            ]
            assert all(
                DATASETS_PRINCIPAL not in text and LIVE_PRINCIPAL not in text for text in texts
            )


# auth hook


async def test_auth_headers_ride_every_route_once_per_call_beside_the_principal() -> None:
    calls = Calls(
        problem(503, "service_unavailable", retryable=True),
        ok({"status": "current", "frame": FRAME.as_json()}),
        ok({"stored_intent": None}),
        httpx2.Response(200, content=b"x"),
    )
    auth = CountingAuth()

    async with make_client(calls, auth=auth, max_attempts=3) as client:
        await client.refresh_frame(spec_id=SPEC, content_version=CV)
        await client.read_component(component_id="cmp_1")
        async with client.stream_render_data(component_id="cmp_1", revision=3, frame=FRAME):
            pass

    sent = [
        (
            r.url.path.rsplit("/", 1)[-1],
            r.headers.get(TOKEN_HEADER),
            r.headers.get(PRINCIPAL_HEADER),
        )
        for r in calls.requests
    ]
    assert sent == [
        ("refresh", "Bearer token-1", LIVE_PRINCIPAL),
        ("refresh", "Bearer token-1", LIVE_PRINCIPAL),
        ("cmp_1", "Bearer token-2", None),
        ("data", "Bearer token-3", LIVE_PRINCIPAL),
    ]


async def _failing_auth() -> Mapping[str, str]:
    raise ApisAuthError("metadata server answered 500")


async def _refresh(client: LiveApisClient) -> None:
    await client.refresh_frame(spec_id=SPEC, content_version=CV)


async def _stream(client: LiveApisClient) -> None:
    async with client.stream_render_data(component_id="cmp_1", revision=3, frame=FRAME):
        pass


@pytest.mark.parametrize("call", [_refresh, _stream], ids=["request", "stream"])
async def test_a_failed_mint_is_unreachable_and_sends_nothing(
    call: Callable[[LiveApisClient], Awaitable[None]],
) -> None:
    calls = Calls(ok({"status": "current", "frame": FRAME.as_json()}))

    async with make_client(calls, auth=_failing_auth, max_attempts=3) as client:
        with pytest.raises(LiveApisTransportError) as raised:
            await call(client)

    assert (raised.value.status, raised.value.retryable) == (None, True)
    assert isinstance(raised.value.__cause__, ApisAuthError)
    assert calls.requests == []


# failures


async def test_transport_failure_raises_unreachable_after_the_attempts() -> None:
    calls = Calls(httpx2.ConnectError("refused"))

    async with make_client(calls, max_attempts=3) as client:
        with pytest.raises(LiveApisTransportError) as raised:
            await client.refresh_frame(spec_id=SPEC, content_version=CV)

    assert (raised.value.status, raised.value.code, raised.value.retryable) == (None, None, True)
    assert len(calls.requests) == 3


async def test_a_retryable_5xx_is_retried_until_it_succeeds() -> None:
    calls = Calls(
        problem(503, "service_unavailable", retryable=True),
        ok({"status": "current", "frame": FRAME.as_json()}),
    )

    async with make_client(calls, max_attempts=3) as client:
        result = await client.refresh_frame(spec_id=SPEC, content_version=CV)

    assert result == Current(FRAME)
    assert len(calls.requests) == 2


@pytest.mark.parametrize(
    "answer",
    [problem(409, RENDER_TARGET_MISMATCH), problem(500, "internal_error")],
    ids=["refusal-4xx", "non-retryable-5xx"],
)
async def test_a_refusal_that_repeats_is_not_retried(answer: httpx2.Response) -> None:
    calls = Calls(answer)

    async with make_client(calls, max_attempts=3) as client:
        with pytest.raises(LiveApisError):
            await client.render(component_id="cmp_1", revision=1, frame=FRAME)

    assert len(calls.requests) == 1


async def test_an_unparsed_5xx_is_retryable_and_carries_no_code() -> None:
    calls = Calls(httpx2.Response(502, content=b"bad gateway"))

    async with make_client(calls) as client:
        with pytest.raises(LiveApisError) as raised:
            await client.list_versions(spec_id=SPEC)

    assert (raised.value.status, raised.value.code, raised.value.retryable) == (502, None, True)


async def test_an_id_cannot_alter_the_route() -> None:
    calls = Calls(ok({"schema_version": 1, "spec_id": SPEC, "versions": []}))

    async with make_client(calls) as client:
        await client.list_versions(spec_id="spec_x/../ops_y")

    assert calls.requests[0].url.raw_path == b"/api/datasets/spec_x%2F..%2Fops_y/versions"
