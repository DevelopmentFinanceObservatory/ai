"""`common.live_apis` against requests and answers recorded from apis.

`fixtures/dfo_apis_live_refresh_wire_pairs.json` is apis' own recording of
its live-refresh routes (`tests/live_refresh/wire_pairs.py` there). Each test
sends the client's call for one recorded pair into a transport that answers the
recorded response, and compares the request the client built with the recorded
one field by field: method, path, query, JSON body, and headers. A route or
answer that changes in apis fails here once the fixture is copied again.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from common.live_apis import (
    PRINCIPAL_HEADER,
    ComponentRead,
    Current,
    DatasetVersion,
    FramePair,
    LiveApisClient,
    LiveApisError,
    Moved,
    Pending,
)

DATASETS_PRINCIPAL = "datasets-principal-7f3a"
LIVE_PRINCIPAL = "live-principal-91bc"
_PRINCIPALS = {
    "<DATASETS_FIRST_PARTY_PRINCIPAL>": DATASETS_PRINCIPAL,
    "<LIVE_REFRESH_PRINCIPAL>": LIVE_PRINCIPAL,
}
PAIRS: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "dfo_apis_live_refresh_wire_pairs.json").read_text()
)["pairs"]

DATASET = "spec_176b1b132147fff98b37810b759f6f90"
FIRST = FramePair(DATASET, "cv_" + "0" * 31 + "1")
SECOND = FramePair(DATASET, "cv_" + "0" * 31 + "2")
DERIVED_FIRST = FramePair("ops_" + "0" * 31 + "1", "ocv_" + "0" * 31 + "1")
DERIVED_SECOND = FramePair("ops_" + "0" * 31 + "2", "ocv_" + "0" * 31 + "2")
CARD = "cmp_" + "0" * 23 + "1"
DERIVED_CARD = "cmp_" + "0" * 23 + "2"
REPLAY_KEY = "rpk_" + "0" * 31 + "2"

# The one recorded pair this client does not send: presentation-mcp posts the
# publish (`presentation_mcp.client.PlatformAssembler`).
NOT_THIS_CLIENT = {"publish.with_recipe"}
# Recorded with no principal header, which this client cannot leave out. Every
# other field of the request is compared.
RECORDED_WITHOUT_PRINCIPAL = {"frame_refresh.refresh_not_permitted"}

Call = Callable[[LiveApisClient], Awaitable[object]]


async def _render_data(client: LiveApisClient, **options: Any) -> dict[str, Any]:
    async with client.stream_render_data(
        component_id=CARD, revision=1, frame=SECOND, **options
    ) as stream:
        return {
            "content-type": stream.content_type,
            "content-disposition": stream.content_disposition,
            "body": b"".join([chunk async for chunk in stream.chunks]),
        }


def _refresh(frame: FramePair) -> Call:
    return lambda client: client.refresh_frame(
        spec_id=frame.spec_id, content_version=frame.content_version
    )


def _render(component_id: str, frame: FramePair) -> Call:
    return lambda client: client.render(component_id=component_id, revision=1, frame=frame)


def _read_recipe(frame: FramePair) -> Call:
    return lambda client: client.read_recipe(
        spec_id=frame.spec_id, content_version=frame.content_version
    )


CALLS: dict[str, Call] = {
    "component_read.at_revision": lambda c: c.read_component(component_id=CARD, revision=1),
    "recipe_read.found": _read_recipe(DERIVED_FIRST),
    "recipe_read.recipe_not_found": _read_recipe(DERIVED_SECOND),
    "replay_lookup.found": lambda c: c.find_replay(replay_key=REPLAY_KEY),
    "replay_lookup.recipe_not_found": lambda c: c.find_replay(replay_key=REPLAY_KEY),
    "versions_list": lambda c: c.list_versions(spec_id=DATASET),
    "frame_refresh.current": _refresh(FIRST),
    "frame_refresh.moved": _refresh(FIRST),
    "frame_refresh.pending": _refresh(FIRST),
    "frame_refresh.frame_not_refreshable": _refresh(DERIVED_FIRST),
    "frame_refresh.rendering_not_recorded": _refresh(SECOND),
    "frame_refresh.refresh_not_permitted": _refresh(FIRST),
    "render.dataset_card": _render(CARD, SECOND),
    "render.derived_card_with_anchor_warning": _render(DERIVED_CARD, DERIVED_SECOND),
    "render.render_target_mismatch": _render(CARD, DERIVED_FIRST),
    "render_data.csv": lambda c: _render_data(c, format="csv", bundle=False),
    "render_data.bundle": _render_data,
}


def _component(response: dict[str, Any]) -> object:
    body = response["body"]
    return ComponentRead(frame_ref=FramePair(**body["stored_intent"]["frame_ref"]), body=body)


def _versions(response: dict[str, Any]) -> object:
    return tuple(
        DatasetVersion(
            version["content_version"],
            version["base_content_version"],
            version["rendering"],
            version["groupby"],
        )
        for version in response["body"]["versions"]
    )


def _refreshed(response: dict[str, Any]) -> object:
    body = response["body"]
    if body["status"] == "pending":
        return Pending(operation_id=body["operation_id"], retry_after_s=body["retry_after_s"])
    return {"current": Current, "moved": Moved}[body["status"]](FramePair(**body["frame"]))


def _export(response: dict[str, Any]) -> object:
    return {
        "content-type": response["headers"]["content-type"],
        "content-disposition": response["headers"]["content-disposition"],
        "body": _recorded_content(response),
    }


# What the client returns for a recorded 2xx answer of each route.
RESULTS: dict[str, Callable[[dict[str, Any]], object]] = {
    "component_read": _component,
    "recipe_read": lambda response: response["body"],
    "replay_lookup": lambda response: FramePair(**response["body"]),
    "versions_list": _versions,
    "frame_refresh": _refreshed,
    "render": lambda response: response["body"],
    "render_data": _export,
}


def _expected(name: str, response: dict[str, Any]) -> object:
    """What the client returns for the recorded `response`, or the
    `(status, code, retryable)` of the error it raises."""
    if name.endswith(".recipe_not_found"):
        return None
    if response["status"] >= 400:
        return response["status"], response["body"]["code"], response["body"]["retryable"]
    return RESULTS[name.partition(".")[0]](response)


def _recorded_content(response: dict[str, Any]) -> bytes:
    if "body_starts_hex" in response:
        return bytes.fromhex(response["body_starts_hex"])
    if "body_starts" in response:
        return response["body_starts"].encode()
    return json.dumps(response["body"]).encode()


def _sent(request: httpx2.Request) -> dict[str, Any]:
    """The fields of `request` a recorded request states."""
    return {
        "method": request.method,
        "path": request.url.path,
        "query": sorted(request.url.params.multi_items()),
        "body": json.loads(request.content) if request.content else None,
        "principal": request.headers.get(PRINCIPAL_HEADER),
        "content-type": request.headers.get("content-type"),
    }


def _recorded(request: dict[str, Any]) -> dict[str, Any]:
    url = httpx2.URL(request["path"])
    headers = request["headers"]
    return {
        "method": request["method"],
        "path": url.path,
        "query": sorted(url.params.multi_items()),
        "body": request.get("body"),
        "principal": _PRINCIPALS.get(headers.get(PRINCIPAL_HEADER, "")),
        "content-type": headers.get("content-type"),
    }


def test_every_recorded_pair_is_compared() -> None:
    assert set(PAIRS) - NOT_THIS_CLIENT == set(CALLS)


@pytest.mark.parametrize("name", sorted(CALLS))
async def test_the_client_sends_the_recorded_request_and_reads_the_recorded_answer(
    name: str,
) -> None:
    pair = PAIRS[name]
    requests: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            pair["response"]["status"],
            headers=pair["response"]["headers"],
            content=_recorded_content(pair["response"]),
        )

    async with LiveApisClient(
        base_url="http://apis.test",
        datasets_principal=DATASETS_PRINCIPAL,
        live_refresh_principal=LIVE_PRINCIPAL,
        transport=httpx2.MockTransport(answer),
    ) as client:
        try:
            result = await CALLS[name](client)
        except LiveApisError as exc:
            result = (exc.status, exc.code, exc.retryable)

    (request,) = requests
    sent, recorded = _sent(request), _recorded(pair["request"])
    if name in RECORDED_WITHOUT_PRINCIPAL:
        assert recorded["principal"] is None
        recorded["principal"] = sent["principal"]
    assert sent == recorded
    assert result == _expected(name, pair["response"])
