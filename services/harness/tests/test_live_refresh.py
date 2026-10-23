"""`POST /api/studio/refresh`: the credential, each answer and refusal, and the
route's limits, with apis as `live_refresh_support.RecordedApis` and the
replayer as a double. `test_live_refresh_integration.py` runs the real replayer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from typing import Any

import httpx2
import pytest
from analysis_mcp.execution_identity import current_execution_identity
from common.apis_auth import ApisAuthError, IdTokenProvider
from common.live_apis import LiveApisClient
from common.materialisation import FrameRef, MintRecord
from common.recipe import replay_key, root_identity
from common.row_budget import RowBudget, row_budget
from compiler.surface import ToolSurfaceUnavailableError
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    InMemoryDefinitionStore,
    StoredDefinition,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.frame_materialisation import KIND_COLUMNS, frame_ref_for
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from harness import server as server_module
from harness.auth import principal_for
from harness.live_refresh import (
    HARD_STOP_GRACE_S,
    REFRESH_CODES,
    LiveRefresh,
    build_live_refresh,
    create_live_refresh_router,
)
from harness.recipe_replay import Pending, Refused, Replayed
from harness.server import create_app
from harness.settings import (
    LIVE_REFRESH_CALLER_TIMEOUT_S,
    LIVE_REFRESH_DEADLINE_MAX_S,
    Settings,
)
from presentation_mcp.client import PlatformAssembler

from .live_refresh_support import (
    CARD,
    DATASET_CARD,
    LIVE_PRINCIPAL,
    RECORDED_RENDER,
    RecordedApis,
    problem,
)
from .replay_support import FIRST, OPTIONS, PRINCIPAL, SECOND, SPEC, Pair

STUDIO_TOKEN = "studio-token-55d0"
OTHER_SPEC = "spec_" + "b" * 32
STORED: Pair = ("ops_" + "a" * 32, "ocv_" + "a" * 32)
REPLAYED: Pair = ("ops_" + "c" * 32, "ocv_" + "c" * 32)
NORMAL_FORM = "rnf_" + "1" * 32
SETTINGS = Settings(live_refresh_principal=LIVE_PRINCIPAL)
_STATUS = {row.code: row.status for row in REFRESH_CODES}


def recipe_of(frame: Pair, *datasets: str) -> dict[str, Any]:
    """A stored recipe of `frame` over `datasets` (default the one dataset), each
    recorded at its first version."""
    roots = [
        {
            "schema_version": 1,
            "frame": {"spec_id": spec_id, "content_version": FIRST},
            "dataset": {"spec_id": spec_id, "content_version": FIRST},
            "options": OPTIONS,
            "scope": {},
        }
        for spec_id in datasets or (SPEC,)
    ]
    return {
        "schema_version": 1,
        "normal_form_hash": NORMAL_FORM,
        "replay_key": f"rpk_of_{frame[0]}",
        "execution_identity": "engine|grammar|reduction",
        "tree": {"frame": {"spec_id": frame[0], "content_version": frame[1]}},
        "roots": roots,
        "anchor": None,
    }


class Replayer:
    """`replay_recipe`'s double: counts its entries and answers each from
    `outcomes`, the last one repeating. A `Replayed` stores its own recipe on `apis`
    under its frame, as the real replayer's publish does; one that carries none stores
    `recipe_of(REPLAYED)`, which has the stored recipe's normal form. `gate`, when set,
    holds every entry until it is released."""

    def __init__(self, apis: RecordedApis, *outcomes: Replayed | Pending | Refused | Exception):
        self.apis = apis
        self.outcomes = list(outcomes) or [
            Replayed(
                frame=FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]),
                recipe={},
                newly_published=True,
            )
        ]
        self.entered = 0
        self.gate: asyncio.Event | None = None

    async def __call__(self, recipe: Mapping[str, Any], **_: Any) -> Replayed | Pending | Refused:
        self.entered += 1
        if self.gate is not None:
            await self.gate.wait()
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Replayed):
            self.apis.recipes[(outcome.frame.spec_id, outcome.frame.content_version)] = (
                outcome.recipe or recipe_of(REPLAYED)
            )
        return outcome


@pytest.fixture
def apis() -> RecordedApis:
    """apis holding the card on a derived frame whose one dataset has moved."""
    fake = RecordedApis()
    fake.store_card(FrameRef(spec_id=STORED[0], content_version=STORED[1]))
    fake.recipes[STORED] = recipe_of(STORED)
    fake.heads[SPEC] = SECOND
    return fake


def service_over(fake: RecordedApis, replayer: Replayer, **overrides: Any) -> LiveRefresh:
    async def no_tool(**_: Any) -> Any:
        raise AssertionError("the replayer double calls no tool")

    async def no_publish(**_: Any) -> bool:
        raise AssertionError("the replayer double publishes nothing")

    options: dict[str, Any] = {
        "apis": fake.client(),
        "call_tool": no_tool,
        "publish": no_publish,
        "row_budget": RowBudget(max_rows=1_000),
        "deadline_s": 30.0,
        "requests_per_s": 100.0,
        "burst": 100,
        "max_in_flight": 100,
        "replay": replayer,
    }
    return LiveRefresh(**(options | overrides))


def client_of(service: LiveRefresh, *, settings: Settings = SETTINGS) -> httpx2.AsyncClient:
    app = FastAPI()
    app.state.live_refresh = service
    app.include_router(create_live_refresh_router(settings=settings))
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://harness")


async def refresh(
    client: httpx2.AsyncClient,
    *,
    component_id: str = CARD,
    token: str | None = LIVE_PRINCIPAL,
    body: Any = None,
) -> httpx2.Response:
    return await client.post(
        "/api/studio/refresh",
        json=body if body is not None else {"component_id": component_id, "revision": 1},
        headers={"Authorization": f"Bearer {token}"} if token is not None else {},
    )


def assert_refused(response: httpx2.Response, code: str) -> None:
    """`response` is the refusal `code` under the status `REFRESH_CODES` lists for
    it, or 422 for a code passed through from another service."""
    assert (response.status_code, response.json()["code"]) == (_STATUS.get(code, 422), code)
    assert isinstance(response.json()["detail"], str)


# --- the credential -------------------------------------------------------------


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer ",
        "Bearer wrong",
        f"Bearer {STUDIO_TOKEN}",
        f"Bearer {PRINCIPAL}",
        f"Basic {LIVE_PRINCIPAL}",
        LIVE_PRINCIPAL,
        f"Bearer {LIVE_PRINCIPAL}x",
        "Bearer principál",
    ],
)
async def test_any_other_credential_is_refused_before_dfo_apis_is_asked(
    apis: RecordedApis, authorization: str | None
) -> None:
    replayer = Replayer(apis)
    settings = Settings(live_refresh_principal=LIVE_PRINCIPAL, studio_api_token=STUDIO_TOKEN)
    async with client_of(service_over(apis, replayer), settings=settings) as client:
        response = await client.post(
            "/api/studio/refresh",
            json={"component_id": CARD, "revision": 1},
            headers={"Authorization": authorization.encode()} if authorization is not None else {},
        )

    assert_refused(response, "unauthorized")
    assert response.headers["www-authenticate"] == "Bearer"
    assert (apis.requests, replayer.entered) == ([], 0)


@pytest.mark.parametrize("unset", [None, "", "   "])
@pytest.mark.parametrize("token", ["", "anything", "None"])
async def test_an_unset_principal_refuses_every_caller(
    apis: RecordedApis, unset: str | None, token: str
) -> None:
    settings = Settings(live_refresh_principal=unset)
    async with client_of(service_over(apis, Replayer(apis)), settings=settings) as client:
        response = await refresh(client, token=token)

    assert_refused(response, "unauthorized")
    assert apis.requests == []


def test_a_server_without_the_principal_says_at_startup_that_the_route_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        server_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )

    create_app(settings=Settings(studio_api_token=STUDIO_TOKEN))
    closed = [w for w in warnings if "LIVE REFRESH IS CLOSED" in w]
    create_app(settings=Settings(studio_api_token=STUDIO_TOKEN, live_refresh_principal="set"))

    assert len(closed) == 1
    assert len([w for w in warnings if "LIVE REFRESH IS CLOSED" in w]) == 1


def test_the_studio_token_does_not_open_the_served_route() -> None:
    app = create_app(
        settings=Settings(studio_api_token=STUDIO_TOKEN, live_refresh_principal=LIVE_PRINCIPAL)
    )
    response = TestClient(app).post(
        "/api/studio/refresh",
        json={"component_id": CARD, "revision": 1},
        headers={"Authorization": f"Bearer {STUDIO_TOKEN}"},
    )

    assert (response.status_code, response.json()["code"]) == (401, "unauthorized")


@pytest.mark.parametrize("studio_api_token", [None, STUDIO_TOKEN])
def test_the_live_principal_opens_no_other_studio_route(studio_api_token: str | None) -> None:
    settings = Settings(studio_api_token=studio_api_token, live_refresh_principal=LIVE_PRINCIPAL)

    with pytest.raises(HTTPException) as refused:
        principal_for(f"Bearer {LIVE_PRINCIPAL}", settings=settings)
    assert refused.value.status_code == 401

    response = TestClient(create_app(settings=settings)).post(
        "/api/studio/turn",
        json={
            "kind": "turn.request",
            "schema_version": 1,
            "conversation_id": "c_live_refresh",
            "message": "kenya oda 2023",
        },
        headers={"Authorization": f"Bearer {LIVE_PRINCIPAL}"},
    )
    assert response.status_code == 401


def test_one_value_for_both_studio_credentials_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="LIVE_REFRESH_PRINCIPAL"):
        Settings(studio_api_token="same", live_refresh_principal="same")


def test_a_deadline_too_short_to_mint_a_root_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="live_refresh_deadline_s"):
        Settings(live_refresh_deadline_s=29.0)


def test_a_deadline_whose_hard_stop_outlasts_the_websites_call_fails_at_startup() -> None:
    longest = Settings(live_refresh_deadline_s=LIVE_REFRESH_DEADLINE_MAX_S)

    assert longest.live_refresh_deadline_s + HARD_STOP_GRACE_S == LIVE_REFRESH_CALLER_TIMEOUT_S
    with pytest.raises(ValueError, match="live_refresh_deadline_s"):
        Settings(live_refresh_deadline_s=LIVE_REFRESH_DEADLINE_MAX_S + 0.5)


async def test_neither_principal_reaches_a_log_line(
    apis: RecordedApis, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    async with client_of(service_over(apis, Replayer(apis))) as client:
        assert (await refresh(client)).json()["status"] == "redrawn"
        apis.render_answer = problem("render.render_target_mismatch", status=403, code="x")
        await refresh(client)
        await refresh(client, token="wrong")

    assert "studio refresh: component=" in caplog.text
    assert LIVE_PRINCIPAL not in caplog.text
    assert PRINCIPAL not in caplog.text


# --- the request and the card ---------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"component_id": CARD},
        {"component_id": CARD, "revision": 0},
        {"component_id": CARD, "revision": "1"},
        {"component_id": CARD, "revision": True},
        {"component_id": "../recipes", "revision": 1},
        {"component_id": CARD, "revision": 1, "frame": {}},
    ],
)
async def test_a_malformed_body_is_refused(apis: RecordedApis, body: Any) -> None:
    async with client_of(service_over(apis, Replayer(apis))) as client:
        response = await client.post(
            "/api/studio/refresh", json=body, headers={"Authorization": f"Bearer {LIVE_PRINCIPAL}"}
        )

    assert_refused(response, "request_invalid")
    assert apis.requests == []


def unknown_card(apis: RecordedApis) -> str:
    return "cmp_unknown"


def dataset_card(apis: RecordedApis) -> str:
    return DATASET_CARD


def card_on_no_known_frame(apis: RecordedApis) -> str:
    apis.cards[CARD] = ("note_1", "v1")
    return CARD


def card_without_a_recipe(apis: RecordedApis) -> str:
    del apis.recipes[STORED]
    return CARD


def card_whose_recipe_has_no_mint_record(apis: RecordedApis) -> str:
    apis.recipes[STORED]["roots"] = [{"spec_id": SPEC}]
    return CARD


def card_whose_recipe_has_no_normal_form(apis: RecordedApis) -> str:
    del apis.recipes[STORED]["normal_form_hash"]
    return CARD


@pytest.mark.parametrize(
    ("arrange", "code"),
    [
        (unknown_card, "component_not_found"),
        (dataset_card, "dataset_frame"),
        (card_on_no_known_frame, "frame_unsupported"),
        (card_without_a_recipe, "recipe_missing"),
        (card_whose_recipe_has_no_mint_record, "recipe_invalid"),
        (card_whose_recipe_has_no_normal_form, "recipe_invalid"),
    ],
)
async def test_a_card_that_cannot_be_replayed_is_refused_before_any_root_is_asked_about(
    apis: RecordedApis, arrange: Any, code: str
) -> None:
    replayer = Replayer(apis)
    component_id = arrange(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client, component_id=component_id)

    assert_refused(response, code)
    assert response.headers["cache-control"] == "no-store"
    assert apis.calls("/frames/refresh") == []
    assert replayer.entered == 0


# --- the roots ------------------------------------------------------------------


async def test_a_card_whose_roots_are_all_current_stands_and_nothing_else_is_asked(
    apis: RecordedApis, flow_definition: SuspendingDefinitionStore
) -> None:
    apis.recipes[STORED] = recipe_of(STORED, SPEC, OTHER_SPEC)
    apis.heads = {SPEC: FIRST, OTHER_SPEC: FIRST}
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert (response.status_code, response.json()) == (200, {"status": "current"})
    assert len(apis.calls("/frames/refresh")) == 2
    assert flow_definition.reads == []
    assert apis.calls("/versions") == []
    assert apis.calls("/recipes/replays/") == []
    assert apis.calls("/render") == []
    assert replayer.entered == 0


async def test_a_rebuilding_root_answers_pending_with_the_longest_wait(apis: RecordedApis) -> None:
    third = "spec_" + "d" * 32
    recipe = recipe_of(STORED, SPEC, OTHER_SPEC, third)
    recipe["roots"][0] |= {
        "options": {**OPTIONS, "columns": list(KIND_COLUMNS)},
        "enrichment_stamp": "stamp-old",
    }
    apis.recipes[STORED] = recipe
    apis.heads = {SPEC: SECOND}
    apis.rebuilding = {OTHER_SPEC: 1.5, third: 4.0}
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert (response.status_code, response.json()) == (
        200,
        {"status": "pending", "retry_after_s": 4.0},
    )
    assert apis.calls("/versions") == []
    assert apis.calls("/recipes/replays/") == []
    assert replayer.entered == 0


async def test_a_frame_already_replayed_from_these_root_versions_is_rendered_without_a_replay(
    apis: RecordedApis, unavailable_definition: SuspendingDefinitionStore
) -> None:
    # The root records neither columns nor a stamp, and its version lists no stamp, so
    # its pair names it whatever its definition says, and the store is not read.
    apis.versions[SPEC] = listed((SECOND, None), (FIRST, None))
    key = replay_key(
        normal_form_hash=NORMAL_FORM,
        root_datasets=[{"spec_id": SPEC, "content_version": SECOND}],
        execution_identities=[current_execution_identity().canonical_string()],
    )
    apis.recipes[REPLAYED] = recipe_of(REPLAYED) | {"replay_key": key}
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert response.json() == {"status": "redrawn", "render": RECORDED_RENDER}
    assert [request.url.path for request in apis.calls("/recipes/replays/")] == [
        f"/api/datasets/recipes/replays/{key}"
    ]
    assert apis.rendered_on() == [REPLAYED]
    assert replayer.entered == 0
    assert unavailable_definition.reads == []


@pytest.mark.parametrize(
    ("versions_answer", "outcome"),
    [
        (None, "current"),
        (
            httpx2.Response(
                503,
                json={"code": "internal_error", "retryable": False},
                headers={"content-type": "application/problem+json"},
            ),
            None,
        ),
    ],
    ids=["listed", "versions-unavailable"],
)
async def test_kind_roots_on_one_dataset_read_its_versions_once(
    apis: RecordedApis, versions_answer: httpx2.Response | None, outcome: str | None
) -> None:
    """Two stamped roots on one dataset whose pairs stand. The listed stamp says
    whether their kinds still stand, so a refresh that cannot read it is refused."""
    recipe = recipe_of(STORED, SPEC, SPEC)
    for root in recipe["roots"]:
        root.update(
            options={**OPTIONS, "columns": list(KIND_COLUMNS)}, enrichment_stamp="stamp-old"
        )
    apis.recipes[STORED] = recipe
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((FIRST, "stamp-old"))
    apis.versions_answer = versions_answer
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    if outcome is None:
        assert_refused(response, "upstream_unavailable")
    else:
        assert response.json() == {"status": outcome}
    assert len(apis.calls("/versions")) == 1
    assert replayer.entered == 0


@pytest.mark.parametrize(
    ("listed_stamp", "status", "replayed"),
    [("stamp-old", "current", False), ("stamp-new", "redrawn", True)],
    ids=["same-stamp", "stamp-moved"],
)
async def test_a_kind_root_on_a_standing_version_is_current_only_under_its_own_stamp(
    apis: RecordedApis, listed_stamp: str, status: str, replayed: bool
) -> None:
    kinds = {**OPTIONS, "columns": list(KIND_COLUMNS)}
    recipe = recipe_of(STORED)
    recipe["roots"][0] |= {
        "frame": frame_ref_for(
            spec_id=SPEC, content_version=FIRST, columns=KIND_COLUMNS, enrichment_stamp="stamp-old"
        ).model_dump(),
        "options": kinds,
        "enrichment_stamp": "stamp-old",
    }
    apis.recipes[STORED] = recipe
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((FIRST, listed_stamp))
    apis.enrichment_stamp = listed_stamp
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    key = replay_key(
        normal_form_hash=NORMAL_FORM,
        root_datasets=[
            root_identity(
                {"spec_id": SPEC, "content_version": FIRST}, enrichment_stamp=listed_stamp
            )
        ],
        execution_identities=[current_execution_identity().canonical_string()],
    )
    looked_up = [f"/api/datasets/recipes/replays/{key}"] if replayed else []

    assert response.json()["status"] == status
    assert replayer.entered == int(replayed)
    assert [request.url.path for request in apis.calls("/recipes/replays/")] == looked_up


def kind_recipe(
    frame: Pair, *, content_version: str, stamp: str | None, keyed: bool = False
) -> dict[str, Any]:
    """A recipe of `frame` whose root records the kind columns and `stamp`, the stamp of
    the kind file its mint read. `keyed` gives it the replay key a replay's publish
    names it under (`common.recipe.build_recipe`), for a recipe a replay publishes."""
    recipe = recipe_of(frame)
    recipe["roots"][0] |= {
        "frame": frame_ref_for(
            spec_id=SPEC,
            content_version=content_version,
            columns=KIND_COLUMNS,
            enrichment_stamp=stamp,
        ).model_dump(),
        "dataset": {"spec_id": SPEC, "content_version": content_version},
        "options": {**OPTIONS, "columns": list(KIND_COLUMNS)},
        **({} if stamp is None else {"enrichment_stamp": stamp}),
    }
    if keyed:
        recipe["replay_key"] = replay_key(
            normal_form_hash=NORMAL_FORM,
            root_datasets=[
                root_identity(
                    {"spec_id": SPEC, "content_version": content_version}, enrichment_stamp=stamp
                )
            ],
            execution_identities=[current_execution_identity().canonical_string()],
        )
    return recipe


def replaying(apis: RecordedApis, recipe: dict[str, Any]) -> Replayer:
    """A replayer whose replay publishes `recipe` on `REPLAYED`."""
    return Replayer(
        apis,
        Replayed(
            frame=FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]),
            recipe=recipe,
            newly_published=True,
        ),
    )


def listed(*versions: tuple[str, str | None]) -> list[dict[str, Any]]:
    """Versions answer entries, newest first, each with the stamp a kind file of it
    carries, or none for a frame with no party role."""
    return [
        {"content_version": version, "rendering": "plain"}
        | ({} if stamp is None else {"enrichment_stamp": stamp})
        for version, stamp in versions
    ]


async def test_a_kind_root_of_a_frame_with_no_party_role_is_replayed_once_per_dataset_move(
    apis: RecordedApis,
) -> None:
    # apis serves the root's kind file unstamped, and its versions answer lists the
    # code index's stamp at the top and none on any version.
    apis.recipes[STORED] = kind_recipe(STORED, content_version=FIRST, stamp=None)
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((SECOND, None), (FIRST, None))
    apis.enrichment_stamp = "stamp-new"
    replayer = replaying(
        apis, kind_recipe(REPLAYED, content_version=SECOND, stamp=None, keyed=True)
    )
    async with client_of(service_over(apis, replayer)) as client:
        standing = await refresh(client)
        apis.heads[SPEC] = SECOND
        after_move = [(await refresh(client)).json()["status"] for _ in range(3)]

    assert standing.json() == {"status": "current"}
    assert after_move == 3 * ["redrawn"]
    assert replayer.entered == 1


async def test_a_stamped_root_whose_dataset_moves_is_replayed_once(apis: RecordedApis) -> None:
    apis.recipes[STORED] = kind_recipe(STORED, content_version=FIRST, stamp="stamp-new")
    apis.heads[SPEC] = SECOND
    apis.versions[SPEC] = listed((SECOND, "stamp-new"), (FIRST, "stamp-new"))
    apis.enrichment_stamp = "stamp-new"
    published = kind_recipe(REPLAYED, content_version=SECOND, stamp="stamp-new", keyed=True)
    replayer = replaying(apis, published)
    async with client_of(service_over(apis, replayer)) as client:
        statuses = [(await refresh(client)).json()["status"] for _ in range(3)]

    assert statuses == 3 * ["redrawn"]
    assert replayer.entered == 1
    assert [request.url.path for request in apis.calls("/recipes/replays/")] == 3 * [
        f"/api/datasets/recipes/replays/{published['replay_key']}"
    ]
    assert apis.rendered_on() == 3 * [REPLAYED]


async def test_a_stamped_root_whose_kinds_change_on_a_standing_version_is_replayed_once(
    apis: RecordedApis,
) -> None:
    apis.recipes[STORED] = kind_recipe(STORED, content_version=FIRST, stamp="stamp-old")
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((FIRST, "stamp-new"))
    apis.enrichment_stamp = "stamp-new"
    replayer = replaying(
        apis, kind_recipe(REPLAYED, content_version=FIRST, stamp="stamp-new", keyed=True)
    )
    async with client_of(service_over(apis, replayer)) as client:
        statuses = [(await refresh(client)).json()["status"] for _ in range(3)]

    assert statuses == 3 * ["redrawn"]
    assert replayer.entered == 1


async def test_an_unstamped_kind_root_whose_version_now_holds_a_party_role_is_replayed_once(
    apis: RecordedApis,
) -> None:
    # Recorded from an unstamped file, as a mint under an apis that sent no stamps
    # names one, while the version's frame holds a party role whose kinds carry a stamp.
    apis.recipes[STORED] = kind_recipe(STORED, content_version=FIRST, stamp=None)
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((FIRST, "stamp-new"))
    apis.enrichment_stamp = "stamp-new"
    replayer = replaying(
        apis, kind_recipe(REPLAYED, content_version=FIRST, stamp="stamp-new", keyed=True)
    )
    async with client_of(service_over(apis, replayer)) as client:
        statuses = [(await refresh(client)).json()["status"] for _ in range(3)]
        apis.store_card(FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]))
        on_replayed = await refresh(client)

    assert statuses == 3 * ["redrawn"]
    assert on_replayed.json() == {"status": "current"}
    assert replayer.entered == 1


async def test_a_stamped_root_whose_version_lists_no_stamp_is_replayed_once_and_logged(
    apis: RecordedApis, caplog: pytest.LogCaptureFixture
) -> None:
    """The version's frame holds no party role, so its mint serves an unstamped kind
    file, and the refresh names the root unstamped too."""
    apis.recipes[STORED] = kind_recipe(STORED, content_version=FIRST, stamp="stamp-old")
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((FIRST, None))
    apis.enrichment_stamp = "stamp-new"
    replayer = replaying(apis, kind_recipe(REPLAYED, content_version=FIRST, stamp=None, keyed=True))
    with caplog.at_level(logging.WARNING, logger="harness.live_refresh"):
        async with client_of(service_over(apis, replayer)) as client:
            statuses = [(await refresh(client)).json()["status"] for _ in range(3)]

    assert statuses == 3 * ["redrawn"]
    assert replayer.entered == 1
    assert any("lists no enrichment_stamp" in record.message for record in caplog.records)


class SuspendingDefinitionStore(InMemoryDefinitionStore):
    """A definition store whose read suspends, as `GcsDefinitionStore.get` does in
    `asyncio.to_thread`, so the read's timeout can fire. `reads` lists the
    `spec_id` of every read."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        self.reads.append(spec_id)
        await asyncio.sleep(0)
        return await super().get(spec_id, schema_version=schema_version)


@pytest.fixture
def flow_definition() -> Iterator[SuspendingDefinitionStore]:
    """The definition store holding `SPEC` as a flow dataset, so a replay mints its
    root with party kinds (`data_mcp.frame_materialisation.replayed_mint_options`).
    Its read suspends, so a refresh that reads it under too small a share of its
    deadline answers `pending`."""
    held = SuspendingDefinitionStore()
    identity = {"entities": {"origin": ["country/FRA"]}}
    asyncio.run(
        held.put_many(
            [
                StoredDefinition(
                    spec_id=SPEC, schema_version=1, canonical_bytes=json.dumps(identity).encode()
                )
            ]
        )
    )
    set_definition_store(held)
    yield held
    set_definition_store(None)


class UnavailableDefinitionStore(SuspendingDefinitionStore):
    """A definition store that is down: every read raises."""

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        self.reads.append(spec_id)
        raise DefinitionStoreUnavailable("the definition store is down")


@pytest.fixture
def unavailable_definition() -> Iterator[SuspendingDefinitionStore]:
    """The definition store as down, so a refresh that reads a definition is refused."""
    held = UnavailableDefinitionStore()
    set_definition_store(held)
    yield held
    set_definition_store(None)


def replayed_under(stamp: str, *, content_version: str = FIRST) -> dict[str, Any]:
    """The recipe a replay of `recipe_of(STORED)` publishes when its flow root mints
    `content_version` with party kinds under `stamp`: the root states its recorded
    options and the stamp its mint record holds, and the key names the stamp
    (`common.recipe.build_recipe`)."""
    root = MintRecord(
        frame=frame_ref_for(
            spec_id=SPEC,
            content_version=content_version,
            columns=KIND_COLUMNS,
            enrichment_stamp=stamp,
        ),
        dataset=FrameRef(spec_id=SPEC, content_version=content_version),
        options=OPTIONS,
        enrichment_stamp=stamp,
    )
    key = replay_key(
        normal_form_hash=NORMAL_FORM,
        root_datasets=[
            root_identity(
                {"spec_id": SPEC, "content_version": content_version}, enrichment_stamp=stamp
            )
        ],
        execution_identities=[current_execution_identity().canonical_string()],
    )
    return recipe_of(REPLAYED) | {"replay_key": key, "roots": [root.model_dump(mode="json")]}


async def test_a_flow_root_recorded_without_kinds_stands_until_its_dataset_moves(
    apis: RecordedApis, flow_definition: SuspendingDefinitionStore
) -> None:
    apis.heads[SPEC] = FIRST
    apis.versions[SPEC] = listed((SECOND, "stamp-new"), (FIRST, "stamp-new"))
    apis.enrichment_stamp = "stamp-new"
    replayed = replayed_under("stamp-new", content_version=SECOND)
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        standing = await refresh(client)
        assert flow_definition.reads == []
        apis.heads[SPEC] = SECOND
        first = await refresh(client)
        apis.recipes[REPLAYED] = replayed
        again = await refresh(client)
        apis.store_card(FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]))
        on_replayed = await refresh(client)

    assert standing.json() == {"status": "current"}
    # Read on the two refreshes that replay or look up a replay key, and not once the
    # card's root records its stamp.
    assert flow_definition.reads == [SPEC, SPEC]
    assert [first.json()["status"], again.json()["status"]] == ["redrawn", "redrawn"]
    assert on_replayed.json() == {"status": "current"}
    assert replayer.entered == 1
    assert [request.url.path for request in apis.calls("/recipes/replays/")] == 2 * [
        f"/api/datasets/recipes/replays/{replayed['replay_key']}"
    ]
    assert apis.rendered_on() == [REPLAYED, REPLAYED]


async def test_a_replay_published_under_another_key_is_logged(
    apis: RecordedApis, caplog: pytest.LogCaptureFixture
) -> None:
    published = replayed_under("stamp-other")
    replayer = Replayer(
        apis,
        Replayed(
            frame=FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]),
            recipe=published,
            newly_published=True,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="harness.live_refresh"):
        async with client_of(service_over(apis, replayer)) as client:
            response = await refresh(client)

    assert response.json()["status"] == "redrawn"
    [logged] = [record for record in caplog.records if record.name == "harness.live_refresh"]
    assert published["replay_key"] in logged.args


# --- the replay and the render --------------------------------------------------


async def test_a_replayed_card_is_rendered_on_the_new_frame(apis: RecordedApis) -> None:
    replayer = Replayer(apis)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert (response.status_code, response.json()) == (
        200,
        {"status": "redrawn", "render": RECORDED_RENDER},
    )
    assert apis.rendered_on() == [REPLAYED]
    assert replayer.entered == 1


@pytest.mark.parametrize("reason", ["mint_queued", "deadline", "row_budget", "busy"])
async def test_a_replay_that_has_not_finished_answers_pending_with_its_wait(
    apis: RecordedApis, reason: Any
) -> None:
    replayer = Replayer(apis, Pending(retry_after_s=2.5, reason=reason))
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert (response.status_code, response.json()) == (
        200,
        {"status": "pending", "retry_after_s": 2.5},
    )
    assert apis.calls("/render") == []


async def test_a_replay_refusal_is_answered_under_its_own_code(apis: RecordedApis) -> None:
    replayer = Replayer(
        apis, Refused(code="growth_gap_year", detail="2021 is missing", stage="verb")
    )
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert_refused(response, "growth_gap_year")
    assert response.json()["detail"] == "2021 is missing"
    assert apis.calls("/render") == []


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (1, "redrawn"),
        (2, "pending"),
    ],
)
async def test_a_dataset_that_moves_under_a_replay_is_replayed_once_more_then_waits(
    apis: RecordedApis, outcomes: int, expected: str
) -> None:
    moved = Refused(code="upstream_version_changed", detail="moved", stage="mint")
    done = Replayed(
        frame=FrameRef(spec_id=REPLAYED[0], content_version=REPLAYED[1]),
        recipe={},
        newly_published=True,
    )
    replayer = Replayer(apis, *[moved] * outcomes, done)
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert (response.status_code, response.json()["status"]) == (200, expected)
    assert replayer.entered == 2


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (problem("render.render_target_mismatch", status=409, code="render_target_mismatch"), None),
        (problem("render.render_target_mismatch", status=422, code="binding_invalid"), None),
        (problem("render.render_target_mismatch", status=403, code="refresh_not_permitted"), None),
        (httpx2.Response(418, json={"retryable": False}), "dfo_apis_refused"),
        (
            problem(
                "render.render_target_mismatch",
                status=409,
                code="frame_unavailable",
                retryable=True,
            ),
            "upstream_unavailable",
        ),
        (
            problem("render.render_target_mismatch", status=500, code="internal_error"),
            "upstream_unavailable",
        ),
        (httpx2.Response(502, text="bad gateway"), "upstream_unavailable"),
        (httpx2.ConnectError("refused"), "upstream_unavailable"),
    ],
)
async def test_a_render_refusal_keeps_dfo_apis_code_and_a_failure_is_upstream_unavailable(
    apis: RecordedApis, answer: httpx2.Response | Exception, code: str | None
) -> None:
    apis.render_answer = answer
    async with client_of(service_over(apis, Replayer(apis))) as client:
        response = await refresh(client)

    if code is None:
        assert isinstance(answer, httpx2.Response)
        code = answer.json()["code"]
    assert_refused(response, code)


async def test_a_tool_call_that_does_not_complete_is_upstream_unavailable(
    apis: RecordedApis,
) -> None:
    replayer = Replayer(apis, ToolSurfaceUnavailableError("analysis-mcp did not answer"))
    async with client_of(service_over(apis, replayer)) as client:
        response = await refresh(client)

    assert_refused(response, "upstream_unavailable")


async def test_without_the_datasets_principal_the_route_says_it_is_not_configured(
    apis: RecordedApis,
) -> None:
    unconfigured = LiveApisClient(
        base_url="http://apis.test",
        live_refresh_principal=LIVE_PRINCIPAL,
        transport=httpx2.MockTransport(apis),
    )
    async with client_of(service_over(apis, Replayer(apis), apis=unconfigured)) as client:
        response = await refresh(client)

    assert_refused(response, "refresh_not_configured")


async def test_a_failed_token_mint_is_upstream_unavailable_and_sends_nothing(
    apis: RecordedApis,
) -> None:
    async def no_token() -> Mapping[str, str]:
        raise ApisAuthError("metadata server answered 500")

    unauthenticated = LiveApisClient(
        base_url="http://apis.test",
        datasets_principal=PRINCIPAL,
        live_refresh_principal=LIVE_PRINCIPAL,
        auth=no_token,
        transport=httpx2.MockTransport(apis),
    )
    async with client_of(service_over(apis, Replayer(apis), apis=unauthenticated)) as client:
        response = await refresh(client)

    assert_refused(response, "upstream_unavailable")
    assert apis.requests == []


# --- time and load --------------------------------------------------------------


async def test_a_step_that_never_ends_is_cut_and_the_request_answers_pending(
    apis: RecordedApis,
) -> None:
    """The deadline stops new steps; a verb already in flight is cut at the hard
    stop, so the request ends on its own."""
    cut = asyncio.Event()

    async def slow_verb(**_: Any) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cut.set()

    async def replay(recipe: Mapping[str, Any], *, call_tool: Any, **_: Any) -> Any:
        return await call_tool(name="growth", arguments={})

    service = service_over(
        apis, Replayer(apis), replay=replay, call_tool=slow_verb, deadline_s=0.05, grace_s=0.05
    )
    async with client_of(service) as client, asyncio.timeout(5):
        response = await refresh(client)

    assert (response.status_code, response.json()) == (
        200,
        {"status": "pending", "retry_after_s": 1.0},
    )
    assert cut.is_set()
    assert apis.calls("/render") == []


async def test_requests_over_the_rate_answer_pending_and_reach_nothing(apis: RecordedApis) -> None:
    replayer = Replayer(apis)
    service = service_over(apis, replayer, burst=1, requests_per_s=0.5)
    async with client_of(service) as client:
        first = await refresh(client)
        asked = len(apis.requests)
        second = await refresh(client)

    assert first.json()["status"] == "redrawn"
    assert (second.status_code, second.json()) == (200, {"status": "pending", "retry_after_s": 2.0})
    assert len(apis.requests) == asked


async def test_requests_over_the_in_flight_cap_answer_pending(apis: RecordedApis) -> None:
    other = "cmp_other"
    apis.cards[other] = STORED
    replayer = Replayer(apis)
    replayer.gate = asyncio.Event()
    async with client_of(service_over(apis, replayer, max_in_flight=1)) as client:
        held = asyncio.create_task(refresh(client))
        while replayer.entered == 0:
            await asyncio.sleep(0)
        over = await refresh(client, component_id=other)
        replayer.gate.set()
        first = await held
        after = await refresh(client, component_id=other)

    assert (over.status_code, over.json()) == (200, {"status": "pending", "retry_after_s": 2.0})
    assert first.json()["status"] == "redrawn"
    assert after.json()["status"] == "redrawn"


async def test_a_run_holds_its_in_flight_slot_until_it_ends(apis: RecordedApis) -> None:
    """The cap counts runs. A request joining a run takes no slot, and a run
    whose callers have all gone keeps its slot until it ends."""
    other = "cmp_other"
    apis.cards[other] = STORED
    replayer = Replayer(apis)
    replayer.gate = asyncio.Event()
    service = service_over(apis, replayer, max_in_flight=1)

    first = asyncio.create_task(service.answer(component_id=CARD, revision=1))
    while replayer.entered == 0:
        await asyncio.sleep(0)
    joined = asyncio.create_task(service.answer(component_id=CARD, revision=1))
    await asyncio.sleep(0)
    first.cancel()
    joined.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.gather(first, joined)
    async with asyncio.timeout(5):
        over = await service.answer(component_id=other, revision=1)
    run = service._runs[(CARD, 1)]
    replayer.gate.set()
    await run
    after = await service.answer(component_id=other, revision=1)

    assert over.outcome == "pending:in_flight"
    assert run.result().outcome == "redrawn"
    assert after.outcome == "redrawn"
    assert replayer.entered == 2
    assert (service._in_flight, service._runs) == (0, {})


async def test_concurrent_requests_for_one_card_share_one_replay(apis: RecordedApis) -> None:
    replayer = Replayer(apis)
    replayer.gate = asyncio.Event()
    async with client_of(service_over(apis, replayer)) as client:
        requests = [asyncio.create_task(refresh(client)) for _ in range(6)]
        while replayer.entered == 0:
            await asyncio.sleep(0)
        # Every request has read the map of runs before the one replay ends.
        for _ in range(50):
            await asyncio.sleep(0)
        replayer.gate.set()
        responses = await asyncio.gather(*requests)

    assert [response.json() for response in responses] == [
        {"status": "redrawn", "render": RECORDED_RENDER}
    ] * 6
    assert replayer.entered == 1
    assert [request.method for request in apis.calls("/components/")] == ["GET", "POST"]
    assert apis.rendered_on() == [REPLAYED]


# --- the service built from settings --------------------------------------------


async def test_the_served_refresh_has_a_row_budget_and_a_publisher_of_its_own() -> None:
    service = build_live_refresh(settings=SETTINGS)
    try:
        assert service.row_budget is not row_budget()
        assert service.row_budget.max_rows == SETTINGS.live_refresh_rows_in_flight_max
        assert service.row_budget.max_rows < row_budget().max_rows
        assert getattr(service.publish, "__self__", None).__class__ is PlatformAssembler
    finally:
        await service.aclose()


async def test_on_cloud_run_both_dfo_apis_clients_share_one_id_token_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "dfo-dev-ai")
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-dev-apis-abc123.a.run.app/")

    service = build_live_refresh(settings=SETTINGS)
    try:
        auth = service._apis._auth
        assert isinstance(auth, IdTokenProvider)
        assert auth.audience == "https://dfo-dev-apis-abc123.a.run.app"
        assembler = getattr(service.publish, "__self__", None)
        assert isinstance(assembler, PlatformAssembler)
        assert assembler._auth is auth
    finally:
        await service.aclose()


def test_a_refresh_row_budget_too_small_for_a_two_input_verb_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="LIVE_REFRESH_ROWS_IN_FLIGHT_MAX"):
        build_live_refresh(settings=Settings(live_refresh_rows_in_flight_max=499_999))
