"""What the `harness.live_refresh` tests share: apis as a fake whose answers
on the live-refresh routes are the ones apis recorded.

`RecordedApis` extends `replay_support.FakeApis` (the publish door and versions
route) with the component read, the recipe read, the frame refresh, the replay
lookup, and the render. Each answer is a response body of
`packages/common/tests/fixtures/dfo_apis_live_refresh_wire_pairs.json`, with
only the ids a scenario names put in. The recording holds no component read of a
derived card and no `not_found` refusal, so two answers are derived:

- a derived card's component read is the recorded
  `render.derived_card_with_anchor_warning` body, which has a component read's
  keys, with `stored_intent.frame_ref` set to the card's frame;
- an unknown component is the recorded `recipe_read.recipe_not_found` problem
  with `code` `not_found` and its `type` and `title` to match.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx2
from common.live_apis import PRINCIPAL_HEADER, LiveApisClient
from common.materialisation import FrameRef

from .replay_support import PRINCIPAL, FakeApis, Pair, pair

LIVE_PRINCIPAL = "live-refresh-principal-91bc"
CARD = "cmp_000000000000000000000002"
DATASET_CARD = "cmp_000000000000000000000001"

RECORDED: dict[str, dict[str, Any]] = json.loads(
    (
        Path(__file__).parents[3]
        / "packages/common/tests/fixtures/dfo_apis_live_refresh_wire_pairs.json"
    ).read_text()
)["pairs"]
# The render apis recorded for a derived card, which every redraw here answers.
RECORDED_RENDER: dict[str, Any] = RECORDED["render.derived_card_with_anchor_warning"]["response"][
    "body"
]

_REFRESH_PATH = "/api/datasets/frames/refresh"
_COMPONENT_PREFIX = "/api/presentation/components/"
_RECIPE_SUFFIX = "/recipe"
_REPLAY_PREFIX = "/api/datasets/recipes/replays/"


def recorded(name: str, *, status: int | None = None, **members: Any) -> httpx2.Response:
    """The recorded response `name`, with `members` replacing members of its body."""
    response = RECORDED[name]["response"]
    return httpx2.Response(
        status or response["status"],
        json={**response["body"], **members},
        headers=response["headers"],
    )


def problem(name: str, *, status: int, code: str, retryable: bool = False) -> httpx2.Response:
    """The recorded problem `name` under another status and code."""
    return recorded(
        name,
        status=status,
        code=code,
        retryable=retryable,
        type=f"/codes/{code}",
        title=code.replace("_", " ").title(),
    )


class RecordedApis(FakeApis):
    """apis for a live refresh. `cards` maps a component id to the frame its
    revision 1 is stored on, `heads` a dataset to its current version, and
    `rebuilding` a dataset whose rebuild is queued to the wait its answer names.
    `render_answer` replaces the render's answer, or is raised in its place, and
    `versions_answer` the versions route's."""

    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, Pair] = {}
        self.heads: dict[str, str] = {}
        self.rebuilding: dict[str, float] = {}
        self.render_answer: httpx2.Response | Exception | None = None
        self.versions_answer: httpx2.Response | None = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        answer = self._recorded_answer(request)
        if answer is None:
            return super().__call__(request)
        self.requests.append(request)
        return answer

    def _recorded_answer(self, request: httpx2.Request) -> httpx2.Response | None:
        """The answer of a route this class serves, `None` for one `FakeApis` does."""
        path, principal = request.url.path, request.headers.get(PRINCIPAL_HEADER)
        if path == _REFRESH_PATH or path.endswith("/render"):
            return self._live(path, request, admitted=principal == LIVE_PRINCIPAL)
        if path.startswith(_COMPONENT_PREFIX):
            return self._component(request)
        if path.endswith("/versions") and self.versions_answer is not None:
            return self.versions_answer
        if principal == PRINCIPAL and path.endswith(_RECIPE_SUFFIX):
            return self._recipe(path)
        if (
            principal == PRINCIPAL
            and path.startswith(_REPLAY_PREFIX)
            and self._replayed(path) is None
        ):
            return recorded("replay_lookup.recipe_not_found")
        return None

    def _live(self, path: str, request: httpx2.Request, *, admitted: bool) -> httpx2.Response:
        if not admitted:
            return recorded("frame_refresh.refresh_not_permitted")
        body = json.loads(request.content)
        return self._refresh(body) if path == _REFRESH_PATH else self._render(path, body)

    def _recipe(self, path: str) -> httpx2.Response:
        spec_id, content_version = path.split("/")[-3:-1]
        held = self.recipes.get((spec_id, content_version))
        if held is None:
            return recorded("recipe_read.recipe_not_found")
        return httpx2.Response(
            200, json=held, headers=RECORDED["recipe_read.found"]["response"]["headers"]
        )

    def _replayed(self, path: str) -> Pair | None:
        key = path.rsplit("/", 1)[1]
        return next((p for p, recipe in self.recipes.items() if recipe["replay_key"] == key), None)

    def _component(self, request: httpx2.Request) -> httpx2.Response:
        component_id = request.url.path.removeprefix(_COMPONENT_PREFIX)
        if component_id == DATASET_CARD:
            return recorded("component_read.at_revision")
        frame = self.cards.get(component_id)
        if frame is None or request.url.params.get("revision") != "1":
            return problem("recipe_read.recipe_not_found", status=404, code="not_found")
        body = copy.deepcopy(RECORDED_RENDER)
        body["stored_intent"]["frame_ref"] = {"spec_id": frame[0], "content_version": frame[1]}
        return httpx2.Response(200, json=body)

    def _refresh(self, body: dict[str, str]) -> httpx2.Response:
        spec_id = body["spec_id"]
        if spec_id in self.rebuilding:
            return recorded("frame_refresh.pending", retry_after_s=self.rebuilding[spec_id])
        head = self.heads[spec_id]
        name = "frame_refresh.current" if head == body["content_version"] else "frame_refresh.moved"
        return recorded(name, frame={"spec_id": spec_id, "content_version": head})

    def _render(self, path: str, body: dict[str, Any]) -> httpx2.Response:
        """The recorded render, for a target whose recipe has the normal form of the
        stored frame's, which is apis' acceptance rule for a derived card."""
        if isinstance(self.render_answer, Exception):
            raise self.render_answer
        if self.render_answer is not None:
            return self.render_answer
        component_id = path.split("/")[-4]
        stored = self.recipes.get(self.cards[component_id])
        target = self.recipes.get((body["frame"]["spec_id"], body["frame"]["content_version"]))
        if (
            stored is None
            or target is None
            or stored["normal_form_hash"] != target["normal_form_hash"]
        ):
            return recorded("render.render_target_mismatch")
        return recorded("render.derived_card_with_anchor_warning")

    def calls(self, fragment: str) -> list[httpx2.Request]:
        """Every request whose path holds `fragment`."""
        return [request for request in self.requests if fragment in request.url.path]

    def rendered_on(self) -> list[Pair]:
        """The frame of each render request, in order."""
        return [
            (frame["spec_id"], frame["content_version"])
            for frame in (json.loads(request.content)["frame"] for request in self.calls("/render"))
        ]

    def store_card(self, frame: FrameRef, *, component_id: str = CARD) -> None:
        self.cards[component_id] = pair(frame)

    def client(self) -> LiveApisClient:
        async def no_sleep(_: float) -> None:
            return None

        return LiveApisClient(
            base_url="http://apis.test",
            datasets_principal=PRINCIPAL,
            live_refresh_principal=LIVE_PRINCIPAL,
            transport=httpx2.MockTransport(self),
            sleep=no_sleep,
        )
