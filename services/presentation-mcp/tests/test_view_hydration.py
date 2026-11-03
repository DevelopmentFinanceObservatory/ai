"""The view page in headless Chromium: what Prefab's bundled renderer draws from
a host's tool result once DFO's hydration script has run on it.

The page is the one `theme.renderer_html` builds, made from Prefab's bundled
page in the installed wheel, so a prefab-ui bump runs these cases against the
new renderer. Every request is answered in process and anything else is
aborted, so the run is offline. A missing Chromium fails the run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, FrameLocator, Page, Route, expect, sync_playwright
from prefab_ui.renderer import get_renderer_html
from presentation_mcp import theme

pytestmark = pytest.mark.browser

FIXTURES = Path(__file__).parent / "fixtures"
ORIGIN = "http://view.test"
COMPONENT_ID = "cmp_01M6GWWZ42CJ68FDEDEY0VGK0Z"
TITLE = "Net ODA as a share of GNI, 2010\u20132025 (UK, France, Germany)"
SERIES = 2
NO_DATA_NOTICE = "This view could not read its data."
LOAD_NOTICE = theme._HYDRATION_NOTICE
WRITE_TOKEN = "cmw_01M6GWWZ42CJ68FDEDEY0VGK10"
DRAW_TIMEOUT_MS = 5_000

# Records what reaches Prefab: each tool result as Prefab receives it, and the
# id of each message that carries one. It runs after the hydration script's
# listener and before Prefab's, which registers only once its module script
# has run.
PROBE_SCRIPT = """<script>
window.seen = [];
window.ids = [];
window.addEventListener("message", (event) => {
  const message = event.data;
  if (message && message.method === "ui/notifications/tool-result") {
    window.seen.push(structuredClone(message.params));
  }
  if (message && "id" in message) window.ids.push(message.id);
});
</script>
"""


@pytest.fixture(scope="module")
def bundled_page() -> str:
    return get_renderer_html(mode="bundled")


def _view_page(bundled_page: str, *, reread_deadline_ms: int | None = None) -> str:
    """The view page, the hydration script's re-read deadline shortened when given."""
    config = theme._hydration_config(ORIGIN)
    if reread_deadline_ms is not None:
        config["rereadDeadlineMs"] = reread_deadline_ms
    return theme._with_hydration_script(
        bundled_page, theme._hydration_script(config) + PROBE_SCRIPT
    )


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def described() -> dict[str, Any]:
    """A saved `describe_component` result for a two-series line chart."""
    return json.loads((FIXTURES / "describe_component_result.json").read_text())


@pytest.fixture
def console() -> dict[Page, list[str]]:
    """Console messages by page, filled by `open_view`."""
    return {}


@pytest.fixture
def open_view(
    browser: Browser, bundled_page: str, console: dict[Page, list[str]]
) -> Iterator[Callable[..., Page]]:
    """Opens the host page on a case and returns it, the view loading in its frame.

    Every console message the page or its frame logs lands in `console[page]`.
    """
    host_page = (FIXTURES / "view_host.html").read_text()
    contexts = []

    def open_(
        *,
        result: dict[str, Any] | None,
        tool_name: str = "describe_component",
        tool_input: dict[str, Any] | None = None,
        host_capabilities: dict[str, Any] | None = None,
        tool_call_results: dict[str, Any] | None = None,
        tool_call_replies: dict[str, Any] | None = None,
        cancelled: bool = False,
        result_on_tool_call: dict[str, Any] | None = None,
        reread_deadline_ms: int | None = None,
    ) -> Page:
        """`tool_call_results` gives a result per tool name, `tool_call_replies`
        the JSON-RPC response's `result` or `error` members instead."""
        view_page = _view_page(bundled_page, reread_deadline_ms=reread_deadline_ms)
        pages = {f"{ORIGIN}/host.html": host_page, f"{ORIGIN}/view.html": view_page}

        def serve(route: Route) -> None:
            body = pages.get(route.request.url)
            if body is None:
                route.abort()
            else:
                route.fulfill(status=200, content_type="text/html", body=body)

        case = {
            "toolName": tool_name,
            "toolInput": tool_input or {"component_id": COMPONENT_ID},
            "result": result,
            "cancelled": cancelled,
            "hostCapabilities": host_capabilities or {"serverTools": {}},
            "toolCallReplies": {
                name: {"result": reply} for name, reply in (tool_call_results or {}).items()
            }
            | (tool_call_replies or {}),
            "resultOnToolCall": result_on_tool_call,
        }
        context = browser.new_context(viewport={"width": 800, "height": 600})
        contexts.append(context)
        context.route("**/*", serve)
        context.add_init_script(f"window.CASE = {json.dumps(case)};")
        page = context.new_page()
        logged: list[str] = []
        page.on("console", lambda message: logged.append(message.text))
        console[page] = logged
        page.goto(f"{ORIGIN}/host.html")
        return page

    yield open_
    for context in contexts:
        context.close()


def _view(page: Page) -> FrameLocator:
    return page.frame_locator("#view")


def _expect_chart_drawn(page: Page) -> None:
    view = _view(page)
    expect(view.locator(".recharts-line")).to_have_count(SERIES, timeout=DRAW_TIMEOUT_MS)
    expect(view.locator(".dfo-title")).to_have_text(TITLE)
    expect(view.get_by_text(NO_DATA_NOTICE)).to_have_count(0)


def _results_prefab_saw(page: Page) -> list[dict[str, Any]]:
    frame = page.frame(url=f"{ORIGIN}/view.html")
    assert frame is not None
    return frame.evaluate("window.seen")


def test_a_result_sent_per_the_spec_draws_the_chart_with_its_data(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    page = open_view(result=described)

    _expect_chart_drawn(page)


def test_a_result_with_nulls_in_its_optional_fields_draws_the_chart(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    # anthropics/claude-ai-mcp#647: claude.ai writes `null` into optional fields.
    result = copy.deepcopy(described)
    first, second = result["content"]
    first["annotations"] = {"audience": None, "priority": 0.5}
    first["_meta"] = None
    second["annotations"] = None
    result["content"].insert(0, None)
    result["isError"] = None

    page = open_view(result=result)

    _expect_chart_drawn(page)
    (seen,) = _results_prefab_saw(page)
    assert [block.get("annotations") for block in seen["content"]] == [{"priority": 0.5}, None]
    assert "_meta" not in seen["content"][0]
    assert "isError" not in seen


def test_a_result_whose_meta_is_null_draws_the_shell_with_the_no_data_notice(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    result = copy.deepcopy(described)
    result["_meta"] = None

    page = open_view(result=result)

    view = _view(page)
    expect(view.get_by_text(NO_DATA_NOTICE)).to_be_visible(timeout=DRAW_TIMEOUT_MS)
    expect(view.locator(".dfo-title")).to_have_text(TITLE)


def test_nulls_inside_structured_content_and_meta_reach_prefab_untouched(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    result = copy.deepcopy(described)
    result["structuredContent"]["state"]["note"] = None
    result["_meta"]["org.one.dfo/view_state"]["data"]["rows"][0]["s1"] = None
    result["_meta"]["org.one.dfo/trace"] = None

    page = open_view(result=result)

    _expect_chart_drawn(page)
    (seen,) = _results_prefab_saw(page)
    assert seen["structuredContent"]["state"]["note"] is None
    assert seen["structuredContent"]["state"]["data"]["rows"][0]["s1"] is None
    assert seen["_meta"]["org.one.dfo/trace"] is None


def _without_view(result: dict[str, Any]) -> dict[str, Any]:
    """`result` as claude.ai delivers it under anthropics/claude-ai-mcp#1158:
    no `structuredContent`, and no `_meta` either."""
    return {
        key: value for key, value in result.items() if key not in {"structuredContent", "_meta"}
    }


def _created(described: dict[str, Any], *, revision: int) -> dict[str, Any]:
    """A write_component create's result for the described chart at `revision`,
    without its view: a lead, the payload as JSON, then the write token."""
    lead, payload_block = described["content"]
    payload = {**json.loads(payload_block["text"]), "revision": revision}
    return {
        "content": [
            lead,
            {"type": "text", "text": json.dumps(payload)},
            {"type": "text", "text": f"Keep this write_token.\nwrite_token: {WRITE_TOKEN}"},
        ],
        "isError": False,
    }


def _rereads(page: Page) -> list[dict[str, Any]]:
    """The `tools/call` requests the view sent its host."""
    return [
        message for message in page.evaluate("window.up") if message.get("method") == "tools/call"
    ]


@pytest.mark.parametrize(
    ("result", "tool_input", "pin"),
    [
        pytest.param(
            "create",
            {"intent": {"type": "line_chart"}},
            {"component_id": COMPONENT_ID, "revision": 2},
            id="create-pinned-by-its-payload",
        ),
        pytest.param(
            "describe",
            {"component_id": COMPONENT_ID, "revision": 1},
            {"component_id": COMPONENT_ID, "revision": 1},
            id="describe-pinned-by-its-arguments",
        ),
    ],
)
def test_a_result_without_its_view_is_read_again_through_the_host_and_drawn(
    open_view: Callable[..., Page],
    console: dict[Page, list[str]],
    described: dict[str, Any],
    result: str,
    tool_input: dict[str, Any],
    pin: dict[str, Any],
) -> None:
    lost = _created(described, revision=2) if result == "create" else _without_view(described)
    if result == "describe":
        del lost["content"]

    page = open_view(
        result=lost,
        tool_name="write_component" if result == "create" else "describe_component",
        tool_input=tool_input,
        tool_call_results={"describe_component": described},
    )

    _expect_chart_drawn(page)
    (request,) = _rereads(page)
    assert request["params"] == {
        "name": "describe_component",
        "arguments": {**pin, "include_projection": "none"},
    }
    # Prefab sees the re-read result alone.
    (seen,) = _results_prefab_saw(page)
    assert seen["structuredContent"]["view"] == described["structuredContent"]["view"]
    frame = page.frame(url=f"{ORIGIN}/view.html")
    assert frame is not None
    assert request["id"] not in frame.evaluate("window.ids")
    assert not [text for text in console[page] if request["id"] in text]


@pytest.mark.parametrize(
    ("result", "tool_input"),
    [
        pytest.param(
            {
                "content": [{"type": "text", "text": "The request was refused: write_refused."}],
                "isError": True,
            },
            {"component_id": COMPONENT_ID, "revision": 1},
            id="error-result",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "Line chart."}]},
            {"intent": {"type": "line_chart"}, "component_id": COMPONENT_ID},
            id="write-without-its-payload",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "Line chart."}]},
            {"component_id": COMPONENT_ID},
            id="describe-without-a-revision",
        ),
        pytest.param(None, {"component_id": COMPONENT_ID, "revision": 1}, id="tool-cancelled"),
    ],
)
def test_a_result_without_its_view_that_cannot_be_read_again_draws_the_notice(
    open_view: Callable[..., Page],
    described: dict[str, Any],
    result: dict[str, Any] | None,
    tool_input: dict[str, Any],
) -> None:
    page = open_view(
        result=result,
        # With no result to send, the host cancels the call.
        cancelled=result is None,
        tool_input=tool_input,
        tool_call_results={"describe_component": described},
    )

    expect(_view(page).get_by_text(LOAD_NOTICE)).to_be_visible(timeout=DRAW_TIMEOUT_MS)
    assert _rereads(page) == []


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(None, id="no-reply"),
        pytest.param("viewless", id="reply-without-a-view"),
        pytest.param("error", id="error-reply"),
        pytest.param("jsonrpc-error", id="json-rpc-error-reply"),
    ],
)
def test_a_re_read_that_fails_draws_the_notice(
    open_view: Callable[..., Page], described: dict[str, Any], reply: str | None
) -> None:
    replies = {
        None: {},
        "viewless": {"describe_component": {"result": _without_view(described)}},
        "error": {"describe_component": {"result": {**described, "isError": True}}},
        "jsonrpc-error": {
            "describe_component": {"error": {"code": -32603, "message": "Internal error"}}
        },
    }[reply]

    page = open_view(
        result=_without_view(described),
        tool_input={"component_id": COMPONENT_ID},
        tool_call_replies=replies,
        reread_deadline_ms=300,
    )

    expect(_view(page).get_by_text(LOAD_NOTICE)).to_be_visible(timeout=DRAW_TIMEOUT_MS)
    (request,) = _rereads(page)
    assert request["params"]["arguments"] == {
        "component_id": COMPONENT_ID,
        "revision": 1,
        "include_projection": "none",
    }


def test_a_re_read_reply_with_a_null_error_beside_its_result_draws_the_chart(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    page = open_view(
        result=_without_view(described),
        tool_call_replies={"describe_component": {"result": described, "error": None}},
    )

    _expect_chart_drawn(page)


def test_a_re_read_overtaken_by_a_good_result_leaves_that_result_drawn(
    open_view: Callable[..., Page], described: dict[str, Any]
) -> None:
    # The host answers the re-read with a full result of its own and never
    # replies to the request, so the re-read's deadline passes after it.
    deadline_ms = 300
    page = open_view(
        result=_without_view(described),
        result_on_tool_call=described,
        reread_deadline_ms=deadline_ms,
    )

    _expect_chart_drawn(page)
    page.wait_for_timeout(deadline_ms * 3)
    assert len(_results_prefab_saw(page)) == 1
    expect(_view(page).get_by_text(LOAD_NOTICE)).to_have_count(0)
