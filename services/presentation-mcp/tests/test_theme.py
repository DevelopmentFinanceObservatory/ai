"""The theme: the CSS and fonts every view payload carries, and the CDN renderer resource."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterator
from importlib.metadata import version
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastmcp.apps import AppConfig, ResourceCSP, app_config_to_meta_dict
from prefab_ui.components import Text
from prefab_ui.renderer import get_renderer_csp, get_renderer_html
from presentation_mcp import theme
from presentation_mcp.app_view import app_shell
from presentation_mcp.platform_contract import VIEW_STATE_META_KEY

from .stub_assembler import resolved_revision

BASE_URL = "https://example.test:8443/mcp/presentation"
SERIES_PALETTE = ["#6467de", "#f6935a", "#f46f58", "#bb6fdc", "#662d6e", "#463138"]
SERIES_PALETTE_LIGHT = ["#8285e7", "#f9b170", "#f78f76", "#cf94e8", "#7f4487", "#5e474f"]

CLASS_CONTRACT = {
    "dfo-card",
    "dfo-head",
    "dfo-title",
    "dfo-caption",
    "dfo-legend",
    "dfo-legend-item",
    "dfo-legend-swatch",
    "dfo-chrome",
    "dfo-more",
    "dfo-built-with",
    "dfo-details",
    "dfo-stat-value",
    "dfo-stat-unit",
}
MEMBERS = ("line_chart", "bar_chart", "table", "stat_card", "note_narration", "source_card")


def _css_classes(tree: Any) -> Iterator[str]:
    if isinstance(tree, dict):
        if "cssClass" in tree:
            yield from tree["cssClass"].split()
        for value in tree.values():
            yield from _css_classes(value)
    elif isinstance(tree, list):
        for item in tree:
            yield from _css_classes(item)


@pytest.fixture(scope="module")
def bundle() -> str:
    return get_renderer_html(mode="bundled")


@pytest.fixture(scope="module")
def css() -> str:
    return theme.THEME_CSS


def test_the_renderer_resource_is_prefabs_cdn_page_pinned_to_the_installed_version() -> None:
    page = theme.renderer_html()
    base = f"https://cdn.jsdelivr.net/npm/@prefecthq/prefab-ui@{version('prefab-ui')}/dist/app"

    assert re.sub(r"<script>.*?</script>\n", "", page, flags=re.DOTALL) == get_renderer_html(
        mode="cdn"
    )
    assert f'href="{base}/renderer.css"' in page
    assert f'src="{base}/renderer.js"' in page
    assert len(page) < 1500


def test_the_view_state_script_sits_in_the_head_ahead_of_prefabs_module_script() -> None:
    page = theme.renderer_html()

    (script,) = re.findall(r"<script>(.*?)</script>", page, flags=re.DOTALL)
    assert page.index("<script>") < page.index("</head>") < page.index('<script type="module"')
    assert 'addEventListener("message"' in script
    assert "event.source !== window.parent" in script
    assert '"ui/notifications/tool-result"' in script
    assert json.dumps(VIEW_STATE_META_KEY) in script
    assert "structuredContent" in script
    assert "state" in script


def test_the_theme_is_unlayered_so_it_beats_prefabs_tailwind_layers(css: str) -> None:
    assert "@layer" not in css


def test_the_fonts_url_asks_google_fonts_for_outfit_and_dm_mono(css: str) -> None:
    assert theme.FONTS_URL == (
        "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;800&family=DM+Mono:wght@400;500&display=swap"
    )
    assert "&amp;" not in theme.FONTS_URL
    assert '--font-sans: "Outfit", system-ui, sans-serif;' in css
    assert '--font-mono: "DM Mono", ui-monospace, monospace;' in css


def test_the_csp_admits_the_fonts_url_its_font_files_the_theme_origin_and_jsdelivr() -> None:
    url_origin = "{0.scheme}://{0.netloc}".format(urlsplit(theme.FONTS_URL))
    meta = app_config_to_meta_dict(AppConfig(csp=ResourceCSP(**theme.renderer_csp(BASE_URL))))
    assert meta["csp"] == {
        "resourceDomains": [
            url_origin,
            "https://fonts.gstatic.com",
            "https://example.test:8443",
            "https://cdn.jsdelivr.net",
        ]
    }


@pytest.mark.parametrize(
    ("base_url", "origin"),
    [
        ("https://example.test:8443/mcp/presentation", "https://example.test:8443"),
        ("https://Example.Test/mcp/presentation", "https://example.test"),
        ("http://localhost:8080/mcp/presentation", "http://localhost:8080"),
        ("http://[::1]:8080/mcp/presentation", "http://[::1]:8080"),
    ],
)
def test_the_csp_theme_origin_is_the_scheme_host_and_explicit_port(
    base_url: str, origin: str
) -> None:
    assert theme.renderer_csp(base_url)["resource_domains"][2] == origin


def test_the_csp_admits_every_origin_prefabs_cdn_page_needs() -> None:
    needed = get_renderer_csp(mode="cdn")["resource_domains"]

    assert needed
    assert set(needed) <= set(theme.renderer_csp(BASE_URL)["resource_domains"])


def test_the_theme_url_is_the_base_url_and_the_content_hash() -> None:
    hashed = f"{BASE_URL}/theme-{theme.THEME_HASH}.css"
    assert theme.theme_url(BASE_URL) == hashed
    assert theme.theme_url(f"{BASE_URL}/") == hashed
    assert re.fullmatch(r"[0-9a-f]{12}", theme.THEME_HASH)
    assert hashlib.sha256(theme.THEME_CSS.encode()).hexdigest()[:12] == theme.THEME_HASH


def test_a_themed_app_names_the_fonts_and_the_theme_by_url_in_light_mode() -> None:
    payload = theme.themed_app(Text("x"), base_url=BASE_URL, state={"key": "value"}).to_json()

    assert payload["stylesheets"] == [theme.FONTS_URL, theme.theme_url(BASE_URL)]
    assert "css" not in payload
    assert theme.THEME_CSS not in json.dumps(payload)
    assert payload["mode"] == "light"
    assert payload["state"] == {"key": "value"}


def test_the_series_palette_is_the_websites_default() -> None:
    assert list(theme.SERIES_COLOURS) == SERIES_PALETTE
    assert list(theme.SERIES_COLOURS_LIGHT) == SERIES_PALETTE_LIGHT


def test_the_css_defines_no_series_custom_properties(css: str) -> None:
    assert "--dfo-series" not in css


def test_the_icon_and_logo_are_inline_svg_masks(css: str) -> None:
    urls = re.findall(r'url\("data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)"\)', css)
    svgs = {base64.b64decode(url).decode() for url in urls}
    assert svgs == {theme._INFO_ICON, theme.DFO_LOGO}
    assert all(svg.startswith("<svg ") and svg.endswith("</svg>") for svg in svgs)
    assert "http" not in css.replace("http://www.w3.org/2000/svg", "")


def test_every_prefab_and_recharts_class_the_theme_targets_is_in_the_pinned_bundle(
    bundle: str, css: str
) -> None:
    targeted = set(re.findall(r"\.((?:recharts|pf)-[a-z0-9-]+)", css))
    assert {"pf-accordion-trigger", "recharts-cartesian-axis-tick-value"} <= targeted
    missing = {name for name in targeted if not re.search(rf"{name}(?![a-z0-9-])", bundle)}
    assert missing == set()


def test_the_theme_styles_every_class_in_the_view_contract(css: str) -> None:
    assert set(re.findall(r"\.(dfo-[a-z-]+)", css)) == CLASS_CONTRACT


def test_the_views_set_exactly_the_contracts_classes() -> None:
    views = [app_shell(resolved_revision(member), base_url=BASE_URL) for member in MEMBERS]

    used = {name for name in _css_classes(views) if name.startswith("dfo-")}

    assert used == CLASS_CONTRACT


def test_the_uri_names_the_prefab_version_and_csp_by_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    uri = theme.renderer_uri(BASE_URL)
    assert re.fullmatch(r"ui://presentation-mcp/renderer-[0-9a-f]{8}\.html", uri)
    assert theme.renderer_uri(BASE_URL) == uri

    monkeypatch.setattr(theme, "version", lambda _name: "0.0.0")
    assert theme.renderer_uri(BASE_URL) != uri


def test_the_uri_changes_with_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    uri = theme.renderer_uri(BASE_URL)

    monkeypatch.setattr(theme, "renderer_html", lambda: "<html>another page</html>")

    assert theme.renderer_uri(BASE_URL) != uri


def test_the_uri_changes_with_the_theme_origin_and_not_with_its_path() -> None:
    uri = theme.renderer_uri(BASE_URL)

    assert theme.renderer_uri("https://other.example.test/mcp/presentation") != uri
    assert theme.renderer_uri("http://example.test:8443/mcp/presentation") != uri
    assert theme.renderer_uri("https://example.test:8443/elsewhere") == uri
