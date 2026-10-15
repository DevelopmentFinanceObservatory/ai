"""The website Studio card look, carried by every view's payload by URL.

Prefab's renderer takes its styling only from the payload's `css` and
`stylesheets`, and some hosts (ChatGPT) pass a tool result's
`structuredContent` to the model, so 7 KB of inline CSS would cost model
context on every view. `themed_app` therefore builds each `PrefabApp` with
`stylesheets=[FONTS_URL, theme_url(base_url)]` and no `css`: the payload
names the theme and the renderer's iframe fetches it.

presentation-mcp serves the CSS itself, at `/theme-<THEME_HASH>.css` beside
`/healthz` (`server.theme_css`). `THEME_HASH` is a short sha256 of
`THEME_CSS`, so a changed theme is a new URL and the current URL is cached
for a year. A view saved in an old chat still names an old hash. The route
answers it with the current CSS and `no-cache`. The route is open without
the MCP bearer token, since the iframe cannot send one. `base_url` is the
address at which the service's routes are reachable from a browser
(`Settings.presentation_public_url`). Its origin joins the font origins in
the renderer's CSP (`renderer_csp`).

The renderer resource is Prefab's CDN page, `renderer_html()`: a stub of about
470 bytes that loads `renderer.js` and `renderer.css` from jsDelivr, pinned to
the installed prefab-ui version, plus a short inline script that merges the
data a result carries in its `_meta` into the view's state. A view therefore
needs jsDelivr reachable from its iframe. `renderer_csp` declares that origin, from Prefab's own
`get_renderer_csp(mode="cdn")`, beside the font and theme origins.

Prefab's own CSS is Tailwind v4 in `@layer`s, so the unlayered `THEME_CSS`
wins over every layered rule. Its selectors stay on Prefab's theme tokens
(`:root`), Recharts' public `recharts-*` class names, and the `dfo-*` classes
`app_view` sets through `css_class`. The series palette is Python
(`SERIES_COLOURS`, `SERIES_COLOURS_LIGHT`): `app_view` writes the hex values
into each chart series and legend swatch. Views force light mode, so there is
no dark variant.

Colours, sizes and weights are website's Default Studio theme
(`src/lib/studio/theme.ts`, `Card.module.scss`, `CardChrome.module.scss`,
`ChartCard.module.scss`, `chartSetup.ts`). The info icon and the DFO logo are
the website's `src/assets/svg/info.svg` and `logo.svg`, inlined as `data:`
masks, which the MCP Apps default CSP admits under `img-src`, so they cost no
request.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from importlib.metadata import version
from string import Template
from typing import Any
from urllib.parse import urlsplit

from prefab_ui.app import PrefabApp
from prefab_ui.components import Component
from prefab_ui.renderer import get_renderer_csp, get_renderer_html

from presentation_mcp.platform_contract import VIEW_STATE_META_KEY

_FONT_ORIGINS = ["https://fonts.googleapis.com", "https://fonts.gstatic.com"]

# Outfit and DM Mono in the weights the style below uses. `Theme(font=...)`
# would load Outfit at 400-700 only, so the URL is set by hand.
FONTS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Outfit:wght@300;400;700;800"
    "&family=DM+Mono:wght@400;500"
    "&display=swap"
)

# The website's default series palette, in series order, and each colour's
# lighter half for the legend's checkered swatch.
SERIES_COLOURS = ("#6467de", "#f6935a", "#f46f58", "#bb6fdc", "#662d6e", "#463138")
SERIES_COLOURS_LIGHT = ("#8285e7", "#f9b170", "#f78f76", "#cf94e8", "#7f4487", "#5e474f")

_INFO_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0.91 0.91 8.18 8.18">'
    '<g stroke="currentColor" stroke-linecap="square" stroke-linejoin="round">'
    '<path d="M5 6.69229L5 5"/><path d="M4.99414 3.41435L4.99414 3.46204"/>'
    '<circle cx="4.99924" cy="5" r="3.58713"/></g></svg>'
)

# Drawn by the theme's "Built with" mark and by `chart_image`'s footer.
DFO_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 58 22">'
    '<path fill="currentColor" d="M.394 21.395V.413h8.841c6.572 0 11.375 4.125 11.375 10.49 '
    "0 6.337-4.803 10.492-11.316 10.492zM9.235 5.422H6.288v10.963h2.947c3.3 0 5.364-1.827 "
    '5.364-5.51 0-3.655-2.093-5.453-5.364-5.453"/>'
    '<path fill="currentColor" d="M20.362 21.395V.413h15.53v4.892h-9.636v3.742h9.195v4.892h'
    '-9.195v7.456z"/>'
    '<path fill="currentColor" d="M46.012 21.807c-6.571 0-11.434-4.685-11.434-10.933C34.578 '
    "4.627 39.382 0 45.953 0 52.525 0 57.33 4.627 57.33 10.904c0 6.247-4.804 10.903-11.317 "
    "10.903m-.059-5.186c3.301 0 5.364-2.24 5.364-5.747 0-3.477-2.092-5.687-5.364-5.687-3.3 "
    '0-5.363 2.21-5.363 5.687 0 3.507 2.063 5.747 5.363 5.747"/></svg>'
)


def _svg_url(svg: str) -> str:
    return f'url("data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}")'


THEME_CSS = Template("""
:root {
  --font-sans: "Outfit", system-ui, sans-serif;
  --font-mono: "DM Mono", ui-monospace, monospace;
  --background: #ffffff;
  --foreground: #484842;
  --dfo-line: #ececdb;
  --dfo-muted: #5e5e52;
  --dfo-caption: #abb4b7;
}

.recharts-cartesian-grid-vertical,
.recharts-cartesian-axis-line,
.recharts-cartesian-axis-tick-line {
  display: none;
}

/* Prefab's chart injects `[data-chart] .recharts-cartesian-grid
   line[stroke='#ccc']` into the body at render time, later in the document
   and more specific than any scoped selector here. */
.recharts-cartesian-grid-horizontal line {
  stroke: var(--dfo-line) !important;
}

.recharts-cartesian-axis-tick-value {
  font-family: var(--font-sans);
  font-size: 8px;
  font-weight: 700;
  fill: var(--foreground);
}

/* The website's legend, above the plot. `app_view` sets each swatch's
   --series-dark and --series-light to its series' pair. */
.dfo-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.dfo-legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-family: var(--font-sans);
  font-size: 10px;
  font-weight: 400;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--foreground);
}

.dfo-legend-swatch {
  flex-shrink: 0;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background-image: repeating-conic-gradient(
    var(--series-light) 0% 25%,
    var(--series-dark) 0% 50%
  );
  background-size: 6px 6px;
}

.dfo-card {
  gap: 16px;
  padding: 16px;
  background-color: var(--background);
  color: var(--foreground);
}

/* The title and, on a chart, the caption naming its metric and unit, held
   closer together than the card's 16px gap. */
.dfo-head {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.dfo-caption {
  font-family: var(--font-sans);
  font-size: 10px;
  font-weight: 400;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--dfo-muted);
}

.dfo-title {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 300;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--dfo-muted);
}

/* The footer row. The accordion's trigger is the source line, and
   "Built with" sits at the right end of that line, clear of the details
   that open below it. A footer without the accordion keeps the line's
   height, so the mark stays inside the card. */
.dfo-chrome {
  position: relative;
  min-height: calc(12px * 1.3);
}

.dfo-more .pf-accordion-trigger {
  align-items: center;
  justify-content: flex-start;
  gap: 4px;
  padding: 0 80px 0 0;
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 300;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--dfo-caption);
  cursor: help;
  background: none;
  transition: color 0.2s ease;
}

.dfo-more .pf-accordion-trigger:hover {
  color: var(--dfo-muted);
  background: none;
}

.dfo-more .pf-accordion-trigger::before {
  content: "";
  flex-shrink: 0;
  width: 10px;
  height: 10px;
  background-color: currentColor;
  -webkit-mask: $info_icon center / contain no-repeat;
  mask: $info_icon center / contain no-repeat;
}

/* Prefab's chevron; the info icon marks the line as the way in. */
.dfo-more .pf-accordion-trigger > svg {
  display: none;
}

.dfo-more .pf-accordion-content-inner {
  padding-top: 8px;
}

.dfo-built-with {
  position: absolute;
  top: calc((12px * 1.3 - 26px * 22 / 58) / 2);
  right: 0;
  display: flex;
  align-items: flex-end;
  gap: 4px;
  font-family: var(--font-sans);
  font-size: 8px;
  font-weight: 300;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--dfo-caption);
  white-space: nowrap;
}

.dfo-built-with::after {
  content: "";
  flex-shrink: 0;
  width: 26px;
  aspect-ratio: 58 / 22;
  background-color: currentColor;
  -webkit-mask: $dfo_logo center / contain no-repeat;
  mask: $dfo_logo center / contain no-repeat;
}

/* The panel's lines are Prefab Muted/Small/Link components whose own text
   classes would beat inherited type, so the rule reaches each line. */
.dfo-details,
.dfo-details * {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 400;
  line-height: 1.4;
  letter-spacing: -0.02em;
  color: var(--foreground);
  overflow-wrap: anywhere;
}

.dfo-stat-value,
.dfo-stat-unit {
  font-family: var(--font-sans);
  font-size: 28px;
  line-height: 0.95;
  letter-spacing: -0.05em;
}

.dfo-stat-value {
  font-weight: 800;
}

/* The website sets 200; 300 is the lightest Outfit weight loaded. */
.dfo-stat-unit {
  font-weight: 300;
}

/* The space between amount and unit. Prefab trims a bound string, so the
   space cannot ride in the unit's text. */
.dfo-stat-unit::before {
  content: " ";
}
""").substitute(info_icon=_svg_url(_INFO_ICON), dfo_logo=_svg_url(DFO_LOGO))

THEME_HASH = hashlib.sha256(THEME_CSS.encode()).hexdigest()[:12]


def theme_url(base_url: str) -> str:
    """The URL the theme stylesheet is served from under `base_url`."""
    return f"{base_url.rstrip('/')}/theme-{THEME_HASH}.css"


def _origin(url: str) -> str:
    """`url`'s scheme, host and explicit port, as a CSP host source."""
    parts = urlsplit(url)
    host = f"[{parts.hostname}]" if ":" in (parts.hostname or "") else parts.hostname
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}"


def renderer_csp(base_url: str) -> dict[str, list[str]]:
    """The renderer resource's CSP: the origins its page and the payload's
    stylesheets load from, keyed as `prefab_ui.renderer.get_renderer_csp` keys
    them. They are the font hosts, the theme's origin under `base_url`, and
    the origin Prefab's CDN page needs (jsDelivr). `fastmcp.apps.ResourceCSP`
    accepts the dict as keyword arguments.
    """
    cdn = get_renderer_csp(mode="cdn")["resource_domains"]
    origins = [*_FONT_ORIGINS, _origin(base_url), *cdn]
    return {"resource_domains": list(dict.fromkeys(origins))}


def themed_app(view: Component, *, base_url: str, state: dict[str, Any] | None = None) -> PrefabApp:
    """A light-mode `PrefabApp` of `view` naming the theme's fonts and
    stylesheet by URL, the stylesheet under `base_url`.

    Every view is built here, so none can reach a renderer without its styling.
    """
    return PrefabApp(
        view=view,
        state=state,
        stylesheets=[FONTS_URL, theme_url(base_url)],
        mode="light",
    )


# Prefab's renderer builds a view's state from `structuredContent.state` alone.
# This listener runs before Prefab's module script, so it sees each
# `ui/notifications/tool-result` message first and merges the result's
# `_meta[VIEW_STATE_META_KEY]` into that state in place. A host that keeps
# `_meta` from the view leaves the state as the shell wrote it. Like Prefab's
# own bridge transport, it ignores a message whose source is not the host window.
_VIEW_STATE_SCRIPT = Template("""<script>
window.addEventListener("message", function (event) {
  if (event.source !== window.parent) return;
  var message = event.data;
  if (!message || message.method !== "ui/notifications/tool-result") return;
  var params = message.params;
  var content = params && params.structuredContent;
  var incoming = params && params._meta && params._meta[$key];
  if (!content || !incoming || typeof incoming !== "object" || Array.isArray(incoming)) return;
  var state = content.state || (content.state = {});
  for (var name in incoming) {
    if (Object.prototype.hasOwnProperty.call(incoming, name)) state[name] = incoming[name];
  }
});
</script>
""").substitute(key=json.dumps(VIEW_STATE_META_KEY))


@functools.cache
def renderer_html() -> str:
    """Prefab's CDN renderer page, which loads `renderer.js` and `renderer.css`
    from jsDelivr at the installed prefab-ui version, with `_VIEW_STATE_SCRIPT`
    before `</head>`, ahead of Prefab's module script."""
    page = get_renderer_html(mode="cdn")
    if "</head>" not in page:
        raise ValueError("Prefab's renderer page has no </head> to place the view-state script in")
    return page.replace("</head>", _VIEW_STATE_SCRIPT + "</head>", 1)


def renderer_uri(base_url: str) -> str:
    """The renderer resource's URI, named by a hash of Prefab's version,
    `renderer_csp(base_url)`, and the page `renderer_html` returns, so a host
    that caches UI resources by URI fetches a changed page or CSP.
    """
    csp = json.dumps(renderer_csp(base_url), sort_keys=True)
    content = f"{version('prefab-ui')}\n{csp}\n{renderer_html()}".encode()
    return f"ui://presentation-mcp/renderer-{hashlib.sha256(content).hexdigest()[:8]}.html"
