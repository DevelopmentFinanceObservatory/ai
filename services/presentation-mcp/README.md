# presentation-mcp

Presentation MCP provides canvas and presentation-layer tooling over the Model Context Protocol.

## Overview

The service connects the agent loop to the platform assembler hosted by the website backend. It publishes tools for writing UI components and inspecting frame views, and loads the platform release schema dynamically on startup.

## Exposed tools

The server registers three tools when the platform assembler release is admitted.

### write_component

Creates a presentation component from an intent specification or updates an existing component with a patch.

`intent` is one of six typed members. The table lists the fields required to create each type. Revisions send only the fields to change. The live `tools/list` schema defines the shapes this release accepts.

| Type             | Required on create            | `select`                                          |
| ---------------- | ----------------------------- | ------------------------------------------------- |
| `line_chart`     | `type`, `frame_ref`, `select` | at most one dimension as an array, up to 8 values |
| `bar_chart`      | `type`, `frame_ref`, `select` | at most one dimension as an array                 |
| `table`          | `type`, `frame_ref`, `select` | any dimension may be an array                     |
| `stat_card`      | `type`, `frame_ref`, `select` | names exactly one series, no arrays               |
| `note_narration` | `type`, `text`                | none, the type carries no `frame_ref`             |
| `source_card`    | `type`, `frame_ref`           | none, the type carries no `select`                |

apis refuses a component whose `select` resolves to more than 24 series. It estimates the count over every dimension the frame declares, multiplying the array's length for an array, 1 for a single value, and the dimension's whole value count for a dimension `select` leaves unbound. A `bar_chart` arraying 24 donors over a frame that also holds two sectors, with sector left unbound, estimates 48 series and is refused.

`line_chart`, `bar_chart`, `table`, and `stat_card` also accept an optional `window` and `title`. `fact_summary.change` in the reply populates only for a `stat_card` intent that authors a `change` object with a `basis`. The other five types carry no `change` property.

The `component_id` a write returns lets any call describe that component, and no tool asks who is calling. The server sends no caller or session header to apis. Revising a component also needs its write token, a `cmw_` string. The server mints it on the create and apis returns it once, in the create's reply. `write_component` sends it to apis as the `x-dfo-write-token` header on a revise. A revise that names `component_id` without `write_token` is refused `request_invalid` before any request leaves the server, and apis refuses a wrong token `403 write_refused`, which reaches the caller as a `do_not_retry` refusal whose `fix` says to create a new component, starting from the intent `describe_component` returns. The server writes that refusal's `detail` itself, since the apis service's own names REST routes and a header the caller cannot use. The model has to keep both the id and the token. An id that names no component refuses `not_found`.

A create is retried like any other call. The server mints the token once per create, sends it in the request body on every attempt as `write_token`, and apis answers a repeat with the draft the first attempt stored, so a transport failure or a retryable 5xx such as the platform's `timeout` cannot store a second draft. One duplicate path remains: when every attempt fails, the model may call `write_component` again, which mints a fresh token and can store a second draft.

A create's result carries the token on the last line of its `content`, for every client, as `write_token: cmw_...`. When `structuredContent` is the model vocabulary, the token is also its top-level `write_token` field. The token is never in `_meta`, including the resolved revision, never in a Prefab shell, and never in a revise or `describe_component` result.

An MCP Apps host hands the whole tool result, `content` included, to the view's iframe, so a create's token reaches the page the iframe runs. That page loads Prefab's renderer from jsDelivr (see below), and the renderer loads 17 further script chunks at run time, which Subresource Integrity cannot cover. A tampered file served for the pinned prefab-ui version could read the token of the component on screen. The token revises that one component's words, series selection or note text; it cannot change the component's frame or type, and every earlier revision stays readable. This risk is accepted: it needs jsDelivr itself to serve altered files for a published npm version, which cannot change once published.

#### Parameters

- `intent`: Dictionary describing the visual component, its type, layout, and bound data references.
- `component_id`: Optional string identifying an existing component to revise.
- `write_token`: The write token the component's create returned. Required with `component_id`.
- `include_projection`: `auto` (the default), `full`, or `none`. Picks the text projection that leads the result's `content`, as described under [The resolved revision and its renderings](#the-resolved-revision-and-its-renderings).

#### Return format

Returns a dictionary containing the created or updated component payload, its assigned identifier, and revision metadata. A create also returns the component's write token, as described above.

### describe_frame

Retrieves the structural schema and sample slice of a materialised frame for presentation planning.

A `frame_ref` is the `spec_id` and `content_version` pair a data tool's reply returned. Any caller holding a `frame_ref` can use it in `describe_frame`, whichever caller minted it. `write_component` refuses a `frame_ref` an analysis-mcp verb minted, whose `spec_id` is prefixed `ops_` and whose `content_version` is prefixed `ocv_`, since those name an Analysis Toolkit op-spec.

#### Parameters

- `frame_ref`: Dictionary specifying the `spec_id` and `content_version` of the target frame.
- `dimension`: Optional string name of a specific dimension to inspect.
- `query`: Optional search query to filter dimension values.

#### Return format

Returns a dictionary detailing available dimensions, observed values, and period bounds.

### describe_component

Fetches stored component state for inspection or subsequent revision.

`component_id` must be non-empty. An empty or whitespace-only value is refused before any request reaches the platform. `revision` is the count the platform increments on every write, starting at 1 on create. Omit it to read the current revision.

#### Parameters

- `component_id`: String identifier of the component.
- `revision`: Optional integer revision number.
- `include_projection`: `auto` (the default), `full`, or `none`. Picks the text projection that leads the result's `content`, as described under [The resolved revision and its renderings](#the-resolved-revision-and-its-renderings).

#### Return format

Returns the component intent and metadata corresponding to the requested revision.

## The resolved revision and its renderings

A successful `write_component` or `describe_component` result carries the platform's resolved revision, the envelope `{schema_version, catalog_version, component_id, revision, provenance, component}`, in its `_meta` under `org.one.dfo/resolved_payload`. Hosts keep `_meta` out of model context, and the envelope appears nowhere else in the result.

Every client gets the model vocabulary as JSON in the last `content` block, led by a text projection. The text projection is bounded Markdown built from the envelope: a one-line description stamped with the component id and revision, the model-authored title marked as the model's, the body, and the freshness, warnings, and citation. The body of a `line_chart`, `bar_chart`, or `table` is a table of the visible data. A `stat_card` body states its value and change, a `note_narration` body its text, and a `source_card` body its sources. It shows at most 20 rows and 5 value columns, stays under 12 KiB for the largest payload the catalogue admits, and names what it cut and the frame that holds the rest. A projection that would pass 12 KiB, which only an envelope outside the catalogue's limits can produce, is cut after its last whole line within the bound and says so.

A create and a `describe_component` lead with the projection, and so does a revise, unless every field the patch changed is presentation-only for the component's type: `title`, `scale`, or `range` on a `line_chart` or `bar_chart`, and `title` on a `table` or `stat_card`. Those change how the component looks and none of the facts it shows, so that revise leads with one line naming what changed, such as `` scale linear to log, values unchanged (`cmp_1`, revision 3). `` A patch that changes any other field, such as `select`, `window`, a `stat_card`'s `change`, or a `note_narration`'s `text`, gets the full projection, and so does any revise of a member type the server does not know. A client that declared the `io.modelcontextprotocol/ui` extension draws the data in its app view, so under `auto` its projection leaves out the data table of a `line_chart`, `bar_chart`, or `table`, together with the line saying what the table cut. A line in its place says that `describe_component` with `include_projection` set to `full` returns the table. A client whose capabilities the server cannot see, such as a 2025-11-25 request to the stateless combined server, gets the same projection with a different line in the table's place: the server cannot tell whether the client draws views, and `describe_component` with `include_projection` set to `full` returns the table. The description, title, footer, and every other type's body stay. A client that declared capabilities without the extension gets the table. The `include_projection` argument of both tools overrides this rule. `full` forces the projection with its table for any client, and `none` leaves `content` to the model vocabulary. The harness always sends `none`, since its event stream carries the render.

Both tools are Prefab MCP Apps. Their `tools/list` entries carry `_meta.ui.resourceUri`, the URI `presentation_mcp.theme.renderer_uri(base_url)` returns, called with `PRESENTATION_PUBLIC_URL`, a resource serving Prefab's CDN renderer page. The page is a stub of about 470 bytes that loads `renderer.js` and `renderer.css` from `https://cdn.jsdelivr.net/npm/@prefecthq/prefab-ui@<installed version>/dist/app/`, pinned to the installed prefab-ui version, so a view needs jsDelivr reachable from its iframe. The URI carries a hash of the Prefab version, the resource's CSP, and the page, so a host that caches resources by URI fetches a changed page or CSP. The DFO card theme is a stylesheet the payload names by URL. Every `PrefabApp` is built by `presentation_mcp.theme.themed_app`, which sets `stylesheets` to the Google Fonts URL (Outfit 300, 400, 700, 800 and DM Mono 400, 500) and `<PRESENTATION_PUBLIC_URL>/theme-<hash>.css`, sets `mode` to `light`, and sets no `css`; the series colours are hex values in the chart and legend props. The CSS is a URL because some hosts pass `structuredContent` to the model, and 7 KB of CSS in every view would cost about 3,000 tokens of model context each. Prefab's renderer takes styling only from the payload, so the payload names the theme and the view's iframe fetches it. This server serves the CSS at `/theme-<hash>.css` beside `/healthz` (`presentation_mcp.server.theme_css`), as `text/css` with `Access-Control-Allow-Origin: *`. `<hash>` is the first 12 hex digits of the sha256 of `THEME_CSS`, so a changed theme is a new URL. The current hash is served with `Cache-Control: public, max-age=31536000, immutable`. Any other hash, from a view saved in an older chat, gets the current CSS with `Cache-Control: no-cache`. The route sits outside the MCP bearer gate, like `/healthz`, because the iframe cannot send the token. `PRESENTATION_PUBLIC_URL` is the URL at which a browser reaches this service, mount path included; under the harness that is the service's public origin plus `/mcp/presentation`. It has no default. A view renders unstyled, with no error, where a browser cannot reach that URL, so Settings construction fails when the variable is unset, blank, or not an absolute `http(s)` URL. The URL needs a host that is a hostname (letters, digits, hyphens, and dots) or a bracketed IPv6 literal, a port from 1 to 65535 if it has one, and no query, fragment, credentials, whitespace, or control character. `just dev` and the other justfile recipes that start the harness set it to `http://localhost:$PORT/mcp/presentation`, and a deployment sets its public `https` URL plus `/mcp/presentation`. The resource's `_meta.ui.csp.resourceDomains` declares the origins the page and the payload's stylesheets load from: `https://fonts.googleapis.com`, `https://fonts.gstatic.com`, the origin of `PRESENTATION_PUBLIC_URL`, and `https://cdn.jsdelivr.net`, which is what Prefab's `get_renderer_csp(mode="cdn")` lists. The listing and the read carry the same set. The renderer URI hashes that CSP, so it changes when the origin does. A client that declared the `io.modelcontextprotocol/ui` extension at `initialize` gets a Prefab shell as `structuredContent`. Some hosts pass `structuredContent` to the model, so the shell holds the component's layout and words and no data value: the title, a chart's metric and shared unit under it, the series and column labels, and a footer line naming the sources and, where the provenance carries warnings, how many caveats there are. That line opens the details: the derived description, a table's shared unit, each source with its licence and link, the freshness, every warning, and the citation. A client whose capabilities the server cannot see gets the same shell. A client that declared capabilities without the extension gets the model vocabulary there.

The values the shell binds travel in the same result's `_meta`, under `org.one.dfo/view_state` (`presentation_mcp.platform_contract.VIEW_STATE_META_KEY`): `{"data": <values>, "loaded": true}`. `data` holds `rows` for a line chart, bar chart, or table, and `value`, `unit`, and, where the card has one, `change` for a stat card, the value and change written with website's `compact()` rule. Hosts keep `_meta` from the model and hand it to the view, so no data value reaches a model through `structuredContent`. Prefab builds a view's state only from `structuredContent.state`, so the renderer page carries a short inline script before Prefab's own. On a `ui/notifications/tool-result` message whose `params._meta` holds the key as an object, the script shallow-merges that object into `params.structuredContent.state`, creating `state` if absent. The shell's own state holds the same keys empty and `loaded` false, and an `If` branch on `!loaded` draws a sentence saying the view could not read its data and that restarting the app loads the current view. A host that drops `_meta`, or draws with an older cached renderer page, shows that sentence in place of a blank chart. The sentence sits in the layout, not the state, because a host may pass `structuredContent` to a model, which would take a state string as a live error. Claude Desktop keeps a renderer page until the app restarts, whatever URI the tools name. A `note_narration` or `source_card` holds no data value, so its shell draws whole.

The view makes no tool call, so it draws in any host and any MCP session, without reading the platform again. A host such as ChatGPT that opens a new MCP session per request gets the whole view from the write's result.

A host may keep the resource URI from when its user added the app and read it after the server has moved on. The server answers every `ui://presentation-mcp/renderer-<version>.html` with the current page, as `text/html;profile=mcp-app`, with the CSP in the content's `_meta.ui.csp` (`resourceDomains`). The renderer URI hashes the page as well as the Prefab version and the CSP.

A client that declared no MCP Apps extension in its capabilities (`ViewSupport.UNDECLARED`, as Antigravity does) cannot draw the view, so a line or bar chart also reaches it as a 1000-pixel-wide `image/png` content block, after the model vocabulary and before a create's write token block (`presentation_mcp.chart_image.chart_png`). A host that declared the extension draws the view, and one whose capabilities the server cannot see (`ViewSupport.UNKNOWN`, such as any 2025-11-25 client through the stateless `/mcp/`, which sends its capabilities on initialize alone) may draw it, so neither gets the image and neither pays its tokens. `include_projection="none"` also withholds it, a log-axis chart and any other member have none, and a render that raises is logged and leaves the result without the image. The render runs in one worker process (`presentation_mcp.chart_worker`), since resvg holds the GIL and would otherwise stall the event loop every server in the harness process shares. The server's lifespan starts the worker, spawned as a fresh interpreter and warmed, and stops it, killing it if it has not exited within 5 s. Once warm, the worker's resident memory and whether it imported `harness.server` go to the log at INFO. Renders take turns on the worker, and one not done within `CHART_RENDER_TIMEOUT_S` (2 s) of its call, queue time included, leaves the result without the image and logs a warning. So does a worker that died. A render that overran kills the worker, and a dead or killed worker is replaced, and warmed, for the next call. A chart past `chart_image`'s size limits has no image: more than 8 series, 24 horizontal bars, 200 vertical bars, 2400 line points, or 1200 px of height, or value ticks outside the float range or too wide to write. Inside the limits a render took 33 to 105 ms through the worker on an Apple M5 Pro.

Prefab's charts have no logarithmic axis. A `line_chart` or `bar_chart` whose value axis is `log` draws as a table of the same rows, under a line saying the axis is logarithmic and the view cannot draw it.

The write has committed by the time the result is built, so a result never refuses over its rendering. A platform reply without an envelope succeeds with the model vocabulary alone, no `_meta`, no projection, and no view, and the server logs a warning. An envelope that fails to render succeeds the same way, with the envelope still in `_meta`, and the server logs the exception. The renderer still opens for an Apps client on these results and on refusals, so there `structuredContent` is a one-line Prefab view saying why nothing is drawn, and `content` keeps the model vocabulary or the refusal.

## Startup schema admission

During startup, the server lifespan queries the upstream platform release endpoint.

1. The service fetches the active platform release document containing JSON schemas for presentation tools.
1. The lifespan compares the release `schema_version` against `max_schema_version`.
1. Valid schemas are assembled into tool definitions and registered on the FastMCP instance.

The server answers its first request only after startup ends. What happens when the platform is unreachable or returns an incompatible schema depends on `PRESENTATION_RELEASE_REQUIRED`.

- If it is `true`, the default and what every deployment runs, an unreachable platform is retried before startup ends, first after `RELEASE_RETRY_BASE_S` and then doubling the wait up to `RELEASE_RETRY_MAX_S`. No fetch starts after `RELEASE_STARTUP_DEADLINE_S`, and a fetch still running then is cancelled. A release that answers and fails admission is not retried. Any failure fails startup with a `RuntimeError` and ends the process, so Cloud Run keeps the old revision or refuses the new instance, and every `tools/list` the server serves lists all three tools. The 90-second default deadline fits inside Cloud Run's default 240-second startup probe.
- If it is `false`, for local development and tests, the release is fetched once. On any failure the server starts with no tools and `/healthz` reports status `degraded` until the process restarts.

## Error handling and refusals

A refusal reaches the caller as an MCP error result, `isError: true`. Its `content` and `structuredContent` carry the same envelope of `error`, `retry`, and `detail` fields, plus `fix` and `pointer` when the refusal has a correction to offer, and `retry_after_s` when it names a wait. A coded platform refusal whose body carried a trace id logs it as a warning and leaves it out of the envelope.

Callers that previously received refusals as non-error results should pass `raise_on_error=False` to `client.call_tool` and read the envelope from `structuredContent`. FastMCP otherwise raises `ToolError` for an `isError` result.

Presentation MCP, data-mcp, and analysis-mcp all set `isError: true` for refusals. Presentation MCP uses `error`, `retry`, and `detail` in both response forms. data-mcp and analysis-mcp use `code` and `retry_mode` in `structuredContent` and return different text content.

- Expected validation errors return an `error` code such as `request_invalid` or `not_found`, with `detail` naming the problem and `fix` and `pointer` naming a correction where one exists.
- A revise with a missing or wrong write token returns `request_invalid` or `write_refused`, as described under [write_component](#write_component).
- Unhandled upstream failures return a `transient_unavailable` refusal.
- A publish whose frame rows this process's frame row budget (`common.row_budget`) has no room for returns `service_unavailable` with `retry_after_delay`, `retry_after_s`, and a `fix` naming the rows requested and the rows free.

## Configuration

Settings load from environment variables using `presentation_mcp.settings.Settings`.

| Environment variable             | Default                 | Description                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DFO_APIS_URL`                   | `http://localhost:8083` | Base URL for the upstream platform API                                                                                                                                                                                                                                                                                                                              |
| `UPSTREAM_CONNECT_TIMEOUT_S`     | `5.0`                   | Upstream connection timeout in seconds                                                                                                                                                                                                                                                                                                                              |
| `UPSTREAM_READ_TIMEOUT_S`        | `50.0`                  | Upstream read timeout in seconds                                                                                                                                                                                                                                                                                                                                    |
| `UPSTREAM_RETRY_BUDGET_S`        | `120.0`                 | Wall-clock retry budget for upstream requests                                                                                                                                                                                                                                                                                                                       |
| `UPSTREAM_MAX_ATTEMPTS`          | `3`                     | Maximum retry attempts for upstream calls                                                                                                                                                                                                                                                                                                                           |
| `UPSTREAM_BACKOFF_BASE_S`        | `1.0`                   | Base backoff duration in seconds                                                                                                                                                                                                                                                                                                                                    |
| `MAX_SCHEMA_VERSION`             | `1`                     | Maximum supported platform schema version                                                                                                                                                                                                                                                                                                                           |
| `PRESENTATION_RELEASE_REQUIRED`  | `true`                  | Retries an unreachable platform until `RELEASE_STARTUP_DEADLINE_S`, then fails startup. `false` fetches once and starts tool-less on failure                                                                                                                                                                                                                        |
| `RELEASE_RETRY_BASE_S`           | `1.0`                   | Wait in seconds before the first retry of a failed startup release fetch                                                                                                                                                                                                                                                                                            |
| `RELEASE_RETRY_MAX_S`            | `60.0`                  | Longest wait in seconds between retries of the startup release fetch                                                                                                                                                                                                                                                                                                |
| `RELEASE_STARTUP_DEADLINE_S`     | `90.0`                  | Seconds after which no startup release fetch starts, and a running one is cancelled                                                                                                                                                                                                                                                                                 |
| `DATASETS_FIRST_PARTY_PRINCIPAL` | unset                   | Principal sent as `x-dfo-principal` on the apis service's publish door; must match that deployment's own `DATASETS_FIRST_PARTY_PRINCIPAL`. Unset means an analysis-mcp-minted (`ops_`/`ocv_`) frame can never be published                                                                                                                                          |
| `PRESENTATION_PUBLIC_URL`        | required                | URL a browser reaches this service at, mount path included, e.g. `https://<host>/mcp/presentation`. It is the base of every view's theme stylesheet URL and the origin the renderer's CSP admits. Startup fails when it is unset or not an absolute `http(s)` URL with a hostname or IPv6 literal, a valid port, and no query, fragment, credentials, or whitespace |
| `CHART_RENDER_TIMEOUT_S`         | `2.0`                   | Seconds a result waits for its chart image from the worker process before it goes without one                                                                                                                                                                                                                                                                       |
| `CHART_WORKER_WARM`              | `true`                  | Starts the chart worker process and loads its fonts at startup. `false` starts it on the first render, which the test suites use                                                                                                                                                                                                                                    |
| `PORT`                           | `8080`                  | Port for the streamable HTTP transport                                                                                                                                                                                                                                                                                                                              |

## Running tests

Run the test suite with pytest from the workspace root or the service directory.

```bash
pytest services/presentation-mcp/tests
```
