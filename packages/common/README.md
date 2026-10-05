# common

`common` provides shared foundation modules for services in the workspace.

## Modules

The package exports shared utilities across seven areas.

| Area            | Module                   | Purpose                                                                     |
| --------------- | ------------------------ | --------------------------------------------------------------------------- |
| Logging         | `common.logging`         | Environment-aware log formatting for local development and Cloud Run        |
| Telemetry       | `common.telemetry`       | OpenTelemetry and Langfuse trace initialization                             |
| Settings        | `common.settings`        | Base pydantic settings class reading environment variables and `.env` files |
| Generic holder  | `common.holder`          | Process-wide lazy singleton holder with asynchronous cleanup                |
| Materialisation | `common.materialisation` | Content-addressed storage protocol and in-memory store for data frames      |
| Caller identity | `common.caller`          | The caller of the current MCP request, from the `x-dfo-caller` header       |
| Refusal text    | `common.refusal`         | The text of an MCP tool refusal, ending in its JSON envelope                |

## Logging

`configure_logging` initialises standard Python and Uvicorn loggers for the process.

When `K_SERVICE` is present or `LOG_FORMAT=json` is set, the module formats logs as single-line JSON records containing severity, message text, logger name, and optional stack traces. This format keeps multiline tracebacks within a single Cloud Logging record.

In local development, the module outputs plain text logs with timestamps, levels, logger names, and messages.

## Telemetry

`init_telemetry` configures OpenTelemetry tracing from environment variables and resolves one backend: `langfuse`, `otel` or `off`.

When `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` are all set, the module initialises the Langfuse client, which attaches its exporter to the global tracer provider, and the backend is `langfuse`. A deployed service needs all three. `LANGFUSE_TRACING_ENVIRONMENT`, read by the Langfuse SDK, is optional and tags traces with an environment, which separates deployed traces from local ones in a shared project. An authentication check runs on a background thread as a diagnostic: it logs a WARNING (with the traceback when it raised) if Langfuse rejects the keys or cannot be reached. The backend stays `langfuse` and startup does not wait for it. `LANGFUSE_TRACING_ENABLED=false` and `LANGFUSE_SAMPLE_RATE=0` make the SDK export nothing, so with either set the backend is `off` and no client is created. The client is built with a span filter that adds ADK's attribute-less `invocation` span to the SDK's default. That span is the parent of ADK's workflow spans, and the default filter drops it, which would leave those spans as separate roots in the Langfuse trace tree.

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set without all the Langfuse variables, the module creates a standard OpenTelemetry tracer provider with a batch OTLP HTTP span exporter, and the backend is `otel`.

With neither configured the backend is `off`. `OTEL_SDK_DISABLED=true` (compared lowercased and stripped, as the OpenTelemetry SDK does) also makes it `off` for either backend, because that SDK then hands out no-op tracers.

`backend()` returns the resolved state (`"langfuse"`, `"otel"` or `"off"`), and `init_telemetry` logs it once at startup: at INFO when tracing is on, at WARNING when it is off (naming the missing `LANGFUSE_*` variables or the variable that disables export), and at ERROR, with the traceback, when the exporter could not be constructed. The harness serves the state as `telemetry` on `/healthz` and on `/status/telemetry`.

`start_turn_span` and `end_turn_span` open and close one root span per unit of work on the active backend. The span becomes current on a task when `use_turn_span(handle)` makes it current for a `with` block, in whichever task runs the block, so a unit of work that spans several requests stays one trace. `end_turn_span` can be called from any task.

`flush_telemetry()` is an awaitable that exports the queued spans on its own thread and returns after at most three seconds. A call that arrives while a flush runs returns at once and is served by a second pass of that flush. A service calls it before a response ends when the platform throttles CPU between requests. It does nothing when the backend is `off`.

`record_event(name, level=..., metadata=...)` attaches a named event to the current trace, as a Langfuse `EVENT` observation or a span event on an OTLP backend. Outside any live span it does nothing, since Langfuse would turn the event into a one-event trace of its own. Only `str`, `int`, `float` and `bool` metadata values are kept, strings are cut to 500 characters, at most 32 entries are kept, and the call never raises. The first failure in a process logs at WARNING and later ones at DEBUG.

Third-party dependencies load lazily so telemetry initialisation stays lightweight in test environments.

## Base service settings

`BaseServiceSettings` inherits from Pydantic `BaseSettings` and defines configuration shared by all workspace services.

- `port`: Port integer for HTTP services, defaulting to 8080.
- `model_name`: Gemini model id, pinned to a specific version. `.env.example` ships the default.
- `langfuse_public_key`: Optional string for Langfuse public credentials.
- `langfuse_secret_key`: Optional string for Langfuse secret credentials.
- `langfuse_base_url`: Optional string for the Langfuse host endpoint.

The class loads values from process environment variables and `.env` files. Unknown keys are ignored.

## Caller identity

A caller names who an MCP call is made for. It is an opaque string with a kind prefix, sent in the `x-dfo-caller` request header: `conversation:` and 32 hex characters from the harness, the start of the SHA-256 of the principal and conversation id (`conversation_caller`), so neither leaves the process; `ip:`, `key:` and `user:` from the website's edge. The MCP transport session cannot serve as this identity. Under the 2026-07-28 protocol revision a server builds a connection per request, so FastMCP's `Context.session_id` changes on every call.

`current_caller()` returns the caller of the current request, or `None` when the request has no header, a blank one, more than one, or one longer than `MAX_CALLER_CHARS` (256), or when no HTTP request is in flight. A repeated header counts as none so that a client's copy cannot outvote the one a proxy sets. It never generates an identity in place of a missing one. The frame store records `current_writer()` as a stored frame's `writer`: the caller, or `anonymous` with none. `supplied_caller(caller)` sets the caller for a block, for tests that drive a server through an in-memory client, which sends no headers.

presentation-mcp leases each component to the caller that created it, so the caller decides who may revise a component. The harness serves presentation-mcp's mount to loopback clients alone, and the combined server that outside clients use leaves it out, so only the harness names a caller to presentation-mcp. The combined server logs the caller of each call. Behind the platform edge, only the website sets the header. Before it, any holder of the shared MCP bearer token can send any caller value. That is acceptable for the pilot: the token already grants every tool the caller could name itself as, the components a caller's writes touch are drafts over public data, and the combined server's request log records the caller of every call.

## Refusal text

`refusal_text(message=..., code=..., retry_mode=..., fix=..., retry_after_s=None, preamble=None)` returns the `content` text of a refusal: the message, a blank line, then `code`, `retry_mode`, `fix` and `retry_after_s` when set as one line of compact JSON. A `preamble` comes first when given, followed by a blank line. data-mcp's and analysis-mcp's `to_tool_result` and the harness's combined server all use it, so a client that shows its model only `content` reads the same envelope from each.

## Materialisation store

The `common.materialisation` package provides content-addressed storage for data frames exchanged between `data-mcp` and `analysis-mcp`.

### Frame references

`FrameRef` identifies a materialised or computed frame with two attributes.

- `spec_id`: Dataset specification identifier or operation hash matching `^(?:spec|ops)_[0-9a-f]{32}$`.
- `content_version`: Content version string matching `^(?:cv|ocv)_[0-9a-f]{32}$`.

### Store protocol

`MaterialisationStore` defines the asynchronous interface for frame persistence.

- `put(ref, frame=frame, trace=trace, writer=writer)`: Persists a frame, lineage trace, and writer identity. Idempotent: calling `put` again on an already-held reference leaves the stored frame, trace, and retention clock exactly as first written.
- `get(ref)`: Retrieves the `(Frame, StoredFrameTrace)` tuple for a reference. Returns `None` for missing keys and raises `FrameExpired` when a previously stored frame exceeds its retention window, measured from the reference's first `put`.
- `aclose()`: Closes the store and releases held connections.

### In-memory store

`InMemoryMaterialisationStore` implements `MaterialisationStore` using an `OrderedDict` keyed by `(spec_id, content_version)` tuples.

The store enforces a 30-day retention window (`FRAME_TTL`) and a default capacity limit of 256 entries (`DEFAULT_MAX_ENTRIES`). When capacity is reached, the store purges expired entries first, followed by the oldest accessed live entry.

### Store holder

`common.materialisation.holder` manages the process-wide store instance through three module functions.

- `get_store()`: Returns the process-wide store and instantiates `InMemoryMaterialisationStore` lazily on first access.
- `set_store(store)`: Replaces the current store instance.
- `aclose_store()`: Asynchronously closes and clears the active store instance.
