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

When `K_SERVICE` is present or `LOG_FORMAT=json` is set, the module formats logs as single-line JSON records containing severity, message text, logger name, and optional stack traces. This format keeps multiline tracebacks within a single Cloud Logging record. A record can carry extra fields as a mapping under the `JSON_FIELDS` attribute, passed as `extra={JSON_FIELDS: {...}}`. Its entries are written beside the formatter's own keys (`severity`, `message`, `logger`, `stack_trace`), which win a clash. Each key, at any depth of nested mappings and lists, is written as its `str`, as is any value JSON has no form for, such as a datetime or a set, so the record still logs.

In local development, the module outputs plain text logs with timestamps, levels, logger names, and messages.

## Telemetry

`init_telemetry` configures OpenTelemetry tracing from environment variables and resolves one backend: `langfuse`, `otel` or `off`.

When `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` are all set, the module initialises the Langfuse client, which attaches its exporter to the global tracer provider, and the backend is `langfuse`. A deployed service needs all three. `LANGFUSE_TRACING_ENVIRONMENT`, read by the Langfuse SDK, is optional and tags traces with an environment, which separates deployed traces from local ones in a shared project. An authentication check runs on a background thread as a diagnostic: it logs a WARNING (with the traceback when it raised) if Langfuse rejects the keys or cannot be reached. The backend stays `langfuse` and startup does not wait for it. `LANGFUSE_TRACING_ENABLED=false` and `LANGFUSE_SAMPLE_RATE=0` make the SDK export nothing, so with either set the backend is `off` and no client is created. The client keeps the SDK's default span filter, which exports every span the planner and orchestrator agents write, since each carries `gen_ai.*` attributes. `init_telemetry` also sets `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` when the environment leaves it unset, so the agents' model-call and tool spans carry prompts, responses and tool content.

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set without all the Langfuse variables, the module creates a standard OpenTelemetry tracer provider with a batch OTLP HTTP span exporter, and the backend is `otel`.

With neither configured the backend is `off`. `OTEL_SDK_DISABLED=true` (compared lowercased and stripped, as the OpenTelemetry SDK does) also makes it `off` for either backend, because that SDK then hands out no-op tracers.

`backend()` returns the resolved state (`"langfuse"`, `"otel"` or `"off"`), and `init_telemetry` logs it once at startup: at INFO when tracing is on, at WARNING when it is off (naming the missing `LANGFUSE_*` variables or the variable that disables export), and at ERROR, with the traceback, when the exporter could not be constructed. The harness serves the state as `telemetry` on `/healthz` and on `/status/telemetry`.

`turn_segment(external_id, session_id=..., user_id=..., tags=...)` is a context manager that runs its block inside one `turn.segment` span, opened on entry and ended on exit, so no span outlives the request that opened it. Every span opened in the block, on the task or on a task started inside it, nests under the segment. The segment carries no input or output. A block that raises ends it at level `ERROR` with the exception's class name as its status message. A cancel or a close (`CancelledError`, `GeneratorExit`) ends it at the default level, and the unit of work's outcome carries the `cancelled` code. `start_turn_segment`, `use_turn_segment(handle)` and `end_turn_segment(handle, status=..., level=..., attributes=...)` open, enter and end the same `turn.segment` span from different places, for a caller whose segment outlives one block: the harness opens a segment with its claim and ends it once the segment's stream has ended, with the stream's ending as its status message, a level of its own choosing and attributes such as `dfo.stored`. `root_if_span_ended(name)` runs a block inside a short-lived root `name` on the turn's trace when the enclosing `turn_segment`'s span has ended, as for a task started in a segment and still running after it, and yields whether it did, so the caller flushes. While that span is live the block runs as it is, and on OTLP `record_event` puts an event made under an inner span that has ended on the live segment. `record_turn_outcome(external_id, ..., request=..., response=..., error=...)` writes how the unit of work ended as one zero-duration `turn` observation, opened and ended in the call: on Langfuse its input is `request`, its output `response`, and an `error=` code sets its level to `ERROR` with the code as its status message, or to `WARNING` for `cancelled` and `abandoned` (`WARNING_TURN_ERRORS`), which no failure ended. With `paused=True` it records a pause for the user in place of a close, as a `turn.pause` observation (`TURN_PAUSE_NAME`), so only a close is a `turn`. Its metadata holds `dfo.external_id`, `dfo.turn_error` when there is an error, and `dfo.turn_status` `paused` on a pause. Each call writes its own observation, so a unit of work that paused before it closed holds one for each. Segments and outcomes of one `external_id` share one trace, whose id is `Langfuse.create_trace_id(seed=external_id)` on Langfuse and the same sha256 prefix on plain OTLP. On Langfuse each is a root of that trace. On plain OTLP each, and each `root_if_span_ended` root, is a child of the turn's synthetic parent: a remote span whose id is the first 8 bytes of the trace id's sha256, so every segment, outcome and early-mint root of a turn shares the one parent. That synthetic parent is marked sampled, so on the OTLP backend a parent-based sampler always samples a turn's spans. Read a turn's input and output from the `turn` observation in `GET /api/public/v2/observations?traceId=...`. On Langfuse every span of the turn carries the trace name `turn` and the session, `user_id` and `tags` it was given, and `turn_call_meta()` returns the same three as MCP request `_meta` entries for a call made inside a segment. On Langfuse `deploy_release()`, which is `LANGFUSE_RELEASE`, else `K_REVISION`, else `None`, is the client's `release`, so every exported span names the revision that produced it. With neither variable set the Langfuse SDK may take a release from a CI variable such as `GITHUB_SHA`. The exporter sends `x-langfuse-ingestion-version: 4`, which Langfuse v4 needs to process the spans in real time. The Langfuse client exports what the SDK's default filter keeps (its own spans and every span with a `gen_ai.*` attribute) and the `mcp` SDK's `MCP send tools/call` span (scope `mcp-python-sdk`). `tools/call` is the one method whose server span passes the default filter, so the other methods' `MCP send` spans are dropped with their server spans. The `mcp` SDK opens an `MCP send <method>` CLIENT span for each request a client sends and names it as the parent in the request's `_meta` traceparent, so an MCP server's `tools/call` span hangs off it. Dropped, it would leave every tool call's server span with a parent missing from the trace.

`flush_telemetry()` is an awaitable that exports the queued spans on its own thread and returns after at most three seconds. A call that arrives while a flush runs waits, within the same bound, for that flush, whose second pass serves it. A service calls it before a response ends when the platform throttles CPU between requests. It does nothing when the backend is `off`. `start_telemetry_flush()` starts the same flush and returns without waiting, for code that must not await, such as a cancelled task's unwind.

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

`current_caller()` returns the caller of the current request, or `None` when the request has no header, a blank one, more than one, or one longer than `MAX_CALLER_CHARS` (256), or when no HTTP request is in flight. A repeated header counts as none so that a client's copy cannot outvote the one a proxy sets. It never generates an identity in place of a missing one. The frame store records `current_writer()` as a stored frame's `writer`: the caller, or `anonymous` with none. `supplied_caller(caller)` sets the caller for a block, for tests that drive a server through an in-memory client, which sends no headers. `set_supplied_caller(caller)` sets it for the rest of the current context, for code that calls dfo-apis outside any MCP request.

A Studio turn names itself in the `x-dfo-turn` header: its wire trace id, `tr_` and 16 hex characters. `current_turn()` reads it as `current_caller()` reads the caller, and counts a repeated header or any other form as none. `set_supplied_turn(turn)` stands in for it outside an MCP request. `vouched_caller_headers` forwards the turn to dfo-apis beside a `conversation:` caller only, so dfo-apis logs each delivery with the turn that caused it.

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
