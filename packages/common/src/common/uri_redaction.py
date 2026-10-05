"""Cut credentials out of the URIs in text before it leaves the process.

A `dfo://artifact` URI is a bearer credential: any holder can read the
artifact it names. An `http` or `https` URL carries its credential in its
query: a signed Cloud Storage URL, an AWS presigned URL, an Azure SAS URL or an
`access_token` parameter each reads the resource until it expires.
`loggable_uri` is the rule for one URI on a log line: a `ui://` view whole, and
any other URI cut to its scheme and host.

`UriRedactingExporter` applies `redact_uris` to the text of an exported span,
since fastmcp puts a read's whole URI on its spans and trace content carries
ordinary links worth keeping. `redact_uris` finds each URI case-insensitively
and whatever precedes it, so a URI glued to other text (after a JSON `\\n`
escape, say) is still found, and cuts it by these rules:

- An `http` or `https` URL, written out (`https://` or `https:\\/\\/`) or
  percent-encoded once (`https%3A` then `%2F`, `/` or `\\/` twice), keeps its
  scheme, host and path and loses its query and fragment.
- A `dfo` URI (`dfo://`, `dfo:\\/\\/`, `dfo%3A%2F%2F` or `dfo%3A//`) is cut to
  its scheme, separator and host.
- A URI with userinfo (`user@` or `user:password@`, or `%40` in a
  percent-encoded `http` or `https` URL) before its host is cut to its scheme,
  separator and host, with the userinfo dropped. Such a URI, a database DSN
  say, is found by its `://` whatever its scheme and whatever precedes it,
  including a URI nested inside another.

Every other URI stays whole.

An `http` or `https` URL runs from its scheme to the next whitespace, `"`,
JSON-escaped quote (`\\"`), JSON-escaped newline, tab or carriage return, or the
end of the text. Every other character is inside it, including `'`, `]`, `>`
and non-ASCII letters, and so is every other JSON backslash escape (`\\/`,
`\\\\` and `\\uXXXX` for any code point, as a serialiser with `ensure_ascii`
writes `é`). Its host ends at `/`, `?`, `#`, a backslash, `%3F` or `%23`, and a
percent-encoded URL's also at `%2F`, `&`, `;` or `=`. Its query starts at the
first `?` or `#` after its host, written out, percent-encoded (`%3F`, `%23`)
or JSON-escaped (`\\u003f`, `\\u0023`), and everything from there to the
URL's end is dropped except a run of `)`, `]`, `>`, `'`, `.`, `,` or `;` at
its end, which closes a Markdown link, a quoted string or a sentence. So the
path kept holds no query in any of those forms, and text glued after a URL
that has a query is dropped with it.

A `dfo` URI or a URI with userinfo ends at whitespace, a quote, a backslash
other than the JSON escapes `\\/`, `\\u0026` (`&`), `\\u003d` (`=`),
`\\u003f` (`?`) and `\\u0023` (`#`), or a closing `)`, `]` or `>`. Its host
ends at `/`, `?`, `#` or a backslash. A percent-encoded `dfo` URI ends at `&`
or `;`, the end of the query value that carries it, and its host also ends at
`%`, `=`, `&` or `;`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import override

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status
from opentelemetry.util.types import Attributes, AttributeValue

# A written-out URI after its scheme: `://` (or `:\/\/`), the authority, and the rest.
_SEPARATOR = r"(?P<separator>:(?://|\\/\\/))"
_AUTHORITY = r"(?P<authority>[^\s'\"\\)\]>/?#]*)"
_JSON_ESCAPES = r"\\(?:/|u0026|u003[dDfF]|u0023)"
_REST = rf"(?P<rest>(?:[^\s'\"\\)\]>]|{_JSON_ESCAPES})*)"
# A percent-encoded `dfo` URI after its scheme.
_ENCODED_SEPARATOR = r"(?P<separator>%3A(?:%2F|/){2})"
_DFO_URI = re.compile(r"dfo" + _SEPARATOR + _AUTHORITY + _REST, re.IGNORECASE)
_ENCODED_DFO_URI = re.compile(
    r"dfo"
    + _ENCODED_SEPARATOR
    + r"(?P<authority>[^\s'\"\\)\]>%&;=/?#]*)(?P<rest>[^\s'\"\\)\]>&;#]*)",
    re.IGNORECASE,
)
# An `http` or `https` URL after its host: a path that runs to the first query or
# fragment start in any form, then `query`, which runs to the URL's end. Each
# consumes a backslash with the character after it, so `\\"` ends the URL at
# its `"`.
_HTTP_PATH_AND_QUERY = (
    r"(?:[^\s\"\\?#%]|%(?!3f|23)|\\(?![nrt]|u003f|u0023)[^\s\"])*"
    r"(?P<query>(?:[?#]|%3f|%23|\\u003f|\\u0023)(?:[^\s\"\\]|\\(?![nrt])[^\s\"])*)?"
)
_HTTP_URL = re.compile(
    r"https?" + _SEPARATOR + r"(?P<authority>(?:[^\s\"\\/?#%]|%(?!3f|23))*)" + _HTTP_PATH_AND_QUERY,
    re.IGNORECASE,
)
_ENCODED_HTTP_URL = re.compile(
    r"https?"
    + r"(?P<separator>%3A(?:%2F|/|\\/){2})"
    + r"(?P<authority>(?:[^\s\"\\/?#%&;=]|%(?!2f|3f|23))*)"
    + _HTTP_PATH_AND_QUERY,
    re.IGNORECASE,
)
# Starts at the separator and needs an `@` before the authority ends, so the text
# before `://` is left alone, a failed try consumes nothing, and the scan stays linear.
_USERINFO_URI = re.compile(
    _SEPARATOR + r"(?P<authority>[^\s'\"\\)\]>/?#@]*@[^\s'\"\\)\]>/?#]*)" + _REST
)
# The punctuation that can close the text around a URL: a Markdown link, a
# bracketed list, an angle-bracketed URI, a quoted string or a sentence.
_URL_CLOSING = ")]>'.,;"
# Where userinfo ends in an authority, written out or percent-encoded.
_USERINFO_END = re.compile(r"@|%40")
# Where a URI's authority ends (RFC 3986).
_AUTHORITY_END = re.compile(r"[/?#]")


def loggable_uri(uri: str) -> str:
    """`uri` whole for a `ui://` view, and its scheme and host alone otherwise.

    Splits the string itself, so a malformed URI raises nothing, and one with
    no `://` comes back as an empty string. The host ends at the first `/`, `?`
    or `#`, so a query or fragment is dropped even with no path. Userinfo before
    the host is dropped.
    """
    scheme, separator, rest = uri.partition("://")
    if not separator:
        return ""
    if scheme.lower() == "ui":
        return uri
    host = _AUTHORITY_END.split(rest, maxsplit=1)[0].rpartition("@")[2]
    return f"{scheme}{separator}{host}"


def redact_uris(text: str) -> str:
    """`text` with the credentials in its URIs cut out, by the rules the module
    docstring states."""
    text = _DFO_URI.sub(_to_host, text)
    text = _ENCODED_DFO_URI.sub(_to_host, text)
    text = _HTTP_URL.sub(_to_path, text)
    text = _ENCODED_HTTP_URL.sub(_to_path, text)
    return _USERINFO_URI.sub(_to_host, text)


def _to_host(match: re.Match[str]) -> str:
    """The match cut to the text before its authority (scheme, if the pattern
    has one, and separator) and its host."""
    uri = match.group()
    head = uri[: match.start("authority") - match.start()]
    return head + _USERINFO_END.split(match["authority"])[-1]


def _to_path(match: re.Match[str]) -> str:
    """The `http` or `https` URL cut to its scheme and host if it has userinfo,
    and otherwise with its query and fragment dropped. Of the text dropped, only
    the run of closing punctuation that ends it stays."""
    uri = match.group()
    start = match.start()
    if _USERINFO_END.search(match["authority"]):
        host = _USERINFO_END.split(match["authority"])[-1]
        kept = uri[: match.start("authority") - start] + host
        dropped = uri[match.end("authority") - start :]
    elif match["query"] is not None:
        kept = uri[: match.start("query") - start]
        dropped = match["query"]
    else:
        return uri
    return kept + dropped[len(dropped.rstrip(_URL_CLOSING)) :]


class UriRedactingExporter(SpanExporter):
    """Hands `exporter` each span with every credential URI in its name,
    status description, attributes and event attributes cut by `redact_uris`.

    Redacts at export because fastmcp adds a read's URI to its request span
    after the span starts, which a span processor's `on_start` never sees, and
    an ended span is read-only.
    """

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([_redacted(span) for span in spans])

    @override
    def shutdown(self) -> None:
        self._exporter.shutdown()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


def _redacted(span: ReadableSpan) -> ReadableSpan:
    status = span.status
    return ReadableSpan(
        name=redact_uris(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_redacted_attributes(span.attributes),
        events=[
            Event(event.name, _redacted_attributes(event.attributes), event.timestamp)
            for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        status=Status(
            status.status_code,
            redact_uris(status.description) if status.description else status.description,
        ),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _redacted_attributes(attributes: Attributes) -> Attributes:
    if not attributes:
        return attributes
    return {key: _redacted_value(value) for key, value in attributes.items()}


def _redacted_value(value: AttributeValue) -> AttributeValue:
    if isinstance(value, str):
        return redact_uris(value)
    if isinstance(value, Sequence) and all(isinstance(item, str) for item in value):
        return tuple(redact_uris(str(item)) for item in value)
    return value
