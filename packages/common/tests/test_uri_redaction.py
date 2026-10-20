"""`common.uri_redaction`: credentials cut out of the URIs in exported spans."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import pytest
from common import telemetry
from common.uri_redaction import UriRedactingExporter, loggable_uri, redact_uris
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

_ARTIFACT = f"dfo://artifact/spec_{'1' * 32}/cv_2/sha256:{'3' * 64}"
_SECRET = _ARTIFACT.removeprefix("dfo://artifact")
# A V4 signed URL of the shape apis mints, and a V2 one.
_SIGNED_PATH = "https://storage.googleapis.com/dfo-artifacts/working/abc.csv"
_SIGNED_V4 = (
    f"{_SIGNED_PATH}?X-Goog-Algorithm=GOOG4-RSA-SHA256"
    "&X-Goog-Credential=sa%40proj.iam.gserviceaccount.com%2F20260930%2Fauto%2Fstorage%2Fgoog4_request"
    "&X-Goog-Date=20260930T000000Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=deadbeefcafebabe"
)
_SIGNED_V2 = (
    f"{_SIGNED_PATH}?GoogleAccessId=sa%40proj.iam.gserviceaccount.com&Expires=1790000000"
    "&Signature=abc%2Fdef%3D"
)
# An AWS presigned URL, an Azure SAS URL and an OAuth access-token URL.
_AWS_PATH = "https://dfo-exports.s3.eu-west-1.amazonaws.com/working/abc.csv"
_AWS_PRESIGNED = (
    f"{_AWS_PATH}?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260930%2Feu-west-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260930T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=0123456789abcdef"
)
_AZURE_PATH = "https://dfoexports.blob.core.windows.net/working/abc.csv"
_AZURE_SAS = (
    f"{_AZURE_PATH}?sv=2022-11-02&sp=r&sr=b&se=2026-10-01T00%3A00%3A00Z&sig=c2VjcmV0c2lnbmF0dXJl%3D"
)
_TOKEN_PATH = "https://api.example.test/v1/files/abc.csv"
_ACCESS_TOKEN_URL = f"{_TOKEN_PATH}?access_token=ya29.SECRETTOKEN"
_ORDINARY_PATH = "https://www.oecd.org/en/data/indicators/official-development-assistance-oda.html"
_ORDINARY = f"{_ORDINARY_PATH}?oecdcontrol-8027380c62-var3=2023#chart"


@pytest.mark.parametrize(
    ("uri", "logged"),
    [
        (_ARTIFACT, "dfo://artifact"),
        ("ui://presentation/view.html", "ui://presentation/view.html"),
        ("dfo://user:pw@artifact/a/b", "dfo://artifact"),
        ("dfo://[x/y", "dfo://[x"),
        ("artifact/spec_1", ""),
        ("https://example.test?token=secret", "https://example.test"),
        ("https://example.test#secret", "https://example.test"),
        ("https://example.test?token=secret#fragment", "https://example.test"),
        ("https://user:pw@example.test?token=secret", "https://example.test"),
        ("https://user:pw@example.test#secret", "https://example.test"),
    ],
)
def test_loggable_uri_keeps_a_view_whole_and_cuts_any_other_uri_to_its_host(
    uri: str, logged: str
) -> None:
    assert loggable_uri(uri) == logged


def test_redact_uris_cuts_each_uri_in_free_text() -> None:
    text = f"delegate {_ARTIFACT} and Unknown resource: '{_ARTIFACT}' from ui://v/p"
    assert redact_uris(text) == (
        "delegate dfo://artifact and Unknown resource: 'dfo://artifact' from ui://v/p"
    )


@pytest.mark.parametrize(
    ("url", "cut"),
    [
        (_SIGNED_V4, _SIGNED_PATH),
        (_SIGNED_V2, _SIGNED_PATH),
        (
            "https://storage.googleapis.com/b/o?x-goog-signature=deadbeef",
            "https://storage.googleapis.com/b/o",
        ),
        (
            "https://storage.googleapis.com/b/o?a=1&X-GOOG-CREDENTIAL=sa",
            "https://storage.googleapis.com/b/o",
        ),
        ("http://storage.test/b/o?X-Goog-Algorithm=GOOG4-RSA-SHA256", "http://storage.test/b/o"),
        (
            "https://storage.googleapis.com/b/o?googleaccessid=sa#frag",
            "https://storage.googleapis.com/b/o",
        ),
        (
            "https://storage.googleapis.com/b/o?Expires=1&signature=abc",
            "https://storage.googleapis.com/b/o",
        ),
        (
            "https://storage.googleapis.com?X%2DGoog%2DSignature=deadbeef",
            "https://storage.googleapis.com",
        ),
    ],
)
def test_redact_uris_drops_the_query_of_a_signed_url(url: str, cut: str) -> None:
    assert redact_uris(f"fetch {url} now") == f"fetch {cut} now"


# A URL in each form a span carries one: written out in free text, in a JSON
# string with `/`, `&` and `=` escaped, and percent-encoded as the last query
# value in a request line.
_FORMS = [
    pytest.param(lambda url: f"fetch {url} now", id="written"),
    pytest.param(
        lambda url: (
            json.dumps({"url": url})
            .replace("/", "\\/")
            .replace("&", "\\u0026")
            .replace("=", "\\u003d")
        ),
        id="json-escaped",
    ),
    pytest.param(lambda url: f"GET /r?u={quote(url, safe='')} HTTP/1.1", id="percent-encoded"),
]


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize(
    ("url", "path", "secrets"),
    [
        pytest.param(
            _AWS_PRESIGNED,
            _AWS_PATH,
            ("0123456789abcdef", "AKIAIOSFODNN7EXAMPLE"),
            id="aws-presigned",
        ),
        pytest.param(_AZURE_SAS, _AZURE_PATH, ("c2VjcmV0c2lnbmF0dXJl",), id="azure-sas"),
        pytest.param(_ACCESS_TOKEN_URL, _TOKEN_PATH, ("SECRETTOKEN",), id="access-token"),
    ],
)
def test_redact_uris_keeps_the_path_of_a_credential_url_and_drops_its_query(
    form: Callable[[str], str], url: str, path: str, secrets: tuple[str, ...]
) -> None:
    redacted = redact_uris(form(url))

    assert [secret for secret in secrets if secret in redacted] == []
    assert redacted == form(path)


@pytest.mark.parametrize(
    ("uri", "cut"),
    [
        (_ORDINARY, _ORDINARY_PATH),
        ("https://example.test/path?token=secret&signed=1", "https://example.test/path"),
        (
            "https://example.test/x-goog-signature/p?q=1#Signature=a",
            "https://example.test/x-goog-signature/p",
        ),
        ("https://example.test/people/@handle?ref=1", "https://example.test/people/@handle"),
        (
            "[x](https://www.oecd.org/p(1)?a=1) and (https://example.test/b)",
            "[x](https://www.oecd.org/p(1)) and (https://example.test/b)",
        ),
        ("https://example.test/p?mysignature=1&x-goog-signatures=2", "https://example.test/p"),
        ("https://example.test?token=secret", "https://example.test"),
        ("https://example.test#access_token=secret", "https://example.test"),
    ],
)
def test_redact_uris_drops_the_query_and_fragment_of_any_http_url(uri: str, cut: str) -> None:
    assert redact_uris(f"see {uri} now") == f"see {cut} now"


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            "https://proxy.test/fetch%3Ftoken%3Dsecret now",
            "https://proxy.test/fetch now",
            id="encoded-question-mark",
        ),
        pytest.param(
            "https://proxy.test/fetch%23access_token%3Dsecret now",
            "https://proxy.test/fetch now",
            id="encoded-hash",
        ),
        pytest.param(
            '"https:\\/\\/h.test\\/p\\u003ftoken\\u003dsecret"',
            '"https:\\/\\/h.test\\/p"',
            id="json-escaped-question-mark",
        ),
        pytest.param(
            '"https:\\/\\/h.test\\/p\\u0023access_token\\u003dsecret"',
            '"https:\\/\\/h.test\\/p"',
            id="json-escaped-hash",
        ),
        pytest.param(
            "u=https%3A%2F%2Fh.test%2Fp?token=secret",
            "u=https%3A%2F%2Fh.test%2Fp",
            id="encoded-url-raw-question-mark",
        ),
        pytest.param(
            "u=https%3A%2F%2Fh.test%2Fp%23access_token%3Dsecret&x=1",
            "u=https%3A%2F%2Fh.test%2Fp",
            id="encoded-url-encoded-hash",
        ),
        pytest.param(
            "u=https%3A%2F%2Fh.test%2Fp(1)%3Fsig%3Dsecret done",
            "u=https%3A%2F%2Fh.test%2Fp(1) done",
            id="encoded-url-parenthesis",
        ),
        pytest.param(
            "https://proxy.test/r/https%3A%2F%2Fh.test%2Fp%3Fsig%3Dsecret now",
            "https://proxy.test/r/https%3A%2F%2Fh.test%2Fp now",
            id="encoded-url-in-path",
        ),
    ],
)
def test_redact_uris_ends_the_path_it_keeps_at_a_query_in_any_form(
    text: str, redacted: str
) -> None:
    """The path kept holds no `?` or `#`, written out, percent-encoded or
    JSON-escaped, so no query survives in it once decoded."""
    assert redact_uris(text) == redacted


# URLs whose path or query holds a character RFC 3986 allows unencoded (`'`,
# `]`, `>`) or a non-ASCII letter, which a serialiser with `ensure_ascii`
# writes as a `\uXXXX` escape.
_AWKWARD_URLS = [
    pytest.param(
        "https://en.wikipedia.org/wiki/C\u00f4te_d'Ivoire?access_token=SECRET",
        "https://en.wikipedia.org/wiki/C\u00f4te_d'Ivoire",
        id="apostrophe-in-path",
    ),
    pytest.param(
        "https://example.org/donn\u00e9es.csv?X-Amz-Signature=SECRET",  # codespell:ignore
        "https://example.org/donn\u00e9es.csv",  # codespell:ignore
        id="accent-in-path",
    ),
    pytest.param("https://h.test/a]b?sig=SECRET", "https://h.test/a]b", id="bracket-in-path"),
    pytest.param("https://h.test/a>b?sig=SECRET", "https://h.test/a>b", id="angle-in-path"),
    pytest.param(
        "https://h.test/p?q=o'neil&access_token=SECRET",
        "https://h.test/p",
        id="apostrophe-in-query",
    ),
    pytest.param(
        "https://api.example.test/items?filter[a]=1&access_token=SECRET",
        "https://api.example.test/items",
        id="brackets-in-query",
    ),
    pytest.param(
        "https://h.test/p?a=<b>&access_token=SECRET", "https://h.test/p", id="angles-in-query"
    ),
    pytest.param(
        "https://h.test/p?q=caf\u00e9&sig=SECRET",  # codespell:ignore
        "https://h.test/p",
        id="accent-in-query",
    ),
]


@pytest.mark.parametrize(
    "form",
    [
        *_FORMS,
        pytest.param(lambda url: json.dumps({"url": url}), id="json-ensure-ascii"),
        pytest.param(lambda url: f"<{url}>", id="angle-brackets"),
    ],
)
@pytest.mark.parametrize(("url", "path"), _AWKWARD_URLS)
def test_redact_uris_drops_the_query_after_a_character_a_url_may_hold(
    form: Callable[[str], str], url: str, path: str
) -> None:
    """The URL runs past every character but whitespace and quotes, so a query
    after `'`, `]`, `>` or a `\\uXXXX` escape is still found and dropped."""
    redacted = redact_uris(form(url))

    assert "SECRET" not in redacted
    assert redacted == form(path)


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param("see https://h.test%3Fsig=SECRET now", "see https://h.test now", id="host"),
        pytest.param(
            "https%3A\\/\\/h.test\\/p%3Fsig%3DSECRET",
            "https%3A\\/\\/h.test\\/p",
            id="encoded-json-escaped-slashes",
        ),
        pytest.param(
            "https%3A%2F%2Fh.test%2Fp%3Fq%3Do'neil%26sig%3DSECRET",
            "https%3A%2F%2Fh.test%2Fp",
            id="encoded-apostrophe",
        ),
        pytest.param(
            "u=https%3A//h.test/p?a=1&sig=SECRET",
            "u=https%3A//h.test/p",
            id="encoded-raw-ampersand",
        ),
    ],
)
def test_redact_uris_drops_a_query_that_follows_the_host_or_an_encoded_separator(
    text: str, redacted: str
) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            "see [link](https://h.test/p?sig=SECRET).",
            "see [link](https://h.test/p).",
            id="markdown",
        ),
        pytest.param(
            "{'url': 'https://h.test/p?sig=SECRET', 'n': 1}",
            "{'url': 'https://h.test/p', 'n': 1}",
            id="repr",
        ),
        pytest.param(
            json.dumps("[link](https://h.test/p?sig=SECRET)\n\nNext"),
            json.dumps("[link](https://h.test/p)\n\nNext"),
            id="json-newline-after",
        ),
        pytest.param(
            json.dumps(
                json.dumps({"url": "https://h.test/caf\u00e9?sig=SECRET"})  # codespell:ignore
            ),
            json.dumps(json.dumps({"url": "https://h.test/caf\u00e9"})),  # codespell:ignore
            id="json-twice",
        ),
        pytest.param(
            "https://h.test/p?sig=SECRET),glued text", "https://h.test/p text", id="glued-after"
        ),
    ],
)
def test_redact_uris_keeps_the_punctuation_that_closes_a_url_with_a_query(
    text: str, redacted: str
) -> None:
    """What follows a query up to whitespace or a quote is dropped, except the
    closing punctuation at its end."""
    assert redact_uris(text) == redacted


def test_redact_uris_drops_the_rest_of_a_query_string_after_an_encoded_urls_query() -> None:
    """A percent-encoded URL runs to whitespace like a written-out one, so a raw
    `&` in its query cannot end it early."""
    text = f"GET /r?u={quote(_AWS_PRESIGNED, safe='')}&x=1 HTTP/1.1"
    assert redact_uris(text) == f"GET /r?u={quote(_AWS_PATH, safe='')} HTTP/1.1"


@pytest.mark.parametrize(
    "uri",
    [
        _ORDINARY_PATH,
        "https://example.test/x-goog-signature/p",
        "https://example.test/people/@handle",
        "[x](https://www.oecd.org/p(1)) and (https://example.test/b)",
        f"u={quote(_ORDINARY_PATH, safe='')}&x=1",
        "gs://bucket/path/object.csv?generation=1",
        "ui://presentation/view.html",
    ],
)
def test_redact_uris_keeps_a_uri_with_nothing_to_cut_whole(uri: str) -> None:
    assert redact_uris(f"see {uri} now") == f"see {uri} now"


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        (
            "connection to postgresql+asyncpg://dfo:hunter2@10.0.0.5:5432/sessions failed",
            "connection to postgresql+asyncpg://10.0.0.5:5432 failed",
        ),
        ("see https://user:pw@example.test/p?q=1 now", "see https://example.test now"),
        ("see https://user@example.test now", "see https://example.test now"),
        (json.dumps("dsn:\npostgresql://dfo:hunter2@db/s"), '"dsn:\\npostgresql://db"'),
        (
            "u=https%3A%2F%2Fdfo%3Ahunter2%40db.test%2Fp%3Fq%3D1&x=1",
            "u=https%3A%2F%2Fdb.test",
        ),
    ],
)
def test_redact_uris_cuts_a_uri_with_userinfo_and_drops_the_userinfo(
    text: str, redacted: str
) -> None:
    assert redact_uris(text) == redacted


# Text glued to a URI's left: JSON escapes of a newline, a tab and a non-breaking
# space, a plain space, a percent-encoded newline, and a word and full stop.
_GLUED = [
    pytest.param(lambda uri: json.dumps("link:\n" + uri), id="json-newline"),
    pytest.param(lambda uri: json.dumps({"text": "link:\n" + uri}), id="json-object"),
    pytest.param(lambda uri: json.dumps("a\t" + uri), id="json-tab"),
    pytest.param(lambda uri: json.dumps("a\u00a0" + uri), id="json-nbsp"),
    pytest.param(lambda uri: "a " + uri, id="space"),
    pytest.param(lambda uri: "x%0A" + uri, id="percent-newline"),
    pytest.param(lambda uri: "see." + uri, id="full-stop"),
]


@pytest.mark.parametrize("glue", _GLUED)
@pytest.mark.parametrize(
    ("uri", "secret", "cut"),
    [
        pytest.param(_ARTIFACT, _SECRET, "dfo://artifact", id="dfo"),
        pytest.param(_SIGNED_V4, "deadbeefcafebabe", _SIGNED_PATH, id="signed"),
    ],
)
def test_redact_uris_cuts_a_credential_glued_to_the_text_before_it(
    glue: Callable[[str], str], uri: str, secret: str, cut: str
) -> None:
    redacted = redact_uris(glue(uri))

    assert secret not in redacted
    assert redacted == glue(cut)


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            json.dumps(_ARTIFACT).replace("/", "\\/"), '"dfo:\\/\\/artifact"', id="json-slashes"
        ),
        pytest.param("DFO://ARTIFACT/a/b", "DFO://ARTIFACT", id="upper-dfo"),
        pytest.param(
            "HTTPS://Storage.googleapis.com/b/o?X-GOOG-SIGNATURE=deadbeef",
            "HTTPS://Storage.googleapis.com/b/o",
            id="upper-https",
        ),
        pytest.param(
            "https://proxy.test/r?u=" + quote(_SIGNED_V4, safe=""),
            "https://proxy.test/r",
            id="nested-encoded",
        ),
        pytest.param(
            "https://storage.googleapis.com/b/o?a=1&amp;X-Goog-Signature=deadbeef",
            "https://storage.googleapis.com/b/o",
            id="html-amp",
        ),
        pytest.param(
            "https://storage.googleapis.com/b/o?a=1;Signature=deadbeef",
            "https://storage.googleapis.com/b/o",
            id="semicolon",
        ),
    ],
)
def test_redact_uris_cuts_an_escaped_upper_case_or_nested_credential(
    text: str, redacted: str
) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    "escape",
    [
        pytest.param(lambda text: text.replace("&", "\\u0026"), id="ampersand"),
        pytest.param(lambda text: text.replace("/", "\\/"), id="slash"),
        pytest.param(
            lambda text: text.replace("/", "\\/").replace("&", "\\u0026").replace("=", "\\u003d"),
            id="all",
        ),
    ],
)
def test_redact_uris_drops_a_signed_urls_whole_query_through_its_json_escapes(
    escape: Callable[[str], str],
) -> None:
    """A JSON encoder may write `&` as `\\u0026`, `/` as `\\/` and `=` as
    `\\u003d`; the whole query is dropped, not only the part before the first
    escape."""
    redacted = redact_uris(escape(json.dumps({"url": _SIGNED_V4})))

    assert "deadbeefcafebabe" not in redacted
    assert "X-Goog-Credential" not in redacted
    assert redacted == escape(json.dumps({"url": _SIGNED_PATH}))


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            "u=dfo%3A%2F%2Fartifact%2Fspec_1%2Fcv_2&x=1",
            "u=dfo%3A%2F%2Fartifact&x=1",
            id="in-query",
        ),
        pytest.param(quote(_ARTIFACT, safe=""), "dfo%3A%2F%2Fartifact", id="whole"),
        pytest.param("DFO%3a%2f%2fartifact%2Fsecret", "DFO%3a%2f%2fartifact", id="lower-hex"),
        pytest.param("xdfo%3A%2F%2Fartifact%2Fsecret", "xdfo%3A%2F%2Fartifact", id="glued"),
    ],
)
def test_redact_uris_cuts_a_percent_encoded_dfo_uri(text: str, redacted: str) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            "get https://storage.googleapis.com/b/r(1).csv"
            "?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=deadbeef now",
            "get https://storage.googleapis.com/b/r(1).csv now",
            id="path",
        ),
        pytest.param(
            "[csv](https://storage.googleapis.com/b/r(1).csv?X-Goog-Signature=deadbeef) done",
            "[csv](https://storage.googleapis.com/b/r(1).csv) done",
            id="markdown-link",
        ),
    ],
)
def test_redact_uris_keeps_a_parenthesis_in_a_signed_urls_path(text: str, redacted: str) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a" * 200_000, id="letters"),
        pytest.param("a://" * 50_000, id="separators"),
        pytest.param("://" + "u" * 200_000, id="long-authority"),
        pytest.param("://a@" * 40_000, id="userinfo"),
        pytest.param("https%3A%2F%2F" * 15_000, id="encoded"),
        pytest.param("https://a/" + "%3" * 70_000, id="http-percent"),
        pytest.param("https://a/" + "\\u00" * 50_000, id="http-json-escapes"),
        pytest.param("https%3A%2F%2Fa" + "%4" * 70_000, id="encoded-authority"),
        pytest.param("https://" + "%3" * 70_000, id="http-authority-percent"),
        pytest.param("https://a/" + "\\" * 200_000, id="http-backslashes"),
        pytest.param("https://a/?" + "\\" * 200_000, id="query-backslashes"),
        pytest.param("https://a/" + "\\u" * 70_000, id="http-escape-starts"),
        pytest.param("https://a/?" + ")" * 200_000, id="closing-run"),
        pytest.param("https://a/" + "'" * 200_000, id="apostrophes"),
        pytest.param("https://a?" * 20_000, id="queries"),
        pytest.param("https%3A\\/\\/" * 15_000, id="encoded-json-separators"),
    ],
)
def test_redact_uris_scans_long_repetitive_text_in_linear_time(text: str) -> None:
    """A scan that retried a run from each of its characters would take tens of
    seconds here."""
    started = time.perf_counter()
    redact_uris(text)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            "conn%20postgresql://dfo:hunter2@db/x", "conn%20postgresql://db", id="after-escape"
        ),
        pytest.param(
            "x%0Apostgresql://dfo:hunter2@db/x", "x%0Apostgresql://db", id="after-newline"
        ),
        pytest.param("2postgresql://dfo:hunter2@db/x", "2postgresql://db", id="after-digit"),
        pytest.param(
            "https://a.test/x,postgresql://dfo:hunter2@db/x",
            "https://a.test/x,postgresql://db",
            id="in-rest",
        ),
        pytest.param(
            "ftp://proxy.test/?next=https://u:hunter2@db.test/",
            "ftp://proxy.test/?next=https://db.test",
            id="in-query",
        ),
    ],
)
def test_redact_uris_cuts_userinfo_whatever_precedes_its_separator(
    text: str, redacted: str
) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param(
            f"(see https://x.test/#s){_SIGNED_V4}", "(see https://x.test/", id="after-fragment"
        ),
        pytest.param(
            f"https://app.test/#/view?src={_SIGNED_V4}", "https://app.test/", id="hash-route"
        ),
        pytest.param(
            f"https://proxy.test/?u=x#y&next={_SIGNED_V4}",
            "https://proxy.test/",
            id="fragment-param",
        ),
    ],
)
def test_redact_uris_drops_a_fragment_that_holds_a_signed_url(text: str, redacted: str) -> None:
    assert redact_uris(text) == redacted


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        pytest.param("dfo%3A//artifact/spec_1/cv_2", "dfo%3A//artifact", id="dfo-slashes"),
        pytest.param(quote(_ARTIFACT), "dfo%3A//artifact", id="dfo-quote-default"),
        pytest.param(
            f"url={quote(_SIGNED_V4, safe='')}&x=1",
            f"url={quote(_SIGNED_PATH, safe='')}",
            id="signed-form-value",
        ),
        pytest.param(
            quote(_SIGNED_V4),
            "https%3A//storage.googleapis.com/dfo-artifacts/working/abc.csv",
            id="signed-quote-default",
        ),
        pytest.param(
            f"url={quote(_ORDINARY, safe='')}&x=1",
            f"url={quote(_ORDINARY_PATH, safe='')}",
            id="ordinary",
        ),
    ],
)
def test_redact_uris_cuts_a_percent_encoded_uri_with_or_without_encoded_slashes(
    text: str, redacted: str
) -> None:
    assert redact_uris(text) == redacted


def test_redact_uris_cuts_a_query_or_fragment_that_follows_a_dfo_host_directly() -> None:
    text = "see dfo://artifact?token=secret and (dfo://u@artifact#secret) now"
    assert redact_uris(text) == "see dfo://artifact and (dfo://artifact) now"


def test_redact_uris_keeps_json_encoded_text_parseable() -> None:
    inner = json.dumps({"uri": _ARTIFACT})
    outer = json.dumps({"payload": inner})

    redacted = redact_uris(outer)

    assert json.loads(json.loads(redacted)["payload"]) == {"uri": "dfo://artifact"}
    assert _SECRET not in redacted


def test_redact_uris_leaves_the_closing_parenthesis_of_a_markdown_link() -> None:
    assert redact_uris("[x](dfo://artifact/a/b)") == "[x](dfo://artifact)"


def _span_with_the_uri_everywhere(provider: TracerProvider) -> None:
    with provider.get_tracer("test").start_as_current_span(f"delegate {_ARTIFACT}") as span:
        span.set_attributes(
            {
                "mcp.resource.uri": _ARTIFACT,
                "uris": [_ARTIFACT, "ui://v/p"],
                "count": 3,
            }
        )
        span.add_event("exception", {"exception.message": f"Unknown resource: {_ARTIFACT!r}"})
        span.set_status(Status(StatusCode.ERROR, f"read {_ARTIFACT} failed"))


def test_the_exporter_cuts_the_uri_in_every_field_of_a_span() -> None:
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(UriRedactingExporter(exported)))

    _span_with_the_uri_everywhere(provider)

    [span] = exported.get_finished_spans()
    assert _SECRET not in span.to_json()
    assert span.name == "delegate dfo://artifact"
    assert dict(span.attributes or {}) == {
        "mcp.resource.uri": "dfo://artifact",
        "uris": ("dfo://artifact", "ui://v/p"),
        "count": 3,
    }
    [event] = span.events
    assert dict(event.attributes or {}) == {
        "exception.message": "Unknown resource: 'dfo://artifact'"
    }
    assert span.status.description == "read dfo://artifact failed"


def test_the_otlp_backend_exports_through_the_redacting_exporter(
    isolated_telemetry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = InMemorySpanExporter()
    providers: list[TracerProvider] = []
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda: exported)
    monkeypatch.setattr(trace, "set_tracer_provider", providers.append)

    telemetry._init_otel(service_name="test")
    [provider] = providers
    _span_with_the_uri_everywhere(provider)
    provider.shutdown()

    [span] = exported.get_finished_spans()
    assert _SECRET not in span.to_json()


def test_the_exporter_cuts_credentials_and_keeps_a_urls_path() -> None:
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(UriRedactingExporter(exported)))

    with provider.get_tracer("test").start_as_current_span("read") as span:
        span.set_attributes({"signed": _SIGNED_V4, "source": f"see {_ORDINARY}"})
        span.add_event(
            "exception",
            {"exception.message": "connect to 'postgresql://dfo:hunter2@db:5432/s' failed"},
        )

    [span] = exported.get_finished_spans()
    assert "deadbeefcafebabe" not in span.to_json()
    assert "hunter2" not in span.to_json()
    [event] = span.events
    assert dict(event.attributes or {}) == {
        "exception.message": "connect to 'postgresql://db:5432' failed"
    }
    assert dict(span.attributes or {}) == {
        "signed": _SIGNED_PATH,
        "source": f"see {_ORDINARY_PATH}",
    }
