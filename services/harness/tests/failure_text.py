"""What a line of failure text has to be for a user to read it whole."""

import re
from collections.abc import Mapping

from harness.sentences import FAILURE_CLASS, MAX_SENTENCE_CHARS, failure_sentence

# An identifier: a code, tool or field name joined by an underscore, a spec
# or dcid reference, or a run of hexadecimal digits such as a trace id.
_MACHINE_TEXT = re.compile(r"[a-z0-9]_[a-z0-9]|dcid|[0-9a-f]{8,}", re.IGNORECASE)


def is_user_sentence(detail: object) -> bool:
    """Whether `detail` is text the Activity log shows whole and a user can
    read: one line with no full stop before its end, within the log's length,
    naming no identifier."""
    return (
        isinstance(detail, str)
        and 0 < len(detail) <= MAX_SENTENCE_CHARS
        and "\n" not in detail
        and not re.search(r"\.\s", detail)
        and not _MACHINE_TEXT.search(detail)
    )


def classed_sentence(
    code: object, retry_mode: object = None, fix: Mapping[str, object] | None = None
) -> str:
    """`failure_sentence` for a `code` the failure table classes. A code in
    no class fails here, where `failure_sentence` would answer the sentence
    that names no cause."""
    assert code in FAILURE_CLASS, f"{code!r} has no failure class"
    return failure_sentence(code, retry_mode, fix)
