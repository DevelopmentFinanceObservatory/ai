"""Text from a caller or a data row, made safe to quote in what a model reads.

A refusal or a digest interpolates such a value into text a model reads, so
every service quotes it through `truncate_value`, the one guard at that boundary.
"""

from __future__ import annotations

import unicodedata

MAX_LABEL_CHARS = 200


def escape_control_chars(value: str) -> str:
    """`value`, with every Unicode category `Cc` (control) or `Cf` (format)
    character replaced by its `\\uXXXX` escape. `dfo_frame_grammar` places no
    character restriction on a role/entity value beyond rejecting lone UTF-16
    surrogates, so a raw newline, carriage return, or escape/bidi-override
    character can reach here verbatim; escaping them is what makes a
    digest's brackets (`[...]`, `role=value`) or a refusal message's own
    line structure real delimiters a value can never break out of or use to
    forge a line that reads as a directive. Every other codepoint, including
    any non-ASCII letter, passes through unchanged."""
    return "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) in ("Cc", "Cf") else char
        for char in value
    )


def truncate_value(value: str) -> str:
    """`value`, escaped (`escape_control_chars`) and then capped to
    `MAX_LABEL_CHARS` -- a role, unit, facet, or entity-label value read
    straight off an input row, or an id a caller typed, carries no length
    ceiling before it reaches here. Escaping runs first: an escaped control
    character is six characters, so capping first and escaping the kept slice
    would let a value crafted to exploit that ordering render far past
    `MAX_LABEL_CHARS` once escaped."""
    value = escape_control_chars(value)
    if len(value) <= MAX_LABEL_CHARS:
        return value
    return value[:MAX_LABEL_CHARS] + "...(truncated)"
