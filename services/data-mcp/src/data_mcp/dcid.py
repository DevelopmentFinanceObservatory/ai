"""The dcid syntax the graph's node routes accept, held once and free of network imports.

`DCID_PATTERN` and `DCID_MAX_LENGTH` mirror dfo-apis `src/graph_api/node_models.py` `Dcid`,
which is built from `routing.DcidConvertor.regex` and `routing.DCID_MAX_LENGTH`: one or more
`/`-separated segments, each drawn from `[A-Za-z0-9_.-]`, at most 512 characters. The nodes
route refuses a whole batch that holds one id outside it.

`get_data_tool._DCID_PATTERN` is a different test: it decides whether a ref a caller typed is
a dcid for `get_data` rather than a name to resolve, and is narrower (lowercase-initial
namespace). This module decides what the nodes route accepts.
"""

import re

DCID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*$")
DCID_MAX_LENGTH = 512

# The marker `entity_ambiguous` puts before each candidate's dcid, that `get_data` reads as "a
# dcid, not a name to resolve" in `scope.entities`, and that a frame cell carries on a ref.
ENTITY_DCID_MARKER = "dcid:"


def bare_dcid(ref: str) -> str:
    """`ref` without its `ENTITY_DCID_MARKER`, so a ref compares the same with or without it."""
    return ref.removeprefix(ENTITY_DCID_MARKER)


def has_dot_segment(ref: str) -> bool:
    """Whether a `/`-separated segment of `ref` is exactly `.` or `..`, which no dcid has: the
    pattern admits dots, but such a segment is a path traversal, never a node."""
    return any(segment in (".", "..") for segment in ref.split("/"))


def is_dcid(ref: str) -> bool:
    """Whether the nodes route accepts `ref` as a dcid (free text such as a breakdown's
    sector name is not one)."""
    return len(ref) <= DCID_MAX_LENGTH and DCID_PATTERN.fullmatch(ref) is not None
