"""The `describe` tool: what a graph identifier means, by node type.

`search` and `get_data` hand the model dcids inside their own digests --
a StatVar behind a spec_id, a source behind a citation, an entity behind an
acknowledgement line. `describe` is the one tool that takes a dcid as input,
because it is the one tool whose job is to explain what a dcid already
carried by another tool's digest means. Never call it on an invented dcid.

Upstream access sits entirely behind `data_mcp.node_lookup`: this module
never builds a request or reads a wire shape itself, only the outcomes that
module hands back. Rendering those outcomes into the digest lives in
`data_mcp.describe_digest`.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field, ValidationError

from data_mcp import node_lookup
from data_mcp.describe_digest import render_describe_digest
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import NotHeldEntry
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

# At most this many dcids per call. A named module constant so every refusal
# that cites it interpolates the real bound rather than a literal that could
# drift from the check below.
DESCRIBE_ID_CAP = 8

# Mirrors the node batch route's own dcid grammar: one or more `/`-separated
# segments, each drawn from `[A-Za-z0-9_.-]`. The route refuses a whole batch
# that holds one malformed dcid, so validating each id locally, before
# dispatch, names the offending id at its own `/ids/<index>` pointer.
DCID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*$")

# Other tools' own refusals quote a stat_var ref with this prefix, e.g.
# `dcid:sdg/DC_TOF_HLTHL`. Stripping it before grammar validation lets that
# exact quoted form work here unchanged.
_DCID_PREFIX = "dcid:"


def _strip_dcid_prefix(ids: list[str]) -> list[str]:
    """Strip a leading `dcid:` prefix from each id, so the bare dcid reaches validation."""
    return [dcid.removeprefix(_DCID_PREFIX) for dcid in ids]


def _cap_and_grammar_refusal(ids: list[str]) -> ToolResult | None:
    if len(ids) == 0:
        return to_tool_result(
            ToolRefusal.request_invalid(
                tool="describe", pointer="/ids", detail="at least one dcid per call"
            )
        )
    if len(ids) > DESCRIBE_ID_CAP:
        return to_tool_result(
            ToolRefusal.request_invalid(
                tool="describe",
                pointer="/ids",
                detail=f"at most {DESCRIBE_ID_CAP} dcids per call, got {len(ids)}",
            )
        )
    for index, dcid in enumerate(ids):
        if not DCID_PATTERN.fullmatch(dcid):
            return to_tool_result(
                ToolRefusal.request_invalid(
                    tool="describe",
                    pointer=f"/ids/{index}",
                    detail=(
                        f"{dcid!r} is not a well-formed dcid; expected one or more "
                        "'/'-separated segments of letters, digits, '_', '.', or '-'"
                    ),
                )
            )
        # The grammar above admits dots inside a segment, which stay legal,
        # but a segment that is exactly `.` or `..` is never a dcid. Refusing
        # it here names it malformed, where the route would call it unheld.
        if any(segment in (".", "..") for segment in dcid.split("/")):
            return to_tool_result(
                ToolRefusal.request_invalid(
                    tool="describe",
                    pointer=f"/ids/{index}",
                    detail=(
                        f"{dcid!r} is not a well-formed dcid; a '/'-separated segment "
                        "of exactly '.' or '..' is never valid"
                    ),
                )
            )
    return None


async def describe(
    ids: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=DESCRIBE_ID_CAP,
            description=(
                f"1..{DESCRIBE_ID_CAP} ids to describe, read out of another tool's "
                "digest -- never invented."
            ),
        ),
    ],
) -> ToolResult:
    """Describe what one or more graph identifiers mean.

    Call this on a dcid you read out of `search`'s or `get_data`'s own
    digest -- never on one you invented. Pass 1 to 8 ids per call.

    Every node comes back with its label, kind and description, plus what
    its kind carries:
      - a StatVar: what it measures, the roles it declares, the years held
        and its sources.
      - a source, dataset or provenance node: its publisher, licence and
        licence type, freshness, and url.
      - a place: what it's contained in, and a region's member count.
      - a group: its membership authority and member count.
      - an organisation or aggregate: the roles it can fill.
      - a code, code system or unit: what it codes or measures.
      - any other node: its types.
    A value the graph does not record is named under "not known".

    Labels return in English, whatever language the query was in.

    A dcid this deployment's graph does not hold refuses `node_not_held`,
    naming every such dcid -- but any dcid in the same call that did
    resolve is still described in this reply's content. Every other
    failure (a malformed dcid, too many dcids, an upstream outage) fails
    the whole call.

    Args:
        ids: The ids to describe, 1 to 8 per call.

    Returns:
        A digest: one block per dcid, shaped by its node type, plus a
        trailing note for any dcid this deployment's graph does not hold.
    """
    ids = _strip_dcid_prefix(ids)
    refusal = _cap_and_grammar_refusal(ids)
    if refusal is not None:
        return refusal

    try:
        outcomes = await node_lookup.describe_nodes(ids)
    except UpstreamError as exc:
        return to_tool_result(from_upstream_error(exc, tool="describe"))
    except (ValueError, KeyError, TypeError, ValidationError) as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the node response did not match the shape this service expects: {exc}"
            )
        )

    unique_ids = list(dict.fromkeys(ids))
    unheld = [dcid for dcid in unique_ids if isinstance(outcomes[dcid], NotHeldEntry)]
    held_present = len(unheld) < len(unique_ids)
    digest = render_describe_digest(dcids=ids, outcomes=outcomes)

    if not unheld:
        nodes = []
        for dcid in unique_ids:
            node = outcomes[dcid]
            # Guaranteed by `unheld` being empty: every outcome here is a
            # node envelope, never `NotHeldEntry`. The assert narrows the
            # type for the type checker as well as at runtime.
            assert not isinstance(node, NotHeldEntry)
            nodes.append(node.model_dump(mode="json", by_alias=True))
        # The full node data goes in structured content for code that reads it. The
        # combined server drops structured content from a successful result, so the
        # digest must stand alone for an outside model.
        return ToolResult(content=digest, structured_content={"nodes": nodes}, is_error=False)

    node_refusal = ToolRefusal.node_not_held(dcids=unheld)
    if held_present:
        return to_tool_result(node_refusal, preamble=digest)
    return to_tool_result(node_refusal)
