"""Re-resolving a part in another unit through Search, in code.

A unit is a currency, a price basis and, for constant prices, a base year.
`resubmit_unit` states it on the part's identity (`data_mcp.definition_writes.unit_identity`)
and posts the edited identity through data-mcp's `resolve_definition`
(`translator.discovery.resolve_definition`), as a scope switch posts
(`harness.graph.switch`). Search answers with the part in that unit under its
own spec_id.

The identity edited is the reading's stored scope base, posted with the
scope's entries, when the reading holds scope choices: a resubmit of the
compiled identity alone would answer a part with no scope choices. Otherwise
it is the identity of the block the part was read from.

A Search refusal comes back as a `UnitRefused` value.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from compiler.sources import readable_wire_groups
from compiler.surface import ToolSurface
from data_mcp.definition_writes import unit_identity
from planner.workflow import ResolvedGroup
from translator.discovery import DefinitionNotResolvedError, resolve_definition

from harness.turn_records import SearchReading

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnitResubmitted:
    """The part Search answered for the identity in the new unit, and its spec block."""

    spec_id: str
    part: Mapping[str, Any]
    spec: Mapping[str, Any]


@dataclass(frozen=True)
class UnitRefused:
    """A unit Search did not serve: `code` is its refusal's code (`None` when it states none), `reason` one sentence
    saying why, and `retryable` whether the same call can succeed if made again."""

    code: str | None
    reason: str
    retryable: bool


def _groups(
    reading: SearchReading | None, block: Mapping[str, Any]
) -> tuple[ResolvedGroup, ...] | None:
    """The named sets a resubmit posts beside the part: the block's where they read, else the reading's, `None` when neither reads."""
    groups = readable_wire_groups(block)
    if groups:
        return groups
    stored_stands = (
        reading is not None
        and not reading.entity_groups_unreadable
        and (groups is not None or bool(reading.entity_groups))
    )
    if stored_stands and reading is not None:
        return reading.entity_groups
    return groups


async def resubmit_unit(
    surface: ToolSurface,
    *,
    reading: SearchReading | None,
    block: Mapping[str, Any],
    unit: Mapping[str, Any],
    label: str,
) -> UnitResubmitted | UnitRefused:
    """Resolve the part `block` and `reading` stand for in `unit`, named `label` in a refusal.

    `unit` is `{"currency": "EUR", "prices": "constant", "base_year": 2024}`
    (`unit_identity`). The part is posted with its named sets, so Search names
    them again. The new spec_id is the one Search's answer states, since an
    answer compiled from scope entries is minted under its own.

    Returns:
        `UnitResubmitted` for a definite part, else `UnitRefused`.
    """
    scope = reading.scope if reading is not None else None
    if scope is not None and scope.choices:
        base, entries = scope.base.identity, scope.entries
    else:
        definition = block.get("definition")
        base = definition.get("identity") if isinstance(definition, Mapping) else None
        entries = ()
    groups = _groups(reading, block)
    if not isinstance(base, Mapping) or groups is None:
        return UnitRefused(
            code=None, reason=f"Search's record of {label} could not be read.", retryable=False
        )
    try:
        spec_id, identity = unit_identity(base, unit)
    except ValueError as exc:
        return UnitRefused(
            code=None, reason=f"That unit is not one Search states: {exc}.", retryable=False
        )
    try:
        part = await resolve_definition(
            surface,
            spec_id=spec_id,
            identity=identity,
            groups=groups,
            scope_entries=[entry.model_dump(mode="json", exclude_none=True) for entry in entries],
        )
    except DefinitionNotResolvedError as exc:
        logger.info("unit not resolved: %s code=%s: %s", label, exc.code, exc)
        if exc.retryable or exc.not_issued:
            return UnitRefused(
                code=exc.code,
                reason=f"Search could not serve {label} in that unit just now.",
                retryable=True,
            )
        return UnitRefused(
            code=exc.code, reason=f"Search cannot serve {label} in that unit.", retryable=False
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    if not isinstance(spec, Mapping) or not spec.get("spec_id"):
        return UnitRefused(
            code=None, reason=f"Search holds no data for {label} in that unit.", retryable=False
        )
    return UnitResubmitted(spec_id=str(spec["spec_id"]), part=part, spec=spec)
