"""The `skill.read` log line: one INFO line per logical skill read.

`get_skill_tool` (the tool), `skills_resources` (`skill://` reads) and
`skills_extension` (`skills/get`) each call `log_skill_read` once per read.
The fields are JSON members of the entry (`common.logging.JSON_FIELDS`), so
Cloud Logging stores them under `jsonPayload`.
"""

from __future__ import annotations

import logging
from typing import Literal

from common.caller import current_caller, current_mcp_client, current_turn
from common.logging import JSON_FIELDS
from common.mcp_telemetry import is_loopback_request

from data_mcp.tool_errors import ToolErrorCode

logger = logging.getLogger(__name__)

SKILL_READ_EVENT = "skill.read"

type SkillReadVia = Literal["tool", "resource", "extension"]
type SkillReadKind = Literal["load", "manifest", "file"]


def refused_skill_id(code: str, skill_id: str) -> str | None:
    """The `skill_id` a refused read logs; its `path` is always `None`.

    `skill_resource_not_found` means the Skills API found the skill but not
    the file, so it keeps the id. Any other refusal logs none, since the id
    is caller text the API did not confirm.
    """
    return skill_id if code == ToolErrorCode.SKILL_RESOURCE_NOT_FOUND else None


def _on_loopback() -> bool:
    try:
        return is_loopback_request()
    except RuntimeError:  # no HTTP request in flight
        return False


def log_skill_read(
    *,
    via: SkillReadVia,
    kind: SkillReadKind,
    outcome: str,
    skill_id: str | None,
    path: str | None,
) -> None:
    """Write the `skill.read` line for one logical read.

    `skill_id` and `path` are the values the Skills API confirmed, or `None`.
    `surface` is `studio` on a request the harness marked as loopback and
    `external` otherwise; `turn` is logged on loopback only, where the
    harness sets it. `outcome` is `ok` or the refusal code.
    """
    loopback = _on_loopback()
    surface = "studio" if loopback else "external"
    logger.info(
        "%s skill_id=%s kind=%s outcome=%s surface=%s via=%s",
        SKILL_READ_EVENT,
        skill_id,
        kind,
        outcome,
        surface,
        via,
        extra={
            JSON_FIELDS: {
                "event": SKILL_READ_EVENT,
                "skill_id": skill_id,
                "path": path,
                "kind": kind,
                "outcome": outcome,
                "surface": surface,
                "caller": current_caller(),
                "turn": current_turn() if loopback else None,
                "client": current_mcp_client(),
                "via": via,
            }
        },
    )
