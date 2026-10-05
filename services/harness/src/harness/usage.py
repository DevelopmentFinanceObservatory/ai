"""One log line per model call, with the token counts the API reported.

Every seam that calls a model funnels its `usage_metadata` through
`log_usage`, so a grep for "token usage" over a server log reconstructs
what a turn cost. Logging only -- nothing reads these numbers back.
"""

from __future__ import annotations

import logging

from google.genai import types

logger = logging.getLogger(__name__)


def log_usage(seam: str, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
    if usage is None:
        logger.info("token usage seam=%s (no usage metadata reported)", seam)
        return
    logger.info(
        "token usage seam=%s prompt=%s cached=%s output=%s thoughts=%s total=%s",
        seam,
        usage.prompt_token_count or 0,
        usage.cached_content_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
        usage.total_token_count or 0,
    )


__all__ = ["log_usage"]
