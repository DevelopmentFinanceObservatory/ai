"""The one encoder for a json or jsonb SQL parameter."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from executor.ledger import NonFiniteJsonError


def to_json_param(value: Any) -> str:
    """`value` as JSON text for a `CAST(... AS jsonb)` parameter.

    Raises:
        NonFiniteJsonError: `value` holds NaN or an infinity, which Postgres
            refuses in a json column. A code defect, raised before any
            statement runs.
    """
    try:
        return json.dumps(value, allow_nan=False)
    except ValueError as exc:
        if "Out of range float" in str(exc):
            raise NonFiniteJsonError(
                "a non-finite float reached a JSON column; the value holding it is a defect"
            ) from exc
        raise


def json_or_none(value: Any, *, logger: logging.Logger, failure: Callable[[], str]) -> str | None:
    """`value` as `to_json_param` writes it, or `None` when JSON cannot hold
    it (a NaN or an infinity, or a value `json` cannot encode), logged at
    error on `logger` as `failure()`, built only then. For a best-effort row its write
    goes on without."""
    try:
        return to_json_param(value)
    except (TypeError, ValueError):
        logger.exception(failure())
        return None
