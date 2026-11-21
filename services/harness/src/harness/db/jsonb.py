"""The one encoder for a json or jsonb SQL parameter."""

from __future__ import annotations

import json
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
