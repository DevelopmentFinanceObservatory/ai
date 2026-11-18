"""The code and retry mode a stream-fatal `error` frame states for the error
a turn ended on.

`harness.studio_translate` builds the frame, and the turn's record closes on
the frame's code. `harness.studio_store` records a turn whose closing write
failed from its cursor's error alone, with the code the frame the client got
carried (`recorded_code`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from executor.classify import internal_fault

from harness.wire import ErrorCode, RetryMode


def stated_refusal(refusal: Mapping[str, Any] | None, *, transient: bool) -> Mapping[str, Any]:
    """The refusal an error states: its own, or, for an error that states
    none, nothing when it is `transient` and an internal fault otherwise."""
    return refusal or ({} if transient else internal_fault())


def wire_error(
    refusal: Mapping[str, Any], *, timed_out: bool = False
) -> tuple[ErrorCode, RetryMode]:
    """The code and retry mode the frame for `refusal` states. A refusal
    whose code the wire grammar knows keeps it; a timed-out MCP call
    (`timed_out`) reads as `transient_unavailable` whatever its own code.
    Otherwise the refusal's own `retry_mode` decides: `retry_after_delay`,
    or none, becomes `transient_unavailable`, anything else
    `capability_unavailable`, so a do-not-retry refusal tells the user the
    capability is unavailable."""
    wire_code = ErrorCode.TRANSIENT_UNAVAILABLE.value if timed_out else refusal.get("code")
    try:
        return ErrorCode(wire_code), RetryMode(refusal.get("retry_mode", RetryMode.DO_NOT_RETRY))
    except ValueError:
        try:
            retry = RetryMode(refusal.get("retry_mode", RetryMode.RETRY_AFTER_DELAY))
        except ValueError:
            retry = RetryMode.RETRY_AFTER_DELAY
        code = (
            ErrorCode.TRANSIENT_UNAVAILABLE
            if retry is RetryMode.RETRY_AFTER_DELAY
            else ErrorCode.CAPABILITY_UNAVAILABLE
        )
        return code, retry


def recorded_code(error: Mapping[str, Any]) -> ErrorCode:
    """The code of the frame for a turn closed on `error`, the record
    `harness.turn.loop.failure` writes on the cursor
    (`harness.studio_translate.recorded_fatal`)."""
    if error.get("timeout_tool") is not None:
        return ErrorCode.TRANSIENT_UNAVAILABLE
    refusal = stated_refusal(error.get("refusal"), transient=bool(error.get("transient")))
    return wire_error(refusal)[0]
