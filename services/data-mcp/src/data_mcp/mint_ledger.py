"""The upstream mints this process holds a dfo-apis mint slot for on behalf of outside callers.

A `get_data` call under `EXTERNAL_CALL_BUDGET` gives its mint POST at most 32 s. dfo-apis
runs a synchronous mint for up to `UPSTREAM_SYNC_MINT_TIMEOUT_S` whether or not its client
is still waiting, and holds one of its mint slots (`mint_max_concurrency`, 4) until the mint
ends. A call can therefore give up on a mint that holds its slot for about 90 s more, and a
caller that repeats the call every few seconds starts a new mint each time.

`MintLedger` holds one `HeldMint` per mint POST, from the moment it is sent until dfo-apis
answers it, or until `UPSTREAM_SYNC_MINT_TIMEOUT_S` plus `MINT_LEDGER_MARGIN_S` after it was
sent when no answer comes. The margin covers the rendering and store writes dfo-apis does
after the fetch its own timeout bounds, which can run a few seconds past it. It refuses a
mint while it holds `limit` others, and refuses a mint whose request matches one it holds,
which would start the same work twice. A POST whose connection failed started nothing
upstream, but the client cannot tell that from a timeout, so its mint is held for the whole
window too.

`get_data` counts only mints made under a budget that does not wait out dfo-apis'
synchronous mint (`CallBudget.waits_out_sync_mint`), which is `EXTERNAL_CALL_BUDGET`. A
mint under `DIRECT_CALL_BUDGET` waits longer than dfo-apis runs one, so it leaves nothing
running behind it, and the harness's own calls are not limited here. The ledger is per
process; `Settings.external_mint_concurrency` states what that means across instances.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_mcp.deadline import UPSTREAM_SYNC_MINT_TIMEOUT_S
from data_mcp.tool_errors import ToolRefusal

# Margin past `UPSTREAM_SYNC_MINT_TIMEOUT_S` a held mint's slot stays reserved: dfo-apis'
# own timeout bounds the fetch alone, and its rendering and store writes after the fetch
# can hold the mint's permit a few seconds longer.
MINT_LEDGER_MARGIN_S = 5.0


@dataclass(frozen=True, kw_only=True, eq=False)
class HeldMint:
    """One mint POST the ledger holds a slot for, in seconds on the call deadline's clock.

    `answer_by` is the latest the POST can still be waiting for its answer, the end of
    the deadline it was sent under. `expires_at` is when dfo-apis has stopped running it.
    """

    request_key: str
    started_at: float
    answer_by: float
    expires_at: float

    def wait_s(self, now: float) -> float:
        """Seconds until this mint's slot may be free: until its POST must have ended
        while the POST may still be waiting, and until its window ends after that."""
        return (self.answer_by if now < self.answer_by else self.expires_at) - now


class MintLedger:
    """The mints one process holds dfo-apis mint slots for.

    `admit` and `answered` run on the event loop with no await between a check and
    its update, so concurrent calls see a consistent ledger without a lock.
    """

    def __init__(self) -> None:
        self._held: list[HeldMint] = []

    def admit(
        self, *, request_key: str, spec_id: str, now: float, answer_by: float, limit: int
    ) -> HeldMint | ToolRefusal:
        """Hold a slot for a mint POST about to be sent at `now`, or refuse it.

        `request_key` identifies the request body, so a repeat of the same `get_data`
        call matches it. The refusal's `retry_after_s` is the wait until the slot it
        waits on may be free (`HeldMint.wait_s`): the matching mint's for a repeat, and
        the soonest of all held mints' when `limit` are held.
        """
        self._held = [mint for mint in self._held if mint.expires_at > now]
        for mint in self._held:
            if mint.request_key == request_key:
                return ToolRefusal.mint_still_running(
                    spec_id=spec_id,
                    started_s_ago=now - mint.started_at,
                    retry_after_s=mint.wait_s(now),
                )
        if len(self._held) >= limit:
            return ToolRefusal.mint_slots_full(
                limit=limit, retry_after_s=min(mint.wait_s(now) for mint in self._held)
            )
        mint = HeldMint(
            request_key=request_key,
            started_at=now,
            answer_by=answer_by,
            expires_at=now + UPSTREAM_SYNC_MINT_TIMEOUT_S + MINT_LEDGER_MARGIN_S,
        )
        self._held.append(mint)
        return mint

    def answered(self, mint: HeldMint) -> None:
        """Free `mint`'s slot: dfo-apis answered its POST, so the mint holds no slot there.

        An answer that arrives after the mint's window has ended finds it already
        dropped.
        """
        if mint in self._held:
            self._held.remove(mint)
