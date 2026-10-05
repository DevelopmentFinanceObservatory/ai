from data_mcp.deadline import UPSTREAM_SYNC_MINT_TIMEOUT_S
from data_mcp.mint_ledger import MINT_LEDGER_MARGIN_S, HeldMint, MintLedger
from data_mcp.tool_errors import ToolRefusal


def _admit(ledger: MintLedger, key: str, *, now: float) -> HeldMint | ToolRefusal:
    return ledger.admit(
        request_key=key, spec_id="spec_" + "a" * 32, now=now, answer_by=now + 35.0, limit=1
    )


def test_a_full_ledger_waits_for_a_running_post_until_it_must_have_ended() -> None:
    """While the held mint's POST may still be waiting, its slot may free as soon as
    dfo-apis answers, so the wait runs to the end of that POST's deadline."""
    ledger = MintLedger()
    assert isinstance(_admit(ledger, "a", now=0.0), HeldMint)

    refused = _admit(ledger, "b", now=10.0)

    assert isinstance(refused, ToolRefusal)
    assert refused.retry_after_s == 25.0


def test_an_answered_mint_frees_its_slot_at_once() -> None:
    ledger = MintLedger()
    held = _admit(ledger, "a", now=0.0)
    assert isinstance(held, HeldMint)

    ledger.answered(held)

    assert isinstance(_admit(ledger, "a", now=1.0), HeldMint)


def test_an_unanswered_mint_holds_its_slot_past_dfo_apis_own_timeout() -> None:
    """dfo-apis' rendering and store writes after the fetch its timeout bounds can run a
    few seconds past it, so the slot stays held for that margin too."""
    ledger = MintLedger()
    _admit(ledger, "a", now=0.0)

    still_held = _admit(ledger, "a", now=UPSTREAM_SYNC_MINT_TIMEOUT_S)

    assert isinstance(still_held, ToolRefusal)

    freed = _admit(ledger, "a", now=UPSTREAM_SYNC_MINT_TIMEOUT_S + MINT_LEDGER_MARGIN_S)

    assert isinstance(freed, HeldMint)
