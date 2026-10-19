"""A DAC2A donor-side frame with `financingProvider_party_kind`, for the tests that sum,
rank and divide across donors."""

from __future__ import annotations

from collections.abc import Mapping

from dfo_frame_grammar import Frame

ODA = "dcid:Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
PROVIDER = "financingProvider"
PROVIDER_KINDS = f"{PROVIDER}_party_kind"


def provider_frame(
    donors: Mapping[str, tuple[str, float]], *, years: tuple[str, ...] = ("2021",)
) -> Frame:
    """ODA to Ethiopia from `donors`, each a `(kind, value)` pair, in every year of `years`."""
    return Frame.build(
        role_keys=[PROVIDER, "financingRecipient", PROVIDER_KINDS],
        rows=[
            {
                "stat_var": ODA,
                PROVIDER: donor,
                "financingRecipient": "country/ETH",
                PROVIDER_KINDS: kind,
                "time_period": year,
                "value": value,
                "unit": "USD",
            }
            for year in years
            for donor, (kind, value) in donors.items()
        ],
    )
