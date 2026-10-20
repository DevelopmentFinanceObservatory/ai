"""A DAC2A donor-side frame with `financingProvider_party_kind`, `_party_class` and
`_party_whole`, for the tests that sum, rank and divide across donors."""

from __future__ import annotations

from collections.abc import Mapping

from dfo_frame_grammar import Frame

ODA = "dcid:Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
PROVIDER = "financingProvider"
PROVIDER_KINDS = f"{PROVIDER}_party_kind"
PROVIDER_CLASSES = f"{PROVIDER}_party_class"
PROVIDER_WHOLE = f"{PROVIDER}_party_whole"

# A donor's `(kind, class, whole)` cells as apis serves them. A residual's class is `residual`;
# a total holding more than one class has none, which data-mcp writes `untyped`.
OFFICIAL = ("party", "official", "false")
PRIVATE = ("party", "private", "false")
RESIDUAL = ("residual", "residual", "false")
GROUP_TOTAL = ("total", "official", "false")
OFFICIAL_WHOLE = ("total", "official", "true")
PRIVATE_WHOLE = ("total", "private", "true")


def provider_frame(
    donors: Mapping[str, tuple[tuple[str, str, str], float]],
    *,
    years: tuple[str, ...] = ("2021",),
    classes: bool = True,
) -> Frame:
    """ODA to Ethiopia from `donors`, each a `(cells, value)` pair, in every year of `years`.
    `classes=False` leaves out the class column, as a frame minted before it."""
    enrichment = [PROVIDER_KINDS, *([PROVIDER_CLASSES] if classes else []), PROVIDER_WHOLE]
    return Frame.build(
        role_keys=[PROVIDER, "financingRecipient", *enrichment],
        rows=[
            {
                "stat_var": ODA,
                PROVIDER: donor,
                "financingRecipient": "country/ETH",
                PROVIDER_KINDS: kind,
                **({PROVIDER_CLASSES: party_class} if classes else {}),
                PROVIDER_WHOLE: whole,
                "time_period": year,
                "value": value,
                "unit": "USD",
            }
            for year in years
            for donor, ((kind, party_class, whole), value) in donors.items()
        ],
    )
