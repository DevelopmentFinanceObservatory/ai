"""Graph unit refs that dfo-ai code writes or checks by name.

A frame's `unit` cell holds a unit ref as the graph names it, and verbs compare
units as exact strings, so a ref written here must match the form dfo-apis
puts in a frame it mints.
"""

from __future__ import annotations

from typing import Final

# The unit of a value stated in per cent (0-100). dfo-apis writes this bare
# ref, without a `dcid:` prefix, in the `unit` column of a published share
# such as World Development Indicators' "% of GDP" series. `share_of` writes
# it on every row it computes, so a computed share compares with a published
# one.
PERCENT_UNIT: Final = "Percent"
