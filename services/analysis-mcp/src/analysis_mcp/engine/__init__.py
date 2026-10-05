"""The verb engines: Arrow kernels over `Frame.table`, with per-group and
per-distinct-value arithmetic in Python. No numpy, no pandas, no SQL.

`analysis_mcp.engine.common` holds what every verb needs before it computes
(narrowing a frame to the caller's scope, resolving unit/facet
heterogeneity, recording exclusions). `analysis_mcp.engine.columnar` holds
the Arrow kernels the verbs share (grouping, masks, output columns and
frames). `aggregate`, `compare`, `filter`, `growth`, `join`, `rank`, and
`share_of` are the seven verb engines built on top of them.
"""

from __future__ import annotations
