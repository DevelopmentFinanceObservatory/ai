"""The labels of breakdown cells, for text a model or a reader sees.

A breakdown column `stat_var_<axis>` (`stat_var_crsPurpose`) holds the code each row sums
within. dfo-apis writes that cell as the code's label (`civilian peace-building, conflict
prevention and resolution`) or as the code's dcid (`CRSPurposeCode15220`). Text shows a
dcid cell as its label with the dcid after it (`civilian peace-building, conflict prevention
and resolution (CRSPurposeCode15220)`), so a reader still holds the value a selection or a
filter binds, and a label cell as it is.

`label_cells` reads the dcid cells through dfo-apis' `POST /api/graph/nodes`, in batches of
`NODE_BATCH_DCIDS`, and keeps a label only from a `code` node of the cell's own axis
(`code.property`). Any other cell keeps its value: a label cell, which is no dcid or no code
of the axis, and a dcid the graph does not hold or could not be read. The graph names a code
with its system after it (`... (CRS Purpose)`). The label drops that suffix, so a dcid cell
reads as the label cell dfo-apis writes from the code index.

Each service passes its own `ReadNodes`, its client's post of one batch. A read that fails
returns no entries, so text falls back to the cell values and the answer it labels stands. At most `MAX_LABELLED_CELLS` distinct dcids are read per call, which bounds one answer to
`MAX_LABELLED_CELLS / NODE_BATCH_DCIDS` node requests.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.dcid import bare_dcid, has_dot_segment, is_dcid

BREAKDOWN_COLUMN_PREFIX = "stat_var_"

# dfo-apis' own limit on one `POST /api/graph/nodes` (`caps.max_node_batch_dcids`).
NODE_BATCH_DCIDS = 50
MAX_LABELLED_CELLS = 4 * NODE_BATCH_DCIDS

NODES_PATH = "/api/graph/nodes"

# How long one nodes read may take. A label only names a cell, so a slow graph costs the
# labels and leaves the answer's own deadline to the answer.
NODES_READ_TIMEOUT_S = 2.5

type Cell = tuple[str, str]
# Reads one batch of at most `NODE_BATCH_DCIDS` bare dcids, answering the node entries the
# route returned, or none when the read failed.
type ReadNodes = Callable[[list[str]], Awaitable[Sequence[Mapping[str, Any]]]]


def breakdown_axis(column: str) -> str | None:
    """The axis a breakdown column names (`crsPurpose` for `stat_var_crsPurpose`), or None
    for any other column."""
    axis = column.removeprefix(BREAKDOWN_COLUMN_PREFIX)
    return axis if axis and axis != column else None


@dataclass(frozen=True)
class CellLabels:
    """The label of each breakdown cell read as a code of its axis, by `(column, value)`."""

    labels: Mapping[Cell, str] = field(default_factory=dict)

    def label(self, column: str, value: str) -> str | None:
        """The label read for a cell, or None for a cell with none."""
        return self.labels.get((column, value))

    def text(self, column: str, value: str) -> str:
        """The text a cell shows: its label with its value after it where one was read
        (`basic education (CRSPurposeCode11220)`), else its value."""
        label = self.label(column, value)
        return value if label is None else f"{label} ({value})"


NO_LABELS = CellLabels()


def _labellable(cell: Cell) -> bool:
    column, value = cell
    dcid = bare_dcid(value)
    return breakdown_axis(column) is not None and is_dcid(dcid) and not has_dot_segment(dcid)


def code_label(node: Mapping[str, Any], *, axis: str) -> str | None:
    """The label of `node`, a `/api/graph/nodes` entry, when it is a code of `axis`, without
    the ` (<code system>)` the graph adds after it, or None for any other node."""
    code = node.get("code")
    label = node.get("label")
    if node.get("kind") != "code" or not isinstance(code, Mapping) or not isinstance(label, str):
        return None
    if code.get("property") != axis:
        return None
    system = code.get("code_system")
    system_label = system.get("label") if isinstance(system, Mapping) else None
    if isinstance(system_label, str):
        label = label.removesuffix(f" ({system_label})")
    return label or None


async def label_cells(
    cells: Iterable[Cell],
    *,
    read_nodes: ReadNodes,
    max_cells: int = MAX_LABELLED_CELLS,
) -> CellLabels:
    """The labels of those of `cells` that are code dcids of their breakdown axis.

    Only the first `max_cells` distinct dcids, in the order `cells` names them, are read,
    in concurrent batches of `NODE_BATCH_DCIDS`. A cell past them keeps its value.
    """
    wanted = [cell for cell in dict.fromkeys(cells) if _labellable(cell)]
    dcids = list(dict.fromkeys(bare_dcid(value) for _, value in wanted))[:max_cells]
    if not dcids:
        return NO_LABELS
    batches = [list(batch) for batch in itertools.batched(dcids, NODE_BATCH_DCIDS, strict=False)]
    answers = await asyncio.gather(*(read_nodes(batch) for batch in batches))
    nodes = {
        str(node.get("ref")): node
        for answer in answers
        for node in answer
        if isinstance(node, Mapping)
    }
    labels: dict[Cell, str] = {}
    for column, value in wanted:
        node = nodes.get(bare_dcid(value))
        axis = breakdown_axis(column)
        if node is None or axis is None:
            continue
        if (label := code_label(node, axis=axis)) is not None:
            labels[column, value] = label
    return CellLabels(labels)
