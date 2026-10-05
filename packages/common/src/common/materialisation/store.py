"""The materialisation seam: the store protocol data-mcp's `get_data` writes
frames through and every analysis-mcp verb reads them back from (and writes
its own results back through, keyed by its own `ops_`/`ocv_` pair).

`MaterialisationStore` is a `typing.Protocol`, not a base class -- there is
no shared implementation here, only the shape. Two implementations ship:
`InMemoryMaterialisationStore` and the bucket-backed `GcsMaterialisationStore`,
chosen by `build_store`. Writers and readers see only the holder
(`set_store`/`get_store`/`aclose_store`, built in each service's own FastMCP
`lifespan`), so the choice is invisible to them.

`put`/`get`/`open` are keyed by `FrameRef` -- content-addressed, not
session-scoped: a pair names the exact frame it names regardless of who
registered it or who is asking, so this protocol's job is "give back whatever
frame this pair names," with no ownership check gating the read.

`open` returns a `HeldFrame`: the pair found, with its row count when the store
recorded one, before any of its bytes are read. A caller claims rows from the
process's frame row budget (`common.row_budget`) before it calls `HeldFrame.read`,
the recorded count or, without one, its own row ceiling, and passes the claim as
`read`'s `max_rows`, so the download, the checks and the decode all run under a
claim no decode exceeds. `get` is `open` followed by `read` up to the store's own
row ceiling, for a caller that claims nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from common.materialisation.frame_ref import FrameRef
from common.materialisation.trace import StoredFrameTrace

if TYPE_CHECKING:
    from dfo_frame_grammar import Frame


class FrameIntegrityError(Exception):
    """A `get` reached a stored object that fails an integrity check: the
    recorded sha256 of its own bytes doesn't match what was actually read
    back, its own envelope names a different reference than the one
    requested, or (when a store checks it) a recomputed `content_version`
    doesn't match the value the check compares against. Every cause means
    the object itself is wrong, not merely large, so a caller sees this as
    a `do_not_retry` refusal naming the reference -- retrying the identical
    reference reaches the identical wrong object. Distinct from
    `FrameTooLarge`: that names bytes that are exactly what `put` wrote,
    just over a size-shaped ceiling this store enforces before decoding."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.ref = ref
        self.reason = reason
        super().__init__(
            f"frame integrity check failed: {ref.spec_id}, {ref.content_version}: {reason}"
        )


class FrameTooLarge(Exception):  # noqa: N818 -- named for the refusal it maps to, not a generic "Error"
    """A `get` refused a stored object before decoding it: its encoded
    byte size, row count, column count, or row-group uncompressed size
    exceeded this store's own ceiling. The stored bytes are exactly what
    `put` wrote -- nothing about them is corrupt or mismatched, they are
    simply too large for this store's contract to serve back, so a caller
    sees this as the retryable `frame_too_large` refusal: a narrower
    `frame_ref` can succeed, unlike `FrameIntegrityError`'s `do_not_retry`.
    """

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.ref = ref
        self.reason = reason
        super().__init__(f"frame too large: {ref.spec_id}, {ref.content_version}: {reason}")


class FrameExpired(Exception):  # noqa: N818 -- named for the refusal it maps to, not a generic "Error"
    """A `get` reached a pair whose frame aged out, measured from the pair's first
    `put` -- retention is absolute, not extended by later reads. Distinct from `get`
    returning `None`, which covers both a pair never registered and one evicted under
    the implementation's own entry cap: `FrameExpired` means the pair WAS held, and
    this implementation's own ageout policy is the only reason it's gone -- safe to
    disclose, since naming the pair at all already means the caller obtained it from a
    real prior tool response."""

    def __init__(self, ref: FrameRef) -> None:
        self.ref = ref
        super().__init__(f"frame expired: {ref.spec_id}, {ref.content_version}")


class FrameReader(Protocol):
    """`HeldFrame.read`'s signature."""

    def __call__(self, *, max_rows: int) -> Awaitable[tuple[Frame, StoredFrameTrace]]: ...


@dataclass(frozen=True, kw_only=True)
class HeldFrame:
    """A pair a store holds, found before any of its bytes are read.

    `num_rows` is the count `put` recorded beside the object, or `None` for an object
    stored without one. `read(max_rows=...)` downloads the object, runs every check
    the store makes, and returns the frame with its trace. It raises `FrameTooLarge`
    before decoding when the object holds more than `max_rows` rows, the rows its
    caller claimed, or fails one of the store's own ceilings; `FrameIntegrityError`
    for an object that fails an integrity check; and `FrameExpired` for one deleted
    since `open` found it.
    """

    num_rows: int | None
    read: FrameReader


class MaterialisationStore(Protocol):
    """Persists and retrieves a `(Frame, StoredFrameTrace)` pair by `FrameRef`."""

    async def put(
        self, ref: FrameRef, *, frame: Frame, trace: StoredFrameTrace, writer: str
    ) -> None:
        """Idempotent: if `ref` is already held, keep the stored frame and
        trace exactly as first written -- never overwritten, and retention
        runs from that first write, not from this call. `writer` records only
        the identity of the first `put` to land for `ref` (the MCP caller,
        `common.caller.current_writer`, or any stable string); a later
        `put` under the same `ref`, from any other writer, does not change
        what's recorded. Nothing in this codebase reads it back today --
        it's kept on the signature for a later retrieval-route entitlement
        check."""
        ...

    async def get(self, ref: FrameRef) -> tuple[Frame, StoredFrameTrace] | None:
        """`None` means the pair is not currently held: either it was never
        registered, or it was evicted under this implementation's own entry
        cap. A read never extends retention -- a pair ages out a fixed
        interval after its first `put`, however often it's read in between.
        Raises `FrameExpired` for a pair that WAS held and aged out past this
        implementation's own retention window -- disclosing "this existed and
        expired" is safe once a caller already names the exact pair, and
        stays true on every read of the same expired pair until a `put`
        heals it or capacity pressure purges it."""
        ...

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        """`get` up to the first byte of the frame: the same `None` and `FrameExpired`,
        and `FrameTooLarge` or `FrameIntegrityError` only for what the store knows
        before it reads the object. Every other check runs in `HeldFrame.read`."""
        ...

    async def aclose(self) -> None:
        """Release any resources this store holds (connections, clients)."""
        ...
