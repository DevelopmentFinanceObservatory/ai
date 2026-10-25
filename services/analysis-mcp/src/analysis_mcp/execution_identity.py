"""The execution identity: the engine-side third of a toolkit result's identity.

`op_spec_hash` and a result's `content_version` name *what* was asked and
*what came back*; `ExecutionIdentity` names *which engine computed it* --
the toolkit's own build, the frame grammar it validated against, and the
reduction policy its arithmetic followed. Any change that could move a
computed number moves one of these three fields; a test that perturbs the
engine and watches `content_version` move confirms the binding holds.

`frame_grammar_version` is read from `dfo_frame_grammar.FRAME_GRAMMAR_VERSION`
rather than pinned here, so a frame-grammar bump is picked up automatically
instead of drifting out of sync with a hand-copied literal.
"""

from __future__ import annotations

from dfo_frame_grammar import FRAME_GRAMMAR_VERSION
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# The engine's own version, hand-set as a literal.
ENGINE_VERSION = "0.9.0"

# The reduction policy's version -- separate from `ENGINE_VERSION` because a
# reduction policy (e.g. "skip nulls, use fsum") can change independently of
# the engine's own build. At 0.2.0: a row leaving `unit` unbound is excluded
# from every verb's own computation once it resolves to a real unit. The
# exclusion runs on `aggregate`'s unpinned path (matching what its pinned path
# already did), then extends to `rank`, `share_of`, `growth`, `join`, and
# `compare`.
REDUCTION_POLICY_VERSION = "0.2.0"


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class ExecutionIdentity:
    """The engine-version binding of a toolkit result's identity.

    Frozen and closed to exactly these three fields -- nothing else describes
    "which engine computed this."
    """

    engine_version: str
    frame_grammar_version: str
    reduction_policy_version: str

    def canonical_string(self) -> str:
        """The pipe-joined form hashed into a toolkit result's `content_version`.

        A plain string, not a nested JSON object: the three fields already
        have a fixed order and none may contain a `|` (dotted/semver version
        strings only), so this is an unambiguous, order-independent-by-
        construction encoding -- simpler than asking the JCS preimage builder
        to sort or validate a fourth nested object.
        """
        return f"{self.engine_version}|{self.frame_grammar_version}|{self.reduction_policy_version}"


def current_execution_identity() -> ExecutionIdentity:
    """The engine's execution identity as of this build."""
    return ExecutionIdentity(
        engine_version=ENGINE_VERSION,
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        reduction_policy_version=REDUCTION_POLICY_VERSION,
    )
