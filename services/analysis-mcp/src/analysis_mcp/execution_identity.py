"""The execution identity: the engine-side third of a toolkit result's identity.

`op_spec_hash` and a result's `content_version` name *what* was asked and
*what came back*; `ExecutionIdentity` names *which engine computed it* --
the engine's build, the frame grammar it validated against, and the
reduction policy its arithmetic followed. Any change that could move a
computed number moves one of these three fields, and `content_version` is
minted over them (`identity.mint_toolkit_content_version`).

dfo-apis' analysis runner computes every verb and states its identity in each
summary (`summary.VerbSummary.execution_identity`) and in apis' capabilities
document (`common.live_apis.LiveApisClient.analysis_execution_identity`), which
`harness.recipe_replay` reads to predict the pair a recomputed step mints. A
stored frame that an in-process analysis-mcp verb computed carries that engine's
identity in its trace (`trace.Trace.execution_identity`).
"""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


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
