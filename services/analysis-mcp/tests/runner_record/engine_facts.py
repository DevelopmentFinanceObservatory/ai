"""Compute the digest facts the digest tests ask for with dfo-apis' own engine.

Runs in a dfo-apis checkout, from its root, like `answer.py`. Reads the requests
`digest_facts.computed` writes under `DIGEST_FACTS_RECORD`, runs each through its
verb's `run_<verb>` in `dfo_analysis_engine.engine`, and writes the facts the
runner states in a summary's `result` (`dfo_analysis_engine.calls._facts`) beside
the fixture's own, by the request's key.

    uv run python engine_facts.py <requests.json> <fixtures/digest_engine_facts.json>
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

from dfo_analysis_engine.calls import _facts
from dfo_analysis_engine.engine.aggregate import run_aggregate
from dfo_analysis_engine.engine.quotient_of import run_quotient_of
from dfo_analysis_engine.engine.rank import run_rank
from dfo_analysis_engine.engine.share_of import run_share_of
from materialise.formats import read_canonical_frame

_RUNS: dict[str, Any] = {
    "aggregate": run_aggregate,
    "rank": run_rank,
    "share_of": run_share_of,
    "quotient_of": run_quotient_of,
}


def main(requests_path: str, facts_path: str) -> None:
    requests = json.loads(Path(requests_path).read_text())
    written = Path(facts_path)
    facts = json.loads(written.read_text()) if written.exists() else {}
    for key, request in requests.items():
        frames = [read_canonical_frame(base64.b64decode(data)) for data in request["frames"]]
        result = _RUNS[request["verb"]](*frames, **request["arguments"])
        facts[key] = _facts(result, top=True)
    written.write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
