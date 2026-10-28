"""Answer recorded analysis-mcp requests with dfo-apis' own runner.

Runs in a dfo-apis checkout, from its root with `PYTHONPATH=.` so its `tests`
package imports (`record.py` runs it so). Reads the accumulated requests
(`record.py` writes them, each with the round it was first sent in) and writes
the answers: each request runs through `datasets_api.analysis.run_analysis` or
`run_anchor` over a root source holding the recorded frames, in round order and
over one set of stores, so a derived root finds the recipe of the call that
returned it. A refusal or a pending root is the analysis door's own body.

    PYTHONPATH=. uv run python answer.py <requests.json> <replies.json>
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

from datasets_api.analysis import (
    AnalysisPending,
    AnalysisRequest,
    AnchorRequest,
    run_analysis,
    run_anchor,
)
from datasets_api.doors.analysis import (
    _REFUSED,
    AnalysisReady,
    AnchorAnswer,
    LeftOutSeriesBody,
    _pending,
    _refused,
)
from materialise.formats import read_canonical_frame
from tests.datasets_api._analysis_doubles import FakeRootSource, Stores


def _door_answer(outcome: Any) -> dict[str, Any]:
    return {
        "status": outcome.status_code,
        "body": json.loads(outcome.body),
        "retry_after": outcome.headers.get("retry-after"),
    }


def _answer(entry: dict[str, Any], *, source: FakeRootSource, stores: Stores) -> dict[str, Any]:
    body = entry["body"]
    try:
        if entry["path"].endswith("/anchor"):
            anchor, roots = run_anchor(AnchorRequest.model_validate(body), source=source)
            answer = AnchorAnswer(
                period=anchor.period,
                rule=anchor.rule,
                left_out=[
                    LeftOutSeriesBody(root=series.root, coordinates=dict(series.coordinates))
                    for series in anchor.left_out
                ],
                roots=[root.dataset.model_dump() for root in roots],
            ).model_dump(mode="json")
        else:
            request = AnalysisRequest.model_validate(body)
            result = run_analysis(request, source=source, stores=stores.analysis())
            answer = AnalysisReady.model_validate(result.body()).model_dump(mode="json")
    except AnalysisPending as pending:
        return _door_answer(_pending(pending))
    except _REFUSED as refused:
        return _door_answer(_refused(refused))
    return {"status": 200, "body": answer, "retry_after": None}


def main() -> None:
    requests = json.loads(Path(sys.argv[1]).read_text())
    frames = {
        tuple(pair.split(" ")): read_canonical_frame(base64.b64decode(data))
        for entry in requests.values()
        for pair, data in entry["frames"].items()
    }
    stores = Stores()
    source = FakeRootSource(
        frames=frames,
        newest=dict(frames.keys()),
        recipes=stores.recipe_store,
    )
    ordered = sorted(requests.items(), key=lambda item: (item[1]["round"], item[0]))
    replies = {key: _answer(entry, source=source, stores=stores) for key, entry in ordered}
    Path(sys.argv[2]).write_text(json.dumps(replies, indent=1))
    refused = sum(reply["status"] != 200 for reply in replies.values())
    print(f"{len(replies)} answered, {refused} refused")


if __name__ == "__main__":
    main()
