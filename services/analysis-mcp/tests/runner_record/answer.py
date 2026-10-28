"""Answer recorded analysis-mcp requests with dfo-apis' own runner.

Runs in a dfo-apis checkout, from its root with `PYTHONPATH=.` so its `tests`
package imports (`record.py` runs it so). Reads the accumulated requests
(`record.py` writes them, each with the round it was first sent in) and writes
the answers: each request runs through `datasets_api.analysis.run_analysis` or
`run_anchor` over a root source holding the recorded frames, in round order and
over one set of stores, so a derived root finds the recipe of the call that
returned it. A refusal or a pending root is the analysis door's own body.

A root's frame is recorded under its pair and the columns it is read with, and a
root read with columns is the frame-layout read `analysis_roots` makes: the
recorded frame, and a mint record naming the frame pair those columns give
(`analysis_roots.frame_ref_for`, with no enrichment stamp).

A GET reads what the calls stored, after every call is answered, as the door
serving its path answers it: a derived frame's manifest and its Parquet file (`doors.derived`),
a frame's recipe and the frame a replay key names (`doors.recipe`). A refusal
is the problem document of its code, at the status the code registry maps.

    PYTHONPATH=. uv run python answer.py <requests.json> <replies.json>
"""

from __future__ import annotations

import base64
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from datasets_api.analysis import (
    AnalysisPending,
    AnalysisRequest,
    AnchorRequest,
    DatasetRoot,
    ResolvedRoot,
    SpecRoot,
    recorded_dataset_root,
    run_analysis,
    run_anchor,
)
from datasets_api.analysis_roots import frame_ref_for
from datasets_api.derived import (
    DerivedReads,
    manifest_body,
    read_derived_manifest,
    render_derived_file,
    root_rendering,
)
from datasets_api.doors.analysis import (
    _REFUSED,
    AnalysisReady,
    AnchorAnswer,
    LeftOutSeriesBody,
    _pending,
    _refused,
)
from datasets_api.doors.derived import DerivedManifest
from datasets_api.doors.files import selector_options
from datasets_api.doors.recipe import _read_recipe, _read_replay
from datasets_api.refusals import classify_datasets_refusal
from dfo_frame_grammar import Frame
from dfo_platform.config import PublicSiteSettings
from materialise.formats import read_canonical_frame
from tests.datasets_api._analysis_doubles import FakeRootSource, Stores

_DERIVED = re.compile(r"/api/datasets/derived/(ops_\w+)/(ocv_\w+)(/file)?")
_RECIPE = re.compile(r"/api/datasets/frames/(ops_\w+)/(ocv_\w+)/recipe")
_REPLAY = re.compile(r"/api/datasets/recipes/replays/(rpk_\w+)")
_SITE = PublicSiteSettings(public_site_url="https://dfo.test")


type _Rendered = tuple[str, str, tuple[str, ...]]


class _RenderedRoots(FakeRootSource):
    """`FakeRootSource` over frames by pair and by the columns each is read with."""

    def __init__(self, *, rendered: Mapping[_Rendered, Frame], stores: Stores) -> None:
        frames = {(spec_id, version): frame for (spec_id, version, _), frame in rendered.items()}
        super().__init__(frames=frames, newest=dict(frames), recipes=stores.recipe_store)
        self.rendered = dict(rendered)

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        resolved = super().resolve(root, latest=latest, trace_id=trace_id)
        columns = tuple(root.rendering.columns or ())
        dataset = resolved.dataset
        if not columns:
            return resolved
        frame = frame_ref_for(
            spec_id=dataset.spec_id,
            content_version=dataset.content_version,
            columns=columns,
            enrichment_stamp=None,
        )
        options = {**resolved.record["options"], "columns": list(columns)}
        record = {**resolved.record, "frame": frame.model_dump(), "options": options}
        return replace(
            resolved,
            frame=frame,
            record=record,
            rendering=root_rendering(options),
            source=columns,
        )

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        """The dataset root a stored recipe's root `record` names, read with the
        rendering its options state, as the live source reads it."""
        return recorded_dataset_root(record)

    def read(self, root: ResolvedRoot) -> Frame:
        if isinstance(root.source, tuple):
            key = (root.dataset.spec_id, root.dataset.content_version, root.source)
            if key in self.rendered:
                self.reads += 1
                return self.rendered[key]
        return super().read(root)


def _rendered_key(recorded: str) -> _Rendered:
    """A recorded frame's key, `<spec_id> <content_version> <columns JSON>`."""
    spec_id, content_version, *columns = recorded.split(" ", 2)
    return spec_id, content_version, tuple(json.loads(columns[0]) if columns else ())


def _door_answer(outcome: Any) -> dict[str, Any]:
    return {
        "status": outcome.status_code,
        "body": json.loads(outcome.body),
        "retry_after": outcome.headers.get("retry-after"),
    }


def _read(entry: dict[str, Any], *, stores: Stores) -> dict[str, Any]:
    """The answer to a recorded GET. See the module docstring."""
    path = entry["path"]
    reads = DerivedReads(
        snapshot_reader=stores.snapshot_reader,
        artifact_store=stores.artifact_store,
        recipe_reader=stores.recipe_store,
        now=lambda: stores.now,
    )
    try:
        if derived := _DERIVED.fullmatch(path):
            spec_id, content_version, file = derived.groups()
            if file:
                rendered = render_derived_file(
                    spec_id=spec_id,
                    content_version=content_version,
                    options=selector_options(parse_qsl(entry["query"])),
                    reads=reads,
                    site=_SITE,
                )
                content = base64.b64encode(rendered.data).decode()
                return {"status": 200, "content": content, "retry_after": None}
            facts = read_derived_manifest(
                spec_id=spec_id, content_version=content_version, reads=reads
            )
            body = DerivedManifest.model_validate(manifest_body(facts)).model_dump(mode="json")
        elif recipe := _RECIPE.fullmatch(path):
            spec_id, content_version = recipe.groups()
            body = _read_recipe(
                spec_id=spec_id, content_version=content_version, recipe_store=stores.recipe_store
            )
        elif replay := _REPLAY.fullmatch(path):
            body = _read_replay(
                replay_key=replay.group(1), recipe_store=stores.recipe_store
            ).model_dump(mode="json")
        else:
            raise SystemExit(f"no door answers GET {path}")
    except Exception as exc:
        spec = classify_datasets_refusal(exc)
        if spec is None:
            raise
        problem = {
            "code": spec.code,
            "retryable": spec.retryable,
            "detail": spec.detail or str(exc),
        }
        return {"status": spec.status, "body": problem, "retry_after": None}
    return {"status": 200, "body": body, "retry_after": None}


def _answer(entry: dict[str, Any], *, source: FakeRootSource, stores: Stores) -> dict[str, Any]:
    if entry.get("method") == "GET":
        return _read(entry, stores=stores)
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
    rendered = {
        _rendered_key(recorded): read_canonical_frame(base64.b64decode(data))
        for entry in requests.values()
        for recorded, data in entry["frames"].items()
    }
    stores = Stores()
    source = _RenderedRoots(rendered=rendered, stores=stores)
    # A GET reads what the calls stored, so it is answered after every call.
    ordered = sorted(
        requests.items(),
        key=lambda item: (item[1].get("method") == "GET", item[1]["round"], item[0]),
    )
    replies = {key: _answer(entry, source=source, stores=stores) for key, entry in ordered}
    Path(sys.argv[2]).write_text(json.dumps(replies, indent=1))
    refused = sum(reply["status"] != 200 for reply in replies.values())
    print(f"{len(replies)} answered, {refused} refused")


if __name__ == "__main__":
    main()
