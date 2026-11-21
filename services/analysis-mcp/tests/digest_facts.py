"""The facts a digest renderer reads, as the digest tests build them.

`facts` builds a summary facts model (`analysis_mcp.summary.AggregateFacts` and the
rest) from the fields a test names, read through their JSON form as a runner
summary's `result` is. A field the model does not hold, such as a frame, is left
out.

`computed` is the facts dfo-apis' engine computes for one verb over a test's
frames, replayed from `fixtures/digest_engine_facts.json`, keyed by the verb, the
Parquet bytes of each frame and the RFC 8785 form of the arguments. To record
facts a test asks for and the fixture lacks, run the tests with
`DIGEST_FACTS_RECORD=<path>`: each request is written there with its frames as
base64 Parquet, and the test fails. Compute the facts in a dfo-apis checkout with
`runner_record/engine_facts.py`, merge them into the fixture, and run again.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import rfc8785
from analysis_mcp.summary import (
    AggregateFacts,
    QuotientOfFacts,
    RankFacts,
    ShareOfFacts,
)
from common.materialisation.frame_codec import frame_to_parquet_bytes
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest
from pydantic import BaseModel

FACTS_PATH = Path(__file__).resolve().parent / "fixtures" / "digest_engine_facts.json"

# The facts model of each verb `computed` replays.
_MODELS: Mapping[str, type[BaseModel]] = {
    "aggregate": AggregateFacts,
    "rank": RankFacts,
    "share_of": ShareOfFacts,
    "quotient_of": QuotientOfFacts,
}


def _json(value: object) -> object:
    """`value` as JSON: a model as its JSON dump, a unit in its pin form."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, UnitRequest | ComposedUnit):
        return value.as_json()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    return value


def facts[T: BaseModel](model: type[T], **fields: object) -> T:
    """`model` holding `fields`, each through its JSON form. A frame is left out."""
    return model.model_validate(
        {name: _json(value) for name, value in fields.items() if not isinstance(value, Frame)}
    )


def _key(verb: str, frames: tuple[Frame, ...], arguments: Mapping[str, Any]) -> str:
    digests = [hashlib.sha256(frame_to_parquet_bytes(frame)).hexdigest() for frame in frames]
    body = {"frames": digests, "arguments": _json(arguments)}
    return f"{verb} {rfc8785.dumps(body).decode()}"


def _record(key: str, verb: str, frames: tuple[Frame, ...], arguments: Mapping[str, Any]) -> None:
    """Write the request under `key` to `DIGEST_FACTS_RECORD`, one file per
    pytest-xdist worker."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    path = Path(os.environ["DIGEST_FACTS_RECORD"] + (f".{worker}" if worker else ""))
    written = json.loads(path.read_text()) if path.exists() else {}
    written[key] = {
        "verb": verb,
        "frames": [base64.b64encode(frame_to_parquet_bytes(frame)).decode() for frame in frames],
        "arguments": _json(arguments),
    }
    path.write_text(json.dumps(written, indent=1, sort_keys=True))


def computed(verb: str, *frames: Frame, **arguments: Any) -> Any:
    """The facts dfo-apis' engine computes for `verb` over `frames` with
    `arguments`, the keyword arguments of its `run_<verb>`."""
    key = _key(verb, frames, arguments)
    recorded = json.loads(FACTS_PATH.read_text()) if FACTS_PATH.exists() else {}
    if key not in recorded:
        if os.environ.get("DIGEST_FACTS_RECORD"):
            _record(key, verb, frames, arguments)
        pytest.fail(f"no engine facts are recorded for {key}")
    return _MODELS[verb].model_validate(recorded[key])
