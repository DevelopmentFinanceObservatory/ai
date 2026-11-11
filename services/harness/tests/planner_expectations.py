"""Stored expectations for the tests that run one planner script and check its output.

Each test runs the Pydantic AI planner and checks what the run produced
against one JSON file, `fixtures/planner_runs/<name>.json`, named for the test
module, the test and its parameters (`expectation_name`).

A value many files would repeat whole, such as a request's config with the
planner's instruction and tool declarations, is stored once as
`fixtures/planner_runs/shared/<name>.json`, named for what it holds, and a file
holds `{"shared": "<name>"}` in its place (`shared`). Comparisons put each such
value back first, so they stay whole.

Two kinds of value are stored in a fixed form, so that an edit to them moves no
file (`as_json`): the seam's brief, harness prose with tests of its own, is
stored as `BRIEF`, and a package version in a request's HTTP headers as
`<version>` (`versionless_headers`).

Rewrite the files after a change that moves what the planner produces, then run
the same tests again without the variable to check them:

    rm -rf services/harness/tests/fixtures/planner_runs && PLANNER_EXPECTATIONS=write \\
        uv run --package harness pytest services/harness/tests/test_planner_node.py \\
        services/harness/tests/test_planner_turns.py \\
        services/harness/tests/test_planner_events.py \\
        services/harness/tests/planner/test_recorded_requests.py \\
        services/harness/tests/planner/test_pydantic_agent.py \\
        services/harness/tests/planner/test_pydantic_endings.py \\
        services/harness/tests/telemetry/test_planner_spans.py

A write run fails each test whose file it wrote or changed, so it cannot pass
while an expectation changed.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from harness.planner_seam import _READINGS_CONTEXT

PLANNER_RUNS = Path(__file__).parent / "fixtures" / "planner_runs"
_SHARED = PLANNER_RUNS / "shared"
_WRITING = os.environ.get("PLANNER_EXPECTATIONS") == "write"

# What a stored value holds in place of the seam's brief, the text the planner is
# shown after the turn's question (`planner.pydantic_agent.run_planner`), which opens
# with `BRIEF_OPENING`.
BRIEF = "<brief>"
BRIEF_OPENING = _READINGS_CONTEXT.partition(".")[0] + "."
# A package version in an HTTP header: `pydantic-ai/2.66.0`, `gl-python/3.13.14`.
_PACKAGE_VERSION = re.compile(r"\b([A-Za-z][\w.-]*)/\d[\w.+-]*")

# Each shared value this session produced, by name.
_PRODUCED: dict[str, Any] = {}
# The shared files a write run changed since the last `expect`.
_CHANGED: list[Path] = []


def expectation_name(request: pytest.FixtureRequest) -> str:
    """`<module>/<test>` for an unparametrised test, `<module>/<test>/<ids>` for a
    parametrised one."""
    node = request.node
    name = f"{request.module.__name__.rsplit('.', 1)[-1]}/{node.originalname}"
    callspec = getattr(node, "callspec", None)
    if callspec is None:
        return name
    return f"{name}/{''.join(c if c.isalnum() or c in '._-' else '_' for c in callspec.id)}"


def as_json(value: Any) -> Any:
    """`value` as a JSON file holds it: tuples as lists, any mapping as an object,
    and each string that opens with `BRIEF_OPENING` as `BRIEF`."""
    return _marked(json.loads(json.dumps(value, default=_mapping, allow_nan=False)))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not JSON: {value!r}")


def _marked(value: Any) -> Any:
    if isinstance(value, str):
        return BRIEF if value.startswith(BRIEF_OPENING) else value
    if isinstance(value, dict):
        return {key: _marked(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_marked(item) for item in value]
    return value


def versionless_headers(config: Mapping[str, Any]) -> dict[str, Any]:
    """`config`, a dumped request config, with each package version in its HTTP
    headers put as `<version>`, the header names and the rest kept."""
    options = config.get("http_options")
    if not options or "headers" not in options:
        return dict(config)
    headers = {
        name: _PACKAGE_VERSION.sub(r"\1/<version>", value)
        for name, value in options["headers"].items()
    }
    return {**config, "http_options": {**options, "headers": headers}}


def _dumped(value: Any) -> str:
    """The file's text for `value`, in the form the `pretty-format-json` hook keeps."""
    return json.dumps(value, indent=2, ensure_ascii=True) + "\n"


def shared(value: Any, name: str) -> dict[str, str]:
    """A reference to `value`, stored once as `shared/<name>.json`.

    A write run fails when one session gives two different values one name.
    """
    produced = as_json(value)
    if _WRITING:
        if name in _PRODUCED and _PRODUCED[name] != produced:
            pytest.fail(f"two values named shared/{name}.json", pytrace=False)
        path = _SHARED / f"{name}.json"
        text = _dumped(produced)
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            _CHANGED.append(path)
    _PRODUCED[name] = produced
    return {"shared": name}


def resolved(value: Any, *, produced: bool = False) -> Any:
    """`value` with each `shared` reference put back as the value it stands for:
    the stored one, or with `produced` the one this session last produced."""
    if isinstance(value, dict):
        if value.keys() == {"shared"} and isinstance(value["shared"], str):
            name = value["shared"]
            if produced:
                return _PRODUCED[name]
            path = _SHARED / f"{name}.json"
            if not path.exists():
                pytest.fail(f"no shared value {path.relative_to(PLANNER_RUNS)}", pytrace=False)
            return json.loads(path.read_text())
        return {key: resolved(item, produced=produced) for key, item in value.items()}
    if isinstance(value, list):
        return [resolved(item, produced=produced) for item in value]
    return value


def expected(name: str) -> Any:
    """The stored expectation `name`, its `shared` references put back."""
    path = PLANNER_RUNS / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"no stored expectation {path.relative_to(PLANNER_RUNS)}: "
            "write it with PLANNER_EXPECTATIONS=write",
            pytrace=False,
        )
    return resolved(json.loads(path.read_text()))


def assert_json_equal(produced: Any, stored: Any, *, name: str) -> None:
    """Fail with the difference of the two when `produced` as JSON differs from
    `stored`, either side's `shared` references named alike and put back."""
    produced = as_json(produced)
    whole = resolved(produced, produced=True)
    if produced == stored and whole == resolved(stored):
        return
    diff = difflib.unified_diff(
        _dumped(resolved(stored)).splitlines(),
        _dumped(whole).splitlines(),
        fromfile=f"{name} (stored)",
        tofile=f"{name} (produced)",
        lineterm="",
    )
    pytest.fail("\n".join(diff) or f"{name}: a shared value is named differently", pytrace=False)


def expect(name: str, value: Any) -> None:
    """Assert that `value` equals the stored expectation `name`.

    With `PLANNER_EXPECTATIONS=write`, write `value` to the file, and fail when
    that, or a `shared` value written for it, changed a file.
    """
    produced = as_json(value)
    path = PLANNER_RUNS / f"{name}.json"
    if _WRITING:
        changed = [*_CHANGED]
        _CHANGED.clear()
        text = _dumped(produced)
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            changed.append(path)
        if changed:
            wrote = ", ".join(str(each.relative_to(PLANNER_RUNS)) for each in changed)
            pytest.fail(f"wrote {wrote}", pytrace=False)
        return
    if not path.exists():
        expected(name)
    assert_json_equal(produced, json.loads(path.read_text()), name=name)
