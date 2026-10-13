"""Rename `dcid` to `ref` on every Search graph reference in dfo-ai's recorded JSON fixtures.

Search serves and accepts a graph reference as `{ref, label}`, the shape the Graph API
serves. The script rewrites the JSON files under `services/*/tests` and the planner entry
eval's recordings (`evals/planner_entry/recordings/*.json`) in place, renaming the key only
at the paths below. Every other `dcid` stays: Graph node and Graph resolve
fixtures (already `{ref, label}`), the flat `ref` of the catalogue and Datasets
manifests, `stat_var` columns, graph request logs, and data-mcp's own refusal payloads.

Rules:

- A Spec is found where an envelope holds one (`parts[].spec`, `parts[].candidates.specs[]`,
  `parts[].no_data.flagged_match`, `parts[].no_data.alternate_match`,
  `parts[].no_data.nearest_real[]`), and where a fixture holds one directly: a document
  with `resolved`, its `spec`, `specs` (list or map) or `seed_spec[].spec`, or a top-level
  map of envelopes. Under the Spec's `resolved`, `SPEC_RESOLVED_PATHS` are renamed, and
  each `resolved.equivalent_readings[]` is a Spec of its own.
- A document holding `parts` or `scope` is an envelope; `ENVELOPE_PATHS` are renamed.
- A document holding `input` is a request; `REQUEST_PATHS` are renamed.
- A document holding `searches` is a planner entry recording; each `searches[].envelope`
  is an envelope.
- A node holding both `dcid` and `ref` is refused, and so is a file whose text is not
  `json.dumps(indent=2)` output, with or without `sort_keys`, since rewriting it would
  reformat the whole file. A file written with sorted keys is rewritten with them.

The script is idempotent. Re-run it after replaying the branch onto a base that records
new fixtures, then run the tests:

    uv run --no-sync python scripts/codemod_graphref_ref.py --check   # list, exit 1 if any
    uv run --no-sync python scripts/codemod_graphref_ref.py           # rewrite

Python test literals are out of its reach; find them with `rg '"dcid"' services/*/tests`.
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Paths under a Spec's `resolved` whose node is a graph reference. `[]` steps into each
# list item, `*` into each map value.
SPEC_RESOLVED_PATHS = [
    "shape.population_type",
    "shape.measured_property",
    "shape.stat_type",
    "shape.measurement_qualifier",
    "shape.measurement_denominator",
    "shape.slot_keys[].property",
    "slots[].key.property",
    "slots[].binding.value.ref",
    "slots[].binding.values[].ref",
    "slots[].binding.set_ref",
    "slots[].expansion.added[]",
    "slot_options[].key.property",
    "slot_options[].options[].value.ref",
    "entity_options[].options[].value.ref",
    "facet_options[].options[].value.ref",
    "entity_groups.*",
    "measure_axis_values[].value",
    "alternative_measures[].measure_axis_values[].value",
    "open_axis_facts[].value",
    "open_axis_facts[].values[].value",
    "related_group",
    "related_measures[].stat_var",
    "observations.series[].source_refs[]",
    "trace.resolved_sources[]",
    "trace.resolved_entities[]",
    "trace.slot_filters[].key.property",
    "trace.slot_filters[].refs[]",
    "trace.query_qualifiers[].named[]",
    "trace.query_qualifiers[].selected[]",
]

ENVELOPE_PATHS = [
    "scope.place",
    "parts[].no_data.ambiguous_entities[].candidates[]",
    "parts[].no_data.denominator.asked",
    "parts[].no_data.denominator.found",
    "parts[].no_data.unbound_qualifiers[].named[]",
    "parts[].no_data.unbound_qualifiers[].selected[]",
]

REQUEST_PATHS = [
    "input.entries[].entity_groups.*",
    "input.entries[].candidates[].entity_groups.*",
]

# Where an envelope holds a Spec.
ENVELOPE_SPEC_PATHS = [
    "parts[].spec",
    "parts[].candidates.specs[]",
    "parts[].no_data.flagged_match",
    "parts[].no_data.alternate_match",
    "parts[].no_data.nearest_real[]",
]


def _steps(path: str) -> list[str]:
    steps = []
    for segment in path.split("."):
        steps.append(segment.removesuffix("[]"))
        if segment.endswith("[]"):
            steps.append("[]")
    return steps


def _select(node: Any, path: str) -> Iterator[Any]:
    nodes = [node]
    for step in _steps(path):
        following = []
        for current in nodes:
            if step == "[]":
                if isinstance(current, list):
                    following.extend(current)
            elif step == "*":
                if isinstance(current, dict):
                    following.extend(current.values())
            elif isinstance(current, dict) and step in current:
                following.append(current[step])
        nodes = following
    yield from nodes


def _rename(ref: Any) -> int:
    if not isinstance(ref, dict) or "dcid" not in ref:
        return 0
    if "ref" in ref:
        raise ValueError(f"a graph reference carries both dcid and ref: {ref!r}")
    renamed = {("ref" if key == "dcid" else key): value for key, value in ref.items()}
    ref.clear()
    ref.update(renamed)
    return 1


def _spec(spec: Any) -> int:
    resolved = spec.get("resolved") if isinstance(spec, dict) else None
    if not isinstance(resolved, dict):
        return 0
    count = sum(_rename(ref) for path in SPEC_RESOLVED_PATHS for ref in _select(resolved, path))
    return count + sum(_spec(reading) for reading in _select(resolved, "equivalent_readings[]"))


def _envelope(envelope: dict[str, Any]) -> int:
    count = sum(_spec(spec) for path in ENVELOPE_SPEC_PATHS for spec in _select(envelope, path))
    return count + sum(_rename(ref) for path in ENVELOPE_PATHS for ref in _select(envelope, path))


def _document(document: Any) -> int:
    """Rename in a fixture: an envelope, a request, a Spec, a seed, a planner entry
    recording, or a map or list of them."""
    if isinstance(document, list):
        return sum(_document(item) for item in document)
    if not isinstance(document, dict):
        return 0
    if "parts" in document or "scope" in document:
        return _envelope(document)
    if "input" in document:
        return sum(_rename(ref) for path in REQUEST_PATHS for ref in _select(document, path))
    if "resolved" in document:
        return _spec(document)
    count = 0
    for key in ("spec", "seed_spec"):
        if key in document:
            count += _document(document[key])
    if "specs" in document:
        specs = document["specs"]
        count += _document(list(specs.values()) if isinstance(specs, dict) else specs)
    if "searches" in document:
        count += _document(list(_select(document, "searches[].envelope")))
    if count == 0 and not ({"spec", "seed_spec", "specs", "searches"} & document.keys()):
        count = sum(_document(value) for value in document.values() if isinstance(value, dict))
    return count


def main() -> int:
    check = "--check" in sys.argv[1:]
    files = sorted(ROOT.glob("services/*/tests/**/*.json")) + sorted(
        ROOT.glob("evals/planner_entry/recordings/*.json")
    )
    total = changed = 0
    for path in files:
        text = path.read_text()
        document = json.loads(text)
        count = _document(document)
        if count:
            original = json.loads(text)
            sort_keys = json.dumps(original, indent=2, sort_keys=True) + "\n" == text
            if not sort_keys and json.dumps(original, indent=2) + "\n" != text:
                raise SystemExit(
                    f"{path} is not two-space-indented ASCII JSON; rewriting it would reformat it"
                )
            total += count
            changed += 1
            print(f"{count:4} {path.relative_to(ROOT)}")
            if not check:
                path.write_text(json.dumps(document, indent=2, sort_keys=sort_keys) + "\n")
    print(f"{total} references renamed in {changed} of {len(files)} files")
    return 1 if check and total else 0


if __name__ == "__main__":
    raise SystemExit(main())
