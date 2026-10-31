"""Write a record under every pydantic-ai-slim release in the range, then read each record
under every release, and report the writer-by-reader matrix.

Run: python3 matrix.py [--first 2.55.0] [--last 2.64.0]

Needs `uv` and network access (PyPI). Writes records/, controls/, matrix.json, matrix.md.
Exits 1 if a forward pair fails or a control passes cleanly.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).parent
RECORDS = HERE / "records"
CONTROLS = HERE / "controls"
PYPI = "https://pypi.org/pypi/pydantic-ai-slim/json"


def parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def releases(*, first: str, last: str) -> list[str]:
    """Every final release in [first, last] that PyPI still serves, oldest first."""
    with urllib.request.urlopen(PYPI, timeout=60) as response:
        data = json.load(response)
    found = [
        version
        for version, files in data["releases"].items()
        if files
        and re.fullmatch(r"\d+\.\d+\.\d+", version)
        and parse(first) <= parse(version) <= parse(last)
        and not any(f["yanked"] for f in files)
    ]
    return sorted(found, key=parse)


def run_under(version: str, script: str, *args: str) -> dict:
    """Run a script in a throwaway env holding exactly `pydantic-ai-slim==version`."""
    command = [
        "uv", "run", "--no-project", "--with", f"pydantic-ai-slim=={version}",
        "python", script, *args,
    ]  # fmt: skip
    env = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1"}
    done = subprocess.run(
        command, cwd=HERE, env=env, capture_output=True, text=True, timeout=300, check=False
    )
    if done.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def mutate(source: Path, name: str, change: Callable[[dict], None]) -> Path:
    record = json.loads(source.read_text())
    change(record)
    target = CONTROLS / name
    target.write_text(json.dumps(record, indent=2))
    return target


def rename_thinking_kind(record: dict) -> None:
    for message in record["messages"]:
        for part in message["parts"]:
            if part["part_kind"] == "thinking":
                part["part_kind"] = "thinkng"


def add_unknown_field(record: dict) -> None:
    record["messages"][0]["field_from_the_future"] = "x"


def drop_pending_request(record: dict) -> None:
    record["deferred_tool_requests"] = None


# name -> (mutation, what a working check must report)
CONTROL_SPECS = {
    "corrupt-part-kind.json": (rename_thinking_kind, "load fails"),
    "unknown-field.json": (add_unknown_field, "round-trip diff reports the field removed"),
    "no-pending-request.json": (drop_pending_request, "loads, resume fails"),
}


def control_fired(name: str, result: dict) -> bool:
    match name:
        case "corrupt-part-kind.json":
            return not result["loaded"]
        case "unknown-field.json":
            return not result["loaded"] or any(
                d["kind"] == "removed" and "field_from_the_future" in d["path"]
                for d in result["diff"]
            )
        case _:
            return result["loaded"] and not result["resumed"]


def collapse(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def verdict(result: dict) -> str:
    if not result["loaded"]:
        return "LOAD FAIL"
    if not result["resumed"]:
        return "RESUME FAIL"
    return "ok" if result["round_trip_equal"] else f"ok, {len(result['diff'])} diffs"


def diff_summary(cells: dict, writers: list[str], readers: list[str]) -> list[str]:
    """Per writer, the distinct changed fields and the first reader that shows each."""
    lines = []
    for writer in writers:
        first_seen: dict[tuple[str, str], str] = {}
        valued: set[tuple[str, str]] = set()
        for reader in readers:
            for d in cells[writer][reader]["diff"]:
                key = (d["kind"], collapse(d["path"]))
                first_seen.setdefault(key, reader)
                if d.get("before") not in (None, 0):
                    valued.add(key)
        if first_seen:
            lines.append(f"- writer {writer}:")
            lines += [
                f"  - {kind} `{path}`, first under reader {reader}"
                + (", a stored value is dropped" if (kind, path) in valued else "")
                for (kind, path), reader in sorted(first_seen.items())
            ]
    return lines


def render(*, writers: list[str], readers: list[str], cells: dict, controls: dict) -> str:
    out = ["# Stability matrix", "", "Rows are writer versions, columns are reader versions.", ""]
    out.append("| writer \\ reader | " + " | ".join(readers) + " |")
    out.append("|---|" + "---|" * len(readers))
    for writer in writers:
        row = []
        for reader in readers:
            text = verdict(cells[writer][reader])
            if parse(reader) < parse(writer):
                text = f"({text})"
            row.append(text)
        out.append(f"| {writer} | " + " | ".join(row) + " |")
    out += [
        "",
        "Cells in brackets are rollback pairs (reader older than writer), reported only.",
        "",
    ]

    forward = [(w, r) for w in writers for r in readers if parse(r) >= parse(w)]
    rollback = [(w, r) for w in writers for r in readers if parse(r) < parse(w)]
    bad_forward = [(w, r) for w, r in forward if not cells[w][r]["resumed"]]
    bad_rollback = [(w, r) for w, r in rollback if not cells[w][r]["resumed"]]
    changed = [(w, r) for w, r in forward + rollback if cells[w][r]["diff"]]
    out += [
        f"- Forward pairs: {len(forward)}, failing to load or resume: {len(bad_forward)}.",
        f"- Rollback pairs: {len(rollback)}, failing to load or resume: {len(bad_rollback)}.",
        f"- Pairs with a round-trip difference: {len(changed)}.",
        "",
    ]
    if bad_forward or bad_rollback:
        out.append("## Failures")
        out.append("")
        for w, r in bad_forward + bad_rollback:
            cell = cells[w][r]
            out.append(f"- writer {w}, reader {r}: {cell['load_error'] or cell['resume_error']}")
        out.append("")
    summary = diff_summary(cells, writers, readers)
    if summary:
        out += ["## Round-trip differences", "", *summary, ""]
    out += ["## Controls (read under each reader)", ""]
    for name, per_reader in controls.items():
        fired = [r for r, ok in per_reader.items() if ok]
        out.append(
            f"- {name}: {CONTROL_SPECS[name][1]}; reported by {len(fired)} of {len(per_reader)} readers."
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", default="2.55.0")
    parser.add_argument("--last", default="2.64.0")
    args = parser.parse_args()

    versions = releases(first=args.first, last=args.last)
    if not versions:
        sys.exit("no releases found in range")
    print(f"versions ({len(versions)}): {', '.join(versions)}", flush=True)

    RECORDS.mkdir(exist_ok=True)
    CONTROLS.mkdir(exist_ok=True)
    for version in versions:
        run_under(version, "make_record.py", f"records/{version}.json")
    base = RECORDS / f"{versions[-1]}.json"
    controls = [mutate(base, name, change) for name, (change, _) in CONTROL_SPECS.items()]

    paths = [*(f"records/{v}.json" for v in versions), *(f"controls/{c.name}" for c in controls)]
    cells: dict[str, dict[str, dict]] = {w: {} for w in versions}
    control_hits: dict[str, dict[str, bool]] = {c.name: {} for c in controls}
    for reader in versions:
        report = run_under(reader, "load_record.py", *paths)
        assert report["reader"] == reader, report["reader"]
        for result in report["results"]:
            name = result["record"]
            if name in control_hits:
                control_hits[name][reader] = control_fired(name, result)
            else:
                cells[name.removesuffix(".json")][reader] = result
        print(f"read under {reader}", flush=True)

    (HERE / "matrix.json").write_text(
        json.dumps(
            {"writers": versions, "readers": versions, "cells": cells, "controls": control_hits},
            indent=2,
        )
    )
    (HERE / "matrix.md").write_text(
        render(writers=versions, readers=versions, cells=cells, controls=control_hits)
    )

    failed_forward = [
        (w, r)
        for w in versions
        for r in versions
        if parse(r) >= parse(w) and not cells[w][r]["resumed"]
    ]
    dead_controls = [n for n, hits in control_hits.items() if not all(hits.values())]
    print(f"forward failures: {failed_forward}; controls that did not fire: {dead_controls}")
    return 1 if failed_forward or dead_controls else 0


if __name__ == "__main__":
    sys.exit(main())
