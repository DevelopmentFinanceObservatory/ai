"""Record dfo-apis' runner answers for the requests these tests send.

    uv run --package analysis_mcp python services/analysis-mcp/tests/runner_record/record.py \\
        <dfo-apis checkout> <scratch dir>

The fixture and the scratch dir's requests are emptied first, so every reply is
recorded from dfo-apis afresh. Each round runs the analysis-mcp tests with
`RUNNER_RECORD` set, so each request with no recorded answer is written with the
frames its roots name (`runner_replay`). The round's new requests join the accumulated ones in the
scratch dir, `answer.py` answers every one of them in the dfo-apis checkout, and
the answers replace `fixtures/runner_replies.json`. A request that needs the
answer to an earlier one, such as a call on the frame a call returned, is first
sent in a later round, so rounds run until one sends no new request.

Another suite records its own replies with `--package`, `--fixture` and the
test paths to run, as the harness does:

    uv run --package harness python services/analysis-mcp/tests/runner_record/record.py \\
        <dfo-apis checkout> <scratch dir> --package harness \\
        --fixture services/harness/tests/fixtures/runner_replies.json services/harness/tests
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TESTS = Path(__file__).resolve().parents[1]
_WORKSPACE = _TESTS.parents[2]
_FIXTURE = _TESTS / "fixtures" / "runner_replies.json"
_ANSWER = Path(__file__).resolve().with_name("answer.py")
_MAX_ROUNDS = 20


@dataclass(frozen=True, kw_only=True)
class _Suite:
    """What a round runs, and where its answers go."""

    package: str
    fixture: Path
    tests: tuple[str, ...]
    workers: int


def _round(apis: Path, scratch: Path, suite: _Suite) -> int:
    """One round; the number of requests it added."""
    recorded = scratch / "record.json"
    for stale in scratch.glob("record.json*"):
        stale.unlink()
    workers = ["--with", "pytest-xdist"] if suite.workers else []
    spread = ["-n", str(suite.workers)] if suite.workers else []
    subprocess.run(
        [
            "uv",
            "run",
            *workers,
            "--package",
            suite.package,
            "pytest",
            *suite.tests,
            *spread,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_WORKSPACE,
        # pytest-xdist workers must collect parameters in one order.
        env={**os.environ, "RUNNER_RECORD": str(recorded), "PYTHONHASHSEED": "0"},
        check=False,
        capture_output=True,
    )
    written: dict[str, Any] = {}
    for part in sorted(scratch.glob("record.json*")):
        written.update(json.loads(part.read_text()))
    accumulated_path = scratch / "requests.json"
    accumulated = json.loads(accumulated_path.read_text()) if accumulated_path.exists() else {}
    number = 1 + max((entry["round"] for entry in accumulated.values()), default=0)
    new = {key: entry for key, entry in written.items() if key not in accumulated}
    if not new:
        return 0
    for entry in new.values():
        entry["round"] = number
    accumulated.update(new)
    accumulated_path.write_text(json.dumps(accumulated))
    replies = scratch / "replies.json"
    subprocess.run(
        ["uv", "run", "python", str(_ANSWER), str(accumulated_path), str(replies)],
        cwd=apis,
        env={**os.environ, "PYTHONPATH": "."},
        check=True,
    )
    answers = json.loads(replies.read_text())
    suite.fixture.write_text(json.dumps(dict(sorted(answers.items())), indent=1) + "\n")
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("apis", type=Path, help="a dfo-apis checkout")
    parser.add_argument("scratch", type=Path, help="where the rounds keep their requests")
    parser.add_argument("tests", nargs="*", default=[str(_TESTS)], help="the test paths to run")
    parser.add_argument("--package", default="analysis_mcp", help="the package to run them in")
    parser.add_argument("--fixture", type=Path, default=_FIXTURE, help="the replies to write")
    parser.add_argument(
        "--workers", type=int, default=0, help="pytest-xdist workers per round, none when 0"
    )
    args = parser.parse_args()
    apis, scratch = args.apis.resolve(), args.scratch.resolve()
    suite = _Suite(
        package=args.package,
        fixture=(_WORKSPACE / args.fixture).resolve(),
        tests=tuple(args.tests),
        workers=args.workers,
    )
    scratch.mkdir(parents=True, exist_ok=True)
    # Every reply is recorded afresh: a request an old reply answers would be sent
    # with no new answer, and keep a reply dfo-apis no longer gives.
    for stale in ("requests.json", "replies.json"):
        (scratch / stale).unlink(missing_ok=True)
    suite.fixture.write_text("{}\n")
    for number in range(1, _MAX_ROUNDS + 1):
        added = _round(apis, scratch, suite)
        print(f"round {number}: {added} new request(s)")
        if added == 0:
            return
    raise SystemExit(f"requests still new after {_MAX_ROUNDS} rounds")


if __name__ == "__main__":
    main()
