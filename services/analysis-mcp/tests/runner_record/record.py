"""Record dfo-apis' runner answers for the requests these tests send.

    uv run --package analysis_mcp python services/analysis-mcp/tests/runner_record/record.py \\
        <dfo-apis checkout> <scratch dir>

Each round runs the analysis-mcp tests with `RUNNER_RECORD` set, so each request
with no recorded answer is written with the frames its roots name
(`runner_replay`). The round's new requests join the accumulated ones in the
scratch dir, `answer.py` answers every one of them in the dfo-apis checkout, and
the answers replace `fixtures/runner_replies.json`. A request that needs the
answer to an earlier one, such as a call on the frame a call returned, is first
sent in a later round, so rounds run until one sends no new request.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_WORKSPACE = _TESTS.parents[2]
_FIXTURE = _TESTS / "fixtures" / "runner_replies.json"
_ANSWER = Path(__file__).resolve().with_name("answer.py")
_MAX_ROUNDS = 10


def _round(apis: Path, scratch: Path) -> int:
    """One round; the number of requests it added."""
    recorded = scratch / "record.json"
    recorded.unlink(missing_ok=True)
    subprocess.run(
        [
            "uv",
            "run",
            "--package",
            "analysis_mcp",
            "pytest",
            str(_TESTS),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_WORKSPACE,
        env={**os.environ, "RUNNER_RECORD": str(recorded)},
        check=False,
        capture_output=True,
    )
    if not recorded.exists():
        return 0
    accumulated_path = scratch / "requests.json"
    accumulated = json.loads(accumulated_path.read_text()) if accumulated_path.exists() else {}
    number = 1 + max((entry["round"] for entry in accumulated.values()), default=0)
    new = {
        key: entry
        for key, entry in json.loads(recorded.read_text()).items()
        if key not in accumulated
    }
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
    _FIXTURE.write_text(json.dumps(dict(sorted(answers.items())), indent=1) + "\n")
    return len(new)


def main() -> None:
    apis, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    for number in range(1, _MAX_ROUNDS + 1):
        added = _round(apis, scratch)
        print(f"round {number}: {added} new request(s)")
        if added == 0:
            return
    raise SystemExit(f"requests still new after {_MAX_ROUNDS} rounds")


if __name__ == "__main__":
    main()
