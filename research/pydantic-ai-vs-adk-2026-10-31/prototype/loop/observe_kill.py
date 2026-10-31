"""Observation: SIGKILL a process while `propose_plan` runs, then recover.

    uv run --no-project --with 'pydantic-ai-slim==2.64.0' python observe_kill.py

Prints one JSON object per scenario: which checkpoint the turn fell back to, whether
progress frames repeated, which tool bodies ran twice, and whether the finished turn
equals the unkilled reference.
"""

import asyncio
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from agents import Deps
from checks import APPROVE, QUESTION, SCOPE, history
from loop import Cursor
from procs import run_driver
from reference import run_reference
from store import Store

CID = "obs"


def counts(frames: list[dict]) -> dict[str, int]:
    return dict(Counter(f["type"] for f in frames))


def checkpoint(db: Path) -> dict:
    store = Store(db)
    row = store.read_turn(CID)
    assert row is not None
    cursor_json, version = row
    cursor = Cursor.model_validate_json(cursor_json)
    return {
        "state": cursor.state,
        "version": version,
        "planner_input": cursor.planner_input.kind if cursor.planner_input else None,
        "planner_conversation_stored": store.load_conversation(CID, "planner") is not None,
    }


async def scenario(hold_at: int, directory: Path) -> dict:
    db, log = directory / "obs.db", directory / "tools.log"
    log_args = ("--tool-log", str(log))
    seen: dict = {"hold_at": hold_at}
    client: list[dict] = []

    def is_progress(frame: dict) -> bool:
        return frame["type"] == "activity.step"

    if hold_at == 1:
        killed = run_driver(
            db,
            "start",
            "--cid",
            CID,
            "--text",
            QUESTION,
            global_args=(*log_args, "--hold-at", "1"),
            kill_when=is_progress,
        )
    else:
        first = run_driver(db, "start", "--cid", CID, "--text", QUESTION, global_args=log_args)
        client += first.frames
        interrupt = first.frames[-1]["interrupt_id"]
        killed = run_driver(
            db,
            "resume",
            "--cid",
            CID,
            "--interrupt",
            interrupt,
            "--answer",
            json.dumps(SCOPE),
            global_args=(*log_args, "--hold-at", "2"),
            kill_when=is_progress,
        )
    client += killed.frames
    seen["killed_returncode"] = killed.returncode
    seen["frames_before_kill"] = counts(killed.frames)
    seen["checkpoint_after_kill"] = checkpoint(db)

    recovered = run_driver(db, "recover", "--cid", CID, global_args=log_args)
    client += recovered.frames
    seen["recover_returncode"] = recovered.returncode
    seen["frames_from_recover"] = counts(recovered.frames)
    seen["recover_first_frames"] = [
        f"{f['type']}:{f.get('done', f.get('text', ''))}" for f in recovered.frames[:5]
    ]
    seen["progress_frames_the_client_saw"] = [
        f["done"] for f in client if f["type"] == "activity.step"
    ]

    last = recovered.frames[-1]
    if last["type"] == "elicitation.request":
        resumed = run_driver(
            db,
            "resume",
            "--cid",
            CID,
            "--interrupt",
            last["interrupt_id"],
            "--answer",
            json.dumps(SCOPE),
            global_args=log_args,
        )
        last = resumed.frames[-1]
    final = run_driver(
        db,
        "resume",
        "--cid",
        CID,
        "--interrupt",
        last["interrupt_id"],
        "--answer",
        json.dumps(APPROVE),
        global_args=log_args,
    )
    seen["final_frame"] = final.frames[-1]["type"]
    seen["tool_body_entries"] = dict(Counter(log.read_text().splitlines()))

    reference = await run_reference(CID, QUESTION, SCOPE, Deps())
    store = Store(db)
    seen["equals_reference"] = {
        name: history(store.require_conversation(CID, name)) == history(conversation)
        for name, conversation in zip(("orchestrator", "planner"), reference[:2], strict=True)
    }
    return seen


async def main() -> None:
    for hold_at in (1, 2):
        with tempfile.TemporaryDirectory() as directory:
            print(json.dumps(await scenario(hold_at, Path(directory)), indent=2))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
