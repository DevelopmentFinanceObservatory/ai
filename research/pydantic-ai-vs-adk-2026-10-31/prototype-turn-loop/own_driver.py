"""One request against the own loop, as a child process: one JSON line per outcome.

Commands: `start` opens a turn and runs it to its first pause; `answer` answers
a pause and runs on; `resume` continues a turn stopped between pauses;
`cancel` writes a cancel. `--barrier DIR` holds an `answer` between reading
the cursor and writing its claim until `DIR/go` exists.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from loop import Options, S, run_turn
from store import Conflict, Store, connect


def emit(line: dict) -> None:
    sys.stdout.write(json.dumps({"t": time.time(), **line}) + "\n")
    sys.stdout.flush()


def barrier(directory: Path) -> Callable[[], Awaitable[None]]:
    async def wait() -> None:
        emit({"type": "read"})
        while not (directory / "go").exists():
            await asyncio.sleep(0.005)

    return wait


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "answer", "resume", "cancel"])
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--turn", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--interrupt", default="approve")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--decision", default="approve")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--step-s", type=float, default=0.05)
    parser.add_argument("--ttl", type=float, default=2.0)
    parser.add_argument("--barrier", type=Path)
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--no-cas", action="store_true")
    parser.add_argument("--id-only", action="store_true")
    parser.add_argument("--pause-checkpoints-only", action="store_true")
    parser.add_argument("--in-process-cancel-only", action="store_true")
    args = parser.parse_args()

    pool = await connect(args.dsn)
    store = Store(
        pool,
        ttl=args.ttl,
        cas=not args.no_cas,
        match_round=not args.id_only,
        after_read=barrier(args.barrier) if args.barrier else None,
    )
    options = Options(
        version=args.version,
        from_start=args.from_start,
        checkpoint_between_pauses=not args.pause_checkpoints_only,
        cross_instance_cancel=not args.in_process_cancel_only,
    )
    owner = f"pid-{os.getpid()}"
    answer = None
    try:
        if args.command == "cancel":
            # `at` is taken before the write starts, so a stop measured from
            # it is an upper bound on the time from the commit.
            at = time.time()
            await store.request_cancel(args.turn)
            emit({"type": "cancel", "at": at, "committed": time.time()})
            return
        if args.command == "start":
            data = {"steps": args.steps, "step_s": args.step_s}
            cursor = await store.create(args.turn, state=S.SEED, owner=owner, data=data)
        elif args.command == "answer":
            response = {"decision": args.decision, "round": args.round}
            cursor, answer = await store.claim_answer(
                args.turn,
                interrupt_id=args.interrupt,
                round=args.round,
                answer=response,
                owner=owner,
            )
        else:
            cursor = await store.claim_resume(args.turn, owner=owner)
        emit({"type": "claimed", "state": cursor.state, "records": cursor.records})
        outcome = await run_turn(
            store,
            cursor,
            dsn=args.dsn,
            owner=owner,
            options=options,
            log=args.log,
            emit=emit,
            answer=answer,
        )
        line = {"type": outcome.kind, "stopped_at": outcome.stopped_at}
        if outcome.cursor is not None:
            line |= {"interrupt": outcome.cursor.pending_interrupt, "round": outcome.cursor.round}
        emit(line)
    except Conflict as conflict:
        emit({"type": "conflict", "reason": conflict.reason})
    finally:
        await pool.close()


asyncio.run(main())
