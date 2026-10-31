"""Tests for the tap, the analysis checks and the driver's own logic.

Run from the worktree root with the pins the prototype uses:

    uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness pytest \\
        research/pai-prototype/gemini/test_tap.py -p no:cacheprovider

The tap tests run google-genai's real classes against an `httpx.MockTransport`, so the
patched methods, the SSE parser and the response models are the library's own. Every
check also has a deliberately broken variant that it must catch.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import errors, types

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse
import driver
import tap

SIGNATURE = "c2lnbmF0dXJl"  # base64 of "signature"
CHUNKS = [
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "weigh", "thought": True}]}}]},
    {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "propose_plan", "args": {"a": 1}},
                            "thoughtSignature": SIGNATURE,
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 60,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 9,
        },
    },
]


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "badmodel" in url:
        return httpx.Response(400, json={"error": {"code": 400, "message": "no", "status": "BAD"}})
    if "streamGenerateContent" in url:
        body = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in CHUNKS)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )
    if "cachedContents" in url:
        return httpx.Response(200, json={"name": "cachedContents/abc"})
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 5},
        },
    )


def make_client() -> genai.Client:
    transport = httpx.MockTransport(handler)
    options = types.HttpOptions(async_client_args={"transport": transport})
    return genai.Client(api_key="test", http_options=options)


def dump(value: Any) -> Any:
    return value.model_dump(mode="json", exclude={"sdk_http_response"})


async def scenario(client: genai.Client) -> dict[str, Any]:
    """Every patched method once, plus an error, reduced to comparable data."""
    models = client.aio.models
    config = types.GenerateContentConfig(system_instruction="sys", tools=None)
    contents = [types.Content(role="user", parts=[types.Part(text="hello")])]
    out: dict[str, Any] = {"plain": dump(await models.generate_content(model="m", contents="hi"))}
    stream = await models.generate_content_stream(model="m", contents=contents, config=config)
    out["chunks"] = [dump(chunk) async for chunk in stream]
    cache_config = types.CreateCachedContentConfig(system_instruction="sys", ttl="60s")
    out["cache"] = dump(await client.aio.caches.create(model="m", config=cache_config))
    with pytest.raises(errors.ClientError) as caught:
        await models.generate_content(model="badmodel", contents="hi")
    out["error"] = (caught.value.code, str(caught.value))
    return out


def tapped_run(path: Path) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        with tap.installed(path), tap.labelled(arm="a", task="t", rep=1, entry_id="e1"):
            return await scenario(make_client())

    return asyncio.run(go())


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- The tap ---------------------------------------------------------------------------


def test_tap_returns_exactly_what_the_untapped_library_returns(tmp_path: Path) -> None:
    untapped = asyncio.run(scenario(make_client()))
    assert tapped_run(tmp_path / "tap.jsonl") == untapped


def test_the_comparison_catches_a_tap_that_drops_a_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lossy(stream: Any, call: Any) -> Any:
        chunks = [chunk async for chunk in stream]
        for chunk in chunks[:-1]:
            call.observe(chunk)
            yield chunk
        call.finish()

    monkeypatch.setattr(tap, "_tapped", lossy)
    untapped = asyncio.run(scenario(make_client()))
    assert tapped_run(tmp_path / "tap.jsonl") != untapped


def test_the_patch_is_removed_on_exit_and_cannot_nest(tmp_path: Path) -> None:
    before = genai.models.AsyncModels.generate_content_stream
    with tap.installed(tmp_path / "a.jsonl"):
        assert genai.models.AsyncModels.generate_content_stream is not before
        with pytest.raises(RuntimeError), tap.installed(tmp_path / "b.jsonl"):
            pass
    assert genai.models.AsyncModels.generate_content_stream is before


def test_records_hold_what_the_wire_carried(tmp_path: Path) -> None:
    path = tmp_path / "tap.jsonl"
    tapped_run(path)
    stream, cache = (
        next(r for r in read(path) if r["stream"]),
        next(r for r in read(path) if r["kind"] == "cache_create"),
    )
    assert (stream["arm"], stream["task"], stream["rep"], stream["entry_id"]) == ("a", "t", 1, "e1")
    assert stream["has_system_instruction"] and not stream["has_tools"]
    assert stream["cached_content"] is None
    assert stream["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert stream["chunks"] == 2
    assert 0 < stream["first_chunk_s"] <= stream["last_chunk_s"] <= stream["seconds"]
    assert stream["usage"] == {"prompt": 100, "cached": 60, "candidates": 7, "thoughts": 9}
    assert stream["response_thought_parts"] == 1
    (call,) = stream["response_function_calls"]
    assert call["name"] == "propose_plan" and call["has_sig"] and call["sig_sha"]
    assert cache["model"] == "m" and cache["name"] == "cachedContents/abc"
    assert cache["system_instruction"] == '"sys"'
    assert cache["system_instruction_sha"] == tap.sha('"sys"')
    assert cache["tools"] is None and cache["tool_config"] is None
    failed = next(r for r in read(path) if r["error"])
    assert failed["error"]["class"] == "ClientError" and failed["error"]["code"] == 400
    assert all(r["tap_errors"] == [] for r in read(path))


def test_request_function_calls_and_thought_parts_are_read_from_contents(tmp_path: Path) -> None:
    contents = [
        {"role": "user", "parts": [{"text": "q"}]},
        {
            "role": "model",
            "parts": [
                {"text": "summary", "thought": True},
                {
                    "function_call": {"name": "get_skill", "id": "x1", "args": {"s": "a"}},
                    "thought_signature": b"signature",
                },
                {"function_call": {"name": "get_skill", "args": {"s": "b"}}},
            ],
        },
    ]

    async def go() -> None:
        with tap.installed(tmp_path / "tap.jsonl"):
            await make_client().aio.models.generate_content(model="m", contents=contents)

    asyncio.run(go())
    (record,) = read(tmp_path / "tap.jsonl")
    first, second = record["request_function_calls"]
    assert (first["name"], first["id"], first["has_sig"], first["content_index"]) == (
        "get_skill",
        "x1",
        True,
        1,
    )
    assert (second["id"], second["has_sig"], second["sig_sha"]) == (None, False, None)
    assert record["request_thought_parts"] == 1
    # Bytes in a dict and in a model reach the same base64.
    assert record["contents"][1]["parts"][1]["thought_signature"] == SIGNATURE


def test_calls_made_in_tasks_inherit_the_labels(tmp_path: Path) -> None:
    async def go() -> None:
        client = make_client()
        with tap.installed(tmp_path / "tap.jsonl"):

            async def entry(n: int) -> None:
                with tap.labelled(arm="a", task="t", rep=n, entry_id=f"e{n}"):
                    await asyncio.create_task(
                        client.aio.models.generate_content(model="m", contents="hi")
                    )

            await asyncio.gather(entry(1), entry(2))

    asyncio.run(go())
    assert sorted(r["entry_id"] for r in read(tmp_path / "tap.jsonl")) == ["e1", "e2"]


class _Models:
    """A fake with the real methods' shape, recording what it was handed."""

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}
        self.closed = False

    async def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.seen = {"model": model, "contents": contents, "config": config}
        return object()

    async def generate_content_stream(
        self, *, model: str, contents: Any, config: Any = None
    ) -> Any:
        self.seen = {"model": model, "contents": contents, "config": config}

        async def chunks() -> Any:
            try:
                for n in range(3):
                    yield n
            finally:
                self.closed = True

        return chunks()


class _Caches:
    async def create(self, *, model: str, config: Any = None) -> Any:
        raise TimeoutError("slow")


def test_arguments_pass_through_untouched_and_errors_and_closing_propagate(
    tmp_path: Path,
) -> None:
    config, contents = types.GenerateContentConfig(system_instruction="s"), ["x"]

    async def go() -> None:
        with tap.installed(tmp_path / "tap.jsonl", models_cls=_Models, caches_cls=_Caches):
            models = _Models()
            stream = await models.generate_content_stream(
                model="m", contents=contents, config=config
            )
            assert models.seen["config"] is config and models.seen["contents"] is contents
            assert [chunk async for chunk in stream] == [0, 1, 2]
            assert models.closed
            # A consumer that stops early closes the original stream and the tap says so.
            models = _Models()
            stream = await models.generate_content_stream(model="m", contents=contents)
            async for chunk in stream:
                if chunk == 1:
                    break
            await stream.aclose()
            assert models.closed
            # A response the tap cannot read still reaches the caller.
            assert type(await models.generate_content(model="m", contents=contents)) is object
            with pytest.raises(TimeoutError):
                await _Caches().create(model="m")

    asyncio.run(go())
    records = read(tmp_path / "tap.jsonl")
    assert [r["error"]["class"] if r["error"] else None for r in records] == [
        None,
        "GeneratorExit",
        None,
        "TimeoutError",
    ]
    assert records[0]["chunks"] == 3 and records[1]["chunks"] == 2


# --- The checks ------------------------------------------------------------------------


def fc(name: str, args: str, *, sig: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "id": None,
        "args_sha": args,
        "has_sig": sig is not None,
        "sig_sha": sig,
    }


def gen(
    seq: int,
    *,
    cached: str | None = "c",
    prefix: str | None = None,
    replay: list[dict[str, Any]] | None = None,
    thoughts: int = 0,
    responds: list[dict[str, Any]] | None = None,
    prompt: int = 100,
    hit: int = 90,
    first: float | None = 1.0,
) -> dict[str, Any]:
    return {
        "kind": "generate",
        "seq": seq,
        "cached_content": cached,
        "has_system_instruction": prefix == "system_instruction",
        "has_tools": prefix == "tools",
        "has_tool_config": prefix == "tool_config",
        "request_function_calls": replay or [],
        "request_thought_parts": thoughts,
        "response_function_calls": responds or [],
        "stream": first is not None,
        "first_chunk_s": first,
        "usage": {"prompt": prompt, "cached": hit, "candidates": 1, "thoughts": 1},
    }


@pytest.mark.parametrize("prefix", ["system_instruction", "tools", "tool_config"])
def test_w2_counts_a_cached_request_that_also_carries_the_prefix(prefix: str) -> None:
    good = [[gen(1), gen(2)]]
    assert analyse.w2(good) == {"requests": 2, "cached": 2, "violations": 0}
    broken = [[gen(1), gen(2, prefix=prefix)]]
    assert analyse.w2(broken)["violations"] == 1
    base = analyse.w2(good)
    assert analyse._w2_verdict("pai", analyse.w2(good), base).ok is True
    assert analyse._w2_verdict("pai", analyse.w2(broken), base).ok is False
    assert analyse._w2_verdict("pai", analyse.w2([[gen(1, cached=None)]]), base).ok is None
    # Without a cache the same fields are what a request should carry.
    assert analyse.w2([[gen(1, cached=None, prefix=prefix)]])["violations"] == 0


def test_w3_matches_replays_to_their_originals_and_flags_a_lost_signature() -> None:
    a, b = ("f", "arg-a"), ("f", "arg-b")
    answer = [fc(*a, sig="s1"), fc(*b, sig=None)]
    kept = [fc(*a, sig="s1"), fc(*b, sig=None)]
    good = [[gen(1, responds=answer), gen(2, replay=kept, responds=[fc("g", "x", sig="s2")])]]
    counts = analyse.w3(good)
    assert (counts["replayed"], counts["signed"], counts["lost"], counts["thought_parts"]) == (
        2,
        1,
        0,
        0,
    )
    lost = [[gen(1, responds=answer), gen(2, replay=[fc(*a, sig=None), fc(*b, sig=None)])]]
    assert analyse.w3(lost)["lost"] == 1
    swapped = [[gen(1, responds=answer), gen(2, replay=[fc(*a, sig="other"), fc(*b, sig="x")])]]
    assert (analyse.w3(swapped)["changed"], analyse.w3(swapped)["added"]) == (1, 1)
    replayed_thought = [[gen(1, responds=answer), gen(2, replay=kept, thoughts=2)]]
    assert analyse.w3(replayed_thought)["thought_parts"] == 2
    base = analyse.w3(good)
    assert analyse._w3_verdict("pai", analyse.w3(good), base).ok is True
    assert analyse._w3_verdict("pai", analyse.w3(lost), base).ok is False
    assert analyse._w3_verdict("pai", analyse.w3(replayed_thought), base).ok is False
    nothing = analyse.w3([[gen(1), gen(2)]])
    assert analyse._w3_verdict("pai", nothing, base).ok is None


def test_w3_pairs_repeated_identical_calls_in_order() -> None:
    twice = [fc("f", "x", sig="s1"), fc("f", "x", sig="s2")]
    entry = [gen(1, responds=twice[:1]), gen(2, responds=twice[1:]), gen(3, replay=twice)]
    assert analyse.w3([entry])["changed"] == 0
    reordered = [gen(1, responds=twice[:1]), gen(2, responds=twice[1:]), gen(3, replay=twice[::-1])]
    assert analyse.w3([reordered])["changed"] == 2


def test_w4_reads_only_second_and_later_calls_and_fails_a_lower_share() -> None:
    base = analyse.w4([[gen(1, hit=0), gen(2, hit=90), gen(3, hit=90)]])
    assert base["share"] == 0.9 and base["calls"] == 2
    close = analyse.w4([[gen(1, hit=0), gen(2, hit=86)]])
    far = analyse.w4([[gen(1, hit=0), gen(2, hit=80)]])
    assert analyse._w4_verdict("pai", close, base).ok is True
    assert analyse._w4_verdict("pai", far, base).ok is False
    higher = analyse.w4([[gen(1, hit=0), gen(2, hit=99)]])
    assert analyse._w4_verdict("pai", higher, base).ok is True


def test_w5_reads_streams_only_and_fails_a_slower_median() -> None:
    entries = [[gen(1, first=1.0), gen(2, first=2.0), gen(3, first=None)]]
    base = analyse.w5(entries)
    assert base["streams"] == 2 and base["p50"] == 1.5 and base["first_call_p50"] == 1.0
    slow = analyse.w5([[gen(1, first=2.0), gen(2, first=2.0)]])
    fine = analyse.w5([[gen(1, first=1.5), gen(2, first=1.6)]])
    assert analyse._w5_verdict("pai", fine, base).ok is True
    assert analyse._w5_verdict("pai", slow, base).ok is False


def test_parity_names_where_first_requests_differ() -> None:
    same = [{"role": "user", "parts": [{"text": "q"}]}]
    other = [{"role": "user", "parts": [{"text": "q!"}]}]
    assert analyse.first_difference(same, same) is None
    assert analyse.first_difference(same, other) == "/0/parts/0/text"
    split = [
        {"role": "user", "parts": [{"text": "q"}]},
        {"role": "user", "parts": [{"text": "r"}]},
    ]
    joined = [{"role": "user", "parts": [{"text": "q"}, {"text": "r"}]}]
    assert analyse.first_difference(split, joined) is not None
    assert analyse.first_difference(analyse.merged(split), analyse.merged(joined)) is None
    firsts = {
        ("baseline", "e"): [split],
        ("pai", "e"): [joined],
        ("baseline", "a"): [same],
        ("pai", "a"): [same],
        ("baseline", "b"): [same],
        ("pai", "b"): [other],
        ("baseline", "c"): [same],
        ("pai", "c"): [same, other],
        ("baseline", "d"): [same],
    }
    found = analyse.parity(firsts, "pai")
    assert {task: v["status"] for task, v in found.items()} == {
        "a": "identical",
        "b": "differs",
        "e": "regrouped",
        "c": "unstable",
        "d": "missing",
    }
    assert found["b"]["where"] == "/0/parts/0/text"
    assert analyse._parity_verdict("pai", found).ok is False
    assert analyse._parity_verdict("pai", {"a": found["a"]}).ok is True
    assert analyse._parity_verdict("pai", {"e": found["e"]}).ok is False


# --- Whole analysis on synthetic records -----------------------------------------------


def record(arm: str, task: str, entry_id: str, *, status: str = "ok") -> dict[str, Any]:
    first = {"outcome": "ok", "code": None}
    return {
        "schema": driver.RECORD_SCHEMA,
        "arm": arm,
        "task": task,
        "rep": 1,
        "model": "m",
        "revision": "r",
        "experiment": arm,
        "entry_id": entry_id,
        "inputs": "i",
        "status": status,
        "error": None if status == "ok" else "boom",
        "stop_reason": "accepted",
        "plan_kinds": ["dataset"],
        "proposals": [{"kinds": ["dataset"], "args": {}, "outcome": "ok"}],
        "first_proposal": first,
        "calls": [],
        "budget_closed": None,
        "surface_misses": [],
        "wall_s": 1.0,
    }


def test_a_retried_entry_counts_only_its_latest_attempt() -> None:
    records = [
        record("baseline", "t", "old", status="error"),
        record("baseline", "t", "new"),
    ]
    tap_calls = [
        {**gen(1), "entry_id": "old"},
        {**gen(2), "entry_id": "new"},
        {**gen(3), "entry_id": "new"},
    ]
    entries = analyse.chosen_entries(records, tap_calls)
    assert [[c["seq"] for c in entry] for entry in entries["baseline"]] == [[2, 3]]


def test_falls_names_tasks_the_baseline_passes_and_the_arm_does_not() -> None:
    tasks = driver.load_tasks()
    task = next(t for t in tasks if t.first_proposal_refusal is None)
    needs = sorted(task.required_kinds)[0]
    passing, failing = {"plan_kinds": sorted(task.required_kinds)}, {"plan_kinds": [needs, "zzz"]}
    records = [
        {**record("baseline", task.id, "b"), **passing},
        {**record("pai", task.id, "p"), **passing},
    ]
    assert analyse.falls(records, tasks, "pai") == []
    records[1] = {**record("pai", task.id, "p"), **failing}
    assert analyse.falls(records, tasks, "pai") == [task.id]


# --- The driver ------------------------------------------------------------------------


def test_arm_order_is_seeded_per_task_and_repetition_and_covers_every_arm_once() -> None:
    tasks = driver.load_tasks()[:12]
    first = driver.shuffled_triples(tasks, driver.ARMS, 2, seed=1)
    assert first == driver.shuffled_triples(tasks, driver.ARMS, 2, seed=1)
    assert first != driver.shuffled_triples(tasks, driver.ARMS, 2, seed=2)
    assert sorted(first) == sorted(
        (arm, task.id, rep) for arm in driver.ARMS for task in tasks for rep in (1, 2)
    )
    orders = {tuple(arm for arm, *_ in first[i : i + 3]) for i in range(0, len(first), 3)}
    assert len(orders) > 1


def test_chunks_partition_the_tasks() -> None:
    tasks = driver.load_tasks()
    parts = [driver.select_tasks(tasks, only=None, chunk=f"{k}/8") for k in range(1, 9)]
    assert sorted(t.id for part in parts for t in part) == sorted(t.id for t in tasks)
    assert max(len(part) for part in parts) <= 8
    with pytest.raises(SystemExit):
        driver.select_tasks(tasks, only=None, chunk="9/8")
    with pytest.raises(SystemExit):
        driver.select_tasks(tasks, only="nope", chunk=None)


def test_a_pai_record_missing_a_field_is_stamped_as_an_error() -> None:
    complete = dict.fromkeys(driver.REQUIRED)
    complete |= {"status": "ok", "calls": [], "surface_misses": []}
    stamped = driver.stamp(
        complete, arm="pai", rep=2, entry_id="e", model="m", revision="r", experiment="x"
    )
    assert stamped["status"] == "ok" and stamped["rep"] == 2 and stamped["entry_id"] == "e"
    incomplete = {key: value for key, value in complete.items() if key != "proposals"}
    broken = driver.stamp(
        incomplete, arm="pai", rep=2, entry_id="e", model="m", revision="r", experiment="x"
    )
    assert broken["status"] == "error" and "proposals" in broken["error"]


def test_the_pai_arms_raise_not_implemented_while_the_port_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pai_entry", None)
    with pytest.raises(NotImplementedError):
        driver.load_pai_record()
