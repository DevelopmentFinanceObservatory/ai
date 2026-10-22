"""`harness.recipe_backfill`: the walk of the materialisation store and what each
derived frame ends as.

The store is `InMemoryMaterialisationStore`, filled by real analysis-mcp verbs so
the traces are the ones a deployment stores. apis is `FakeApis` behind
`httpx2.MockTransport`: the recipe and versions routes, answering as apis'
publish door does.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import plain_single_unit_frame
from analysis_mcp.growth_tool import growth
from common.live_apis import PRINCIPAL_HEADER, LiveApisClient, LiveApisError
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from common.recipe import build_recipe
from harness import recipe_backfill
from harness.recipe_backfill import FRAMES_IN_FLIGHT, BackfillReport, Reason, run_backfill

PRINCIPAL = "datasets-principal-7f3a"
SPEC = "spec_" + "a" * 32
ROOT = FrameRef(spec_id=SPEC, content_version="cv_" + "1" * 32)
OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}
START = datetime(2026, 9, 14, tzinfo=UTC)
NOW = datetime(2026, 10, 7, tzinfo=UTC)

Pair = tuple[str, str]


def pair(ref: FrameRef) -> Pair:
    return ref.spec_id, ref.content_version


def problem(status: int, code: str, **members: object) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"code": code, "retryable": False, "detail": "refused", **members},
        headers={"content-type": "application/problem+json"},
    )


def _listed(version: dict[str, str | None]) -> dict[str, Any]:
    """`version` as the versions route lists it: a raw version is `plain`, and a
    collapse here has no recorded groupby."""
    rendering = "plain" if version["base_content_version"] is None else "not_recorded"
    return {**version, "rendering": rendering, "groupby": None}


class FakeApis:
    """apis' recipe and versions routes over plain dicts.

    `published` holds the pairs with a version row, `recipes` each pair's stored
    recipe and `versions` each dataset's stored versions. `answers` replaces the
    answer to every request for one frame, keyed by its `spec_id`.
    """

    def __init__(self, *, published: Iterable[FrameRef] = ()) -> None:
        self.published: set[Pair] = {pair(ref) for ref in published}
        self.recipes: dict[Pair, dict[str, Any]] = {}
        self.versions: dict[str, list[dict[str, str | None]]] = {}
        self.answers: dict[str, httpx2.Response | Exception] = {}
        self.requests: list[httpx2.Request] = []
        self.in_flight = 0
        self.most_in_flight = 0

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            return self._answer(request)
        finally:
            self.in_flight -= 1

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        if request.headers.get(PRINCIPAL_HEADER) != PRINCIPAL:
            return problem(403, "publish_not_permitted")
        parts = request.url.path.removeprefix("/api/datasets/").split("/")
        if parts[-1] == "versions":
            return httpx2.Response(
                200,
                json={
                    "schema_version": 1,
                    "spec_id": parts[0],
                    "versions": [_listed(version) for version in self.versions.get(parts[0], [])],
                },
            )
        _, spec_id, content_version, _ = parts
        override = self.answers.get(spec_id)
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return override
        key = (spec_id, content_version)
        if request.method == "GET":
            held = self.recipes.get(key)
            return problem(404, "recipe_not_found") if held is None else _ok(held)
        return self._put(key, json.loads(request.content))

    def _put(self, key: Pair, recipe: dict[str, Any]) -> httpx2.Response:
        if key not in self.published:
            return problem(404, "snapshot_not_found")
        held = self.recipes.get(key)
        answer = {"spec_id": key[0], "content_version": key[1]}
        if held is not None:
            for member in ("normal_form_hash", "replay_key"):
                if held[member] != recipe[member]:
                    return problem(
                        409, "publish_conflict", constraint_id=f"recipe_requires_matching_{member}"
                    )
            return _ok({**answer, "created": False})
        if any(other["replay_key"] == recipe["replay_key"] for other in self.recipes.values()):
            return problem(409, "publish_conflict", constraint_id="replay_key_requires_one_frame")
        self.recipes[key] = recipe
        return _ok({**answer, "created": True})

    def sent(self, method: str) -> list[httpx2.Request]:
        return [request for request in self.requests if request.method == method]

    def client(self, *, principal: str = PRINCIPAL) -> LiveApisClient:
        async def no_sleep(_: float) -> None:
            return None

        return LiveApisClient(
            base_url="http://apis.test",
            datasets_principal=principal,
            transport=httpx2.MockTransport(self),
            sleep=no_sleep,
        )


def _ok(body: object) -> httpx2.Response:
    return httpx2.Response(200, json=body)


class Clock:
    """A clock a minute later on every reading, so each frame written is older than
    the next."""

    def __init__(self) -> None:
        self._at = START

    def __call__(self) -> datetime:
        self._at += timedelta(minutes=1)
        return self._at


@pytest.fixture
async def store() -> AsyncIterator[InMemoryMaterialisationStore]:
    held = InMemoryMaterialisationStore(now=Clock())
    set_store(held)
    yield held
    set_store(None)


async def seed_root(
    store: InMemoryMaterialisationStore, root: FrameRef = ROOT, *, mint_record: bool = True
) -> FrameRef:
    trace = MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": root.content_version}],
            "facets": [],
            "conversion": None,
        },
        spec_id=root.spec_id,
        content_version=root.content_version,
    )
    await store.put(root, frame=plain_single_unit_frame(), trace=trace, writer="test")
    if mint_record:
        await store.put_mint_record(MintRecord(frame=root, dataset=root, options=OPTIONS))
    return root


def _result_ref(result: Any) -> FrameRef:
    assert result.is_error is False, result.content
    content = result.structured_content
    return FrameRef(spec_id=content["spec_id"], content_version=content["content_version"])


async def summed(ref: FrameRef, *, aggregation: str = "sum") -> FrameRef:
    """`aggregate` over `ref`, by reporter and year."""
    return _result_ref(
        await aggregate(
            frame_ref=ref.model_dump(),
            aggregation=aggregation,
            group_by=("reporter", "time_period"),
        )
    )


async def grown(ref: FrameRef) -> FrameRef:
    """`growth` over `ref` from 2021 to 2023."""
    return _result_ref(
        await growth(
            frame_ref=ref.model_dump(),
            window_start="2021",
            window_end="2023",
            entity_by=("reporter",),
        )
    )


async def without(
    store: InMemoryMaterialisationStore, missing: FrameRef, kept: Iterable[FrameRef]
) -> InMemoryMaterialisationStore:
    """A store holding `kept` as `store` holds them, in the order given, and not
    `missing`: the bucket after its age rule deleted one frame."""
    thinned = InMemoryMaterialisationStore(now=Clock())
    for ref in kept:
        assert ref != missing
        held = await store.get(ref)
        assert held is not None
        await thinned.put(ref, frame=held[0], trace=held[1], writer="test")
        if (record := await store.get_mint_record(ref)) is not None:
            await thinned.put_mint_record(record)
    return thinned


async def backfill(
    store: InMemoryMaterialisationStore, apis: FakeApis, **options: Any
) -> BackfillReport:
    async with apis.client() as client:
        return await run_backfill(store=store, client=client, now=lambda: NOW, **options)


def reasons(report: BackfillReport) -> list[tuple[FrameRef, Reason, str | None]]:
    return [(outcome.ref, outcome.reason, outcome.detail) for outcome in report.outcomes]


async def test_a_write_run_stores_each_published_frames_recipe_oldest_first(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store)
    total = await summed(root)
    change = await grown(total)
    apis = FakeApis(published=[total, change])

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [(total, Reason.STORED, None), (change, Reason.STORED, None)]
    assert apis.recipes == {
        pair(total): await build_recipe(total, store=store),
        pair(change): await build_recipe(change, store=store),
    }
    assert report.frames_held == 3
    assert report.derived_held == 2
    assert report.oldest_created_at == (await store.list_frames())[0].created_at
    assert report.settled


async def test_a_dry_run_writes_nothing_to_dfo_apis_or_the_store(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store, mint_record=False)
    total = await summed(root)
    apis = FakeApis(published=[total])
    apis.versions[SPEC] = [{"content_version": root.content_version, "base_content_version": None}]

    report = await backfill(store, apis)

    assert reasons(report) == [(total, Reason.WOULD_STORE, None)]
    assert report.write is False
    assert apis.sent("PUT") == []
    assert apis.recipes == {}
    assert await store.get_mint_record(root) is None


async def test_a_second_write_run_finds_every_recipe_already_stored(
    store: InMemoryMaterialisationStore,
) -> None:
    total = await summed(await seed_root(store))
    change = await grown(total)
    apis = FakeApis(published=[total, change])
    await backfill(store, apis, write=True)
    stored = copy.deepcopy(apis.recipes)
    puts = len(apis.sent("PUT"))

    again = await backfill(store, apis, write=True)
    dry = await backfill(store, apis)

    for report in (again, dry):
        assert [outcome.reason for outcome in report.outcomes] == [Reason.ALREADY_STORED] * 2
    assert apis.recipes == stored
    assert len(apis.sent("PUT")) == puts + 2


async def test_a_root_with_no_mint_record_is_recovered_once_and_written_back(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store, mint_record=False)
    total = await summed(root)
    change = await grown(total)
    apis = FakeApis(published=[total, change])
    apis.versions[SPEC] = [
        {"content_version": "cv_" + "9" * 32, "base_content_version": None},
        {"content_version": root.content_version, "base_content_version": None},
    ]

    report = await backfill(store, apis, write=True)

    recovered = MintRecord(frame=root, dataset=root, options=OPTIONS, stat_var_count=1)
    assert [outcome.reason for outcome in report.outcomes] == [Reason.STORED] * 2
    assert await store.get_mint_record(root) == recovered
    assert apis.recipes[pair(change)]["roots"] == [recovered.model_dump(mode="json")]
    versions_reads = [r for r in apis.requests if r.url.path.endswith(f"{SPEC}/versions")]
    assert len(versions_reads) == 1

    # A later run reads the record from the store and asks apis for no version.
    later = FakeApis(published=[total, change])
    await backfill(store, later, write=True)
    assert [r for r in later.requests if r.url.path.endswith(f"{SPEC}/versions")] == []


@pytest.mark.parametrize(
    ("versions", "why"),
    [
        ([], "no_version_matches"),
        (
            [{"content_version": ROOT.content_version, "base_content_version": "cv_" + "8" * 32}],
            "groupby_unrecorded",
        ),
    ],
)
async def test_a_frame_over_an_unrecoverable_root_is_reported_and_sends_no_recipe(
    store: InMemoryMaterialisationStore, versions: list[dict[str, str | None]], why: str
) -> None:
    root = await seed_root(store, mint_record=False)
    total = await summed(root)
    apis = FakeApis(published=[total])
    apis.versions[SPEC] = versions

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [
        (total, Reason.ROOT_UNRECOVERABLE, f"{root.spec_id} {root.content_version} {why}")
    ]
    assert apis.sent("PUT") == []
    assert await store.get_mint_record(root) is None
    assert report.settled


async def test_a_frame_with_a_missing_ancestor_names_it_and_sends_no_recipe(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store)
    total = await summed(root)
    change = await grown(total)
    thinned = await without(store, total, [root, change])
    apis = FakeApis(published=[change])

    report = await backfill(thinned, apis, write=True)

    assert reasons(report) == [
        (change, Reason.MISSING_ANCESTOR, f"{total.spec_id} {total.content_version}")
    ]
    assert apis.sent("PUT") == []


async def test_a_frame_that_cannot_be_rebuilt_but_holds_a_recipe_counts_as_stored(
    store: InMemoryMaterialisationStore,
) -> None:
    """A frame published with its recipe needs no ancestor left in the bucket."""
    root = await seed_root(store)
    total = await summed(root)
    change = await grown(total)
    apis = FakeApis(published=[change])
    apis.recipes[pair(change)] = await build_recipe(change, store=store)
    thinned = await without(store, total, [root, change])

    report = await backfill(thinned, apis, write=True)

    assert reasons(report) == [
        (
            change,
            Reason.ALREADY_STORED,
            f"not rebuilt: missing_ancestor {total.spec_id} {total.content_version}",
        )
    ]
    assert apis.sent("PUT") == []


async def test_a_frame_dfo_apis_never_published_is_not_a_failure(
    store: InMemoryMaterialisationStore,
) -> None:
    total = await summed(await seed_root(store))
    apis = FakeApis()

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [(total, Reason.NOT_PUBLISHED, None)]
    assert report.settled


@pytest.mark.parametrize("write", [True, False])
@pytest.mark.parametrize("member", ["normal_form_hash", "replay_key"])
async def test_a_different_stored_recipe_is_a_conflict_with_its_constraint_id(
    store: InMemoryMaterialisationStore, member: str, write: bool
) -> None:
    total = await summed(await seed_root(store))
    apis = FakeApis(published=[total])
    held = {**await build_recipe(total, store=store), member: "another"}
    apis.recipes[pair(total)] = held

    report = await backfill(store, apis, write=write)

    assert reasons(report) == [(total, Reason.CONFLICT, f"recipe_requires_matching_{member}")]
    assert apis.recipes[pair(total)] == held


async def test_a_replay_key_another_frame_holds_is_a_conflict(
    store: InMemoryMaterialisationStore,
) -> None:
    total = await summed(await seed_root(store))
    apis = FakeApis(published=[total])
    apis.recipes[("ops_" + "f" * 32, "ocv_" + "f" * 32)] = await build_recipe(total, store=store)

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [(total, Reason.CONFLICT, "replay_key_requires_one_frame")]


@pytest.mark.parametrize(
    ("answer", "detail", "attempts"),
    [
        (
            problem(503, "service_unavailable", retryable=True),
            "status=503 code=service_unavailable",
            3,
        ),
        (httpx2.ConnectError("no route"), "status=None code=None", 3),
        (problem(400, "request_invalid"), "status=400 code=request_invalid", 1),
    ],
)
async def test_a_failed_call_is_retried_as_the_client_retries_then_recorded_and_the_run_goes_on(
    store: InMemoryMaterialisationStore,
    answer: httpx2.Response | Exception,
    detail: str,
    attempts: int,
) -> None:
    root = await seed_root(store)
    total = await summed(root)
    low = await summed(root, aggregation="min")
    apis = FakeApis(published=[total, low])
    apis.answers[total.spec_id] = answer

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [(total, Reason.APIS_FAILURE, detail), (low, Reason.STORED, None)]
    assert len([r for r in apis.sent("PUT") if total.spec_id in r.url.path]) == attempts
    assert not report.settled


async def test_an_error_on_one_frame_is_recorded_and_the_run_goes_on(
    store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = await seed_root(store)
    total = await summed(root)
    low = await summed(root, aggregation="min")
    apis = FakeApis(published=[total, low])
    read_trace = store.get_trace

    async def unreadable(ref: FrameRef) -> Any:
        if ref == total:
            raise OSError("bucket read failed")
        return await read_trace(ref)

    monkeypatch.setattr(store, "get_trace", unreadable)

    report = await backfill(store, apis, write=True)

    assert reasons(report) == [
        (total, Reason.UNEXPECTED_ERROR, "OSError"),
        (low, Reason.STORED, None),
    ]
    assert not report.settled


async def test_bounds_take_the_oldest_frames_in_range(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store)
    frames = [await summed(root, aggregation=name) for name in ("sum", "min", "max", "mean")]
    written = {pair(listing.ref): listing.created_at for listing in await store.list_frames()}
    apis = FakeApis(published=frames)

    middle = await backfill(
        store, apis, since=written[pair(frames[1])], until=written[pair(frames[3])]
    )
    first = await backfill(store, apis, limit=1)

    assert [outcome.ref for outcome in middle.outcomes] == frames[1:3]
    assert [outcome.ref for outcome in first.outcomes] == frames[:1]
    assert first.derived_held == 4


async def test_no_more_than_the_fixed_number_of_frames_are_in_flight(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store)
    totals = [await summed(root, aggregation=name) for name in ("sum", "min", "max", "mean")]
    frames = totals + [await grown(total) for total in totals]
    apis = FakeApis(published=frames)

    report = await backfill(store, apis, write=True)

    assert len(report.outcomes) == 8 > FRAMES_IN_FLIGHT
    assert apis.most_in_flight == FRAMES_IN_FLIGHT


async def test_each_frames_trace_is_read_from_the_store_once(
    store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace read is a whole object downloaded from the bucket, so an ancestor's is
    kept for the frames built on it."""
    total = await summed(await seed_root(store))
    change = await grown(total)
    reads: list[FrameRef] = []
    read_trace = store.get_trace

    async def counted(ref: FrameRef) -> Any:
        reads.append(ref)
        return await read_trace(ref)

    monkeypatch.setattr(store, "get_trace", counted)

    await backfill(store, FakeApis(published=[total, change]), write=True)

    assert sorted(reads, key=pair) == sorted([total, change], key=pair)


async def test_the_report_counts_what_holds_a_recipe_and_what_does_not_and_shows_no_secret(
    store: InMemoryMaterialisationStore,
) -> None:
    root = await seed_root(store)
    total = await summed(root)
    low = await summed(root, aggregation="min")
    high = await summed(root, aggregation="max")
    apis = FakeApis(published=[total, high])
    apis.answers[high.spec_id] = problem(403, "publish_not_permitted")

    report = await backfill(store, apis, write=True)
    text = report.render()

    assert report.counts() == {
        Reason.STORED: 1,
        Reason.NOT_PUBLISHED: 1,
        Reason.APIS_FAILURE: 1,
    }
    assert "Derived frames with a recipe stored in dfo-apis: 1" in text
    assert "Derived frames without a stored recipe: 1" in text
    assert "Not published in dfo-apis, so no card draws on them: 1" in text
    # The oldest object is the root, written a minute after 14 September.
    assert "Oldest frame object: 23.0 days (written 2026-09-14T00:01:00Z)" in text
    for ref, reason in ((total, "stored"), (low, "not_published"), (high, "apis_failure")):
        assert any(
            line.startswith(reason) and f"{ref.spec_id} {ref.content_version}" in line
            for line in text.splitlines()
        )
    assert PRINCIPAL not in text


async def test_a_refused_principal_stops_the_run_before_any_frame_is_read(
    store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await summed(await seed_root(store))
    apis = FakeApis()

    async def never(*_: object) -> Any:
        raise AssertionError("the store is not read once dfo-apis refuses the principal")

    monkeypatch.setattr(store, "list_frames", never)

    async with apis.client(principal="not-the-principal") as client:
        with pytest.raises(LiveApisError) as refused:
            await run_backfill(store=store, client=client, write=True)

    assert refused.value.code == "publish_not_permitted"
    assert len(apis.requests) == 1


@pytest.mark.parametrize(
    "unset", ["DFO_APIS_URL", "DATASETS_FIRST_PARTY_PRINCIPAL", "FRAME_STORE_BUCKET"]
)
def test_a_missing_setting_fails_at_start_naming_it_and_printing_no_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], unset: str
) -> None:
    values = {
        "DFO_APIS_URL": "http://apis.test",
        "DATASETS_FIRST_PARTY_PRINCIPAL": PRINCIPAL,
        "FRAME_STORE_BUCKET": "frames-bucket",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, " " if name == unset else value)

    def never(**_: object) -> Any:
        raise AssertionError("no store is built when a setting is missing")

    monkeypatch.setattr(recipe_backfill, "build_store", never)

    assert recipe_backfill.main(["--write"]) == 2

    printed = capsys.readouterr()
    assert printed.out == ""
    assert unset in printed.err
    assert PRINCIPAL not in printed.err


def test_the_default_is_a_dry_run() -> None:
    parse = recipe_backfill._arguments

    assert parse([]).write is False
    assert parse(["--dry-run"]).write is False
    assert parse(["--write"]).write is True
    assert parse(["--since", "2026-09-20"]).since == datetime(2026, 9, 20, tzinfo=UTC)
