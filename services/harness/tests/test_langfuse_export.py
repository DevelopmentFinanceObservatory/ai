"""`harness.langfuse_export`: the row mapping, the window rule, paging and pacing
against Langfuse, and the per-day partition replace in BigQuery.

Langfuse is `httpx2.MockTransport` replaying `fixtures/langfuse_observations_turn.json`,
two real pages (limit 3) of prod observations from one turn, recorded with
`statusMessage` removed, and answering the day-grouped metrics query from a count
map. BigQuery is `FakeBigQuery`, which records the calls the
job makes.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from google.cloud import bigquery
from harness import langfuse_export
from harness.langfuse_export import (
    ACCESS_DAYS,
    BACKFILL_FLOOR,
    REQUEST_INTERVAL_S,
    SCHEMA,
    EmptyWindowError,
    IncompleteDaysError,
    LangfuseError,
    LangfuseReader,
    check_day,
    export_days,
    retained_from,
    run_export,
)

RECORDING = json.loads(
    (Path(__file__).parent / "fixtures" / "langfuse_observations_turn.json").read_text()
)
RECORDED_DAY = date(2026, 10, 20)
TABLE_ID = "proj.langfuse_export.observations"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def recorded_langfuse(
    requests: list[httpx2.Request], counts: dict[date, int] | None = None
) -> httpx2.MockTransport:
    """Answers the recorded pages for `RECORDED_DAY` by cursor, an empty page for any
    other day, and the day-grouped metrics query with `counts`, by default the
    recording's four observations on `RECORDED_DAY`."""
    pages = {page["cursor"]: page["answer"] for page in RECORDING["pages"]}
    counts = {RECORDED_DAY: 4} if counts is None else counts

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path == "/api/public/v2/metrics":
            data = [
                {"time_dimension": day.isoformat(), "count_count": n}
                for day, n in sorted(counts.items())
            ]
            return httpx2.Response(200, json={"data": data})
        if not request.url.params["fromStartTime"].startswith(RECORDED_DAY.isoformat()):
            return httpx2.Response(200, json={"data": [], "meta": {}})
        return httpx2.Response(200, json=pages[request.url.params.get("cursor")])

    return httpx2.MockTransport(handle)


def reader_for(transport: httpx2.MockTransport, clock: FakeClock | None = None) -> LangfuseReader:
    clock = clock or FakeClock()
    return LangfuseReader(
        client=httpx2.Client(base_url="https://langfuse.test", transport=transport),
        environment="dfo-prod-ai",
        sleep=clock.sleep,
        clock=clock.clock,
    )


class FakeJob:
    def result(self) -> None:
        return None


class FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def result(self) -> list[dict[str, Any]]:
        return self._rows


class FakeBigQuery:
    def __init__(self, *, latest: date | None, held: dict[date, int] | None = None) -> None:
        self.latest = latest
        self.held = held or {}
        self.created: list[bigquery.Table] = []
        self.queries: list[str] = []
        self.loads: list[tuple[str, list[dict[str, Any]], bigquery.LoadJobConfig]] = []
        self.deleted: list[str] = []

    def create_table(self, table: bigquery.Table, *, exists_ok: bool) -> None:
        assert exists_ok
        self.created.append(table)

    def query(self, sql: str) -> FakeQuery:
        self.queries.append(sql)
        if "COUNT(*)" in sql:
            first = date.fromisoformat(sql.split("TIMESTAMP('")[1][:10])
            return FakeQuery(
                [{"day": day, "row_count": n} for day, n in self.held.items() if day >= first]
            )
        return FakeQuery([{"day": self.latest}])

    def load_table_from_json(
        self, rows: list[dict[str, Any]], destination: str, *, job_config: bigquery.LoadJobConfig
    ) -> FakeJob:
        self.loads.append((destination, rows, job_config))
        return FakeJob()

    def delete_table(self, table: str, *, not_found_ok: bool) -> None:
        assert not_found_ok
        self.deleted.append(table)


def test_recorded_pages_map_to_rows_without_free_text():
    requests: list[httpx2.Request] = []
    reader = reader_for(recorded_langfuse(requests))

    rows = [langfuse_export.observation_row(o) for o in reader.observations(RECORDED_DAY)]

    assert len(requests) == 2
    first = requests[0].url.params
    assert first["fields"].split(",") == [
        "core",
        "basic",
        "model",
        "usage",
        "metrics",
        "trace_context",
    ]
    assert first["environment"] == "dfo-prod-ai"
    assert first["fromStartTime"] == "2026-10-20T00:00:00Z"
    assert first["toStartTime"] == "2026-10-21T00:00:00Z"
    assert "cursor" not in first
    assert requests[1].url.params["cursor"] == RECORDING["pages"][0]["answer"]["meta"]["cursor"]

    assert len(rows) == 4
    columns = {field.name for field in SCHEMA}
    required = {field.name for field in SCHEMA if field.mode == "REQUIRED"}
    for row in rows:
        assert set(row) == columns
        assert all(row[name] is not None for name in required)

    generation = next(row for row in rows if row["type"] == "GENERATION")
    assert generation == {
        "id": "0000000000000006",
        "trace_id": "00000000000000000000000000000002",
        "parent_observation_id": "0000000000000007",
        "is_root_observation": False,
        "session_id": "00000000-0000-4000-8000-000000000004",
        "trace_name": "turn",
        "name": "generate_content gemini-3.8-flash",
        "type": "GENERATION",
        "level": "DEFAULT",
        "environment": "dfo-prod-ai",
        "model": "gemini-3.8-flash",
        "start_time": "2026-10-20T22:03:17.037Z",
        "end_time": "2026-10-20T22:03:22.835Z",
        "latency_seconds": 5.798,
        "time_to_first_token_seconds": 4.33,
        "input_tokens": 17352,
        "cached_input_tokens": 10319,
        "output_tokens": 238,
        "total_tokens": 17590,
        "input_cost_usd": 0.006048675,
        "output_cost_usd": 0.0008925,
        "total_cost_usd": 0.006941175,
    }
    # Langfuse's empty strings land as NULL.
    tool = next(row for row in rows if row["type"] == "TOOL")
    assert tool["model"] is None


@pytest.mark.parametrize(
    ("usage", "cached"),
    [
        ({"usageDetails": {"input": 718, "input_cached_tokens": 10825}}, 10825),
        # Spans carry `{}`.
        ({"usageDetails": {}}, None),
        ({"usageDetails": None}, None),
        ({}, None),
    ],
)
def test_cached_input_tokens_come_from_usage_details(usage, cached):
    observation = {
        key: value
        for key, value in RECORDING["pages"][0]["answer"]["data"][0].items()
        if key != "usageDetails"
    }

    row = langfuse_export.observation_row({**observation, **usage})

    assert row["cached_input_tokens"] == cached


def test_cached_input_tokens_follow_input_tokens():
    names = [field.name for field in SCHEMA]

    assert names[names.index("input_tokens") + 1] == "cached_input_tokens"
    assert next(f for f in SCHEMA if f.name == "cached_input_tokens").field_type == "INT64"


def test_mapping_drops_status_message_and_user_id():
    observation = {
        **RECORDING["pages"][0]["answer"]["data"][0],
        "statusMessage": "echo of the user's question",
        "userId": "someone",
        "input": "prompt",
        "metadata": {"k": "v"},
    }

    row = langfuse_export.observation_row(observation)

    assert "echo of the user's question" not in json.dumps(row)
    assert "someone" not in json.dumps(row)


@pytest.mark.parametrize(
    ("latest", "today", "first", "last"),
    [
        (None, date(2026, 10, 23), BACKFILL_FLOOR, date(2026, 10, 23)),
        (date(2026, 10, 20), date(2026, 10, 23), date(2026, 10, 19), date(2026, 10, 23)),
        (date(2026, 10, 23), date(2026, 10, 23), date(2026, 10, 22), date(2026, 10, 23)),
        (BACKFILL_FLOOR, BACKFILL_FLOOR, BACKFILL_FLOOR, BACKFILL_FLOOR),
    ],
)
def test_export_days_replace_from_latest_minus_one(latest, today, first, last):
    days = export_days(latest_loaded=latest, today=today, langfuse={}, held={})

    assert days[0] == first
    assert days[-1] == last
    assert len(days) == (last - first).days + 1


@pytest.mark.parametrize(
    ("langfuse", "held", "added"),
    [
        # A day a run read short stays in every later window until its counts match.
        ({date(2026, 10, 25): 4}, {date(2026, 10, 25): 3}, [date(2026, 10, 25)]),
        # A day Langfuse holds and BigQuery lacks.
        ({date(2026, 10, 24): 4}, {}, [date(2026, 10, 24)]),
        # A partition Langfuse no longer counts.
        ({}, {date(2026, 10, 24): 4}, [date(2026, 10, 24)]),
        ({date(2026, 10, 25): 4}, {date(2026, 10, 25): 4}, []),
        # The oldest day the access window shows whole is compared, the day before
        # it is not.
        ({date(2026, 10, 22): 4}, {date(2026, 10, 22): 3}, [date(2026, 10, 22)]),
        ({date(2026, 10, 21): 4}, {date(2026, 10, 21): 3}, []),
    ],
)
def test_export_days_add_retained_days_whose_counts_differ(langfuse, held, added):
    today = date(2026, 11, 20)
    assert retained_from(today) == date(2026, 10, 22)
    latest = date(2026, 11, 18)

    days = export_days(latest_loaded=latest, today=today, langfuse=langfuse, held=held)

    assert days == [*added, *export_days(latest_loaded=latest, today=today, langfuse={}, held={})]


@pytest.mark.parametrize(
    ("since", "first"),
    [
        (date(2026, 10, 5), date(2026, 10, 5)),
        # A day inside the usual window adds nothing.
        (date(2026, 11, 19), date(2026, 11, 17)),
    ],
)
def test_from_widens_the_window_to_its_day(since, first):
    today = date(2026, 11, 20)

    days = export_days(
        latest_loaded=date(2026, 11, 18), today=today, langfuse={}, held={}, since=since
    )

    assert days == [first + timedelta(days=offset) for offset in range((today - first).days + 1)]


def test_from_reads_old_days_and_keeps_a_partition_that_reads_short(capsys):
    """`--from` re-reads days before the usual window, and a day that reads fewer rows
    than its partition holds keeps the partition."""
    today = RECORDED_DAY + timedelta(days=ACCESS_DAYS)
    bq = FakeBigQuery(latest=today - timedelta(days=1), held={RECORDED_DAY: 5})
    requests: list[httpx2.Request] = []

    loaded = run_export(
        reader=reader_for(recorded_langfuse(requests, {})),
        bq=bq,
        table_id=TABLE_ID,
        today=today,
        since=BACKFILL_FLOOR,
    )

    assert loaded == []
    assert bq.loads == []
    days_read = {
        r.url.params["fromStartTime"][:10] for r in requests if "fromStartTime" in r.url.params
    }
    assert len(days_read) == (today - BACKFILL_FLOOR).days + 1
    recorded_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("2026-10-20")
    )
    assert recorded_line == (
        "2026-10-20: partition kept; read 4 rows, the partition holds 5, "
        "older than Langfuse's access window"
    )


@pytest.mark.parametrize(
    ("today", "first"),
    [
        (date(2026, 11, 30), date(2026, 11, 1)),
        # The floor holds whole: nothing before it was ever exported.
        (date(2026, 10, 23), BACKFILL_FLOOR),
    ],
)
def test_retained_from_is_the_oldest_whole_day(today, first):
    assert retained_from(today) == first
    assert (today - first).days <= ACCESS_DAYS - 1


def logged_days(output: str) -> list[str]:
    return [line.split(":")[0] for line in output.splitlines()]


def test_run_replaces_each_day_partition_and_keeps_empty_days(capsys):
    bq = FakeBigQuery(latest=date(2026, 10, 21))
    reader = reader_for(recorded_langfuse([]))

    loaded = run_export(reader=reader, bq=bq, table_id=TABLE_ID, today=date(2026, 10, 22))

    # The table holds 10-21, so the run reaches back to 10-20 and on to today.
    assert loaded == [(date(2026, 10, 20), 4)]
    assert logged_days(capsys.readouterr().out) == ["2026-10-20", "2026-10-21", "2026-10-22"]
    [table] = bq.created
    assert [field.name for field in table.schema] == [field.name for field in SCHEMA]
    assert table.time_partitioning.field == "start_time"
    assert table.time_partitioning.type_ == bigquery.TimePartitioningType.DAY
    # The partition counts reach back to the oldest day Langfuse holds whole.
    assert bq.queries == [
        f"SELECT DATE(MAX(start_time)) AS day FROM `{TABLE_ID}`",
        f"SELECT DATE(start_time) AS day, COUNT(*) AS row_count FROM `{TABLE_ID}` "
        f"WHERE start_time >= TIMESTAMP('{BACKFILL_FLOOR}') GROUP BY day",
    ]

    [(destination, rows, config)] = bq.loads
    assert destination == f"{TABLE_ID}$20261020"
    assert len(rows) == 4
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert config.source_format == bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    assert config.time_partitioning.field == "start_time"
    # 10-21 and 10-22 read empty, so their partitions are left as they are.
    assert bq.deleted == []


def test_a_load_adds_cached_input_tokens_to_an_older_table():
    """A table made before `cached_input_tokens` gains the column through the load of
    one partition, which leaves the other partitions as they are."""
    bq = FakeBigQuery(latest=date(2026, 10, 21))

    run_export(
        reader=reader_for(recorded_langfuse([])), bq=bq, table_id=TABLE_ID, today=date(2026, 10, 21)
    )

    [(destination, rows, config)] = bq.loads
    assert destination == f"{TABLE_ID}$20261020"
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert config.schema_update_options == [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    assert "cached_input_tokens" in [field.name for field in config.schema]
    assert all("cached_input_tokens" in row for row in rows)


def test_an_empty_window_fails_and_touches_no_partition():
    """A wrong environment or keys for another project read as empty."""
    bq = FakeBigQuery(latest=date(2026, 10, 21), held={date(2026, 10, 21): 9})
    empty = httpx2.MockTransport(
        lambda request: httpx2.Response(200, json={"data": [], "meta": {}})
    )

    with pytest.raises(EmptyWindowError, match="2026-10-20 to 2026-10-23"):
        run_export(reader=reader_for(empty), bq=bq, table_id=TABLE_ID, today=date(2026, 10, 23))

    assert bq.loads == []
    assert bq.deleted == []


@pytest.mark.parametrize("held", [4, 3])
def test_a_read_holding_the_partition_replaces_it(capsys, held):
    bq = FakeBigQuery(latest=date(2026, 10, 21), held={RECORDED_DAY: held})

    loaded = run_export(
        reader=reader_for(recorded_langfuse([])),
        bq=bq,
        table_id=TABLE_ID,
        today=date(2026, 10, 21),
    )

    assert loaded == [(RECORDED_DAY, 4)]
    assert [destination for destination, _, _ in bq.loads] == [f"{TABLE_ID}$20261020"]
    assert capsys.readouterr().out.splitlines()[0].startswith("2026-10-20: replaced")


@pytest.mark.parametrize(
    ("held", "counts", "replaced"),
    [
        # Fewer rows than the partition holds: the partition is kept.
        ({RECORDED_DAY: 5}, {RECORDED_DAY: 4}, False),
        # Fewer rows than Langfuse counts: the read still grows the partition.
        ({RECORDED_DAY: 3}, {RECORDED_DAY: 6}, True),
    ],
)
def test_a_retained_day_read_short_fails_after_the_other_days(capsys, held, counts, replaced):
    bq = FakeBigQuery(latest=date(2026, 10, 21), held=held)

    with pytest.raises(IncompleteDaysError) as failure:
        run_export(
            reader=reader_for(recorded_langfuse([], counts)),
            bq=bq,
            table_id=TABLE_ID,
            today=date(2026, 10, 22),
        )

    message = str(failure.value)
    assert "2026-10-20 read 4 rows" in message
    assert f"Langfuse counts {counts[RECORDED_DAY]}" in message
    assert f"the partition holds {held[RECORDED_DAY]}" in message
    assert [destination for destination, _, _ in bq.loads] == (
        [f"{TABLE_ID}$20261020"] if replaced else []
    )
    # The later days were still read.
    assert logged_days(capsys.readouterr().out) == ["2026-10-20", "2026-10-21", "2026-10-22"]


def test_a_day_missing_from_bigquery_is_loaded():
    """A run that stopped part-way left 10-20 unloaded while later days loaded."""
    bq = FakeBigQuery(latest=date(2026, 10, 25))
    requests: list[httpx2.Request] = []

    loaded = run_export(
        reader=reader_for(recorded_langfuse(requests)),
        bq=bq,
        table_id=TABLE_ID,
        today=date(2026, 10, 26),
    )

    assert loaded == [(RECORDED_DAY, 4)]
    days_read = sorted(
        {r.url.params["fromStartTime"][:10] for r in requests if "fromStartTime" in r.url.params}
    )
    assert days_read == ["2026-10-20", "2026-10-24", "2026-10-25", "2026-10-26"]
    [metrics] = [r for r in requests if r.url.path == "/api/public/v2/metrics"]
    query = json.loads(metrics.url.params["query"])
    assert query["timeDimension"] == {"granularity": "day"}
    assert query["fromTimestamp"] == f"{BACKFILL_FLOOR}T00:00:00Z"
    assert query["toTimestamp"] == "2026-10-27T00:00:00Z"
    assert query["filters"][0]["value"] == ["dfo-prod-ai"]


@pytest.mark.parametrize(("age", "fails"), [(ACCESS_DAYS - 1, True), (ACCESS_DAYS, False)])
def test_a_short_day_fails_inside_the_access_window_and_is_kept_past_it(capsys, age, fails):
    """Hobby's access window hides older observations, so a day past it reads partial."""
    today = RECORDED_DAY + timedelta(days=age)
    bq = FakeBigQuery(latest=date(2026, 10, 21), held={RECORDED_DAY: 5})
    reader = reader_for(recorded_langfuse([], {}))

    if fails:
        with pytest.raises(IncompleteDaysError, match="2026-10-20 read 4 rows"):
            run_export(reader=reader, bq=bq, table_id=TABLE_ID, today=today)
    else:
        assert run_export(reader=reader, bq=bq, table_id=TABLE_ID, today=today) == []

    assert bq.loads == []
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith("2026-10-20: partition kept")
    assert ("access window" in first_line) is not fails


def test_a_failure_partway_through_a_day_leaves_its_partition(capsys):
    recorded = recorded_langfuse([])

    def handle(request: httpx2.Request) -> httpx2.Response:
        # 10-21's first page answers and its second page keeps failing.
        if request.url.params.get("fromStartTime", "").startswith("2026-10-21"):
            if "cursor" in request.url.params:
                return httpx2.Response(502)
            return httpx2.Response(200, json=RECORDING["pages"][0]["answer"])
        return recorded.handle_request(request)

    bq = FakeBigQuery(latest=date(2026, 10, 21))

    with pytest.raises(LangfuseError):
        run_export(
            reader=reader_for(httpx2.MockTransport(handle)),
            bq=bq,
            table_id=TABLE_ID,
            today=date(2026, 10, 21),
        )

    assert [destination for destination, _, _ in bq.loads] == [f"{TABLE_ID}$20261020"]
    assert bq.deleted == []
    assert logged_days(capsys.readouterr().out) == ["2026-10-20", "2026-10-21"]


def test_empty_table_backfills_from_the_floor():
    bq = FakeBigQuery(latest=None)
    requests: list[httpx2.Request] = []

    loaded = run_export(
        reader=reader_for(recorded_langfuse(requests)),
        bq=bq,
        table_id=TABLE_ID,
        today=date(2026, 10, 20),
    )

    days_read = sorted(
        {
            request.url.params["fromStartTime"][:10]
            for request in requests
            if "fromStartTime" in request.url.params
        }
    )
    assert days_read[0] == BACKFILL_FLOOR.isoformat()
    assert len(days_read) == 16
    assert loaded == [(RECORDED_DAY, 4)]


def test_pages_are_paced_and_429_waits_for_retry_after():
    answers = [
        httpx2.Response(429, headers={"retry-after": "7"}),
        httpx2.Response(200, json=RECORDING["pages"][0]["answer"]),
        httpx2.Response(503),
        httpx2.Response(200, json=RECORDING["pages"][1]["answer"]),
    ]
    clock = FakeClock()
    reader = reader_for(httpx2.MockTransport(lambda request: answers.pop(0)), clock)

    observations = list(reader.observations(RECORDED_DAY))

    assert len(observations) == 4
    assert not answers
    # 429 waits its Retry-After, which already covers the pacing interval. The 503
    # backs off two seconds, after which the retry needs no further pacing wait.
    assert clock.sleeps == [7.0, REQUEST_INTERVAL_S, langfuse_export.BACKOFF_S]


def test_persistent_server_errors_raise():
    reader = reader_for(httpx2.MockTransport(lambda request: httpx2.Response(502)))

    with pytest.raises(LangfuseError):
        list(reader.observations(RECORDED_DAY))


@pytest.mark.parametrize(
    ("totals", "cached", "matched"),
    [
        ({"count_count": 4, "sum_totalCost": 0.006941175}, 10319, True),
        ({"count_count": 5, "sum_totalCost": 0.006941175}, 10319, False),
        ({"count_count": 4, "sum_totalCost": 0.007}, 10319, False),
        ({"count_count": 4, "sum_totalCost": 0.006941175}, 10320, False),
    ],
)
def test_check_day_compares_rows_with_metrics(totals, cached, matched):
    observations = recorded_langfuse([])
    # The usage query answers one row per usage key of the day.
    usage = [
        {"usageType": "input", "sum_usageByType": 7033},
        {"usageType": "input_cached_tokens", "sum_usageByType": cached},
        {"usageType": "output", "sum_usageByType": 238},
    ]

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/public/v2/metrics":
            query = json.loads(request.url.params["query"])
            assert query["fromTimestamp"] == "2026-10-20T00:00:00Z"
            assert query["filters"][0]["value"] == ["dfo-prod-ai"]
            if query["metrics"] == [{"measure": "usageByType", "aggregation": "sum"}]:
                assert query["dimensions"] == [{"field": "usageType"}]
                return httpx2.Response(200, json={"data": usage})
            return httpx2.Response(200, json={"data": [totals]})
        return observations.handle_request(request)

    ok, line = check_day(reader=reader_for(httpx2.MockTransport(handle)), day=RECORDED_DAY)

    assert ok is matched
    assert "rows 4 cost 0.006941175 cached input tokens 10319" in line


def test_load_refuses_without_bigquery_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("EXPORT_BQ_PROJECT", "EXPORT_BQ_DATASET"):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_EXPORT_ENVIRONMENT",
    ):
        monkeypatch.setenv(name, "set")

    assert langfuse_export.main([]) == 2
