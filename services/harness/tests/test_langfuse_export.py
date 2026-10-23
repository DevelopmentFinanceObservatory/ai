"""`harness.langfuse_export`: the row mapping, the window rule, paging and pacing
against Langfuse, and the per-day partition replace in BigQuery.

Langfuse is `httpx2.MockTransport` replaying `fixtures/langfuse_observations_turn.json`,
two real pages (limit 3) of prod observations from one turn, recorded with
`statusMessage` removed. BigQuery is `FakeBigQuery`, which records the calls the
job makes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import pytest
from google.cloud import bigquery
from harness import langfuse_export
from harness.langfuse_export import (
    BACKFILL_FLOOR,
    REQUEST_INTERVAL_S,
    SCHEMA,
    EmptyWindowError,
    LangfuseError,
    LangfuseReader,
    check_day,
    export_days,
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


def recorded_langfuse(requests: list[httpx2.Request]) -> httpx2.MockTransport:
    """Answers the recorded pages for `RECORDED_DAY` by cursor, and an empty page for
    any other day."""
    pages = {page["cursor"]: page["answer"] for page in RECORDING["pages"]}

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
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
            return FakeQuery([{"day": day, "row_count": n} for day, n in self.held.items()])
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
        "output_tokens": 238,
        "total_tokens": 17590,
        "input_cost_usd": 0.006048675,
        "output_cost_usd": 0.0008925,
        "total_cost_usd": 0.006941175,
    }
    # Langfuse's empty strings land as NULL.
    tool = next(row for row in rows if row["type"] == "TOOL")
    assert tool["model"] is None


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
    days = export_days(latest_loaded=latest, today=today)

    assert days[0] == first
    assert days[-1] == last
    assert len(days) == (last - first).days + 1


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
    assert bq.queries == [
        f"SELECT DATE(MAX(start_time)) AS day FROM `{TABLE_ID}`",
        f"SELECT DATE(start_time) AS day, COUNT(*) AS row_count FROM `{TABLE_ID}` "
        "WHERE start_time >= TIMESTAMP('2026-10-20') GROUP BY day",
    ]

    [(destination, rows, config)] = bq.loads
    assert destination == f"{TABLE_ID}$20261020"
    assert len(rows) == 4
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert config.source_format == bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    assert config.time_partitioning.field == "start_time"
    assert config.schema_update_options == [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    # 10-21 and 10-22 read empty, so their partitions are left as they are.
    assert bq.deleted == []


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


@pytest.mark.parametrize(("held", "replaced"), [(5, False), (4, True), (3, True)])
def test_a_replace_never_shrinks_a_partition(capsys, held, replaced):
    """Langfuse deletes data by age, so an old day can come back partial."""
    bq = FakeBigQuery(latest=date(2026, 10, 21), held={RECORDED_DAY: held})

    loaded = run_export(
        reader=reader_for(recorded_langfuse([])),
        bq=bq,
        table_id=TABLE_ID,
        today=date(2026, 10, 21),
    )

    assert loaded == ([(RECORDED_DAY, 4)] if replaced else [])
    assert [destination for destination, _, _ in bq.loads] == (
        [f"{TABLE_ID}$20261020"] if replaced else []
    )
    first_line = capsys.readouterr().out.splitlines()[0]
    assert ("replaced" in first_line) is replaced


def test_a_failure_partway_through_a_day_leaves_its_partition(capsys):
    recorded = recorded_langfuse([])

    def handle(request: httpx2.Request) -> httpx2.Response:
        # 10-21's first page answers and its second page keeps failing.
        if request.url.params["fromStartTime"].startswith("2026-10-21"):
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

    days_read = sorted({request.url.params["fromStartTime"][:10] for request in requests})
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
    ("metrics", "matched"),
    [
        ({"count_count": 4, "sum_totalCost": 0.006941175}, True),
        ({"count_count": 5, "sum_totalCost": 0.006941175}, False),
        ({"count_count": 4, "sum_totalCost": 0.007}, False),
    ],
)
def test_check_day_compares_rows_with_metrics(metrics, matched):
    observations = recorded_langfuse([])

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/public/v2/metrics":
            query = json.loads(request.url.params["query"])
            assert query["fromTimestamp"] == "2026-10-20T00:00:00Z"
            assert query["filters"][0]["value"] == ["dfo-prod-ai"]
            return httpx2.Response(200, json={"data": [metrics]})
        return observations.handle_request(request)

    ok, line = check_day(reader=reader_for(httpx2.MockTransport(handle)), day=RECORDED_DAY)

    assert ok is matched
    assert "rows 4 cost 0.006941175" in line


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
