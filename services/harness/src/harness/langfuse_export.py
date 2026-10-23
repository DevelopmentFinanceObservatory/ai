"""Loads Langfuse observation rows into BigQuery. Run as `python -m harness.langfuse_export`.

Each run reads `GET /api/public/v2/observations` for one Langfuse environment,
one UTC day at a time, and replaces that day's partition of the BigQuery table
`observations`, which is day-partitioned on `start_time`. The days are the latest
loaded day minus one through today, plus every day Langfuse still holds whole
(`retained_from` on) whose observation count, from one `GET /api/public/v2/metrics`
query grouped by day, differs from its partition's row count. An empty table is
backfilled from `BACKFILL_FLOOR`, the first day the prod project holds data. A
run that fails part-way leaves every day it reached replaced whole and every
other day as it was, and the next run's count comparison picks up the gap.

A replace never shrinks a partition: a day that reads fewer rows than its
partition holds, none included, keeps the partition and prints both counts.
Langfuse deletes data by age, so a day before `retained_from` can come back
partial, and a deleted row could not be reloaded; such a day is kept and logged.
A day from `retained_from` on that reads fewer rows than its partition holds or
than Langfuse counts is a fault. The run processes the other days and then
raises `IncompleteDaysError` naming every such day with its counts, and the day
stays in later runs until the counts match. When every day of the run reads no
rows, the run raises `EmptyWindowError` naming the days and
`LANGFUSE_EXPORT_ENVIRONMENT`, since a wrong environment or keys for another
project read as empty and prod has had turns every day since `BACKFILL_FLOOR`.
Each day prints a line as it is replaced or kept, and a failure prints the day it
stopped on before the error.

The rows carry ids, names, the model, times, tokens and cost. Requests omit the
`io` and `metadata` field groups, so prompts and answers stay in
Langfuse, and `observation_row` copies named fields only, which keeps
`statusMessage` (it can echo user input) and `userId` out of BigQuery.

Langfuse Cloud counts every page against the organisation's General API bucket
(30 requests a minute on Hobby), so pages go out at most one per
`REQUEST_INTERVAL_S`. A 429 waits for its `Retry-After`. A 5xx or a transport
error is retried with backoff.

`--check-day DAY` reads one day and compares the row count and summed cost the
mapping produces with `GET /api/public/v2/metrics` for the same day and
environment. It writes nothing.

Settings come from the environment or `.env`: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_EXPORT_ENVIRONMENT`, and for
a load also `EXPORT_BQ_PROJECT` and `EXPORT_BQ_DATASET`.

Exit status: 0 when every day loaded or kept or the check matched, 1 when the
check found a difference, a request failed, a retained day read short or the
whole run read no rows, 2
when a setting is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx2
from google.cloud import bigquery
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKFILL_FLOOR = date(2026, 10, 5)
# Langfuse Cloud's Hobby plan holds 30 days of data
# (https://langfuse.com/docs/administration/data-retention), deleting
# observations by `start_time` age.
RETENTION_DAYS = 30
TABLE = "observations"

# Every group except `io` and `metadata` (free text) and `time` and `prompt`,
# which no column reads.
FIELD_GROUPS = "core,basic,model,usage,metrics,trace_context"
PAGE_LIMIT = 1000
# One page every two seconds stays under Hobby's 30 requests a minute.
REQUEST_INTERVAL_S = 2.0
ATTEMPTS = 5
BACKOFF_S = 2.0

SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("trace_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("parent_observation_id", "STRING"),
    bigquery.SchemaField("is_root_observation", "BOOL"),
    bigquery.SchemaField("session_id", "STRING"),
    bigquery.SchemaField("trace_name", "STRING"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "STRING"),
    bigquery.SchemaField("environment", "STRING"),
    bigquery.SchemaField("model", "STRING"),
    bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("end_time", "TIMESTAMP"),
    bigquery.SchemaField("latency_seconds", "FLOAT64"),
    bigquery.SchemaField("time_to_first_token_seconds", "FLOAT64"),
    bigquery.SchemaField("input_tokens", "INT64"),
    bigquery.SchemaField("output_tokens", "INT64"),
    bigquery.SchemaField("total_tokens", "INT64"),
    bigquery.SchemaField("input_cost_usd", "FLOAT64"),
    bigquery.SchemaField("output_cost_usd", "FLOAT64"),
    bigquery.SchemaField("total_cost_usd", "FLOAT64"),
]
PARTITIONING = bigquery.TimePartitioning(
    type_=bigquery.TimePartitioningType.DAY, field="start_time"
)


class ExportSettings(BaseSettings):
    """The job's settings, read from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    langfuse_export_environment: str | None = None
    export_bq_project: str | None = None
    export_bq_dataset: str | None = None

    def missing(self, *, load: bool) -> list[str]:
        """The env var of every setting the mode needs that is unset or blank."""
        names = [name for name in type(self).model_fields if load or not name.startswith("export_")]
        return [name.upper() for name in names if not (getattr(self, name) or "").strip()]


class LangfuseError(RuntimeError):
    """Langfuse kept failing a request after every retry, or refused it."""


class EmptyWindowError(RuntimeError):
    """Every day of the run's window read no rows, which a misconfiguration causes."""


class IncompleteDaysError(RuntimeError):
    """A day Langfuse still holds whole read fewer rows than its partition or
    Langfuse's own count."""


def _text(value: object) -> str | None:
    """Langfuse sends an unset string field as `""`, and BigQuery holds it as NULL."""
    return value if isinstance(value, str) and value else None


def observation_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    """The BigQuery row for one observation from the v2 API."""
    return {
        "id": observation["id"],
        "trace_id": observation["traceId"],
        "parent_observation_id": _text(observation.get("parentObservationId")),
        "is_root_observation": observation.get("isRootObservation"),
        "session_id": _text(observation.get("sessionId")),
        "trace_name": _text(observation.get("traceName")),
        "name": _text(observation.get("name")),
        "type": observation["type"],
        "level": _text(observation.get("level")),
        "environment": _text(observation.get("environment")),
        "model": _text(observation.get("model")),
        "start_time": observation["startTime"],
        "end_time": observation.get("endTime"),
        "latency_seconds": observation.get("latency"),
        "time_to_first_token_seconds": observation.get("timeToFirstToken"),
        "input_tokens": observation.get("inputUsage"),
        "output_tokens": observation.get("outputUsage"),
        "total_tokens": observation.get("totalUsage"),
        "input_cost_usd": observation.get("inputCost"),
        "output_cost_usd": observation.get("outputCost"),
        "total_cost_usd": observation.get("totalCost"),
    }


def retained_from(today: date) -> date:
    """The oldest day Langfuse still holds whole. Its age at the end of `today` is
    `RETENTION_DAYS`, so the day before it is the oldest Langfuse may have partly
    deleted."""
    return max(today - timedelta(days=RETENTION_DAYS - 1), BACKFILL_FLOOR)


def _catch_up_from(latest_loaded: date | None) -> date:
    if latest_loaded is None:
        return BACKFILL_FLOOR
    return max(latest_loaded - timedelta(days=1), BACKFILL_FLOOR)


def export_days(
    *,
    latest_loaded: date | None,
    today: date,
    langfuse: Mapping[date, int],
    held: Mapping[date, int],
) -> list[date]:
    """The days a run replaces: the latest loaded day minus one (or `BACKFILL_FLOOR`
    when nothing is loaded) through today, and every day from `retained_from` on
    whose Langfuse count differs from its partition's row count."""
    first = _catch_up_from(latest_loaded)
    days = {first + timedelta(days=offset) for offset in range((today - first).days + 1)}
    retained = retained_from(today)
    days.update(
        day
        for day in langfuse.keys() | held.keys()
        if retained <= day <= today and langfuse.get(day, 0) != held.get(day, 0)
    )
    return sorted(days)


def _day_bounds(day: date) -> tuple[str, str]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return (
        start.isoformat().replace("+00:00", "Z"),
        (start + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )


class LangfuseReader:
    """Reads observations and metrics from the Langfuse public API, one request at a
    time and at most one per `REQUEST_INTERVAL_S`."""

    def __init__(
        self,
        *,
        client: httpx2.Client,
        environment: str,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._environment = environment
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def _get(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        for attempt in range(1, ATTEMPTS + 1):
            if self._last_request is not None:
                wait = self._last_request + REQUEST_INTERVAL_S - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            try:
                response = self._client.get(path, params=params)
            except httpx2.TransportError as exc:
                if attempt == ATTEMPTS:
                    raise LangfuseError(f"GET {path} failed: {exc!r}") from exc
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            if response.status_code == 429 and attempt < ATTEMPTS:
                self._sleep(float(response.headers.get("retry-after", BACKOFF_S)))
                continue
            if response.status_code >= 500 and attempt < ATTEMPTS:
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            if response.status_code != 200:
                raise LangfuseError(f"GET {path} answered {response.status_code}")
            return response.json()
        raise AssertionError("unreachable: the last attempt returns or raises")

    def observations(self, day: date) -> Iterator[dict[str, Any]]:
        """Every observation of the environment whose start time falls on `day` (UTC)."""
        start, end = _day_bounds(day)
        params: dict[str, str | int] = {
            "fields": FIELD_GROUPS,
            "environment": self._environment,
            "fromStartTime": start,
            "toStartTime": end,
            "limit": PAGE_LIMIT,
        }
        while True:
            page = self._get("/api/public/v2/observations", params)
            yield from page["data"]
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or not page["data"]:
                return
            params["cursor"] = cursor

    def _metrics(
        self, *, first: date, last: date, metrics: list[dict[str, str]], by_day: bool
    ) -> list[dict[str, Any]]:
        """The metrics rows for the environment's observations from `first` through
        `last` (UTC), one row a day when `by_day` and one row otherwise."""
        query: dict[str, Any] = {
            "view": "observations",
            "metrics": metrics,
            "dimensions": [],
            "filters": [
                {
                    "column": "environment",
                    "operator": "any of",
                    "value": [self._environment],
                    "type": "stringOptions",
                }
            ],
            "fromTimestamp": _day_bounds(first)[0],
            "toTimestamp": _day_bounds(last)[1],
        }
        if by_day:
            query["timeDimension"] = {"granularity": "day"}
        return self._get("/api/public/v2/metrics", {"query": json.dumps(query)})["data"]

    def day_counts(self, *, first: date, last: date) -> dict[date, int]:
        """The observation count Langfuse's metrics report for each day from `first`
        through `last` that holds observations. The metrics bucket a day by UTC
        start time, the bounds `observations` reads."""
        rows = self._metrics(
            first=first,
            last=last,
            metrics=[{"measure": "count", "aggregation": "count"}],
            by_day=True,
        )
        return {date.fromisoformat(row["time_dimension"]): int(row["count_count"]) for row in rows}

    def day_totals(self, day: date) -> tuple[int, float]:
        """The observation count and summed total cost Langfuse's metrics report for `day`."""
        rows = self._metrics(
            first=day,
            last=day,
            metrics=[
                {"measure": "count", "aggregation": "count"},
                {"measure": "totalCost", "aggregation": "sum"},
            ],
            by_day=False,
        )
        if not rows:
            return 0, 0.0
        return int(rows[0]["count_count"]), float(rows[0]["sum_totalCost"] or 0.0)


def _table_id(settings: ExportSettings) -> str:
    return f"{settings.export_bq_project}.{settings.export_bq_dataset}.{TABLE}"


def latest_loaded_day(bq: bigquery.Client, table_id: str) -> date | None:
    """The latest day with a loaded row, or None for an empty table."""
    rows = list(bq.query(f"SELECT DATE(MAX(start_time)) AS day FROM `{table_id}`").result())
    return rows[0]["day"] if rows else None


def partition_row_counts(bq: bigquery.Client, table_id: str, first: date) -> dict[date, int]:
    """The loaded row count of every day from `first` on that holds rows."""
    sql = (
        f"SELECT DATE(start_time) AS day, COUNT(*) AS row_count FROM `{table_id}` "
        f"WHERE start_time >= TIMESTAMP('{first.isoformat()}') GROUP BY day"
    )
    return {row["day"]: row["row_count"] for row in bq.query(sql).result()}


def replace_day(bq: bigquery.Client, table_id: str, day: date, rows: list[dict[str, Any]]) -> None:
    """Replace `day`'s partition with `rows` in one load job."""
    partition = f"{table_id}${day:%Y%m%d}"
    config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        time_partitioning=PARTITIONING,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    bq.load_table_from_json(rows, partition, job_config=config).result()


def run_export(
    *, reader: LangfuseReader, bq: bigquery.Client, table_id: str, today: date
) -> list[tuple[date, int]]:
    """Create the table if absent and replace every day `export_days` names whose
    read holds rows and at least as many as its partition. Returns each replaced day
    with its row count. Raises `EmptyWindowError` when every day reads no rows, and
    otherwise `IncompleteDaysError` when a day from `retained_from` on read fewer
    rows than its partition or Langfuse's count."""
    table = bigquery.Table(table_id, schema=SCHEMA)
    table.time_partitioning = PARTITIONING
    bq.create_table(table, exists_ok=True)
    latest = latest_loaded_day(bq, table_id)
    retained = retained_from(today)
    langfuse = reader.day_counts(first=retained, last=today)
    held = partition_row_counts(bq, table_id, min(_catch_up_from(latest), retained))
    days = export_days(latest_loaded=latest, today=today, langfuse=langfuse, held=held)
    loaded = []
    short = []
    read_any = False
    for day in days:
        try:
            rows = [observation_row(observation) for observation in reader.observations(day)]
            replaced = bool(rows) and len(rows) >= held.get(day, 0)
            if replaced:
                replace_day(bq, table_id, day, rows)
        except Exception:
            print(f"{day}: failed; this day and later days were not replaced", flush=True)
            raise
        read_any = read_any or bool(rows)
        if day >= retained:
            counts = (
                f"read {len(rows)} rows, Langfuse counts {langfuse.get(day, 0)}, "
                f"the partition holds {held.get(day, 0)}"
            )
            if len(rows) < max(held.get(day, 0), langfuse.get(day, 0)):
                short.append(f"{day} {counts}")
        else:
            counts = (
                f"read {len(rows)} rows, the partition holds {held.get(day, 0)}, "
                "past Langfuse's retention edge"
            )
        if replaced:
            print(f"{day}: replaced; {counts}", flush=True)
            loaded.append((day, len(rows)))
        else:
            print(f"{day}: partition kept; {counts}", flush=True)
    if not read_any:
        raise EmptyWindowError(
            f"Langfuse returned no rows for any day from {days[0]} to {days[-1]}; check "
            "LANGFUSE_EXPORT_ENVIRONMENT and that the Langfuse keys belong to the project "
            "holding it"
        )
    if short:
        raise IncompleteDaysError(f"days read short of their counts: {', '.join(short)}")
    return loaded


def check_day(*, reader: LangfuseReader, day: date) -> tuple[bool, str]:
    """Compare the rows the mapping produces for `day` with Langfuse's metrics.
    Returns whether they match and a line stating both sides."""
    rows = [observation_row(observation) for observation in reader.observations(day)]
    cost = math.fsum(row["total_cost_usd"] or 0.0 for row in rows)
    metric_count, metric_cost = reader.day_totals(day)
    matched = len(rows) == metric_count and math.isclose(cost, metric_cost, abs_tol=1e-9)
    line = (
        f"{day}: rows {len(rows)} cost {cost:.9f}; "
        f"metrics count {metric_count} totalCost {metric_cost:.9f}; "
        f"{'match' if matched else 'MISMATCH'}"
    )
    return matched, line


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.langfuse_export",
        description="Replace recent days of Langfuse observations in BigQuery.",
    )
    parser.add_argument(
        "--check-day",
        type=date.fromisoformat,
        metavar="DAY",
        help="compare one day's mapped rows with the metrics API and write nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the load, or the check with `--check-day`. Returns the exit status."""
    arguments = _arguments(argv)
    settings = ExportSettings()
    if missing := settings.missing(load=arguments.check_day is None):
        print(f"langfuse export: set {', '.join(missing)}", file=sys.stderr)
        return 2
    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    assert settings.langfuse_base_url is not None
    assert settings.langfuse_export_environment is not None
    with httpx2.Client(
        base_url=settings.langfuse_base_url,
        auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
        timeout=60.0,
    ) as client:
        reader = LangfuseReader(client=client, environment=settings.langfuse_export_environment)
        if arguments.check_day is not None:
            matched, line = check_day(reader=reader, day=arguments.check_day)
            print(line)
            return 0 if matched else 1
        run_export(
            reader=reader,
            bq=bigquery.Client(project=settings.export_bq_project),
            table_id=_table_id(settings),
            today=datetime.now(UTC).date(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
