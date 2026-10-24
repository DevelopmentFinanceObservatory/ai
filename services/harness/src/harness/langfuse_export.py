"""Loads Langfuse observation rows into BigQuery. Run as `python -m harness.langfuse_export`.

Each run reads `GET /api/public/v2/observations` for one Langfuse environment, one
UTC day at a time, and replaces that day's partition of the BigQuery table
`observations`, which is day-partitioned on `start_time`. A load adds any column
of `SCHEMA` the table lacks, which rows loaded earlier hold as NULL until their
day is replaced. The days are the latest loaded day minus one through today, plus
every day from `BACKFILL_FLOOR` on whose observation count, from
`GET /api/public/v2/metrics` queries grouped by day, each covering at most
`DAY_COUNT_CHUNK_DAYS` days, differs from its partition's row count.
`BACKFILL_FLOOR` is the first day the prod project holds data, and an empty table
is backfilled from it. `--from DAY` adds every day from DAY (or
`BACKFILL_FLOOR`, when DAY is earlier) through today to the window, under the
same rules, so a column added to `SCHEMA` reaches history. A run that fails
part-way leaves every day it reached replaced whole and every other day as it
was. For the scheduled window, the next run's count comparison picks up the gap.
A `--from` run that stops part-way is repaired by rerunning it from the first day
it did not log, since the scheduled runs skip days whose counts already match and
a rerun from the original DAY replaces the same leading days again.

`LangfuseReader.observations` returns each observation id once, so an observation
two pages both hold loads once and `--check-day` counts it once. Of two copies of
one id it keeps the one with an `endTime` over one without, and otherwise the
copy seen last, since an observation Langfuse was still updating when an early
page was cut has no `endTime` yet. A replace never shrinks a partition: a day that
reads fewer rows than its partition holds, none included, keeps the partition and
prints both counts. A request error ends the run as `LangfuseError`, and
`run_export` returns each day's `DayOutcome`.
A day that reads fewer rows than its partition holds or than Langfuse counts is a
fault. The run processes the other days and then raises `IncompleteDaysError`
naming every such day with its counts, and the day stays in later runs until the
counts match. A partition holding rows Langfuse no
longer counts, such as an observation deleted in Langfuse, fails every run until
that day's rows are deleted from the table, after which the next run reloads the
day. When every day of the run reads no rows, the run raises `EmptyWindowError`
naming the days and `LANGFUSE_EXPORT_ENVIRONMENT`, since a wrong environment or
keys for another project read as empty and prod has had turns every day since
`BACKFILL_FLOOR`. Each day prints a line as it is replaced or kept, and a failure
prints the day it stopped on before the error.

When the day-count query fails, the run still reads the window from the latest
loaded day minus one (or `--from` when earlier) through today, which needs no
count, under the never-shrink rule. It then raises
`EmptyWindowError` chained from the query's `LangfuseError` when every day read no
rows, so a wrong environment is always named, and otherwise the query's
`LangfuseError`, or an `IncompleteDaysError` chained from it and naming it when a
day read fewer rows than its partition holds.

The rows carry ids, names, the model, times, tokens and cost. `input_tokens` is
Langfuse's `inputUsage`, which includes the cached input tokens, and
`cached_input_tokens` is `usageDetails.input_cached_tokens`, NULL when the
observation has no such key. Requests omit the
`io` and `metadata` field groups, so prompts and answers stay in
Langfuse, and `observation_row` copies named fields only, which keeps
`statusMessage` (it can echo user input) and `userId` out of BigQuery.

Langfuse Cloud's limits are stated beside `REQUEST_INTERVAL_S`. Every
observations page counts against the organisation's General API bucket, so
requests go out at most one per `REQUEST_INTERVAL_S`. Metrics queries draw from
the separate Metrics API bucket and make at most `METRICS_ATTEMPTS` attempts. A
run makes one day-count query for each `DAY_COUNT_CHUNK_DAYS` days since
`BACKFILL_FLOOR`, and each `--check-day` makes two queries, spending two to four.
A 429 waits when its `Retry-After`, in seconds or as an HTTP date, is at most
`MAX_RETRY_AFTER_S`, and backs off with a doubling wait from `RATE_LIMIT_BACKOFF_S`
capped at `MAX_RETRY_AFTER_S` when the header is missing or unusable. It ends the
run with `LangfuseError` when it asks for longer, as an exhausted Metrics bucket
can, or when the attempts run out. Any other 4xx ends the run at once. A 5xx or a
transport error is retried with backoff.

`--check-day DAY` reads one day and compares the row count, summed cost and summed
cached input tokens the mapping produces with `GET /api/public/v2/metrics` for the
same day and environment. It writes nothing.

Settings come from the environment or `.env`: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_EXPORT_ENVIRONMENT`, and for
a load also `EXPORT_BQ_PROJECT` and `EXPORT_BQ_DATASET`.

Exit status: 0 when every day loaded or kept or the check matched, 1 when the
check found a difference, a request failed, a day read short or the
whole run read no rows, and 2 when a setting is missing.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

import httpx2
from google.cloud import bigquery
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKFILL_FLOOR = date(2026, 10, 5)
TABLE = "observations"

# Every group except `io` and `metadata` (free text) and `time` and `prompt`,
# which no column reads.
FIELD_GROUPS = "core,basic,model,usage,metrics,trace_context"
PAGE_LIMIT = 1000
# The limits of Langfuse Cloud's Pro plan, which the dfo organisation holds
# (https://langfuse.com/faq/all/api-limits): data access for 3 years, which
# reaches back past `BACKFILL_FLOOR`; a General API bucket of 1,000 requests a
# minute; and a Metrics API bucket of 500 queries an hour.
# One page every 0.1 seconds is 600 requests a minute, under the General API bucket.
REQUEST_INTERVAL_S = 0.1
ATTEMPTS = 5
BACKOFF_S = 2.0
# A metrics query's attempts.
METRICS_ATTEMPTS = 2
# The most rows a metrics query asks for, Langfuse's maximum `config.row_limit`
# (https://langfuse.com/docs/metrics/features/metrics-api); the default is 100.
METRICS_ROW_LIMIT = 1000
# The days one day-count query covers, one row each, below `METRICS_ROW_LIMIT` so
# a whole answer never fills it. A run makes one query through 2029-03-22 and two
# through 2031-09-08, so with `METRICS_ATTEMPTS` an hourly run spends at most 4 of
# the Metrics bucket's 500 queries an hour until then.
DAY_COUNT_CHUNK_DAYS = 900
# A 429 without a usable `Retry-After` waits this long, doubling with each
# consecutive 429.
RATE_LIMIT_BACKOFF_S = 15.0
# The longest a 429 waits. The General API bucket refills within a minute, and a
# longer `Retry-After`, such as the minutes an exhausted hourly Metrics bucket can
# name, fails the request at once.
MAX_RETRY_AFTER_S = 60.0
# How much of a refusal's body a `LangfuseError` message carries.
ERROR_BODY_CHARS = 500

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
    bigquery.SchemaField("cached_input_tokens", "INT64"),
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
    """Langfuse kept failing a request after every retry, refused it, or answered
    what the job cannot read. `status` is the last answer's HTTP status, None after
    a transport error, and `body` its first `ERROR_BODY_CHARS` characters."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class DayOutcome(StrEnum):
    """What a run did with one day's partition."""

    REPLACED = "replaced"
    KEPT = "partition kept"


class EmptyWindowError(RuntimeError):
    """Every day of the run's window read no rows, which a misconfiguration causes."""


class IncompleteDaysError(RuntimeError):
    """A day read fewer rows than its partition or Langfuse's own count; `days`
    lists them. A partition holding rows Langfuse no longer counts, such as an
    observation deleted in Langfuse, fails every run until that day's rows are
    deleted from the table, after which the next run reloads the day."""

    def __init__(self, message: str, *, days: list[date]) -> None:
        super().__init__(message)
        self.days = days


@contextmanager
def _readable(shown: object) -> Iterator[None]:
    """Turns a KeyError, TypeError or ValueError from reading a metrics answer into a
    `LangfuseError` carrying the first `ERROR_BODY_CHARS` characters of `shown`."""
    try:
        yield
    except (KeyError, TypeError, ValueError) as exc:
        raise LangfuseError(
            f"the metrics query answered rows it cannot read: {exc!r}",
            status=200,
            body=json.dumps(shown)[:ERROR_BODY_CHARS],
        ) from exc


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
        "cached_input_tokens": (observation.get("usageDetails") or {}).get("input_cached_tokens"),
        "output_tokens": observation.get("outputUsage"),
        "total_tokens": observation.get("totalUsage"),
        "input_cost_usd": observation.get("inputCost"),
        "output_cost_usd": observation.get("outputCost"),
        "total_cost_usd": observation.get("totalCost"),
    }


def window_start(*, latest_loaded: date | None, since: date | None) -> date:
    """The first day of the run's window, which is read whatever the counts say: the
    latest loaded day minus one (`BACKFILL_FLOOR` when nothing is loaded), or `since`
    when earlier, and never before `BACKFILL_FLOOR`. The count comparison sees only
    observations added or removed, and Langfuse updates an existing observation in
    place as later events for it arrive (its end time, usage and cost), so the day
    before the latest loaded day is re-read for the updates a count misses."""
    first = BACKFILL_FLOOR if latest_loaded is None else latest_loaded - timedelta(days=1)
    if since is not None:
        first = min(first, since)
    return max(first, BACKFILL_FLOOR)


def export_days(
    *,
    first: date,
    today: date,
    langfuse: Mapping[date, int] | None,
    held: Mapping[date, int],
) -> list[date]:
    """The days a run replaces: `first` (the run's `window_start`) through today, and
    every day from `BACKFILL_FLOOR` on whose Langfuse count differs from its
    partition's row count. `langfuse` is None when the counts could not be read,
    which leaves the window."""
    days = {first + timedelta(days=offset) for offset in range((today - first).days + 1)}
    if langfuse is not None:
        days.update(
            day
            for day in langfuse.keys() | held.keys()
            if BACKFILL_FLOOR <= day <= today and langfuse.get(day, 0) != held.get(day, 0)
        )
    return sorted(days)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rate_limit_wait(retry_after: str | None, consecutive: int, now: datetime) -> float:
    """How long the `consecutive`th 429 in a row asks to wait: its `Retry-After`, in
    seconds or as an HTTP date measured from `now`, or a wait doubling from
    `RATE_LIMIT_BACKOFF_S`, at most `MAX_RETRY_AFTER_S`, when the header is missing or
    unusable."""
    seconds = _retry_after_seconds(retry_after, now)
    if seconds is None:
        return min(RATE_LIMIT_BACKOFF_S * 2 ** (consecutive - 1), MAX_RETRY_AFTER_S)
    return seconds


def _retry_after_seconds(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        # A date already past asks for no wait beyond the pacing interval.
        return max((at.replace(tzinfo=at.tzinfo or UTC) - now).total_seconds(), 0.0)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


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
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._client = client
        self._environment = environment
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self._last_request: float | None = None

    def _get(
        self, path: str, params: Mapping[str, str | int], *, attempts: int = ATTEMPTS
    ) -> dict[str, Any]:
        rate_limited = 0
        for attempt in range(1, attempts + 1):
            if self._last_request is not None:
                wait = self._last_request + REQUEST_INTERVAL_S - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            try:
                response = self._client.get(path, params=params)
            except httpx2.TransportError as exc:
                rate_limited = 0
                if attempt == attempts:
                    raise LangfuseError(f"GET {path} failed: {exc!r}") from exc
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            rate_limited = rate_limited + 1 if response.status_code == 429 else 0
            if response.status_code == 429 and attempt < attempts:
                wait = _rate_limit_wait(
                    response.headers.get("retry-after"), rate_limited, self._wall_clock()
                )
                if wait <= MAX_RETRY_AFTER_S:
                    self._sleep(wait)
                    continue
            if response.status_code >= 500 and attempt < attempts:
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            if response.status_code != 200:
                body = response.text[:ERROR_BODY_CHARS]
                raise LangfuseError(
                    f"GET {path} answered {response.status_code}: {body}",
                    status=response.status_code,
                    body=body,
                )
            try:
                return response.json()
            except ValueError as exc:
                raise LangfuseError(
                    f"GET {path} answered 200 with a body that is not JSON: {exc!r}",
                    status=200,
                    body=response.text[:ERROR_BODY_CHARS],
                ) from exc
        raise AssertionError("unreachable: the last attempt returns or raises")

    def observations(self, day: date) -> list[dict[str, Any]]:
        """Every observation of the environment whose start time falls on `day` (UTC),
        each id once. Of two copies of an id, which overlapping pages can hold, the one
        with an `endTime` wins over one without, and otherwise the later-seen copy."""
        start, end = _day_bounds(day)
        found: dict[str, dict[str, Any]] = {}
        params: dict[str, str | int] = {
            "fields": FIELD_GROUPS,
            "environment": self._environment,
            "fromStartTime": start,
            "toStartTime": end,
            "limit": PAGE_LIMIT,
        }
        while True:
            page = self._get("/api/public/v2/observations", params)
            for observation in page["data"]:
                kept = found.get(observation["id"])
                if kept is None or observation.get("endTime") or not kept.get("endTime"):
                    found[observation["id"]] = observation
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or not page["data"]:
                return list(found.values())
            params["cursor"] = cursor

    def _metrics(
        self,
        *,
        first: date,
        last: date,
        metrics: list[dict[str, str]],
        dimensions: list[dict[str, str]] | None = None,
        by_day: bool,
    ) -> list[dict[str, Any]]:
        """The metrics rows for the environment's observations from `first` through
        `last` (UTC), one row a day when `by_day` and one row otherwise, for each value
        of `dimensions`, in at most `METRICS_ATTEMPTS` attempts. Raises `LangfuseError`
        carrying the answer's first `ERROR_BODY_CHARS` characters when it holds no
        `data`."""
        query: dict[str, Any] = {
            "view": "observations",
            "metrics": metrics,
            "dimensions": dimensions or [],
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
            "config": {"row_limit": METRICS_ROW_LIMIT},
        }
        if by_day:
            query["timeDimension"] = {"granularity": "day"}
        answer = self._get(
            "/api/public/v2/metrics", {"query": json.dumps(query)}, attempts=METRICS_ATTEMPTS
        )
        with _readable(answer):
            return answer["data"]

    def day_counts(self, *, first: date, last: date) -> dict[date, int]:
        """The observation count Langfuse's metrics report for each day from `first`
        through `last` that holds observations, read in consecutive queries of at most
        `DAY_COUNT_CHUNK_DAYS` days. The metrics bucket a day by UTC start time, the
        bounds `observations` reads. Raises `LangfuseError` when a query fails,
        answers rows without a readable day and count, or answers `METRICS_ROW_LIMIT`
        rows, which Langfuse may have cut."""
        counts: dict[date, int] = {}
        while first <= last:
            chunk_last = min(first + timedelta(days=DAY_COUNT_CHUNK_DAYS - 1), last)
            rows = self._metrics(
                first=first,
                last=chunk_last,
                metrics=[{"measure": "count", "aggregation": "count"}],
                by_day=True,
            )
            with _readable({"data": rows}):
                if len(rows) >= METRICS_ROW_LIMIT:
                    raise LangfuseError(
                        f"the day-count query for {first} through {chunk_last} answered "
                        f"{len(rows)} rows, the row limit, so it may be cut",
                        status=200,
                    )
                counts.update(
                    (date.fromisoformat(row["time_dimension"]), int(row["count_count"]))
                    for row in rows
                )
            first = chunk_last + timedelta(days=1)
        return counts

    def day_cached_input_tokens(self, day: date) -> int:
        """The summed `input_cached_tokens` usage Langfuse's metrics report for `day`.
        The `usageByType` measure sums each usage key, grouped by `usageType`. Raises
        `LangfuseError` naming the measure when Langfuse refuses it with a 400."""
        try:
            rows = self._metrics(
                first=day,
                last=day,
                metrics=[{"measure": "usageByType", "aggregation": "sum"}],
                dimensions=[{"field": "usageType"}],
                by_day=False,
            )
        except LangfuseError as exc:
            if exc.status != 400:
                raise
            raise LangfuseError(
                "the metrics API does not accept the `usageByType` measure grouped by "
                f"`usageType`, which sums cached input tokens: {exc}",
                status=exc.status,
                body=exc.body,
            ) from exc
        with _readable({"data": rows}):
            return sum(
                int(row["sum_usageByType"] or 0)
                for row in rows
                if row["usageType"] == "input_cached_tokens"
            )

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
        with _readable({"data": rows}):
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
    """Replace `day`'s partition with `rows` in one load job, adding any `SCHEMA`
    column the table lacks."""
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
    *,
    reader: LangfuseReader,
    bq: bigquery.Client,
    table_id: str,
    today: date,
    since: date | None = None,
) -> dict[date, DayOutcome]:
    """Create the table if absent and replace every day `export_days` names, with
    `since` widening the window, whose read holds rows and at least as many as its
    partition. Returns each day's `DayOutcome`. Raises a read's `LangfuseError` at
    once, then `EmptyWindowError` when every day reads no rows, chained from the
    day-count query's `LangfuseError` when that failed, then that `LangfuseError` or
    an `IncompleteDaysError` chained from it when a day read fewer rows than its
    partition holds, and then `IncompleteDaysError` when a day read fewer rows than
    its partition or Langfuse's count."""
    table = bigquery.Table(table_id, schema=SCHEMA)
    table.time_partitioning = PARTITIONING
    bq.create_table(table, exists_ok=True)
    first = window_start(latest_loaded=latest_loaded_day(bq, table_id), since=since)
    outcomes: dict[date, DayOutcome] = {}
    counts_error = None
    langfuse: dict[date, int] | None
    try:
        langfuse = reader.day_counts(first=BACKFILL_FLOOR, last=today)
    except LangfuseError as exc:
        print(f"day counts: failed, so only {first} through {today} is read; {exc}", flush=True)
        counts_error = exc
        langfuse = None
    held = partition_row_counts(bq, table_id, BACKFILL_FLOOR)
    days = export_days(first=first, today=today, langfuse=langfuse, held=held)
    short = []
    read_any = False
    for day in days:
        count = None if langfuse is None else langfuse.get(day, 0)
        try:
            rows = [observation_row(observation) for observation in reader.observations(day)]
            replaced = bool(rows) and len(rows) >= held.get(day, 0)
            if replaced:
                replace_day(bq, table_id, day, rows)
        except Exception:
            print(f"{day}: failed; this day and later days were not replaced", flush=True)
            raise
        read_any = read_any or bool(rows)
        counts = f"read {len(rows)} rows, the partition holds {held.get(day, 0)}"
        if count is not None:
            counts = (
                f"read {len(rows)} rows, Langfuse counts {count}, "
                f"the partition holds {held.get(day, 0)}"
            )
        if len(rows) < max(held.get(day, 0), count or 0):
            short.append((day, counts))
        outcomes[day] = DayOutcome.REPLACED if replaced else DayOutcome.KEPT
        print(f"{day}: {outcomes[day]}; {counts}", flush=True)
    if not read_any:
        raise EmptyWindowError(
            f"Langfuse returned no rows for any day from {days[0]} to {days[-1]}; check "
            "LANGFUSE_EXPORT_ENVIRONMENT and that the Langfuse keys belong to the project "
            "holding it"
        ) from counts_error
    incomplete = "days read short of their counts: " + ", ".join(
        f"{day} {counts}" for day, counts in short
    )
    if counts_error is not None:
        if not short:
            raise counts_error
        raise IncompleteDaysError(
            f"{incomplete}; the day-count query failed: {counts_error}",
            days=[day for day, _ in short],
        ) from counts_error
    if short:
        raise IncompleteDaysError(incomplete, days=[day for day, _ in short])
    return outcomes


def check_day(*, reader: LangfuseReader, day: date) -> tuple[bool, str]:
    """Compare the rows the mapping produces for `day` with Langfuse's metrics.
    Returns whether they match and a line stating both sides."""
    rows = [observation_row(observation) for observation in reader.observations(day)]
    cost = math.fsum(row["total_cost_usd"] or 0.0 for row in rows)
    cached = sum(row["cached_input_tokens"] or 0 for row in rows)
    metric_count, metric_cost = reader.day_totals(day)
    metric_cached = reader.day_cached_input_tokens(day)
    matched = (
        len(rows) == metric_count
        and math.isclose(cost, metric_cost, abs_tol=1e-9)
        and cached == metric_cached
    )
    line = (
        f"{day}: rows {len(rows)} cost {cost:.9f} cached input tokens {cached}; "
        f"metrics count {metric_count} totalCost {metric_cost:.9f} "
        f"input_cached_tokens {metric_cached}; "
        f"{'match' if matched else 'MISMATCH'}"
    )
    return matched, line


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.langfuse_export",
        description="Replace recent days of Langfuse observations in BigQuery.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-day",
        type=date.fromisoformat,
        metavar="DAY",
        help="compare one day's mapped rows with the metrics API and write nothing",
    )
    mode.add_argument(
        "--from",
        dest="since",
        type=date.fromisoformat,
        metavar="DAY",
        help="also replace every day from DAY through today",
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
            since=arguments.since,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
