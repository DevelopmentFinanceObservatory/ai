"""Write replies for tests, as `harness.mcp_client.McpExecutor` assembles the
component of a `write_component` outcome: the resolved revision as `payload`
and the wire fact summary mapped from the platform's.

The platform's summary is replayed from the payload under the apis service's rules
(`assembler.reply.build_fact_summary`): at most `FACT_SUMMARY_SERIES_CAP`
series, each with its first and last value at three significant figures and
the coordinates the platform stored for it, and `series_cut` counting the
rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from harness.mcp_client import _wire_fact_summary

from .resolved_revision import resolved_revision

# The apis service's `FACT_SUMMARY_SERIES_CAP` (dfo-data-tools' `assembler/limits.py`).
FACT_SUMMARY_SERIES_CAP = 12

STAT_VAR = "dcid:LifeExpectancy_Person"


def series_id(code: str) -> str:
    return f"s_{code.rpartition('/')[2].lower()}"


def coordinates(code: str) -> dict[str, str]:
    return {"observationAbout": code, "stat_var": STAT_VAR, "unit": "Year"}


def stored(codes: Iterable[str]) -> dict[str, dict[str, str]]:
    """The coordinates the platform stored for the series of `codes`, by series id."""
    return {series_id(code): coordinates(code) for code in codes}


def chart(member: str, series: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    """The `member` fixture (`line_chart` or `bar_chart`) drawing `series`,
    each place code's values by period, a None value a point marked missing.
    A series is labelled with its code's last segment."""
    envelope = resolved_revision(member)
    template = envelope["component"]["series"][0]
    envelope["component"]["series"] = [
        {
            **template,
            "id": series_id(code),
            "label": code.rpartition("/")[2],
            "points": [
                {"period": period, "status": "missing"}
                if value is None
                else {"period": period, "value": value}
                for period, value in points.items()
            ],
        }
        for code, points in series.items()
    ]
    return envelope


def table(series: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """The `table` fixture with one row for each place code and period of `series`."""
    envelope = resolved_revision("table")
    template = envelope["component"]["rows"][0]
    envelope["component"]["rows"] = [
        {
            **template,
            "series_id": series_id(code),
            "dimensions": {"recipient": code.rpartition("/")[2]},
            "period": period,
            "value": value,
        }
        for code, points in series.items()
        for period, value in points.items()
    ]
    return envelope


def _platform_series(
    component: Mapping[str, Any], by_id: Mapping[str, Mapping[str, str]]
) -> list[dict[str, Any]]:
    """The series the platform's summary reads from `component`, uncapped. A
    stat card's payload names no series, so its id is the one `by_id` holds."""
    match component["type"]:
        case "stat_card":
            drawn = [
                (
                    next(iter(by_id), None),
                    component["label"],
                    [{"period": component["period"], "value": component["value"]}],
                )
            ]
        case "table":
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in component["rows"]:
                grouped.setdefault(row["series_id"], []).append(row)
            drawn = [
                (key, " / ".join(rows[0]["dimensions"].values()), rows)
                for key, rows in grouped.items()
            ]
        case _:
            drawn = [(one["id"], one["label"], one["points"]) for one in component["series"]]
    listed = []
    for key, label, points in drawn:
        observed = [point for point in points if point.get("value") is not None]
        listed.append(
            {
                "id": key,
                "label": label,
                "coordinates": by_id.get(key, {}),
                "point_count": len(observed),
                "first_value": f"{observed[0]['value']:.3g}" if observed else None,
                "last_value": f"{observed[-1]['value']:.3g}" if observed else None,
            }
        )
    return listed


def write_reply(
    envelope: dict[str, Any], coordinates: Mapping[str, Mapping[str, str]] | None = None
) -> dict[str, Any]:
    """The component of a `write_component` outcome for `envelope`.
    `coordinates` are those the platform stored for its series, by series id
    (`stored`); a series it holds none for is summarised with none."""
    component = envelope["component"]
    listed = _platform_series(component, coordinates or {})
    platform_facts = {
        "series": listed[:FACT_SUMMARY_SERIES_CAP],
        "series_cut": max(0, len(listed) - FACT_SUMMARY_SERIES_CAP),
        "window": (component.get("x_axis") or {}).get("window"),
        "unit": "Year",
    }
    return {
        "component_id": envelope["component_id"],
        "revision": envelope["revision"],
        "type": component["type"],
        "payload": envelope,
        "fact_summary": _wire_fact_summary(
            platform_facts,
            component=component,
            component_id=envelope["component_id"],
            revision=envelope["revision"],
            manifest=None,
            indicator_count=1,
        ),
    }
