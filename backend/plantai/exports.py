"""CSV export helpers shared by the API routes."""

from __future__ import annotations

import io

import pandas as pd

from .analytics import StandMetrics, plot_summary, rows_table


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def detections_csv(frame: pd.DataFrame) -> bytes:
    return to_csv_bytes(frame)


def stand_csv(metrics: StandMetrics, scope: str, plot_id: str) -> bytes:
    record = metrics.as_dict()
    record.update({"scope": scope, "plot_id": plot_id})
    return to_csv_bytes(pd.DataFrame([record]))


def plots_csv(metrics: StandMetrics, plot_id: str, scope: str) -> bytes:
    return to_csv_bytes(pd.DataFrame([plot_summary(metrics, plot_id, scope)]))


def rows_csv(metrics: StandMetrics, plot_id: str) -> bytes:
    rows = rows_table(metrics)
    if not rows:
        return to_csv_bytes(pd.DataFrame(columns=["plot_id", "row_index"]))
    frame = pd.DataFrame(rows)
    frame.insert(0, "plot_id", plot_id)
    return to_csv_bytes(frame)
