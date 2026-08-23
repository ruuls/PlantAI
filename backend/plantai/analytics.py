"""Stand-count agronomy on top of the detections.

The detector answers "where are the plants"; this module answers the questions
a breeder or agronomist actually asks — plants per acre, plants per foot of
row, weed pressure, and whether the stand cleared its target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geometry import ACRE_FT2, M2_PER_ACRE, bbox_area_acres

CONFIDENCE_BANDS: list[tuple[float, float, str]] = [
    (0.0, 0.5, "<50"),
    (0.5, 0.6, "50-60"),
    (0.6, 0.7, "60-70"),
    (0.7, 0.8, "70-80"),
    (0.8, 0.9, "80-90"),
    (0.9, 1.01, "90+"),
]


@dataclass
class RowGeometry:
    """A research plot described the way it is measured in the field."""

    rows: int = 0
    row_length_ft: float = 0.0
    row_spacing_in: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.rows > 0 and self.row_length_ft > 0 and self.row_spacing_in > 0

    @property
    def total_row_length_ft(self) -> float:
        return float(self.rows) * float(self.row_length_ft) if self.is_complete else 0.0

    @property
    def area_ft2(self) -> float:
        if not self.is_complete:
            return 0.0
        return self.total_row_length_ft * (self.row_spacing_in / 12.0)

    @property
    def area_acres(self) -> float:
        return self.area_ft2 / ACRE_FT2 if self.is_complete else 0.0


@dataclass
class StandMetrics:
    plants: int = 0
    weeds: int = 0
    total: int = 0
    weed_pressure_pct: float = 0.0
    area_acres: float = 0.0
    area_ha: float = 0.0
    plants_per_acre: float = 0.0
    weeds_per_acre: float = 0.0
    plants_per_ft: float = 0.0
    weeds_per_ft: float = 0.0
    total_row_length_ft: float = 0.0
    target_plants_acre: float = 0.0
    threshold_plants_acre: float = 0.0
    pct_of_target: float = 0.0
    status: str = "No stand estimate"
    tone: str = "amber"
    basis: str = "area"
    geometry: RowGeometry = field(default_factory=RowGeometry)

    def as_dict(self) -> dict:
        return {
            "plants": self.plants,
            "weeds": self.weeds,
            "total": self.total,
            "weed_pressure_pct": round(self.weed_pressure_pct, 1),
            "area_acres": round(self.area_acres, 6),
            "area_ha": round(self.area_ha, 6),
            "plants_per_acre": round_density(self.plants_per_acre),
            "weeds_per_acre": round_density(self.weeds_per_acre),
            "plants_per_ft": round(self.plants_per_ft, 3),
            "weeds_per_ft": round(self.weeds_per_ft, 3),
            "total_row_length_ft": round(self.total_row_length_ft, 2),
            "target_plants_acre": round(self.target_plants_acre, 1),
            "threshold_plants_acre": round_density(self.threshold_plants_acre),
            "pct_of_target": round(self.pct_of_target, 1),
            "status": self.status,
            "tone": self.tone,
            "basis": self.basis,
            "rows": self.geometry.rows,
            "row_length_ft": self.geometry.row_length_ft,
            "row_spacing_in": self.geometry.row_spacing_in,
        }


def round_density(value: float) -> float:
    """Round a per-area figure without flattening a sparse field to zero.

    A dense stand is reported to one decimal (55,176.0/acre), but a scouting
    pass that finds 0.73 weeds per acre must not round to 0.7 and lose a
    quarter of its magnitude, so small values keep four significant figures.
    """
    if not math.isfinite(value) or value == 0:
        return 0.0
    if abs(value) >= 100:
        return round(value, 1)
    decimals = 3 - int(math.floor(math.log10(abs(value))))
    return round(value, min(max(decimals, 1), 8))


def _role_mask(df: pd.DataFrame, role: str) -> np.ndarray:
    if df.empty or "class" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["class"].astype(str).str.lower().to_numpy() == role


def stand_metrics(
    df: pd.DataFrame,
    target_plants_acre: float,
    alert_pct: float,
    area_acres: float = 0.0,
    geometry: RowGeometry | None = None,
) -> StandMetrics:
    """Combine counts with an area basis into a stand assessment.

    Two bases are supported. ``area`` divides by the mapped acreage of the
    selection — right for whole-field density. ``rows`` derives the plot area
    from ``rows x row length x row spacing``, which is how research plots are
    scored and what makes plants-per-foot meaningful.
    """
    geometry = geometry or RowGeometry()
    plants = int(_role_mask(df, "plant").sum())
    weeds = int(_role_mask(df, "weed").sum())
    total = int(len(df))

    metrics = StandMetrics(
        plants=plants,
        weeds=weeds,
        total=total,
        weed_pressure_pct=(100.0 * weeds / total) if total else 0.0,
        target_plants_acre=float(target_plants_acre),
        geometry=geometry,
    )

    if geometry.is_complete:
        metrics.basis = "rows"
        metrics.area_acres = geometry.area_acres
        metrics.total_row_length_ft = geometry.total_row_length_ft
        if metrics.total_row_length_ft > 0:
            metrics.plants_per_ft = plants / metrics.total_row_length_ft
            metrics.weeds_per_ft = weeds / metrics.total_row_length_ft
    else:
        metrics.basis = "area"
        metrics.area_acres = float(max(area_acres, 0.0))

    metrics.area_ha = metrics.area_acres * M2_PER_ACRE / 10_000.0
    if metrics.area_acres > 0:
        metrics.plants_per_acre = plants / metrics.area_acres
        metrics.weeds_per_acre = weeds / metrics.area_acres

    if target_plants_acre > 0:
        metrics.pct_of_target = 100.0 * metrics.plants_per_acre / target_plants_acre
        metrics.threshold_plants_acre = target_plants_acre * float(alert_pct) / 100.0

    if metrics.plants_per_acre <= 0:
        metrics.status, metrics.tone = "No stand estimate", "amber"
    elif target_plants_acre <= 0:
        metrics.status, metrics.tone = "No target set", "amber"
    elif metrics.pct_of_target < alert_pct:
        metrics.status, metrics.tone = "Below threshold", "red"
    elif metrics.pct_of_target > 110:
        metrics.status, metrics.tone = "Above target", "amber"
    else:
        metrics.status, metrics.tone = "On target", "green"
    return metrics


def status_detail(metrics: StandMetrics) -> str:
    """The one-line explanation shown under the status banner."""
    return (
        f"{metrics.plants_per_acre:,.0f} plants/acre · "
        f"{metrics.pct_of_target:.1f}% of {metrics.target_plants_acre:,.0f} target · "
        f"{metrics.area_acres:.4f} acres"
    )


def confidence_bands(df: pd.DataFrame) -> list[dict]:
    """Detection counts per confidence band, for the distribution chart."""
    if df is None or df.empty or "confidence" not in df.columns:
        return [{"label": lbl, "count": 0} for _, _, lbl in CONFIDENCE_BANDS]
    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype="float64")
    return [
        {"label": label, "count": int(np.count_nonzero((conf >= lo) & (conf < hi)))}
        for lo, hi, label in CONFIDENCE_BANDS
    ]


def class_confidence(df: pd.DataFrame) -> list[dict]:
    """Per-class count with mean and minimum confidence."""
    out: list[dict] = []
    for role, display in (("plant", "Plant"), ("weed", "Weed")):
        subset = df[_role_mask(df, role)] if not df.empty else df
        conf = (
            pd.to_numeric(subset["confidence"], errors="coerce").dropna()
            if len(subset) and "confidence" in subset.columns
            else pd.Series(dtype="float64")
        )
        out.append(
            {
                "class": display,
                "count": int(len(subset)),
                "avg_conf": round(float(conf.mean()) * 100, 1) if len(conf) else None,
                "min_conf": round(float(conf.min()) * 100, 1) if len(conf) else None,
            }
        )
    return out


def rows_table(metrics: StandMetrics) -> list[dict]:
    """Per-row allocation of the plot's counts.

    Without per-row polygons the split is even, so this reads as a plot average
    broken out by row rather than a measurement of each row. The UI says so.
    """
    n = int(metrics.geometry.rows)
    if n <= 0 or not metrics.geometry.is_complete:
        return []

    plant_base, plant_rem = divmod(metrics.plants, n)
    weed_base, weed_rem = divmod(metrics.weeds, n)
    length = float(metrics.geometry.row_length_ft)

    rows: list[dict] = []
    for i in range(n):
        plants = plant_base + (1 if i < plant_rem else 0)
        weeds = weed_base + (1 if i < weed_rem else 0)
        rows.append(
            {
                "row_index": i + 1,
                "row_length_ft": round(length, 2),
                "row_spacing_in": metrics.geometry.row_spacing_in,
                "plants": plants,
                "weeds": weeds,
                "total": plants + weeds,
                "plants_per_ft": round(plants / length, 3) if length > 0 else 0.0,
                "weeds_per_ft": round(weeds / length, 3) if length > 0 else 0.0,
            }
        )
    return rows


def plot_summary(metrics: StandMetrics, plot_id: str, scope: str) -> dict:
    """One row of ``plots.csv`` — the plot-level export."""
    d = metrics.as_dict()
    return {
        "plot_id": plot_id,
        "scope": scope,
        "basis": metrics.basis,
        "n_rows": metrics.geometry.rows,
        "row_spacing_in": metrics.geometry.row_spacing_in,
        "row_spacing_ft": round(metrics.geometry.row_spacing_in / 12.0, 4),
        "row_length_ft": metrics.geometry.row_length_ft,
        "total_row_length_ft": d["total_row_length_ft"],
        "plot_area_ft2": round(metrics.area_acres * ACRE_FT2, 2),
        "plot_area_acres": d["area_acres"],
        "plot_area_ha": d["area_ha"],
        "plants": d["plants"],
        "weeds": d["weeds"],
        "total_detections": d["total"],
        "plants_per_acre": d["plants_per_acre"],
        "weeds_per_acre": d["weeds_per_acre"],
        "plants_per_ft": d["plants_per_ft"],
        "weeds_per_ft": d["weeds_per_ft"],
        "weed_pressure_pct": d["weed_pressure_pct"],
        "target_plants_acre": d["target_plants_acre"],
        "pct_of_target": d["pct_of_target"],
        "status": d["status"],
    }


def fallback_area_acres(df: pd.DataFrame) -> float:
    """Extent-based acreage, used only when no AOI is drawn."""
    if df is None or df.empty or not {"longitude", "latitude"}.issubset(df.columns):
        return 0.0
    return bbox_area_acres(df["longitude"], df["latitude"])
