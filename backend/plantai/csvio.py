"""Ingesting detection tables produced elsewhere.

Field CSVs arrive in whatever shape the tool that wrote them used, so the
importer accepts the three conventions seen in practice and normalises them
onto one schema.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .detect import DETECTION_COLUMNS
from .geometry import to_wgs84
from .model import classify_label

DEFAULT_SOURCE_CRS = "EPSG:32618"

_LAT_ALIASES = ("latitude", "lat", "y_wgs84")
_LON_ALIASES = ("longitude", "lon", "lng", "long", "x_wgs84")
_X_ALIASES = ("global_x", "center_x", "utm_x", "x", "easting")
_Y_ALIASES = ("global_y", "center_y", "utm_y", "y", "northing")
_CLASS_ALIASES = ("class", "class_name", "label", "category", "species")
_CONF_ALIASES = ("confidence", "conf", "score", "prob")


class CsvIngestError(ValueError):
    """Raised when a CSV cannot be placed on the map."""


def _pick(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_detections(
    data: bytes | str,
    source_crs: str = DEFAULT_SOURCE_CRS,
) -> tuple[pd.DataFrame, dict]:
    """Normalise an uploaded CSV into the internal detection schema.

    Returns the frame plus a short report of what was interpreted, so the UI
    can tell the user which columns it used rather than silently guessing.
    """
    buffer = io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data)
    try:
        df = pd.read_csv(buffer)
    except Exception as exc:  # noqa: BLE001 - message goes straight to the user
        raise CsvIngestError(f"Could not read the CSV: {exc}") from exc

    if df.empty:
        raise CsvIngestError("The CSV has no rows.")

    lat_col = _pick(df, _LAT_ALIASES)
    lon_col = _pick(df, _LON_ALIASES)
    x_col = _pick(df, _X_ALIASES)
    y_col = _pick(df, _Y_ALIASES)

    report: dict = {"rows_in": int(len(df)), "source_crs": None}

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype="float64")
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype="float64")
        gx = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype="float64") if x_col else lon
        gy = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype="float64") if y_col else lat
        report["coordinates"] = f"{lat_col}/{lon_col} (already WGS84)"
        report["source_crs"] = "EPSG:4326"
    elif x_col and y_col:
        gx = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype="float64")
        gy = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype="float64")
        crs = str(df["crs"].dropna().iloc[0]) if "crs" in df.columns and df["crs"].notna().any() else source_crs
        lon, lat = to_wgs84(gx, gy, crs)
        report["coordinates"] = f"{x_col}/{y_col} projected from {crs}"
        report["source_crs"] = crs
    else:
        raise CsvIngestError(
            "No usable coordinates. Provide latitude/longitude, "
            "center_x/center_y, or global_x/global_y columns."
        )

    class_col = _pick(df, _CLASS_ALIASES)
    if class_col:
        labels = df[class_col].astype(str).str.strip().replace("", "plant")
    else:
        labels = pd.Series(["plant"] * len(df), index=df.index)
        report["class_note"] = "No class column found — every row was treated as a plant."

    conf_col = _pick(df, _CONF_ALIASES)
    if conf_col:
        confidence = pd.to_numeric(df[conf_col], errors="coerce").fillna(1.0)
        # Tables that store confidence as a percentage would otherwise fail the
        # 0-1 filter the whole UI is built around.
        if float(confidence.max()) > 1.5:
            confidence = confidence / 100.0
            report["confidence_note"] = f"{conf_col} looked like percentages and was rescaled to 0-1."
    else:
        confidence = pd.Series(np.ones(len(df)), index=df.index)
        report["confidence_note"] = "No confidence column found — every row was scored 1.00."

    out = pd.DataFrame(
        {
            "class": [classify_label(v) for v in labels],
            "label": labels.to_numpy(),
            "confidence": confidence.to_numpy(dtype="float64").clip(0.0, 1.0),
            "global_x": gx,
            "global_y": gy,
            "longitude": lon,
            "latitude": lat,
            "crs": report["source_crs"],
        }
    )

    valid = np.isfinite(out["longitude"]) & np.isfinite(out["latitude"])
    dropped = int((~valid).sum())
    out = out[valid].reset_index(drop=True)
    if out.empty:
        raise CsvIngestError("Every row had missing or invalid coordinates.")

    report["rows_out"] = int(len(out))
    if dropped:
        report["dropped"] = dropped
    report["classes"] = (
        out["label"].astype(str).value_counts().head(8).to_dict()
    )
    return out[DETECTION_COLUMNS], report
