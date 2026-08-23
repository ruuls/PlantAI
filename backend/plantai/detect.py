"""Tiled YOLO inference over an orthomosaic, plus overlap deduplication."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import settings
from .geometry import to_wgs84
from .model import Detector, classify_label
from .raster import RasterInfo, TileSpec, percentile_stretch, plan_tiles, read_tiles

ProgressFn = Callable[[str, float, str], None]

DETECTION_COLUMNS = [
    "class",
    "label",
    "confidence",
    "global_x",
    "global_y",
    "longitude",
    "latitude",
    "crs",
]


@dataclass
class DetectionParams:
    """Everything that changes a run's output, in one place."""

    tile_size: int = settings.tile_size
    conf: float = settings.conf
    iou: float = settings.iou
    side_overlap: float = settings.side_overlap
    front_overlap: float = settings.front_overlap
    dedup_radius_m: float = settings.dedup_radius_m
    batch_tiles: int = settings.batch_tiles

    def clamped(self) -> "DetectionParams":
        return DetectionParams(
            tile_size=int(min(max(self.tile_size, 128), 4096)),
            conf=float(min(max(self.conf, 0.0), 1.0)),
            iou=float(min(max(self.iou, 0.0), 1.0)),
            side_overlap=float(min(max(self.side_overlap, 0.0), 0.9)),
            front_overlap=float(min(max(self.front_overlap, 0.0), 0.9)),
            dedup_radius_m=float(min(max(self.dedup_radius_m, 0.0), 5.0)),
            batch_tiles=int(min(max(self.batch_tiles, 1), 32)),
        )


@dataclass
class RunStats:
    """Counters the progress panel reports back to the UI."""

    tiles: int = 0
    raw: int = 0
    kept: int = 0
    plants: int = 0
    weeds: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "tiles": self.tiles,
            "raw": self.raw,
            "kept": self.kept,
            "plants": self.plants,
            "weeds": self.weeds,
            **self.extras,
        }


def dedupe_mask(x: np.ndarray, y: np.ndarray, conf: np.ndarray, radius_m: float) -> np.ndarray:
    """Greedy spatial non-max suppression on a projected plane.

    Detections are visited highest-confidence first and a point is dropped when
    a stronger one already sits within ``radius_m``. A grid of ``radius_m``
    cells bounds each lookup to nine buckets, which keeps this near-linear
    instead of the O(n²) pairwise scan the radius test implies.
    """
    n = int(x.size)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    if radius_m <= 0:
        keep[:] = True
        return keep

    r2 = float(radius_m) ** 2
    xs = np.asarray(x, dtype="float64")
    ys = np.asarray(y, dtype="float64")
    order = np.argsort(-np.asarray(conf, dtype="float64"), kind="stable")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in order:
        xi = xs[idx]
        yi = ys[idx]
        ci = int(np.floor(xi / radius_m))
        cj = int(np.floor(yi / radius_m))

        clash = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for other in buckets.get((ci + di, cj + dj), ()):
                    dx = xs[other] - xi
                    dy = ys[other] - yi
                    if dx * dx + dy * dy <= r2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break

        if not clash:
            keep[idx] = True
            buckets[(ci, cj)].append(int(idx))
    return keep


def deduplicate(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """Drop overlap duplicates, one class at a time.

    Classes are suppressed independently: a weed growing beside a crop plant is
    a real second detection, not a duplicate of it.
    """
    if df.empty or radius_m <= 0:
        return df.reset_index(drop=True)

    keep = np.zeros(len(df), dtype=bool)
    positions = np.arange(len(df))
    for _, group in df.groupby("class", sort=False):
        idx = positions[df["class"].to_numpy() == group["class"].iloc[0]]
        mask = dedupe_mask(
            group["global_x"].to_numpy(dtype="float64"),
            group["global_y"].to_numpy(dtype="float64"),
            group["confidence"].to_numpy(dtype="float64"),
            radius_m,
        )
        keep[idx] = mask
    return df[keep].reset_index(drop=True)


def empty_detections() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c in {"class", "label", "crs"} else "float64") for c in DETECTION_COLUMNS})


def run_detection(
    info: RasterInfo,
    detector: Detector,
    params: DetectionParams,
    region: tuple[int, int, int, int] | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[pd.DataFrame, RunStats]:
    """Tile the raster, score every tile, deduplicate, and georeference.

    ``region`` restricts the work to a pixel window — that is what makes the
    AOI quick-count fast enough to use plot by plot.
    """
    params = params.clamped()
    stats = RunStats()

    def emit(step: str, frac: float, note: str = "") -> None:
        if progress is not None:
            progress(step, max(0.0, min(1.0, frac)), note)

    emit("tile", 0.0, "Planning windows")
    tiles = plan_tiles(
        info,
        tile_size=params.tile_size,
        side_overlap=params.side_overlap,
        front_overlap=params.front_overlap,
        region=region,
    )
    stats.tiles = len(tiles)
    if not tiles:
        emit("tile", 1.0, "Nothing to scan")
        return empty_detections(), stats

    stretch = None if info.dtype == "uint8" else percentile_stretch(info.path)
    emit("tile", 1.0, f"{len(tiles):,} tiles")

    rows: list[tuple[int, float, float, float]] = []
    batch_imgs: list[np.ndarray] = []
    batch_tf: list[object] = []
    done = 0

    def flush() -> None:
        nonlocal batch_imgs, batch_tf
        if not batch_imgs:
            return
        results = detector.predict_batch(
            batch_imgs, conf=params.conf, iou=params.iou, imgsz=params.tile_size
        )
        for dets, affine in zip(results, batch_tf):
            for class_id, confidence, px, py in dets:
                gx, gy = affine * (px, py)
                rows.append((class_id, confidence, gx, gy))
        batch_imgs = []
        batch_tf = []

    emit("infer", 0.0, "Starting model")
    for spec, image, affine in read_tiles(info.path, tiles, stretch):
        if should_stop is not None and should_stop():
            emit("infer", done / len(tiles), "Cancelled")
            return empty_detections(), stats

        batch_imgs.append(image)
        batch_tf.append(affine)
        if len(batch_imgs) >= params.batch_tiles:
            flush()
        done += 1
        if done % max(1, len(tiles) // 100) == 0 or done == len(tiles):
            emit("infer", done / len(tiles), f"{len(rows):,} raw")
    flush()

    stats.raw = len(rows)
    emit("infer", 1.0, f"{stats.raw:,} raw")

    if not rows:
        emit("dedupe", 1.0, "0 kept")
        emit("project", 1.0, "EPSG:4326")
        return empty_detections(), stats

    class_ids = np.fromiter((r[0] for r in rows), dtype="int64", count=len(rows))
    confidence = np.fromiter((r[1] for r in rows), dtype="float64", count=len(rows))
    gx = np.fromiter((r[2] for r in rows), dtype="float64", count=len(rows))
    gy = np.fromiter((r[3] for r in rows), dtype="float64", count=len(rows))

    labels = [detector.label_for(int(cid)) for cid in class_ids]
    roles = [classify_label(lbl) for lbl in labels]

    raw_df = pd.DataFrame(
        {
            "class": roles,
            "label": labels,
            "confidence": confidence,
            "global_x": gx,
            "global_y": gy,
        }
    )

    emit("dedupe", 0.3, "Suppressing overlap")
    kept = deduplicate(raw_df, params.dedup_radius_m)
    stats.kept = len(kept)
    emit("dedupe", 1.0, f"{stats.kept:,} kept")

    emit("project", 0.4, "Reprojecting")
    lon, lat = to_wgs84(kept["global_x"].to_numpy(), kept["global_y"].to_numpy(), info.crs)
    kept = kept.assign(longitude=lon, latitude=lat, crs=info.crs)
    kept = kept[DETECTION_COLUMNS]

    stats.plants = int((kept["class"] == "plant").sum())
    stats.weeds = int((kept["class"] == "weed").sum())
    emit("project", 1.0, "EPSG:4326")
    return kept.reset_index(drop=True), stats


def summarise(df: pd.DataFrame) -> dict:
    """Headline counts for the map's KPI strip."""
    if df is None or df.empty:
        return {"total": 0, "plants": 0, "weeds": 0, "weed_pressure_pct": 0.0, "avg_confidence": 0.0}
    classes = df["class"].astype(str).str.lower()
    plants = int((classes == "plant").sum())
    weeds = int((classes == "weed").sum())
    total = int(len(df))
    return {
        "total": total,
        "plants": plants,
        "weeds": weeds,
        "weed_pressure_pct": round(100.0 * weeds / total, 1) if total else 0.0,
        "avg_confidence": round(float(df["confidence"].mean()), 4),
    }
