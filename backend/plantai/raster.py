"""Reading, describing and windowing georeferenced orthomosaics."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.ops import transform as shapely_transform

from .config import settings
from .geometry import transformer


class RasterError(RuntimeError):
    """Raised for a raster the pipeline cannot work with."""


@dataclass(frozen=True)
class RasterInfo:
    """Everything the UI needs to describe an orthomosaic without opening it."""

    path: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str
    bounds_native: tuple[float, float, float, float]
    bounds_wgs84: tuple[float, float, float, float]
    center: tuple[float, float]
    gsd_m: float
    size_bytes: int
    has_overviews: bool
    overview_levels: list[int]
    nodata: float | None

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        west, south, east, north = self.bounds_wgs84
        return [[south, west], [north, east]]

    @property
    def area_m2(self) -> float:
        return float(self.width) * float(self.height) * (self.gsd_m**2)

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds_wgs84": list(self.bounds_wgs84),
            "leaflet_bounds": self.leaflet_bounds,
            "center": list(self.center),
            "gsd_cm": round(self.gsd_m * 100.0, 3),
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1e9, 3),
            "has_overviews": self.has_overviews,
            "overview_levels": self.overview_levels,
            "area_m2": round(self.area_m2, 2),
            "area_acres": round(self.area_m2 / 4_046.8564224, 3),
        }


def file_version(path: str | os.PathLike) -> str:
    """Cache key that changes whenever the file does."""
    try:
        stat = os.stat(path)
        raw = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        raw = str(path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def describe(path: str | os.PathLike) -> RasterInfo:
    """Open a GeoTIFF and summarise it, or explain why it cannot be used."""
    path = str(Path(path).expanduser())
    if not os.path.isfile(path):
        raise RasterError(f"File not found: {path}")

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise RasterError(
                    "This GeoTIFF carries no CRS. Re-export it as a georeferenced "
                    "GeoTIFF so detections can be placed on the map."
                )
            if src.count < 3:
                raise RasterError(
                    f"Expected at least 3 bands (RGB), found {src.count}. "
                    "The detector needs a true-colour orthomosaic."
                )

            b = src.bounds
            crs = src.crs.to_string()
            overviews = list(src.overviews(1))
            gsd = float(abs(src.transform.a))
            nodata = src.nodata
            width, height = src.width, src.height
            band_count, dtype = src.count, src.dtypes[0]
    except rasterio.RasterioIOError as exc:
        raise RasterError(f"Could not open GeoTIFF: {exc}") from exc

    tf = transformer(crs, "EPSG:4326")
    west, south = tf.transform(b.left, b.bottom)
    east, north = tf.transform(b.right, b.top)

    return RasterInfo(
        path=path,
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
        crs=crs,
        bounds_native=(b.left, b.bottom, b.right, b.top),
        bounds_wgs84=(west, south, east, north),
        center=((west + east) / 2.0, (south + north) / 2.0),
        gsd_m=gsd,
        size_bytes=os.path.getsize(path),
        has_overviews=bool(overviews),
        overview_levels=overviews,
        nodata=None if nodata is None else float(nodata),
    )


def build_overviews(path: str, levels: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)) -> list[int]:
    """Add overview pyramids in place so the map can render at low zooms.

    Without them every map tile has to read full-resolution pixels, which is
    what makes a multi-gigabyte mosaic feel frozen when panning.
    """
    # Skip levels that would shrink the image below one tile.
    with rasterio.open(path) as src:
        smallest = min(src.width, src.height)
    usable = [lv for lv in levels if smallest // lv >= 64] or [2]

    with rasterio.open(path, "r+") as dst:
        dst.build_overviews(usable, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    return usable


def percentile_stretch(
    path: str,
    lo_pct: float = settings.stretch_lo,
    hi_pct: float = settings.stretch_hi,
    max_sample: int = 1024,
) -> list[tuple[float, float]]:
    """Per-band display range from a downsampled read.

    Sampling keeps this cheap on huge mosaics, and the percentile clip is what
    makes a flat UAV mosaic look like the imagery people expect in QGIS.
    """
    with rasterio.open(path) as src:
        scale = max(src.width / max_sample, src.height / max_sample, 1.0)
        out_h = max(1, int(src.height / scale))
        out_w = max(1, int(src.width / scale))
        sample = src.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            masked=True,
            resampling=Resampling.average,
        ).astype("float32")

    ranges: list[tuple[float, float]] = []
    for band in range(3):
        values = np.ma.filled(sample[band], np.nan).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges.append((0.0, 255.0))
            continue
        # Mosaic borders are padded with zeros; excluding them keeps the
        # stretch from being dragged down by the background.
        positive = values[values > 0]
        if positive.size > max(100, values.size * 0.05):
            values = positive
        lo, hi = np.percentile(values, [lo_pct, hi_pct])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi <= lo:
            hi = lo + 1.0
        ranges.append((float(lo), float(hi)))
    return ranges


@dataclass(frozen=True)
class TileSpec:
    """One inference window, in raster pixel space."""

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def plan_tiles(
    info: RasterInfo,
    tile_size: int,
    side_overlap: float,
    front_overlap: float,
    region: tuple[int, int, int, int] | None = None,
) -> list[TileSpec]:
    """Lay out overlapping inference windows across the raster or a sub-region.

    ``region`` is ``(col0, row0, col1, row1)`` in pixels. Overlap is what stops
    a plant sitting on a tile seam from being missed; the duplicates it creates
    are removed later by :func:`plantai.detect.deduplicate`.
    """
    col0, row0, col1, row1 = region or (0, 0, info.width, info.height)
    col0 = max(0, int(col0))
    row0 = max(0, int(row0))
    col1 = min(info.width, int(col1))
    row1 = min(info.height, int(row1))
    if col1 <= col0 or row1 <= row0:
        return []

    span_x = col1 - col0
    span_y = row1 - row0
    tw = min(int(tile_size), span_x)
    th = min(int(tile_size), span_y)
    if tw < 1 or th < 1:
        return []

    step_x = max(int(tw * (1.0 - min(max(side_overlap, 0.0), 0.95))), 1)
    step_y = max(int(th * (1.0 - min(max(front_overlap, 0.0), 0.95))), 1)

    xs = list(range(col0, col1, step_x)) or [col0]
    ys = list(range(row0, row1, step_y)) or [row0]

    seen: set[tuple[int, int]] = set()
    tiles: list[TileSpec] = []
    for y in ys:
        for x in xs:
            # Clamp the last row/column inward so every window stays full-size.
            cx = min(x, col1 - tw)
            cy = min(y, row1 - th)
            key = (cx, cy)
            if key in seen:
                continue
            seen.add(key)
            tiles.append(TileSpec(cx, cy, tw, th))
    return tiles


def region_for_geometry(info: RasterInfo, geom, buffer_m: float = 0.5) -> tuple[int, int, int, int]:
    """Pixel window covering a lon/lat geometry, padded by ``buffer_m``."""
    project = transformer("EPSG:4326", info.crs).transform
    native = shapely_transform(project, geom)
    if buffer_m > 0:
        native = native.buffer(float(buffer_m))
    minx, miny, maxx, maxy = native.bounds

    with rasterio.open(info.path) as src:
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    win = win.round_offsets().round_lengths()
    col0 = max(int(win.col_off), 0)
    row0 = max(int(win.row_off), 0)
    col1 = min(int(win.col_off + win.width), info.width)
    row1 = min(int(win.row_off + win.height), info.height)
    return col0, row0, col1, row1


def read_tiles(
    path: str,
    tiles: list[TileSpec],
    stretch: list[tuple[float, float]] | None = None,
) -> Iterator[tuple[TileSpec, np.ndarray, object]]:
    """Yield ``(spec, uint8 HWC array, affine transform)`` for each window.

    Opening the dataset once and streaming windows keeps peak memory at one
    tile, which is what lets a 2 GB mosaic run on a laptop.
    """
    with rasterio.open(path) as src:
        for spec in tiles:
            data = src.read([1, 2, 3], window=spec.window, boundless=True, fill_value=0)
            img = to_uint8_rgb(data, stretch)
            yield spec, img, window_transform(spec.window, src.transform)


def to_uint8_rgb(data: np.ndarray, stretch: list[tuple[float, float]] | None) -> np.ndarray:
    """Convert a (3, H, W) band read into a contiguous uint8 HWC image."""
    if data.dtype == np.uint8 and not stretch:
        return np.ascontiguousarray(np.moveaxis(data, 0, -1))

    arr = np.asarray(data, dtype="float32")
    out = np.zeros(arr.shape, dtype="uint8")
    for band in range(min(3, arr.shape[0])):
        lo, hi = (stretch[band] if stretch else (0.0, 255.0))
        scaled = np.nan_to_num(arr[band], nan=lo, posinf=hi, neginf=lo)
        scaled = (scaled - lo) / max(hi - lo, 1e-9) * 255.0
        out[band] = np.clip(scaled, 0, 255).astype("uint8")
    return np.ascontiguousarray(np.moveaxis(out, 0, -1))
