"""Coordinate transforms, area-of-interest handling and point-in-polygon tests.

Everything the UI draws arrives as GeoJSON in EPSG:4326. Areas and distances
have to be measured on a plane, so AOIs are reprojected into the UTM zone that
contains them before any area is computed.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep

M2_PER_ACRE = 4_046.856_422_4
M2_PER_HA = 10_000.0
ACRE_FT2 = 43_560.0
FT_PER_M = 3.280_839_895


@lru_cache(maxsize=64)
def transformer(src: str, dst: str) -> Transformer:
    """Transformers are expensive to build and safe to share."""
    return Transformer.from_crs(src, dst, always_xy=True)


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """The UTM zone EPSG code covering a longitude/latitude."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def to_wgs84(x: np.ndarray, y: np.ndarray, src_crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Project arrays of projected coordinates to lon/lat."""
    if str(src_crs).upper().replace(" ", "") in {"EPSG:4326", "OGC:CRS84"}:
        return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    lon, lat = transformer(str(src_crs), "EPSG:4326").transform(
        np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    )
    return np.atleast_1d(lon).astype("float64"), np.atleast_1d(lat).astype("float64")


def parse_aoi(geojson: Any) -> Polygon | None:
    """Return the most recently drawn polygon from a GeoJSON payload.

    The map hands back every shape still on the canvas. Unioning them would let
    a forgotten polygon inflate the next plot's counts, so only the last
    polygon/multipolygon is ever used — matching what the user sees selected.
    """
    if geojson is None:
        return None
    if isinstance(geojson, (str, bytes)):
        try:
            geojson = json.loads(geojson)
        except (ValueError, TypeError):
            return None
    if not isinstance(geojson, dict):
        if isinstance(geojson, list):
            geojson = {"type": "FeatureCollection", "features": geojson}
        else:
            return None

    gtype = geojson.get("type")
    if gtype in {"Polygon", "MultiPolygon"}:
        candidates = [geojson]
    elif gtype == "Feature":
        candidates = [geojson.get("geometry") or {}]
    else:
        candidates = [
            f.get("geometry") or {}
            for f in geojson.get("features", [])
            if isinstance(f, dict)
        ]

    polygons = [c for c in candidates if c.get("type") in {"Polygon", "MultiPolygon"}]
    if not polygons:
        return None
    try:
        geom = shape(polygons[-1])
    except (ValueError, TypeError, AttributeError):
        return None
    if geom.is_empty:
        return None
    # Self-intersecting hand-drawn polygons would otherwise raise on `contains`.
    return geom if geom.is_valid else geom.buffer(0)


def aoi_utm_crs(geom) -> str:
    """The metric CRS to measure a lon/lat geometry in."""
    lon, lat = geom.centroid.x, geom.centroid.y
    return f"EPSG:{utm_epsg_for_lonlat(lon, lat)}"


def area_metrics(geom) -> dict[str, float]:
    """Area of a lon/lat geometry in m², hectares and acres."""
    if geom is None or geom.is_empty:
        return {"area_m2": 0.0, "area_ha": 0.0, "area_acres": 0.0}
    try:
        project = transformer("EPSG:4326", aoi_utm_crs(geom)).transform
        area_m2 = float(shapely_transform(project, geom).area)
    except Exception:
        # Degenerate geometry: fall back to a local flat-earth approximation.
        lat = math.radians(geom.centroid.y)
        area_m2 = float(geom.area) * (111_320.0**2) * math.cos(lat)
    return {
        "area_m2": area_m2,
        "area_ha": area_m2 / M2_PER_HA,
        "area_acres": area_m2 / M2_PER_ACRE,
    }


def points_within(geom, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Boolean mask of the points falling inside ``geom``.

    The bounding box is tested first with numpy; only the survivors go through
    shapely. On a 400k-point field that is the difference between a few
    milliseconds and half a minute.
    """
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    mask = np.zeros(lon.shape, dtype=bool)
    if geom is None or geom.is_empty or lon.size == 0:
        return mask

    minx, miny, maxx, maxy = geom.bounds
    in_box = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    if not in_box.any():
        return mask

    candidates = np.flatnonzero(in_box)
    try:
        from shapely import contains_xy  # shapely >= 2.0

        mask[candidates] = contains_xy(geom, lon[candidates], lat[candidates])
    except ImportError:  # pragma: no cover - shapely 1.x fallback
        from shapely.geometry import Point

        prepared = prep(geom)
        mask[candidates] = [
            prepared.contains(Point(float(lon[i]), float(lat[i]))) for i in candidates
        ]
    return mask


def bbox_area_acres(lon: Iterable[float], lat: Iterable[float]) -> float:
    """Rough acreage of the extent covered by a point cloud.

    Only a fallback for when no AOI is drawn — the true mapped area is smaller
    than its bounding box whenever the field is not rectangular.
    """
    lon = np.asarray(list(lon), dtype="float64")
    lat = np.asarray(list(lat), dtype="float64")
    if lon.size < 2 or lat.size < 2:
        return 0.0
    lon_span = float(lon.max() - lon.min())
    lat_span = float(lat.max() - lat.min())
    if lon_span <= 0 or lat_span <= 0:
        return 0.0
    mid = math.radians(float(lat.mean()))
    area_m2 = (lat_span * 111_320.0) * (lon_span * 111_320.0 * math.cos(mid))
    return max(area_m2 / M2_PER_ACRE, 0.0)


def geojson_bounds(geom) -> list[list[float]] | None:
    """``[[south, west], [north, east]]`` — the order Leaflet expects."""
    if geom is None or geom.is_empty:
        return None
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]
