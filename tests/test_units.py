"""Unit tests for the pure-logic modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plantai.analytics import (
    RowGeometry,
    class_confidence,
    confidence_bands,
    round_density,
    rows_table,
    stand_metrics,
)
from plantai.csvio import CsvIngestError, load_detections
from plantai.detect import dedupe_mask, deduplicate, summarise
from plantai.geometry import area_metrics, bbox_area_acres, parse_aoi, points_within, utm_epsg_for_lonlat
from plantai.model import classify_label
from plantai.raster import TileSpec, plan_tiles, RasterInfo
from plantai.store import pack_points


# ─────────────────────────────────────────────────────── geometry ──
def test_utm_zone_selection() -> None:
    assert utm_epsg_for_lonlat(-76.6, 36.7) == 32618       # Virginia, north
    assert utm_epsg_for_lonlat(-58.4, -34.6) == 32721      # Buenos Aires, south
    assert utm_epsg_for_lonlat(179.9, 0.0) == 32660        # last northern zone


def test_parse_aoi_takes_only_the_last_polygon() -> None:
    big = {"type": "Polygon", "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]}
    small = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": big}, {"type": "Feature", "geometry": small}],
    }
    assert parse_aoi(fc).bounds == (0.0, 0.0, 1.0, 1.0)


def test_parse_aoi_accepts_the_shapes_the_map_returns() -> None:
    geometry = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
    assert parse_aoi(geometry) is not None                                    # bare geometry
    assert parse_aoi({"type": "Feature", "geometry": geometry}) is not None    # feature
    assert parse_aoi([{"type": "Feature", "geometry": geometry}]) is not None  # bare list
    assert parse_aoi('{"type": "Feature", "geometry": %s}' % str(geometry).replace("'", '"')) is not None
    assert parse_aoi(None) is None
    assert parse_aoi({"type": "FeatureCollection", "features": []}) is None
    assert parse_aoi({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}) is None


def test_parse_aoi_repairs_a_self_intersecting_polygon() -> None:
    """A hand-drawn bowtie must not blow up the containment test."""
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    geom = parse_aoi(bowtie)
    assert geom is not None and geom.is_valid


def test_area_metrics_are_consistent() -> None:
    square = parse_aoi(
        {"type": "Polygon", "coordinates": [[[-76.6, 36.7], [-76.6, 36.71], [-76.59, 36.71], [-76.59, 36.7], [-76.6, 36.7]]]}
    )
    area = area_metrics(square)
    assert area["area_ha"] == pytest.approx(area["area_m2"] / 10_000)
    assert area["area_acres"] == pytest.approx(area["area_m2"] / 4046.8564224)
    # ~1.11 km x 0.89 km at this latitude
    assert 90 < area["area_ha"] < 110


def test_points_within_matches_the_bounding_box_shortcut() -> None:
    square = parse_aoi({"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]})
    lon = np.array([0.5, 2.0, 0.99, -0.01, 0.5])
    lat = np.array([0.5, 0.5, 0.99, 0.5, 5.0])
    assert list(points_within(square, lon, lat)) == [True, False, True, False, False]
    assert points_within(square, np.array([]), np.array([])).size == 0


def test_bbox_area_needs_two_distinct_points() -> None:
    assert bbox_area_acres([1.0], [1.0]) == 0.0
    assert bbox_area_acres([1.0, 1.0], [2.0, 2.0]) == 0.0
    assert bbox_area_acres([-76.6, -76.59], [36.7, 36.71]) > 0


# ────────────────────────────────────────────────────────── model ──
@pytest.mark.parametrize(
    "label,expected",
    [("plant", "plant"), ("Plant", "plant"), ("soybean", "plant"), ("CROP", "plant"),
     ("weed", "weed"), ("Palmer", "weed"), ("ragweed", "weed"),
     ("unknown_species", "weed")],  # unrecognised labels are the safer error as weeds
)
def test_class_roles(label: str, expected: str) -> None:
    assert classify_label(label) == expected


# ───────────────────────────────────────────────────────── detect ──
def test_dedupe_keeps_the_most_confident_of_a_cluster() -> None:
    x = np.array([0.0, 0.05, 0.30, 10.0])
    y = np.zeros(4)
    conf = np.array([0.5, 0.9, 0.7, 0.6])
    mask = dedupe_mask(x, y, conf, 0.10)
    assert list(conf[mask]) == [0.9, 0.7, 0.6]


def test_dedupe_radius_zero_keeps_everything() -> None:
    x = np.zeros(5)
    assert dedupe_mask(x, x, x, 0.0).all()
    assert dedupe_mask(np.array([]), np.array([]), np.array([]), 0.1).size == 0


def test_dedupe_is_per_class() -> None:
    """A weed beside a plant is a second detection, not a duplicate."""
    df = pd.DataFrame({
        "class": ["plant", "plant", "weed", "weed"],
        "confidence": [0.5, 0.9, 0.8, 0.4],
        "global_x": [0.0, 0.02, 0.0, 0.03],
        "global_y": [0.0, 0.0, 0.0, 0.0],
    })
    out = deduplicate(df, 0.10)
    assert sorted(out["class"]) == ["plant", "weed"]
    assert sorted(out["confidence"]) == [0.8, 0.9]


def test_dedupe_scales_to_a_field_sized_run() -> None:
    rng = np.random.default_rng(3)
    n = 100_000
    x = rng.uniform(0, 300, n)
    y = rng.uniform(0, 300, n)
    conf = rng.uniform(0, 1, n)
    kept = dedupe_mask(x, y, conf, 0.10)
    assert 0 < int(kept.sum()) <= n


def test_summarise_handles_an_empty_frame() -> None:
    assert summarise(pd.DataFrame()) == {
        "total": 0, "plants": 0, "weeds": 0, "weed_pressure_pct": 0.0, "avg_confidence": 0.0
    }


# ───────────────────────────────────────────────────────── raster ──
def _fake_info(width: int, height: int) -> RasterInfo:
    return RasterInfo(
        path="", width=width, height=height, band_count=3, dtype="uint8", crs="EPSG:32618",
        bounds_native=(0, 0, 1, 1), bounds_wgs84=(0, 0, 1, 1), center=(0, 0),
        gsd_m=0.01, size_bytes=0, has_overviews=False, overview_levels=[], nodata=None,
    )


def test_tiles_cover_the_raster_and_stay_in_bounds() -> None:
    info = _fake_info(2000, 1500)
    tiles = plan_tiles(info, tile_size=640, side_overlap=0.3, front_overlap=0.3)
    assert tiles
    for tile in tiles:
        assert tile.col_off >= 0 and tile.row_off >= 0
        assert tile.col_off + tile.width <= info.width
        assert tile.row_off + tile.height <= info.height
    # The bottom-right corner has to be reachable, or edge plants go uncounted.
    assert any(t.col_off + t.width == info.width for t in tiles)
    assert any(t.row_off + t.height == info.height for t in tiles)


def test_tiles_shrink_to_fit_a_small_region() -> None:
    info = _fake_info(2000, 1500)
    tiles = plan_tiles(info, 1280, 0.3, 0.3, region=(100, 100, 400, 300))
    assert tiles
    assert all(t.width <= 300 and t.height <= 200 for t in tiles)


def test_tiles_are_deduplicated_and_empty_regions_are_skipped() -> None:
    info = _fake_info(640, 640)
    # One window covers the whole raster; overlap must not produce copies.
    assert plan_tiles(info, 640, 0.9, 0.9) == [TileSpec(0, 0, 640, 640)]
    assert plan_tiles(info, 640, 0.3, 0.3, region=(10, 10, 10, 10)) == []


def test_higher_overlap_produces_more_tiles() -> None:
    info = _fake_info(4000, 4000)
    light = plan_tiles(info, 1280, 0.3, 0.3)
    heavy = plan_tiles(info, 1280, 0.7, 0.8)
    assert len(heavy) > len(light)


# ────────────────────────────────────────────────────── analytics ──
def test_row_geometry_area() -> None:
    geometry = RowGeometry(rows=4, row_length_ft=15.0, row_spacing_in=15.0)
    assert geometry.total_row_length_ft == 60.0
    assert geometry.area_ft2 == pytest.approx(75.0)
    assert geometry.area_acres == pytest.approx(75.0 / 43_560)
    assert not RowGeometry(rows=4, row_length_ft=0, row_spacing_in=15).is_complete


@pytest.mark.parametrize(
    "plants,target,alert,expected_status,expected_tone",
    [
        (95, 150_000, 85, "Below threshold", "red"),
        (0, 150_000, 85, "No stand estimate", "amber"),
    ],
)
def test_stand_status_thresholds(plants, target, alert, expected_status, expected_tone) -> None:
    df = pd.DataFrame({"class": ["plant"] * plants, "confidence": [0.9] * plants})
    metrics = stand_metrics(df, target, alert, geometry=RowGeometry(4, 15, 15))
    assert metrics.status == expected_status
    assert metrics.tone == expected_tone


def test_stand_status_on_and_above_target() -> None:
    geometry = RowGeometry(rows=4, row_length_ft=15.0, row_spacing_in=15.0)
    acres = geometry.area_acres

    on_target = int(round(150_000 * acres))
    df = pd.DataFrame({"class": ["plant"] * on_target, "confidence": [0.9] * on_target})
    assert stand_metrics(df, 150_000, 85, geometry=geometry).status == "On target"

    high = int(round(200_000 * acres))
    df = pd.DataFrame({"class": ["plant"] * high, "confidence": [0.9] * high})
    metrics = stand_metrics(df, 150_000, 85, geometry=geometry)
    assert metrics.status == "Above target"
    assert metrics.tone == "amber"


def test_rows_allocate_every_detection() -> None:
    df = pd.DataFrame({"class": ["plant"] * 97 + ["weed"] * 5, "confidence": [0.9] * 102})
    metrics = stand_metrics(df, 150_000, 85, geometry=RowGeometry(4, 15, 15))
    rows = rows_table(metrics)
    assert len(rows) == 4
    assert sum(r["plants"] for r in rows) == 97
    assert sum(r["weeds"] for r in rows) == 5
    # The remainder is spread over the first rows rather than dropped.
    assert max(r["plants"] for r in rows) - min(r["plants"] for r in rows) == 1


def test_rows_table_needs_row_geometry() -> None:
    df = pd.DataFrame({"class": ["plant"], "confidence": [0.9]})
    assert rows_table(stand_metrics(df, 150_000, 85, area_acres=1.0)) == []


def test_confidence_bands_partition_every_detection() -> None:
    conf = np.linspace(0.0, 1.0, 501)
    df = pd.DataFrame({"class": ["plant"] * 501, "confidence": conf})
    bands = confidence_bands(df)
    assert len(bands) == 6
    assert sum(b["count"] for b in bands) == 501


def test_class_confidence_reports_missing_classes() -> None:
    df = pd.DataFrame({"class": ["plant"] * 3, "confidence": [0.5, 0.7, 0.9]})
    stats = {entry["class"]: entry for entry in class_confidence(df)}
    assert stats["Plant"]["count"] == 3
    assert stats["Plant"]["avg_conf"] == pytest.approx(70.0)
    assert stats["Plant"]["min_conf"] == pytest.approx(50.0)
    assert stats["Weed"]["count"] == 0
    assert stats["Weed"]["avg_conf"] is None


@pytest.mark.parametrize(
    "value,expected",
    [(0.73492628, 0.7349), (55176.4321, 55176.4), (0.0345678, 0.03457), (0.0, 0.0)],
)
def test_density_rounding_keeps_sparse_values(value: float, expected: float) -> None:
    assert round_density(value) == expected


# ────────────────────────────────────────────────────────── csvio ──
def test_csv_accepts_the_three_coordinate_conventions() -> None:
    utm, report = load_detections(b"center_x,center_y,class\n357000.5,4065000.2,plant\n")
    assert report["source_crs"] == "EPSG:32618"
    assert -77 < utm["longitude"].iloc[0] < -76

    latlon, report = load_detections(b"latitude,longitude,class\n36.7,-76.6,weed\n")
    assert report["source_crs"] == "EPSG:4326"
    assert latlon["class"].iloc[0] == "weed"

    globals_, _ = load_detections(b"global_x,global_y\n357000.5,4065000.2\n")
    assert len(globals_) == 1


def test_csv_rescales_percentage_confidence() -> None:
    df, report = load_detections(b"lat,lon,score\n36.7,-76.6,91\n36.71,-76.59,44\n")
    assert df["confidence"].max() <= 1.0
    assert "percentages" in report["confidence_note"]


def test_csv_defaults_are_reported_not_hidden() -> None:
    _, report = load_detections(b"lat,lon\n36.7,-76.6\n")
    assert "class_note" in report and "confidence_note" in report


def test_csv_uses_a_crs_column_when_present() -> None:
    df, report = load_detections(b"global_x,global_y,crs\n500000,4649776,EPSG:32631\n")
    assert report["source_crs"] == "EPSG:32631"
    assert 2.9 < float(df["longitude"].iloc[0]) < 3.1  # UTM 31N central meridian


def test_csv_drops_unusable_rows_and_reports_it() -> None:
    df, report = load_detections(b"latitude,longitude\n36.7,-76.6\n,\n")
    assert len(df) == 1
    assert report["dropped"] == 1


def test_csv_rejects_unusable_files() -> None:
    with pytest.raises(CsvIngestError):
        load_detections(b"a,b\n1,2\n")
    with pytest.raises(CsvIngestError):
        load_detections(b"latitude,longitude\n")
    with pytest.raises(CsvIngestError):
        load_detections(b"latitude,longitude\nnope,nope\n")


# ────────────────────────────────────────────────────────── store ──
def test_pack_points_layout_and_precision() -> None:
    df = pd.DataFrame({
        "class": ["plant", "weed", "plant"],
        "confidence": [0.9, 0.4, 0.75],
        "longitude": [-76.601140, -76.601135, -76.601150],
        "latitude": [36.719850, 36.719855, 36.719840],
    })
    blob = pack_points(df)
    assert blob[:4] == b"PAI1"

    count = int(np.frombuffer(blob, dtype="<u4", count=1, offset=4)[0])
    assert count == 3

    ref = np.frombuffer(blob, dtype="<f8", count=2, offset=8)
    lon = np.frombuffer(blob, dtype="<f4", count=count, offset=24) + ref[0]
    cls = np.frombuffer(blob, dtype="u1", count=count, offset=24 + 12 * count)

    assert list(cls) == [0, 1, 0]
    # float32 offsets must stay far below a centimetre (~1e-7 degrees).
    assert float(np.abs(lon - df["longitude"].to_numpy()).max()) < 1e-9


def test_pack_points_filters_and_handles_empty() -> None:
    df = pd.DataFrame({
        "class": ["plant", "weed"], "confidence": [0.9, 0.1],
        "longitude": [-76.6, -76.6], "latitude": [36.7, 36.7],
    })
    filtered = pack_points(df, min_conf=0.5)
    assert int(np.frombuffer(filtered, dtype="<u4", count=1, offset=4)[0]) == 1

    for blob in (pack_points(pd.DataFrame()), pack_points(df, min_conf=0.99)):
        assert blob[:4] == b"PAI1"
        assert int(np.frombuffer(blob, dtype="<u4", count=1, offset=4)[0]) == 0
