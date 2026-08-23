"""End-to-end checks against the HTTP surface, using a synthetic orthomosaic."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from main import app

# The fixture raster sits in UTM 18N near the Virginia coast, at 1 cm/px.
ORIGIN_X, ORIGIN_Y, GSD = 357_000.0, 4_065_000.0, 0.01


@pytest.fixture(scope="module")
def ortho(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("raster") / "fixture.tif"
    size = 1024
    rng = np.random.default_rng(11)
    img = rng.integers(60, 120, size=(3, size, size)).astype("uint8")
    img[1, ::32, :] = 200  # crop rows
    profile = dict(
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:32618",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, GSD, GSD),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img)
    return str(path)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def dataset(client: TestClient, ortho: str) -> dict:
    res = client.post("/api/datasets/path", json={"path": ortho})
    assert res.status_code == 200, res.text
    return res.json()


@pytest.fixture(scope="module")
def detections(client: TestClient) -> dict:
    """A 200-point CSV: a grid of plants with a cluster of weeds inside it."""
    lines = ["center_x,center_y,class,confidence"]
    for i in range(180):
        x = ORIGIN_X + 0.5 + (i % 18) * 0.4
        y = ORIGIN_Y - 0.5 - (i // 18) * 0.4
        lines.append(f"{x},{y},plant,{0.5 + (i % 40) * 0.0125:.4f}")
    for i in range(20):
        x = ORIGIN_X + 1.0 + (i % 5) * 0.3
        y = ORIGIN_Y - 1.0 - (i // 5) * 0.3
        lines.append(f"{x},{y},weed,{0.4 + i * 0.02:.4f}")
    payload = "\n".join(lines).encode()

    res = client.post(
        "/api/detections/csv",
        files={"file": ("detections.csv", io.BytesIO(payload), "text/csv")},
        data={"source_crs": "EPSG:32618"},
    )
    assert res.status_code == 200, res.text
    return res.json()


def _aoi_around(west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[west, south], [west, north], [east, north], [east, south], [west, south]]
                    ],
                },
            }
        ],
    }


# ------------------------------------------------------------------ service
def test_health_and_config(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert "model" in health

    config = client.get("/api/config").json()
    assert config["tile_size"] > 0
    assert config["stand_target"] > 0


def test_model_reports_missing_checkpoint(client: TestClient) -> None:
    res = client.post("/api/model", json={"path": "/nonexistent/best.pt"})
    assert res.status_code == 400
    assert "not found" in res.json()["detail"].lower()


# ----------------------------------------------------------------- datasets
def test_dataset_registration(dataset: dict) -> None:
    assert dataset["crs"] == "EPSG:32618"
    assert dataset["width"] == dataset["height"] == 1024
    assert dataset["band_count"] == 3
    assert dataset["gsd_cm"] == pytest.approx(1.0, abs=0.01)
    south, west = dataset["leaflet_bounds"][0]
    north, east = dataset["leaflet_bounds"][1]
    assert south < north and west < east
    assert 36.0 < south < 37.0 and -77.0 < west < -76.0


def test_dataset_rejects_non_raster(client: TestClient, tmp_path) -> None:
    junk = tmp_path / "notes.tif"
    junk.write_bytes(b"not a geotiff")
    res = client.post("/api/datasets/path", json={"path": str(junk)})
    assert res.status_code == 400


def test_tiles_render_png(client: TestClient, dataset: dict) -> None:
    import math

    lon, lat = dataset["center"]
    z = 20
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
    )
    res = client.get(f"/api/datasets/{dataset['id']}/tiles/{z}/{x}/{y}.png")
    assert res.status_code == 200
    assert res.content[:4] == b"\x89PNG"

    # A tile on the other side of the planet is transparent, not an error.
    off = client.get(f"/api/datasets/{dataset['id']}/tiles/2/0/0.png")
    assert off.status_code == 200
    assert off.content[:4] == b"\x89PNG"


def test_overviews_can_be_built(client: TestClient, dataset: dict) -> None:
    res = client.post(f"/api/datasets/{dataset['id']}/overviews")
    assert res.status_code == 200
    body = res.json()
    assert body["has_overviews"] is True
    # Second call is a no-op rather than an error.
    again = client.post(f"/api/datasets/{dataset['id']}/overviews").json()
    assert again["built"] == []


# --------------------------------------------------------------- detections
def test_csv_import(detections: dict) -> None:
    assert detections["count"] == 200
    assert detections["summary"]["plants"] == 180
    assert detections["summary"]["weeds"] == 20
    assert detections["summary"]["weed_pressure_pct"] == pytest.approx(10.0)
    assert "projected from EPSG:32618" in detections["import"]["coordinates"]


def test_csv_import_rejects_missing_coordinates(client: TestClient) -> None:
    res = client.post(
        "/api/detections/csv",
        files={"file": ("bad.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert res.status_code == 400
    assert "coordinates" in res.json()["detail"].lower()


def test_confidence_filter_applies(client: TestClient, detections: dict) -> None:
    everything = client.get(f"/api/detections/{detections['id']}").json()
    filtered = client.get(f"/api/detections/{detections['id']}?min_conf=0.9").json()
    assert filtered["summary"]["total"] < everything["summary"]["total"]
    assert filtered["summary"]["total"] > 0


def test_binary_points_roundtrip(client: TestClient, detections: dict) -> None:
    res = client.get(f"/api/detections/{detections['id']}/points")
    assert res.status_code == 200
    blob = res.content
    assert blob[:4] == b"PAI1"

    count = int(np.frombuffer(blob, dtype="<u4", count=1, offset=4)[0])
    assert count == 200

    ref = np.frombuffer(blob, dtype="<f8", count=2, offset=8)
    lon = np.frombuffer(blob, dtype="<f4", count=count, offset=24) + ref[0]
    lat = np.frombuffer(blob, dtype="<f4", count=count, offset=24 + 4 * count) + ref[1]
    cls = np.frombuffer(blob, dtype="u1", count=count, offset=24 + 12 * count)

    assert int((cls == 0).sum()) == 180  # plants
    assert int((cls == 1).sum()) == 20  # weeds
    # The float32 offsets must stay well under a centimetre of error.
    assert float(np.abs(lon - lon.mean()).max()) < 0.001
    assert 36.0 < lat.mean() < 37.0


def test_export_csv(client: TestClient, detections: dict) -> None:
    res = client.get(f"/api/detections/{detections['id']}/export.csv")
    assert res.status_code == 200
    header = res.text.splitlines()[0].split(",")
    assert {"class", "confidence", "longitude", "latitude", "global_x", "global_y"} <= set(header)


# ---------------------------------------------------------------------- AOI
def test_aoi_counts_only_the_latest_polygon(client: TestClient, detections: dict) -> None:
    """Old drawings must not inflate the count — only the last shape counts."""
    all_points = _aoi_around(-180, -85, 180, 85)
    tiny = client.get(f"/api/detections/{detections['id']}/points").content
    ref = np.frombuffer(tiny, dtype="<f8", count=2, offset=8)
    lon0, lat0 = float(ref[0]), float(ref[1])
    small = _aoi_around(lon0 - 0.00002, lat0 - 0.00002, lon0 + 0.00002, lat0 + 0.00002)

    stale_then_current = {
        "type": "FeatureCollection",
        "features": all_points["features"] + small["features"],
    }
    both = client.post(
        f"/api/detections/{detections['id']}/aoi",
        json={"aoi": stale_then_current, "min_conf": 0.0},
    ).json()
    only_small = client.post(
        f"/api/detections/{detections['id']}/aoi", json={"aoi": small, "min_conf": 0.0}
    ).json()

    assert both["total"] == only_small["total"]
    assert both["total"] < 200


def test_aoi_area_and_density(client: TestClient, detections: dict) -> None:
    box = _aoi_around(-76.61, 36.71, -76.60, 36.72)
    res = client.post(
        f"/api/detections/{detections['id']}/aoi", json={"aoi": box, "min_conf": 0.0}
    ).json()
    assert res["area_acres"] > 0
    assert res["area_m2"] > res["area_ha"]
    if res["plants"]:
        assert res["plants_per_acre"] == pytest.approx(
            res["plants"] / res["area_acres"], rel=0.01
        )


def test_aoi_requires_a_polygon(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/aoi",
        json={"aoi": {"type": "FeatureCollection", "features": []}},
    )
    assert res.status_code == 400


# ----------------------------------------------------------------- analysis
def test_analysis_area_basis(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/analysis",
        json={"scope": "all", "min_conf": 0.0, "basis": "area", "target_plants_acre": 150000},
    ).json()
    metrics = res["metrics"]
    assert metrics["plants"] == 180
    assert metrics["basis"] == "area"
    assert metrics["area_acres"] > 0
    assert metrics["plants_per_acre"] > 0
    assert res["status_detail"]
    assert len(res["confidence_bands"]) == 6
    assert [c["class"] for c in res["class_confidence"]] == ["Plant", "Weed"]
    assert res["rows"] == []


def test_analysis_row_basis(client: TestClient, detections: dict) -> None:
    body = {
        "scope": "all",
        "min_conf": 0.0,
        "basis": "rows",
        "geometry": {"rows": 4, "row_length_ft": 15.0, "row_spacing_in": 15.0},
        "target_plants_acre": 150000,
        "plot_id": "plot_042",
    }
    res = client.post(f"/api/detections/{detections['id']}/analysis", json=body).json()
    metrics = res["metrics"]

    assert metrics["basis"] == "rows"
    # 4 rows x 15 ft x 1.25 ft spacing = 75 ft^2 = 0.001722 acres
    assert metrics["area_acres"] == pytest.approx(75 / 43560, rel=1e-3)
    assert metrics["total_row_length_ft"] == pytest.approx(60.0)
    assert metrics["plants_per_ft"] == pytest.approx(180 / 60.0, rel=1e-3)
    assert len(res["rows"]) == 4
    # Every plant lands in exactly one row.
    assert sum(r["plants"] for r in res["rows"]) == metrics["plants"]
    assert sum(r["weeds"] for r in res["rows"]) == metrics["weeds"]


def test_analysis_rejects_incomplete_geometry(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/analysis",
        json={"scope": "all", "basis": "rows", "geometry": {"rows": 0}},
    )
    assert res.status_code == 400


def test_analysis_aoi_scope_needs_a_polygon(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/analysis", json={"scope": "aoi", "aoi": None}
    )
    assert res.status_code == 400


@pytest.mark.parametrize("kind", ["stand", "plots", "rows"])
def test_analysis_exports(client: TestClient, detections: dict, kind: str) -> None:
    body = {
        "scope": "all",
        "basis": "rows",
        "geometry": {"rows": 3, "row_length_ft": 20.0, "row_spacing_in": 30.0},
        "plot_id": "plot_007",
    }
    res = client.post(f"/api/detections/{detections['id']}/exports/{kind}", json=body)
    assert res.status_code == 200
    lines = res.text.strip().splitlines()
    assert len(lines) >= 2
    assert "plot_007" in res.text


def test_rows_export_requires_row_basis(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/exports/rows",
        json={"scope": "all", "basis": "area"},
    )
    assert res.status_code == 400


def test_pdf_report(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/report.pdf",
        json={
            "scope": "all",
            "basis": "rows",
            "geometry": {"rows": 4, "row_length_ft": 15, "row_spacing_in": 15},
            "plot_id": "plot_001",
            "dataset_name": "fixture.tif",
        },
    )
    assert res.status_code == 200
    assert res.content[:5] == b"%PDF-"
    assert len(res.content) > 1000


# ------------------------------------------------------------------ detect
def test_detect_without_a_model_is_refused(client: TestClient, dataset: dict) -> None:
    res = client.post("/api/detect", json={"dataset_id": dataset["id"], "mode": "field"})
    assert res.status_code == 400
    assert "model" in res.json()["detail"].lower()


def test_detect_unknown_dataset(client: TestClient) -> None:
    res = client.post("/api/detect", json={"dataset_id": "nope", "mode": "field"})
    assert res.status_code == 404


def test_unknown_ids_are_404(client: TestClient) -> None:
    assert client.get("/api/detections/nope").status_code == 404
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_frontend_is_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "PlantAI" in res.text


def test_analysis_warns_when_row_geometry_mismatches_the_area(
    client: TestClient, detections: dict
) -> None:
    """A plot geometry far smaller than the selection is the classic way a
    stand count comes out at several hundred percent of target."""
    box = _aoi_around(-76.61, 36.71, -76.60, 36.72)  # ~245 acres
    res = client.post(
        f"/api/detections/{detections['id']}/analysis",
        json={
            "scope": "aoi",
            "aoi": box,
            "basis": "rows",
            "geometry": {"rows": 4, "row_length_ft": 15, "row_spacing_in": 15},
        },
    ).json()
    assert res["warning"]
    assert "row geometry" in res["warning"]

    # A geometry that matches the selection stays quiet.
    matched = client.post(
        f"/api/detections/{detections['id']}/analysis",
        json={
            "scope": "aoi",
            "aoi": box,
            "basis": "rows",
            "geometry": {"rows": 2134, "row_length_ft": 2000, "row_spacing_in": 30},
        },
    ).json()
    assert matched["warning"] == ""


def test_analysis_area_basis_never_warns(client: TestClient, detections: dict) -> None:
    res = client.post(
        f"/api/detections/{detections['id']}/analysis",
        json={"scope": "all", "basis": "area"},
    ).json()
    assert res["warning"] == ""


def test_tiles_are_safe_under_concurrent_requests(client: TestClient, dataset: dict) -> None:
    """A browser opens six connections at once.

    GDAL dataset handles are not thread-safe, and FastAPI runs sync endpoints
    in a threadpool. Sharing one reader across those threads returns torn
    blocks rather than an error, so this guards the per-thread handles.
    """
    import math
    from concurrent.futures import ThreadPoolExecutor

    lon, lat = dataset["center"]
    z = 20
    n = 2**z
    x0 = int((lon + 180.0) / 360.0 * n)
    y0 = int(
        (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
    )
    requests = [
        (x0 + dx, y0 + dy)
        for dx in range(2)
        for dy in range(2)
        for _ in range(6)
    ]

    def fetch(coords):
        x, y = coords
        return client.get(f"/api/datasets/{dataset['id']}/tiles/{z}/{x}/{y}.png")

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(fetch, requests))

    assert all(r.status_code == 200 for r in responses)
    assert all(r.content[:4] == b"\x89PNG" for r in responses)
