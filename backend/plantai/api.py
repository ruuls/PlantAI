"""HTTP surface for the PlantAI detection console."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import exports, tiles
from .analytics import (
    RowGeometry,
    class_confidence,
    confidence_bands,
    fallback_area_acres,
    round_density,
    stand_metrics,
    status_detail,
    rows_table,
)
from .config import settings
from .csvio import CsvIngestError, load_detections
from .detect import DetectionParams, summarise
from .geometry import M2_PER_ACRE, M2_PER_HA, area_metrics, parse_aoi, points_within
from .jobs import jobs
from .model import ModelError, detector
from .raster import RasterError, build_overviews, region_for_geometry
from .report import ReportError, build_report
from .schemas import (
    AnalysisRequest,
    AoiRequest,
    DetectRequest,
    LoadModelRequest,
    RegisterPathRequest,
    ReportRequest,
)
from .store import pack_points, store

router = APIRouter(prefix="/api")

RASTER_SUFFIXES = {".tif", ".tiff"}
MODEL_SUFFIXES = {".pt"}


# ────────────────────────────────────────────────────────────── service ──
@router.get("/health")
def health() -> dict:
    return {"status": "ok", "model": detector.status().as_dict()}


@router.get("/config")
def config() -> dict:
    """Defaults the UI starts from, so both halves agree on one source."""
    return {
        "tile_size": settings.tile_size,
        "conf": settings.conf,
        "iou": settings.iou,
        "side_overlap": settings.side_overlap,
        "front_overlap": settings.front_overlap,
        "dedup_radius_m": settings.dedup_radius_m,
        "stand_target": settings.stand_target,
        "stand_alert_pct": settings.stand_alert_pct,
        "allow_local_paths": settings.allow_local_paths,
        "library_dirs": settings.library_dirs,
        "max_upload_mb": settings.max_upload_mb,
        "plant_classes": settings.plant_classes,
        "weed_classes": settings.weed_classes,
    }


# ──────────────────────────────────────────────────────────────── model ──
@router.get("/model")
def model_status() -> dict:
    return detector.status().as_dict()


@router.post("/model")
def load_model(payload: LoadModelRequest) -> dict:
    status = detector.load(payload.path)
    if not status.loaded:
        raise HTTPException(status_code=400, detail=status.error or "Could not load the model.")
    return status.as_dict()


@router.get("/library")
def library(kind: str = Query("raster", pattern="^(raster|model)$")) -> dict:
    """List candidate files under the configured library directories.

    Only these directories are readable, so the browser can offer server-side
    files without turning the endpoint into a filesystem viewer.
    """
    suffixes = RASTER_SUFFIXES if kind == "raster" else MODEL_SUFFIXES
    entries: list[dict] = []
    for root in settings.library_dirs:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "dir": str(path.parent),
                    "size_bytes": size,
                    "size_label": _size_label(size),
                }
            )
            if len(entries) >= 200:
                break
    return {"kind": kind, "roots": settings.library_dirs, "files": entries}


def _size_label(size: int) -> str:
    if size >= 1e9:
        return f"{size / 1e9:.2f} GB"
    if size >= 1e6:
        return f"{size / 1e6:.1f} MB"
    return f"{size / 1e3:.0f} KB"


# ───────────────────────────────────────────────────────────── datasets ──
@router.get("/datasets")
def list_datasets() -> dict:
    return {"datasets": [d.as_dict() for d in store.datasets()]}


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "ortho.tif").suffix.lower()
    if suffix not in RASTER_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a .tif or .tiff orthomosaic.")

    target = settings.uploads_dir / f"{int(time.time() * 1000)}{suffix}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as sink:
            while chunk := await file.read(4 * 1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the {settings.max_upload_mb} MB upload limit. "
                            "Register it by path instead."
                        ),
                    )
                sink.write(chunk)
        dataset = store.add_dataset(target, Path(file.filename or target.name).name, "upload")
    except RasterError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return dataset.as_dict()


@router.post("/datasets/path")
def register_dataset(payload: RegisterPathRequest) -> dict:
    if not settings.allow_local_paths:
        raise HTTPException(
            status_code=403, detail="Server-side paths are disabled. Upload the file instead."
        )
    path = Path(payload.path).expanduser()
    try:
        dataset = store.add_dataset(path, path.name, "path")
    except RasterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dataset.as_dict()


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    dataset = _dataset_or_404(dataset_id)
    payload = dataset.as_dict()
    try:
        payload["min_zoom"], payload["max_zoom"] = tiles.zoom_range(dataset.info.path)
    except Exception:
        payload["min_zoom"], payload["max_zoom"] = 0, 24
    return payload


@router.post("/datasets/{dataset_id}/overviews")
def make_overviews(dataset_id: str) -> dict:
    """Build the pyramid in place — the fix for a mosaic that pans slowly."""
    dataset = _dataset_or_404(dataset_id)
    if dataset.info.has_overviews:
        return {"built": [], "message": "This raster already has overviews.", **dataset.as_dict()}
    if not os.access(dataset.info.path, os.W_OK):
        raise HTTPException(
            status_code=400,
            detail="The GeoTIFF is not writable, so overviews cannot be added in place.",
        )
    # The open reader holds the file; drop it before writing.
    tiles.close(dataset.info.path)
    try:
        levels = build_overviews(dataset.info.path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not build overviews: {exc}") from exc
    refreshed = store.refresh_dataset(dataset_id)
    return {"built": levels, "message": "Overviews built.", **refreshed.as_dict()}


@router.get("/datasets/{dataset_id}/tiles/{z}/{x}/{y}.png")
def raster_tile(dataset_id: str, z: int, x: int, y: int) -> Response:
    dataset = _dataset_or_404(dataset_id)
    request = tiles.TileRequest(z=z, x=x, y=y)
    try:
        png = tiles.render(dataset.info.path, request, dataset.info.dtype)
    except tiles.TileOutOfRange:
        # A transparent pixel is what Leaflet expects outside the footprint.
        return Response(content=_EMPTY_PNG, media_type="image/png")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Tile render failed: {exc}") from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


_EMPTY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


# ─────────────────────────────────────────────────────────── detections ──
@router.post("/detections/csv")
async def upload_detections(
    file: UploadFile = File(...),
    source_crs: str = Form("EPSG:32618"),
    dataset_id: str = Form(""),
) -> dict:
    raw = await file.read()
    try:
        frame, report = load_detections(raw, source_crs=source_crs)
    except CsvIngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = store.add_detections(
        frame,
        source="csv",
        dataset_id=dataset_id or None,
        stats={"kept": int(len(frame)), **summarise(frame)},
        note=file.filename or "uploaded.csv",
    )

    # A table exported at conf 0.1 loses most of its rows to a 0.25 display
    # filter. Hand the UI the distribution so it can say so instead of just
    # showing a smaller number than the file contains.
    conf = frame["confidence"].to_numpy(dtype="float64")
    report["confidence"] = {
        "min": round(float(conf.min()), 4),
        "median": round(float(np.median(conf)), 4),
        "max": round(float(conf.max()), 4),
    }
    return {**record.as_dict(), "summary": summarise(frame), "import": report}


@router.get("/detections")
def list_detections() -> dict:
    return {"detections": [d.as_dict() for d in store.detection_sets()]}


@router.get("/detections/{detection_id}")
def get_detections(detection_id: str, min_conf: float = 0.0) -> dict:
    record = _detections_or_404(detection_id)
    frame = _filter_conf(record.frame, min_conf)
    return {
        **record.as_dict(),
        "summary": summarise(frame),
        "labels": (
            frame["label"].astype(str).value_counts().head(12).to_dict() if not frame.empty else {}
        ),
    }


@router.delete("/detections/{detection_id}")
def delete_detections(detection_id: str) -> dict:
    return {"deleted": store.drop_detections(detection_id)}


@router.get("/detections/{detection_id}/points")
def detection_points(detection_id: str, min_conf: float = 0.0) -> Response:
    """The binary point payload the map's canvas layer renders."""
    record = _detections_or_404(detection_id)
    return Response(
        content=pack_points(record.frame, min_conf),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/detections/{detection_id}/export.csv")
def export_detections(detection_id: str, min_conf: float = 0.0) -> Response:
    record = _detections_or_404(detection_id)
    frame = _filter_conf(record.frame, min_conf)
    return _csv_response(exports.detections_csv(frame), "detections.csv")


@router.post("/detections/{detection_id}/aoi")
def aoi_counts(detection_id: str, payload: AoiRequest) -> dict:
    """Counts and density inside the most recently drawn polygon."""
    record = _detections_or_404(detection_id)
    geom = parse_aoi(payload.aoi)
    if geom is None:
        raise HTTPException(status_code=400, detail="No polygon found in the AOI payload.")

    frame = _filter_conf(record.frame, payload.min_conf)
    inside = _subset(frame, geom)
    counts = summarise(inside)

    # Round the area first and derive the densities from that reported value,
    # so the figures on screen divide out exactly when someone checks them.
    area = {"area_m2": round(area_metrics(geom)["area_m2"], 2)}
    area["area_ha"] = round(area["area_m2"] / M2_PER_HA, 6)
    area["area_acres"] = round(area["area_m2"] / M2_PER_ACRE, 6)
    acres = area["area_acres"]
    hectares = area["area_ha"]

    return {
        **counts,
        **area,
        "plants_per_acre": round_density(counts["plants"] / acres) if acres > 0 else 0.0,
        "weeds_per_acre": round_density(counts["weeds"] / acres) if acres > 0 else 0.0,
        "plants_per_ha": round_density(counts["plants"] / hectares) if hectares > 0 else 0.0,
        "weeds_per_ha": round_density(counts["weeds"] / hectares) if hectares > 0 else 0.0,
    }


@router.post("/detections/{detection_id}/aoi/export.csv")
def export_aoi(detection_id: str, payload: AoiRequest) -> Response:
    record = _detections_or_404(detection_id)
    geom = parse_aoi(payload.aoi)
    if geom is None:
        raise HTTPException(status_code=400, detail="No polygon found in the AOI payload.")
    inside = _subset(_filter_conf(record.frame, payload.min_conf), geom)
    return _csv_response(exports.detections_csv(inside), "aoi_detections.csv")


# ───────────────────────────────────────────────────────────── analysis ──
def _analysis(detection_id: str, payload: AnalysisRequest):
    record = _detections_or_404(detection_id)
    frame = _filter_conf(record.frame, payload.min_conf)

    geom = parse_aoi(payload.aoi) if payload.scope == "aoi" else None
    if payload.scope == "aoi":
        if geom is None:
            raise HTTPException(
                status_code=400,
                detail="Draw an area of interest on the map before scoping the analysis to it.",
            )
        frame = _subset(frame, geom)
        area_acres = area_metrics(geom)["area_acres"]
        scope_label = "Drawn AOI"
    else:
        area_acres = fallback_area_acres(frame)
        scope_label = "All detections"

    geometry = RowGeometry()
    if payload.basis == "rows":
        geometry = RowGeometry(
            rows=int(payload.geometry.rows),
            row_length_ft=float(payload.geometry.row_length_ft),
            row_spacing_in=float(payload.geometry.row_spacing_in),
        )
        if not geometry.is_complete:
            raise HTTPException(
                status_code=400,
                detail="Row geometry needs a row count, a row length and a row spacing above zero.",
            )

    metrics = stand_metrics(
        frame,
        target_plants_acre=payload.target_plants_acre,
        alert_pct=payload.alert_pct,
        area_acres=area_acres,
        geometry=geometry,
    )

    # Row geometry describes the plot the selection is supposed to cover. When
    # the two disagree by a wide margin the density is being computed over the
    # wrong footprint, and the stand reads far above or below target for no
    # agronomic reason. Say so rather than reporting a confident wrong number.
    warning = ""
    if metrics.basis == "rows" and area_acres > 0 and metrics.area_acres > 0:
        ratio = area_acres / metrics.area_acres
        if ratio > 3 or ratio < 1 / 3:
            warning = (
                f"The selected area measures {area_acres:.4f} acres but the row geometry "
                f"describes {metrics.area_acres:.4f} acres. Densities use the row geometry, "
                "so check that the rows, row length and spacing match the area you selected."
            )
    return frame, metrics, scope_label, warning


@router.post("/detections/{detection_id}/analysis")
def analysis(detection_id: str, payload: AnalysisRequest) -> dict:
    frame, metrics, scope_label, warning = _analysis(detection_id, payload)
    return {
        "scope": scope_label,
        "warning": warning,
        "metrics": metrics.as_dict(),
        "status_detail": status_detail(metrics),
        "rows": rows_table(metrics),
        "confidence_bands": confidence_bands(frame),
        "class_confidence": class_confidence(frame),
        "area_note": (
            "Plot area = rows x row length x row spacing."
            if metrics.basis == "rows"
            else "Density uses the mapped area of the selected scope."
        ),
    }


@router.post("/detections/{detection_id}/exports/{kind}")
def analysis_export(detection_id: str, kind: str, payload: AnalysisRequest) -> Response:
    if kind not in {"stand", "plots", "rows"}:
        raise HTTPException(status_code=404, detail="Unknown export.")
    _, metrics, scope_label, _warning = _analysis(detection_id, payload)
    if kind == "stand":
        return _csv_response(
            exports.stand_csv(metrics, scope_label, payload.plot_id), "stand_summary.csv"
        )
    if kind == "plots":
        return _csv_response(exports.plots_csv(metrics, payload.plot_id, scope_label), "plots.csv")
    if not metrics.geometry.is_complete:
        raise HTTPException(
            status_code=400, detail="rows.csv needs the plot row geometry basis."
        )
    return _csv_response(exports.rows_csv(metrics, payload.plot_id), "rows.csv")


@router.post("/detections/{detection_id}/report.pdf")
def report(detection_id: str, payload: ReportRequest) -> Response:
    frame, metrics, scope_label, _warning = _analysis(detection_id, payload)
    try:
        pdf = build_report(
            frame,
            metrics,
            dataset_name=payload.dataset_name,
            plot_id=payload.plot_id,
            scope=scope_label,
            min_conf=payload.min_conf,
            rows=rows_table(metrics),
        )
    except ReportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="plantai_report.pdf"'},
    )


# ────────────────────────────────────────────────────────────── detect ──
@router.post("/detect")
def start_detection(payload: DetectRequest) -> dict:
    dataset = _dataset_or_404(payload.dataset_id)
    status = detector.status()
    if not status.loaded:
        status = detector.load()
    if not status.loaded:
        raise HTTPException(
            status_code=400,
            detail=status.error or "No detection model is loaded. Choose a .pt checkpoint first.",
        )

    params = DetectionParams(
        tile_size=payload.tile_size,
        conf=payload.conf,
        iou=payload.iou,
        side_overlap=payload.side_overlap,
        front_overlap=payload.front_overlap,
        dedup_radius_m=payload.dedup_radius_m,
    )

    region = None
    geom = None
    if payload.mode == "aoi":
        geom = parse_aoi(payload.aoi)
        if geom is None:
            raise HTTPException(
                status_code=400,
                detail="Draw a polygon on the map before running a quick count.",
            )
        region = region_for_geometry(dataset.info, geom)
        if region[2] <= region[0] or region[3] <= region[1]:
            raise HTTPException(
                status_code=400, detail="The AOI does not overlap this orthomosaic."
            )

    job = jobs.submit(dataset.info, dataset.id, params, payload.mode, region=region, aoi=geom)
    return job.snapshot()


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job.snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return {"cancelled": jobs.cancel(job_id)}


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    """Server-sent progress for a running detection."""
    if jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job.")

    def stream():
        for snapshot in jobs.stream(job_id):
            yield f"data: {json.dumps(snapshot)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ────────────────────────────────────────────────────────────── images ──
@router.post("/images/detect")
async def detect_images(
    files: list[UploadFile] = File(...),
    conf: float = Form(settings.conf),
    iou: float = Form(settings.iou),
) -> dict:
    status = detector.status()
    if not status.loaded:
        status = detector.load()
    if not status.loaded:
        raise HTTPException(status_code=400, detail=status.error or "No detection model is loaded.")

    import base64
    import io as _io

    from PIL import Image

    results = []
    for upload in files:
        raw = await upload.read()
        try:
            image = np.asarray(Image.open(_io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            results.append({"name": upload.filename, "error": f"Unreadable image: {exc}"})
            continue

        try:
            preview, rows = detector.annotate(image, conf=conf, iou=iou)
        except ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plants = sum(1 for r in rows if r["role"] == "plant")
        weeds = sum(1 for r in rows if r["role"] == "weed")
        total = len(rows)

        buffer = _io.BytesIO()
        Image.fromarray(preview).save(buffer, format="JPEG", quality=80)
        results.append(
            {
                "name": upload.filename,
                "plants": plants,
                "weeds": weeds,
                "total": total,
                "weed_pressure_pct": round(100.0 * weeds / total, 1) if total else 0.0,
                "avg_confidence": (
                    round(float(np.mean([r["confidence"] for r in rows])) * 100, 1) if rows else 0.0
                ),
                "preview": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
            }
        )
    return {"results": results}


# ─────────────────────────────────────────────────────────────── shared ──
def _dataset_or_404(dataset_id: str):
    dataset = store.dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset. Load an orthomosaic first.")
    return dataset


def _detections_or_404(detection_id: str):
    record = store.detections(detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown detection set.")
    return record


def _filter_conf(frame: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    if frame.empty or min_conf <= 0:
        return frame
    keep = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0) >= float(min_conf)
    return frame[keep.to_numpy()].reset_index(drop=True)


def _subset(frame: pd.DataFrame, geom) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = points_within(geom, frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    return frame[mask].reset_index(drop=True)


def _csv_response(payload: bytes, filename: str) -> Response:
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
