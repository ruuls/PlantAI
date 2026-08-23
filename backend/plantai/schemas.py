"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import settings


class LoadModelRequest(BaseModel):
    path: str = Field(..., description="Absolute path to an Ultralytics .pt checkpoint")


class RegisterPathRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a GeoTIFF on the server")


class DetectRequest(BaseModel):
    dataset_id: str
    mode: Literal["field", "aoi"] = "field"
    aoi: Any | None = Field(default=None, description="GeoJSON for the AOI when mode is 'aoi'")
    tile_size: int = settings.tile_size
    conf: float = settings.conf
    iou: float = settings.iou
    side_overlap: float = settings.side_overlap
    front_overlap: float = settings.front_overlap
    dedup_radius_m: float = settings.dedup_radius_m


class AoiRequest(BaseModel):
    aoi: Any
    min_conf: float = 0.0


class RowGeometryModel(BaseModel):
    rows: int = 0
    row_length_ft: float = 0.0
    row_spacing_in: float = 0.0


class AnalysisRequest(BaseModel):
    scope: Literal["aoi", "all"] = "all"
    aoi: Any | None = None
    min_conf: float = 0.0
    basis: Literal["area", "rows"] = "area"
    geometry: RowGeometryModel = Field(default_factory=RowGeometryModel)
    target_plants_acre: float = float(settings.stand_target)
    alert_pct: float = float(settings.stand_alert_pct)
    plot_id: str = "plot_001"


class ReportRequest(AnalysisRequest):
    dataset_name: str = ""
