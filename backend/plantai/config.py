"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    """Every knob the service exposes. Instantiated once as ``settings``."""

    # --- storage -------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "/tmp/plantai")).expanduser()
    )

    # --- model ---------------------------------------------------------
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "").strip())
    # Directories the UI is allowed to browse for .pt / .tif files.
    library_dirs: list[str] = field(
        default_factory=lambda: _env_list("LIBRARY_DIRS", "/models,/data")
    )
    # Allow the UI to register a GeoTIFF that already sits on the server's disk.
    allow_local_paths: bool = field(
        default_factory=lambda: _env_bool("ALLOW_LOCAL_PATHS", True)
    )

    # --- inference defaults --------------------------------------------
    tile_size: int = field(default_factory=lambda: _env_int("TILE_SIZE", 1280))
    batch_tiles: int = field(default_factory=lambda: _env_int("BATCH_TILES", 4))
    conf: float = field(default_factory=lambda: _env_float("YOLO_CONF", 0.25))
    iou: float = field(default_factory=lambda: _env_float("YOLO_IOU", 0.45))
    dedup_radius_m: float = field(default_factory=lambda: _env_float("DEDUP_RADIUS_M", 0.10))
    side_overlap: float = field(default_factory=lambda: _env_float("SIDE_OVERLAP", 0.30))
    front_overlap: float = field(default_factory=lambda: _env_float("FRONT_OVERLAP", 0.30))

    # --- class semantics ------------------------------------------------
    # Names are matched case-insensitively against the checkpoint's labels.
    plant_classes: list[str] = field(
        default_factory=lambda: _env_list("PLANT_CLASSES", "plant,crop,soybean,faba,fababean")
    )
    weed_classes: list[str] = field(
        default_factory=lambda: _env_list("WEED_CLASSES", "weed,palmer,ragweed,grass")
    )

    # --- agronomy defaults ----------------------------------------------
    stand_target: int = field(default_factory=lambda: _env_int("STAND_TARGET", 150_000))
    stand_alert_pct: int = field(default_factory=lambda: _env_int("STAND_ALERT_PCT", 85))

    # --- http ------------------------------------------------------------
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
        )
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 8192))
    serve_frontend: bool = field(default_factory=lambda: _env_bool("SERVE_FRONTEND", True))

    # --- tile rendering ---------------------------------------------------
    tile_px: int = field(default_factory=lambda: _env_int("MAP_TILE_PX", 256))
    stretch_lo: float = field(default_factory=lambda: _env_float("STRETCH_LO", 2.0))
    stretch_hi: float = field(default_factory=lambda: _env_float("STRETCH_HI", 98.0))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.sessions_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
