"""In-process registry of the rasters and detection sets a session is using.

State lives on the server because the alternative — shipping several hundred
thousand detections back and forth as JSON — is what makes browser dashboards
of this kind fall over.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings
from .raster import RasterInfo, describe


@dataclass
class Dataset:
    """A registered orthomosaic."""

    id: str
    info: RasterInfo
    name: str
    origin: str  # "upload" | "path"
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        payload = self.info.as_dict()
        payload.update({"id": self.id, "name": self.name, "origin": self.origin})
        return payload


@dataclass
class DetectionSet:
    """A table of detections plus how it was produced."""

    id: str
    frame: pd.DataFrame
    source: str  # "csv" | "field" | "aoi"
    dataset_id: str | None = None
    stats: dict = field(default_factory=dict)
    note: str = ""
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "dataset_id": self.dataset_id,
            "count": int(len(self.frame)),
            "stats": self.stats,
            "note": self.note,
            "created": self.created,
        }


class Store:
    """Thread-safe, bounded, process-local storage."""

    def __init__(self, max_datasets: int = 8, max_detection_sets: int = 12) -> None:
        self._lock = threading.RLock()
        self._datasets: dict[str, Dataset] = {}
        self._detections: dict[str, DetectionSet] = {}
        self._max_datasets = max_datasets
        self._max_detection_sets = max_detection_sets

    # -------------------------------------------------------------- rasters
    def add_dataset(self, path: str | Path, name: str, origin: str) -> Dataset:
        info = describe(path)
        dataset = Dataset(id=uuid.uuid4().hex[:12], info=info, name=name, origin=origin)
        with self._lock:
            self._datasets[dataset.id] = dataset
            self._evict(self._datasets, self._max_datasets, self._drop_dataset)
        return dataset

    def refresh_dataset(self, dataset_id: str) -> Dataset:
        """Re-read a raster's header, e.g. after overviews were built."""
        with self._lock:
            current = self._datasets[dataset_id]
            refreshed = Dataset(
                id=current.id,
                info=describe(current.info.path),
                name=current.name,
                origin=current.origin,
                created=current.created,
            )
            self._datasets[dataset_id] = refreshed
        return refreshed

    def dataset(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def datasets(self) -> list[Dataset]:
        with self._lock:
            return sorted(self._datasets.values(), key=lambda d: d.created, reverse=True)

    def _drop_dataset(self, dataset: Dataset) -> None:
        from . import tiles

        tiles.close(dataset.info.path)
        # Only files this service copied in are ours to delete; a raster the
        # user pointed at by path stays exactly where it is.
        if dataset.origin == "upload":
            try:
                Path(dataset.info.path).unlink(missing_ok=True)
            except OSError:
                pass

    # ----------------------------------------------------------- detections
    def add_detections(
        self,
        frame: pd.DataFrame,
        source: str,
        dataset_id: str | None = None,
        stats: dict | None = None,
        note: str = "",
    ) -> DetectionSet:
        record = DetectionSet(
            id=uuid.uuid4().hex[:12],
            frame=frame.reset_index(drop=True),
            source=source,
            dataset_id=dataset_id,
            stats=stats or {},
            note=note,
        )
        with self._lock:
            self._detections[record.id] = record
            self._evict(self._detections, self._max_detection_sets, lambda _r: None)
        return record

    def detections(self, detection_id: str) -> DetectionSet | None:
        with self._lock:
            return self._detections.get(detection_id)

    def detection_sets(self) -> list[DetectionSet]:
        with self._lock:
            return sorted(self._detections.values(), key=lambda d: d.created, reverse=True)

    def drop_detections(self, detection_id: str) -> bool:
        with self._lock:
            return self._detections.pop(detection_id, None) is not None

    # --------------------------------------------------------------- helper
    @staticmethod
    def _evict(bucket: dict, limit: int, on_drop) -> None:
        """Drop the oldest entries once the bucket exceeds ``limit``."""
        if len(bucket) <= limit:
            return
        ordered = sorted(bucket.values(), key=lambda item: item.created)
        for stale in ordered[: len(bucket) - limit]:
            bucket.pop(stale.id, None)
            on_drop(stale)


store = Store()


def pack_points(df: pd.DataFrame, min_conf: float = 0.0) -> bytes:
    """Serialise detections into the compact binary the map layer decodes.

    JSON would cost roughly 60 bytes per detection and a parse pause the user
    can feel; this is 13 bytes and decodes straight into typed arrays.

    Layout (little-endian)::

        magic  'PAI1'   4 bytes
        count  uint32   4 bytes
        ref    2x float64  reference lon/lat
        lon    float32 * count   degrees offset from ref
        lat    float32 * count   degrees offset from ref
        conf   float32 * count
        cls    uint8   * count   0 = plant, 1 = weed
    """
    if df is None or df.empty:
        return b"PAI1" + np.uint32(0).tobytes() + np.array([0.0, 0.0], dtype="<f8").tobytes()

    frame = df
    if min_conf > 0:
        frame = frame[frame["confidence"].to_numpy(dtype="float64") >= float(min_conf)]
    if frame.empty:
        return b"PAI1" + np.uint32(0).tobytes() + np.array([0.0, 0.0], dtype="<f8").tobytes()

    lon = frame["longitude"].to_numpy(dtype="float64")
    lat = frame["latitude"].to_numpy(dtype="float64")
    ref_lon = float(lon.mean())
    ref_lat = float(lat.mean())

    conf = frame["confidence"].to_numpy(dtype="float32")
    cls = (frame["class"].astype(str).str.lower().to_numpy() != "plant").astype("uint8")

    return b"".join(
        [
            b"PAI1",
            np.uint32(len(frame)).tobytes(),
            np.array([ref_lon, ref_lat], dtype="<f8").tobytes(),
            (lon - ref_lon).astype("<f4").tobytes(),
            (lat - ref_lat).astype("<f4").tobytes(),
            conf.astype("<f4").tobytes(),
            cls.tobytes(),
        ]
    )
