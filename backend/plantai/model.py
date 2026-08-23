"""YOLO checkpoint loading and class-label semantics.

The detector is optional: the CSV workflows and the map work with no weights
present, so a missing model degrades one feature rather than the whole app.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings

PLANT = "plant"
WEED = "weed"


class ModelError(RuntimeError):
    """Raised when inference is requested but no usable checkpoint is loaded."""


@dataclass
class ModelStatus:
    loaded: bool
    path: str = ""
    name: str = ""
    class_names: list[str] | None = None
    device: str = "cpu"
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "path": self.path,
            "name": self.name,
            "class_names": self.class_names or [],
            "device": self.device,
            "error": self.error,
        }


def classify_label(label: str) -> str:
    """Map a checkpoint label onto the two roles the dashboard reasons about.

    Anything not explicitly listed as a crop is treated as a weed: an
    unrecognised label is far more likely to be a weed species than the crop,
    and over-counting weeds is the safer error for a stand-count decision.
    """
    norm = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    for name in settings.plant_classes:
        if norm == name.strip().lower():
            return PLANT
    for name in settings.weed_classes:
        if norm == name.strip().lower():
            return WEED
    return WEED


class Detector:
    """Thread-safe lazy wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._path = ""
        self._names: list[str] = []
        self._device = "cpu"
        self._error = ""

    # ---------------------------------------------------------------- state
    @property
    def path(self) -> str:
        return self._path

    def status(self) -> ModelStatus:
        if self._model is None:
            return ModelStatus(
                loaded=False,
                path=self._path or settings.model_path,
                error=self._error,
            )
        return ModelStatus(
            loaded=True,
            path=self._path,
            name=Path(self._path).name,
            class_names=list(self._names),
            device=self._device,
        )

    # ---------------------------------------------------------------- load
    def load(self, path: str | None = None) -> ModelStatus:
        """Load (or reload) weights. Idempotent for an unchanged path."""
        target = str(path or settings.model_path or "").strip()
        if not target:
            self._error = "No model path configured. Set MODEL_PATH or pick a .pt file."
            return self.status()

        resolved = str(Path(target).expanduser())
        if not Path(resolved).is_file():
            self._error = f"Model file not found: {resolved}"
            return self.status()

        with self._lock:
            if self._model is not None and self._path == resolved:
                return self.status()
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                self._error = (
                    "ultralytics is not installed in this environment "
                    f"({exc}). Install it to run detection."
                )
                return self.status()

            try:
                model = YOLO(resolved)
                names = getattr(model, "names", None)
                if isinstance(names, dict):
                    labels = [str(names[k]) for k in sorted(names)]
                elif names:
                    labels = [str(n) for n in names]
                else:
                    labels = []

                device = "cpu"
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                        model.to(device)
                except ImportError:
                    pass

                self._model = model
                self._path = resolved
                self._names = labels
                self._device = device
                self._error = ""
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
                self._model = None
                self._error = f"Could not load checkpoint: {exc}"
        return self.status()

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._path = ""
            self._names = []

    # ----------------------------------------------------------- inference
    def predict_batch(
        self,
        images: list[np.ndarray],
        conf: float,
        iou: float,
        imgsz: int,
    ) -> list[list[tuple[int, float, float, float]]]:
        """Run a batch of tiles.

        Returns, per image, a list of ``(class_id, confidence, x_px, y_px)``
        where the coordinates are box centres in that tile's pixel space.
        """
        if self._model is None:
            raise ModelError(self._error or "No model loaded.")
        if not images:
            return []

        results = self._model.predict(
            images,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            verbose=False,
            device=self._device,
        )

        batch: list[list[tuple[int, float, float, float]]] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                batch.append([])
                continue
            xywh = boxes.xywh.cpu().numpy().astype("float64")
            cls = boxes.cls.cpu().numpy().astype("int64")
            confs = boxes.conf.cpu().numpy().astype("float64").ravel()
            batch.append(
                [
                    (int(cls[i]), float(confs[i]), float(xywh[i, 0]), float(xywh[i, 1]))
                    for i in range(len(cls))
                ]
            )
        return batch

    def label_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self._names):
            return self._names[class_id]
        return f"class_{class_id}"

    def annotate(self, image: np.ndarray, conf: float, iou: float) -> tuple[np.ndarray, list[dict]]:
        """Single-image inference returning a rendered preview and rows."""
        if self._model is None:
            raise ModelError(self._error or "No model loaded.")
        res = self._model.predict(
            image, conf=float(conf), iou=float(iou), verbose=False, device=self._device
        )[0]
        rows: list[dict] = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype("int64")
            confs = boxes.conf.cpu().numpy().astype("float64").ravel()
            xyxy = boxes.xyxy.cpu().numpy().astype("float64")
            for i in range(len(cls)):
                label = self.label_for(int(cls[i]))
                rows.append(
                    {
                        "label": label,
                        "role": classify_label(label),
                        "confidence": float(confs[i]),
                        "bbox": [float(v) for v in xyxy[i]],
                    }
                )
        return res.plot()[:, :, ::-1], rows  # ultralytics renders BGR


detector = Detector()
