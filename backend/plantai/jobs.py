"""Background detection runs with progress the UI can subscribe to.

Inference over a full mosaic takes minutes. Running it inside the request
would block the worker and time the browser out, so a run becomes a job the
frontend follows over server-sent events.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue

import pandas as pd

from .detect import DetectionParams, RunStats, run_detection
from .model import detector
from .raster import RasterInfo
from .store import store

STEPS = [
    ("tile", "Tiling raster window"),
    ("infer", "Model inference"),
    ("dedupe", "Removing duplicates"),
    ("project", "Converting to map coordinates"),
]
STEP_INDEX = {key: i for i, (key, _) in enumerate(STEPS)}


@dataclass
class Job:
    """One detection run and everything the UI shows about it."""

    id: str
    dataset_id: str
    mode: str  # "field" | "aoi"
    state: str = "queued"  # queued | running | done | error | cancelled
    step: str = "tile"
    progress: float = 0.0
    steps: dict = field(default_factory=lambda: {key: {"progress": 0.0, "note": "queued"} for key, _ in STEPS})
    stats: dict = field(default_factory=dict)
    detection_id: str | None = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _events: Queue = field(default_factory=Queue, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "mode": self.mode,
            "state": self.state,
            "step": self.step,
            "progress": round(self.progress, 4),
            "steps": [
                {
                    "key": key,
                    "label": label,
                    "progress": round(self.steps[key]["progress"], 4),
                    "note": self.steps[key]["note"],
                }
                for key, label in STEPS
            ],
            "stats": self.stats,
            "detection_id": self.detection_id,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 2),
        }

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "error", "cancelled"}


class JobManager:
    """Runs one detection at a time and fans progress out to listeners."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._order[-1]) if self._order else None

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.is_terminal:
            return False
        job._cancel.set()
        return True

    def submit(
        self,
        info: RasterInfo,
        dataset_id: str,
        params: DetectionParams,
        mode: str,
        region: tuple[int, int, int, int] | None = None,
        aoi=None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], dataset_id=dataset_id, mode=mode)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Keep a short history so the UI can still read a finished run.
            for stale in self._order[:-6]:
                self._jobs.pop(stale, None)
            self._order = self._order[-6:]

        thread = threading.Thread(
            target=self._run,
            args=(job, info, params, region, aoi),
            name=f"plantai-detect-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    # ------------------------------------------------------------- internal
    def _publish(self, job: Job) -> None:
        job._events.put(job.snapshot())

    def _run(
        self,
        job: Job,
        info: RasterInfo,
        params: DetectionParams,
        region: tuple[int, int, int, int] | None,
        aoi,
    ) -> None:
        job.state = "running"
        self._publish(job)

        def progress(step: str, frac: float, note: str) -> None:
            job.step = step
            job.steps[step] = {"progress": frac, "note": note or ""}
            # Earlier steps are implicitly complete once a later one starts.
            for key, _ in STEPS[: STEP_INDEX.get(step, 0)]:
                if job.steps[key]["progress"] < 1.0:
                    job.steps[key] = {"progress": 1.0, "note": job.steps[key]["note"]}
            done = sum(job.steps[key]["progress"] for key, _ in STEPS)
            job.progress = done / len(STEPS)
            self._publish(job)

        try:
            frame, stats = run_detection(
                info,
                detector,
                params,
                region=region,
                progress=progress,
                should_stop=job._cancel.is_set,
            )

            if job._cancel.is_set():
                job.state = "cancelled"
                job.error = "Cancelled before completion."
            else:
                if aoi is not None and not frame.empty:
                    # A tiled window is rectangular; clip back to the drawn
                    # shape so an angled plot does not pick up its neighbours.
                    from .geometry import points_within

                    mask = points_within(
                        aoi, frame["longitude"].to_numpy(), frame["latitude"].to_numpy()
                    )
                    frame = frame[mask].reset_index(drop=True)
                    stats.kept = int(len(frame))
                    stats.plants = int((frame["class"] == "plant").sum())
                    stats.weeds = int((frame["class"] == "weed").sum())

                record = store.add_detections(
                    frame,
                    source=job.mode,
                    dataset_id=job.dataset_id,
                    stats=stats.as_dict(),
                    note=f"{job.mode} run on {info.path}",
                )
                job.detection_id = record.id
                job.stats = stats.as_dict()
                job.state = "done"
                job.progress = 1.0
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished = time.time()
            self._publish(job)
            job._events.put(None)  # sentinel: stream is closed

    def stream(self, job_id: str, timeout: float = 1.0):
        """Yield snapshots until the job finishes; used by the SSE endpoint."""
        job = self.get(job_id)
        if job is None:
            return
        yield job.snapshot()
        while True:
            try:
                item = job._events.get(timeout=timeout)
            except Empty:
                if job.is_terminal:
                    return
                yield job.snapshot()  # heartbeat keeps proxies from closing us
                continue
            if item is None:
                return
            yield item


jobs = JobManager()
