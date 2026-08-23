"""PlantAI service entry point.

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from plantai import __version__, tiles
from plantai.api import router
from plantai.config import settings
from plantai.model import detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("plantai")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    if settings.model_path:
        status = detector.load()
        if status.loaded:
            log.info("Model loaded: %s (%s)", status.name, ", ".join(status.class_names or []))
        else:
            log.warning("Model not loaded: %s", status.error)
    else:
        log.info("No MODEL_PATH set — CSV workflows are available, detection needs a checkpoint.")
    yield
    tiles.close_all()


app = FastAPI(
    title="PlantAI",
    version=__version__,
    description="Plant and weed detection on georeferenced orthomosaics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/healthz", include_in_schema=False)
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


if settings.serve_frontend and FRONTEND_DIR.is_dir():
    # One origin for the API and the console keeps deployment to a single
    # container; the separate nginx image stays available for split hosting.
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIR),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    for _static in ("css", "js", "vendor"):
        _path = FRONTEND_DIR / _static
        if _path.is_dir():
            app.mount(f"/{_static}", StaticFiles(directory=_path), name=_static)
