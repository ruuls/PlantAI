"""XYZ map tiles rendered straight from the orthomosaic.

Serving the raster as web-mercator tiles is what lets a multi-gigabyte mosaic
appear under the detections without converting it first: rio-tiler reads only
the pixels a tile needs, using the overview pyramid when one exists.

Readers are held open because re-opening the dataset per tile is the single
biggest source of pan/zoom stutter — but they are held *per thread*. A GDAL
dataset handle is not thread-safe, and the browser opens six connections at
once, so a shared handle returns torn blocks (`TIFFReadEncodedTile failed`,
`ZIPDecode` errors) rather than an honest exception.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader

from .config import settings
from .raster import percentile_stretch

_local = threading.local()
_lock = threading.Lock()
_stretches: dict[str, list[tuple[float, float]]] = {}
# Bumped whenever a raster is invalidated, so every thread drops its handle
# on next use instead of reading a file that has been rewritten underneath it.
_generation: dict[str, int] = {}


@dataclass(frozen=True)
class TileRequest:
    z: int
    x: int
    y: int


class TileOutOfRange(Exception):
    """The requested tile does not intersect the raster."""


def _readers() -> dict[str, tuple[int, Reader]]:
    cache = getattr(_local, "readers", None)
    if cache is None:
        cache = {}
        _local.readers = cache
    return cache


def _reader(path: str) -> Reader:
    with _lock:
        generation = _generation.setdefault(path, 0)

    cache = _readers()
    entry = cache.get(path)
    if entry is not None:
        cached_generation, reader = entry
        if cached_generation == generation:
            return reader
        _safe_close(reader)
        cache.pop(path, None)

    reader = Reader(path)
    cache[path] = (generation, reader)
    return reader


def _safe_close(reader: Reader | None) -> None:
    if reader is None:
        return
    try:
        reader.close()
    except Exception:
        pass


def _stretch(path: str, dtype: str) -> list[tuple[float, float]] | None:
    """8-bit mosaics are already display-ready; anything else needs scaling."""
    if dtype == "uint8":
        return None
    with _lock:
        cached = _stretches.get(path)
    if cached is None:
        cached = percentile_stretch(path)
        with _lock:
            _stretches[path] = cached
    return cached


def close(path: str) -> None:
    """Invalidate a raster's readers, e.g. before rewriting it in place.

    Only this thread's handle can be closed directly; the generation bump makes
    every other thread discard its own on the next request.
    """
    with _lock:
        _generation[path] = _generation.get(path, 0) + 1
        _stretches.pop(path, None)

    cache = _readers()
    entry = cache.pop(path, None)
    if entry is not None:
        _safe_close(entry[1])


def close_all() -> None:
    with _lock:
        paths = list(_generation)
    for path in paths:
        close(path)

    cache = _readers()
    for _, reader in list(cache.values()):
        _safe_close(reader)
    cache.clear()


def render(path: str, req: TileRequest, dtype: str = "uint8") -> bytes:
    """Render one PNG tile, with an alpha mask everywhere the raster has no data."""
    reader = _reader(path)
    try:
        img = reader.tile(
            req.x,
            req.y,
            req.z,
            indexes=(1, 2, 3),
            tilesize=settings.tile_px,
        )
    except TileOutsideBounds as exc:
        raise TileOutOfRange(str(exc)) from exc

    stretch = _stretch(path, dtype)
    if stretch is not None:
        img.rescale(in_range=tuple(stretch), out_range=((0, 255),) * 3)
    elif img.data.dtype != np.uint8:
        img.rescale(in_range=((0, 255),) * 3, out_range=((0, 255),) * 3)

    return img.render(img_format="PNG", add_mask=True)


def zoom_range(path: str) -> tuple[int, int]:
    """Native zoom span, used to cap the map's zoom controls."""
    reader = _reader(path)
    return int(reader.minzoom), int(reader.maxzoom)
