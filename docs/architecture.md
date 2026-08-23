# Architecture notes

Three problems shaped this codebase. Each has an obvious solution that does not
survive a real field, so the reasoning is recorded here.

## Rendering several hundred thousand detections

A full-field run over a UAV mosaic returns 10⁵–10⁶ detections. The obvious
approach — one Leaflet marker per detection, or a clustering plugin — creates
that many DOM nodes and locks the browser up long before the field is drawn.

`frontend/js/map.js` draws them onto a canvas overlay instead:

- points live in typed arrays (`Float64Array` for coordinates, `Float32Array`
  for confidence, `Uint8Array` for class), never as objects;
- each repaint culls to the viewport plus a 15% margin before projecting
  anything;
- each marker is one `drawImage` of a pre-rendered sprite rather than an
  `arc()` + `fill()` + `stroke()`;
- a hard ceiling of 250k drawn points keeps a pathological zoom-out from
  stalling a frame.

The confidence slider filters in the same loop, so it responds without a
network round trip.

## Moving the detections to the browser

JSON costs roughly 60 bytes per detection and a parse pause the user can feel.
`GET /api/detections/{id}/points` returns a binary payload instead:

```
magic  'PAI1'          4 bytes
count  uint32          4 bytes
ref    2 x float64    16 bytes   reference lon/lat
lon    float32 x count            degrees offset from ref
lat    float32 x count            degrees offset from ref
conf   float32 x count
cls    uint8   x count            0 = plant, 1 = weed
```

About 13 bytes per detection, decoding straight into typed arrays.

Storing coordinates as raw float32 would be a mistake — float32 carries about
seven significant digits, so latitude 36.7284 lands on a ~1 m grid, which is
larger than the plants being counted. Encoding them as *offsets from a float64
reference* spends the precision on the field's own extent instead: measured
error is under 10⁻⁹ degrees, roughly a tenth of a millimetre.

`tests/test_units.py::test_pack_points_layout_and_precision` pins this.

## Serving map tiles safely

`plantai/tiles.py` keeps rio-tiler `Reader` objects open, because re-opening a
multi-gigabyte GeoTIFF per tile is the single biggest source of pan/zoom
stutter.

Keeping *one shared* reader is a trap. GDAL dataset handles are not
thread-safe, FastAPI runs sync endpoints in a threadpool, and a browser opens
six connections at once. A shared handle does not raise — it returns torn
blocks, surfacing as `TIFFReadEncodedTile() failed` and `ZIPDecode` errors and
intermittent 500s under exactly the load that matters.

So readers are **thread-local**, with a per-path generation counter: bumping it
makes every thread discard its handle on next use, which is what makes it safe
to rewrite a raster in place when building overviews.

`tests/test_api.py::test_tiles_are_safe_under_concurrent_requests` runs 24
overlapping tile requests across 12 threads.

## Smaller decisions

**Only the newest AOI counts.** `geometry.parse_aoi` takes the last polygon in
the payload rather than unioning them. The map hands back every shape still on
the canvas, and unioning lets a polygon the user has forgotten about silently
inflate the next plot's numbers.

**Deduplication is per class, on a grid.** Greedy non-max suppression in the
projected plane, highest confidence first, with a spatial hash of `radius`-sized
cells bounding each lookup to nine buckets — near-linear rather than the O(n²)
the radius test implies (200k points in well under a second). Classes are
suppressed independently, because a weed growing beside a crop plant is a
second detection, not a duplicate of it.

**Unknown labels are weeds.** `model.classify_label` maps a checkpoint's labels
onto the two roles the dashboard reasons about. A label in neither list is
treated as a weed: an unrecognised species is far more likely to be a weed than
the crop, and for a stand-count decision over-counting weeds is the safer
error.

**Density rounding keeps significant figures.** `round_density` reports a dense
stand to one decimal (55,176.0/acre) but keeps four significant figures below
100, so a scouting pass finding 0.73 weeds per acre does not round to 0.7 and
lose a quarter of its magnitude.

**Row geometry is checked against the selection.** When the plot geometry the
user entered and the area they selected disagree by more than 3×, the analysis
response carries a warning. That mismatch is the usual reason a stand count
comes back at several hundred percent of target.
