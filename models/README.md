# Model checkpoints

Drop your Ultralytics-compatible YOLO checkpoint here, then point the service
at it with `MODEL_PATH` (or pick it from the sidebar's model browser, which
lists everything under `LIBRARY_DIRS`).

```
models/
└── YOLO12n_plantweed1280-best.pt
```

Weights are not tracked in git — `.gitignore` excludes `*.pt`.

## Class labels

The console reasons about two roles, **plant** and **weed**, derived from the
checkpoint's own labels:

- a label listed in `PLANT_CLASSES` is a crop plant;
- a label listed in `WEED_CLASSES` is a weed;
- **anything else is treated as a weed**, because for a stand-count decision
  over-counting weeds is the safer error.

So a single-class `plant` model works out of the box, and a multi-species weed
model works by adding its labels to `WEED_CLASSES`.

## Checkpoint forks

Weights trained with the [`sunsmarterjie/yolov12`](https://github.com/sunsmarterjie/yolov12)
fork need that package rather than PyPI `ultralytics` — it imports under the
same name, so install one or the other, never both:

```bash
pip uninstall -y ultralytics
pip install "git+https://github.com/sunsmarterjie/yolov12.git"
```
