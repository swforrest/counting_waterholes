# Repository Overview — Counting Waterholes

> This is a **descriptive summary** of the repo for humans, written to reflect the
> current state of the codebase (including recent, not-yet-committed AlphaEarth work).
> It complements — but does not replace — [`CLAUDE.md`](CLAUDE.md), which is the
> agent-facing instructions file and covers the original `counting_boats/` (YOLOv5)
> pipeline in detail.

## 1. What this project does

The repo maps **waterholes in Northern Australia** (Arnhem Land, Cape York) from
satellite imagery, with the goal of identifying waterholes **susceptible to damage
from invasive herbivores** (water buffalo, feral cattle, pigs). It does this via
**two parallel, complementary pipelines** that both consume Sentinel-2 / Planet
imagery but answer different questions:

| | Pipeline A — `counting_boats/` | Pipeline B — `cookie-cutting/` |
|---|---|---|
| **Question** | *Where are the waterholes?* (object detection) | *What state is each waterhole in?* (per-pixel classification) |
| **Method** | YOLOv5 bounding-box detector on tiled PNGs | Per-pixel classifier (e.g. gradient boosting) using spectral indices + temporal features + AlphaEarth embeddings |
| **Output unit** | One bounding box per waterhole | Per-pixel class (dry / swamp / wet / sink) across a chip |
| **Imagery source** | Planet imagery (primary), Sentinel-2 (alt.) | Sentinel-2 chips + AlphaEarth embeddings, aligned to a fixed raster grid |
| **Status** | Mature, adapted from [CountingBoats](https://github.com/charlie-turner-314/CountingBoats) | Newer, actively evolving (AlphaEarth integration ongoing, uncommitted) |

---

## 2. Pipeline A — `counting_boats/` (YOLOv5 object detection)

Detects individual waterholes as bounding boxes and classifies each into one of 5
condition classes. Adapted from the CountingBoats project.

### Data flow

```
Image Acquisition (planet_download.ipynb / S2_download_GEE.ipynb)
  → images/RawImages/*.tif
  → Training data prep (From_tif_to_trainable_*.ipynb)
      TIFF → padded PNG (416px tiles, 104px stride) → LabelMe annotation
      → tiled train/val split → background culling → YOLO folder structure
  → Model Training (YOLOv5, same notebook)
      → runs/exp/weights/best.pt
  → Validation (post_training*.ipynb)
      prepare → segment → detect → backwards-annotate → compare vs ground truth
      → confusion matrix / plots
  → Deployment (deployment.ipynb)
      New TIFFs → detect → JSON/CSV of detections
  → Analysis (analysis.ipynb, visualisation/)
      Timeseries, spatial plots, coverage heatmaps
```

### Package structure (`counting_boats/`)

```
counting_boats/
├── train.py               # CLI (typer): prepare, segment, describe, cull_AF,
│                           #   reorganize_folders, train
├── classify.py             # Deployment CLI (typer)
├── analysis.py              # Analysis & visualization functions
├── plot_output.py           # Draw bounding boxes on images
├── testing.py                # Top-level testing orchestration
├── reconcile_val_mistakes.py # Validation error correction
├── tile.py                   # Tiling helpers
└── boat_utils/
    ├── config.py              # YAML config loader
    ├── planet_utils.py        # Planet API (search/order/download)
    ├── classifier.py          # YOLOv5 inference & clustering
    ├── image_cutting_support.py  # TIFF→PNG, tiling, coord transforms
    ├── testing.py              # Validation & evaluation utilities
    ├── auto_helpers.py          # Pipeline automation helpers
    ├── spatial_helpers.py       # Geospatial utilities
    ├── heatmap.py                # Coverage heatmap generation
    ├── stitch_PNGs.py            # Reassemble tiles into full images
    └── user_io_helpers.py        # I/O convenience functions
```

### Key functions

| Module | Function | Purpose |
|---|---|---|
| `planet_utils.py` | `PlanetSearch/Select/Order/Download` | Full Planet API order lifecycle |
| `classifier.py` | `process_tif_waterhole(tif_path, config)` | Waterhole-specific detection with per-class clustering |
| `classifier.py` | `cluster_AF(detections, threshold_px)` | Merge overlapping detections |
| `image_cutting_support.py` | `create_padded_png(...)` | TIFF → padded, tile-aligned PNG |
| `testing.py` | `compare_detections_to_ground_truth(...)` | Match predictions to labels |
| `auto_helpers.py` | `search/select/order/download/count/archive/analyse` | End-to-end automation steps |

### Waterhole classes (5)

`Dry_WH`, `WH_swamp`, `WH_wet`, `WH_sink`, `U` (unclassified). Per-class clustering
distance thresholds range 120–170 px. See `CLAUDE.md` for full details, config keys
(`config.yml`), and known environment issues.

---

## 3. Pipeline B — `cookie-cutting/` (per-pixel waterhole state classifier)

A second, largely undocumented pipeline that classifies **every pixel** inside a
waterhole chip by surface state, rather than drawing a single bounding box. Built to
exploit **temporal behaviour** (how a pixel's spectral signature changes across
seasons/years) and **AlphaEarth embeddings** as features, on top of classic spectral
indices.

### Core modules (`cookie-cutting/`)

| Module | Purpose |
|---|---|
| `wh_config.py` | Loads `waterhole_seg_config.yaml`, resolves data paths |
| `wh_naming.py` | Parses/builds chip filenames (`<prefix>_S2_<site>_<lat>_<lon>_<YYYY-MM>`) |
| `wh_tiles.py` | Reads GeoTIFF chips, writes label masks on the raster's native grid; handles two nodata conventions |
| `wh_indices.py` | Spectral index functions — MNDWI, NDVI, NDTI, NDMI, red-edge index (NaN-aware) |
| `wh_inventory.py` | Builds a manifest of exported chips (site/month/grid/quality checks) |
| `wh_bbox.py` | Converts LabelMe bounding boxes into per-site masks, constraining footprint/composition calculations |
| `wh_temporal.py` | Per-pixel temporal features — each pixel normalized against its own multi-year history. Foundational module the rest of the design leans on |
| `wh_features.py` | Assembles the full per-pixel feature table: instantaneous + local context + temporal + AlphaEarth embeddings |
| `wh_footprint.py` | Derives one basin "footprint" per site from seasonal behaviour (robust z-scores vs. the site's tile matrix), not a simple threshold |
| `wh_pseudo.py` | Auto-generates high-confidence pseudo-labels for easy classes (conservative open-water/matrix rule) |
| `wh_label.py` | Interactive matplotlib tool for sparse manual labelling, writing masks on the raster's native grid |
| `wh_train.py` | Trains/evaluates the per-pixel classifier; enforces **site-grouped cross-validation** (no pixel-level random splits) because of spatial autocorrelation |
| `wh_plots.py` | Diagnostic plots — temporal features, harmonic fits, footprints, and an AlphaEarth RGB/PCA panel in the labelling UI |

### Driving notebooks

- `labelling.ipynb` — sparse labelling of waterhole surface state
- `footprint_estimation.ipynb` — basin footprint estimation/exploration
- `pseudo_labelling.ipynb` — pseudo-label generation
- `model_training.ipynb` — trains/compares classifiers (incl. gradient boosting), with a feature-ablation study (e.g. raw features vs. +AlphaEarth embeddings)

### Tests

`cookie-cutting/tests/` — `test_wh_bbox.py`, `test_wh_features.py`,
`test_wh_indices.py`, `test_wh_naming.py`, `test_wh_temporal.py`, `test_wh_train.py`,
plus `conftest.py`.

### Data directories

`AlphaEarth_tif/`, `derived/`, `full_images/`, `images/`, `images_tif/`,
`images_tif_v2/`, `labels/`, `labels_pseudo/`.

---

## 4. AlphaEarth embeddings

**AlphaEarth (Google Satellite Embedding V1)** is a 64-band annual, 10 m composite
produced by a geospatial foundation model from Sentinel-2, SAR, and other sources.
It's used in Pipeline B as an additional feature panel alongside raw spectral
bands/indices, on the hypothesis that the embeddings encode useful land-cover context
beyond what's derivable from raw reflectance alone.

How it plugs in:

- **`S2_download_AlphaEarth_GEE.ipynb`** — exports AlphaEarth embedding bands
  (A00–A63) from Google Earth Engine, locked to the same raster grid as the
  Sentinel-2 tiles, to Google Drive.
- **`cookie-cutting_AlphaEarth_download.ipynb`** — cookie-cuts/aligns the downloaded
  AlphaEarth rasters to match chip/PNG padding and LabelMe coordinates.
- **`wh_features.py`** — `ALPHAEARTH_PREFIX` / `load_alphaearth()` fold the embedding
  bands into the feature table; two named feature sets support ablation comparisons:
  `embeddings_only` vs. `instantaneous_plus_embeddings`.
- **`wh_train.py` / `wh_label.py`** — a `params.use_alphaearth` flag controls whether
  embeddings are loaded for a given training/labelling session, with a clear error if
  a model trained *with* embeddings is later used *without* them.
- **`wh_plots.py`** — the interactive labelling UI has a dedicated AlphaEarth panel
  (RGB composite from 3 chosen bands, or PCA) to help labellers see embedding
  structure.

Related but separate exploratory work: **`sentinel-2_embeddings.ipynb`** (by Scott
Forrest) uses pretrained geospatial foundation models to embed S2 pixels for
clustering/classification — conceptually related, possibly precursor work to the
AlphaEarth integration.

---

## 5. Other top-level notebooks (not covered by `CLAUDE.md`)

| File | Purpose |
|---|---|
| `S2_download_AlphaEarth_GEE.ipynb` | Export AlphaEarth embeddings aligned to the S2 grid |
| `sentinel-2_embeddings.ipynb` | Exploratory foundation-model embedding of S2 pixels |
| `classification_labelling_tif_to_png.ipynb` | Renders S2 chips to labelling PNGs using the `wh_*` modules |
| `cookie-cutting_AlphaEarth_download.ipynb` | Aligns/cookie-cuts AlphaEarth rasters to PNG padding |
| `cookie-cutting_S2_download.ipynb` | Defines the chip-naming convention used by `wh_naming.py` |
| `json_re-labelling.ipynb` | Re-labelling workflow for existing annotations |
| `satellite_map_alphaearth.html`, `satellite_map_alphaearth_aoi.html`, `satellite_map2/3/4.html` | Standalone interactive map outputs |

---

## 6. Repository directory map (consolidated)

```
counting_waterholes/
├── counting_boats/          # Pipeline A — YOLOv5 detection package (see §2)
├── cookie-cutting/          # Pipeline B — per-pixel classifier package (see §3)
│   ├── wh_*.py               # Core modules
│   ├── waterhole_seg_config.yaml
│   ├── tests/
│   ├── AlphaEarth_tif/ derived/ full_images/ images/
│   ├── images_tif/ images_tif_v2/ labels/ labels_pseudo/
│   └── *.ipynb                # Driving notebooks
├── data/                      # AOI polygons, ground truth CSVs, NN weights
│   └── polygons/               # 15+ AOI GeoJSON files (mimal, bulman, peel, ...)
├── AOIs/                       # Additional AOI resources
├── training/                   # Pipeline A training datasets & run outputs
├── testing/                    # Pipeline A test/validation dataset
├── visualisation/               # Analysis & plotting tools
├── images/                      # Runtime raw image storage (RawImages/)
├── images_labelled/ images_unlabelled/  # Labelling staging areas
├── alphaearth_images/           # AlphaEarth raster staging
├── counting_wh/                 # (support/output dir — see notebooks for usage)
├── results/                     # Output results
├── docs/                        # Documentation assets
├── Website_buildup/              # Website/dashboard build materials
├── debugging/                    # Experimental configs & scripts
├── .cometml-runs/                 # Comet ML experiment tracking output
├── tempDL/                        # Temporary download staging
├── config.yml                     # Pipeline A main runtime config
├── config_train_*.yaml / config_test_*.yaml / config_deploy_*.yaml
├── env.yaml                       # Conda environment definition
├── api_key.env                    # Planet API key (not in git)
├── CLAUDE.md                      # Agent instructions (Pipeline A focus)
└── REPO_OVERVIEW.md                # This file
```

---

## 7. Configuration files

- **`config.yml`** — Pipeline A runtime config: YOLOv5/Planet paths, tile size (416),
  stride (104), confidence threshold, clustering cutoffs, AOI selection. See
  `CLAUDE.md` for the full key reference.
- **`config_train_*.yaml` / `config_test_*.yaml` / `config_deploy_*.yaml`** —
  Pipeline A stage-specific configs (raw image paths, class names, batch size,
  epochs, weights, comparison cutoffs).
- **`waterhole_seg_config.yaml`** (in `cookie-cutting/`) — Pipeline B config,
  loaded via `wh_config.py`.

---

## 8. Environment

```bash
conda env create -f env.yaml   # Python 3.10
```

Key dependencies: `torch`, `torchvision`, `gdal`, `rasterio`, `numpy`, `pandas`,
`PIL`, `matplotlib`, `scipy`, `opencv`, `typer`, `comet-ml`.

Known issues: set `KMP_DUPLICATE_LIB_OK=TRUE` on some systems; GPU training requires
manual PyTorch/CUDA version alignment; the `deployment.ipynb` notebook (Pipeline A)
is incomplete and still in progress.

---

## 9. Authorship

Code marked `# AF` indicates additions/modifications by Adriano Fossati (e.g.
`cluster_AF`, `cull_AF`, `backwards_annotation_AF`, and the entirety of the
`cookie-cutting/` Pipeline B and AlphaEarth integration). The base Pipeline A code is
adapted from the CountingBoats project (charlie-turner-314).
