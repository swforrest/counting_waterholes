# CLAUDE.md — Counting Waterholes Project

## Project Overview

End-to-end satellite imagery pipeline for detecting and classifying waterholes in Northern Australia (Arnhem Land and Cape York regions). Uses Planet satellite imagery and YOLOv5 object detection. Adapted from the [CountingBoats](https://github.com/charlie-turner-314/CountingBoats) project. Primary interface is Jupyter notebooks; the `counting_boats/` Python package provides the underlying logic.

**Goal:** Map waterholes susceptible to damage from invasive herbivores (water buffalo, feral cattle, pigs).

---

## Repository Structure

```
counting_waterholes/
├── counting_boats/              # Core Python package
│   ├── boat_utils/             # Utility modules
│   │   ├── config.py           # YAML config loader
│   │   ├── planet_utils.py     # Planet API (search/order/download)
│   │   ├── classifier.py       # YOLOv5 inference & clustering
│   │   ├── image_cutting_support.py  # TIFF→PNG, tiling, coord transforms
│   │   ├── testing.py          # Validation & evaluation utilities
│   │   ├── auto_helpers.py     # Pipeline automation helpers
│   │   ├── spatial_helpers.py  # Geospatial utilities
│   │   └── heatmap.py          # Coverage heatmap generation
│   ├── train.py                # Training workflow (CLI via typer)
│   ├── classify.py             # Deployment CLI (typer app)
│   ├── analysis.py             # Analysis and visualization functions
│   ├── plot_output.py          # Draw bounding boxes on images
│   ├── testing.py              # Top-level testing orchestration
│   └── reconcile_val_mistakes.py  # Validation error correction
│
├── data/
│   ├── NN_weights.pt           # Pre-trained YOLOv5 model weights
│   ├── polygons/               # AOI boundary GeoJSON files
│   │   ├── mimal*.geojson      # Mimal study area variants
│   │   ├── bulman.geojson
│   │   ├── peel.json
│   │   └── [15+ other AOI files]
│   └── Rapid Waterhole Assessment 2024*.csv  # Ground truth data
│
├── training/                   # Training datasets and run outputs
│   ├── run_3_archive_GPU/      # Previous training run (archived)
│   └── run_4/                  # Current training run
│       └── {images,labels}/{train,val}/  # YOLO-structured dataset
│
├── testing/                    # Test/validation dataset
│   ├── pngs/                   # Test PNGs (prepared)
│   ├── raw_images/             # Raw TIFFs for testing
│   └── unlabelled_yet/         # Awaiting annotation
│
├── visualisation/              # Analysis and plotting tools
│   ├── create_coverage_heatmaps.py
│   ├── exploring.ipynb
│   ├── timeseries.ipynb
│   ├── spatial_plots.ipynb
│   ├── clean_data.ipynb
│   └── check_valid.ipynb
│
├── images/                     # Runtime image storage
│   └── RawImages/              # Input TIFFs for inference
│
├── debugging/                  # Experimental configs and scripts
│   ├── config_*.yaml           # Various test configurations
│   └── run_script_AF.py        # Legacy training script
│
├── Notebooks (root level)      # Primary user-facing workflow
│   ├── planet_download.ipynb       # Step 1: Acquire Planet imagery
│   ├── S2_download_GEE.ipynb       # Alt: Download Sentinel-2 via GEE
│   ├── S2_download_GEE_random_sampling.ipynb  # GEE random sampling
│   ├── From_tif_to_trainable_*.ipynb  # Step 2: Training data prep
│   ├── cookie-cutting_S2_download.ipynb  # S2 image cookie-cutting
│   ├── post_training*.ipynb        # Step 3: Validation/testing
│   ├── deployment.ipynb            # Step 4: Inference on new images
│   ├── analysis.ipynb              # Step 5: Analyse results
│   ├── Image_boxes_checks.ipynb    # Annotation validation
│   └── csv_to_json.ipynb           # Convert CSV detections to JSON
│
├── config.yml                  # Main project config
├── config_train_*.yaml         # Training-specific configs
├── config_test_*.yaml          # Testing-specific configs
├── config_deploy_*.yaml        # Deployment configs
├── env.yaml                    # Conda environment definition
└── api_key.env                 # Planet API key (not in git)
```

---

## Pipeline: End-to-End Data Flow

### Step 1 — Image Acquisition (`planet_download.ipynb`)
```
Define AOI polygon (data/polygons/*.geojson)
  → PlanetSearch()      — find available imagery by date/cloud cover
  → PlanetSelect()      — filter by area coverage (>90%) and date
  → PlanetOrder()       — submit Planet API order
  → PlanetDownload()    — download & extract TIFF files
  → images/RawImages/   — raw multi-band GeoTIFF files
```
Sentinel-2 alternative: `S2_download_GEE.ipynb` via Google Earth Engine.

### Step 2 — Training Data Preparation (`From_tif_to_trainable_*.ipynb`)
```
images/RawImages/*.tif
  → train.prepare()         — TIFF → padded PNG (tile-aligned, 416px tiles, 104px stride)
  → Manual annotation       — LabelMe JSON with bounding boxes
  → train.segment()         — Split PNG+JSON into 416×416 tiles (80/20 train/val)
  → train.describe()        — Report class distribution
  → train.cull_AF()         — Remove excess background tiles (keep ≤10% background)
  → train.reorganize_folders()  — Structure into YOLO format
  → training/run_X/{images,labels}/{train,val}/
```

### Step 3 — Model Training (`From_tif_to_trainable_*.ipynb`, final cells)
```
training/run_X/
  → train.train(config)     — Execute YOLOv5 training
  → runs/exp/weights/best.pt  — Best checkpoint saved here
```

### Step 4 — Validation (`post_training*.ipynb`)
```
testing/raw_images/*.tif
  → testing.prepare()       — TIFF → PNG
  → Manual annotation       — Ground truth labels
  → testing.segment()       — Tile images/labels
  → testing.run_detection() — YOLOv5 inference
  → testing.backwards_annotation_AF()  — Predictions → LabelMe JSON
  → testing.compare_detections_to_ground_truth()  — Match preds vs GT
  → testing.confusion_matrix_AF()      — Metrics per class
  → testing.plot_waterholes()          — Visual comparison
```

### Step 5 — Deployment / Inference (`deployment.ipynb`)
```
New images/RawImages/*.tif
  → testing.prepare()       — TIFF → PNG
  → testing.run_detection() — Inference with trained weights
  → testing.backwards_annotation_AF()  — Detections → JSON
  → classifications/        — Final output (CSV + JSON)
```

### Step 6 — Analysis (`analysis.ipynb`, `visualisation/`)
```
classifications/*.csv
  → timeseries.ipynb        — Temporal trends
  → spatial_plots.ipynb     — Geographic distribution
  → create_coverage_heatmaps.py  — Density heatmaps
```

---

## Key Functions by Module

### `counting_boats/boat_utils/planet_utils.py`
| Function | Purpose |
|---|---|
| `PlanetSearch(polygon_file, min_date, max_date, cloud_cover)` | Query Planet API for available scenes |
| `PlanetSelect(options, polygon_file, date, area_coverage)` | Filter results by coverage and date |
| `PlanetOrder(polygon_file, items, name)` | Submit download order |
| `PlanetCheckOrder(order_id)` | Poll order status |
| `PlanetDownload(order_id, download_dir)` | Download and extract imagery |
| `extract_zip_AF(zip_path, out_dir)` | Extract TIFFs from zip archives |

### `counting_boats/boat_utils/classifier.py`
| Function | Purpose |
|---|---|
| `process_tif(tif_path, config)` | Main entry: detect on a single TIFF |
| `detect_from_tif(tif_path, config)` | Run YOLOv5 on tiled image |
| `cluster_AF(detections, threshold_px)` | Merge overlapping detections |
| `process_clusters_AF(clusters)` | Convert clusters to center coords |
| `pixel2latlong(px, py, geotransform)` | Pixel → lat/long |
| `process_tif_waterhole(tif_path, config)` | Waterhole-specific detection with per-class clustering |

### `counting_boats/boat_utils/image_cutting_support.py`
| Function | Purpose |
|---|---|
| `create_padded_png(tif_path, out_dir, tile_size, stride)` | TIFF → padded PNG |
| `create_padded_png_S2(tif_path, out_dir, tile_size, stride)` | Sentinel-2 variant |
| `segment_image(image_path, label_path, out_dir, tile_size, stride)` | Tile image+labels |
| `add_margin(image, margin)` | Pad image edges |
| `get_required_padding(dim, tile_size, stride)` | Calculate padding amount |
| `latlong2coord(lat, lon, geotransform)` | Lat/long → pixel |
| `coord2latlong(px, py, geotransform)` | Pixel → lat/long |
| `Classification` | Bounding box data class |

### `counting_boats/boat_utils/testing.py`
| Function | Purpose |
|---|---|
| `prepare(tif_dir, config)` | TIFF → PNG for labelling |
| `segment(tif_dir, config)` | Tile labelled images |
| `run_detection(tif_dir, config)` | Run YOLOv5 inference |
| `backwards_annotation_AF(tif_dir, config)` | Detections → LabelMe JSON |
| `compare_detections_to_ground_truth(tif_dir, config)` | Match preds to GT |
| `confusion_matrix_AF(tif_dir, config)` | Compute confusion matrix |
| `plot_waterholes(tif_dir, config)` | Visualise detections vs labels |
| `waterholes_count_compare(tif_dir, config)` | Compare count totals |

### `counting_boats/train.py` (CLI: `python -m counting_boats.train`)
| Command | Purpose |
|---|---|
| `prepare(config)` | Convert TIFFs to padded PNGs |
| `prepare_S2(config)` | Sentinel-2 variant |
| `segment(config, train_val_split=0.8)` | Tile and split train/val |
| `describe(config)` | Report dataset statistics |
| `cull_AF(config)` | Remove excess background tiles |
| `reorganize_folders(config)` | Restructure for YOLO |
| `train(config)` | Execute YOLOv5 training |

### `counting_boats/boat_utils/auto_helpers.py`
| Function | Purpose |
|---|---|
| `search(config)` | Search for new imagery |
| `select(config)` | Select from search results |
| `order(config)` | Place Planet orders |
| `download(config)` | Download ordered imagery |
| `count(config)` | Run detection |
| `archive(config)` | Archive raw data |
| `analyse(config)` | Summarise detection results |

### `counting_boats/boat_utils/spatial_helpers.py`
| Function | Purpose |
|---|---|
| `area_coverage_tif(polygon, tif_path)` | % of polygon covered by TIFF |
| `area_coverage_poly(poly1, poly2)` | % intersection of two polygons |
| `combine_polygons(polygon_list)` | Merge overlapping polygons |
| `polygons_to_32756(polygons)` | Reproject to UTM zone 56S |

---

## Configuration Files

### `config.yml` (main runtime config)
Key parameters:
```yaml
yolo_dir:              # Path to cloned YOLOv5 repo
python:                # Python interpreter path
proj_root:             # Project root directory
weights:               # Model weights (data/NN_weights.pt or runs/exp/weights/best.pt)
planet.api_key:        # Planet API key (or 'ENV' to read from api_key.env)
output_dir:            # Results output directory
tif_dir:               # Raw TIFFs location
download_dir:          # Planet download staging area

TILE_SIZE: 416         # Tile size for inference
STRIDE: 104            # Overlap stride (25% of tile size)
CONFIDENCE_THRESHOLD: 0.5
STAT_DISTANCE_CUTOFF_PIX: 3       # Clustering threshold (pixels)
STAT_DISTANCE_CUTOFF_LATLONG: 0.00025
AUTO_MODE: "batch"     # or "single"
ALLOWED_CLOUD_COVER: 0.1
MINIMUM_AREA_COVERAGE: 0.9
AOIS: "all"            # or comma-separated AOI names
```

### `config_train_*.yaml` (training configs)
Defines raw image paths, output directories, class names, batch size, epochs, image size. Multiple variants exist for different environments (GPU server, Google Drive, local/SF).

### `config_test_*.yaml` / `config_deploy_*.yaml`
Defines test image paths, trained weights path, confidence thresholds per class, comparison distance cutoffs, and which pipeline stages to run.

---

## Waterhole Classes

| ID | Class Name | Description |
|---|---|---|
| 0 | `Dry_WH` | Dry waterhole |
| 1 | `WH_swamp` | Swampy / vegetated waterhole |
| 2 | `WH_wet` | Water-filled waterhole |
| 3 | `WH_sink` | Sinkhole-type waterhole |
| 4 | `U` | Unclassified / uncertain |

Per-class clustering distance thresholds (pixels):
- Dry: 120, Wet: 120, Swamp: 170, Sink: 120, U: 120

---

## Key Technical Details

- **Tile size:** 416×416 px, stride 104 px (25% overlap) — ensures features at tile edges are still detected
- **Padding:** Images are padded before tiling so dimensions are exactly divisible by stride
- **Clustering:** Detections within threshold distance are merged; cluster centroid becomes the waterhole location
- **Coordinate system:** TIF geotransform used to convert pixel → lat/long; spatial work in UTM zone 56S (EPSG:32756)
- **Model:** YOLOv5s (small variant); input 416×416 RGB; outputs bounding boxes + class probabilities
- **Background culling:** Training tiles with no labels are culled so they make up ≤10% of dataset

---

## Output Formats

**Detection CSV:**
```
latitude, longitude, confidence, class, date, image
-12.95, 134.60, 0.87, 0, 2024-06-07, 20240607_mimal.tif
```

**LabelMe JSON (annotations):**
```json
{
  "shapes": [{"label": "Dry_WH", "points": [[x1,y1],[x2,y2]], "shape_type": "rectangle"}],
  "imagePath": "20240607_mimal.png"
}
```

---

## Environment

```bash
conda env create -f env.yaml   # Python 3.10
```

Key dependencies: `torch`, `torchvision`, `gdal`, `rasterio`, `numpy`, `pandas`, `PIL`, `matplotlib`, `scipy`, `opencv`, `typer`, `comet-ml`

**Known issues:**
- Set `KMP_DUPLICATE_LIB_OK=TRUE` on some systems
- GPU training requires manual PyTorch/CUDA version alignment
- Deployment notebook is incomplete and still in progress

---

## Study Areas (AOI Polygons in `data/polygons/`)

Primary areas: Mimal (Arnhem Land), Bulman, Peel, Cape York. 15+ polygon files covering different sub-regions and variants used in different model runs.

---

## Authorship Notes

Code marked with `# AF` indicates additions/modifications by Adriano Fossati (e.g., `cluster_AF`, `cull_AF`, `backwards_annotation_AF`). The base pipeline is from the CountingBoats project (charlie-turner-314).
