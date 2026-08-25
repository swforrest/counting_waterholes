"""Apply the trained classifier across every site-month and aggregate the result.

This is the step the project exists for: turning a classifier into a per-waterhole
time series of surface composition.

Three things here are deliberate and worth knowing before changing them.

**Feature parameters come from the model's own manifest, never from a notebook.**
A model applied with features in a different order, or built under a different
config, does not fail — it returns confident nonsense. `wh_train.save_model`
records exactly what it was trained on, and this module rebuilds from that and
asserts the assembled feature list matches.

**Counts are emitted for both the bounding box and the footprint.** Once a class
raster exists, counting it in a second region costs nothing, and the alternative
is re-running an hour of prediction to answer "what about the footprint?".

**The temporal consistency pass flags; it never smooths.** An isolated wet month
between dry ones, at low observation count, is more likely a compositing artefact
than a rainfall event — but it might not be, and silently rewriting it would hide
exactly the events the study is looking for.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage

import wh_bbox
import wh_features
import wh_footprint
import wh_naming
import wh_train
from wh_config import Config
from wh_features import FeatureParams

WGS84 = "EPSG:4326"


@dataclass
class PredictParams:
    """Everything tunable about a prediction run."""

    model_name: str = "classifier"

    # 0 = off. A 3x3 majority filter is standard despeckling for landscape
    # classification and wrong at this scale: 39% of the water patches here are
    # 1-2 px, and the filter would delete them. Same reasoning that turned
    # erosion off in the pseudo-labeller.
    majority_filter_px: int = 0

    # Max class probability as uint8 0-100. One extra small band rather than six
    # float32 probability bands: ~19 MB against ~460 MB, and it is the layer the
    # dashboard actually needs.
    write_confidence: bool = True

    # Colourised PNG beside each GeoTIFF. A browser cannot draw a GeoTIFF without
    # a tile server, but it can place a PNG as an image overlay given bounds —
    # which is what makes a backend-free dashboard possible.
    write_png: bool = True

    # Which region the notebook's plots summarise. Both are always written to the
    # CSV; this only chooses the default view.
    denominator: str = "bbox"

    # An isolated wet month between dry ones at low observation count.
    flag_isolated_wet: bool = True
    low_obs_threshold: float = 2.0
    wet_classes: tuple[str, ...] = ("open_water", "turbid_water")

    # Data-quality banding for the CSV.
    good_mean_obs: float = 3.0
    poor_gap_fraction: float = 0.5

    sites: tuple[str, ...] | None = None
    overwrite: bool = False

    # Sites are independent, so the run parallelises cleanly across them.
    # Single-threaded the archive takes ~115 min, most of it in predict_proba
    # for the confidence band; on 6 workers it is closer to 20.
    #
    # Each worker is pinned to one BLAS/OpenMP thread. Without that, six
    # processes each spawning ten threads oversubscribe the machine and the
    # whole run gets slower, not faster.
    workers: int = 6

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "majority_filter_px": self.majority_filter_px,
            "write_confidence": self.write_confidence,
            "write_png": self.write_png,
            "denominator": self.denominator,
            "flag_isolated_wet": self.flag_isolated_wet,
            "low_obs_threshold": self.low_obs_threshold,
            "good_mean_obs": self.good_mean_obs,
            "poor_gap_fraction": self.poor_gap_fraction,
        }


# --- getting the model's own feature definition ----------------------------


def feature_params_from_manifest(meta: dict) -> FeatureParams:
    """Rebuild the FeatureParams a model was trained with.

    Applying a model with a different feature set is not an error the model can
    detect — it happily consumes whatever it is given and returns something. The
    manifest is the only authority on what it was fitted to.
    """
    stored = meta["feature_params"]
    bands = stored.get("alphaearth_bands", "all")

    return FeatureParams(
        reflectance_bands=tuple(stored["reflectance_bands"]),
        indices=tuple(stored["indices"]),
        context_windows=tuple(stored["context_windows"]),
        context_indices=tuple(stored["context_indices"]),
        temporal_indices=tuple(stored["temporal_indices"]),
        include_n_obs=bool(stored.get("include_n_obs", False)),
        use_alphaearth=bool(stored.get("use_alphaearth", False)),
        alphaearth_year=int(stored.get("alphaearth_year", 2025)),
        alphaearth_bands=None if bands in (None, "all") else tuple(bands),
    )


def check_scheme(meta: dict, cfg: Config) -> None:
    """Refuse to predict under a class scheme the model was not trained on."""
    trained = int(meta.get("class_scheme_version", -1))
    current = int(cfg["classes"]["scheme_version"])
    if trained != current:
        raise ValueError(
            f"the model was trained under class scheme version {trained} but the "
            f"config is now version {current}. Class ids would not mean the same "
            f"thing; retrain before predicting."
        )


# --- output paths ----------------------------------------------------------


def site_dir(cfg: Config, site_id: str) -> Path:
    return cfg.paths["predictions"] / "pixel_predictions" / f"site_{site_id}"


def raster_path(cfg: Config, site_id: str, stem: str, kind: str = "pred") -> Path:
    suffix = ".png" if kind == "png" else ".tif"
    name = "pred" if kind == "png" else kind
    return site_dir(cfg, site_id) / f"{stem}_{name}{suffix}"


def composition_path(cfg: Config) -> Path:
    return cfg.paths["predictions"] / "waterhole_composition.csv"


# --- writing ---------------------------------------------------------------


def _write_uint8(path: Path, data: np.ndarray, tile, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1, dtype="uint8",
        crs=tile.crs, transform=tile.transform, nodata=None, compress="deflate",
    ) as dataset:
        dataset.write(data.astype(np.uint8), 1)
        dataset.descriptions = (description,)


def _class_rgba(classes: np.ndarray, cfg: Config) -> np.ndarray:
    """Colourise a class raster using the config's colours, class 0 transparent."""
    import matplotlib.colors as mcolors

    rgba = np.zeros(classes.shape + (4,), dtype=np.uint8)
    for definition in cfg.classes:
        if definition.ignore:
            continue
        selected = classes == definition.id
        if selected.any():
            colour = mcolors.to_rgba(definition.colour)
            rgba[selected] = [int(round(channel * 255)) for channel in colour]
    return rgba


def write_png(path: Path, classes: np.ndarray, cfg: Config) -> None:
    """Colourised PNG for display. Unclassified pixels stay transparent."""
    import matplotlib.image

    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.image.imsave(path, _class_rgba(classes, cfg))


def write_bounds(cfg: Config, site_id: str, tile) -> Path:
    """WGS84 bounds for the site, written once — every month shares one grid.

    This is what lets a web map place the PNGs as image overlays without a tile
    server, which is the whole reason the dashboard can be a static site.
    """
    with rasterio.open(tile.path) as dataset:
        bounds = dataset.bounds

    west, south, east, north = transform_bounds(tile.crs, WGS84, *bounds)
    path = site_dir(cfg, site_id) / "bounds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "site_id": site_id,
        "crs": str(tile.crs),
        "shape": [int(tile.shape[0]), int(tile.shape[1])],
        # Leaflet imageOverlay wants [[south, west], [north, east]].
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "leaflet_bounds": [[south, west], [north, east]],
    }, indent=1))
    return path


# --- counting --------------------------------------------------------------


def _safe_nanmean(values: np.ndarray | None) -> float:
    """Mean ignoring NaN, or NaN when a tile is unobserved throughout.

    A fully clouded month has no confidence to average; that is expected here,
    not a symptom, so it does not warn.
    """
    if values is None:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


def count_classes(
    classes: np.ndarray, region: np.ndarray, cfg: Config, prefix: str
) -> dict[str, object]:
    """Pixel counts and fractions per class within a region.

    Fractions are over the CLASSIFIED pixels of the region, not its full area, so
    a month with cloud gaps reports composition rather than a diluted version of
    it. The denominator is reported alongside so the dilution is recoverable.
    """
    inside = region & (classes > 0)
    total = int(inside.sum())

    record: dict[str, object] = {f"{prefix}_n_classified": total}
    for definition in cfg.classes:
        if definition.ignore:
            continue
        count = int((inside & (classes == definition.id)).sum())
        record[f"{prefix}_n_{definition.name}"] = count
        record[f"{prefix}_frac_{definition.name}"] = (
            count / total if total else np.nan
        )
    return record


def majority_filter(classes: np.ndarray, size: int) -> np.ndarray:
    """Replace each pixel with the most common class in its window.

    Off by default. See PredictParams.majority_filter_px for why.
    """
    if size <= 1:
        return classes

    counts = [
        ndimage.uniform_filter(
            (classes == definition_id).astype(np.float32), size=size, mode="nearest"
        )
        for definition_id in range(int(classes.max()) + 1)
    ]
    smoothed = np.argmax(np.stack(counts), axis=0).astype(np.uint8)
    # Never invent a class where nothing was classified.
    return np.where(classes > 0, smoothed, 0).astype(np.uint8)


# --- per-site prediction ---------------------------------------------------


def predict_site(
    manifest: pd.DataFrame,
    cfg: Config,
    model,
    meta: dict,
    params: PredictParams,
    site_id: str,
    boxes: pd.DataFrame,
) -> list[dict[str, object]]:
    """Predict every month of one site, write the rasters, return the records."""
    feature_params = feature_params_from_manifest(meta)
    feature_names = list(meta["feature_names"])

    predictor = wh_train.SitePredictor(manifest, cfg, feature_params, site_id)

    try:
        box_region = wh_bbox.load_mask(cfg, site_id)
    except FileNotFoundError:
        box_region = None
    try:
        footprint_region = wh_footprint.load_mask(cfg, site_id)
    except FileNotFoundError:
        footprint_region = None

    box_row = boxes.loc[site_id] if site_id in boxes.index else None
    pixel_area = None
    records = []
    wrote_bounds = False

    for year_month in predictor.months:
        stem = Path(
            manifest[
                (manifest["site_id"] == site_id)
                & (manifest["year_month"] == year_month)
            ].iloc[0]["tif_path"]
        ).stem

        target = raster_path(cfg, site_id, stem, "pred")
        tile, classes, confidence = predictor.predict(
            model, feature_names, year_month,
            with_confidence=params.write_confidence,
        )

        if params.majority_filter_px > 1:
            classes = majority_filter(classes, params.majority_filter_px)

        if params.overwrite or not target.exists():
            _write_uint8(target, classes, tile, "class_id")
            if params.write_confidence and confidence is not None:
                _write_uint8(
                    raster_path(cfg, site_id, stem, "conf"),
                    np.nan_to_num(confidence * 100, nan=0).round(),
                    tile, "confidence_pct",
                )
            if params.write_png:
                write_png(raster_path(cfg, site_id, stem, "png"), classes, cfg)

        if not wrote_bounds:
            write_bounds(cfg, site_id, tile)
            pixel_area = abs(tile.transform.a * tile.transform.e)
            wrote_bounds = True

        box_mask = box_region if box_region is not None else np.ones(tile.shape, bool)
        foot_mask = (
            footprint_region if footprint_region is not None
            else np.zeros(tile.shape, bool)
        )

        record: dict[str, object] = {
            "site_id": site_id,
            "year_month": year_month,
            "year": int(year_month[:4]),
            "month": int(year_month[5:]),
            "n_pixels_tile": int(tile.shape[0] * tile.shape[1]),
            "n_pixels_bbox": int(box_mask.sum()),
            "n_pixels_footprint": int(foot_mask.sum()),
            "has_footprint": footprint_region is not None and bool(foot_mask.any()),
            "footprint_area_ha": float(foot_mask.sum() * pixel_area / 1e4),
            "gap_fraction": float(tile.gap_fraction),
            "mean_n_obs": (
                float(tile.n_obs[tile.valid].mean()) if tile.n_obs is not None
                and tile.valid.any() else np.nan
            ),
            "n_observed_px": int(tile.valid.sum()),
            "mean_confidence": _safe_nanmean(confidence),
        }
        record.update(count_classes(classes, box_mask, cfg, "bbox"))
        record.update(count_classes(classes, foot_mask, cfg, "footprint"))

        if box_row is not None:
            for column in ("label", "center_lon", "center_lat", "lon_min",
                           "lon_max", "lat_min", "lat_max", "width_m", "height_m"):
                record[
                    "bbox_width_m" if column == "width_m"
                    else "bbox_height_m" if column == "height_m"
                    else column
                ] = box_row[column]

        records.append(record)

    return records


def site_is_done(cfg: Config, site_id: str, manifest: pd.DataFrame) -> bool:
    """Whether every month of a site already has a class raster."""
    expected = len(manifest[manifest["site_id"] == site_id])
    return len(list(site_dir(cfg, site_id).glob("*_pred.tif"))) >= expected


def _report_progress(position: int, total: int, started: datetime) -> None:
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    rate = position / elapsed if elapsed else 0.0
    remaining = (total - position) / rate / 60 if rate else float("nan")
    print(f"  {position}/{total} sites  ({elapsed / 60:.1f} min elapsed, "
          f"~{remaining:.0f} min left)", flush=True)


# --- the whole archive -----------------------------------------------------


# Per-process cache. joblib's loky backend reuses its workers, so each process
# loads the model, manifest and boxes once and reuses them across its sites.
_WORKER: dict[str, object] = {}


def _worker_state(config_path: str, model_name: str) -> dict[str, object]:
    """Load this process's shared state on first use, then reuse it.

    Pinning each worker to a single BLAS/OpenMP thread matters: the parallelism
    is across sites, and letting every process also spawn ten threads
    oversubscribes the machine badly enough to be slower than running serially.
    """
    key = f"{config_path}|{model_name}"
    if _WORKER.get("key") == key:
        return _WORKER

    import os

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    import wh_bbox as bbox_module
    import wh_config as config_module
    import wh_inventory as inventory_module
    import wh_train as train_module

    cfg = config_module.load(config_path)
    model, meta = train_module.load_model(cfg, model_name)

    _WORKER.clear()
    _WORKER.update({
        "key": key,
        "cfg": cfg,
        "manifest": inventory_module.load_manifest(cfg),
        "model": model,
        "meta": meta,
        "boxes": bbox_module.load_boxes(cfg),
    })
    return _WORKER


def _predict_one(
    config_path: str, params: PredictParams, site_id: str
) -> tuple[str, list[dict[str, object]], str]:
    """Worker entry point. Returns (site_id, records, error message).

    Defined at module level, not in the notebook, so the worker processes can
    import it — a function defined in a notebook cell cannot be sent to a
    spawned process.
    """
    try:
        state = _worker_state(config_path, params.model_name)
        records = predict_site(
            state["manifest"], state["cfg"], state["model"], state["meta"],
            params, site_id, state["boxes"],
        )
        return site_id, records, ""
    except Exception as error:  # noqa: BLE001 — one bad site must not stop the run
        return site_id, [], f"{type(error).__name__}: {error}"


def run(
    manifest: pd.DataFrame,
    cfg: Config,
    params: PredictParams,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predict every site, write rasters, and return the composition table.

    Resumable: a site whose rasters all exist is skipped unless overwrite is set,
    so an interrupted run picks up where it stopped. Note the composition record
    is still recomputed for skipped sites — it is derived from the rasters and
    costs seconds, whereas re-predicting would cost minutes.
    """
    model, meta = wh_train.load_model(cfg, params.model_name)
    check_scheme(meta, cfg)
    boxes = wh_bbox.load_boxes(cfg)

    sites = sorted(params.sites or manifest["site_id"].unique())
    records: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    started = datetime.now(timezone.utc)

    if params.workers > 1 and len(sites) > 1:
        # joblib/loky rather than ProcessPoolExecutor: loky does not need
        # __main__ to be an importable file, which it is not inside a Jupyter
        # kernel, so this works from a notebook cell as well as a script.
        from joblib import Parallel, delayed

        if verbose:
            print(f"predicting {len(sites)} sites on {params.workers} workers", flush=True)

        results = Parallel(n_jobs=params.workers, backend="loky", verbose=0)(
            delayed(_predict_one)(str(cfg.source_path), params, site_id)
            for site_id in sites
        )
        for site_id, site_records, error in results:
            records.extend(site_records)
            if error:
                failures.append((site_id, error))
        if verbose:
            _report_progress(len(sites), len(sites), started)
    else:
        for position, site_id in enumerate(sites, start=1):
            try:
                records.extend(
                    predict_site(manifest, cfg, model, meta, params, site_id, boxes)
                )
            except Exception as error:  # noqa: BLE001
                failures.append((site_id, f"{type(error).__name__}: {error}"))
            if verbose and position % 10 == 0:
                _report_progress(position, len(sites), started)

    if failures:
        print(f"\n{len(failures)} site(s) failed:")
        for site_id, message in failures[:10]:
            print(f"  {site_id}: {message[:110]}")

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("no site produced any predictions")

    table["model_name"] = params.model_name
    table["cv_macro_f1"] = meta.get("cv_macro_f1")
    table["config_hash"] = cfg.hash
    table["predicted_at"] = started.isoformat(timespec="seconds")

    table = add_quality_flags(table, cfg, params)
    return table.sort_values(["site_id", "year", "month"]).reset_index(drop=True)


def add_quality_flags(
    table: pd.DataFrame, cfg: Config, params: PredictParams
) -> pd.DataFrame:
    """Data-quality banding and the isolated-wet flag."""
    table = table.copy()

    table["data_quality"] = np.select(
        [
            table["gap_fraction"] > params.poor_gap_fraction,
            table["mean_n_obs"] < params.low_obs_threshold,
            table["mean_n_obs"] >= params.good_mean_obs,
        ],
        ["poor", "thin", "good"],
        default="fair",
    )

    table["flag_isolated_wet"] = False
    if not params.flag_isolated_wet:
        return table

    wet_columns = [
        f"{params.denominator}_frac_{name}" for name in params.wet_classes
        if f"{params.denominator}_frac_{name}" in table.columns
    ]
    if not wet_columns:
        return table

    table["wet_fraction"] = table[wet_columns].sum(axis=1)

    for site_id, group in table.groupby("site_id"):
        ordered = group.sort_values(["year", "month"])
        wet = ordered["wet_fraction"].to_numpy()
        thin = (ordered["mean_n_obs"].to_numpy() < params.low_obs_threshold)

        previous = np.roll(wet, 1)
        following = np.roll(wet, -1)
        neighbours_dry = (previous < wet / 3) & (following < wet / 3)
        neighbours_dry[0] = neighbours_dry[-1] = False

        # A wet spike with dry months either side is only suspicious when the
        # median it came from was thin. A well-observed wet month is a rainfall
        # event, not an artefact.
        flagged = neighbours_dry & thin & (wet > 0.05)
        table.loc[ordered.index, "flag_isolated_wet"] = flagged

    return table


def save_table(table: pd.DataFrame, cfg: Config) -> Path:
    path = composition_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_table(cfg: Config) -> pd.DataFrame:
    path = composition_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no composition table at {path}; run the prediction first")
    return pd.read_csv(path, dtype={"site_id": str})


# --- dashboard assets ------------------------------------------------------


def export_boxes_geojson(cfg: Config, boxes: pd.DataFrame) -> Path:
    """All 187 boxes as WGS84 GeoJSON — the dashboard's map layer."""
    features = []
    for site_id, box in boxes.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [box["lon_min"], box["lat_min"]],
                    [box["lon_max"], box["lat_min"]],
                    [box["lon_max"], box["lat_max"]],
                    [box["lon_min"], box["lat_max"]],
                    [box["lon_min"], box["lat_min"]],
                ]],
            },
            "properties": {
                "site_id": site_id,
                "label": box["label"],
                "center_lon": float(box["center_lon"]),
                "center_lat": float(box["center_lat"]),
                "width_m": round(float(box["width_m"]), 1),
                "height_m": round(float(box["height_m"]), 1),
            },
        })

    path = cfg.paths["predictions"] / "waterhole_boxes.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def export_class_colours(cfg: Config) -> Path:
    """The class scheme as JSON, so the dashboard legend cannot drift from it."""
    path = cfg.paths["predictions"] / "class_colours.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scheme_version": cfg["classes"]["scheme_version"],
        "classes": [
            {"id": d.id, "name": d.name, "colour": d.colour, "ignore": d.ignore}
            for d in cfg.classes
        ],
    }, indent=1))
    return path
