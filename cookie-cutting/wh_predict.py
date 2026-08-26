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
import wh_tiles
import wh_train
from wh_config import Config
from wh_features import FeatureParams

WGS84 = "EPSG:4326"

# The display layers written beside each class raster. All three share one grid
# and one bounds.json, so a web map can stack or swap them freely.
#
#   pred  the classification, class 0 transparent
#   rgb   true colour — what the classifier actually saw
#   conf  max class probability — how much of it to believe
#
# rgb and conf exist because a class overlay on a basemap is unfalsifiable on its
# own: the basemap is a different sensor from a different year, so a viewer has
# no way to tell a correct classification from a confident wrong one. Flipping
# between the three answers "what was this made from, and how sure is it?"
PNG_LAYERS = ("pred", "rgb", "conf")

# Fixed so the notebook's confidence panel and the dashboard's cannot drift
# apart. The floor is 0.4 rather than 0 because a six-class model is never much
# below chance, and stretching from zero wastes most of the ramp.
CONFIDENCE_CMAP = "cividis"
CONFIDENCE_VMIN = 0.4
CONFIDENCE_VMAX = 1.0

# Display-image encoding. Measured on a 150x151 chip, per site-month:
#
#            PNG      WebP
#   pred    2.1 KB    0.9 KB   lossless
#   conf   18.6 KB    4.8 KB   lossless
#   rgb    49.4 KB    4.7 KB   lossy q80
#
# Across 15,708 site-months that is ~1.2 GB against ~150 MB. PNG is the default
# because it is what already exists on disk and it needs no decisions about
# browser support; WebP is what makes the whole archive comfortably static-
# hostable. Every browser since 2020 (Safari 14) reads WebP.
IMAGE_FORMATS = ("png", "webp")

# Only the true-colour layer is encoded lossily. Class and confidence images are
# flat colour, where lossy compression would invent intermediate colours that
# decode back to the wrong class or the wrong confidence.
WEBP_RGB_QUALITY = 80


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

    # Colourised PNGs beside each GeoTIFF. A browser cannot draw a GeoTIFF
    # without a tile server, but it can place a PNG as an image overlay given
    # bounds — which is what makes a backend-free dashboard possible.
    #
    # See PNG_LAYERS. Empty tuple writes none; ("pred",) is the old behaviour.
    # "conf" requires write_confidence.
    png_layers: tuple[str, ...] = PNG_LAYERS

    # "png" or "webp" — see IMAGE_FORMATS for the measured sizes. webp is ~8x
    # smaller across the archive and is what keeps it static-hostable.
    image_format: str = "png"

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

    def __post_init__(self) -> None:
        unknown = [layer for layer in self.png_layers if layer not in PNG_LAYERS]
        if unknown:
            raise ValueError(
                f"unknown png_layers {unknown}; choose from {list(PNG_LAYERS)}"
            )
        if self.image_format not in IMAGE_FORMATS:
            raise ValueError(
                f"unknown image_format {self.image_format!r}; "
                f"choose from {list(IMAGE_FORMATS)}"
            )
        if "conf" in self.png_layers and not self.write_confidence:
            # Refused rather than skipped: silently dropping a requested layer is
            # how you discover at the dashboard that the trust overlay is missing
            # and the archive needs another hour.
            raise ValueError(
                "png_layers includes 'conf' but write_confidence is False, so no "
                "confidence is computed to draw. Either set write_confidence=True "
                "or drop 'conf' from png_layers."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "majority_filter_px": self.majority_filter_px,
            "write_confidence": self.write_confidence,
            "png_layers": list(self.png_layers),
            "image_format": self.image_format,
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


def raster_path(
    cfg: Config,
    site_id: str,
    stem: str,
    kind: str = "pred",
    image_format: str = "png",
) -> Path:
    """Path of one output layer for one site-month.

    `kind` is "pred" or "conf" for the GeoTIFFs, or "pred_png" / "rgb_png" /
    "conf_png" for the display images — which take `image_format`, recorded in
    the site's bounds.json so the dashboard knows the extension. Every path is
    derivable from a site_id and a year_month, so nothing has to list a directory.
    """
    if kind.endswith("_png"):
        return site_dir(cfg, site_id) / f"{stem}_{kind[:-4]}.{image_format}"
    return site_dir(cfg, site_id) / f"{stem}_{kind}.tif"


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


def _confidence_rgba(confidence: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Colourise confidence, transparent wherever nothing was classified.

    Masked to the classified pixels rather than the whole tile so the confidence
    overlay covers exactly what the prediction overlay covers — flipping between
    them then compares like with like instead of shifting shape.
    """
    import matplotlib

    span = CONFIDENCE_VMAX - CONFIDENCE_VMIN
    scaled = np.clip(
        np.nan_to_num((confidence - CONFIDENCE_VMIN) / span, nan=0.0), 0, 1
    )
    rgba = (matplotlib.colormaps[CONFIDENCE_CMAP](scaled) * 255).round().astype(np.uint8)
    rgba[..., 3] = np.where((classes > 0) & np.isfinite(confidence), 255, 0)
    return rgba


def _rgb_rgba(tile) -> np.ndarray:
    """True colour on the prediction grid, transparent where unobserved.

    The same fixed stretch as the labelling PNGs and the notebook plots, so what
    the dashboard shows is what the labels were drawn on.
    """
    import wh_plots

    rgb = wh_plots.rgb_composite(tile)
    rgba = np.zeros(tuple(tile.shape) + (4,), dtype=np.uint8)
    rgba[..., :3] = (np.nan_to_num(rgb, nan=0.0) * 255).round().astype(np.uint8)
    rgba[..., 3] = np.where(tile.valid, 255, 0)
    return rgba


def _save_image(path: Path, rgba: np.ndarray, lossy: bool = False) -> None:
    """Write an RGBA array, encoded from the path's own extension.

    Only the true-colour layer passes lossy=True. Encoding a class or confidence
    image lossily would invent colours between the discrete ones, which decode
    back to the wrong class or the wrong confidence.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(rgba.astype(np.uint8)))

    if path.suffix == ".webp":
        if lossy:
            # alpha_quality=100 keeps the transparency mask crisp even though
            # the colour channels are lossy.
            image.save(path, "WEBP", quality=WEBP_RGB_QUALITY, alpha_quality=100)
        else:
            image.save(path, "WEBP", lossless=True)
    else:
        image.save(path, "PNG", optimize=True)


def write_class_png(path: Path, classes: np.ndarray, cfg: Config) -> None:
    """Colourised class image for display. Unclassified pixels stay transparent."""
    _save_image(path, _class_rgba(classes, cfg))


def write_rgb_png(path: Path, tile) -> None:
    """True-colour image, so a viewer can see what a classification was made from."""
    _save_image(path, _rgb_rgba(tile), lossy=True)


def write_confidence_png(path: Path, confidence: np.ndarray, classes: np.ndarray) -> None:
    """Confidence image, so a viewer can see how much to trust it."""
    _save_image(path, _confidence_rgba(confidence, classes))


def write_bounds(
    cfg: Config,
    site_id: str,
    tile,
    png_layers: tuple[str, ...] = PNG_LAYERS,
    image_format: str = "png",
) -> Path:
    """WGS84 bounds for the site, written once — every month shares one grid.

    This is what lets a web map place the PNGs as image overlays without a tile
    server, which is the whole reason the dashboard can be a static site. All
    layers share these bounds, so they can be stacked or swapped without
    recomputing anything.
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
        # Which overlays this site actually has, and their extension, so the UI
        # offers only what exists and can derive every filename.
        "png_layers": list(png_layers),
        "image_format": image_format,
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

        # Each artefact is gated on its own existence, not on the class raster's.
        # Otherwise adding a display layer to an already-predicted archive would
        # write nothing, because the .tif it used to be gated on is already there.
        def needed(path: Path) -> bool:
            return params.overwrite or not path.exists()

        if needed(target):
            _write_uint8(target, classes, tile, "class_id")
        if params.write_confidence and confidence is not None:
            conf_target = raster_path(cfg, site_id, stem, "conf")
            if needed(conf_target):
                _write_uint8(
                    conf_target,
                    np.nan_to_num(confidence * 100, nan=0).round(),
                    tile, "confidence_pct",
                )

        for layer in params.png_layers:
            png_target = raster_path(
                cfg, site_id, stem, f"{layer}_png", params.image_format
            )
            if not needed(png_target):
                continue
            if layer == "pred":
                write_class_png(png_target, classes, cfg)
            elif layer == "rgb":
                write_rgb_png(png_target, tile)
            elif layer == "conf" and confidence is not None:
                write_confidence_png(png_target, confidence, classes)

        if not wrote_bounds:
            write_bounds(cfg, site_id, tile, params.png_layers, params.image_format)
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


def site_is_done(
    cfg: Config,
    site_id: str,
    manifest: pd.DataFrame,
    params: PredictParams | None = None,
) -> bool:
    """Whether a site already has every artefact the params ask for.

    Counts each layer, not just the class raster: a site predicted before a
    display layer was added is not done under the new params, and reporting it
    as done is how the dashboard ends up missing overlays for half the sites.
    """
    params = params or PredictParams()
    expected = len(manifest[manifest["site_id"] == site_id])
    directory = site_dir(cfg, site_id)

    patterns = ["*_pred.tif"]
    if params.write_confidence:
        patterns.append("*_conf.tif")
    patterns.extend(
        f"*_{layer}.{params.image_format}" for layer in params.png_layers
    )

    return all(len(list(directory.glob(p))) >= expected for p in patterns)


class _Progress:
    """Incremental progress for a run that takes tens of minutes.

    A bar via tqdm when it is importable, and a throttled plain-text line when it
    is not — the fallback matters because this is the only window onto an hour of
    work, and it should never be the thing that fails.

    `write` exists so a failure can be reported the moment it happens without
    corrupting the bar. Holding failures until the end of an hour-long run means
    watching a progress bar advance while already-broken work accumulates behind
    it.
    """

    def __init__(self, total: int, description: str, verbose: bool = True):
        self.total = total
        self.verbose = verbose
        self.started = datetime.now(timezone.utc)
        self.position = 0
        self._last_print = 0.0
        self._bar = None

        if not verbose:
            return

        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, desc=description, unit="site",
                             smoothing=0.1, dynamic_ncols=True)
        except Exception:  # noqa: BLE001 — progress must never break the run
            print(f"{description}: {total} sites", flush=True)

    def update(self, count: int = 1, **postfix) -> None:
        self.position += count
        if not self.verbose:
            return

        if self._bar is not None:
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.update(count)
            return

        # Plain text: at most one line every 15 s, plus the last one, so an
        # hour-long run leaves a readable trail rather than 187 lines.
        elapsed = (datetime.now(timezone.utc) - self.started).total_seconds()
        if elapsed - self._last_print < 15 and self.position < self.total:
            return
        self._last_print = elapsed

        rate = self.position / elapsed if elapsed else 0.0
        remaining = (self.total - self.position) / rate / 60 if rate else float("nan")
        extra = "  " + "  ".join(f"{k}={v}" for k, v in postfix.items()) if postfix else ""
        print(f"  {self.position}/{self.total} sites  {elapsed / 60:5.1f} min elapsed  "
              f"~{remaining:4.0f} min left{extra}", flush=True)

    def write(self, text: str) -> None:
        if not self.verbose:
            return
        if self._bar is not None:
            self._bar.write(text)
        else:
            print(text, flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def elapsed_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.started).total_seconds() / 60


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

    Restartable rather than resumable: every site is predicted again, but a file
    that already exists is not rewritten unless overwrite is set. So an
    interrupted run finishes correctly, and a re-run after adding a display layer
    writes only that layer — at the cost of the prediction itself, which is where
    the time goes. Every site must be visited regardless, since the composition
    record is built from the prediction rather than read back from disk.
    """
    model, meta = wh_train.load_model(cfg, params.model_name)
    check_scheme(meta, cfg)
    boxes = wh_bbox.load_boxes(cfg)

    sites = sorted(params.sites or manifest["site_id"].unique())
    records: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    started = datetime.now(timezone.utc)
    todo = [site for site in sites if not site_is_done(cfg, site, manifest, params)]

    if verbose:
        print(f"{len(sites)} sites, {len(manifest[manifest['site_id'].isin(sites)]):,} "
              f"site-months", flush=True)
        if todo:
            print(f"{len(todo)} site(s) still need files written; "
                  f"{len(sites) - len(todo)} already complete", flush=True)
        else:
            print("every requested file already exists — this pass recomputes the "
                  "composition table only", flush=True)
        if params.workers > 1 and len(sites) > 1:
            # The first result takes noticeably longer than the rest: each worker
            # loads the config, manifest and model before its first site. Saying
            # so up front stops that looking like a hang.
            print(f"starting {params.workers} workers (each loads the model once, "
                  f"so the first sites take longer)", flush=True)

    progress = _Progress(len(sites), "predicting", verbose)

    def absorb(site_id: str, site_records: list, error: str) -> None:
        records.extend(site_records)
        if error:
            failures.append((site_id, error))
            # Reported as it happens: watching a bar advance for another 40
            # minutes while failures pile up invisibly helps nobody.
            progress.write(f"  site {site_id} FAILED: {error[:100]}")
        progress.update(months=len(records), failed=len(failures))

    if params.workers > 1 and len(sites) > 1:
        # joblib/loky rather than ProcessPoolExecutor: loky does not need
        # __main__ to be an importable file, which it is not inside a Jupyter
        # kernel, so this works from a notebook cell as well as a script.
        #
        # generator_unordered yields each site the moment it finishes rather than
        # returning a list at the end, which is what makes progress advance
        # during the run instead of all at once when it is over.
        from joblib import Parallel, delayed

        results = Parallel(
            n_jobs=params.workers, backend="loky", verbose=0,
            return_as="generator_unordered",
        )(
            delayed(_predict_one)(str(cfg.source_path), params, site_id)
            for site_id in sites
        )
        for site_id, site_records, error in results:
            absorb(site_id, site_records, error)
    else:
        for site_id in sites:
            try:
                absorb(
                    site_id,
                    predict_site(manifest, cfg, model, meta, params, site_id, boxes),
                    "",
                )
            except Exception as error:  # noqa: BLE001
                absorb(site_id, [], f"{type(error).__name__}: {error}")

    progress.close()

    if verbose:
        print(f"\n{len(records):,} site-months from {len(sites) - len(failures)} sites "
              f"in {progress.elapsed_minutes:.1f} min", flush=True)

    if failures:
        print(f"\n{len(failures)} site(s) failed:")
        for site_id, message in failures[:10]:
            print(f"  {site_id}: {message[:110]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("no site produced any predictions")

    table["model_name"] = params.model_name
    table["cv_macro_f1"] = meta.get("cv_macro_f1")
    table["config_hash"] = cfg.hash
    table["predicted_at"] = started.isoformat(timespec="seconds")

    table = add_quality_flags(table, cfg, params)
    return table.sort_values(["site_id", "year", "month"]).reset_index(drop=True)


# --- adding display layers to an archive already predicted -----------------


def backfill_site(
    manifest: pd.DataFrame, cfg: Config, params: PredictParams, site_id: str
) -> dict[str, int]:
    """Write one site's missing display PNGs from rasters that already exist.

    Every display layer is derivable from files on disk — the class raster, the
    confidence raster, and the source chip — so adding a layer to an archive that
    has already been predicted does not need the model, the features or the
    temporal stack. That is the difference between minutes and an hour, and it is
    why this exists separately from `run`.

    A month whose class raster is missing is counted and skipped, never invented.
    """
    rows = manifest[manifest["site_id"] == site_id]
    written = {layer: 0 for layer in params.png_layers}
    written["missing_pred"] = 0
    written["missing_conf"] = 0
    wrote_bounds = False

    for _, row in rows.iterrows():
        stem = Path(row["tif_path"]).stem
        pred_path = raster_path(cfg, site_id, stem, "pred")
        if not pred_path.exists():
            written["missing_pred"] += 1
            continue

        wanted = {
            layer: raster_path(
                cfg, site_id, stem, f"{layer}_png", params.image_format
            )
            for layer in params.png_layers
        }
        wanted = {
            layer: path for layer, path in wanted.items()
            if params.overwrite or not path.exists()
        }
        if not wanted and wrote_bounds:
            continue

        classes = None
        if {"pred", "conf"} & set(wanted):
            with rasterio.open(pred_path) as dataset:
                classes = dataset.read(1)

        if "pred" in wanted:
            write_class_png(wanted["pred"], classes, cfg)
            written["pred"] += 1

        # The chip is read only when something actually needs it.
        if "rgb" in wanted or not wrote_bounds:
            tile = wh_tiles.read_tile(row["tif_path"], cfg)
            if "rgb" in wanted:
                write_rgb_png(wanted["rgb"], tile)
                written["rgb"] += 1
            if not wrote_bounds:
                write_bounds(
                    cfg, site_id, tile, params.png_layers, params.image_format
                )
                wrote_bounds = True

        if "conf" in wanted:
            conf_path = raster_path(cfg, site_id, stem, "conf")
            if not conf_path.exists():
                written["missing_conf"] += 1
            else:
                with rasterio.open(conf_path) as dataset:
                    confidence = dataset.read(1).astype(np.float32) / 100.0
                write_confidence_png(wanted["conf"], confidence, classes)
                written["conf"] += 1

    return written


def _backfill_one(
    config_path: str, params: PredictParams, site_id: str
) -> tuple[str, dict[str, int], str]:
    """Worker entry point for backfill_pngs."""
    try:
        import wh_config as config_module
        import wh_inventory as inventory_module

        cfg = config_module.load(config_path)
        manifest = inventory_module.load_manifest(cfg)
        return site_id, backfill_site(manifest, cfg, params, site_id), ""
    except Exception as error:  # noqa: BLE001 — one bad site must not stop the run
        return site_id, {}, f"{type(error).__name__}: {error}"


def backfill_pngs(
    manifest: pd.DataFrame,
    cfg: Config,
    params: PredictParams,
    verbose: bool = True,
) -> dict[str, int]:
    """Add the display PNGs to an archive that has already been predicted.

    Use this after adding a layer to `png_layers`; use `run` when the class
    rasters themselves need to change. Returns totals per layer.
    """
    sites = sorted(params.sites or manifest["site_id"].unique())
    totals: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    if verbose:
        print(f"backfilling {list(params.png_layers)} as {params.image_format} "
              f"for {len(sites)} sites", flush=True)

    progress = _Progress(len(sites), "backfilling", verbose)

    def absorb(site_id: str, counts: dict[str, int], error: str) -> None:
        if error:
            failures.append((site_id, error))
            progress.write(f"  site {site_id} FAILED: {error[:100]}")
        for layer, count in counts.items():
            totals[layer] = totals.get(layer, 0) + count
        written = sum(v for k, v in totals.items() if not k.startswith("missing_"))
        progress.update(images=written)

    if params.workers > 1 and len(sites) > 1:
        from joblib import Parallel, delayed

        results = Parallel(
            n_jobs=params.workers, backend="loky", verbose=0,
            return_as="generator_unordered",
        )(
            delayed(_backfill_one)(str(cfg.source_path), params, site_id)
            for site_id in sites
        )
        for site_id, counts, error in results:
            absorb(site_id, counts, error)
    else:
        for site_id in sites:
            absorb(site_id, backfill_site(manifest, cfg, params, site_id), "")

    progress.close()

    if failures:
        print(f"\n{len(failures)} site(s) failed:")
        for site_id, message in failures[:10]:
            print(f"  {site_id}: {message[:110]}")

    if verbose:
        wrote = {k: v for k, v in totals.items() if not k.startswith("missing_")}
        print(f"\nwrote {wrote} in {progress.elapsed_minutes:.1f} min", flush=True)

        if totals.get("missing_pred"):
            print(f"\n{totals['missing_pred']} month(s) have no class raster — "
                  f"run() has not covered those sites yet.")
        if totals.get("missing_conf"):
            # The likely cause, and the one that matters: an archive predicted
            # with write_confidence=False has no confidence to draw, and no
            # amount of backfilling can invent it.
            print(f"\n{totals['missing_conf']} month(s) have no confidence raster, "
                  f"so they have no confidence overlay.\nConfidence cannot be "
                  f"backfilled — it comes from predict_proba, not from the class\n"
                  f"raster. To get it, re-run run() with write_confidence=True.")
    return totals


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


def export_footprints_geojson(cfg: Config, sites: list[str] | None = None) -> Path:
    """Every derived basin footprint as one WGS84 GeoJSON — a second map layer.

    The per-site files `wh_footprint` writes are already WGS84, but there are 176
    of them and each carries the full parameter set that derived it. This
    combines them and keeps only what a map needs, so the dashboard fetches one
    file the size of the boxes layer rather than 176.

    Outlines belong here, as vectors, rather than drawn into the overlay images:
    a site's box and footprint are the same for all 84 of its months, so baking
    them into 15,708 images would repeat them needlessly, destroy the pixels
    underneath, and leave them impossible to toggle off.
    """
    features = []
    missing = []

    for path in sorted(cfg.paths["derived"].glob("footprints/*_footprint.geojson")):
        collection = json.loads(path.read_text())
        for feature in collection.get("features", []):
            source = feature.get("properties", {})
            site_id = source.get("site_id")
            if sites is not None and site_id not in sites:
                continue
            area_m2 = float(source.get("area_m2", 0.0))
            features.append({
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "site_id": site_id,
                    "n_pixels": source.get("n_pixels"),
                    "area_m2": area_m2,
                    "area_ha": round(area_m2 / 1e4, 3),
                    "succeeded": source.get("succeeded"),
                    "notes": source.get("notes", ""),
                },
            })

    found = {feature["properties"]["site_id"] for feature in features}
    if sites is not None:
        missing = sorted(set(sites) - found)

    out = cfg.paths["predictions"] / "waterhole_footprints.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "type": "FeatureCollection",
        # Named so the dashboard can say which sites have no footprint rather
        # than silently drawing 176 outlines over 187 boxes.
        "sites_without_footprint": missing,
        "features": features,
    }))
    return out


def export_class_colours(cfg: Config) -> Path:
    """The class scheme as JSON, so the dashboard legend cannot drift from it.

    Carries the confidence ramp too: the confidence PNGs are already colourised,
    so the dashboard needs the stops to draw a colourbar but has no way to
    recompute them.
    """
    import matplotlib
    import matplotlib.colors as mcolors

    colour_map = matplotlib.colormaps[CONFIDENCE_CMAP]
    stops = [mcolors.to_hex(colour_map(value)) for value in np.linspace(0, 1, 9)]

    path = cfg.paths["predictions"] / "class_colours.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scheme_version": cfg["classes"]["scheme_version"],
        "classes": [
            {"id": d.id, "name": d.name, "colour": d.colour, "ignore": d.ignore}
            for d in cfg.classes
        ],
        "confidence": {
            "cmap": CONFIDENCE_CMAP,
            "vmin": CONFIDENCE_VMIN,
            "vmax": CONFIDENCE_VMAX,
            "stops": stops,
            "note": "max class probability; values are clipped to [vmin, vmax]",
        },
    }, indent=1))
    return path
