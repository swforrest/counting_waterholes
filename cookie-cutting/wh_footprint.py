"""Derive one basin footprint per site from multi-year seasonal behaviour.

The footprint answers "where is the waterhole", once, so that every month only
has to answer "what state is it in". It also gives composition fractions a
denominator that does not move through time.

Deliberately NOT a threshold on maximum MNDWI. Emergent sedges and melaleuca
routinely cover standing water at these sites and drag MNDWI far negative, so a
water-index threshold would give a sedge-choked basin no footprint at all —
exactly the sites that matter most. Instead a basin is found by how a pixel
BEHAVES across years:

  * seasonal range (wet max - dry min) of several indices — the basin swings
    with the season far more than the savanna matrix does;
  * dry-season NDVI anomaly — a vegetated basin stays green while the matrix
    browns off, which is the strongest single signal for the hard sites.

Both are measured as robust z-scores against the tile's own matrix, so no
absolute threshold has to hold across sites with different soils and cover.

Parameters are passed in explicitly rather than read from the config, so the
footprint notebook can hold them where they are visible and editable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.warp import transform_geom
from scipy import ndimage

import wh_temporal
from wh_config import Config
from wh_temporal import SiteStack

# Scale factor making the median absolute deviation a consistent estimator of
# the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826


@dataclass
class FootprintParams:
    """Everything tunable about footprint derivation.

    Held as a dataclass so the notebook can show and edit the values in one
    visible block, and so a run records exactly what produced it.
    """

    seasonal_range_indices: tuple[str, ...] = ("mndwi", "ndvi", "ndti")
    seasonal_range_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    dry_ndvi_anomaly_weight: float = 1.5
    score_threshold: float = 2.0
    min_basin_pixels: int = 4
    closing_radius_px: int = 1
    buffer_px: int = 3
    seed_search_radius_px: int = 15
    min_valid_months: int = 24
    max_basin_fraction: float = 0.25

    def as_dict(self) -> dict[str, object]:
        return {
            "seasonal_range_indices": list(self.seasonal_range_indices),
            "seasonal_range_weights": list(self.seasonal_range_weights),
            "dry_ndvi_anomaly_weight": self.dry_ndvi_anomaly_weight,
            "score_threshold": self.score_threshold,
            "min_basin_pixels": self.min_basin_pixels,
            "closing_radius_px": self.closing_radius_px,
            "buffer_px": self.buffer_px,
            "seed_search_radius_px": self.seed_search_radius_px,
            "min_valid_months": self.min_valid_months,
            "max_basin_fraction": self.max_basin_fraction,
        }


@dataclass
class Footprint:
    """The derived footprint for one site, plus the layers that produced it."""

    site_id: str
    mask: np.ndarray  # bool, the buffered basin
    core_mask: np.ndarray  # bool, before buffering
    score: np.ndarray  # float, combined robust z-score
    components: dict[str, np.ndarray] = field(default_factory=dict)
    n_pixels: int = 0
    area_m2: float = 0.0
    succeeded: bool = True
    reason: str = ""
    transform: object = None
    crs: object = None
    params: FootprintParams | None = None

    @property
    def fraction_of_tile(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0


def robust_z(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Z-score against the tile's own median and MAD.

    Median and MAD rather than mean and SD because the basin itself is the
    outlier we are looking for — letting it into the centre and spread estimate
    would hide it. Comparing each pixel to its own tile also removes the
    between-site variation in soil and cover that no absolute threshold could
    survive.
    """
    finite = np.isfinite(values)
    if valid is not None:
        finite &= valid

    if not finite.any():
        return np.full(values.shape, np.nan)

    sample = values[finite]
    centre = np.median(sample)
    spread = MAD_TO_SIGMA * np.median(np.abs(sample - centre))

    if spread <= 0:
        # A flat tile has no anomalies to find; say so rather than dividing by
        # zero and returning infinities that would pass any threshold.
        return np.where(finite, 0.0, np.nan)

    return np.where(finite, (values - centre) / spread, np.nan)


def basin_score(
    features: dict[str, np.ndarray],
    params: FootprintParams,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine seasonal-range and dry-greenness anomalies into one score.

    Returns the weighted-mean z-score and the individual layers, so the notebook
    can plot what each contributed rather than trusting a single number.
    """
    layers: dict[str, np.ndarray] = {}
    weighted_sum = None
    total_weight = 0.0

    if len(params.seasonal_range_weights) != len(params.seasonal_range_indices):
        raise ValueError(
            f"{len(params.seasonal_range_weights)} weights for "
            f"{len(params.seasonal_range_indices)} indices"
        )

    for index_name, weight in zip(
        params.seasonal_range_indices, params.seasonal_range_weights
    ):
        key = f"{index_name}_seasonal_range"
        if key not in features:
            raise KeyError(
                f"{key} not in the temporal features; load the site stack with "
                f"indices including {index_name!r}"
            )
        layer = robust_z(features[key], valid)
        layers[key] = layer
        contribution = np.where(np.isfinite(layer), layer, 0.0) * weight
        weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        total_weight += weight

    # Dry-season greenness anomaly. Positive means greener than the matrix while
    # the matrix is browning off, which is what a vegetated basin looks like.
    if params.dry_ndvi_anomaly_weight > 0:
        if "ndvi_dry_median" not in features:
            raise KeyError(
                "ndvi_dry_median not in the temporal features; recompute them "
                "with a wh_temporal that emits seasonal medians"
            )
        layer = robust_z(features["ndvi_dry_median"], valid)
        layers["ndvi_dry_anomaly"] = layer
        weighted_sum = weighted_sum + np.where(np.isfinite(layer), layer, 0.0) * (
            params.dry_ndvi_anomaly_weight
        )
        total_weight += params.dry_ndvi_anomaly_weight

    score = weighted_sum / total_weight
    any_finite = np.any(
        [np.isfinite(layer) for layer in layers.values()], axis=0
    )
    return np.where(any_finite, score, np.nan), layers


def _disk(radius: int) -> np.ndarray:
    """Circular structuring element of the given radius in pixels."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    span = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    return (yy**2 + xx**2) <= radius**2


def _select_central_component(
    candidate: np.ndarray, params: FootprintParams
) -> tuple[np.ndarray, str]:
    """Keep only the basin at the tile centre, where the AOI was seeded.

    Tiles routinely contain more than one basin — site 025 has three. The AOI
    was buffered from a labelled waterhole centre, so the central component is
    the site; the others belong to different sites or to nothing.
    """
    labelled, n_components = ndimage.label(candidate)
    if n_components == 0:
        return np.zeros_like(candidate), "no pixels passed the score threshold"

    height, width = candidate.shape
    centre = (height // 2, width // 2)

    # Prefer a component containing the centre; otherwise the nearest one within
    # the search radius, since the labelled centre is only accurate to the
    # original bounding box.
    label_at_centre = labelled[centre]
    if label_at_centre != 0:
        chosen = label_at_centre
    else:
        distance, indices = ndimage.distance_transform_edt(
            labelled == 0, return_indices=True
        )
        if distance[centre] > params.seed_search_radius_px:
            return (
                np.zeros_like(candidate),
                f"nearest candidate is {distance[centre]:.0f} px from the tile "
                f"centre, beyond the {params.seed_search_radius_px} px search radius",
            )
        chosen = labelled[indices[0][centre], indices[1][centre]]

    selected = labelled == chosen
    if selected.sum() < params.min_basin_pixels:
        return (
            np.zeros_like(candidate),
            f"central component is {selected.sum()} px, below the "
            f"{params.min_basin_pixels} px minimum",
        )

    return selected, ""


def derive_footprint(
    stack: SiteStack,
    features: dict[str, np.ndarray],
    params: FootprintParams,
) -> Footprint:
    """Derive the basin footprint for one site.

    Returns a Footprint with succeeded=False and a reason when no plausible
    basin is found, rather than an empty mask that would silently become a
    zero denominator downstream.
    """
    # A pixel needs enough observed months for its seasonal statistics to mean
    # anything. The reprojection border and persistent cloud both fail here.
    valid_months = np.isfinite(stack.stacks["mndwi"]).sum(axis=0)
    valid = valid_months >= params.min_valid_months

    score, layers = basin_score(features, params, valid=valid)

    candidate = np.isfinite(score) & (score >= params.score_threshold) & valid

    if params.closing_radius_px > 0:
        candidate = ndimage.binary_closing(
            candidate, structure=_disk(params.closing_radius_px)
        )

    core, reason = _select_central_component(candidate, params)

    footprint = Footprint(
        site_id=stack.site_id,
        mask=core.copy(),
        core_mask=core,
        score=score,
        components=layers,
        transform=stack.transform,
        crs=stack.crs,
        params=params,
    )

    if reason:
        footprint.succeeded = False
        footprint.reason = reason
        return footprint

    mask = core
    if params.buffer_px > 0:
        mask = ndimage.binary_dilation(core, structure=_disk(params.buffer_px))
    mask &= valid

    fraction = float(mask.mean())
    if fraction > params.max_basin_fraction:
        footprint.succeeded = False
        footprint.reason = (
            f"basin covers {100 * fraction:.0f}% of the tile, above the "
            f"{100 * params.max_basin_fraction:.0f}% ceiling — the score "
            f"threshold is probably too low for this site"
        )
        footprint.mask = mask
        return footprint

    pixel_area = abs(stack.transform.a * stack.transform.e)

    footprint.mask = mask
    footprint.n_pixels = int(mask.sum())
    footprint.area_m2 = float(mask.sum() * pixel_area)
    return footprint


# --- persistence -----------------------------------------------------------


def mask_path(cfg: Config, site_id: str) -> Path:
    return cfg.paths["derived"] / "footprints" / f"site_{site_id}_footprint.tif"


def geojson_path(cfg: Config, site_id: str) -> Path:
    return cfg.paths["derived"] / "footprints" / f"site_{site_id}_footprint.geojson"


def save_footprint(footprint: Footprint, cfg: Config) -> tuple[Path, Path]:
    """Write the footprint as an aligned uint8 raster and a WGS84 GeoJSON."""
    raster_path = mask_path(cfg, footprint.site_id)
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=footprint.mask.shape[0],
        width=footprint.mask.shape[1],
        count=1,
        dtype="uint8",
        crs=footprint.crs,
        transform=footprint.transform,
        nodata=None,
        compress="deflate",
    ) as dataset:
        dataset.write(footprint.mask.astype(np.uint8), 1)
        dataset.descriptions = ("basin_footprint",)

    vector_path = geojson_path(cfg, footprint.site_id)
    vector_path.write_text(json.dumps(_to_geojson(footprint), indent=1))

    return raster_path, vector_path


def _to_geojson(footprint: Footprint) -> dict[str, object]:
    """Polygonise the mask and reproject to WGS84 for interoperability."""
    shapes = list(
        rio_features.shapes(
            footprint.mask.astype(np.uint8),
            mask=footprint.mask,
            transform=footprint.transform,
        )
    )

    features_out = []
    for geometry, value in shapes:
        if value != 1:
            continue
        wgs84 = transform_geom(footprint.crs, "EPSG:4326", geometry, precision=7)
        features_out.append(
            {
                "type": "Feature",
                "geometry": wgs84,
                "properties": {
                    "site_id": footprint.site_id,
                    "n_pixels": footprint.n_pixels,
                    "area_m2": round(footprint.area_m2, 1),
                    "source_crs": str(footprint.crs),
                    "succeeded": footprint.succeeded,
                    **(footprint.params.as_dict() if footprint.params else {}),
                },
            }
        )

    return {"type": "FeatureCollection", "features": features_out}


def load_mask(cfg: Config, site_id: str) -> np.ndarray:
    """Read a saved footprint mask as a boolean array."""
    path = mask_path(cfg, site_id)
    if not path.exists():
        raise FileNotFoundError(f"no footprint at {path}; derive it first")
    with rasterio.open(path) as dataset:
        return dataset.read(1).astype(bool)


def run_site(
    manifest,
    site_id: str,
    cfg: Config,
    params: FootprintParams,
    indices: list[str] | None = None,
) -> tuple[Footprint, SiteStack, dict[str, np.ndarray]]:
    """Load a site, compute its temporal features, and derive its footprint.

    Returns the stack and features too, so the notebook can plot the layers that
    produced the result without recomputing them.
    """
    if indices is None:
        indices = sorted(set(params.seasonal_range_indices) | {"ndvi", "mndwi"})

    stack = wh_temporal.load_site_stack(manifest, site_id, cfg, indices=indices)
    features = wh_temporal.temporal_feature_stack(stack, cfg)
    footprint = derive_footprint(stack, features, params)
    return footprint, stack, features
