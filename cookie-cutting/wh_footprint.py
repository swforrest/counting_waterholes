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
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.warp import transform_geom
from scipy import ndimage

import wh_features
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

    # Indices whose seasonal range feeds the basin score, mapped to their weight.
    # One dict rather than parallel lists of names and weights, so dropping an
    # index cannot leave the two out of step.
    seasonal_range_weights: dict[str, float] = field(
        default_factory=lambda: {"mndwi": 1.0, "ndvi": 1.0, "ndti": 1.0}
    )
    dry_ndvi_anomaly_weight: float = 1.5

    # --- AlphaEarth embeddings ---------------------------------------------
    # The embeddings are ANNUAL and static, so they cannot contribute a seasonal
    # range like the other layers. What they contribute instead is a SPATIAL
    # anomaly: how unlike the surrounding savanna a pixel's embedding vector is.
    # See embedding_anomaly() for the construction.
    use_alphaearth: bool = False
    alphaearth_year: int = 2025
    # None uses all 64 bands. A subset is legitimate, but note that bands ranked
    # important for pixel CLASSIFICATION are not necessarily the ones that
    # delineate a basin — those are different questions.
    alphaearth_bands: tuple[str, ...] | None = None
    alphaearth_weight: float = 1.5
    score_threshold: float = 2.0
    min_basin_pixels: int = 4
    closing_radius_px: int = 1
    buffer_px: int = 3
    seed_search_radius_px: int = 15
    min_valid_months: int = 24
    max_basin_fraction: float = 0.25

    # Reduce the footprint to one connected region, keeping the largest.
    #
    # Without this a site can end up as a main basin plus a scatter of stray
    # pixels that survived the score threshold nearby, which polygonises into
    # several features and makes "the waterhole's area" ambiguous. The fragments
    # are rarely the waterhole; they are speckle in the anomaly score.
    single_component: bool = True

    @property
    def seasonal_range_indices(self) -> tuple[str, ...]:
        """Indices contributing a seasonal range, in dict order."""
        return tuple(self.seasonal_range_weights)

    def as_dict(self) -> dict[str, object]:
        return {
            "seasonal_range_weights": dict(self.seasonal_range_weights),
            "dry_ndvi_anomaly_weight": self.dry_ndvi_anomaly_weight,
            "score_threshold": self.score_threshold,
            "min_basin_pixels": self.min_basin_pixels,
            "closing_radius_px": self.closing_radius_px,
            "buffer_px": self.buffer_px,
            "seed_search_radius_px": self.seed_search_radius_px,
            "min_valid_months": self.min_valid_months,
            "max_basin_fraction": self.max_basin_fraction,
            "single_component": self.single_component,
            "use_alphaearth": self.use_alphaearth,
            "alphaearth_year": self.alphaearth_year,
            "alphaearth_bands": (
                list(self.alphaearth_bands) if self.alphaearth_bands else "all"
            ),
            "alphaearth_weight": self.alphaearth_weight,
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
    reason: str = ""  # why it FAILED; empty on success
    notes: str = ""  # anything worth saying about a footprint that did succeed
    transform: object = None
    crs: object = None
    params: FootprintParams | None = None
    box_mask: np.ndarray | None = None  # this site's extent, when one was used

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


def embedding_anomaly(
    alphaearth: dict[str, np.ndarray],
    valid: np.ndarray | None = None,
    bands: tuple[str, ...] | None = None,
) -> np.ndarray:
    """How unlike the surrounding savanna each pixel's embedding vector is.

    The other score layers measure how much a pixel CHANGES through the year.
    The embeddings are annual and static, so they cannot say that — but they can
    say how far a pixel sits from the landscape around it, which is a different
    and complementary piece of evidence. A basin that never floods visibly still
    looks unlike savanna to a model trained on a year of multi-sensor data.

    Construction, per tile:

      1. each band is centred on its own median and scaled by its own MAD, so a
         high-variance dimension cannot dominate purely by being large;
      2. the per-pixel distance is the ROOT MEAN square across bands, not the
         sum — that keeps the magnitude comparable whether 3 bands are selected
         or all 64, so changing the subset does not silently rescale the layer's
         weight in the combined score;
      3. the result is robust-z'd like every other layer, against the tile's own
         median and MAD.

    Median and MAD are taken over the whole tile, which is mostly matrix — the
    basin is the outlier being looked for, and letting it into its own baseline
    would hide it.
    """
    names = sorted(bands and [f"{wh_features.ALPHAEARTH_PREFIX}{b}" for b in bands]
                   or alphaearth)
    missing = [name for name in names if name not in alphaearth]
    if missing:
        raise KeyError(f"embedding bands not loaded: {missing}")

    scaled = []
    for name in names:
        values = alphaearth[name].astype(np.float64)
        finite = np.isfinite(values)
        if valid is not None:
            finite &= valid
        if not finite.any():
            continue

        sample = values[finite]
        centre = np.median(sample)
        spread = MAD_TO_SIGMA * np.median(np.abs(sample - centre))
        if spread <= 0:
            continue
        scaled.append(np.where(finite, (values - centre) / spread, np.nan))

    if not scaled:
        raise ValueError("no embedding band on this tile had any usable spread")

    stack = np.stack(scaled, axis=0)

    # Pixels excluded by `valid` — too few observed months for their seasonal
    # statistics to mean anything — are NaN in every band, so there is nothing to
    # average and numpy warns. NaN is the right answer for them and robust_z
    # keeps it, so the warning is expected rather than a symptom.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        distance = np.sqrt(np.nanmean(stack**2, axis=0))

    return robust_z(distance, valid)


def basin_score(
    features: dict[str, np.ndarray],
    params: FootprintParams,
    valid: np.ndarray | None = None,
    alphaearth: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine seasonal-range and dry-greenness anomalies into one score.

    Returns the weighted-mean z-score and the individual layers, so the notebook
    can plot what each contributed rather than trusting a single number.
    """
    layers: dict[str, np.ndarray] = {}
    weighted_sum = None
    total_weight = 0.0

    has_embeddings = alphaearth is not None and params.alphaearth_weight > 0
    if (
        not params.seasonal_range_weights
        and params.dry_ndvi_anomaly_weight <= 0
        and not has_embeddings
    ):
        raise ValueError(
            "the basin score has no contributing layers: set at least one entry "
            "in seasonal_range_weights, a positive dry_ndvi_anomaly_weight, or "
            "enable the AlphaEarth layer"
        )

    for index_name, weight in params.seasonal_range_weights.items():
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
        contribution = (
            np.where(np.isfinite(layer), layer, 0.0) * params.dry_ndvi_anomaly_weight
        )
        # weighted_sum is still None if seasonal_range_weights was left empty and
        # the dry anomaly is the only contributing layer.
        weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        total_weight += params.dry_ndvi_anomaly_weight

    if has_embeddings:
        layer = embedding_anomaly(alphaearth, valid, params.alphaearth_bands)
        layers["alphaearth_anomaly"] = layer
        contribution = (
            np.where(np.isfinite(layer), layer, 0.0) * params.alphaearth_weight
        )
        weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        total_weight += params.alphaearth_weight

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
    candidate: np.ndarray,
    params: FootprintParams,
    box_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Keep only this site's basin, discarding neighbouring waterholes.

    Tiles routinely contain more than one basin — 93 of 187 contain more than one
    waterhole, and site 025 has three. Two ways to pick the right one:

    With a `box_mask` (this site's labelled extent, buffered), the choice is made
    on evidence: keep components that intersect the box. That is strictly better
    than guessing from distance, and it is why `seed_search_radius_px` exists only
    as the fallback.

    Without one, fall back to the component at or nearest the tile centre, since
    the AOI was buffered from the labelled waterhole centre.
    """
    labelled, n_components = ndimage.label(candidate)
    if n_components == 0:
        return np.zeros_like(candidate), "no pixels passed the score threshold"

    if box_mask is not None:
        return _select_by_box(labelled, box_mask, params)

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


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keep only the largest connected region.

    Returns (mask, n_regions_dropped, pixels_dropped) so the caller can say what
    was discarded rather than silently shrinking the footprint.
    """
    if not mask.any():
        return mask, 0, 0

    labelled, n_components = ndimage.label(mask)
    if n_components <= 1:
        return mask, 0, 0

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background
    keep = int(sizes.argmax())

    largest = labelled == keep
    return largest, n_components - 1, int(mask.sum() - largest.sum())


def _select_by_box(
    labelled: np.ndarray, box_mask: np.ndarray, params: FootprintParams
) -> tuple[np.ndarray, str]:
    """Keep the components that fall inside this site's labelled extent.

    Several components may legitimately belong to one waterhole — a basin broken
    into fragments by a bar of vegetation, say — so every component intersecting
    the box is kept rather than only the largest. Components belonging to a
    neighbouring waterhole are excluded because they sit outside the box.
    """
    inside = np.unique(labelled[box_mask & (labelled > 0)])
    if inside.size == 0:
        return (
            np.zeros_like(labelled, dtype=bool),
            "no candidate pixels fall inside this site's bounding box",
        )

    selected = np.isin(labelled, inside) & box_mask
    if selected.sum() < params.min_basin_pixels:
        return (
            np.zeros_like(labelled, dtype=bool),
            f"in-box component is {int(selected.sum())} px, below the "
            f"{params.min_basin_pixels} px minimum",
        )

    return selected, ""


def derive_footprint(
    stack: SiteStack,
    features: dict[str, np.ndarray],
    params: FootprintParams,
    box_mask: np.ndarray | None = None,
    alphaearth: dict[str, np.ndarray] | None = None,
) -> Footprint:
    """Derive the basin footprint for one site.

    `box_mask` is this site's labelled extent from wh_bbox, buffered. When given,
    it decides which connected component belongs to this waterhole rather than
    the tile-centre heuristic, and bounds the final footprint — so a neighbouring
    waterhole in the same tile cannot leak into this site's composition fractions.

    Note it constrains *selection* only. The anomaly scores are still computed
    against the whole tile, because the buffered box is a median 10% of a chip
    and there is nowhere near enough matrix inside it to estimate a baseline.

    Returns a Footprint with succeeded=False and a reason when no plausible
    basin is found, rather than an empty mask that would silently become a
    zero denominator downstream.
    """
    # A pixel needs enough observed months for its seasonal statistics to mean
    # anything. The reprojection border and persistent cloud both fail here.
    valid_months = np.isfinite(stack.stacks["mndwi"]).sum(axis=0)
    valid = valid_months >= params.min_valid_months

    score, layers = basin_score(features, params, valid=valid, alphaearth=alphaearth)

    candidate = np.isfinite(score) & (score >= params.score_threshold) & valid

    if params.closing_radius_px > 0:
        candidate = ndimage.binary_closing(
            candidate, structure=_disk(params.closing_radius_px)
        )

    if box_mask is not None and box_mask.shape != candidate.shape:
        raise ValueError(
            f"box mask shape {box_mask.shape} does not match the tile {candidate.shape}"
        )

    core, reason = _select_central_component(candidate, params, box_mask)

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
    if box_mask is not None:
        # The dilation must not push the footprint back out past this site's own
        # extent and into a neighbouring waterhole.
        mask &= box_mask

    # Applied last, after dilation and both maskings — each of which can split a
    # region or leave fragments behind — so the saved footprint is exactly one
    # polygon and "the waterhole's area" is unambiguous.
    if params.single_component:
        mask, n_dropped, dropped_px = _largest_component(mask)
        if n_dropped:
            footprint.notes = (
                f"kept the largest of {n_dropped + 1} regions "
                f"({dropped_px} px discarded as fragments)"
            )

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


# --- picking pixels to inspect --------------------------------------------
#
# Both helpers fall back rather than raising when a footprint is empty. A site
# whose footprint FAILED is exactly the site you most want to plot pixels from,
# to see why it failed — so refusing to pick one would be backwards.


def strongest_pixel(footprint: Footprint) -> tuple[int, int, str]:
    """The most basin-like pixel: highest score inside the footprint if there is
    one, otherwise the highest anywhere on the tile.

    Returns (row, col, note) where note says which rule applied.
    """
    if footprint.mask.any():
        scores = np.where(footprint.mask, footprint.score, np.nan)
        note = "strongest pixel inside the footprint"
    else:
        scores = footprint.score
        note = "no footprint — strongest pixel anywhere on the tile"

    if not np.isfinite(scores).any():
        raise ValueError(
            f"site {footprint.site_id}: no finite basin score anywhere; the tile "
            f"is probably unobserved throughout"
        )

    row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
    return int(row), int(col), note


def background_pixel(
    footprint: Footprint,
    observed_months: np.ndarray,
    min_distance_px: int = 25,
    min_months: int = 70,
) -> tuple[int, int, str]:
    """A well-observed savanna-matrix pixel, for contrast against a basin pixel.

    Prefers a well-observed pixel far from the footprint. With no footprint to
    move away from, falls back to the lowest-scoring well-observed pixel, which
    is the least basin-like thing on the tile.
    """
    well_observed = observed_months >= min_months
    if not well_observed.any():
        well_observed = observed_months >= np.nanpercentile(observed_months, 75)

    if footprint.mask.any():
        distance = ndimage.distance_transform_edt(~footprint.mask)
        candidates = np.where(well_observed & (distance > min_distance_px), distance, np.nan)
        if np.isfinite(candidates).any():
            row, col = np.unravel_index(np.nanargmax(candidates), candidates.shape)
            return int(row), int(col), f"matrix pixel >{min_distance_px} px from the footprint"

    scores = np.where(well_observed, footprint.score, np.nan)
    if not np.isfinite(scores).any():
        raise ValueError(f"site {footprint.site_id}: no well-observed pixel to compare against")

    row, col = np.unravel_index(np.nanargmin(scores), scores.shape)
    return int(row), int(col), "lowest-scoring well-observed pixel"


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
                    "notes": footprint.notes,
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
    use_box: bool = True,
) -> tuple[Footprint, SiteStack, dict[str, np.ndarray]]:
    """Load a site, compute its temporal features, and derive its footprint.

    Uses the site's bounding-box mask when one has been built, falling back to
    the tile-centre heuristic when it has not — so this keeps working before
    wh_bbox has been run, and improves once it has.

    Returns the stack and features too, so the notebook can plot the layers that
    produced the result without recomputing them.
    """
    if indices is None:
        indices = sorted(set(params.seasonal_range_indices) | {"ndvi", "mndwi"})

    stack = wh_temporal.load_site_stack(manifest, site_id, cfg, indices=indices)
    features = wh_temporal.temporal_feature_stack(stack, cfg)

    box_mask = None
    if use_box:
        import wh_bbox

        try:
            box_mask = wh_bbox.load_mask(cfg, site_id)
        except FileNotFoundError:
            box_mask = None
        if box_mask is not None and box_mask.shape != stack.shape:
            raise ValueError(
                f"site {site_id}: box mask {box_mask.shape} does not match the "
                f"tile grid {stack.shape}; rebuild the box masks"
            )

    alphaearth = None
    if params.use_alphaearth:
        import wh_features

        feature_params = wh_features.FeatureParams(
            use_alphaearth=True,
            alphaearth_year=params.alphaearth_year,
            alphaearth_bands=params.alphaearth_bands,
        )
        alphaearth = wh_features.load_alphaearth(
            cfg, site_id, feature_params, expected_shape=stack.shape
        )

    footprint = derive_footprint(
        stack, features, params, box_mask=box_mask, alphaearth=alphaearth
    )
    footprint.box_mask = box_mask
    return footprint, stack, features
