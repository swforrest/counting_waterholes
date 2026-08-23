"""Assemble the per-pixel feature table.

Three groups of features per pixel, per month:

  * instantaneous — reflectance bands and spectral indices from that month's tile;
  * local context — windowed mean and SD of a couple of indices, so a pixel knows
    whether it sits in a uniform patch or on an edge;
  * temporal — that pixel's own history, from wh_temporal.

The temporal group is the point of the whole design: a pixel reading MNDWI -0.4
in October says almost nothing on its own, but the same pixel reaching +0.5 every
February is unambiguously a basin floor.

Everything here is NaN-aware. A month with no clear observation must not enter a
statistic as if it were a reading, and a windowed mean near a nodata gap must be
the mean of the pixels that *were* observed, not an average dragged toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

import wh_indices
from wh_config import Config
from wh_tiles import Tile

# Columns that identify a row rather than describe it. Kept out of the feature
# matrix at training time.
ID_COLUMNS = ("site_id", "year_month", "row", "col", "class_id", "source")


@dataclass
class FeatureParams:
    """Which features to build. Passed explicitly so a notebook can show them."""

    reflectance_bands: tuple[str, ...] = (
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    )
    indices: tuple[str, ...] = (
        "mndwi", "ndwi", "ndvi", "ndti", "ndmi", "nd_rededge",
        "red_green_ratio", "awei_sh", "awei_nsh",
    )
    context_windows: tuple[int, ...] = (3, 9)
    context_indices: tuple[str, ...] = ("mndwi", "ndvi")
    temporal_indices: tuple[str, ...] = ("mndwi", "ndvi", "ndti", "ndmi")

    # n_obs is NOT a training feature by default. It is a property of the
    # observing system, not of the ground: wet-season months have both fewer
    # clear scenes and more water, so a classifier can learn "few observations
    # therefore wet" instead of reading the surface. Permutation importance found
    # it the second most influential feature of 121, which is what that shortcut
    # looks like from the outside. It would also transfer badly to a year with
    # different cloud cover.
    #
    # It remains in use elsewhere, where it belongs: weighting the temporal
    # statistics (wh_temporal.observation_weights), gating pseudo-labels, and
    # filtering the labelling queue.
    include_n_obs: bool = False

    # AlphaEarth annual embeddings: 64 static learned dimensions per pixel.
    use_alphaearth: bool = False
    alphaearth_year: int = 2025
    # None means all 64. A subset selected by band importance goes here.
    alphaearth_bands: tuple[str, ...] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> "FeatureParams":
        settings = cfg["features"]
        bands = settings.get("alphaearth_bands", "all")
        return cls(
            reflectance_bands=tuple(settings["reflectance_bands"]),
            indices=tuple(settings["indices"]),
            context_windows=tuple(settings["context_windows"]),
            context_indices=tuple(settings["context_indices"]),
            temporal_indices=tuple(settings["temporal"]["indices"]),
            include_n_obs=bool(settings.get("include_n_obs", False)),
            use_alphaearth=bool(settings.get("use_alphaearth", False)),
            alphaearth_year=int(settings.get("alphaearth_year", 2025)),
            alphaearth_bands=None if bands in (None, "all") else tuple(bands),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reflectance_bands": list(self.reflectance_bands),
            "indices": list(self.indices),
            "context_windows": list(self.context_windows),
            "context_indices": list(self.context_indices),
            "temporal_indices": list(self.temporal_indices),
            "include_n_obs": self.include_n_obs,
            "use_alphaearth": self.use_alphaearth,
            "alphaearth_year": self.alphaearth_year,
            "alphaearth_bands": (
                list(self.alphaearth_bands) if self.alphaearth_bands else "all"
            ),
        }


def local_context(
    values: np.ndarray, window: int, statistic: str = "mean"
) -> np.ndarray:
    """Windowed mean or SD over the OBSERVED pixels in the window.

    A plain uniform_filter would treat NaN as zero and pull every window near a
    nodata gap toward zero, inventing an edge where there is only missing data.
    Instead the filter runs over the zero-filled values and over the validity
    mask, and the two are divided — giving the mean of however many pixels were
    actually observed, and NaN where none were.

    SD is computed from the same windowed sums (E[x^2] - E[x]^2), clipped at zero
    to absorb floating-point noise.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number of pixels, got {window}")
    if statistic not in ("mean", "sd"):
        raise ValueError(f"statistic must be 'mean' or 'sd', got {statistic!r}")

    values = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(values)

    if not observed.any():
        return np.full(values.shape, np.nan)

    # Centre on the field's own mean before filtering. The variance below is a
    # difference of two large, nearly equal numbers; without the shift it loses
    # most of its significant digits, and the SD of a constant field comes out
    # around 1e-8 instead of 0.
    shift = float(np.mean(values[observed]))
    centred = np.where(observed, values - shift, 0.0)

    # uniform_filter returns the mean over the window; multiplying by the window
    # area recovers the sum, which is what has to be divided by the true count.
    area = float(window * window)
    count = ndimage.uniform_filter(observed.astype(np.float64), size=window, mode="nearest") * area
    total = ndimage.uniform_filter(centred, size=window, mode="nearest") * area

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_centred = np.where(count > 0, total / count, np.nan)

    if statistic == "mean":
        return mean_centred + shift

    total_squares = ndimage.uniform_filter(centred**2, size=window, mode="nearest") * area
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where(count > 0, total_squares / count - mean_centred**2, np.nan)
    return np.sqrt(np.clip(variance, 0.0, None))


def instantaneous_features(tile: Tile, params: FeatureParams) -> dict[str, np.ndarray]:
    """Reflectance, indices, local context and n_obs for one site-month."""
    features: dict[str, np.ndarray] = {}

    for band in params.reflectance_bands:
        if band not in tile.bands:
            raise KeyError(f"band {band!r} not in the tile; have {sorted(tile.bands)}")
        features[f"refl_{band}"] = tile.bands[band]

    computed = wh_indices.compute_many(params.indices, tile.bands)
    features.update(computed)

    for name in params.context_indices:
        if name not in computed:
            computed[name] = wh_indices.compute(name, tile.bands)
        for window in params.context_windows:
            features[f"{name}_mean{window}"] = local_context(computed[name], window, "mean")
            features[f"{name}_sd{window}"] = local_context(computed[name], window, "sd")

    if params.include_n_obs:
        # How much evidence is behind this month's median. A one-scene median and
        # a six-scene one are different measurements, and the classifier should
        # be able to tell.
        features["n_obs"] = (
            tile.n_obs.astype(np.float64)
            if tile.n_obs is not None
            else np.full(tile.shape, np.nan)
        )

    return features


ALPHAEARTH_PREFIX = "ae_"
ALL_ALPHAEARTH_BANDS = tuple(f"A{i:02d}" for i in range(64))


def alphaearth_path(cfg: Config, site_id: str, year: int) -> Path:
    """Where a site's embedding chip lives."""
    prefix = cfg["tiles"]["filename_prefix"]
    matches = sorted(cfg.paths["alphaearth"].glob(f"{prefix}_{site_id}_*_{year}.tif"))
    if not matches:
        raise FileNotFoundError(
            f"no AlphaEarth chip for site {site_id} year {year} in "
            f"{cfg.paths['alphaearth']}"
        )
    return matches[0]


def load_alphaearth(
    cfg: Config,
    site_id: str,
    params: FeatureParams,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Read a site's 64-band embedding as {ae_A00: array, ...}.

    Static across time — one annual image applies to every month — so this is
    loaded once per site alongside the temporal features and broadcast, exactly
    as the per-pixel temporal features are.

    The shape is asserted rather than resampled. The chips were downloaded onto
    the Sentinel-2 grid deliberately, because interpolating a learned embedding
    produces a vector corresponding to no real surface; silently resampling here
    would throw that away.
    """
    path = alphaearth_path(cfg, site_id, params.alphaearth_year)
    nodata = float(cfg["tiles"]["nodata"])

    with rasterio.open(path) as dataset:
        names = list(dataset.descriptions)
        if not all(names) or len(names) != len(ALL_ALPHAEARTH_BANDS):
            names = list(ALL_ALPHAEARTH_BANDS)
        wanted = params.alphaearth_bands or ALL_ALPHAEARTH_BANDS

        missing = [band for band in wanted if band not in names]
        if missing:
            raise KeyError(f"{path}: embedding bands not present: {missing}")

        if expected_shape is not None and dataset.shape != expected_shape:
            raise ValueError(
                f"{path}: embedding shape {dataset.shape} does not match the tile "
                f"{expected_shape}. The chips are meant to be grid-locked to the "
                f"Sentinel-2 export; re-download rather than resampling."
            )

        features = {}
        for band in wanted:
            values = dataset.read(names.index(band) + 1).astype(np.float64)
            features[f"{ALPHAEARTH_PREFIX}{band}"] = np.where(
                np.isclose(values, nodata) | ~np.isfinite(values), np.nan, values
            )

    return features


def alphaearth_composite(
    alphaearth: dict[str, np.ndarray],
    bands: tuple[str, str, str] = ("A60", "A24", "A63"),
    mode: str = "bands",
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Render the 64-band embedding as one false-colour image, (H, W, 3) in [0, 1].

    Two modes:

      "bands" — three named bands to R, G, B. Defaults to the three that ranked
        highest on permutation importance. Directly interpretable: a colour
        difference is a difference in those three specific dimensions.

      "pca"  — the first three principal components of all 64 bands. Uses more of
        the information, at the cost of axes that mean nothing in themselves.

    Each channel is percentile-stretched independently, so colours are comparable
    within a tile but not between tiles. That matters less than it sounds here:
    the embedding is annual and static, so a site's composite is identical in
    every month, and stepping through time in the labeller will not make it
    flicker.
    """
    if mode not in ("bands", "pca"):
        raise ValueError(f"mode must be 'bands' or 'pca', got {mode!r}")

    if mode == "bands":
        missing = [b for b in bands if f"{ALPHAEARTH_PREFIX}{b}" not in alphaearth]
        if missing:
            raise KeyError(
                f"embedding bands {missing} not loaded; available "
                f"{sorted(alphaearth)[:4]}..."
            )
        channels = [alphaearth[f"{ALPHAEARTH_PREFIX}{b}"] for b in bands]
    else:
        channels = _embedding_principal_components(alphaearth)

    stretched = [_stretch(channel, percentiles) for channel in channels]
    return np.dstack(stretched)


def _embedding_principal_components(
    alphaearth: dict[str, np.ndarray], n_components: int = 3
) -> list[np.ndarray]:
    """First n principal components of the embedding stack, as 2-D arrays.

    Fitted on this tile alone, so the components are not comparable between
    sites — good for looking at one waterhole, not for stacking.
    """
    from sklearn.decomposition import PCA

    names = sorted(alphaearth)
    stack = np.stack([alphaearth[name] for name in names], axis=-1)
    height, width, n_bands = stack.shape

    flat = stack.reshape(-1, n_bands)
    observed = np.isfinite(flat).all(axis=1)
    if observed.sum() < n_components:
        raise ValueError("too few observed pixels to fit a PCA on this tile")

    components = PCA(n_components=n_components).fit_transform(flat[observed])

    out = []
    for index in range(n_components):
        channel = np.full(flat.shape[0], np.nan)
        channel[observed] = components[:, index]
        out.append(channel.reshape(height, width))
    return out


def _stretch(values: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Scale to [0, 1] between two percentiles; NaN stays NaN."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)

    low, high = np.percentile(finite, percentiles)
    if high <= low:
        return np.where(np.isfinite(values), 0.5, np.nan)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def assemble_features(
    tile: Tile,
    temporal_features: dict[str, np.ndarray],
    month_position: int,
    params: FeatureParams,
    alphaearth: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Instantaneous features merged with this month's slice of the temporal ones.

    Temporal features come in two shapes. Per-pixel (H, W) features — seasonal
    extremes, harmonic coefficients — describe what kind of place a pixel is and
    are the same in every month, so they broadcast. Per-month (T, H, W) features
    — rank, residual, distance from the wet maximum — describe the pixel's state
    now and are indexed at this month.
    """
    features = instantaneous_features(tile, params)

    for name, values in temporal_features.items():
        if values.ndim == 2:
            features[name] = values
        elif values.ndim == 3:
            if not 0 <= month_position < values.shape[0]:
                raise IndexError(
                    f"month_position {month_position} outside the {values.shape[0]}-month "
                    f"stack for feature {name!r}"
                )
            features[name] = values[month_position]
        else:
            raise ValueError(f"temporal feature {name!r} has unexpected shape {values.shape}")

    # Embeddings are static: one annual image broadcasts across every month.
    if alphaearth:
        features.update(alphaearth)

    shapes = {values.shape for values in features.values()}
    if len(shapes) != 1:
        raise ValueError(f"features disagree on shape: {shapes}")

    return features


def extract_pixels(
    features: dict[str, np.ndarray],
    selection: np.ndarray,
    site_id: str,
    year_month: str,
    class_ids: np.ndarray | None = None,
    source: str = "manual",
) -> pd.DataFrame:
    """Tidy DataFrame of the selected pixels.

    site_id and year_month travel with every row because the cross-validation
    splitter groups on site_id, and losing that column would make an ungrouped
    split possible by accident.
    """
    rows, cols = np.nonzero(selection)
    if rows.size == 0:
        return pd.DataFrame(columns=list(ID_COLUMNS) + sorted(features))

    # Every column is assembled first and the frame built once. Adding them one
    # at a time is an insert per feature per tile — with ~120 features over
    # hundreds of labelled tiles that fragments the frame badly enough for pandas
    # to warn about it, thousands of times.
    columns: dict[str, object] = {
        "site_id": site_id,
        "year_month": year_month,
        "row": rows,
        "col": cols,
        "source": source,
    }
    if class_ids is not None:
        columns["class_id"] = class_ids[rows, cols].astype(np.int16)

    for name in sorted(features):
        columns[name] = features[name][rows, cols]

    return pd.DataFrame(columns)


def instantaneous_feature_names(params: FeatureParams) -> list[str]:
    """Names of the features derived from a single month's tile.

    Derived from the params rather than pattern-matched, so the ablation can
    split instantaneous from temporal features without guessing from suffixes.
    """
    names = [f"refl_{band}" for band in params.reflectance_bands]
    names += list(params.indices)
    for name in params.context_indices:
        for window in params.context_windows:
            names += [f"{name}_mean{window}", f"{name}_sd{window}"]
    if params.include_n_obs:
        names.append("n_obs")
    return names


def alphaearth_feature_names(params: FeatureParams) -> list[str]:
    """Names of the embedding columns this params object would produce."""
    if not params.use_alphaearth:
        return []
    bands = params.alphaearth_bands or ALL_ALPHAEARTH_BANDS
    return [f"{ALPHAEARTH_PREFIX}{band}" for band in bands]


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The feature columns of a training table: everything that is not an id."""
    return [column for column in table.columns if column not in ID_COLUMNS]


def describe_missing(table: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Fraction of NaN per feature, worst first.

    Worth looking at before training: a temporal feature that is mostly NaN means
    too few pixels had enough observed months to fit, which is a data problem
    rather than something for the imputer to paper over.
    """
    columns = feature_columns(table)
    missing = table[columns].isna().mean().sort_values(ascending=False)
    return missing[missing > threshold].to_frame("nan_fraction")
