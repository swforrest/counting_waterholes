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

import numpy as np
import pandas as pd
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
    include_n_obs: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> "FeatureParams":
        settings = cfg["features"]
        return cls(
            reflectance_bands=tuple(settings["reflectance_bands"]),
            indices=tuple(settings["indices"]),
            context_windows=tuple(settings["context_windows"]),
            context_indices=tuple(settings["context_indices"]),
            temporal_indices=tuple(settings["temporal"]["indices"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reflectance_bands": list(self.reflectance_bands),
            "indices": list(self.indices),
            "context_windows": list(self.context_windows),
            "context_indices": list(self.context_indices),
            "temporal_indices": list(self.temporal_indices),
            "include_n_obs": self.include_n_obs,
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


def assemble_features(
    tile: Tile,
    temporal_features: dict[str, np.ndarray],
    month_position: int,
    params: FeatureParams,
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

    table = pd.DataFrame({
        "site_id": site_id,
        "year_month": year_month,
        "row": rows,
        "col": cols,
        "source": source,
    })
    if class_ids is not None:
        table["class_id"] = class_ids[rows, cols].astype(np.int16)

    for name in sorted(features):
        table[name] = features[name][rows, cols]

    return table


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
