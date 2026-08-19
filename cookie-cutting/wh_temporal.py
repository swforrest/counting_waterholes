"""Per-pixel temporal features: each pixel normalised against its own history.

This is the part of the design everything else leans on. A pixel reading MNDWI
-0.4 in October says almost nothing on its own. The same pixel reaching +0.5
every February is unambiguously a basin floor. Comparing a pixel to itself
across years removes the between-site variation in soil, shade and vegetation
that otherwise swamps the between-class variation.

It also side-steps the weakness of the water indices themselves. A sedge-choked
waterhole never crosses an MNDWI threshold, but it still has a seasonal
*signature* — an annual amplitude, and a dry-season greenness anomaly — that
distinguishes it from the savanna matrix around it.

Everything here works on a stack of shape (n_months, height, width) with NaN
where a month has no clear observation, and is careful never to let a gap enter
a statistic as if it were a reading.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import wh_tiles
from wh_config import Config

MONTHS_PER_YEAR = 12


@dataclass
class SiteStack:
    """One site's full time series for one or more indices."""

    site_id: str
    month_index: np.ndarray  # (n_months,) int, months since year 0
    year_month: list[str]  # (n_months,) 'YYYY-MM'
    stacks: dict[str, np.ndarray]  # index name -> (n_months, H, W) float, NaN = gap
    n_obs: np.ndarray  # (n_months, H, W) int, clear observations per pixel
    shape: tuple[int, int]
    transform: object
    crs: object

    @property
    def n_months(self) -> int:
        return len(self.month_index)

    @property
    def calendar_month(self) -> np.ndarray:
        """(n_months,) calendar month 1-12, for wet/dry season selection."""
        return (self.month_index % MONTHS_PER_YEAR) + 1

    @property
    def decimal_year(self) -> np.ndarray:
        """(n_months,) time in years, for harmonic and trend terms."""
        return self.month_index / MONTHS_PER_YEAR


def load_site_stack(
    manifest: pd.DataFrame,
    site_id: str,
    cfg: Config,
    indices: list[str] | None = None,
) -> SiteStack:
    """Read every month of one site into aligned per-index stacks.

    Months whose grid disagrees with the site's own siblings are dropped rather
    than stacked, because a shifted grid silently mixes different ground
    locations into one pixel's history.
    """
    if indices is None:
        indices = list(cfg["features"]["temporal"]["indices"])

    rows = manifest[manifest["site_id"] == site_id].sort_values("month_index")
    if rows.empty:
        raise KeyError(f"site {site_id!r} not present in the manifest")

    if "grid_matches_site" in rows.columns:
        dropped = rows[~rows["grid_matches_site"]]
        if not dropped.empty:
            print(
                f"  site {site_id}: dropping {len(dropped)} month(s) on a "
                f"different grid: {sorted(dropped['year_month'])}"
            )
        rows = rows[rows["grid_matches_site"]]
        if rows.empty:
            raise ValueError(f"site {site_id}: no months share a common grid")

    first = wh_tiles.read_tile(rows.iloc[0]["tif_path"], cfg)
    shape = first.shape
    n_months = len(rows)

    stacks = {
        name: np.full((n_months, *shape), np.nan, dtype=np.float32)
        for name in indices
    }
    n_obs = np.zeros((n_months, *shape), dtype=np.int16)

    for position, (_, row) in enumerate(rows.iterrows()):
        tile = first if position == 0 else wh_tiles.read_tile(row["tif_path"], cfg)
        if tile.shape != shape:
            raise ValueError(
                f"site {site_id}: {row['year_month']} has shape {tile.shape}, "
                f"expected {shape}"
            )
        for name, values in tile.indices(indices).items():
            stacks[name][position] = values
        if tile.n_obs is not None:
            n_obs[position] = tile.n_obs

    return SiteStack(
        site_id=site_id,
        month_index=rows["month_index"].to_numpy(),
        year_month=rows["year_month"].tolist(),
        stacks=stacks,
        n_obs=n_obs,
        shape=shape,
        transform=first.transform,
        crs=first.crs,
    )


# --- observation weighting -------------------------------------------------


def observation_weights(
    stack: SiteStack,
    values: np.ndarray,
    min_obs: int,
    weight_by_obs: bool = True,
) -> np.ndarray:
    """Weights for a (n_months, H, W) stack: 0 where the value cannot be trusted.

    A month-pixel is given zero weight if it has no value, or fewer than
    min_obs clear observations. Remaining weights rise with the square root of
    the observation count, so a six-scene median counts for more than a
    two-scene one without letting it dominate.
    """
    usable = np.isfinite(values) & (stack.n_obs >= min_obs)
    if not weight_by_obs:
        return usable.astype(np.float64)
    return np.where(usable, np.sqrt(np.maximum(stack.n_obs, 1)), 0.0)


# --- harmonic regression ---------------------------------------------------


def harmonic_design(
    decimal_year: np.ndarray,
    orders: int = 1,
    include_trend: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix for an annual harmonic fit, with an optional linear trend.

    Columns: intercept, [trend], then cos/sin at 1..orders cycles per year.
    Time is centred so the intercept is the series mean rather than the value
    extrapolated back to year zero, and so the trend coefficient is not
    correlated with it.

    orders=1 (annual only) is the sensible default. Each extra order adds two
    parameters that have to be paid for out of the wet-season months, which are
    exactly the months the cloud masking thins out.
    """
    if orders < 0:
        raise ValueError(f"orders must be >= 0, got {orders}")

    centred = decimal_year - decimal_year.mean()
    columns = [np.ones_like(centred)]
    names = ["intercept"]

    if include_trend:
        columns.append(centred)
        names.append("trend_per_year")

    for order in range(1, orders + 1):
        angle = 2.0 * np.pi * order * decimal_year
        columns.extend([np.cos(angle), np.sin(angle)])
        names.extend([f"cos{order}", f"sin{order}"])

    return np.column_stack(columns), names


@dataclass
class HarmonicFit:
    """Per-pixel harmonic fit results."""

    coefficients: np.ndarray  # (n_terms, H, W)
    term_names: list[str]
    fitted: np.ndarray  # (n_months, H, W)
    residual: np.ndarray  # (n_months, H, W)
    n_used: np.ndarray  # (H, W) months that contributed
    amplitude: np.ndarray  # (H, W) annual amplitude, sqrt(cos1^2 + sin1^2)
    phase: np.ndarray  # (H, W) annual phase in radians


def fit_harmonic(
    values: np.ndarray,
    design: np.ndarray,
    term_names: list[str],
    weights: np.ndarray | None = None,
    min_months: int | None = None,
) -> HarmonicFit:
    """Weighted least squares per pixel, solving all pixels at once.

    Each pixel has its own pattern of missing months, so this cannot be a single
    lstsq. Instead the normal equations are accumulated per pixel — with a
    small number of terms, the resulting stack of tiny systems solves quickly
    and exactly.

    Pixels with fewer usable months than terms (or than min_months) get NaN
    coefficients rather than a fit that interpolates its own noise.
    """
    n_months, height, width = values.shape
    n_terms = design.shape[1]

    if design.shape[0] != n_months:
        raise ValueError(
            f"design has {design.shape[0]} rows but the stack has {n_months} months"
        )
    if len(term_names) != n_terms:
        raise ValueError(f"{len(term_names)} term names for {n_terms} design columns")

    if weights is None:
        weights = np.isfinite(values).astype(np.float64)
    weights = np.where(np.isfinite(values), weights, 0.0)

    clean = np.where(np.isfinite(values), values, 0.0).astype(np.float64)

    flat_weights = weights.reshape(n_months, -1)  # (T, P)
    flat_values = clean.reshape(n_months, -1)

    # A = X' W X per pixel -> (P, K, K);  b = X' W y per pixel -> (P, K)
    normal_matrix = np.einsum("tk,tp,tj->pkj", design, flat_weights, design)
    normal_vector = np.einsum("tk,tp->pk", design, flat_weights * flat_values)

    n_used = (flat_weights > 0).sum(axis=0)
    required = max(n_terms, min_months or 0)
    solvable = n_used >= required

    coefficients = np.full((flat_weights.shape[1], n_terms), np.nan)
    if solvable.any():
        solved, ok = _solve_stack(normal_matrix[solvable], normal_vector[solvable])
        indices = np.flatnonzero(solvable)
        coefficients[indices[ok]] = solved[ok]

    fitted_flat = coefficients @ design.T  # (P, T)
    fitted = fitted_flat.T.reshape(n_months, height, width)
    residual = np.where(np.isfinite(values), values - fitted, np.nan)

    coefficient_stack = coefficients.T.reshape(n_terms, height, width)

    amplitude, phase = _annual_amplitude_phase(coefficient_stack, term_names)

    return HarmonicFit(
        coefficients=coefficient_stack,
        term_names=term_names,
        fitted=fitted,
        residual=residual,
        n_used=n_used.reshape(height, width),
        amplitude=amplitude,
        phase=phase,
    )


def _solve_stack(
    matrices: np.ndarray, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a stack of small linear systems, flagging the singular ones."""
    solutions = np.full(vectors.shape, np.nan)

    # A rank-deficient pixel (e.g. every usable month in the same season) has a
    # near-zero determinant; solving it would return arbitrary large numbers.
    determinants = np.linalg.det(matrices)
    scale = np.abs(matrices).max(axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    invertible = np.abs(determinants) > 1e-12 * scale ** matrices.shape[1]

    if invertible.any():
        # The trailing axis is made explicit: numpy 2 reads a 2-D right-hand
        # side as a stack of matrices, not as a stack of vectors.
        solved = np.linalg.solve(
            matrices[invertible], vectors[invertible][..., np.newaxis]
        )
        solutions[invertible] = solved[..., 0]
    return solutions, invertible


def _annual_amplitude_phase(
    coefficients: np.ndarray, term_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude and phase of the annual cycle, from its cos/sin coefficients.

    Amplitude is the single most useful number the fit produces here: it is how
    strongly a pixel swings with the season, which is what separates a basin
    floor from the matrix around it regardless of whether it ever holds visible
    water.
    """
    if "cos1" not in term_names or "sin1" not in term_names:
        empty = np.full(coefficients.shape[1:], np.nan)
        return empty, empty.copy()

    cosine = coefficients[term_names.index("cos1")]
    sine = coefficients[term_names.index("sin1")]
    return np.hypot(cosine, sine), np.arctan2(sine, cosine)


# --- summary statistics over a pixel's own history -------------------------


_SEASONAL_REDUCERS = {
    "max": np.nanmax,
    "min": np.nanmin,
    "median": np.nanmedian,
    "mean": np.nanmean,
}


def seasonal_statistic(
    values: np.ndarray,
    calendar_month: np.ndarray,
    months: list[int],
    statistic: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel statistic over a set of calendar months across all years.

    Months outside the selection, and pixel-months with zero weight, are
    excluded. Pixels with nothing usable return NaN.

    'median' matters as much as the extremes here: a vegetated basin is
    distinguished from the savanna matrix by staying green through the dry
    season, which is a shift in the typical value, not in the extreme.
    """
    if statistic not in _SEASONAL_REDUCERS:
        raise ValueError(
            f"statistic must be one of {sorted(_SEASONAL_REDUCERS)}, got {statistic!r}"
        )

    selected = np.isin(calendar_month, months)
    if not selected.any():
        raise ValueError(f"no months in the stack match {months}")

    subset = values[selected]
    if weights is not None:
        subset = np.where(weights[selected] > 0, subset, np.nan)

    usable = np.isfinite(subset).any(axis=0)
    # A pixel with no usable month in the season is expected, not exceptional;
    # it is handled by the np.where below rather than warned about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _SEASONAL_REDUCERS[statistic](subset, axis=0)
    return np.where(usable, result, np.nan)


def seasonal_extreme(
    values: np.ndarray,
    calendar_month: np.ndarray,
    months: list[int],
    statistic: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel max or min over a set of calendar months. See seasonal_statistic."""
    if statistic not in ("max", "min"):
        raise ValueError(f"statistic must be 'max' or 'min', got {statistic!r}")
    return seasonal_statistic(values, calendar_month, months, statistic, weights)


def percentile_rank(
    values: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Where each month sits within that pixel's own distribution, in [0, 1].

    This is the self-normalising feature: 0.9 means "wetter than 90% of the
    months this particular pixel has ever had", which is comparable across
    sites in a way that a raw index value is not.
    """
    usable = np.isfinite(values)
    if weights is not None:
        usable &= weights > 0

    masked = np.where(usable, values, np.nan)
    n_usable = usable.sum(axis=0)

    # Rank by counting how many of a pixel's own usable months it exceeds.
    below = np.zeros(values.shape, dtype=np.float64)
    for position in range(values.shape[0]):
        current = masked[position]
        with np.errstate(invalid="ignore"):
            below[position] = np.nansum(masked < current[None, :, :], axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ranks = below / np.maximum(n_usable - 1, 1)[None, :, :]

    return np.where(usable & (n_usable[None, :, :] > 1), ranks, np.nan)


def months_since_threshold(
    values: np.ndarray,
    month_index: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Months since each pixel last exceeded a threshold, per month.

    Left-censored: a pixel that has never exceeded the threshold within the
    exported record returns NaN, not a large number. Treating "never observed
    wet" as "wet a very long time ago" would be a fabrication.
    """
    exceeds = values > threshold
    if weights is not None:
        exceeds &= weights > 0
    exceeds &= np.isfinite(values)

    n_months = values.shape[0]
    result = np.full(values.shape, np.nan)
    last_seen = np.full(values.shape[1:], np.nan)

    for position in range(n_months):
        # Evaluated before this month's own exceedance, so a wet month reads 0
        # only after the update below.
        gap = month_index[position] - last_seen
        result[position] = gap
        last_seen = np.where(exceeds[position], month_index[position], last_seen)
        result[position] = np.where(exceeds[position], 0.0, result[position])

    return result


def temporal_feature_stack(
    stack: SiteStack, cfg: Config
) -> dict[str, np.ndarray]:
    """All temporal features for one site, as {name: (n_months, H, W) or (H, W)}.

    Per-month features vary with the month; per-pixel features (the fit
    coefficients, the seasonal extremes) are constant down the time axis and are
    returned with shape (H, W). wh_features.py broadcasts them when assembling
    the training table.
    """
    settings = cfg["features"]["temporal"]
    min_obs = int(settings["min_obs_for_temporal"])
    wet_months = list(settings["wet_season_months"])
    dry_months = list(settings["dry_season_months"])
    orders = int(settings["harmonic_orders"])
    threshold = float(settings["water_threshold_mndwi"])
    harmonic_enabled = bool(settings.get("harmonic_enabled", True))
    keep_trend = bool(settings.get("keep_trend_when_disabled", True))

    features: dict[str, np.ndarray] = {}
    design, term_names = harmonic_design(stack.decimal_year, orders=orders)

    for name in stack.stacks:
        values = stack.stacks[name].astype(np.float64)
        weights = observation_weights(stack, values, min_obs=min_obs)

        wet_max = seasonal_extreme(values, stack.calendar_month, wet_months, "max", weights)
        dry_min = seasonal_extreme(values, stack.calendar_month, dry_months, "min", weights)

        features[f"{name}_wet_max"] = wet_max
        features[f"{name}_dry_min"] = dry_min
        features[f"{name}_seasonal_range"] = wet_max - dry_min
        features[f"{name}_minus_wet_max"] = values - wet_max[None, :, :]
        features[f"{name}_rank"] = percentile_rank(values, weights)

        # Dry-season median, not an extreme: a sedge-choked basin is marked out
        # by staying green while the matrix browns off, which shifts the typical
        # value rather than the extreme one. This is the feature that finds the
        # basins MNDWI cannot see.
        features[f"{name}_dry_median"] = seasonal_statistic(
            values, stack.calendar_month, dry_months, "median", weights
        )
        features[f"{name}_wet_median"] = seasonal_statistic(
            values, stack.calendar_month, wet_months, "median", weights
        )

        if not harmonic_enabled and not keep_trend:
            continue

        fit = fit_harmonic(values, design, term_names, weights=weights,
                           min_months=len(term_names) + 2)

        if not harmonic_enabled:
            # Trend only: the one harmonic output with no model-free substitute.
            features[f"{name}_harm_trend_per_year"] = fit.coefficients[
                fit.term_names.index("trend_per_year")
            ]
            continue

        for position, term in enumerate(fit.term_names):
            features[f"{name}_harm_{term}"] = fit.coefficients[position]
        features[f"{name}_harm_amplitude"] = fit.amplitude
        features[f"{name}_harm_phase"] = fit.phase
        features[f"{name}_harm_fitted"] = fit.fitted
        features[f"{name}_harm_residual"] = fit.residual
        features[f"{name}_harm_n_used"] = fit.n_used.astype(np.float64)

    # Recency of water is defined on MNDWI only, and is a FEATURE — never a
    # label, and never a footprint criterion. Emergent vegetation keeps MNDWI
    # low over standing water, so a large value here does not mean dry.
    if "mndwi" in stack.stacks:
        mndwi = stack.stacks["mndwi"].astype(np.float64)
        features["months_since_water"] = months_since_threshold(
            mndwi,
            stack.month_index,
            threshold,
            observation_weights(stack, mndwi, min_obs=min_obs),
        )

    return features


# --- feature grouping, for ablation ---------------------------------------

HARMONIC_MARKER = "_harm_"
TREND_MARKER = "_harm_trend_per_year"


def is_harmonic_feature(name: str) -> bool:
    """Whether a feature came out of the harmonic fit."""
    return HARMONIC_MARKER in name


def is_trend_feature(name: str) -> bool:
    """Whether a feature is the multi-year trend coefficient.

    Kept separate from the rest of the harmonic block: it is the only harmonic
    output with no model-free substitute, so an ablation that drops the harmonic
    usually still wants this.
    """
    return name.endswith(TREND_MARKER)


def split_feature_names(names: list[str]) -> dict[str, list[str]]:
    """Partition temporal feature names into ablation groups.

    Groups: 'model_free' (seasonal extremes, medians, ranks, recency), 'trend'
    (the multi-year coefficient), and 'harmonic' (everything else the fit
    produced). wh_train uses these to score the harmonic block's contribution
    rather than assuming it earns its place.
    """
    groups: dict[str, list[str]] = {"model_free": [], "trend": [], "harmonic": []}
    for name in names:
        if is_trend_feature(name):
            groups["trend"].append(name)
        elif is_harmonic_feature(name):
            groups["harmonic"].append(name)
        else:
            groups["model_free"].append(name)
    return groups


def cache_path(cfg: Config, site_id: str) -> Path:
    """Where a site's computed temporal features are cached."""
    return cfg.paths["derived"] / "temporal" / f"site_{site_id}.npz"


def save_features(
    features: dict[str, np.ndarray],
    cfg: Config,
    site_id: str,
    include_per_month: bool = False,
) -> Path:
    """Cache a site's temporal features.

    By default only the per-pixel (H, W) features are written — the harmonic
    coefficients, seasonal extremes and amplitudes. The per-month (n_months,
    H, W) features are excluded because they are ~90 MB per site, which is
    12 GB across the archive, to save the ~1 second it takes to recompute them
    from the cached statistics and the tiles.
    """
    selected = {
        name: values.astype(np.float32)
        for name, values in features.items()
        if include_per_month or values.ndim == 2
    }

    path = cache_path(cfg, site_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **selected)
    return path


def load_features(cfg: Config, site_id: str) -> dict[str, np.ndarray]:
    """Load cached temporal features for a site."""
    path = cache_path(cfg, site_id)
    if not path.exists():
        raise FileNotFoundError(f"no cached temporal features at {path}")
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}
