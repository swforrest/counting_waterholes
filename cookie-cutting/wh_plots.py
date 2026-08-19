"""Diagnostic plots for the temporal features, the harmonic fit and footprints.

These exist to make the features legible before they are trusted. The harmonic
fit in particular is easy to accept on faith and hard to reason about, so
plot_pixel_timeseries draws the observations, the fitted curve and the residuals
together — the residual is the feature that carries sharp change, and it is
worth seeing how large it gets where a basin fills quickly.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import wh_tiles
from wh_config import Config
from wh_footprint import Footprint
from wh_temporal import SiteStack, harmonic_design

# Diverging map for anomaly scores, centred on zero.
SCORE_CMAP = "RdYlBu_r"
RGB_MAX_REFLECTANCE = 0.30
RGB_GAMMA = 0.85


def read_month_tile(manifest: pd.DataFrame, site_id: str, year_month: str, cfg: Config):
    """Read one site-month chip, for RGB context in a plot."""
    rows = manifest[
        (manifest["site_id"] == site_id) & (manifest["year_month"] == year_month)
    ]
    if rows.empty:
        raise KeyError(f"no chip for site {site_id} in {year_month}")
    return wh_tiles.read_tile(rows.iloc[0]["tif_path"], cfg)


def rgb_composite(tile) -> np.ndarray:
    """Fixed-stretch true colour, matching the labelling PNGs."""
    stack = np.dstack([tile.bands["B4"], tile.bands["B3"], tile.bands["B2"]])
    stack = np.clip(stack.astype("float32") / RGB_MAX_REFLECTANCE, 0, 1)
    return np.power(stack, RGB_GAMMA)


def _show(axis, data, title, cmap="viridis", diverging=False, **kwargs):
    """Imshow with a colourbar and no tick clutter."""
    if diverging:
        finite = data[np.isfinite(data)]
        limit = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
        limit = max(limit, 1e-6)
        kwargs["norm"] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    image = axis.imshow(data, cmap=cmap, interpolation="nearest", **kwargs)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    return image


def plot_footprint_diagnostics(
    footprint: Footprint,
    stack: SiteStack,
    features: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    cfg: Config,
    wet_month: str | None = None,
    dry_month: str | None = None,
    figsize: tuple[float, float] = (15, 8.5),
):
    """Every layer that produced a footprint, in one figure.

    Top row is what the eye can check: wet and dry true colour, with the derived
    outline drawn on the dry image where the basin is easiest to see. Bottom row
    is what the algorithm actually used.
    """
    wet_month = wet_month or _pick_month(stack, [1, 2, 3])
    dry_month = dry_month or _pick_month(stack, [9, 10, 8])

    layer_names = list(footprint.components)
    n_bottom = len(layer_names) + 1
    n_columns = max(3, n_bottom)

    figure, axes = plt.subplots(2, n_columns, figsize=figsize, constrained_layout=True)
    status = "OK" if footprint.succeeded else f"FAILED — {footprint.reason}"
    figure.suptitle(
        f"site {stack.site_id}   footprint {footprint.n_pixels} px "
        f"({footprint.area_m2 / 1e4:.2f} ha, {100 * footprint.fraction_of_tile:.1f}% of tile)"
        f"   [{status}]",
        fontsize=11,
    )

    for axis in axes.ravel():
        axis.set_axis_off()

    wet_tile = read_month_tile(manifest, stack.site_id, wet_month, cfg)
    dry_tile = read_month_tile(manifest, stack.site_id, dry_month, cfg)

    for axis, tile, label in (
        (axes[0, 0], wet_tile, f"RGB wet {wet_month}"),
        (axes[0, 1], dry_tile, f"RGB dry {dry_month}"),
    ):
        axis.set_axis_on()
        axis.imshow(rgb_composite(tile))
        axis.set_title(label, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    # Outline rather than fill, so the surface underneath stays readable.
    for axis in (axes[0, 0], axes[0, 1]):
        if footprint.mask.any():
            axis.contour(footprint.mask, levels=[0.5], colors="#ff00ff", linewidths=1.2)
        if footprint.core_mask.any():
            axis.contour(
                footprint.core_mask, levels=[0.5], colors="#ffffff",
                linewidths=0.8, linestyles="dashed",
            )

    axes[0, 2].set_axis_on()
    _show(axes[0, 2], footprint.score, "combined basin score (z)", SCORE_CMAP, diverging=True)
    if footprint.core_mask.any():
        axes[0, 2].contour(footprint.core_mask, levels=[0.5], colors="#000000", linewidths=0.8)

    for position, name in enumerate(layer_names):
        axis = axes[1, position]
        axis.set_axis_on()
        _show(axis, footprint.components[name], f"{name} (z)", SCORE_CMAP, diverging=True)

    axis = axes[1, len(layer_names)]
    axis.set_axis_on()
    valid_months = np.isfinite(stack.stacks["mndwi"]).sum(axis=0)
    _show(axis, valid_months, "observed months", "cividis")

    return figure


def _pick_month(stack: SiteStack, preferred: list[int]) -> str:
    """The best-observed month of a site matching one of the preferred months."""
    best_month, best_score = stack.year_month[0], -1.0
    for position, label in enumerate(stack.year_month):
        month = stack.calendar_month[position]
        if month not in preferred:
            continue
        observed = float(np.isfinite(stack.stacks["mndwi"][position]).mean())
        # Earlier entries in `preferred` win ties, so the intended season leads.
        score = observed - 0.01 * preferred.index(month)
        if score > best_score:
            best_month, best_score = label, score
    return best_month


def plot_pixel_timeseries(
    stack: SiteStack,
    features: dict[str, np.ndarray],
    row: int,
    col: int,
    indices: list[str] | None = None,
    cfg: Config | None = None,
    figsize: tuple[float, float] = (13, 3.2),
):
    """One pixel's history, with the harmonic fit and its residuals.

    This is the plot to read before deciding what the harmonic block is worth.
    A basin pixel fills sharply and drains slowly; the sinusoid cannot follow
    that, and the residual panel shows exactly where it fails and by how much.
    """
    indices = indices or [name for name in ("mndwi", "ndvi", "ndti") if name in stack.stacks]

    figure, axes = plt.subplots(
        1, len(indices), figsize=(figsize[0], figsize[1]), constrained_layout=True,
        squeeze=False,
    )
    times = stack.decimal_year

    for position, name in enumerate(indices):
        axis = axes[0][position]
        observed = stack.stacks[name][:, row, col].astype(float)

        axis.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)

        wet = np.isin(stack.calendar_month, [11, 12, 1, 2, 3, 4])
        axis.scatter(times[wet], observed[wet], s=16, color="#1D6FA5",
                     label="wet-season obs", zorder=3)
        axis.scatter(times[~wet], observed[~wet], s=16, color="#B5651D",
                     label="dry-season obs", zorder=3)
        axis.plot(times, observed, color="#666666", linewidth=0.7, alpha=0.6, zorder=2)

        fitted_key = f"{name}_harm_fitted"
        if fitted_key in features:
            fitted = features[fitted_key][:, row, col].astype(float)
            axis.plot(times, fitted, color="#111111", linewidth=1.4,
                      label="harmonic fit", zorder=4)
            residual = observed - fitted
            axis.vlines(times, fitted, observed, color="#CC3311", linewidth=0.8,
                        alpha=0.7, zorder=1)
            rmse = float(np.sqrt(np.nanmean(residual**2)))
            worst = float(np.nanmax(np.abs(residual))) if np.isfinite(residual).any() else np.nan
            subtitle = f"  resid RMSE {rmse:.3f}, max |resid| {worst:.3f}"
        else:
            subtitle = "  (harmonic disabled)"

        wet_max_key = f"{name}_wet_max"
        if wet_max_key in features:
            axis.axhline(
                float(features[wet_max_key][row, col]),
                color="#1D6FA5", linestyle="dashed", linewidth=1.0,
                label="wet max / dry min",
            )
        dry_min_key = f"{name}_dry_min"
        if dry_min_key in features:
            axis.axhline(
                float(features[dry_min_key][row, col]),
                color="#B5651D", linestyle="dashed", linewidth=1.0,
            )

        axis.set_title(f"{name} at ({row}, {col}){subtitle}", fontsize=9)
        axis.set_xlabel("year")
        axis.tick_params(labelsize=8)
        if position == 0:
            axis.legend(fontsize=7, loc="best", framealpha=0.9)

    figure.suptitle(f"site {stack.site_id} — pixel time series", fontsize=11)
    return figure


def plot_feature_maps(
    features: dict[str, np.ndarray],
    names: list[str],
    n_columns: int = 4,
    mask: np.ndarray | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 2.9),
):
    """A grid of per-pixel feature maps, for eyeballing what each one encodes."""
    available = [name for name in names if name in features]
    missing = [name for name in names if name not in features]
    if missing:
        print(f"  not present, skipped: {missing}")
    if not available:
        raise KeyError("none of the requested features are present")

    n_rows = int(np.ceil(len(available) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(figsize_per_panel[0] * n_columns, figsize_per_panel[1] * n_rows),
        constrained_layout=True, squeeze=False,
    )

    for axis in axes.ravel():
        axis.set_axis_off()

    for position, name in enumerate(available):
        axis = axes[position // n_columns][position % n_columns]
        axis.set_axis_on()
        data = features[name]
        if data.ndim == 3:
            # Pixels unobserved in every month are expected (the reprojection
            # border, persistent cloud); they plot as blank rather than warn.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data = np.nanmedian(data, axis=0)
            title = f"{name}\n(median over months)"
        else:
            title = name
        diverging = any(
            token in name for token in ("residual", "trend", "phase", "minus", "anomaly")
        )
        _show(axis, data, title, SCORE_CMAP if diverging else "viridis", diverging=diverging)
        if mask is not None and mask.any():
            axis.contour(mask, levels=[0.5], colors="#000000", linewidths=0.7)

    return figure


def plot_harmonic_vs_model_free(
    features: dict[str, np.ndarray],
    index_name: str = "mndwi",
    mask: np.ndarray | None = None,
    figsize: tuple[float, float] = (5.5, 5.0),
):
    """Harmonic amplitude against raw seasonal range, basin pixels highlighted.

    Both claim to measure "how much does this pixel swing seasonally". If the
    basin separates further along one axis than the other, that is the axis
    worth keeping.
    """
    amplitude = features[f"{index_name}_harm_amplitude"]
    seasonal_range = features[f"{index_name}_seasonal_range"]

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    finite = np.isfinite(amplitude) & np.isfinite(seasonal_range)

    background = finite if mask is None else finite & ~mask
    axis.scatter(
        seasonal_range[background], amplitude[background],
        s=3, alpha=0.25, color="#888888", label="matrix", rasterized=True,
    )

    if mask is not None and mask.any():
        inside = finite & mask
        axis.scatter(
            seasonal_range[inside], amplitude[inside],
            s=10, alpha=0.85, color="#1D6FA5", label="basin footprint",
        )
        separation = (
            f"\nbasin median: range {np.nanmedian(seasonal_range[inside]):.3f}, "
            f"amplitude {np.nanmedian(amplitude[inside]):.3f}"
        )
    else:
        separation = ""

    axis.set_xlabel(f"{index_name} seasonal range (wet max - dry min)")
    axis.set_ylabel(f"{index_name} harmonic amplitude")
    axis.set_title(f"model-free vs harmonic{separation}", fontsize=10)
    axis.legend(fontsize=8)
    return figure


def plot_footprint_summary(results: pd.DataFrame, figsize: tuple[float, float] = (11, 3.6)):
    """Distribution of derived footprint sizes, and how many sites failed."""
    figure, (size_axis, status_axis) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )

    succeeded = results[results["succeeded"]]
    size_axis.hist(succeeded["n_pixels"], bins=40, color="#1D6FA5")
    size_axis.set_xlabel("footprint size (pixels, 100 m² each)")
    size_axis.set_ylabel("sites")
    size_axis.set_title(
        f"{len(succeeded)} of {len(results)} sites footprinted", fontsize=10
    )

    counts = results["reason"].replace("", "succeeded").value_counts()
    labels = [label[:44] + ("..." if len(label) > 44 else "") for label in counts.index]
    status_axis.barh(labels[::-1], counts.to_numpy()[::-1], color="#B5651D")
    status_axis.set_xlabel("sites")
    status_axis.set_title("outcome", fontsize=10)
    status_axis.tick_params(labelsize=8)

    return figure
