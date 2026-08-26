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
        # The site's own labelled extent: everything outside belongs to a
        # neighbouring waterhole or to nothing, and is excluded from the footprint.
        if getattr(footprint, "box_mask", None) is not None and footprint.box_mask.any():
            axis.contour(
                footprint.box_mask, levels=[0.5], colors="#ffff00",
                linewidths=1.0, linestyles="dotted",
            )
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
        hint = ""
        if any("_harm_" in name for name in missing):
            hint = (
                "\nEvery requested feature is a harmonic one, and "
                "features.temporal.harmonic_enabled is false in the config, so none "
                "were computed. Set it true and rebuild to plot these."
            )
        raise KeyError(
            f"none of the requested features are present. Asked for "
            f"{len(names)}, none of which exist.{hint}"
        )

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
    amplitude_key = f"{index_name}_harm_amplitude"
    if amplitude_key not in features:
        raise KeyError(
            f"{amplitude_key} was not computed, so there is no harmonic to compare "
            f"against. features.temporal.harmonic_enabled is false in the config — "
            f"set it true and rebuild the temporal features to use this plot."
        )

    amplitude = features[amplitude_key]
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


def class_overlay(mask: np.ndarray, cfg: Config, alpha: float = 0.75) -> np.ndarray:
    """Label mask as an RGBA overlay; class 0 is transparent."""
    rgba = np.zeros(mask.shape + (4,))
    for definition in cfg.classes:
        if definition.ignore:
            continue
        selected = mask == definition.id
        if selected.any():
            colour = plt.matplotlib.colors.to_rgba(definition.colour)
            rgba[selected] = (*colour[:3], alpha)
    return rgba


def class_legend(axis, cfg: Config, present: set[int] | None = None) -> None:
    """Legend of class colours, restricted to the classes actually drawn."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=d.colour, edgecolor="#333333", label=d.name)
        for d in cfg.classes
        if not d.ignore and (present is None or d.id in present)
    ]
    if handles:
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=7, frameon=True)


def plot_pseudo_labels(
    tile,
    pseudo_mask: np.ndarray,
    cfg: Config,
    params,
    footprint: np.ndarray | None = None,
    manual_mask: np.ndarray | None = None,
    figsize: tuple[float, float] = (16, 4.2),
):
    """Show a pseudo-label mask against the imagery and the rules that made it.

    Panels: true colour with the pseudo labels overlaid; MNDWI with the
    open-water threshold drawn as a contour; NDVI with the vegetation cutoff;
    and, where hand labels exist for the same tile, those alongside for
    comparison.

    The contours are the point — they show *why* a pixel was claimed, so a bad
    threshold is visible as a contour in the wrong place rather than as an
    unexplained blob.
    """
    import wh_indices

    mndwi = wh_indices.compute("mndwi", tile.bands)
    ndvi = wh_indices.compute("ndvi", tile.bands)

    n_panels = 4 if manual_mask is not None else 3
    figure, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)

    present = {int(v) for v in np.unique(pseudo_mask) if v}

    # --- RGB with the pseudo labels on top
    axes[0].imshow(rgb_composite(tile))
    axes[0].imshow(class_overlay(pseudo_mask, cfg), interpolation="nearest")
    axes[0].set_title(f"RGB + pseudo labels ({int((pseudo_mask > 0).sum())} px)", fontsize=9)

    # --- MNDWI, with the open-water rule drawn
    water_image = axes[1].imshow(
        mndwi, cmap="RdBu", norm=TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.6),
        interpolation="nearest",
    )
    plt.colorbar(water_image, ax=axes[1], fraction=0.046, pad=0.03)
    if np.isfinite(mndwi).any() and np.nanmax(mndwi) > params.open_water_mndwi_min:
        axes[1].contour(np.nan_to_num(mndwi, nan=-1.0),
                        levels=[params.open_water_mndwi_min],
                        colors="#00ff00", linewidths=1.2)
    axes[1].set_title(f"MNDWI (green = >{params.open_water_mndwi_min} rule)", fontsize=9)

    # --- NDVI, with the vegetation cutoff actually used for this tile
    veg_image = axes[2].imshow(ndvi, cmap="YlGn", vmin=-0.1, vmax=0.9, interpolation="nearest")
    plt.colorbar(veg_image, ax=axes[2], fraction=0.046, pad=0.03)
    observed = np.isfinite(ndvi)
    if observed.any():
        cutoff = max(
            float(np.percentile(ndvi[observed], params.vegetation_ndvi_percentile)),
            params.vegetation_ndvi_min,
        )
        axes[2].contour(np.nan_to_num(ndvi, nan=-1.0), levels=[cutoff],
                        colors="#0000ff", linewidths=0.9)
        axes[2].set_title(f"NDVI (blue = >{cutoff:.2f} cutoff)", fontsize=9)
    else:
        axes[2].set_title("NDVI", fontsize=9)

    # --- hand labels for the same tile, if any
    if manual_mask is not None:
        axes[3].imshow(rgb_composite(tile))
        axes[3].imshow(class_overlay(manual_mask, cfg), interpolation="nearest")
        axes[3].set_title(
            f"hand labels ({int((manual_mask > 0).sum())} px)", fontsize=9
        )
        present |= {int(v) for v in np.unique(manual_mask) if v}

    # The basin footprint bounds where surrounding vegetation may be claimed.
    for axis in axes:
        if footprint is not None and footprint.any():
            axis.contour(footprint, levels=[0.5], colors="#00ffff", linewidths=1.0)
        axis.set_xticks([])
        axis.set_yticks([])

    class_legend(axes[-1], cfg, present)
    figure.suptitle(
        f"site {tile.key.site_id}  {tile.key.year_month}  "
        f"(gaps {100 * tile.gap_fraction:.0f}%)", fontsize=11,
    )
    return figure


def plot_pseudo_agreement(
    agreement: pd.DataFrame, figsize: tuple[float, float] = (6.5, 5.0)
):
    """Heatmap of hand label vs pseudo label on the same pixels.

    The diagonal is agreement. Off-diagonal cells are where the automatic rule
    claimed something a person called different — the column that matters for
    deciding whether the pseudo-labels are safe to train on.
    """
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    normalised = agreement.div(agreement.sum(axis=1).replace(0, np.nan), axis=0)

    image = axis.imshow(normalised.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(agreement.columns)))
    axis.set_xticklabels(agreement.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(agreement.index)))
    axis.set_yticklabels(agreement.index, fontsize=8)
    axis.set_xlabel("pseudo label")
    axis.set_ylabel("hand label")
    axis.set_title("where both exist: does the rule agree with you?", fontsize=10)

    for row in range(agreement.shape[0]):
        for col in range(agreement.shape[1]):
            count = agreement.iat[row, col]
            if count:
                axis.text(col, row, f"{count:,}", ha="center", va="center", fontsize=7,
                          color="white" if normalised.iat[row, col] > 0.5 else "#333333")

    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="row fraction")
    return figure


# --- model diagnostics -----------------------------------------------------


def plot_confusion(
    evaluation,
    normalise: str = "true",
    figsize: tuple[float, float] = (7.5, 6.2),
):
    """Confusion matrix as a heatmap.

    Normalised by true class (row) by default, which is the reading that matters:
    "of the pixels that really were mud, where did they go?". Raw counts are
    printed in each cell so the support behind each rate stays visible — a 100%
    rate over 12 pixels is not the same claim as 100% over 12,000.
    """
    matrix = evaluation.confusion
    counts = matrix.to_numpy()

    if normalise == "true":
        totals = counts.sum(axis=1, keepdims=True)
        shown = np.divide(counts, np.where(totals == 0, np.nan, totals))
        label = "fraction of true class"
    elif normalise == "pred":
        totals = counts.sum(axis=0, keepdims=True)
        shown = np.divide(counts, np.where(totals == 0, np.nan, totals))
        label = "fraction of predicted class"
    else:
        shown = counts.astype(float)
        label = "pixels"

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalise else None)

    names = [name.replace("true_", "").replace("pred_", "") for name in matrix.index]
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(
        [c.replace("pred_", "") for c in matrix.columns], rotation=45, ha="right", fontsize=8
    )
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(
        f"{evaluation.model_name} — {evaluation.strategy}\n"
        f"macro F1 {evaluation.macro_f1:.3f}", fontsize=10,
    )

    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            value = shown[row, col]
            if not np.isfinite(value):
                continue
            axis.text(
                col, row, f"{counts[row, col]:,}", ha="center", va="center",
                fontsize=7, color="white" if value > 0.5 else "#333333",
            )

    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label=label)
    return figure


def plot_class_scores(evaluation, figsize: tuple[float, float] = (9, 4.2)):
    """Per-class F1 and IoU, with support shown so rarity is visible."""
    scores = evaluation.per_class.sort_values("f1")
    positions = np.arange(len(scores))

    figure, (score_axis, support_axis) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    width = 0.38
    score_axis.barh(positions + width / 2, scores["f1"], height=width,
                    color="#1D6FA5", label="F1")
    score_axis.barh(positions - width / 2, scores["iou"], height=width,
                    color="#B5651D", label="IoU")
    score_axis.set_yticks(positions)
    score_axis.set_yticklabels(scores.index, fontsize=8)
    score_axis.set_xlim(0, 1)
    score_axis.set_xlabel("score")
    score_axis.legend(fontsize=8)
    score_axis.set_title(f"{evaluation.model_name} — per class", fontsize=10)

    support_axis.barh(positions, scores["support"], color="#888888")
    support_axis.set_yticks(positions)
    support_axis.set_yticklabels([])
    support_axis.set_xscale("log")
    support_axis.set_xlabel("labelled pixels (log)")
    support_axis.set_title("support", fontsize=10)

    return figure


def plot_site_scores(evaluation, figsize: tuple[float, float] = (8, 4.0)):
    """Per-site macro F1 — where generalisation is actually failing.

    Each bar is a whole waterhole held out. An average over sites hides the case
    where the model works on six and fails on three, which is exactly what
    matters before applying it to 187.
    """
    scores = evaluation.per_site.sort_values("macro_f1")

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    colours = ["#B5651D" if value < 0.4 else "#1D6FA5" for value in scores["macro_f1"]]
    bars = axis.barh(np.arange(len(scores)), scores["macro_f1"], color=colours)

    axis.set_yticks(np.arange(len(scores)))
    axis.set_yticklabels(
        [f"{site} ({int(row.n_pixels):,} px, {int(row.n_classes)} cls)"
         for site, row in scores.iterrows()],
        fontsize=8,
    )
    axis.axvline(evaluation.macro_f1, color="#333333", linestyle="dashed", linewidth=1,
                 label=f"overall {evaluation.macro_f1:.3f}")
    axis.set_xlim(0, 1)
    axis.set_xlabel("macro F1 when this site is held out")
    axis.set_title(f"{evaluation.model_name} — per site", fontsize=10)
    axis.legend(fontsize=8)

    for bar, value in zip(bars, scores["macro_f1"]):
        axis.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                  f"{value:.2f}", va="center", fontsize=7)

    return figure


def plot_ablation(ablation: pd.DataFrame, figsize: tuple[float, float] = (8, 3.6)):
    """Macro F1 by feature set, with the instantaneous-only baseline marked."""
    # Sets that were not separately scored (identical columns to another set)
    # would otherwise plot as duplicate bars implying independent evidence.
    if "identical_to" in ablation.columns:
        ablation = ablation[ablation["identical_to"] == ""]
    ordered = ablation.sort_values("macro_f1")

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    colours = [
        "#B5651D" if name == "instantaneous_only" else "#1D6FA5"
        for name in ordered.index
    ]
    axis.barh(np.arange(len(ordered)), ordered["macro_f1"], color=colours)
    axis.set_yticks(np.arange(len(ordered)))
    axis.set_yticklabels(
        [f"{name}  ({int(n)} feat)" for name, n in
         zip(ordered.index, ordered["n_features"])], fontsize=8,
    )
    axis.set_xlabel("macro F1 (leave-one-site-out)")
    axis.set_xlim(0, max(0.7, ordered["macro_f1"].max() * 1.15))
    axis.set_title("does the temporal design earn its keep?", fontsize=10)

    if "instantaneous_only" in ordered.index:
        baseline = ordered.loc["instantaneous_only", "macro_f1"]
        axis.axvline(baseline, color="#B5651D", linestyle="dashed", linewidth=1)

    for position, value in enumerate(ordered["macro_f1"]):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=7)

    return figure


def plot_band_importance(
    importance: pd.DataFrame,
    n_bands: int = 25,
    reference: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (9, 7),
):
    """Ranked embedding-band importance, with the across-site spread shown.

    The error bars are the point. A band that looks essential in one fold and
    useless in the rest is not a finding, and a bare ranking of means would
    present it as one. A bar whose spread crosses zero has not earned its place.

    `reference` is the full importance table; when given, the best few
    non-embedding features are drawn alongside so "A17 matters" is calibrated
    against how much anything matters.
    """
    import wh_features

    bands = importance.head(n_bands).iloc[::-1]
    positions = np.arange(len(bands))

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    colours = [
        "#B5651D" if row.mean_importance - row.std_across_folds <= 0 else "#1D6FA5"
        for row in bands.itertuples()
    ]
    axis.barh(
        positions, bands["mean_importance"], xerr=bands["std_across_folds"],
        color=colours, error_kw={"ecolor": "#666666", "elinewidth": 0.8},
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{name}  ({int(row.n_folds_positive)}/{int(row.n_folds)})"
         for name, row in zip(bands.index, bands.itertuples())],
        fontsize=8,
    )
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel("drop in macro F1 when permuted (held-out sites)")
    axis.set_title(
        "AlphaEarth band importance\n"
        "orange = spread crosses zero; (n/N) = folds where it helped",
        fontsize=10,
    )

    if reference is not None:
        others = reference[
            ~reference.index.str.startswith(wh_features.ALPHAEARTH_PREFIX)
        ].head(5)
        if len(others):
            label = ", ".join(
                f"{name} {row.mean_importance:.3f}"
                for name, row in zip(others.index, others.itertuples())
            )
            axis.text(
                0.98, 0.02, f"best non-embedding features:\n{label}",
                transform=axis.transAxes, fontsize=7, ha="right", va="bottom",
                bbox={"facecolor": "#FFFFFF", "alpha": 0.85, "edgecolor": "#CCCCCC"},
            )

    return figure


# --- spatial inspection ----------------------------------------------------


def plot_prediction_map(
    tile,
    predicted: np.ndarray,
    cfg: Config,
    manual_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    footprint: np.ndarray | None = None,
    held_out: bool = True,
    alphaearth: np.ndarray | None = None,
    box_mask: np.ndarray | None = None,
    figsize: tuple[float, float] = (17, 4.3),
):
    """Where the model is right and wrong, spatially.

    Panels: true colour, the predicted class raster, your hand labels, and an
    agreement map showing which labelled pixels were got right. Optionally the
    model's confidence.

    The agreement panel is the point. Aggregate metrics say a class is weak;
    this says *where* — a basin margin, a shadowed edge, one corner of the tile —
    which is what tells you whether it is a model problem or a label problem.
    """
    panels = ["rgb"]
    if alphaearth is not None:
        panels.append("alphaearth")
    panels.append("predicted")
    if manual_mask is not None:
        panels += ["manual", "agreement"]
    if confidence is not None:
        panels.append("confidence")

    figsize = (max(figsize[0], 3.4 * len(panels)), figsize[1])

    figure, axes = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    present = {int(v) for v in np.unique(predicted) if v}

    for axis, panel in zip(axes, panels):
        if panel == "rgb":
            axis.imshow(rgb_composite(tile))
            axis.set_title("RGB", fontsize=9)

        elif panel == "alphaearth":
            # Static per site, so this panel is the same in every month — it is
            # structural context, not a reading of this month's surface.
            axis.imshow(alphaearth, interpolation="nearest")
            axis.set_title("AlphaEarth", fontsize=9)

        elif panel == "predicted":
            axis.imshow(rgb_composite(tile))
            axis.imshow(class_overlay(predicted, cfg, alpha=0.8), interpolation="nearest")
            axis.set_title("predicted", fontsize=9)

        elif panel == "manual":
            axis.imshow(rgb_composite(tile))
            axis.imshow(class_overlay(manual_mask, cfg, alpha=0.8), interpolation="nearest")
            axis.set_title(f"hand labels ({int((manual_mask > 0).sum())} px)", fontsize=9)
            present |= {int(v) for v in np.unique(manual_mask) if v}

        elif panel == "agreement":
            labelled = manual_mask > 0
            correct = labelled & (predicted == manual_mask)
            overlay = np.zeros(tile.shape + (4,))
            overlay[labelled & ~correct] = plt.matplotlib.colors.to_rgba("#D62728")
            overlay[correct] = plt.matplotlib.colors.to_rgba("#2CA02C")
            axis.imshow(rgb_composite(tile))
            axis.imshow(overlay, interpolation="nearest")
            accuracy = correct.sum() / labelled.sum() if labelled.any() else np.nan
            axis.set_title(
                f"green = correct, red = wrong\n{accuracy:.0%} of {int(labelled.sum()):,} "
                f"labelled px", fontsize=9,
            )

        elif panel == "confidence":
            # Most pixels sit at ~1.0, so a fixed 0-1 scale renders as a flat
            # wash and hides the only interesting part — where the model is
            # unsure. Stretched to this tile's own low tail instead.
            finite = confidence[np.isfinite(confidence)]
            low = float(np.percentile(finite, 2)) if finite.size else 0.0
            low = min(low, 0.95)
            image = axis.imshow(confidence, cmap="cividis", vmin=low, vmax=1.0,
                                interpolation="nearest")
            plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            median = float(np.nanmedian(confidence)) if finite.size else np.nan
            axis.set_title(
                f"confidence (median {median:.2f})\nstretched {low:.2f}-1.00", fontsize=9
            )

        # Cyan solid = footprint (this basin), yellow dotted = the labelled
        # bounding box (this site's extent). Anything outside the box belongs to
        # a neighbouring waterhole, which half these tiles contain.
        if box_mask is not None and box_mask.any():
            axis.contour(box_mask, levels=[0.5], colors="#ffff00",
                         linewidths=1.0, linestyles="dotted")
        if footprint is not None and footprint.any():
            axis.contour(footprint, levels=[0.5], colors="#00ffff", linewidths=1.0)
        axis.set_xticks([])
        axis.set_yticks([])

    class_legend(axes[-1], cfg, present)
    provenance = (
        "model has NOT seen this site" if held_out else "WARNING: model was trained on this site"
    )
    figure.suptitle(
        f"site {tile.key.site_id}  {tile.key.year_month}   ({provenance})", fontsize=11
    )
    return figure


def plot_error_by_class_map(
    tile, predicted: np.ndarray, manual_mask: np.ndarray, cfg: Config,
    figsize: tuple[float, float] = (11, 4.3),
):
    """For each hand-labelled class, where its pixels were misclassified to.

    Answers "my mud is being called dry ground — is that everywhere, or only on
    the basin margin?", which the confusion matrix cannot.
    """
    labelled_classes = [int(v) for v in np.unique(manual_mask) if v]
    if not labelled_classes:
        raise ValueError("this tile has no hand labels")

    figure, axes = plt.subplots(
        1, len(labelled_classes), figsize=figsize, constrained_layout=True, squeeze=False
    )

    for axis, class_id in zip(axes[0], labelled_classes):
        selected = manual_mask == class_id
        wrong = selected & (predicted != manual_mask)

        axis.imshow(rgb_composite(tile))
        overlay = np.zeros(tile.shape + (4,))
        overlay[selected & ~wrong] = plt.matplotlib.colors.to_rgba("#2CA02C")
        for other in np.unique(predicted[wrong]) if wrong.any() else []:
            mistaken = wrong & (predicted == other)
            colour = cfg.class_by_id(int(other)).colour
            overlay[mistaken] = plt.matplotlib.colors.to_rgba(colour)
        axis.imshow(overlay, interpolation="nearest")

        name = cfg.class_by_id(class_id).name
        recall = (selected & ~wrong).sum() / selected.sum()
        axis.set_title(
            f"true {name}\n{recall:.0%} correct of {int(selected.sum()):,} px", fontsize=9
        )
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"site {tile.key.site_id} {tile.key.year_month} — "
        f"green = correct, other colours = what it was mistaken for", fontsize=10,
    )
    return figure


# --- composition through time ----------------------------------------------


def plot_site_composition(
    table: pd.DataFrame,
    site_id: str,
    cfg: Config,
    denominator: str = "bbox",
    figsize: tuple[float, float] = (12, 4.2),
):
    """Stacked-area composition for one waterhole through time.

    Months flagged as isolated-wet are marked rather than removed: the flag says
    the value is suspicious, not that it is wrong, and deciding is the analyst's
    job.
    """
    site = table[table["site_id"] == site_id].sort_values(["year", "month"]).copy()
    if site.empty:
        raise KeyError(f"no rows for site {site_id}")

    site["date"] = pd.to_datetime(site["year_month"] + "-01")
    names = [d.name for d in cfg.classes if not d.ignore]
    columns = [f"{denominator}_frac_{name}" for name in names]

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.stackplot(
        site["date"],
        *[site[column].fillna(0.0) for column in columns],
        labels=names,
        colors=[cfg.class_by_name(name).colour for name in names],
    )

    # Unobserved months would otherwise read as an abrupt change in composition.
    missing = site[site[f"{denominator}_n_classified"].fillna(0) == 0]
    for date in missing["date"]:
        axis.axvspan(date, date, color="#ffffff", alpha=0.0)
    if len(missing):
        axis.scatter(missing["date"], np.full(len(missing), 1.02), marker="v",
                     s=18, color="#999999", label="no data")

    flagged = site[site.get("flag_isolated_wet", False) == True]  # noqa: E712
    if len(flagged):
        axis.scatter(flagged["date"], np.full(len(flagged), 1.06), marker="*",
                     s=45, color="#D62728", label="isolated wet (flagged)")

    axis.set_ylim(0, 1.10)
    axis.set_ylabel(f"fraction within the {denominator}")
    axis.set_title(
        f"site {site_id} — surface composition through time "
        f"({site['label'].iloc[0] if 'label' in site else ''})", fontsize=10,
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    return figure


def plot_composition_heatmap(
    table: pd.DataFrame,
    cfg: Config,
    class_name: str = "open_water",
    denominator: str = "bbox",
    figsize: tuple[float, float] = (13, 9),
):
    """Site x month heatmap of one class's fraction, all waterholes at once.

    The seasonal banding should be obvious as vertical stripes; a site that looks
    unlike its neighbours is either genuinely different or a site the model does
    not handle, and the confidence column is the way to tell those apart.
    """
    column = f"{denominator}_frac_{class_name}"
    if column not in table.columns:
        raise KeyError(f"{column} not in the table")

    wide = table.pivot_table(
        index="site_id", columns="year_month", values=column, aggfunc="mean"
    )

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    image = axis.imshow(wide.to_numpy(), aspect="auto", cmap="YlGnBu",
                        vmin=0, vmax=float(np.nanpercentile(wide.to_numpy(), 99)) or 1)

    step = max(1, len(wide.columns) // 14)
    axis.set_xticks(range(0, len(wide.columns), step))
    axis.set_xticklabels(wide.columns[::step], rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(wide.index)))
    axis.set_yticklabels(wide.index, fontsize=5)
    axis.set_title(f"{class_name} fraction within the {denominator}", fontsize=10)
    plt.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label="fraction")
    return figure


def plot_composition_quality(table: pd.DataFrame, figsize: tuple[float, float] = (11, 3.6)):
    """Where the composition series can and cannot be trusted."""
    figure, (quality_axis, confidence_axis) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )

    order = ["good", "fair", "thin", "poor"]
    counts = table["data_quality"].value_counts().reindex(order).fillna(0)
    colours = {"good": "#2CA02C", "fair": "#1D6FA5", "thin": "#B5651D", "poor": "#D62728"}
    quality_axis.bar(order, counts, color=[colours[k] for k in order])
    quality_axis.set_ylabel("site-months")
    quality_axis.set_title(
        f"data quality ({100 * counts.get('good', 0) / len(table):.0f}% good)", fontsize=10
    )

    # Confidence is optional — write_confidence=False halves the run time and
    # leaves this column entirely NaN — so the panel explains its absence rather
    # than failing on an empty histogram.
    by_site = (
        table.groupby("site_id")["mean_confidence"].mean().dropna().sort_values()
    )

    if by_site.empty:
        confidence_axis.text(
            0.5, 0.5,
            "no confidence recorded\n\nset write_confidence=True in PredictParams\n"
            "and re-run to see which sites the model\nis least sure about",
            ha="center", va="center", fontsize=9, color="#666666",
            transform=confidence_axis.transAxes,
        )
        confidence_axis.set_xticks([])
        confidence_axis.set_yticks([])
        confidence_axis.set_title("prediction confidence (not written)", fontsize=10)
        return figure

    confidence_axis.hist(by_site, bins=min(30, max(5, len(by_site))), color="#1D6FA5")
    confidence_axis.set_xlabel("mean prediction confidence, per site")
    confidence_axis.set_ylabel("sites")
    confidence_axis.set_title(
        f"prediction confidence over {len(by_site)} sites\n"
        "the model saw far fewer than it is applied to; low values mark the unlike ones",
        fontsize=9,
    )
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
