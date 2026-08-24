"""Interactive sparse labeller for waterhole surface state.

Opens a native matplotlib window driven from a notebook cell. Painting happens on
the GeoTIFF's own grid — the panels are rendered live from the raster, and the
mask is written back on exactly that grid, at exactly that size. Nothing is ever
resampled, because a resampled label is a wrong label.

Design notes worth knowing before you use it:

  * Month stepping keeps the label buffer. Buffers are held per (site, month) in
    memory for the whole session, so you can walk back and forth through a site's
    history to decide what a pixel is, without losing work. Nothing is written to
    disk until you save.

  * The pre-rendered PNG cannot be co-registered with the live panels. It is a
    matplotlib figure with padding, titles and colourbars, not a raster, so it
    opens in its own window rather than as an aligned overlay.

  * Only paint what you are sure of. At 10 m most basin margins are mixed pixels;
    leaving them as class 0 is the correct answer, not a failure to finish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Patch
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath

import wh_indices
import wh_tiles
from wh_config import Config

RGB_PANEL = "rgb"
ALPHAEARTH_PANEL = "alphaearth"

# Matplotlib binds a lot of single letters by default, and almost every one of
# them collides with a labelling key: 'p' toggles pan, 's' and 'ctrl+s' open the
# save-figure dialog, 'f' goes fullscreen, 'h' resets the view, and left/right/'v'
# walk the view history. Left in place they either shadow a binding or fire in
# addition to it. These are cleared for the labelling window and restored when it
# closes, so other figures in the session keep their normal behaviour.
_MPL_KEYMAPS_TO_CLEAR = (
    "keymap.save",
    "keymap.fullscreen",
    "keymap.home",
    "keymap.back",
    "keymap.forward",
    "keymap.pan",
    "keymap.zoom",
    "keymap.grid",
    "keymap.grid_minor",
    "keymap.quit",
    "keymap.quit_all",
    "keymap.xscale",
    "keymap.yscale",
    "keymap.copy",
    "keymap.help",
)


def _normalise_key(key: str | None) -> str | None:
    """Fold the platform's modifier spellings onto one name.

    On macOS the Qt backend reports the Command key as 'ctrl' (Qt swaps Control
    and Command there), while the macosx backend reports it as 'cmd'. Accepting
    every spelling means the same binding works on either backend and on Linux,
    and that Cmd behaves the way a Mac user expects.
    """
    if key is None:
        return None
    for alias in ("cmd+", "super+", "meta+"):
        key = key.replace(alias, "ctrl+")
    return key


@dataclass
class LabelParams:
    """Everything tunable about the labelling session.

    Held here rather than in the YAML so the notebook can show and edit it in one
    visible block. The class scheme itself stays in the config, because training
    and prediction have to agree with it.
    """

    # Order is reading order across the grid: with panel_rows=2 the first three
    # fill the top row, the rest the bottom.
    panels: tuple[str, ...] = (
        RGB_PANEL, "mndwi", ALPHAEARTH_PANEL, "ndvi", "ndti", "ndmi",
    )
    # Panels are laid out on a grid rather than one long row: with six panels a
    # single row leaves each one too small to paint into on a normal screen.
    panel_rows: int = 2

    # Which embedding dimensions the AlphaEarth panel shows.
    #   "bands" -> the three named below, straight to R, G, B
    #   "pca"   -> the first three principal components of all 64
    alphaearth_mode: str = "bands"
    alphaearth_bands: tuple[str, str, str] = ("A60", "A24", "A63")
    alphaearth_year: int = 2025
    rgb_bands: tuple[str, str, str] = ("B4", "B3", "B2")
    rgb_max_reflectance: float = 0.30
    rgb_gamma: float = 0.85
    display_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "mndwi": (-0.8, 0.6),
            "ndwi": (-0.8, 0.6),
            "ndvi": (-0.1, 0.9),
            "ndti": (-0.4, 0.4),
            "ndmi": (-0.6, 0.6),
        }
    )
    diverging_panels: tuple[str, ...] = ("mndwi", "ndwi", "ndti", "ndmi")
    brush_radius_px: int = 2
    min_brush_radius_px: int = 0
    max_brush_radius_px: int = 20
    undo_depth: int = 200
    label_alpha: float = 0.55
    initial_zoom_half_width_px: int = 40
    nodata_colour: str = "#ff00ff"
    labeller_name: str = "scott"

    # Saving is automatic by default: any tile with unsaved labels is written
    # when you navigate away from it, and everything outstanding is written when
    # the window closes. Painting is absorbing enough without having to remember
    # a keystroke, and an empty mask is never written, so autosave cannot create
    # spurious files.
    autosave_on_navigate: bool = True
    save_on_quit: bool = True

    # Which month a site opens on: the best-observed month among these calendar
    # months. Late dry season by default — the basin floor and its margin are
    # most interpretable when the water has drawn down.
    start_month_preference: tuple[int, ...] = (9, 10, 8, 7)


class _UndoStack:
    """Bounded undo/redo over sparse mask edits.

    Stores only the pixels a stroke touched and their previous values, so a
    session's history costs a few kilobytes rather than a copy of the mask per
    stroke.
    """

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self._undo: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._redo: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def record(self, rows: np.ndarray, cols: np.ndarray, before: np.ndarray,
               after: np.ndarray) -> None:
        if rows.size == 0:
            return
        self._undo.append((rows, cols, before))
        self._redo.clear()
        if len(self._undo) > self.depth:
            self._undo.pop(0)

    def undo(self, mask: np.ndarray) -> bool:
        if not self._undo:
            return False
        rows, cols, before = self._undo.pop()
        current = mask[rows, cols].copy()
        mask[rows, cols] = before
        self._redo.append((rows, cols, current))
        return True

    def redo(self, mask: np.ndarray) -> bool:
        if not self._redo:
            return False
        rows, cols, after = self._redo.pop()
        current = mask[rows, cols].copy()
        mask[rows, cols] = after
        self._undo.append((rows, cols, current))
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class Labeller:
    """The interactive labelling session.

    Construct via `launch()` from a notebook cell rather than directly.
    """

    def __init__(
        self,
        queue: pd.DataFrame,
        manifest: pd.DataFrame,
        cfg: Config,
        params: LabelParams,
        footprints: dict[str, np.ndarray] | None = None,
    ) -> None:
        if queue.empty:
            raise ValueError("the work queue is empty")

        self.cfg = cfg
        self.params = params
        self.queue = queue.reset_index(drop=True)
        # The full manifest, not the queue: month stepping must reach every month
        # of a site, including the ones filtered out of the labelling queue.
        self.all_months = manifest.sort_values(["site_id", "month_index"])
        self.footprints = footprints or {}

        self.classes = cfg.classes
        self.class_ids = [definition.id for definition in self.classes]
        self.key_to_class = {definition.key: definition.id for definition in self.classes}
        self.active_class = next(
            definition.id for definition in self.classes if not definition.ignore
        )

        self.position = 0
        self.brush_radius = params.brush_radius_px
        self.mode = "brush"
        self.show_labels = True
        self.show_footprint = True
        self.png_window = None

        # Label buffers survive month and tile navigation for the whole session.
        self.buffers: dict[tuple[str, str], np.ndarray] = {}
        self.dirty: set[tuple[str, str]] = set()
        self.undo = _UndoStack(params.undo_depth)

        self.tile = None
        self.index_cache: dict[str, np.ndarray] = {}
        self._alphaearth = None
        self._alphaearth_site = None
        self._painting = False

        self._set_current_from_queue()
        self._saved_keymaps = {
            name: list(plt.rcParams[name]) for name in _MPL_KEYMAPS_TO_CLEAR
        }
        for name in _MPL_KEYMAPS_TO_CLEAR:
            plt.rcParams[name] = []

        self._build_figure()
        self._load_current()
        self._connect()

    def _restore_keymaps(self, event=None) -> None:
        """Give matplotlib its default shortcuts back when the window closes.

        Connected to close_event so shutting the window from its title bar is
        enough, and called directly by close() because not every backend emits
        close_event.
        """
        for name, value in self._saved_keymaps.items():
            plt.rcParams[name] = value

    def _on_close(self, event=None) -> None:
        """Window closed from its title bar: save outstanding work, restore keys."""
        if self.dirty and self.params.save_on_quit:
            written = self.save_all(announce=False)
            print(f"saved {len(written)} tile(s) on close")
        self._restore_keymaps()

    def close(self) -> None:
        """Save outstanding work, then close the window and restore shortcuts."""
        if self.dirty:
            if self.params.save_on_quit:
                written = self.save_all(announce=False)
                print(f"saved {len(written)} tile(s) on quit")
            else:
                tiles = ", ".join(f"{site} {month}" for site, month in sorted(self.dirty))
                print(
                    f"warning: closing with unsaved labels on {len(self.dirty)} "
                    f"tile(s): {tiles}"
                )
        self._restore_keymaps()
        plt.close(self.figure)
        if self.png_window is not None:
            plt.close(self.png_window)
            self.png_window = None

    # --- current position ------------------------------------------------

    @property
    def row(self) -> dict:
        """The tile currently on screen.

        Not the queue row itself: browsing months moves this without disturbing
        the queue, so 'next tile' always returns to the tile you were sent to
        label rather than wherever you happened to browse to.
        """
        return self.current

    @property
    def queue_row(self) -> pd.Series:
        return self.queue.iloc[self.position]

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.current["site_id"]), str(self.current["year_month"]))

    @property
    def mask(self) -> np.ndarray:
        return self.buffers[self.key]

    def _set_current_from_queue(self) -> None:
        """Point the view at the queued site's starting month."""
        row = self.queue_row
        self.current = {
            "site_id": str(row["site_id"]),
            "year_month": str(row["start_year_month"]),
            "tif_path": row["start_tif_path"],
            "month_index": int(row["start_month_index"]),
        }

    def _site_months(self) -> pd.DataFrame:
        """All months of the current site, for temporal stepping.

        Drawn from the full manifest rather than the filtered queue, so stepping
        through time is not limited to the tiles selected for labelling.
        """
        site = self.current["site_id"]
        return self.all_months[self.all_months["site_id"] == site].sort_values(
            "month_index"
        )

    # --- loading ----------------------------------------------------------

    def _load_current(self) -> None:
        path = self.row["tif_path"]
        try:
            self.tile = wh_tiles.read_tile(path, self.cfg)
        except OSError as error:
            self._set_status(f"CANNOT READ TILE — {str(error).splitlines()[0]}")
            raise

        self.index_cache = {}
        for panel in self.params.panels:
            if panel not in (RGB_PANEL, ALPHAEARTH_PANEL):
                self.index_cache[panel] = wh_indices.compute(panel, self.tile.bands)

        # Static per site, so only reloaded when the site changes — stepping
        # through months leaves it untouched.
        if ALPHAEARTH_PANEL in self.params.panels:
            site = self.current["site_id"]
            if self._alphaearth_site != site:
                self._alphaearth = self._load_alphaearth(site)
                self._alphaearth_site = site

        message = ""
        if self.key not in self.buffers:
            # The buffer must be registered before anything reads self.mask,
            # which _set_status does.
            mask, message = self._load_existing_mask()
            self.buffers[self.key] = mask

        self.undo.clear()
        self._draw()
        if message:
            self._set_status(message)

    def _load_alphaearth(self, site_id: str):
        """The site's embedding composite, or None if it has not been downloaded."""
        import wh_features

        params = wh_features.FeatureParams(
            use_alphaearth=True, alphaearth_year=self.params.alphaearth_year
        )
        try:
            bands = wh_features.load_alphaearth(
                self.cfg, site_id, params, expected_shape=self.tile.shape
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"  no AlphaEarth panel for site {site_id}: {error}")
            return None

        return wh_features.alphaearth_composite(
            bands, bands=self.params.alphaearth_bands, mode=self.params.alphaearth_mode
        )

    def _load_existing_mask(self) -> tuple[np.ndarray, str]:
        """Reopen an existing mask for editing, so sessions are resumable."""
        path = wh_tiles.label_path_for(self.row["tif_path"], self.cfg)
        if path.exists():
            mask = wh_tiles.read_mask(path, self.tile.shape)
            return mask, f"loaded existing mask ({int((mask > 0).sum())} px)"
        return np.zeros(self.tile.shape, dtype=np.uint8), ""

    # --- figure -----------------------------------------------------------

    def _build_figure(self) -> None:
        n_panels = len(self.params.panels)
        rows = max(1, min(self.params.panel_rows, n_panels))
        columns = int(np.ceil(n_panels / rows))

        self.figure, grid = plt.subplots(
            rows, columns,
            figsize=(3.6 * columns, 3.8 * rows),
            constrained_layout=True,
        )
        grid = np.atleast_1d(grid).ravel()

        # Spare cells when the panels do not fill the grid exactly.
        for spare in grid[n_panels:]:
            spare.set_axis_off()
        self.axes = grid[:n_panels]

        # Shared axes keep pan and zoom locked across every panel, including
        # across rows.
        for axis in self.axes[1:]:
            axis.sharex(self.axes[0])
            axis.sharey(self.axes[0])

        colours = ["#00000000"] + [
            definition.colour for definition in self.classes if not definition.ignore
        ]
        self.label_cmap = ListedColormap(colours)

        self.base_images: list = []
        self.label_images: list = []
        self.nodata_images: list = []
        self.footprint_artists: list = []

        self.status = self.figure.text(
            0.01, 0.005, "", fontsize=8, family="monospace", va="bottom"
        )
        self.readout = self.figure.text(
            0.99, 0.005, "", fontsize=8, family="monospace", va="bottom", ha="right"
        )
        self.brush_marker = None

    def _draw(self) -> None:
        """Full redraw. Tiles are ~150x150 px, so this is cheap enough to do whole."""
        for axis in self.axes:
            axis.clear()

        self.base_images = []
        self.label_images = []
        self.nodata_images = []
        self.footprint_artists = []

        gap = ~self.tile.valid
        nodata_rgba = np.zeros(self.tile.shape + (4,))
        nodata_rgba[gap] = plt.matplotlib.colors.to_rgba(self.params.nodata_colour)

        for axis, panel in zip(self.axes, self.params.panels):
            if panel == RGB_PANEL:
                base = axis.imshow(self._rgb(), interpolation="nearest")
                axis.set_title("RGB", fontsize=9)
            elif panel == ALPHAEARTH_PANEL:
                if self._alphaearth is None:
                    base = axis.imshow(
                        np.zeros(self.tile.shape), cmap="gray", interpolation="nearest"
                    )
                    axis.set_title("AlphaEarth (not downloaded)", fontsize=9)
                else:
                    base = axis.imshow(self._alphaearth, interpolation="nearest")
                    label = (
                        "PCA 1-3" if self.params.alphaearth_mode == "pca"
                        else "/".join(self.params.alphaearth_bands)
                    )
                    axis.set_title(f"AlphaEarth {label}", fontsize=9)
            else:
                data = self.index_cache[panel]
                low, high = self.params.display_ranges.get(panel, (-1.0, 1.0))
                if panel in self.params.diverging_panels:
                    norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
                    base = axis.imshow(data, cmap="RdBu", norm=norm, interpolation="nearest")
                else:
                    base = axis.imshow(
                        data, cmap="YlGn", vmin=low, vmax=high, interpolation="nearest"
                    )
                axis.set_title(panel.upper(), fontsize=9)

            self.base_images.append(base)
            self.nodata_images.append(axis.imshow(nodata_rgba, interpolation="nearest"))

            label_image = axis.imshow(
                self._label_rgba(), interpolation="nearest", zorder=5
            )
            self.label_images.append(label_image)

            mask = self.footprints.get(self.row["site_id"])
            if mask is not None and mask.shape == self.tile.shape and mask.any():
                artist = axis.contour(
                    mask, levels=[0.5], colors="#00ffff", linewidths=1.0, zorder=6
                )
                self.footprint_artists.append(artist)

            axis.set_xticks([])
            axis.set_yticks([])

        if self.params.initial_zoom_half_width_px:
            height, width = self.tile.shape
            half = self.params.initial_zoom_half_width_px
            self.axes[0].set_xlim(width / 2 - half, width / 2 + half)
            self.axes[0].set_ylim(height / 2 + half, height / 2 - half)

        self._draw_legend()
        self._set_status()
        self.figure.canvas.draw_idle()

    def _rgb(self) -> np.ndarray:
        stack = np.dstack([self.tile.bands[band] for band in self.params.rgb_bands])
        stack = np.clip(
            stack.astype("float32") / self.params.rgb_max_reflectance, 0, 1
        )
        return np.power(stack, self.params.rgb_gamma)

    def _label_rgba(self) -> np.ndarray:
        """Labels as a translucent overlay; class 0 is fully transparent."""
        rgba = np.zeros(self.tile.shape + (4,))
        if not self.show_labels:
            return rgba
        for definition in self.classes:
            if definition.ignore:
                continue
            selected = self.mask == definition.id
            if selected.any():
                colour = plt.matplotlib.colors.to_rgba(definition.colour)
                rgba[selected] = (*colour[:3], self.params.label_alpha)
        return rgba

    def _draw_legend(self) -> None:
        handles = [
            Patch(
                facecolor=definition.colour,
                edgecolor="#333333",
                label=f"[{definition.key}] {definition.name}",
            )
            for definition in self.classes
            if not definition.ignore
        ]
        self.axes[-1].legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=7, frameon=True,
        )

    # --- painting ---------------------------------------------------------

    def _brush_indices(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        radius = self.brush_radius
        height, width = self.tile.shape
        row_low, row_high = max(0, row - radius), min(height, row + radius + 1)
        col_low, col_high = max(0, col - radius), min(width, col + radius + 1)

        rows, cols = np.mgrid[row_low:row_high, col_low:col_high]
        within = (rows - row) ** 2 + (cols - col) ** 2 <= radius**2
        return rows[within], cols[within]

    def _paint(self, row: int, col: int) -> None:
        rows, cols = self._brush_indices(row, col)
        if rows.size == 0:
            return

        value = 0 if self.mode == "erase" else self.active_class
        before = self.mask[rows, cols].copy()
        if np.all(before == value):
            return

        self.mask[rows, cols] = value
        self.undo.record(rows, cols, before, np.full(rows.shape, value, dtype=np.uint8))
        self.dirty.add(self.key)
        self._refresh_labels()

    def _fill_polygon(self, vertices: list[tuple[float, float]]) -> None:
        if len(vertices) < 3:
            return
        height, width = self.tile.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        points = np.column_stack([cols.ravel(), rows.ravel()])
        inside = MplPath(vertices).contains_points(points).reshape(height, width)

        rows_idx, cols_idx = np.nonzero(inside)
        if rows_idx.size == 0:
            return

        value = 0 if self.mode == "erase" else self.active_class
        before = self.mask[rows_idx, cols_idx].copy()
        self.mask[rows_idx, cols_idx] = value
        self.undo.record(
            rows_idx, cols_idx, before, np.full(rows_idx.shape, value, dtype=np.uint8)
        )
        self.dirty.add(self.key)
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        rgba = self._label_rgba()
        for image in self.label_images:
            image.set_data(rgba)
        self._set_status()
        self.figure.canvas.draw_idle()

    # --- saving -----------------------------------------------------------

    def save(self) -> Path | None:
        """Write the current tile's mask and its JSON sidecar."""
        if not self.mask.any():
            self._set_status("nothing to save (mask is empty)")
            return None

        path = wh_tiles.label_path_for(self.row["tif_path"], self.cfg)
        wh_tiles.write_mask(path, self.mask, self.tile)
        self._write_sidecar(
            self.row["tif_path"], path,
            self.current["site_id"], self.current["year_month"], self.mask,
        )

        self.dirty.discard(self.key)
        self._set_status(f"saved {int((self.mask > 0).sum())} px -> {path.name}")
        return path

    def _write_sidecar(
        self, tile_path, mask_path: Path, site_id: str, year_month: str,
        mask: np.ndarray,
    ) -> None:
        """Record what produced a mask, beside it."""
        sidecar = {
            "source_tile": str(tile_path),
            "label_mask": str(mask_path),
            "site_id": str(site_id),
            "year_month": str(year_month),
            "class_scheme_version": self.cfg["classes"]["scheme_version"],
            "config_hash": self.cfg.hash,
            "labeller": self.params.labeller_name,
            "source": "manual",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pixel_counts": {
                definition.name: int((mask == definition.id).sum())
                for definition in self.classes
            },
            "n_labelled": int((mask > 0).sum()),
        }
        wh_tiles.sidecar_path_for(tile_path, self.cfg).write_text(
            json.dumps(sidecar, indent=1)
        )

    def save_all(self, announce: bool = True) -> list[Path]:
        """Write every tile with unsaved labels, not just the one on screen."""
        written = []
        for site_id, year_month in sorted(self.dirty):
            written.append(self._save_buffer(site_id, year_month))

        written = [path for path in written if path is not None]
        if announce:
            self._set_status(
                f"saved {len(written)} tile(s)" if written else "nothing to save"
            )
        return written

    def _save_buffer(self, site_id: str, year_month: str) -> Path | None:
        """Save one buffer by key, which may not be the tile on screen.

        Writing a mask needs the source tile's grid, so a buffer for a tile that
        is not loaded is written against a fresh read of that tile rather than
        against whatever happens to be on screen.
        """
        key = (site_id, year_month)
        mask = self.buffers.get(key)
        if mask is None or not mask.any():
            self.dirty.discard(key)
            return None

        if key == self.key:
            return self.save()

        rows = self.all_months[
            (self.all_months["site_id"] == site_id)
            & (self.all_months["year_month"] == year_month)
        ]
        if rows.empty:
            print(f"warning: cannot locate the chip for {site_id} {year_month}; not saved")
            return None

        tile_path = rows.iloc[0]["tif_path"]
        reference = wh_tiles.read_tile(tile_path, self.cfg)
        path = wh_tiles.label_path_for(tile_path, self.cfg)
        wh_tiles.write_mask(path, mask, reference)
        self._write_sidecar(tile_path, path, site_id, year_month, mask)
        self.dirty.discard(key)
        return path

    # --- navigation -------------------------------------------------------

    def _leave_current(self) -> None:
        """Autosave the tile being navigated away from, if it has unsaved work."""
        if self.params.autosave_on_navigate and self.key in self.dirty:
            self.save()

    def _step_month(self, delta: int) -> None:
        """Browse the current site's months, keeping every label buffer.

        This moves the view only. The queue holds waterholes, not months, so
        browsing never advances your place in it.
        """
        self._leave_current()
        months = self._site_months().reset_index(drop=True)
        matches = months.index[months["year_month"] == self.current["year_month"]]
        if len(matches) == 0:
            self._set_status("current month is not in this site's month list")
            return

        location = int(matches[0]) + delta
        if not 0 <= location < len(months):
            self._set_status("no further months for this site")
            return

        target = months.iloc[location]
        self.current = {
            **self.current,
            "year_month": target["year_month"],
            "tif_path": target["tif_path"],
            "month_index": target["month_index"],
        }
        self._load_current()

    def _step_site(self, delta: int) -> None:
        """Move to the next or previous waterhole in the queue.

        The queue holds one entry per site, because that is the unit of work:
        you pick a waterhole, look through its history, label what is
        informative, and move on. Months are browsed within a site, not queued.
        """
        target = self.position + delta
        if not 0 <= target < len(self.queue):
            self._set_status("end of the queue — no further waterholes")
            return

        self._leave_current()
        self.position = target
        self._set_current_from_queue()
        self._load_current()

    def _return_to_start_month(self) -> None:
        """Snap back to the month this site opened on."""
        if self.current["year_month"] == str(self.queue_row["start_year_month"]):
            self._set_status("already on this site's starting month")
            return
        self._leave_current()
        self._set_current_from_queue()
        self._load_current()

    def _site_saved_months(self) -> int:
        """How many months of the current site already have a saved mask."""
        site_id = self.current["site_id"]
        on_disk = sum(
            1
            for path in self.cfg.paths["labels"].glob(f"*_S2_{site_id}_*_labels.tif")
        ) if self.cfg.paths["labels"].exists() else 0
        unsaved = sum(1 for site, _ in self.dirty if site == site_id)
        return on_disk + unsaved

    def _toggle_png(self) -> None:
        """Open the pre-rendered chip in its own window.

        Not an overlay: the PNG is a matplotlib figure with padding, titles and
        colourbars, so it has no pixel correspondence with the raster panels.
        """
        if self.png_window is not None:
            plt.close(self.png_window)
            self.png_window = None
            self._set_status("PNG closed")
            return

        png_path = wh_tiles.png_path_for(self.row["tif_path"], self.cfg)
        if not png_path.exists():
            self._set_status(f"no PNG rendered for {Path(self.row['tif_path']).stem}")
            return

        image = plt.imread(png_path)
        self.png_window = plt.figure(figsize=(11, 4.2))
        axis = self.png_window.add_subplot(111)
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(f"pre-rendered chip — {png_path.name}", fontsize=9)
        self.png_window.show()

    # --- events -----------------------------------------------------------

    def _connect(self) -> None:
        canvas = self.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("key_press_event", self._on_key)
        # Closing from the window's title bar must save too, not just via 'q'.
        canvas.mpl_connect("close_event", self._on_close)

    def _toolbar_active(self) -> bool:
        """True while a navigation tool (pan or zoom) is engaged.

        Without this, dragging to pan also paints a stroke along the drag path,
        which is both surprising and tedious to undo.
        """
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return True
        # Belt and braces: the zoom/pan widgets take the canvas widget lock, and
        # some backends report an empty toolbar mode regardless.
        return bool(self.figure.canvas.widgetlock.locked())

    def _event_pixel(self, event) -> tuple[int, int] | None:
        if event.inaxes not in list(self.axes) or event.xdata is None:
            return None
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        height, width = self.tile.shape
        if not (0 <= row < height and 0 <= col < width):
            return None
        return row, col

    def _on_press(self, event) -> None:
        if event.button != 1 or self.mode == "polygon" or self._toolbar_active():
            return
        pixel = self._event_pixel(event)
        if pixel is None:
            return
        self._painting = True
        self._paint(*pixel)

    def _on_release(self, event) -> None:
        self._painting = False

    def _on_motion(self, event) -> None:
        pixel = self._event_pixel(event)
        if pixel is None:
            return
        row, col = pixel
        # The readout still tracks the cursor while panning; only painting stops.
        if self._painting and not self._toolbar_active():
            self._paint(row, col)
        elif self._toolbar_active():
            self._painting = False
        self._update_readout(row, col)

    def _update_readout(self, row: int, col: int) -> None:
        """Live band and index values for the pixel under the cursor."""
        if not self.tile.valid[row, col]:
            self.readout.set_text(f"({row},{col}) NO OBSERVATION")
            self.figure.canvas.draw_idle()
            return

        bands = " ".join(
            f"{name}:{self.tile.bands[name][row, col]:.3f}"
            for name in ("B2", "B3", "B4", "B8", "B11", "B12")
        )
        indices = " ".join(
            f"{name}:{values[row, col]:+.3f}"
            for name, values in self.index_cache.items()
        )
        obs = self.tile.n_obs[row, col] if self.tile.n_obs is not None else -1
        current = self.mask[row, col]
        name = self.cfg.class_by_id(int(current)).name
        self.readout.set_text(
            f"({row},{col}) n_obs:{obs} label:{name}\n{bands}\n{indices}"
        )
        self.figure.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = _normalise_key(event.key)
        if key is None:
            return

        if key in self.key_to_class:
            self.active_class = self.key_to_class[key]
            self.mode = "brush"
            self._set_status()
        elif key == "[":
            self.brush_radius = max(self.params.min_brush_radius_px, self.brush_radius - 1)
            self._set_status()
        elif key == "]":
            self.brush_radius = min(self.params.max_brush_radius_px, self.brush_radius + 1)
            self._set_status()
        elif key == "e":
            self.mode = "erase"
            self._set_status()
        elif key == "b":
            self.mode = "brush"
            self._set_status()
        elif key == "g":
            self._start_polygon()
        elif key in ("ctrl+z", "z"):
            if self.undo.undo(self.mask):
                self.dirty.add(self.key)
                self._refresh_labels()
            else:
                self._set_status("nothing to undo")
        elif key in ("ctrl+y", "ctrl+Z", "Z"):
            if self.undo.redo(self.mask):
                self.dirty.add(self.key)
                self._refresh_labels()
            else:
                self._set_status("nothing to redo")
        elif key in ("ctrl+s", "w"):
            self.save()
        elif key in ("ctrl+shift+s", "W"):
            self.save_all()
        elif key == "right":
            self._step_month(1)
        elif key == "left":
            self._step_month(-1)
        elif key in ("n", "N"):
            self._step_site(1)
        elif key in ("p", "P"):
            self._step_site(-1)
        elif key == "s":
            self._step_site(1)
        elif key == "c":
            self._return_to_start_month()
        elif key == "h":
            self.show_labels = not self.show_labels
            self._refresh_labels()
        elif key == "f":
            self.show_footprint = not self.show_footprint
            # ContourSet is itself an Artist in matplotlib 3.8+.
            for artist in self.footprint_artists:
                artist.set_visible(self.show_footprint)
            self.figure.canvas.draw_idle()
        elif key == "v":
            self._toggle_png()
        elif key == "q":
            self.close()

    def _start_polygon(self) -> None:
        self.mode = "polygon"
        self._set_status("polygon: drag to enclose an area")

        def on_select(vertices):
            self._fill_polygon(vertices)
            self.selector.disconnect_events()
            self.mode = "brush"
            self._set_status()

        self.selector = LassoSelector(self.axes[0], onselect=on_select)

    # --- status -----------------------------------------------------------

    def _set_status(self, message: str = "") -> None:
        definition = self.cfg.class_by_id(self.active_class)
        labelled = int((self.mask > 0).sum()) if self.tile is not None else 0
        marker = "*" if self.key in self.dirty else " "
        unsaved = len(self.dirty)

        saved_here = self._site_saved_months() if self.tile is not None else 0
        autosave = "autosave on" if self.params.autosave_on_navigate else "MANUAL SAVE"

        self.status.set_text(
            f"[waterhole {self.position + 1}/{len(self.queue)}]{marker} site "
            f"{self.current['site_id']}  {self.current['year_month']}  "
            f"({saved_here} month(s) labelled here)\n"
            f"class [{definition.key}] {definition.name} | {self.mode} r={self.brush_radius} "
            f"| {labelled} px on this month | {unsaved} unsaved | {autosave}\n{message}"
        )
        if self.tile is not None:
            self.figure.canvas.draw_idle()


def build_queue(
    manifest: pd.DataFrame,
    sites: list[str] | None = None,
    max_gap_fraction: float = 0.2,
    min_mean_obs: float = 2.0,
    start_month: str = "first",
    start_month_preference: tuple[int, ...] = (9, 10, 8, 7),
) -> pd.DataFrame:
    """One row per waterhole: the unit of labelling work.

    The queue is sites, not site-months, because that is how the work actually
    goes — you pick a waterhole, look through its history, label the months that
    are informative, and move on. Months are browsed inside a site rather than
    queued, so `n` always means "next waterhole".

    It also matches how the results are validated: with grouped-by-site
    cross-validation the effective sample size is the number of labelled SITES,
    not pixels or months, so breadth across sites is what buys statistical power.

    `start_month` picks which month each site opens on:

      "first"          the earliest month of the record — start at the beginning
                       and work forward, which keeps a site's history in order.
      "best_observed"  the best-observed month among start_month_preference
                       (late dry season by default), where the basin floor and
                       pugged margin are most interpretable.

    Either way the choice is made among months passing the quality filters, so a
    site never opens on a chip that is mostly cloud. Every month remains
    browsable with the arrow keys regardless.
    """
    if start_month not in ("first", "best_observed"):
        raise ValueError(
            f"start_month must be 'first' or 'best_observed', got {start_month!r}"
        )

    candidates = manifest.copy()
    if sites:
        candidates = candidates[candidates["site_id"].isin(sites)]

    good = candidates[
        (candidates["gap_fraction"] <= max_gap_fraction)
        & (candidates["mean_n_obs"] >= min_mean_obs)
    ]

    rows = []
    for site_id, group in good.groupby("site_id", sort=True):
        start = _pick_start_month(group, start_month, start_month_preference)
        rows.append({
            "site_id": site_id,
            "start_year_month": start["year_month"],
            "start_tif_path": start["tif_path"],
            "start_month_index": int(start["month_index"]),
            "n_good_months": len(group),
            "mean_n_obs": float(group["mean_n_obs"].mean()),
        })

    if not rows:
        return pd.DataFrame(
            columns=["site_id", "start_year_month", "start_tif_path",
                     "start_month_index", "n_good_months", "mean_n_obs"]
        )

    return pd.DataFrame(rows).sort_values("site_id").reset_index(drop=True)


def _pick_start_month(
    group: pd.DataFrame, strategy: str, preference: tuple[int, ...]
) -> pd.Series:
    """The month a site opens on. See build_queue for the strategies."""
    if strategy == "first":
        return group.sort_values("month_index").iloc[0]

    # Best-observed month in the preferred season, falling back to any season if
    # a site has no qualifying month there — better to open somewhere useful than
    # to skip the site entirely.
    preferred = group[group["month"].isin(preference)]
    pool = preferred if not preferred.empty else group
    return pool.sort_values(
        ["mean_n_obs", "gap_fraction"], ascending=[False, True]
    ).iloc[0]


def launch(
    queue: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: Config,
    params: LabelParams | None = None,
    footprints: dict[str, np.ndarray] | None = None,
) -> Labeller:
    """Open the labelling window. Call from a notebook cell after %matplotlib qt.

    `manifest` is passed separately from `queue` so month stepping can reach every
    month of a site, not just the ones selected for labelling.
    """
    params = params or LabelParams()
    labeller = Labeller(queue, manifest, cfg, params, footprints=footprints)
    plt.show()
    return labeller


KEY_HELP = """
  PAINTING
    0-6              select class (0 = unlabelled)
    b / e            brush / eraser
    g                polygon fill (drag to enclose)
    [ / ]            brush smaller / larger
    z    or cmd+z    undo
    Z    or cmd+y    redo

  NAVIGATION
    left / right     previous / next MONTH of this waterhole
    c                return to this waterhole's starting month
    p / n            previous / next WATERHOLE
    s                skip to the next waterhole

  ACTIONS
    w    or cmd+s    save this month
    W                save EVERY month with unsaved labels
    h                hide/show labels
    f                hide/show basin footprint
    v                open/close the pre-rendered PNG (separate window)
    q                close

  On macOS cmd and ctrl are interchangeable for every binding above.
  Matplotlib's own shortcuts are disabled inside this window (they collide
  with p, s, f, h, g, v and the arrow keys) and restored when it closes.

  SAVING IS AUTOMATIC: a month with unsaved labels is written whenever you
  navigate away from it, and everything outstanding is written when the window
  closes. An empty mask is never written. Set autosave_on_navigate=False or
  save_on_quit=False in LabelParams if you would rather save by hand.
"""
