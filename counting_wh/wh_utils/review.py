"""
Interactive review of labelled vs detected waterholes.

A contact sheet: a 4 x 4 grid of 16 waterholes per page, each cell zoomed in on
ONE waterhole, showing your label and the model detection as coloured boxes.
Page through it to assess accuracy by eye in a couple of minutes.

    solid box   = YOUR LABEL,     coloured by the class you assigned
    dashed box  = MODEL DETECTION, coloured by the class it predicted

Two different colours in a cell means the classes disagree - visible instantly,
without reading anything. A dashed box alone is a false positive; a solid box
alone is a waterhole the model missed.

Usage from a notebook:

    %matplotlib qt
    from counting_boats.wh_utils import review
    reviewer = review.launch(RUN_FOLDER, CONFIG)

Reads only the "*.details.csv" files written by
compare_detections_to_ground_truth() plus the padded scene PNGs, so it changes
nothing and is safe to re-run.

Style follows Image_boxes_checks.ipynb (class-coloured rectangles, corner name
tags, ticks stripped, dpi=300 saves); mechanics follow wh_label.py (a params
dataclass, launch(), keyboard navigation, matplotlib keymaps disabled while the
window is open).
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
from PIL import Image

from .evaluation import load_details
from .waterhole_classes import load_class_registry

# The padded scene PNGs are ~180 megapixels, well past Pillow's decompression
# bomb guard. image_cutting_support.py already lifts it for the same reason.
PIL.Image.MAX_IMAGE_PIXELS = None

FILTERS = ("all", "errors", "mismatch", "false_positive", "false_negative")
SORTS = ("image", "worst_overlap", "smallest")

# Title colour by outcome, so a page reads at a glance
OUTCOME_COLOURS = {
    "agree": "#2e7d32",      # green  - matched, classes agree
    "mismatch": "#ef6c00",   # orange - matched, classes differ
    "false_positive": "#c62828",  # red    - detected, nothing labelled
    "false_negative": "#1565c0",  # blue   - labelled, nothing detected
}


@dataclass
class ReviewParams:
    """Everything tunable about a review session, in one visible block."""

    # grid -- 4 x 4 = 16 waterholes per page
    panel_rows: int = 4
    panel_cols: int = 4

    # how far to zoom out around each waterhole
    crop_mode: str = "adaptive"   # "adaptive" (window = box * margin) | "fixed"
    crop_margin: float = 3.0
    min_crop_px: int = 64
    fixed_crop_px: int = 256

    # what goes in the queue
    filter: str = "all"
    sort: str = "image"
    only_classes: tuple = ()      # restrict by LABEL class name, empty = all

    # drawing
    label_linewidth: float = 2.0
    detection_linewidth: float = 1.6
    detection_linestyle: str = "--"
    show_titles: bool = True
    show_tags: bool = True
    tag_fontsize: int = 8
    tag_alpha: float = 0.7
    title_fontsize: int = 8
    class_colours: dict = field(default_factory=dict)

    # output
    # Sized so a 4 x 4 grid gives roughly square cells once the legend column
    # is accounted for; a taller figure letterboxes square crops.
    figure_size: tuple = (16.0, 13.0)
    save_dpi: int = 300


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------
def class_colours(registry, params=None):
    """
    A distinct colour per class, generated rather than hardcoded.

    Classes are config-driven and arbitrary, so colours come from a qualitative
    colormap indexed by class id (tab10, or tab20 past ten classes). Override
    any of them by name via ReviewParams.class_colours.
    """
    overrides = dict(getattr(params, "class_colours", {}) or {})
    n = len(registry.ids)
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    colours = {}
    for cid in registry.ids:
        name = registry.id_to_name[cid]
        colours[cid] = overrides.get(name, cmap(cid % cmap.N))
    return colours


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------
def _resolve_png(config, image):
    return os.path.normpath(
        os.path.join(config["path"], config["pngs"], f"{image}.png")
    )


def _outcome(row):
    if row["match_type"] != "matched":
        return row["match_type"]
    return "agree" if row["ml_class"] == row["manual_class"] else "mismatch"


def build_queue(run_folder, config, params=None):
    """
    Every waterhole comparison in the run, filtered and sorted, with its scene
    PNG resolved. Touches no imagery.

    Args:
        run_folder: folder compare_detections_to_ground_truth() wrote into
        config: parsed config dict (or path to one)
        params: ReviewParams; its filter/sort/only_classes are applied here

    Returns:
        DataFrame with the details columns plus "png_path" and "outcome".
    """
    from .testing import parse_config

    if isinstance(config, str):
        config = parse_config(config)
    params = params or ReviewParams()
    registry = load_class_registry(config)

    df = load_details(run_folder).copy()
    df["png_path"] = df["image"].map(lambda im: _resolve_png(config, im))
    df["outcome"] = df.apply(_outcome, axis=1)

    if params.filter not in FILTERS:
        raise ValueError(f"filter must be one of {list(FILTERS)}, got {params.filter!r}")
    if params.sort not in SORTS:
        raise ValueError(f"sort must be one of {list(SORTS)}, got {params.sort!r}")

    if params.filter == "errors":
        df = df[df["outcome"] != "agree"]
    elif params.filter == "mismatch":
        df = df[df["outcome"] == "mismatch"]
    elif params.filter in ("false_positive", "false_negative"):
        df = df[df["match_type"] == params.filter]

    if params.only_classes:
        wanted = {registry.name_to_id[n] for n in params.only_classes}
        df = df[df["manual_class"].isin(wanted) | df["ml_class"].isin(wanted)]

    if params.sort == "image":
        df = df.sort_values(["image", "y", "x"])
    elif params.sort == "worst_overlap":
        # unmatched has no overlap at all, so it sorts first as the worst case
        df = df.sort_values(
            "overlap", ascending=True, na_position="first", kind="stable"
        )
    else:  # smallest
        size = df["manual_area"].fillna(df["ml_area"])
        df = df.assign(_size=size).sort_values("_size", kind="stable").drop(columns="_size")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------
def _boxes(row):
    """(label_box, detection_box) as (x1, y1, x2, y2), either may be None."""
    out = []
    for px, py, pw, ph in (
        ("x", "y", "manual_w", "manual_h"),
        ("x", "y", "ml_w", "ml_h"),
    ):
        w, h = row[pw], row[ph]
        if pd.isna(w) or pd.isna(h):
            out.append(None)
        else:
            cx, cy = float(row[px]), float(row[py])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out[0], out[1]


def _crop_window(row, params):
    """Pixel window to cut around this waterhole, as (x0, y0, x1, y1)."""
    label, det = _boxes(row)
    present = [b for b in (label, det) if b is not None]
    if not present:
        side = params.fixed_crop_px
    elif params.crop_mode == "fixed":
        side = params.fixed_crop_px
    else:
        biggest = max(max(b[2] - b[0], b[3] - b[1]) for b in present)
        side = max(biggest * params.crop_margin, params.min_crop_px)
    side = int(round(side))
    cx, cy = float(row["x"]), float(row["y"])
    half = side // 2
    return int(round(cx)) - half, int(round(cy)) - half, int(round(cx)) + half, int(round(cy)) + half


def extract_crops(queue, params=None, progress=True):
    """
    Cut a small image around every waterhole, decoding each scene only once.

    The scenes are ~180 MP (roughly 540 MB decoded), so caching them and cutting
    on demand while paging would exhaust memory within a few pages. Instead each
    scene is opened once, every crop it owes is taken, and it is released before
    the next one. A few hundred crops come to tens of megabytes in total, so
    paging afterwards is instant.

    Args:
        queue: output of build_queue()
        params: ReviewParams (controls the window size)
        progress: show a tqdm bar, since a 180 MP decode takes seconds

    Returns:
        dict mapping queue index -> (array, x0, y0), where x0/y0 are the crop
        origin in scene pixels so boxes can be shifted into crop space. A scene
        that cannot be opened yields None for its rows, which draw as a clearly
        marked placeholder rather than aborting the session.
    """
    params = params or ReviewParams()
    crops = {}
    groups = list(queue.groupby("image", sort=False))
    iterator = groups
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(groups, desc="extracting crops", unit="scene")
        except ImportError:
            pass

    for image, group in iterator:
        png = group["png_path"].iloc[0]
        if not os.path.exists(png):
            print(f"  scene PNG not found, skipping: {png}")
            for idx in group.index:
                crops[idx] = None
            continue
        try:
            with Image.open(png) as scene:
                width, height = scene.size
                for idx, row in group.iterrows():
                    x0, y0, x1, y1 = _crop_window(row, params)
                    # clip to the scene; PIL pads outside the bounds with black
                    cx0, cy0 = max(x0, 0), max(y0, 0)
                    cx1, cy1 = min(x1, width), min(y1, height)
                    if cx1 <= cx0 or cy1 <= cy0:
                        crops[idx] = None
                        continue
                    patch = scene.crop((cx0, cy0, cx1, cy1)).convert("RGB")
                    crops[idx] = (np.asarray(patch), cx0, cy0)
        except Exception as exc:  # a corrupt or unreadable scene must not kill the run
            print(f"  could not read {png}: {exc}")
            for idx in group.index:
                crops[idx] = None
    return crops


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _draw_cell(ax, row, crop, colours, registry, params):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])

    if crop is None:
        ax.text(0.5, 0.5, "image\nunavailable", ha="center", va="center",
                fontsize=8, color="#888888", transform=ax.transAxes)
        ax.set_facecolor("#f0f0f0")
        if params.show_titles:
            ax.set_title(str(row["image"])[:24], fontsize=params.title_fontsize,
                         color="#888888")
        return

    array, x0, y0 = crop
    ax.imshow(array)

    if getattr(params, "_boxes_on", True):
        label_box, det_box = _boxes(row)
        specs = [
            (label_box, row["manual_class"], "-", params.label_linewidth,
             "above_left"),
            (det_box, row["ml_class"], params.detection_linestyle,
             params.detection_linewidth, "below_right"),
        ]
        for box, cls, style, lw, corner in specs:
            if box is None or pd.isna(cls) or int(cls) < 0:
                continue
            colour = colours.get(int(cls), "#999999")
            bx0, by0, bx1, by1 = box
            rect = patches.Rectangle(
                (bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0,
                linewidth=lw, edgecolor=colour, facecolor="none", linestyle=style,
            )
            ax.add_patch(rect)
            if params.show_tags:
                # The label tag sits above the top-left corner and the detection
                # tag below the bottom-right, so the two can never overlap even
                # when the boxes are nested - which they usually are.
                if corner == "above_left":
                    tx, ty, ha, va = bx0 - x0, by0 - y0, "left", "bottom"
                else:
                    tx, ty, ha, va = bx1 - x0, by1 - y0, "right", "top"
                ax.text(
                    tx, ty, registry.id_to_name.get(int(cls), str(cls)),
                    color="white", fontsize=params.tag_fontsize,
                    bbox=dict(facecolor=colour, alpha=params.tag_alpha, pad=1),
                    ha=ha, va=va,
                )

    if params.show_titles:
        ax.set_title(_cell_title(row, registry), fontsize=params.title_fontsize,
                     color=OUTCOME_COLOURS.get(row["outcome"], "black"))


def _cell_title(row, registry):
    name = registry.id_to_name.get
    outcome = row["outcome"]
    if outcome == "agree":
        return f"{name(int(row['manual_class']), '?')}  IoU {row['overlap']:.2f}"
    if outcome == "mismatch":
        return (f"label {name(int(row['manual_class']), '?')} -> "
                f"det {name(int(row['ml_class']), '?')}  IoU {row['overlap']:.2f}")
    if outcome == "false_positive":
        conf = row.get("ml_confidence", float("nan"))
        conf_txt = "" if pd.isna(conf) else f"  conf {conf:.2f}"
        return f"FALSE POSITIVE  {name(int(row['ml_class']), '?')}{conf_txt}"
    return f"MISSED  {name(int(row['manual_class']), '?')}"


def _draw_legend(ax, registry, colours, params, status_lines):
    """
    Class colours, the box-style key and live session state, stacked as one
    column on the right.

    Block positions are MEASURED from the rendered legends rather than
    estimated: the legend axes spans the full figure height, so a guessed
    fractional offset leaves the blocks drifting apart, and the class list
    changes height with the number of classes in the config.
    """
    ax.clear()
    ax.axis("off")
    fig = ax.figure

    handles = [
        patches.Patch(facecolor=colours[cid], edgecolor="#333333",
                      label=registry.id_to_name[cid])
        for cid in registry.ids
    ]
    first = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.0),
                      fontsize=9, frameon=True, title="Classes")
    first.get_title().set_fontweight("bold")
    ax.add_artist(first)

    style_handles = [
        plt.Line2D([], [], color="#333333", linestyle="-",
                   linewidth=params.label_linewidth, label="your label"),
        plt.Line2D([], [], color="#333333", linestyle=params.detection_linestyle,
                   linewidth=params.detection_linewidth, label="model detection"),
    ]

    def _bottom_of(artist, fallback):
        """Bottom edge of a drawn artist in axes fraction, or a fallback."""
        try:
            fig.canvas.draw()
            bb = artist.get_window_extent()
            return bb.transformed(ax.transAxes.inverted()).y0
        except Exception:
            return fallback

    gap = 0.03
    y = _bottom_of(first, 1.0 - 0.02 * (len(registry.ids) + 2)) - gap
    second = ax.legend(handles=style_handles, loc="upper left",
                       bbox_to_anchor=(0, y), fontsize=9, frameon=True,
                       title="Box style")
    second.get_title().set_fontweight("bold")
    ax.add_artist(second)

    y = _bottom_of(second, y - 0.08) - gap
    ax.text(0, y, "\n".join(status_lines), transform=ax.transAxes,
            fontsize=9, va="top", family="monospace")
# ---------------------------------------------------------------------------
# the interactive window
# ---------------------------------------------------------------------------
class Reviewer:
    """Paged contact sheet of label-vs-detection comparisons."""

    def __init__(self, run_folder, config, params=None):
        from .testing import parse_config

        if isinstance(config, str):
            config = parse_config(config)
        self.run_folder = run_folder
        self.config = config
        self.params = params or ReviewParams()
        self.params._boxes_on = True
        self.registry = load_class_registry(config)
        self.colours = class_colours(self.registry, self.params)

        self.full = build_queue(run_folder, config, self.params)
        if self.full.empty:
            raise ValueError(
                f"No waterholes to review with filter={self.params.filter!r}. "
                "Try filter='all'."
            )
        self.crops = extract_crops(self.full, self.params)
        self.queue = self.full
        self.page_index = 0
        self._saved_keymaps = None
        self.fig = None
        self._build_figure()
        self.draw()

    # --- geometry ---
    @property
    def per_page(self):
        return self.params.panel_rows * self.params.panel_cols

    @property
    def n_pages(self):
        return max(1, int(np.ceil(len(self.queue) / self.per_page)))

    def _build_figure(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = plt.figure(figsize=self.params.figure_size)
        outer = self.fig.add_gridspec(
            1, 2, width_ratios=[self.params.panel_cols, 1.15], wspace=0.02
        )
        grid = outer[0].subgridspec(
            self.params.panel_rows, self.params.panel_cols, hspace=0.22, wspace=0.06
        )
        self.axes = [
            self.fig.add_subplot(grid[r, c])
            for r in range(self.params.panel_rows)
            for c in range(self.params.panel_cols)
        ]
        self.legend_ax = self.fig.add_subplot(outer[1])
        self.legend_ax.axis("off")
        self._disable_keymaps()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._restore_keymaps)

    # --- matplotlib keymaps collide with our single-key bindings ---
    def _disable_keymaps(self):
        if self._saved_keymaps is None:
            self._saved_keymaps = {
                k: list(v) for k, v in plt.rcParams.items() if k.startswith("keymap.")
            }
        for k in self._saved_keymaps:
            plt.rcParams[k] = []

    def _restore_keymaps(self, event=None):
        if self._saved_keymaps:
            for k, v in self._saved_keymaps.items():
                plt.rcParams[k] = v
            self._saved_keymaps = None

    # --- drawing ---
    def _status(self):
        q = self.queue
        counts = q["outcome"].value_counts()
        start = self.page_index * self.per_page
        return [
            f"FILTER  {self.params.filter}",
            f"SORT    {self.params.sort}",
            "",
            f"page    {self.page_index + 1}/{self.n_pages}",
            f"showing {start + 1}-{min(start + self.per_page, len(q))} of {len(q)}",
            "",
            f"agree     {int(counts.get('agree', 0))}",
            f"mismatch  {int(counts.get('mismatch', 0))}",
            f"false pos {int(counts.get('false_positive', 0))}",
            f"missed    {int(counts.get('false_negative', 0))}",
            "",
            "arrows/n/p  page",
            "a e m f x   filter",
            "o           sort",
            "+/- b t s q",
        ]

    def draw(self):
        start = self.page_index * self.per_page
        rows = self.queue.iloc[start:start + self.per_page]
        for i, ax in enumerate(self.axes):
            if i < len(rows):
                row = rows.iloc[i]
                _draw_cell(ax, row, self.crops.get(row.name), self.colours,
                           self.registry, self.params)
                ax.set_visible(True)
            else:
                ax.clear()
                ax.set_visible(False)
        _draw_legend(self.legend_ax, self.registry, self.colours, self.params,
                     self._status())
        self.fig.canvas.draw_idle()

    # --- navigation ---
    def page(self, n):
        self.page_index = int(np.clip(n, 0, self.n_pages - 1))
        self.draw()

    def next_page(self):
        self.page(self.page_index + 1)

    def prev_page(self):
        self.page(self.page_index - 1)

    def set_filter(self, name):
        """Re-queue without re-extracting crops (they are keyed by row id)."""
        if name not in FILTERS:
            raise ValueError(f"filter must be one of {list(FILTERS)}")
        self.params.filter = name
        self.queue = self._requeue()
        self.page_index = 0
        self.draw()

    def set_sort(self, name):
        if name not in SORTS:
            raise ValueError(f"sort must be one of {list(SORTS)}")
        self.params.sort = name
        self.queue = self._requeue()
        self.page_index = 0
        self.draw()

    def _requeue(self):
        p = self.params
        df = self.full
        if p.filter == "errors":
            df = df[df["outcome"] != "agree"]
        elif p.filter == "mismatch":
            df = df[df["outcome"] == "mismatch"]
        elif p.filter in ("false_positive", "false_negative"):
            df = df[df["match_type"] == p.filter]
        if p.sort == "image":
            df = df.sort_values(["image", "y", "x"], kind="stable")
        elif p.sort == "worst_overlap":
            df = df.sort_values("overlap", ascending=True, na_position="first",
                                kind="stable")
        else:
            size = df["manual_area"].fillna(df["ml_area"])
            df = df.assign(_s=size).sort_values("_s", kind="stable").drop(columns="_s")
        if df.empty:
            print(f"  no waterholes match filter={p.filter!r}; keeping the previous page")
            return self.queue
        return df

    def set_grid(self, rows, cols):
        self.params.panel_rows, self.params.panel_cols = rows, cols
        self._build_figure()
        self.page_index = 0
        self.draw()

    def save_page(self, path=None):
        outdir = os.path.join(self.run_folder, "plots", "review")
        os.makedirs(outdir, exist_ok=True)
        path = path or os.path.join(
            outdir, f"review_{self.params.filter}_page{self.page_index + 1:03d}.png"
        )
        self.fig.savefig(path, dpi=self.params.save_dpi, bbox_inches="tight")
        print(f"saved {path}")
        return path

    def to_dataframe(self):
        """What is currently in the queue, in display order."""
        return self.queue.copy()

    def close(self):
        self._restore_keymaps()
        if self.fig is not None:
            plt.close(self.fig)

    # --- keys ---
    def _on_key(self, event):
        k = event.key
        if k in ("right", "n"):
            self.next_page()
        elif k in ("left", "p"):
            self.prev_page()
        elif k == "home":
            self.page(0)
        elif k == "end":
            self.page(self.n_pages - 1)
        elif k == "a":
            self.set_filter("all")
        elif k == "e":
            self.set_filter("errors")
        elif k == "m":
            self.set_filter("mismatch")
        elif k == "f":
            self.set_filter("false_positive")
        elif k == "x":
            self.set_filter("false_negative")
        elif k == "o":
            self.set_sort(SORTS[(SORTS.index(self.params.sort) + 1) % len(SORTS)])
        elif k in ("+", "="):
            side = min(self.params.panel_rows + 1, 6)
            self.set_grid(side, side)
        elif k == "-":
            side = max(self.params.panel_rows - 1, 1)
            self.set_grid(side, side)
        elif k == "b":
            self.params._boxes_on = not self.params._boxes_on
            self.draw()
        elif k == "t":
            self.params.show_titles = not self.params.show_titles
            self.draw()
        elif k == "s":
            self.save_page()
        elif k == "q":
            self.close()


KEY_HELP = """
  PAGING
    right / n        next page
    left  / p        previous page
    home / end       first / last page

  FILTER
    a                all
    e                errors only (false pos + missed + class disagreement)
    m                class disagreements only
    f                false positives only
    x                missed waterholes only

  SORT
    o                cycle: image order -> worst overlap first -> smallest first

  VIEW
    + / -            more / fewer panels per page
    b                toggle boxes (see the raw imagery underneath)
    t                toggle titles
    s                save this page as PNG into <run>/plots/review/
    q                close

  Matplotlib's own shortcuts are disabled inside this window (they collide with
  p, s, f, o, b and the arrows) and restored when it closes.
"""




def launch(run_folder, config, params=None, show_help=True):
    """
    Open the review window. Call from a notebook cell after %matplotlib qt.

    Args:
        run_folder: folder compare_detections_to_ground_truth() wrote into
        config: parsed config dict, or a path to the config file
        params: ReviewParams; defaults to a 4 x 4 grid of everything
        show_help: print the key bindings

    Returns:
        Reviewer - keep the reference alive or the window may be garbage
        collected. Use .page(), .set_filter(), .to_dataframe() from later cells.
    """
    reviewer = Reviewer(run_folder, config, params)
    if show_help:
        print(KEY_HELP)
    plt.show()
    return reviewer


def save_contact_sheet(run_folder, config, params=None, page=0, out_path=None):
    """
    Render one page to a PNG without opening a window.

    Works headlessly (Agg backend), so it is the way to put a review page into a
    report or run this on a machine with no display.

    Args:
        run_folder: folder compare_detections_to_ground_truth() wrote into
        config: parsed config dict, or a path to the config file
        params: ReviewParams
        page: zero-based page number
        out_path: destination PNG; defaults into <run>/plots/review/

    Returns:
        Path to the PNG written.
    """
    from .testing import parse_config

    if isinstance(config, str):
        config = parse_config(config)
    params = params or ReviewParams()
    params._boxes_on = True
    registry = load_class_registry(config)
    colours = class_colours(registry, params)

    queue = build_queue(run_folder, config, params)
    crops = extract_crops(queue, params)

    per_page = params.panel_rows * params.panel_cols
    rows = queue.iloc[page * per_page:(page + 1) * per_page]

    fig = plt.figure(figsize=params.figure_size)
    outer = fig.add_gridspec(1, 2, width_ratios=[params.panel_cols, 1.15], wspace=0.02)
    grid = outer[0].subgridspec(params.panel_rows, params.panel_cols,
                                hspace=0.22, wspace=0.06)
    axes = [fig.add_subplot(grid[r, c])
            for r in range(params.panel_rows) for c in range(params.panel_cols)]
    for i, ax in enumerate(axes):
        if i < len(rows):
            row = rows.iloc[i]
            _draw_cell(ax, row, crops.get(row.name), colours, registry, params)
        else:
            ax.set_visible(False)

    counts = queue["outcome"].value_counts()
    status = [
        f"FILTER  {params.filter}",
        f"SORT    {params.sort}",
        "",
        f"page    {page + 1}/{max(1, int(np.ceil(len(queue) / per_page)))}",
        f"total   {len(queue)}",
        "",
        f"agree     {int(counts.get('agree', 0))}",
        f"mismatch  {int(counts.get('mismatch', 0))}",
        f"false pos {int(counts.get('false_positive', 0))}",
        f"missed    {int(counts.get('false_negative', 0))}",
    ]
    _draw_legend(fig.add_subplot(outer[1]), registry, colours, params, status)

    outdir = os.path.join(run_folder, "plots", "review")
    os.makedirs(outdir, exist_ok=True)
    out_path = out_path or os.path.join(
        outdir, f"contact_sheet_{params.filter}_page{page + 1:03d}.png"
    )
    fig.savefig(out_path, dpi=params.save_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path
