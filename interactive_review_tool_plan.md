# Plan — Interactive label-vs-detection review tool

A contact-sheet browser for the testing pipeline: a **4 x 4 grid of 16
waterholes per page**, each cell zoomed in on one waterhole showing **your
label** and **the model's detection** as coloured boxes, paged through for fast
visual assessment of accuracy.

Two sources of inspiration, deliberately:

- **`Image_boxes_checks.ipynb`** for the *look*: a square mosaic of 16 tiles,
  class-coloured `patches.Rectangle` boxes with a small coloured name tag at the
  corner, ticks stripped, saved at `dpi=300`. That notebook is the closest thing
  to what this tool should feel like — this version swaps its random training
  tiles for zoomed waterhole comparisons, and its bottom legend for a side one.
- **`cookie-cutting/labelling.ipynb` + `wh_label.py`** for the *mechanics*: a
  visible `@dataclass` parameter block, a `launch()` call, a Qt matplotlib
  window, keyboard navigation, and a legend down the side.

---

## 1. Why

`confusion_matrix_AF` and the `evaluation` functions say *how much* is wrong.
They cannot say *what* is wrong — whether a false positive is a genuine miss or
a labelling slip, whether a class disagreement is the model's fault or an
ambiguous waterhole, whether boxes are loose or the waterhole is just hard.

That judgement needs eyes on the imagery. `plot_waterholes` already draws boxes,
but it renders whole scenes to PNG files that you then open one by one — no
paging, no filtering, no way to sweep 300 waterholes quickly.

**Goal:** look at every waterhole in a run in a few minutes, and land on the
handful that need a decision.

---

## 2. Where it fits

```
prepare -> segment -> run_detection -> backwards_annotation_AF
        -> compare_detections_to_ground_truth      <-- writes *.details.csv
        -> confusion_matrix_AF
        -> evaluation.{classification_report, localisation_quality, recall_by_size}
        -> review.launch()                          <-- THIS TOOL
```

Read-only over existing outputs. Runs after
`compare_detections_to_ground_truth`, changes nothing, safe to re-run.

---

## 3. Data — all verified present

| Need | Source | Verified |
|---|---|---|
| One row per waterhole comparison | `<run>/<image>.details.csv` | 15 columns incl. `match_type`, `ml_class`, `manual_class`, `overlap`, box `w`/`h`/`area`, `ml_confidence` |
| Which image a row came from | `evaluation.load_details()` adds an `image` column from the filename | yes |
| The imagery | `<path>/<pngs>/<image>.png` | `testing/pngs/` holds `20240101_mimal_test.png` etc. |
| Box coordinates | `x`, `y`, `*_w`, `*_h` in **full padded-image pixels** | `parse_classifications_AF` does `x * TILE_SIZE + across`, so they are already in padded-PNG space — no transform needed |
| Class names and ids | `load_class_registry(config)` | config-driven, any number/names |

The image-name to PNG mapping is the same join `backwards_annotation_AF` already
does (`config["path"] / config["pngs"] / f"{image}.png"`), so it is proven, not
assumed.

### The one hard constraint

**The padded PNGs are ~180 megapixels** (`20240101_mimal_test.png` is
180,670,464 px). Two consequences:

1. PIL refuses to open them until `Image.MAX_IMAGE_PIXELS` is raised —
   `image_cutting_support.py` already sets it to `None`, so the tool must do the
   same before opening.
2. A decoded RGB array is roughly **540 MB per image**. Caching several decoded
   scenes would exhaust memory.

**Therefore the tool must not page against live full images.** See §6.

---

## 4. What one cell shows

**One waterhole per cell**, zoomed in — a crop of the scene centred on that
waterhole, not a whole tile. Sixteen of them fill a page.

```
 +--------------------------------+
 |  matched   IoU 0.81            |   <- title, coloured by outcome
 |                                |
 |     +---------------+[WH_wet]  |   solid  = YOUR LABEL, coloured by
 |     |  ...........  |          |           the class you assigned
 |     |  .  ~~~~~  .  |          |
 |     |  . ~~~~~~~ .  |          |   dashed = MODEL DETECTION, coloured by
 |     |  .  ~~~~~  .  |          |           the class it predicted
 |     +---------------+          |
 |    [WH_wet]                    |   <- corner name tags, as in
 +--------------------------------+      Image_boxes_checks
```

- **Solid box** = your label, coloured by `manual_class`
- **Dashed box** = model detection, coloured by `ml_class`
- Same colour, boxes nested → agreement
- **Two different colours → class disagreement, visible instantly**
- Dashed only → false positive
- Solid only → false negative (missed)

Panel title carries `match_type`, both class names when they differ, IoU, and
detection confidence. Title text is colour-coded: agreement / class mismatch /
missed / spurious.

### Visual conventions carried over from `Image_boxes_checks.ipynb`

| Element | Setting |
|---|---|
| Box | `patches.Rectangle(..., linewidth=2, edgecolor=colour, facecolor="none")` |
| Name tag | class name at the box corner, white text, `fontsize=8`, `bbox=dict(facecolor=colour, alpha=0.7)` |
| Ticks | `set_xticks([])`, `set_yticks([])` on every cell |
| Title | `fontsize=8` |
| Layout | `tight_layout()`, square `figsize` (default `(15, 16)` — a little taller than wide to leave room for the side legend) |
| Save | `dpi=300`, `bbox_inches="tight"` |

Two deliberate departures from that notebook:

1. **Tags show the class *name*, not `Class 3`.** Classes are config-driven now,
   so `registry.id_to_name` is available and far more readable at a glance.
   Tags can be turned off (`show_tags=False`) when boxes are tightly packed.
2. **The legend goes on the side, not `loc="lower center"`.** A side legend has
   room for the class list *and* the line-style key *and* live filter/sort/page
   state, which a one-row bottom legend does not.

---

## 5. Layout and interaction

```
+---------------------------------------------------+---------------------+
|  [wh 1]   [wh 2]   [wh 3]   [wh 4]                |  CLASSES            |
|                                                   |   # Dry_WH          |
|  [wh 5]   [wh 6]   [wh 7]   [wh 8]                |   # WH_swamp        |
|                                                   |   # WH_wet          |
|  [wh 9]   [wh10]   [wh11]   [wh12]                |   # WH_sink         |
|                                                   |   # U               |
|  [wh13]   [wh14]   [wh15]   [wh16]                |                     |
|                                                   |  ---- your label    |
|                                                   |  - -  detection     |
|                                                   |                     |
|                                                   |  FILTER: errors     |
|                                                   |  SORT:   worst      |
|                                                   |  page 3/11  (163)   |
|                                                   |  TP 84 FP 23 FN 36  |
+---------------------------------------------------+---------------------+
```

**4 x 4 = 16 waterholes per page** by default. At 163 waterholes that is 11
pages — a full run sweeps in a couple of minutes. `+`/`-` steps the grid between
3x3, 4x4 and 5x5 for a closer look or a faster sweep.

Legend is a dedicated axes on the right (not `ax.legend`), so it can hold class
patches, the line-style key, live filter/sort state, page position and running
counts together.

### Keys

```
  PAGING
    right / n        next page
    left  / p        previous page
    home / end       first / last page
    g                jump to page (prompt in the status line)

  FILTER (what goes in the queue)
    a                all
    e                errors only  (FP + FN + class disagreement)
    m                class disagreements only
    f                false positives only
    x                false negatives only
    1-9              only this class (by label class)

  SORT
    o                cycle: image order -> worst overlap first -> smallest first

  VIEW
    +/-              more / fewer panels per page
    b                toggle boxes on/off (see the raw imagery underneath)
    t                toggle titles
    s                save the current page as PNG into <run>/plots/review/

    q                close
```

Matplotlib's own single-key shortcuts collide with `p`, `s`, `f`, `g`, `o`, `b`
and the arrows, so the tool disables its keymaps on open and restores them on
close — exactly as `wh_label.Labeller._restore_keymaps` does.

---

## 6. Architecture — crop extraction up front

The 540 MB-per-scene constraint drives this. **Do not** cache decoded scenes and
crop on demand while paging.

Instead, a **prepare pass** before the window opens:

```
for each image in the queue (typically 3-10 scenes):
    open the padded PNG once  (raise MAX_IMAGE_PIXELS first)
    for every waterhole belonging to that image:
        crop a small window around it, convert to a small array
    release the scene            <- big array freed before the next image
```

Crops are ~200x200 px, so a full run of 300 waterholes is roughly **36 MB**
total. Paging then reads pre-extracted arrays and is instant.

- Progress is reported with `tqdm`, since decoding 180 MP PNGs takes seconds each.
- Optional on-disk cache (`<run>/plots/review/crops.npz`) so reopening the tool
  is immediate; invalidated by details-file mtime.
- If a scene PNG is missing, its waterholes get a clearly marked placeholder
  panel rather than aborting the whole session.

### Crop sizing

`crop_mode`:
- `"adaptive"` (default) — window = largest of the two boxes × `crop_margin`
  (default 3.0), floored at `min_crop_px`. Every waterhole fills its panel
  regardless of size, which matters given the size range here.
- `"fixed"` — constant `fixed_crop_px` window, so apparent sizes stay comparable
  across panels.

Boxes are clipped to the scene edge; the crop is padded where it runs off.

---

## 7. Colours

Classes are config-driven and arbitrary, so colours must be **generated**, not
hardcoded — consistent with the class-flexibility work.

- Default: `matplotlib` `tab10` indexed by class id, giving a stable distinct
  colour per class for up to 10 classes; `tab20` beyond that.
- Override per class by name in the params block:
  `class_colours={"WH_wet": "#1f77b4", ...}` — partial overrides fall back to the
  generated colour.
- `-1` (absent) is never drawn: a missing side simply has no box.

---

## 8. Public API

New module `counting_boats/boat_utils/review.py`, mirrored to
`counting_wh/wh_utils/review.py` (the two packages are kept in sync).

```python
@dataclass
class ReviewParams:
    # grid -- 4 x 4 = 16 waterholes per page, as in Image_boxes_checks
    panel_rows: int = 4
    panel_cols: int = 4

    # crops: how far to zoom out around each waterhole
    crop_mode: str = "adaptive"        # "adaptive" | "fixed"
    crop_margin: float = 3.0           # adaptive: window = largest box * this
    min_crop_px: int = 64
    fixed_crop_px: int = 256

    # what to show
    filter: str = "all"                # all|errors|mismatch|false_positive|false_negative
    sort: str = "image"                # image|worst_overlap|smallest
    only_classes: tuple[str, ...] = () # restrict by LABEL class, empty = all

    # drawing -- matches Image_boxes_checks conventions
    label_linewidth: float = 2.0
    detection_linewidth: float = 1.6
    detection_linestyle: str = "--"
    show_titles: bool = True
    show_tags: bool = True             # class name tag at each box corner
    tag_fontsize: int = 8
    tag_alpha: float = 0.7
    title_fontsize: int = 8
    class_colours: dict = field(default_factory=dict)

    # output
    figure_size: tuple[float, float] = (15.0, 16.0)
    save_dpi: int = 300

    # caching
    cache_crops: bool = True


def build_queue(run_folder, config, params) -> pd.DataFrame:
    """Details rows + resolved PNG path, filtered and sorted. No I/O on imagery."""

def launch(run_folder, config, params=None) -> Reviewer:
    """Extract crops, open the window. Call after %matplotlib qt."""

class Reviewer:
    def page(self, n): ...
    def set_filter(self, name): ...
    def to_dataframe(self): ...      # what is currently in the queue
    def save_page(self, path=None): ...
    def close(self): ...
```

Notebook usage mirrors `labelling.ipynb`:

```python
%matplotlib qt
from counting_boats.boat_utils import review

PARAMS = review.ReviewParams(panel_rows=4, panel_cols=4,   # 16 per page
                             filter="errors", sort="worst_overlap")
reviewer = review.launch(RUN_FOLDER, CONFIG, PARAMS)
```

---

## 9. Build order

**Phase 1 — data layer** (no UI)
`build_queue()` + crop extraction + `save_contact_sheet()` writing a static PNG
grid. Testable headlessly with the Agg backend and synthetic details/PNGs.
Already useful on its own: it produces shareable figures.

**Phase 2 — interactive window**
`Reviewer` class, figure with panel grid + legend axes, paging, key handling,
keymap save/restore, status line.

**Phase 3 — filters and sorting**
Live re-queue without re-extracting crops (crops keyed by row id, so filtering
is a view over the same store).

**Phase 4 — notebook integration**
Params + launch cells appended to `post_training_UN.ipynb`, in the style of the
other steps, with a markdown cell explaining the colour/linestyle convention.

**Phase 5 (optional, decide later) — flagging**
Press a key to mark a panel as "my label was wrong" / "detection is right",
writing `<run>/review_flags.csv`. This is the useful half of the boat-era
`counting_boats/reconcile_val_mistakes.py`, which its own docstring calls "not
super useful" and which predates the multi-class scheme. Worth doing only if
the review actually turns up systematic labelling errors — otherwise it is
scope for its own sake.

---

## 10. Testing

Headless, synthetic, no real data — same approach as the `evaluation` tests:

- Generate small fake scene PNGs and matching `*.details.csv` rows.
- `build_queue` — filters select the right rows; sorts order correctly; missing
  PNG produces a placeholder rather than an exception.
- Crop extraction — a box near a scene edge is clipped and padded, not wrapped;
  crop geometry matches the requested window.
- Colour assignment — every class in `names` gets a distinct colour; overrides
  apply; a 7-class config works as well as a 5-class one.
- Contact sheet — renders under `matplotlib.use("Agg")` and writes a PNG.
- Memory — assert no full-scene array is retained after prepare (crop store size
  is bounded).

Interactive key handling is checked by driving `Reviewer` methods directly
(`page`, `set_filter`) rather than synthesising Qt events.

---

## 11. Risks and decisions to confirm

| Risk | Handling |
|---|---|
| 180 MP scene decode is slow (seconds each) | One decode per scene in a single prepare pass, `tqdm` progress, optional on-disk crop cache |
| `%matplotlib qt` needs a Qt binding | `labelling.ipynb` already relies on it, so the environment supports it. Fallback: `%matplotlib widget` (ipympl), or Phase 1 static sheets which need no backend |
| Details files only exist for runs made since the reporting was added | `load_details` already raises a clear error telling you to re-run `compare_detections_to_ground_truth` |
| Scene PNG deleted after processing | Placeholder panel naming the missing file |
| Many classes make the legend crowded | Legend axes scrolls with `+/-` panel count; beyond ~12 classes, switch to `tab20` and a two-column legend |

**Settled:** one waterhole per cell, zoomed in, 4 x 4 = 16 per page, styled
after `Image_boxes_checks.ipynb`.

**Still to confirm before Phase 2:**

1. **Default filter** — plan opens on `all` in image order. Opening on `errors`
   sorted `worst_overlap` gets to the interesting cases faster, at the cost of a
   skewed first impression of overall quality.
2. **Phase 5 flagging** — build it, or leave review as read-only?
3. **Crop margin** — `crop_margin=3.0` means the window is three times the box,
   so you see the waterhole plus its surroundings and can judge whether a
   "false positive" is actually an unlabelled waterhole. Tighter (2.0) fills the
   cell with the waterhole; wider (5.0) shows more context but shrinks it. Easy
   to change once you have seen a page of real crops.
