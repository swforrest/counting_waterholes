# Waterhole dashboard — plan

A browser map of the 187 waterholes in central Arnhem Land. Click a bounding box to see that
waterhole's per-pixel predictions month by month, and how its surface composition has changed
seasonally and over the seven-year record.

This document exists so the prediction pipeline's output format *is* the dashboard's input
format. Nothing below requires re-running anything upstream.

---

## The headline decision: no backend

The entire dataset is small enough to serve as static files — provided the true-colour layer
is not encoded as PNG. Measured on a real 84-month site and scaled to all 187:

| | PNG | WebP |
|---|---|---|
| `pred` overlays | 38 MB | **16 MB** |
| `conf` overlays | 292 MB | **75 MB** |
| `rgb` overlays | 776 MB | **74 MB** |
| GeoTIFFs (`_pred.tif`, `_conf.tif`) | 92 MB | 92 MB (unchanged) |
| **total** | **~1.2 GB** | **~260 MB** |

| | |
|---|---|
| `waterhole_composition.csv` | ~15,700 rows, a few MB |
| `waterhole_boxes.geojson` | 187 polygons, <100 KB |
| `waterhole_footprints.geojson` | 176 polygons, ~580 KB |

True colour is photographic and compresses terribly as PNG — 49 KB a chip against 4.7 KB as
lossy WebP, for a difference no one can see at this scale. **Set `image_format="webp"`**
unless something in the hosting chain cannot serve it; every browser since 2020 reads it.
Only the `rgb` layer is lossy. Class and confidence images stay lossless in either format,
because lossy encoding of flat colour invents intermediate values that decode to the wrong
class or the wrong confidence — wrong in a way that looks fine.

The GeoTIFFs are not served to the browser at all; they are the authoritative product for
analysis and could be excluded from a deploy, halving the WebP figure again.

That means a static site — GitHub Pages, Netlify, an S3 bucket, or a QUT web share — with no
server, no database and no tile service. It can also run from a local directory over
`python -m http.server` for internal review.

The single design choice that makes this possible is writing a **PNG plus WGS84 bounds**
alongside each GeoTIFF. A browser cannot draw a GeoTIFF without a tile server or a WASM
decoder, but every mapping library can place a PNG given its corners. The GeoTIFF stays the
authoritative product for analysis; the PNG is purely for display.

---

## What the pipeline already emits

```
cookie-cutting/predictions/
  waterhole_boxes.geojson          map layer: one box per site (all 187)
  waterhole_footprints.geojson     map layer: one basin outline per site (176)
  class_colours.json               classes + the confidence ramp, so legends cannot drift
  waterhole_composition.csv        one row per (site_id, year_month)
  pixel_predictions/
    site_025/
      bounds.json                  WGS84 bounds + layer list; ONE per site
      <stem>_2019-01_pred.png      class overlay, class 0 transparent
      <stem>_2019-01_rgb.png       true colour, transparent where unobserved
      <stem>_2019-01_conf.png      confidence, masked to the classified pixels
      <stem>_2019-01_pred.tif      authoritative class raster
      <stem>_2019-01_conf.tif      confidence 0-100
```

**`bounds.json`** carries `leaflet_bounds` as `[[south, west], [north, east]]`, which is
exactly what `L.imageOverlay` takes. One file per site rather than per month because the
inventory verifies every month of a site shares one grid. It also lists `png_layers`, so the
UI offers only the layers that were actually written.

### The three display layers

All three share one grid and one `bounds.json`, so switching between them is a source swap on
a single `L.imageOverlay` — nothing moves, nothing reprojects.

| layer | shows | why it is there |
|---|---|---|
| `pred` | the classification | the product |
| `rgb` | Sentinel-2 true colour, same month, same pixels | what the classifier saw |
| `conf` | max class probability | how much to believe it |

`rgb` and `conf` are not decoration. **A class overlay on a satellite basemap is
unfalsifiable on its own**: the basemap is a different sensor from a different year, usually
a different season, so a viewer comparing the two cannot distinguish a correct classification
from a confident wrong one — the imagery underneath disagrees either way. Flipping between
the prediction, the actual input, and the model's own uncertainty is what turns the overlay
from something to be taken on trust into something that can be checked. Given the model saw
27 of 187 sites, that check is the difference between a dashboard and a demo.

**Confidence is masked to the classified pixels**, not the whole chip, so `conf` and `pred`
cover exactly the same area and the flip compares like with like. `rgb` is instead masked to
the *observed* pixels, so a cloud gap reads as a hole rather than as black ground.

### The two counting regions, as vector layers

Every composition number is counted inside one of two regions, and the dashboard should be
able to draw both over any layer:

- **bounding box** — `waterhole_boxes.geojson`, the hand-drawn label extent. Defined for all
  187 sites; the denominator behind every `bbox_*` column.
- **basin footprint** — `waterhole_footprints.geojson`, derived from seasonal amplitude and
  dry-season NDVI anomaly. Tighter and more meaningful, but only 176 sites have one; the
  denominator behind every `footprint_*` column. The file names the 11 that do not in
  `sites_without_footprint`, so the UI can say so rather than leaving a box mysteriously
  bare.

**These are vectors, deliberately, and must not be drawn into the overlay images.** A site's
box and footprint are identical across all 84 of its months, so baking them in would repeat
the same outline in 15,708 images, destroy the pixels underneath, and leave them impossible
to toggle. As GeoJSON they are two more map layers costing ~650 KB together, they stay crisp
at every zoom, and clicking one can select the site.

Drawing them matters more than it sounds: the region is *what the number means*. A site whose
box contains two basins, or whose footprint has drifted off the water, produces a composition
series that is internally consistent and describing the wrong ground — and the only way to
see that is to look at the outline sitting on the imagery.

**Filenames are stable and derivable.** Given a `site_id` and `year_month` from the CSV, the
overlay path follows from the naming convention in `wh_naming.py` — the app never has to
list a directory.

### CSV columns the dashboard uses

- Identity and placement: `site_id`, `year_month`, `label`, `center_lon`, `center_lat`,
  `lon_min`/`lon_max`/`lat_min`/`lat_max`
- Composition: `bbox_frac_<class>` for the six classes (and `footprint_frac_<class>` if the
  footprint denominator is preferred), plus `bbox_n_classified` as the denominator
- Trust: `mean_confidence`, `data_quality`, `flag_isolated_wet`, `gap_fraction`

---

## Views

### 1. Map

`waterhole_boxes.geojson` over a satellite basemap (MapLibre GL with an ESRI or Mapbox
raster source; Leaflet is equally fine at this scale). Boxes coloured by the currently
selected month's dominant class, or by a chosen class's fraction, both read from the CSV.

A month slider at the top recolours all 187 boxes at once — an overview of the whole
population drying and refilling, which is the view no single-site plot gives.

### 2. Site panel, on click

- **Overlay**: `_pred.png` for the selected month, placed with `bounds.json`, over the
  basemap with an opacity slider. Stepping the month swaps the image source.
- **Layer switch**: `pred` / `rgb` / `conf`, from `bounds.json`'s `png_layers`. Worth making
  this a single key or a click rather than a menu — the whole value is in flipping quickly
  between them on the same month, and anything slower will not be used. The confidence
  colourbar comes from `class_colours.json` → `confidence.stops`.
- **Region outlines**: the site's box and footprint over whichever layer is showing, from the
  two GeoJSON layers. Keep them on by default and tie the outline highlight to the
  denominator the composition plot is using, so it is always visible which region the numbers
  on screen are counted within.
- **Composition through time**: stacked area of the six class fractions, colours from
  `class_colours.json`. This mirrors `wh_plots.plot_site_composition`, so the notebook and
  the dashboard cannot disagree.
- **Long-term trend**: a linear fit per class fraction, with the slope reported per year.
  See the caveat below before putting a number on screen.
- **Quality strip**: a thin band under the time axis, shaded by `data_quality`, with
  `flag_isolated_wet` months marked. The trend must be visibly conditional on this.

---

## Three things the UI has to be honest about

These are the reasons the pipeline emits the columns it does. A dashboard that shows only
the fractions would be more confident than the data supports.

**1. The model saw 27 sites and is applied to 187.** A waterhole unlike anything labelled
still gets a confident-looking classification. Surface `mean_confidence` per site — a small
badge on the map and a line on the site panel — and consider desaturating boxes below a
threshold rather than letting them read as equally reliable. The `rgb` and `conf` overlays
are the per-pixel version of the same warning.

*Measured, and it changes how the confidence layer should be drawn:* over 6.5 M classified
pixels sampled across 300 site-months, **93% sit above 0.9 confidence** and the 5th
percentile is 0.82. Gradient boosting is poorly calibrated and saturates; the model is
nominally near-certain almost everywhere, and uncertainty is confined to a thin speckle at
class boundaries. So a continuous 0.4–1.0 ramp renders as a flat field with edges picked out.
That is honest — it is what the model thinks — but it means the useful view is probably a
**low-confidence mask** (say, everything under 0.9 highlighted) rather than a smooth ramp.
The PNG supports either; the choice belongs to whoever builds the panel. What it must not
become is a rescaled ramp that manufactures visible doubt where the model has none, or
implies calibration the model does not have.

**2. Wet-season months are the thin ones.** January and February average ~2 clear scenes
against ~6 in August, and 103 site-months are entirely unobserved. Composition in the wet
season rests on less evidence than in the dry, which is precisely backwards for a study of
water. The quality strip must be visible on the same axis as the composition, not hidden
behind a toggle.

**3. `flag_isolated_wet` marks suspicion, not error.** The pipeline flags an isolated wet
month between dry ones on a thin median because it is more likely a compositing artefact
than a rainfall event — but it might be the event. Mark it; never drop it, and never smooth
it away.

**On the trend line.** Seven years of monthly data with strong seasonality is not many
independent observations, and a naive OLS slope on autocorrelated monthly fractions will
have badly understated uncertainty. Either fit on annual dry-season means (about 7 points,
honest), or show the slope with an interval that accounts for autocorrelation. Do not put a
bare "declining 2%/year" on screen without one.

---

## Build order

1. **Static map + boxes + footprints.** Prove the basemap, both GeoJSON layers and
   click-to-select.
2. **Overlay a single month** for one site from its `_pred` image and `bounds.json`. This is
   the step that validates the whole no-backend premise; do it early.
3. **Layer switch** between `pred`, `rgb` and `conf` on that one month — cheap once step 2
   works, since all three share the bounds, and it is what makes every later view checkable.
4. **Month slider** over the overlays, with preloading of adjacent months.
5. **Composition plot** from the CSV, colours from `class_colours.json`.
6. **Quality strip and confidence badges** — before the trend line, not after, so the
   caveats ship with the numbers rather than following them.
7. **Trend**, with the interval.

---

## Open questions for that stage

- **Hosting.** Static is enough; whether that is GitHub Pages, a QUT web share or an S3
  bucket affects only the deploy step. Note GitHub Pages' 1 GB soft limit — comfortable at
  WebP sizes, not at PNG sizes, which is the practical reason the format matters.
- **CSV or per-site JSON.** One ~15,700-row CSV parsed client-side is fine for a desktop
  browser, but pre-splitting into `site_XXX.json` would make the site panel instant and
  avoid parsing the whole archive to draw one waterhole. Cheap to add to `wh_predict` as an
  extra export if it proves necessary.
- **Basemap imagery and attribution.** ESRI World Imagery is convenient and free for
  non-commercial use; check the licence against how this will be published.
- **Overlay resolution.** ~150×150 px overlays are tiny but blocky when zoomed in. The fix
  is CSS, not bytes: `image-rendering: pixelated` on the `pred` and `conf` layers keeps class
  boundaries honest at any zoom, where the browser's default smoothing would blend classes
  into colours that are not in the legend. Upsampling at write time is the fallback if that
  proves insufficient, at ~16× the bytes.
- **Whether `rgb` should also carry the basemap's job.** With a true-colour layer per month,
  the satellite basemap is arguably redundant inside a box — and it is the misleading half,
  being a different sensor from a different year. Possibly the basemap should fade out when
  a site is selected.
- **Does the footprint or the bounding box make the better default denominator?** Both are
  in the CSV. The footprint is the tighter, more meaningful region; the box is defined for
  all 187 sites where the footprint is not.
