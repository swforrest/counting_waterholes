# Waterhole dashboard — plan

A browser map of the 187 waterholes in central Arnhem Land. Click a bounding box to see that
waterhole's per-pixel predictions month by month, and how its surface composition has changed
seasonally and over the seven-year record.

This document exists so the prediction pipeline's output format *is* the dashboard's input
format. Nothing below requires re-running anything upstream.

---

## The headline decision: no backend

The entire dataset is small enough to serve as static files.

| | |
|---|---|
| class rasters + PNGs + confidence, all 187 sites | **~130 MB** |
| `waterhole_composition.csv` | ~15,700 rows, a few MB |
| `waterhole_boxes.geojson` | 187 polygons, <100 KB |

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
  waterhole_boxes.geojson          map layer: one polygon per site
  class_colours.json               id -> name -> hex, so the legend cannot drift
  waterhole_composition.csv        one row per (site_id, year_month)
  pixel_predictions/
    site_025/
      bounds.json                  WGS84 bounds; ONE per site, all months share a grid
      <stem>_2019-01_pred.png      display overlay, class 0 transparent
      <stem>_2019-01_pred.tif      authoritative class raster
      <stem>_2019-01_conf.tif      confidence 0-100
```

**`bounds.json`** carries `leaflet_bounds` as `[[south, west], [north, east]]`, which is
exactly what `L.imageOverlay` takes. One file per site rather than per month because the
inventory verifies every month of a site shares one grid.

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
threshold rather than letting them read as equally reliable.

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

1. **Static map + boxes.** Prove the basemap, the GeoJSON layer and click-to-select.
2. **Overlay a single month** for one site from its PNG and `bounds.json`. This is the step
   that validates the whole no-backend premise; do it early.
3. **Month slider** over the overlays, with preloading of adjacent months.
4. **Composition plot** from the CSV, colours from `class_colours.json`.
5. **Quality strip and confidence badges** — before the trend line, not after, so the
   caveats ship with the numbers rather than following them.
6. **Trend**, with the interval.

---

## Open questions for that stage

- **Hosting.** Static is enough; whether that is GitHub Pages, a QUT web share or an S3
  bucket affects only the deploy step.
- **CSV or per-site JSON.** One ~15,700-row CSV parsed client-side is fine for a desktop
  browser, but pre-splitting into `site_XXX.json` would make the site panel instant and
  avoid parsing the whole archive to draw one waterhole. Cheap to add to `wh_predict` as an
  extra export if it proves necessary.
- **Basemap imagery and attribution.** ESRI World Imagery is convenient and free for
  non-commercial use; check the licence against how this will be published.
- **PNG size.** ~150×150 px overlays are tiny but blocky when zoomed in. Upsampling them
  4× at write time, nearest-neighbour so no class is invented, would look better at the cost
  of ~16× the PNG bytes — still well under a gigabyte.
- **Does the footprint or the bounding box make the better default denominator?** Both are
  in the CSV. The footprint is the tighter, more meaningful region; the box is defined for
  all 187 sites where the footprint is not.
