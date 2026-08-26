# Waterhole dashboard

A browser map of the waterholes in central Arnhem Land. Click a bounding box to
see that waterhole's per-pixel predictions month by month, and how its surface
composition has changed seasonally and across the seven-year record.

Static site — no server, no database, no tile service. New to this?
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) explains the concepts; this page is the
commands. The design brief is [dashboard_plan.md](dashboard_plan.md).

---

## Run it locally

```bash
cd dashboard_webapp
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

That is the whole toolchain. No npm, no build, no install — Leaflet is committed
under `vendor/`, and the charts are hand-written SVG.

> Do not open `index.html` by double-clicking it. A `file://` page is not allowed
> to `fetch` anything, so you get a blank screen. The tiny web server above is
> what makes the browser willing.

`data_sample/` (8 sites) is committed, so a fresh clone runs immediately.

### Useful URLs

The view is in the address bar, so any state can be bookmarked or sent to someone:

```
?site=004&month=2023-11&layer=conf&denominator=footprint
```

### Keyboard

| Key | |
|---|---|
| `1` `2` `3` | prediction / true colour / confidence |
| `←` `→` | previous / next month |
| `space` | play through the months |
| `Esc` | close the site panel |

---

## Rebuild the data

The app reads `data/` if it exists and falls back to `data_sample/`. Neither is
committed except the sample; both are generated from `cookie-cutting/predictions/`.

```bash
# the committed sample (fast, a few seconds)
python3 tools/build_dashboard_data.py \
    --sites 000,001,002,003,004,005,006,008 --out data_sample

# everything (~220 MB, a few minutes — OneDrive may need to hydrate files first)
python3 tools/build_dashboard_data.py --sites all --out data

# everything, without copying 47,000 files: symlinks instead. Local use only —
# these cannot be zipped into a release bundle.
python3 tools/build_dashboard_data.py --sites all --out data --symlink

# rebuild only the JSON, leaving existing images alone
python3 tools/build_dashboard_data.py --sites all --out data --no-overlays
```

Standard library only, so the system Python is fine — no conda environment needed.

Then check what was built:

```bash
python3 tools/check_data.py --data data_sample
```

It verifies that every overlay the app can construct a URL for exists on disk,
that composition fractions sum to 1, that the map's boxes all have records behind
them, and that any site without a footprint is declared as such. These are the
failure modes that otherwise show up as a blank panel or a missing image rather
than as an error.

Re-run it after regenerating predictions, after the WebP conversion finishes, or
after the composition table changes. The script probes the disk for what actually
exists, so a partially converted or partially written archive is safe to build
from: missing layers are recorded as missing and the app greys out that toggle.

---

## Deploy

Once, by hand: **Settings → Pages → Build and deployment → Source: GitHub
Actions.** (Not "Deploy from a branch" — that mode can only publish the repo root
or `/docs`, so it cannot serve this subdirectory.)

Then, whenever the data changes:

```bash
python3 tools/build_dashboard_data.py --sites all --out data
python3 tools/make_release_bundle.py                    # -> dist/dashboard-data.zip

cd ..
gh release create dashboard-data-v1 dashboard_webapp/dist/dashboard-data.zip \
    --title "Dashboard data v1" --notes "187 sites, 2019-01 to 2025-12"
# or, replacing the data on an existing tag:
gh release upload dashboard-data-v1 dashboard_webapp/dist/dashboard-data.zip --clobber
```

Pushing any change under `dashboard_webapp/` deploys automatically. To redeploy
without a commit — after uploading new data, say — run the **Deploy dashboard**
workflow from the Actions tab.

Live at <https://swforrest.github.io/counting_waterholes/>.

The overlays are not in git: ~220 MB across ~47,000 files would sit in the
history forever, and again after every model run. They ride as a release asset
instead. If the release is missing, the deploy still succeeds using
`data_sample/` and logs a warning — a stale site beats a failed one.

---

## When something goes wrong

**Blank page, console says "Failed to fetch".** No data directory. Run the build
script, or check you are on `http://localhost:8000` rather than `file://`.

**Page loads but has no styling, only on the live site.** A root-absolute path
(`/css/app.css`) — correct on localhost, wrong under `/counting_waterholes/`.
Always write `css/app.css`. The deploy workflow greps for this and fails rather
than publishing it.

**Changes do not appear.** The browser cached the old JavaScript. Hard-reload
(<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd>).

**A layer button is greyed out.** That month has no image for that layer — either
it was never observed, or the layer has not been written for that site yet.
`manifest.json` → `layer_counts` shows the totals.

**Blocky overlays when zoomed in.** Intended. The overlays are ~150×150 px, and
`image-rendering: pixelated` keeps class boundaries exactly where the raster puts
them. Letting the browser smooth them would blend adjacent classes into colours
that are not in the legend and were never in the model's output.

---

## Layout

```
index.html          the page
css/app.css         layout and appearance
js/
  main.js           state, controls, wiring — the only module that decides things
  data.js           fetching and caching
  colours.js        classes and the confidence ramp, read from class_colours.json
  map.js            basemap, boxes, footprints, month recolouring
  overlay.js        the image overlay: placement, layer switch, preloading
  panel.js          the site panel
  charts.js         SVG composition chart, confidence line, dry-season trend
vendor/leaflet/     Leaflet 1.9.4, committed so the site has no CDN dependency
tools/
  build_dashboard_data.py    predictions/ -> data/
  make_release_bundle.py     data/ -> dist/dashboard-data.zip
data_sample/        committed 8-site subset
data/               full build (gitignored)
```

---

## What the dashboard is careful about

The pipeline emits quality and confidence columns for a reason, and the UI is
built to keep them attached to the numbers rather than tucked behind a toggle.

- **The model saw 27 sites and is applied to 187.** A waterhole unlike anything
  labelled still gets a confident-looking classification. Mean confidence is on
  every site panel, and the `rgb` layer is there so a prediction can be checked
  against what the classifier actually saw.
- **Confidence is saturated, not calibrated.** 93% of pixels sit above 0.9. The
  confidence chart is drawn on a 0.7–1.0 axis and marks months below 0.9,
  because the information is entirely in the dips. Read it as relative.
- **Wet-season months rest on less evidence.** January and February average ~2
  clear scenes against ~6 in August — backwards for a study of water. The quality
  strip sits directly under the composition chart, on the same axis.
- **Unobserved months are gaps, not zeroes.** The chart breaks; it never
  interpolates across a month that was not measured.
- **`flag_isolated_wet` marks suspicion, not error.** Flagged months are marked
  and kept. Deciding is yours.
- **The trend is fitted on annual dry-season means**, roughly 7 points, not on
  84 autocorrelated monthly values that would report an interval several times
  too narrow.
