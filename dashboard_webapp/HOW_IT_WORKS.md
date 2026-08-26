# How this web app works

Written for someone who is comfortable with Python and geospatial data but has
not built a browser application before. It explains the concepts this dashboard
is made of, and why each choice was made — `README.md` covers the commands.

---

## 1. Static vs dynamic, and why it is the first question

These two words get used as if they were a quality ranking. They are not; they
describe *where the code runs*.

**A dynamic site runs code on a server, per request.** You visit a URL, a machine
somewhere wakes up, runs Python (Flask, Django, Shiny) or JavaScript (Node),
probably queries a database, builds an HTML page on the spot and sends it to you.
Every visitor triggers work on that machine. This is what you need if the data is
too large to send, changes constantly, is private per user, or if the visitor
submits something that must be stored.

**A static site is just files.** The server does no thinking. It receives "give
me `/index.html`" and returns those bytes, exactly as they sit on disk. Same
bytes for everyone.

The trap is assuming static means *not interactive*. It does not. Once the
browser has the files, it runs the JavaScript inside them, and that JavaScript
can do arbitrary computation, redraw the page, fetch more files, respond to
clicks — all on the visitor's own machine. **Static/dynamic is about where the
code runs; interactive/inert is about whether there is any code at all.** They
are independent, and this dashboard is static *and* fully interactive.

That matters here because GitHub Pages will only ever serve files. It will not
run Python. So the question was never "can we have interactivity" — it was "can
every computation be done in the browser?" For this dashboard the answer is yes:

| What the dashboard does | Where it happens |
|---|---|
| Recolour 187 boxes for the selected month | Browser, reading a pre-computed array |
| Show a month's prediction | Browser, swapping one image URL |
| Stacked-area composition chart | Browser, ~200 lines of SVG maths |
| Dry-season trend fit | Browser, least squares over 7 numbers |
| Run the segmentation model | **Never** — done offline, in the notebooks |

The heavy work (running the classifier over 15,708 site-months) already happened
in `cookie-cutting/`. The dashboard only *displays* its output. That asymmetry is
what makes a no-backend design honest rather than a compromise.

The practical payoff: nothing to pay for, nothing to keep running, nothing to
patch, and no server to fall over the week before a deadline. The cost: the whole
dataset must be small enough to ship to the browser, which is the constraint that
drove the WebP conversion.

---

## 2. What actually happens when someone opens the page

```
   browser                                    GitHub Pages (a file server)
      |                                                  |
      |  GET /counting_waterholes/                       |
      |------------------------------------------------->|
      |<------------------------------------------------- index.html
      |
      |  reads the HTML, finds <link> and <script> tags, asks for those too
      |------------------------------------------------->|
      |<------------------------------------------------- app.css, leaflet.js, main.js
      |
      |  now main.js is RUNNING, and it asks for data:
      |  GET data/manifest.json, data/overview.json, data/boxes.geojson
      |------------------------------------------------->|
      |<------------------------------------------------- the JSON files
      |
      |  draws the map. user clicks a box:
      |  GET data/sites/site_004.json
      |  GET data/overlays/site_004/..._2023-11_pred.webp
      |------------------------------------------------->|
      |<------------------------------------------------- more files
```

Every arrow is the same operation: *ask for a file by name, receive its bytes*.
The server is identical whether it is `python3 -m http.server` on your laptop or
GitHub's global network. That is why local testing is trustworthy here — it is
not a simulation of the real thing, it is the real thing with a different file
server.

### The three kinds of file, and the division of labour

- **HTML** (`index.html`) — *structure*. The elements that exist: a header, a map
  container, a side panel. Think of it as the skeleton, written once and mostly
  static.
- **CSS** (`css/app.css`) — *appearance*. Rules of the form "elements matching
  this description look like this". No logic. When JavaScript wants something to
  look different, it usually adds or removes a class name and lets CSS decide
  what that means.
- **JavaScript** (`js/*.js`) — *behaviour*. The only part that makes decisions.

Keeping them separate is not ceremony. The `image-rendering: pixelated` rule that
stops the browser blurring class boundaries lives in one line of CSS; if
appearance were tangled into the drawing logic it would have to be repeated
everywhere an overlay is created.

### Why you cannot just double-click index.html

Opening the file directly gives it a `file://` address, and browsers refuse to
let a `file://` page `fetch` other files — a security rule, because otherwise any
downloaded HTML could read your disk. You get a blank page and a console error.

Running `python3 -m http.server` gives the same files an `http://` address, and
the restriction lifts. The server is doing nothing clever; it exists only to make
the browser willing.

---

## 3. The data layer, and the one constraint that shaped everything

A static host has no index. It answers "give me this exact path" and nothing
else — you cannot ask "what files are in this folder?" There is no query, no
filter, no API.

So **the app can only read files whose names it can work out in advance.** That
single limitation explains most of `tools/build_dashboard_data.py`:

- The composition table is one 15,700-row CSV. The browser cannot ask for "the
  rows where site_id = 004", so the build script splits it into
  `sites/site_004.json` — a name the app can construct from a site id.
- Overlay filenames embed rounded lat/lon tags
  (`..._004_S13p52_E134p51_2023-11_pred.webp`) that cannot be derived from the
  site id. The build script writes each site's stem into its JSON, so the app
  reads the prefix rather than guessing it.
- Whether a given month has a `conf` image is a fact about the disk. The build
  script probes for it and records the answer, so the app knows a layer is
  missing without having to request it and handle a 404.

That last one has a specific reason. The archive is mid-migration from PNG to
WebP, and `bounds.json` names the format the site is being converted *to* — so a
site can advertise `webp` while some of its months are still `png`. Recording the
real extension per site-month per layer means a half-converted archive works, and
the conversion can keep running while you use the dashboard.

### Why the file formats matter more than they look

The overlays are written twice: as GeoTIFF (the authoritative product, used for
analysis) and as an ordinary image (purely for display). The browser cannot read
a GeoTIFF — it has no concept of a coordinate reference system, and decoding one
would need a tile server or a WebAssembly library. But it can stretch a normal
image between two corners, which is all the display needs.

Format choice per layer is not cosmetic:

| Layer | Format | Why |
|---|---|---|
| `rgb` | lossy WebP | photographic; lossy is invisible here and ~10× smaller |
| `pred` | lossless WebP | lossy encoding invents colours *between* the class colours, which decode to the wrong class |
| `conf` | lossless WebP | same problem: an invented intermediate value is a wrong confidence |

A lossy class raster is wrong in the worst way — it still looks fine.

---

## 4. The map

Leaflet is the mapping library. Its job is the coordinate arithmetic: you hand it
latitudes and longitudes, and it works out where they fall on screen at the
current zoom and pan, and keeps that correct as the user drags.

Three kinds of thing sit on it:

1. **The basemap** — ESRI World Imagery, fetched as tiles from ESRI's servers as
   you pan. The one part of the app that depends on an outside service.
2. **Vector layers** — the bounding boxes and basin footprints, as GeoJSON.
   Shapes with coordinates, drawn as crisp lines at any zoom, clickable.
3. **Image overlays** — the monthly prediction, stretched between the corners in
   `bounds.json`.

### `L.imageOverlay` is the load-bearing piece

```js
L.imageOverlay(url, [[south, west], [north, east]])
```

Give it an image and two corners, and it places it geographically. The prediction
pipeline already writes precisely this, as `leaflet_bounds` in each site's
`bounds.json` — which is why there is no coordinate maths anywhere in this app.

Because all three layers of all 84 months of a site share one pixel grid and one
`bounds.json`, changing month or layer is a *source swap* on a single overlay
object: `overlay.setUrl(newURL)`. Nothing moves, nothing reprojects. That is what
makes flipping between prediction, true colour and confidence fast enough to
actually use — and that flip is the point, because a class overlay on a satellite
basemap is unfalsifiable on its own. The basemap is a different sensor from a
different year and season, so it disagrees with the prediction whether the
prediction is right or wrong.

### Why the outlines are vectors, not drawn into the images

A site's box and footprint are identical across all 84 of its months. Baking them
into the overlays would repeat one outline across 15,708 images, destroy the
pixels underneath, and make them impossible to switch off. As GeoJSON they cost
~650 KB once, stay sharp at every zoom, and can be clicked to select a site.

---

## 5. How the JavaScript is organised

### Modules

The code is split across `js/*.js` files that import each other:

```js
import * as colours from "./colours.js";
```

This is the browser's own module system (ES modules) — the `type="module"`
attribute on the `<script>` tag switches it on. It works like Python's `import`:
each file has its own namespace, and only what it `export`s is visible outside.

Larger projects usually run a *bundler* (Vite, webpack) that merges modules into
one file and installs dependencies from npm. **This project has none, on purpose.**
There is no Node.js on the machine this was built on; a bundler would add an
install step, a lockfile and a version-drift problem to a project whose actual
dependency list is "Leaflet". Leaflet is instead vendored — the file is committed
under `vendor/` — so the app has no CDN to go down, no npm install, no build, and
what you edit is exactly what ships.

The charts are hand-written SVG for the same reason: the smallest capable
charting library is larger than this entire application.

### One state object

Every module here is a drawer of things. The decisions live in `js/main.js`, in
one object:

```js
const state = {
  monthIndex: 0, siteId: null, layer: "pred",
  denominator: "bbox", colourBy: "dominant", ...
};
```

Every control does two things: change `state`, then call `apply()`. `apply()` is
the only function that pushes state onto the screen. Nothing reads the current
month off the slider or the selected site out of the page.

This matters more than it sounds. The alternative — each control updating the
bits it thinks it owns — is how you get a chart showing June while the map shows
July. When the state and the screen can only disagree in one place, that class of
bug stops existing. It is the same discipline as keeping one authoritative
dataframe instead of several partial copies.

### Asynchrony

Fetching a file takes time. `async`/`await` marks the waiting:

```js
const site = await data.loadSite("004");
```

The browser is free to keep drawing and responding to clicks while that runs,
rather than freezing. `Promise.all([...])` runs several fetches at once and waits
for all of them — used at startup so the manifest, overview and GeoJSON download
in parallel rather than one after another.

---

## 6. Deploying to GitHub Pages from a subdirectory

GitHub Pages has two modes, and the difference is the thing most people trip on.

**Deploy from a branch** (the old way) lets you pick a branch and *either* its
root *or* a folder named `/docs`. Those are the only two choices. It cannot
publish `dashboard_webapp/`, and `/docs` here already holds the pdoc API
documentation.

**GitHub Actions** (what this repo uses) runs a workflow on GitHub's machines
whenever you push. The workflow can assemble the site however it likes and then
say "publish this directory". `.github/workflows/deploy-dashboard.yml` does:

1. Check out the repository.
2. Download the data bundle from a GitHub Release and unzip it into
   `dashboard_webapp/data/`.
3. Fail early if there is no data, or if any asset path is root-absolute.
4. Upload `dashboard_webapp/` as the site and deploy it.

The result is served at `https://swforrest.github.io/counting_waterholes/`, and
contains only what was in `dashboard_webapp/`.

One setting must be changed by hand, once, because a workflow is not allowed to
grant itself the right to publish: **Settings → Pages → Source: GitHub Actions**.

### Why the data comes from a Release, not from git

Git stores every version of every file forever. Committing ~220 MB of overlays
would add 220 MB to the history *permanently*, and doing it again after the next
model run would add another 220 MB — for a repository already carrying ~300 MB,
on a OneDrive-backed working copy that is slow to begin with. Release assets live
outside the git object store: replaced in place, downloaded only when asked for,
invisible to `git clone`.

`data_sample/` (8 sites, ~9.5 MB) *is* committed, so a fresh clone works
immediately and a deploy still produces a working site if the release is missing.

### The subpath trap

Locally the site is at `http://localhost:8000/`, so `/css/app.css` resolves
correctly. Published, the site is at `/counting_waterholes/`, and `/css/app.css`
now points at `swforrest.github.io/css/app.css` — which does not exist. The page
loads, unstyled, and looks broken in a way that is confusing to diagnose.

The fix is to always write paths *relative*: `css/app.css`, never `/css/app.css`.
The workflow greps for the mistake and fails the build rather than publishing it.

---

## 7. Where to change things

| You want to | Edit |
|---|---|
| Change a class colour | `cookie-cutting/predictions/class_colours.json`, then re-run the build — the app never hard-codes one |
| Add a chart | `js/charts.js`, called from `js/panel.js` |
| Change what the map shows | `js/map.js` |
| Add a control | `index.html` for the element, `js/main.js` to wire it into `state` |
| Change the data shape | `tools/build_dashboard_data.py`, then `js/data.js` |
| Change layout or colour | `css/app.css` |

The rule the code follows: **anything that carries meaning comes from the
pipeline, not from the app.** Class colours, the confidence ramp, class names and
quality levels are all read at runtime from files the prediction code wrote. A
legend that disagrees with the pixels it describes is worse than no legend, and
hard-coding is how that happens.
