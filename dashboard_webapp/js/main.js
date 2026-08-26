/*
  Wiring. Holds the application state, listens to the controls, and tells the
  other modules what to draw.

  The shape is deliberate and worth naming, because it is the pattern most small
  browser apps end up wanting: there is exactly one `state` object, every control
  does nothing but change it and call apply(), and apply() is the only thing that
  touches the map and the panel. Nothing reads the current month off a slider or
  the selected site off the DOM. When state and screen can only disagree in one
  place, the bugs where a chart shows June and the map shows July stop happening.
*/

import * as data from "./data.js";
import * as colours from "./colours.js";
import * as mapModule from "./map.js";
import * as overlay from "./overlay.js";
import * as panel from "./panel.js";

const state = {
  manifest: null,
  overview: null,
  monthIndex: 0,
  siteId: null,
  site: null,      // the selected site's full record, once fetched
  bounds: null,    // its bounds.json, for placing the overlay
  layer: "pred",
  denominator: "bbox",
  colourBy: "dominant",
  opacity: 0.85,
  playing: false,
};

const element = (id) => document.getElementById(id);

// ---------------------------------------------------------------- boot

async function boot() {
  const { root, manifest } = await data.boot();
  state.manifest = manifest;
  colours.init(manifest);

  const [overview, boxes, footprints] = await Promise.all([
    data.loadOverview(), data.loadBoxes(), data.loadFootprints(),
  ]);
  state.overview = overview;

  const map = mapModule.create("map", { overview, onSelect: selectSite });
  overlay.attach(map);
  mapModule.addVectors(boxes, footprints);
  panel.attach(element("panel"));

  buildColourByOptions();
  buildLegend();
  wireControls();

  // Restore whatever the URL asks for, so a view can be shared or bookmarked.
  const params = new URLSearchParams(location.search);
  const month = params.get("month");
  state.monthIndex = month && manifest.months.includes(month)
    ? manifest.months.indexOf(month)
    : manifest.months.length - 1;
  if (["pred", "rgb", "conf"].includes(params.get("layer"))) state.layer = params.get("layer");
  if (["bbox", "footprint"].includes(params.get("denominator"))) {
    state.denominator = params.get("denominator");
  }

  const slider = element("month-slider");
  slider.max = String(manifest.months.length - 1);
  slider.value = String(state.monthIndex);
  element("denominator").value = state.denominator;

  const note = element("boot-note");
  if (note) {
    note.textContent = manifest.subset
      ? `Showing a ${manifest.sites.length}-site sample built from ${root}/. ` +
        `Run the build script with --sites all for the full ${187} sites.`
      : `${manifest.sites.length} sites, ${manifest.months.length} months.`;
  }

  apply();

  const requested = params.get("site");
  if (requested && manifest.sites.includes(requested)) selectSite(requested);
}

// ------------------------------------------------------------- controls

function buildColourByOptions() {
  const select = element("colour-by");
  for (const name of colours.classNames()) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${colours.prettyName(name)} fraction`;
    select.appendChild(option);
  }
}

function buildLegend() {
  const legend = element("map-legend");
  legend.innerHTML = "<h3>Classes</h3>";
  for (const name of colours.classNames()) {
    const row = document.createElement("div");
    row.className = "legend-row";
    row.innerHTML =
      `<span class="swatch" style="background:${colours.colourOf(name)}"></span>` +
      `<span>${colours.prettyName(name)}</span>`;
    legend.appendChild(row);
  }
}

function showConfidenceLegend(show) {
  const legend = element("map-legend");
  const existing = legend.querySelector(".confidence-key");
  if (existing) existing.remove();
  if (!show) return;

  const ramp = colours.confidenceRamp();
  const block = document.createElement("div");
  block.className = "confidence-key";
  block.innerHTML =
    `<h3 style="margin-top:10px">Confidence</h3>` +
    `<div class="legend-ramp" style="background:${colours.confidenceGradient()}"></div>` +
    `<div class="legend-ends"><span>${ramp.vmin}</span><span>${ramp.vmax}</span></div>`;
  legend.appendChild(block);
}

function wireControls() {
  element("month-slider").addEventListener("input", (event) => {
    setMonth(Number(event.target.value));
  });

  element("play").addEventListener("click", togglePlay);

  element("colour-by").addEventListener("change", (event) => {
    state.colourBy = event.target.value;
    apply();
  });

  element("denominator").addEventListener("change", (event) => {
    state.denominator = event.target.value;
    // A different denominator is a different question, so the panel is rebuilt
    // rather than patched: every series in it changes.
    if (state.site) panel.render(state.site, state, panelHandlers());
    mapModule.highlightDenominator(state.denominator);
    syncURL();
  });

  element("show-boxes").addEventListener("change", (event) => {
    mapModule.setLayerVisible("boxes", event.target.checked);
  });
  element("show-footprints").addEventListener("change", (event) => {
    mapModule.setLayerVisible("footprints", event.target.checked);
  });
  element("dim-basemap").addEventListener("change", (event) => {
    mapModule.dimBasemap(event.target.checked);
  });

  element("opacity").addEventListener("input", (event) => {
    state.opacity = Number(event.target.value) / 100;
    overlay.setOpacity(state.opacity);
  });

  for (const button of document.querySelectorAll(".layer-button")) {
    button.addEventListener("click", () => setLayer(button.dataset.layer));
  }

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    const keys = { 1: "pred", 2: "rgb", 3: "conf" };
    if (keys[event.key]) { setLayer(keys[event.key]); event.preventDefault(); return; }
    if (event.key === "ArrowLeft") { setMonth(state.monthIndex - 1); event.preventDefault(); }
    if (event.key === "ArrowRight") { setMonth(state.monthIndex + 1); event.preventDefault(); }
    if (event.key === "Escape") deselect();
    if (event.key === " ") { togglePlay(); event.preventDefault(); }
  });
}

// ---------------------------------------------------------------- state

function setMonth(index) {
  const clamped = Math.max(0, Math.min(state.manifest.months.length - 1, index));
  if (clamped === state.monthIndex) return;
  state.monthIndex = clamped;
  element("month-slider").value = String(clamped);
  apply();
}

function setLayer(layer) {
  if (!state.site) return;
  // A layer with no image for this month is not offered; the archive is written
  // per site-month and can legitimately be incomplete.
  if (!state.site.layers[layer] || !state.site.layers[layer][state.monthIndex]) return;
  state.layer = layer;
  apply();
}

async function selectSite(siteId) {
  if (!state.manifest.sites.includes(siteId)) return;
  state.siteId = siteId;
  const [site, bounds] = await Promise.all([data.loadSite(siteId), data.loadBounds(siteId)]);
  state.site = site;
  state.bounds = bounds;

  element("overlay-controls").hidden = false;
  panel.render(site, state, panelHandlers());
  mapModule.setSelected(siteId, state.monthIndex, state.colourBy);
  mapModule.highlightDenominator(state.denominator);
  mapModule.flyToSite(bounds.leaflet_bounds);
  apply();
}

function deselect() {
  state.siteId = null;
  state.site = null;
  state.bounds = null;
  overlay.remove();
  element("overlay-controls").hidden = true;
  showConfidenceLegend(false);
  panel.clear();
  mapModule.setSelected(null, state.monthIndex, state.colourBy);
  syncURL();
}

const panelHandlers = () => ({ onClose: deselect, onScrub: setMonth });

/**
 * Push the state onto the screen. The single place that draws.
 */
function apply() {
  const months = state.manifest.months;
  element("month-label").textContent = months[state.monthIndex];

  mapModule.recolour(state.monthIndex, state.colourBy);

  if (state.site) {
    const drawn = overlay.show(state.site, state.monthIndex, state.layer, state.bounds);
    updateLayerButtons(drawn);
    showConfidenceLegend(state.layer === "conf" && drawn);
    panel.updateMonth(state.site, state);
  }
  syncURL();
}

function updateLayerButtons(drawn) {
  const note = element("overlay-note");
  for (const button of document.querySelectorAll(".layer-button")) {
    const layer = button.dataset.layer;
    const available = Boolean(
      state.site.layers[layer] && state.site.layers[layer][state.monthIndex]
    );
    button.disabled = !available;
    button.classList.toggle("active", available && layer === state.layer);
  }

  if (!drawn) {
    note.className = "note warn";
    note.textContent =
      `No ${state.layer} image for ${state.manifest.months[state.monthIndex]}. ` +
      "The month was not observed, or that layer has not been written yet.";
  } else if (state.layer === "conf") {
    note.className = "note";
    note.textContent =
      "Confidence is masked to the classified pixels, so it covers exactly the same " +
      "ground as the prediction and the two can be compared directly.";
  } else if (state.layer === "rgb") {
    note.className = "note";
    note.textContent =
      "True colour for this same month, masked to observed pixels — holes are cloud gaps, " +
      "not dark ground. This is what the classifier actually saw.";
  } else {
    note.className = "note";
    note.textContent = "Flip to true colour (2) to check the classification against the input.";
  }
}

function togglePlay() {
  state.playing = !state.playing;
  element("play").classList.toggle("playing", state.playing);
  element("play").innerHTML = state.playing ? "&#10074;&#10074;" : "&#9654;";
  if (state.playing) step();
}

function step() {
  if (!state.playing) return;
  const next = state.monthIndex + 1;
  setMonth(next >= state.manifest.months.length ? 0 : next);
  setTimeout(step, 420);
}

/**
 * Keep the address bar in step with the view, so a particular waterhole in a
 * particular month can be sent to someone as a link. replaceState updates the
 * URL without adding a history entry — otherwise dragging the slider would bury
 * the back button under 84 of them.
 */
function syncURL() {
  const params = new URLSearchParams();
  if (state.siteId) params.set("site", state.siteId);
  params.set("month", state.manifest.months[state.monthIndex]);
  params.set("layer", state.layer);
  params.set("denominator", state.denominator);
  history.replaceState(null, "", `?${params}`);
}

// ----------------------------------------------------------------- go

boot().catch((error) => {
  document.getElementById("app").innerHTML =
    `<div class="boot-error"><h2>Could not start</h2><pre>${error.message}</pre>
     <p>If this is a fresh checkout, the data directory has to be built first — it is
     not committed. See <code>README.md</code>.</p></div>`;
  console.error(error);
});
