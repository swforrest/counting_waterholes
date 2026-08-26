/*
  The site panel: everything known about one waterhole.

  Its job is not only to show the composition but to keep the caveats attached
  to it. The model was trained on 27 sites and applied to 187, wet-season months
  rest on a third of the evidence dry-season months do, and an isolated wet month
  is more often a compositing artefact than a rainfall event. A panel showing
  only the fractions would be more confident than the data supports, so the
  quality strip, the confidence series and the flag callouts are not optional
  extras here — they ship alongside the numbers, and the trend comes last.
*/

import * as colours from "./colours.js";
import * as charts from "./charts.js";

const format = {
  percent: (value) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`),
  fixed: (value, digits = 2) => (value == null ? "—" : value.toFixed(digits)),
};

let container = null;
let current = null;   // { site, denominator, chart }

export function attach(element) {
  container = element;
}

export function clear(message) {
  current = null;
  container.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder";
  placeholder.innerHTML = `
    <h2>Pick a waterhole</h2>
    <p>Click a box on the map to see its per-pixel predictions month by month,
       and how its surface composition has changed across the record.</p>`;
  if (message) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = message;
    placeholder.appendChild(note);
  }
  container.appendChild(placeholder);
}

function section(title, hint) {
  const node = document.createElement("section");
  node.className = "section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  if (hint) {
    const span = document.createElement("span");
    span.className = "hint";
    span.textContent = hint;
    heading.appendChild(span);
  }
  node.appendChild(heading);
  return node;
}

/** Build the whole panel for one site. Called on selection, not on every month. */
export function render(site, state, handlers) {
  current = { site, denominator: state.denominator };
  container.innerHTML = "";

  const manifest = state.manifest;
  const hasFootprint = site.has_footprint;
  const withoutFootprint = manifest.sites_without_footprint.includes(site.site_id);

  // ---- heading ----------------------------------------------------------
  const head = document.createElement("div");
  head.className = "site-head";
  head.innerHTML = `
    <div>
      <h2>site ${site.site_id}</h2>
      <div class="label">${site.label} &middot; ${format.fixed(site.bbox_width_m, 0)} &times; ${format.fixed(site.bbox_height_m, 0)} m</div>
    </div>`;
  const close = document.createElement("button");
  close.className = "close-button";
  close.title = "Close (Esc)";
  close.textContent = "×";
  close.addEventListener("click", handlers.onClose);
  head.appendChild(close);
  container.appendChild(head);

  // ---- badges -----------------------------------------------------------
  const badges = document.createElement("div");
  badges.className = "badges";

  const observed = site.mean_confidence.filter((v) => v != null);
  const meanConfidence = observed.length
    ? observed.reduce((sum, v) => sum + v, 0) / observed.length
    : null;

  if (meanConfidence != null) {
    const badge = document.createElement("span");
    // The threshold is a warning, not a verdict. Across 6.5 M sampled pixels
    // the 5th percentile is 0.82, so a site averaging below ~0.9 is unusual
    // enough to look at, and gradient boosting's saturation means the absolute
    // number should not be read as a probability.
    badge.className = meanConfidence < 0.9 ? "badge warn" : "badge";
    badge.innerHTML = `mean confidence <strong>${format.fixed(meanConfidence, 3)}</strong>`;
    badges.appendChild(badge);
  }

  const footprintBadge = document.createElement("span");
  footprintBadge.className = hasFootprint ? "badge" : "badge warn";
  footprintBadge.innerHTML = hasFootprint
    ? `basin <strong>${format.fixed(site.footprint_area_ha, 2)} ha</strong>`
    : "no basin footprint";
  badges.appendChild(footprintBadge);

  const flagged = site.flag_isolated_wet.filter(Boolean).length;
  if (flagged) {
    const badge = document.createElement("span");
    badge.className = "badge flag";
    badge.innerHTML = `<strong>${flagged}</strong> flagged month${flagged > 1 ? "s" : ""}`;
    badges.appendChild(badge);
  }

  const unobserved = site.bbox_n_classified.filter((n) => !n).length;
  if (unobserved) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.innerHTML = `<strong>${unobserved}</strong> month${unobserved > 1 ? "s" : ""} unobserved`;
    badges.appendChild(badge);
  }

  container.appendChild(badges);

  if (withoutFootprint) {
    const callout = document.createElement("div");
    callout.className = "callout";
    callout.textContent =
      "Footprint estimation did not converge for this site, so only the bounding-box " +
      "denominator is available. The box is the hand-drawn label extent and may take in " +
      "ground beyond the basin.";
    container.appendChild(callout);
  }

  // ---- composition this month ------------------------------------------
  const now = section("Composition this month", "");
  const bars = document.createElement("div");
  bars.className = "composition-now";
  now.appendChild(bars);
  container.appendChild(now);

  // ---- composition through time ----------------------------------------
  const denominatorLabel = state.denominator === "bbox" ? "bounding box" : "basin footprint";
  const history = section("Composition through time", `within the ${denominatorLabel}`);
  const chart = charts.compositionChart(
    site, state.denominator, state.monthIndex, handlers.onScrub
  );
  history.appendChild(chart);

  const key = document.createElement("div");
  key.className = "note";
  key.style.marginTop = "6px";
  key.innerHTML =
    "The strip under the axis shades each month by data quality — darker amber is thinner " +
    "evidence. Red triangles mark months flagged as isolated-wet; dots mark months with no " +
    "clear observation at all.";
  history.appendChild(key);
  container.appendChild(history);

  // ---- confidence -------------------------------------------------------
  if (observed.length) {
    const confidence = section("Mean confidence", "0.7 – 1.0");
    confidence.appendChild(charts.confidenceChart(site));
    const note = document.createElement("div");
    note.className = "note";
    note.style.marginTop = "6px";
    note.textContent =
      "The classifier is poorly calibrated and saturates: most pixels sit above 0.9 " +
      "whether or not the site resembles anything it was trained on. Read this as " +
      "relative, not as a probability.";
    confidence.appendChild(note);
    container.appendChild(confidence);
  }

  // ---- trend ------------------------------------------------------------
  const trends = charts.dryseasonTrends(site, state.denominator);
  const trendSection = section("Dry-season trend", "Jun–Sep means, % per year");
  const table = document.createElement("table");
  table.className = "trend";
  table.innerHTML = `
    <thead><tr><th>Class</th><th style="text-align:right">Slope</th>
    <th style="text-align:right">&plusmn;1 s.e.</th></tr></thead>`;
  const body = document.createElement("tbody");

  for (const name of colours.classNames()) {
    const result = trends[name];
    const row = document.createElement("tr");
    const swatch = `<span class="swatch" style="background:${colours.colourOf(name)}"></span>`;
    row.innerHTML = `
      <td><span class="cls">${swatch}${colours.prettyName(name)}</span></td>
      <td class="num">${result.slope == null ? "—" : (result.slope * 100).toFixed(2)}</td>
      <td class="num">${result.stderr == null ? "—" : (result.stderr * 100).toFixed(2)}</td>`;
    body.appendChild(row);
  }
  table.appendChild(body);
  trendSection.appendChild(table);

  const caveat = document.createElement("div");
  caveat.className = "callout";
  const yearCount = (trends._years || []).length;
  caveat.innerHTML =
    `Fitted on ${yearCount} annual dry-season means, not on the ${site.months.length} monthly ` +
    "values. Monthly fractions are strongly seasonal and autocorrelated, so a fit across all " +
    "of them would report an interval several times too narrow. Even here the standard error " +
    "rests on a handful of points — treat a slope smaller than its own s.e. as no trend.";
  trendSection.appendChild(caveat);
  container.appendChild(trendSection);

  // ---- provenance -------------------------------------------------------
  const provenance = section("Provenance", "");
  const list = document.createElement("dl");
  list.className = "stat-grid";
  list.innerHTML = `
    <dt>Model</dt><dd>${site.model.name}</dd>
    <dt>CV macro F1</dt><dd>${format.fixed(site.model.cv_macro_f1, 3)}</dd>
    <dt>Box pixels</dt><dd>${site.n_pixels_bbox ?? "—"}</dd>
    <dt>Basin pixels</dt><dd>${site.n_pixels_footprint ?? "—"}</dd>
    <dt>Centre</dt><dd>${site.center[1].toFixed(4)}, ${site.center[0].toFixed(4)}</dd>
    <dt>Predicted</dt><dd>${(site.model.predicted_at || "").slice(0, 10)}</dd>`;
  provenance.appendChild(list);

  const trained = document.createElement("div");
  trained.className = "callout";
  trained.textContent =
    "The classifier was trained on hand-painted labels from 27 sites and applied to all 187. " +
    "A waterhole unlike anything labelled still receives a confident-looking classification, " +
    "so check the prediction against the true-colour layer before trusting a site you have " +
    "not seen before.";
  provenance.appendChild(trained);
  container.appendChild(provenance);

  current.chart = chart;
  current.bars = bars;
  updateMonth(site, state);
}

/**
 * Update only what depends on the month.
 *
 * Rebuilding the whole panel on every slider step would throw away and re-make
 * several hundred SVG nodes 84 times a drag; this touches the cursor line and
 * six bars instead.
 */
export function updateMonth(site, state) {
  if (!current || current.site.site_id !== site.site_id) return;

  const index = state.monthIndex;
  const denominator = state.denominator;
  const classified = site[`${denominator}_n_classified`][index];
  const fractions = site[`${denominator}_frac`];

  if (current.chart && current.chart.updateCursor) current.chart.updateCursor(index);

  const bars = current.bars;
  bars.innerHTML = "";

  if (!classified) {
    const note = document.createElement("div");
    note.className = "note warn";
    note.textContent =
      `No clear observation for ${site.months[index]} within the ` +
      `${denominator === "bbox" ? "bounding box" : "basin footprint"} — nothing was ` +
      "classified, so this month has no composition. It is left as a gap rather than " +
      "interpolated.";
    bars.appendChild(note);
    return;
  }

  for (const name of colours.classNames()) {
    const value = fractions[name][index] || 0;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-track">
        <div class="bar-fill" style="width:${(value * 100).toFixed(1)}%;background:${colours.colourOf(name)}"></div>
        <span class="bar-name">${colours.prettyName(name)}</span>
      </div>
      <span class="bar-value">${format.percent(value)}</span>`;
    bars.appendChild(row);
  }

  const meta = document.createElement("div");
  meta.className = "note";
  meta.style.marginTop = "6px";
  const quality = site.data_quality[index];
  const gap = site.gap_fraction[index];
  const observations = site.mean_n_obs[index];
  const confidence = site.mean_confidence[index];
  meta.innerHTML =
    `${classified} px classified &middot; quality <strong>${quality}</strong> ` +
    `&middot; ${observations == null ? "no" : format.fixed(observations, 1)} clear scenes ` +
    `&middot; gap ${format.percent(gap)}` +
    (confidence == null ? "" : ` &middot; confidence ${format.fixed(confidence, 3)}`);
  bars.appendChild(meta);

  if (site.flag_isolated_wet[index]) {
    const callout = document.createElement("div");
    callout.className = "callout flagged";
    callout.textContent =
      "Flagged as an isolated wet month: wet between two dry months, on a thin median. " +
      "That is more often a compositing artefact than a rainfall event — but it may be the " +
      "event. It is marked, not removed, and deciding is yours.";
    bars.appendChild(callout);
  }
}

export const currentSiteId = () => (current ? current.site.site_id : null);
