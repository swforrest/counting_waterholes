/*
  Charts, drawn as SVG by hand.

  No charting library. Two reasons: the smallest capable one is larger than the
  entire rest of this app, and what is drawn here is specific enough — fixed
  class colours, a quality strip sharing the composition's time axis, flags that
  must not be smoothed away — that configuring a library would be more code than
  the ~200 lines below, with less control.

  SVG is drawn in its own coordinate space: we compute a viewBox in "chart
  units" and the browser scales it to whatever width the panel happens to be.
  So there are no pixel measurements here and the charts stay sharp at any size.

  The time axis is *month index*, not date. Every site has the same 84 months in
  the same order, gaps included, so position is enough — and an unobserved month
  keeps its slot instead of being closed up, which is the point.
*/

import * as colours from "./colours.js";

const NS = "http://www.w3.org/2000/svg";

// One shared geometry for the composition chart and the quality strip beneath
// it, so the two line up exactly along the time axis.
const W = 400;
const PLOT_H = 118;
const STRIP_H = 9;
const PAD_L = 26;
const PAD_R = 6;
const PAD_T = 10;
const MARK_H = 9;      // room above the plot for flags and no-data marks

function element(name, attributes = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  return node;
}

const xOf = (index, count) =>
  PAD_L + (count <= 1 ? 0 : (index / (count - 1)) * (W - PAD_L - PAD_R));

/**
 * Stacked area of the class fractions through time, with the quality strip on
 * the same axis and flags marked above.
 *
 * Mirrors wh_plots.plot_site_composition so the notebook and the dashboard
 * cannot disagree about what a site's history looks like.
 */
export function compositionChart(site, denominator, monthIndex, onScrub) {
  const names = colours.classNames();
  const months = site.months;
  const count = months.length;
  const classified = site[`${denominator}_n_classified`];
  const fractions = site[`${denominator}_frac`];

  const totalHeight = PAD_T + MARK_H + PLOT_H + STRIP_H + 14;
  const svg = element("svg", {
    class: "chart",
    viewBox: `0 0 ${W} ${totalHeight}`,
    preserveAspectRatio: "none",
  });

  const top = PAD_T + MARK_H;
  const yOf = (cumulative) => top + PLOT_H * (1 - cumulative);

  // Stack the classes in legend order. Each band is drawn as a closed polygon
  // running left along its own upper edge and back right along the one below.
  const running = new Array(count).fill(0);
  for (const name of names) {
    const lower = running.slice();
    const upper = running.map((base, index) => {
      const value = classified[index] ? (fractions[name][index] || 0) : 0;
      return base + value;
    });

    // Break the band at unobserved months so a gap reads as a gap rather than
    // as a straight line implying measurements that were never made.
    let segment = [];
    const segments = [];
    for (let index = 0; index < count; index += 1) {
      if (classified[index]) segment.push(index);
      else if (segment.length) { segments.push(segment); segment = []; }
    }
    if (segment.length) segments.push(segment);

    for (const indices of segments) {
      const forward = indices.map((i) => `${xOf(i, count)},${yOf(upper[i])}`);
      const back = indices.slice().reverse().map((i) => `${xOf(i, count)},${yOf(lower[i])}`);
      svg.appendChild(element("polygon", {
        points: forward.concat(back).join(" "),
        fill: colours.colourOf(name),
        "shape-rendering": "geometricPrecision",
      }));
    }
    running.splice(0, count, ...upper);
  }

  // Axis furniture: 0/50/100% guides and year ticks.
  for (const level of [0, 0.5, 1]) {
    svg.appendChild(element("line", {
      class: "axis-line", x1: PAD_L, x2: W - PAD_R,
      y1: yOf(level), y2: yOf(level),
      opacity: level === 0 ? 1 : 0.35,
    }));
    const text = element("text", { x: PAD_L - 4, y: yOf(level) + 3, "text-anchor": "end" });
    text.textContent = `${level * 100}`;
    svg.appendChild(text);
  }

  months.forEach((month, index) => {
    if (!month.endsWith("-01")) return;
    const x = xOf(index, count);
    svg.appendChild(element("line", {
      class: "axis-line", x1: x, x2: x,
      y1: top + PLOT_H, y2: top + PLOT_H + 3, opacity: 0.6,
    }));
    const text = element("text", { x, y: totalHeight - 3, "text-anchor": "middle" });
    text.textContent = month.slice(0, 4);
    svg.appendChild(text);
  });

  // The quality strip, directly under the composition and on the same axis —
  // not behind a toggle. Wet-season months rest on ~2 clear scenes against ~6
  // in the dry, which is exactly backwards for a study of water, and that has
  // to be visible next to the numbers it qualifies.
  const stripY = top + PLOT_H + 1;
  const bandWidth = (W - PAD_L - PAD_R) / count;
  site.data_quality.forEach((level, index) => {
    svg.appendChild(element("rect", {
      x: xOf(index, count) - bandWidth / 2,
      y: stripY, width: bandWidth + 0.5, height: STRIP_H,
      fill: classified[index] ? colours.qualityColour(level) : "rgba(255,255,255,0.02)",
    }));
  });

  // Flagged and unobserved months, marked above the plot.
  site.flag_isolated_wet.forEach((flagged, index) => {
    if (!flagged) return;
    const x = xOf(index, count);
    svg.appendChild(element("path", {
      class: "flag-mark",
      d: `M ${x} ${PAD_T + 1} l 3.4 6 h -6.8 z`,
    }));
  });
  classified.forEach((n, index) => {
    if (n) return;
    svg.appendChild(element("circle", {
      class: "nodata-mark", cx: xOf(index, count), cy: PAD_T + 4, r: 1.6, opacity: 0.7,
    }));
  });

  // The month cursor, tying the chart to whatever the map is showing.
  const cursor = element("line", {
    class: "cursor-line",
    x1: xOf(monthIndex, count), x2: xOf(monthIndex, count),
    y1: PAD_T, y2: stripY + STRIP_H,
  });
  svg.appendChild(cursor);

  // Scrubbing: drag anywhere on the chart to move the map through time.
  const hit = element("rect", {
    x: PAD_L, y: PAD_T, width: W - PAD_L - PAD_R, height: PLOT_H + MARK_H + STRIP_H,
    fill: "transparent", style: "cursor: col-resize",
  });
  const indexFromEvent = (event) => {
    const box = svg.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    const unit = (fraction * W - PAD_L) / (W - PAD_L - PAD_R);
    return Math.max(0, Math.min(count - 1, Math.round(unit * (count - 1))));
  };
  let dragging = false;
  hit.addEventListener("pointerdown", (event) => {
    dragging = true;
    hit.setPointerCapture(event.pointerId);
    onScrub(indexFromEvent(event));
  });
  hit.addEventListener("pointermove", (event) => {
    if (dragging) onScrub(indexFromEvent(event));
  });
  hit.addEventListener("pointerup", (event) => {
    dragging = false;
    hit.releasePointerCapture(event.pointerId);
  });
  svg.appendChild(hit);

  svg.updateCursor = (index) => {
    cursor.setAttribute("x1", xOf(index, count));
    cursor.setAttribute("x2", xOf(index, count));
  };
  return svg;
}

/**
 * Mean confidence through time, as a line.
 *
 * A bar chart is wrong here and it is worth saying why: the classifier is
 * saturated — 93% of pixels sit above 0.9 confidence — so bars from any sensible
 * baseline are all nearly full height and the chart becomes a solid block that
 * says nothing. The information is entirely in the dips, so a line against a
 * 0.7-1.0 axis, with months below 0.9 picked out, is what actually shows them.
 *
 * The axis is deliberately not 0-1: stretching a saturated series across the
 * full range would flatten it into a straight line at the top.
 */
export function confidenceChart(site) {
  const count = site.months.length;
  const height = 42;
  const svg = element("svg", {
    class: "chart", viewBox: `0 0 ${W} ${height}`, preserveAspectRatio: "none",
  });

  const low = 0.7;
  const top = 5;
  const bottom = height - 9;
  const yOf = (value) =>
    bottom - (bottom - top) * Math.min(1, Math.max(0, (value - low) / (1 - low)));

  for (const level of [0.8, 0.9]) {
    svg.appendChild(element("line", {
      class: "axis-line", x1: PAD_L, x2: W - PAD_R,
      y1: yOf(level), y2: yOf(level), opacity: 0.55, "stroke-dasharray": "2 3",
    }));
    const text = element("text", { x: PAD_L - 4, y: yOf(level) + 3, "text-anchor": "end" });
    text.textContent = level.toFixed(1);
    svg.appendChild(text);
  }

  // Break the line at unobserved months, for the same reason the composition
  // bands break: a straight run across a gap asserts measurements that do not
  // exist.
  let run = [];
  const runs = [];
  site.mean_confidence.forEach((value, index) => {
    if (value == null) { if (run.length) { runs.push(run); run = []; } return; }
    run.push(index);
  });
  if (run.length) runs.push(run);

  for (const indices of runs) {
    if (indices.length === 1) {
      const index = indices[0];
      svg.appendChild(element("circle", {
        cx: xOf(index, count), cy: yOf(site.mean_confidence[index]), r: 1.2,
        fill: colours.confidenceColour(site.mean_confidence[index]),
      }));
      continue;
    }
    svg.appendChild(element("polyline", {
      points: indices
        .map((i) => `${xOf(i, count)},${yOf(site.mean_confidence[i])}`)
        .join(" "),
      fill: "none",
      stroke: colours.confidenceColour(0.97),
      "stroke-width": 1.2,
      "stroke-linejoin": "round",
    }));
  }

  // Months the model was least sure about. These are the ones worth opening the
  // confidence overlay on.
  site.mean_confidence.forEach((value, index) => {
    if (value == null || value >= 0.9) return;
    svg.appendChild(element("circle", {
      cx: xOf(index, count), cy: yOf(value), r: 2, fill: "#ff6b6b",
    }));
  });

  return svg;
}

/**
 * Trend per class, fitted on annual dry-season means.
 *
 * Deliberately not OLS on the monthly series. Seven years of monthly fractions
 * with strong seasonality are nothing like 84 independent observations: the
 * residuals are heavily autocorrelated, and a naive fit reports an interval
 * several times too narrow. Restricting to one dry-season mean per year gives
 * ~7 roughly independent points, and comparing like season with like season
 * removes the seasonal cycle rather than averaging over it.
 *
 * Returns per class: slope in fraction-per-year, its standard error, and the
 * number of years actually used.
 */
export function dryseasonTrends(site, denominator, dryMonths = [6, 7, 8, 9]) {
  const names = colours.classNames();
  const classified = site[`${denominator}_n_classified`];
  const fractions = site[`${denominator}_frac`];

  const byYear = new Map();
  site.months.forEach((month, index) => {
    const [year, monthNumber] = month.split("-").map(Number);
    if (!dryMonths.includes(monthNumber)) return;
    if (!classified[index]) return;
    if (!byYear.has(year)) byYear.set(year, []);
    byYear.get(year).push(index);
  });

  const years = [...byYear.keys()].sort();
  const results = {};

  for (const name of names) {
    const points = years.map((year) => {
      const indices = byYear.get(year);
      const mean = indices.reduce((sum, i) => sum + (fractions[name][i] || 0), 0) / indices.length;
      return [year, mean];
    });

    if (points.length < 4) {
      results[name] = { slope: null, stderr: null, years: points.length };
      continue;
    }

    const n = points.length;
    const meanX = points.reduce((sum, p) => sum + p[0], 0) / n;
    const meanY = points.reduce((sum, p) => sum + p[1], 0) / n;
    let sxx = 0;
    let sxy = 0;
    for (const [x, y] of points) {
      sxx += (x - meanX) ** 2;
      sxy += (x - meanX) * (y - meanY);
    }
    const slope = sxx === 0 ? 0 : sxy / sxx;
    const intercept = meanY - slope * meanX;

    let residual = 0;
    for (const [x, y] of points) residual += (y - (intercept + slope * x)) ** 2;
    const variance = n > 2 ? residual / (n - 2) : 0;
    const stderr = sxx === 0 ? null : Math.sqrt(variance / sxx);

    results[name] = { slope, stderr, years: n, points, intercept };
  }

  results._years = years;
  return results;
}
