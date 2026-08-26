/*
  The legend's single source of truth.

  Every colour that means something — the six classes, the confidence ramp — is
  read from class_colours.json, which the prediction pipeline wrote when it drew
  the rasters. Hard-coding a hex value here would let the legend drift away from
  the images it describes, and a legend that disagrees with the pixels is worse
  than no legend.
*/

let classes = [];
let byName = new Map();
let confidence = null;

export function init(manifest) {
  classes = manifest.classes.filter((entry) => !entry.ignore);
  byName = new Map(classes.map((entry) => [entry.name, entry]));
  confidence = manifest.confidence;
}

export const classList = () => classes;
export const classNames = () => classes.map((entry) => entry.name);
export const colourOf = (name) => (byName.get(name) || {}).colour || "#888888";
export const confidenceRamp = () => confidence;

/** "aquatic_vegetation" -> "Aquatic vegetation" */
export function prettyName(name) {
  const text = String(name).replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * "#1f4ea1" -> "rgba(31, 78, 161, 0.7)".
 *
 * Needed because the map fills boxes with a translucent version of the class
 * colour so the imagery underneath stays visible.
 */
export function withAlpha(hex, alpha) {
  const clean = hex.replace("#", "");
  const value = parseInt(clean.slice(0, 6), 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * A CSS gradient matching the confidence PNG's colour ramp, for the legend.
 *
 * The stops come from the same cividis sampling used to draw the images, so the
 * bar under the map and the pixels on it mean the same thing.
 */
export function confidenceGradient() {
  return `linear-gradient(to right, ${confidence.stops.join(", ")})`;
}

/**
 * Sample the confidence ramp at a 0-1 value, for the sparkline in the panel.
 * Values outside [vmin, vmax] clamp, matching how the rasters were written.
 */
export function confidenceColour(value) {
  const { vmin, vmax, stops } = confidence;
  if (value == null || Number.isNaN(value)) return "#3a4453";
  const fraction = Math.min(1, Math.max(0, (value - vmin) / (vmax - vmin)));
  const index = Math.min(stops.length - 1, Math.round(fraction * (stops.length - 1)));
  return stops[index];
}

/**
 * Shading for the data-quality strip. Deliberately monochrome: quality is an
 * ordinal warning about how much evidence a month rests on, and giving it hues
 * would make it compete with the class colours, which carry the actual meaning.
 */
const QUALITY_SHADES = {
  good: "rgba(255, 255, 255, 0.06)",
  fair: "rgba(255, 180, 84, 0.28)",
  thin: "rgba(255, 180, 84, 0.55)",
  poor: "rgba(255, 107, 107, 0.65)",
};

export const qualityColour = (level) => QUALITY_SHADES[level] || "rgba(0,0,0,0)";
