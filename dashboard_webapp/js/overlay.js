/*
  Placing one month's prediction on the map.

  This is the step the whole no-backend design rests on. A browser cannot draw a
  GeoTIFF: it has no idea what a CRS is, and reading one would need a tile server
  or a WASM decoder. But every mapping library can stretch an ordinary image
  between two corners — and that is exactly what the pipeline's bounds.json
  provides, as `leaflet_bounds` in the [[south, west], [north, east]] form
  L.imageOverlay expects.

  So the GeoTIFF stays the authoritative product for analysis, and the WebP is
  purely for display. No server, no tiles, no reprojection.

  Because all three layers of all 84 months share one grid and one bounds.json,
  changing month or layer is a source swap on a single overlay object. Nothing
  moves; nothing is recomputed.
*/

import { overlayURL, preload } from "./data.js";

let map = null;
let overlay = null;
let bounds = null;
let opacity = 0.85;

export function attach(mapInstance) {
  map = mapInstance;
}

/** Point the overlay at a new site, creating it on first use. */
export function show(site, monthIndex, layer, siteBounds) {
  bounds = siteBounds.leaflet_bounds;
  const url = overlayURL(site, monthIndex, layer);

  if (!url) {
    hide();
    return false;
  }

  if (!overlay) {
    overlay = L.imageOverlay(url, bounds, {
      opacity,
      interactive: false,
      className: `overlay-${layer}`,
      // Above the basemap, below the vector outlines, so a box or footprint is
      // never hidden by the thing it is supposed to delimit.
      zIndex: 350,
    }).addTo(map);
  } else {
    overlay.setBounds(bounds);
    // setUrl swaps the image in place. Leaflet keeps the old one visible until
    // the new one decodes, so stepping months does not flash.
    overlay.setUrl(url);
    setLayerClass(layer);
    if (!map.hasLayer(overlay)) overlay.addTo(map);
  }

  preload(site, monthIndex, layer);
  return true;
}

/**
 * Swap the CSS class so `image-rendering: pixelated` applies to the class and
 * confidence layers but not to true colour.
 *
 * Smoothing a class raster invents colours between the discrete ones — a pixel
 * that reads as halfway between "mud" and "open water" is not in the legend and
 * was never in the model's output. Smoothing a photograph is fine.
 */
function setLayerClass(layer) {
  const element = overlay.getElement();
  if (!element) return;
  // classList, not className: Leaflet puts its own classes on this element
  // (leaflet-image-layer, leaflet-zoom-animated) and overwriting the lot would
  // silently break zoom animation.
  element.classList.remove("overlay-pred", "overlay-rgb", "overlay-conf");
  element.classList.add(`overlay-${layer}`);
}

export function hide() {
  if (overlay && map.hasLayer(overlay)) map.removeLayer(overlay);
}

export function remove() {
  if (overlay) {
    map.removeLayer(overlay);
    overlay = null;
    bounds = null;
  }
}

export function setOpacity(value) {
  opacity = value;
  if (overlay) overlay.setOpacity(value);
}

export const getBounds = () => bounds;
