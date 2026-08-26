/*
  The map: basemap, the two vector layers, and recolouring boxes as the month
  changes.

  Leaflet's job is the coordinate maths. You hand it latitudes and longitudes;
  it works out where those land on screen at the current zoom and pan, and keeps
  that true while the user drags. Nothing here converts coordinates by hand.

  Two vector layers, deliberately not baked into the overlay images:

    boxes      the hand-drawn label extent, all 187 sites, denominator for bbox_*
    footprints the derived basin outline, 176 sites, denominator for footprint_*

  They are the same for all 84 of a site's months, so drawing them into the
  images would repeat one outline across 15,708 files, destroy the pixels
  underneath, and make them impossible to switch off. As vectors they cost ~650
  KB once, stay sharp at every zoom, and can be clicked.
*/

import * as colours from "./colours.js";

// ESRI World Imagery. Free for non-commercial use with this attribution kept
// visible — which is why the attribution string is not optional decoration.
const BASEMAP_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const BASEMAP_ATTRIBUTION =
  'Imagery &copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics';

let map = null;
let basemap = null;
let boxLayer = null;
let footprintLayer = null;
let overview = null;
let siteIndex = new Map();   // site_id -> row in the overview arrays
let selectedId = null;
let onSelect = () => {};

// Style constants kept together so the visual language is easy to read off:
// selection is a bright ring, footprints are dashed, everything else is quiet.
const STYLE = {
  box: { weight: 1, color: "#ffffff", opacity: 0.55, fillOpacity: 0.55 },
  boxSelected: { weight: 2.5, color: "#4da3ff", opacity: 1, fillOpacity: 0.35 },
  boxNoData: { fillColor: "#4a5462", fillOpacity: 0.22, color: "#8a95a5", opacity: 0.4, weight: 1 },
  footprint: { weight: 1.4, color: "#7fe3c0", opacity: 0.85, fill: false, dashArray: "3 3" },
  footprintSelected: { weight: 2.2, color: "#7fe3c0", opacity: 1, fill: false, dashArray: null },
};

export function create(elementId, options) {
  overview = options.overview;
  // Built once: recolour() runs on every slider step over every box, and
  // searching a 187-entry array inside that loop would be 35,000 string
  // comparisons per frame.
  siteIndex = new Map(overview.sites.map((id, index) => [id, index]));
  onSelect = options.onSelect || onSelect;

  map = L.map(elementId, {
    zoomControl: true,
    minZoom: 4,
    maxZoom: 19,
    // The overlays are ~10 m/px; letting the user zoom past the basemap's own
    // detail is fine, because at that point the overlay is the subject.
    zoomSnap: 0.5,
  });

  basemap = L.tileLayer(BASEMAP_URL, {
    attribution: BASEMAP_ATTRIBUTION,
    maxNativeZoom: 19,
    maxZoom: 22,
  }).addTo(map);

  return map;
}

export const instance = () => map;

/**
 * Fade the basemap down.
 *
 * Worth having because the basemap is the misleading half of the picture: it is
 * a different sensor, from a different year and usually a different season, so
 * a classification laid over it can look wrong when it is right, or right when
 * it is wrong. When a site is open and its own true-colour layer is available,
 * the basemap has little left to contribute.
 */
export function dimBasemap(dim) {
  if (basemap) basemap.setOpacity(dim ? 0.25 : 1);
}

export function addVectors(boxes, footprints) {
  boxLayer = L.geoJSON(boxes, {
    style: () => STYLE.box,
    onEachFeature: (feature, layer) => {
      const { site_id: siteId, label } = feature.properties;
      layer.bindTooltip(`site ${siteId} &middot; ${label}`, { sticky: true, direction: "top" });
      layer.on("click", () => onSelect(siteId));
    },
  }).addTo(map);

  footprintLayer = L.geoJSON(footprints, {
    style: () => STYLE.footprint,
    onEachFeature: (feature, layer) => {
      layer.on("click", () => onSelect(feature.properties.site_id));
    },
  }).addTo(map);

  map.fitBounds(boxLayer.getBounds(), { padding: [40, 40] });
}

export function setLayerVisible(which, visible) {
  const layer = which === "boxes" ? boxLayer : footprintLayer;
  if (!layer) return;
  if (visible) layer.addTo(map);
  else map.removeLayer(layer);
}

/**
 * Recolour every box for one month.
 *
 * This is the view no single-site plot gives: the whole population drying and
 * refilling at once. It runs on every slider step, so it reads from
 * overview.json's flat integer arrays rather than fetching or parsing anything.
 */
export function recolour(monthIndex, mode) {
  if (!boxLayer || !overview) return;

  boxLayer.eachLayer((layer) => {
    const siteId = layer.feature.properties.site_id;
    const row = siteIndex.get(siteId);
    if (row === undefined) return;

    const classified = overview.n_classified[row][monthIndex];
    const isSelected = siteId === selectedId;

    // An unobserved month is not a composition of zero — it is an absence, and
    // must not be coloured as though the ground were measured and found empty.
    if (!classified) {
      layer.setStyle({
        ...STYLE.boxNoData,
        ...(isSelected ? { color: STYLE.boxSelected.color, weight: STYLE.boxSelected.weight } : {}),
      });
      return;
    }

    let fillColour;
    if (mode === "dominant") {
      const dominant = overview.dominant[row][monthIndex];
      fillColour = colours.colourOf(overview.classes[dominant]);
    } else {
      // Colouring by one class's fraction: the class's own colour, with the
      // fraction as opacity. Same hue as the legend, so it stays readable.
      const permille = overview.frac[mode][row][monthIndex];
      const fraction = permille < 0 ? 0 : permille / overview.scale;
      fillColour = colours.withAlpha(colours.colourOf(mode), Math.max(0.08, fraction));
    }

    layer.setStyle({
      ...(isSelected ? STYLE.boxSelected : STYLE.box),
      fillColor: fillColour,
    });
    if (isSelected) layer.bringToFront();
  });
}

export function setSelected(siteId, monthIndex, mode) {
  selectedId = siteId;
  recolour(monthIndex, mode);

  if (footprintLayer) {
    footprintLayer.eachLayer((layer) => {
      const selected = layer.feature.properties.site_id === siteId;
      layer.setStyle(selected ? STYLE.footprintSelected : STYLE.footprint);
      if (selected) layer.bringToFront();
    });
  }
}

/**
 * Emphasise whichever region the numbers on screen are counted within.
 *
 * The brief is firm about this: the region *is* what the number means. A site
 * whose box holds two basins, or whose footprint has drifted off the water,
 * gives a series that is internally consistent and about the wrong ground — and
 * the only way to notice is to see the outline sitting on the imagery.
 */
export function highlightDenominator(denominator) {
  if (boxLayer) {
    boxLayer.eachLayer((layer) => {
      if (layer.feature.properties.site_id !== selectedId) return;
      layer.setStyle({ dashArray: denominator === "bbox" ? null : "4 4" });
    });
  }
  if (footprintLayer) {
    footprintLayer.eachLayer((layer) => {
      if (layer.feature.properties.site_id !== selectedId) return;
      layer.setStyle({
        weight: denominator === "footprint" ? 3 : 1.6,
        dashArray: denominator === "footprint" ? null : "3 3",
      });
    });
  }
}

/** Frame one site, leaving room for its surroundings so context stays visible. */
export function flyToSite(bounds) {
  map.flyToBounds(bounds, { padding: [90, 90], duration: 0.6, maxZoom: 17 });
}
