/*
  Everything that reads from disk.

  There is no server here. `fetch` asks the static host for a file by its path,
  exactly as typing that path into the address bar would, and gets back its
  bytes. That is the whole data layer: no queries, no API, no database. It works
  because build_dashboard_data.py already shaped the files so that the app knows
  each one's name in advance — a static host cannot list a directory, so anything
  the app cannot name, it cannot read.

  `async`/`await` is how JavaScript waits: the network takes time, and an
  `await` lets the browser keep drawing the page instead of freezing until the
  bytes arrive.
*/

// Where the data lives. The full build goes to data/, the committed sample to
// data_sample/. Rather than a config file that would have to be edited for
// local work and again for deploy, the app tries the full build and falls back.
const DATA_ROOTS = ["data", "data_sample"];

// Fetched-once caches. Stepping through months revisits the same site
// constantly, and re-parsing its JSON every time would be wasted work.
const siteCache = new Map();
const boundsCache = new Map();

let root = null;
let manifest = null;

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${path}`);
  }
  return response.json();
}

/**
 * Find the data directory and read its manifest. Must be called before anything
 * else here.
 */
export async function boot() {
  const failures = [];
  for (const candidate of DATA_ROOTS) {
    try {
      manifest = await getJSON(`${candidate}/manifest.json`);
      root = candidate;
      return { root, manifest };
    } catch (error) {
      failures.push(`${candidate}: ${error.message}`);
    }
  }
  throw new Error(
    `no dashboard data found.\n${failures.join("\n")}\n\n` +
    `Build it with:  python3 tools/build_dashboard_data.py --sites 000,001 --out data_sample`
  );
}

export function getManifest() {
  return manifest;
}

export function getRoot() {
  return root;
}

export function overviewURL() {
  return `${root}/overview.json`;
}

export const loadOverview = () => getJSON(`${root}/overview.json`);
export const loadBoxes = () => getJSON(`${root}/boxes.geojson`);
export const loadFootprints = () => getJSON(`${root}/footprints.geojson`);

/** One site's full record: identity, geometry, 84 months of composition. */
export async function loadSite(siteId) {
  if (!siteCache.has(siteId)) {
    siteCache.set(siteId, getJSON(`${root}/sites/site_${siteId}.json`));
  }
  return siteCache.get(siteId);
}

/**
 * The site's georeferencing. One file per site, not per month, because every
 * month of a site shares one pixel grid — which is also why switching layer or
 * month never moves the overlay.
 */
export async function loadBounds(siteId) {
  if (!boundsCache.has(siteId)) {
    boundsCache.set(siteId, getJSON(`${root}/overlays/site_${siteId}/bounds.json`));
  }
  return boundsCache.get(siteId);
}

/**
 * The URL of one overlay image, or null if that month/layer was never written.
 *
 * The extension comes from the site record rather than from bounds.json:
 * bounds.json reports the format the site is *being converted to*, and during
 * the PNG-to-WebP migration a site can advertise webp while some months are
 * still png. The build script probed the actual files; this trusts that.
 */
export function overlayURL(site, monthIndex, layer) {
  const extension = site.layers[layer] && site.layers[layer][monthIndex];
  if (!extension) return null;
  const month = site.months[monthIndex];
  return `${root}/overlays/site_${site.site_id}/${site.stem_prefix}_${month}_${layer}.${extension}`;
}

/**
 * Ask the browser to download images it is about to need.
 *
 * Constructing an Image and setting .src starts the fetch and fills the HTTP
 * cache; when the slider reaches that month, the swap is instant. Nothing is
 * added to the page, and the objects are discarded immediately.
 */
export function preload(site, monthIndex, layer, radius = 2) {
  for (let offset = -radius; offset <= radius; offset += 1) {
    const index = monthIndex + offset;
    if (index < 0 || index >= site.months.length || offset === 0) continue;
    const url = overlayURL(site, index, layer);
    if (url) new Image().src = url;
  }
}
