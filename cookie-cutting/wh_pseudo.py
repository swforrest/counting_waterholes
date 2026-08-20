"""Generate high-confidence labels automatically for the easy classes.

The point is to spend hand-labelling effort where judgement is actually needed —
aquatic vegetation, wet mud, pugged margin — and let a threshold handle the cases
that are not ambiguous: unmistakable open water, and savanna matrix well away
from the basin.

Two things to be clear about.

**This is deliberately blind to vegetated water.** The open-water rule is an
MNDWI threshold, and sedges over standing water push MNDWI far negative. That is
the exact failure the hand labels exist to cover, so the rule must stay
conservative rather than being widened until it "finds" the hard cases wrongly.

**It buys site coverage, not volume.** Open water is genuinely rare at 10 m in
these waterholes: sampling 120 wet-season tiles across 12 sites yields ~170
pixels even at a permissive threshold. What it does buy is open_water appearing
at 6 sites instead of 2, and with grouped-by-site cross-validation the number of
sites is the thing that limits what can be learned.

Masks are written to their own directory and tagged `source: "pseudo"` in the
sidecar, so they can be included or excluded at will and can never overwrite
hand-painted work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import wh_footprint
import wh_indices
import wh_naming
import wh_tiles
from wh_config import Config


@dataclass
class PseudoParams:
    """Thresholds for automatic labelling. See the module docstring for calibration."""

    min_obs: int = 2
    open_water_mndwi_min: float = 0.15
    open_water_ndvi_max: float = 0.10
    # Off by default: 39% of water patches here are 1-2 pixels, so eroding
    # deletes 35% of the water and two whole sites. The MNDWI threshold already
    # excludes mixed waterline pixels, which sit near 0.
    open_water_erode_px: int = 0

    vegetation_buffer_px: int = 20
    vegetation_ndvi_min: float = 0.35
    vegetation_ndvi_percentile: float = 60.0

    max_pixels_per_class_per_tile: int = 300
    random_state: int = 42

    @classmethod
    def from_config(cls, cfg: Config) -> "PseudoParams":
        settings = cfg["pseudo_labels"]
        water = settings["open_water"]
        vegetation = settings["surrounding_vegetation"]
        return cls(
            min_obs=int(settings["min_obs"]),
            open_water_mndwi_min=float(water["mndwi_min"]),
            open_water_ndvi_max=float(water["ndvi_max"]),
            open_water_erode_px=int(water["erode_px"]),
            vegetation_buffer_px=int(vegetation["outside_footprint_buffer_px"]),
            vegetation_ndvi_min=float(vegetation["ndvi_min"]),
            vegetation_ndvi_percentile=float(vegetation.get("ndvi_percentile", 60.0)),
            max_pixels_per_class_per_tile=int(settings["max_pixels_per_class_per_tile"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_obs": self.min_obs,
            "open_water_mndwi_min": self.open_water_mndwi_min,
            "open_water_ndvi_max": self.open_water_ndvi_max,
            "open_water_erode_px": self.open_water_erode_px,
            "vegetation_buffer_px": self.vegetation_buffer_px,
            "vegetation_ndvi_min": self.vegetation_ndvi_min,
            "vegetation_ndvi_percentile": self.vegetation_ndvi_percentile,
            "max_pixels_per_class_per_tile": self.max_pixels_per_class_per_tile,
        }


def pseudo_label_dir(cfg: Config) -> Path:
    """Separate from the hand labels, so the two can never collide."""
    return cfg.paths["labels"].parent / "labels_pseudo"


def open_water_mask(tile: wh_tiles.Tile, params: PseudoParams) -> np.ndarray:
    """Unambiguous open water: high MNDWI, low NDVI, enough observations.

    Eroded afterwards so the mixed pixels around the waterline are dropped — a
    half-water pixel is exactly the ambiguity that hand labelling exists for.
    """
    mndwi = wh_indices.compute("mndwi", tile.bands)
    ndvi = wh_indices.compute("ndvi", tile.bands)

    mask = (
        (mndwi > params.open_water_mndwi_min)
        & (ndvi < params.open_water_ndvi_max)
        & tile.valid
    )
    mask &= _enough_observations(tile, params)
    mask = np.nan_to_num(mask, nan=False).astype(bool)

    if params.open_water_erode_px > 0 and mask.any():
        mask = ndimage.binary_erosion(
            mask, structure=_disk(params.open_water_erode_px), border_value=0
        )
    return mask


def surrounding_vegetation_mask(
    tile: wh_tiles.Tile,
    footprint: np.ndarray | None,
    params: PseudoParams,
    neighbour_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Savanna matrix: greener than most of the tile, well away from any basin.

    Returns (mask, reason). Without a footprint there is no defensible way to say
    "outside the basin", so nothing is generated and the reason says why — a
    guess here would put basin pixels into the majority class and quietly teach
    the classifier the opposite of what we want.

    `neighbour_mask` is the union of OTHER waterholes' extents in the same tile,
    from wh_bbox. 93 of 187 tiles contain a second waterhole, and the footprint
    only covers this site's basin — so without this, a neighbouring waterhole
    would be claimed as savanna matrix, which is precisely the mistake the class
    exists to avoid.

    The NDVI cut is a percentile of the tile's own distribution rather than a
    fixed value, because dry-season savanna NDVI runs ~0.30-0.45 and any fixed
    threshold high enough for the wet season yields nothing in the dry.
    """
    if footprint is None:
        return np.zeros(tile.shape, dtype=bool), "no basin footprint for this site"
    if footprint.shape != tile.shape:
        return (
            np.zeros(tile.shape, dtype=bool),
            f"footprint shape {footprint.shape} does not match tile {tile.shape}",
        )

    far_from_basin = (
        ndimage.distance_transform_edt(~footprint) > params.vegetation_buffer_px
        if footprint.any()
        else np.ones(tile.shape, dtype=bool)
    )

    ndvi = wh_indices.compute("ndvi", tile.bands)
    observed = tile.valid & np.isfinite(ndvi)
    if not observed.any():
        return np.zeros(tile.shape, dtype=bool), "tile has no observed pixels"

    cutoff = max(
        float(np.percentile(ndvi[observed], params.vegetation_ndvi_percentile)),
        params.vegetation_ndvi_min,
    )

    mask = far_from_basin & observed & (ndvi > cutoff) & _enough_observations(tile, params)

    if neighbour_mask is not None:
        if neighbour_mask.shape != tile.shape:
            raise ValueError(
                f"neighbour mask {neighbour_mask.shape} does not match tile {tile.shape}"
            )
        mask &= ~neighbour_mask

    return np.nan_to_num(mask, nan=False).astype(bool), ""


def _enough_observations(tile: wh_tiles.Tile, params: PseudoParams) -> np.ndarray:
    """Pixels whose monthly median rests on enough clear scenes to be trusted."""
    if tile.n_obs is None:
        return np.ones(tile.shape, dtype=bool)
    return tile.n_obs >= params.min_obs


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    span = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    return (yy**2 + xx**2) <= radius**2


def _subsample(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    """Cap a class's contribution from one tile, keeping a random subset.

    Without this a single wet tile could contribute tens of thousands of matrix
    pixels and swamp every hand label in the table.
    """
    count = int(mask.sum())
    if count <= limit:
        return mask

    rows, cols = np.nonzero(mask)
    keep = rng.choice(count, size=limit, replace=False)
    trimmed = np.zeros_like(mask)
    trimmed[rows[keep], cols[keep]] = True
    return trimmed


def evaluate_tile(
    tile_path: str | Path,
    cfg: Config,
    params: PseudoParams,
    footprint: np.ndarray | None,
    rng: np.random.Generator | None = None,
    neighbours: np.ndarray | None = None,
) -> dict[str, object]:
    """Count what would be generated for one tile, without writing anything.

    Reports counts AFTER the per-tile cap, so the yield table is what would
    actually land on disk rather than an uncapped number several times larger.
    Used by the notebook to show yields before committing, which matters here:
    the originally configured thresholds produced almost nothing.
    """
    tile = wh_tiles.read_tile(tile_path, cfg)
    key = wh_naming.parse_path(tile_path)
    rng = rng or np.random.default_rng(params.random_state)

    water = _subsample(
        open_water_mask(tile, params), params.max_pixels_per_class_per_tile, rng
    )
    vegetation, reason = surrounding_vegetation_mask(
        tile, footprint, params, neighbour_mask=neighbours
    )
    vegetation = _subsample(vegetation, params.max_pixels_per_class_per_tile, rng)
    vegetation &= ~water

    return {
        "site_id": key.site_id,
        "year_month": key.year_month,
        "open_water": int(water.sum()),
        "surrounding_vegetation": int(vegetation.sum()),
        "vegetation_skipped": reason,
        "tif_path": str(tile_path),
    }


def survey(
    manifest: pd.DataFrame,
    cfg: Config,
    params: PseudoParams,
    sites: list[str] | None = None,
    months: list[str] | None = None,
    max_tiles_per_site: int | None = None,
) -> pd.DataFrame:
    """Yield table across many tiles, without writing. See evaluate_tile."""
    selected = _select(manifest, sites, months)
    footprints = _load_footprints(selected["site_id"].unique(), cfg)
    neighbours = _load_neighbour_masks(selected["site_id"].unique(), manifest, cfg)
    rng = np.random.default_rng(params.random_state)

    records = []
    for site_id, group in selected.groupby("site_id", sort=True):
        if max_tiles_per_site:
            # Spread the sample across the record rather than taking the first
            # N months, which would land entirely in one year and one season.
            step = max(1, len(group) // max_tiles_per_site)
            group = group.iloc[::step].head(max_tiles_per_site)
        for _, row in group.iterrows():
            try:
                records.append(
                    evaluate_tile(row["tif_path"], cfg, params, footprints.get(site_id),
                                  rng, neighbours.get(site_id))
                )
            except OSError as error:
                print(f"  skipped {row['tif_path']}: {str(error).splitlines()[0]}")

    return pd.DataFrame(records)


def generate(
    manifest: pd.DataFrame,
    cfg: Config,
    params: PseudoParams,
    sites: list[str] | None = None,
    months: list[str] | None = None,
    min_pixels: int = 5,
) -> pd.DataFrame:
    """Write pseudo-label masks and sidecars. Returns what was written.

    Tiles yielding fewer than min_pixels in total are skipped rather than
    written: a mask with three pixels in it is noise with a filename.
    """
    output_dir = pseudo_label_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    open_water_id = cfg.class_by_name("open_water").id
    vegetation_id = cfg.class_by_name("surrounding_vegetation").id

    selected = _select(manifest, sites, months)
    footprints = _load_footprints(selected["site_id"].unique(), cfg)
    neighbours = _load_neighbour_masks(selected["site_id"].unique(), manifest, cfg)
    rng = np.random.default_rng(params.random_state)

    records = []
    for _, row in selected.iterrows():
        tile_path = row["tif_path"]
        try:
            tile = wh_tiles.read_tile(tile_path, cfg)
        except OSError as error:
            print(f"  skipped {Path(tile_path).name}: {str(error).splitlines()[0]}")
            continue

        water = _subsample(
            open_water_mask(tile, params), params.max_pixels_per_class_per_tile, rng
        )
        vegetation, reason = surrounding_vegetation_mask(
            tile, footprints.get(row["site_id"]), params,
            neighbour_mask=neighbours.get(row["site_id"]),
        )
        vegetation = _subsample(vegetation, params.max_pixels_per_class_per_tile, rng)

        # Water wins any overlap: it is the rarer and more specific claim.
        vegetation &= ~water

        mask = np.zeros(tile.shape, dtype=np.uint8)
        mask[vegetation] = vegetation_id
        mask[water] = open_water_id

        if int((mask > 0).sum()) < min_pixels:
            continue

        mask_path = wh_naming.sibling_path(tile_path, output_dir, "_labels.tif")
        wh_tiles.write_mask(mask_path, mask, tile)
        _write_sidecar(tile_path, mask_path, mask, cfg, params, output_dir)

        records.append({
            "site_id": row["site_id"],
            "year_month": row["year_month"],
            "open_water": int(water.sum()),
            "surrounding_vegetation": int(vegetation.sum()),
            "vegetation_skipped": reason,
            "mask_path": str(mask_path),
        })

    return pd.DataFrame(records)


def written_masks(cfg: Config) -> pd.DataFrame:
    """Index of the pseudo masks on disk, with their per-class counts."""
    directory = pseudo_label_dir(cfg)
    if not directory.exists():
        return pd.DataFrame()

    rows = []
    for sidecar in sorted(directory.glob("*_labels.json")):
        meta = json.loads(sidecar.read_text())
        row = {
            "site_id": meta["site_id"],
            "year_month": meta["year_month"],
            "n_labelled": meta["n_labelled"],
            "mask_path": meta["label_mask"],
            "tif_path": meta["source_tile"],
        }
        row.update({k: v for k, v in meta["pixel_counts"].items() if k != "unlabelled"})
        rows.append(row)
    return pd.DataFrame(rows)


def most_informative(cfg: Config, n: int = 6, class_name: str = "open_water") -> pd.DataFrame:
    """Tiles worth looking at: the ones where the rule actually claimed something.

    Defaults to open water, because that is the rule most likely to be wrong and
    the one whose thresholds were hardest to calibrate. A random sample would
    mostly show tiles with nothing but matrix vegetation on them.
    """
    written = written_masks(cfg)
    if written.empty or class_name not in written.columns:
        return written

    # One tile per site, so the sample spans waterholes rather than showing six
    # months of the same one.
    ranked = written[written[class_name] > 0].sort_values(class_name, ascending=False)
    return ranked.groupby("site_id", as_index=False).head(1).head(n)


def agreement_with_manual(cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cross-tabulate hand labels against pseudo labels on the same pixels.

    This is the real test of whether the automatic rules are trustworthy. Only
    pixels labelled by BOTH are counted, so the table is small — but where a
    person and a threshold disagree about the same pixel, the threshold is the
    one to doubt.

    Returns (crosstab, summary).
    """
    manual_dir = cfg.paths["labels"]
    pseudo_dir = pseudo_label_dir(cfg)

    names = {d.id: d.name for d in cfg.classes}
    pairs: list[tuple[int, int]] = []
    n_tiles = 0

    for pseudo_path in sorted(pseudo_dir.glob("*_labels.tif")):
        manual_path = manual_dir / pseudo_path.name
        if not manual_path.exists():
            continue

        manual = wh_tiles.read_mask(manual_path)
        pseudo = wh_tiles.read_mask(pseudo_path, manual.shape)

        both = (manual > 0) & (pseudo > 0)
        if not both.any():
            continue

        n_tiles += 1
        pairs.extend(zip(manual[both].tolist(), pseudo[both].tolist()))

    if not pairs:
        return pd.DataFrame(), {"overlapping_tiles": n_tiles, "overlapping_pixels": 0}

    frame = pd.DataFrame(pairs, columns=["manual", "pseudo"])
    crosstab = pd.crosstab(
        frame["manual"].map(names), frame["pseudo"].map(names)
    )

    agreed = int((frame["manual"] == frame["pseudo"]).sum())
    summary = {
        "overlapping_tiles": n_tiles,
        "overlapping_pixels": len(frame),
        "agreed": agreed,
        "agreement_rate": agreed / len(frame),
    }
    return crosstab, summary


def _select(
    manifest: pd.DataFrame, sites: list[str] | None, months: list[str] | None
) -> pd.DataFrame:
    selected = manifest
    if sites:
        selected = selected[selected["site_id"].isin(sites)]
    if months:
        selected = selected[selected["year_month"].isin(months)]
    return selected.sort_values(["site_id", "month_index"])


def _load_footprints(site_ids, cfg: Config) -> dict[str, np.ndarray]:
    footprints = {}
    for site_id in site_ids:
        try:
            footprints[site_id] = wh_footprint.load_mask(cfg, site_id)
        except FileNotFoundError:
            continue
    return footprints


def _load_neighbour_masks(
    site_ids, manifest: pd.DataFrame, cfg: Config
) -> dict[str, np.ndarray]:
    """Per-site union of the OTHER waterholes' extents in that site's tile.

    Built once per site — every month of a site shares one grid. Returns an empty
    mapping when the box table has not been built, so pseudo-labelling still runs
    (just without neighbour exclusion) rather than failing.
    """
    import wh_bbox

    try:
        boxes = wh_bbox.load_boxes(cfg)
    except FileNotFoundError:
        print("  no bounding-box table; neighbouring waterholes will not be excluded")
        return {}

    params = wh_bbox.BoxParams.from_config(cfg)
    masks = {}
    for site_id in site_ids:
        rows = manifest[manifest["site_id"] == site_id]
        if rows.empty:
            continue
        try:
            tile = wh_tiles.read_tile(rows.iloc[0]["tif_path"], cfg)
        except OSError:
            continue
        masks[site_id] = wh_bbox.neighbour_mask(site_id, tile, boxes, params)
    return masks


def _write_sidecar(
    tile_path, mask_path: Path, mask: np.ndarray, cfg: Config,
    params: PseudoParams, output_dir: Path,
) -> None:
    key = wh_naming.parse_path(tile_path)
    sidecar = {
        "source_tile": str(tile_path),
        "label_mask": str(mask_path),
        "site_id": key.site_id,
        "year_month": key.year_month,
        "class_scheme_version": cfg["classes"]["scheme_version"],
        "config_hash": cfg.hash,
        "labeller": "wh_pseudo",
        "source": "pseudo",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pixel_counts": {
            definition.name: int((mask == definition.id).sum())
            for definition in cfg.classes
        },
        "n_labelled": int((mask > 0).sum()),
        "params": params.as_dict(),
    }
    path = wh_naming.sibling_path(mask_path, output_dir, ".json")
    path.write_text(json.dumps(sidecar, indent=1))
