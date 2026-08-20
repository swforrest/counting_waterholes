"""Per-site waterhole bounding boxes, and masks derived from them.

The AOIs were built by buffering each labelled bounding-box *centre* by 750 m, so
a 1.5 km chip often catches neighbouring waterholes — 93 of 187 tiles contain more
than one, up to 5. A composition fraction computed over a whole tile therefore
mixes several waterholes together, which confounds the per-site time series the
project exists to produce.

The same labelme JSON carries each waterhole's rectangular *extent*, which is what
bounds a site to its own basin. This module turns those boxes into masks on each
tile's exact grid.

Two things the masks are deliberately NOT for:

  * They never filter training pixels. `surrounding_vegetation` is 50.7% outside
    the box by definition — it *is* the matrix — while only 10 of 29,314
    waterhole-class labelled pixels fall outside. Filtering on the box would
    delete the majority class.
  * They never replace the tile when computing statistics. `wh_footprint.robust_z`
    compares a pixel to its tile's median and MAD, and the buffered box is a
    median 10% of the tile — far too little matrix to estimate a baseline from.

What they ARE for: constraining which connected component the footprint selects,
bounding the denominator for composition fractions, and keeping the pseudo-labeller
from claiming a neighbouring waterhole as savanna matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from PIL import Image

import wh_tiles
from wh_config import Config

# The reference mosaic is ~5400x4200; PIL reads its size from the header without
# decoding, but the default bomb guard trips on the pixel count.
Image.MAX_IMAGE_PIXELS = None

WGS84 = "EPSG:4326"


@dataclass
class BoxParams:
    """Everything tunable about the box masks."""

    buffer_m: float = 100.0

    @classmethod
    def from_config(cls, cfg: Config) -> "BoxParams":
        return cls(buffer_m=float(cfg["bounding_boxes"]["buffer_m"]))

    def as_dict(self) -> dict[str, object]:
        return {"buffer_m": self.buffer_m}


# --- extracting the boxes from the labelme JSON ----------------------------


def extract_boxes(cfg: Config) -> pd.DataFrame:
    """Parse the labelme JSON into one row per waterhole, indexed by site_id.

    Reproduces the padding arithmetic from cookie-cutting_S2_download.ipynb: the
    JSON stores corners in PNG pixel coordinates, and that PNG was padded before
    tiling, so the offset has to be removed before the reference raster's
    transform can turn them into coordinates.

    The row order of the JSON is the site id. That is not an assumption — it is
    checked against the exported filenames by `verify_against_tiles`.
    """
    settings = cfg["bounding_boxes"]
    reference_tif = _resolve(cfg, settings["reference_tif"])
    reference_png = _resolve(cfg, settings["reference_png"])
    json_path = _resolve(cfg, settings["labelme_json"])

    for path in (reference_tif, reference_png, json_path):
        if not path.exists():
            raise FileNotFoundError(f"bounding-box source missing: {path}")

    with rasterio.open(reference_tif) as dataset:
        raster_height, raster_width = dataset.height, dataset.width
        transform = dataset.transform

    png_width, png_height = Image.open(reference_png).size

    # The mosaic was padded on all sides before tiling; the right and bottom
    # padding is always tile_size - stride, so the rest is the left/top offset.
    minimum_pad = int(settings["tile_size"]) - int(settings["stride"])
    left_pad = (png_width - raster_width) - minimum_pad
    top_pad = (png_height - raster_height) - minimum_pad

    if left_pad < 0 or top_pad < 0:
        raise ValueError(
            f"negative padding (left {left_pad}, top {top_pad}) from PNG "
            f"{png_width}x{png_height} and raster {raster_width}x{raster_height}; "
            f"tile_size/stride in the config probably do not match this mosaic"
        )

    shapes = json.loads(json_path.read_text())["shapes"]

    rows = []
    for index, shape in enumerate(shapes):
        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        west, north = transform * (x_min - left_pad, y_min - top_pad)
        east, south = transform * (x_max - left_pad, y_max - top_pad)

        rows.append({
            "site_id": f"{index:03d}",
            "label": shape["label"],
            "lon_min": min(west, east),
            "lon_max": max(west, east),
            "lat_min": min(north, south),
            "lat_max": max(north, south),
        })

    boxes = pd.DataFrame(rows)
    boxes["center_lon"] = (boxes["lon_min"] + boxes["lon_max"]) / 2
    boxes["center_lat"] = (boxes["lat_min"] + boxes["lat_max"]) / 2
    boxes["width_m"] = (boxes["lon_max"] - boxes["lon_min"]) * _metres_per_degree_lon(
        boxes["center_lat"]
    )
    boxes["height_m"] = (boxes["lat_max"] - boxes["lat_min"]) * 110574.0

    return boxes.set_index("site_id", drop=False)


def _metres_per_degree_lon(latitude):
    return 111320.0 * np.cos(np.radians(latitude))


def _resolve(cfg: Config, relative: str) -> Path:
    from wh_config import REPO_ROOT

    return (REPO_ROOT / relative).resolve()


def boxes_path(cfg: Config) -> Path:
    return cfg.paths["derived"] / "bounding_boxes.csv"


def save_boxes(boxes: pd.DataFrame, cfg: Config) -> Path:
    """Cache the box table so nothing downstream re-parses a 37 MB JSON."""
    path = boxes_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes.to_csv(path, index=False)
    return path


def load_boxes(cfg: Config) -> pd.DataFrame:
    path = boxes_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no box table at {path}; run extract_boxes first")
    return pd.read_csv(path, dtype={"site_id": str}).set_index("site_id", drop=False)


def verify_against_tiles(boxes: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Check that box row order really is the site id.

    The filenames encode each AOI's centre to two decimals, which is exactly the
    box centre the export used. Any disagreement means the JSON has been re-saved
    in a different order and every mask would be attached to the wrong waterhole.
    """
    sites = manifest.drop_duplicates("site_id").set_index("site_id")

    problems = []
    for site_id, box in boxes.iterrows():
        if site_id not in sites.index:
            problems.append({"site_id": site_id, "issue": "no tiles for this site"})
            continue
        tile = sites.loc[site_id]
        if abs(round(box["center_lat"], 2) - round(tile["lat"], 2)) > 0.005 or abs(
            round(box["center_lon"], 2) - round(tile["lon"], 2)
        ) > 0.005:
            problems.append({
                "site_id": site_id,
                "issue": (
                    f"box centre ({box['center_lat']:.3f}, {box['center_lon']:.3f}) "
                    f"does not match tile ({tile['lat']:.2f}, {tile['lon']:.2f})"
                ),
            })

    return pd.DataFrame(problems)


# --- rasterising boxes onto a tile grid ------------------------------------


def _box_bounds_in_tile_crs(box: pd.Series, crs, buffer_m: float):
    """The buffered box as (left, bottom, right, top) in the tile's CRS."""
    lons = [box["lon_min"], box["lon_max"], box["lon_max"], box["lon_min"]]
    lats = [box["lat_max"], box["lat_max"], box["lat_min"], box["lat_min"]]
    xs, ys = rasterio.warp.transform(WGS84, crs, lons, lats)

    return (
        min(xs) - buffer_m,
        min(ys) - buffer_m,
        max(xs) + buffer_m,
        max(ys) + buffer_m,
    )


def box_mask(
    box: pd.Series,
    tile: wh_tiles.Tile,
    params: BoxParams,
) -> tuple[np.ndarray, bool]:
    """Rasterise one buffered box onto a tile's exact grid.

    Returns (mask, clipped). `clipped` is True when the buffered box reaches
    beyond the tile, which happens for 7 of 187 sites at a 100 m buffer — those
    waterholes are larger than the AOI that was cut for them, and the mask cannot
    bound them. Reported rather than silently accepted, because for those sites a
    composition fraction is over the whole tile and means something different.
    """
    left, bottom, right, top = _box_bounds_in_tile_crs(box, tile.crs, params.buffer_m)

    inverse = ~tile.transform
    col_min, row_max = inverse * (left, bottom)
    col_max, row_min = inverse * (right, top)

    height, width = tile.shape
    clipped = col_min < 0 or row_min < 0 or col_max > width or row_max > height

    col_start = int(np.clip(np.floor(col_min), 0, width))
    col_stop = int(np.clip(np.ceil(col_max), 0, width))
    row_start = int(np.clip(np.floor(row_min), 0, height))
    row_stop = int(np.clip(np.ceil(row_max), 0, height))

    mask = np.zeros(tile.shape, dtype=bool)
    mask[row_start:row_stop, col_start:col_stop] = True
    return mask, clipped


def neighbour_mask(
    site_id: str,
    tile: wh_tiles.Tile,
    boxes: pd.DataFrame,
    params: BoxParams,
) -> np.ndarray:
    """Union of every OTHER site's box that overlaps this tile.

    Half the tiles contain a second waterhole. Without this, the pseudo-labeller
    would happily claim a neighbouring waterhole's margin as savanna matrix and
    teach the classifier the opposite of what is wanted.
    """
    mask = np.zeros(tile.shape, dtype=bool)
    for other_id, box in boxes.iterrows():
        if other_id == site_id:
            continue
        other, _ = box_mask(box, tile, params)
        mask |= other
    return mask


# --- persistence -----------------------------------------------------------


def mask_path(cfg: Config, site_id: str) -> Path:
    return cfg.paths["derived"] / "bounding_boxes" / f"site_{site_id}_box.tif"


def save_mask(
    mask: np.ndarray, tile: wh_tiles.Tile, cfg: Config, site_id: str
) -> Path:
    """Write a box mask on the tile's exact grid, like the footprint masks."""
    path = mask_path(cfg, site_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=mask.shape[0], width=mask.shape[1], count=1, dtype="uint8",
        crs=tile.crs, transform=tile.transform, nodata=None, compress="deflate",
    ) as dataset:
        dataset.write(mask.astype(np.uint8), 1)
        dataset.descriptions = ("bounding_box",)

    return path


def load_mask(cfg: Config, site_id: str) -> np.ndarray:
    path = mask_path(cfg, site_id)
    if not path.exists():
        raise FileNotFoundError(f"no box mask at {path}; derive it first")
    with rasterio.open(path) as dataset:
        return dataset.read(1).astype(bool)


def build_all(
    manifest: pd.DataFrame,
    cfg: Config,
    boxes: pd.DataFrame,
    params: BoxParams,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rasterise and save a box mask for every site. Returns a summary table."""
    records = []
    for site_id, box in boxes.iterrows():
        rows = manifest[manifest["site_id"] == site_id]
        if rows.empty:
            records.append({"site_id": site_id, "status": "no tiles", "n_pixels": 0})
            continue

        try:
            tile = wh_tiles.read_tile(rows.iloc[0]["tif_path"], cfg)
        except OSError as error:
            records.append({
                "site_id": site_id, "status": str(error).splitlines()[0], "n_pixels": 0,
            })
            continue

        mask, clipped = box_mask(box, tile, params)
        save_mask(mask, tile, cfg, site_id)

        pixel_area = abs(tile.transform.a * tile.transform.e)
        records.append({
            "site_id": site_id,
            "label": box["label"],
            "width_m": round(float(box["width_m"]), 1),
            "height_m": round(float(box["height_m"]), 1),
            "n_pixels": int(mask.sum()),
            "area_ha": round(float(mask.sum() * pixel_area / 1e4), 2),
            "tile_fraction": round(float(mask.mean()), 4),
            "clipped_by_tile": clipped,
            "status": "clipped" if clipped else "ok",
        })

    summary = pd.DataFrame(records)
    if verbose:
        clipped = summary.get("clipped_by_tile")
        n_clipped = int(clipped.sum()) if clipped is not None else 0
        print(f"built {len(summary)} box masks; {n_clipped} clipped by the tile edge")
    return summary
