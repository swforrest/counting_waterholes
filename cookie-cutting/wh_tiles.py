"""Read chips and write label masks.

The GeoTIFF is authoritative throughout: bands, grid, CRS and transform all come
from the file, and label masks are written back onto exactly that grid. Nothing
here reads or trusts the rendered PNGs.

Two nodata conventions exist in the wild:

  * the re-export (images_tif_v2) sets nodata = -9999 explicitly and carries an
    n_obs band;
  * the original 2024 export (images_tif) sets no nodata at all, so masked
    pixels arrive as 0.0 and are indistinguishable from a genuinely dark pixel
    except by the fact that every band is exactly zero.

The second is a guess and is treated as one: reading a legacy chip records a
warning on the Tile rather than pretending the data is clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS

import wh_indices
import wh_naming
from wh_config import Config
from wh_naming import TileKey

LEGACY_NODATA_WARNING = (
    "no nodata tag: gaps inferred from all-zero pixels (legacy EPSG:4326 export)"
)


@dataclass
class Tile:
    """One site-month chip, read into memory."""

    key: TileKey
    path: Path
    bands: dict[str, np.ndarray]  # float64, NaN where there is no observation
    n_obs: np.ndarray | None  # int16 count of clear observations, or None if absent
    valid: np.ndarray  # bool, True where every band has an observation
    crs: CRS
    transform: rasterio.Affine
    shape: tuple[int, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def gap_fraction(self) -> float:
        """Fraction of the chip with no clear observation in this month."""
        return float((~self.valid).mean())

    def index(self, name: str) -> np.ndarray:
        """Compute one spectral index for this tile."""
        return wh_indices.compute(name, self.bands)

    def indices(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        """Compute several spectral indices for this tile."""
        return wh_indices.compute_many(names, self.bands)


def read_tile(path: str | Path, cfg: Config) -> Tile:
    """Read a chip, converting nodata to NaN. Raises on a malformed file.

    Band identity is taken from the file's band descriptions when they are
    present and checked against the configured order; when they are absent
    (legacy export) the configured order is applied positionally.
    """
    path = Path(path)
    key = wh_naming.parse_path(path)

    band_names: list[str] = list(cfg["tiles"]["bands"])
    obs_band: str = cfg["tiles"]["obs_band"]
    nodata_value: float = float(cfg["tiles"]["nodata"])

    warnings: list[str] = []

    with rasterio.open(path) as dataset:
        descriptions = list(dataset.descriptions)
        raw = dataset.read().astype(np.float64)
        file_nodata = dataset.nodata
        crs = dataset.crs
        transform = dataset.transform
        shape = (dataset.height, dataset.width)

    n_bands = raw.shape[0]
    has_obs = n_bands == len(band_names) + 1

    if not has_obs and n_bands != len(band_names):
        raise ValueError(
            f"{path}: expected {len(band_names)} bands (or "
            f"{len(band_names) + 1} with '{obs_band}'), found {n_bands}"
        )

    expected = band_names + ([obs_band] if has_obs else [])
    if all(description is not None for description in descriptions):
        if descriptions != expected:
            raise ValueError(
                f"{path}: band descriptions {descriptions} do not match the "
                f"configured order {expected}"
            )
    else:
        warnings.append("no band descriptions: band identity assumed from position")

    optical = raw[: len(band_names)]

    if file_nodata is not None:
        gap = np.isclose(optical, file_nodata) | ~np.isfinite(optical)
    else:
        # Legacy chips: a gap is a pixel that is exactly zero in every band.
        warnings.append(LEGACY_NODATA_WARNING)
        gap = np.broadcast_to(
            np.all(optical == 0.0, axis=0), optical.shape
        ).copy()
        gap |= ~np.isfinite(optical)

    optical = np.where(gap, np.nan, optical)
    valid = ~np.any(gap, axis=0)

    if has_obs:
        n_obs = raw[len(band_names)]
        n_obs = np.where(np.isfinite(n_obs), n_obs, 0).astype(np.int16)
    else:
        n_obs = None
        warnings.append(
            f"no '{obs_band}' band: cannot tell a one-scene median from a six-scene one"
        )

    _check_reflectance_range(path, optical, band_names, cfg, warnings)

    if nodata_value >= 0:
        raise ValueError(
            f"{cfg.source_path}: tiles.nodata must be negative so it cannot "
            f"collide with reflectance, got {nodata_value}"
        )

    return Tile(
        key=key,
        path=path,
        bands={name: optical[i] for i, name in enumerate(band_names)},
        n_obs=n_obs,
        valid=valid,
        crs=crs,
        transform=transform,
        shape=shape,
        warnings=warnings,
    )


def _check_reflectance_range(
    path: Path,
    optical: np.ndarray,
    band_names: list[str],
    cfg: Config,
    warnings: list[str],
) -> None:
    """Fail loudly if the data is not on the reflectance scale we expect.

    Guards the exact failure the renderer would hit silently: if the GEE export
    stops dividing by 10000, values become ~2000 instead of ~0.2 and every index
    threshold in the config quietly means something else.
    """
    finite = optical[np.isfinite(optical)]
    if finite.size == 0:
        warnings.append("chip is entirely nodata")
        return

    low = float(cfg["tiles"]["reflectance_min"])
    high = float(cfg["tiles"]["reflectance_max"])
    observed_max = float(finite.max())

    if observed_max > high:
        raise ValueError(
            f"{path}: maximum reflectance {observed_max:.1f} exceeds "
            f"{high}. The chip is probably still scaled by 10000 — check the "
            f"'.divide(10000)' step in cookie-cutting_S2_download.ipynb."
        )
    if float(finite.min()) < low:
        warnings.append(f"negative reflectance present (min {finite.min():.4f})")


def describe_geometry(path: str | Path) -> dict[str, object]:
    """Cheap header-only read of the properties the inventory compares.

    Kept separate from read_tile so building a manifest over thousands of chips
    does not pull every band into memory.
    """
    path = Path(path)
    with rasterio.open(path) as dataset:
        pixel_width, pixel_height = dataset.res
        return {
            "path": str(path),
            "height": dataset.height,
            "width": dataset.width,
            "n_bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "nodata": dataset.nodata,
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "bounds_left": dataset.bounds.left,
            "bounds_bottom": dataset.bounds.bottom,
            "bounds_right": dataset.bounds.right,
            "bounds_top": dataset.bounds.top,
            "has_band_names": all(d is not None for d in dataset.descriptions),
        }


def write_mask(
    path: str | Path,
    mask: np.ndarray,
    reference: Tile,
    compress: str = "deflate",
) -> Path:
    """Write a uint8 label mask on the reference tile's exact grid.

    0 means unlabelled. The mask must already match the tile's shape — this
    function will not resample, because a resampled label is a wrong label.
    """
    path = Path(path)
    if mask.shape != reference.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match tile shape "
            f"{reference.shape}; labels are never resampled"
        )
    if mask.dtype != np.uint8:
        if mask.min() < 0 or mask.max() > 255:
            raise ValueError(f"mask values {mask.min()}..{mask.max()} do not fit in uint8")
        mask = mask.astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype="uint8",
        crs=reference.crs,
        transform=reference.transform,
        nodata=None,  # 0 means 'unlabelled', which is data, not absence
        compress=compress,
    ) as dataset:
        dataset.write(mask, 1)
        dataset.descriptions = ("class_id",)

    return path


def read_mask(path: str | Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a uint8 label mask, checking it is on the grid the caller expects."""
    path = Path(path)
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"{path}: label mask must have one band, has {dataset.count}")
        mask = dataset.read(1)

    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(
            f"{path}: mask shape {mask.shape} does not match the tile it is "
            f"being loaded against {expected_shape}"
        )
    return mask.astype(np.uint8)


def label_path_for(tile_path: str | Path, cfg: Config) -> Path:
    """Where the label mask for a chip lives, mirroring the tile layout."""
    return wh_naming.sibling_path(tile_path, cfg.paths["labels"], "_labels.tif")


def sidecar_path_for(tile_path: str | Path, cfg: Config) -> Path:
    """Where the JSON sidecar for a chip's label mask lives."""
    return wh_naming.sibling_path(tile_path, cfg.paths["labels"], "_labels.json")


def png_path_for(tile_path: str | Path, cfg: Config) -> Path:
    """The pre-rendered 3-panel chip for a tile, if one exists."""
    return wh_naming.sibling_path(tile_path, cfg.paths["chips_png"], ".png")
