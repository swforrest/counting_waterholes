"""Build and summarise a manifest of the exported chips.

One row per GeoTIFF, carrying its parsed identity, its georeferencing, its
pairing with a pre-rendered PNG, and its observation quality. Everything
downstream selects work from this table rather than globbing directories.

The checks that matter most:

  * Does every site have every month?
  * Is each site's grid IDENTICAL across its own months? Per-pixel temporal
    features are meaningless if a site's rows and columns shift between months,
    so this is verified rather than assumed.
  * How well observed is each monthly median? A median over one surviving scene
    is a different measurement from a median over six.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import wh_naming
from wh_config import Config

# Threads, not processes: this is dominated by file-open latency, and the tiles
# usually live on a synced network drive.
DEFAULT_WORKERS = 16


def _scan_one(path: Path, obs_band_index: int) -> dict[str, object]:
    """Header plus the n_obs band for a single chip.

    Only one of the thirteen bands is read. n_obs is enough to derive the gap
    fraction (no clear observation means a count of zero), which keeps a scan of
    ~16k chips to minutes rather than an hour.
    """
    key = wh_naming.parse_path(path)

    with rasterio.open(path) as dataset:
        pixel_width, pixel_height = dataset.res
        record: dict[str, object] = {
            "site_id": key.site_id,
            "year": key.year,
            "month": key.month,
            "year_month": key.year_month,
            "month_index": key.month_index,
            "lat": key.lat,
            "lon": key.lon,
            "tif_path": str(path),
            "height": dataset.height,
            "width": dataset.width,
            "n_bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "nodata": dataset.nodata,
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "left": dataset.bounds.left,
            "top": dataset.bounds.top,
            "has_band_names": all(d is not None for d in dataset.descriptions),
        }

        if dataset.count > obs_band_index:
            n_obs = dataset.read(obs_band_index + 1).astype(np.float64)
        else:
            n_obs = None

    if n_obs is None:
        record.update(
            has_obs_band=False,
            gap_fraction=np.nan,
            edge_invalid_px=0,
            mean_n_obs=np.nan,
            max_n_obs=np.nan,
        )
        return record

    # -inf appears in a thin border strip where the reprojected AOI does not
    # fill the raster's bounding box. Counted separately from real cloud gaps.
    finite = np.isfinite(n_obs)
    observed = finite & (n_obs > 0)

    record.update(
        has_obs_band=True,
        gap_fraction=float((~observed).mean()),
        edge_invalid_px=int((~finite).sum()),
        mean_n_obs=float(n_obs[observed].mean()) if observed.any() else 0.0,
        max_n_obs=float(n_obs[finite].max()) if finite.any() else 0.0,
    )
    return record


def build_manifest(
    cfg: Config,
    tiles_dir: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    progress: bool = True,
) -> pd.DataFrame:
    """Scan the tile directory and return the manifest, sorted by site then month.

    Prints progress by default: a full scan opens ~16k files, which on a synced
    network drive takes minutes and is otherwise indistinguishable from a hang.
    """
    directory = Path(tiles_dir) if tiles_dir else cfg.paths["tiles"]
    if not directory.exists():
        raise FileNotFoundError(f"tile directory not found: {directory}")

    paths = sorted(directory.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"no .tif files in {directory}")

    unparsed = [p.name for p in paths if wh_naming.try_parse_path(p) is None]
    if unparsed:
        raise ValueError(
            f"{len(unparsed)} file(s) in {directory} do not match the chip "
            f"naming convention, e.g. {unparsed[:3]}"
        )

    obs_band_index = len(cfg["tiles"]["bands"])  # n_obs sits after the optical bands

    records = _scan_all(paths, obs_band_index, workers, progress)

    manifest = pd.DataFrame.from_records(records)
    manifest = _add_png_pairing(manifest, cfg)
    manifest = _add_grid_consistency(manifest)

    return manifest.sort_values(["site_id", "month_index"]).reset_index(drop=True)


def _scan_all(
    paths: list[Path],
    obs_band_index: int,
    workers: int,
    progress: bool,
) -> list[dict[str, object]]:
    """Scan every chip concurrently, reporting progress as results land."""
    records: list[dict[str, object]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_one, path, obs_band_index) for path in paths]

        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())

            if progress and (completed % 250 == 0 or completed == len(paths)):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0.0
                remaining = (len(paths) - completed) / rate if rate else 0.0
                print(
                    f"\r  scanned {completed:,}/{len(paths):,} chips "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                    end="",
                    flush=True,
                    file=sys.stdout,
                )

    if progress:
        print(f"\r  scanned {len(paths):,} chips in {time.time() - started:.0f}s"
              f"{' ' * 30}")

    return records


def _add_png_pairing(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pair each chip with its pre-rendered PNG, if one exists."""
    png_dir = cfg.paths["chips_png"]
    available = (
        {p.stem: str(p) for p in png_dir.glob("*.png")} if png_dir.exists() else {}
    )

    stems = manifest["tif_path"].map(lambda p: Path(p).stem)
    manifest["png_path"] = stems.map(available).fillna("")
    manifest["has_png"] = manifest["png_path"] != ""
    return manifest


def _add_grid_consistency(manifest: pd.DataFrame) -> pd.DataFrame:
    """Flag chips whose grid differs from the rest of their own site.

    The site's modal grid is taken as the reference. A month that disagrees
    cannot be stacked with its siblings, so it is excluded from temporal
    features rather than silently misaligned.
    """
    grid = manifest[["height", "width", "left", "top", "crs"]].round(
        {"left": 3, "top": 3}
    )
    manifest["grid_signature"] = [
        f"{row.crs}|{row.height}x{row.width}|{row.left}|{row.top}"
        for row in grid.itertuples()
    ]

    modal = manifest.groupby("site_id")["grid_signature"].agg(
        lambda values: values.mode().iat[0]
    )
    manifest["site_grid_signature"] = manifest["site_id"].map(modal)
    manifest["grid_matches_site"] = (
        manifest["grid_signature"] == manifest["site_grid_signature"]
    )
    return manifest


def save_manifest(manifest: pd.DataFrame, cfg: Config, filename: str = "manifest.csv") -> Path:
    """Write the manifest to the derived directory as CSV."""
    output_path = cfg.paths["derived"] / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return output_path


def load_manifest(cfg: Config, filename: str = "manifest.csv") -> pd.DataFrame:
    """Read a previously saved manifest, keeping site_id as a zero-padded string."""
    path = cfg.paths["derived"] / filename
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}; run build_manifest first")
    manifest = pd.read_csv(path, dtype={"site_id": str})
    return _reroot_paths(manifest, cfg)


def _reroot_paths(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Re-anchor stored paths whose directory no longer exists.

    The manifest records absolute paths, but the OneDrive mount root is not
    stable: re-linking the shared library remounts it under a new name (the
    original comes back as '...QueenslandUniversityofTechnology 2'), which
    strands every path in a manifest built before the rename. The failure then
    surfaces as a per-chip 'No such file or directory', which reads like missing
    data rather than a moved mount point.

    Only paths whose parent directory has genuinely disappeared are rewritten,
    and only onto a directory of the same name from the config, so a truly
    missing chip still fails against the path it was recorded under.
    """
    known = {directory.name: directory for directory in cfg.paths.values()}
    parent_exists: dict[str, bool] = {}

    def reroot(value: object) -> str:
        if not isinstance(value, str) or not value:
            return ""
        old = Path(value)
        parent = str(old.parent)
        if parent not in parent_exists:
            parent_exists[parent] = old.parent.exists()
        if parent_exists[parent]:
            return value
        replacement = known.get(old.parent.name)
        return str(replacement / old.name) if replacement else value

    for column in ("tif_path", "png_path"):
        if column in manifest.columns:
            manifest[column] = manifest[column].map(reroot)

    return manifest


def summarise(manifest: pd.DataFrame, cfg: Config) -> None:
    """Print the summary. Reports problems prominently rather than burying them."""
    sites = manifest["site_id"].unique()
    months = sorted(manifest["year_month"].unique())

    print("=" * 72)
    print(f"{len(manifest):,} chips | {len(sites)} sites | {len(months)} months "
          f"({months[0]} to {months[-1]})")
    print("=" * 72)

    _report_missing_months(manifest, months)
    _report_grid(manifest, cfg)
    _report_bands(manifest, cfg)
    _report_observations(manifest)
    _report_png_pairing(manifest, cfg)


def _report_missing_months(manifest: pd.DataFrame, months: list[str]) -> None:
    print("\n-- month coverage --")
    counts = manifest.groupby("site_id")["year_month"].nunique()
    incomplete = counts[counts < len(months)]

    if incomplete.empty:
        print(f"  every site has all {len(months)} months")
        return

    print(f"  {len(incomplete)} site(s) missing months:")
    present_by_site = manifest.groupby("site_id")["year_month"].apply(set)
    for site_id, count in incomplete.items():
        missing = sorted(set(months) - present_by_site[site_id])
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"    site {site_id}: {count}/{len(months)} present, missing {shown}")


def _report_grid(manifest: pd.DataFrame, cfg: Config) -> None:
    print("\n-- georeferencing --")
    expected_crs = str(cfg["tiles"]["expected_crs"])
    expected_size = float(cfg["tiles"]["expected_pixel_size_m"])
    tolerance = float(cfg["tiles"]["pixel_size_tolerance_m"])

    wrong_crs = manifest[manifest["crs"] != expected_crs]
    if wrong_crs.empty:
        print(f"  all chips are {expected_crs}")
    else:
        print(f"  !! {len(wrong_crs)} chip(s) not in {expected_crs}: "
              f"{sorted(wrong_crs['crs'].unique())}")

    off_size = manifest[
        (manifest["pixel_width"] - expected_size).abs().gt(tolerance)
        | (manifest["pixel_height"] - expected_size).abs().gt(tolerance)
    ]
    if off_size.empty:
        print(f"  all chips have {expected_size} m square pixels")
    else:
        print(f"  !! {len(off_size)} chip(s) with unexpected pixel size")

    mismatched = manifest[~manifest["grid_matches_site"]]
    if mismatched.empty:
        print("  every site's grid is identical across all of its months")
    else:
        print(f"  !! {len(mismatched)} chip(s) whose grid differs from their site's "
              f"siblings — these cannot be stacked temporally:")
        for site_id, group in mismatched.groupby("site_id"):
            print(f"    site {site_id}: {sorted(group['year_month'])[:6]}")

    shapes = manifest.groupby(["height", "width"]).size().sort_values(ascending=False)
    print(f"  chip shapes across sites: "
          + ", ".join(f"{h}x{w} ({n})" for (h, w), n in shapes.items()))


def _report_bands(manifest: pd.DataFrame, cfg: Config) -> None:
    print("\n-- bands --")
    expected_bands = len(cfg["tiles"]["bands"]) + 1

    wrong_count = manifest[manifest["n_bands"] != expected_bands]
    if wrong_count.empty:
        print(f"  all chips have {expected_bands} bands "
              f"({len(cfg['tiles']['bands'])} optical + {cfg['tiles']['obs_band']})")
    else:
        print(f"  !! {len(wrong_count)} chip(s) without {expected_bands} bands")

    unnamed = manifest[~manifest["has_band_names"]]
    if not unnamed.empty:
        print(f"  !! {len(unnamed)} chip(s) carry no band descriptions")

    missing_obs = manifest[~manifest["has_obs_band"]]
    if missing_obs.empty:
        print(f"  all chips carry the {cfg['tiles']['obs_band']} band")
    else:
        print(f"  !! {len(missing_obs)} chip(s) have no observation-count band")

    no_nodata = manifest[manifest["nodata"].isna()]
    if not no_nodata.empty:
        print(f"  !! {len(no_nodata)} chip(s) have no nodata tag")


def _report_observations(manifest: pd.DataFrame) -> None:
    print("\n-- observation quality --")
    if not manifest["has_obs_band"].any():
        print("  no observation-count band; nothing to report")
        return

    empty = manifest[manifest["gap_fraction"] >= 1.0]
    print(f"  fully unobserved chips: {len(empty)}")
    if not empty.empty:
        by_site = empty.groupby("site_id").size().sort_values(ascending=False)
        print(f"    worst sites: "
              + ", ".join(f"{s} ({n})" for s, n in by_site.head(5).items()))

    heavy = manifest[manifest["gap_fraction"] > 0.5]
    print(f"  chips more than half unobserved: {len(heavy)} "
          f"({100 * len(heavy) / len(manifest):.1f}%)")

    thin = manifest[manifest["mean_n_obs"] < 2]
    print(f"  chips whose median rests on <2 scenes on average: {len(thin)} "
          f"({100 * len(thin) / len(manifest):.1f}%)")

    print("\n  by calendar month (mean across sites and years):")
    by_month = manifest.groupby("month").agg(
        gap=("gap_fraction", "mean"),
        obs=("mean_n_obs", "mean"),
        empty=("gap_fraction", lambda values: (values >= 1.0).sum()),
    )
    print(f"    {'mo':>4} {'gap':>7} {'n_obs':>7} {'empty':>7}")
    for month, row in by_month.iterrows():
        print(f"    {month:>4} {row['gap']:>7.3f} {row['obs']:>7.2f} "
              f"{int(row['empty']):>7}")

    edge = manifest["edge_invalid_px"]
    print(f"\n  reprojection border (-inf) pixels per chip: "
          f"median {edge.median():.0f}, max {edge.max():.0f} "
          f"— treated as nodata on read")


def _report_png_pairing(manifest: pd.DataFrame, cfg: Config) -> None:
    print("\n-- pre-rendered PNGs --")
    n_paired = int(manifest["has_png"].sum())
    print(f"  {n_paired:,} of {len(manifest):,} chips have a PNG")

    if n_paired < len(manifest):
        sites_without = (
            manifest[~manifest["has_png"]]["site_id"].nunique()
        )
        print(f"  {len(manifest) - n_paired:,} chips have none "
              f"(affecting {sites_without} sites)")

    png_dir = cfg.paths["chips_png"]
    if png_dir.exists():
        all_pngs = {p.stem for p in png_dir.glob("*.png")}
        tif_stems = set(manifest["tif_path"].map(lambda p: Path(p).stem))
        orphans = all_pngs - tif_stems
        if orphans:
            print(f"  !! {len(orphans)} PNG(s) with no matching chip in this "
                  f"tile directory, e.g. {sorted(orphans)[:2]}")
