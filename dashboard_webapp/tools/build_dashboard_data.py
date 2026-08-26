"""Turn `cookie-cutting/predictions/` into the web-shaped `data/` the dashboard reads.

The prediction pipeline writes what is convenient for analysis: one 15,708-row CSV
and 47,000 loose images in per-site folders. A browser wants the opposite — a few
small files it can fetch by name, and no directory listing, because a static host
does not offer one. This script is the bridge, and it is the only build step the
project has; the app itself is plain files with no compiler.

Three things it does that are not just copying:

1. **It probes the disk for every overlay** rather than trusting `bounds.json`.
   The WebP migration writes `bounds.json` per site as it goes, so a site being
   converted right now advertises `"image_format": "webp"` while some of its
   months are still PNG. Recording the *actual* extension per site-month per
   layer means a half-migrated archive works, and the app never requests a file
   that is not there.

2. **It records each site's filename stem.** The stem embeds lat/lon tags rounded
   to two decimals (`..._000_S13p54_E134p50_2019-01_pred.webp`) which cannot be
   reconstructed from `site_id`, so the app would otherwise have to guess.

3. **It splits the CSV per site** and builds one compact overview array, so
   drawing a single waterhole does not mean parsing the whole archive.

Standard library only, so it runs under the system Python as well as the conda
environment.

Usage
-----
    python3 tools/build_dashboard_data.py --sites 000,001,002 --out data_sample
    python3 tools/build_dashboard_data.py --sites all --out data --symlink
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# The display layers, in the order the UI offers them. `pred` is the product,
# `rgb` is what the classifier saw, `conf` is how much to believe it.
LAYERS = ("pred", "rgb", "conf")

# Extensions to probe for, best first. The migration is PNG -> WebP, so a site
# that has both (mid-write) should be reported as the WebP one.
EXTENSIONS = ("webp", "png")

# data_quality is ordinal; the app needs the order to shade the quality strip.
QUALITY_ORDER = ("poor", "thin", "fair", "good")

# Fractions are stored as integer permille (0-1000) in overview.json. Three
# significant figures is far beyond what a 150x150 px count can support, and it
# roughly halves the file against writing floats.
PERMILLE = 1000


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def to_float(value):
    """CSV cell -> float or None. Blank cells are real: 103 months are unobserved."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value):
    number = to_float(value)
    return None if number is None else int(number)


def to_bool(value):
    return str(value).strip().lower() == "true"


def round_or_none(value, digits=4):
    return None if value is None else round(value, digits)


def permille(value):
    """Fraction -> 0..1000, or -1 for 'no data'.

    -1 rather than null keeps overview.json a dense array of numbers, which is
    both smaller and faster for the map to scan when recolouring 187 boxes.
    """
    if value is None:
        return -1
    return int(round(value * PERMILLE))


# --------------------------------------------------------------------------
# reading the pipeline's output
# --------------------------------------------------------------------------

def load_class_colours(predictions_dir: Path) -> dict:
    with open(predictions_dir / "class_colours.json") as handle:
        return json.load(handle)


def load_composition(predictions_dir: Path) -> list:
    with open(predictions_dir / "waterhole_composition.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def find_stem_prefix(site_dir: Path):
    """Recover the shared filename stem for a site, minus the `_YYYY-MM_layer.ext`.

    Every file in a site folder shares one prefix, so the first overlay found
    settles it. Returns None for a site whose folder holds no overlays at all.
    """
    for path in sorted(site_dir.iterdir()):
        if path.suffix.lstrip(".") not in EXTENSIONS:
            continue
        stem = path.stem  # e.g. 2024-06_..._000_S13p54_E134p50_2019-01_pred
        parts = stem.rsplit("_", 2)  # -> [prefix, '2019-01', 'pred']
        if len(parts) == 3 and parts[2] in LAYERS:
            return parts[0]
    return None


def probe_layers(site_dir: Path, stem_prefix: str, months: list) -> dict:
    """For each layer, the extension present per month, or None where absent.

    This is the step that makes a mixed PNG/WebP archive safe. One `os.listdir`
    per site rather than 252 `Path.exists()` calls, because this runs over 187
    site folders on a OneDrive-backed filesystem where every stat is a network
    round trip.
    """
    present = set(os.listdir(site_dir))
    layers = {}
    for layer in LAYERS:
        per_month = []
        for month in months:
            found = None
            for extension in EXTENSIONS:
                if f"{stem_prefix}_{month}_{layer}.{extension}" in present:
                    found = extension
                    break
            per_month.append(found)
        layers[layer] = per_month
    return layers


# --------------------------------------------------------------------------
# writing the web-shaped data
# --------------------------------------------------------------------------

def write_json(path: Path, payload, compact=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if compact:
            json.dump(payload, handle, separators=(",", ":"))
        else:
            json.dump(payload, handle, indent=1)


def build_site_json(site_id, rows_by_month, months, class_names, stem_prefix, layers):
    """One site's whole record: identity, geometry, 84 months of composition.

    Every time series is a plain array aligned to `months`, so the app indexes by
    position and never searches. Missing months are null, not dropped — a gap in
    the record is information the brief insists stays visible.
    """
    first = next((rows_by_month[m] for m in months if m in rows_by_month), None)
    if first is None:
        raise ValueError(f"site {site_id} has no rows in the composition table")

    def series(column, cast=to_float, digits=4):
        values = []
        for month in months:
            row = rows_by_month.get(month)
            value = None if row is None else cast(row[column])
            values.append(round_or_none(value, digits) if cast is to_float else value)
        return values

    record = {
        "site_id": site_id,
        "label": first["label"],
        "stem_prefix": stem_prefix,
        "months": months,
        "center": [to_float(first["center_lon"]), to_float(first["center_lat"])],
        "bbox": [
            to_float(first["lon_min"]), to_float(first["lat_min"]),
            to_float(first["lon_max"]), to_float(first["lat_max"]),
        ],
        "bbox_width_m": to_float(first["bbox_width_m"]),
        "bbox_height_m": to_float(first["bbox_height_m"]),
        "has_footprint": to_bool(first["has_footprint"]),
        "footprint_area_ha": to_float(first["footprint_area_ha"]),
        "n_pixels_bbox": to_int(first["n_pixels_bbox"]),
        "n_pixels_footprint": to_int(first["n_pixels_footprint"]),
        "layers": layers,
        "model": {
            "name": first["model_name"],
            "cv_macro_f1": to_float(first["cv_macro_f1"]),
            "config_hash": first["config_hash"],
            "predicted_at": first["predicted_at"],
        },
        # Trust and coverage, on the same axis as the composition.
        "gap_fraction": series("gap_fraction"),
        "mean_n_obs": series("mean_n_obs", digits=2),
        "mean_confidence": series("mean_confidence"),
        "data_quality": series("data_quality", cast=str),
        "flag_isolated_wet": [
            bool(rows_by_month[m] and to_bool(rows_by_month[m]["flag_isolated_wet"]))
            if m in rows_by_month else False
            for m in months
        ],
        "wet_fraction": series("wet_fraction"),
    }

    # Both denominators travel with the site, so the UI can toggle without a
    # second fetch. The brief is explicit that which region a number is counted
    # in is part of what the number means.
    for denominator in ("bbox", "footprint"):
        record[f"{denominator}_n_classified"] = series(
            f"{denominator}_n_classified", cast=to_int
        )
        record[f"{denominator}_frac"] = {
            name: series(f"{denominator}_frac_{name}") for name in class_names
        }

    return record


def build_overview(site_records, months, class_names):
    """One file the map can recolour every box from, for any month.

    Parallel integer arrays rather than 15,708 objects: about a quarter of the
    size, and no per-object property lookup while dragging the month slider.
    Only the bbox denominator is here — it is defined for all 187 sites, and the
    footprint numbers live in the per-site files for when a site is open.
    """
    site_ids = [record["site_id"] for record in site_records]
    quality_index = {name: i for i, name in enumerate(QUALITY_ORDER)}

    overview = {
        "months": months,
        "sites": site_ids,
        "classes": list(class_names),
        "quality_levels": list(QUALITY_ORDER),
        "denominator": "bbox",
        "scale": PERMILLE,
        "note": "fractions are integer permille; -1 means no data for that month",
        "frac": {name: [] for name in class_names},
        "dominant": [],
        "n_classified": [],
        "quality": [],
        "flagged": [],
    }

    for record in site_records:
        fractions = record["bbox_frac"]
        for name in class_names:
            overview["frac"][name].append([permille(v) for v in fractions[name]])

        dominant, classified, quality, flagged = [], [], [], []
        for index in range(len(months)):
            count = record["bbox_n_classified"][index]
            classified.append(count if count is not None else 0)

            if not count:
                dominant.append(-1)
            else:
                values = [fractions[name][index] or 0.0 for name in class_names]
                dominant.append(max(range(len(values)), key=lambda i: values[i]))

            level = record["data_quality"][index]
            quality.append(quality_index.get(level, -1))
            flagged.append(1 if record["flag_isolated_wet"][index] else 0)

        overview["dominant"].append(dominant)
        overview["n_classified"].append(classified)
        overview["quality"].append(quality)
        overview["flagged"].append(flagged)

    return overview


def copy_overlays(site_dir: Path, out_dir: Path, stem_prefix, months, layers, symlink):
    """Place a site's images and bounds.json where the app can fetch them.

    Symlinking is for local development against the full 947 MB archive; the
    release bundle must be copies, because a zip of symlinks is not a zip of
    images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(site_dir / "bounds.json", out_dir / "bounds.json")

    copied = 0
    for layer in LAYERS:
        for month, extension in zip(months, layers[layer]):
            if extension is None:
                continue
            name = f"{stem_prefix}_{month}_{layer}.{extension}"
            destination = out_dir / name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            if symlink:
                destination.symlink_to(os.path.relpath(site_dir / name, out_dir))
            else:
                shutil.copy2(site_dir / name, destination)
            copied += 1
    return copied


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    here = Path(__file__).resolve().parent
    app_root = here.parent
    default_predictions = app_root.parent / "cookie-cutting" / "predictions"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=default_predictions,
                        help="the pipeline's predictions/ directory")
    parser.add_argument("--out", type=Path, default=app_root / "data",
                        help="output directory, relative to the app root")
    parser.add_argument("--sites", default="all",
                        help="comma-separated site ids, or 'all'")
    parser.add_argument("--symlink", action="store_true",
                        help="symlink overlays instead of copying (local dev only)")
    parser.add_argument("--no-overlays", action="store_true",
                        help="skip images entirely; rebuild only the JSON")
    arguments = parser.parse_args(argv)

    predictions_dir = arguments.predictions.resolve()
    out_dir = (arguments.out if arguments.out.is_absolute()
               else app_root / arguments.out).resolve()
    pixel_dir = predictions_dir / "pixel_predictions"

    if not pixel_dir.is_dir():
        parser.error(f"no pixel_predictions/ under {predictions_dir}")

    colours = load_class_colours(predictions_dir)
    class_names = [c["name"] for c in colours["classes"] if not c.get("ignore")]

    print(f"reading  {predictions_dir}")
    rows = load_composition(predictions_dir)
    months = sorted({row["year_month"] for row in rows})

    by_site = {}
    for row in rows:
        by_site.setdefault(row["site_id"], {})[row["year_month"]] = row

    if arguments.sites == "all":
        wanted = sorted(by_site)
    else:
        wanted = [s.strip() for s in arguments.sites.split(",") if s.strip()]
        missing = [s for s in wanted if s not in by_site]
        if missing:
            parser.error(f"unknown site ids: {', '.join(missing)}")

    print(f"writing  {out_dir}")
    print(f"         {len(wanted)} site(s), {len(months)} months "
          f"({months[0]} to {months[-1]})")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_colours.json", "waterhole_boxes.geojson",
                 "waterhole_footprints.geojson"):
        shutil.copy2(predictions_dir / name, out_dir / name.replace("waterhole_", ""))

    # Trim the vector layers to the sites being built, so a sample deploy does
    # not draw 187 boxes that cannot be opened.
    selected = set(wanted)
    if arguments.sites != "all":
        for name in ("boxes.geojson", "footprints.geojson"):
            with open(out_dir / name) as handle:
                collection = json.load(handle)
            collection["features"] = [
                f for f in collection["features"]
                if f["properties"]["site_id"] in selected
            ]
            write_json(out_dir / name, collection)

    site_records = []
    tally = {layer: {"webp": 0, "png": 0, "missing": 0} for layer in LAYERS}
    images_placed = 0
    skipped = []

    for site_id in wanted:
        site_dir = pixel_dir / f"site_{site_id}"
        if not site_dir.is_dir():
            skipped.append((site_id, "no overlay folder"))
            continue

        stem_prefix = find_stem_prefix(site_dir)
        if stem_prefix is None:
            skipped.append((site_id, "no overlays written yet"))
            continue

        layers = probe_layers(site_dir, stem_prefix, months)
        for layer in LAYERS:
            for extension in layers[layer]:
                tally[layer][extension or "missing"] += 1

        record = build_site_json(
            site_id, by_site[site_id], months, class_names, stem_prefix, layers
        )
        site_records.append(record)
        write_json(out_dir / "sites" / f"site_{site_id}.json", record)

        if not arguments.no_overlays:
            images_placed += copy_overlays(
                site_dir, out_dir / "overlays" / f"site_{site_id}",
                stem_prefix, months, layers, arguments.symlink,
            )

        done = len(site_records)
        if done % 10 == 0 or done == len(wanted):
            print(f"         {done}/{len(wanted)} sites, "
                  f"{images_placed} images", flush=True)

    if not site_records:
        print("nothing built — no site had overlays on disk", file=sys.stderr)
        return 1

    write_json(out_dir / "overview.json",
               build_overview(site_records, months, class_names))

    with open(out_dir / "footprints.geojson") as handle:
        footprints = json.load(handle)

    # An empty mean_confidence column is not an error to hide: the badge the
    # brief asks for simply has nothing to draw until the composition table is
    # regenerated, and the app needs to know that rather than render blanks.
    has_confidence = any(
        value is not None
        for record in site_records for value in record["mean_confidence"]
    )

    write_json(out_dir / "manifest.json", {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(predictions_dir),
        "subset": arguments.sites != "all",
        "months": months,
        "sites": [record["site_id"] for record in site_records],
        "classes": colours["classes"],
        "class_names": class_names,
        "confidence": colours["confidence"],
        "quality_levels": list(QUALITY_ORDER),
        "denominators": ["bbox", "footprint"],
        "sites_without_footprint": footprints.get("sites_without_footprint", []),
        "layer_counts": tally,
        "has_mean_confidence": has_confidence,
        "overlays_symlinked": bool(arguments.symlink),
    }, compact=False)

    print(f"\nbuilt {len(site_records)} site(s), {images_placed} images")
    for layer in LAYERS:
        counts = tally[layer]
        total = counts["webp"] + counts["png"]
        print(f"  {layer:5s} {total:6d} present "
              f"({counts['webp']} webp, {counts['png']} png), "
              f"{counts['missing']} missing")
    if not has_confidence:
        print("\n  note: mean_confidence is empty in the composition table, so the\n"
              "        per-site confidence badge will stay hidden.")
    for site_id, reason in skipped:
        print(f"  skipped site {site_id}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
