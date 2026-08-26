"""Verify a built data directory before serving or deploying it.

The failure mode this guards against is a site that loads, looks right, and is
quietly wrong — an overlay the app will request and not find, a composition row
whose fractions do not sum, a site listed on the map that has no record behind
it. All of those produce a blank panel or a missing image rather than an error,
which is a slow way to discover them.

Run it after every build, and before cutting a release bundle.

    python3 tools/check_data.py                  # checks data/
    python3 tools/check_data.py --data data_sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = ("pred", "rgb", "conf")


def main(argv=None):
    app_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=app_root / "data")
    parser.add_argument("--sample-overlays", type=int, default=0,
                        help="check only N overlays per site (0 = all)")
    arguments = parser.parse_args(argv)

    root = (arguments.data if arguments.data.is_absolute()
            else app_root / arguments.data).resolve()

    problems = []
    notes = []

    def require(path):
        if not path.exists():
            problems.append(f"missing {path.relative_to(root)}")
            return None
        with open(path) as handle:
            return json.load(handle)

    manifest = require(root / "manifest.json")
    if manifest is None:
        print(f"FAIL: {root} is not a built data directory")
        return 1

    overview = require(root / "overview.json")
    boxes = require(root / "boxes.geojson")
    footprints = require(root / "footprints.geojson")
    require(root / "class_colours.json")
    if None in (overview, boxes, footprints):
        print("FAIL: core files missing")
        return 1

    months = manifest["months"]
    class_names = manifest["class_names"]
    sites = manifest["sites"]

    # 1. The map and the data must agree on which sites exist. A box with no
    #    record behind it opens an empty panel.
    box_ids = {f["properties"]["site_id"] for f in boxes["features"]}
    if box_ids != set(sites):
        problems.append(
            f"boxes.geojson has {len(box_ids)} sites, manifest has {len(sites)}; "
            f"only in boxes: {sorted(box_ids - set(sites))[:5]}, "
            f"only in manifest: {sorted(set(sites) - box_ids)[:5]}"
        )

    if overview["sites"] != sites:
        problems.append("overview.json site order differs from the manifest")
    if overview["months"] != months:
        problems.append("overview.json months differ from the manifest")

    # 2. Footprints are optional, but must be optional in a declared way.
    footprint_ids = {f["properties"]["site_id"] for f in footprints["features"]}
    declared_missing = set(manifest["sites_without_footprint"])
    unexplained = set(sites) - footprint_ids - declared_missing
    if unexplained:
        problems.append(
            f"{len(unexplained)} site(s) have no footprint and are not listed in "
            f"sites_without_footprint: {sorted(unexplained)[:5]}"
        )

    checked_overlays = 0
    missing_overlays = []
    bad_sums = []
    format_mix = {"webp": 0, "png": 0}

    for site_id in sites:
        record = require(root / "sites" / f"site_{site_id}.json")
        if record is None:
            continue

        if record["months"] != months:
            problems.append(f"site {site_id}: month axis differs from the manifest")
            continue

        overlay_dir = root / "overlays" / f"site_{site_id}"
        if require(overlay_dir / "bounds.json") is None:
            continue

        # 3. Every overlay the app can construct a URL for must exist. The app
        #    only asks for those the build recorded, so a mismatch here is a
        #    404 the user would see as a blank map.
        for layer in LAYERS:
            extensions = record["layers"].get(layer, [])
            indices = range(len(months))
            if arguments.sample_overlays:
                step = max(1, len(months) // arguments.sample_overlays)
                indices = range(0, len(months), step)
            for index in indices:
                extension = extensions[index] if index < len(extensions) else None
                if extension is None:
                    continue
                format_mix[extension] = format_mix.get(extension, 0) + 1
                name = f"{record['stem_prefix']}_{months[index]}_{layer}.{extension}"
                checked_overlays += 1
                if not (overlay_dir / name).exists():
                    missing_overlays.append(f"site {site_id}: {name}")

        # 4. Composition must be a composition: fractions over a denominator
        #    sum to 1 wherever anything was classified.
        for denominator in ("bbox", "footprint"):
            classified = record[f"{denominator}_n_classified"]
            fractions = record[f"{denominator}_frac"]
            for index, count in enumerate(classified):
                if not count:
                    continue
                total = sum(fractions[name][index] or 0.0 for name in class_names)
                if abs(total - 1.0) > 1e-3:
                    bad_sums.append(
                        f"site {site_id} {months[index]} {denominator}: sum={total:.4f}"
                    )

    problems.extend(missing_overlays[:10])
    if len(missing_overlays) > 10:
        problems.append(f"...and {len(missing_overlays) - 10} more missing overlays")
    problems.extend(bad_sums[:10])
    if len(bad_sums) > 10:
        problems.append(f"...and {len(bad_sums) - 10} more bad fraction sums")

    # Observations that are not failures but are worth seeing.
    if format_mix.get("png") and format_mix.get("webp"):
        notes.append(
            f"mixed image formats: {format_mix['webp']} webp, {format_mix['png']} png "
            "(the WebP conversion is still in progress; the app handles both)"
        )
    if not manifest.get("has_mean_confidence"):
        notes.append("mean_confidence is empty — the confidence badge will be hidden")
    if manifest.get("overlays_symlinked"):
        notes.append("overlays are symlinks — fine locally, but cannot be zipped for release")
    if manifest.get("subset"):
        notes.append(f"this is a {len(sites)}-site subset, not the full archive")

    print(f"checked {root}")
    print(f"  {len(sites)} sites, {len(months)} months, {checked_overlays} overlay paths")
    for note in notes:
        print(f"  note: {note}")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
