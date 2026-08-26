"""Zip the built `data/` directory into a release asset.

Why a release asset rather than committing the images:

The full overlay set is ~220 MB across ~47,000 files. Committing that to the
repository would work exactly once. Git keeps every version of every file
forever, so each regeneration of the predictions would add another 220 MB to the
history that every future clone has to download — on top of the ~300 MB this
repository already carries. Release assets live outside the git object store:
they are downloaded on demand, replaced in place, and never touch a clone.

The deploy workflow fetches this zip and unpacks it next to the app, so the
published site has the full dataset while `main` stays small.

Usage
-----
    python3 tools/make_release_bundle.py                 # zips data/
    python3 tools/make_release_bundle.py --data data_sample --out sample.zip

Then, from the repository root:

    gh release create dashboard-data-v1 dashboard_webapp/dist/dashboard-data.zip \\
        --title "Dashboard data v1" --notes "187 sites, 2019-01 to 2025-12"

To replace the data for an existing tag:

    gh release upload dashboard-data-v1 dashboard_webapp/dist/dashboard-data.zip --clobber
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def main(argv=None):
    app_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=app_root / "data",
                        help="the built data directory to bundle")
    parser.add_argument("--out", type=Path,
                        default=app_root / "dist" / "dashboard-data.zip")
    arguments = parser.parse_args(argv)

    data_dir = (arguments.data if arguments.data.is_absolute()
                else app_root / arguments.data).resolve()
    out_path = (arguments.out if arguments.out.is_absolute()
                else app_root / arguments.out).resolve()

    if not (data_dir / "manifest.json").exists():
        parser.error(f"{data_dir} has no manifest.json — run build_dashboard_data.py first")

    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in files):
        parser.error(
            "the data directory contains symlinks (built with --symlink).\n"
            "Rebuild without it: a zip of symlinks is not a zip of images."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = sum(p.stat().st_size for p in files)
    print(f"bundling {len(files)} files ({human(raw)}) from {data_dir}")

    # The overlays are already compressed (WebP/PNG), so re-deflating them costs
    # minutes and saves almost nothing. Store images, deflate the JSON.
    with zipfile.ZipFile(out_path, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            stored = path.relative_to(data_dir).as_posix()
            compression = (zipfile.ZIP_STORED
                           if path.suffix in (".webp", ".png", ".tif")
                           else zipfile.ZIP_DEFLATED)
            archive.write(path, f"data/{stored}", compress_type=compression)
            if index % 5000 == 0:
                print(f"  {index}/{len(files)}", flush=True)

    size = out_path.stat().st_size
    print(f"\nwrote {out_path}  ({human(size)})")
    if size > 2 * 1024 ** 3:
        print("  WARNING: over GitHub's 2 GB per-asset limit.")
    print("\nUpload with:\n"
          f"  gh release create dashboard-data-v1 {out_path} \\\n"
          f"      --title 'Dashboard data v1' --notes 'built from predictions/'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
