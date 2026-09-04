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
    python3 tools/make_release_bundle.py --publish       # zip, then upload it

`--publish` creates the release on first use and clobbers the asset on the tag
thereafter, so the same command works for the first upload and every rebuild.
It always passes an explicit `--repo`; see resolve_repo below for why that is
not optional here.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

# The tag the deploy workflow fetches by default (see .github/workflows/
# deploy-dashboard.yml, input `data_tag`). Changing this means changing both.
DEFAULT_TAG = "dashboard-data-v1"

# GitHub publishes a Pages site of at most 1 GB, and the workflow unzips this
# asset *into* the site, so the uncompressed total is what counts against it —
# not the zip. Warn well before the wall rather than failing in Actions.
PAGES_LIMIT = 1024 ** 3
PAGES_WARN_AT = 0.85 * PAGES_LIMIT


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def resolve_repo(explicit: str | None) -> str:
    """The `owner/name` to publish to, never left for `gh` to infer.

    This checkout is a fork (`swforrest/counting_waterholes`) and also carries an
    `upstream` remote pointing at the project it was adapted from. Given a fork,
    `gh` resolves an unqualified command to the *parent*, so a bare
    `gh release create` uploads to the upstream repository instead of this one.
    Deriving the target from `origin` and passing it explicitly removes the
    guess entirely.
    """
    if explicit:
        return explicit

    result = subprocess.run(["git", "remote", "get-url", "origin"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("no `origin` remote — pass --repo owner/name")

    # https://github.com/owner/name.git  |  git@github.com:owner/name.git
    match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?/?$", result.stdout.strip())
    if not match:
        raise SystemExit(f"cannot parse owner/name from origin: {result.stdout.strip()}"
                         "\npass --repo owner/name")
    return f"{match.group(1)}/{match.group(2)}"


def publish(zip_path: Path, tag: str, repo: str, notes: str) -> None:
    """Create the release, or replace the asset if the tag already exists."""
    if shutil.which("gh") is None:
        raise SystemExit("gh is not installed — see https://cli.github.com")

    exists = subprocess.run(["gh", "release", "view", tag, "--repo", repo],
                            capture_output=True).returncode == 0

    if exists:
        print(f"\nreplacing the asset on {repo} tag {tag} ...")
        command = ["gh", "release", "upload", tag, str(zip_path),
                   "--clobber", "--repo", repo]
    else:
        print(f"\ncreating release {tag} on {repo} ...")
        command = ["gh", "release", "create", tag, str(zip_path), "--repo", repo,
                   "--title", "Dashboard data", "--notes", notes]

    if subprocess.run(command).returncode != 0:
        raise SystemExit("upload failed")

    print(f"\nuploaded. Deploy the site with:\n"
          f"  gh workflow run deploy-dashboard.yml --repo {repo}")


def main(argv=None):
    app_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=app_root / "data",
                        help="the built data directory to bundle")
    parser.add_argument("--out", type=Path,
                        default=app_root / "dist" / "dashboard-data.zip")
    parser.add_argument("--publish", action="store_true",
                        help="upload the zip to the GitHub release after building it")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"release tag to publish to (default: {DEFAULT_TAG})")
    parser.add_argument("--repo", default=None,
                        help="owner/name to publish to (default: derived from origin)")
    arguments = parser.parse_args(argv)

    # Resolved before the zip is built: a bad --repo should fail in a second,
    # not after several minutes of archiving.
    repo = resolve_repo(arguments.repo) if arguments.publish else None

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
    if raw > PAGES_WARN_AT:
        print(f"  WARNING: {human(raw)} unpacks into the Pages site, which is capped\n"
              f"           at {human(PAGES_LIMIT)}. Converting the remaining PNG overlays\n"
              f"           to WebP is what brings this down.")

    notes = (f"{len(files)} files, {human(raw)} unpacked. "
             f"Built from {data_dir.name} by tools/make_release_bundle.py.")
    if arguments.publish:
        publish(out_path, arguments.tag, repo, notes)
    else:
        print(f"\nPublish with:\n"
              f"  python3 tools/make_release_bundle.py --data {arguments.data} "
              f"--out {out_path} --publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
