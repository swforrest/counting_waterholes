"""Build the browser-tab icon set in `icons/` from one real prediction tile.

The favicon is a crop of an actual model output rather than a drawn logo, so the
tab shows the same thing the app does: open water in `#1f4ea1` against
surrounding vegetation in `#7f9f5a`. Two details make it survive being shrunk to
16 px, and are the reason this is a script instead of a one-off crop:

1. **The crop is square and centred on the open-water blob**, not on the tile.
   A 151 px tile scaled whole to 16 px is a green square with a smudge in it;
   cropping to the basin first means the water still fills most of the icon.

2. **Small sizes are downsampled with BOX (area averaging), the 180 px Apple
   touch icon with NEAREST.** Averaging keeps the blob's outline readable when
   there are only 16 pixels to say it in; nearest-neighbour at 180 px keeps the
   blocky per-pixel look that is the point of the image.

Any transparent pixels (tiles are padded with alpha 0 where the scene has no
data) are flattened onto the vegetation green first, so the icon never shows a
white notch against a dark browser theme.

Re-run this if the class colours in `data/class_colours.json` change, or to
point at a different tile. Unlike the other tools here it needs Pillow, so run
it from the conda environment rather than the system Python.

Usage
-----
    python tools/make_favicon.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data_sample/overlays/site_002/2024-06_mimal_test_S2_002_S13p52_E134p51_2024-05_pred.webp"
OUT = ROOT / "icons"

# surrounding_vegetation, from data/class_colours.json — the backdrop colour
VEG = (127, 159, 90)

# Square crop of the 151 px source, centred on the open-water blob (which spans
# roughly x 44-98, y 61-90 in the tile above).
CROP = (37, 42, 105, 110)

PNG_SIZES = (16, 32, 48, 64)
ICO_SIZES = (16, 32, 48)
TOUCH_SIZE = 180


def main() -> None:
    OUT.mkdir(exist_ok=True)

    tile = Image.open(SRC).convert("RGBA")
    flat = Image.new("RGBA", tile.size, VEG + (255,))
    flat.alpha_composite(tile)
    crop = flat.crop(CROP).convert("RGB")
    crop.save(OUT / "favicon-source.png")

    for n in PNG_SIZES:
        crop.resize((n, n), Image.BOX).save(OUT / f"favicon-{n}.png")
    crop.resize((TOUCH_SIZE, TOUCH_SIZE), Image.NEAREST).save(OUT / "apple-touch-icon.png")

    # One .ico holding all three legacy sizes, for browsers that ask for it by
    # name rather than reading the <link> tags.
    frames = [Image.open(OUT / f"favicon-{n}.png") for n in ICO_SIZES]
    frames[0].save(
        OUT / "favicon.ico",
        format="ICO",
        sizes=[(n, n) for n in ICO_SIZES],
        append_images=frames[1:],
    )

    for f in sorted(OUT.iterdir()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
