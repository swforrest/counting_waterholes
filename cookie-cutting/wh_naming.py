"""Parse and build chip filenames.

Convention, set by cookie-cutting_S2_download.ipynb:

    <prefix>_S2_<site>_<lat>_<lon>_<YYYY-MM>.<ext>
    2024-06_mimal_test_S2_019_S13p40_E134p47_2024-09.tif

The lat/lon tags are rounded to two decimals and exist to make filenames
readable; they are NOT a usable position. site_id is the authoritative key, and
the georeferencing in the GeoTIFF is the authoritative position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# <prefix>_S2_<3-digit site>_<lat tag>_<lon tag>_<YYYY-MM>
TILE_STEM_PATTERN = re.compile(
    r"^(?P<prefix>.+)_S2_"
    r"(?P<site_id>\d{3})_"
    r"(?P<lat_tag>[NS]\d+p\d+)_"
    r"(?P<lon_tag>[EW]\d+p\d+)_"
    r"(?P<year>\d{4})-(?P<month>\d{2})$"
)

_COORD_TAG_PATTERN = re.compile(r"^(?P<hemisphere>[NSEW])(?P<degrees>\d+)p(?P<fraction>\d+)$")


@dataclass(frozen=True, order=True)
class TileKey:
    """Identity of one site-month chip."""

    site_id: str
    year: int
    month: int
    lat: float
    lon: float
    prefix: str

    @property
    def year_month(self) -> str:
        """'YYYY-MM', the form used in filenames, manifests and output tables."""
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def month_index(self) -> int:
        """Months since year 0. Gives a sortable, gap-aware time axis."""
        return self.year * 12 + (self.month - 1)

    def stem(self) -> str:
        """Rebuild the filename stem this key came from."""
        return (
            f"{self.prefix}_S2_{self.site_id}_"
            f"{format_coord_tag(self.lat, is_latitude=True)}_"
            f"{format_coord_tag(self.lon, is_latitude=False)}_"
            f"{self.year_month}"
        )


def parse_coord_tag(tag: str) -> float:
    """'S13p54' -> -13.54, 'E134p50' -> 134.50."""
    match = _COORD_TAG_PATTERN.match(tag)
    if match is None:
        raise ValueError(f"not a coordinate tag: {tag!r}")

    value = float(f"{match['degrees']}.{match['fraction']}")
    return -value if match["hemisphere"] in ("S", "W") else value


def format_coord_tag(value: float, is_latitude: bool) -> str:
    """-13.54 -> 'S13p54' (latitude), 134.50 -> 'E134p50' (longitude).

    Mirrors coord_tag() in cookie-cutting_S2_download.ipynb, including its two
    decimal places, so a parsed key round-trips to the filename it came from.
    """
    if is_latitude:
        hemisphere = "S" if value < 0 else "N"
    else:
        hemisphere = "W" if value < 0 else "E"
    return f"{hemisphere}{abs(value):.2f}".replace(".", "p")


def parse_stem(stem: str) -> TileKey:
    """Parse a filename stem into a TileKey. Raises on anything unrecognised."""
    match = TILE_STEM_PATTERN.match(stem)
    if match is None:
        raise ValueError(
            f"filename does not match the chip convention "
            f"'<prefix>_S2_<site>_<lat>_<lon>_<YYYY-MM>': {stem!r}"
        )

    month = int(match["month"])
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range in {stem!r}: {month}")

    return TileKey(
        site_id=match["site_id"],
        year=int(match["year"]),
        month=month,
        lat=parse_coord_tag(match["lat_tag"]),
        lon=parse_coord_tag(match["lon_tag"]),
        prefix=match["prefix"],
    )


def parse_path(path: str | Path) -> TileKey:
    """Parse any chip path (.tif, .png, .json) into a TileKey."""
    return parse_stem(Path(path).stem)


def try_parse_path(path: str | Path) -> TileKey | None:
    """Like parse_path but returns None instead of raising.

    Only for walking directories that may hold unrelated files. Anywhere a chip
    is expected, use parse_path and let a bad name raise.
    """
    try:
        return parse_path(path)
    except ValueError:
        return None


def sibling_path(path: str | Path, directory: str | Path, extension: str) -> Path:
    """The same chip's file in another directory, e.g. the PNG beside a TIF.

    Relies on the export and the renderer sharing a filename stem, which is what
    keeps chip.tif -> chip.png -> chip.json aligned.
    """
    return Path(directory) / f"{Path(path).stem}{extension}"
