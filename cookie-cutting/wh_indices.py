"""Spectral indices, as pure functions of band arrays.

Inputs are surface reflectance (already divided by 10000 by the GEE export) as
float arrays with NaN wherever the monthly composite has no observation.
wh_tiles.read_tile does that conversion; nothing here re-invents a nodata rule.

NaN propagates: a pixel with no observation gets NaN in every index, rather than
a plausible-looking value derived from a fill value.

A note on why this module carries more than the original spec asked for:
MNDWI is not a reliable wetness proxy at these sites, because emergent sedges
and melaleuca routinely cover standing water and drag the index far negative.
NDMI and the red-edge index are included because sedge-over-water separates from
dry sedge in the SWIR and red edge much better than in any green-based water
index. Treat every water index here as a feature, never as an answer.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

# Denominators smaller than this are treated as degenerate and give NaN.
# Reflectance sums this small do not occur in real data.
_MIN_DENOMINATOR = 1e-6

BandArrays = Mapping[str, np.ndarray]


def normalised_difference(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(high - low) / (high + low), NaN where the denominator degenerates.

    Scale-invariant, so it does not matter whether the inputs are reflectance or
    raw DN — unlike the AWEI variants below, which do care.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    denominator = high + low
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (high - low) / denominator
    return np.where(np.abs(denominator) < _MIN_DENOMINATOR, np.nan, result)


def mndwi(bands: BandArrays) -> np.ndarray:
    """Modified NDWI (Xu 2006): green vs SWIR1. High = open water."""
    return normalised_difference(bands["B3"], bands["B11"])


def ndwi(bands: BandArrays) -> np.ndarray:
    """NDWI (McFeeters 1996): green vs NIR. High = open water."""
    return normalised_difference(bands["B3"], bands["B8"])


def ndvi(bands: BandArrays) -> np.ndarray:
    """NDVI: NIR vs red. High = dense green vegetation."""
    return normalised_difference(bands["B8"], bands["B4"])


def ndti(bands: BandArrays) -> np.ndarray:
    """Normalised Difference Turbidity Index: red vs green. High = turbid."""
    return normalised_difference(bands["B4"], bands["B3"])


def ndmi(bands: BandArrays) -> np.ndarray:
    """NDMI / NDII: NIR vs SWIR1. High = moist canopy or soil.

    Unlike MNDWI this responds to water *under* a vegetation canopy, which is
    the case MNDWI gets wrong at these sites.
    """
    return normalised_difference(bands["B8"], bands["B11"])


def nd_rededge(bands: BandArrays) -> np.ndarray:
    """Red-edge normalised difference: B8A vs B5.

    Separates stressed/senescent emergent vegetation from vigorous growth, which
    is the aquatic-vegetation vs dry-sedge distinction.
    """
    return normalised_difference(bands["B8A"], bands["B5"])


def red_green_ratio(bands: BandArrays) -> np.ndarray:
    """Red / green. A simple turbidity ratio; rises with suspended sediment."""
    red = np.asarray(bands["B4"], dtype=np.float64)
    green = np.asarray(bands["B3"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = red / green
    return np.where(np.abs(green) < _MIN_DENOMINATOR, np.nan, result)


def awei_nsh(bands: BandArrays) -> np.ndarray:
    """AWEI, no-shadow variant (Feyisa et al. 2014).

    4*(green - SWIR1) - (0.25*NIR + 2.75*SWIR2).
    Scale-dependent: only valid on reflectance, not on DN.
    """
    green = np.asarray(bands["B3"], dtype=np.float64)
    nir = np.asarray(bands["B8"], dtype=np.float64)
    swir1 = np.asarray(bands["B11"], dtype=np.float64)
    swir2 = np.asarray(bands["B12"], dtype=np.float64)
    return 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)


def awei_sh(bands: BandArrays) -> np.ndarray:
    """AWEI, shadow variant (Feyisa et al. 2014).

    blue + 2.5*green - 1.5*(NIR + SWIR1) - 0.25*SWIR2.
    Scale-dependent: only valid on reflectance, not on DN.
    """
    blue = np.asarray(bands["B2"], dtype=np.float64)
    green = np.asarray(bands["B3"], dtype=np.float64)
    nir = np.asarray(bands["B8"], dtype=np.float64)
    swir1 = np.asarray(bands["B11"], dtype=np.float64)
    swir2 = np.asarray(bands["B12"], dtype=np.float64)
    return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2


INDEX_FUNCTIONS: dict[str, Callable[[BandArrays], np.ndarray]] = {
    "mndwi": mndwi,
    "ndwi": ndwi,
    "ndvi": ndvi,
    "ndti": ndti,
    "ndmi": ndmi,
    "nd_rededge": nd_rededge,
    "red_green_ratio": red_green_ratio,
    "awei_nsh": awei_nsh,
    "awei_sh": awei_sh,
}

# Bands each index consumes, so a missing band is caught before the maths.
INDEX_BANDS: dict[str, tuple[str, ...]] = {
    "mndwi": ("B3", "B11"),
    "ndwi": ("B3", "B8"),
    "ndvi": ("B8", "B4"),
    "ndti": ("B4", "B3"),
    "ndmi": ("B8", "B11"),
    "nd_rededge": ("B8A", "B5"),
    "red_green_ratio": ("B4", "B3"),
    "awei_nsh": ("B3", "B8", "B11", "B12"),
    "awei_sh": ("B2", "B3", "B8", "B11", "B12"),
}


def compute(name: str, bands: BandArrays) -> np.ndarray:
    """Compute one index by name."""
    if name not in INDEX_FUNCTIONS:
        raise KeyError(
            f"unknown index {name!r}; known: {sorted(INDEX_FUNCTIONS)}"
        )

    missing = [band for band in INDEX_BANDS[name] if band not in bands]
    if missing:
        raise KeyError(f"index {name!r} needs bands {missing} which were not supplied")

    return INDEX_FUNCTIONS[name](bands)


def compute_many(names: Sequence[str], bands: BandArrays) -> dict[str, np.ndarray]:
    """Compute several indices, returning {name: array}."""
    return {name: compute(name, bands) for name in names}
