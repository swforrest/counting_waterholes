"""Tests for spectral indices, against values computed by hand."""

import numpy as np
import pytest

import wh_indices

# One pixel, chosen so every expected value below is exact or easy to verify.
#   B2 blue 0.04, B3 green 0.06, B4 red 0.10, B5 red-edge 0.12,
#   B8 NIR 0.30, B8A NIR-narrow 0.28, B11 SWIR1 0.24, B12 SWIR2 0.05
PIXEL = {
    "B2": np.array([[0.04]]),
    "B3": np.array([[0.06]]),
    "B4": np.array([[0.10]]),
    "B5": np.array([[0.12]]),
    "B8": np.array([[0.30]]),
    "B8A": np.array([[0.28]]),
    "B11": np.array([[0.24]]),
    "B12": np.array([[0.05]]),
}


@pytest.mark.parametrize(
    "name,expected",
    [
        # (0.06 - 0.24) / 0.30
        ("mndwi", -0.6),
        # (0.06 - 0.30) / 0.36
        ("ndwi", -2.0 / 3.0),
        # (0.30 - 0.10) / 0.40
        ("ndvi", 0.5),
        # (0.10 - 0.06) / 0.16
        ("ndti", 0.25),
        # (0.30 - 0.24) / 0.54
        ("ndmi", 1.0 / 9.0),
        # (0.28 - 0.12) / 0.40
        ("nd_rededge", 0.4),
        # 0.10 / 0.06
        ("red_green_ratio", 5.0 / 3.0),
        # 4*(0.06 - 0.24) - (0.25*0.30 + 2.75*0.05)
        ("awei_nsh", -0.9325),
        # 0.04 + 2.5*0.06 - 1.5*(0.30 + 0.24) - 0.25*0.05
        ("awei_sh", -0.6325),
    ],
)
def test_index_values_match_hand_calculation(name, expected):
    result = wh_indices.compute(name, PIXEL)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected)


def test_normalised_difference_is_scale_invariant():
    """ND indices give the same answer on DN as on reflectance."""
    scaled = {name: value * 10000 for name, value in PIXEL.items()}
    assert wh_indices.compute("ndvi", scaled)[0, 0] == pytest.approx(
        wh_indices.compute("ndvi", PIXEL)[0, 0]
    )


def test_awei_is_not_scale_invariant():
    """AWEI is a weighted sum, so it is only meaningful on reflectance.

    Pinned deliberately: it is the trap if the export ever stops dividing by 10000.
    """
    scaled = {name: value * 10000 for name, value in PIXEL.items()}
    assert wh_indices.compute("awei_sh", scaled)[0, 0] != pytest.approx(
        wh_indices.compute("awei_sh", PIXEL)[0, 0]
    )


def test_nan_input_propagates():
    bands = {name: value.copy() for name, value in PIXEL.items()}
    bands["B11"] = np.array([[np.nan]])
    assert np.isnan(wh_indices.compute("mndwi", bands)[0, 0])
    # An index that does not use B11 is unaffected.
    assert not np.isnan(wh_indices.compute("ndvi", bands)[0, 0])


def test_degenerate_denominator_gives_nan_not_a_number():
    bands = {"B3": np.array([[0.0]]), "B11": np.array([[0.0]])}
    assert np.isnan(wh_indices.compute("mndwi", bands)[0, 0])


def test_zero_green_gives_nan_ratio():
    bands = {"B4": np.array([[0.1]]), "B3": np.array([[0.0]])}
    assert np.isnan(wh_indices.compute("red_green_ratio", bands)[0, 0])


def test_normalised_difference_bounds():
    rng = np.random.default_rng(0)
    high = rng.uniform(0.01, 0.5, size=(20, 20))
    low = rng.uniform(0.01, 0.5, size=(20, 20))
    result = wh_indices.normalised_difference(high, low)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_unknown_index_raises():
    with pytest.raises(KeyError, match="unknown index"):
        wh_indices.compute("ndxi", PIXEL)


def test_missing_band_raises_before_the_maths():
    with pytest.raises(KeyError, match="needs bands"):
        wh_indices.compute("mndwi", {"B3": np.array([[0.06]])})


def test_compute_many_returns_every_requested_index():
    names = ["mndwi", "ndvi", "ndmi"]
    result = wh_indices.compute_many(names, PIXEL)
    assert sorted(result) == sorted(names)


def test_every_registered_index_declares_its_bands():
    assert sorted(wh_indices.INDEX_FUNCTIONS) == sorted(wh_indices.INDEX_BANDS)
    for name, bands in wh_indices.INDEX_BANDS.items():
        assert wh_indices.compute(name, PIXEL) is not None
        assert all(band in PIXEL for band in bands), name
