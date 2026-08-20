"""Tests for local context windows and feature assembly."""

import numpy as np
import pandas as pd
import pytest

import wh_features


# --- local_context: hand-computed values ----------------------------------


def test_mean_of_a_uniform_field_is_the_value():
    values = np.full((7, 7), 0.4)
    result = wh_features.local_context(values, 3, "mean")
    assert np.allclose(result, 0.4)


def test_sd_of_a_uniform_field_is_zero():
    values = np.full((7, 7), 0.4)
    result = wh_features.local_context(values, 3, "sd")
    assert np.allclose(result, 0.0, atol=1e-12)


def test_mean_matches_hand_calculation_at_the_centre():
    # 3x3 window over the centre of a 3x3 field: mean of 1..9 is 5.
    values = np.arange(1, 10, dtype=float).reshape(3, 3)
    result = wh_features.local_context(values, 3, "mean")
    assert result[1, 1] == pytest.approx(5.0)


def test_sd_matches_hand_calculation_at_the_centre():
    values = np.arange(1, 10, dtype=float).reshape(3, 3)
    result = wh_features.local_context(values, 3, "sd")
    # Population SD of 1..9 is sqrt(60/9).
    assert result[1, 1] == pytest.approx(np.sqrt(60.0 / 9.0))


def test_window_of_one_returns_the_input():
    values = np.arange(1, 10, dtype=float).reshape(3, 3)
    assert np.allclose(wh_features.local_context(values, 1, "mean"), values)


# --- the NaN case, which is the whole reason this function exists ----------


def test_nan_is_excluded_from_the_mean_not_treated_as_zero():
    """A naive uniform_filter would give 40/9; the answer is the mean of what was seen."""
    values = np.full((3, 3), 5.0)
    values[0, 0] = np.nan

    result = wh_features.local_context(values, 3, "mean")

    assert result[1, 1] == pytest.approx(5.0)          # mean of the 8 observed pixels
    assert result[1, 1] != pytest.approx(40.0 / 9.0)   # what zero-filling would give


def test_nan_excluded_from_sd():
    values = np.full((3, 3), 5.0)
    values[0, 0] = np.nan
    result = wh_features.local_context(values, 3, "sd")
    assert result[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_window_with_no_observations_is_nan():
    values = np.full((5, 5), np.nan)
    result = wh_features.local_context(values, 3, "mean")
    assert np.all(np.isnan(result))


def test_partially_observed_window_uses_only_observed_pixels():
    values = np.full((3, 3), np.nan)
    values[1, 1] = 2.0
    values[1, 2] = 4.0
    result = wh_features.local_context(values, 3, "mean")
    assert result[1, 1] == pytest.approx(3.0)


def test_nan_pixel_still_gets_a_context_value_from_its_neighbours():
    """The centre pixel being unobserved does not stop its window having a mean."""
    values = np.full((3, 3), 6.0)
    values[1, 1] = np.nan
    result = wh_features.local_context(values, 3, "mean")
    assert result[1, 1] == pytest.approx(6.0)


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize("window", [0, -1, 2, 4])
def test_even_or_nonpositive_windows_are_rejected(window):
    with pytest.raises(ValueError, match="positive odd"):
        wh_features.local_context(np.zeros((5, 5)), window, "mean")


def test_unknown_statistic_is_rejected():
    with pytest.raises(ValueError, match="mean"):
        wh_features.local_context(np.zeros((5, 5)), 3, "median")


def test_larger_window_smooths_more():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=(41, 41))
    narrow = wh_features.local_context(values, 3, "mean")
    wide = wh_features.local_context(values, 9, "mean")
    assert np.nanstd(wide) < np.nanstd(narrow)


# --- assembly -------------------------------------------------------------


class _FakeTile:
    """Minimal stand-in for wh_tiles.Tile."""

    def __init__(self, shape=(6, 6)):
        self.shape = shape
        rng = np.random.default_rng(1)
        self.bands = {
            name: rng.uniform(0.02, 0.4, size=shape)
            for name in ("B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12")
        }
        self.n_obs = np.full(shape, 4, dtype=np.int16)
        self.valid = np.ones(shape, dtype=bool)


PARAMS = wh_features.FeatureParams(
    reflectance_bands=("B3", "B4", "B8"),
    indices=("mndwi", "ndvi"),
    context_windows=(3,),
    context_indices=("mndwi",),
    temporal_indices=("mndwi",),
)


def test_instantaneous_features_cover_every_requested_input():
    features = wh_features.instantaneous_features(_FakeTile(), PARAMS)
    for expected in ("refl_B3", "refl_B4", "refl_B8", "mndwi", "ndvi",
                     "mndwi_mean3", "mndwi_sd3"):
        assert expected in features, expected


def test_n_obs_is_excluded_by_default():
    """n_obs describes the observing system, not the ground; off unless asked for.

    Wet months have both fewer clear scenes and more water, so including it lets
    a classifier learn "few observations therefore wet" instead of reading the
    surface.
    """
    assert not wh_features.FeatureParams().include_n_obs
    features = wh_features.instantaneous_features(_FakeTile(), PARAMS)
    assert "n_obs" not in features
    assert "n_obs" not in wh_features.instantaneous_feature_names(PARAMS)


def test_n_obs_is_included_when_explicitly_asked_for():
    params = wh_features.FeatureParams(
        reflectance_bands=("B3",), indices=("mndwi",),
        context_windows=(3,), context_indices=("mndwi",),
        temporal_indices=("mndwi",), include_n_obs=True,
    )
    features = wh_features.instantaneous_features(_FakeTile(), params)
    assert "n_obs" in features
    assert "n_obs" in wh_features.instantaneous_feature_names(params)


def test_missing_band_raises():
    tile = _FakeTile()
    del tile.bands["B3"]
    with pytest.raises(KeyError, match="B3"):
        wh_features.instantaneous_features(tile, PARAMS)


def test_assemble_broadcasts_per_pixel_and_indexes_per_month():
    tile = _FakeTile()
    temporal = {
        "mndwi_wet_max": np.full(tile.shape, 0.5),                 # per-pixel
        "mndwi_rank": np.stack([np.full(tile.shape, float(t)) for t in range(4)]),
    }

    features = wh_features.assemble_features(tile, temporal, month_position=2, params=PARAMS)

    assert np.allclose(features["mndwi_wet_max"], 0.5)
    assert np.allclose(features["mndwi_rank"], 2.0)
    assert features["mndwi_rank"].shape == tile.shape


def test_assemble_rejects_an_out_of_range_month():
    tile = _FakeTile()
    temporal = {"mndwi_rank": np.zeros((3, *tile.shape))}
    with pytest.raises(IndexError, match="month_position"):
        wh_features.assemble_features(tile, temporal, month_position=5, params=PARAMS)


def test_extract_pixels_carries_the_grouping_columns():
    tile = _FakeTile()
    features = wh_features.instantaneous_features(tile, PARAMS)
    selection = np.zeros(tile.shape, dtype=bool)
    selection[1:3, 1:3] = True
    class_ids = np.full(tile.shape, 4, dtype=np.uint8)

    table = wh_features.extract_pixels(features, selection, "025", "2024-08", class_ids)

    assert len(table) == 4
    assert set(table["site_id"]) == {"025"}
    assert set(table["year_month"]) == {"2024-08"}
    assert set(table["class_id"]) == {4}
    assert "mndwi" in table.columns


def test_extract_pixels_with_nothing_selected_returns_an_empty_frame():
    tile = _FakeTile()
    features = wh_features.instantaneous_features(tile, PARAMS)
    table = wh_features.extract_pixels(
        features, np.zeros(tile.shape, dtype=bool), "025", "2024-08"
    )
    assert table.empty


def test_feature_columns_excludes_identifiers():
    table = pd.DataFrame({
        "site_id": ["025"], "year_month": ["2024-08"], "row": [1], "col": [2],
        "class_id": [3], "source": ["manual"], "mndwi": [0.1], "ndvi": [0.5],
    })
    assert sorted(wh_features.feature_columns(table)) == ["mndwi", "ndvi"]


# --- AlphaEarth composite -------------------------------------------------


def _fake_embedding(shape=(8, 8), n_bands=64):
    rng = np.random.default_rng(3)
    return {
        f"{wh_features.ALPHAEARTH_PREFIX}A{i:02d}": rng.normal(0, 0.1, size=shape)
        for i in range(n_bands)
    }


def test_band_composite_is_three_channels_in_unit_range():
    image = wh_features.alphaearth_composite(_fake_embedding())
    assert image.shape == (8, 8, 3)
    assert np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0


def test_band_composite_uses_the_bands_it_was_given():
    embedding = _fake_embedding()
    a = wh_features.alphaearth_composite(embedding, bands=("A00", "A01", "A02"))
    b = wh_features.alphaearth_composite(embedding, bands=("A10", "A11", "A12"))
    assert not np.allclose(a, b)


def test_band_order_maps_to_rgb():
    embedding = _fake_embedding()
    forward = wh_features.alphaearth_composite(embedding, bands=("A00", "A01", "A02"))
    reversed_ = wh_features.alphaearth_composite(embedding, bands=("A02", "A01", "A00"))
    assert np.allclose(forward[..., 0], reversed_[..., 2])


def test_pca_composite_is_three_channels():
    image = wh_features.alphaearth_composite(_fake_embedding(), mode="pca")
    assert image.shape == (8, 8, 3)
    assert np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0


def test_unknown_band_raises():
    with pytest.raises(KeyError, match="not loaded"):
        wh_features.alphaearth_composite(_fake_embedding(), bands=("A00", "ZZZ", "A02"))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="bands.*pca"):
        wh_features.alphaearth_composite(_fake_embedding(), mode="umap")


def test_composite_is_deterministic():
    embedding = _fake_embedding()
    assert np.array_equal(
        wh_features.alphaearth_composite(embedding),
        wh_features.alphaearth_composite(embedding),
    )


def test_stretch_handles_a_constant_channel():
    """A band with no variation must not divide by zero."""
    flat = np.full((5, 5), 0.3)
    assert np.allclose(wh_features._stretch(flat, (2.0, 98.0)), 0.5)
