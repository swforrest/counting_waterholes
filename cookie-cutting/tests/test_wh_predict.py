"""Tests for prediction, region counting and the quality flags."""

import numpy as np
import pandas as pd
import pytest

import wh_features
import wh_predict


class _Cfg:
    """Config stand-in exposing the class scheme and nothing else."""

    def __init__(self, scheme_version=1):
        from wh_config import ClassDef

        self.classes = [
            ClassDef(0, "unlabelled", "#00000000", "0", True, ""),
            ClassDef(1, "open_water", "#1f4ea1", "1", False, ""),
            ClassDef(2, "turbid_water", "#8c6d3f", "2", False, ""),
            ClassDef(3, "aquatic_vegetation", "#2e8b57", "3", False, ""),
            ClassDef(4, "mud", "#a0522d", "4", False, ""),
            ClassDef(5, "dry_bare", "#d9b382", "5", False, ""),
            ClassDef(6, "surrounding_vegetation", "#7f9f5a", "6", False, ""),
        ]
        self.raw = {"classes": {"scheme_version": scheme_version}}

    def __getitem__(self, key):
        return self.raw[key]


# --- region counting ------------------------------------------------------


def test_counts_and_fractions_within_a_region():
    cfg = _Cfg()
    classes = np.zeros((10, 10), dtype=np.uint8)
    classes[0:2, :] = 1     # 20 px open water
    classes[2:5, :] = 5     # 30 px dry bare
    region = np.ones((10, 10), dtype=bool)

    record = wh_predict.count_classes(classes, region, cfg, "bbox")

    assert record["bbox_n_classified"] == 50
    assert record["bbox_n_open_water"] == 20
    assert record["bbox_n_dry_bare"] == 30
    assert record["bbox_frac_open_water"] == pytest.approx(0.4)
    assert record["bbox_frac_dry_bare"] == pytest.approx(0.6)


def test_fractions_sum_to_one():
    cfg = _Cfg()
    rng = np.random.default_rng(0)
    classes = rng.integers(1, 7, size=(20, 20)).astype(np.uint8)

    record = wh_predict.count_classes(classes, np.ones((20, 20), bool), cfg, "bbox")
    total = sum(v for k, v in record.items() if k.startswith("bbox_frac_"))
    assert total == pytest.approx(1.0)


def test_the_region_actually_restricts_the_count():
    cfg = _Cfg()
    classes = np.full((10, 10), 1, dtype=np.uint8)
    region = np.zeros((10, 10), dtype=bool)
    region[0:3, 0:4] = True

    record = wh_predict.count_classes(classes, region, cfg, "bbox")
    assert record["bbox_n_open_water"] == 12
    assert record["bbox_n_classified"] == 12


def test_unclassified_pixels_are_excluded_from_the_denominator():
    """Fractions are over classified pixels, so cloud gaps do not dilute them."""
    cfg = _Cfg()
    classes = np.zeros((10, 10), dtype=np.uint8)
    classes[0:5, :] = 1     # half classified, half class 0

    record = wh_predict.count_classes(classes, np.ones((10, 10), bool), cfg, "bbox")
    assert record["bbox_n_classified"] == 50
    assert record["bbox_frac_open_water"] == pytest.approx(1.0)


def test_an_empty_region_gives_nan_fractions_not_zero():
    """Nothing observed is unknown composition, not a composition of zeroes."""
    cfg = _Cfg()
    record = wh_predict.count_classes(
        np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10), bool), cfg, "footprint"
    )
    assert record["footprint_n_classified"] == 0
    assert np.isnan(record["footprint_frac_open_water"])


def test_counts_never_exceed_the_region_size():
    cfg = _Cfg()
    rng = np.random.default_rng(1)
    classes = rng.integers(0, 7, size=(30, 30)).astype(np.uint8)
    region = rng.random((30, 30)) > 0.5

    record = wh_predict.count_classes(classes, region, cfg, "bbox")
    counted = sum(v for k, v in record.items() if k.startswith("bbox_n_") and "classified" not in k)
    assert counted == record["bbox_n_classified"] <= int(region.sum())


# --- majority filter ------------------------------------------------------


def test_majority_filter_is_a_no_op_when_off():
    classes = np.random.default_rng(2).integers(0, 7, size=(20, 20)).astype(np.uint8)
    assert np.array_equal(wh_predict.majority_filter(classes, 0), classes)
    assert np.array_equal(wh_predict.majority_filter(classes, 1), classes)


def test_majority_filter_removes_an_isolated_pixel():
    classes = np.full((20, 20), 5, dtype=np.uint8)
    classes[10, 10] = 1

    filtered = wh_predict.majority_filter(classes, 3)
    assert filtered[10, 10] == 5


def test_majority_filter_never_classifies_an_unclassified_pixel():
    classes = np.full((20, 20), 5, dtype=np.uint8)
    classes[0:4, 0:4] = 0

    filtered = wh_predict.majority_filter(classes, 3)
    assert (filtered[0:4, 0:4] == 0).all()


# --- quality flags --------------------------------------------------------


def _series(wet, obs, site="000"):
    return pd.DataFrame({
        "site_id": site,
        "year": 2024,
        "month": np.arange(1, len(wet) + 1),
        "year_month": [f"2024-{m:02d}" for m in range(1, len(wet) + 1)],
        "mean_n_obs": obs,
        "gap_fraction": 0.0,
        "bbox_frac_open_water": wet,
        "bbox_frac_turbid_water": 0.0,
    })


def test_isolated_wet_month_on_a_thin_median_is_flagged():
    table = _series([0.0, 0.0, 0.5, 0.0, 0.0], [5, 5, 1, 5, 5])
    flagged = wh_predict.add_quality_flags(table, _Cfg(), wh_predict.PredictParams())
    assert bool(flagged.loc[2, "flag_isolated_wet"])


def test_a_well_observed_wet_month_is_not_flagged():
    """A wet month backed by several clear scenes is a rainfall event."""
    table = _series([0.0, 0.0, 0.5, 0.0, 0.0], [5, 5, 6, 5, 5])
    flagged = wh_predict.add_quality_flags(table, _Cfg(), wh_predict.PredictParams())
    assert not flagged["flag_isolated_wet"].any()


def test_a_sustained_wet_run_is_not_flagged():
    table = _series([0.0, 0.4, 0.5, 0.45, 0.0], [1, 1, 1, 1, 1])
    flagged = wh_predict.add_quality_flags(table, _Cfg(), wh_predict.PredictParams())
    assert not flagged["flag_isolated_wet"].any()


def test_flagging_can_be_switched_off():
    table = _series([0.0, 0.0, 0.5, 0.0, 0.0], [5, 5, 1, 5, 5])
    params = wh_predict.PredictParams(flag_isolated_wet=False)
    flagged = wh_predict.add_quality_flags(table, _Cfg(), params)
    assert not flagged["flag_isolated_wet"].any()


def test_flagging_never_alters_the_values():
    table = _series([0.0, 0.0, 0.5, 0.0, 0.0], [5, 5, 1, 5, 5])
    before = table["bbox_frac_open_water"].copy()
    flagged = wh_predict.add_quality_flags(table, _Cfg(), wh_predict.PredictParams())
    pd.testing.assert_series_equal(flagged["bbox_frac_open_water"], before)


def test_data_quality_bands():
    table = _series([0.0] * 3, [6, 1, 4])
    table.loc[0, "gap_fraction"] = 0.9
    banded = wh_predict.add_quality_flags(table, _Cfg(), wh_predict.PredictParams())
    assert list(banded["data_quality"]) == ["poor", "thin", "good"]


# --- the model manifest is the authority on features ----------------------


def test_feature_params_round_trip_through_a_manifest():
    params = wh_features.FeatureParams(
        reflectance_bands=("B3", "B4"), indices=("mndwi", "ndvi"),
        context_windows=(3,), context_indices=("mndwi",),
        temporal_indices=("mndwi", "ndti"), include_n_obs=False,
        use_alphaearth=True, alphaearth_year=2025,
        alphaearth_bands=("A28", "A32", "A63"),
    )
    rebuilt = wh_predict.feature_params_from_manifest({"feature_params": params.as_dict()})

    assert rebuilt.as_dict() == params.as_dict()
    assert rebuilt.alphaearth_bands == ("A28", "A32", "A63")


def test_all_bands_round_trips_as_none():
    params = wh_features.FeatureParams(use_alphaearth=True, alphaearth_bands=None)
    rebuilt = wh_predict.feature_params_from_manifest({"feature_params": params.as_dict()})
    assert rebuilt.alphaearth_bands is None


def test_a_class_scheme_mismatch_is_refused():
    with pytest.raises(ValueError, match="class scheme version"):
        wh_predict.check_scheme({"class_scheme_version": 1}, _Cfg(scheme_version=2))


def test_a_matching_class_scheme_passes():
    wh_predict.check_scheme({"class_scheme_version": 1}, _Cfg(scheme_version=1))
