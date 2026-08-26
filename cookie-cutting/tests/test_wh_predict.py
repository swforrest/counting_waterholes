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


# --- display layers -------------------------------------------------------


class _Tile:
    """Minimal tile stand-in: three bands and a validity mask."""

    def __init__(self, shape=(6, 6), valid=None):
        self.shape = shape
        self.bands = {
            "B4": np.full(shape, 0.12, dtype=np.float32),
            "B3": np.full(shape, 0.09, dtype=np.float32),
            "B2": np.full(shape, 0.06, dtype=np.float32),
        }
        self.valid = np.ones(shape, bool) if valid is None else valid
        for band in self.bands.values():
            band[~self.valid] = np.nan


def test_png_layer_names_are_validated():
    with pytest.raises(ValueError, match="unknown png_layers"):
        wh_predict.PredictParams(png_layers=("pred", "rgba"))


def test_confidence_png_without_confidence_is_refused():
    """A silently dropped trust layer is discovered an hour too late."""
    with pytest.raises(ValueError, match="write_confidence"):
        wh_predict.PredictParams(write_confidence=False, png_layers=("pred", "conf"))


def test_dropping_the_confidence_png_allows_the_fast_run():
    params = wh_predict.PredictParams(
        write_confidence=False, png_layers=("pred", "rgb")
    )
    assert params.png_layers == ("pred", "rgb")


def test_layer_paths_are_distinct_and_derivable(tmp_path):
    class Cfg:
        paths = {"predictions": tmp_path}

    names = {
        kind: wh_predict.raster_path(Cfg(), "025", "chip_2019-01", kind).name
        for kind in ("pred", "conf", "pred_png", "rgb_png", "conf_png")
    }
    assert names["pred"] == "chip_2019-01_pred.tif"
    assert names["conf"] == "chip_2019-01_conf.tif"
    assert names["pred_png"] == "chip_2019-01_pred.png"
    assert names["rgb_png"] == "chip_2019-01_rgb.png"
    assert names["conf_png"] == "chip_2019-01_conf.png"
    assert len(set(names.values())) == 5


def test_rgb_png_is_transparent_where_the_tile_is_unobserved():
    valid = np.ones((6, 6), bool)
    valid[0:2, :] = False
    rgba = wh_predict._rgb_rgba(_Tile(valid=valid))

    assert rgba.shape == (6, 6, 4)
    assert (rgba[0:2, :, 3] == 0).all()
    assert (rgba[2:, :, 3] == 255).all()
    assert rgba[2:, :, :3].max() > 0          # something was actually drawn


def test_confidence_png_covers_exactly_the_classified_pixels():
    classes = np.zeros((6, 6), dtype=np.uint8)
    classes[2:4, :] = 1
    confidence = np.full((6, 6), 0.9)

    rgba = wh_predict._confidence_rgba(confidence, classes)
    assert (rgba[2:4, :, 3] == 255).all()
    assert (rgba[0:2, :, 3] == 0).all()


def test_confidence_png_does_not_colour_nan_confidence():
    classes = np.ones((4, 4), dtype=np.uint8)
    confidence = np.full((4, 4), np.nan)
    assert (wh_predict._confidence_rgba(confidence, classes)[..., 3] == 0).all()


def test_confidence_colours_increase_with_confidence():
    """Low and high confidence must not render the same."""
    classes = np.ones((1, 2), dtype=np.uint8)
    rgba = wh_predict._confidence_rgba(np.array([[0.45, 0.99]]), classes)
    assert not np.array_equal(rgba[0, 0, :3], rgba[0, 1, :3])


def test_confidence_is_clipped_not_wrapped():
    """Below vmin renders as the floor colour, not as a bright value."""
    classes = np.ones((1, 3), dtype=np.uint8)
    rgba = wh_predict._confidence_rgba(
        np.array([[0.05, wh_predict.CONFIDENCE_VMIN, 5.0]]), classes
    )
    assert np.array_equal(rgba[0, 0, :3], rgba[0, 1, :3])
    assert rgba[0, 2, 3] == 255


# --- image encoding -------------------------------------------------------


def test_image_format_is_validated():
    with pytest.raises(ValueError, match="unknown image_format"):
        wh_predict.PredictParams(image_format="jpeg")


def test_webp_changes_only_the_display_extensions(tmp_path):
    class Cfg:
        paths = {"predictions": tmp_path}

    def name(kind):
        return wh_predict.raster_path(Cfg(), "025", "chip", kind, "webp").name

    assert name("pred_png") == "chip_pred.webp"
    assert name("rgb_png") == "chip_rgb.webp"
    # The GeoTIFFs are the authoritative product and never change format.
    assert name("pred") == "chip_pred.tif"
    assert name("conf") == "chip_conf.tif"


@pytest.mark.parametrize("suffix", [".png", ".webp"])
def test_class_colours_survive_the_encoder_exactly(tmp_path, suffix):
    """Every visible pixel must decode back to the colour it was written with.

    Lossy encoding would blend adjacent classes into colours that map to a third
    class — silently wrong rather than visibly broken — so the class and
    confidence layers are always lossless whatever the container.

    The guarantee is over *visible* pixels: WebP's lossless mode zeroes the
    colour channels under fully transparent pixels, which is invisible by
    definition, and the alpha channel itself is exact.
    """
    from PIL import Image

    cfg = _Cfg()
    classes = np.arange(0, 7, dtype=np.uint8).repeat(8).reshape(7, 8).repeat(4, 0)
    original = wh_predict._class_rgba(classes, cfg)

    path = tmp_path / f"chip_pred{suffix}"
    wh_predict.write_class_png(path, classes, cfg)
    decoded = np.array(Image.open(path).convert("RGBA"))

    assert (classes == 0).any(), "the fixture must exercise transparency"
    visible = original[..., 3] > 0
    assert np.array_equal(decoded[..., 3], original[..., 3])
    assert np.array_equal(decoded[visible][:, :3], original[visible][:, :3])


def test_transparency_survives_the_encoder(tmp_path):
    from PIL import Image

    cfg = _Cfg()
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[4:, :] = 3

    path = tmp_path / "chip_pred.webp"
    wh_predict.write_class_png(path, classes, cfg)
    decoded = np.array(Image.open(path).convert("RGBA"))

    assert (decoded[:4, :, 3] == 0).all()
    assert (decoded[4:, :, 3] == 255).all()


def test_the_rgb_layer_is_the_only_lossy_one(tmp_path):
    """It is photographic, and lossless PNG of noise is ~10x the bytes."""
    from PIL import Image

    tile = _Tile(shape=(64, 64))
    rng = np.random.default_rng(3)
    for band in tile.bands.values():
        band[:] = rng.random(band.shape).astype(np.float32) * 0.3

    lossy = tmp_path / "chip_rgb.webp"
    wh_predict.write_rgb_png(lossy, tile)
    lossless = tmp_path / "chip_pred.webp"
    wh_predict._save_image(lossless, wh_predict._rgb_rgba(tile))

    assert lossy.stat().st_size < lossless.stat().st_size
    assert Image.open(lossy).size == (64, 64)


# --- the footprint map layer ----------------------------------------------


def _write_footprint(directory, site_id, area_m2=1000.0, succeeded=True):
    import json

    path = directory / f"site_{site_id}_footprint.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [134.0, -13.0], [134.1, -13.0], [134.1, -13.1], [134.0, -13.0],
            ]]},
            "properties": {
                "site_id": site_id, "n_pixels": 10, "area_m2": area_m2,
                "succeeded": succeeded, "notes": "kept the largest of 2 regions",
                # The tuning parameters must not survive into the map layer.
                "score_threshold": 1.75, "buffer_px": 3, "use_alphaearth": True,
            },
        }],
    }))


class _PathsCfg:
    def __init__(self, tmp_path):
        (tmp_path / "derived" / "footprints").mkdir(parents=True)
        (tmp_path / "predictions").mkdir(parents=True)
        self.paths = {
            "derived": tmp_path / "derived",
            "predictions": tmp_path / "predictions",
        }


def test_footprints_combine_into_one_layer(tmp_path):
    import json

    cfg = _PathsCfg(tmp_path)
    for site_id in ("000", "001", "002"):
        _write_footprint(cfg.paths["derived"] / "footprints", site_id)

    path = wh_predict.export_footprints_geojson(cfg)
    layer = json.loads(path.read_text())

    assert path.name == "waterhole_footprints.geojson"
    assert len(layer["features"]) == 3
    assert {f["properties"]["site_id"] for f in layer["features"]} == {"000", "001", "002"}


def test_sites_without_a_footprint_are_named_not_silently_absent(tmp_path):
    """176 outlines over 187 boxes is a difference the UI has to be able to state."""
    import json

    cfg = _PathsCfg(tmp_path)
    _write_footprint(cfg.paths["derived"] / "footprints", "000")

    path = wh_predict.export_footprints_geojson(cfg, sites=["000", "001", "002"])
    layer = json.loads(path.read_text())

    assert layer["sites_without_footprint"] == ["001", "002"]


def test_the_map_layer_drops_the_tuning_parameters(tmp_path):
    """They are provenance for the footprint, not something a map needs 176 times."""
    import json

    cfg = _PathsCfg(tmp_path)
    _write_footprint(cfg.paths["derived"] / "footprints", "000")

    layer = json.loads(wh_predict.export_footprints_geojson(cfg).read_text())
    properties = layer["features"][0]["properties"]

    assert set(properties) == {
        "site_id", "n_pixels", "area_m2", "area_ha", "succeeded", "notes"
    }
    assert properties["area_ha"] == pytest.approx(0.1)


def test_no_footprints_at_all_is_an_empty_layer_not_a_crash(tmp_path):
    import json

    cfg = _PathsCfg(tmp_path)
    layer = json.loads(wh_predict.export_footprints_geojson(cfg, sites=["000"]).read_text())

    assert layer["features"] == []
    assert layer["sites_without_footprint"] == ["000"]


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


# --- progress reporting ---------------------------------------------------


def test_progress_falls_back_to_text_when_tqdm_is_unavailable(monkeypatch, capsys):
    """The bar is the only window onto an hour of work; it must never be the
    thing that breaks the run."""
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)

    progress = wh_predict._Progress(3, "predicting")
    # Cumulative, the way run() passes it — the reporter displays, it does not
    # accumulate.
    for site in range(1, 4):
        progress.update(months=84 * site)
    progress.close()

    printed = capsys.readouterr().out
    assert "3 sites" in printed
    # Intermediate lines are throttled to one per 15 s, but the last one always
    # prints, so a finished run never ends on a stale count.
    assert "3/3 sites" in printed
    assert "months=252" in printed


def test_progress_is_silent_when_not_verbose(capsys):
    progress = wh_predict._Progress(2, "predicting", verbose=False)
    progress.update()
    progress.write("this must not appear")
    progress.close()
    assert capsys.readouterr().out == ""


def test_progress_counts_every_update():
    progress = wh_predict._Progress(5, "predicting", verbose=False)
    for _ in range(5):
        progress.update()
    assert progress.position == 5


def test_progress_reports_elapsed_time():
    progress = wh_predict._Progress(1, "predicting", verbose=False)
    assert progress.elapsed_minutes >= 0
