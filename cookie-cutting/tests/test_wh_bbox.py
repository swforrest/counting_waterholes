"""Tests for waterhole bounding boxes and the masks derived from them."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.warp
from rasterio.transform import Affine

import wh_bbox

# A synthetic tile in UTM 53S: 100x100 px at 10 m, origin at a round coordinate
# so pixel arithmetic in the tests is exact.
TILE_CRS = "EPSG:32753"
ORIGIN_X, ORIGIN_Y = 500000.0, 8500000.0
PIXEL = 10.0
SHAPE = (100, 100)


class _FakeTile:
    """Minimal stand-in for wh_tiles.Tile: only the grid matters here."""

    def __init__(self):
        self.crs = rasterio.crs.CRS.from_string(TILE_CRS)
        self.transform = Affine(PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y)
        self.shape = SHAPE


def _box_from_pixels(row_start, row_stop, col_start, col_stop) -> pd.Series:
    """Build a box row covering exactly this pixel window of the fake tile.

    Constructed by converting the window's UTM corners to lon/lat, so the test
    exercises the real reprojection path rather than bypassing it.
    """
    left = ORIGIN_X + col_start * PIXEL
    right = ORIGIN_X + col_stop * PIXEL
    top = ORIGIN_Y - row_start * PIXEL
    bottom = ORIGIN_Y - row_stop * PIXEL

    lons, lats = rasterio.warp.transform(
        TILE_CRS, wh_bbox.WGS84, [left, right, right, left], [top, top, bottom, bottom]
    )
    return pd.Series({
        "site_id": "000",
        "label": "WH_wet",
        "lon_min": min(lons), "lon_max": max(lons),
        "lat_min": min(lats), "lat_max": max(lats),
    })


NO_BUFFER = wh_bbox.BoxParams(buffer_m=0.0)


# --- rasterisation against known pixel coordinates ------------------------


def test_box_lands_on_the_expected_pixels():
    tile = _FakeTile()
    box = _box_from_pixels(20, 40, 30, 50)

    mask, clipped = wh_bbox.box_mask(box, tile, NO_BUFFER)

    rows, cols = np.nonzero(mask)
    assert not clipped
    # Reprojection to lon/lat and back is not bit-exact, so allow one pixel.
    assert rows.min() == pytest.approx(20, abs=1)
    assert rows.max() == pytest.approx(39, abs=1)
    assert cols.min() == pytest.approx(30, abs=1)
    assert cols.max() == pytest.approx(49, abs=1)


def test_mask_is_a_solid_rectangle():
    tile = _FakeTile()
    mask, _ = wh_bbox.box_mask(_box_from_pixels(10, 30, 10, 30), tile, NO_BUFFER)

    rows, cols = np.nonzero(mask)
    window = mask[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    assert window.all()


def test_mask_shape_matches_the_tile():
    tile = _FakeTile()
    mask, _ = wh_bbox.box_mask(_box_from_pixels(10, 20, 10, 20), tile, NO_BUFFER)
    assert mask.shape == tile.shape
    assert mask.dtype == bool


# --- buffer arithmetic ----------------------------------------------------


def test_buffer_grows_the_box_by_the_right_number_of_pixels():
    """100 m on every side is 10 pixels at 10 m resolution."""
    tile = _FakeTile()
    box = _box_from_pixels(40, 60, 40, 60)

    plain, _ = wh_bbox.box_mask(box, tile, wh_bbox.BoxParams(buffer_m=0.0))
    buffered, _ = wh_bbox.box_mask(box, tile, wh_bbox.BoxParams(buffer_m=100.0))

    plain_rows, plain_cols = np.nonzero(plain)
    buffered_rows, buffered_cols = np.nonzero(buffered)

    assert buffered_rows.min() == pytest.approx(plain_rows.min() - 10, abs=1)
    assert buffered_rows.max() == pytest.approx(plain_rows.max() + 10, abs=1)
    assert buffered_cols.min() == pytest.approx(plain_cols.min() - 10, abs=1)
    assert buffered_cols.max() == pytest.approx(plain_cols.max() + 10, abs=1)


def test_buffer_is_applied_on_every_side_not_to_the_total():
    """A 20 px box with a 100 m buffer becomes 40 px across, not 30."""
    tile = _FakeTile()
    mask, _ = wh_bbox.box_mask(
        _box_from_pixels(40, 60, 40, 60), tile, wh_bbox.BoxParams(buffer_m=100.0)
    )
    cols = np.nonzero(mask)[1]
    assert (cols.max() - cols.min() + 1) == pytest.approx(40, abs=2)


def test_zero_buffer_changes_nothing():
    tile = _FakeTile()
    box = _box_from_pixels(30, 50, 30, 50)
    a, _ = wh_bbox.box_mask(box, tile, wh_bbox.BoxParams(buffer_m=0.0))
    b, _ = wh_bbox.box_mask(box, tile, NO_BUFFER)
    assert np.array_equal(a, b)


# --- clipping -------------------------------------------------------------


def test_box_reaching_past_the_tile_is_flagged_and_clipped():
    tile = _FakeTile()
    # A window running off the right and bottom edges.
    mask, clipped = wh_bbox.box_mask(_box_from_pixels(80, 130, 80, 130), tile, NO_BUFFER)

    assert clipped
    assert mask.shape == tile.shape
    assert mask[-1, -1]


def test_a_box_covering_everything_gives_an_all_true_mask():
    tile = _FakeTile()
    mask, clipped = wh_bbox.box_mask(
        _box_from_pixels(-50, 150, -50, 150), tile, NO_BUFFER
    )
    assert clipped
    assert mask.all()


def test_buffer_can_push_an_interior_box_into_clipping():
    tile = _FakeTile()
    box = _box_from_pixels(2, 20, 2, 20)
    _, without = wh_bbox.box_mask(box, tile, wh_bbox.BoxParams(buffer_m=0.0))
    _, with_buffer = wh_bbox.box_mask(box, tile, wh_bbox.BoxParams(buffer_m=100.0))
    assert not without
    assert with_buffer


# --- neighbours -----------------------------------------------------------


def _boxes_frame(entries: dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for site_id, window in entries.items():
        box = _box_from_pixels(*window)
        box["site_id"] = site_id
        rows.append(box)
    return pd.DataFrame(rows).set_index("site_id", drop=False)


def test_neighbour_mask_excludes_the_site_itself():
    tile = _FakeTile()
    boxes = _boxes_frame({
        "000": (10, 30, 10, 30),
        "001": (60, 80, 60, 80),
    })

    neighbours = wh_bbox.neighbour_mask("000", tile, boxes, NO_BUFFER)
    own, _ = wh_bbox.box_mask(boxes.loc["000"], tile, NO_BUFFER)

    assert not (neighbours & own).any()
    assert neighbours[70, 70]


def test_neighbour_mask_unions_several_boxes():
    tile = _FakeTile()
    boxes = _boxes_frame({
        "000": (10, 20, 10, 20),
        "001": (40, 50, 40, 50),
        "002": (70, 80, 70, 80),
    })

    neighbours = wh_bbox.neighbour_mask("000", tile, boxes, NO_BUFFER)
    assert neighbours[45, 45]
    assert neighbours[75, 75]
    assert not neighbours[15, 15]


def test_neighbour_mask_is_empty_when_a_site_is_alone():
    tile = _FakeTile()
    boxes = _boxes_frame({"000": (10, 30, 10, 30)})
    assert not wh_bbox.neighbour_mask("000", tile, boxes, NO_BUFFER).any()


# --- params ---------------------------------------------------------------


def test_params_round_trip_through_a_dict():
    params = wh_bbox.BoxParams(buffer_m=250.0)
    assert params.as_dict()["buffer_m"] == 250.0


# --- single-component footprints ------------------------------------------


def test_largest_component_keeps_only_the_biggest():
    import wh_footprint

    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, 2:8] = True      # 36 px — the basin
    mask[15, 15] = True        # 1 px fragment
    mask[17:19, 17:19] = True  # 4 px fragment

    kept, n_dropped, dropped_px = wh_footprint._largest_component(mask)

    assert kept.sum() == 36
    assert n_dropped == 2
    assert dropped_px == 5
    assert kept[3, 3] and not kept[15, 15]


def test_largest_component_leaves_a_single_region_alone():
    import wh_footprint

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    kept, n_dropped, dropped_px = wh_footprint._largest_component(mask)

    assert np.array_equal(kept, mask)
    assert n_dropped == 0 and dropped_px == 0


def test_largest_component_handles_an_empty_mask():
    import wh_footprint

    kept, n_dropped, dropped_px = wh_footprint._largest_component(
        np.zeros((10, 10), dtype=bool)
    )
    assert not kept.any() and n_dropped == 0 and dropped_px == 0


def test_diagonal_touching_regions_count_as_separate():
    """scipy's default connectivity is 4-way, so a diagonal touch is two regions."""
    import wh_footprint

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    mask[4:7, 4:7] = True

    kept, n_dropped, _ = wh_footprint._largest_component(mask)

    assert n_dropped == 1
    assert kept.sum() == 9


# --- AlphaEarth embedding anomaly -----------------------------------------


def _embedding_with_patch(shape=(30, 30), n_bands=8, offset=6.0):
    """Uniform-ish embedding with one region shifted away from the rest."""
    import wh_features

    rng = np.random.default_rng(7)
    embedding = {}
    for i in range(n_bands):
        band = rng.normal(0.0, 1.0, size=shape)
        band[10:16, 10:16] += offset          # the "basin"
        embedding[f"{wh_features.ALPHAEARTH_PREFIX}A{i:02d}"] = band
    return embedding


def test_anomaly_is_high_where_the_embedding_differs():
    import wh_footprint

    layer = wh_footprint.embedding_anomaly(_embedding_with_patch())

    patch = layer[10:16, 10:16]
    background = np.concatenate([layer[:8].ravel(), layer[20:].ravel()])
    assert np.nanmedian(patch) > np.nanmedian(background) + 3


def test_anomaly_is_flat_when_nothing_differs():
    import wh_footprint

    layer = wh_footprint.embedding_anomaly(_embedding_with_patch(offset=0.0))
    finite = layer[np.isfinite(layer)]
    # No structure to find, so nothing should reach the usual score threshold.
    assert np.nanpercentile(finite, 99) < 6


def test_rms_keeps_the_magnitude_comparable_across_band_counts():
    """Selecting a subset must not silently rescale the layer's weight."""
    import wh_features
    import wh_footprint

    embedding = _embedding_with_patch(n_bands=16)
    all_bands = wh_footprint.embedding_anomaly(embedding)
    three = wh_footprint.embedding_anomaly(
        embedding, bands=("A00", "A01", "A02")
    )

    peak_all = np.nanmedian(all_bands[10:16, 10:16])
    peak_three = np.nanmedian(three[10:16, 10:16])
    assert 0.4 < peak_three / peak_all < 2.5


def test_band_subset_uses_only_those_bands():
    import wh_features
    import wh_footprint

    embedding = _embedding_with_patch(n_bands=8)
    # Give one band its anomaly somewhere else entirely; including it must move
    # the result.
    rng = np.random.default_rng(11)
    odd = rng.normal(0.0, 1.0, size=(30, 30))
    odd[22:28, 22:28] += 8.0
    embedding[f"{wh_features.ALPHAEARTH_PREFIX}A00"] = odd

    with_it = wh_footprint.embedding_anomaly(embedding, bands=("A00", "A01"))
    without = wh_footprint.embedding_anomaly(embedding, bands=("A01",))

    assert not np.allclose(with_it, without, equal_nan=True)
    # The odd band's own anomaly shows only when it is included.
    assert np.nanmedian(with_it[22:28, 22:28]) > np.nanmedian(without[22:28, 22:28])


def test_a_band_with_no_spread_is_skipped_not_counted_as_zero():
    """A constant band carries no information; averaging a zero in would dilute."""
    import wh_features
    import wh_footprint

    embedding = _embedding_with_patch(n_bands=4)
    informative = wh_footprint.embedding_anomaly(embedding, bands=("A01", "A02"))

    embedding[f"{wh_features.ALPHAEARTH_PREFIX}A00"] = np.full((30, 30), 0.5)
    with_flat = wh_footprint.embedding_anomaly(
        embedding, bands=("A00", "A01", "A02")
    )

    assert np.allclose(informative, with_flat, equal_nan=True)


def test_unknown_band_raises():
    import wh_footprint

    with pytest.raises(KeyError, match="not loaded"):
        wh_footprint.embedding_anomaly(_embedding_with_patch(), bands=("A00", "ZZZ"))


def test_valid_mask_restricts_the_baseline():
    import wh_footprint

    embedding = _embedding_with_patch()
    valid = np.ones((30, 30), dtype=bool)
    valid[:5] = False

    layer = wh_footprint.embedding_anomaly(embedding, valid=valid)
    assert np.all(np.isnan(layer[:5]))
    assert np.isfinite(layer[10:16, 10:16]).all()


def test_a_constant_embedding_raises_rather_than_dividing_by_zero():
    import wh_features
    import wh_footprint

    flat = {
        f"{wh_features.ALPHAEARTH_PREFIX}A{i:02d}": np.full((10, 10), 0.5)
        for i in range(4)
    }
    with pytest.raises(ValueError, match="usable spread"):
        wh_footprint.embedding_anomaly(flat)


def test_score_refuses_when_every_layer_is_switched_off():
    import wh_footprint

    params = wh_footprint.FootprintParams(
        seasonal_range_weights={}, dry_ndvi_anomaly_weight=0.0, use_alphaearth=False
    )
    with pytest.raises(ValueError, match="no contributing layers"):
        wh_footprint.basin_score({}, params)
