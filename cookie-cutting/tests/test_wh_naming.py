"""Tests for chip filename parsing."""

import pytest

import wh_naming

REAL_STEM = "2024-06_mimal_test_S2_019_S13p40_E134p47_2024-09"


def test_parses_a_real_filename():
    key = wh_naming.parse_stem(REAL_STEM)
    assert key.site_id == "019"
    assert key.year == 2024
    assert key.month == 9
    assert key.year_month == "2024-09"
    assert key.lat == pytest.approx(-13.40)
    assert key.lon == pytest.approx(134.47)
    assert key.prefix == "2024-06_mimal_test"


def test_parses_paths_with_any_extension():
    for extension in (".tif", ".png", ".json"):
        key = wh_naming.parse_path(f"/some/where/{REAL_STEM}{extension}")
        assert key.site_id == "019"


def test_stem_round_trips():
    assert wh_naming.parse_stem(REAL_STEM).stem() == REAL_STEM


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("S13p40", -13.40),
        ("N13p40", 13.40),
        ("E134p47", 134.47),
        ("W134p47", -134.47),
        ("S0p05", -0.05),
    ],
)
def test_coord_tag_parsing(tag, expected):
    assert wh_naming.parse_coord_tag(tag) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,is_latitude,expected",
    [
        (-13.40, True, "S13p40"),
        (134.47, False, "E134p47"),
        (-134.4, False, "W134p40"),
    ],
)
def test_coord_tag_formatting(value, is_latitude, expected):
    assert wh_naming.format_coord_tag(value, is_latitude) == expected


@pytest.mark.parametrize(
    "stem",
    [
        "not_a_chip",
        "2024-06_mimal_test_S2_19_S13p40_E134p47_2024-09",  # site not zero-padded
        "2024-06_mimal_test_S2_019_13p40_E134p47_2024-09",  # no hemisphere
        "2024-06_mimal_test_S2_019_S13p40_E134p47_2024",  # no month
        "2024-06_mimal_test_S2_019_S13p40_E134p47_2024-9",  # month not padded
    ],
)
def test_bad_names_raise(stem):
    with pytest.raises(ValueError):
        wh_naming.parse_stem(stem)


def test_month_out_of_range_raises():
    with pytest.raises(ValueError, match="month out of range"):
        wh_naming.parse_stem("p_S2_019_S13p40_E134p47_2024-13")


def test_month_index_orders_across_a_year_boundary():
    december = wh_naming.parse_stem("p_S2_000_S13p40_E134p47_2024-12")
    january = wh_naming.parse_stem("p_S2_000_S13p40_E134p47_2025-01")
    assert january.month_index - december.month_index == 1


def test_try_parse_returns_none_for_unrelated_files():
    assert wh_naming.try_parse_path("/some/where/.DS_Store") is None
    assert wh_naming.try_parse_path(f"/x/{REAL_STEM}.tif") is not None


def test_sibling_path_keeps_the_stem():
    sibling = wh_naming.sibling_path(f"/tifs/{REAL_STEM}.tif", "/labels", "_labels.tif")
    assert sibling.name == f"{REAL_STEM}_labels.tif"
    assert str(sibling.parent) == "/labels"
