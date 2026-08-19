"""Tests for the per-pixel temporal statistics and the harmonic fit."""

import numpy as np
import pytest

import wh_temporal

# Seven years of monthly steps, matching the real export.
MONTH_INDEX = np.arange(2019 * 12, 2019 * 12 + 84)
DECIMAL_YEAR = MONTH_INDEX / 12.0


def _stack(series: np.ndarray) -> np.ndarray:
    """Turn a 1-D series into a (n_months, 1, 1) single-pixel stack."""
    return series.reshape(-1, 1, 1).astype(np.float64)


# --- design matrix --------------------------------------------------------


def test_design_matrix_shape_and_names():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    assert names == ["intercept", "trend_per_year", "cos1", "sin1"]
    assert design.shape == (84, 4)


def test_design_matrix_second_order_adds_two_terms():
    _, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=2)
    assert names == ["intercept", "trend_per_year", "cos1", "sin1", "cos2", "sin2"]


def test_design_matrix_trend_is_centred():
    """A centred trend keeps the intercept interpretable as the series mean."""
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    assert design[:, names.index("trend_per_year")].mean() == pytest.approx(0.0)


def test_design_matrix_without_trend():
    _, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1, include_trend=False)
    assert names == ["intercept", "cos1", "sin1"]


# --- harmonic fit ---------------------------------------------------------


def test_recovers_known_coefficients_exactly():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    angle = 2.0 * np.pi * DECIMAL_YEAR
    truth = 0.2 + 0.5 * np.cos(angle) + 0.3 * np.sin(angle)

    fit = wh_temporal.fit_harmonic(_stack(truth), design, names)

    assert fit.coefficients[names.index("intercept"), 0, 0] == pytest.approx(0.2)
    assert fit.coefficients[names.index("cos1"), 0, 0] == pytest.approx(0.5)
    assert fit.coefficients[names.index("sin1"), 0, 0] == pytest.approx(0.3)
    assert fit.coefficients[names.index("trend_per_year"), 0, 0] == pytest.approx(0.0, abs=1e-9)


def test_recovers_a_linear_trend():
    """The trend term is what makes a multi-year degradation signal estimable."""
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    centred = DECIMAL_YEAR - DECIMAL_YEAR.mean()
    truth = 0.4 - 0.05 * centred + 0.2 * np.cos(2.0 * np.pi * DECIMAL_YEAR)

    fit = wh_temporal.fit_harmonic(_stack(truth), design, names)

    assert fit.coefficients[names.index("trend_per_year"), 0, 0] == pytest.approx(-0.05)
    assert fit.coefficients[names.index("intercept"), 0, 0] == pytest.approx(0.4)


def test_amplitude_and_phase():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    angle = 2.0 * np.pi * DECIMAL_YEAR
    truth = 0.5 * np.cos(angle) + 0.3 * np.sin(angle)

    fit = wh_temporal.fit_harmonic(_stack(truth), design, names)

    assert fit.amplitude[0, 0] == pytest.approx(np.hypot(0.5, 0.3))
    assert fit.phase[0, 0] == pytest.approx(np.arctan2(0.3, 0.5))


def test_fitted_and_residual_are_consistent():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    rng = np.random.default_rng(0)
    truth = 0.3 + 0.4 * np.cos(2.0 * np.pi * DECIMAL_YEAR) + rng.normal(0, 0.02, 84)

    fit = wh_temporal.fit_harmonic(_stack(truth), design, names)

    assert np.allclose(fit.fitted[:, 0, 0] + fit.residual[:, 0, 0], truth)
    assert np.abs(fit.residual[:, 0, 0]).max() < 0.1


def test_gaps_do_not_bias_the_fit():
    """Missing months must be skipped, not treated as zeros."""
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    angle = 2.0 * np.pi * DECIMAL_YEAR
    truth = 0.2 + 0.5 * np.cos(angle) + 0.3 * np.sin(angle)

    gappy = truth.copy()
    gappy[::5] = np.nan  # drop every fifth month

    fit = wh_temporal.fit_harmonic(_stack(gappy), design, names)

    assert fit.coefficients[names.index("cos1"), 0, 0] == pytest.approx(0.5)
    assert fit.coefficients[names.index("sin1"), 0, 0] == pytest.approx(0.3)
    assert fit.n_used[0, 0] == np.isfinite(gappy).sum()


def test_weights_are_respected():
    """A zero-weight month must not influence the fit at all."""
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    angle = 2.0 * np.pi * DECIMAL_YEAR
    truth = 0.2 + 0.5 * np.cos(angle) + 0.3 * np.sin(angle)

    corrupted = truth.copy()
    corrupted[10] = 99.0
    weights = np.ones(84)
    weights[10] = 0.0

    fit = wh_temporal.fit_harmonic(
        _stack(corrupted), design, names, weights=_stack(weights)
    )

    assert fit.coefficients[names.index("cos1"), 0, 0] == pytest.approx(0.5)
    assert fit.n_used[0, 0] == 83


def test_too_few_months_gives_nan_not_a_fit():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    sparse = np.full(84, np.nan)
    sparse[:3] = [0.1, 0.2, 0.3]  # fewer months than the four terms

    fit = wh_temporal.fit_harmonic(_stack(sparse), design, names)

    assert np.all(np.isnan(fit.coefficients[:, 0, 0]))
    assert np.isnan(fit.amplitude[0, 0])


def test_rank_deficient_pixel_gives_nan():
    """Every usable month in one season cannot constrain an annual cycle."""
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    values = np.full(84, np.nan)
    values[[0, 12, 24, 36, 48]] = 0.5  # same calendar month every year

    fit = wh_temporal.fit_harmonic(_stack(values), design, names)
    assert np.all(np.isnan(fit.coefficients[:, 0, 0]))


def test_solves_many_pixels_independently():
    design, names = wh_temporal.harmonic_design(DECIMAL_YEAR, orders=1)
    angle = 2.0 * np.pi * DECIMAL_YEAR
    stack = np.empty((84, 2, 2))
    stack[:, 0, 0] = 0.5 * np.cos(angle)
    stack[:, 0, 1] = 0.1 * np.cos(angle)
    stack[:, 1, 0] = np.nan  # unfittable
    stack[:, 1, 1] = 0.3 * np.sin(angle)

    fit = wh_temporal.fit_harmonic(stack, design, names)

    assert fit.amplitude[0, 0] == pytest.approx(0.5)
    assert fit.amplitude[0, 1] == pytest.approx(0.1)
    assert np.isnan(fit.amplitude[1, 0])
    assert fit.amplitude[1, 1] == pytest.approx(0.3)


# --- seasonal extremes ----------------------------------------------------


def test_seasonal_extreme_selects_the_right_months():
    calendar_month = (MONTH_INDEX % 12) + 1
    values = np.zeros(84)
    values[calendar_month == 2] = 0.8  # February peaks
    values[calendar_month == 9] = -0.6  # September troughs

    wet_max = wh_temporal.seasonal_extreme(
        _stack(values), calendar_month, [11, 12, 1, 2, 3, 4], "max"
    )
    dry_min = wh_temporal.seasonal_extreme(
        _stack(values), calendar_month, [5, 6, 7, 8, 9, 10], "min"
    )

    assert wet_max[0, 0] == pytest.approx(0.8)
    assert dry_min[0, 0] == pytest.approx(-0.6)


def test_seasonal_extreme_ignores_gaps():
    calendar_month = (MONTH_INDEX % 12) + 1
    values = np.full(84, 0.1)
    values[calendar_month == 2] = np.nan

    wet_max = wh_temporal.seasonal_extreme(
        _stack(values), calendar_month, [1, 2, 3], "max"
    )
    assert wet_max[0, 0] == pytest.approx(0.1)


def test_seasonal_extreme_all_missing_gives_nan():
    calendar_month = (MONTH_INDEX % 12) + 1
    values = np.full(84, np.nan)
    result = wh_temporal.seasonal_extreme(_stack(values), calendar_month, [1, 2], "max")
    assert np.isnan(result[0, 0])


def test_seasonal_extreme_rejects_a_bad_statistic():
    calendar_month = (MONTH_INDEX % 12) + 1
    with pytest.raises(ValueError, match="max"):
        wh_temporal.seasonal_extreme(
            _stack(np.zeros(84)), calendar_month, [1], "mean"
        )


# --- percentile rank ------------------------------------------------------


def test_percentile_rank_spans_zero_to_one():
    values = np.arange(84, dtype=float)
    ranks = wh_temporal.percentile_rank(_stack(values))

    assert ranks[0, 0, 0] == pytest.approx(0.0)
    assert ranks[-1, 0, 0] == pytest.approx(1.0)
    assert ranks[42, 0, 0] == pytest.approx(42 / 83)


def test_percentile_rank_ignores_gaps():
    values = np.full(84, np.nan)
    values[:4] = [0.0, 1.0, 2.0, 3.0]
    ranks = wh_temporal.percentile_rank(_stack(values))

    assert ranks[0, 0, 0] == pytest.approx(0.0)
    assert ranks[3, 0, 0] == pytest.approx(1.0)
    assert np.isnan(ranks[10, 0, 0])


# --- recency --------------------------------------------------------------


def test_months_since_water_counts_from_the_last_exceedance():
    values = np.full(12, -0.5)
    values[2] = 0.4  # wet in the third month
    month_index = np.arange(12)

    result = wh_temporal.months_since_threshold(_stack(values), month_index, 0.0)

    assert result[2, 0, 0] == pytest.approx(0.0)
    assert result[3, 0, 0] == pytest.approx(1.0)
    assert result[7, 0, 0] == pytest.approx(5.0)


def test_months_since_water_is_left_censored():
    """Never seen wet must be NaN, not a large number invented from nothing."""
    values = np.full(12, -0.5)
    result = wh_temporal.months_since_threshold(_stack(values), np.arange(12), 0.0)
    assert np.all(np.isnan(result[:, 0, 0]))


def test_months_since_water_before_first_exceedance_is_nan():
    values = np.full(12, -0.5)
    values[6] = 0.4
    result = wh_temporal.months_since_threshold(_stack(values), np.arange(12), 0.0)

    assert np.all(np.isnan(result[:6, 0, 0]))
    assert result[6, 0, 0] == pytest.approx(0.0)
