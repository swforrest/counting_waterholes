"""Tests for the grouped cross-validation splitter.

The splitter is the load-bearing piece of the evaluation: if a site can appear on
both sides of a split, every number this pipeline reports is inflated. These
tests exist to make that failure impossible to introduce quietly.
"""

import numpy as np
import pandas as pd
import pytest

import wh_train


def _table(n_sites: int, rows_per_site: int = 20) -> pd.DataFrame:
    """Synthetic training table with a known site structure."""
    rng = np.random.default_rng(0)
    sites = [f"{index:03d}" for index in range(n_sites)]
    return pd.DataFrame({
        "site_id": np.repeat(sites, rows_per_site),
        "year_month": np.tile(
            [f"2024-{1 + (i % 12):02d}" for i in range(rows_per_site)], n_sites
        ),
        "row": rng.integers(0, 100, n_sites * rows_per_site),
        "col": rng.integers(0, 100, n_sites * rows_per_site),
        "class_id": rng.integers(1, 7, n_sites * rows_per_site),
        "source": "manual",
        "feature_a": rng.normal(size=n_sites * rows_per_site),
        "feature_b": rng.normal(size=n_sites * rows_per_site),
    })


def _Cfg(strategy="leave_one_site_out", max_for_loso=12, n_splits=5):
    """Training params for the splitter tests, named for the old stand-in."""
    return wh_train.TrainParams(
        cv_strategy=strategy, max_sites_for_loso=max_for_loso, n_splits=n_splits,
    )


# --- the guarantee that matters -------------------------------------------


def test_no_site_ever_appears_on_both_sides():
    table = _table(9)
    for train_index, test_index, _ in wh_train.site_splits(table, _Cfg()):
        train_sites = set(table.iloc[train_index]["site_id"])
        test_sites = set(table.iloc[test_index]["site_id"])
        assert train_sites.isdisjoint(test_sites)


def test_leave_one_site_out_gives_one_fold_per_site():
    table = _table(9)
    splits = list(wh_train.site_splits(table, _Cfg()))
    assert len(splits) == 9
    held_out = [label for _, _, label in splits]
    assert sorted(held_out) == sorted(table["site_id"].unique())


def test_each_loso_fold_holds_out_exactly_one_site():
    table = _table(9)
    for _, test_index, label in wh_train.site_splits(table, _Cfg()):
        assert set(table.iloc[test_index]["site_id"]) == {label}


def test_every_row_is_tested_exactly_once_under_loso():
    table = _table(6)
    tested = np.concatenate(
        [test_index for _, test_index, _ in wh_train.site_splits(table, _Cfg())]
    )
    assert sorted(tested) == list(range(len(table)))


# --- refusing to produce a meaningless split ------------------------------


@pytest.mark.parametrize("n_sites", [1, 2])
def test_too_few_sites_raises(n_sites):
    table = _table(n_sites)
    with pytest.raises(ValueError, match="at least 3 sites"):
        list(wh_train.site_splits(table, _Cfg()))


def test_the_error_says_what_to_do_about_it():
    with pytest.raises(ValueError, match="memorisation"):
        list(wh_train.site_splits(_table(2), _Cfg()))


def test_unknown_strategy_raises_rather_than_defaulting():
    table = _table(9)
    with pytest.raises(ValueError, match="unknown cv strategy"):
        list(wh_train.site_splits(table, _Cfg(), strategy="random"))


def test_there_is_no_ungrouped_splitter_imported():
    """An ungrouped split must not be reachable, even by importing one.

    Checked against the parsed imports rather than the file text, so the module
    can still discuss these names in its docstring — which it does, to explain
    why they are absent.
    """
    import ast

    tree = ast.parse(open(wh_train.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    forbidden = {"train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit",
                 "StratifiedShuffleSplit"}
    assert not (imported & forbidden), (
        f"wh_train imports ungrouped splitter(s): {sorted(imported & forbidden)}"
    )


def test_only_grouped_splitters_are_imported():
    import ast

    tree = ast.parse(open(wh_train.__file__).read())
    from_sklearn_ms = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.model_selection"
        for alias in node.names
    }
    assert from_sklearn_ms == {"GroupKFold", "LeaveOneGroupOut"}


# --- strategy selection ---------------------------------------------------


def test_group_kfold_holds_out_whole_sites():
    table = _table(20)
    for train_index, test_index, _ in wh_train.site_splits(
        table, _Cfg(strategy="group_kfold_by_site", n_splits=4)
    ):
        train_sites = set(table.iloc[train_index]["site_id"])
        test_sites = set(table.iloc[test_index]["site_id"])
        assert train_sites.isdisjoint(test_sites)


def test_loso_switches_to_group_kfold_above_the_threshold():
    """With many sites, LOSO would mean many folds; it degrades to k-fold."""
    table = _table(30)
    splits = list(wh_train.site_splits(table, _Cfg(max_for_loso=12, n_splits=5)))
    assert len(splits) == 5


def test_group_kfold_folds_are_capped_by_site_count():
    table = _table(4)
    splits = list(wh_train.site_splits(table, _Cfg(strategy="group_kfold_by_site", n_splits=10)))
    assert len(splits) == 4


# --- temporal holdout -----------------------------------------------------


def test_temporal_holdout_splits_by_calendar_month():
    table = _table(5, rows_per_site=24)
    train_index, test_index = wh_train.temporal_holdout_split(table, [8, 9, 10])

    months = pd.to_datetime(table["year_month"] + "-01").dt.month
    assert set(months.iloc[test_index]) <= {8, 9, 10}
    assert not set(months.iloc[train_index]) & {8, 9, 10}


def test_temporal_holdout_refuses_an_empty_or_total_split():
    table = _table(5, rows_per_site=24)
    with pytest.raises(ValueError, match="split the table"):
        wh_train.temporal_holdout_split(table, list(range(1, 13)))


# --- ablation feature sets ------------------------------------------------


def test_ablation_sets_are_strictly_nested():
    blocks = {
        "instantaneous": ["a", "b"],
        "temporal_model_free": ["c"],
        "temporal_trend": ["d"],
        "temporal_harmonic": ["e", "f"],
        "alphaearth": [],
    }
    sets = wh_train.ablation_sets(blocks)

    assert set(sets["instantaneous_only"]) < set(sets["model_free_temporal"])
    assert set(sets["model_free_temporal"]) < set(sets["no_harmonic"])
    assert set(sets["no_harmonic"]) < set(sets["all_features"])
    assert set(sets["all_features"]) == {"a", "b", "c", "d", "e", "f"}


def test_every_ablation_set_contains_exactly_what_its_name_says():
    """A set called 'instantaneous_only' must not quietly include embeddings."""
    blocks = {
        "instantaneous": ["a", "b"],
        "temporal_model_free": ["c"],
        "temporal_trend": ["d"],
        "temporal_harmonic": ["e"],
        "alphaearth": ["ae_A00", "ae_A01"],
    }
    sets = wh_train.ablation_sets(blocks)
    embeddings = set(blocks["alphaearth"])

    assert set(sets["instantaneous_only"]) == {"a", "b"}
    assert not set(sets["instantaneous_only"]) & embeddings
    assert not set(sets["spectral_only"]) & embeddings
    assert not set(sets["no_harmonic"]) & embeddings
    assert not set(sets["model_free_temporal"]) & embeddings

    assert set(sets["embeddings_only"]) == embeddings
    assert set(sets["instantaneous_plus_embeddings"]) == {"a", "b"} | embeddings
    assert embeddings < set(sets["all_features"])


def test_embedding_sets_are_absent_when_there_are_no_embeddings():
    blocks = {
        "instantaneous": ["a"], "temporal_model_free": ["c"],
        "temporal_trend": [], "temporal_harmonic": [], "alphaearth": [],
    }
    sets = wh_train.ablation_sets(blocks)
    assert "embeddings_only" not in sets
    assert "instantaneous_plus_embeddings" not in sets


def test_spectral_only_is_all_features_minus_embeddings():
    blocks = {
        "instantaneous": ["a"], "temporal_model_free": ["c"],
        "temporal_trend": ["d"], "temporal_harmonic": ["e"],
        "alphaearth": ["ae_A00"],
    }
    sets = wh_train.ablation_sets(blocks)
    assert set(sets["all_features"]) - set(sets["spectral_only"]) == {"ae_A00"}


def test_no_harmonic_keeps_the_trend_term():
    blocks = {
        "instantaneous": ["a"],
        "temporal_model_free": ["c"],
        "temporal_trend": ["mndwi_harm_trend_per_year"],
        "temporal_harmonic": ["mndwi_harm_amplitude"],
        "alphaearth": [],
    }
    sets = wh_train.ablation_sets(blocks)
    assert "mndwi_harm_trend_per_year" in sets["no_harmonic"]
    assert "mndwi_harm_amplitude" not in sets["no_harmonic"]


# --- strategy override ----------------------------------------------------


def test_explicit_loso_is_not_overruled_by_the_size_cap():
    """The cap guards the config default; it must not overrule a direct request."""
    table = _table(30)
    splits = list(wh_train.site_splits(
        table, _Cfg(max_for_loso=15), strategy="leave_one_site_out"
    ))
    assert len(splits) == 30


def test_config_default_still_falls_back_above_the_cap():
    table = _table(30)
    splits = list(wh_train.site_splits(table, _Cfg(max_for_loso=15, n_splits=5)))
    assert len(splits) == 5


def test_n_splits_can_be_overridden_per_call():
    table = _table(30)
    for wanted in (3, 4, 6):
        splits = list(wh_train.site_splits(
            table, _Cfg(n_splits=5), strategy="group_kfold_by_site", n_splits=wanted
        ))
        assert len(splits) == wanted


def test_explicit_kfold_below_the_cap_is_honoured():
    """Asking for k-fold with few sites must not silently give LOSO."""
    table = _table(6)
    splits = list(wh_train.site_splits(
        table, _Cfg(), strategy="group_kfold_by_site", n_splits=3
    ))
    assert len(splits) == 3
