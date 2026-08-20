"""Train and honestly evaluate the per-pixel classifier.

The single most important thing in this module is that **every split is grouped
by site**. Pixels within a tile are almost perfectly autocorrelated, so a random
pixel-level split reports the model's ability to memorise a texture, not to
generalise to a waterhole it has never seen. With 9 labelled sites the effective
sample size is 9, not 31,838, however many pixels the table contains.

That is enforced structurally rather than by convention: this module never
imports `train_test_split` or `KFold`, and `site_splits` is the only way to
produce a split. It raises rather than falling back if it is handed too few
groups to be meaningful.

A second, subtler caveat applies to the temporal holdout: a held-out month's
temporal features were computed from that pixel's whole history, including the
training months. That leak is reported rather than hidden — the grouped-site CV
has no equivalent problem, because holding out a site holds out its history too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wh_features
import wh_naming
import wh_pseudo
import wh_temporal
import wh_tiles
from wh_config import Config
from wh_features import FeatureParams

MIN_GROUPS = 3


# --- training table --------------------------------------------------------


def _label_sources(cfg: Config, include_pseudo: bool) -> list[tuple[Path, str]]:
    directories = [(cfg.paths["labels"], "manual")]
    if include_pseudo:
        directories.append((wh_pseudo.pseudo_label_dir(cfg), "pseudo"))
    return [(directory, source) for directory, source in directories if directory.exists()]


def build_training_table(
    manifest: pd.DataFrame,
    cfg: Config,
    params: FeatureParams,
    include_pseudo: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per labelled pixel, with every feature and its site and month.

    Tiles are processed a site at a time so each site's temporal features are
    computed once rather than once per labelled month.
    """
    wanted: dict[str, list[tuple[Path, str]]] = {}
    for directory, source in _label_sources(cfg, include_pseudo):
        for mask_path in sorted(directory.glob("*_labels.tif")):
            key = wh_naming.parse_stem(mask_path.stem.replace("_labels", ""))
            wanted.setdefault(key.site_id, []).append((mask_path, source))

    if not wanted:
        raise FileNotFoundError(
            f"no label masks found in {[str(d) for d, _ in _label_sources(cfg, include_pseudo)]}"
        )

    tables = []
    for site_id in sorted(wanted):
        stack = wh_temporal.load_site_stack(
            manifest, site_id, cfg, indices=list(params.temporal_indices)
        )
        temporal = wh_temporal.temporal_feature_stack(stack, cfg)
        month_positions = {label: index for index, label in enumerate(stack.year_month)}

        # Static across months, so read once per site rather than per label mask.
        alphaearth = None
        if params.use_alphaearth:
            alphaearth = wh_features.load_alphaearth(
                cfg, site_id, params, expected_shape=stack.shape
            )

        for mask_path, source in wanted[site_id]:
            key = wh_naming.parse_stem(mask_path.stem.replace("_labels", ""))
            position = month_positions.get(key.year_month)
            if position is None:
                print(f"  {mask_path.name}: month not in the site stack, skipped")
                continue

            rows = manifest[
                (manifest["site_id"] == site_id)
                & (manifest["year_month"] == key.year_month)
            ]
            if rows.empty:
                print(f"  {mask_path.name}: no chip in the manifest, skipped")
                continue

            try:
                tile = wh_tiles.read_tile(rows.iloc[0]["tif_path"], cfg)
            except OSError as error:
                print(f"  {mask_path.name}: {str(error).splitlines()[0]}")
                continue

            mask = wh_tiles.read_mask(mask_path, tile.shape)
            selection = mask > 0
            if not selection.any():
                continue

            features = wh_features.assemble_features(
                tile, temporal, position, params, alphaearth=alphaearth
            )
            tables.append(
                wh_features.extract_pixels(
                    features, selection, site_id, key.year_month, mask, source
                )
            )

        if verbose:
            total = sum(len(t) for t in tables)
            print(f"  site {site_id}: {len(wanted[site_id])} mask(s), {total:,} rows so far")

    if not tables:
        raise ValueError("no labelled pixels could be read")

    table = pd.concat(tables, ignore_index=True)
    table["class_id"] = table["class_id"].astype(np.int16)
    return table


def save_table(table: pd.DataFrame, cfg: Config, name: str = "training_table.csv") -> Path:
    path = cfg.paths["derived"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_table(cfg: Config, name: str = "training_table.csv") -> pd.DataFrame:
    path = cfg.paths["derived"] / name
    if not path.exists():
        raise FileNotFoundError(f"no training table at {path}; build it first")
    return pd.read_csv(path, dtype={"site_id": str})


# --- splitting -------------------------------------------------------------


def site_splits(
    table: pd.DataFrame,
    cfg: Config | None = None,
    strategy: str | None = None,
    n_splits: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """The ONLY splitter in this module. Always grouped by site.

    Yields (train_index, test_index, held_out_label). Raises if there are too
    few sites for a grouped split to mean anything, rather than silently
    degrading into something that looks like a result.
    """
    groups = table["site_id"].to_numpy()
    unique = np.unique(groups)

    if unique.size < MIN_GROUPS:
        raise ValueError(
            f"grouped cross-validation needs at least {MIN_GROUPS} sites, got "
            f"{unique.size} ({sorted(unique)}). Label more sites — a split within "
            f"one or two sites measures memorisation, not generalisation."
        )

    settings = (cfg["training"]["cv"] if cfg else {}) or {}
    strategy = strategy or settings.get("strategy", "leave_one_site_out")
    max_for_loso = int(settings.get("max_sites_for_loso", 12))

    if strategy == "leave_one_site_out" and unique.size > max_for_loso:
        strategy = "group_kfold_by_site"

    indices = np.arange(len(table))

    if strategy == "leave_one_site_out":
        splitter = LeaveOneGroupOut()
        for train_index, test_index in splitter.split(indices, groups=groups):
            yield train_index, test_index, str(groups[test_index][0])
    elif strategy == "group_kfold_by_site":
        folds = min(int(n_splits or settings.get("n_splits", 5)), unique.size)
        splitter = GroupKFold(n_splits=folds)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(indices, groups=groups)
        ):
            held = sorted(set(groups[test_index]))
            yield train_index, test_index, f"fold{fold}:{'+'.join(held)}"
    else:
        raise ValueError(
            f"unknown cv strategy {strategy!r}; use 'leave_one_site_out' or "
            f"'group_kfold_by_site'"
        )


def temporal_holdout_split(
    table: pd.DataFrame, holdout_months: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Train on months outside the holdout, test on months inside it.

    Reported alongside the grouped CV and explicitly biased optimistically: the
    temporal features of a held-out month were computed from that pixel's whole
    history, training months included. Removing that leak would mean recomputing
    every temporal feature per fold from training months only.
    """
    month = pd.to_datetime(table["year_month"] + "-01").dt.month.to_numpy()
    test = np.isin(month, holdout_months)
    if not test.any() or test.all():
        raise ValueError(
            f"temporal holdout {holdout_months} selects {test.sum()} of {len(table)} rows; "
            f"pick months that split the table"
        )
    return np.flatnonzero(~test), np.flatnonzero(test)


# --- models ----------------------------------------------------------------


def make_model(name: str, cfg: Config) -> Pipeline:
    """One of the three baselines, in increasing order of capacity.

    Logistic regression is a diagnostic, not a contender: if it does badly while
    the trees do well, the classes are not linearly separable in this feature
    space, which is worth knowing before adding more features.
    """
    seed = int(cfg["training"].get("random_state", 42))
    weight = cfg["training"].get("class_weight", "balanced")

    if name == "logistic_regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight=weight, random_state=seed,
            )),
        ])

    if name == "random_forest":
        return Pipeline([
            # Random forests in sklearn cannot take NaN, so temporal features
            # that could not be fitted are imputed. HistGBM below keeps them as
            # NaN, which is more faithful — "no fit" is itself informative.
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                class_weight="balanced_subsample", random_state=seed,
            )),
        ])

    if name == "gradient_boosting":
        return Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, class_weight=weight,
                random_state=seed,
            )),
        ])

    raise ValueError(f"unknown model {name!r}")


# --- evaluation ------------------------------------------------------------


@dataclass
class Evaluation:
    """Scores for one model over one splitting scheme."""

    model_name: str
    strategy: str
    per_class: pd.DataFrame
    per_site: pd.DataFrame
    confusion: pd.DataFrame
    macro_f1: float
    weighted_f1: float
    n_train: int
    n_test: int
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (
            f"{self.model_name:22s} macro F1 {self.macro_f1:.3f}  "
            f"weighted F1 {self.weighted_f1:.3f}  ({self.n_test:,} test rows)"
        )


def _class_names(cfg: Config, present: np.ndarray) -> list[str]:
    return [cfg.class_by_id(int(class_id)).name for class_id in present]


def evaluate_predictions(
    truth: np.ndarray,
    predicted: np.ndarray,
    sites: np.ndarray,
    cfg: Config,
    model_name: str,
    strategy: str,
) -> Evaluation:
    """Per-class F1 and IoU, a confusion matrix, and a per-site breakdown.

    The per-site table is the one to read: an average over sites hides the case
    where the model works on six waterholes and fails on three, which is exactly
    the failure mode that matters for deploying this across 187 of them.
    """
    labels = np.unique(np.concatenate([truth, predicted]))
    names = _class_names(cfg, labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    iou = jaccard_score(truth, predicted, labels=labels, average=None, zero_division=0)

    per_class = pd.DataFrame({
        "class": names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": support,
    }).set_index("class")

    per_site_rows = []
    for site in np.unique(sites):
        selected = sites == site
        per_site_rows.append({
            "site_id": site,
            "n_pixels": int(selected.sum()),
            "n_classes": int(np.unique(truth[selected]).size),
            "macro_f1": f1_score(
                truth[selected], predicted[selected], average="macro", zero_division=0
            ),
            "accuracy": float((truth[selected] == predicted[selected]).mean()),
        })
    per_site = pd.DataFrame(per_site_rows).set_index("site_id").sort_values("macro_f1")

    matrix = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=labels),
        index=[f"true_{name}" for name in names],
        columns=[f"pred_{name}" for name in names],
    )

    return Evaluation(
        model_name=model_name,
        strategy=strategy,
        per_class=per_class,
        per_site=per_site,
        confusion=matrix,
        macro_f1=f1_score(truth, predicted, average="macro", zero_division=0),
        weighted_f1=f1_score(truth, predicted, average="weighted", zero_division=0),
        n_train=0,
        n_test=len(truth),
    )


def cross_validate(
    table: pd.DataFrame,
    cfg: Config,
    model_name: str,
    feature_names: list[str] | None = None,
    strategy: str | None = None,
    verbose: bool = True,
) -> Evaluation:
    """Grouped cross-validation. Every fold holds out whole sites."""
    features = feature_names or wh_features.feature_columns(table)
    values = table[features].to_numpy(dtype=np.float64)
    target = table["class_id"].to_numpy()
    sites = table["site_id"].to_numpy()

    truth_parts, predicted_parts, site_parts = [], [], []
    n_train = 0

    for train_index, test_index, held_out in site_splits(table, cfg, strategy):
        if np.unique(target[train_index]).size < 2:
            print(f"  {held_out}: only one class in training, fold skipped")
            continue

        model = make_model(model_name, cfg)
        model.fit(values[train_index], target[train_index])
        predicted = model.predict(values[test_index])

        truth_parts.append(target[test_index])
        predicted_parts.append(predicted)
        site_parts.append(sites[test_index])
        n_train += len(train_index)

        if verbose:
            fold_f1 = f1_score(
                target[test_index], predicted, average="macro", zero_division=0
            )
            print(f"  held out {held_out:>18s}: macro F1 {fold_f1:.3f} "
                  f"({len(test_index):,} px, {np.unique(target[test_index]).size} classes)")

    if not truth_parts:
        raise ValueError("no usable folds; check the class coverage per site")

    evaluation = evaluate_predictions(
        np.concatenate(truth_parts), np.concatenate(predicted_parts),
        np.concatenate(site_parts), cfg, model_name,
        strategy or cfg["training"]["cv"]["strategy"],
    )
    evaluation.n_train = n_train
    return evaluation


# --- ablation --------------------------------------------------------------


def feature_blocks(table: pd.DataFrame, params: FeatureParams) -> dict[str, list[str]]:
    """Partition the feature columns into the blocks the ablation compares."""
    available = set(wh_features.feature_columns(table))
    instantaneous = [
        name for name in wh_features.instantaneous_feature_names(params) if name in available
    ]
    embeddings = sorted(
        name for name in available if name.startswith(wh_features.ALPHAEARTH_PREFIX)
    )
    temporal = sorted(available - set(instantaneous) - set(embeddings))
    groups = wh_temporal.split_feature_names(temporal)

    return {
        "instantaneous": instantaneous,
        "temporal_model_free": groups["model_free"],
        "temporal_trend": groups["trend"],
        "temporal_harmonic": groups["harmonic"],
        "alphaearth": embeddings,
    }


def ablation_sets(blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    """The feature sets to compare, from richest to leanest.

    'instantaneous_only' is the important one: it tests the design's central
    claim, that normalising a pixel against its own history is what makes the
    ambiguous dry-season months separable. If it scores as well as everything
    else, the temporal machinery is not earning its keep.
    """
    everything = sum(blocks.values(), [])
    embeddings = blocks.get("alphaearth", [])
    spectral = (
        blocks["instantaneous"] + blocks["temporal_model_free"] + blocks["temporal_trend"]
        + blocks["temporal_harmonic"]
    )

    # Every name here describes exactly the columns it contains. An earlier
    # version quietly folded the embeddings into "instantaneous_only", which made
    # a 92-column set look like a 28-column one and the comparison meaningless.
    sets = {
        "all_features": everything,
        "spectral_only": spectral,
        "no_harmonic": blocks["instantaneous"] + blocks["temporal_model_free"]
                       + blocks["temporal_trend"],
        "model_free_temporal": blocks["instantaneous"] + blocks["temporal_model_free"],
        "instantaneous_only": blocks["instantaneous"],
    }

    if embeddings:
        # What the embeddings add on top of the spectral features, and how far
        # they get without them.
        sets["embeddings_only"] = embeddings
        sets["instantaneous_plus_embeddings"] = blocks["instantaneous"] + embeddings

    return sets


def run_ablation(
    table: pd.DataFrame,
    cfg: Config,
    params: FeatureParams,
    model_name: str = "gradient_boosting",
    verbose: bool = False,
) -> pd.DataFrame:
    """Cross-validate each feature set and compare. Returns a tidy table.

    Feature sets that turn out to contain exactly the same columns are scored
    once and marked, rather than cross-validated repeatedly and reported as
    separate results. This happens whenever a block is empty — with
    `harmonic_enabled: false`, for instance, 'all_features', 'no_harmonic' and
    'model_free_temporal' are the same 57 columns, and running all three would
    triple the work to produce three identical numbers that look like a bug.
    """
    blocks = feature_blocks(table, params)
    empty = [name for name, columns in blocks.items() if not columns]
    if empty:
        print(f"  empty feature block(s): {', '.join(empty)}")

    scored: dict[tuple[str, ...], str] = {}
    rows = []

    for label, features in ablation_sets(blocks).items():
        if not features:
            print(f"  {label:22s} skipped — no features")
            continue

        signature = tuple(sorted(features))
        duplicate_of = scored.get(signature)

        if duplicate_of is not None:
            rows.append({
                "feature_set": label,
                "n_features": len(features),
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "identical_to": duplicate_of,
            })
            print(f"  {label:22s} {len(features):>4d} features  "
                  f"identical to '{duplicate_of}', not re-run")
            continue

        scored[signature] = label
        evaluation = cross_validate(table, cfg, model_name, features, verbose=verbose)
        rows.append({
            "feature_set": label,
            "n_features": len(features),
            "macro_f1": evaluation.macro_f1,
            "weighted_f1": evaluation.weighted_f1,
            "identical_to": "",
        })
        print(f"  {label:22s} {len(features):>4d} features  "
              f"macro F1 {evaluation.macro_f1:.3f}")

    result = pd.DataFrame(rows).set_index("feature_set")

    # Carry the score onto the rows that were not re-run, so the table reads
    # sensibly while still saying which numbers came from the same fit.
    for label, row in result.iterrows():
        if row["identical_to"]:
            result.loc[label, ["macro_f1", "weighted_f1"]] = result.loc[
                row["identical_to"], ["macro_f1", "weighted_f1"]
            ].to_numpy()

    baseline = (
        result.loc["instantaneous_only", "macro_f1"]
        if "instantaneous_only" in result.index
        else np.nan
    )
    result["gain_over_instantaneous"] = result["macro_f1"] - baseline

    if len(scored) < 2:
        print("\n  Only one distinct feature set was scored, so this ablation is not "
              "informative.\n  Set features.temporal.harmonic_enabled: true in the config "
              "and rebuild the\n  training table to compare against the harmonic block.")

    return result


# --- predicting whole tiles, for spatial inspection ------------------------


def fit_without_site(
    table: pd.DataFrame,
    cfg: Config,
    model_name: str,
    held_out_site: str,
    feature_names: list[str] | None = None,
):
    """Fit on every site except one, so that site can be predicted honestly.

    Looking at predictions from a model that was trained on the same waterhole
    tells you nothing — it will look excellent and mean nothing. Every spatial
    check should use a model that has never seen the site it is drawing.
    """
    features = feature_names or wh_features.feature_columns(table)
    training = table[table["site_id"] != held_out_site]

    if training.empty:
        raise ValueError(f"no training rows left after holding out site {held_out_site}")
    if training["site_id"].nunique() < 2:
        raise ValueError(
            f"only {training['site_id'].nunique()} site(s) left after holding out "
            f"{held_out_site}; the fit would be meaningless"
        )

    model = make_model(model_name, cfg)
    model.fit(training[features].to_numpy(dtype=np.float64), training["class_id"].to_numpy())
    return model


def predict_tile(
    model,
    feature_names: list[str],
    tile: wh_tiles.Tile,
    temporal_features: dict[str, np.ndarray],
    month_position: int,
    params: FeatureParams,
    with_confidence: bool = True,
    alphaearth: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Classify every observed pixel of one tile.

    Returns (class raster, confidence raster). Pixels with no clear observation
    are left as 0 rather than guessed at, so a gap never appears as a confident
    surface class.
    """
    features = wh_features.assemble_features(
        tile, temporal_features, month_position, params, alphaearth=alphaearth
    )

    missing = [name for name in feature_names if name not in features]
    if missing:
        raise KeyError(
            f"the model needs {len(missing)} feature(s) the tile does not provide, "
            f"e.g. {missing[:5]}. "
            + (
                "The model was trained with AlphaEarth embeddings but "
                "params.use_alphaearth is False here — set it True."
                if any(n.startswith(wh_features.ALPHAEARTH_PREFIX) for n in missing)
                else "Was the model trained under a different config?"
            )
        )

    stacked = np.stack([features[name] for name in feature_names], axis=-1)
    flat = stacked.reshape(-1, len(feature_names)).astype(np.float64)
    observed = tile.valid.reshape(-1)

    predicted = np.zeros(flat.shape[0], dtype=np.uint8)
    confidence = np.full(flat.shape[0], np.nan) if with_confidence else None

    if observed.any():
        predicted[observed] = model.predict(flat[observed]).astype(np.uint8)
        if with_confidence and hasattr(model, "predict_proba"):
            confidence[observed] = model.predict_proba(flat[observed]).max(axis=1)

    return (
        predicted.reshape(tile.shape),
        confidence.reshape(tile.shape) if confidence is not None else None,
    )


class SitePredictor:
    """Predict any month of one site, loading that site's history once.

    Temporal features need the whole 84-month stack, which is most of the cost
    of a prediction. Constructing this once per site and calling predict() per
    month turns an 8-second-per-month job into a one-off load plus well under a
    second each.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cfg: Config,
        params: FeatureParams,
        site_id: str,
    ) -> None:
        self.manifest = manifest
        self.cfg = cfg
        self.params = params
        self.site_id = site_id
        self.stack = wh_temporal.load_site_stack(
            manifest, site_id, cfg, indices=list(params.temporal_indices)
        )
        self.temporal = wh_temporal.temporal_feature_stack(self.stack, cfg)

        # Static across months, so loaded once with the rest of the site. Without
        # this a model trained with embeddings cannot predict anything, because
        # the tile would be missing 64 of its features.
        self.alphaearth = None
        if params.use_alphaearth:
            self.alphaearth = wh_features.load_alphaearth(
                cfg, site_id, params, expected_shape=self.stack.shape
            )

    @property
    def months(self) -> list[str]:
        return list(self.stack.year_month)

    def tile(self, year_month: str) -> tuple[wh_tiles.Tile, int]:
        """The chip for one month, plus its position in the temporal stack."""
        if year_month not in self.stack.year_month:
            raise KeyError(
                f"site {self.site_id} has no month {year_month}; available "
                f"{self.stack.year_month[0]}..{self.stack.year_month[-1]}"
            )
        position = self.stack.year_month.index(year_month)
        rows = self.manifest[
            (self.manifest["site_id"] == self.site_id)
            & (self.manifest["year_month"] == year_month)
        ]
        return wh_tiles.read_tile(rows.iloc[0]["tif_path"], self.cfg), position

    def predict(
        self, model, feature_names: list[str], year_month: str,
        with_confidence: bool = True,
    ) -> tuple[wh_tiles.Tile, np.ndarray, np.ndarray | None]:
        """Returns (tile, class raster, confidence raster) for one month."""
        tile, position = self.tile(year_month)
        predicted, confidence = predict_tile(
            model, feature_names, tile, self.temporal, position, self.params,
            with_confidence=with_confidence, alphaearth=self.alphaearth,
        )
        return tile, predicted, confidence


def prepare_site_prediction(
    manifest: pd.DataFrame,
    cfg: Config,
    params: FeatureParams,
    site_id: str,
    year_month: str,
):
    """One-shot convenience wrapper around SitePredictor.

    Returns (tile, temporal_features, month_position). For more than one month
    of the same site, build a SitePredictor instead — this reloads the whole
    84-month stack every call.
    """
    predictor = SitePredictor(manifest, cfg, params, site_id)
    tile, position = predictor.tile(year_month)
    return tile, predictor.temporal, position


def labelled_tiles(cfg: Config, include_pseudo: bool = False) -> pd.DataFrame:
    """Index of tiles that have hand labels, for choosing what to inspect."""
    rows = []
    for directory, source in _label_sources(cfg, include_pseudo):
        for sidecar in sorted(directory.glob("*_labels.json")):
            meta = json.loads(sidecar.read_text())
            rows.append({
                "site_id": meta["site_id"],
                "year_month": meta["year_month"],
                "n_labelled": meta["n_labelled"],
                "n_classes": sum(
                    1 for name, count in meta["pixel_counts"].items()
                    if count and name != "unlabelled"
                ),
                "source": source,
                "mask_path": meta["label_mask"],
            })
    return pd.DataFrame(rows)


# --- feature importance ----------------------------------------------------


def permutation_importance_by_site(
    table: pd.DataFrame,
    cfg: Config,
    model_name: str = "logistic_regression",
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Permutation importance measured on each held-out site, then pooled.

    Importance is computed on the site the model has NOT seen, so it ranks
    features by how much they help generalise to a new waterhole — which is the
    question — rather than how much they help fit the sites already labelled.

    The across-fold spread matters as much as the mean: a feature that looks
    essential in one fold and useless in the rest is not a finding, and averaging
    alone would hide that. `std_across_folds` and `n_folds_positive` are reported
    for exactly that reason.

    Columns are named for FOLDS, not sites, because they are only the same thing
    under leave-one-site-out. Above `max_sites_for_loso` the splitter falls back
    to k-fold and a fold holds several sites.
    """
    from sklearn.inspection import permutation_importance

    features = feature_names or wh_features.feature_columns(table)
    values = table[features].to_numpy(dtype=np.float64)
    target = table["class_id"].to_numpy()

    per_fold = []
    for train_index, test_index, held_out in site_splits(table, cfg):
        if np.unique(target[train_index]).size < 2 or len(test_index) < 20:
            continue

        model = make_model(model_name, cfg)
        model.fit(values[train_index], target[train_index])

        result = permutation_importance(
            model, values[test_index], target[test_index],
            scoring="f1_macro", n_repeats=n_repeats,
            random_state=int(cfg["training"].get("random_state", 42)),
            n_jobs=-1,
        )
        per_fold.append(pd.Series(result.importances_mean, index=features, name=held_out))

        if verbose:
            top = per_fold[-1].nlargest(3)
            print(f"  held out {held_out}: top {', '.join(f'{k} {v:.3f}' for k, v in top.items())}")

    if not per_fold:
        raise ValueError("no usable folds for permutation importance")

    folds = pd.concat(per_fold, axis=1)
    return pd.DataFrame({
        "mean_importance": folds.mean(axis=1),
        "std_across_folds": folds.std(axis=1),
        "n_folds_positive": (folds > 0).sum(axis=1),
        "n_folds": folds.shape[1],
    }).sort_values("mean_importance", ascending=False)


def band_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Restrict an importance table to the AlphaEarth bands, named A00..A63."""
    embeddings = importance[
        importance.index.str.startswith(wh_features.ALPHAEARTH_PREFIX)
    ].copy()
    embeddings.index = embeddings.index.str.removeprefix(wh_features.ALPHAEARTH_PREFIX)
    return embeddings


def select_top_bands(importance: pd.DataFrame, n: int = 10) -> list[str]:
    """The n best-ranked embedding bands.

    A caution that belongs with the result rather than after it: choosing bands
    on the same folds that then report the score is circular, and the reported
    score will be optimistic. To quote a number for a selected subset, either
    nest the selection inside each fold or re-score on sites held out of the
    selection entirely.
    """
    return band_importance(importance).head(n).index.tolist()


# --- persistence -----------------------------------------------------------


def save_model(
    model,
    feature_names: list[str],
    cfg: Config,
    params: FeatureParams,
    evaluation: Evaluation,
    table: pd.DataFrame,
    name: str = "classifier",
) -> tuple[Path, Path]:
    """Persist the fitted model beside everything needed to interpret it.

    The feature list and config hash travel with the model because a model
    applied with features in a different order, or built under a different
    config, produces confident nonsense rather than an error.
    """
    directory = cfg.paths["derived"] / "models"
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / f"{name}.joblib"
    joblib.dump(model, model_path)

    manifest = {
        "name": name,
        "model_class": type(model[-1]).__name__ if hasattr(model, "__getitem__") else type(model).__name__,
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "config_hash": cfg.hash,
        "class_scheme_version": cfg["classes"]["scheme_version"],
        "classes": {d.id: d.name for d in cfg.classes},
        "feature_params": params.as_dict(),
        "cv_strategy": evaluation.strategy,
        "cv_macro_f1": evaluation.macro_f1,
        "cv_weighted_f1": evaluation.weighted_f1,
        "training_sites": sorted(table["site_id"].unique().tolist()),
        "n_training_pixels": int(len(table)),
        "label_sources": table["source"].value_counts().to_dict(),
        "class_counts": table["class_id"].value_counts().sort_index().to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = directory / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))

    return model_path, manifest_path


def load_model(cfg: Config, name: str = "classifier") -> tuple[object, dict]:
    """Load a persisted model and its manifest."""
    directory = cfg.paths["derived"] / "models"
    model_path = directory / f"{name}.joblib"
    manifest_path = directory / f"{name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"no model at {model_path}")
    return joblib.load(model_path), json.loads(manifest_path.read_text())
