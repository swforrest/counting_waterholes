"""
Model assessment for the waterhole detector.

These build on the per-image "*.details.csv" files written alongside the normal
comparison CSVs by compare_detections_to_ground_truth(). Run that step first;
everything here only reads its output, so the analyses are cheap to re-run and
never touch the detection pipeline.

Three questions the confusion matrix alone does not answer:

  classification_report_AF  Per-class precision / recall / F1. The confusion
                            matrix reports only overall accuracy, which lumps
                            "missed the waterhole entirely" together with "found
                            it but called it the wrong type" - two problems with
                            completely different fixes.

  localisation_quality_AF   How TIGHT the boxes are, via the distribution of
                            overlap among matched pairs, plus a sweep of the
                            matching threshold. This is how you choose
                            COMPARE_IOU_THRESHOLD and class_iou_threshold on
                            evidence rather than guesswork.

  recall_by_size_AF         Recall broken down by waterhole size. An aggregate
                            recall of 0.85 can hide 0.95 on large waterholes and
                            0.40 on small ones, which would change how the
                            ecological results should be read.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .waterhole_classes import load_class_registry

DETAILS_SUFFIX = ".details.csv"


def _plots_dir(run_folder: str) -> str:
    out = os.path.join(run_folder, "plots")
    os.makedirs(out, exist_ok=True)
    return out


def load_details(run_folder: str) -> pd.DataFrame:
    """
    Load every per-image "*.details.csv" in a run folder into one DataFrame.

    Args:
        run_folder: the folder compare_detections_to_ground_truth() wrote into

    Returns:
        One row per comparison entry, with an added "image" column naming the
        source file. Raises FileNotFoundError if the details files are missing,
        which means compare_detections_to_ground_truth() has not been run since
        this reporting was added.
    """
    files = [f for f in os.listdir(run_folder) if f.endswith(DETAILS_SUFFIX)]
    if not files:
        raise FileNotFoundError(
            f"No '*{DETAILS_SUFFIX}' files in {run_folder}. Run "
            "compare_detections_to_ground_truth() first - it writes them next to "
            "the comparison CSVs."
        )
    frames = []
    for f in sorted(files):
        df = pd.read_csv(os.path.join(run_folder, f))
        df["image"] = f[: -len(DETAILS_SUFFIX)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Per-class precision / recall / F1
# ---------------------------------------------------------------------------
def classification_report_AF(run_folder, config, save=True):
    """
    Per-class precision, recall and F1, plus the detection/classification split.

    Reports two different things, because they fail for different reasons:

      Detection      Did we find the waterhole at all, ignoring which class we
                     called it? Poor recall here means the model is blind to
                     waterholes, or the clustering is dropping them.
      Classification Given that we found it, did we get the type right? Computed
                     over matched pairs only, so it is not contaminated by
                     misses.

    Args:
        run_folder: folder holding the comparison output
        config: parsed config dict (or path), for the class names
        save: write a CSV of the per-class table into the run folder

    Returns:
        DataFrame of the per-class table.
    """
    from .testing import parse_config

    if isinstance(config, str):
        config = parse_config(config)
    registry = load_class_registry(config)
    df = load_details(run_folder)

    matched = df[df["match_type"] == "matched"]
    n_fp = int((df["match_type"] == "false_positive").sum())
    n_fn = int((df["match_type"] == "false_negative").sum())
    n_tp = len(matched)

    # --- detection, ignoring class ---
    det_precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else float("nan")
    det_recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else float("nan")
    det_f1 = (
        2 * det_precision * det_recall / (det_precision + det_recall)
        if det_precision and det_recall and not np.isnan(det_precision + det_recall)
        else float("nan")
    )

    # --- classification, over matched pairs only ---
    cls_acc = (
        float((matched["ml_class"] == matched["manual_class"]).mean())
        if n_tp
        else float("nan")
    )

    print("=" * 66)
    print("DETECTION (did we find the waterhole at all, class ignored)")
    print("=" * 66)
    print(f"  true positives        {n_tp}")
    print(f"  false positives       {n_fp}   (detected, no matching label)")
    print(f"  false negatives       {n_fn}   (labelled, not detected)")
    print(f"  precision             {det_precision:.3f}")
    print(f"  recall                {det_recall:.3f}")
    print(f"  F1                    {det_f1:.3f}")
    print()
    print("=" * 66)
    print("CLASSIFICATION (of the ones we found, did we get the type right)")
    print("=" * 66)
    print(f"  accuracy on matched   {cls_acc:.3f}  over {n_tp} matched pairs")
    print()

    # --- per class ---
    rows = []
    for cid in registry.ids:
        name = registry.id_to_name[cid]
        # TP: matched AND both sides agree on this class
        tp = int(((matched["manual_class"] == cid) & (matched["ml_class"] == cid)).sum())
        # FP: predicted this class but the label says otherwise (or nothing there)
        fp = int(((df["ml_class"] == cid) & (df["manual_class"] != cid)).sum())
        # FN: labelled this class but we predicted otherwise (or missed it)
        fn = int(((df["manual_class"] == cid) & (df["ml_class"] != cid)).sum())
        support = int((df["manual_class"] == cid).sum())

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (tp + fp) and (tp + fn) and (precision + recall) > 0
            else float("nan")
        )
        rows.append(
            {
                "class": name,
                "class_id": cid,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 3) if not np.isnan(precision) else np.nan,
                "recall": round(recall, 3) if not np.isnan(recall) else np.nan,
                "f1": round(f1, 3) if not np.isnan(f1) else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    print("=" * 66)
    print("PER CLASS (detection + correct type together)")
    print("=" * 66)
    print(table.to_string(index=False))
    print()
    print("'support' is how many waterholes of that class you labelled.")
    print("Treat classes with small support as indicative only.")

    if save:
        out = os.path.join(run_folder, "classification_report.csv")
        table.to_csv(out, index=False)
        print(f"\nSaved {out}")
    return table


# ---------------------------------------------------------------------------
# 2. Localisation quality and threshold sensitivity
# ---------------------------------------------------------------------------
def localisation_quality_AF(run_folder, config, thresholds=None, save=True):
    """
    How well the boxes line up, and how sensitive the results are to the
    matching threshold.

    Two outputs:

      Overlap distribution  For matched pairs, the spread of the overlap score.
                            Values bunched near the threshold mean the boxes are
                            only just qualifying and localisation is weak; values
                            up near 0.8-0.9 mean tight boxes and room to raise
                            COMPARE_IOU_THRESHOLD.

      Threshold sweep       Precision, recall and F1 as the matching threshold
                            varies. A sharp fall between 0.5 and 0.6 says the
                            model finds waterholes but boxes them loosely.

    The sweep uses each box's BEST overlap with any box on the other side, so it
    approximates the strict one-to-one matching used by compare(). The two agree
    unless several boxes contend for the same partner, which is rare once
    detections have been clustered.

    Args:
        run_folder: folder holding the comparison output
        config: parsed config dict (or path)
        thresholds: iterable of overlap thresholds to sweep. Defaults to
            0.05 ... 0.95 in steps of 0.05.
        save: write the sweep CSV and figures

    Returns:
        (summary dict, sweep DataFrame)
    """
    from .testing import parse_config

    if isinstance(config, str):
        config = parse_config(config)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    df = load_details(run_folder)
    matched = df[df["match_type"] == "matched"]
    overlaps = matched["overlap"].dropna().to_numpy(dtype=float)

    current = config.get("COMPARE_IOU_THRESHOLD", None)
    metric = config.get("OVERLAP_METRIC", "iou")

    summary = {
        "overlap_metric": metric,
        "current_threshold": current,
        "n_matched": int(len(overlaps)),
        "mean": float(np.mean(overlaps)) if len(overlaps) else float("nan"),
        "median": float(np.median(overlaps)) if len(overlaps) else float("nan"),
        "p25": float(np.percentile(overlaps, 25)) if len(overlaps) else float("nan"),
        "p75": float(np.percentile(overlaps, 75)) if len(overlaps) else float("nan"),
        "min": float(np.min(overlaps)) if len(overlaps) else float("nan"),
        "max": float(np.max(overlaps)) if len(overlaps) else float("nan"),
    }

    print("=" * 66)
    print(f"LOCALISATION QUALITY  (metric: {metric})")
    print("=" * 66)
    print(f"  matched pairs         {summary['n_matched']}")
    print(f"  current threshold     {current}")
    print(f"  overlap  mean         {summary['mean']:.3f}")
    print(f"           median       {summary['median']:.3f}")
    print(f"           25th / 75th  {summary['p25']:.3f} / {summary['p75']:.3f}")
    print(f"           min / max    {summary['min']:.3f} / {summary['max']:.3f}")
    if not np.isnan(summary["median"]) and current is not None:
        headroom = summary["median"] - float(current)
        if headroom < 0.1:
            print(
                "\n  NOTE: the median overlap is barely above the threshold, so the\n"
                "  boxes are only just qualifying. Localisation is the weak point,\n"
                "  and raising the threshold would drop many true matches."
            )
        else:
            print(
                f"\n  Median sits {headroom:.2f} above the threshold, so there is room\n"
                "  to tighten it if you want stricter matching."
            )

    # --- sweep ---
    gt = df[df["match_type"].isin(["matched", "false_negative"])]
    ml = df[df["match_type"].isin(["matched", "false_positive"])]
    gt_best = gt["manual_best_overlap"].fillna(0.0).to_numpy(dtype=float)
    ml_best = ml["ml_best_overlap"].fillna(0.0).to_numpy(dtype=float)
    n_gt, n_ml = len(gt_best), len(ml_best)

    rows = []
    for t in thresholds:
        tp_r = int((gt_best >= t).sum())      # labels with a good enough detection
        tp_p = int((ml_best >= t).sum())      # detections with a good enough label
        recall = tp_r / n_gt if n_gt else float("nan")
        precision = tp_p / n_ml if n_ml else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(t),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "matched_labels": tp_r,
                "total_labels": n_gt,
                "matched_detections": tp_p,
                "total_detections": n_ml,
            }
        )
    sweep = pd.DataFrame(rows)

    print("\n" + "=" * 66)
    print("THRESHOLD SWEEP")
    print("=" * 66)
    print(sweep[["threshold", "precision", "recall", "f1"]].to_string(index=False))
    if len(sweep):
        # Deliberately NOT reporting "best F1". In this sweep a lower threshold
        # can only add matches, so precision, recall and F1 all rise as the
        # threshold falls - F1 is maximal at the lowest threshold by
        # construction, and quoting it would just recommend 0.05 every time.
        # What the curve genuinely shows is how fast matches disappear as you
        # demand tighter boxes, i.e. how well localised the detections are.
        peak = float(sweep["recall"].max())
        knee_90 = sweep[sweep["recall"] >= 0.9 * peak]["threshold"].max()
        knee_75 = sweep[sweep["recall"] >= 0.75 * peak]["threshold"].max()
        print(
            f"\n  Recall peaks at {peak:.3f} and holds within 10% of that up to a"
            f"\n  threshold of {knee_90:.2f}, and within 25% up to {knee_75:.2f}."
        )
        print(
            "\n  Read this as localisation headroom, NOT a recommended threshold:\n"
            "  because a lower threshold can only add matches, F1 here is always\n"
            "  highest at the smallest threshold and must not be used to pick one.\n"
            "  A high knee means tight boxes and room to demand stricter overlap;\n"
            "  a knee close to your current setting means the boxes are loose and\n"
            "  raising COMPARE_IOU_THRESHOLD would discard genuine matches.\n"
            "  Choose the operating point by purpose: recall for an exhaustive\n"
            "  survey, precision for a map people will act on."
        )

    if save:
        outdir = _plots_dir(run_folder)

        if len(overlaps):
            plt.figure(figsize=(8, 5))
            plt.hist(overlaps, bins=20, range=(0, 1), edgecolor="black")
            if current is not None:
                plt.axvline(
                    float(current),
                    color="red",
                    linestyle="--",
                    label=f"COMPARE_IOU_THRESHOLD = {current}",
                )
                plt.legend()
            plt.xlabel(f"{metric} of matched pairs")
            plt.ylabel("number of waterholes")
            plt.title("Localisation quality: overlap of matched detections")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "overlap_distribution.png"), dpi=150)
            plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(sweep["threshold"], sweep["precision"], marker="o", label="precision")
        plt.plot(sweep["threshold"], sweep["recall"], marker="s", label="recall")
        plt.plot(sweep["threshold"], sweep["f1"], marker="^", label="F1")
        if current is not None:
            plt.axvline(float(current), color="red", linestyle="--", label="current")
        plt.xlabel(f"matching threshold ({metric})")
        plt.ylabel("score")
        plt.title("Sensitivity to the matching threshold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_sweep.png"), dpi=150)
        plt.close()

        csv_out = os.path.join(run_folder, "threshold_sweep.csv")
        sweep.to_csv(csv_out, index=False)
        print(f"\nSaved {csv_out}")
        print(f"Saved figures to {outdir}")

    return summary, sweep


# ---------------------------------------------------------------------------
# 3. Recall stratified by waterhole size
# ---------------------------------------------------------------------------
def recall_by_size_AF(run_folder, config, bins=None, by_class=False, save=True):
    """
    Recall broken down by how big the labelled waterhole is.

    Answers "are we systematically missing the small ones?", which no aggregate
    metric can show. Sizes come from the ground-truth box area in pixels, so they
    are directly comparable across images at the same resolution.

    Args:
        run_folder: folder holding the comparison output
        config: parsed config dict (or path)
        bins: explicit area bin edges in square pixels. Default splits the
            labelled waterholes into five equal-count (quintile) buckets, so
            every bucket carries a usable sample regardless of your size range.
        by_class: also break the table down per class
        save: write the CSV and figure

    Returns:
        DataFrame of recall per size bucket.
    """
    from .testing import parse_config

    if isinstance(config, str):
        config = parse_config(config)
    registry = load_class_registry(config)

    df = load_details(run_folder)
    # Only ground-truth rows have a size: matched pairs and missed labels.
    gt = df[df["match_type"].isin(["matched", "false_negative"])].copy()
    gt = gt[gt["manual_area"].notna() & (gt["manual_area"] > 0)]
    if gt.empty:
        raise ValueError(
            "No labelled waterholes with a usable box area were found. Check that "
            "the ground-truth label files were read correctly."
        )
    gt["found"] = (gt["match_type"] == "matched").astype(int)

    if bins is None:
        # quintiles of the observed size distribution
        edges = np.unique(np.quantile(gt["manual_area"], [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        if len(edges) < 3:
            edges = np.linspace(gt["manual_area"].min(), gt["manual_area"].max(), 4)
    else:
        edges = np.asarray(bins, dtype=float)

    gt["size_bucket"] = pd.cut(
        gt["manual_area"], bins=edges, include_lowest=True, duplicates="drop"
    )

    grouped = (
        gt.groupby("size_bucket", observed=True)
        .agg(
            labelled=("found", "size"),
            detected=("found", "sum"),
            median_area_px=("manual_area", "median"),
        )
        .reset_index()
    )
    grouped["recall"] = (grouped["detected"] / grouped["labelled"]).round(3)
    # side length of an equivalent square, easier to picture than an area
    grouped["typical_side_px"] = np.sqrt(grouped["median_area_px"]).round(1)

    print("=" * 78)
    print("RECALL BY WATERHOLE SIZE  (ground-truth box area, pixels squared)")
    print("=" * 78)
    print(grouped.to_string(index=False))

    overall = gt["found"].mean()
    worst = grouped.loc[grouped["recall"].idxmin()]
    best = grouped.loc[grouped["recall"].idxmax()]
    print(f"\n  overall recall  {overall:.3f}")
    print(
        f"  worst bucket    {worst['recall']:.3f}  "
        f"(around {worst['typical_side_px']:.0f}px across, n={int(worst['labelled'])})"
    )
    print(
        f"  best bucket     {best['recall']:.3f}  "
        f"(around {best['typical_side_px']:.0f}px across, n={int(best['labelled'])})"
    )
    spread = float(best["recall"] - worst["recall"])
    if spread > 0.15:
        print(
            f"\n  NOTE: a {spread:.2f} spread across sizes. The aggregate recall is\n"
            "  hiding a real size bias - worth reporting recall per size bucket\n"
            "  rather than a single number."
        )

    result = grouped
    if by_class:
        per_class = (
            gt.assign(
                class_name=gt["manual_class"].map(registry.id_to_name).fillna("Unknown")
            )
            .groupby(["class_name", "size_bucket"], observed=True)
            .agg(labelled=("found", "size"), detected=("found", "sum"))
            .reset_index()
        )
        per_class["recall"] = (per_class["detected"] / per_class["labelled"]).round(3)
        print("\n" + "=" * 78)
        print("RECALL BY SIZE AND CLASS")
        print("=" * 78)
        print(per_class.to_string(index=False))
        result = (grouped, per_class)

    if save:
        outdir = _plots_dir(run_folder)
        plt.figure(figsize=(9, 5))
        labels = [str(b) for b in grouped["size_bucket"]]
        plt.bar(range(len(grouped)), grouped["recall"], edgecolor="black")
        plt.xticks(range(len(grouped)), labels, rotation=30, ha="right")
        plt.axhline(overall, color="red", linestyle="--", label=f"overall {overall:.2f}")
        for i, (r, n) in enumerate(zip(grouped["recall"], grouped["labelled"])):
            plt.text(i, r + 0.02, f"n={int(n)}", ha="center", fontsize=9)
        plt.ylim(0, 1.08)
        plt.ylabel("recall")
        plt.xlabel("ground-truth box area (pixels squared)")
        plt.title("Are small waterholes being missed?")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "recall_by_size.png"), dpi=150)
        plt.close()

        csv_out = os.path.join(run_folder, "recall_by_size.csv")
        grouped.to_csv(csv_out, index=False)
        print(f"\nSaved {csv_out}")
        print(f"Saved figure to {outdir}")

    return result
