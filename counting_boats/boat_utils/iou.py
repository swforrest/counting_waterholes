"""
Box-overlap helpers for reconciling waterhole detections.

The pipeline used to reconcile detections purely by the euclidean distance
between box centres, which ignores box size and shape entirely - two very
differently sized waterholes with nearby centres looked identical to it.
These helpers replace that with an area-overlap measure, which accounts for both.

Box convention
--------------
Detections travel through the pipeline as rows of
(x_centre, y_centre, confidence, class, width, height) in absolute pixels of the
full padded image (see parse_classifications_AF), so columns [0, 1, 4, 5] are the
(cx, cy, w, h) box. Helpers here take that (cx, cy, w, h) form and convert to
corners internally.

Overlap metrics
---------------
Which measure counts as "the same waterhole" is configurable via the config key
OVERLAP_METRIC, because the right answer depends on the shapes involved. Both
options are strict about non-overlap: boxes that do not physically touch score 0
and are never merged or matched, however close their centres are.

"iou" (Intersection over Union, the default)
    intersection / union. Penalises a size mismatch, so a small waterhole whose
    box happens to sit inside a big neighbour box scores LOW and stays a separate
    waterhole. This is what you want when large irregular waterholes can engulf
    smaller nearby ones.

"iomin" (Intersection over Minimum area)
    intersection / area of the smaller box. A box fully contained in another
    scores 1.0 regardless of the size difference, so containment always merges.
    Use this only if the detector emits a tight box and a loose box for the SAME
    waterhole and you want them collapsed - be aware it will also swallow a
    genuinely distinct small waterhole that falls inside a bigger box.
"""
import numpy as np
from scipy.spatial.distance import squareform

# Column indices of (cx, cy, w, h) within a classification row
BOX_COLUMNS = [0, 1, 4, 5]

# Overlap measures selectable from the config's OVERLAP_METRIC key
OVERLAP_METRICS = ("iou", "iomin")


def validate_metric(metric: str) -> str:
    """
    Check an overlap metric name, raising a clear error naming the valid options.
    """
    if metric not in OVERLAP_METRICS:
        raise ValueError(
            f"unknown overlap metric {metric!r}; expected one of {list(OVERLAP_METRICS)}. "
            "Set this with the 'OVERLAP_METRIC' key in the config file."
        )
    return metric


def extract_boxes(classifications: np.ndarray) -> np.ndarray:
    """
    Pull the (cx, cy, w, h) boxes out of classification rows.

    Args:
        classifications: array of rows (x, y, confidence, class, width, height)

    Returns:
        (N, 4) float array of (cx, cy, w, h). Empty (0, 4) if there are no rows.
    """
    classifications = np.asarray(classifications)
    if classifications.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    return classifications[:, BOX_COLUMNS].astype(np.float64)


def boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2) corners.

    Args:
        boxes: (N, 4) array of (cx, cy, w, h)

    Returns:
        (N, 4) array of (x1, y1, x2, y2)
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    half_w, half_h = np.abs(w) / 2.0, np.abs(h) / 2.0
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def overlap_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray, metric: str = "iou"
) -> np.ndarray:
    """
    Pairwise overlap between two sets of (cx, cy, w, h) boxes.

    Args:
        boxes_a: (N, 4) array of (cx, cy, w, h)
        boxes_b: (M, 4) array of (cx, cy, w, h)
        metric: "iou" (intersection / union) or "iomin" (intersection / smaller
            box area). See the module docstring for when each is appropriate.

    Returns:
        (N, M) array where entry [i, j] scores boxes_a[i] against boxes_b[j],
        in [0, 1]. Zero-area boxes score 0 rather than raising a divide error.
    """
    validate_metric(metric)
    a = boxes_to_corners(boxes_a)
    b = boxes_to_corners(boxes_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    # Intersection rectangle of every (a, b) pair
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0.0, None) * np.clip(inter_y2 - inter_y1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)

    if metric == "iou":
        denominator = area_a[:, None] + area_b[None, :] - inter
    else:  # "iomin" - containment scores 1.0
        denominator = np.minimum(area_a[:, None], area_b[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denominator > 0, inter / denominator, 0.0)
    return np.clip(score, 0.0, 1.0)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise Intersection over Union. Shorthand for overlap_matrix(..., "iou")."""
    return overlap_matrix(boxes_a, boxes_b, metric="iou")


def overlap_condensed_distance(boxes: np.ndarray, metric: str = "iou") -> np.ndarray:
    """
    Condensed (1 - overlap) distance matrix for scipy hierarchical clustering.

    Args:
        boxes: (N, 4) array of (cx, cy, w, h)
        metric: overlap measure, see overlap_matrix()

    Returns:
        Condensed distance vector of length N*(N-1)/2, as scipy's linkage()
        expects. Identical boxes are distance 0; non-overlapping boxes are 1.
    """
    distance = 1.0 - overlap_matrix(boxes, boxes, metric=metric)
    # squareform requires an exactly-zero diagonal; floating point can leave
    # a self-distance a hair off zero.
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 1.0)
    # Enforce exact symmetry so squareform's tolerance check can't trip.
    distance = (distance + distance.T) / 2.0
    return squareform(distance, checks=False)


def greedy_match(
    boxes_a: np.ndarray, boxes_b: np.ndarray, threshold: float, metric: str = "iou"
) -> list:
    """
    Greedily pair boxes one-to-one by descending overlap (COCO-style matching).

    Each box in either set is used at most once, so the result gives a clean
    true-positive / false-positive / false-negative split.

    Args:
        boxes_a: (N, 4) array of (cx, cy, w, h) - typically the detections
        boxes_b: (M, 4) array of (cx, cy, w, h) - typically the ground truth
        threshold: minimum overlap score for a pair to be considered a match
        metric: overlap measure, see overlap_matrix()

    Returns:
        List of (index_a, index_b) pairs, highest overlap first. Any index not
        appearing is unmatched.
    """
    scores = overlap_matrix(boxes_a, boxes_b, metric=metric)
    if scores.size == 0:
        return []

    candidates = np.argwhere(scores >= threshold)
    if len(candidates) == 0:
        return []

    # Highest overlap first; stable so equal scores resolve deterministically
    order = np.argsort(-scores[candidates[:, 0], candidates[:, 1]], kind="stable")

    used_a, used_b, pairs = set(), set(), []
    for k in order:
        i, j = int(candidates[k, 0]), int(candidates[k, 1])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs
