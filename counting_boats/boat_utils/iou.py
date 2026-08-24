"""
Intersection over Union (IoU) helpers for reconciling waterhole bounding boxes.

The pipeline used to reconcile detections purely by the euclidean distance
between box centres, which ignores box size and shape entirely - two very
differently sized waterholes with nearby centres looked identical to it.
These helpers replace that with IoU, which accounts for both.

Box convention
--------------
Detections travel through the pipeline as rows of
``(x_centre, y_centre, confidence, class, width, height)`` in absolute pixels
of the full padded image (see ``parse_classifications_AF``), so columns
``[0, 1, 4, 5]`` are the ``(cx, cy, w, h)`` box. Helpers here take that
``(cx, cy, w, h)`` form and convert to corners internally.

Strict IoU is used: boxes that do not physically overlap score 0 and are never
merged or matched, no matter how close their centres are.
"""
import numpy as np
from scipy.spatial.distance import squareform

# Column indices of (cx, cy, w, h) within a classification row
BOX_COLUMNS = [0, 1, 4, 5]


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


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of (cx, cy, w, h) boxes.

    Args:
        boxes_a: (N, 4) array of (cx, cy, w, h)
        boxes_b: (M, 4) array of (cx, cy, w, h)

    Returns:
        (N, M) array where entry [i, j] is the IoU of boxes_a[i] and boxes_b[j],
        in [0, 1]. Zero-area boxes give an IoU of 0 rather than a divide error.
    """
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
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return np.clip(iou, 0.0, 1.0)


def iou_condensed_distance(boxes: np.ndarray) -> np.ndarray:
    """
    Condensed (1 - IoU) distance matrix for scipy hierarchical clustering.

    Args:
        boxes: (N, 4) array of (cx, cy, w, h)

    Returns:
        Condensed distance vector of length N*(N-1)/2, as scipy's linkage()
        expects. Identical boxes are distance 0; non-overlapping boxes are 1.
    """
    distance = 1.0 - iou_matrix(boxes, boxes)
    # squareform requires an exactly-zero diagonal; floating point can leave
    # a self-distance a hair off zero.
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 1.0)
    # Enforce exact symmetry so squareform's tolerance check can't trip.
    distance = (distance + distance.T) / 2.0
    return squareform(distance, checks=False)


def greedy_match(boxes_a: np.ndarray, boxes_b: np.ndarray, iou_threshold: float) -> list:
    """
    Greedily pair boxes one-to-one by descending IoU (COCO-style matching).

    Each box in either set is used at most once, so the result gives a clean
    true-positive / false-positive / false-negative split.

    Args:
        boxes_a: (N, 4) array of (cx, cy, w, h) - typically the detections
        boxes_b: (M, 4) array of (cx, cy, w, h) - typically the ground truth
        iou_threshold: minimum IoU for a pair to be considered a match

    Returns:
        List of (index_a, index_b) pairs, highest IoU first. Any index not
        appearing is unmatched.
    """
    scores = iou_matrix(boxes_a, boxes_b)
    if scores.size == 0:
        return []

    candidates = np.argwhere(scores >= iou_threshold)
    if len(candidates) == 0:
        return []

    # Highest IoU first; stable so equal scores resolve deterministically
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
