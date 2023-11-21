import os
import argparse
import numpy as np
import scipy.cluster
import scipy.spatial
import matplotlib.pyplot as plt
import imageCuttingSupport as ics
from sklearn.metrics import ConfusionMatrixDisplay, f1_score 
from NNclassifier import read_classifications, cluster, process_clusters

"""
given:
  - a directory of classifications pertaining to a single image (e.g 416x416 ims with 104 overlap)
  - a directory of labels pertaining to that same image (e.g 416x416 ims with 104 overlap)
we want to:
  - cluster the classifications for the image (so that overlap does not count as multiple boats)
  - cluster the labels for the image (so that overlap does not count as multiple boats)
  - compare the clusters to each other (more clustering)
and output:
  - a csv for each image with the following columns:
      - x
      - y
      - classification class
      - label class
  - a summary csv
      - total boats labelled (over all images)
      - total boats detected (over all images)
      - real boats (discarding overlap)
      - detected boats (discarding overlap)
      - true positives
      - false positives
      - false negatives
"""

STAT_DISTANCE_CUTOFF_PIX = 6
MOVING_DISTANCE_CUTOFF_PIX = 10
CONF_THRESHOLD = 0.5

def process_image(path, labels, summary):
    label_dir = os.path.join(labels, "/".join(path.split("/")[-2:]))
    # check if it exists
    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} does not exist, skipping...")
        return []
    # ML classifications
    ML_classifications, _ = read_classifications("ml", class_folder=path, confidence_threshold=CONF_THRESHOLD)
    ML_classifications_stat = list(filter(lambda x: x[3] == 0, ML_classifications))
    summary["detected_stat"] += len(ML_classifications_stat)
    ML_classifications_moving = list(filter(lambda x: x[3] == 1, ML_classifications))
    summary["detected_mov"] += len(ML_classifications_moving)
    # cluster
    ML_clusters_stat = cluster(ML_classifications_stat, STAT_DISTANCE_CUTOFF_PIX)
    ML_clusters_moving = cluster(ML_classifications_moving, MOVING_DISTANCE_CUTOFF_PIX)
    # manual annotations
    manual_annotations, _ = read_classifications("manual", class_folder=label_dir)
    manual_annotations_stat = list(filter(lambda x: x[3] == 0, manual_annotations))
    summary["labelled_stat"] += len(manual_annotations_stat)
    manual_annotations_moving = list(filter(lambda x: x[3] == 1, manual_annotations))
    summary["labelled_mov"] += len(manual_annotations_moving)
    # cluster
    manual_clusters_stat = cluster(manual_annotations_stat, STAT_DISTANCE_CUTOFF_PIX)
    manual_clusters_moving = cluster(manual_annotations_moving, MOVING_DISTANCE_CUTOFF_PIX)
    # process
    ML_clusters_stat = process_clusters(ML_clusters_stat)
    summary["detected_stat_clusters"] += len(ML_clusters_stat)
    ML_clusters_moving = process_clusters(ML_clusters_moving)
    summary["detected_mov_clusters"] += len(ML_clusters_moving)
    manual_clusters_stat = process_clusters(manual_clusters_stat)
    summary["labelled_stat_clusters"] += len(manual_clusters_stat)
    manual_clusters_moving = process_clusters(manual_clusters_moving)
    summary["labelled_mov_clusters"] += len(manual_clusters_moving)
    stat = compare(ML_clusters_stat, manual_clusters_stat, STAT_DISTANCE_CUTOFF_PIX)
    moving = compare(ML_clusters_moving, manual_clusters_moving, MOVING_DISTANCE_CUTOFF_PIX)
    return stat + moving

def main(classifications, labels, outdir):
    """
    Walk the directories given to match up classifications and labels.
    For each pair, cluster, process, and compare
    Write to csv
    """
    i = 0
    summary = {
            "detected_stat": 0,
            "detected_mov": 0,
            "labelled_stat": 0,
            "labelled_mov": 0,
            "detected_stat_clusters": 0,
            "detected_mov_clusters": 0,
            "labelled_stat_clusters": 0,
            "labelled_mov_clusters": 0,
            }
    for root, subdirs, files in os.walk(classifications):
        if len(files) > 0 and files[0].endswith(".txt"):
            # we are in a directory with classifications
            # get the corresponding label directory
            this_img = os.path.basename(root)
            write_to_csv(process_image(root, labels, summary), this_img, outdir)
            i += 1
            print(f"Processed {i} images", end="\r")
    summarize(outdir, summary)

def summarize(outdir, summary):
    """
    Open all csv files in the directory and summarize them
    """
    # get all lines (except headers) from all files using np list comp
    all_boats = np.asarray([line.strip().split(",") for file in os.listdir(outdir) if (file.endswith(".csv") and "summary" not in file) for line in open(os.path.join(outdir, file)) if line[0] != "x"])
    pred = all_boats[:, 2].astype(float).astype(int)
    true = all_boats[:, 3].astype(float).astype(int)
    # get the confusion matrix
    f1 = f1_score(y_pred=pred, y_true=true, average="weighted")
    acc = np.sum(true == pred) / len(true)
    ConfusionMatrixDisplay.from_predictions(y_pred=pred, y_true=true, 
            labels=[-1, 0, 1], display_labels=["Undetected", "Static Boat", "Moving Boat"])
    fig = plt.gcf()
    fig.suptitle(f"Confusion Matrix (F1: {round(f1, 3)}, Accuracy: {round(acc, 3)})")
    fig.tight_layout()
    # save the confusion matrix image
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    # save a summary csv with:
    # - class                   (all, static, moving)
    # - total boats labelled    (over all images)
    # - total boats detected    (over all images)
    # - real boats              (discarding overlap)
    # - detected boats          (discarding overlap)
    # - detected boat images    (e.g max would be 16)
    # - true positives
    # - false positives
    # - false negatives
    labelled_all = summary["labelled_stat"] + summary["labelled_mov"]
    detected_all = summary["detected_stat"] + summary["detected_mov"]
    with open(os.path.join(outdir, "summary.csv"), "w+") as file:
        file.write("class, total_boats_labelled, total_boats_detected, real_boats,  \
                detected_boats, im_per_boat_detected, im_per_boat_labelled,         \
                true_positives, false_positives, false_negatives\n")
        # all
        file.write(f"all, {labelled_all}, {detected_all}, {len(true[true != -1])},  \
                {len(pred[pred != -1])}, {detected_all/len(pred[pred != -1])},      \
                {labelled_all/len(true[true != -1])}, {np.sum(true == pred)},       \
                {len(pred[(pred != -1) & (true == -1)])},                           \
                {len(pred[(pred == -1) & (true != -1)])}\n")
        # static
        file.write(f"static, {summary['labelled_stat']}, {summary['detected_stat']},\
                {len(true[true == 0])}, {len(pred[pred == 0])},                     \
                {summary['detected_stat']/len(pred[pred == 0])},                    \
                {summary['labelled_stat']/len(true[true == 0])},                    \
                {np.sum((true == pred) & (true == 0))},                             \
                {len(pred[(pred == 0) & (true != 0)])},                             \
                {len(true[(pred != 0) & (true == 0)])}\n")
        # moving
        file.write(f"moving, {summary['labelled_mov']}, {summary['detected_mov']},  \
                {len(true[(true == 1)])}, {len(pred[pred==1])},                     \
                {summary['detected_mov']/len(pred[pred == 1])},                     \
                {summary['labelled_mov']/len(true[true == 1])},                     \
                {np.sum((true == pred) & (true == 1))},                             \
                {len(pred[(pred == 1) & (true != 1)])},                             \
                {len(true[(pred != 1) & (true == 1)])}\n")


def compare(ml:list, manual:list, cutoff):
    """
    given two lists of clusters, compare them (cluster them and note the results)
    e.g if ml has the point (52, 101), and manual has (51.8, 101.2), they should be clustered together
    , and this boat should be noted as being in both sets
    :param ml: list of clusters in form [x, y, confidence, class, width, height, filename]
    :param manual: list of clusters in form [x, y, confidence, class, width, height, filename]
    :return list of clusters in form [x, y, confidence, class, width, height, filename, in_ml, in_manual]
    """
    all = ml + manual
    points_ml = np.asarray(ml)[:, :2] if len(ml) > 0 else np.empty((0, 2))
    points_man = np.asarray(manual)[:, :2] if len(manual) > 0 else np.empty((0, 2))
    # join together
    all_points = points_man.tolist() + points_ml.tolist()
    all_points = np.asarray(all_points, dtype=np.float64)
    if len(all_points) < 2:
        # if its 1, still need to pretend cluster
        if len(all_points) == 1:
            list(all_points[0]).append(0)
            clusters = [0]
            points_with_cluster = np.c_[all_points, np.asarray(all)[:, 2:], clusters]
        else:
            return []
    else:
        # cluster
        distances           = scipy.spatial.distance.pdist(all_points, metric='euclidean')
        clustering          = scipy.cluster.hierarchy.linkage(distances, 'average')
        clusters            = scipy.cluster.hierarchy.fcluster(clustering, cutoff, criterion='distance')
        points_with_cluster = np.c_[all_points, np.asarray(all)[:, 2:], clusters]
    # for each cluster, note if it is in ml, manual, or both
    results = []
    for cluster in np.unique(clusters):
        res = [0., 0., -1, -1] # x, y, ml, manual
        points = points_with_cluster[points_with_cluster[:, -1] == str(cluster)]
        # 6th is the source, 3 is the class
        x = 0
        y = 0
        for point in points:
            x += float(point[0])
            y += float(point[1])
            if point[6] == "ml":
                res[2] = point[3]
            elif point[6] == "manual":
                res[3] = point[3]
        res[0] = round(x / len(points), 3)
        res[1] = round(y / len(points), 3)
        results.append(res)
    return results

def write_to_csv(boats, filename, outdir):
    path = os.path.join(outdir, filename + ".csv")
    with open(path, "w+") as file:
        file.write("x, y, ml_class, manual_class\n")
        for boat in boats:
            file.write(f"{boat[0]}, {boat[1]}, {boat[2]}, {boat[3]}\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
    argparser.add_argument("-c", "--classifications", help="Directory of classifications from yolo (can contain multiple direcories)", required=True)
    argparser.add_argument("-l", "--labels", help="Directory of manual annotations (can contain multiple direcories)", required=True) 
    argparser.add_argument("-o", "--outdir", help="Directory to dump output csvs", required=True)

    args = argparser.parse_args()
    classifications = args.classifications
    labels = args.labels
    outdir = args.outdir
    main(classifications, labels, outdir)



################################
# Used to separate out the files into days
# Used once because everything was mixed, unused now
def segregate_by_day(directory, into=None):
    """
    Bunch of files in a directory, need to separate into days.
    """
    if into is None:
        into = directory
    days = []
    print("Segregating by day...")
    for file in os.listdir(directory):
        if (date := ics.get_date_from_filename(file).replace("/", "_")) not in days:
            print(date)
            days.append(date)
            os.mkdir(os.path.join(into, date))
        os.rename(os.path.join(directory, file), os.path.join(into, date, file))
    # return the directories
    return [os.path.join(directory, day) for day in days]

def segregate_by_image(directory, into=None):
    """
    Bunch of files in a directory, need to separate into same image.
    """
    if into is None:
        into = directory
    imgs = []
    print("Segregating by image...")
    for file in os.listdir(directory):
        # everything before the 2nd last underscore is the image name
        img = file[:file.rfind("_", 0, file.rfind("_"))]
        print(img)
        if img not in imgs:
            print(img)
            imgs.append(img)
            os.mkdir(os.path.join(into, img))
        os.rename(os.path.join(directory, file), os.path.join(into, img, file))
    # return the directories
    return [os.path.join(directory, img) for img in imgs]

