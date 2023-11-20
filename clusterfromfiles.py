import imageCuttingSupport as ics
import os
from os import path 
import argparse
from NNclassifier import read_classifications, cluster, process_clusters
import numpy as np
import scipy.cluster
import scipy.spatial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
  - a csv
      - number of true positives
      - number of true negatives
      - number of false positives
      - number of false negatives
"""

STAT_DISTANCE_CUTOFF_PIX = 6
MOVING_DISTANCE_CUTOFF_PIX = 10

def main(classifications, labels, outdir):
    """
    Walk the directories given to match up classifications and labels.
    For each pair, cluster, process, and compare
    Write to csv
    """
    i =0
    for root, subdirs, files in os.walk(classifications):
        if len(files) > 0 and files[0].endswith(".txt"):
            # we are in a directory with classifications
            # get the corresponding label directory
            label_dir = path.join(labels, "/".join(root.split("/")[-2:]))
            this_img = os.path.basename(root)
            # check if it exists
            if not path.exists(label_dir):
                print(f"Label directory {label_dir} does not exist, skipping...")
                continue
            # ML classifications
            ML_classifications, _ = read_classifications("ml", class_folder=root)
            ML_classifications_stat = list(filter(lambda x: x[3] == 0, ML_classifications))
            ML_classifications_moving = list(filter(lambda x: x[3] == 1, ML_classifications))
            # cluster
            ML_clusters_stat = cluster(ML_classifications_stat, STAT_DISTANCE_CUTOFF_PIX)
            ML_clusters_moving = cluster(ML_classifications_moving, MOVING_DISTANCE_CUTOFF_PIX)
            # manual annotations
            manual_annotations, _ = read_classifications("manual", class_folder=label_dir)
            manual_annotations_stat = list(filter(lambda x: x[3] == 0, manual_annotations))
            manual_annotations_moving = list(filter(lambda x: x[3] == 1, manual_annotations))
            # cluster
            manual_clusters_stat = cluster(manual_annotations_stat, STAT_DISTANCE_CUTOFF_PIX)
            manual_clusters_moving = cluster(manual_annotations_moving, MOVING_DISTANCE_CUTOFF_PIX)
            # process
            ML_clusters_stat = process_clusters(ML_clusters_stat)
            ML_clusters_moving = process_clusters(ML_clusters_moving)
            manual_clusters_stat = process_clusters(manual_clusters_stat)
            manual_clusters_moving = process_clusters(manual_clusters_moving)
            # TODO: compare 
            stat = compare(ML_clusters_stat, manual_clusters_stat, STAT_DISTANCE_CUTOFF_PIX)
            moving = compare(ML_clusters_moving, manual_clusters_moving, MOVING_DISTANCE_CUTOFF_PIX)
            # write to csv
            allboats = stat + moving
            write_to_csv(allboats, this_img, outdir)
            i += 1
            print(f"Processed {i} images", end="\r")
    summarize(outdir)

def summarize(outdir):
    """
    Open all csv files in the directory and summarize them
    """
    # get all lines (except headers) from all files using np list comp
    all_boats = np.asarray([line.strip().split(",") for file in os.listdir(outdir) if file.endswith(".csv") for line in open(os.path.join(outdir, file)) if line[0] != "x"])
    pred = all_boats[:, 2].astype(float).astype(int)
    true = all_boats[:, 3].astype(float).astype(int)
    # get the confusion matrix
    ConfusionMatrixDisplay.from_predictions(pred, true, labels=[-1, 0, 1], display_labels=["Undetected", "Static Boat", "Moving Boat"])
    # save the confusion matrix image
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.show()


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
    argparser.add_argument("-c", "--classifications", help="Directory of classifications from yolo (can contain multiple direcories)")
    argparser.add_argument("-l", "--labels", help="Directory of manual annotations (can contain multiple direcories)")
    argparser.add_argument("-o", "--outdir", help="Directory to dump output csvs")

    args = argparser.parse_args()
    classifications = args.classifications
    labels = args.labels
    outdir = args.outdir
    main(classifications, labels, outdir)



################################
# Used to separate out the files into days (unused now)
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
            os.mkdir(path.join(into, date))
        os.rename(path.join(directory, file), path.join(into, date, file))
    # return the directories
    return [path.join(directory, day) for day in days]

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
            os.mkdir(path.join(into, img))
        os.rename(path.join(directory, file), path.join(into, img, file))
    # return the directories
    return [path.join(directory, img) for img in imgs]

