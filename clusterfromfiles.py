import imageCuttingSupport as ics
import os
from os import path 
import argparse
from NNclassifier import read_classifications, cluster, process_clusters
import numpy as np
import scipy.cluster
import scipy.spatial
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

argparser = argparse.ArgumentParser()
argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
argparser.add_argument("-c", "--classifications", help="Directory of classifications from yolo (can contain multiple direcories)")
argparser.add_argument("-l", "--labels", help="Directory of manual annotations (can contain multiple direcories)")

args = argparser.parse_args()
classifications = args.classifications
labels = args.labels

def main():
    """
    Walk the directories given to match up classifications and labels.
    For each pair, cluster, process, and compare
    Write to csv
    """
    for root, subdirs, files in os.walk(classifications):
        if len(files) > 0 and files[0].endswith(".txt"):
            # we are in a directory with classifications
            # get the corresponding label directory
            label_dir = path.join(labels, "/".join(root.split("/")[-2:]))
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
    points_ml = np.asarray(ml)[:, :2] # all x y coords
    points_man = np.asarray(manual)[:, :2] # all x y coords
    # join together
    all_points = points_man.tolist() + points_ml.tolist()
    all_points = np.asarray(all_points, dtype=np.float64)
    print(all_points)
    # cluster
    distances           = scipy.spatial.distance.pdist(all_points, metric='euclidean')
    clustering          = scipy.cluster.hierarchy.linkage(distances, 'average')
    clusters            = scipy.cluster.hierarchy.fcluster(clustering, cutoff, criterion='distance')
    points_with_cluster = np.c_[all_points, np.asarray(all)[:, 2:], clusters]

if __name__ == "__main__":
    main()



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

