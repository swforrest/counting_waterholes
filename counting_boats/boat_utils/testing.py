"""
Utility functions for training/validation pipeline.  Includes: Preparing tif files for labelling Using labelme 
"""

import os
import shutil

import numpy as np
import pandas as pd
import scipy
import random

from .classifier import cluster, process_clusters, read_classifications, pixel2latlong
from .config import cfg
from . import image_cutting_support as ics
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import json


def prepare(run_folder, config):
    """
    Given a folder, find all the tif files are create a png for each one.
    Also rename tif files if required.

    Args:
        run_folder (str): The folder to prepare
        config (dict): The configuration dictionary

    Returns:
        None
    """
    img_folder = config["raw_images"]  # folder with the tif files
    save_folder = os.path.join(config["path"], config["pngs"])
    os.makedirs(save_folder, exist_ok=True)
    for root, _, files in os.walk(img_folder):
        for file in files:
            if file == "composite.tif":
                # find the json file:
                date_file = [f for f in files if f.endswith("xml")][0]
                date = date_file.split("_")[0]
                aoi = os.path.basename(root).split("_")[-2].split("/")[-1]
                print(root, aoi)
                name = f"{date}_{aoi}.tif"
                print(name)
                os.rename(os.path.join(root, file), os.path.join(root, name))
                # want to create a png for this
                new_name = os.path.join(save_folder, f"{name.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    ics.create_padded_png(
                        root,
                        save_folder,
                        name,
                        tile_size=config["img_size"],
                        stride=config["img_stride"],
                    )
            # if the file is a tif and first part is a date, don't need to rename
            elif file.endswith("tif") and file.split("_")[0].isdigit():
                # check if we have already created a png for this
                new_name = os.path.join(save_folder, f"{file.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    print(f"Creating png for {file}")
                    ics.create_padded_png(
                        root,
                        save_folder,
                        file,
                        tile_size=config["img_size"],
                        stride=config["img_stride"],
                    )


def segment(run_folder, config):
    """
    Segment (labelled) png's in the given base.
    Places segmented images in the 'SegmentedImages' folder, and Labels in the 'Labels' folder.

    Args:
        run_folder (str): The folder to segment
        config (dict): The configuration dictionary

    Returns:
        None
    """
    tile_size = config.get("img_size", 416)
    stride = config.get("img_stride", 104)
    pngs = os.path.join(config["path"], config["pngs"])
    im_save_folder = os.path.join(config["path"], config["segmented_images"])
    label_save_folder = os.path.join(config["path"], config["labels"])
    if not os.path.exists(im_save_folder):
        os.makedirs(im_save_folder, exist_ok=True)
    if not os.path.exists(label_save_folder):
        os.makedirs(label_save_folder, exist_ok=True)
    for filename in os.listdir(pngs):
        if filename.endswith(".json"):  # Grab the LabelMe Label file
            # get all dir names in the segmentation folder (recursively)
            dirs = [x[0].split(os.path.sep)[-1] for x in os.walk(im_save_folder)]
            if filename[:-5] in dirs or filename[:8] in [
                n[:8] for n in os.listdir(im_save_folder)
            ]:
                # skip this file if it has already been segmented (segmenting takes a while)
                continue
            # find the corresponding image file
            img_file = os.path.join(pngs, filename[:-5] + ".png")
            if os.path.isfile(img_file):  # Check exists
                ics.segment_image(
                    img_file,
                    os.path.join(pngs, filename),
                    tile_size,
                    stride,
                    remove_empty=0,
                    im_outdir=im_save_folder,
                    labels_outdir=label_save_folder,
                )
            else:
                print(f"Could not find image file for {filename}")
    # Separate the folders into individual images for fun
    segregate(im_save_folder)
    segregate(label_save_folder)


def run_detection(run_folder, run_config):
    """
    Run the YoloV5 detection on the segmented images, and move
    the detections to a sibling directory for analysis.

    Args:
        run_folder (str): The folder to run detection on.
        run_config (dict): The configuration dictionary.

    Raises:
        Exception: If there is an error running detection on a directory.

    Returns:
        None
    """
    weights = run_config["weights"]
    yolo = cfg["yolo_dir"]
    python = cfg["python"]
    classification_dir = os.path.join(run_folder, run_config["classifications"])
    img_dir = os.path.join(run_folder, run_config["segmented_images"])
    for root, _, files in os.walk(img_dir):
        if len(files) > 0 and files[0].endswith(".png"):
            this_classification_dir = os.path.join(
                classification_dir, os.path.sep.join(root.split(os.path.sep)[-2:])
            )
            if os.path.exists(this_classification_dir):  # don't double classify
                print(f"Already classified {this_classification_dir}")
                continue
            os.makedirs(this_classification_dir, exist_ok=True)
            device = run_config.get("device", "cuda:0")
            tile_size = run_config.get("img_size", 416)
            res = os.system(
                f"{python} {yolo}/detect.py --imgsz {tile_size} --save-txt --save-conf --weights {weights} --source {root} --device {device} --nosave --conf-thres 0.15"
            )
            if res != 0:
                raise Exception(f"Error running detection on {root}")
            latest_exp = (
                max(
                    [
                        int(f.split("exp")[1]) if f != "exp" else 0
                        for f in os.listdir(os.path.join(yolo, "runs", "detect"))
                        if "exp" in f
                    ]
                )
                or ""
            )
            for file in os.listdir(
                os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels")
            ):
                shutil.move(
                    os.path.join(
                        yolo, "runs", "detect", f"exp{latest_exp}", "labels", file
                    ),
                    this_classification_dir,
                )
            print(f"Classified {root}, saved to {this_classification_dir}")


def backwards_annotation(run_folder, config):
    """
    Generate labelme style annotations (json) from the classifications.
    1. Read classifications
    2. Generate json file {image}_labelme_auto.json with:
    """
    detection_dir = os.path.join(config["path"], config["classifications"])
    for root, _, files in os.walk(detection_dir):
        # skip if json file exists
        if os.path.exists(
            os.path.join(
                config["path"],
                config["pngs"],
                f"{os.path.basename(root)}_labelme_auto.json",
            )
        ):
            continue
        if len(files) > 0 and files[0].endswith(".txt"):
            this_image = os.path.basename(root)
            ML_classifications, _ = read_classifications(
                class_folder=root, confidence_threshold=0.5
            )  # read all
            ML_classifications_stat = ML_classifications[
                ML_classifications[:, 3] == 0.0
            ]
            ML_classifications_moving = ML_classifications[
                ML_classifications[:, 3] == 1.0
            ]
            # cluster
            ML_clusters_stat = cluster(
                ML_classifications_stat, STAT_DISTANCE_CUTOFF_PIX
            )
            ML_clusters_moving = cluster(
                ML_classifications_moving, MOVING_DISTANCE_CUTOFF_PIX
            )
            # condense
            ML_clusters_stat = process_clusters(ML_clusters_stat)
            ML_clusters_moving = process_clusters(ML_clusters_moving)
            # get image metadata (width and height)
            img = Image.open(
                os.path.join(config["path"], config["pngs"], this_image + ".png")
            )
            width, height = img.size

            json_data = {}
            json_data["version"] = "5.2.1"
            json_data["flags"] = {}
            json_data["imagePath"] = this_image
            json_data["imageHeight"] = height
            json_data["imageWidth"] = width
            # put in the shapes
            json_data["shapes"] = []
            for c in ML_clusters_stat:
                x, y, _, _, w, h = c
                w = int(w / 2)
                h = int(h / 2)
                json_data["shapes"].append(
                    {
                        "label": "boat",
                        "points": [[x - w, y - h], [x + w, y + h]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )
            for c in ML_clusters_moving:
                (
                    x,
                    y,
                    _,
                    _,
                    w,
                    h,
                ) = c
                w = int(w / 2)
                h = int(h / 2)
                json_data["shapes"].append(
                    {
                        "label": "movingBoat",
                        "points": [[x - w, y - h], [x + w, y + h]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )
            # also need to get the "image_data" key. There will also be a {this_image}.json we can grab this from
            with open(
                os.path.join(config["path"], config["pngs"], f"{this_image}.json"), "r"
            ) as f:
                image_data = json.load(f)["imageData"]
            json_data["imageData"] = image_data
            # save the json
            json_path = os.path.join(
                config["path"], config["pngs"], f"{this_image}_labelme_auto.json"
            )
            with open(json_path, "w+") as f:
                json.dump(json_data, f)


def compare_detections_to_ground_truth(run_folder, config):
    """
    Match up labels and detections, compare them, and save the results

    Args:
        run_folder (str): The folder to run detection on.
        config (dict): The configuration dictionary.

    Returns:
        None

    """
    label_dir = os.path.join(config["path"], config["labels"])
    detection_dir = os.path.join(config["path"], config["classifications"])
    for root, _, files in os.walk(detection_dir):
        if len(files) > 0 and files[0].endswith(".txt"):
            this_img = os.path.basename(root)
            data = process_image(root, label_dir)
            comparisons_to_csv(data, os.path.join(run_folder, this_img + ".csv"))
    # create an overall file, with all boats in lat long
    if config.get("raw_images", False):
        classifications_to_lat_long(run_folder, config)


def confusion_matrix(run_folder, config):
    """
    Summarize the results of the comparison. Reads all csvs and creates a confusion matrix

    Args:
        run_folder (str): The folder to run detection on.
        config (dict): The configuration dictionary.

    Returns:
        None
    """
    if os.path.exists(os.path.join(run_folder, "all_boats.csv")):
        all_data = pd.read_csv(os.path.join(run_folder, "all_boats.csv"))
    else:
        # read all the csvs in the run folder that start with a date (8 numbers)
        all_data = pd.concat(
            [
                pd.read_csv(os.path.join(run_folder, file))
                for file in os.listdir(run_folder)
                if file.endswith(".csv") and file[:8].isdigit()
            ]
        )
    # create confusion matrix
    true = all_data["manual_class"]
    pred = all_data["ml_class"]
    # save image of confusion matrix
    acc = np.sum(true == pred) / len(true)
    ConfusionMatrixDisplay.from_predictions(
        y_pred=pred,
        y_true=true,
        labels=[-1, 0, 1],
        display_labels=["Not a Boat", "Static Boat", "Moving Boat"],
    )
    fig = plt.gcf()
    fig.suptitle(
        f"{len(true[true != -1])} Labelled Boats (Detection Accuracy: {round(acc, 3)})"
    )
    fig.tight_layout()
    # save the confusion matrix image
    plt.savefig(os.path.join(run_folder, "plots", "confusion_matrix.png"))


### Clustering Helpers
STAT_DISTANCE_CUTOFF_PIX = 6
MOVING_DISTANCE_CUTOFF_PIX = 10
COMPARE_DISTANCE_CUTOFF_PIX = 8
CONF_THRESHOLD = 0.5


def process_image(
    detections,
    labels_root,
):
    """
    Compare the detections and labels for a single image

    Args:
        detections (str): The directory of detections for the iamge
        labels_root (str): The root directory of labels

    Returns:
        list of clusters in form [x, y, confidence, class, width, height, filename, in_ml, in_manual]
    """
    # labels will be in a parallel directory to detections
    # e.g detections = "Detections/b/../d", labels = "Labels/b/../d"
    label_dir = os.path.join(
        labels_root, os.path.sep.join(detections.split(os.path.sep)[-2:])
    )
    # check if it exists
    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} does not exist, skipping image...")
        return []
    # ML classifications
    ML_classifications, _ = read_classifications(class_folder=detections)
    ML_classifications_stat = ML_classifications[ML_classifications[:, 3] == 0.0]
    ML_classifications_moving = ML_classifications[ML_classifications[:, 3] == 1.0]
    # cluster
    ML_clusters_stat = cluster(ML_classifications_stat, STAT_DISTANCE_CUTOFF_PIX)
    ML_clusters_moving = cluster(ML_classifications_moving, MOVING_DISTANCE_CUTOFF_PIX)
    # save clusters as csv for later analysis
    if not os.path.exists(os.path.join(detections, "clusters")):
        os.makedirs(os.path.join(detections, "clusters"))
    statoutfile = os.path.join(detections, "clusters", "stat_clusters.csv")
    movingoutfile = os.path.join(detections, "clusters", "moving_clusters.csv")
    with open(statoutfile, "w") as f:
        for c in ML_clusters_stat:
            f.write(",".join([str(i) for i in c]) + "\n")
    with open(movingoutfile, "w") as f:
        for c in ML_clusters_moving:
            f.write(",".join([str(i) for i in c]) + "\n")

    # manual annotations
    manual_annotations, _ = read_classifications(class_folder=label_dir)
    if len(manual_annotations) == 0:
        manual_annotations_stat = np.empty((0, 7))
        manual_annotations_moving = np.empty((0, 7))
    else:
        manual_annotations_stat = manual_annotations[manual_annotations[:, 3] == 0.0]
        manual_annotations_moving = manual_annotations[manual_annotations[:, 3] == 1.0]
    # cluster
    manual_clusters_stat = cluster(manual_annotations_stat, STAT_DISTANCE_CUTOFF_PIX)
    manual_clusters_moving = cluster(
        manual_annotations_moving, MOVING_DISTANCE_CUTOFF_PIX
    )

    # process
    ML_clusters_stat = process_clusters(ML_clusters_stat)
    ML_clusters_moving = process_clusters(ML_clusters_moving)
    manual_clusters_stat = process_clusters(manual_clusters_stat)
    manual_clusters_moving = process_clusters(manual_clusters_moving)

    ML_clusters = np.concatenate((ML_clusters_stat, ML_clusters_moving))
    manual_clusters = np.concatenate((manual_clusters_stat, manual_clusters_moving))
    comparison = compare(ML_clusters, manual_clusters, COMPARE_DISTANCE_CUTOFF_PIX)
    return comparison


def compare(ml: np.ndarray, manual: np.ndarray, cutoff):
    """
    given two lists of clusters, compare them (cluster them and note the results)
    e.g if ml has the point (52, 101), and manual has (51.8, 101.2), they should be clustered together
    , and this boat should be noted as being in both sets

    Args:
        ml: list of clusters in form [x, y, confidence, class, width, height, filename]
        manual: list of clusters in form [x, y, confidence, class, width, height, filename]

    Returns:
        list of clusters in form [x, y, ml_class, manual_class]
    """
    all_clusters, all_points = combine_detections_and_labels(ml, manual)
    if len(all_points) < 2:
        # if its 1, still need to pretend cluster
        if len(all_points) == 1:
            list(all_points[0]).append(0)
            clusters = [0]
            points_with_cluster = np.c_[
                all_points, np.asarray(all_clusters)[:, 2:], clusters
            ]
        else:
            return []
    else:
        # cluster
        distances = scipy.spatial.distance.pdist(all_points, metric="euclidean")
        clustering = scipy.cluster.hierarchy.linkage(distances, "average")
        clusters = scipy.cluster.hierarchy.fcluster(
            clustering, cutoff, criterion="distance"
        )
        points_with_cluster = np.c_[
            all_points, np.asarray(all_clusters)[:, 2:], clusters
        ]
    # for each cluster, note if it is in ml, manual, or both
    results = []
    for cluster in np.unique(clusters):
        res = [0.0, 0.0, -1, -1]  # x, y, ml class, manual class
        points = points_with_cluster[points_with_cluster[:, -1] == str(cluster)]
        if len(points) == 0:
            print("No points in cluster")
            continue
        # 6th is the source, 3 is the class
        ml_cls = []
        manual_cls = []
        x = 0
        y = 0
        for point in points:
            x += float(point[0])
            y += float(point[1])
            if point[6] == "ml":
                ml_cls.append(int(float(point[3])))
            elif point[6] == "manual":
                manual_cls.append(int(float(point[3])))
        res[0] = round(x / len(points), 3)
        res[1] = round(y / len(points), 3)
        # class should be most common class
        if len(ml_cls) > 0:
            res[2] = max(set(ml_cls), key=ml_cls.count)
        if len(manual_cls) > 0:
            res[3] = max(set(manual_cls), key=manual_cls.count)
        results.append(res)
    return results


def combine_detections_and_labels(ml, manual):
    """
    Combine the detections and labels into one list of annotated clusters for comparison

    Args:
        ml: list of clusters in form [x, y, confidence, class, width, height, filename]
        manual: list of clusters in form [x, y, confidence, class, width, height, filename]

    Returns:
        list of clusters in form [x, y, confidence, class, width, height, filename, source]
    """
    # add "ml" to the end of each ml cluster
    if len(ml) > 0:
        ml = np.c_[ml, np.full(len(ml), "ml")]
    # add "manual" to the end of each manual cluster
    if len(manual) > 0:
        manual = np.c_[manual, np.full(len(manual), "manual")]
    # one of ml or manual could be empty so we need to check
    if len(ml) == 0:
        all = manual
    elif len(manual) == 0:
        all = ml
    else:
        all = np.concatenate((ml, manual))
    points_ml = ml[:, :2] if len(ml) > 0 else np.empty((0, 2))
    points_man = manual[:, :2] if len(manual) > 0 else np.empty((0, 2))
    # join together
    all_points = np.concatenate((points_ml, points_man)).astype(float)
    return all, all_points


def comparisons_to_csv(comparisons, filename):
    """
    Write the comparisons to a csv file

    Args:
        comparisons: list of clusters in form [x, y, ml_class, manual_class]
        filename: the name of the file to write to

    Returns:
        None

    """
    df = pd.DataFrame(comparisons, columns=["x", "y", "ml_class", "manual_class"])
    df.to_csv(filename, index=False)


def classifications_to_lat_long(run_folder, run_config):
    """
    Convert x and y of image classifications to lat/long and saves csv

    Args:
        run_folder (str): The folder to run detection on.
        run_config (dict): The configuration dictionary.

    Returns:
        None
    """
    # Initialise dataframe
    all_boats = pd.DataFrame(
        columns=[
            "date",
            "latitude",
            "longitude",
            "ml_class",
            "manual_class",
            "agree",
            "filename",
        ]
    )
    # Get all the images that are relevant
    # Involves reading all of the output files from the detections function
    raw_images = run_config["raw_images"]
    for file in os.listdir(run_folder):
        if not file.endswith(".csv"):
            continue
        # try to parse the date, if not, skip
        date = file.split("_")[0]
        if len(date) != 8:
            print(f"Could not parse date from {file}")
            continue
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        image_name = file.split(".")[0]
        boats = pd.read_csv(os.path.join(run_folder, file)).values.tolist()
        # remove index
        if len(boats) == 0:
            continue
        im_name = image_name.split(".")[0] + ".tif"
        image = [
            os.path.join(root, i)
            for root, dirs, files in os.walk(raw_images)
            for i in files
            if i == im_name
        ]
        if len(image) == 0:
            print(f"Could not find image {im_name} for {file}")
            continue
        image = image[0]
        boats = pixel2latlong(boats, image)
        # append the boats to the 'all_boats' dataframe
        boats = pd.DataFrame(
            boats, columns=["longitude", "latitude", "ml_class", "manual_class"]
        )
        boats["date"] = date
        boats["filename"] = image_name
        # Agree: -1 if disagree, 0 if agree stationary, 1 if agree moving
        boats["agree"] = boats["ml_class"] == boats["manual_class"]
        # boats["agree"] = boats.apply(lambda x: x["ml_class"] if x["agree"] else -1, axis=1)
        all_boats = pd.concat([all_boats, boats]) if all_boats.size != 0 else boats
    if all_boats.size == 0:
        return
    all_boats.to_csv(os.path.join(run_folder, "all_boats.csv"), index=False)


def boat_count_compare(run_folder, config):
    """
    Column graph with each group being one image, showing number of labelled and number of detected boats next to each other

    Args:
        run_folder (str): The folder to run detection on.
        config (dict): The configuration dictionary.

    Returns:
        None
    """
    if os.path.exists(os.path.join(run_folder, "all_boats.csv")):
        all_data = pd.read_csv(os.path.join(run_folder, "all_boats.csv"))
    else:
        csvs = [
            os.path.join(run_folder, file)
            for file in os.listdir(run_folder)
            if file.endswith(".csv") and file[0:8].isdigit()
        ]
        # add a 'filename' column to each csv and concatenate them
        all_data = pd.concat(
            [
                pd.read_csv(csv).assign(
                    filename=csv.split(os.path.sep)[-1].split(".")[0]
                )
                for csv in csvs
            ]
        )
    # x axis should be filename
    # y axis is count of boats
    # one column for manual, one for ml for each image
    all_data["manual"] = all_data["manual_class"].apply(lambda x: 1 if x != -1 else 0)
    all_data["ml"] = all_data["ml_class"].apply(lambda x: 1 if x != -1 else 0)
    all_data = (
        all_data.groupby(["filename"]).agg({"manual": "sum", "ml": "sum"}).reset_index()
    )
    all_data.plot(
        x="filename",
        y=["manual", "ml"],
        kind="bar",
        title="Boat Counts by Image",
        figsize=(20, 10),
        fontsize=20,
    )
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "plots", "count_by_image_column.png"))


### Metrics Helpers


def plot_boats(csvs: str, imgs: str, **kwargs):
    """
    given a directory of csvs, plot the boats on the images and save the images

    Args:
        csvs: directory containing csvs. Must be of form: x, y, ml_class, manual_class
        imgs: base folder with the images (png), or a folder with subfolders with images (stitched.png)

    Returns:
        None
    """
    if "outdir" in kwargs:
        outdir = kwargs["outdir"]
    else:
        outdir = csvs
    all_csvs = [
        os.path.join(csvs, file)
        for file in os.listdir(csvs)
        if file.endswith(".csv") and "summary" not in file
    ]
    all_images = [
        os.path.join(imgs, file) for file in os.listdir(imgs) if file.endswith(".png")
    ]
    all_images = [im for im in all_images if "heron" not in im]
    # filter to images which have a csv
    all_images = [
        image
        for image in all_images
        if any(
            [
                image.split(os.path.sep)[-1].split(".")[0] in csv
                for csv in [s.split(os.path.sep)[-1].split(".")[0] for s in all_csvs]
            ]
        )
    ]
    print(all_images)
    if len(all_images) == 0:
        # try to see if the stitched images exist
        all_images = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(imgs)
            for file in files
            if file == "stitched.png"
        ]
    i = 0
    for csv in all_csvs:
        # get the corresponding image
        img = [image for image in all_images if csv.split()[1].split(".")[0] in image]
        if len(img) == 0:
            continue
        img = img[0]
        # get the boats
        boats = np.asarray(
            [line.strip().split(",") for line in open(csv) if line[0] != "x"]
        )
        # plot the image
        fig, ax = plt.subplots()
        ax.imshow(plt.imread(img))
        # draw a box around the boat. 10x10 pixels.
        #   Green if : detected and labelled static
        #   Blue If  : detected and labelled moving
        #   Orange if: detected and labelled but disagree
        #   Red if   : detected but not labelled
        #   Yellow if: labelled but not detected
        correct = 0
        incorrect = 0
        for boat in boats:
            x = float(boat[0])
            y = float(boat[1])
            ml = int(float(boat[2]))
            manual = int(float(boat[3]))
            if ml == manual:
                correct += 1
            else:
                incorrect += 1
            if ml == 0 and ml == manual:  # Agree Static
                # green
                color = "g"
            elif ml == 1 and ml == manual:  # Agree Moving
                # blue
                color = "b"
            elif ml != -1 and manual != -1 and ml != manual:  # Disagreement
                # orange
                color = "orange"
            elif ml != -1 and manual == -1:  # Detected but not Labelled
                # red
                color = "r"
            else:  # Labelled but not Detected
                # yellow
                color = "y"
            if "skip" in kwargs and kwargs["skip"] == True and color == "g":
                continue
            rect = plt.Rectangle(
                (x - 5, y - 5), 10, 10, linewidth=0.1, edgecolor=color, facecolor="none"
            )
            if color == "r":
                # also draw a big circle around the boat (50x50)
                circ = plt.Circle(
                    (x, y), 50, linewidth=0.3, edgecolor=color, facecolor="none"
                )
                ax.add_patch(circ)
                # and annotate the detection as "ML: 0"
                ax.annotate(f"ML: {ml}", (x, y), color=color, fontsize=6)
            if color == "y":
                # also draw a big star around the boat (50x50)
                star = plt.Polygon(
                    np.array([[x - 50, y - 50], [x + 50, y - 50], [x, y + 50]]),
                    linewidth=0.3,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(star)
                # and annotate the label as "Label: 1"
                ax.annotate(f"Label: {manual}", (x, y), color=color, fontsize=6)
            if color == "orange":
                # also draw a big square
                square = plt.Rectangle(
                    (x - 50, y - 50),
                    100,
                    100,
                    linewidth=0.3,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(square)
                # and annotate the detection as "ML: 0, Label: 1"
                ax.annotate(
                    f"ML: {ml}, Label: {manual}", (x, y), color=color, fontsize=6
                )
            ax.add_patch(rect)
            if color == "orange":
                # also annotate the boat with the classes as "ML: 0, Label: 1"
                ax.annotate(
                    f"ML: {ml}, Label: {manual}", (x, y), color=color, fontsize=6
                )
        # save the image in really high quality with no axis labels
        plt.axis("off")
        # add a legend below the image (outside). Make it very small and 2 rows
        plt.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color="g"),
                plt.Rectangle((0, 0), 1, 1, color="b"),
                plt.Rectangle((0, 0), 1, 1, color="orange"),
                plt.Rectangle((0, 0), 1, 1, color="r"),
                plt.Rectangle((0, 0), 1, 1, color="y"),
            ],
            labels=[
                "Detected and Labelled Static",
                "Detected and Labelled Moving",
                "Disagreement",
                "Detected but not Labelled",
                "Labelled but not Detected",
            ],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.05),
            fontsize=6,
        )
        # make the title the correct, incorrect, and accuracy. Put the title at the bottom
        plt.title(
            f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {round(correct/(correct+incorrect), 3)}"
        )
        plt.savefig(
            os.path.join(outdir, csv.split()[1].split(".")[0] + ".png"),
            dpi=1000,
            bbox_inches="tight",
        )
        plt.close()
        i += 1
        print(f"Plotted {i}/{len(all_images)} images", end="\r")


def all_mistakes(run_folder, config):
    """
    Given a summary csv, find the images where there is a mistake made.
    Since the x and y in the summary refer to the entire image, we need to
    calculate the subimage(s) that the boat is in. save the best subimage (most central)
    to a new directory with the type of mistake (e.g "false_positive")

    Args:
        run_folder (str): The folder to run detection on.
        config (dict): The configuration dictionary.

    Returns:
        None
    """
    output_dir = os.path.join(run_folder, "mistakes")
    os.makedirs(output_dir, exist_ok=True)
    # for image:
    #   for boat:
    #       if mistake:
    #           find the subimage
    #           draw a box around the boat
    #           save the image
    summary_dir = run_folder
    img_dir = os.path.join(config["path"], config["segmented_images"])
    csvs = [
        os.path.join(summary_dir, file)
        for file in os.listdir(summary_dir)
        if file.endswith(".csv") and "summary" not in file
    ]
    for csv in csvs:
        csv_name = os.path.basename(csv)
        day = csv_name.split("_")[0][-2:]
        month = csv_name.split("_")[0][-4:-2]
        year = csv_name.split("_")[0][-8:-4]
        this_img_dir = os.path.join(
            img_dir, f"{day}_{month}_{year}", csv_name.split(".")[0]
        )
        if not os.path.exists(this_img_dir):
            print(f"Could not find image directory for {csv_name}")
            print(f"Expected {this_img_dir}")
            continue
        boats = np.asarray(
            [line.strip().split(",") for line in open(csv) if line[0] != "x"]
        )
        id = 0
        stride = cfg["STRIDE"]
        for boat in boats:
            if boat[2] != boat[3]:
                x = float(boat[0])
                y = float(boat[1])
                # get the best subimage
                row = max(y // stride - 1, 1)
                col = max(x // stride - 1, 1)
                # get the image
                img_path = os.path.join(
                    this_img_dir,
                    csv_name.split(".")[0]
                    + "_"
                    + str(int(row))
                    + "_"
                    + str(int(col))
                    + ".png",
                )
                if not os.path.exists(img_path):
                    all_imgs = all_possible_imgs(x, y)
                    # find one img that does exists
                    for row, col in all_imgs:
                        img_path = os.path.join(
                            this_img_dir,
                            csv_name.split(".")[0]
                            + "_"
                            + str(int(row))
                            + "_"
                            + str(int(col))
                            + ".png",
                        )
                        if os.path.exists(img_path):
                            break
                        img_path = ""
                    if img_path == "":
                        print(
                            f"Could not find section for {csv_name} with x={x}, y={y}"
                        )
                        print("*" * 80)
                        continue
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(plt.imread(img_path))
                # draw a box around the boat. 10x10 pixels.
                #   Red if   : detected but not labelled
                #   Yellow if: labelled but not detected
                ml = int(float(boat[2]))
                manual = int(float(boat[3]))
                rel_x = x - (col * 104)
                rel_y = y - (row * 104)
                if ml != -1 and manual != -1 and ml != manual:
                    # also draw a big square
                    rect = plt.Rectangle(
                        (rel_x - 10, rel_y - 10),
                        20,
                        20,
                        linewidth=0.3,
                        edgecolor="gray",
                        facecolor="none",
                    )
                    square = plt.Rectangle(
                        (rel_x - 50, rel_y - 50),
                        100,
                        100,
                        linewidth=0.3,
                        edgecolor="orange",
                        facecolor="none",
                    )
                    ax.add_patch(square)
                    # and annotate the detection as "ML: 0, Label: 1"
                    ax.annotate(
                        f"ML: {ml}, Label: {manual}",
                        (rel_x, rel_y),
                        color="orange",
                        fontsize=6,
                    )
                elif ml != -1 and manual == -1:
                    # also draw a big circle around the boat (50x50)
                    rect = plt.Rectangle(
                        (rel_x - 10, rel_y - 10),
                        20,
                        20,
                        linewidth=0.3,
                        edgecolor="gray",
                        facecolor="none",
                    )
                    circ = plt.Circle(
                        (rel_x, rel_y),
                        50,
                        linewidth=0.3,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(circ)
                    # and annotate the detection as "ML: 0"
                    ax.annotate(f"ML: {ml}", (rel_x, rel_y), color="r", fontsize=6)
                else:
                    # also draw a big star around the boat (50x50)
                    rect = plt.Rectangle(
                        (rel_x - 10, rel_y - 10),
                        20,
                        20,
                        linewidth=0.3,
                        edgecolor="gray",
                        facecolor="none",
                    )
                    star = plt.Polygon(
                        np.array(
                            [
                                [rel_x - 50, rel_y - 50],
                                [rel_x + 50, rel_y - 50],
                                [rel_x, rel_y + 50],
                            ]
                        ),
                        linewidth=0.3,
                        edgecolor="y",
                        facecolor="none",
                    )
                    ax.add_patch(star)
                    # and annotate the label as "Label: 1"
                    ax.annotate(
                        f"Label: {manual}", (rel_x, rel_y), color="y", fontsize=6
                    )
                ax.add_patch(rect)
                # save the image in really high quality with no axis labels
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        output_dir,
                        csv_name.split(".")[0]
                        + "_"
                        + str(id)
                        + "_"
                        + str(int(row))
                        + "_"
                        + str(int(col))
                        + ".png",
                    ),
                    dpi=1000,
                    bbox_inches="tight",
                )
                # also save a text file with the x, y, ml, manual
                with open(
                    os.path.join(
                        output_dir, csv_name.split(".")[0] + "_" + str(id) + ".txt"
                    ),
                    "w+",
                ) as file:
                    file.write(f"{x}, {y}, {manual}")
                plt.close()
                id += 1


def subimage_confidence(run_folder, config):
    """
    For config["subimage_confidence"] (int) boats, plot the 16 squares around the boat
    and their confidence scores.
    """
    class_dir = config["classifications"]
    days = [
        f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))
    ]
    n_outputs = config["tasks"]["analyse"]["images"]["subimage_confidence"]
    images = []
    for d in days:
        day_dir = os.path.join(class_dir, d)
        images.extend(
            [f for f in os.listdir(day_dir) if os.path.isdir(os.path.join(day_dir, f))]
        )
    for _ in range(n_outputs):
        img = random.choice(images)
        im_dir = os.path.join(day_dir, img)
        cluster_dir = os.path.join(im_dir, "clusters")
        if not os.path.exists(cluster_dir):
            continue
        # read the csv which has x, y, confidence, class, width, height, cluster
        try:
            stat_clusters = pd.read_csv(
                os.path.join(cluster_dir, "moving_clusters.csv"), header=None
            )
            mov_clusters = pd.read_csv(
                os.path.join(cluster_dir, "stat_clusters.csv"), header=None
            )
            together = pd.concat([stat_clusters, mov_clusters])
            clusters = cluster(together.to_numpy()[:, :6], MOVING_DISTANCE_CUTOFF_PIX)
            clusters = pd.DataFrame(clusters)
        except:
            continue
        boat = clusters.sample()
        # work out which subimage the cluster is in
        # get the image
        img_path = os.path.join(config["pngs"], f"{img}.png")
        if not os.path.exists(img_path):
            print(f"Could not find image {img_path}")
            continue
        # cropped image bounds
        W = 200
        H = 200
        x1 = boat[0].values[0] - W / 2
        y1 = boat[1].values[0] - H / 2
        x2 = x1 + W
        y2 = y1 + H
        all_boats = clusters.where(
            (clusters[0] > x1)
            & (clusters[0] < x2)
            & (clusters[1] > y1)
            & (clusters[1] < y2)
        )
        print(all_boats.shape)
        # make a cropped image around the boat
        img_data = Image.open(img_path)
        img_data = img_data.crop((x1, y1, x2, y2))
        # upscale the image
        scale = 4
        img_data = img_data.resize((W * scale, H * scale))
        # save the image
        subimg_path = os.path.join(run_folder, "imgs", f"{img}_{int(x1)}_{int(y1)}.png")
        img_data.save(subimg_path)
        # upscale the clusters (each x and y is how far away from the top left corner of the subimage * scale)
        all_boats[0] = all_boats[0] * scale
        all_boats[1] = all_boats[1] * scale
        all_boats[0] = all_boats[0] - (x1 * scale)
        all_boats[1] = all_boats[1] - (y1 * scale)
        # grab the image
        with Image.open(subimg_path) as im:
            # for each in the cluster, draw a rectangle around the boat
            draw = ImageDraw.Draw(im)
            grouped = all_boats.groupby(6)
            for _, c in grouped:
                print("G")
                for _, row in c.iterrows():
                    w = row[4] * 1.05 * scale
                    h = row[5] * 1.05 * scale
                    # Bounds of the boat
                    x0 = row[0] - w / 2
                    y0 = row[1] - h / 2
                    x1 = x0 + w
                    y1 = y0 + h
                    # draw a rectangle around the boat (0.5 opacity)
                    conf = row[2]
                    if conf > 0.9:  # Green
                        color = (0, 255, 0, 100)
                    elif conf > 0.7:  # Orange
                        color = (255, 165, 0, 100)
                    elif conf > 0.5:  # Red
                        color = (255, 0, 0, 100)
                    draw.rectangle((x0, y0, x1, y1), width=1, outline=color)
                num_boats = len(c)
                avg_conf = c[2].mean()
                max_conf = c[2].max()
                min_conf = c[2].min()
                # draw the text just above the boat
                # "Detections: {}, Avg Confidence: {}, Range: {} - {}".format(num_boats, avg_conf, min_conf, max_conf)
                stats = f"Detections: {num_boats}, Avg Confidence: {avg_conf:.2f}, Range: {min_conf:.2f} - {max_conf:.2f}"
                x = max(c[0].min() - (c[4].max() * scale) / 2, 0)
                y = max(c[1].min() - (c[5].max() * scale) / 2 - 15, 0)
                draw.text((x, y), stats, fill=(255, 255, 255, 255))
            # save the image again
            im.save(subimg_path)


import boat_utils.heatmap as hm


def coverage_heatmap(run_folder, config):
    """
    Generate the coverage heatmap for all TIF files used in the run
    """
    tif_dir = config["raw_images"]
    # walk
    tifs = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(tif_dir)
        for file in files
        if file.endswith(".tif")
    ]
    tifs = [tif for tif in tifs if "composite" not in tif]
    tifs = tifs[0:1]

    polygons = [hm.polygons_from_tif(tif) for tif in tifs]
    # flatten
    polygons = [poly for sublist in polygons for poly in sublist]
    # convert to
    # get the coverage
    hm.create_heatmap_from_polygons(
        polygons, os.path.join(run_folder, "images", "heatmap.tif")
    )


def all_possible_imgs(x, y, stride=104):
    """
    return a list of tuples (row, col) that would contain the given x and y coords

    Args:
        x: x coord
        y: y coord
        stride: stride of the images

    Returns:
        list of tuples (row, col)
    """
    row = y // stride - 1
    col = x // stride - 1
    options = []
    # NOTE: we do it like this to try to keep the 'best' subimages as highest priority
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            options.append((row + i, col + j))
    for i in [-2, 2]:
        for j in [0, 1, -1, -2, 2]:
            options.append((row + i, col + j))

    return options


### Preparation Helpers


def segregate(directory):
    """
    Segregate the directory by day and image

    Args:
        directory (str): The directory to segregate

    """
    # separate by day
    days = segregate_by_day(directory)
    # separate by image
    for day in days:
        segregate_by_image(day, day)


def segregate_by_day(directory, into=None):
    """
    Separate files in a directory into subdirectories by day.
    Files must have the date as the last part of the filename.

    Args:
        directory (str): The directory to segregate
        into (str): The directory to move the files into

    Returns:
        list of directories
    """
    if into is None:
        into = directory
    days = []
    print("Segregating by day...")
    for file in os.listdir(directory):
        if not (file.endswith(".png") or file.endswith(".txt")):
            continue
        date = ics.get_date_from_filename(file)
        if date is None:
            print(f"Could not get date from {file}")
            continue
        if (date := date.replace("/", "_")) not in days:
            print(date)
            days.append(date)
            os.mkdir(os.path.join(into, date))
        os.rename(os.path.join(directory, file), os.path.join(into, date, file))
    # return the directories
    return [os.path.join(directory, day) for day in days]


def segregate_by_image(directory, into=None):
    """
    Separate files in a directory into subdirectories by image.
    Files must have the image name as the middle part of the filename.

    Args:
        directory (str): The directory to segregate
        into (str): The directory to move the files into

    Returns:
        list of directories
    """
    if into is None:
        into = directory
    imgs = []
    print("Segregating by image...")
    for file in os.listdir(directory):
        if not (file.endswith(".png") or file.endswith(".txt")):
            continue
        # everything before the 2nd last underscore is the image name
        img = file[: file.rfind("_", 0, file.rfind("_"))]
        if img not in imgs:
            imgs.append(img)
            os.mkdir(os.path.join(into, img))
        os.rename(os.path.join(directory, file), os.path.join(into, img, file))
    # return the directories
    return [os.path.join(directory, img) for img in imgs]
