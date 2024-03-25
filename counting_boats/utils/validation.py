"""
Utility functions for training/validation pipeline.  Includes: Preparing tif files for labelling Using labelme 
"""
import os
import shutil

import numpy as np
import pandas as pd
import scipy

from .classifier import cluster, process_clusters, read_classifications
from config import cfg
from . import image_cutting_support as ics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def prepare(run_folder, config):
    """
    Given a folder, find all the tif files are create a png for each one.
    Also rename tif files if required.
    """
    img_folder = config["raw_images"] # folder with the tif files
    save_folder = os.path.join(config["path"], config["pngs"])
    for root, _, files in os.walk(img_folder):
        for file in files:
            if file == 'composite.tif':
                # find the json file:
                date_file = [f for f in files if f.endswith('xml')][0]
                date = date_file.split('_')[0]
                aoi = root.split('_')[-2].split('/')[-1]
                name = f"{date}_{aoi}.tif"
                print(name)
                os.rename(os.path.join(root, file), 
                          os.path.join(root, name))
                # want to create a png for this
                new_name = os.path.join(save_folder, f"{name.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    ics.create_padded_png(root, save_folder, name)
            # if the file is a tif and first part is a date, don't need to rename
            elif file.endswith('tif') and file.split('_')[0].isdigit():
                # check if we have already created a png for this
                new_name = os.path.join(save_folder, f"{file.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    ics.create_padded_png(root, save_folder, file)

def segment(run_folder, config, segment_size=416, stride=104):
    """
    Segment (labelled) png's in the given base. 
    Places segmented images in the 'SegmentedImages' folder, and Labels in the 'Labels' folder.
    """
    pngs = os.path.join(config["path"], config["pngs"])
    im_save_folder = os.path.join(config["path"], config["segmented_images"])
    label_save_folder = os.path.join(config["path"], config["labels"])
    for filename in os.listdir(pngs):
        if filename.endswith(".json"): # Grab the LabelMe Label file
            # get all dir names in the segmentation folder (recursively)
            dirs = [x[0].split(os.path.sep)[-1] for x in os.walk(im_save_folder)]
            if filename[:-5] in dirs:
                # skip this file if it has already been segmented (segmenting takes a while)
                continue
            # find the corresponding image file
            img_file = os.path.join(im_save_folder, filename[:-5] + ".png")
            if os.path.isfile(img_file): # Check exists
                ics.segment_image(img_file, os.path.join(pngs, filename), segment_size, stride, remove_empty=0, 
                                  im_outdir=im_save_folder, labels_outdir=label_save_folder)
            else:
                print(f"Could not find image file for {filename}")
    # Separate the folders into individual images for fun
    segregate(im_save_folder)
    segregate(label_save_folder)
    


def run_detection(run_folder, run_config, img_dir = "SegmentedImages"):
    """
    Run the YoloV5 detection on the segmented images, and move 
    the detections to a sibling directory for analysis.
    """
    weights = cfg["weights"]
    yolo = cfg["yolo_dir"]
    python = cfg["python"]
    classification_dir = os.path.join(run_config["path"], run_config["classifications"])
    for root, _, files in os.walk(img_dir):
        if len(files) > 0 and files[0].endswith(".png"):
            this_classification_dir = os.path.join(classification_dir, os.path.sep.join(root.split(os.path.sep)[-2:]))
            if os.path.exists(this_classification_dir): # don't double classify
                continue
            os.makedirs(this_classification_dir, exist_ok=True)
            os.system(f"{python} {yolo}/detect.py --imgsz 416 --save-txt --save-conf --weights {weights} --source {root}")
            latest_exp = max([int(f.split("exp")[1]) if f != "exp" else 0 for f in os.listdir(os.path.join(yolo, "runs", "detect" )) if "exp" in f]) or ""
            for file in os.listdir(os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels")):
                shutil.move(os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels", file), this_classification_dir)
            print(f"Classified {root}, saved to {this_classification_dir}")

def compare_detections_to_ground_truth(folder, labels="Labels", detections="Classifications"):
    """
    Match up labels and detections, compare them, and save the results
    """
    label_dir = os.path.join(folder, labels)
    detection_dir = os.path.join(folder, detections)
    for root, _, files in os.walk(detection_dir):
        if len(files) > 0 and files[0].endswith(".txt"):
            this_img = os.path.basename(root)
            data = process_image(root, label_dir)
            comparisons_to_csv(data, os.path.join(folder, "output", this_img + ".csv"))

def summarize(folder):
    """
    Summarize the results of the comparison. Reads all csvs and creates a confusion matrix
    """
    csvs = [os.path.join(folder, "output", f) for f in os.listdir(os.path.join(folder, "output")) if f.endswith(".csv")]
    all_data = []
    for csv in csvs:
        all_data.append(pd.read_csv(csv))
    all_data = pd.concat(all_data) # combine all the data into one dataframke
    all_data = all_data.dropna()
    # create confusion matrix
    true = all_data["manual_class"]
    pred = all_data["ml_class"]
    # save image of confusion matrix
    acc = np.sum(true == pred) / len(true)
    ConfusionMatrixDisplay.from_predictions(y_pred=pred, y_true=true, 
            labels=[-1, 0, 1], display_labels=["Undetected", "Static Boat", "Moving Boat"])
    fig = plt.gcf()
    fig.suptitle(f"{len(true[true != -1])} Labelled Boats (Accuracy: {round(acc, 3)})")
    fig.tight_layout()
    # save the confusion matrix image
    plt.savefig(os.path.join("output", "confusion_matrix.png"))

### Clustering Helpers
STAT_DISTANCE_CUTOFF_PIX = 6
MOVING_DISTANCE_CUTOFF_PIX = 10
COMPARE_DISTANCE_CUTOFF_PIX = 8
CONF_THRESHOLD = 0.5

def process_image(detections, labels_root) -> list[list]:
    """
    :return list of clusters in form [x, y, confidence, class, width, height, filename, in_ml, in_manual]
    """
    # labels will be in a parallel directory to detections
    # e.g detections = "Detections/b/../d", labels = "Labels/b/../d"
    label_dir = os.path.join(labels_root, os.path.sep.join(detections.split(os.path.sep)[-2:]))
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
    manual_clusters_moving = cluster(manual_annotations_moving, MOVING_DISTANCE_CUTOFF_PIX)
    # process
    ML_clusters_stat = process_clusters(ML_clusters_stat)
    ML_clusters_moving = process_clusters(ML_clusters_moving)
    manual_clusters_stat = process_clusters(manual_clusters_stat)
    manual_clusters_moving = process_clusters(manual_clusters_moving)

    ML_clusters = np.concatenate((ML_clusters_stat, ML_clusters_moving))
    manual_clusters = np.concatenate((manual_clusters_stat, manual_clusters_moving))
    comparison = compare(ML_clusters, manual_clusters, COMPARE_DISTANCE_CUTOFF_PIX)
    return comparison

def compare(ml:np.ndarray, manual:np.ndarray, cutoff):
    """
    given two lists of clusters, compare them (cluster them and note the results)
    e.g if ml has the point (52, 101), and manual has (51.8, 101.2), they should be clustered together
    , and this boat should be noted as being in both sets
    :param ml: list of clusters in form [x, y, confidence, class, width, height, filename]
    :param manual: list of clusters in form [x, y, confidence, class, width, height, filename]
    :return list of clusters in form [x, y, ml_class, manual_class]
    """
    all_clusters, all_points = combine_detections_and_labels(ml, manual)
    if len(all_points) < 2:
        # if its 1, still need to pretend cluster
        if len(all_points) == 1:
            list(all_points[0]).append(0)
            clusters = [0]
            points_with_cluster = np.c_[all_points, np.asarray(all_clusters)[:, 2:], clusters]
        else:
            return []
    else:
        # cluster
        distances           = scipy.spatial.distance.pdist(all_points, metric='euclidean')
        clustering          = scipy.cluster.hierarchy.linkage(distances, 'average')
        clusters            = scipy.cluster.hierarchy.fcluster(clustering, cutoff, criterion='distance')
        points_with_cluster = np.c_[all_points, np.asarray(all_clusters)[:, 2:], clusters]
    # for each cluster, note if it is in ml, manual, or both
    results = []
    for cluster in np.unique(clusters):
        res = [0., 0., -1, -1] # x, y, ml class, manual class
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
    """
    df = pd.DataFrame(comparisons, columns=["x", "y", "ml_class", "manual_class"])
    df.to_csv(filename)


def classifications_to_lat_long(folder, boats, filename, date):
    """
    Convert x and y of an image to latlong AND saves it to a csv
    """
    if os.path.exists(os.path.join(folder, "all_boats.csv")):
        all_boats = pd.read_csv(os.path.join(folder, "all_boats.csv"))
    else:
        all_boats = pd.DataFrame(columns=["date", "latitude", "longitude", "ml_class", "manual_class", "filename"])
    # find the image of the boats
    image = None
    for root, _, files in os.walk(os.path.join(folder, "RawImages")):
        if f"{filename}.tif" in files:
            image = os.path.join(root, f"{filename}.tif")
            break
    if image is None:
        print(f"Could not find image {filename} at {os.path.join(folder, 'RawImages')}")
        return
    crs = ics.get_crs(image)
    for boat in boats:
        x = boat[0]
        y = boat[1]
        x, y = ics.pixel2coord(x, y, image)
        lat, long = ics.coord2latlong(x, y, crs)
        boat[0] = lat
        boat[1] = long
    # append the baots to the 'all_boats' dataframe
    boats = pd.DataFrame(boats, columns=["latitude", "longitude", "ml_class", "manual_class"])
    boats["date"] = date
    boats["filename"] = filename
    all_boats = pd.concat([all_boats, boats])
    all_boats.to_csv(os.path.join(folder, "all_boats.csv"), index=False)

### Metrics Helpers

def plot_boats(csvs:str, imgs:str, **kwargs):
    """
    given a directory of csvs, plot the boats on the images and save the images
    :param csvs: directory containing csvs. Must be of form: x, y, ml_class, manual_class
    :param imgs: base folder with the images (png), or a folder with subfolders with images (stitched.png)
    """
    if "outdir" in kwargs:
        outdir = kwargs["outdir"]
    else:
        outdir = csvs
    all_csvs = [os.path.join(csvs, file) for file in os.listdir(csvs) if file.endswith(".csv") and "summary" not in file]
    all_images = [os.path.join(imgs, file) for file in os.listdir(imgs) if file.endswith(".png")]
    all_images = [im for im in all_images if "heron" not in im]
    # filter to images which have a csv
    all_images = [image for image in all_images if any([image.split(os.path.sep)[-1].split(".")[0] in csv for csv in [s.split(os.path.sep)[-1].split(".")[0] for s in all_csvs]])]
    print(all_images)
    if len(all_images) == 0:
        # try to see if the stitched images exist
        all_images = [os.path.join(root, file) for root, dirs, files in os.walk(imgs) for file in files if file == "stitched.png"]
    i = 0
    for csv in all_csvs:
        # get the corresponding image
        img = [image for image in all_images if csv.split()[1].split(".")[0] in image]
        if len(img) == 0:
            print(f"Could not find image for {csv}")
            continue
        img = img[0]
        # get the boats
        boats = np.asarray([line.strip().split(",") for line in open(csv) if line[0] != "x"])
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
            if ml == manual: correct += 1
            else: incorrect += 1
            if ml == 0 and ml == manual:                        # Agree Static
                # green
                color = "g"
            elif ml == 1 and ml == manual:                      # Agree Moving
                # blue
                color = "b"
            elif ml != -1 and manual != -1 and ml != manual:    # Disagreement
                # orange
                color = "orange"
            elif ml != -1 and manual == -1:                     # Detected but not Labelled
                # red
                color = "r"
            else:                                               # Labelled but not Detected
                # yellow
                color = "y"
            if "skip" in kwargs and kwargs["skip"] == True and color == "g":
                continue
            rect = plt.Rectangle((x-5, y-5), 10, 10, linewidth=0.1, edgecolor=color, facecolor="none")
            if color == "r":
                # also draw a big circle around the boat (50x50)
                circ = plt.Circle((x, y), 50, linewidth=0.3, edgecolor=color, facecolor="none")
                ax.add_patch(circ)
                # and annotate the detection as "ML: 0"
                ax.annotate(f"ML: {ml}", (x, y), color=color, fontsize=6)
            if color == "y":
                # also draw a big star around the boat (50x50)
                star = plt.Polygon(np.array([[x-50, y-50], [x+50, y-50], [x, y+50]]), linewidth=0.3, edgecolor=color, facecolor="none")
                ax.add_patch(star)
                # and annotate the label as "Label: 1"
                ax.annotate(f"Label: {manual}", (x, y), color=color, fontsize=6)
            if color == "orange":
                # also draw a big square
                square = plt.Rectangle((x-50, y-50), 100, 100, linewidth=0.3, edgecolor=color, facecolor="none")
                ax.add_patch(square)
                # and annotate the detection as "ML: 0, Label: 1"
                ax.annotate(f"ML: {ml}, Label: {manual}", (x, y), color=color, fontsize=6)
            ax.add_patch(rect)
            if color == "orange":
                # also annotate the boat with the classes as "ML: 0, Label: 1"
                ax.annotate(f"ML: {ml}, Label: {manual}", (x, y), color=color, fontsize=6)
        # save the image in really high quality with no axis labels
        plt.axis("off")
        # add a legend below the image (outside). Make it very small and 2 rows
        plt.legend(
                handles=[plt.Rectangle((0,0), 1, 1, color="g"), 
                         plt.Rectangle((0,0), 1, 1, color="b"), 
                         plt.Rectangle((0,0), 1, 1, color="orange"), 
                         plt.Rectangle((0,0), 1, 1, color="r"), 
                         plt.Rectangle((0,0), 1, 1, color="y")], 
                labels=["Detected and Labelled Static", "Detected and Labelled Moving", "Disagreement", "Detected but not Labelled", "Labelled but not Detected"], 
                loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=6)
        # make the title the correct, incorrect, and accuracy. Put the title at the bottom
        plt.title(f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {round(correct/(correct+incorrect), 3)}")
        plt.savefig(os.path.join(outdir, csv.split()[1].split(".")[0] + ".png"), dpi=1000, bbox_inches="tight")
        plt.close()
        i += 1
        print(f"Plotted {i}/{len(all_images)} images", end="\r")

def highlight_mistakes(folder):
    """
    Given a summary csv, find the images where there is a mistake made.
    Since the x and y in the summary refer to the entire image, we need to 
    calculate the subimage(s) that the boat is in. save the best subimage (most central)
    to a new directory with the type of mistake (e.g "false_positive")
    """
    output_dir = os.path.join(folder, "mistakes")
    os.makedirs(output_dir, exist_ok=True)
    # for image:
    #   for boat:
    #       if mistake:
    #           find the subimage
    #           draw a box around the boat
    #           save the image
    summary_dir = os.path.join(folder, "output")
    img_dir = os.path.join(folder, "SegmentedImages")
    csvs = [os.path.join(summary_dir, file) for file in os.listdir(summary_dir) if file.endswith(".csv") and "summary" not in file]
    for csv in csvs:
        csv_name = os.path.basename(csv)
        day = csv_name.split("_")[0][-2:]
        month = csv_name.split("_")[0][-4:-2]
        year = csv_name.split("_")[0][-8:-4]
        this_img_dir = os.path.join(img_dir, f"{day}_{month}_{year}", csv_name.split(".")[0])
        if not os.path.exists(this_img_dir):
            print(f"Could not find image directory for {csv_name}")
            print(f"Expected {this_img_dir}")
            continue
        boats = np.asarray([line.strip().split(",") for line in open(csv) if line[0] != "x"])
        id = 0
        for boat in boats:
            if boat[2] != boat[3]:
                x = float(boat[0])
                y = float(boat[1])
                # get the best subimage
                row = max(y // 104 - 1, 1)
                col = max(x // 104 - 1, 1)
                # get the image
                img_path = os.path.join(this_img_dir, csv_name.split(".")[0] + "_" + str(int(row)) + "_" + str(int(col)) + ".png")
                if not os.path.exists(img_path):
                    all_imgs = all_possible_imgs(x, y)
                    # find one img that does exists
                    for row, col in all_imgs:
                        img_path = os.path.join(this_img_dir, csv_name.split(".")[0] + "_" + str(int(row)) + "_" + str(int(col)) + ".png")
                        if os.path.exists(img_path):
                            break
                        img_path = ""
                    if img_path == "":
                        print(f"Could not find section for {csv_name} with x={x}, y={y}")
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
                    rect = plt.Rectangle((rel_x-10, rel_y-10), 20, 20, linewidth=0.3, edgecolor="gray", facecolor="none")
                    square = plt.Rectangle((rel_x-50, rel_y-50), 100, 100, linewidth=0.3, edgecolor="orange", facecolor="none")
                    ax.add_patch(square)
                    # and annotate the detection as "ML: 0, Label: 1"
                    ax.annotate(f"ML: {ml}, Label: {manual}", (rel_x, rel_y), color="orange", fontsize=6)
                elif ml != -1 and manual == -1:
                    # also draw a big circle around the boat (50x50)
                    rect = plt.Rectangle((rel_x-10, rel_y-10), 20, 20, linewidth=0.3, edgecolor="gray", facecolor="none")
                    circ = plt.Circle((rel_x, rel_y), 50, linewidth=0.3, edgecolor="r", facecolor="none")
                    ax.add_patch(circ)
                    # and annotate the detection as "ML: 0"
                    ax.annotate(f"ML: {ml}", (rel_x, rel_y), color="r", fontsize=6)
                else:
                    # also draw a big star around the boat (50x50)
                    rect = plt.Rectangle((rel_x-10, rel_y-10), 20, 20, linewidth=0.3, edgecolor="gray", facecolor="none")
                    star = plt.Polygon(np.array([[rel_x-50, rel_y-50], [rel_x+50, rel_y-50], [rel_x, rel_y+50]]), linewidth=0.3, edgecolor="y", facecolor="none")
                    ax.add_patch(star)
                    # and annotate the label as "Label: 1"
                    ax.annotate(f"Label: {manual}", (rel_x, rel_y), color="y", fontsize=6)
                ax.add_patch(rect)
                # save the image in really high quality with no axis labels
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, csv_name.split(".")[0] + "_" + str(id) + "_" + str(int(row)) + "_" + str(int(col)) + ".png"), dpi=1000, bbox_inches="tight")
                # also save a text file with the x, y, ml, manual
                with open(os.path.join(output_dir, csv_name.split(".")[0] + "_" + str(id) + ".txt"), "w+") as file:
                    file.write(f"{x}, {y}, {manual}")
                plt.close()
                id += 1
    print("Done")

def all_possible_imgs(x, y, stride=104):
    """
    return a list of tuples (row, col) that would contain the given x and y coords
    """
    row = y // stride - 1
    col = x // stride - 1
    options = []
    # NOTE: we do it like this to try to keep the 'best' subimages as highest priority
    for i in [0,-1, 1]:
        for j in [0, -1, 1]:
            options.append((row + i, col + j))
    for i in [-2, 2]:
        for j in [0, 1, -1, -2, 2]:
            options.append((row + i, col + j))

    return options

### Preparation Helpers

def segregate(directory):
    # separate by day
    days = segregate_by_day(directory)
    # separate by image
    for day in days:
        segregate_by_image(day, day)

def segregate_by_day(directory, into=None):
    """
    Bunch of files in a directory, need to separate into days.
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
    Bunch of files in a directory, need to separate into same image.
    """
    if into is None:
        into = directory
    imgs = []
    print("Segregating by image...")
    for file in os.listdir(directory):
        if not (file.endswith(".png") or file.endswith(".txt")):
            continue
        # everything before the 2nd last underscore is the image name
        img = file[:file.rfind("_", 0, file.rfind("_"))]
        if img not in imgs:
            imgs.append(img)
            os.mkdir(os.path.join(into, img))
        os.rename(os.path.join(directory, file), os.path.join(into, img, file))
    # return the directories
    return [os.path.join(directory, img) for img in imgs]

