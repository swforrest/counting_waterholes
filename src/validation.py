import os
import argparse
import numpy as np
import scipy.cluster
import scipy.spatial
import matplotlib.pyplot as plt
import shutil
import utils.image_cutting_support as ics
from sklearn.metrics import ConfusionMatrixDisplay, f1_score 
from utils.classifier import read_classifications, cluster, process_clusters

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
COMPARE_DISTANCE_CUTOFF_PIX = 8
CONF_THRESHOLD = 0.5

def process_image(path, labels, summary):
    label_dir = os.path.join(labels, os.path.sep.join(path.split("/")[-2:]))
    # check if it exists
    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} does not exist, skipping...")
        return []
    # ML classifications
    ML_classifications, _ = read_classifications(class_folder=path, confidence_threshold=CONF_THRESHOLD)
    ML_classifications_stat = ML_classifications[ML_classifications[:, 3] == 0.0] 
    summary["detected_stat"] += len(ML_classifications_stat)
    ML_classifications_moving = ML_classifications[ML_classifications[:, 3] == 1.0]
    summary["detected_mov"] += len(ML_classifications_moving)
    # cluster
    ML_clusters_stat = cluster(ML_classifications_stat, STAT_DISTANCE_CUTOFF_PIX)
    ML_clusters_moving = cluster(ML_classifications_moving, MOVING_DISTANCE_CUTOFF_PIX)
    # manual annotations
    manual_annotations, _ = read_classifications(class_folder=label_dir)
    manual_annotations_stat = manual_annotations[manual_annotations[:, 3] == 0.0]
    summary["labelled_stat"] += len(manual_annotations_stat)
    manual_annotations_moving = manual_annotations[manual_annotations[:, 3] == 1.0]
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

    ML_clusters = np.concatenate((ML_clusters_stat, ML_clusters_moving))
    manual_clusters = np.concatenate((manual_clusters_stat, manual_clusters_moving))
    comparison = compare(ML_clusters, manual_clusters, COMPARE_DISTANCE_CUTOFF_PIX)
    return comparison

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
    print(f"Processed {i} images")
    summarize(outdir, summary)

def summarize(outdir, summary):
    """
    Open all csv files in the directory and summarize them
    """
    # get all lines (except headers) from all files using np list comp
    all_boats = np.asarray([line.strip().split(",") for file in os.listdir(outdir) if (file.endswith(".csv") and "summary" not in file) for line in open(os.path.join(outdir, file)) if line[0] != "x"])
    if len(all_boats) == 0:
        print("No boats found")
        return
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


def compare(ml:np.ndarray, manual:np.ndarray, cutoff):
    """
    given two lists of clusters, compare them (cluster them and note the results)
    e.g if ml has the point (52, 101), and manual has (51.8, 101.2), they should be clustered together
    , and this boat should be noted as being in both sets
    :param ml: list of clusters in form [x, y, confidence, class, width, height, filename]
    :param manual: list of clusters in form [x, y, confidence, class, width, height, filename]
    :return list of clusters in form [x, y, confidence, class, width, height, filename, in_ml, in_manual]
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

def write_to_csv(boats, filename, outdir):
    path = os.path.join(outdir, filename + ".csv")
    with open(path, "w+") as file:
        file.write("x, y, ml_class, manual_class\n")
        for boat in boats:
            file.write(f"{boat[0]}, {boat[1]}, {boat[2]}, {boat[3]}\n")

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
    # filter to images which have a csv
    all_images = [image for image in all_images if any([image.split("/")[-1].split(".")[0] in csv for csv in [s.split("/")[-1].split(".")[0] for s in all_csvs]])]
    print(all_images)
    if len(all_images) == 0:
        # try to see if the stitched images exist
        all_images = [os.path.join(root, file) for root, dirs, files in os.walk(imgs) for file in files if file == "stitched.png"]
    i = 0
    for csv in all_csvs:
        # get the corresponding image
        img = [image for image in all_images if csv.split("/")[-1].split(".")[0] in image]
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
        plt.savefig(os.path.join(outdir, csv.split("/")[-1].split(".")[0] + ".png"), dpi=1000, bbox_inches="tight")
        plt.close()
        i += 1
        print(f"Plotted {i}/{len(all_images)} images", end="\r")

def segregate(directory):
    # separate by day
    days = segregate_by_day(directory)
    # separate by image
    for day in days:
        segregate_by_image(day, day)




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

def all_possible_imgs(x, y):
    """
    return a list of tuples (row, col) that would contain the given x and y coords
    """
    row = y // 104 - 1
    col = x // 104 - 1
    options = []
    # NOTE: we do it like this to try to keep the 'best' subimages as highest priority
    for i in [0,-1, 1]:
        for j in [0, -1, 1]:
            options.append((row + i, col + j))
    for i in [-2, 2]:
        for j in [0, 1, -1, -2, 2]:
            options.append((row + i, col + j))

    return options

def find_ml_mistakes(summary_dir, img_dir):
    """
    Given a summary csv, find the images where there is a mistake made.
    Since the x and y in the summary refer to the entire image, we need to 
    calculate the subimage(s) that the boat is in. save the best subimage (most central)
    to a new directory with the type of mistake (e.g "false_positive")
    """
    output_dir = os.path.join(summary_dir, "mistakes")
    os.makedirs(output_dir, exist_ok=True)
    # for image:
    #   for boat:
    #       if mistake:
    #           find the subimage
    #           draw a box around the boat
    #           save the image
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



def infer_from_imgs(img_dir, classification_dir):
    # img dir has subdirectories with images (416x416) and stitched.png
    # classification dir should end up mirroring img dir, but with classifications
    weights = "/Users/charlieturner/Documents/CountingBoats/best_weights.pt"
    for root, subdirs, files in os.walk(img_dir):
        if len(files) > 0 and files[0].endswith(".png") and "stitched" not in root:
            # we are in a directory with images
            this_classification_dir = os.path.join(classification_dir, "/".join(root.split("/")[-2:]))
            # if the dir exists, skip it
            if os.path.exists(this_classification_dir):
                continue
            os.makedirs(this_classification_dir, exist_ok=True)
            # move stitched.png to stitched/stiched.png
            if os.path.exists(os.path.join(root, "stitched.png")):
                os.mkdir(os.path.join(root, "stitched"))
                os.rename(os.path.join(root, "stitched.png"), os.path.join(root, "stitched", "stitched.png"))
            # classify the imgs in this dir
            yolo = "/Users/charlieturner/yolov5"
            os.system(f"python {yolo}/detect.py --imgsz 416 --save-txt --save-conf --weights {weights} --source {root}")
            # move the classifications to the classification dir
            # currently in ~/yolov5/runs/detect/exp{""|num}/labels
            exps = [int(f.split("exp")[1]) if f != "exp" else 0 for f in os.listdir(os.path.join(yolo, "runs", "detect" )) if "exp" in f]
            latest_exp = max(exps) if max(exps) != 0 else ""
            # os.rename(os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels"), this_classification_dir)
            # dir isn't empty, use shutil instead:
            for file in os.listdir(os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels")):
                shutil.move(os.path.join(yolo, "runs", "detect", f"exp{latest_exp}", "labels", file), this_classification_dir)
            print(f"Classified {root}, saved to {this_classification_dir}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
    argparser.add_argument("-f", "--folder", help="Directory of classifications and labels", required=False)
    argparser.add_argument("-c", "--classifications", help="Directory of classifications from yolo (can contain multiple direcories)", required=False)
    argparser.add_argument("-l", "--labels", help="Directory of manual annotations (can contain multiple direcories)", required=False) 
    argparser.add_argument("-o", "--outdir", help="Directory to dump output csvs", required=False)
    argparser.add_argument("-i", "--imgs", help="Directory with images", required=False)
    argparser.add_argument("-s", "--summaries", help="Directory with summary CSVs", required=False)
    args = argparser.parse_args()
    if args.folder is not None:
        args.classifications = os.path.join(args.folder, "Classifications")
        args.labels = os.path.join(args.folder, "Labels")
        args.outdir = os.path.join(args.folder, "Summary")
        args.imgs = os.path.join(args.folder, "SegmentedImages")
        args.summaries = os.path.join(args.folder, "Summary")
    classifications = args.classifications
    labels = args.labels
    outdir = args.outdir
    imgs = args.imgs
    summaries = args.summaries

    # check what we want to do
    print("1. Cluster and compare")
    print("2. Plot boats")
    print("3. Separate Files")
    print("4. Infer From Images")
    print("5. Highlight ML Mistakes")
    choice = input("Enter a number: ")
    if choice == "1":
        main(classifications, labels, outdir)
    elif choice == "2":
        folder = args.folder if args.folder is not None else imgs
        plot_boats(summaries, folder)
    elif choice == "3":
        segregate(labels)
        segregate(imgs)
    elif choice == "4":
        infer_from_imgs(imgs, classifications)
    elif choice == "5":
        find_ml_mistakes(summaries, imgs)


