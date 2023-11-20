import imageCuttingSupport as ics
import numpy as np
import os
import os.path as path
import scipy.cluster
import scipy.spatial
import argparse
from dotenv import load_dotenv
from datetime import datetime

"""
Usage: python NNclassifier.py -d <.tif directory> -o <output file name>
"""

## Helper functions ## 

def remove(path, del_folder=True):
    """
    Removes all files in a folder, and optionally the folder itself, recursively.
    """
    if not os.path.exists(path):
        return
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            remove(file_path)
    if del_folder:
        os.rmdir(path)

def pixel2latlong(classifications, file):
    """
    Convert the given classifications from pixel coordinates to lat/long.
    :param classifications: The classifications to convert, must have x, y as first two columns.
    :param file: The file these classifications came from.
    """
    leftPad, _, topPad, _ = ics.get_required_padding(os.path.join(os.getcwd(), TIF_DIRECTORY, file))
    for c in classifications:
        x = float(c[0]) - leftPad
        y = float(c[1]) - topPad
        xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), TIF_DIRECTORY, file))
        c[0], c[1] = ics.coord2latlong(xp, yp)
    return classifications

# ENVIRONMENT
# read paths in from .env
load_dotenv()
PYTHON_PATH     = os.getenv("PYTHON_PATH")
YOLO_PATH       = os.getenv("YOLO_PATH")
DATA_PATH       = os.getenv("DATA_PATH")
CLASS_PATH      = os.getenv("CLASSIFICATION_PATH")
WEIGHTS_PATH    = os.getenv("YOLO_WEIGHTS")
# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory containing .tif files to be classified", required=False)
    parser.add_argument("-i", "--images", help="Directory of segmented 416x416 images", required=False)
    parser.add_argument("-t", "--text", help="Directory of text files with classifications", required=False)
    parser.add_argument("-o", "--output", help="Output file name", required=False, default="BoatsClassified_<DATETIME>.csv")
    args = parser.parse_args()
# Set variables
    IMG_DIRECTORY = args.images
    TIF_DIRECTORY = args.directory
    TEXT_DIRECTORY = args.text


    OUTFILE = args.output if args.output != "BoatsClassified_<DATETIME>.csv" else f"BoatsClassified_{datetime.now().strftime('%Y%m%d%H%M%S')}"

## CONSTANTS 
CONF_THRESHOLD                  = 0.5
STAT_DISTANCE_CUTOFF_PIX        = 6
MOVING_DISTANCE_CUTOFF_PIX      = 10
STAT_DISTANCE_CUTOFF_LATLONG    = 0.00025
MOVING_DISTANCE_CUTOFF_LATLONG  = 0.0003

days = []
def main():
    """
    Validate input.
    Run the classifier on each image in the directory.
    """
    # Validate input
    if not validate_input():
        exit(1)
    if IMG_DIRECTORY:
        classify_images(IMG_DIRECTORY)
    elif TIF_DIRECTORY:
        classify_directory(TIF_DIRECTORY)
    elif TEXT_DIRECTORY:
        classify_text(TEXT_DIRECTORY)

def classify_directory(directory):
    """
    Use for directory of tiff images. Preprocesses, classifies, clusters.
    """
    if TIF_DIRECTORY is None:
        print("write_to_csv: This function cannot be called from an external script")
        exit(1)
    init()
    for i, day in enumerate(days):
        print(f"Classifying day {i+1} of {len(days)} - {day} ({i/len(days)*100:.2f}%)")
        allFiles = os.listdir(TIF_DIRECTORY)
        files = [file for file in allFiles if ics.get_date_from_filename(file) == day]
        all_static_boats = []
        all_moving_boats = []
        for file in files:
            classifications, _ = detect(file=file)
            # split into moving and static boats
            static = [c for c in classifications if c[3] == 0]
            moving = [c for c in classifications if c[3] == 1]
            # cluster each set separately
            static_clusters = cluster(static, STAT_DISTANCE_CUTOFF_PIX)
            moving_clusters = cluster(moving, MOVING_DISTANCE_CUTOFF_PIX)
            # process each set separately
            static_boats = process_clusters(static_clusters)
            moving_boats = process_clusters(moving_clusters)
            # within the file, need to convert coordinates to lat/long
            static_boats = pixel2latlong(static_boats, file)
            moving_boats = pixel2latlong(moving_boats, file)
            # add to the list of all boats
            all_static_boats.extend(static_boats)
            all_moving_boats.extend(moving_boats)
            remove(DATA_PATH)
            remove("tempPNG")
            remove("classifier")
            remove(CLASS_PATH)
        # once a day has been classified, need to cluster again
        static_boats = cluster(all_static_boats, STAT_DISTANCE_CUTOFF_LATLONG)
        moving_boats = cluster(all_moving_boats, MOVING_DISTANCE_CUTOFF_LATLONG)
        # process again
        static_boats = process_clusters(static_boats)
        moving_boats = process_clusters(moving_boats)
        # write to csv
        write_to_csv(static_boats, day)
        write_to_csv(moving_boats, day)

def classify_images(images_dir):
    """
        Use when images are already split into 416x416 images.
        Simply runs the classifier and clusters.
    """
    dirs = [path.join(images_dir, dir) for dir in os.listdir(images_dir)]
    # read a day of images at a time
    for dir in dirs:
        day = path.basename(dir)
        day = day.split("_").join("/")
        classifications, _ = detect(dir=dir) 
        # cluster
        clusters = cluster(classifications, STAT_DISTANCE_CUTOFF_PIX)
        # process
        boats = process_clusters(clusters)
        # write to csv
        write_to_csv(boats, day)

def classify_text(dir):
    """
    If images have been classified and the text files are available, use this.
    Clusters and collates based of yolov5 text files.
    """
    classifications, _ = read_classifications("prerun", class_folder=dir)
    # cluster
    clusters = cluster(classifications, STAT_DISTANCE_CUTOFF_PIX)
    # process
    boats = process_clusters(clusters)
    # write to csv
    write_to_csv(boats, "unknown")

def validate_input():
    """
    Validate that all input is correct.
    :return: True if all input is valid, False otherwise.
    """
    if IMG_DIRECTORY is None and TIF_DIRECTORY is None and TEXT_DIRECTORY is None:
        print("Please provide either a directory of image segments (-i) or a directory of tif images (-d) or a directory of text files (-t)")
        exit(1)
    # yolo path should exist
    if not YOLO_PATH or not os.path.isdir(YOLO_PATH):
        print("YOLO_PATH is not a valid directory")
        return False
    # WEIGHTS_PATH should be a valid .pt file
    if not WEIGHTS_PATH or not os.path.isfile(WEIGHTS_PATH):
        print("WEIGHTS_PATH is not a valid .pt file")
        return False
    # CLASS_PATH should be a valid directory
    if not CLASS_PATH:
        print("CLASS_PATH is not a valid directory")
        return False
    return True

def init():
    """
    Populate global variables and initialise directories.
    """
    if TIF_DIRECTORY is None:
        print("write_to_csv: This function cannot be called from an external script")
        exit(1)
    # get date from filename for each image
    for image in os.listdir(TIF_DIRECTORY):
        # check actually valid image:
        if image[-4:] not in [".tif", ".png", "tiff"]:
            continue
        if (date := ics.get_date_from_filename(image)) not in days:
            days.append(date)

def detect(file=None, dir=None):
    """
    Run the classifier 
    :param file: Tiff file to run the classifier on.
    :param dir: Directory of images to run the classifier on.
    :param day: Day to run the classifier on, if dir given.
    :return:    Tuple of classifications > 50%, and low confidence classifications
                The classifications as x, y, confidence, class, width, height, filename
    """
    if TIF_DIRECTORY is None:
        print("write_to_csv: This function cannot be called from an external script")
        exit(1)
    if file is None and dir is None:
        print("detect: Please provide either a file or a directory")
        exit(1)

    # case 1 - tiff image (file) 
    source = DATA_PATH
    if file:
        remove(DATA_PATH)
        if DATA_PATH: os.mkdir(DATA_PATH) 
        remove("tempPNG")
        os.mkdir("tempPNG")
        ics.create_padded_png(TIF_DIRECTORY, "tempPNG", file)
        png_path = path.join(os.getcwd(), "tempPNG", f"{file[0:-4]}.png")
        ics.segment_image_for_classification(png_path, DATA_PATH, 416, 104)
    # case 2 - directory of images (dir)
    elif dir:
        source = dir
        file = "dir" # for labelling in csv
    detect_path = path.join(YOLO_PATH or "", "detect.py")
    weights_path = path.join(WEIGHTS_PATH or "")
    # run the command
    os.system(f"{PYTHON_PATH} {detect_path} --imgsz 416 --save-txt --save-conf --weights {weights_path} --source {source}")
    return read_classifications(file)

def read_classifications(file, class_folder=None):
    if class_folder is None:
        # Classifications are stored in the CLASS_PATH directory in the latest exp folder
        exps = [int(f.split("exp")[1]) if f != "exp" else 0 for f in os.listdir(CLASS_PATH) if "exp" in f]
        latest_exp = max(exps) if max(exps) != 0 else ""
        classification_path = path.join(CLASS_PATH or "", f"exp{latest_exp}", "labels")
    else:
        classification_path = path.join(class_folder)
    # Store classifications as: (x, y, confidence, class, width, height)
    classifications = []
    low_confidence = []
    for classificationFile in os.listdir(classification_path):
        with open(os.path.join(classification_path, classificationFile)) as f:
            fileSplit = classificationFile.split(".txt")[0].split("_")
            row = int(fileSplit[-2])
            col = int(fileSplit[-1])
            across = col * 104
            down = row * 104
            lines = [line.rstrip() for line in f]
            for line in lines:
                classType, xMid, yMid, xWid, yWid, *conf = line.split(" ")
                if len(conf) == 0:
                    conf = 1
                else:
                    conf = conf[0]
                if float(conf) > CONF_THRESHOLD:
                    classifications.append(
                            [float(xMid) * 416 + across, 
                             float(yMid) * 416 + down, 
                             float(conf), int(classType), 
                             float(xWid), float(yWid), os.path.basename(file)])
                else:
                    low_confidence.append([float(xMid) * 416 + across, 
                                           float(yMid) * 416 + down, 
                                           float(conf), int(classType), 
                                           float(xWid), float(yWid), os.path.basename(file)])
    return classifications, low_confidence

def cluster(classifications, cutoff):
    """
    Cluster the given classifications using the given cutoff.
    """
    if len(classifications) < 2:
        # add cluster = 1 to point
        if len(classifications) == 1:
            classifications[0].append(1)
        return classifications
    points              = np.asarray(classifications)[:, [0, 1]].astype(np.float64)
    distances           = scipy.spatial.distance.pdist(points, metric='euclidean')
    clustering          = scipy.cluster.hierarchy.linkage(distances, 'average')
    clusters            = scipy.cluster.hierarchy.fcluster(clustering, cutoff, criterion='distance')
    points_with_cluster = np.c_[points, np.asarray(classifications)[:, 2:], clusters]
    return points_with_cluster

def process_clusters(classifications_with_clusters):
    """
    Process the given classifications with clusters. Condenses each cluster into a single point.
    :param classifications_with_clusters: The classifications as x, y, confidence, class, width, height, filename, cluster
    :return: A list of the condensed classifications in the form: x, y, confidence, class, width, height, filenames
    """
    boats = []
    classifications_with_clusters = np.array(classifications_with_clusters)
    if len(classifications_with_clusters) == 0:
        return []
    # for i in np.unique(classifications_with_clusters[:, -1]):
    #     thisBoat = [line for line in classifications_with_clusters if line[-1] == i]
    #     boats.append(condense(thisBoat))
    # as a comprehension:
    boats = [condense([line for line in classifications_with_clusters if line[-1] == i]) 
             for i in np.unique(classifications_with_clusters[:, -1])]
    return boats

def condense(cluster):
    files = np.unique(np.asarray(cluster)[:, -2])
    # remove cluster number
    thisBoat = np.asarray(cluster)[:, [0, 1, 2, 3, 4, 5]].astype(np.float64)
    thisBoatMean = np.mean(thisBoat, axis=0)
    # class label has turned back into float for some reason, round it
    thisBoatMean[3] = round(thisBoatMean[3])
    # using maximum confidence as the cluster confidence
    maxVals = np.max(thisBoat, axis=0)
    thisBoatMean[2] = maxVals[2]
    return np.append(thisBoatMean, " ".join(files))

def write_to_csv(classifications, day):
    if OUTFILE is None:
        print("write_to_csv: This function cannot be called from an external script")
        exit(1)
    # Write to output csv
    # Create output csv if it doesn't exist
    print(len(classifications), len(classifications[0]))
    print(classifications[0])
    if not os.path.isfile(f"{OUTFILE}.csv"):
        with open(f"{OUTFILE}.csv", "a+") as outFile:
            outFile.writelines("date,class,images,latitude,longitude,confidence,w,h\n")

    # Write the data for that day to a csv
    lines = [f"{day},{boat[3]},{boat[6]},{boat[1]},{boat[0]},{boat[2]},{float(boat[4])*416},{float(boat[5])*416}\n" for boat in classifications]
    with open(f"{OUTFILE}.csv", "a+") as outFile:
        outFile.writelines(lines)

if __name__ == "__main__":
    main()
    exit(0)
