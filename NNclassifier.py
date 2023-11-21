import imageCuttingSupport as ics
import numpy as np
import os
import os.path as path
import scipy.cluster
import scipy.spatial
import argparse
import yaml
from datetime import datetime

"""
Intended usage:
    Automatically called once per day
    Downloads .tif files into directory
    Runs the classifier on each .tif file (with preprocessing)
    Processes the outputs
    Saves the results to a csv file. 

Optionally?:
    Save some random images to a folder for manual checking after the fact

Usage: python NNclassifier.py -d <.tif directory> -o <output file name>
"""


def main():
    """
    Validate input.
    Run the classifier on each image in the directory.
    """
    classify_directory(config.get("tif_dir"))

def process_tif(file, data_path, stat_cutoff, moving_cutoff):
    """
    Processes a single tif file
    """
    classifications, _ = detect(python=config["python"], weights=config["weights"], yolo_dir=config["yolo_dir"], tif_dir = config["tif_dir"] , file_name = file)
    # split into moving and static boats
    static = [c for c in classifications if c[3] == 0]
    moving = [c for c in classifications if c[3] == 1]
    # cluster each set separately
    static_clusters = cluster(static, stat_cutoff)
    moving_clusters = cluster(moving, moving_cutoff)
    # process each set separately
    static_boats = process_clusters(static_clusters)
    moving_boats = process_clusters(moving_clusters)
    # within the file, need to convert coordinates to lat/long
    static_boats = pixel2latlong(static_boats, file, config["tif_dir"])
    moving_boats = pixel2latlong(moving_boats, file, config["tif_dir"])
    # Clean Up Required Directories
    remove(data_path)
    remove("tempPNG")
    # add to the list of all boats
    return static_boats, moving_boats

def classify_directory(directory):
    """
    Use for directory of tiff images. Preprocesses, classifies, clusters.
    """
    days = {ics.get_date_from_filename(file) for file in os.listdir(directory)}
    days.remove(None)
    allFiles = os.listdir(directory)
    for i, day in enumerate(days):
        print(f"Classifying day {i+1} of {len(days)} - {day} ({i/len(days)*100:.2f}%)")
        files = [file for file in allFiles if ics.get_date_from_filename(file) == day]
        all_static_boats, all_moving_boats = zip(*[process_tif(file, 
                                                               config["img_path"], 
                                                               config["STAT_DISTANCE_CUTOFF_PIX"], 
                                                               config["MOVING_DISTANCE_CUTOFF_PIX"]) 
                                                   for file in files])
        # once a day has been classified, need to cluster again
        static_boats = cluster(all_static_boats, config["STAT_DISTANCE_CUTOFF_LATLONG"])
        moving_boats = cluster(all_moving_boats, config["MOVING_DISTANCE_CUTOFF_LATLONG"])
        # process again
        static_boats = process_clusters(static_boats)
        moving_boats = process_clusters(moving_boats)
        # write to csv
        write_to_csv(static_boats, day, config["OUTFILE"])
        write_to_csv(moving_boats, day, config["OUTFILE"])

def classify_images(images_dir, STAT_DISTANCE_CUTOFF_PIX, OUTFILE):
    """
    Use when images are already split into 416x416 images.
    Simply runs the classifier and clusters.
    """
    dirs = [path.join(images_dir, dir) for dir in os.listdir(images_dir)]
    # read a day of images at a time
    for dir in dirs:
        day = path.basename(dir)
        day = day.split("_").join("/")
        classifications, _ = detect(python=config["python"], weights=config["weights"], yolo_dir=config["yolo_dir"], img_dir = dir) 
        # cluster
        clusters = cluster(classifications, STAT_DISTANCE_CUTOFF_PIX)
        # process
        boats = process_clusters(clusters)
        # write to csv
        write_to_csv(boats, day, OUTFILE)

def classify_text(dir, STAT_DISTANCE_CUTOFF_PIX, OUTFILE):
    """
    If images have been classified and the text files are available, use this.
    Clusters and collates from yolov5 text files.
    """
    classifications, _ = read_classifications("prerun", class_folder=dir)
    # cluster
    clusters = cluster(classifications, STAT_DISTANCE_CUTOFF_PIX)
    # process
    boats = process_clusters(clusters)
    # write to csv
    write_to_csv(boats, "unknown", OUTFILE)

def detect(
        python, 
        weights,
        yolo_dir,
        temp_dir = None,
        file_name = None,
        tif_dir = None,
        img_dir = None, 
        ):
    """
    Run the classifier 
    :param file: Tiff file to run the classifier on.
    :param dir: Directory of images to run the classifier on.
    :param day: Day to run the classifier on, if dir given.
    :return:    Tuple of classifications > 50%, and low confidence classifications
                The classifications as x, y, confidence, class, width, height, filename
    """
    if temp_dir is None:
        temp_dir = "temp"
    remove(temp_dir)
    source = temp_dir
    # case 1 - tiff image (file)
    if file_name:
        os.mkdir(temp_dir) 
        remove("tempPNG")
        os.mkdir("tempPNG")
        ics.create_padded_png(tif_dir, "tempPNG", file_name)
        png_path = path.join(os.getcwd(), "tempPNG", f"{file_name[0:-4]}.png")
        ics.segment_image_for_classification(png_path, temp_dir, 416, 104)
    # case 2 - directory of segmented images (dir)
    elif img_dir:
        source = img_dir
        file_name = "dir"                   # for labelling in csv
    detect_path = path.join(yolo_dir, "detect.py")
    # run the command
    os.system(f"{python} {detect_path} --imgsz 416 --save-txt --save-conf --weights {weights} --source {source}")
    return read_classifications(file_name, yolo_dir=yolo_dir, confidence_threshold=config["CONF_THRESHOLD"])

def read_classifications(file, yolo_dir=None, class_folder=None, confidence_threshold=0.5):
    if class_folder is None:
        assert yolo_dir is not None, "Must provide yolo_dir if class_folder is not provided"
        # Classifications are stored in the CLASS_PATH directory in the latest exp folder
        exps = [int(f.split("exp")[1]) if f != "exp" else 0 for f in os.listdir(os.path.join(yolo_dir, "runs", "detect" )) if "exp" in f]
        latest_exp = max(exps) if max(exps) != 0 else ""
        classification_path = path.join(os.path.join(yolo_dir, "runs", "detect"), f"exp{latest_exp}", "labels")
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
                if float(conf) > confidence_threshold:
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

def write_to_csv(classifications, day, file):
    # Write to output csv
    # Create output csv if it doesn't exist
    print(len(classifications), len(classifications[0]))
    print(classifications[0])
    if not os.path.isfile(f"{file}.csv"):
        with open(f"{file}.csv", "a+") as outFile:
            outFile.writelines("date,class,images,latitude,longitude,confidence,w,h\n")

    # Write the data for that day to a csv
    lines = [f"{day},{boat[3]},{boat[6]},{boat[1]},{boat[0]},{boat[2]},{float(boat[4])*416},{float(boat[5])*416}\n" for boat in classifications]
    with open(f"{file}.csv", "a+") as outFile:
        outFile.writelines(lines)

## Helper functions ## 
def get_config():
    """
    parses command line args, and gets config.yml. 
    Command line args have precedence and can be used to override config.yml
    """
    config:dict = yaml.safe_load(open("config.yml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",    help="Directory containing .tif files to be classified", required=False)
    parser.add_argument("-i", "--images",       help="Directory of segmented 416x416 images", required=False)
    parser.add_argument("-t", "--text",         help="Directory of text files with classifications", required=False)
    parser.add_argument("-o", "--output",       help="Output file name", required=False)
    args = parser.parse_args()
    # command line args override config.yml
    if args.directory:
        config["tif_dir"] = args.directory
    if args.images:
        config["img_path"] = args.images
    if args.text:
        config["text_dir"] = args.text
    if args.output:
        config["OUTFILE"] = args.output
    else:
        if config.get("OUTFILE") is None:
            config["OUTFILE"] = config["OUTFILE"].replace("<DATETIME>", datetime.now().strftime("%Y%m%d_%H%M%S"))
    # replace <DATETIME> with current datetime
    return config

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

def pixel2latlong(classifications, file, tif_dir):
    """
    Convert the given classifications from pixel coordinates to lat/long.
    :param classifications: The classifications to convert, must have x, y as first two columns.
    :param file: The file these classifications came from.
    """
    leftPad, _, topPad, _ = ics.get_required_padding(os.path.join(os.getcwd(), tif_dir, file))
    for c in classifications:
        x = float(c[0]) - leftPad
        y = float(c[1]) - topPad
        xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), tif_dir, file))
        c[0], c[1] = ics.coord2latlong(xp, yp)
    return classifications

if __name__ == "__main__":
    global config 
    config = get_config()
    main()

