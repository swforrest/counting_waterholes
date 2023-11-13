import os
import scipy.cluster
import scipy.spatial

import numpy as np
import matplotlib.pyplot as plt
import imageCuttingSupport as ics

from matplotlib.patches import Rectangle
from PIL import Image
from osgeo import gdal

python_path = "C:\\Users\\mcclymaw\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
yolo_path = "C:\\Users\\mcclymaw\\yolov5"
data_path = "C:\\Users\\mcclymaw\\yolov5\\NNdata"
classification_path = "C:\\Users\\mcclymaw\\yolov5\\runs\\detect\\exp"

current_directory = os.getcwd()


def plot_single_TIF_clustered(TIF_directory, image_name, include_low_conf=False):
    """
    Classifies and then plots the clustered classifications from a single TIF file (single satellite image)
    :param TIF_directory: The directory where the TIF files to be classified and displayed is kept.
    :param image_name: The file name of the image to classify/display.
    :param include_low_conf: True/False to include low confidence classifications (<50% confidence)
    :return: None
    """

    finalStatBoats = []
    finalMovingBoats = []

    # Initialise temp directories
    if os.path.isdir("classifier"):
        os.rmdir("classifier")
    os.mkdir("classifier")
    if os.path.isdir("tempPNG"):
        os.rmdir("tempPNG")
    os.mkdir("tempPNG")

    statClassifications = []
    statConfidences = []
    statClasses = []

    movingClassifications = []
    movingConfidences = []
    movingClasses = []

    finalBoats = []

    ics.create_padded_png(f"{TIF_directory}", "tempPNG", image_name)
    ics.segment_image_for_classification(f"{os.getcwd()}\\tempPNG\\{image_name[0:-4]}.png", data_path, 416, 104)

    # Actually run the yolov5 classifier on a single image from a single day
    os.system(
        f"{python_path} {yolo_path}\\detect.py --imgsz 416 --save-txt --save-conf --weights {yolo_path}\\best.pt --source {data_path}")

    # Extract this file's classifications from the yolov5 directory
    for classificationFile in os.listdir(f"{classification_path}\\labels"):
        with open(f"{classification_path}\\labels\\{classificationFile}", 'r') as f:

            fileSplit = classificationFile.split(".txt")[0].split("_")

            row = int(fileSplit[-2])
            col = int(fileSplit[-1])

            across = col * 104
            down = row * 104

            lines = [line.rstrip() for line in f]
            for line in lines:
                classType, xMid, yMid, xWid, yWid, conf = line.split(" ")
                if float(conf) > 0.50:
                    if int(classType) == 0:
                        statClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                        statConfidences.append(float(conf))
                        statClasses.append(int(classType))
                    elif int(classType) == 1:
                        movingClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                        movingConfidences.append(float(conf))
                        movingClasses.append(int(classType))

    # Cluster non-moving boats first
    if statClassifications != [] and len(statClassifications) > 1:
        statClassifications = np.asarray(statClassifications)
        statDistances = scipy.spatial.distance.pdist(statClassifications, metric='euclidean')
        statDistanceCutoff = 6
        statClustering = scipy.cluster.hierarchy.linkage(statDistances, 'average')
        statClusters = scipy.cluster.hierarchy.fcluster(statClustering, statDistanceCutoff, criterion='distance')
        statPointsWithConf = np.c_[statClassifications, statConfidences, statClusters, statClasses]

        for i in range(1, max(statClusters) + 1):
            thisBoat = [line for line in statPointsWithConf if int(line[3]) == i]
            thisBoatMean = np.mean(thisBoat, axis=0)
            thisBoatMean[4] = round(thisBoatMean[4])
            maxVals = np.max(thisBoat, axis=0)
            thisBoatMean[2] = maxVals[2]
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{TIF_directory}\\{image_name}")
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            thisBoatMean[0], thisBoatMean[1] = x, y
            finalStatBoats.append(thisBoatMean)
    elif len(statClassifications) == 1:
        thisBoat = np.c_[statClassifications, statConfidences, 0, statClasses]
        thisBoatMean = np.mean(thisBoat, axis=0)
        leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
            f"{current_directory}\\{TIF_directory}\\{image_name}")
        x = thisBoatMean[0] - leftPad
        y = thisBoatMean[1] - topPad
        thisBoatMean[0], thisBoatMean[1] = x, y
        finalStatBoats.append(thisBoatMean)

    # Cluster moving boats second
    if movingClassifications != [] and len(movingClassifications) > 1:
        movingClassifications = np.asarray(movingClassifications)
        movingDistances = scipy.spatial.distance.pdist(movingClassifications, metric='euclidean')
        movingDistanceCutoff = 10
        movingClustering = scipy.cluster.hierarchy.linkage(movingDistances, 'average')
        movingClusters = scipy.cluster.hierarchy.fcluster(movingClustering, movingDistanceCutoff,
                                                          criterion='distance')
        movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClusters, movingClasses]

        for i in range(1, max(movingClusters) + 1):
            thisBoat = [line for line in movingPointsWithConf if int(line[3]) == i]
            thisBoatMean = np.mean(thisBoat, axis=0)
            thisBoatMean[4] = round(thisBoatMean[4])
            maxVals = np.max(thisBoat, axis=0)
            thisBoatMean[2] = maxVals[2]
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{TIF_directory}\\{image_name}")
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            thisBoatMean[0], thisBoatMean[1] = x, y
            finalMovingBoats.append(thisBoatMean)
    elif len(movingClassifications) == 1:
        thisBoat = np.c_[movingClassifications, movingConfidences, 0, movingClasses]
        thisBoatMean = np.mean(thisBoat, axis=0)
        leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
            f"{current_directory}\\{TIF_directory}\\{image_name}")
        x = thisBoatMean[0] - leftPad
        y = thisBoatMean[1] - topPad
        thisBoatMean[0], thisBoatMean[1] = x, y
        finalMovingBoats.append(thisBoatMean)

    for boat in finalMovingBoats:
        finalBoats.append(boat)
    for boat in finalStatBoats:
        finalBoats.append(boat)
    for file in os.listdir("tempPNG"):
        os.remove(f"tempPNG\\{file}")
    ics.create_unpadded_png(f"{TIF_directory}", "tempPNG", image_name)
    im = Image.open(os.getcwd() + "\\tempPNG\\" + os.listdir("tempPNG")[0])
    plt.imshow(im)

    ax = plt.gca()

    # PLOT CLUSTERED BOATS
    for boat in finalBoats:
        if boat[2] > 0.75:
            if int(boat[4] == 0):
                rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='g', facecolor='none')
            elif int(boat[4] == 1):
                rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='g', facecolor='none')
        elif boat[2] > 0.01:
            if int(boat[4] == 0):
                rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='y', facecolor='none')
            elif int(boat[4] == 1):
                rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='y', facecolor='none')
        else:  # Red Boxes
            if include_low_conf == True:
                if int(boat[4] == 0):
                    rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
                elif int(boat[4] == 1):
                    rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.title('Boat Classification Confidences')
    labels = ["> 75% Confidence", "> 50% Confidence"]
    if include_low_conf == True:
        labels.append("< 50% Confidence")
    plt.legend(labels, loc="lower left")
    leg = ax.get_legend()
    leg.legend_handles[0].set_color('green')
    leg.legend_handles[1].set_color('yellow')
    if include_low_conf == True:
        leg.legend_handles[2].set_color('red')
    plt.show()

    # Cleanup temp directories
    for file in os.listdir(f"{classification_path}\\labels"):
        os.remove(f"{classification_path}\\labels\\{file}")
    os.rmdir(f"{classification_path}\\labels")
    for file in os.listdir(classification_path):
        os.remove(f"{classification_path}\\{file}")
    os.rmdir(classification_path)
    for file in os.listdir("classifier"):
        os.remove(f"classifier\\{file}")
    os.rmdir("classifier")
    for file in os.listdir("tempPNG"):
        os.remove(f"tempPNG\\{file}")
    os.rmdir("tempPNG")
    for file in os.listdir(data_path):
        os.remove(f"{data_path}\\{file}")


def plot_single_TIF_unclustered(TIF_directory, image_name, include_low_conf=False):
    """
    Classifies and then plots the unclustered classifications from a single TIF file (single satellite image)
    :param TIF_directory: The directory where the TIF files to be classified and displayed is kept.
    :param image_name: The file name of the image to classify/display.
    :param include_low_conf: True/False to include low confidence classifications (<50% confidence)
    :return: None
    """

    finalStatBoats = []
    finalMovingBoats = []

    # Initialise temp directories
    if os.path.isdir("classifier"):
        os.rmdir("classifier")
    os.mkdir("classifier")
    if os.path.isdir("tempPNG"):
        os.rmdir("tempPNG")
    os.mkdir("tempPNG")

    statClassifications = []
    statConfidences = []
    statClasses = []

    movingClassifications = []
    movingConfidences = []
    movingClasses = []

    finalBoats = []

    ics.create_padded_png(f"{TIF_directory}", "tempPNG", image_name)
    ics.segment_image_for_classification(f"{os.getcwd()}\\tempPNG\\{image_name[0:-4]}.png", data_path, 416, 104)

    # Actually run the yolov5 classifier on a single image from a single day
    os.system(
        f"{python_path} {yolo_path}\\detect.py --imgsz 416 --save-txt --save-conf --weights {yolo_path}\\best.pt --source {data_path}")

    # Extract this file's classifications from the yolov5 directory
    for classificationFile in os.listdir(f"{classification_path}\\labels"):
        with open(f"{classification_path}\\labels\\{classificationFile}", 'r') as f:

            fileSplit = classificationFile.split(".txt")[0].split("_")

            row = int(fileSplit[-2])
            col = int(fileSplit[-1])

            across = col * 104
            down = row * 104

            lines = [line.rstrip() for line in f]
            for line in lines:
                classType, xMid, yMid, xWid, yWid, conf = line.split(" ")
                if float(conf) > 0.50:
                    if int(classType) == 0:
                        statClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                        statConfidences.append(float(conf))
                        statClasses.append(int(classType))
                    elif int(classType) == 1:
                        movingClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                        movingConfidences.append(float(conf))
                        movingClasses.append(int(classType))

    if statClassifications != [] and len(statClassifications) > 1:
        statClassifications = np.asarray(statClassifications)
        statPointsWithConf = np.c_[statClassifications, statConfidences, statClasses]

        for boat in statPointsWithConf:
            thisBoat = boat
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{TIF_directory}\\{image_name}")
            x = thisBoat[0] - leftPad
            y = thisBoat[1] - topPad
            thisBoat[0], thisBoat[1] = x, y
            finalStatBoats.append(thisBoat)

    elif len(statClassifications) == 1:
        thisBoat = np.c_[statClassifications, statConfidences, statClasses]
        leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
            f"{current_directory}\\{TIF_directory}\\{image_name}")
        x = thisBoat[0] - leftPad
        y = thisBoat[1] - topPad
        thisBoat[0], thisBoat[1] = x, y
        finalStatBoats.append(thisBoat)

    if movingClassifications != [] and len(movingClassifications) > 1:
        movingClassifications = np.asarray(movingClassifications)
        movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClasses]

        for boat in movingPointsWithConf:
            thisBoat = boat
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{TIF_directory}\\{image_name}")
            x = thisBoat[0] - leftPad
            y = thisBoat[1] - topPad
            thisBoat[0], thisBoat[1] = x, y
            finalMovingBoats.append(thisBoat)
    elif len(movingClassifications) == 1:
        thisBoat = np.c_[movingClassifications, movingConfidences, movingClasses]
        leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
            f"{current_directory}\\{TIF_directory}\\{image_name}")
        x = thisBoat[0] - leftPad
        y = thisBoat[1] - topPad
        thisBoat[0], thisBoat[1] = x, y
        finalMovingBoats.append(thisBoat)

    for boat in finalMovingBoats:
        finalBoats.append(boat)
    for boat in finalStatBoats:
        finalBoats.append(boat)
    for file in os.listdir("tempPNG"):
        os.remove(f"tempPNG\\{file}")

    ics.create_unpadded_png(f"{TIF_directory}", "tempPNG", image_name)
    im = Image.open(os.getcwd() + "\\tempPNG\\" + os.listdir("tempPNG")[0])
    plt.imshow(im)

    ax = plt.gca()

    # PLOT CLUSTERED BOATS
    for boat in finalBoats:
        if boat[2] > 0.75:
            if int(boat[3] == 0):
                rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='g', facecolor='none')
            elif int(boat[3] == 1):
                rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='g', facecolor='none')
        elif boat[2] > 0.01:
            if int(boat[3] == 0):
                rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='y', facecolor='none')
            elif int(boat[3] == 1):
                rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='y', facecolor='none')
        else:  # Red Boxes
            if include_low_conf == True:
                if int(boat[3] == 0):
                    rect = Rectangle((boat[0] - 5, boat[1] - 5), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
                elif int(boat[3] == 1):
                    rect = Rectangle((boat[0] - 10, boat[1] - 10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.title('Boat Classification Confidences')
    labels = ["> 75% Confidence", "> 50% Confidence"]
    if include_low_conf == True:
        labels.append("< 50% Confidence")
    plt.legend(labels, loc="lower left")
    leg = ax.get_legend()
    leg.legend_handles[0].set_color('green')
    leg.legend_handles[1].set_color('yellow')
    if include_low_conf == True:
        leg.legend_handles[2].set_color('red')
    plt.show()

    # Cleanup temp directories
    for file in os.listdir(f"{classification_path}\\labels"):
        os.remove(f"{classification_path}\\labels\\{file}")
    os.rmdir(f"{classification_path}\\labels")
    for file in os.listdir(classification_path):
        os.remove(f"{classification_path}\\{file}")
    os.rmdir(classification_path)
    for file in os.listdir("classifier"):
        os.remove(f"classifier\\{file}")
    os.rmdir("classifier")
    for file in os.listdir("tempPNG"):
        os.remove(f"tempPNG\\{file}")
    os.rmdir("tempPNG")
    for file in os.listdir(data_path):
        os.remove(f"{data_path}\\{file}")


def data_traceback(detection_from_csv, TIF_source_directory):
    """
    Takes in a full boat classification string from an output csv and finds all images where that classifications
    may have originated from.
    :param detection_from_csv: A string of the (row) from a full classifications csv - this row is an individual boat
    classification
    :return: probable_images - A list of images where the classifications in detection_from_csv may have originated from
    """
    detection_data = detection_from_csv.split(",")

    date = detection_data[0]
    lat = detection_data[2]
    long = detection_data[3]

    day, month, year = date.split("/")

    new_date_string = year + month + day

    right_day_images = []

    for file in os.listdir(TIF_source_directory):
        if file[0:8] == new_date_string:
            right_day_images.append(file)

    probable_images = []
    for file in right_day_images:
        ds = gdal.Open(f"{TIF_source_directory}\\{file}")
        metadata = gdal.Info(ds)
        metadata_components = metadata.split("\n")

        top, left = ics.get_cartesian_top_left(metadata_components)
        bottom, right = ics.get_cartesian_bottom_right(metadata_components)

        if float(lat) < top and float(lat) > bottom and float(long) > left and float(long) < right:
            probable_images.append(file)

    return probable_images
