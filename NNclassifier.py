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

def remove(path, del_folder=True):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    if del_folder: 
        os.rmdir(path)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="Directory containing .tif files to be classified", required=True)
parser.add_argument("-o", "--output", help="Output file name", required=False, default="BoatsClassified_<DATETIME>.csv")
args = parser.parse_args()

tifDir = args.directory
outfileName = args.output if args.output != "BoatsClassified_<DATETIME>.csv" else f"BoatsClassified_{datetime.now().strftime('%Y%m%d%H%M%S')}"

days = []

# read paths in from .env
load_dotenv()

python_path = os.getenv("PYTHON_PATH")
yolo_path = os.getenv("YOLO_PATH")
data_path = os.getenv("DATA_PATH")
classification_path = os.getenv("CLASSIFICATION_PATH")
weights_path = os.getenv("YOLO_WEIGHTS")

# if any are None, exit
if None in [python_path, yolo_path, data_path, classification_path, weights_path]:
    print("One or more paths are not set in .env. Exiting...")
    exit(1)


# get date from filename for each image
# NOTE: could just do as a set comprehension
for image in os.listdir(f"{tifDir}"):
    if (date := ics.get_date_from_filename(image)) not in days:
        days.append(date)

i = 0
tot = len(os.listdir(tifDir))
for day in days:
    # NOTE: classifying 'date', not image. Multiple images per date are possible
    print(f"Classifying day {i+1} of {len(days)} - {day} ({i/tot*100:.2f}%)")
    allFiles = os.listdir(tifDir)
    files = [file for file in allFiles if ics.get_date_from_filename(file) == day]

    finalStatBoats = []
    finalMovingBoats = []

    # Iterate over all the images present for the current day
    for file in files:
        # Initialise temp directories
        if os.path.isdir("classifier"):
            remove("classifier")
        os.mkdir("classifier")
        if os.path.isdir("tempPNG"):
            remove("tempPNG")
        os.mkdir("tempPNG")

        statClassifications = []
        statSizes = []
        statConfidences = []
        statClasses = []

        movingClassifications = []
        movingSizes = []
        movingConfidences = []
        movingClasses = []

        # ics.create_padded_png(f"{tifDir}", "tempPNG", file)
        # png_path = path.join(os.getcwd(), "tempPNG", f"{file[0:-4]}.png")
        # ics.segment_image_for_classification(png_path, data_path, 416, 104)

        # Actually run the yolov5 classifier on a single image from a single day
        detect_path = path.join(yolo_path, "detect.py")
        # turn weights path into a path object
        weights_path = path.join(os.getcwd(), weights_path)

        #os.system(
        #    f"{python_path} {detect_path} --imgsz 416 --save-txt --save-conf --weights {weights_path} --source {data_path}")

        # Extract this file's classifications from the yolov5 directory
        # latest classifications are classification_path/exp{num}
        # get that folder
        exps = [int(f.split("exp")[1]) if f != "exp" else 0 for f in os.listdir(classification_path) if "exp" in f]
        latest_exp = max(exps) if max(exps) != 0 else ""
        classification_path = path.join(classification_path, f"exp{latest_exp}")
        for classificationFile in os.listdir(os.path.join(classification_path, "labels")):
            with open(os.path.join(classification_path, "labels", classificationFile)) as f:

                fileSplit = classificationFile.split(".txt")[0].split("_")

                row = int(fileSplit[-2])
                col = int(fileSplit[-1])

                across = col * 104
                down = row * 104

                lines = [line.rstrip() for line in f]
                for line in lines:
                    classType, xMid, yMid, xWid, yWid, conf = line.split(" ")
                    if float(conf) > 0.50:
                        # IF NEEDED CODE FOR TANKERS GOES IN HERE
                        if int(classType) == 0:
                            statClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                            statSizes.append([float(xWid), float(yWid)])
                            statConfidences.append(float(conf))
                            statClasses.append(int(classType))
                        elif int(classType) == 1:
                            movingClassifications.append([float(xMid) * 416 + across, float(yMid) * 416 + down])
                            movingSizes.append([float(xWid), float(yWid)])
                            movingConfidences.append(float(conf))
                            movingClasses.append(int(classType))

        # Cluster non-moving boats first
        if statClassifications != [] and len(statClassifications) > 1:
            statClassifications = np.asarray(statClassifications)
            statDistances = scipy.spatial.distance.pdist(statClassifications, metric='euclidean')
            statDistanceCutoff = 6
            statClustering = scipy.cluster.hierarchy.linkage(statDistances, 'average')
            statClusters = scipy.cluster.hierarchy.fcluster(statClustering, statDistanceCutoff, criterion='distance')
            statPointsWithConf = np.c_[statClassifications, statConfidences, statClusters, statClasses, statSizes]
            # NOTE: this is where we could grab out the per-image stats?

            # clusters are labelled 1, 2, ... n
            for i in range(1, max(statClusters) + 1):
                thisBoat = [line for line in statPointsWithConf if int(line[3]) == i]
                thisBoatMean = np.mean(thisBoat, axis=0)
                # class label has turned back into float for some reason, round it
                thisBoatMean[4] = round(thisBoatMean[4])

                # using maximum confidence as the cluster confidence
                # NOTE: this could be changed to avg, or something else
                maxVals = np.max(thisBoat, axis=0)
                thisBoatMean[2] = maxVals[2]
                leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                        os.path.join(os.getcwd(), tifDir, file))
                x = thisBoatMean[0] - leftPad
                y = thisBoatMean[1] - topPad
                xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), tifDir, file))
                thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
                finalStatBoats.append(thisBoatMean)
        elif len(statClassifications) == 1:
            thisBoat = np.c_[statClassifications, statConfidences, 0, statClasses]
            thisBoatMean = np.mean(thisBoat, axis=0)
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                    os.path.join(os.getcwd(), tifDir, file))
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), tifDir, file))
            thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
            finalStatBoats.append(thisBoatMean)

        # Cluster moving boats second
        if movingClassifications != [] and len(movingClassifications) > 1:
            movingClassifications = np.asarray(movingClassifications)
            movingDistances = scipy.spatial.distance.pdist(movingClassifications, metric='euclidean')
            movingDistanceCutoff = 10
            movingClustering = scipy.cluster.hierarchy.linkage(movingDistances, 'average')
            movingClusters = scipy.cluster.hierarchy.fcluster(movingClustering, movingDistanceCutoff,
                                                              criterion='distance')
            movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClusters, movingClasses, movingSizes]

            for i in range(1, max(movingClusters) + 1):
                thisBoat = [line for line in movingPointsWithConf if int(line[3]) == i]
                thisBoatMean = np.mean(thisBoat, axis=0)
                thisBoatMean[4] = round(thisBoatMean[4])
                maxVals = np.max(thisBoat, axis=0)
                thisBoatMean[2] = maxVals[2]
                leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                        os.path.join(os.getcwd(), tifDir, file))
                x = thisBoatMean[0] - leftPad
                y = thisBoatMean[1] - topPad
                xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), tifDir, file))
                thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
                finalMovingBoats.append(thisBoatMean)
        elif len(movingClassifications) == 1:
            thisBoat = np.c_[movingClassifications, movingConfidences, 0, movingClasses]
            thisBoatMean = np.mean(thisBoat, axis=0)
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                os.path.join(os.getcwd(), tifDir, file))
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            xp, yp = ics.pixel2coord(x, y, os.path.join(os.getcwd(), tifDir, file))
            thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
            finalMovingBoats.append(thisBoatMean)

        # Cleanup temp directories
        # rmdir: classification_path
        # rmdir: classifier
        # rmdir: tempPNG
        # rm: data_path contents
        if os.getenv("REMOVE_RUNS") == 1:
            print("Cleaning Temp Directories")
            try:
                remove(classification_path)
            except:
                print("Classification path could not be removed")
            try:
                remove("classifier")
            except:
                print("Classifier could not be removed")
            try:
                remove("tempPNG")
            except:
                print("tempPNG path could not be removed")
            try:
                remove(data_path, False)
            except:
                print("data path could not be removed")


    # Once all images for that day have been classified, we cluster again for boats that are overlapping between images
    finalBoats = []
    thisBoatMean = None
    # Cluster static boats first
    if finalStatBoats != [] and len(finalStatBoats) > 1:
        statClassifications = np.asarray(finalStatBoats)
        statConfidences = statClassifications[:, 2]
        statClusters = statClassifications[:, 3]
        statClasses = statClassifications[:, 4]
        statSizes = statClassifications[:, [5, 6]]
        statClassifications = statClassifications[:, [0, 1]]
        statDistances = scipy.spatial.distance.pdist(statClassifications, metric='euclidean')
        statDistanceCutoff = 0.00025
        statClustering = scipy.cluster.hierarchy.linkage(statDistances, 'average')
        statClusters = scipy.cluster.hierarchy.fcluster(statClustering, statDistanceCutoff, criterion='distance')
        statPointsWithConf = np.c_[statClassifications, statConfidences, statClusters, statClasses, statSizes]

        for i in range(1, max(statClusters) + 1):
            thisBoat = [line for line in statPointsWithConf if int(line[3]) == i]
            thisBoatMean = np.mean(thisBoat, axis=0)
            thisBoatMean[4] = round(thisBoatMean[4])
            maxVals = np.max(thisBoat, axis=0)
            thisBoatMean[2] = maxVals[2]
            finalBoats.append(thisBoatMean)
    elif len(finalStatBoats) == 1:
        statClassifications = np.asarray(finalStatBoats)
        statConfidences = statClassifications[:, 2]
        statClusters = statClassifications[:, 3]
        statClasses = statClassifications[:, 4]
        thisBoat = np.c_[statClassifications, statConfidences, 0, statClasses]
        finalStatBoats.append(thisBoatMean)

    # Final moving boats cluster
    if finalMovingBoats != [] and len(finalMovingBoats) > 1:
        movingClassifications = np.asarray(finalMovingBoats)
        movingConfidences = movingClassifications[:, 2]
        movingClusters = movingClassifications[:, 3]
        movingClasses = movingClassifications[:, 4]
        movingSizes = movingClassifications[:, [5, 6]]
        movingClassifications = movingClassifications[:, [0, 1]]
        movingDistances = scipy.spatial.distance.pdist(movingClassifications, metric='euclidean')
        movingDistanceCutoff = 0.0003
        movingClustering = scipy.cluster.hierarchy.linkage(movingDistances, 'average')
        movingClusters = scipy.cluster.hierarchy.fcluster(movingClustering, movingDistanceCutoff, criterion='distance')
        movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClusters, movingClasses, movingSizes]

        for i in range(1, max(movingClusters) + 1):
            thisBoat = [line for line in movingPointsWithConf if int(line[3]) == i]
            thisBoatMean = np.mean(thisBoat, axis=0)
            thisBoatMean[4] = round(thisBoatMean[4])
            maxVals = np.max(thisBoat, axis=0)
            thisBoatMean[2] = maxVals[2]
            finalBoats.append(thisBoatMean)
    elif len(finalMovingBoats) == 1:
        movingClassifications = np.asarray(finalMovingBoats)
        movingConfidences = movingClassifications[:, 2]
        movingClusters = movingClassifications[:, 3]
        movingClasses = movingClassifications[:, 4]
        thisBoat = np.c_[movingClassifications, movingConfidences, 0, movingClasses]
        thisBoatMean = np.mean(thisBoat, axis=0)
        finalMovingBoats.append(thisBoatMean)

    # Write to output csv
    # Create output csv if it doesn't exist
    if not os.path.isfile(f"{outfileName}.csv"):
        with open(f"{outfileName}.csv", "a+") as outFile:
            outFile.writelines("date,class,latitude,longitude,confidence,w,h\n")

    # Write the data for that day to a csv
    with open(f"{outfileName}.csv", "a+") as outFile:
        for boat in finalBoats:
            outFile.writelines(f"{day},{boat[4]},{boat[1]},{boat[0]},{boat[2]},{boat[5]},{boat[6]}\n")
