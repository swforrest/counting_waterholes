import imageCuttingSupport as ics
import numpy as np
import os
import scipy.cluster
import scipy.spatial

current_directory = os.getcwd()

days = []

tifDir = input("What is the name of the directory where the .tif files are stored: ")
outfileName = input("What is the desired name of the output .csv file (do not include .csv as part of the name): ")

python_path = "C:\\Users\\mcclymaw\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
yolo_path = "C:\\Users\\mcclymaw\\yolov5"
data_path = "C:\\Users\\mcclymaw\\yolov5\\NNdata"
classification_path = "C:\\Users\\mcclymaw\\yolov5\\runs\\detect\\exp"

for image in os.listdir(f"{current_directory}\\{tifDir}"):
    if (date := ics.get_date_from_filename(image)) not in days:
        days.append(date)

i = 0
tot = len(os.listdir(f"{current_directory}\\{tifDir}"))
for day in days:
    print(f"Classifying image {i} of {len(days)}")
    allFiles = os.listdir(f"{current_directory}\\{tifDir}")
    files = [file for file in allFiles if ics.get_date_from_filename(file) == day]

    finalStatBoats = []
    finalMovingBoats = []

    # Iterate over all the images present for the current day
    for file in files:
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

        ics.create_padded_png(f"{tifDir}", "tempPNG", file)
        ics.segment_image_for_classification(f"{os.getcwd()}\\tempPNG\\{file[0:-4]}.png", data_path, 416, 104)

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
                        # IF NEEDED CODE FOR TANKERS GOES IN HERE
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
                    f"{current_directory}\\{tifDir}\\{file}")
                x = thisBoatMean[0] - leftPad
                y = thisBoatMean[1] - topPad
                xp, yp = ics.pixel2coord(x, y, f"{current_directory}\\{tifDir}\\{file}")
                thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
                finalStatBoats.append(thisBoatMean)
        elif len(statClassifications) == 1:
            thisBoat = np.c_[statClassifications, statConfidences, 0, statClasses]
            thisBoatMean = np.mean(thisBoat, axis=0)
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{tifDir}\\{file}")
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            xp, yp = ics.pixel2coord(x, y, f"{current_directory}\\{tifDir}\\{file}")
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
            movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClusters, movingClasses]

            for i in range(1, max(movingClusters) + 1):
                thisBoat = [line for line in movingPointsWithConf if int(line[3]) == i]
                thisBoatMean = np.mean(thisBoat, axis=0)
                thisBoatMean[4] = round(thisBoatMean[4])
                maxVals = np.max(thisBoat, axis=0)
                thisBoatMean[2] = maxVals[2]
                leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                    f"{current_directory}\\{tifDir}\\{file}")
                x = thisBoatMean[0] - leftPad
                y = thisBoatMean[1] - topPad
                xp, yp = ics.pixel2coord(x, y, f"{current_directory}\\{tifDir}\\{file}")
                thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
                finalMovingBoats.append(thisBoatMean)
        elif len(movingClassifications) == 1:
            thisBoat = np.c_[movingClassifications, movingConfidences, 0, movingClasses]
            thisBoatMean = np.mean(thisBoat, axis=0)
            leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
                f"{current_directory}\\{tifDir}\\{file}")
            x = thisBoatMean[0] - leftPad
            y = thisBoatMean[1] - topPad
            xp, yp = ics.pixel2coord(x, y, f"{current_directory}\\{tifDir}\\{file}")
            thisBoatMean[0], thisBoatMean[1] = ics.coord2latlong(xp, yp)
            finalMovingBoats.append(thisBoatMean)

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

    # Once all images for that day have been classified, we cluster again for boats that are overlapping between images
    finalBoats = []
    # Cluster static boats first
    if finalStatBoats != [] and len(finalStatBoats) > 1:
        statClassifications = np.asarray(finalStatBoats)
        statConfidences = statClassifications[:, 2]
        statClusters = statClassifications[:, 3]
        statClasses = statClassifications[:, 4]
        statClassifications = statClassifications[:, [0, 1]]
        statDistances = scipy.spatial.distance.pdist(statClassifications, metric='euclidean')
        statDistanceCutoff = 0.00025
        statClustering = scipy.cluster.hierarchy.linkage(statDistances, 'average')
        statClusters = scipy.cluster.hierarchy.fcluster(statClustering, statDistanceCutoff, criterion='distance')
        statPointsWithConf = np.c_[statClassifications, statConfidences, statClusters, statClasses]

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
        movingClassifications = movingClassifications[:, [0, 1]]
        movingDistances = scipy.spatial.distance.pdist(movingClassifications, metric='euclidean')
        movingDistanceCutoff = 0.0003
        movingClustering = scipy.cluster.hierarchy.linkage(movingDistances, 'average')
        movingClusters = scipy.cluster.hierarchy.fcluster(movingClustering, movingDistanceCutoff, criterion='distance')
        movingPointsWithConf = np.c_[movingClassifications, movingConfidences, movingClusters, movingClasses]

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
            outFile.writelines("date,class,latitude,longitude,confidence\n")

    # Write the data for that day to a csv
    with open(f"{outfileName}.csv", "a+") as outFile:
        for boat in finalBoats:
            outFile.writelines(f"{day},{boat[4]},{boat[1]},{boat[0]},{boat[2]}\n")
