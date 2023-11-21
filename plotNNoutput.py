import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import imageCuttingSupport as ics
import os
from datetime import datetime

"""
usage: python plotNNoutput.py -d <detections> -i <image> 
"""

parser = argparse.ArgumentParser(description='Plot the output of the neural network')
parser.add_argument('-d', '--detections', help='The path to the detections csv file', required=True)
parser.add_argument('-i', '--image', help='The path to the image file', required=True)
args = parser.parse_args()

NNcsv = args.detections
NNpicture = args.image

IMAGE_NAME = os.path.basename(NNpicture)

temp_path = "./temp" + datetime.now().strftime("%Y%m%d%H%M%S%f")

# open the csv file, get all rows with image matching image name

matches = []

with open(NNcsv, 'r') as f:
    for line in f:
        if IMAGE_NAME in line:
            matches.append(line)
# matches have the form:
# date,class,images,latitude,longitude,confidence,w,h
# use ics to convert lat long into image coords
# then plot the image with the bounding boxes

# coords will be (x, y, w, h, conf) - center of box, width, height
# all in relation to the image
# NOTE: include class here at some stage
print("Converting coordinates...")
leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(NNpicture)
coords = []
for match in matches:
    match = match.split(',')
    class_type = match[1]
    lat = float(match[3])
    long = float(match[4])
    conf = float(match[5])
    w = float(match[6])
    h = float(match[7])
    x, y = ics.latlong2coord(long, lat)
    x, y = ics.coord2pixel(x, y, NNpicture)
    # now have to adjust for padding
    x = x + leftPad
    y = y + topPad
    coords.append((x, y, w, h, conf, class_type))


# now plot the image with plt
# and put the bounding boxes on it

# first have to create the png if doesn't exist
if IMAGE_NAME.split(".")[1] != "png":
    print("Creating temporary png...")
    im = None
    path = os.path.join(temp_path, IMAGE_NAME.split(".")[0] + ".png")
    relative_dir = os.path.dirname(NNpicture)
    os.mkdir(temp_path)
    ics.create_padded_png(relative_dir, temp_path, IMAGE_NAME)
    im = Image.open(os.path.join(temp_path, IMAGE_NAME.split(".")[0] + ".png"))
else:
    im = Image.open(NNpicture)

fig, ax = plt.subplots(1)
ax.imshow(im)
colors = ['r', 'g']
for coord in coords:
    # plot different classes in different colours (numbered)
    x, y, w, h, conf, class_type = coord
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, fill=False, edgecolor=colors[int(float(class_type))], linewidth=2)
    ax.add_patch(rect)
    plt.text(x - w/2, y - h/2, round(conf, 2), color=colors[int(float(class_type))])

# legend with colors
legend = []
for i in range(len(colors)):
    legend.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i]))
ax.legend(legend, ['Stationary', 'Moving'])
plt.axis('off')

plt.show()

# remove the temp png
print("Removing temporary files...")
os.remove(path)
os.rmdir(temp_path)








