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

def main():
    parser = argparse.ArgumentParser(description='Plot the output of the neural network')
    parser.add_argument('-d', '--detections', help='The path to the detections csv file', required=True)
    parser.add_argument('-i', '--image', help='The path to the image file', required=True)
    args = parser.parse_args()

    # if the image is a folder, run for each image in folder
    if os.path.isdir(args.image):
        for image in os.listdir(args.image):
            if not image.split(".")[1] in ["png", "tif"]:
                continue
            # or if already saved in output
            if os.path.exists(os.path.join("ImgDetections", image.split(".")[0] + ".png")):
                continue
            plot(args.detections, os.path.join(args.image, image), save=True, show=False)
    else:
        plot(args.detections, args.image, save=True, show=False)

def plot(NNcsv, NNpicture, save=False, show=True):
    IMAGE_NAME = os.path.basename(NNpicture)
    temp_path = "./temp" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    # open the csv file, get all rows with image matching image name
    matches = []
    with open(NNcsv, 'r') as f:
        for line in f:
            if IMAGE_NAME in line:
                matches.append(line)
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
    path = None
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
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, fill=False, edgecolor=colors[int(float(class_type))], linewidth=0.2)
        ax.add_patch(rect)
        #plt.text(x - w/2, y - h/2, round(conf, 2), color=colors[int(float(class_type))], size=6)
# legend with colors
    legend = []
    for i in range(len(colors)):
        legend.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i]))
    ax.legend(legend, ['Stationary', 'Moving'])
    plt.axis('off')
    # save 
    if save:
        os.makedirs("ImgDetections", exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join("ImgDetections", IMAGE_NAME.split(".")[0] + ".png"), bbox_inches='tight',
                    dpi=1000)
    if show:
        plt.show()

# remove the temp png
    print("Removing temporary files...")
    if path is not None:
        os.remove(path)
    os.rmdir(temp_path)


if __name__ == "__main__":
    main()





