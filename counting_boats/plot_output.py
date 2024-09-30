import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os
import zipfile
from datetime import datetime
from .boat_utils.config import cfg
from .boat_utils import image_cutting_support as ics

"""
usage: python plotNNoutput.py -d <detections> -i <image> 

Plots all of the detections on the image. The detections are read from the csv file
and only detections with the given image name are plotted. Image can be a folder of
images (tif or png) or a single image. If a folder, the detections are plotted for each
image in the folder. If the image is already saved in the output folder, it is skipped.

Make sure to name the image the same as the image in the detections csv file
    - this is probably {YYYYMMDD}_{aoi}.{tif,png}

Having a config file is a good idea too, so that the tile size and stride are the same
as the ones used in the neural network./

Usage: python counting_boats/plot_output.py -d <detections> -i <image>

"""


def main():
    parser = argparse.ArgumentParser(
        description="Plot the output of the neural network"
    )
    parser.add_argument(
        "-d", "--detections", help="The path to the detections csv file", required=True
    )
    parser.add_argument(
        "-i",
        "--image",
        help="The path to the image file, or folder containing images either tif or png. Must be named the same as the image in the detections csv file",
        required=False,
    )
    parser.add_argument(
        "-z",
        "--zip",
        help="The path to a zip file containing the image. Zip file should be named the same as the image in the detections csv file",
        required=False,
    )

    args = parser.parse_args()

    if args.zip:
        # if a zip, need to extract the zip.
        # extract into ./temp folder
        if not os.path.exists(args.zip):
            print(f"Zip file {args.zip} does not exist")
            return
        # make temp folder and make it empty
        if not os.path.exists("temp"):
            os.mkdir("temp")
        else:
            for file in os.listdir("temp"):
                os.remove(os.path.join("temp", file))
        print("Extracting zip file...")
        with zipfile.ZipFile(args.zip, "r") as zip_ref:
            zip_ref.extractall("temp")
        print("Zip file extracted")
        # get the zipfile name
        zip_name = os.path.basename(args.zip).split(".")[0]
        date = zip_name.split("_")[1]
        aoi = zip_name.split("_")[0]
        # name the image the same as the zip file
        os.rename(
            os.path.join("temp", "composite" + ".tif"),
            os.path.join("temp", f"{date}_{aoi}.tif"),
        )
        args.image = os.path.join("temp", f"{date}_{aoi}.tif")

    # if the image is a folder, run for each image in folder
    if os.path.isdir(args.image):
        print("Plotting detections for all images in folder...")
        for image in os.listdir(args.image):
            if not (".png" in image or ".tif" in image):
                continue
            # or if already saved in output
            if os.path.exists(
                os.path.join("ImgDetections", image.split(".")[0] + ".png")
            ):
                print(f"Skipping {image} as already saved")
                continue
            print(f"Plotting detections for {image}")
            plot(
                args.detections, os.path.join(args.image, image), save=True, show=False
            )
    else:
        plot(args.detections, args.image, save=True, show=False)


def plot(NNcsv, NNpicture, save=False, show=True):
    IMAGE_NAME = os.path.basename(NNpicture)
    temp_path = os.path.join(
        os.path.dirname(NNpicture), "temp" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    )
    # open the csv file, get all rows with image matching image name
    matches = []
    with open(NNcsv, "r") as f:
        for line in f:
            if IMAGE_NAME in line:
                matches.append(line)
    if len(matches) == 0:
        print(f"No detections for {IMAGE_NAME}")
        return
    print("Converting coordinates...")
    tile_size = cfg["TILE_SIZE"]
    stride = cfg["STRIDE"]
    leftPad, rightPad, topPad, bottomPad = ics.get_required_padding(
        NNpicture, tile_size, stride
    )
    coords = []
    for match in matches:
        match = match.split(",")
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
        ics.create_padded_png(
            relative_dir,
            temp_path,
            IMAGE_NAME,
            tile_size=cfg["TILE_SIZE"],
            stride=cfg["STRIDE"],
        )
        im = Image.open(os.path.join(temp_path, IMAGE_NAME.split(".")[0] + ".png"))
    else:
        im = Image.open(NNpicture)

    # Create a drawing context
    draw = ImageDraw.Draw(im)

    # Define colors for different classes
    colors = ["red", "green"]

    # Loop through coordinates and draw rectangles
    for coord in coords:
        x, y, w, h, conf, class_type = coord
        color = colors[int(float(class_type))]

        # Draw rectangle (x - w, y - h) to (x + w, y + h)
        draw.rectangle([(x - w, y - h), (x + w, y + h)], outline=color, width=1)

        # Optional: Add confidence text near the rectangle
        # draw.text((x - w / 2, y - h / 2), f"{round(conf, 2)}", fill=color)

    # Save the image with full quality
    if save:
        os.makedirs("ImgDetections", exist_ok=True)
        im.save(
            os.path.join("ImgDetections", IMAGE_NAME.split(".")[0] + ".png"),
            quality=100,
        )

    # Optionally display the image
    if show:
        im.show()

    # remove the temp png
    print("Removing temporary files...")
    if path is not None:
        os.remove(path)
    os.rmdir(temp_path)


if __name__ == "__main__":
    main()
