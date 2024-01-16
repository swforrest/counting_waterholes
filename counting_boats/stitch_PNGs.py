import os
import datetime
from PIL import Image
"""
Given the training image grids, stitch them back together into one image
"""

def main():
    # get the current directory
    dir = input("Enter the directory of images to stitch: ")
    # recursively search the directory for a directory containing images
    # if there is a directory, add to list
    dirs = []
    for root, subdirs, files in os.walk(dir):
        if len(files) > 0 and files[0].endswith(".png"):
            dirs.append(root)
    # for each directory, stitch the images
    for dir in dirs:
        stitch(dir)


def stitch(dir):
    """
    for each png in the directory, pull out the x, y coords
    images have names like imagessdfasdf_x_y.png
    They are all 416x416, and overlap 104 pixels.
    need to stich them back together into one image
    """
    # get list of files in directory
    if "stitched.png" in os.listdir(dir):
        print("Already stitched {}".format(dir))
        return
    imgs = os.listdir(dir)
    imgs = filter(lambda x: x.endswith(".png"), imgs)
    # for each file, get the x and y coords
    coords = [
        (img, int(img.split("_")[-1].split(".")[0]), int(img.split("_")[-2])) for img in imgs if img != "stitched.png"
            ]
    # get the maximum x and y values
    max_x = max([x[1] for x in coords])
    max_y = max([x[2] for x in coords])
    print("Max x: {}, Max y: {}".format(max_x, max_y))
    # we want to use every 4th of each from 0-max_x and 0-max_y
    image = Image.new("RGB", (104*(max_x+1), 104*(max_y+1)))
    # for each image, paste it into the new image
    [image.paste(Image.open(os.path.join(dir, img)), (x*104, y*104)) for img, x, y in coords]
    image.save(os.path.join(dir, "stitched.png"))

if __name__ == "__main__":
    main()

