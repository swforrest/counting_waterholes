import os
import datetime
from PIL import Image



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
    imgs = os.listdir(dir)
    imgs = filter(lambda x: x.endswith(".png"), imgs)
    # for each file, get the x and y coords
    coords = []
    for img in imgs:
        if img == "stitched.png":
            continue
        y, x = img.split("_")[-2:]
        x = x.split(".")[0]
        coords.append((img, x, y))
    
    # get the maximum x and y values
    max_x = max([int(x[1]) for x in coords])
    max_y = max([int(x[2]) for x in coords])
    # we want to use every 4th of each from 0-max_x and 0-max_y
    rows = range(0, max_x+1, 4)
    cols = range(0, max_y+1, 4)
    image = Image.new("RGB", (416*len(cols), 416*len(rows)))
    # for each image, paste it into the new image
    for img, x, y in coords:
        x = int(x)
        y = int(y)
        # paste the image into the new image
        im = Image.open(os.path.join(dir, img))
        image.paste(im, (x*104, y*104))
    image.save(os.path.join(dir, "stitched.png"))


if __name__ == "__main__":
    main()


