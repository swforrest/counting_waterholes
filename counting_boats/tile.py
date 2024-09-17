"""
This one was used create a collage image from multiple smaller images. 
Not documented, not tested, but left here for reference.

Author: Charlie Turner
Date: 02/09/2024
"""

from PIL import Image
from PIL import ImageDraw
import os


def main():
    """
    Tile the images in the folder.
    images are named {x}_{y}_{row}_{col}.png
    We want to create a super image which is a grid of the images increasing column left to right and row top to bottom.
    """
    folder = (
        "/Users/charlieturner/Documents/CountingBoats/runs/val/val0/imgs/tiles/boat_10"
    )
    images = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            images.append(file)
    images = sorted(
        images
    )  # this sorts row first then column, so can work top to bottom
    im_size = 416
    num_ims = 4
    master = Image.new("RGB", (im_size * num_ims, im_size * num_ims))
    print(images)
    min_row = min([int(image.split("_")[2]) for image in images])
    min_col = min([int(image.split("_")[3].split(".")[0]) for image in images])
    for image in images:
        image = image.split(".")[0]
        row, col = image.split("_")[2], image.split("_")[3]
        row, col = int(row), int(col)
        im = Image.open(folder + "/" + image + ".png")
        master.paste(im, ((col - min_col) * im_size, (row - min_row) * im_size))
    master.save(folder + "/master.png")


def magnify():
    image = Image.open(
        "/Users/charlieturner/Documents/CountingBoats/runs/val/val0/imgs/tiles/boat_10/master.png"
    )
    # rotate 180
    image = image.rotate(180)
    # make a 'magnifying glass' effect over the top right corner of the image.
    # new image will have 25% more height and width, and a circle of diameter 50% over the bottom left will be magnified by 2x over the corner
    width, height = image.size
    new_width = int(width * 1.25)
    new_height = int(height * 1.25)
    # new image should be transparent
    new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    # paste the image at the bottom left
    new_image.paste(image, (0, new_height - height))
    # draw a circle radius 0.25 * height at the bottom left corner
    circle_diameter = int(0.5 * height * 0.95)
    margin = int(0.05 * height)
    draw = ImageDraw.Draw(new_image)
    draw.ellipse(
        (
            margin,
            new_height - circle_diameter - margin,
            margin + circle_diameter,
            new_height - margin,
        ),
        outline="white",
        width=5,
    )
    # paste the magnified image at the top right
    sub_image = image.crop(
        (
            margin,
            height - (circle_diameter + margin),
            margin + circle_diameter,
            height - margin,
        )
    )
    sub_image = sub_image.resize((circle_diameter * 2, circle_diameter * 2))
    # paste as circle
    mask = Image.new("L", sub_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + sub_image.size, fill=255)
    new_image.paste(
        sub_image, (new_width - 2 * circle_diameter - margin, 0 + margin), mask
    )
    draw = ImageDraw.Draw(new_image)
    # white circle around big circle
    draw.ellipse(
        (
            new_width - 2 * circle_diameter - margin,
            margin,
            new_width - margin,
            2 * circle_diameter + margin,
        ),
        outline="white",
        width=5,
    )

    new_image.save(
        "/Users/charlieturner/Documents/CountingBoats/runs/val/val0/imgs/tiles/boat_10/magnified.png"
    )


if __name__ == "__main__":
    # main()
    magnify()
