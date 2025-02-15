"""
Simple GUI used to reconcile issues with ML detections for labelled images.
1. Displays an image, and three buttons: "Boat", "Moving Boat", "Neither"
2. User clicks the appropriate button, and the result is saved to a CSV file, and that image's labels updated.

Not super useful but left here for reference. General usage was to:
    label images -> run ML inference -> check labels with this tool -> train the model
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from .boat_utils.config import cfg

label_json_str = """{
      "label": "boat",
      "points": [
        [
          0,
          0
        ],
        [
          0,
          0
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {}
    }
    """


def get_inputs():
    folder = input("Enter img folder: ")

    labels = input("Enter labels root folder: ")
    return folder, labels


def build_ui():

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    plt.axis("off")
    title = plt.title("Select what you see in the center of the square.")


def all_possible_imgs(x, y):
    """
    return a list of tuples (row, col) that would contain the given x and y coords
    """
    stride = cfg["STRIDE"]
    row = y // stride - 1
    col = x // stride - 1
    options = []
    # NOTE: we do it like this to try to keep the 'best' subimages as highest priority
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            options.append((row + i, col + j))
    for i in [-2, 2]:
        for j in [0, 1, -1, -2, 2]:
            options.append((row + i, col + j))

    return options


class Reconcile:
    """Class to reconcile ML detections with labelled images."""

    def __init__(self, folder):
        self.folder = folder
        self.imgs = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if file.endswith(".png") and not file.endswith("done.png")
        ]
        self.img_idx = -1
        self.next_img = None
        self.display_next_image()

    def display_next_image(self):
        """Display next image in folder."""
        self.img_idx += 1
        if self.img_idx >= len(self.imgs) - 1:
            print("No more images to display.")
            return
        if self.next_img is not None:
            # display the image
            ax.imshow(self.next_img)
            plt.draw()
        else:
            img = Image.open(self.imgs[self.img_idx])
            img = np.array(img)
            ax.imshow(img)
            plt.draw()
        # get the next image
        img = Image.open(self.imgs[self.img_idx + 1])
        img = np.array(img)
        self.next_img = img

    def write_result(self, result):
        """Write result"""
        this_img = self.imgs[self.img_idx]
        this_img_name = os.path.basename(this_img)
        # want up to third underscore
        this_img_name = "_".join(this_img_name.split("_")[:3])
        txt_file = os.path.join(folder, this_img_name + ".txt")
        # get x and y from txt file has format: x, y
        with open(txt_file, "r") as f:
            x, y, label = f.read().split(", ")
            x = float(x)
            y = float(y)
            label = int(label)
        # update the labels if necessary
        if label == result:
            print("No change needed")
            os.rename(this_img, this_img[:-4] + "done.png")
            return
        this_img_name = "_".join(this_img_name.split("_")[:2])  # remove ID from name
        label_file = os.path.join(labels, this_img_name + ".json")
        data = json.loads(label_json_str)
        w = 5
        # update the json
        data["label"] = "boat" if result == 0 else "movingBoat"
        data["points"] = [[x - w, y - w], [x + w, y + w]]
        print(label_file)
        with open(label_file, "r+") as f:
            json_data = json.load(f)
            # check if there is already a boat label at this location (+- 5 pixels)
            updated = False
            for shape in json_data["shapes"]:
                x1, y1 = shape["points"][0]
                x2, y2 = shape["points"][1]
                if x1 - w <= x <= x2 + w and y1 - w <= y <= y2 + w:
                    # update this label
                    if result == -1:
                        json_data["shapes"].remove(shape)
                        print("Removing label")
                    else:
                        shape["label"] = data["label"]
                        print("Updating label")
                    updated = True
            # if we didn't find an existing label, add this one
            if not updated:
                json_data["shapes"].append(data)
                print("Adding new label")
            # write the updated json
            f.seek(0)
            json.dump(json_data, f, indent=4)
            f.truncate()
        # update the name of the image to end with 'done.png'
        os.rename(this_img, this_img[:-4] + "done.png")

    def is_boat(self, event):
        """Callback function for "Boat" button."""
        self.write_result(0)
        self.display_next_image()

    def is_moving_boat(self, event):
        """Callback function for "Moving Boat" button."""
        self.write_result(1)
        self.display_next_image()

    def is_neither(self, event):
        """Callback function for "Neither" button."""
        self.write_result(-1)
        self.display_next_image()

if __name__ == "__main__":
    folder, labels = get_inputs()
    build_ui()

    callback = Reconcile(folder)
    # build the buttons
    boat_btn_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
    boat_btn = Button(boat_btn_ax, "Boat")
    boat_btn.on_clicked(callback.is_boat)

    moving_boat_btn_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    moving_boat_btn = Button(moving_boat_btn_ax, "Moving Boat")
    moving_boat_btn.on_clicked(callback.is_moving_boat)

    neither_btn_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
    neither_btn = Button(neither_btn_ax, "Neither")
    neither_btn.on_clicked(callback.is_neither)

    plt.show()
