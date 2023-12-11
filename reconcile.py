"""
Simple GUI used to reconcile issues with ML detections for labelled images.
1. Displays an image, and three buttons: "Boat", "Moving Boat", "Neither"
2. User clicks the appropriate button, and the result is saved to a CSV file, and that image's labels updated.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

folder = input("Enter img folder: ")

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
plt.axis('off')

def all_possible_imgs(x, y):
    """
    return a list of tuples (row, col) that would contain the given x and y coords
    """
    row = y // 104 - 1
    col = x // 104 - 1
    options = []
    # NOTE: we do it like this to try to keep the 'best' subimages as highest priority
    for i in [0,-1, 1]:
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
        self.imgs = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        self.img_idx = -1
        self.results = self.load_results()
        self.display_next_image()

    def load_results(self):
        path = os.path.join(self.folder, 'results.csv')
        if os.path.exists(path):
            self.results = pd.read_csv(path)
        else:
            self.results = pd.DataFrame(columns=['img', 'result'])
        return self.results

    def display_next_image(self):
        """Display next image in folder."""
        self.img_idx += 1
        img = Image.open(self.imgs[self.img_idx])
        img = np.array(img)
        ax.imshow(img)
        existing_result = self.results[self.results['img'] == self.imgs[self.img_idx]]
        if len(existing_result) > 0:
            self.result = existing_result['result'].iloc[0]
            ax.set_title(f"Charlie Thinks: {self.result}")
        plt.draw()

    def write_result(self, result):
        """Write result"""
        self.results = pd.concat([self.results, pd.DataFrame([{'img': self.imgs[self.img_idx], 'result': result}])], ignore_index=True)
        self.results.to_csv(os.path.join(self.folder, 'results.csv'), index=False)

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

callback = Reconcile(folder)
# build the buttons
boat_btn_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
boat_btn = Button(boat_btn_ax, 'Boat')
boat_btn.on_clicked(callback.is_boat)

moving_boat_btn_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
moving_boat_btn = Button(moving_boat_btn_ax, 'Moving Boat')
moving_boat_btn.on_clicked(callback.is_moving_boat)

neither_btn_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
neither_btn = Button(neither_btn_ax, 'Neither')
neither_btn.on_clicked(callback.is_neither)

plt.show()
