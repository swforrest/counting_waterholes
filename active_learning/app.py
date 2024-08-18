from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
import os
import random
import datetime
from PIL import Image, ImageDraw

app = Flask(__name__)

# Path to the directory containing images
IMAGE_DIRECTORY = "C:\\ML_Software\\active_learning\\data\\segmented_images"
# Path to the log file
LOG_FILE = "click_log.txt"


def read_classifications(remove_annotated=True):
    """Walk the classifications dir and read all of them. Keep track of
    the path (as this points to the image as well) and sort by confidence ascending.
    """
    classifications = []
    for root, _, files in os.walk("data/classifications"):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file)) as f:
                    boats = [line for line in f.readlines()]
                    if len(boats) == 0:
                        continue
                    for b in boats:
                        boat = [float(a) for a in b.split()]
                        boat[0] = int(boat[0])
                        classifications.append(
                            {
                                "class": boat[0],
                                "confidence": boat[5],
                                "x": boat[1],
                                "y": boat[2],
                                "im_path": os.path.normpath(
                                    os.path.join(root, file)
                                    .replace("classifications", "segmented_images")
                                    .replace(".txt", ".png")
                                ),
                            }
                        )
    # sort by confidence ascending
    classifications.sort(key=lambda x: x["confidence"])
    # remove any images which are present in click_log.txt
    if remove_annotated and os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            clicked_images = [line.split() for line in f]
            clicked_images_x_y = [(c[5], c[6], c[7]) for c in clicked_images]
        classifications = [
            c
            for c in classifications
            if (c["im_path"], str(c["x"]), str(c["y"])) not in clicked_images_x_y
        ]
    return classifications


def get_classification(image_path, x, y):
    """Return the classification for a given image."""
    classifications = read_classifications(remove_annotated=False)
    for c in classifications:
        if c["im_path"] == image_path and str(c["x"]) == x and str(c["y"]) == y:
            return c
    return None


import numpy as np

image_path = None


def get_next_image():
    """Return the next image to be annotated, and its confidence in the network."""

    classifications = read_classifications(remove_annotated=True)

    idx = None
    global image_path
    if image_path is not None:
        # serve an image with that path:
        for i, c in enumerate(classifications):
            if c["im_path"] == image_path:
                idx = i
                break
        # if the image is not in the list, fall down to the random image selection
    if idx is None:
        # Serve an image with a probability proportional to the inverse of the confidence
        # I.e images with low confidence are more likely to be served
        confidences = [c["confidence"] for c in classifications]
        # get inverses
        inv_confidences = [1.0 / c for c in confidences]
        cumsum = [sum(inv_confidences[: i + 1]) for i in range(len(inv_confidences))]
        # get a random number from 0 to the sum of all (inverse) confidences
        r = random.uniform(0, cumsum[-1])
        # find the index of the first element greater than r
        idx = np.searchsorted(cumsum, r)
        # return the image with the corresponding index
        image_path = classifications[idx]["im_path"]

    if classifications:
        image = Image.open(classifications[idx]["im_path"])
        draw = ImageDraw.Draw(image)
        # draw a rectangle around the boat
        x = classifications[idx]["x"] * image.width
        y = classifications[idx]["y"] * image.height
        draw.rectangle([x - 10, y - 10, x + 10, y + 10], outline="red", width=2)
        image.save("static/image.png")
        with open("static/image.txt", "w") as f:
            f.write(
                str(classifications[idx]["x"]) + " " + str(classifications[idx]["y"])
            )
        return classifications[idx]["im_path"], classifications[idx]["confidence"]
    return None, None


@app.route("/")
def index():
    """Serve a random image with buttons."""
    image, conf = get_next_image()
    return render_template("index.html", image=image, conf=conf)


@app.route("/click/<button>/<image>")
def click(button, image):
    """Log the button click and serve a new random image."""
    with open("static/image.txt", "r") as f:
        x, y = f.read().split()
    classification = get_classification(image, x, y)
    with open(LOG_FILE, "a+") as f:
        f.write(
            f"{datetime.datetime.now()}: {button} clicked on {image} {classification['x']} {classification['y']} (detected:{classification['class']},{classification['confidence']})\n"
        )
    if button == "unsure":
        # If we are unsure about anything in the image, remove this image from the list
        global image_path
        image_path = None
    return redirect(url_for("index"))


@app.route("/images/<filename>")
def image(filename):
    """Serve the image file at static/image.png."""
    # directory = os.path.dirname(filename)
    # return send_from_directory(directory, os.path.basename(filename))
    return send_from_directory("static", "image.png")


if __name__ == "__main__":
    app.run(debug=True)
