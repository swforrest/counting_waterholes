import os
import imageCuttingSupport as ics

folder = "/Users/charlieturner/Documents/CountingBoats/TestGBR"

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        # a label file
        # find the corresponding image file
        img_file = os.path.join(folder, filename[:-5] + ".png")
        if os.path.isfile(img_file):
            # ensure we aren't deleting parts of the image
            ics.segment_image(img_file, os.path.join(folder, filename), 416, 104, remove_empty=0)

