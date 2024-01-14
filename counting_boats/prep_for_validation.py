import os
import utils.image_cutting_support as ics

folder = "/Users/charlieturner/Documents/CountingBoats/TestGBR"

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        # get all dir names in "SegmentedImages" folder (recursively)
        dirs = [x[0].split("/")[-1] for x in os.walk(os.path.join(folder, "SegmentedImages"))]
        if filename[:-5] in dirs:
            continue
        # a label file
        # find the corresponding image file
        img_file = os.path.join(folder, filename[:-5] + ".png")
        if os.path.isfile(img_file):
            ics.segment_image(img_file, os.path.join(folder, filename), 416, 104, remove_empty=0)

