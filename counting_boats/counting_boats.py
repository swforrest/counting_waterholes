import os
import utils.planet_utils as pu
import utils.classifier as cl
from config import cfg
from CountTheBoats import archive

"""
Ideally this becomes the one file to rule them all, currently CountTheBoats is the main file
But is composed of lots of functions that could be in utils
"""

def classify_zips(folder):
    """
    Run the pipeline on all the zips in the folder
    Assuming zips come from Planet, and so are named in the format:
    AOI_YYYYMMDD.zip
    """
    zips = [f for f in os.listdir(folder) if f.endswith('.zip')]
    for z in zips:
        if not os.path.exists(os.path.join(folder, z.split('.')[0])):
            pu.extract_zip(os.path.join(folder, z))
    # classify tifs
    cl.main()
    # archive the tifs
    coverage_path = os.path.join(cfg["proj_root"], "outputs", "coverage.csv")
    archive(folder, "coverage.csv")


if __name__ == "__main__":
    classify_zips("/Users/charlieturner/Documents/CountingBoats/images/downloads")

