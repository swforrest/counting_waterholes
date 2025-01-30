import os
print("Python works with Miniconda!")
import shutil

import numpy as np
import pandas as pd
import scipy
import random

from counting_boats.boat_utils.classifier import cluster, process_clusters, read_classifications, pixel2latlong
from counting_boats.boat_utils.config import cfg
from counting_boats.boat_utils import image_cutting_support as ics
from counting_boats.boat_utils import heatmap as hm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import json 

#all good manages to run those imports. 
#Had to specify the direction of the py scripts in the Boat_utils as it wasn't found alone. 

def prepare(run_folder, config):
    """
    Given a folder, find all the tif files are create a png for each one.
    Also rename tif files if required.

    Args:

        run_folder (str): The folder to prepare
        config (dict): The configuration dictionary

    Returns:

        None
    """
    img_folder = config["raw_images"]  # folder with the tif files
    save_folder = os.path.join(config["path"], config["pngs"])
    os.makedirs(save_folder, exist_ok=True)
    for root, _, files in os.walk(img_folder):
        for file in files:
            if file == "composite.tif":
                # find the json file:
                date_file = [f for f in files if f.endswith("xml")][0]
                date = date_file.split("_")[0]
                aoi = os.path.basename(root).split("_")[-2].split("/")[-1]
                print(root, aoi)
                name = f"{date}_{aoi}.tif"
                print(name)
                os.rename(os.path.join(root, file), os.path.join(root, name))
                # want to create a png for this
                new_name = os.path.join(save_folder, f"{name.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    ics.create_padded_png(
                        root,
                        save_folder,
                        name,
                        tile_size=config["img_size"],
                        stride=config["img_stride"],
                    )
            # if the file is a tif and first part is a date, don't need to rename
            elif file.endswith("tif") and file.split("_")[0].isdigit():
                # check if we have already created a png for this
                new_name = os.path.join(save_folder, f"{file.split('.')[0]}.png")
                if not os.path.exists(new_name):
                    print(f"Creating png for {file}")
                    ics.create_padded_png(
                        root,
                        save_folder,
                        file,
                        tile_size=config["img_size"],
                        stride=config["img_stride"],
                    )

prepare(r"C:\Users\adria\OneDrive - Queensland University of Technology\FirstByte Waterholes WD\counting_waterholes\images\RawImages\waterhole_test_20250116_psscene_analytic_8b_sr_udm2.zip", 
        r"C:\Users\adria\OneDrive - Queensland University of Technology\FirstByte Waterholes WD\counting_waterholes")

