"""
AF: runing parts of the boat_detection workflow to prepare tif images to annotate them as PNGs with labelme.

Script to run with pythonin order to subselect some function in different .py files in the repo. 

want to a. extract zip from our downloaded image. And b. prepare the image to cut it as a png. 

WIP

"""
#Script to run with pythonin order to subselect some function in different .py files in the repo. 
#want to a. extract zip from our downloaded image. And b. prepare the image to cut it as a png. 

import sys
import os
import yaml

# Add the project root to sys.path (adjust as needed)
sys.path.append(os.path.abspath(r"C:\Users\adria\OneDrive - AdrianoFossati\Documents\MASTER Australia\RA\Waterholes_project\counting_waterholes"))


from counting_boats.boat_utils.planet_utils import extract_zip
from counting_boats.boat_utils.testing import prepare
from counting_boats.boat_utils.testing import segment


# Define the path to your zip file
zip_path = r"C:\Users\adria\OneDrive - AdrianoFossati\Documents\MASTER Australia\RA\Waterholes_project\counting_waterholes\images\raw_images\NT_050225.zip"


# Run extraction
#extract_zip(zip_path)


#cfg config:
with open("config_train.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg["tif_dir"] = cfg.get(
        "tif_dir", os.path.join(cfg["proj_root"], "images", "RawImages")
    )  # This is generated so not included in the config file

#Run preparation of the tif files into png and renamed the tif. 
prepare(r"C:\Users\adria\OneDrive - AdrianoFossati\Documents\MASTER Australia\RA\Waterholes_project\counting_waterholes\images\RawImages", cfg)
 
#segment(r"C:\Users\adria\OneDrive - AdrianoFossati\Documents\MASTER Australia\RA\Waterholes_project\counting_waterholes\images", cfg)


print('hello')

