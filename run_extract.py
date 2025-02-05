#Script to run with pythonin order to subselect some function in different .py files in the repo. 
#want to a. extract zip from our downloaded image. And b. prepare the image to cut it as a png. 

import sys
import os

# Add the project root to sys.path (adjust as needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from counting_boats.boats_utils.planet_utils import extract_zip

# Define the path to your zip file
zip_path = r"C:\Users\adria\OneDrive - AdrianoFossati\Documents\MASTER Australia\RA\Waterholes_project\counting_waterholes\images\raw_images\waterhole_test_20250116_psscene_analytic_8b_sr_udm2.zip"

# Run extraction
extract_zip(zip_path)
