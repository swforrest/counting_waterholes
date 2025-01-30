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

