# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr, gdal
import json
from src.utils.imageCuttingSupport import latlong2coord

polygon = "./Polygons/peel.json"
tif = "./Rawimages/processed/20231114_230520_95_2415_3B_AnalyticMS.tif"


# Area of polygon:
with open(polygon) as f:
    # read as json
    geoJSON = json.load(f)

# convert from lat long to EPSG:32756
for i, val in enumerate(geoJSON['coordinates'][0]):
    lat, long = val
    x, y = latlong2coord(lat, long)
    geoJSON['coordinates'][0][i] = [x, y]

poly = ogr.CreateGeometryFromJson(str(geoJSON))
area = poly.Area()
print("Area of polygon: ", area)

# Same idea with the tif. Get the area of the tif
import rasterio
import cv2
with rasterio.open(tif) as src:
    array = src.read()
    meta = gdal.Info(tif)
    coords = meta.split("Corner Coordinates")[1].split("\n")[1:5]
    # Each looks like:
    # 'Upper Left  (  523650.000, 6961995.000) (153d14\'21.71"E, 27d27\'55.37"S)'
    # we want just the numbers in the first brackets
    coords = [x.split("(")[1].split(")")[0].split(",") for x in coords]
    # upper left, lower left, upper right, lower right
    coords = [(float(x), float(y)) for x, y in coords]
    real_w = coords[2][0] - coords[0][0]
    real_h = coords[0][1] - coords[1][1]
    # flatten array as average of all bands
    array = np.mean(array, axis=0)
    array[array > 0] = 1
    # get the area of the tif (taking into account the real world size)
    tif_area = np.sum(array) * real_w * real_h / array.shape[0] / array.shape[1]
    print("Area of tif: ", tif_area)



# plot the tif
print("Percentage of area covered: ", f"{round(tif_area/area * 100)}%")
print("Disclaimer: Rough estimate with multiple conversions, values over 100% are possible")











