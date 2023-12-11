# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr
import json
from imageCuttingSupport import latlong2coord

polygon = "./Polygons/peel.json"
tif = "./Rawimages/processed/20231114_230520_95_2415_3B_AnalyticMS.tif"


# Area of polygon:
with open(polygon) as f:
    # read as json
    geoJSON = json.load(f)
    print(geoJSON)

# convert from lat long to EPSG:32756
for i, val in enumerate(geoJSON['coordinates'][0]):
    lat, long = val
    x, y = latlong2coord(lat, long)
    geoJSON['coordinates'][0][i] = [x, y]

poly = ogr.CreateGeometryFromJson(str(geoJSON))
area = poly.GetArea()
print("Area of polygon: ", area)

# Same idea with the tif. Get the area of the tif
import rasterio
with rasterio.open(tif) as src:
    array = src.read()
    meta = src.meta
    tif_area = np.abs(meta['width'] * meta['height'] * meta['transform'][0] * meta['transform'][4])
    print("Area of tif: ", tif_area)
    # get the outline of the tif as a polygon

print("Percentage of area covered: ", tif_area/area * 100)











