"""
Given a heap of polygons, create a heatmap of the density of seeing each area.

1. Bounding box of all the polygons
    - This is then the area for the 'heatmap'
    - Can alternatively just use a big polygon that covers the area
2. Represent the bounding box by a grid of pixels 
    - This so that the heatmap can be represented by a 2D array
3. For each polygon:
    - 'paint' the polygon onto the heatmap by adding 1 to each pixel that is
        covered by the polygon

"""

import json
import math
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr

from .area_coverage import polygons_to_32756
from .image_cutting_support import coord2latlong

gdal.UseExceptions()

def get_bbox(polygons):
    """
    Get the bounding box of a list of polygons.
    :param polygons: List of polygons, converted to 32756 global coords
    """
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for poly in polygons:
        env = poly.GetEnvelope()
        x_min = min(x_min, env[0])
        x_max = max(x_max, env[1])
        y_min = min(y_min, env[2])
        y_max = max(y_max, env[3])
    return x_min, x_max, y_min, y_max

def get_polygons_from_folder(folder, name=None):
    polygons = []
    for root, _, files in os.walk(folder): 
        for file in files:
            if name is None:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        polygons.extend(polygons_to_32756(json.load(f)))
            else:
                if name in file:
                    with open(os.path.join(root, file), "r") as f:
                        polygons.extend(polygons_to_32756(json.load(f)))
    return polygons

def get_polygons_from_file(csv_path):
    df = pd.read_csv(csv_path)
    if "polygon" not in df.columns:
        raise ValueError("No column named 'polygon' in file")
    polygons = []
    for poly in df["polygon"]:
        polygons.append(ogr.CreateGeometryFromJson(poly))
    return polygons

def create_grid(x_min, x_max, y_min, y_max, size=1000):
    """
    Create's a grid of pixels that covers the area of the bounding box.
    """
    # make the grid using (size x size)m^2 sized pixels
    x_range = x_max - x_min
    y_range = y_max - y_min
    cols = math.ceil(x_range / size) 
    rows = math.ceil(y_range / size)
    # make the grid
    grid = np.zeros((cols, rows))
    print(grid.shape)
    x_step = x_range / cols
    y_step = y_range / rows
    return grid, x_step, y_step

def paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly):
    """
    Populate the grid as per whether each pixel's center is covered by the polygon.
    """
    for x in np.arange(x_min, x_max, x_step):
        for y in np.arange(y_min, y_max, y_step):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(x + x_step / 2, y + y_step / 2)
            if point is not None and poly.Contains(point):
                grid[int((x - x_min) / x_step), int((y - y_min) / y_step)] += 1

def export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename="heatmap.tif", geojson=True):
    """
    Export the grid as a GEOTIFF.
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = grid.shape
    new_raster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    # set origin and pixel size
    origin_x = x_min
    origin_y = y_max
    new_raster.SetGeoTransform((origin_x, x_step, 0, origin_y, 0, -y_step))
    # write data
    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(grid)
    band.SetScale(1)
    # set projection
    new_raster_srs = osr.SpatialReference()
    new_raster_srs.ImportFromEPSG(32756)
    new_raster.SetProjection(new_raster_srs.ExportToWkt())
    # close raster file
    band.FlushCache()
    if geojson:
        # save GeoJSON file where each feature has either an id field or some identifying value in properties
        # where each feature is a square in the grid
        grid_to_geojson(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename=filename.replace(".tif", ".geojson"))

def grid_to_geojson(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename="heatmap.geojson"):
    """
    Convert the grid to a geojson file.
    """
    features = []
    # transform grid into coordinates
    x_min, y_min = coord2latlong(x_min, y_min)
    x_max, y_max = coord2latlong(x_max, y_max)
    x_step = (x_max - x_min) / grid.shape[0]
    y_step = (y_max - y_min) / grid.shape[1]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x = x_min + i * x_step
            y = y_min + j * y_step
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [x, y],
                            [x + x_step, y],
                            [x + x_step, y + y_step],
                            [x, y + y_step],
                            [x, y]
                        ]
                    ]
                },
                "properties": {
                    "id": f"{i}_{j}",
                    "value": grid[i, j]
                }
            })
    with open(filename, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": features
        }, f)


def polygon_from_tif(tif):
    """
    Get the coordinates of a polygon from a tif file.
    """
    ds = gdal.Open(tif)
    # polygonize the raster
    srcband = ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    dst_layername = "polygonized"
    drv = ogr.GetDriverByName("Memory")
    dst_ds = drv.CreateDataSource(dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    gdal.Polygonize(srcband, None, dst_layer, -1, [], callback=None)
    # get the largest polygon
    max_area = 0
    max_poly_coords = None
    for feature in dst_layer:
        poly = feature.GetGeometryRef()
        if poly.GetArea() > max_area:
            max_area = poly.GetArea()
            for ring in poly:
                max_poly_coords = ring.GetPoints()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*zip(*max_poly_coords))
    plt.show()

def add_to_heatmap(heatmap, polygons):
    """
    Add polygons to the heatmap. Does this by creating a new raster, 
    then adding the polygons to the raster, and then adding the rasters
    together.
    @param heatmap: Path to raster file e.g. 'heatmap.tif'
    @param polygons: List of polygons
    """
    x_min, x_max, y_min, y_max = get_bbox(polygons)
    grid, x_step, y_step = create_grid(x_min, x_max, y_min, y_max)
    for poly in polygons:
        paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly)
    grid = np.rot90(grid)
    export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename="temp.tif")
    # add the rasters together
    ds1 = gdal.Open(heatmap, gdal.GA_Update)
    ds2 = gdal.Open("temp.tif")
    band1 = ds1.GetRasterBand(1)
    band2 = ds2.GetRasterBand(1)
    # add the rasters together
    band1.WriteArray(band1.ReadAsArray() + band2.ReadAsArray())
    band1.FlushCache()
    # remove the temp file
    os.remove("temp.tif")

def create_heatmap_from_polygons(polygons, outdir):
    if len(polygons) == 0:
        raise ValueError("No polygons found in folder")
    # Get the bounding box
    x_min, x_max, y_min, y_max = get_bbox(polygons)
    print(x_min, x_max, y_min, y_max)
    # Get the grid
    grid, x_step, y_step = create_grid(x_min, x_max, y_min, y_max, size=1000)
    # Paint the polygons onto the grid
    for poly in polygons:
        paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly)
    plt.imshow(grid)
    plt.show()
    # Export the grid
    filename = os.path.join(outdir, "heatmap.tif")
    export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename=filename)


if __name__ == "__main__":
    # Load the polygons
    folder = input("Enter the folder with the polygons: ")
    polygons = get_polygons_from_folder(folder, name="composite_metadata.json") 
    create_heatmap_from_polygons(polygons, os.getcwd())

