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

import numpy as np
import os
import json
from osgeo import ogr, gdal, osr
import matplotlib.pyplot as plt
from utils.area_coverage import polygon_to_32756

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
                        polygons.append(polygon_to_32756(json.load(f)))
            else:
                if name in file:
                    with open(os.path.join(root, file), "r") as f:
                        polygons.append(polygon_to_32756(json.load(f)))
    return polygons

def create_grid(x_min, x_max, y_min, y_max):
    # make shortest side 100px, and scale the other
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < y_range:
        cols = 1000
        rows = int(cols * y_range / x_range)
    else:
        rows = 1000
        cols = int(rows * x_range / y_range)
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

def export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step):
    """
    Export the grid as a GEOTIFF.
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = grid.shape
    print(rows, cols)
    new_raster = driver.Create("heatmap.tif", cols, rows, 1, gdal.GDT_Float32)
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

if __name__ == "__main__":
    # Load the polygons
    polygons = get_polygons_from_folder("/Users/charlieturner/Documents/CountingBoats/TestMoreton/RawImgs", name="composite_metadata.json") 
    if len(polygons) == 0:
        raise ValueError("No polygons found in folder")
    # Get the bounding box
    x_min, x_max, y_min, y_max = get_bbox(polygons)
    print(x_min, x_max, y_min, y_max)
    # Get the grid
    grid, x_step, y_step = create_grid(x_min, x_max, y_min, y_max)
    # Paint the polygons onto the grid
    for poly in polygons:
        paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly)
    # Need to rotate the grid
    grid = np.rot90(grid)
    # Plot the grid
    plt.imshow(grid)
    plt.show()
    # Export the grid
    export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step)
