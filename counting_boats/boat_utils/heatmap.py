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

from .spatial_helpers import polygon_latlong2crs
from .image_cutting_support import coord2latlong
from tqdm import tqdm

gdal.UseExceptions()


def get_bbox(polygons: list[ogr.Geometry | None]):
    """
    Get the bounding box of a list of polygons.

    Args:

        polygons: List of polygons, must be in global coordinate system (not latlong)

    Returns:

        x_min: Minimum x value

        x_max: Maximum x value

        y_min: Minimum y value

        y_max: Maximum y value
    """
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for poly in polygons:
        if poly is None:
            continue
        env = poly.GetEnvelope()
        x_min = min(x_min, env[0])
        x_max = max(x_max, env[1])
        y_min = min(y_min, env[2])
        y_max = max(y_max, env[3])
    return x_min, x_max, y_min, y_max


def get_polygons_from_folder(folder, name=None):
    """
    Get the polygons from a folder of json files.

    Args:

        folder: Path to folder

        name: Optional, name of file to search for


    Returns:

        List of polygons in global coordinate system EPSG:32756
    """
    polygons = []
    for root, _, files in os.walk(folder):
        for file in files:
            if name is None:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        polygons.extend(polygon_latlong2crs(json.load(f)))
            else:
                if name in file:
                    with open(os.path.join(root, file), "r") as f:
                        polygons.extend(polygon_latlong2crs(json.load(f)))
    return polygons


def get_polygons_from_file(csv_path, group=None):
    """
    Get the polygons from a csv file with 'polygon' column.
    Group optionally is a group of aois to use

    Args:

        csv_path: Path to csv file
        group: Optional, list of aois to use

    Returns:

        List of polygons in global coordinate system EPSG:32756

    """
    df = pd.read_csv(csv_path)
    return get_polygons_from_df(df, group)


def get_polygons_from_df(df, group=None) -> list[ogr.Geometry | None]:
    """
    Get the polygons from a dataframe with 'polygon' column.
    Group optionally is a group of aois to use

    Args:

        df: Dataframe with 'polygon' column
        group: Optional, list of aois to use

    Returns:

        List of polygons in global coordinate system EPSG:32756
    """
    # filter where df[aoi] is in group
    if group is not None:
        df = df[df["aoi"].isin(group)]
    if "polygon" not in df.columns:
        raise ValueError("No column named 'polygon' in file")
    polygons = []
    for poly in df["polygon"]:
        polygons.extend(polygon_latlong2crs(poly))
    return polygons


def create_grid(
    x_min, x_max, y_min, y_max, size=1000
) -> tuple[np.ndarray, float, float]:
    """
    Create's a grid of pixels that covers the area of the bounding box.

    Args:

        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        size: Size of each pixel in meters

    Returns:

        A tuple of:
            grid: 2D numpy array of zeros
            x_step: Size of each pixel in x direction
            y_step: Size of each pixel in y direction
    """
    # make the grid using (size x size)m^2 sized pixels
    x_range = x_max - x_min  # in meters
    y_range = y_max - y_min  # in meters
    cols = math.ceil(x_range / size)
    rows = math.ceil(y_range / size)
    # make the grid
    grid = np.zeros((cols, rows))
    print(grid.shape)
    x_step = x_range / cols  # in meters
    y_step = y_range / rows  # in meters
    return grid, x_step, y_step


def paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly):
    """
    Populate the given grid as per whether each pixel's center is covered by the polygon.

    Args:

        grid: 2D numpy array of zeros
        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        x_step: Size of each pixel in x direction
        y_step: Size of each pixel in y direction
        poly: Polygon to paint onto the grid

    Returns:

        None
    """

    cols = range(grid.shape[0])
    rows = range(grid.shape[1])

    for row, col in [(row, col) for row in rows for col in cols]:
        x = x_min + col * x_step
        y = y_min + row * y_step
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x + x_step / 2, y + y_step / 2)
        if point is not None and poly.Contains(point):
            grid[col, row] += 1


def export_data(
    grid,
    x_min,
    x_max,
    y_min,
    y_max,
    x_step,
    y_step,
    filename="heatmap.tif",
    geojson=True,
) -> None:
    """
    Export the grid as a GEOTIFF.

    Args:

        grid: 2D numpy array of zeros
        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        x_step: Size of each pixel in x direction
        y_step: Size of each pixel in y direction
        filename: Name of file to save
        geojson: Whether to save a geojson file as well

    Returns:

        None
    """
    driver = gdal.GetDriverByName("GTiff")
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
    print(f"Saved heatmap to {filename}")
    if geojson:
        print("Creating GeoJSON file")
        # save GeoJSON file where each feature has either an id field or some identifying value in properties
        # where each feature is a square in the grid
        grid_to_geojson(
            grid,
            x_min,
            x_max,
            y_min,
            y_max,
            x_step,
            y_step,
            filename=filename.replace(".tif", ".geojson"),
        )


def grid_to_geojson(
    grid, x_min, x_max, y_min, y_max, x_step, y_step, filename="heatmap.geojson"
):
    """
    Convert the grid to a geojson file, where each feature is a square in the grid and has:
    - id: The id of the square
    - value: The value of the square (number of times this square was painted on)

    Args:
    
        grid: 2D numpy array of zeros
        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        x_step: Size of each pixel in x direction
        y_step: Size of each pixel in y direction
        filename: Name of file to save

    Returns:

        None
    """
    features = []
    # rotate the grid back
    grid = np.rot90(grid, k=3)
    # transform grid into coordinates
    x_min, y_min = coord2latlong(x_min, y_min)
    x_max, y_max = coord2latlong(x_max, y_max)
    x_step = (x_max - x_min) / grid.shape[0]
    y_step = (y_max - y_min) / grid.shape[1]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x = x_min + i * x_step
            y = y_min + j * y_step
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y],
                                [x + x_step, y],
                                [x + x_step, y + y_step],
                                [x, y + y_step],
                                [x, y],
                            ]
                        ],
                    },
                    "properties": {"id": f"{i}_{j}", "value": grid[i, j]},
                }
            )
    with open(filename, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def polygon_from_tif(tif) -> list:
    """
    Get the coordinates of a polygon from a tif file.

    Args:

        tif: Path to tif file

    Returns:

        List of polygons as ogr polygons in the source coordinate system of the tif
    """
    ds = gdal.Open(tif)
    print("Getting polygons from: ", tif, end="")
    # polygonize the raster
    #   consider red, green, and blue bands as one
    # Get the source band
    polygons = []
    src_band = ds.GetRasterBand(1)

    src = osr.SpatialReference()
    src.ImportFromWkt(ds.GetProjection())
    driver = ogr.GetDriverByName("Memory")
    dst_ds = driver.CreateDataSource("temp")
    out_layer = dst_ds.CreateLayer("temp", srs=src)

    if out_layer is None:
        print("Failed to create layer.")
        exit(1)

    # Call the Polygonize function
    gdal.Polygonize(src_band, None, out_layer, -1, [], None, None)
    # Get the polygons
    for feature in out_layer:
        geom = feature.GetGeometryRef()
        polygons.append(geom)

    print(len(polygons))
    exit()
    return polygons


def add_to_heatmap(heatmap: str, polygons: list):
    """
    Add polygons to the heatmap. Does this by creating a new raster,
    then adding the polygons to the raster, and then adding the rasters
    together.

    Args:

        heatmap: Path to raster file e.g. 'heatmap.tif'
        polygons: List of polygons as ogr polygons

    Returns:

        None
    """
    x_min, x_max, y_min, y_max = get_bbox(polygons)
    grid, x_step, y_step = create_grid(x_min, x_max, y_min, y_max)
    for poly in polygons:
        paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly)
    grid = np.rot90(grid)  # Grid has to be rotated for some reason
    export_data(
        grid,
        x_min,
        x_max,
        y_min,
        y_max,
        x_step,
        y_step,
        filename="temp.tif",
        geojson=False,
    )
    # add the rasters together
    ds1 = gdal.Open(heatmap, gdal.GA_Update)
    ds2 = gdal.Open("temp.tif")
    band1 = ds1.GetRasterBand(1)
    band2 = ds2.GetRasterBand(1)
    # add the rasters together
    array = band1.ReadAsArray() + band2.ReadAsArray()
    ds1 = None
    ds2 = None
    # remove the temp file
    os.remove("temp.tif")
    os.remove(heatmap)
    # export the new raster
    export_data(array, x_min, x_max, y_min, y_max, x_step, y_step, filename=heatmap)


def create_heatmap_from_polygons(
    polygons: list[ogr.Geometry | None], save_file="heatmap.tif", show=False, size=1000
):
    """
    Create and save a heatmap from a list of polygons.
    Optionally show the heatmap.

    Args:

        polygons: List of polygons
        save_file: Name of file to save
        show: Whether to show the heatmap

    Returns:

        None
    """
    if len(polygons) == 0:
        print("No polygons to create heatmap from")
        return
    # Get the bounding box
    x_min, x_max, y_min, y_max = get_bbox(polygons)
    # Get the grid
    grid, x_step, y_step = create_grid(x_min, x_max, y_min, y_max, size=size)
    # Paint the polygons onto the grid
    print("Painting polygons onto grid")
    for poly in polygons:
        # print the polygon's coordinates
        paint_grid(grid, x_min, x_max, y_min, y_max, x_step, y_step, poly)
    if show:
        plt.imshow(grid)
        plt.show()
    grid = np.rot90(grid)  # Grid has to be rotated for some reason for the tif file
    # Export the grid
    print("Exporting heatmap")
    export_data(grid, x_min, x_max, y_min, y_max, x_step, y_step, filename=save_file)


if __name__ == "__main__":
    # Load the polygons
    folder = input("Enter coverage file with 'polygon' column:")
    polygons = get_polygons_from_file(folder)
    create_heatmap_from_polygons(polygons, os.path.join(os.getcwd(), "coverage.tif"))
