from multiprocessing import Pool
from functools import partial
import rasterio
import json
import os
import numpy as np
import pandas as pd
import sys
import pickle
from tqdm import tqdm
import pyproj
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))

from counting_boats.boat_utils.spatial_helpers import use_udm_2, polygon_latlong2crs
from counting_boats.boat_utils.image_cutting_support import latlong2coord, coord2latlong

DETECTION_CSV = "C:/ML_Software/All_Results/boat_detections.csv"
COVERAGE_CSV = "C:/ML_Software/All_Results/coverage.csv"
ORDERS_CSV = "C:/ML_Software/All_Results/orders.csv"
UDM_DIR = "D:/Results/UDM"
GRID_SIZE = 500  # Meters
RESULTS_DIR = "D:/Results/udm_results"


def udm_clear(udm_name):
    """
    Load the UDM clear band for an image or from cached .npz file.
    Return: a numpy array of the UDM clear band, the minx, miny, maxx, maxy of the image, and the resolution.
    """
    path = os.path.join(UDM_DIR, udm_name)
    npz_path = os.path.join(UDM_DIR, f"{udm_name}.npz")

    if os.path.exists(npz_path):
        # Load from .npz file
        try:
            data = np.load(npz_path)
            mask = data["mask"]
            minx = data["minx"].item()
            miny = data["miny"].item()
            maxx = data["maxx"].item()
            maxy = data["maxy"].item()
            resx = data["resx"].item()
            resy = data["resy"].item()
            return mask, minx, miny, maxx, maxy, resx, resy
        except Exception as e:
            print(f"Error loading cached UDM {udm_name}: {e}")
            # Proceed to reprocess if loading fails

    try:
        # Open the mask
        with rasterio.open(path) as udm:
            # Check if UDM1 or UDM2.1
            if not use_udm_2(path):
                # In band 8: bit 0 is not imaged, bit 1 is cloud.
                mask = udm.read(8) & 0b11 == 0
            else:
                # UDM2 has a clear band we can return directly
                mask = udm.read(1)

            # Get the extent of the image
            minx, miny, maxx, maxy = udm.bounds
            # Get the resolution of the image
            resx, resy = udm.res

        # Save the mask and metadata for future use in a single .npz file
        np.savez_compressed(
            npz_path,
            mask=mask,
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            resx=resx,
            resy=resy,
        )

        return mask, minx, miny, maxx, maxy, resx, resy
    except Exception as e:
        print(f"Error processing UDM {udm_name}: {e}")
        return None, None, None, None, None, None, None


def process_udm(
    udm, group_func, grid_minx, grid_miny, grid_maxx, grid_maxy, grid_rows, grid_cols
):
    """
    Process a single UDM file and return the coverage counts for its group.
    """
    output_file = os.path.join(RESULTS_DIR, f"{udm}_{GRID_SIZE}.pkl")
    if os.path.exists(output_file):
        # Already processed
        return

    try:
        key = group_func(udm)
        result = udm_clear(udm)
        if result[0] is None:
            return  # Skip if there was an error loading the UDM

        mask, udm_minx, udm_miny, udm_maxx, udm_maxy, resx, resy = result

        counts = {}

        # Get grid cell indices covering the UDM image
        gridx1, gridy1 = point_to_grid(
            np.array([udm_minx]),
            np.array([udm_maxy]),
            grid_minx,
            grid_miny,
            grid_rows,
            grid_cols,
        )
        gridx2, gridy2 = point_to_grid(
            np.array([udm_maxx]),
            np.array([udm_miny]),
            grid_minx,
            grid_miny,
            grid_rows,
            grid_cols,
        )

        # Ensure indices are within grid bounds
        gridx1 = max(gridx1[0], 0)
        gridy1 = max(gridy1[0], 0)
        gridx2 = min(gridx2[0], grid_cols - 1)
        gridy2 = min(gridy2[0], grid_rows - 1)

        # Iterate over the grid cells covered by the image
        for x in range(gridx1, gridx2 + 1):
            for y in range(gridy1, gridy2 + 1):
                # Get the bounds of the grid cell in meters
                x1 = grid_minx + x * GRID_SIZE
                y1 = grid_maxy - y * GRID_SIZE
                x2 = x1 + GRID_SIZE
                y2 = y1 - GRID_SIZE

                # Calculate mask pixel indices corresponding to the grid cell
                mask_x1 = max(int((x1 - udm_minx) / resx), 0)
                mask_x2 = min(int((x2 - udm_minx) / resx), mask.shape[1])
                mask_y1 = max(int((udm_maxy - y1) / resy), 0)
                mask_y2 = min(int((udm_maxy - y2) / resy), mask.shape[0])

                # Ensure indices are integers
                mask_x1 = int(mask_x1)
                mask_x2 = int(mask_x2)
                mask_y1 = int(mask_y1)
                mask_y2 = int(mask_y2)

                if mask_x1 >= mask_x2 or mask_y1 >= mask_y2:
                    # No overlap between grid cell and mask
                    continue

                # Sum of mask within these bounds
                coverage = mask[mask_y1:mask_y2, mask_x1:mask_x2].sum()
                # Total number of pixels in the grid cell area in the mask
                total_pixels = (mask_y2 - mask_y1) * (mask_x2 - mask_x1)
                # If more than 50% of the grid cell is covered, consider it imaged
                if coverage > 0.5 * total_pixels:
                    counts[(x, y)] = counts.get((x, y), 0) + 1

        # Save counts to file
        with open(output_file, "wb") as f:
            pickle.dump((key, counts), f)
    except Exception as e:
        print(f"Error processing {udm}: {e}")


def generate_coverage_heatmaps(
    UDM_DIR,
    group_func,
    grid_minx,
    grid_miny,
    grid_maxx,
    grid_maxy,
    grid_rows,
    grid_cols,
):
    heatmaps = {}
    udm_files = os.listdir(UDM_DIR)

    # Filter out UDM files that have already been processed
    udm_files_to_process = [
        udm
        for udm in udm_files
        if udm.endswith(".tif")
        and not os.path.exists(os.path.join(RESULTS_DIR, f"{udm}_{GRID_SIZE}.pkl"))
    ]

    if udm_files_to_process:
        # Prepare arguments for multiprocessing
        process_func = partial(
            process_udm,
            group_func=group_func,
            grid_minx=grid_minx,
            grid_miny=grid_miny,
            grid_maxx=grid_maxx,
            grid_maxy=grid_maxy,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

        # Use a multiprocessing Pool to process UDM files in parallel
        with Pool() as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_func, udm_files_to_process),
                    total=len(udm_files_to_process),
                )
            )

    # Aggregate results from saved files
    result_files = [
        f for f in os.listdir(RESULTS_DIR) if f.endswith(f"{GRID_SIZE}.pkl")
    ]
    for result_file in tqdm(result_files, desc="Aggregating results"):
        with open(os.path.join(RESULTS_DIR, result_file), "rb") as f:
            key, counts = pickle.load(f)
            if key not in heatmaps:
                heatmaps[key] = counts
            else:
                # Merge counts
                for cell, count in counts.items():
                    heatmaps[key][cell] = heatmaps[key].get(cell, 0) + count

    return heatmaps


def point_to_grid(x, y, grid_minx, grid_miny, grid_rows, grid_cols):
    """Given a list of points in meters (32756),
    return the grid cells they belong to in image coordinates (origin top left)
    """
    gridx = (x - grid_minx) // GRID_SIZE
    gridy = grid_rows - ((y - grid_miny) // GRID_SIZE) - 1
    # Ensure indices are integers
    gridx = gridx.astype(int)
    gridy = gridy.astype(int)
    return gridx, gridy


def key_function(x):
    # Extract year from the UDM filename
    return int(x.split("_")[1][:4])


if __name__ == "__main__":
    # Extent should be the minimum and maximum possible extent, use the geojson for this
    geojson_file = "../data/polygons/moreton.geojson"
    min_lat, min_lon, max_lat, max_lon = 9999, 9999, -9999, -9999
    with open(geojson_file, "r") as f:
        geojson = json.load(f)
        for poly in geojson["coordinates"][0]:
            for coord in poly:
                if coord[0] < min_lon:
                    min_lon = coord[0]
                if coord[0] > max_lon:
                    max_lon = coord[0]
                if coord[1] < min_lat:
                    min_lat = coord[1]
                if coord[1] > max_lat:
                    max_lat = coord[1]

    # Convert lat long to meters (x, y)
    grid_minx, grid_miny = latlong2coord(min_lon, min_lat)
    grid_maxx, grid_maxy = latlong2coord(max_lon, max_lat)

    # Meters go left and up, so minx, miny is the bottom left corner
    MINLON, MINLAT = coord2latlong(grid_minx, grid_miny)
    MAXLON, MAXLAT = coord2latlong(grid_maxx, grid_maxy)

    # Round to nearest grid size
    # Floor the minimums
    grid_minx = int(grid_minx // GRID_SIZE) * GRID_SIZE
    grid_miny = int(grid_miny // GRID_SIZE) * GRID_SIZE

    # Ceil the maximums
    grid_maxx = int(grid_maxx // GRID_SIZE) * GRID_SIZE + GRID_SIZE
    grid_maxy = int(grid_maxy // GRID_SIZE) * GRID_SIZE + GRID_SIZE

    # Define the grid
    grid_rows = int((grid_maxy - grid_miny) // GRID_SIZE) + 1
    grid_cols = int((grid_maxx - grid_minx) // GRID_SIZE) + 1

    assert point_to_grid(
        np.array([grid_minx]),
        np.array([grid_maxy]),
        grid_minx,
        grid_miny,
        grid_rows,
        grid_cols,
    ) == (0, 0)
    assert point_to_grid(
        np.array([grid_maxx]),
        np.array([grid_miny]),
        grid_minx,
        grid_miny,
        grid_rows,
        grid_cols,
    ) == (grid_cols - 1, grid_rows - 1)

    geojson_file = "../data/polygons/moreton.geojson"
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)
        # Assuming the GeoJSON is a FeatureCollection or a Feature
        if geojson_data["type"] == "FeatureCollection":
            polygons = [
                shape(feature["geometry"]) for feature in geojson_data["features"]
            ]
        elif geojson_data["type"] == "Feature":
            polygons = [shape(geojson_data["geometry"])]
        else:
            polygons = [shape(geojson_data)]

    # **Coordinate Transformation from EPSG:4326 to EPSG:32756**
    # Define the coordinate transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)

    # Transform the polygons to EPSG:32756
    transformed_polygons = []
    for polygon in polygons:
        transformed_polygon = transform(transformer.transform, polygon)
        transformed_polygons.append(transformed_polygon)

    # Merge all transformed polygons into a single geometry
    polygon_union = unary_union(transformed_polygons)

    # **Precompute the grid mask**
    grid_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    for x in tqdm(range(grid_cols), desc="Creating grid mask"):
        for y in range(grid_rows):
            # Compute the bounds of the grid cell
            x1 = grid_minx + x * GRID_SIZE
            y1 = grid_miny + y * GRID_SIZE
            x2 = x1 + GRID_SIZE
            y2 = y1 + GRID_SIZE

            # Create a shapely polygon for the grid cell
            cell_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # Check if cell_polygon intersects with the transformed polygon(s)
            if cell_polygon.intersects(polygon_union):
                grid_mask[y, x] = True
    # need to flip the mask vertically
    grid_mask = np.flipud(grid_mask)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    heatmaps = generate_coverage_heatmaps(
        UDM_DIR,
        key_function,
        grid_minx,
        grid_miny,
        grid_maxx,
        grid_maxy,
        grid_rows,
        grid_cols,
    )

    # Save the heatmaps to a file
    with open("heatmaps.csv", "w") as f:
        f.write("year,gridx,gridy,count\n")
        for year, heatmap in heatmaps.items():
            for (x, y), count in heatmap.items():
                if grid_mask[y, x]:  # Check the grid_mask before writing
                    f.write(f"{year},{x},{y},{count}\n")
