"""
This module provides functions for calculating the coverage of polygons and TIFF files.

The main functions in this module are:
- `area_coverage_tif`: Calculates the intersection of a polygon and a TIFF file, as a percentage of the polygon area.
- `area_coverage_poly`: Computes the intersection of a polygon and a reference polygon, as a percentage of the reference polygon area.
- `combine_polygons`: Combines two or more (intersecting) polygons into one.
- `polygons_to_32756`: Converts a polygon from latitude-longitude coordinates to EPSG:32756 coordinate system.

These functions are useful for analyzing and measuring the coverage of geographic areas by polygons and TIFF files.

Author: Charlie Turner
Date: 16/09/24
"""

import json
import numpy as np
from osgeo import gdal, ogr
import rasterio
import os
import matplotlib.pyplot as plt

from . import image_cutting_support as ics

UDM1_CLOUD = 2
UDM1_CLEAR = 0


def create_clear_coverage_tif(udm_path: str, filename="clear.tif"):
    """
    Same as add_udm_clear_to_tif, but creates a new tif file instead of modifying an existing one.
    """
    udm = gdal.Open(udm_path)
    # New tif with the full bounds of both
    width = udm.RasterXSize
    height = udm.RasterYSize
    # Create the new raster
    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(filename, width, height, 1, gdal.GDT_Byte)
    # Set the projection and geotransform
    out_raster.SetProjection(udm.GetProjection())
    out_raster.SetGeoTransform(udm.GetGeoTransform())
    # Read the rasters
    in_band_udm = udm.GetRasterBand(1)
    out_band = out_raster.GetRasterBand(1)
    # Create the new array
    out_band.WriteArray(in_band_udm.ReadAsArray())
    out_band.FlushCache()
    # Close the rasters
    out_raster = None
    udm = None


def get_array_from_tif(tif_path: str, band=1, grid_size=None):
    """
    Get the band from a tif as an array. If grid_size is not None, resample the array to squares of that size in meters.
    """
    tif = gdal.Open(tif_path)
    in_band = tif.GetRasterBand(band)
    array = in_band.ReadAsArray()
    x_top_left, _, _, y_top_left, _, _ = tif.GetGeoTransform()

    if grid_size is not None:
        transform = (
            tif.GetGeoTransform()
        )  # (top left x, x resolution, x rotation, top left y, y rotation, y resolution)
        x_top_left = transform[0]
        y_top_left = transform[3]
        x_res = transform[1] or 3
        y_res = transform[5] or -3
        x_size = tif.RasterXSize
        y_size = tif.RasterYSize
        # Get the real world size of the tif
        real_w = x_res * x_size
        real_h = y_res * y_size
        # Get the number of squares in each direction
        x_squares = int(real_w // grid_size)
        y_squares = int(real_h // grid_size)
        # Get the new size of each square
        new_x_res = real_w / x_squares
        new_y_res = real_h / y_squares
        # Create the new array with gdal Warp majority
        warp_options = gdal.WarpOptions(
            format="GTiff",
            xRes=new_x_res,
            yRes=new_y_res,
            resampleAlg=gdal.GRA_Mode,
        )
        gdal.Warp("temp.tif", tif, options=warp_options)
        tif = gdal.Open("temp.tif")
        in_band = tif.GetRasterBand(1)
        array = in_band.ReadAsArray()
        # Get the new top left corner
        x_top_left, _, _, y_top_left, _, _ = tif.GetGeoTransform()

    tif = None
    in_band = None
    return x_top_left, y_top_left, array


def add_udm_clear_to_tif(udm_path: str, tif_file):
    """
    Given a udm, add the clear coverage to the tif file. E.g, the tif file will
    have a raster with 1s for clear pixels and 0s for non-clear pixels. So will
    the udm. This function will alter the tif to then have a 2 for clear pixels in both
    rasters, or 1 if only one raster was clear.
    """
    if not os.path.exists(tif_file):
        print("TIF file does not exist, creating new")
        create_clear_coverage_tif(udm_path, tif_file)
        return
    udm: gdal.Dataset = gdal.Open(udm_path)
    tif: gdal.Dataset = gdal.Open(tif_file)
    # New tif with the full extent
    udm_transform = udm.GetGeoTransform()
    tif_transform = tif.GetGeoTransform()
    assert udm_transform[1] == tif_transform[1], "Pixel sizes do not match"
    udm_width = udm.RasterXSize
    udm_height = udm.RasterYSize
    tif_width = tif.RasterXSize
    tif_height = tif.RasterYSize
    top_left_x = min(udm_transform[0], tif_transform[0])
    top_left_y = max(udm_transform[3], tif_transform[3])
    bottom_right_x = max(
        udm_transform[0] + udm_width * udm_transform[1],
        tif_transform[0] + tif_width * tif_transform[1],
    )
    bottom_right_y = min(
        udm_transform[3] + udm_height * udm_transform[5],
        tif_transform[3] + tif_height * tif_transform[5],
    )
    # Using gdal warp to resize to max size and a common top left corner
    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=[top_left_x, bottom_right_y, bottom_right_x, top_left_y],
        xRes=udm_transform[1],
        yRes=udm_transform[5],
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    # warp the tif and udm
    gdal.Warp(udm_path.replace(".tif", "_w.tif"), udm, options=warp_options)
    gdal.Warp(tif_file.replace(".tif", "_w.tif"), tif, options=warp_options)
    # read the rasters
    udm = gdal.Open(udm_path.replace(".tif", "_w.tif"))
    tif = gdal.Open(tif_file.replace(".tif", "_w.tif"), gdal.GA_Update)
    if use_udm_2(udm_path):
        in_band_udm = udm.GetRasterBand(1)
    else:
        in_band_udm = udm.GetRasterBand(8)
    in_band_tif = tif.GetRasterBand(1)
    out_band = tif.GetRasterBand(1)
    udm_array = in_band_udm.ReadAsArray()
    if not use_udm_2(udm_path):
        udm_array[udm_array == UDM1_CLOUD] = 0
        udm_array[udm_array == UDM1_CLEAR] = 1
    tif_array = in_band_tif.ReadAsArray()
    # Create the new array
    new_array = udm_array + tif_array
    out_band.WriteArray(new_array)
    out_band.FlushCache()
    # Close the rasters
    tif = None
    udm = None
    # Rename the file to match the tif
    os.remove(tif_file)
    os.rename(tif_file.replace(".tif", "_w.tif"), tif_file)
    return


def use_udm_2(udm_path: str):
    """
    Open a udm, sum the clear pixels (band 1). If 0 return False (old udm has nothing in band 1), else True
    """
    udm = gdal.Open(udm_path)
    in_band = udm.GetRasterBand(1)
    array = in_band.ReadAsArray()
    return np.sum(array) > 0


def cloud_coverage_udm(udm_path: str) -> tuple[float, np.ndarray]:
    """
    Using the usable data mask (UDM) from Planet, calculate the cloud coverage of the image.
    The UDM is a raster file with 8 bands, each representing a different variable. Band 6 should
    be the cloud mask. This function checks the metadata to confirm, then calculates the cloud
    percentage as the proportion of pixels in band 6 that are clouds, and returns the cloud coverage
    mask as a binary array.
    Note: the cloud mask is over the entire extent of the image. Use `area_coverage_tif` to calculate
    the coverage of the tif, and then multiply cloud coverage with image coverage to get cloud coverage over
    the AOI.

    Args:

        udm_path: path to UDM file (from Planet)

    Returns

        A tuple containing, a float (cloud coverage as a proportion of the image), and a numpy array (cloud mask)
        with 1s for cloud pixels and 0s for non-cloud pixels.
    """
    with rasterio.open(udm_path) as src:
        # check that band 6 is the cloud mask
        if use_udm_2(udm_path):
            if "cloud" not in src.descriptions[5]:
                print(src.descriptions)
                raise ValueError("Band 6 of UDM is not the cloud mask")
            # read the cloud mask
            cloud_mask = src.read(6)
        else:
            udm1 = src.read(8)
            # cloud mask is any pixels with a 1 in bit 1 of the 8 bit binary value
            # bit 0: not imaged (1 indicates not imaged)
            # bit 1: cloud
            # bit 2-8: missing in a color band
            cloud_mask = udm1.astype(np.uint8) & 0b00000010

        imaged_mask = (
            src.read(8).astype(np.uint8) & 0b00000001
        )  # 1 for not imaged, 0 for imaged at this stage

        imaged_mask = imaged_mask == 0  # 1 for imaged, 0 for not imaged
        # calculate cloud coverage over imaged area
        cloud_coverage = np.sum(cloud_mask) / np.sum(imaged_mask)
        return cloud_coverage, cloud_mask


def area_coverage_tif(polygon: str, tif: str) -> tuple[float, float, float]:
    """
    Calculate the intersection of a polygon and a tif, as a percentage of the polygon area.
    To be used when calculating the coverage of a tif file for an AOI, after the TIF has already
    been obtained. Assumes the tif is clipped to the polygon (does not check whether the tif is
    actually inside the polygon, just calculates the areas).

    Args:

        polygon : path to polygon file (geojson format)
        tif     : path to tif file (from Planet)

    Returns

        A tuple containing, a float (proportion of polygon covered by tif),
        float (area of polygon, m2), float (area of tif, m2)
    """
    # Area of polygon:
    poly = polygon_latlong2crs(polygon)[0]
    area = poly.Area()
    # Same idea with the tif. Get the area of the tif
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
        coverage = tif_area / area
    return coverage, area, tif_area


def area_coverage_poly(reference: str, polygon: str) -> float:
    """
    Computes the intersection of a polygon and a reference polygon, as a proportion of the reference polygon area.

    Args:

        reference: path to reference polygon file (geojson format) SHOULD BE A SINGLE POLYGON
        polygon: path to polygon file (geojson format) Can be a multipolygon
    Returns

        The proportion of the reference polygon covered by the polygon
    """
    ref_poly = polygon_latlong2crs(reference)[0]
    # area of reference polygon
    ref_area = ref_poly.Area()
    polys = polygon_latlong2crs(polygon)
    coverage = 0
    for poly in polys:
        # intersection
        intersection = ref_poly.Intersection(poly)
        if intersection is None:
            raise ValueError("Polygons do not intersect")
        # area of intersection
        area = intersection.Area()
        # coverage
        coverage = coverage + (area / ref_area)
    return coverage


def combine_polygons(polygons: list[str]) -> ogr.Geometry:
    """
    Combines two or more polygons into one

    Args:

    polygons: list of paths to polygon file (geojson format) or polygon strings (stringified geojson)

    Returns

    The combined polygon in EPSG:32756 coordinate system
    """
    # convert to ogr polygons
    ogr_polys = [polygon_latlong2crs(poly)[0] for poly in polygons]
    if len(ogr_polys) == 0:
        print("No polygons to combine")
        print(ogr_polys)
        print(polygons)
        exit()
    # combine
    poly = ogr_polys[0]
    if len(ogr_polys) == 1:
        return poly
    for i in range(1, len(ogr_polys)):
        poly = poly.Union(ogr_polys[i])
    if poly is None:
        raise ValueError("Polygons do not intersect")
    return poly


def polygon_latlong2crs(
    polygon: str | dict | ogr.Geometry, crs=32756
) -> list[ogr.Geometry]:
    """
    Converts a polygon from lat long to a coordinate system (default EPSG:32756).
    Often polygons will be in geojson, which uses latitude and longitude.

    Args:

        polygon: one of: path to polygon file (geojson format), polygon string (stringified geojson), or dict representing the polygon. Polygon must be in lat long coordinates.
        crs: the coordinate system to convert to (default EPSG:32756)

    Returns

        The same polygon, converted to the specified coordinate system, or EPSG:32756 by default
    """
    if type(polygon) == ogr.Geometry:
        return [polygon]
    geoJSON: dict = {}
    if type(polygon) != str and type(polygon) != dict:
        raise ValueError(
            "polygons_to_32756: Received polygon of type {}, must be string or dict to convert".format(
                type(polygon)
            )
        )
    # check if the polygon is a string or a path
    if (
        type(polygon) == str and polygon[0] == "{"
    ):  # } <- obligatory bracket to fix linting
        # if string, convert to json
        geoJSON = json.loads(polygon)
    elif type(polygon) == str:
        with open(polygon) as f:
            # read as json
            geoJSON = json.load(f)
    elif type(polygon) == dict:
        geoJSON = polygon
    # convert from lat long to EPSG:32756
    if "geometry" in geoJSON:
        geoJSON = geoJSON["geometry"]
    if "geometries" in geoJSON:
        geoJSON = geoJSON["geometries"]
        polygons = []
        for shape in geoJSON:
            if not (shape["type"] == "Polygon" or shape["type"] == "MultiPolygon"):
                print("Shape type not supported", shape["type"])
                continue
            for i, poly in enumerate(shape["coordinates"]):
                if shape["type"] == "MultiPolygon":
                    poly = poly[0]  # geoJSONpologons are nested once more bit
                poly_dict = {"coordinates": [[None] * len(poly)], "type": "Polygon"}
                for i, val in enumerate(poly):
                    lat, long = val
                    x, y = ics.latlong2coord(lat, long)
                    poly_dict["coordinates"][0][i] = [x, y]
                polygons.append(ogr.CreateGeometryFromJson(json.dumps(poly_dict)))
        return polygons

    polygons = []
    for i, poly in enumerate(geoJSON["coordinates"]):
        if geoJSON["type"] == "MultiPolygon":
            poly = poly[0]  # geoJSONpologons are nested once more bit
        poly_dict = {"coordinates": [[None] * len(poly)], "type": "Polygon"}
        for i, val in enumerate(poly):
            lat, long = val
            x, y = ics.latlong2coord(lat, long)
            poly_dict["coordinates"][0][i] = [x, y]
        polygons.append(ogr.CreateGeometryFromJson(json.dumps(poly_dict)))
    return polygons


def is_inside(polygon, point):
    """
    Check if a point is inside a polygon
    :param polygon: polygon or list of polygons
    :param point: (lat, long)
    :return: True if point is inside polygon, False otherwise
    """
    # poly = polygon_latlong2crs(polygon)[0]
    point_obj = ogr.Geometry(ogr.wkbPoint)
    point_obj.AddPoint(point[0], point[1])
    # point is in lat long, convert to EPSG:32756
    print(point_obj)
    point_obj.TransformTo("EPSG:32756")
    if type(polygon) == list:
        for poly in polygon:
            if poly.Contains(point_obj):
                return True
    else:
        if polygon.Contains(point_obj):
            return True
    return False


if __name__ == "__main__":
    tif_file = input("tif File:")
    udm_file = input("udm:")
    cloud_cov, cloud_mask = cloud_coverage_udm(udm_file)
    cov, _, _ = area_coverage_tif(
        "/Users/charlieturner/Documents/Countingwhs/data/polygons/peel.json", tif_file
    )
    print("Cloud coverage: ", cloud_cov)
    print("Coverage: ", cov)
    print(cloud_mask)
    print(np.sum(cloud_mask))
    # Cloud coverage / coverage
    print(cloud_cov / cov)
