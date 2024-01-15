# import pandas as pd
import json

import numpy as np

import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import rasterio

import utils.image_cutting_support as ics

def area_coverage_tif(polygon, tif):
    """
    Calculate the intersection of a polygon and a tif, as a percentage of the polygon area.
    To be used when calculating the coverage of a tif file for an AOI, after the TIF has already
    been obtained. Assumes the tif is clipped to the polygon (does not check whether the tif is 
    actually inside the polygon, just calculates the areas).
    :param polygon: path to polygon file (geojson format)
    :param tif: path to tif file (from Planet)
    :return: coverage (decimal), area of polygon, area of tif
    """
# Area of polygon:
    poly = polygon_to_32756(polygon)
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
        coverage = tif_area/area
        return coverage, area, tif_area

def area_coverage_poly(reference, polygon):
    """
    Computes the intersection of a polygon and a reference polygon, as a percentage of the reference polygon area.
    :param reference: path to reference polygon file (geojson format)
    :param polygon: path to polygon file (geojson format)
    :return: coverage (decimal), area of reference polygon, area of polygon
    """
    ref_poly = polygon_to_32756(reference)
    poly = polygon_to_32756(polygon)
    # intersection
    intersection = ref_poly.Intersection(poly)
    if intersection is None:
        raise ValueError("Polygons do not intersect")
    # area of intersection
    area = intersection.Area()
    # area of reference polygon
    ref_area = ref_poly.Area()
    # coverage
    coverage = area/ref_area
    return coverage, intersection

def combine_polygons(polygons):
    """
    Combines two or more polygons into one
    :param polygons: list of paths to polygon file (geojson format) or polygon strings
    :return: combined polygon in EPSG:32756
    """
    # convert to ogr polygons
    ogr_polys = [polygon_to_32756(poly) for poly in polygons]
    # combine
    poly = ogr_polys[0]
    if len(ogr_polys) == 1:
        print("combine_polygons: Received only one polygon, returning it")
        return poly
    for i in range(1, len(ogr_polys)):
        poly = poly.Union(ogr_polys[i])
    if poly is None:
        raise ValueError("Polygons do not intersect")
    return poly

def polygon_to_32756(polygon:str|dict) -> ogr.Geometry:
    """
    Converts a polygon from lat long to EPSG:32756
    :param polygon: path to polygon file (geojson format) or polygon string
        e.g "{ "type": "Polygon", "coordinates": [...]}"
    :return: polygon in EPSG:32756
    """
    geoJSON = polygon
    if type(polygon) != str and type(polygon) != dict:
        print("polygon_to_32756: Received polygon of type {}, must be string or dict to convert".format(type(polygon)))
        return polygon
    # check if the polygon is a string or a path
    if type(polygon) == str and polygon[0] == "{":
        # if string, convert to json
        geoJSON = json.loads(polygon)
    elif type(polygon) == str:
        with open(polygon) as f:
            # read as json
            geoJSON = json.load(f)
# convert from lat long to EPSG:32756
    for i, val in enumerate(geoJSON['coordinates'][0]):
        lat, long = val
        x, y = ics.latlong2coord(lat, long)
        geoJSON['coordinates'][0][i] = [x, y]
    poly = ogr.CreateGeometryFromJson(str(geoJSON))
    return poly












