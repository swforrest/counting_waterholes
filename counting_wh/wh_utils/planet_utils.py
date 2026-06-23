"""
A collection of utilities to interact with the Planet API specifically for the counting whs project.
Opinionated in that there are assumptions made about the desired workflow.

The main functions are:
- PlanetSearch: search for imagery in a given area of interest
- PlanetSelect: select a subset of items from a search result
- PlanetOrder: order a given search result
- PlanetCheckOrder: check the status of an order

Author: Charlie Turner
Date: 17/09/2024
"""

import requests
import os
import yaml
import json
from .config import cfg
from dotenv import load_dotenv
from . import spatial_helpers
import zipfile
import argparse #addition from gpt to allow execution from the command line 

load_dotenv(override=True)

# api key is either set in the environment variable or config.yml
API_KEY = os.environ.get("PLANET_API_KEY", cfg["planet"]["api_key"])
#if API_KEY is None or API_KEY == "ENV":
    #raise Exception("Planet API key not found in environment variable or config.yml")
#AF: deleted for now

def PlanetSearch(
    polygon_file: str,
    min_date: str,
    max_date: str,
    cloud_cover: float = 0.1,
):
    """
    Search a given area of interest for Planet imagery

    Args:

        polygon_file: the path to the polygon file
        min_date: the minimum date to search for (inclusive)
        max_date: the maximum date to search for (inclusive)
        cloud_cover: the maximum cloud cover to search for

    Requires:

        Planet API key be set in environment variable

    Returns:

        a list of Planet items (json result from API)

    Raises:

        Exception: if the API returns a non-200 status code
    """
    # get the polygon
    polygon = None

    with open(polygon_file, "r") as f:
        geojson_data = json.load(f)

    # # Load GeoJSON from file
    # with open(selected_polygon, 'r') as f:
    #     geojson_data = json.load(f)

    # Extract the geometry from the GeoJSON
    # Depending on your GeoJSON structure, you might need one of these approaches:
    if 'geometry' in geojson_data:
        # For a Feature
        polygon = geojson_data['geometry']
    elif 'type' in geojson_data and geojson_data['type'] == 'FeatureCollection':
        # For a FeatureCollection, use the first feature's geometry
        polygon = geojson_data['features'][0]['geometry']
    elif 'coordinates' in geojson_data and 'type' in geojson_data:
        # If it's already a bare geometry
        polygon = geojson_data


    # Get the format-string json file as text
    search_body = SEARCH_BODY.replace("MIN_DATE", min_date)
    search_body = search_body.replace("MAX_DATE", max_date)
    search_body = search_body.replace("CLOUD_COVER", str(cloud_cover))
    search_body = search_body.replace("POLYGON", json.dumps(polygon))
    # parse as json
    headers = {"content-type": "application/json"}
    auth = (API_KEY, "")
    response = requests.post(
        "https://api.planet.com/data/v1/quick-search",
        data=search_body,
        headers=headers,
        auth=auth,
    )
    if response.status_code != 200:
        print(response.text)
        raise Exception("Planet API returned non-200 status code in search call")
    return response.json()["features"]


def PlanetSelect(
    items: list, polygon: str, date: str | None = None, area_coverage: float = 0.9
):
    """
    Select a subset of items from a search result

    Args:

        items: the list of items to select from
        polygon: the area of interest to check coverage against (polygon file)
        date: the date to select
        area_coverage: the minimum area coverage to select

    Returns:

        list of selected items
    """
    selected = None
    if date is None:
        coverage = 0
        offset = 1
        while coverage <= area_coverage:
            try:
                items.sort(key=lambda x: x["properties"]["acquired"])
                selected = [
                    item
                    for item in items
                    if item["properties"]["acquired"]
                    == items[-offset]["properties"]["acquired"]
                ]
                offset += len(selected)
                # check the area coverage
                coverage = items_area_coverage(selected, polygon)
            except Exception as e:
                print(e)
                selected = None
                break
    else:
        selected = [item for item in items if date in item["properties"]["acquired"]]
        if len(selected) == 0:
            return None
        # check the area coverage
        coverage = items_area_coverage(selected, polygon)
        if coverage < area_coverage:
            selected = None
    return selected


def PlanetOrder(items: list, polygon_file: str, name: str):
    """
    Order a given search result.

    Args:

        itemIDs: a list of item IDs to order
        polygon_file: a geojson file containing the area of interest to clip to
            must be of format: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
        name: the name to give the order

    Returns:

        a list of order IDs
    """
    pids = [item["id"] for item in items]
    products = [
        {"item_ids": pids, "item_type": "PSScene", "product_bundle": "analytic_sr_udm2"}
    ]
    # get the polygon
    polygon = None
    with open(polygon_file, "r") as f:
        polygon = json.load(f)
    # create the order body
    order_body = {
        "name": name,
        "products": products,
        "tools": [
            {
                "clip": {"aoi": polygon},
            },
            {
                "harmonize": {"target_sensor": "Sentinel-2"},
            },
            {"composite": {}},
        ],
        "notifications": {"email": False},
        "metadata": {"stac": {}},
        "order_type": "partial",
        "delivery": {
            "archive_filename": "planet_image_api",
            "archive_type": "zip",
            "single_archive": True,
        },
    }
    # set the headers
    headers = {"content-type": "application/json"}
    # make the request
    response = requests.post(
        "https://api.planet.com/compute/ops/orders/v2",
        data=json.dumps(order_body),
        headers=headers,
        auth=(API_KEY, ""),
    )
    # return that
    return response.json()


def PlanetCheckOrder(orderID: str):
    """
    Check the status of a given order

    Args:

        orderID: the order ID to check

    Returns:

        the status of the order
    """
    uri = f"https://api.planet.com/compute/ops/orders/v2/{orderID}"
    response = requests.get(uri, auth=(API_KEY, ""))
    return response.json()["state"]


def PlanetDownload(orderID: str, aoi=None, date=None, downloadPath="tempDL"):
    """
    Download a given order and move the tif file to the raw tiffs directory

    Args:

        orderID: the order ID to download
        aoi: the area of interest
        date: the date of the image
        downloadPath: the path to download the file to

    Returns:

        the name of the tif file after extracting
    """
    uri = f"https://api.planet.com/compute/ops/orders/v2/{orderID}"
    response = requests.get(uri, auth=(API_KEY, ""))
    if response.status_code != 200:
        raise Exception("Planet API returned non-200 status code")
    # grab the location uri
    links = response.json()["_links"]["results"]
    download_link = [
        link["location"] for link in links if "manifest" not in link["name"]
    ][0]
    # make the directory if it doesn't exist
    downloadFile = os.path.join(downloadPath, f"{aoi}_{date}.zip")
    os.makedirs(downloadPath, exist_ok=True)
    # download the file
    progress = 0
    with requests.get(download_link, stream=True) as r:
        with open(downloadFile, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress += len(chunk)
                    # print(f"{orderID} \t Downloaded {progress} bytes", end="\r")
    print(f"{orderID} \t Downloaded {progress} bytes", end="\r")
    # extract the file
    return extract_zip(downloadFile, aoi, date)


def extract_zip(downloadFile, aoi=None, date=None):
    """
    Extract the zip file and move the tif file to the raw tiffs directory

    Args:

        downloadFile: the path to the zip file
        aoi: the area of interest
        date: the date of the image

    Returns:

        the name of the tif file after extracting
    """
    if aoi is None or date is None:
        try:
            seps = os.path.basename(downloadFile).split(".zip")[0].split("_")
            date = seps[-1]
            aoi = "_".join(seps[:-1])
        except:
            raise Exception("AOI and date must be provided")
        
    extractPath = downloadFile.split(".zip")[0]
    print(downloadFile, extractPath)
    
    with zipfile.ZipFile(downloadFile, "r") as zip_ref:
        zip_ref.extractall(extractPath)
    
    # move the tif file to the raw tiffs directory, naming it (date)_(aoi).tif
    newfname = f"{date}_{aoi}.tif"
    tif = os.path.join(extractPath, "composite.tif")
    
    if not os.path.isfile(tif):
        raise Exception(f"No tif file found in {extractPath}")

    if not os.path.exists(cfg["tif_dir"]):
        os.makedirs(cfg["tif_dir"], exist_ok=True)

    try:
        os.replace(
            tif,
            os.path.join(cfg["tif_dir"], newfname),
        )
    except:
        if os.path.exists(os.path.join(cfg["tif_dir"], newfname)):
            print(f"File {newfname} already exists, skipping")
        else:
            raise Exception(f"Error moving {tif} to {newfname}")
    return newfname


def parse_config_AF(config: str) -> dict:
    """
    Parse the config file

    Args:
        config (str): path to the config file

    Returns:
        dict: the parsed config file
    """
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    

def extract_zip_AF(downloadFile, aoi, date, cfg):
    """
    Extract the zip file and move the tif file to the raw tiffs directory

    Args:

        downloadFile: the path to the zip file
        aoi: the area of interest
        date: the date of the image

    Returns:

        the name of the tif file after extracting
    """
    cfg=parse_config_AF(cfg)
    
    # Ensure tif_dir exists
    if not os.path.exists(cfg["tif_dir"]):
        os.makedirs(cfg["tif_dir"], exist_ok=True)
    
    # List to store processed tif filenames
    processed_tifs = []
    
    # Handle both single file and directory inputs
    if os.path.isfile(downloadFile):
        zip_files = [downloadFile]
    elif os.path.isdir(downloadFile):
        zip_files = [
            os.path.join(downloadFile, f) 
            for f in os.listdir(downloadFile) 
            if f.endswith('.zip')
        ]
    else:
        raise ValueError(f"Invalid input path: {downloadFile}")
    
    # Process each zip file
    for zip_path in zip_files:
        # Determine AOI and date if not provided
        current_aoi = aoi
        current_date = date
        
        if current_aoi is None or current_date is None:
            try:
                # Extract AOI and date from filename
                filename = os.path.basename(zip_path).split(".zip")[0]
                seps = filename.split("_")
                current_date = seps[-1]
                current_aoi = "_".join(seps[:-1])
            except:
                raise Exception(f"Could not extract AOI and date from {zip_path}")
        
        # Create extraction path
        extractPath = zip_path.split(".zip")[0]
        print(f"Processing: {zip_path}, Extract Path: {extractPath}")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extractPath)
        
        # Define paths for composite tif
        tif = os.path.join(extractPath, "composite.tif")
        
        # Check if tif exists
        if not os.path.isfile(tif):
            print(f"Warning: No tif file found in {extractPath}")
            continue
        
        # Create new filename
        newfname = f"{current_date}_{current_aoi}.tif"
        new_tif_path = os.path.join(cfg["tif_dir"], newfname)
        
        # Move file
        try:
            # If file already exists, skip
            if os.path.exists(new_tif_path):
                print(f"File {newfname} already exists, skipping")
                continue
            
            # Move the file
            os.replace(tif, new_tif_path)
            processed_tifs.append(newfname)
            print(f"Successfully moved {newfname} to {cfg['tif_dir']}")
        
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
    
    # Return list of processed tif filenames
    return processed_tifs



def get_with_retry(uri, auth, retries=5):
    """
    Get a URI with retries, doubling the delay each time.

    Args:

        uri: the URI to get
        auth: the auth tuple
        retries: the number of retries to attempt

    Returns:

        the response object
    """
    delay = 1
    for i in range(retries):
        response = requests.get(uri, auth=auth)
        if response.status_code == 200:
            return response
        delay *= 2
    return None


def get_orders():
    """
    Get all orders from the Planet API

    Returns:

        a list of orders which are dictionaries. Contains the order ID, state, etc.
    """
    uri = f"https://api.planet.com/compute/ops/orders/v2"
    response = get_with_retry(uri, (API_KEY, ""))
    if response is None:
        raise Exception("Failed to get orders")
    # write the response to a file
    with open("orders.json", "w") as f:
        f.write(response.text)
    data = response.json()
    orders = data["orders"]
    while "next" in data["_links"]:
        uri = data["_links"]["next"]
        response = get_with_retry(uri, (API_KEY, ""))
        if response is None:
            raise Exception("Failed to get orders")
        data = response.json()
        with open("orders.json", "a") as f:
            f.write(response.text)
        orders += data["orders"]

    return orders


def get_aois():
    """
    Get all the areas of interest

    Returns:

        a list of area of interest names
    """
    return [
        f.split(".")[0]
        for f in os.listdir(os.path.join(cfg["proj_root"], "data", "polygons"))
        if f.endswith("json")
    ]


def get_polygon_file(aoi):
    """
    Get the polygon for a given area of interest

    Args:

        aoi: the area of interest name

    Returns:

        the path to the polygon file

    """
    files = os.listdir(os.path.join(cfg["proj_root"], "data", "polygons"))
    files = [f for f in files if f.endswith("json")]  # only json files
    file = [f for f in files if f.split(".")[0] == aoi]
    if len(file) == 0:
        return None
    else:
        file = file[0]
    return os.path.join(cfg["proj_root"], "data", "polygons", file)


def items_area_coverage(items, AOI) -> float:
    """
    Given all the items for a particular day, and a polygon, compute the area coverage

    Args:

        items: a list of items
        AOI: the area of interest polygon

    Returns:

        the area coverage as a float
    """
    item_polys = [item["geometry"] for item in items]
    # combine the polygons
    combined = spatial_helpers.combine_polygons(item_polys)
    # get the coverage
    coverage = spatial_helpers.area_coverage_poly(AOI, combined)
    return coverage


if __name__ == "__main__":
    # prompt for what to do
    print("What would you like to do?")
    print("1. Check Order")
    print("2. Download")
    choice = input("Enter a number: ")
    if choice == "1":
        orderID = input("Enter the order ID: ")
        state = PlanetCheckOrder(orderID)
        print(f"Order {orderID} is {state}")
    elif choice == "2":
        orderID = input("Enter the order ID: ")
        downloadPath = input("Enter the path to download to: ")
        PlanetDownload(orderID)
    else:
        print("Not implemented yet")


SEARCH_BODY = """
        {
    "filter": {
        "type": "AndFilter",
        "config": [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": POLYGON
            },
            {
                "type": "OrFilter",
                "config": [
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {
                            "gte": "MIN_DATET00:00:00.000Z",
                            "lte": "MAX_DATET23:59:59.999Z"
                        }
                    }
                ]
            },
            {
                "type": "OrFilter",
                "config": [
                    {
                        "type": "AndFilter",
                        "config": [
                            {
                                "type": "AndFilter",
                                "config": [
                                    {
                                        "type": "StringInFilter",
                                        "field_name": "item_type",
                                        "config": [
                                            "PSScene"
                                        ]
                                    },
                                    {
                                        "type": "AndFilter",
                                        "config": [
                                            {
                                                "type": "AssetFilter",
                                                "config": [
                                                    "basic_analytic_4b"
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "type": "RangeFilter",
                                "config": {
                                    "gte": 0,
                                    "lte": CLOUD_COVER
                                },
                                "field_name": "cloud_cover"
                            },
                            {
                                "type": "StringInFilter",
                                "field_name": "publishing_stage",
                                "config": [
                                    "standard",
                                    "finalized"
                                ]
                            }
                        ]
                    },
                    {
                        "type": "AndFilter",
                        "config": [
                            {
                                "type": "StringInFilter",
                                "field_name": "item_type",
                                "config": [
                                    "SkySatCollect"
                                ]
                            },
                            {
                                "type": "RangeFilter",
                                "config": {
                                    "gte": 0,
                                    "lte": CLOUD_COVER
                                },
                                "field_name": "cloud_cover"
                            }
                        ]
                    }
                ]
            },
            {
                "type": "PermissionFilter",
                "config": [
                    "assets:download"
                ]
            }
        ]
    },
    "item_types": [
        "PSScene",
        "SkySatCollect"
    ]
}
"""
