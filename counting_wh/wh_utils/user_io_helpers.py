"""
This module contains functions for user interaction and input/output for a CLI tool for counting whs.
Unfinished, untested, and possibly un-needed.

Author: Charlie Turner
Date: 17/09/2024
"""

import os
from . import planet_utils as planet_utils
from . import classifier as classifier
import traceback
import datetime
import argparse
import pandas as pd
from . import spatial_helpers as ac
from . import heatmap as hm
import json
from .config import cfg

""" Groups for reporting/analysing """
groups = [
    {"name": "MoretonBayRegion", "aois": ["peel", "south_bribie", "tangalooma"]},
    {"name": "GBR", "aois": ["keppel", "whitsundays_island_group"]},
]


def analyse(csv_path, coverage_path):
    # Update:
    # - coverage heatmap raster
    for g in groups:
        heatmap_path = os.path.join("outputs", f"{g['name']}_coverage_heatmap.tif")
        coverage = pd.read_csv(coverage_path)
        polygons = hm.get_polygons_from_file(coverage_path, group=g["aois"])
        if len(polygons) == 0:
            print(f"Analyse: No polygons found for {g['name']}")
            continue
        hm.create_heatmap_from_polygons(polygons, heatmap_path)


def report():
    # create a file for this run
    pass


def save_history(history, csv_path):
    history.to_csv(csv_path, index=False)


def archive(path, coverage_path):
    """
    Deal with folder of raw data after processing
    """
    # We want to delete any folders, but keep zip folders
    if not os.path.exists(coverage_path):
        # create it
        open(coverage_path, "w").write(
            "date,aoi,area_coverage,cloud_coverage,polygon\n"
        )
    coverage = pd.read_csv(coverage_path)
    import shutil

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".zip"):
                # ARCHIVE THIS ZIP
                print(
                    "Sending to archive (not really, make sure we do this later!!!):", f
                )
                continue
        for d in dirs:
            # save the polygon to the coverage file
            # load composite_metadata.json from the directory if it exists
            if "composite_metadata.json" in os.listdir(os.path.join(root, d)):
                meta = json.load(open(os.path.join(root, d, "composite_metadata.json")))
                polygon = meta["geometry"]
                aoi = "_".join(d.split("_")[0:-1])
                date = d.split("_")[-1]
                date = date[:4] + "-" + date[4:6] + "-" + date[6:8]
                # check to see if exists in coverage file already
                if (
                    len(coverage[(coverage["date"] == date) & (coverage["aoi"] == aoi)])
                    > 0
                ):
                    print(f"Already have {date}, {aoi} in coverage. Skipping.")
                else:
                    cov_amount = ac.area_coverage_poly(
                        planet_utils.get_polygon_file(aoi), polygon
                    )
                    # add to coverage
                    coverage = pd.concat(
                        [
                            coverage,
                            pd.DataFrame(
                                {
                                    "aoi": [aoi],
                                    "date": [date],
                                    "area_coverage": [cov_amount],
                                    "polygon": [json.dumps(polygon)],
                                }
                            ),
                        ]
                    )
                    # save the coverage
                    coverage.to_csv(coverage_path, index=False)
            shutil.rmtree(os.path.join(root, d))


def new_order():
    """
    Prompts the user and orders the image from Planet
    """
    aoi = option_select(planet_utils.get_aois(), prompt="Select an AOI:")
    polygon = planet_utils.get_polygon_file(aoi)
    min_date_default = datetime.datetime.now() - datetime.timedelta(days=14)
    min_date_default = min_date_default.strftime("%Y-%m-%d")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    min_date = input(f"Minimum date ({min_date_default}): ")
    if min_date == "":
        min_date = min_date_default
    max_date = input(f"Maximum date ({today}): ")
    if max_date == "":
        max_date = today
    cloud_cover = input("Maximum cloud cover (0.1): ")
    if cloud_cover == "":
        cloud_cover = 0.1
    area_cover = input("Minimum area cover (0.9): ")
    if area_cover == "":
        area_cover = 0.9
    try:
        options = planet_utils.PlanetSearch(
            polygon_file=polygon,
            min_date=min_date,
            max_date=max_date,
            cloud_cover=float(cloud_cover),
        )
        if len(options) == 0:
            print("No images found with filter in search.")
            exit()
        items = planet_utils.PlanetSelect(
            options, polygon=polygon, area_coverage=float(area_cover)
        )
        if items is None or len(items) == 0:
            print("No images found with filter in select.")
            exit()
        order = planet_utils.PlanetOrder(
            polygon_file=polygon,
            items=items,
            name=f"{aoi}_{items[0]['properties']['acquired'][:10]}_{items[-1]['properties']['acquired'][:10]}",
        )
        order_id = order["id"]
        print("Order ID:", order_id)
        return order_id
    except Exception as e:
        traceback.print_exc()
        print(e)
        exit()


def existing_order():
    """
    Checks an order and downloads the files
    """
    order_id = input("Order ID: ")
    # Download
    try:
        planet_utils.PlanetDownload(order_id)
    except Exception as e:
        traceback.print_exc()
        print(e)
        exit(1)


def option_select(options: list, prompt: str = "Select an option:"):
    print(prompt)
    for i in range(len(options)):
        print(i + 1, options[i])
    choice = input("Choice: ")
    if choice.isdigit():
        choice = int(choice)
        if choice > 0 and choice <= len(options):
            return options[choice - 1]
        else:
            print("Invalid choice. Exiting.")
            exit()
