"""
This module contains functions which help with the automatic detection pipeline.
"""
import traceback
from utils.config import cfg
import pandas as pd
import os
import datetime
import json
from utils import classifier
from utils import planet_utils 
from utils import heatmap as hm
from utils import area_coverage as ac
from utils import planet_utils

def get_history(csv_path: str)  -> pd.DataFrame:
    """
    Parse and return the csv file at the provided path.
    Creates the file if not exists, with the headings:
    "order_id, AOI, date, order_status, area_coverage, cloud_coverage"

    Args:
        csv_path: path to the csv file

    Returns:
        DataFrame of the csv file, or a new DataFrame with the header if the file does not exist.

    """
    
    if not os.path.exists(csv_path):
        # write the header line "order_id, AOI, date, order_status, area_coverage, cloud_coverage"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        open(csv_path, "w").write("order_id,aoi,date,order_status,area_coverage,cloud_coverage\n")
    history = pd.read_csv(csv_path)

    return history

def save_history(history:pd.DataFrame, csv_path:str) -> None:
    """
    Save the history DataFrame to the csv file at the provided path.

    Args:
        history: DataFrame of the history of orders
        csv_path: path to the csv file

    Returns:
        None
    """
    history.to_csv(csv_path, index=False)

def search(aoi: str, csv_path:str) -> tuple[list, list]:
    """
    Search for images for a given AOI

    Args:
        aoi: Area of Interest name
        csv_path: Path to csv of order history

    Returns:
        list of all options returned from the API, and dates that we want to select for
        where the dates are not present in our order history, and are limited to the last
        HISTORY_LENGTH days.
    """
    # First get all the dates that we have for the AOI
    history = get_history(csv_path)
    dates_we_have = history[history["aoi"] == aoi]["date"].unique()
    # If we don't have any, use the default last 14 days
    min_date = datetime.datetime.now() - datetime.timedelta(days=cfg["HISTORY_LENGTH"])
    min_date = min_date.strftime("%Y-%m-%d")

    max_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # create list of dates between min and max
    daterange = pd.date_range(start=min_date, end=max_date).strftime("%Y-%m-%d").tolist()
    # remove any dates we already have
    dates = [d for d in daterange if d not in dates_we_have]
    # search for images
    polygon = planet_utils.get_polygon_file(aoi)
    options = []
    try:
        options = planet_utils.PlanetSearch(polygon_file=polygon, 
                                       min_date=min_date, 
                                       max_date=max_date,
                                       cloud_cover=cfg["ALLOWED_CLOUD_COVER"])
    except Exception as e:
        traceback.print_exc()
        print(e)
        return None, None
    print("Found", len(options), "total images for", aoi)
    # select and order for each date
    if len(options) == 0:
        return None, None
    return options, dates

def select(aoi:str, options:list, dates:list) -> list[list]:
    """
    Select images from the given options that are not in history for the AOI.

    Args:
        aoi: Area of Interest name
        options: list of options
        dates: list of dates that we want to select for

    Returns:
        list of items which 
    """
    polygon = planet_utils.get_polygon_file(aoi)
    # Select images for each date and yield them
    items = []
    for date in dates:
        try:
            it= planet_utils.PlanetSelect(items=options,polygon=polygon, date=date, area_coverage=cfg["MINIMUM_AREA_COVERAGE"])
        except Exception as e:
            traceback.print_exc()
            print(e)
            continue
        if it is None or len(it) == 0:
            continue
        items.append(it)

def order(aoi:str, items:list, csv_path:str) -> str:
    """
    Place a Planet order for the given items

    Args:
        aoi: Area of Interest name
        items: list of items to order
        history: DataFrame of the history of orders

    Returns:
        order_id: the ID of the order placed 
    """
    polygon = planet_utils.get_polygon_file(aoi)
    date = items[0]["properties"]["acquired"][:10]
    fs_date = "".join(date.split("-")) # filesafe date
    try:
        order = planet_utils.PlanetOrder(polygon_file=polygon, 
                                    items=items, 
                                    name=f"{aoi}_{fs_date}")
    except Exception as e:
        traceback.print_exc()
        print(e)
        return
    order_id = order["id"]
    print("Order ID:", order_id)
    # add to history
    history = get_history(csv_path)
    history = pd.concat([history, pd.DataFrame({"order_id": [order_id], "aoi": [aoi], "date": [date], "order_status": ["ordered"], "area_coverage": ["-"], "cloud_coverage": ["-"]})])
    save_history(history, csv_path)
    return order_id

def download(csv_path:str, download_path="tempDL") -> None: 
    """
    Download all completed orders from planet, as per the history csv path
    Places the downloaded files in the download_path. 

    Args:
        csv_path: path to the history csv. Will check this to ensure we don't download the same file twice.
        download_path: path to store the downloaded zip files

    Returns:
        None
    """
    history = get_history(csv_path)
    orders = planet_utils.get_orders()
    for order in [o for o in orders if o["state"] == "success"]:
        # check if we have already downloaded (status == downloaded or complete)
        if order["id"] in history[history["order_status"].isin(["downloaded", "complete"])]["order_id"].tolist():
            # already have this
            continue
        elif order["id"] in history[history["order_status"] == "ordered"]["order_id"].tolist():
            print("Downloading", order["id"])
            # download
            this_order = history[history["order_id"] == order["id"]].iloc[0]
            try:
                planet_utils.PlanetDownload(order["id"], this_order["aoi"], this_order["date"].replace("-",""), downloadPath=download_path)
            except Exception as e:
                traceback.print_exc()
                print(e)
                continue
            # update history
            history.loc[history["order_id"] == order["id"], "order_status"] = "downloaded"
            save_history(history, csv_path)

def extract(download_path):
    """
    Extract any downloaded images we haven't processed. 

    Args:
        download_path: path to the downloaded zip files

    Returns:
        None
    """
    for root, dirs, files in os.walk(download_path):
        for f in files:
            if f.endswith(".zip"):
                planet_utils.extract_zip(os.path.join(root, f))


def count() -> None:
    """
    Run the classifier to count boats in the extracted images

    Essentially calls the main function of the classifier module
    """
    classifier.main()


def save(csv_path) -> None:
    """
    Confirm completion of the process and archive the raw data.
    Saves the csv with everything updated.

    Args:
        csv_path: path to the history csv
    
    Returns:
        None
    """
    history = get_history(csv_path)
    new = history[history["order_status"] == "downloaded"]
    if len(new) == 0:
        print("Save: No new orders this run.")
    # make a list of new file names (row["date"]_row["aoi"] for each row in new)
    new_files = [f"{row['date'].replace('-','')}_{row['aoi']}" for _, row in new.iterrows()]
    # update to complete if file does not exist in rawimages.
    # the classifier will have moved the files if complete
    for new_file in new_files:
        if not os.path.exists(os.path.join("data", "RawImages", new_file)):
            date = new_file[:4] + "-" + new_file[4:6] + "-" + new_file[6:8]
            aoi = "_".join(new_file.split("_")[1:])
            right_aoi = history[history["aoi"] == aoi]
            right_date = right_aoi[right_aoi["date"] == date]
            if len(right_date) == 0:
                print(f"Could not find {new_file} in history. Skipping.")
                continue
            order_id = right_date["order_id"].iloc[0]
            history.loc[history["order_id"] == order_id, "order_status"] = "complete"
    # save the history
    save_history(history, csv_path)


groups = [
    {"name": "moreton_bay", "aois": ["peel", "tangalooma", "bribie"]},
    {"name": "gbr", "aois": ["whitsundays", "keppel", "capricorn", "haymen", "heron"]},
]
''' Groups of AOIs for analysis'''


def analyse(csv_path, coverage_path, **kwargs) -> None:
    """
    do a series of analyses on the data and save it in the output directory

    Args:
        csv_path: path to the history csv
        coverage_path: path to the coverage csv (should be generated when images are archived)
        kwargs: additional arguments to skip or alter steps (look at the code for details)
    
    Returns:
        None
    """
    # Update:
        # - coverage heatmap raster
    for g in groups:
        heatmap_path = os.path.join(cfg["output_dir"], f"{g['name']}_coverage_heatmap.tif")
        coverage = pd.read_csv(coverage_path)
        polygons = hm.get_polygons_from_file(coverage_path, group=g["aois"])
        hm.create_heatmap_from_polygons(polygons=polygons, save_file=heatmap_path)

def archive(path:str, coverage_path:str):
    """
    Deal with folder of raw data after processing. Send zip files to archive,
    delete the folders, update coverage file. 

    Args:
        path: path to the folder of raw data
        coverage_path: path to the coverage csv
    
    Returns:
        None
    """
    # We want to delete any folders, but keep zip folders
    if not os.path.exists(coverage_path):
        # create it
        open(coverage_path, "w").write("date,aoi,area_coverage,polygon\n")
    coverage = pd.read_csv(coverage_path)
    import shutil
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".zip"):
                # Move to ../archive/{whatever}
                print("Sending to archive (not really, just moving to ./archive !)")
                os.rename(os.path.join(root, f), os.path.join("images", "archive", f))
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
                if len(coverage[(coverage["date"] == date) & (coverage["aoi"] == aoi)]) > 0:
                    print(f"Already have {date}, {aoi} in coverage. Skipping.")
                else:
                    cov_amount = ac.area_coverage_poly(planet_utils.get_polygon_file(aoi), polygon)
                    # add to coverage
                    coverage = pd.concat([coverage, pd.DataFrame({"aoi": [aoi], "date": [date], "area_coverage": [cov_amount], "polygon": [json.dumps(polygon)]})])
                    # save the coverage
                    coverage.to_csv(coverage_path, index=False)
            shutil.rmtree(os.path.join(root, d))