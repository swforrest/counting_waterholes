"""
Functions to help with the automatic pipeline.
"""
import traceback
from utils.config import cfg
import pandas as pd
import os
import datetime
import classifier
from . import planet_utils 
import heatmap as hm
import area_coverage as ac
import json

def get_history(csv_path) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # write the header line "order_id, AOI, date, order_status, area_coverage, cloud_coverage"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        open(csv_path, "w").write("order_id,aoi,date,order_status,area_coverage,cloud_coverage\n")
    history = pd.read_csv(csv_path)

    return history

def save_history(history, csv_path):
    history.to_csv(csv_path, index=False)

def search(aoi, csv_path):
    """
    Search for images for a given AOI
    @param aoi: Area of Interest name
    @param csv_path: Path to csv of order history
    @return list: list of options
    """
    # First get all the dates that we have for the AOI
    history = get_history(csv_path)
    dates = history[history["aoi"] == aoi]["date"].unique()
    # If we don't have any, use the default last 14 days
    min_date = datetime.datetime.now() - datetime.timedelta(days=cfg["HISTORY_LENGTH"])
    min_date = min_date.strftime("%Y-%m-%d")

    max_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # create list of dates between min and max
    daterange = pd.date_range(start=min_date, end=max_date).strftime("%Y-%m-%d").tolist()
    # remove any dates we already have
    dates = [d for d in daterange if d not in dates]
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

def select(aoi, options, dates):
    """
    Select images from the given options that are not in history for the AOI.
    @param aoi: Area of Interest name
    @param options: list of options
    @param dates: list of dates that we want to select for
    @return list: list of items
    """
    polygon = planet_utils.get_polygon_file(aoi)
    # Select images for each date and yield them
    for date in dates:
        try:
            items = planet_utils.PlanetSelect(items=options,polygon=polygon, date=date, area_coverage=cfg["MINIMUM_AREA_COVERAGE"])
        except Exception as e:
            traceback.print_exc()
            print(e)
            continue
        if items is None or len(items) == 0:
            continue
        yield items

def order(aoi, items, csv_path):
    """
    Place a Planet order for the given items
    @param aoi: Area of Interest name
    @param items: list of items to order
    @param history: DataFrame of the history of orders
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

def download(csv_path, download_path="tempDL"):
    """
    Download the given order from Planet
    @param order: order_id
    @param history: DataFrame of the history of orders
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

def count():
    """
    Process any downloaded images
    """
    classifier.main()


def save(csv_path):
    """
    Confirm completion of the process and archive the raw data
    """
    history = get_history(csv_path)
    new = history[history["order_status"] == "downloaded"]
    if len(new) == 0:
        print("No new orders this run. Exiting.")
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


def analyse(csv_path, coverage_path):
    # Update:
        # - coverage heatmap raster
    for g in groups:
        heatmap_path = os.path.join(cfg["output_dir"], f"{g['name']}_coverage_heatmap.tif")
        coverage = pd.read_csv(coverage_path)
        polygons = hm.get_polygons_from_file(coverage_path, group=g["aois"])
        hm.create_heatmap_from_polygons(polygons=polygons, save_file=heatmap_path)

def archive(path, coverage_path):
    """
    Deal with folder of raw data after processing
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
                # NOTE:  ARCHIVE THIS ZIP
                # This is where we could send it off to AWS, or another storage location
                print("Sending to archive (not really, just not deleting it!)")
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
                    cov_amount, _ = ac.area_coverage_poly(planet_utils.get_polygon_file(aoi), polygon)
                    # add to coverage
                    coverage = pd.concat([coverage, pd.DataFrame({"aoi": [aoi], "date": [date], "area_coverage": [cov_amount], "polygon": [json.dumps(polygon)]})])
                    # save the coverage
                    coverage.to_csv(coverage_path, index=False)
            shutil.rmtree(os.path.join(root, d))