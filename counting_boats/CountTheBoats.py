import os
import utils.planet_utils as planet_utils
import utils.classifier as classifier
import traceback
import datetime
import argparse
import pandas as pd
import utils.area_coverage as ac
import utils.heatmap as hm
import json
from config import cfg

def main():
    parser = argparse.ArgumentParser(description="Count the boats!")
    parser.add_argument("-a", "--full-auto", action="store_true", help="Run in full auto mode")
    parser.add_argument("-i", "--interest", help="Area of Interest to process", default=None)
    parser.description = \
            """
            Count the boats!
            """
    args = parser.parse_args()
    if args.full_auto:
        # turn interest into a list: e.g "peel,rockingham" -> ["peel", "rockingham"]
        aois = args.interest.split(",") if args.interest is not None else None
        full_auto(aois)
    else:
        choice = input("New order or existing? (N/E): ")
        if choice == "N":
            new_order()
        elif choice == "E":
            existing_order()
        else:
            print("Invalid choice. Exiting.")
            exit()

def full_auto(aois=None):
    csv_path = os.path.join("outputs", "AOI_history.csv")
    archive_path = os.path.join("outputs", "coverage.csv")
    download_path = os.path.join("images", "downloads")
    archive(download_path, archive_path)
    aois = aois if aois is not None else planet_utils.get_aois()
    for aoi in aois:
        options, dates = auto_search(aoi, csv_path)
        if options is None:
            continue
        for items in auto_select(aoi, options, dates):
            auto_order(aoi, items, csv_path )
    auto_download(csv_path=csv_path, download_path=download_path)
    auto_extract(csv_path=csv_path, download_path=download_path)
    auto_count() # Count the boats
    auto_save(csv_path)                 # Save the history
    archive(download_path, archive_path) # Archive the raw data (and save coverage info)
    # analyse
    analyse(csv_path, archive_path)
    # report (per run, and overall)

""" Groups for reporting/analysing """
groups = [
    {
        "name": "MoretonBayRegion", "aois": ["peel", "south_bribie", "tangalooma"]
    },
    {
        "name": "GBR", "aois": ["keppel", "whitsundays_island_group"]
    }
]

def analyse(csv_path, coverage_path):
    # Update:
        # - coverage heatmap raster
    for g in groups:
        heatmap_path = os.path.join("outputs", f"{g['name']}_coverage_heatmap.tif")
        coverage = pd.read_csv(coverage_path)
        polygons = hm.get_polygons_from_file(coverage_path, group=g["aois"])
        hm.create_heatmap_from_polygons(polygons, heatmap_path)

def report():
    # create a file for this run
    pass

def auto_extract(csv_path, download_path):
    """
    Extract any zips that we don't have yet
    """
    all_zips = [f for f in os.listdir(download_path) if f.endswith(".zip")]
    history = get_history(csv_path)
    for zip_file in all_zips:
        aoi = "_".join(zip_file.split("_")[0:-1])
        date = zip_file.split("_")[-1]
        date = date[:4] + "-" + date[4:6] + "-" + date[6:8]
        right_aoi = history[history["aoi"] == aoi]
        right_date = right_aoi[right_aoi["date"] == date]
        if len(right_date) == 0:
            # need to extract
            try:
                planet_utils.extract_zip(downloadFile=os.path.join(download_path, zip_file))
            except Exception as e:
                traceback.print_exc()
                print(e)
                continue
            # update history
            history = get_history(csv_path)
            this_row = history[(history["aoi"] == aoi) & (history["date"] == date)]
            if len(this_row) == 0:
                print(f"Could not find {aoi} {date} in history. What?")
            this_row["order_status"] = "downloaded"
            save_history(history, csv_path)

def get_history(csv_path) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # write the header line "order_id, AOI, date, order_status, area_coverage, cloud_coverage"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        open(csv_path, "w").write("order_id,aoi,date,order_status,area_coverage,cloud_coverage\n")
    history = pd.read_csv(csv_path)
    return history


def auto_search(aoi, csv_path):
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

def auto_select(aoi, options, dates):
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

def auto_order(aoi, items, csv_path):
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

def auto_download(csv_path, download_path="tempDL"):
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

def auto_count():
    """
    Process any downloaded images
    """
    classifier.main()


def auto_save(csv_path):
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

def save_history(history, csv_path):
    history.to_csv(csv_path, index=False)

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
                # ARCHIVE THIS ZIP
                print("Sending to archive (not really, make sure we do this later!!!):", f)
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
    if min_date == "": min_date = min_date_default
    max_date = input(f"Maximum date ({today}): ")
    if max_date == "": max_date = today
    cloud_cover = input("Maximum cloud cover (0.1): ")
    if cloud_cover == "": cloud_cover = 0.1
    area_cover = input("Minimum area cover (0.9): ")
    if area_cover == "": area_cover = 0.9
    try:
        options = planet_utils.PlanetSearch(polygon_file=polygon, min_date=min_date, max_date=max_date, cloud_cover=float(cloud_cover))
        if len(options) == 0:
            print("No images found with filter in search.")
            exit()
        items = planet_utils.PlanetSelect(options, polygon=polygon, area_coverage=float(area_cover))
        if items is None or len(items) == 0:
            print("No images found with filter in select.")
            exit()
        order = planet_utils.PlanetOrder(polygon_file=polygon, items=items, name=f"{aoi}_{items[0]['properties']['acquired'][:10]}_{items[-1]['properties']['acquired'][:10]}")
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

def option_select(options:list, prompt:str="Select an option:"):
    print(prompt)
    for i in range(len(options)):
        print(i+1, options[i])
    choice = input("Choice: ")
    if choice.isdigit():
        choice = int(choice)
        if choice > 0 and choice <= len(options):
            return options[choice-1]
        else:
            print("Invalid choice. Exiting.")
            exit()
        

if __name__ == "__main__":
    main()

