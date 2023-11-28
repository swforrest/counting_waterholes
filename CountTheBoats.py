"""
The semi-automated boat counter.
1. Asks: New order or existing?
NEW:
    1. Which AOI/AOI's?
    2. Which date?
        Latest is default, or specify
    3. Acceptable Cloud Coverage?
        Default is 10%
    4. Acceptable Area Coverage?
        Default is 90%
EXISTING:
    1. Order ID?
    2. Download the zip
    3. Unzip the zip
    4. Move the TIF to the correct folder
    5. Delete the zip
    6. Count them boats
    7. Add boat counts to the database
    8. save a labelled image for review
"""
import os
import PlanetUtils
import traceback
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Count the boats!")
    parser.add_argument("-a", "--full-auto", action="store_true", help="Run in full auto mode")
    args = parser.parse_args()
    if args.full_auto:
        full_auto()
    else:
        choice = input("New order or existing? (N/E): ")
        if choice == "N":
            new_order()
        elif choice == "E":
            existing_order()
        else:
            print("Invalid choice. Exiting.")
            exit()

def full_auto():
    """
    Runs the program in full auto mode
    1. Orders all three AOIs for any days in the last 14 days that we don't have yet
    2. Checks if any orders are complete (that weren't last time)
    3. Downloads the complete orders
    4. Counts the boats
    """
    # for each AOI, check the latest date we have (in boat_history.csv)
    # if we don't have any, use the default (all 14 days)
    # if we do have some, use the latest date we have as the minimum
    # boat_history.csv:
    # order_id, AOI, date, order_status, area_coverage, cloud_coverage
    csv_path = os.path.join("data", "AOI_history.csv")
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        open(csv_path, "w").close()
    entries = open(csv_path).readlines()
    entries = [l.split(",") for l in entries]
    for aoi in PlanetUtils.get_aois():
        # check csv
        dates = [l[2] for l in entries if l[1] == aoi]
        if len(dates) == 0:
            min_date = datetime.datetime.now() - datetime.timedelta(days=14)
            min_date = min_date.strftime("%Y-%m-%d")
        else:
            min_date = max(dates)
        if min_date == datetime.datetime.now().strftime("%Y-%m-%d"):
            print("Already have all dates for", aoi)
            continue
        # search for images
        polygon = PlanetUtils.get_polygon(aoi)
        options = PlanetUtils.PlanetSearch(polygon_file=polygon, 
                                           min_date=min_date, 
                                           max_date=datetime.datetime.now().strftime("%Y-%m-%d"), 
                                           cloud_cover=0.1, 
                                           area_cover=0.9)
        print("Found", len(options), "images for", aoi)
        # select and order for each date
        if len(options) == 0:
            print("No images found with filter.")
            continue
        # build a list of dates from min to max
        dates = [min_date]
        while dates[-1] != datetime.datetime.now().strftime("%Y-%m-%d"):
            dates.append((datetime.datetime.strptime(dates[-1], "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        for date in dates:
            fs_date = "".join(date.split("-")) # filesafe date
            items = PlanetUtils.PlanetSelect(options, date)
            if len(items) == 0:
                continue
            order = PlanetUtils.PlanetOrder(polygon_file=polygon, 
                                            items=items, 
                                            name=f"{aoi}_{fs_date}")
            order_id = order["id"]
            print("Order ID:", order_id)
            # add to csv
            entries.append(
                [order_id, aoi, date, "ordered", "-", "-", "\n"]
                    )
            open(csv_path, "w").writelines([",".join(l) for l in entries])
    # check for complete orders
    orders = PlanetUtils.get_orders()
    new = 0
    for order in [o for o in orders if o["state"] == "success"]:
        if order["id"] in [l[0] for l in entries if l[3] == "complete"]:
            continue
        elif order["id"] in [l[0] for l in entries if l[3] == "ordered"]:
            # download
            this_order = [l for l in entries if l[0] == order["id"]][0]
            PlanetUtils.PlanetDownload(order["id"])
            # update csv
            this_order[3] = "complete"
            entries = [l for l in entries if l[0] != order["id"]]
            entries.append(this_order)
            open(csv_path, "w").writelines([",".join(l) for l in entries])
            new += 1
    # count the boats
    # run NNClassifier.py
    if new > 0:
        print("New orders downloaded. Running NNClassifier.py")
        os.system("python NNClassifier.py")
    else:
        print("No new orders to process.")

def new_order():
    """
    Prompts the user and orders the image from Planet
    """
    aoi = option_select(PlanetUtils.get_aois(), prompt="Select an AOI:")
    polygon = PlanetUtils.get_polygon(aoi)
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
        options = PlanetUtils.PlanetSearch(polygon_file=polygon, min_date=min_date, max_date=max_date, cloud_cover=float(cloud_cover), area_cover=float(area_cover))
        if len(options) == 0:
            print("No images found with filter.")
            exit()
        items = PlanetUtils.PlanetSelect(options)
        order = PlanetUtils.PlanetOrder(polygon_file=polygon, items=items, name=f"{aoi}_{items[0]['properties']['acquired'][:10]}_{items[-1]['properties']['acquired'][:10]}")
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
        PlanetUtils.PlanetDownload(order_id)
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

