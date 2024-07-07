import datetime
import os
import argparse
import time
import typer
import utils.planet_utils as pu
import utils.classifier as cl
import utils.auto_helpers as ah
import pandas as pd
import math
from utils.config import cfg


class COLORS:
    """Colors for the terminal"""

    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    OKCYAN = "\033[96m"
    OKYELLOW = "\033[93m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


"""
Argparse, typer, rich for CLI in this file

utils/whatever for the actual functionality

Workflow for manual:
1. Fill out config file.
2. Searching Planet (search {aoi} {start_date} {end_date} {cloud_cover} {area_cover}):
    Input       : AOI, start date, end date, cloud cover, area cover
    Output      : list of images, total area, output .txt with one image per line
                    runs/{this}/images.txt
3. Ordering Images (order {items}):
    Input       : .txt file | list[items]
    Output      : List of order ids, output .txt with one order id per line 
                    runs/{this}/orders.txt
    Side effects: Orders the images, updates auto_orders.csv file with statuses
4. Downloading Images (download {orders}):
    Input       : .txt file | list[order ids]
    Output      : Zip file with images, list of zip files with images
                    runs/{this}/images.txt
    Side effects: Downloads the images, updates auto_orders.csv file with statuses
5. Classifying Images (classify {images}):
    Input       : .txt file | list[zip files of tif] | list[images (png)]
    Output      : classification .csv file
    Side effects: Creates classified images, updates coverage.csv file, archives zip files
"""

app = typer.Typer(
    name="CountTheBoats",
    help="Count the boats in the images",
    no_args_is_help=True,
)


@app.command()
def auto(
    skip_order: bool = typer.Option(
        False, help="Skip ordering images (Just check existing orders)"
    ),
):
    """
    Run the entire pipeline automatically.
    """
    orders_path = os.path.join(cfg["output_dir"], "orders.csv")
    coverage_path = os.path.join(cfg["output_dir"], "coverage.csv")
    download_path = os.path.join("images", "downloads")
    # For all AOIS
    history_len = cfg["HISTORY_LENGTH"]
    aois = cfg["AOIS"].split(",")
    if "all" in aois:
        aois = pu.get_aois()
    print(
        "Chosen AOIS:",
        aois,
    )
    if cfg["AUTO_MODE"] == "single":  # Do all history at once
        if not skip_order:
            # pause for 3 seconds
            time.sleep(3)
            for aoi in aois:
                options, dates = ah.search(aoi, orders_path, days=history_len)
                if options is None:
                    continue
                for items in ah.select(aoi, options, dates):
                    ah.order(aoi, items, orders_path)
        print("Downloading Images")
        ah.download(csv_path=orders_path, download_path=download_path)
        # ah.extract(csv_path=orders_path, download_path=download_path)
        print("Classifying Images")
        ah.count(
            save_coverage=False
        )  # Count the boats (don't save coverage, we will do that later)
        ah.save(orders_path)  # Save the history
        print("Archiving Images")
        ah.archive(
            download_path, coverage_path
        )  # Archive the raw data (and save coverage info)
        # analyse
        print("Analysing Images")
        ah.analyse(orders_path, coverage_path)
        # report (per run, and overall)
    elif cfg["AUTO_MODE"] == "batch":  # --------------- BATCH MODE ----------------
        batch_mode(history_len=history_len, aois=aois, orders_path=orders_path, download_path=download_path, coverage_path=coverage_path)
    else:
        raise ValueError("Invalid AUTO_MODE in config")
    print("Auto complete. Results in", cfg["output_dir"])


def batch_mode(history_len: int, aois: list[str], orders_path: str, download_path: str, coverage_path: str):
    batch_size = cfg["BATCH_SIZE"]
    # set start date to today - history length
    start_date = datetime.datetime.now() - datetime.timedelta(days=(history_len - 1))
    # end date of first batch
    end_date = start_date + datetime.timedelta(days=(batch_size - 1))
    n_batches = math.ceil(history_len / batch_size)
    print(
        COLORS.OKBLUE,
        "Batch Mode... Processing",
        n_batches,
        "batches of",
        batch_size,
        "days",
        COLORS.ENDC,
    )
    print(
        "Dates:",
        start_date.strftime("%d/%m/%Y"),
        "to",
        datetime.datetime.now().strftime("%d/%m/%Y"),
        "(inclusive)",
    )
    time.sleep(3)
    # for each batch
    for i in range(n_batches):
        print(
            COLORS.OKBLUE,
            f"Batch {i + 1}: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')} (inclusive)",
            COLORS.ENDC,
        )
        # search
        for aoi in aois:
            options, dates = ah.search(
                aoi=aoi,
                start_date=start_date,
                end_date=end_date,
                orders_csv_path=orders_path,
            )
            if options is None:
                continue
            # select
            for items in ah.select(aoi, options, dates):
                ah.order(aoi, items, orders_path)
        # download
        print(COLORS.OKCYAN, "Downloading Images", COLORS.ENDC)
        batch_download_with_wait(orders_path, download_path, start_date, end_date)
        # classify -> We will save coverage during archive step.
        ah.count(save_coverage=False)
        # save
        ah.save(orders_path)
        # archive -> also saves coverage file
        ah.archive(download_path, coverage_path)
        # analyse batch
        boat_csv_path = cfg["output_dir"] + "/boat_detections.csv"
        ah.analyse(
            boat_csv_path,
            coverage_path,
            start_date=start_date,
            end_date=end_date,
            id=f"batch_{i}",
        )
        # report on batch
        start_date = end_date + datetime.timedelta(days=1)
        end_date = start_date + datetime.timedelta(days=batch_size)
        if end_date > datetime.datetime.now():
            end_date = datetime.datetime.now()
        print(COLORS.OKGREEN, "Batch", i + 1, "complete", COLORS.ENDC)
        time.sleep(3)


def batch_download_with_wait(orders_path, download_path, start_date, end_date):
    # wait until all downloaded
    # Strategy  -> check every 10 minutes. If an order comes through, check every 5 minutes until all orders are downloaded
    #           -> if its been over 2 hours, exit and alert user something might be wrong
    # download
    wait_time = 10 * 60  # 10 minutes
    total_wait_time = 0
    timeout = 2 * 60 * 60  # 2 hours
    remaining_orders: pd.DataFrame = ah.download(
        csv_path=orders_path, download_path=download_path
    )
    remaining_orders["date"] = pd.to_datetime(remaining_orders["date"])
    start_date = pd.to_datetime(start_date).floor("D")
    end_date = pd.to_datetime(end_date).ceil("D")
    remaining_orders = remaining_orders.loc[
        (remaining_orders["date"] >= start_date)
        & (remaining_orders["date"] <= end_date)
    ]

    terminal_width = os.get_terminal_size().columns
    string = (
        "Waiting for "
        + str(len(remaining_orders))
        + " orders to download. Sleeping "
        + str(wait_time)
        + " seconds."
    )
    padding = terminal_width - len(string) - 8 - 3  # 8 = hh:mm:ss, 2 = 2 spaces
    padding_string = "=" * padding if padding > 0 else ""
    while len(remaining_orders) > 0:
        print(
            string,
            padding_string,
            datetime.datetime.now().strftime("%H:%M:%S"),  # hh:mm:ss = 8 characters
        )
        time.sleep(wait_time)
        remaining_orders = ah.download(
            csv_path=orders_path, download_path=download_path
        )
        remaining_orders = remaining_orders.loc[
            (remaining_orders["date"] >= start_date)
            & (remaining_orders["date"] <= end_date)
        ]
        total_wait_time += wait_time
        wait_time = 5 * 60
        if total_wait_time > timeout:
            print(remaining_orders)
            print("Waited over 2 hours for orders to download. Exiting.")
            break


@app.command()
def search(aoi: str, start_date: str, end_date: str, cloud_cover: str, area_cover: str):
    """
    Search for images on planet.
    """
    # TODO: implement search functionality
    pass


@app.command()
def order(items: str):
    """
    Order the given images from Planet.
    """
    # TODO: implement order functionality
    pass


@app.command()
def download(orders: str):
    """
    Downloads the images from the orders in the input file.
    """
    # TODO: implement download functionality
    pass


@app.command()
def classify(images: str):
    """
    Run the classifier on the given images.
    """
    # Images can be either:
    # A .txt file with a list of zip files of tif images
    # A .txt file with a list of png images
    # A list of the above
    # A path to a folder with the above
    # if is folder
    if os.path.isdir(images):
        cl.classify_directory(images)
    pass


if __name__ == "__main__":
    app()
