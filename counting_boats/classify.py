"""
The main file for the pipeline. Classifying boats will be done with this file in most cases.

Requires a config file to be present in the root directory of the project (one level up from this file).
See project README for more information on the config file.

Usage:
1. **For Automatic Mode:**

    python -m counting_boats.classify auto

Auto mode will run the entire pipeline: search, order, download, classify, archive, analyse.

2. **For Manual Mode:**

    python -m counting_boats.classify \\<command\\>

    Commands:
    - archive: Archive the images in the downloads folder.
    - search: Search for images on planet. (Not implemented)
    - order: Order the given images from Planet. (Not implemented)
    - download: Downloads the images from the orders in the input file (Not implemented)
    - classify: Run the classifier on the given images in the config file (Not implemented)

Not implemented commands will raise a NotImplementedError - all functionality for them is 
available in either boat_utils/planet_utils.py or boat_utils/user_io_helpers.py.

Author: Charlie Turner

Date: 16/09/2024
"""

import comet_ml
import datetime
import os
import argparse
import time
import typer
import pandas as pd
import numpy as np
import math
from .boat_utils.config import cfg
from .boat_utils import planet_utils as pu
from .boat_utils import classifier as cl
from .boat_utils import auto_helpers as ah


class COLORS:
    """Colors for the terminal"""

    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    OKCYAN = "\033[96m"
    OKYELLOW = "\033[93m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


app = typer.Typer(
    name="CountTheBoats",
    help="Count the boats in the images",
    no_args_is_help=True,
)


@app.command()
def archive():
    """
    Archive the images in the downloads folder.
    """
    download_path = os.path.join("images", "downloads")
    coverage_path = os.path.join(cfg["output_dir"], "coverage.csv")
    ah.archive(download_path, coverage_path)
    print("Archive complete. Results in", cfg["output_dir"])


@app.command(
    help="Run the end to end classification pipeline automatically as per the config.yml file."
)
def auto(
    skip_order: bool = typer.Option(
        False, help="Skip ordering images (Just check existing orders)"
    ),
    clear: bool = typer.Option(
        True,
        help="Clean processed images to save space. Disable to keep tifs around for analysing in images/processed. If True, tifs are stil available in the archive.",
    ),
):
    """
    Run the entire pipeline automatically.
    """
    orders_path = os.path.join(cfg["output_dir"], "orders.csv")
    _ = ah.get_history(orders_path)  # make sure the file exists
    coverage_path = os.path.join(cfg["output_dir"], "coverage.csv")
    download_path = cfg["download_dir"]
    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])
    # For all AOIS
    history_len = None
    if cfg.get("HISTORY_LENGTH") is None:
        start_date = datetime.datetime.strptime(cfg["START_DATE"], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(cfg["END_DATE"], "%Y-%m-%d")
    else:
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
                if history_len is not None:
                    options, dates = ah.search(aoi, orders_path, days=history_len)
                else:
                    options, dates = ah.search(
                        aoi, orders_path, start_date=start_date, end_date=end_date
                    )
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
        # Clear processed images
        if clear:
            processed_dir = os.path.join(cfg["tif_dir"], "processed")
            if os.path.exists(processed_dir):
                for f in os.listdir(processed_dir):
                    if os.path.isfile(os.path.join(processed_dir, f)):
                        try:
                            os.remove(os.path.join(processed_dir, f))
                        except:
                            print(COLORS.FAIL, "Failed to remove", f, COLORS.ENDC)

    elif cfg["AUTO_MODE"] == "batch":  # --------------- BATCH MODE ----------------
        if history_len is not None:
            batch_mode(
                aois=aois,
                orders_path=orders_path,
                download_path=download_path,
                coverage_path=coverage_path,
                history_len=history_len,
            )
        else:
            batch_mode(
                aois=aois,
                orders_path=orders_path,
                download_path=download_path,
                coverage_path=coverage_path,
                start_date=start_date,
                final_date=end_date,
                clear=clear,
            )
    else:
        raise ValueError("Invalid AUTO_MODE in config")
    print("Auto complete. Results in", cfg["output_dir"])


def batch_mode(
    aois: list[str],
    orders_path: str,
    download_path: str,
    coverage_path: str,
    history_len: int | None = None,
    start_date: datetime.datetime | None = None,
    final_date: datetime.datetime | None = None,
    clear: bool = True,
):
    batch_size = cfg["BATCH_SIZE"]
    if history_len is not None:
        # set start date to today - history length
        start_date = datetime.datetime.now() - datetime.timedelta(
            days=(history_len - 1)
        )
        final_date = datetime.datetime.now()
        n_batches = math.ceil(history_len / batch_size)
    else:
        assert start_date is not None
        assert final_date is not None
        n_batches = math.ceil((final_date - start_date).days / batch_size)
    # end date of first batch
    end_date = start_date + datetime.timedelta(days=(batch_size - 1))
    if end_date > final_date:
        end_date = final_date

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
        final_date.strftime("%d/%m/%Y"),
        "(inclusive)",
    )
    time.sleep(3)
    # for each batch
    ordered_batches = np.zeros(n_batches)
    is_batch_ordered = lambda i: ordered_batches[i] == 1

    # start a comet experiment
    experiment = comet_ml.Experiment(project_name="count-the-boats")
    experiment.log_parameters(
        {
            "aois": aois,
            "batch_size": batch_size,
            "start_date": start_date.strftime("%d/%m/%Y"),
            "final_date": final_date.strftime("%d/%m/%Y"),
        }
    )
    experiment.log_parameters(cfg, prefix="cfg")

    for i in range(n_batches):
        print(
            COLORS.OKBLUE,
            f"Batch {i + 1}: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')} (inclusive)",
            COLORS.ENDC,
        )
        # search and order
        # Here we order the following batch, because orders take time to fulfil
        next_start_date = end_date + datetime.timedelta(days=1)
        next_end_date = next_start_date + datetime.timedelta(days=batch_size)
        if next_end_date > final_date:
            next_end_date = final_date
        print(COLORS.OKCYAN, "Searching and Ordering Images", COLORS.ENDC)
        for aoi in aois:
            if i == 0:  # First batch, order this and next batch
                batch_search_and_order(aoi, start_date, end_date, orders_path)
                print(
                    COLORS.OKGREEN,
                    "\t Searching and Ordering Images for Batch 2",
                    COLORS.ENDC,
                )
                batch_search_and_order(aoi, next_start_date, next_end_date, orders_path)
            elif i == n_batches - 1:  # Last batch, don't order anything
                pass
            else:  # order next batch
                batch_search_and_order(aoi, next_start_date, next_end_date, orders_path)
        # download the current batch
        print(COLORS.OKCYAN, "Downloading Images", COLORS.ENDC)
        batch_download_with_wait(orders_path, download_path, start_date, end_date)
        ah.extract(download_path, start_date, end_date)
        # classify -> We will save coverage during archive step.
        # get a list of the days in the batch in DD/MM/YYYY format
        days_in_batch = (
            pd.date_range(start_date, end_date).strftime("%d/%m/%Y").tolist()
        )
        print(COLORS.OKCYAN, "Classifying Downloads", COLORS.ENDC)
        ah.count(save_coverage=False, days=days_in_batch)
        # save
        print(COLORS.OKCYAN, "Marking as Complete", COLORS.ENDC)
        ah.save(orders_path, start_date=start_date, end_date=end_date)
        # archive -> also saves coverage file
        print(COLORS.OKCYAN, "Archiving ZIPS", COLORS.ENDC)
        ah.archive(
            download_path, coverage_path, start_date=start_date, end_date=end_date
        )
        # analyse batch
        print(COLORS.OKCYAN, "Analysing Batch", COLORS.ENDC)
        boat_csv_path = os.path.join(cfg["output_dir"], "boat_detections.csv")
        ah.analyse(
            boat_csv_path,
            coverage_path,
            start_date=start_date,
            end_date=end_date,
            id=f"batch_{i}",
            exp=experiment,
            batch=i,
        )
        # report on batch
        start_date = end_date + datetime.timedelta(days=1)
        end_date = start_date + datetime.timedelta(days=batch_size)
        if end_date > final_date:
            end_date = final_date
        if clear:
            print(COLORS.OKCYAN, "Clearing Processed Images", COLORS.ENDC)
            processed_dir = os.path.join(cfg["tif_dir"], "processed")
            if os.path.exists(processed_dir):
                for f in os.listdir(processed_dir):
                    if os.path.isfile(os.path.join(processed_dir, f)):
                        try:
                            os.remove(os.path.join(processed_dir, f))
                        except:
                            print(COLORS.FAIL, "Failed to remove", f, COLORS.ENDC)
        print(COLORS.OKGREEN, "Batch", i + 1, "complete", COLORS.ENDC)
        time.sleep(3)
    experiment.end()


def batch_search_and_order(aoi, start_date, end_date, orders_path):
    orders = pd.read_csv(orders_path)
    orders["date"] = pd.to_datetime(orders["date"])
    # if there are orders before and after this batch, don't order because probably already ordered
    if (
        len(orders.loc[(orders["date"] < start_date)]) > 0
        and len(orders.loc[(orders["date"] > end_date)]) > 0
    ):
        print(
            COLORS.OKGREEN,
            f"Orders already exist for {aoi} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')} skipping",
            COLORS.ENDC,
        )
        return

    options, dates = ah.search(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        orders_csv_path=orders_path,
    )
    if options is None:
        return
    # select
    orders = 0
    for items in ah.select(aoi, options, dates):
        oid = ah.order(aoi, items, orders_path)
        if not oid == "":
            orders += 1
    print(
        COLORS.OKGREEN,
        f"\t{orders} orders placed for {aoi} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}",
        COLORS.ENDC,
    )


def batch_download_with_wait(orders_path, download_path, start_date, end_date):
    # wait until all downloaded
    # Strategy  -> check every 10 minutes. If an order comes through, check every 5 minutes until all orders are downloaded
    #           -> if its been over 2 hours, exit and alert user something might be wrong
    # start date and end date are inclusive
    # download

    # first, check if we actually need to download anything
    orders = pd.read_csv(orders_path)
    orders["date"] = pd.to_datetime(orders["date"])
    orders = orders.loc[(orders["date"] >= start_date) & (orders["date"] <= end_date)]
    # possible if not marked as complete
    possible_orders = orders.loc[orders["order_status"] != "complete"]
    if len(possible_orders) == 0:
        print("No orders to download")
        return

    wait_time = 10 * 60  # 10 minutes
    total_wait_time = 0
    timeout = 2 * 60 * 60  # 2 hours
    ids = []
    remaining_orders: pd.DataFrame = ah.download(
        csv_path=orders_path,
        download_path=download_path,
        start_date=start_date,
        end_date=end_date,
    )
    remaining_orders["date"] = pd.to_datetime(remaining_orders["date"])
    start_date = pd.to_datetime(start_date).floor("D")
    end_date = pd.to_datetime(end_date).ceil("D")
    remaining_orders = remaining_orders.loc[
        (remaining_orders["date"] >= start_date)
        & (remaining_orders["date"] <= end_date)
    ]

    terminal_width = os.get_terminal_size().columns
    while len(remaining_orders) > 0:
        remaining_orders = ah.download(
            csv_path=orders_path,
            download_path=download_path,
            start_date=start_date,
            end_date=end_date,
        )
        remaining_orders["date"] = pd.to_datetime(remaining_orders["date"])
        remaining_orders = remaining_orders.loc[
            (remaining_orders["date"] >= start_date)
            & (remaining_orders["date"] <= end_date)
        ]
        if len(remaining_orders) == 0:
            break
        if total_wait_time > timeout:
            print(remaining_orders)
            print("Waited over 2 hours for orders to download. Exiting.")
            break
        string = "Waiting for %2d orders to download. Sleeping %3d seconds." % (
            len(remaining_orders),
            wait_time,
        )
        padding = terminal_width - len(string) - 8 - 3  # 8 = hh:mm:ss, 2 = 2 spaces
        padding_string = "=" * padding if padding > 0 else ""
        print(
            string,
            padding_string,
            datetime.datetime.now().strftime("%H:%M:%S"),  # hh:mm:ss = 8 characters
        )
        # print the order ids
        print("Order IDs:")
        print(remaining_orders["order_id"].to_string(index=False))
        time.sleep(wait_time)
        total_wait_time += wait_time
        wait_time = 5 * 60


@app.command()
def search(aoi: str, start_date: str, end_date: str, cloud_cover: str, area_cover: str):
    """
    Search for images on planet.
    """
    # TODO: implement search functionality
    raise NotImplementedError("Search cli not implemented yet.")


@app.command()
def order(items: str):
    """
    Order the given images from Planet.
    """
    # TODO: implement order functionality
    raise NotImplementedError("Order cli not implemented yet.")


@app.command()
def download(orders: str):
    """
    Downloads the images from the orders in the input file.
    """
    # TODO: implement download functionality
    raise NotImplementedError("Download cli not implemented yet.")


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
    else:
        raise NotImplementedError("Classify cli not implemented yet.")


if __name__ == "__main__":
    app()
