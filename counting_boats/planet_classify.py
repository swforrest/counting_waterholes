import datetime
import os
import argparse
import time
import typer
import utils.planet_utils as pu
import utils.classifier as cl
import utils.auto_helpers as ah
from utils.config import cfg

"""
Ideally this becomes the one file to rule them all, currently CountTheBoats is the main file
But is composed of lots of functions that could be in utils

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
    skip_order: bool = typer.Option(False, help="Skip ordering images"),
):
    """
    Run the entire pipeline automatically.
    """
    orders_path     = os.path.join(cfg["output_dir"], "AOI_history.csv")
    archive_path    = os.path.join(cfg["output_dir"], "coverage.csv")
    download_path   = os.path.join("images", "downloads")
    # For all AOIS
    if not skip_order:
        print("Ordering images")
        # pause for 3 seconds
        time.sleep(3)
        aois = pu.get_aois()
        for aoi in aois:
            options, dates = ah.search(aoi, orders_path)
            if options is None:
                continue
            for items in ah.select(aoi, options, dates):
                ah.order(aoi, items, orders_path )
    ah.download(csv_path=orders_path, download_path=download_path)
    ah.extract(csv_path=orders_path, download_path=download_path)
    ah.count()    # Count the boats
    ah.save(orders_path)                 # Save the history
    ah.archive(download_path, archive_path) # Archive the raw data (and save coverage info)
    # analyse
    ah.analyse(orders_path, archive_path)
    # report (per run, and overall)


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

