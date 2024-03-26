import os
import argparse
import typer
import utils.planet_utils as pu
import utils.classifier as cl
from config import cfg
from CountTheBoats import archive

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

app = typer.Typer()

@app.command()
def search(aoi: str, start_date: str, end_date: str, cloud_cover: str, area_cover: str):
    """
    Search for images on planet.
    Usage: search {aoi} {start_date} {end_date} {cloud_cover} {area_cover}
    """
    # TODO: implement search functionality
    pass

@app.command()
def order(items: str):
    """
    Order the given images from Planet.
    Usage: order {items.txt} | {item1} {item2} {item3}
    """
    # TODO: implement order functionality
    pass

@app.command()
def download(orders: str):
    """
    Downloads the images from the orders in the input file.
    Usage: download {orders.txt} | {order1} {order2} {order3}
    """
    # TODO: implement download functionality
    pass

@app.command()
def classify(images: str):
    """
    Run the classifier on the given images.
    Usage: classify {images.txt} | {image1.png} {image2.png} | {zip1.zip} {zip2.zip}
    """
    # TODO: implement classify functionality
    pass

# error message if no command is given
@app.callback()
def callback():
    typer.echo("Welcome to CountTheBoats! Please provide a command.")


if __name__ == "__main__":
    app()

