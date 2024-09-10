import numpy as np
import pandas as pd
import boat_utils.heatmap as hm
import json
from boat_utils.config import cfg
import os
import boat_utils.image_cutting_support as ics
from PIL import Image, ImageDraw
import zipfile

import matplotlib.pyplot as plt


def aoi_of_point(lat, lon, aois):
    """
    given a lat/lon, find the aoi it is in.
    @param lat: latitude of point
    @param lon: longitude of point
    @param aois: list of polygon files
    """
    # NOTE: Doesn't do anything yet
    return "peel"


def get_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["aoi"] = df.apply(
        lambda x: aoi_of_point(x["latitude"], x["longitude"], None), axis=1
    )
    return df


def get_coverage(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")


"""
Plotting Functions:
    1. Total count (per AOI) per week (bar chart)
    2. Total count (per AOI) per day of week (bar chart)
    3. Heatmap of total count on satellite image
    4. Weighted count (weighted by frequency of area coverage)
"""


def plot_total_count_per_week(df):
    """
    Plot the count of boats each week of the year for all years in teh data
    """
    fig = plt.figure()
    years = df.date.dt.year.unique()
    axs = fig.subplots(len(years), 1, sharex=True)
    for i, year in enumerate(years):
        df[df.date.dt.year == year].groupby(df.date.dt.isocalendar().week).size().plot(
            kind="bar", color="b", alpha=0.5, marker="o", ax=axs[i]
        )
        axs[i].set_title(year)
    fig = plt.gcf()
    fig.suptitle("Total Boats Detected Per Week")
    fig.tight_layout()
    plt.show()


def plot_avg_count_per_week(df):
    """
    Plot the average count of boats per week of the year for all given data.
    """
    fig = plt.figure()
    df["week"] = df.date.dt.isocalendar().week
    df = (
        df.groupby(["week", "aoi"])
        .size()
        .reset_index(name="count")
        .groupby("week")
        .mean(numeric_only=True)
    )
    df.plot(kind="line", color="b", alpha=0.5, marker="o")
    fig = plt.gcf()
    fig.suptitle("Average Boats Detected Per Week")
    fig.tight_layout()
    plt.show()


def plot_avg_count_per_dayofweek(df):
    """
    Plot the average count of boats per day of the week for all given data.
    """
    fig = plt.figure()
    df["dayofweek"] = df.date.dt.dayofweek
    n = df.date.unique().dayofweek
    days, count = np.unique(n, return_counts=True)
    # ensure we have 0-6
    for i in range(7):
        if i not in days:
            days = np.insert(days, i, i)
            count = np.insert(count, i, 0)

    df = (
        df.groupby(["dayofweek", "aoi"])
        .size()
        .reset_index(name="count")
        .groupby("dayofweek")
        .mean(numeric_only=True)
        .sort_index()
    )
    df["n"] = count
    # use day names with n e.g 'Monday (n=5)'
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df.index = [f"{day} (n={n})" for day, n in zip(days, count)]
    df.plot(kind="line", color="b", alpha=0.5, marker="o", y="count")
    fig = plt.gcf()
    fig.suptitle("Average Boats Detected Per Day of the Week")
    fig.tight_layout()
    plt.show()


def map_scatter(df):
    """
    Plot a scatter plot of the total count of boats on a satellite image
    """
    df["class"] = df["class"].astype(str)
    df["class"] = df["class"].replace("0", "Stationary")
    df["class"] = df["class"].replace("1", "Moving")

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="class",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=10,
    )

    fig.update_layout(mapbox_style="open-street-map")
    return fig


def chloro_coverage(fig, geojson_file):
    """
    Add coverage grid from geojson to an exisitng figure
    """
    geojson_file = "/Users/charlieturner/Documents/CountingBoats/heatmap.geojson"
    with open(geojson_file) as f:
        geojson = json.load(f)

    df = pd.DataFrame(geojson["features"])
    df["id"] = df["properties"].apply(lambda x: x["id"])
    df["value"] = df["properties"].apply(lambda x: x["value"])
    df = df[["id", "value"]]
    df = df[df["value"] > 0]
    # make value a string
    df["value"] = df["value"].astype(str)
    print(df["value"].unique())

    mapbox = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="id",
        featureidkey="properties.id",
        color="value",
        mapbox_style="open-street-map",
        center={"lat": -27.5, "lon": 153.1},
        zoom=5,
        opacity=0.3,
    )
    # merge with the scatter plot
    fig.update_traces(
        marker=dict(size=5, opacity=0.5), selector=dict(type="scattermapbox")
    )
    for trace in mapbox.data:
        fig.add_trace(trace)
    # remove legend colorbar
    fig.update_layout(
        coloraxis_colorbar=dict(
            yanchor="top",
            y=1,
            x=0,
            ticks="outside",
        )
    )
    return fig


def create_and_show_heatmap(coverage_file, data):
    fig = map_scatter(data)
    # create the heatmap
    polygons = hm.get_polygons_from_file(coverage_file)
    hm.create_heatmap_from_polygons(
        polygons, os.path.join(cfg["output_dir"], "coverage_heatmap.tif")
    )
    geojson = os.path.join(cfg["output_dir"], "coverage_heatmap.geojson")
    fig = chloro_coverage(fig, geojson)
    fig.show()


def plot_boats():
    all_boats = pd.read_csv(os.path.join(cfg["output_dir"], "boat_detections.csv"))
    all_boats["date"] = pd.to_datetime(all_boats["date"], dayfirst=True)
    contents = [p for p in os.listdir("Testing")]
    # extract any in "zips" that haven't been
    for z in contents:
        if z.endswith(".zip") and not os.path.exists(f"Testing/{z[:-4]}"):
            print(f"Extracting {z}")
            with zipfile.ZipFile(f"Testing/{z}", "r") as zip_ref:
                zip_ref.extractall(f"Testing/{z[:-4]}")
    folders = [f for f in os.listdir("Testing") if os.path.isdir(f"Testing/{f}")]
    for folder in folders:
        print(f"Processing {folder}")
        # create the png (if it doesn't exist)
        if not os.path.exists(f"Testing/{folder}/{folder}.png"):
            print("Creating PNG")
            ics.create_unpadded_png(
                f"Testing/{folder}", f"Testing/{folder}", "composite.tif"
            )
            os.rename(
                f"Testing/{folder}/composite.png", f"Testing/{folder}/{folder}.png"
            )
        # plot the boats onto the png
        if os.path.exists(f"Testing/{folder}/{folder}_boats.png"):
            print("Boats already plotted")
            continue
        date = folder.split("_")[1]
        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        date = f"{day}/{month}/{year}"
        day_boats = all_boats[all_boats["date"] == date]
        xcoords, ycoords = ics.latlong2coord(
            day_boats["longitude"], day_boats["latitude"]
        )
        x, y = ics.coord2pixel(xcoords, ycoords, f"Testing/{folder}/composite.tif")
        # assert x and y are iterable
        assert type(x) == np.ndarray and type(y) == np.ndarray
        image = Image.open(f"Testing/{folder}/{folder}.png")
        image = image.convert("RGBA")
        newImage = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(newImage)
        # for each boat, draw a square 10 pixels wide, 50% opacity
        # 50-70% confidence = red, 70-90% = yellow, 90-100% = green
        colors = [
            (
                255,
                0,
                0,
            ),
            (255, 255, 0, 128),
            (0, 255, 0, 128),
        ]
        for i in range(len(x)):
            confidence = day_boats.iloc[i]["confidence"]
            color = (
                colors[0]
                if confidence < 0.7
                else colors[1] if confidence < 0.9 else colors[2]
            )
            draw.rectangle(
                [x[i] - 5, y[i] - 5, x[i] + 5, y[i] + 5],
                fill=None,
                width=1,
                outline=color,
            )
        out = Image.alpha_composite(image, newImage)
        out.save(f"Testing/{folder}/{folder}_boats.png")


if __name__ == "__main__":
    # detections_file = input("Enter the detections file path: ")
    # coverage_file = input("Enter the coverage file path: ")
    # detections = get_data(detections_file)
    plot_boats()
    # all_boats = pd.read_csv(os.path.join(cfg["output_dir"], "boat_detections.csv"))
    # all_boats["date"] = pd.to_datetime(all_boats["date"], dayfirst=True).dt.date.astype(
    #     str
    # )
    # for d in all_boats["date"].unique():
    #     print(d)
