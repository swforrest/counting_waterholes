"""
Script for training pipeline for a model
"""

import typer
import os
import yaml
import numpy as np
from utils import image_cutting_support as ics

app = typer.Typer()


def parse_config(config: str):
    """
    Parse the config file
    """
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@app.command()
def prepare(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Prepare the TIFF images for labelling
    """
    cfg = parse_config(config)
    # Create the directory
    os.makedirs(cfg["output_dir"], exist_ok=True)
    # find all the tif files that we want
    # from cfg["raw_images"] folder
    tif_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(cfg["raw_images"])
        for f in files
        if f.endswith(".tif")
    ]
    # use ics to convert to padded pngs (the padding is specific to the segment size and stride that will be used)
    for i, tif in enumerate(tif_files):
        tif_dir = os.path.dirname(tif)
        tif_name = os.path.basename(tif)
        ics.create_padded_png(
            tif_dir, cfg["output_dir"], tif_name, cfg["TILE_SIZE"], cfg["STRIDE"]
        )
        print(f"Processed {i+1}/{len(tif_files)}", end="\r")
    print(f"Processed {len(tif_files)}/{len(tif_files)} images")


@app.command()
def segment(
    config: str = typer.Option("", help="Path to the config file"),
    train_val_split: float = typer.Option(
        0.8, help="Proportion of images to use for training"
    ),
):
    """
    Segment the images
    """
    cfg = parse_config(config)
    # segment the images that have been prepared
    labels = [
        os.path.join(cfg["output_dir"], f)
        for f in os.listdir(cfg["output_dir"])
        if f.endswith(".json")
    ]
    images = [l.replace(".json", ".png") for l in labels]
    label_out = os.path.join(cfg["output_dir"], "labels")
    image_out = os.path.join(cfg["output_dir"], "images")
    os.makedirs(label_out, exist_ok=True)
    os.makedirs(image_out, exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        ics.segment_image(
            image,
            label,
            cfg["TILE_SIZE"],
            cfg["STRIDE"],
            im_outdir=image_out,
            labels_outdir=label_out,
        )
    # split the images into training and validation
    os.makedirs(os.path.join(image_out, "train"), exist_ok=True)
    os.makedirs(os.path.join(image_out, "val"), exist_ok=True)
    os.makedirs(os.path.join(label_out, "train"), exist_ok=True)
    os.makedirs(os.path.join(label_out, "val"), exist_ok=True)
    for file in os.listdir(image_out):
        if not file.endswith(".png"):
            continue
        im = os.path.join(image_out, file)
        lab = os.path.join(label_out, file.replace(".png", ".txt"))
        rand = np.random.rand()
        if rand < train_val_split:
            os.rename(im, os.path.join(image_out, "train", os.path.basename(im)))
            os.rename(lab, os.path.join(label_out, "train", os.path.basename(lab)))
        else:
            os.rename(im, os.path.join(image_out, "val", os.path.basename(im)))
            os.rename(lab, os.path.join(label_out, "val", os.path.basename(lab)))


@app.command()
def train(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Train the model
    """
    cfg = parse_config(config)
    # train the model on the images in cfg["output_dir"]
    # have to use system calls to train yolov5
    command = f"{cfg['python']} {cfg['yolo_dir']}/train.py --device cuda:0 \
--img {cfg['TILE_SIZE']} --batch {cfg['BATCH_SIZE']} \
--workers {cfg['workers']} \
--epochs {cfg['EPOCHS']} --data {config} \
--weights {cfg['weights']} --save-period 50 --patience=0"
    # print command in yellow:
    print(f"\033[93m{command}\033[0m")
    os.system(command)


if __name__ == "__main__":
    app()
