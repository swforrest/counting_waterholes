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
    if train_val_split == 1:
        return
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
def describe(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Work out and display:
        - Number of original images (unique filenames excluding locations)
        - Number of tiles
        - Number of labels total
        - Number of labels per class
        - Number of images with no labels

    Each also as a percentage of total
    e.g
    -------------------------------------------
    | Training dataset statistics             |
    -------------------------------------------
    | Number of original images: 10           |
    | Number of tiles: 100                    |
    | Number of labels
    |   - Total: 122                          |
    |   - Per class:                          |
    |       - Class 1: 50 (41.67%)            |
    |       - Class 2: 70 (58.33%)            |
    | Number of images with no labels: 2 (3%) |
    -------------------------------------------
    | Validation dataset statistics ...
    """
    cfg = parse_config(config)
    # get the number of original images
    imdirs = cfg["train"]
    num_images = 0
    num_tiles = 0
    num_labels = 0
    class_counts = {}
    num_tiles_no_labels = 0

    for imdir in imdirs:
        if os.path.exists(os.path.join(imdir, "train")):
            imdir = os.path.join(imdir, "train")
        print(imdir)
        all_images = [i for i in os.listdir(imdir) if i.endswith(".png")]
        unique_images = set([im[0 : im.find("_", 9)] for im in all_images])
        num_images += len(unique_images)
        num_tiles += len(all_images)
        # get the number of labels
        a = os.path.join(imdir, "..", "..", "labels", "train")
        b = os.path.join(imdir, "..", "labels")
        labdir = a if os.path.exists(a) else b
        all_labels = [l for l in os.listdir(labdir) if l.endswith(".txt")]
        # count number of lines in all the files
        for lab in all_labels:
            with open(os.path.join(labdir, lab), "r") as f:
                lines = f.readlines()
                num_labels += len(lines)
                if len(lines) == 0:
                    num_tiles_no_labels += 1
                for line in lines:
                    class_id = line.split(" ")[0]
                    if class_id not in class_counts:
                        class_counts[class_id] = 1
                    else:
                        class_counts[class_id] += 1
    # print the statistics
    print("-" * 43)
    print("| Training dataset statistics             |")
    print("-" * 43)
    print(f"| Number of original images: {num_images:<13}|")
    print(f"| Number of tiles: {num_tiles:<23}|")
    print(f"| Number of labels                        |")
    print(f"|   - Total: {num_labels:<29}|")
    print(f"|   - Per class:                          |")
    for class_id, count in class_counts.items():
        print(
            f"|       - Class {class_id}: {count:<6} ({count/num_labels*100:.2f}%)        |"
        )
    print(
        f"| Background Images: {num_tiles_no_labels:<6} ({num_tiles_no_labels/num_tiles*100:.2f}%)      |"
    )
    print("-" * 43)
    return num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels


@app.command()
def cull(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Remove images with no labels
    """
    num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels = describe(
        config
    )
    # remove images with no labels from the training set
    # need to cull enough images so that the percentage of images with no labels is 10%
    all_labels = [
        l for l in os.listdir("active_learning/data/labels") if l.endswith(".txt")
    ]
    while num_tiles_no_labels / num_tiles > 0.1:
        # find an empty label file
        # shuffle the list of labels
        np.random.shuffle(all_labels)
        for lab in all_labels:
            remove = False
            with open(os.path.join("active_learning/data/labels", lab), "r") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    # delete the corresponding image and label file
                    remove = True
            if remove:
                if os.path.exists(
                    os.path.join(
                        "active_learning/data/images",
                        lab.replace(".txt", ".png"),
                    )
                ):
                    os.remove(
                        os.path.join(
                            "active_learning/data/images",
                            lab.replace(".txt", ".png"),
                        )
                    )
                os.remove(os.path.join("active_learning/data/labels", lab))
                num_tiles_no_labels -= 1
                num_tiles -= 1
                # remove lab from all_labels
                all_labels.remove(lab)
                break
    num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels = describe(
        config
    )


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
--weights {cfg['weights']} --save-period 50"
    # print command in yellow:
    print(f"\033[93m{command}\033[0m")
    os.system(command)


if __name__ == "__main__":
    app()
