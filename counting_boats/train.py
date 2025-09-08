"""
Script for training pipeline for a model
Can be easier to just use command line for training, but this script contains
the other steps required - preparing the images, segmenting them, and describing the dataset

Author: Charlie Turner
Date: 16/09/2024

Addition from AF:
Adapted and modified some section of Charlie's original repository to match our requirements. 
I mainly made sure it ran smoothly and correctly to have the whole pipeline to train a model. 
Added two functions 
1. reorganize_folders() {moves segmented and selected images+labels to training folder} and 
2. cull_AF() {faster than the original cull()} 
Date: 28/02/2025
"""

import typer
import os
import yaml
import numpy as np
from  .boat_utils import image_cutting_support as ics

app = typer.Typer()


def parse_config(config: str) -> dict:
    """
    Parse the config file

    Args:
        config (str): path to the config file

    Returns:
        dict: the parsed config file
    """
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@app.command()
def prepare(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Prepare the TIFF images for labelling by converting them to PNGs
    Does this for all TIF images in the raw_images folder specified in the config

    Args:
        config (str): path to the config

    Returns:
        None
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
def prepare_S2(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Prepare the TIFF images for labelling by converting them to PNGs
    Does this for all TIF images in the raw_images folder specified in the config

    Args:
        config (str): path to the config

    Returns:
        None
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
        ics.create_padded_png_S2(
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
    Segment the images and labels into tiles for training.
    Also split the images into training and validation sets.

    Args:
        config (str): path to the config file
        train_val_split (float): proportion of images to use for training

    Returns:
        None
    """
    
    cfg = parse_config(config)
    print(cfg) #AF
    # segment the images that have been prepared
    labels = [
        os.path.join(cfg["output_dir"], f)
        for f in os.listdir(cfg["output_dir"])
        if f.endswith(".json")
    ]
    print(labels)
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
    print("Config path:", imdirs)
    num_images = 0
    num_tiles = 0
    num_labels = 0
    class_counts = {}
    num_tiles_no_labels = 0
    imdirs=os.path.normpath(imdirs) #AF: handeled the itteration problem on the path of the folder. 
    
    for imdir in imdirs:
        imdir=os.path.normpath(imdirs)
        if not os.path.exists(imdir): #AF
            raise FileNotFoundError(f"Directory not found: {imdir}") #AF
        if os.path.exists(os.path.join(imdir, "train")):
            imdir = os.path.join(imdir, "train")
        all_images = [i for i in os.listdir(imdir) if i.endswith(".png")]
        # unique_images = set([im[0 : im.find("_", 9)] for im in all_images]) #AF
        unique_images = set([im[:im.replace('_', 'X', 1).find('_')] for im in all_images if im.count('_') >= 2])        #modified the unique image detection to match all possible naming of AOIs. Extract everzthing before the third _
        num_images = len(unique_images)
        num_tiles = len(all_images)
        # get the number of labels
        a = os.path.join(imdir, "..", "..", "labels", "train")
        b = os.path.join(imdir, "..", "labels")
        labdir = a if os.path.exists(a) else b
        all_labels = [l for l in os.listdir(labdir) if l.endswith(".txt")]
        # count number of lines in all the files
        # for lab in all_labels:
        #     with open(os.path.join(labdir, lab), "r") as f:
        #         lines = f.readlines()
        #         num_labels += len(lines)
        #         if len(lines) == 0:
        #             num_tiles_no_labels += 1
        #         for line in lines:
        #             class_id = line.split(" ")[0]
        #             if class_id not in class_counts:
        #                 class_counts[class_id] = 1
        #             else:
        #                 class_counts[class_id] += 1
        #New version of it, AF:
        # count number of lines in all the files
        for lab in all_labels:
            lab_path = os.path.join(labdir, lab)

            # Check if file is empty first (file size = 0)
            if os.path.getsize(lab_path) == 0:
                num_tiles_no_labels += 1
                continue
                
            with open(lab_path, "r") as f:
                lines = f.readlines()
                num_labels += len(lines)
                
                # Only check content if file isn't empty                             
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
    print(f"| Number of original images: {num_images}            |")
    print(f"| Number of tiles: {num_tiles}                   |")
    print(f"| Number of labels: {len(all_labels)}                  |")
    print(f"| Number of individual labels             |")
    print(f"|   - Total: {num_labels}                         |")
    print(f"|   - Per class:                      |")
    for class_id, count in class_counts.items():
        print(
            f"|       - Class {class_id}: {count} ({count/num_labels*100:.2f}%)        |"
        )
    print(
        f"| Background Images: {num_tiles_no_labels} ({num_tiles_no_labels/num_tiles*100:.2f}%)      |"
    )
    print("-" * 43)
    return num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels


@app.command()
def cull(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    YOLO recommends having 10% of images in the training set with no instances. 
    We can't know how many tiles will have no instances before we segment the images,
    so have to cull down after. This function will remove images with no labels until
    10% of the training set has no labels

    Args:
        config (str): path to the config file

    Returns:
        None

    AF comment: 
        Function was developped with absolute folder paths, which complicated the task. 
        Need to modify them in here too, not only in the config file. 
        Notes as comments are the function bellow to help navigate it. 

    """
    num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels = describe(
        config
    )
    # remove images with no labels from the training set
    # need to cull enough images so that the percentage of images with no labels is 10%
    all_labels = [
        l for l in os.listdir("training/labels/train") if l.endswith(".txt")
    ]
    while num_tiles_no_labels / num_tiles > 0.1:
        # find an empty label file
        # shuffle the list of labels
        np.random.shuffle(all_labels)
        for lab in all_labels:
            remove = False
            with open(os.path.join("training/labels/train", lab), "r") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    # delete the corresponding image and label file
                    remove = True
            if remove:
                if os.path.exists(
                    os.path.join(
                        "training/images/train",
                        lab.replace(".txt", ".png"),
                    )
                ):
                    os.remove(
                        os.path.join(
                            "training/images/train",
                            lab.replace(".txt", ".png"),
                        )
                    )
                os.remove(os.path.join("training/labels/train", lab))
                num_tiles_no_labels -= 1
                num_tiles -= 1
                # remove lab from all_labels
                all_labels.remove(lab)
                break
    num_images, num_tiles, num_labels, class_counts, num_tiles_no_labels = describe(
        config
    )


@app.command()
def cull_AF(
    config_path: str = typer.Option("", help="Path to the config file"),
):
    """
    Addition from AF.
    YOLO recommends having 10% of images in the training set with no instances. 
    We can't know how many tiles will have no instances before we segment the images,
    so have to cull down after. This function will remove images with no labels until
    10% of the training set has no labels. 
    AF: I am modifying Charlie's code to run the culling in a different way which should make it faster. 
    
    Identify empty label files and move them (along with corresponding images)
    to ensure empty labels make up only 10% of the total dataset.
    
    Args:
        config_path: Path to the YAML config file containing paths


    Returns:
        None

 

    """
    import sys
    import os
    import yaml
    import random
    import shutil
    from pathlib import Path 

    cfg = parse_config(config_path)
    # get the number of original images
    labels_dir = Path(cfg["segmented_labels"])
    images_dir = Path(cfg["segmented_images"])
    
    # Create directories for moved files if they don't exist
    moved_labels_dir = labels_dir.parent / 'moved_empty_labels'
    moved_images_dir = images_dir.parent / 'moved_empty_images'
    os.makedirs(moved_labels_dir, exist_ok=True)
    os.makedirs(moved_images_dir, exist_ok=True)
    
    print(f"Analyzing label files in: {labels_dir}")
    print(f"Looking for corresponding images in: {images_dir}")
    
    # Get all text files in the labels directory
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    total_labels = len(label_files)
    
    if total_labels == 0:
        print("No label files found. Exiting.")
        return
    
    # Identify empty label files
    empty_label_files = []
    non_empty_count = 0
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if the file is empty
        if os.path.getsize(label_path) == 0:
            empty_label_files.append(label_file)
        else:
            non_empty_count += 1
    
    print(f"Total label files found: {total_labels}")
    print(f"Empty label files found: {len(empty_label_files)}")
    print(f"Non-empty label files: {non_empty_count}")
    
    # Calculate how many empty files to keep
    # We want empty files to be 10% of total, so 11.11...% of non-empty files
    max_empty_to_keep = int(non_empty_count * 0.1111)
    empty_to_move = len(empty_label_files) - max_empty_to_keep
    
    if empty_to_move <= 0:
        print("Number of empty labels is already below 10% threshold. No action needed.")
        return
    
    # Shuffle empty label files
    random.shuffle(empty_label_files)
    
    # Select files to move
    files_to_move = empty_label_files[:empty_to_move]
    
    print(f"Moving {len(files_to_move)} empty label files to maintain 10% ratio")
    
    # Track moved files
    moved_labels = []
    moved_images = []
    
    # Move the selected label files and their corresponding images
    for label_file in files_to_move:
        # Get corresponding image file
        base_name = os.path.splitext(label_file)[0]
        image_file = f"{base_name}.png"
        
        # Check if corresponding image exists
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found for label: {label_file}")
            continue
        
        # Move label file
        label_src = os.path.join(labels_dir, label_file)
        label_dst = os.path.join(moved_labels_dir, label_file)
        shutil.move(label_src, label_dst)
        moved_labels.append(label_file)
        
        # Move image file
        image_src = os.path.join(images_dir, image_file)
        image_dst = os.path.join(moved_images_dir, image_file)
        shutil.move(image_src, image_dst)
        moved_images.append(image_file)
    
    # Verify result
    remaining_labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    remaining_empty = 0
    
    for label_file in remaining_labels:
        label_path = os.path.join(labels_dir, label_file)
        if os.path.getsize(label_path) == 0:
            remaining_empty += 1
    
    remaining_total = len(remaining_labels)
    empty_percentage = (remaining_empty / remaining_total) * 100 if remaining_total > 0 else 0
    
    print("\n--- SUMMARY ---")
    print(f"Total label files moved: {len(moved_labels)}")
    print(f"Total image files moved: {len(moved_images)}")
    print(f"Remaining total label files: {remaining_total}")
    print(f"Remaining empty label files: {remaining_empty}")
    print(f"Empty labels now make up {empty_percentage:.2f}% of the dataset")
    print(f"Empty labels moved to: {moved_labels_dir}")
    print(f"Corresponding images moved to: {moved_images_dir}")
    
    if empty_percentage <= 10:
        print("\nSUCCESS: Empty labels now make up 10% or less of the dataset.")
    else:
        print("\nWARNING: Something went wrong. Empty labels still exceed 10% of the dataset.")




def reorganize_folders(config_file: str = typer.Option("", help="Path to the config file")):
    """
    Addition from AF.
    Reorganizes folders from {output_dir}/images and {output_dir}/labels 
    to {training_path}/train and {training_path}/val.
    
    Structure transformation:
    - {output_dir}/images/val -> {training_path}/val/images
    - {output_dir}/images/train -> {training_path}/train/images
    - {output_dir}/labels/val -> {training_path}/val/labels
    - {output_dir}/labels/train -> {training_path}/train/labels
    
    Args:
        config_file (str): Path to the YAML config file containing paths
    """
    import os
    import shutil
    import yaml
    from pathlib import Path 

    
    cfg = parse_config(config_file)
    # get the number of original images
    output_dir = Path(cfg["output_dir"])
    training_path = Path(cfg["training_path"])
      
    
    # Define source and destination paths
    source_images = output_dir / "images"
    source_labels = output_dir / "labels"
    
    dest_train = training_path / "train"
    dest_val = training_path / "val"
    
    # Create destination directories if they don't exist
    dest_train.mkdir(parents=True, exist_ok=True)
    dest_val.mkdir(parents=True, exist_ok=True)
    
    # Dictionary mapping source directories to destination directories
    operations = {
        source_images / "val": dest_val / "images",
        source_images / "train": dest_train / "images",
        source_labels / "val": dest_val / "labels",
        source_labels / "train": dest_train / "labels"
    }
    
    successful_operations = 0
    failed_operations = 0
    
    # Perform the copy operations
    for source, destination in operations.items():
        try:
            # Check if source exists
            if not source.exists():
                print(f"Source directory not found: {source}")
                failed_operations += 1
                continue
                
            # Remove destination if it exists
            if destination.exists():
                print(f"Removing existing destination: {destination}")
                shutil.rmtree(destination)
                
            # Copy directory
            print(f"Copying from {source} to {destination}")
            shutil.copytree(source, destination)
            print(f"Successfully copied to {destination}")
            successful_operations += 1
            
        except Exception as e:
            print(f"Error while copying {source} to {destination}: {e}")
            failed_operations += 1
    
    # Print summary
    print("\n----- REORGANIZATION SUMMARY -----")
    print(f"Successful operations: {successful_operations}")
    print(f"Failed operations: {failed_operations}")
    
    if successful_operations == 4:
        print("\nAll folders were successfully reorganized!")
        print("\nNew structure:")
        print(f"- {training_path}/val/images (copied from {output_dir}/images/val)")
        print(f"- {training_path}/val/labels (copied from {output_dir}/labels/val)")
        print(f"- {training_path}/train/images (copied from {output_dir}/images/train)")
        print(f"- {training_path}/train/labels (copied from {output_dir}/labels/train)")
    else:
        print("\nSome operations failed. Please check the error messages above.")

if __name__ == "__main__":
    print("Starting folder reorganization...")
    reorganize_folders()
    print("Process completed.")





@app.command()
def train(
    config: str = typer.Option("", help="Path to the config file"),
):
    """
    Train the model

    Args:
        config (str): path to the config file

    Returns:
        None
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
