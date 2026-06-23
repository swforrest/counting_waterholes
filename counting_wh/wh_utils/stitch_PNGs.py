import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import re

"""
Given the training image grids, stitch them back together into one image

Hopefully never need to do this ever again

Author: Charlie Turner
Date: 20/02/2024
"""


def stitch(dir, output_dir=None):
    """
    Stitches back together images that were segmented into smaller images.
    Handles multiple images in the same directory, differentiating them by their name.
    
    Image naming format should be: 'YYYYMMDD_area_xx_yy.png',
    where area is the name of the area of interest, xx is the row and yy is the column.
    
    The stitched image will be named 'YYYYMMDD_area_stitched.png'.
    
    Args:
        dir (str): directory containing images to stitch
        output_dir (str, optional): directory to save the stitched images. 
                                   If None, saves to input_dir

    Returns:
        None
    
    Comment:
    AF: Modified the above function to plot some waterholes.
    """
    # If output_dir is not provided, use input_dir
    if output_dir is None:
        output_dir = dir

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of files in directory
    all_imgs = [f for f in os.listdir(dir) if f.endswith(".png") and not f.endswith("_stitched.png")]
    
    # Pattern to match date and area name: YYYYMMDD_area
    pattern = r"^(\d{8}_[^_]+)"
    
    # Group images by prefix (date and area name)
    image_groups = {}
    for img in all_imgs:
        match = re.match(pattern, img)
        if match:
            prefix = match.group(1)
            if prefix not in image_groups:
                image_groups[prefix] = []
            image_groups[prefix].append(img)
    
    # List to store paths of stitched images
    stitched_image_paths = []
    
    # Process each group of images
    for prefix, imgs in image_groups.items():
        # Check if stitched image already exists in output directory
        stitched_filename = f"{prefix}_stitched.png"
        stitched_path = os.path.join(output_dir, stitched_filename)
        
        if stitched_filename in os.listdir(output_dir):
            print(f"Already stitched, output exists at {stitched_path}")
            stitched_image_paths.append(stitched_path)
            continue
            
        # Extract coordinates for each image
        coords = []
        for img in imgs:
            parts = img.split('_')
            # The last two parts are row and column
            if len(parts) >= 4:
                try:
                    y = int(parts[-2])  # row
                    x = int(parts[-1].split('.')[0])  # column
                    coords.append((img, x, y))
                except ValueError:
                    print(f"Skipping {img}: could not parse coordinates")
        
        if not coords:
            print(f"No valid images found for {prefix}")
            continue
        
        # Get the maximum x and y values
        max_x = max([x[1] for x in coords])
        max_y = max([x[2] for x in coords])
        print(f"Group {prefix} - Max x: {max_x}, Max y: {max_y}")
        
        # Create a new image with the appropriate size
        # Images are 416x416 with 104 pixels overlap
        width = 416 + (max_x * (416 - 104))
        height = 416 + (max_y * (416 - 104))
        stitched_image = Image.new("RGB", (width, height))
        
        # For each image, paste it into the new image at the correct position
        for img, x, y in coords:
            img_path = os.path.join(dir, img)
            try:
                current_img = Image.open(img_path)
                # Calculate position considering overlap
                pos_x = x * (416 - 104)
                pos_y = y * (416 - 104)
                stitched_image.paste(current_img, (pos_x, pos_y))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save the stitched image to the output directory
        stitched_image.save(stitched_path)
        stitched_image_paths.append(stitched_path)
        print(f"Saved stitched image to {stitched_path}")
    
    return stitched_image_paths


if __name__ == "__main__":
    main()


def main():
    # get the current directory
    dir = input("Enter the directory of images to stitch: ")
    # recursively search the directory for a directory containing images
    # if there is a directory, add to list
    dirs = []
    for root, subdirs, files in os.walk(dir):
        if len(files) > 0 and files[0].endswith(".png"):
            dirs.append(root)
    # for each directory, stitch the images
    for dir in dirs:
        stitch(dir)


def stitch_AF(dir, output_dir=None, prefix=None):
    """
    for each png in the directory, pull out the x, y coords
    images have names like imagessdfasdf_x_y.png
    They are all 416x416, and overlap 104 pixels.
    need to stich them back together into one image

    Args:
        dir (str): directory containing images to stitch
        output_dir (str, optional): directory to save the stitched image. 
                                   If None, saves to input_dir

    Returns:
        None

    Comment: 
    AF: modified to give an output directory of my choice. 
    """
    # If output_dir is not provided, use input_dir
    if output_dir is None:
        output_dir = dir

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the output filename based on prefix
    if prefix:
        output_filename = f"{prefix}_stitched.png"
    else:
        # Use directory name as prefix if none provided
        dir_name = os.path.basename(dir)
        output_filename = f"{dir_name}_stitched.png"
    
    output_path = os.path.join(output_dir, output_filename)

    # Check if stitched image already exists in output directory
    if os.path.exists(output_path):
        print(f"Already stitched, output exists at {output_path}")
        return output_path
    
    # get list of files in directory
    imgs = [f for f in os.listdir(dir) if f.endswith(".png")]
    
    if not imgs:
        print(f"No PNG images found in {dir}")
        return None
    
    # for each file, get the x and y coords
    coords = []
    for img in imgs:
        try:
            parts = img.split("_")
            if len(parts) >= 2:
                # Assume the last two parts are y and x.png
                y = int(parts[-2])
                x = int(parts[-1].split(".")[0])
                coords.append((img, x, y))
        except (ValueError, IndexError) as e:
            print(f"Error parsing coordinates from filename {img}: {e}")
    
    if not coords:
        print(f"No valid image coordinates found in {dir}")
        return None
    
    # get the maximum x and y values
    max_x = max([x[1] for x in coords])
    max_y = max([x[2] for x in coords])
    print("Max x: {}, Max y: {}".format(max_x, max_y))
    
    # we want to use every 4th of each from 0-max_x and 0-max_y
    image = Image.new("RGB", (104 * (max_x + 1), 104 * (max_y + 1)))
    
    # for each image, paste it into the new image
    [
        image.paste(Image.open(os.path.join(dir, img)), (x * 104, y * 104))
        for img, x, y in coords
    ]
    
    # Save the stitched image to the output directory
    image.save(output_path)
    print(f"Saved stitched image to {output_path}")
    
    return output_path