import json
import math
import os
import PIL
import PIL.Image
import pyproj
import re
import random
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
import tqdm

from PIL import Image
from osgeo import gdal

gdal.UseExceptions()

# allow pillow to open huge images
PIL.Image.MAX_IMAGE_PIXELS = None


class Classification(object):
    """
    This class is used to reference each of the classifications made using labelme that are stored in the output csv
    """

    def __init__(self, left, right, top, bottom, label):
        """
        Instantiate the object.

        Args:

            left: The left edge of the classification
            right: The right edge of the classification
            top: The top edge of the classification
            bottom: The bottom edge of the classification
            label: The label assigned to the classification
        """
        self._left = left
        self._right = right
        self._top = top
        self._bottom = bottom
        self._label = label

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def get_top(self):
        return self._top

    def get_bottom(self):
        return self._bottom

    def get_label(self):
        return self._label

    def in_bounds(self, left: float, right: float, top: float, bottom: float):
        """
        Check if this classification is within a larger bounding box defined by the parameters of this function

        Args:

            left: The left edge of the classification
            right: The right edge of the classification
            top: The top edge of the classification
            bottom: The bottom edge of the classification

        Returns
            True if the classification is within the bounding box specified; false otherwise.
        """
        if (
            self.get_bottom() < bottom
            and self.get_top() > top
            and self.get_left() > left
            and self.get_right() < right
        ):
            return True
        else:
            return False

    def serialise(self):
        return {
            "top": self.get_top(),
            "bottom": self.get_bottom(),
            "left": self.get_left(),
            "right": self.get_right(),
            "label": self.get_label(),
        }


def add_margin(
    pil_img: Image.Image, left: int, right: int, top: int, bottom: int, color: tuple
) -> Image.Image:
    """
    Pads an OPEN PIL (pillow/python imaging library) image on each edge by the amount specified.

    Args:

        pil_img: The open PIL image to pad.
        left: The number of pixels to pad the left edge of the image by.
        right: The number of pixels to pad the right edge of the image by.
        top: The number of pixels to pad the top edge of the image by.
        bottom: The number of pixels to pad the bottom edge of the image by.
        color: What colour the padding should be as a tuple (0, 0, 0) for black (255, 255, 255) for white.

    Returns:

        The open PIL image after the padding has been applied
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    print("New Width: ", new_width, "New Height: ", new_height)
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def process_sub_image_with_labels(args):
    """
    segments an image and its labels into tiles for use in a neural network.
    This function is intended to be used with multiprocessing.Pool

    Args:

        args (tuple): A tuple containing the following elements:
            - i (int): The row index of the sub-image.
            - j (int): The column index of the sub-image.
            - stride (int): The stride value used for sub-image extraction.
            - sub_image (np.ndarray): The sub-image array.
            - image (np.ndarray): The original image array.
            - allImageClassifications (list): List of all image classifications.
            - remove_empty (bool): Flag indicating whether to remove empty images.
            - im_outdir (str): Directory path to save processed images.
            - labels_outdir (str): Directory path to save labels.
            - empty_counter (multiprocessing.Value): Counter for empty images.
            - skipped_counter (multiprocessing.Value): Counter for skipped images.
            - progress (multiprocessing.Value): Progress counter.
            - lock (multiprocessing.Lock): Lock for synchronizing access to shared resources.

    Returns:

        None
    """
    (
        (
            i,
            j,
            stride,
            sub_image,
            image,
            allImageClassifications,
            remove_empty,
            im_outdir,
            labels_outdir,
        ),
        empty_counter,
        skipped_counter,
        progress,
        lock,
    ) = args
    sub_image = np.moveaxis(
        sub_image, 0, -1
    ).squeeze()  # need to move the channel axis to the end
    # Still remove the mostly empty images
    threshold = 0.75
    if np.sum(sub_image == 0) / sub_image.size > threshold:
        with lock:
            skipped_counter.value += 1
            progress.value += 1
        return

    # Create the image
    save_path = os.path.join(
        im_outdir, f"{os.path.basename(image).split('.')[0]}_{i}_{j}.png"
    )
    # get the bounding box of this image
    sizex = sub_image.shape[1]
    sizey = sub_image.shape[0]
    x1 = stride * j
    y1 = stride * i
    x2 = x1 + sizex
    y2 = y1 + sizey
    # Get the classifications in the image
    subsetClassifications = [
        classification
        for classification in allImageClassifications
        if classification.in_bounds(x1, x2, y1, y2)
    ]

    skipped = False
    if remove_empty > 0 and (
        subsetClassifications is None
        or subsetClassifications == []
        or all([c.get_label() == "tanker" for c in subsetClassifications])
    ):
        subsetClassifications = []
        if random.uniform(0, 1) < remove_empty:
            skipped = True
            with lock:
                empty_counter.value += 1
                progress.value += 1
                skipped_counter.value += 1
            return  # don't keep this image
        with lock:
            empty_counter.value += 1
    if skipped:
        raise Exception("This should not happen")
    # Definitely keeping this image, so make the image and save it
    out_image = Image.fromarray(sub_image)
    out_image.save(save_path, quality=100, compress_level=0)
    # write out the classifications for this sub-image
    label_path = os.path.join(
        labels_outdir, f"{os.path.basename(image).split('.')[0]}_{i}_{j}.txt"
    )
    outfile = open(label_path, "a+")
    for c in subsetClassifications:
        # if the classification is less than 3x3 pixels, expand it
        if c.get_right() - c.get_left() < 3:
            c._left -= 1
            c._right += 1
        if c.get_bottom() - c.get_top() < 3:
            c._top -= 1
            c._bottom += 1

        classLabel = (
            0 if c.get_label() == "Dry_WH" \
                else 1 if c.get_label() == "WH_swamp" \
                    else 2 if c.get_label() == "WH_wet" \
                        else 3 if c.get_label() == "WH_sink" \
                            else 4 if c.get_label() == "U" \
                                else -1
        )
        if classLabel == -1:
              continue  # Skip tankers #AF: we have no tankers but I also want to skip some introduced -1
        outfile.write(
            str(classLabel)
            + " "
            + str(((c.get_left() + c.get_right()) / 2 - j * stride) / sizex)
            + " "
            + str(((c.get_top() + c.get_bottom()) / 2 - i * stride) / sizey)
            + " "
            + str((c.get_right() - c.get_left()) / sizex) 
            #AF: removed the divison by two which could cause my boxes to be smaller than how I drew them 
            + " "
            + str((c.get_bottom() - c.get_top()) / sizey)
            #AF: removed the divison by two which could cause my boxes to be smaller than how I drew them
            + "\n"
        )
    outfile.close()
    with lock:
        progress.value += 1


def segment_image(
    image,
    json_file,
    tile_size,
    stride,
    metadata_components=None,
    remove_empty=0.9,
    im_outdir=None,
    labels_outdir=None,
):
    """
    Segments a large .tif file into smaller .png files for use in a neural network. Also created
    files in the IMGtxts directory which contain the annotations present in that sub image.

    Args:

        image: The large .tif that is to be segmented
        json_file: The json file that contains all the classifications for the image that were made in labelme.
        size: The desired size (both length and width) of the segmented images.
        overlap_size: The desired amount of overlap that the segmented images should have
        metadata_components: The metadata of the image - useful if the metadata is stripped out in a previous
            operation on the image/file.

    Returns:

        None
    """
    random.seed()
    # Open the image with tifffile - this reads the image in as a 2d array
    openImage = Image.open(image)

    # Get width and height by indexing the image array
    width, height = openImage.size
    width = int(width)
    height = int(height)

    # Open the json classifications file so create a Classification class object for each classification
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

    # Extract the classifications and nothing else
    classifications = data["shapes"]

    # Create an empty array which the Classification objects will be added to.
    allImageClassifications = []

    # Iterate through each classification and transform them into Classification class objects.
    for classification in classifications:
        try:
            x1, y1 = classification["points"][0]
            x2, y2 = classification["points"][1]
            top = round(min(y1, y2))
            left = round(min(x1, x2))
            bottom = round(max(y1, y2))
            right = round(max(x1, x2))
            label = classification["label"]
            allImageClassifications.append(
                Classification(left, right, top, bottom, label)
            )
        except:
            raise Exception(
                "Error in sorting classifications from JSON file, likely that one of the classifications is not a square"
            )

    # Ensure that the image is divisible by the desired size with no remainder.
    #if width % (stride + tile_size) != 0 or height % (stride + tile_size) != 0:
    if width % (tile_size) != 0 or height % (tile_size) != 0: # AF
        raise Exception(
            "The image is not exactly divisible by the desired size of subset images",
            "Width: ",
            width,
            "Height: ",
            height,
            "Stride: ",
            stride,
            "Tile Size: ",
            tile_size,
        )

    print("Cropping Image: " + image)

    image_array = np.array(openImage)
    image_shape = np.array(image_array.shape)
    print(image_shape)
    window_shape = np.array((tile_size, tile_size, 3)).reshape(-1)
    step = np.array((stride, stride, 1)).reshape(-1)
    image_stride = np.array(image_array.strides)
    shape = tuple((image_shape - window_shape) // step + 1) + tuple(window_shape)
    strides = tuple(image_stride * step) + tuple(image_stride)
    as_strided = np.lib.stride_tricks.as_strided
    sub_images = as_strided(image_array, shape=shape, strides=strides)
    print(
        "We will have: ", sub_images.shape[0] * sub_images.shape[1], " images maximum"
    )
    print(f"{remove_empty * 100}% of images without labels will be removed")

    # Flatten the 2d grid of sub-images into a list with indices (for parallel processing)
    sub_images_list = [
        (
            i,
            j,
            stride,
            sub_images[i, j],
            image,
            allImageClassifications,
            remove_empty,
            im_outdir,
            labels_outdir,
        )
        for i in range(sub_images.shape[0])
        for j in range(sub_images.shape[1])
    ]

    with Manager() as manager:
        skipped_counter = manager.Value("i", 0)
        empty_counter = manager.Value("i", 0)
        progress = manager.Value("i", 0)
        lock = manager.Lock()
        total = len(sub_images_list)
        with Pool(cpu_count()) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(
                    process_sub_image_with_labels,
                    [
                        (args, empty_counter, skipped_counter, progress, lock)
                        for args in sub_images_list
                    ],
                ),
                total=total,
                desc="Saving Segments",
            ):
                pass

        print(f"Skipped {skipped_counter.value} images")
        print(f"Empty {empty_counter.value} images")


def segment_image_for_classification_nosave(image, data_path, tile_size, stride):
    """
    Segments a large .tif file into smaller .png files for use in a neural network.
    Does not save the images to disk, but returns them as an array of views into 
    a larger numpy array.

    Args:

        image: The large .tif that is to be segmented
        tile_size: The desired size (both length and width) of the segmented images.
        data_path: The path to the directory where the segmented images will be saved.
        stride: The desired amount of overlap that the segmented images should have

    Returns:

        The images as an array with:
            i, j, sub_image, image, data_path
    """
    # Open the image with pillow - this reads the image in as a 2d array
    openImage = Image.open(image)

    # Get width and height by indexing the image array
    width, height = openImage.size
    width = int(width)
    height = int(height)

    print(width, height)

    # Ensure that the image is divisible by the desired size with no remainder.
    if width % stride != 0 or height % stride != 0:
        raise Exception(
            "The image is not exactly divisible by the desired size of subset images"
        )

    # Ensure that the desired size is divisible by the desired overlap with no remainder.
    if tile_size % stride != 0:
        raise Exception(
            "The subset image size indicated is not divisible by the input overlap size"
        )

    # open the image as a numpy array
    image_array = np.array(openImage)
    # use np.lib.stride_tricks.sliding_window_view to create a sliding window view of the image
    # this will allow us to extract the sub-images
    image_shape = np.array(image_array.shape)
    print(image_shape)
    window_shape = np.array((tile_size, tile_size, 1)).reshape(-1)
    step = np.array((stride, stride, 1)).reshape(-1)
    image_stride = np.array(image_array.strides)
    shape = tuple((image_shape - window_shape) // step + 1) + tuple(window_shape)
    strides = tuple(image_stride * step) + tuple(image_stride)
    as_strided = np.lib.stride_tricks.as_strided
    sub_images = as_strided(image_array, shape=shape, strides=strides)

    print(sub_images.shape)

    # flatten the 2d grid of sub-images into a list with indices (for parallel processing)
    sub_images_list = [
        (i, j, sub_images[i, j], image, data_path)
        for i in range(sub_images.shape[0])
        for j in range(sub_images.shape[1])
    ]
    return sub_images_list


def process_sub_image(args):
    """
    segments an image into tiles for use in a neural network.
    This function is intended to be used with multiprocessing.Pool

    Args:

        args (tuple): A tuple containing the following elements:
            - (i, j, sub_image, image, data_path) (tuple): 
                - i (int): The row index of the sub-image.
                - j (int): The column index of the sub-image.
                - sub_image (np.ndarray): The sub-image array.
                - image (str): The path to the original image file.
                - data_path (str): The directory path to save processed images.
            - skipped_counter (multiprocessing.Value): Counter for skipped images.
            - progress (multiprocessing.Value): Progress counter.
            - lock (multiprocessing.Lock): Lock for synchronizing access to shared resources.

    Returns:

        None
    """
    (i, j, sub_image, image, data_path), skipped_counter, progress, lock = args
    sub_image = np.moveaxis(sub_image, 0, -1).squeeze()
    if np.all(sub_image == 0):
        with lock:
            skipped_counter.value += 1
            progress.value += 1
        return
    out_image = Image.fromarray(sub_image)
    save_path = os.path.join(
        data_path, f"{os.path.basename(image).split('.')[0]}_{i}_{j}.png"
    )
    out_image.save(save_path)

    with lock:
        progress.value += 1


def segment_image_for_classification(image, data_path, tile_size, stride):
    """
    Segments a large .tif file into smaller .png files for use in a neural network.

    Args:

        image: The large .tif that is to be segmented
        tile_size: The desired size (both length and width) of the segmented images.
        data_path: The path to the directory where the segmented images will be saved.
        stride: The desired amount of overlap that the segmented images should have

    Returns:

        None
    """
    # Open the image with pillow - this reads the image in as a 2d array
    openImage = Image.open(image)

    # Get width and height by indexing the image array
    width, height = openImage.size
    width = int(width)
    height = int(height)

    print(width, height)

    # Ensure that the image is divisible by the desired size with no remainder.
    if width % stride != 0 or height % stride != 0:
        raise Exception(
            "The image is not exactly divisible by the desired size of subset images"
        )

    # Ensure that the desired size is divisible by the desired overlap with no remainder.
    if tile_size % stride != 0:
        raise Exception(
            "The subset image size indicated is not divisible by the input overlap size"
        )

    # open the image as a numpy array
    image_array = np.array(openImage)
    # use np.lib.stride_tricks.sliding_window_view to create a sliding window view of the image
    # this will allow us to extract the sub-images
    image_shape = np.array(image_array.shape)
    print(image_shape)
    window_shape = np.array((tile_size, tile_size, 1)).reshape(-1)
    step = np.array((stride, stride, 1)).reshape(-1)
    image_stride = np.array(image_array.strides)
    shape = tuple((image_shape - window_shape) // step + 1) + tuple(window_shape)
    strides = tuple(image_stride * step) + tuple(image_stride)
    as_strided = np.lib.stride_tricks.as_strided
    sub_images = as_strided(image_array, shape=shape, strides=strides)

    print(sub_images.shape)

    # flatten the 2d grid of sub-images into a list with indices (for parallel processing)
    sub_images_list = [
        (i, j, sub_images[i, j], image, data_path)
        for i in range(sub_images.shape[0])
        for j in range(sub_images.shape[1])
    ]

    with Manager() as manager:
        skipped_counter = manager.Value("i", 0)
        progress = manager.Value("i", 0)
        lock = manager.Lock()
        total = len(sub_images_list)
        with Pool(cpu_count()) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(
                    process_sub_image,
                    [
                        (args, skipped_counter, progress, lock)
                        for args in sub_images_list
                    ],
                ),
                total=total,
                desc="Saving Segments",
            ):
                pass
        print(f"Skipped {skipped_counter.value} images")

def create_padded_png(
    raw_dir, output_dir, file_name, tile_size=416, stride=104, rename=False
):
    """
    Creates an image which is padded for use in training/classifying within a neural network. Note: this images pads
    the images expecting that the size of sub-images created from this images will be 416x416 pixels.

    Args:

        raw_dir: Directory where raw .tif files downloaded from Planet are located
        output_dir: The name of the directory where the padded .png files will be created.
        file_name: The name of the .tif file that is to be padded.
        tile_size: The size of the sub-images that will be created from this image.
        stride: The amount of overlap that the sub-images will have.
        rename: The name that the padded image will be saved as. If False, the image will be saved as the original name but with a .png extension.

    Returns:

        None
    """
    rawImageDirectory = os.path.join(os.getcwd(), raw_dir)
    os.makedirs(output_dir, exist_ok=True)
    PNGpath = output_dir

    savePath = os.path.join(PNGpath, file_name.split(".")[0] + ".png")
    if os.path.exists(savePath):
        return

    # Set the options for the gdal.Translate() call
    opsString2 = (
        "-ot UInt16 -of png -b 3 -b 2 -b 1 -scale_1 0 2048 0 65535 -scale_2 0 2048 0 65535 -scale_3 0 2048 0 "
        "65535"
    )
    # Translate original from tif into png
    print("Doing gdal work...")
    gdal.Translate(
        os.path.join(PNGpath, f"colorCorrected{file_name.split('.')[0]}.png"),
        os.path.join(rawImageDirectory, file_name),
        options=opsString2,
    )
    print("\rDone with gdal work for " + file_name)

    # Open the new color corrected PNG we have just made.
    colorImagePath = os.path.join(
        PNGpath, f"colorCorrected{file_name.split('.')[0]}.png"
    )
    colourImage = PIL.Image.open(colorImagePath)

    # Get new width and height in preparation of padding the image
    width, height = colourImage.size

    leftPad, rightPad, topPad, bottomPad = get_required_padding(colorImagePath, tile_size, stride)

    # Pad the edges (add_margin is a function in imageCuttingSupport.py that adds a margin to an opened PIL image.)
    im_new = add_margin(
        colourImage,
        leftPad,
        rightPad,
        topPad,
        bottomPad,
        (0, 0, 0),
    )

    # Save the padded image which is now ready for classification
    if rename != False and type(rename) == str:
        savePath = os.path.join(PNGpath, rename + ".png")
    else:
        savePath = os.path.join(PNGpath, file_name.split(".")[0] + ".png")
    im_new.save(savePath, quality=100, compress_level=0)

    # Close loose file descriptors
    colourImage.close()
    im_new.close()

    # Cleanup any images that aren't the image that is used for classification
    for filename in os.listdir(PNGpath):
        if filename[0:5] == "color":
            os.remove(os.path.join(PNGpath, filename))


def create_padded_png_S2(
    raw_dir, output_dir, file_name, tile_size=416, stride=104, rename=False
):
    """
    Creates an image which is padded for use in training/classifying within a neural network. Note: this images pads
    the images expecting that the size of sub-images created from this images will be 416x416 pixels.

    Args:

        raw_dir: Directory where raw .tif files downloaded from Planet are located
        output_dir: The name of the directory where the padded .png files will be created.
        file_name: The name of the .tif file that is to be padded.
        tile_size: The size of the sub-images that will be created from this image.
        stride: The amount of overlap that the sub-images will have.
        rename: The name that the padded image will be saved as. If False, the image will be saved as the original name but with a .png extension.

    Returns:

        None
    """
    rawImageDirectory = os.path.join(os.getcwd(), raw_dir)
    os.makedirs(output_dir, exist_ok=True)
    PNGpath = output_dir

    savePath = os.path.join(PNGpath, file_name.split(".")[0] + ".png")
    if os.path.exists(savePath):
        return

    # Set the options for the gdal.Translate() call
    opsString2 = (
        # for planet data
        # "-ot UInt16 -of png -b 3 -b 2 -b 1 -scale_1 0 2048 0 65535 -scale_2 0 2048 0 65535 -scale_3 0 2048 0 65535"
        # for sentinel-2 data
        # for 0 to 255 scaling
        # "-ot UInt16 -of png -b 3 -b 2 -b 1 -scale_1 0 255 0 65535 -scale_2 0 255 0 65535 -scale_3 0 255 0 65535"
        # for 80 to 255 scaling (to give higher contrast based on min and max values of the sentinel-2 bands)
        "-ot UInt16 -of png -b 1 -b 2 -b 3 -scale_1 80 255 0 65535 -scale_2 80 255 0 65535 -scale_3 80 255 0 65535"
    )
    # Translate original from tif into png
    print("Doing gdal work...")
    gdal.Translate(
        os.path.join(PNGpath, f"colorCorrected{file_name.split('.')[0]}.png"),
        os.path.join(rawImageDirectory, file_name),
        options=opsString2,
    )
    print("\rDone with gdal work for " + file_name)

    # Open the new color corrected PNG we have just made.
    colorImagePath = os.path.join(
        PNGpath, f"colorCorrected{file_name.split('.')[0]}.png"
    )
    colourImage = PIL.Image.open(colorImagePath)

    # Get new width and height in preparation of padding the image
    width, height = colourImage.size

    leftPad, rightPad, topPad, bottomPad = get_required_padding(colorImagePath, tile_size, stride)

    # Pad the edges (add_margin is a function in imageCuttingSupport.py that adds a margin to an opened PIL image.)
    im_new = add_margin(
        colourImage,
        leftPad,
        rightPad,
        topPad,
        bottomPad,
        (0, 0, 0),
    )

    # Save the padded image which is now ready for classification
    if rename != False and type(rename) == str:
        savePath = os.path.join(PNGpath, rename + ".png")
    else:
        savePath = os.path.join(PNGpath, file_name.split(".")[0] + ".png")
    im_new.save(savePath, quality=100, compress_level=0)

    # Close loose file descriptors
    colourImage.close()
    im_new.close()

    # Cleanup any images that aren't the image that is used for classification
    for filename in os.listdir(PNGpath):
        if filename[0:5] == "color":
            os.remove(os.path.join(PNGpath, filename))


def create_unpadded_png(raw_dir, output_dir, file_name):
    """
    Creates an image which is NOT padded but converts raw .tif to .png files.

    Args:

        raw_dir: Directory where raw .tif files downloaded from Planet are located
        output_dir: The name of the directory where the padded .png files will be created.
        file_name: The name of the .tif file that is to be padded.

    Returns:

        None
    """
    rawImageDirectory = os.path.join(os.getcwd(), raw_dir)
    PNGpath = output_dir

    # Set the options for the gdal.Translate() call
    opsString2 = (
        "-ot UInt16 -of png -b 3 -b 2 -b 1 -scale_1 0 2048 0 65535 -scale_2 0 2048 0 65535 -scale_3 0 2048 0 "
        "65535"
    )
    # Translate original from tif into png
    gdal.Translate(
        os.path.join(PNGpath, file_name.split(".")[0] + ".png"),
        os.path.join(rawImageDirectory, file_name),
        options=opsString2,
    )


def get_required_padding(filepath, tilesize=416, stride=104) -> tuple[int, int, int, int]:
    """
    Determines how much padding is required on each edge of an image so that the image lengths and widths will be
    divisible by 416 and that each part of the images will be seen by the neural networm 16 times in either training
    or classification.

    Args:

        filepath: The path of the file to evaluate.
        tilesize: The size of the sub-images that will be created from this image.
        stride: The amount of overlap that the sub-images will have.

    Returns:

        (L, R, T, B) - A tuple containing the padding required on the left (L), right (R), top (T), and bottom (b)
            of the images to make it suitable for use in the neural network.
    """
    ds = gdal.Open(filepath)

    metadata = gdal.Info(ds)
    ds = None
    metadata_components = metadata.split("\n")

    # Get new width and height in preparation of paddingthe image
    size = filter(lambda x: "Size is" in x, metadata_components)
    if not size:
        print(metadata)
        raise Exception("Size is not in the metadata")
    size = list(size)[0]
    height = int(size.split(",")[1])
    width = int(size.split(",")[0].split(" ")[-1])

    # Calculate individual padding for each edge
    # we have to go at least (tilesize/stride) * stride pixels in each direction
    # in the case of 416/104, this is 4*104 = 416 padding. This means the first tile is all padding but the rest will be seen by the network properly
    min_pad = (tilesize - stride)

    leftPad = min_pad
    topPad = min_pad
    bottomPad = min_pad
    rightPad = min_pad

    min_h = height + 2 * min_pad
    min_w = width + 2 * min_pad
    # ensure the image will be divisible by the tile size and stride
    if min_h % tilesize != 0:
        topPad += tilesize - (min_h % tilesize)
    if min_w % tilesize != 0:
        leftPad += tilesize - (min_w % tilesize)
    
    # ensure integer padding
    leftPad = int(leftPad)
    rightPad = int(rightPad)
    topPad = int(topPad)
    bottomPad = int(bottomPad)


    return leftPad, rightPad, topPad, bottomPad


def get_crs(filepath: str) -> int:
    """
    Get the EPSG code of the coordinate reference system of a .tif file.

    Args:

        filepath: The path of the file to evaluate.

    Returns:

        The EPSG code of the coordinate reference system of the .tif file.
    """
    ds = gdal.Open(filepath)
    metadata = gdal.Info(ds, format="json")
    ds = None
    try:
        crs = metadata["stac"]["proj:projjson"]["id"]["code"]
        return crs
    except:
        raise Exception(
            "The coordinate reference system does not exist in the metadata"
        )


def pixel2coord(x: int, y: int, original_image_path: str) -> tuple[float, float]:
    """
        Returns global coordinates to pixel center using base-0 raster index

        Args:

            x: The x coordinate (pixel coordinates) of the object in the image to be converted to global coordinates.
            y: The y coordinate (pixel coordinates) of the object in the image to be converted to global coordinates.
            original_image_path: The path of the file to evaluate - a .tif files should be located here. This file will
                also need geospatial metadata. Images obtained from Planet have the required metadata.
    e
        Returns:

            (xp, yp) - A tuple containing the global coordinates of the provided pixel coordinates.
    """
    ds = gdal.Open(original_image_path)
    c, a, b, f, d, e = ds.GetGeoTransform()
    xp = a * x + b * y + a * 0.5 + b * 0.5 + c
    yp = d * x + e * y + d * 0.5 + e * 0.5 + f
    return (xp, yp)


def coord2pixel(
    x: float, y: float, original_image_path: str
) -> tuple[int, int] | tuple[list[int], list[int]]:
    """
    Returns pixel coordinates to pixel center using base-0 raster index

    Args:

    
        x: The x coordinate (global coordinates) of the object in the image to be converted to pixel coordinates.
        y: The y coordinate (global coordinates) of the object in the image to be converted to pixel coordinates.
        original_image_path: The path of the file to evaluate - a .tif files should be located here. This file will
            also need geospatial metadata. Images obtained from Planet have the required metadata.

    Returns:


        (xp, yp) - A tuple containing the pixel coordinates of the provided global coordinates.
    """
    ds = gdal.Open(original_image_path)
    c, a, b, f, d, e = ds.GetGeoTransform()
    xp = (x - b * y - a * 0.5 - b * 0.5 - c) / a
    yp = (y - d * x - d * 0.5 - e * 0.5 - f) / e
    return (xp, yp)


def coord2latlong(x: float, y: float, crs: int = 32756) -> tuple[float, float]:
    """
    Converts global coordinates to latitude/longitude coordinates

    Args:

        x: The x coordinate in a pair of global coordinates.
        y: The y coordinate in a pair of global coordinates.
        crs: The coordinate reference system of the global coordinates.

    Returns:

        (long, lat) - A tuple containing the longitude and latitude at the provided global coordinates.
    """
    proj = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    long, lat = proj.transform(x, y)
    return long, lat


def latlong2coord(lat, long, crs: int = 32756) -> tuple[float, float]:
    """
    Converts latitude/longitude coordinates to global coordinates

    Args:

        long: The longitude coordinate in a pair of latitude/longitude coordinates.
        lat: The latitude coordinate in a pair of latitude/longitude coordinates.

    Returns:

        (x, y) - A tuple containing the global coordinates at the provided latitude/longitude coordinates.
    """
    proj = pyproj.Transformer.from_crs(4326, crs)
    x, y = proj.transform(long, lat)
    return x, y


def get_date_from_filename(filename: str):
    """
        Get the date from a file. The file must have a date in the format "yyyymmdd_" at the start of the filename .
    first
        Args:

            filename: The name of the file to extract the date from.

        Returns:

            The date in the format "dd/mm/yyyy" from the filename.
    """
    str_date = filename.split("_")[0]
    if len(str_date) != 8:
        return None
    year = str_date[0:4]
    month = str_date[4:6]
    day = str_date[6:8]
    return f"{day}/{month}/{year}"


def get_cartesian_top_left(metadata_components: list[str]) -> tuple[float, float]:
    """
    Calculates the lat long at the top left point of a given satellite image

    Args:

        metadata_components: The metadata components stripped from a .tif file

    Returns:

        (long, lat) - A tuple containing the longitude and latitude at the top left point of the satellite image.
    """
    for component in metadata_components:
        if len(component) > 20 and component[0:10] == "Upper Left":
            longdms, latdms = component.split("(")[-1][0:-1].split(", ")
            longd = longdms.split("d")[0]
            longm = longdms.split("d")[1].split("'")[0]
            longs = longdms.split('"')[0].split("'")[1]
            longdirec = longdms.split('"')[1]
            latd = latdms.split("d")[0]
            latm = latdms.split("d")[1].split("'")[0]
            lats = latdms.split('"')[0].split("'")[1]
            latdirec = latdms.split('"')[1]
            long = ((int(longd) * 60 * 60) + (int(longm) * 60) + float(longs)) / (
                60 * 60
            )
            lat = ((int(latd) * 60 * 60) + (int(latm) * 60) + float(lats)) / (60 * 60)
            if longdirec == "S":
                long = long * -1
            if latdirec == "W":
                lat = lat * -1
            return long, lat
        else:
            pass
    raise Exception("The top left corner coordinates do not exist in the metadata")


def get_cartesian_top_right(metadata_components):
    """
    Calculates the lat long at the top right point of a given satellite image
    :param metadata_components: The metadata components stripped from a .tif file
    :return: (long, lat) - A tuple containing the longitude and latitude at the top right point of the satellite image.
    """
    for component in metadata_components:
        if len(component) > 20 and component[0:11] == "Upper Right":
            longdms, latdms = component.split("(")[-1][0:-1].split(", ")
            longd = longdms.split("d")[0]
            longm = longdms.split("d")[1].split("'")[0]
            longs = longdms.split('"')[0].split("'")[1]
            longdirec = longdms.split('"')[1]
            latd = latdms.split("d")[0]
            latm = latdms.split("d")[1].split("'")[0]
            lats = latdms.split('"')[0].split("'")[1]
            latdirec = latdms.split('"')[1]
            long = ((int(longd) * 60 * 60) + (int(longm) * 60) + float(longs)) / (
                60 * 60
            )
            lat = ((int(latd) * 60 * 60) + (int(latm) * 60) + float(lats)) / (60 * 60)
            if longdirec == "S":
                long = long * -1
            if latdirec == "W":
                lat = lat * -1
            return long, lat
        else:
            pass
    raise Exception("The top right corner coordinates do not exist in the metadata")


def get_cartesian_bottom_left(metadata_components: list[str]) -> tuple[float, float]:
    """
    Calculates the lat long at the bottom left point of a given satellite image

    Args:

        metadata_components: The metadata components stripped from a .tif file

    Returns:

        (long, lat) - A tuple containing the longitude and latitude at the bottom left point of the satellite image
    """
    for component in metadata_components:
        if len(component) > 20 and component[0:10] == "Lower Left":
            longdms, latdms = component.split("(")[-1][0:-1].split(", ")
            longd = longdms.split("d")[0]
            longm = longdms.split("d")[1].split("'")[0]
            longs = longdms.split('"')[0].split("'")[1]
            longdirec = longdms.split('"')[1]
            latd = latdms.split("d")[0]
            latm = latdms.split("d")[1].split("'")[0]
            lats = latdms.split('"')[0].split("'")[1]
            latdirec = latdms.split('"')[1]
            long = ((int(longd) * 60 * 60) + (int(longm) * 60) + float(longs)) / (
                60 * 60
            )
            lat = ((int(latd) * 60 * 60) + (int(latm) * 60) + float(lats)) / (60 * 60)
            if longdirec == "S":
                long = long * -1
            if latdirec == "W":
                lat = lat * -1
            return long, lat
        else:
            pass
    raise Exception("The bottom left corner coordinates do not exist in the metadata")


def get_cartesian_bottom_right(metadata_components: list[str]) -> tuple[float, float]:
    """
    Calculates the lat long at the bottom right point of a given satellite image

    Args:

        metadata_components: The metadata components stripped from a .tif file

    Returns:

        (long, lat) - A tuple containing the longitude and latitude at the bottom right point of the
            satellite image.
    """
    for component in metadata_components:
        if len(component) > 20 and component[0:11] == "Lower Right":
            longdms, latdms = component.split("(")[-1][0:-1].split(", ")
            longd = longdms.split("d")[0]
            longm = longdms.split("d")[1].split("'")[0]
            longs = longdms.split('"')[0].split("'")[1]
            longdirec = longdms.split('"')[1]
            latd = latdms.split("d")[0]
            latm = latdms.split("d")[1].split("'")[0]
            lats = latdms.split('"')[0].split("'")[1]
            latdirec = latdms.split('"')[1]
            long = ((int(longd) * 60 * 60) + (int(longm) * 60) + float(longs)) / (
                60 * 60
            )
            lat = ((int(latd) * 60 * 60) + (int(latm) * 60) + float(lats)) / (60 * 60)
            if longdirec == "S":
                long = long * -1
            if latdirec == "W":
                lat = lat * -1
            return long, lat
        else:
            pass
    raise Exception("The bottom right corner coordinates do not exist in the metadata")


def metadata_get_w_h(metadata_components: list[str]) -> tuple[int, int] | None:
    """
    Obtains the width and height of a given satellite image

    Args:

        metadata_components: The metadata components stripped from a .tif file

    Returns:

        (width, height) - A tuple containing the width and height of a given satellite image.
    """
    for component in metadata_components:
        if component[0:4] == "Size":
            width = int(component.split(" ")[2].split(",")[0])
            height = int(component.split(" ")[3])
            return width, height
    return None
