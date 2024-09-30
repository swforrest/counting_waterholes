# Counting Boats

## What

This project counts small marine vessels from satellite imagery of the Moreton Bay
region. The counts are recorded and can be analysed or presented later. Tools for
visualising training and inference data are also available.

## How

This project utilises satellite images, and harnesses machine learning
object detection to count small marine vessels (boats) in the Moreton Bay area.
Extendable to any images from any area, the recommended pipeline runs as follows:

1. Using Planet, satellite images of the area of interest are automatically ordered for recent dates
2. Once the orders are available, imagery is automatically downloaded from planet.
3. A pre-processing pipeline prepares imagery for detection
4. Our YOLOv5 model detects and labels both stationary and moving boats in the images
5. We collate and analyse the boat counts as time-series data, outputting a CSV of detected boats and their coordinates

## Usage

### Installation

#### Yolov5

Clone [YoloV5](https://github.com/ultralytics/yolov5). This is used for the Neural Network detections.

#### Python Dependencies

It's recommended to install a conda-based package manager such as [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
Running the following will then install all required dependencies:

```
conda env create --file env.yaml
```

Activate the environment (if not already) with `conda activate Boats`, and you should be good to go.

### Setup

#### Configuration

Set the variables in `config.yaml` to align with your environment and preferences.
Similarly for `config_train.yaml` or `config_test.yaml` for training and testing respectively.

### Running

From the root directory, run the following commands:

#### Training

```
python -m counting_boats.train {prepare|segment|train} --config config_train.yaml
```

#### Testing

```
python -m counting_boats.testing --config config_test.yaml
```

By altering `config_test.yaml`, you can change the test data and test tasks that are run.

#### Deployment

```
python counting_boats.classify auto
```

### Visualisation

There are some visualisation notebooks in the visualisation folder. These can be run to perform some visualisations of the data.
The `plot_output` script is also a useful tool for visualising the output of the detection model on individual images. Run:

```
python -m counting_boats.plot_output --detections {path_to_detections} { --image {path_to_image} | --zip {path_to_zip} }
```

If you have an image already e.g 'AOI_date.tif', use the `--image` flag. If you want to run on a zip file straight from Planet, use the `--zip` flag.
e.g.

```

python -m counting_boats.plot_output --detections Results/boat_detections.csv --zip "U:\Research\Projects\sef\livingplayingmb\Boat Detection TMBF\PlanetArchive\moreton_20171106.zip"

```

## Acknowledgements
