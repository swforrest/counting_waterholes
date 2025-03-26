# Counting Waterholes Project

## What

This project aims to to determine the locations and assess the health of waterholes in Northern Australia, which are susceptible to damage from invasive herbivores such as water buffalo, feral cattle and pigs. 
Satellite imagery of the Northern Territories region is used to train a classification algorithm. 
The counts are recorded and can be analysed or presented later. Tools for
visualising training and infered detection data are also available.

Based on a pipeline to detect and count boats and moving boats in the Moreton Bay area, we attapted the pipeline to waterholes. The [CountingBoats](https://github.com/charlie-turner-314/CountingBoats) repository was originally designed as a whole pipeline developped to detect boats only. For our workflow, we diverged away from the usage of modules and we use notebooks. One notebook is dedicated to a specific section of the whole process. This provides you with more clarity, control on what you run, and allows you to order single functions or commands. 


## How

This project utilises satellite images, and harnesses machine learning
object detection to detect waterholes in the Arnhem Land and Cape York area.
Extendable to any images from any area, the recommended workflow runs as follows:

1. Using Planet, satellite images of the area of interest are automatically ordered for recent dates. 
2. Once the orders are available, imagery is automatically downloaded from planet.
3. A pre-processing pipeline prepares imagery for detection. 
4. Our YOLOv5 pre-trained model detects and classifies waterholes in your processed images. 
5. We collate and analyse the waterholes counts, outputting a CSV of detected locations, classes and coordinates. 


## Usage

### Installation

#### Yolov5

Clone [YoloV5](https://github.com/ultralytics/yolov5). This is used for the Neural Network detections.

#### Python Dependencies

It's recommended to install a conda-based package manager such as [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).  

Running the following will then install all required dependencies (run only once to set the environment up):

```
conda env create --file env.yaml
```

### Setup

To start working on this project, activate the environment with:

```
conda activate Boats
```

#### Configurations

Modify the paths and variables in `config_train_Drive.yaml` to align with your environment and preferences as described in the file. This will allow you to train your own model from your images and area(s) of interest.  
Following a training proces, to test your model, modify the paths and variables in `config_test_Drive.yaml`.   
Finally if you prefer runing the detection of our model directly on your images, you can modify the paths and variables in `config_deploy_Drive.yaml`. 

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
python -m counting_boats.classify auto
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

python -m counting_boats.plot_output --detections "U:\Research\Projects\sef\livingplayingmb\Boat Detection TMBF\BoatDetectionResults\boat_detections.csv" --zip "U:\Research\Projects\sef\livingplayingmb\Boat Detection TMBF\PlanetArchive\moreton_20171106.zip"

```

## Acknowledgements
