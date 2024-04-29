# Counting Boats
## What
This project counts small marine vessels from satellite imagery of the Moreton Bay 
region. The counts are recorded and can be analysed or presented later. Tools for 
visualising training and inference data are also available.

## How

This project utilises satellite images, and harnesses machine learning
object detection to count small marine vessels (boats) in the Moreton Bay area.
Extendable to any images from any area, the reccommended pipeline runs as follows:

1. Using Planet, satellite images of the area of interest are automatically ordered for recent dates
2. Once the orders are available, imagery is automatically downloaded from planet.
3. A preprocessing pipeline prepares imagery for detection
4. Our YOLOv5 model detects and labels both stationary and moving boats in the images
5. We collate and analyse the boat counts as time-series data, outputting a CSV of detected boats and their coordinates

## Usage

### Installation

#### Yolov5

Clone [YoloV5](https://github.com/ultralytics/yolov5). This is used for the Neural Network detections.

#### Python Dependencies
It's reccommended to install a conda-based package manager such as [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
Running the following will then install all required dependencies:

```
conda create --name CountingBoats --file environment.yaml
```

Activate the environment (if not already) with `conda activate CountingBoats`, and you should be good to go.

### Setup

#### Configuration
Set the variables in `config.yaml` to align with your environment and preferences.
Similarly for 'config_train.yaml' and 'config_inference' in `config.yaml`.

### Running

From the root directory, run the following commands:

#### Training
```
python counting_boats/train.py {prepare|segment|train} --config config_train.yaml
```

#### Testing
```
python counting_boats/testing.py --config config_test.yaml
```

#### Deployment

```
python counting_boats/classify.py --config config_classify.yaml
```


## Acknowledgements

