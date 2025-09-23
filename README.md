# Counting Waterholes Project

## What

This project aims to detect and assess the locations of waterholes in Northern Australia using satellite imagery and machine learning. The primary focus is on identifying waterholes susceptible to damage from invasive herbivores such as water buffalo, feral cattle, and pigs.

## Project Background

Adapted from an existing boat detection pipeline, this project leverages satellite image analysis and YOLOv5 object detection to map and evaluate waterholes in the Arnhem Land and Cape York regions. While originally based on the [CountingBoats](https://github.com/charlie-turner-314/CountingBoats) repository, our workflow has been modified to use Jupyter notebooks for increased clarity and control.

## How

The project follows a comprehensive pipeline:

1. Image Acquisition:
- Use Planet satellite imagery service to order recent images of the target area
- Automatically download imagery for the specified region and date range

2. Pre-processing:
- Prepare satellite images for neural network detection
- Convert and normalize imagery to suitable format

3. Detection:
- Utilize a pre-trained YOLOv5 model to detect and classify waterholes
- Generate comprehensive output with waterhole locations, classifications, and coordinates

4. Analysis:
- Collate detection results
- Output CSV with detailed waterhole information

## Usage

### Installation

Clone this [repository](https://github.com/swforrest/counting_waterholes). Will allow you to run all the notebooks and follow the instructions bellow.

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

Modify configuration files to match your specific environment and research requirements: 
- `config_train_Drive.yaml`: Training configuration
- `config_test_Drive.yaml`: Testing configuration  
- `config_deploy_Drive.yaml`: Deployment configuration

### Running

As mentionned, the whole pipeline flow designed by Charlie Turner for the boat detection, is not used here. You will need to run specific blocks of code in the respective jupyter notebooks of each stage of our workflow.  

#### Images dowload

Using the notebook `planet_download.ipynb`, you will: 
- Define Area of Interest (AOI)
- Specify output directory and date range
- Download and extract satellite imagery

#### Prepare for training

Using the notebook `from_tif_to_trainable_AF.ipynb`, you will: 
- Convert .tif files to training-ready format
- Prepare data for neural network training

#### Training

Either simply run the last code block from `from_tif_to_trainable_AF.ipynb` function train() or copy paste the function output in the yolov5 folder command prompt. 

#### Testing

Using the notebook `post_training.ipynb`, the notebook will guide you along the main steps to test the model you just trained. Those steps are:   
- Prepare images for segmentation
- Run model detection
- Generate annotations
- Compare detections to ground truth
- Create confusion matrix
- Visualize waterhole detections

#### Deployment

Originally the repository had a whole section about the classifying process, which included many functions dependent on the classes of the detection. The adaptation of this existing code to waterhole detection, has proven to be more difficult and time consumming than anticipated. As a result, The deployment section of the repository is still under development and requires further debugging. Some functions are created but need refinement for smooth operation.

#### Visualisation

There are some visualisation notebooks in the visualisation folder. These can be run to perform some visualisations of the data.
The `plot_output` script is also a useful tool for visualising the output of the detection model on individual images. 

All thos functions and modules were developped for the visualisation of Boats and was not adapted to waterholes by lack of time. 

### Acknowledgements

We acknowledge Charlie Turner and other collaborators for their work on the Boat detection repository, largely foundamental to the existance of this repository. 

### Future work
- Complete deployment pipeline debugging
- List more... 