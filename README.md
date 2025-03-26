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

As mentionned, the whole pipeline flow designed by Charlie Turner for the boat detection, is not used here. You will need to run specific blocks of code in the respective jupyter notebooks of each stage of our workflow.  

#### Images dowload

Using the notebook `planet_download.ipynb`, you are able to define the Area Of Interest (AOI), the output directory, and the date range before ordering from Planet. Once the order is completed, it can be downloaded and automatically extracts the obtained composite.tif from the .zip file.  

#### Prepare for training

Using the notebook `from_tif_to_trainable_AF.ipynb`, you will follow multiple steps that guide you form the tif file to a product that allows you to train your model. 

#### Training

Either simply run the last code block from `from_tif_to_trainable_AF.ipynb` function train() or copy paste the function output in the yolov5 folder command prompt. 

#### Testing

Using the notebook `post_training.ipynb`, the notebook will guide you along the main steps to test the model you just trained. Thos utilities are:   
    - prepare: Prepare the images for segmentation  
    - segment: Segment the images  
    - run_detection: Run the YoloV5 detection  
    - backwards_annotation_AF: Generate labelme style annotations from the classifications  
    - compare_detections_to_ground_truth: Match up labels and detections, compare them, and save the results  
    - confusion_matrix_AF: Summarize the results of the comparison  
    - plot_waterholes: Plot a comparison of my labels vs the model detection of waterholes on single stitched back together images.   

#### Deployment

Originally the repository had a whole section about the classifying process, which included many functions dependent on the classes of the detection. The adaptation of this existing code to waterhole detection, has proven to be more difficult and time consumming than anticipated. As a result, the deployment section of this repository still requires some debugging work.  
Most of the functions and their dependencies are created, but the smooth running of those needs some work. 

### Visualisation

There are some visualisation notebooks in the visualisation folder. These can be run to perform some visualisations of the data.
The `plot_output` script is also a useful tool for visualising the output of the detection model on individual images. 

All thos functions and modules were developped for the visualisation of Boats and was not adapted to waterholes by lack of time. 

## Acknowledgements

We acknowledge 