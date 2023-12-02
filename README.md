# Counting Boats
## What
This project counts small marine vessels from satellite imagery of the Moreton Bay 
region. The counts are recorded and can be analysed or presented later. Tools for 
visualising training and inference data are also available.

## Why

Moreton Bay is a complex ecosystem, home to a vast array of marine life, and at the 
same time being a heavily used and enjoyed swimming, boating, and fishing area for humans. 
Understanding the interactions between human activity and marine life is a crucial aspect
in ensuring that the bay can be enjoyed by all for years to come. This project 
is a small part of the much wider [Living and Playing Together Project](https://moretonbayfoundation.org/research/living-and-playing-together).

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


#### Python Dependencies
It's reccommended to install a conda-based package manager such as [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
Running the following will then install all required dependencies:
```
conda create --name CountingBoats --file requirements.txt
```
Activate the environment (if not already) with `conda activate CountingBoats`, and you should be good to go.


### Setup

### Running

## Acknowledgements
