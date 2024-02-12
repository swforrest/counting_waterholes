"""
Perform analysis using data obtained from counting boats.
Available data:
    'boats.csv' - CSV of boat detections:
            'date'      - date of detection
            'class'     - class of boat (0:stationary, 1:moving)
            'latitude'  - latitude of detection
            'longitude' - longitude of detection
            'confidence' - confidence of detection
    'coverage.csv' - CSV of area coverage:
            'date'      - date of images
            'aoi'       - name of area of interest
            'coverage'  - coverage of area of interest (0-1)
            'polygon'   - polygon of image coverage
Inputs:
    Date range
Outputs:
    Heatmaps of:
        - Boat detections (total)
        - Boat detections (stationary)
        - Boat detections (moving)
        - Area coverage
        - Boat detections per area coverage
    Point Plots On Basemap:
        - Boat detections
"""
