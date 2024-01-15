"""
Given a heap of polygons, create a heatmap of the density of seeing each area.

1. Bounding box of all the polygons
    - This is then the area for the 'heatmap'
    - Can alternatively just use a big polygon that covers the area
2. Represent the bounding box by a grid of pixels 
    - This so that the heatmap can be represented by a 2D array
3. For each polygon:
    - 'paint' the polygon onto the heatmap by adding 1 to each pixel that is
        covered by the polygon

"""
