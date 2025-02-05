

import counting_boats.boat_utils.planet_utils as planet_utils

bulman_search = planet_utils.PlanetSearch(
    polygon_file='data/polygons/bulman.geojson',
    min_date='2023-06-28',
    max_date='2023-06-29',
    cloud_cover=0.5,
)

# print(bulman_search)

# for item in bulman_search:
#     print(item)

# # let's look at the first result
# print(list(bulman_search)[0])

# extract image IDs only
image_ids = [feature['id'] for feature in bulman_search]
print(image_ids)



bulman_select = planet_utils.PlanetSelect(
    bulman_search,
    polygon='data/polygons/bulman.geojson',
    date='2023-06-29'
)

# print(bulman_select)

image_ids = [feature['id'] for feature in bulman_select]
print(image_ids)


bulman_order = planet_utils.PlanetOrder(
    bulman_select,
    polygon_file='data/polygons/bulman.geojson',
    name='bulman_test_2025-02-05',
)

print(bulman_order)