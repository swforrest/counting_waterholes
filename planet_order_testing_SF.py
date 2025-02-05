

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
# print(image_ids)

order_name = 'bulman_test_2025-02-05'

# bulman_order = planet_utils.PlanetOrder(
#     bulman_select,
#     polygon_file='data/polygons/bulman.geojson',
#     name=order_name,
# )

# print(bulman_order)

# bulman_order_status = planet_utils.PlanetCheckOrder(order_name)

# print(bulman_order_status)

planet_utils.PlanetDownload(orderID='4d9bc8b4-8255-4f7e-800b-40c82ea9b6c3',
                            aoi='bulman',
                            date='2023-06-29')