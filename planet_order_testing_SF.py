

import counting_boats.boat_utils.planet_utils as planet_utils

bulman_search = planet_utils.PlanetSearch(
    polygon_file='data/polygons/bulman.geojson',
    min_date='2022-12-25',
    max_date='2022-12-31',
    cloud_cover=0.5,
)

print(bulman_search)

