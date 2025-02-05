

import counting_boats.boat_utils.planet_utils as planet_utils

moreton_search = planet_utils.PlanetSearch(
    polygon_file='data/polygons/mimal.geojson',
    min_date='2022-01-01',
    max_date='2022-12-31',
    cloud_cover=0.5,
)

print(moreton_search)

