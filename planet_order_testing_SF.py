

import counting_boats.boat_utils.planet_utils as planet_utils

planet_utils.PlanetSearch(
    polygon_file='data/polygons/bulman.geojson',
    min_date='2019-01-01',
    max_date='2019-12-31',
    cloud_cover=0.1,
)


