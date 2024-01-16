from utils import area_coverage

peel = "/Users/charlieturner/Documents/CountingBoats/Polygons/peel.json"
tangalooma = "/Users/charlieturner/Documents/CountingBoats/Polygons/tangalooma.json"
bribie = "/Users/charlieturner/Documents/CountingBoats/Polygons/south_bribie.json"

peel_poly = area_coverage.polygon_to_32756(peel)
tangalooma_poly = area_coverage.polygon_to_32756(tangalooma)
bribie_poly = area_coverage.polygon_to_32756(bribie)

peel_area = peel_poly.Area()
tangalooma_area = tangalooma_poly.Area()
bribie_area = bribie_poly.Area()

print("Peel area: {}".format(peel_area))
print("Tangalooma area: {}".format(tangalooma_area))
print("Bribie area: {}".format(bribie_area))
total_area_labelled = 4* peel_area + 3*tangalooma_area + 4*bribie_area
print(f"Total area (3x each ,only 2x tangalooma): {total_area_labelled}", "m^2")
print(f"Total area (3x each): {total_area_labelled/1000000}", "km^2")




