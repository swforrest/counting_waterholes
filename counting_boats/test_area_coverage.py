from utils import area_coverage
import matplotlib.pyplot as plt

poly1 = '''{"coordinates": [
					[
						[
							153.27565086080142,
							-27.50152138707257
						],
						[
							153.23242533091917,
							-27.682948738705985
						],
						[
							153.57438712142186,
							-27.7464804303293
						],
						[
							153.6166203162487,
							-27.565772193634675
						],
						[
							153.27565086080142,
							-27.50152138707257
						]
					]
				],
				"type": "Polygon"
			}'''

poly2 = '''{"coordinates": [
					[
						[
							153.21887917639867,
							-27.335271688997082
						],
						[
							153.17708033853734,
							-27.51354799271345
						],
						[
							153.51158754655845,
							-27.575414838552476
						],
						[
							153.55284045240194,
							-27.397473220105905
						],
						[
							153.21887917639867,
							-27.335271688997082
						]
					]
				],
				"type": "Polygon"}'''

poly3 = area_coverage.combine_polygons([poly1, poly2])
poly1 = area_coverage.polygon_to_32756(poly1)
poly2 = area_coverage.polygon_to_32756(poly2)

def plot_poly(poly, ax, color='k'):
    for i in range(poly.GetGeometryCount()):
        x = []
        y = []
        points = poly.GetGeometryRef(i)
        for p in range(points.GetPointCount()):
            x.append(points.GetX(p))
            y.append(points.GetY(p))
        ax.plot(x, y, color, alpha=0.5)

# plot the polygons on their own axis
fig = plt.figure()
ax = fig.add_subplot(111)
# plot poly1
plot_poly(poly1, ax, 'r')
# plot poly2
plot_poly(poly2, ax, 'b')
# plot poly3
plot_poly(poly3, ax, 'k')
plt.show(block=False)


# calculate coverage over peel island
peel = "/Users/charlieturner/Documents/CountingBoats/Polygons/moreton_bay_region.geojson"

coverage, intersection = area_coverage.area_coverage_poly(peel, poly3)

print("Coverage: ", coverage)

# plot peel and poly3 and intersection
peel_poly = area_coverage.polygon_to_32756(peel)
fig = plt.figure()
ax = fig.add_subplot(111)
# plot AOI
plot_poly(peel_poly, ax, 'r')
# plot poly
plot_poly(poly3, ax, 'b')
# plot intersection
plot_poly(intersection, ax, 'k')
plt.show(block=False)
plt.title("Intersection of AOI and Polygons: " + str(round(coverage *100, 2)) + "%")


plt.show()








