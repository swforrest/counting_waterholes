from NNclassifier import cluster, read_classifications, process_clusters, pixel2latlong

STAT_DISTANCE_CUTOFF_PIX = 6
MOVING_DISTANCE_CUTOFF_PIX = 10

folder = './TrainImages/text'

classifications = read_classifications('training', folder)

static = [c for c in classifications if c[3] == 0]
moving = [c for c in classifications if c[3] == 1]

# cluster each set separately
static_clusters = cluster(static, STAT_DISTANCE_CUTOFF_PIX)
moving_clusters = cluster(moving, MOVING_DISTANCE_CUTOFF_PIX)
# process each set separately
static_boats = process_clusters(static_clusters)
moving_boats = process_clusters(moving_clusters)
# save as csv
with open('static_boats.csv', 'w') as f:
    for b in static_boats:
        f.write(b + '\n')



