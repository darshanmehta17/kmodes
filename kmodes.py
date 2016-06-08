import numpy as np
import random
from collections import Counter
import operator


def calc_min_dissim(X, centroids):
    dissimilarity = np.zeros(centroids.shape[0])
    for ii in range(centroids.shape[0]):
        for jj in range(centroids.shape[1]):
            if X[jj] != centroids[ii, jj]:
                # dissimilarity[ii] += abs(X[jj] - centroids[ii, jj])
                dissimilarity[ii] += 1
    return min(enumerate(dissimilarity), key=operator.itemgetter(1))[0]


def update_centroids(X, belongs, centroids):
    n_centroids = centroids.shape[0]
    for ii in range(n_centroids):
        points = np.array([X[jj, :] for jj in range(belongs.shape[0]) if belongs[jj] == ii])
        for kk in range(points.shape[1]):
            temp_points = [points[jj, kk] for jj in range(points.shape[0])]
            count = Counter(temp_points)
            centroids[ii, kk] = max(count.iteritems(), key=operator.itemgetter(1))[0]
    return centroids


def kmodes(X, n_clusters=8, max_iter=100):

    # Chooses random cluster centers
    cluster_centers = np.array(random.sample(X, n_clusters))
    print "Initial centroids:"
    print cluster_centers
    print "|-----------------------------------|"

    n_points = X.shape[0]
    belongs_to = np.full(n_points, 0, dtype='int')
    has_changed = False

    # Calculates the belonging of each point among the cluster centers
    for ii in range(max_iter):
        for jj in range(n_points):
            belongs = calc_min_dissim(X[jj, :], cluster_centers)
            if belongs_to[jj] != belongs:
                belongs_to[jj] = belongs
                has_changed = True
        if not has_changed:
            break
        else:
            cluster_centers = update_centroids(X, belongs_to, cluster_centers)
            has_changed = False
    return cluster_centers, belongs_to

# Importing data from dataset and reformatting into attributes and labels
x = np.genfromtxt('soybean.csv', dtype=str, delimiter=',')[:, :-1]
y = np.genfromtxt('soybean.csv', dtype=str, delimiter=',', usecols=(21,))

centroids, y_test = kmodes(x, 4, 100)

print "|-------------------------------------------------|"
print "Centroids:"
print centroids
print "|-------------------------------------------------|"
print "Y train:Y test"
combo = [(ii,jj) for ii,jj in zip(y,y_test)]
for x in combo:
    print x
print "|-------------------------------------------------|"
