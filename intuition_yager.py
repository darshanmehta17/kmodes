import numpy as np
import operator
from collections import Counter
from time import time
import kmodes as km
import random


def initialize_centroids(X, n_clusters=4):
    """
    Performs selection of initial centroids (From random as of now)
    :param X: The dataset of points to choose from
    :param n_clusters: number of initial points to choose and return
    :return: n_clusters initial points selected from X as per the algorithm used
    """

    # centroids, belongs_to = km.kmodes(X, n_clusters, debug=False)

    # return centroids
    return np.array(random.sample(X, n_clusters))


def calculate_dissimilarity(Z, X):
    """
    Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
    :param Z: Record 1
    :param X: Record 2
    :return: Dissimilarity between Z and X
    """
    m = len(Z)

    dissimlarity = 0

    for j in range(m):
        if Z[j] != X[j]:
            dissimlarity += 1

    return dissimlarity


def calculate_cost(W, Z, X, alpha):
    """
    Calculates the cost function of k-modes algorithm as per Huang '99 paper on fuzzy k-modes.
    :param W: Fuzzy partition matrix
    :param Z: Cluster centroids
    :param alpha: Weighing exponent
    :return: Cost of of the current setup using the formula from the paper
    """
    k = W.shape[0]
    n = W.shape[1]

    cost = 0

    for l in range(k):
        for i in range(n):
            cost += pow(W[l][i], alpha) * calculate_dissimilarity(Z[l], X[i])

    return cost


def calculate_partition_matrix(Z, X, alpha):
    """
    Calculates the dissimilarity matrix W for a fixed Z as per Theorem 1 in Huang '99 paper on fuzzy kmodes.
    :param Z: Fixed centroids
    :param X: Dataset points
    :param alpha: Weighing exponent
    :return: Dissimilarity matrix of type Numpy array of dimension k x n.
    """
    k = len(Z)

    n = X.shape[0]

    exponent = 1 / (alpha - 1)

    W = np.zeros((k, n), dtype='float')

    for l in range(k):
        for i in range(n):
            dli = calculate_dissimilarity(Z[l], X[i])
            if dli == 0:
                W[l][i] = 1
            else:
                flag = False
                sum = 0
                for h in range(k):
                    dhi = calculate_dissimilarity(Z[h], X[i])

                    if h != l and dhi == 0:
                        W[l][i] = 0
                        flag = True
                        break

                    sum += pow(dli / dhi, exponent)

                if not flag:
                    W[l][i] = 1 / sum

    delta = 1.4

    K = np.copy(W)

    W_ = 1 - np.power((1 - np.power(K, [delta])), [1 / delta])
    # print "W:"
    # print W
    # print
    # print
    # print "W':"
    # print W_

    return W_


def calculate_centroids(W, X, alpha):
    """
    Calculates the updated value of Z as per Theorem 4 of paper by Huang '99 on fuzzy kmodes.
    :param W: Partition matrix
    :param X: Dataset
    :param alpha: Weighing exponent
    :return: Updated centroid Numpy matrix of dimension k x n.
    """
    k = W.shape[0]
    m = X.shape[1]

    Z = [[None] * m for i in range(k)]

    for l in range(k):
        for j in range(m):
            weights = []
            x_j = X[:, j]
            dom_aj = np.unique(x_j)
            for key in dom_aj.__iter__():
                indexes = [i for i in range(len(x_j)) if x_j[i] == key]
                sum = 0
                for index in indexes:
                    sum += pow(W[l][index], alpha)
                weights.append((key, sum))
            Z[l][j] = max(weights, key=operator.itemgetter(1))[0]

    return Z


def calculate_db_index(X, Y, Z):
    k = Z.__len__()

    dist_i = []

    for ii in range(k):
        centroid = Z[ii]
        points = [X[i] for i in range(len(Y)) if Y[i] - 1 == ii]
        distance = 0

        for jj in points:
            distance += calculate_dissimilarity(centroid, jj)

        if len(points) == 0:
            dist_i.append(0.0)
        else:
            dist_i.append(round(distance * 1.0 / len(points), 4))

    D_ij = []

    for ii in range(k):
        D_i = []
        for jj in range(k):
            if ii == jj or calculate_dissimilarity(Z[ii], Z[jj]) == 0:
                D_i.append(0.0)
            else:
                D_i .append((dist_i[ii] + dist_i[jj]) * 1.0 / calculate_dissimilarity(Z[ii], Z[jj]))
        D_ij.append(D_i)

    db_index = 0

    for ii in range(k):
        db_index += max(D_ij[ii])

    db_index *= 1.0
    db_index /= k

    return db_index


def fuzzy_kmodes(X, Y, n_clusters=4, alpha=1.1):
    """
    Calculates the optimal cost, cluster centers and fuzzy partition matrix for the given dataset.
    :param X: Dataset
    :param n_clusters: number of clusters to form
    :param alpha: Weighing exponent
    :return:
    """
    t0 = time()

    Z = initialize_centroids(X, n_clusters)

    W = calculate_partition_matrix(Z, X, alpha)

    f_old = calculate_cost(W, Z, X, alpha)

    f_new = 0

    while True:
        Z = calculate_centroids(W, X, alpha)

        f_new = calculate_cost(W, Z, X, alpha)

        if f_new == f_old:
            break

        f_old = f_new

        W = calculate_partition_matrix(Z, X, alpha)

        f_new = calculate_cost(W, Z, X, alpha)

        if f_new == f_old:
            break

    assigned_clusters = calculate_cluster_allotment(W)

    t1 = round(time() - t0, 3)

    accuracy = calculate_accuracy(Y, assigned_clusters)

    db_index = calculate_db_index(X, assigned_clusters, Z)

    return t1, f_new, Z, W, accuracy, db_index


def calculate_cluster_allotment(W):
    """
    Calculates the membership of each point to various clusters.
    :param W: Partition matrix
    :return: allotment array of dimension 1xn
    """
    n = W.shape[1]

    allotment = np.zeros(n, dtype='int')

    for i in range(n):
        allotment[i] = np.argmax(W[:, i]) + 1

    return allotment


def calculate_accuracy(labels, prediction):
    labels_values = np.unique(labels)
    count = 0.0

    for key in labels_values.__iter__():
        indices = [prediction[i] for i in range(len(prediction)) if labels[i] == key]
        count += max(Counter(indices).iteritems(), key=operator.itemgetter(1))[1]

    return round(count / len(prediction), 4) * 100


def run(n_iter=100, n_clusters=4, alpha=1.1):

    # Importing data from data set and reformatting into attributes and labels
    # x = np.genfromtxt('soybean.csv', dtype=str, delimiter=',')[:, :-1]
    # y = np.genfromtxt('soybean.csv', dtype=str, delimiter=',', usecols=(21,))
    x = np.genfromtxt('zoo.csv', dtype=str, delimiter=',')[:, :-1]
    y = np.genfromtxt('zoo.csv', dtype=str, delimiter=',', usecols=(17,))

    comp_time = []
    cost = []
    accuracy = []
    db_indexes = []

    for ii in range(n_iter):
        comp_time_temp, f_new, Z, W, acc, db_index = fuzzy_kmodes(x, y, n_clusters, alpha)
        comp_time.append(comp_time_temp)
        cost.append(f_new)
        accuracy.append(acc)
        db_indexes.append(db_index)

    avg_time = sum(comp_time) / len(comp_time)
    avg_cost = sum(cost) / len(cost)
    avg_accuracy = sum(accuracy) / len(accuracy)

    return avg_time, avg_cost, avg_accuracy, db_indexes


if __name__ == "__main__":

    # Number of iterations
    n_iter = 100

    # Number of clusters
    n_clusters = 7

    # Weighing exponent
    alpha = 1.1

    avg_time, avg_cost, avg_accuracy, db_indexes = run(n_iter, n_clusters, alpha)

    print "Average time:", avg_time
    print
    print "Average Cost:", avg_cost
    print
    print "Average Accuracy:", avg_accuracy
    print
    print "Best DB Index:", min(db_indexes)
    print
    print "Average DB Index:", sum(db_indexes) / len(db_indexes)
    print
    print "DB Indexes:", db_indexes