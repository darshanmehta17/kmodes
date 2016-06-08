import numpy as np
import random
from collections import Counter
import operator


def initialize_centroids(X, n_clusters=4):
    """
    Performs selection of initial centroids (Random as of now)
    :param X: The dataset of points to choose from
    :param n_clusters: number of initial points to choose and return
    :return: n_clusters initial points selected from X as per the algorithm used
    """
    return np.array(random.sample(X, n_clusters))


def calculate_dissimilarity(Z, X):
    """
    Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
    :param Z: Record 1
    :param X: Record 2
    :return: Dissimilarity between Z and X
    """
    m = Z.shape[0]

    dissimlarity = 0

    for j in range(m):
        if Z[j] != X[j]:
            dissimlarity += 1

    return dissimlarity


def calculate_cost(W, Z, alpha):
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
    k = Z.shape[0]

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

    return W


def fuzzy_kmodes(X, n_clusters=4):
    Z = initialize_centroids(X, n_clusters)

