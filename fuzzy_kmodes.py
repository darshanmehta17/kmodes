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
    :param alpha: weighing exponent
    :return: Cost of of the current setup using the formula from the paper
    """
    k = W.shape[0]
    n = W.shape[1]

    cost = 0

    for l in range(k):
        for i in range(n):
            cost += pow(W[l][i], alpha) * calculate_dissimilarity(Z[l], X[i])

    return cost


def fuzzy_kmodes(X, n_clusters=4):
    Z = initialize_centroids(X, n_clusters)

