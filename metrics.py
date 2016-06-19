
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