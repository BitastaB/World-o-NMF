from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import NearestNeighbors, DistanceMetric


def init_kmeans(y):
    kmeans_classes = np.unique(y).shape[0]
    kmeans = KMeans(kmeans_classes)

    return kmeans


def print_static(dataset, max_iter, eps_1, eps_2):
    print(f"Dataset : {dataset}")
    print(f"max iterations : {max_iter}")
    print(f"myeps_1 : {eps_1}") if eps_1 else ''
    print(f"myeps_2 : {eps_2}") if eps_2 else ''


def construct_similarity_matrix(gnd):
    m = len(gnd)
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if gnd[i] == gnd[j]:
                S[i][j] = 1

    return S

def KNN(X, k=6):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    dist = DistanceMetric.get_metric('euclidean')
    x = dist.pairwise(X)
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z[i, indices[i]] = 1
        z[indices[i], i] = 1
    n = z.shape[0]
    Q = np.identity(n)
    z -= Q
    return z, x, indices

