import numpy as np
import pandas as pd

from src.distance_functions import euclidean

VALUE_ERROR_MSG = "Either k or the initial cluster (init_c) must be provided"


def _init_c(df, k) -> dict:
    return {
        i + 1: row[1].tolist() for i, row in zip(range(k), df.sample(n=k).iterrows())
    }


def get_cluster(row, centroids, distance_fn):
    d = {ind: distance_fn(row, c) for ind, c in centroids.items()}
    # print(f'{row}\t{d}')
    return min(d, key=d.get)


def _assign_to_clusters(df, centroids, distance_fn):
    return group_rows_by_cluster(
        {i: get_cluster(r.tolist(), centroids, distance_fn) for i, r in df.iterrows()}
    )


def group_rows_by_cluster(rows):
    clusters = {}
    for i, v in rows.items():
        clusters[v] = [i] if v not in clusters.keys() else clusters[v] + [i]
    return clusters


def _get_centers(df, clusters) -> dict:
    if clusters is None:
        return {}
    return {
        c_num: [np.mean(df[df.index.isin(rows)][f]) for f in df]
        for c_num, rows in clusters.items()
    }


def _init_params(df, k, distance_fn, init_c):
    if init_c is None and k is None:
        raise ValueError(VALUE_ERROR_MSG)
    return (
        euclidean if distance_fn is None else distance_fn,
        _init_c(df, k) if init_c is None else init_c,
    )


def hierarchical_clustering(df):
    clusters = {i: [i] for i in df.index}
    while len(clusters) > 1:
        d = {}
        for i, c in clusters.items():
            for j, c2 in clusters.items():
                if i == j:
                    continue
                d[(i, j)] = min(
                    [
                        euclidean(df.loc[i].tolist(), df.loc[j].tolist())
                        for i in c
                        for j in c2
                    ]
                )
        c = min(d, key=d.get)
        clusters[c[0]] = clusters[c[0]] + clusters[c[1]]
        del clusters[c[1]]
        print(clusters)
    return clusters


def _cluster(
    df: pd.DataFrame,
    k=None,
    max_epochs=None,
    distance_fn=None,
    init_c=None,
    verbose=False,
):
    def satisfied(centroids, centers, max_epochs, i):
        return centers == centroids or (max_epochs is not None and i >= max_epochs)

    distance_fn, init_centroids = _init_params(df, k, distance_fn, init_c)
    centers, centroids, clusters = init_centroids, None, None
    i = 0
    while not satisfied(centroids, centers, max_epochs, i):
        centroids = centers
        if verbose:
            print(f"Epoch {i+1}: {centroids}")
        clusters = _assign_to_clusters(df, centroids, distance_fn)
        for c, rows in clusters.items():
            if len(rows) == 0:
                raise ValueError("Empty cluster")
        centers = _get_centers(df, clusters)
        i += 1
    return centers


class KMeans:
    def __init__(
        self, max_epochs=None, distance_fn=euclidean, init_c=None, verbose=False
    ):
        self.max_epochs = max_epochs
        self.distance_fn = distance_fn
        self.init_c = init_c
        self.verbose = verbose
        self.centers = None

    def fit(self, df: pd.DataFrame, k: int):
        self.centers = _cluster(
            df, k, self.max_epochs, self.distance_fn, self.init_c, self.verbose
        )
        return self

    def predict(self, df: pd.DataFrame):
        return get_cluster(df.tolist(), self.centers, self.distance_fn)

    def center_dim(self, dim):
        return [c[dim] for c in self.centers.values()]


class Cluster:
    def __init__(self, centroid, rows):
        self.rows = rows
        self.centroid = centroid

    def __repr__(self):
        return f"Cluster(rows={self.rows}, centroid={self.centroid})"

    def __str__(self):
        return f"Cluster(rows={self.rows}, centroid={self.centroid})"

    def __eq__(self, other):
        return self.rows == other.rows and self.centroid == other.centroid

    def __hash__(self):
        return hash((self.centroid, self.rows))