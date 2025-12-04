import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


def knn_graph(X, n_neighbors, include_self=False):
    """
    Build a symmetric k-NN graph on points X using Euclidean distances.

    Returns
    -------
    G : scipy.sparse.csr_matrix
        Weighted adjacency matrix (undirected).
    """
    n_samples = X.shape[0]
    k = n_neighbors + (1 if include_self else 0)

    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, indices = nbrs.kneighbors(X)

    rows, cols, data = [], [], []
    for i in range(n_samples):
        start = 0 if include_self else 1
        for dist, j in zip(dists[i, start:], indices[i, start:]):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([dist, dist])

    G = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    return G


def all_pairs_shortest_paths(G):
    """
    Compute all-pairs shortest paths on a sparse graph G.

    Handles disconnected components by replacing inf with a large finite value.
    """
    D = shortest_path(G, directed=False, unweighted=False)
    if not np.all(np.isfinite(D)):
        finite_vals = D[np.isfinite(D)]
        if finite_vals.size == 0:
            raise ValueError("All distances are infinite – graph is fully disconnected.")
        max_val = np.max(finite_vals)
        D[~np.isfinite(D)] = max_val * 1.1
    return D