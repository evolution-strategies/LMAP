"""
LMAP: Local PCA Models with Global MDS Embeddings

Core implementation of the LMAP algorithm as described in the paper:
- Landmark sampling
- Local PCA tangent modeling
- Global MDS alignment on a landmark graph
- Smooth out-of-sample mapping via blended tangent charts
"""

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

from .utils import knn_graph, all_pairs_shortest_paths


def lmap(
    X,
    m=500,
    k_local=40,
    graph_k=10,
    d=2,
    q=5,
    standardize=True,
    random_state=0,
):
    """
    Local PCA Models with Global MDS Embeddings (LMAP).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Input data.
    m : int
        Number of landmarks.
    k_local : int
        Number of neighbors for each landmark's local PCA model.
    graph_k : int
        Number of neighbors in the landmark graph.
    d : int
        Target embedding dimension.
    q : int
        Number of nearest landmarks used for out-of-sample blending.
    standardize : bool
        If True, apply StandardScaler to X internally.
    random_state : int
        Random seed for landmark selection and MDS.

    Returns
    -------
    Y : array, shape (n_samples, d)
        Low-dimensional embedding.
    info : dict
        Dictionary with intermediate quantities:
        - "landmarks"
        - "tangents"
        - "scales"
        - "Y_landmarks"
        - "params"
        - "X_scaled"
    """
    X = np.asarray(X, float)
    n, D = X.shape

    rng = np.random.default_rng(random_state)

    # 0) optional standardization
    if standardize:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.copy()

    # 1) Landmark sampling
    m_eff = min(m, n)
    landmark_idx = rng.choice(n, size=m_eff, replace=False)
    C = Xs[landmark_idx]

    # 2) Local tangent modeling via PCA
    k_local_eff = min(k_local, n)
    nbrs_all = NearestNeighbors(n_neighbors=k_local_eff).fit(Xs)

    tangents = []
    scales = []

    for cj in C:
        dists, idx = nbrs_all.kneighbors(cj.reshape(1, -1), return_distance=True)
        patch = Xs[idx[0]]
        pca = PCA(n_components=min(d, D), random_state=random_state)
        pca.fit(patch)
        T = pca.components_.T  # shape (D, d)
        tangents.append(T)
        scales.append(np.median(dists))

    scales = np.asarray(scales).reshape(-1)

    # 3) Landmark graph + global MDS alignment
    G = knn_graph(C, n_neighbors=graph_k, include_self=True)
    Dl = all_pairs_shortest_paths(G)

    mds = MDS(
        n_components=d,
        dissimilarity="precomputed",
        random_state=random_state,
    )
    Y_landmarks = mds.fit_transform(Dl)

    # 4) Out-of-sample mapping via blended tangent charts
    q_eff = min(q, m_eff)
    nbrs_landmarks = NearestNeighbors(n_neighbors=q_eff).fit(C)
    _, nn_idx = nbrs_landmarks.kneighbors(Xs)

    Y = np.zeros((n, d), float)

    for i, xi in enumerate(Xs):
        num = np.zeros(d, float)
        den = 0.0

        for j in nn_idx[i]:
            cj = C[j]
            Tj = tangents[j]
            sj = float(scales[j]) if scales[j] > 0 else 1.0

            diff = xi - cj
            # local linear prediction in embedding space
            y_loc = Y_landmarks[j] + Tj[:, :d].T @ diff

            w = np.exp(-norm(diff) ** 2 / (2.0 * sj ** 2 + 1e-12))
            num += w * y_loc
            den += w

        if den > 0:
            Y[i] = num / den
        else:
            # fallback: use nearest landmark embedding
            Y[i] = Y_landmarks[nn_idx[i, 0]]

    info = {
        "landmarks": C,
        "tangents": tangents,
        "scales": scales,
        "Y_landmarks": Y_landmarks,
        "X_scaled": Xs,
        "params": {
            "m": m,
            "k_local": k_local,
            "graph_k": graph_k,
            "d": d,
            "q": q,
            "standardize": standardize,
            "random_state": random_state,
        },
    }
    return Y, info