# lmap/metrics.py
"""
Metrics for evaluating LMAP embeddings:
- Trustworthiness (local neighborhood preservation)
- Sammon stress (global distance distortion)
"""

import numpy as np
from sklearn.manifold import trustworthiness as _sk_trustworthiness
from scipy.spatial.distance import pdist, squareform


def trustworthiness(X, Y, n_neighbors=10):
    """
    Trustworthiness of an embedding, using sklearn's implementation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        High-dimensional data.
    Y : array-like, shape (n_samples, n_components)
        Low-dimensional embedding.
    n_neighbors : int, optional (default=10)
        Number of neighbors to consider.

    Returns
    -------
    tw : float
        Trustworthiness score in [0, 1]; higher is better.
    """
    return _sk_trustworthiness(X, Y, n_neighbors=n_neighbors)


def sammon_stress(X, Y, eps=1e-9):
    """
    Sammon stress between high-dimensional distances and embedding distances.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        High-dimensional data.
    Y : array-like, shape (n_samples, n_components)
        Low-dimensional embedding.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    stress : float
        Normalized Sammon stress; lower is better.
    """
    # Pairwise distances in original and embedding spaces
    D_high = squareform(pdist(X))
    D_low = squareform(pdist(Y))

    # Ignore zero distances (on the diagonal)
    mask = D_high > 0
    Dh = D_high[mask]
    Dl = D_low[mask]

    scale = Dh.sum() + eps
    stress = np.sum(((Dh - Dl) ** 2) / (Dh + eps)) / scale
    return stress