import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_embedding_2d(Y, color=None, title=None, s=6):
    """
    Simple 2D scatter plot for an embedding.

    Parameters
    ----------
    Y : array, shape (n_samples, 2)
    color : array or None
        Values used to color the points (e.g. intrinsic parameter).
    title : str or None
    s : int
        Marker size.
    """
    Y = np.asarray(Y)
    plt.figure(figsize=(5, 4))
    if color is None:
        plt.scatter(Y[:, 0], Y[:, 1], s=s)
    else:
        sc = plt.scatter(Y[:, 0], Y[:, 1], c=color, s=s, cmap="viridis")
        plt.colorbar(sc, fraction=0.046, pad=0.04)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_local_tangents_3d(X, landmarks, tangents, scales=None, n_show=60,
                           elev=10, azim=-70):
    """
    Visualize local PCA tangent directions at selected landmarks.

    Parameters
    ----------
    X : array, shape (n_samples, 3)
    landmarks : array, shape (m, 3)
    tangents : list of arrays, each shape (D, d)
    scales : list of float or None
        Scale factors per landmark (e.g. median neighbor distance).
    n_show : int
        Number of landmarks to visualize.
    elev, azim : float
        View angles.
    """
    if X.shape[1] != 3:
        raise ValueError("3D visualization expects X with shape (n_samples, 3).")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="lightgray", s=4, alpha=0.3)

    m = landmarks.shape[0]
    idx = np.linspace(0, m - 1, min(n_show, m), dtype=int)

    for i in idx:
        ci = landmarks[i]
        T = tangents[i]
        s = scales[i] if scales is not None else 0.2
        # only first principal direction for clarity
        v = T[:, 0]
        ax.plot(
            [ci[0], ci[0] + s * v[0]],
            [ci[1], ci[1] + s * v[1]],
            [ci[2], ci[2] + s * v[2]],
            color="red",
            lw=1.5,
        )

    ax.view_init(elev, azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("LMAP: Local PCA Tangents", pad=18)
    plt.tight_layout()
    plt.show()