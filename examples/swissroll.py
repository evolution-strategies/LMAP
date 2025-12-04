#!/usr/bin/env python3
"""
Example: LMAP on the Swiss roll manifold.
"""

from lmap.lmap import lmap
from lmap.datasets import make_swiss_roll_standardized
from lmap.visualization import plot_embedding_2d, plot_local_tangents_3d
from lmap.metrics import trustworthiness, sammon_stress


def main():
    # 1. data
    X, color = make_swiss_roll_standardized(n_samples=2000, random_state=0)

    # 2. run LMAP
    Y, info = lmap(
        X,
        m=500,
        k_local=40,
        graph_k=10,
        d=2,
        q=5,
        standardize=False,   # already standardized in dataset helper
        random_state=0,
    )

    # 3. metrics
    tw = trustworthiness(X, Y, n_neighbors=10)
    sa = sammon_stress(X, Y)

    print(f"LMAP on Swiss Roll: TW={tw:.3f}, SA={sa:.3f}")

    # 4. plots
    plot_embedding_2d(
        Y,
        color=color,
        title=f"LMAP on Swiss Roll (TW={tw:.3f}, SA={sa:.3f})",
    )

    plot_local_tangents_3d(
        X,
        info["landmarks"],
        info["tangents"],
        scales=info["scales"],
        n_show=80,
        elev=10,
        azim=-70,
    )


if __name__ == "__main__":
    main()