import numpy as np
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.preprocessing import StandardScaler


def make_swiss_roll_standardized(n_samples=2000, noise=0.05, random_state=0):
    """
    Swiss roll with standardization and color parameter.

    Returns
    -------
    Xs : array, shape (n_samples, 3)
        Standardized coordinates.
    color : array, shape (n_samples,)
        Intrinsic roll parameter (for coloring).
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    Xs = StandardScaler().fit_transform(X)
    return Xs, t


def load_digits_standardized():
    """
    Digits dataset (64D) with standardization.

    Returns
    -------
    Xs : array, shape (n_samples, 64)
    y : array, shape (n_samples,)
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    Xs = StandardScaler().fit_transform(X)
    return Xs, y