# Make train, validation, and test splits

import numpy as np
from typing import Tuple

def data_split(X: np.ndarray, Y: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/val/test partitions.

    Parameters
    ----------
    X, Y : np.ndarray
        Feature matrix and target array with matching first dimension.
    train_ratio : float, default 0.8
        Fraction of samples assigned to the training set.
    val_ratio : float, default 0.1
        Fraction of samples assigned to the validation set.
    seed : int | None
        If provided, enables deterministic shuffling via numpy Generator.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, X_test, Y_test : np.ndarray
        Split arrays preserving alignment.
    """
    n = X.shape[0]
    indices = np.arange(n)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

