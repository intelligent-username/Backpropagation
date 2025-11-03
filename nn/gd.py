# Gradient Descent
from .network import NeuralNetwork
import numpy as np
from math import inf

def fit(network: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, max_epochs: int = 100, learning_rate: float = 0.01, patience: int = 7, min_loss: float = 1e-7) -> float:
    """Train a neural network with (stochastic) gradient descent and early stopping.

    Parameters
    ----------
    network : NeuralNetwork
        The network to train; mutated in-place.
    X_train, y_train : np.ndarray
        Training features and targets. X may be 2D (batch, features); y either
        1D/2D depending on task (regression vs one-hot classification).
    X_val, y_val : np.ndarray
        Validation split used for early stopping.
    X_test, y_test : np.ndarray
        Test split used for final error reporting.
    max_epochs : int, default 100
        Maximum number of passes over the training set.
    learning_rate : float, default 0.01
        Step size for parameter updates.
    patience : int, default 7
        Stop if validation loss worsens this many consecutive epochs.
    min_loss : float, default 1e-7
        Stop if validation loss falls below this threshold.

    Returns
    -------
    test_error:  float
        Final average test loss.
    """

    train_len = len(X_train)
    train_val = len(X_val)

    e = 0
    failz = 0
    prev_loss = inf

    while e < max_epochs and failz < patience and prev_loss > min_loss:
        # --- Shuffle (predetermine stoachstic GD) ---
        indices = np.arange(train_len)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # --- Training (SGD: update after each sample) ---
        train_error = 0
        for x, y in zip(X_train_shuffled, y_train_shuffled):
            # First, forward pass
            output = network.forward(x)
            # Then calculate error
            train_error += network.loss(y, output)

            # Backward pass
            network.backward(y, learning_rate)

        # Calculate L2 penalty for the entire network
        l2_penalty = 0
        for layer in network.layers:
            if hasattr(layer, 'l2_lambda') and layer.l2_lambda > 0:
                l2_penalty += 0.5 * layer.l2_lambda * np.sum(np.square(layer.weights))

        train_error /= train_len
        train_error += l2_penalty

        # --- Validation ---
        val_error = 0
        for x, y in zip(X_val, y_val):
            output = network.forward(x)
            val_error += network.loss(y, output)
        val_error /= train_val
        val_error += l2_penalty

        # --- Early stopping based on conds ---
        if val_error > prev_loss:
            failz += 1
        else:
            failz = 0
            prev_loss = val_error

        # print(f"Epoch {e + 1}/{max_epochs}, Train Error: {train_error:.4f}, Val Error: {val_error:.4f}")

        e += 1

    # Final Test
    test_error = 0
    for x, y in zip(X_test, y_test):
        output = network.forward(x)
        test_error += network.loss(y, output)
    test_error /= len(X_test)
    
    return test_error
