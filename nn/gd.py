# Gradient Descent
from .network import NeuralNetwork
import numpy as np
from math import inf

def fit(network: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, max_epochs: int = 100, learning_rate: float = 0.01, patience: int = 7, min_loss: float = 1e-7) -> float:
    """
    Trains a neural network using gradient descent.

    More robust stopping conditions, regularization, etc. etc. could be added but we're just keeping it simple rn :)

    Args:
        network (NeuralNetwork): The neural network to train.
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training output data.
        X_val (np.ndarray): The validation input data.
        y_val (np.ndarray): The validation output data.
        X_test (np.ndarray): The test input data.
        y_test (np.ndarray): The test output data.
        max_epochs (int): The maximum number of epochs to train for.
        learning_rate (float): The learning rate for gradient descent. Defaults to 0.01.
        patience (int): Tolerance for error increase. If increases for this many epochs, stop early.
        min_loss (float): If validation loss goes below this, stop early.
    
    Returns:
        float: The final test set error.
        Inputted neural-network gets trained in-place.
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
