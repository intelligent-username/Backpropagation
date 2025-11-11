"""Loss functions and their derivatives.

Conventions
-----------
- y_true and y_pred may be 1D or 2D. Functions reshape to (batch, dims) as needed.
- Derivatives return averages over the batch where applicable.
"""

import numpy as np

# Only Mean Squared Error right now, might implement more later but probably unnecessary for this generic demo

def MSE(y_true, y_pred):
    """Mean Squared Error loss.

    Parameters
    ----------
    y_true, y_pred : array-like
        True targets and predictions with broadcastable shape.

    Returns
    -------
    float
        Average 0.5 * (y_true - y_pred)^2 over all elements.
    """
    # Divide by two to make the constant for the derivative simpler
    return (((y_true - y_pred) ** 2).mean()/2)


def MSE_derivative(y_true, y_pred):
    """Derivative of MSE loss with the 0.5 factor: (y_pred - y_true) / N.

    Returns the mean gradient per element, where N is the total number of elements.
    """
    return (y_pred - y_true) / y_true.size


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss (quadratic near 0, linear beyond |error|>delta).

    Parameters
    ----------
    y_true, y_pred : array-like
        Targets and predictions.
    delta : float, default 1.0
        Threshold separating quadratic and linear regions.

    Returns
    -------
    float
        Mean Huber loss over all elements.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = np.square(error) / 2
    linear_loss = delta * (np.abs(error) - delta / 2)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

def huber_loss_derivative(y_true, y_pred, delta=1.0):
    """Derivative of Huber loss, averaged over all elements."""
    error = y_pred - y_true
    derivative = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
    return derivative / y_true.size

def mae(y_true, y_pred):
    """Mean Absolute Error loss: mean(|y_true - y_pred|)."""
    return np.abs(y_true - y_pred).mean()

def mae_derivative(y_true, y_pred):
    """Subgradient of MAE: sign(y_pred - y_true) / N, averaged over elements."""
    return np.sign(y_pred - y_true) / y_true.size

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1e-6) -> float:
    """Cross-entropy loss for one-hot targets and probability predictions.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded true labels of shape (batch, classes) or (classes,).
    y_pred : np.ndarray
        Predicted probabilities of the same shape as y_true.
    delta : float, default 1e-6
        Added for numerical stability to avoid log(0).

    Returns
    -------
    float
        Average cross-entropy across the batch.
    """

    y_pred = np.clip(y_pred, delta, 1.0 - delta)

    # handle single sample (2D) or batch
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of cross-entropy w.r.t. softmax logits: y_pred - y_true.

    When the output layer uses softmax, the gradient of the loss w.r.t.
    the logits simplifies to (y_pred - y_true), assuming y_pred are the
    softmax probabilities.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    return (y_pred - y_true)
