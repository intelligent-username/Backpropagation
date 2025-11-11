"""
Activation functions & their derivatives.
"""

# Activation Functions
# So far [ReLU, Sigmoid, Tanh]

import numpy as np

def relu(x):
    """Rectified Linear Unit activation.

    Parameters
    ----------
    x : np.ndarray or float
        Input tensor or scalar.

    Returns
    -------
    np.ndarray or float
        Elementwise max(0, x) with the same shape as x.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU activation.

    Notes
    -----
    Not defined at 0, just pick 0.
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """
    Sigmoid activation
    1 / (1 + exp(-x)).
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    The derivative of the Sigmoid: 
    s * (1 - s), where s = sigmoid(x).
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """
    Hyperbolic tangent activation applied elementwise.
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of tanh activation: 1 - tanh(x)^2.
    """
    return 1 - (np.tanh(x) ** 2)

def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over classes.

    Parameters
    ----------
    z : np.ndarray, shape (batch, classes)
        Unnormalized logits per class.

    Returns
    -------
    np.ndarray, shape (batch, classes)
        Probabilities that sum to 1 across axis=1. Uses max-shift for stability.
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

# NOTE: dL/dz = (y_pred - y_true)
# That's why we "don't" multiply by the derivative again
def softmax_derivative_passthrough(z: np.ndarray) -> np.ndarray:
    """Placeholder derivative for softmax when paired with cross-entropy.

    Returns an array of ones with the same shape as z. The true gradient is
    handled in the cross-entropy loss derivative.
    """
    return np.ones_like(z)
