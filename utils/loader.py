# Load in data and convert to some 'standard format' (i.e. clean, normalize, split into sets, etc. etc.)

import numpy as np
import pandas as pd
import csv
from PIL import Image
import os


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize each feature column to the [0, 1] range."""
    if matrix.size == 0:
        return matrix.astype(np.float32)

    matrix = matrix.astype(np.float32)
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    denom = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    return (matrix - col_min) / denom


def load_student_dataset(
    file_path: str,
    label_col: str = 'G3',
    drop_columns: list[str] | None = None,
    scale_target: bool = True,
    target_max: float | None = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a student grade CSV with categorical encoding and feature scaling.
    
    By default, drops G1 and G2 to prevent label leakage, and scales G3 to [0,1].
    """
    df = pd.read_csv(file_path, sep=';')

    if label_col not in df.columns:
        raise ValueError(f"Expected column '{label_col}' to be present in the dataset.")

    # Default: drop G1 and G2 to prevent label leakage
    if drop_columns is None:
        drop_columns = ['G1', 'G2']

    # Drop label column and any other specified columns from features
    feature_df = df.drop(columns=[label_col] + drop_columns, errors='ignore')
    y_series = df[label_col]

    # Encode categorical features so the network can consume them.
    categorical_cols = feature_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols)

    X = _normalize_columns(feature_df.to_numpy(dtype=np.float32))
    Y = y_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    # Scale target to [0, 1] by default for better training
    if scale_target and target_max and target_max != 0:
        Y = Y / target_max

    return X, Y

def one_hot_encode(labels, num_classes):
    labels = np.asarray(labels)

    if labels.size == 0:
        return np.zeros((0, num_classes), dtype=float)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(int)

    return np.eye(num_classes, dtype=float)[labels]

def normalize_data(data, mode="scale255", range_min=0, range_max=1, data_min=None, data_max=None):

    # Skip empty arrays
    if data.size == 0:
        return data

    data = data.astype(np.float32)

    if mode == "scale255":
        return data / 255.0

    # Fallback: min-max (optionally with provided bounds)
    if data_min is None:
        data_min = np.min(data)

    if data_max is None:
        data_max = np.max(data)

    if data_max == data_min:
        return np.zeros_like(data) + range_min

    return (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min


def load_digits_dataset(path: str = 'data/digits', target_size: tuple = (16, 16)) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the digits dataset from the specified path.

    Args:
        path (str): The path to the digits dataset directory.
        target_size (tuple): The target size to resize the images to.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the images and labels.
    """

    images = []
    labels = []
    num_classes = 10

    for label in range(num_classes):
        dir_path = os.path.join(path, str(label))
        if not os.path.isdir(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            try:
                img_path = os.path.join(dir_path, filename)
                with Image.open(img_path) as img:
                    # Convert to grayscale, resize, and convert to numpy array
                    img = img.convert('L').resize(target_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    
                    # Flatten the image and append
                    images.append(img_array.flatten())
                    labels.append(label)
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

    # Convert lists to numpy arrays
    X = np.array(images)
    Y = np.array(labels, dtype=int)   # ensure integer indices

    # Normalize image data and one-hot encode labels
    X = normalize_data(X, mode="scale255")
    Y = one_hot_encode(Y, num_classes)

    return X, Y
