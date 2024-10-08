import math

import numpy as np


def calculate_distance(embeddings1: np.ndarray, embeddings2: np.ndarray, distance_metric: int = 0) -> np.ndarray:
    """
    Calculate the distance between two embeddings using either Euclidean or Cosine similarity.

    Args:
        embeddings1: First set of embeddings.
        embeddings2: Second set of embeddings.
        distance_metric: Metric to use (0 for Euclidean, 1 for Cosine similarity).

    Returns:
        A numpy array containing the calculated distances.
    """
    if distance_metric == 0:
        # Euclidean distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), axis=1)
    elif distance_metric == 1:
        # Cosine similarity-based distance
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise ValueError(f"Undefined distance metric: {distance_metric}")

    return dist
