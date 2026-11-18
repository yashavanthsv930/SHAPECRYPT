import numpy as np
import math
from collections import Counter

def shannon_entropy(image_arr: np.ndarray) -> float:
    """Calculate Shannon entropy of an image array.

    Supports grayscale (2D) and color (3D) images. The calculation is performed
    on the flattened byte values (0‑255) across all channels.
    """
    # Ensure dtype is uint8 for histogram counting
    flat = image_arr.astype(np.uint8).ravel()
    # Count occurrences of each byte value
    value, counts = np.unique(flat, return_counts=True)
    probabilities = counts / counts.sum()
    # Shannon entropy in bits
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)
