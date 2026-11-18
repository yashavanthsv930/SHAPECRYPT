import numpy as np
from typing import Tuple

def npcr(original: np.ndarray, altered: np.ndarray) -> float:
    """Number of Pixels Change Rate.

    Computes the proportion of pixels that differ between two images.
    The images must have identical shape and dtype uint8.
    Returns a percentage (0‑100).
    """
    if original.shape != altered.shape:
        raise ValueError('Images must have the same dimensions for NPCR')
    diff = np.not_equal(original, altered)
    changed = np.count_nonzero(diff)
    total = original.size
    return (changed / total) * 100.0

def uaci(original: np.ndarray, altered: np.ndarray) -> float:
    """Unified Average Changing Intensity.

    Computes the average intensity difference per channel, normalized to 0‑100.
    Both images must share shape and dtype uint8.
    """
    if original.shape != altered.shape:
        raise ValueError('Images must have the same dimensions for UACI')
    # Absolute difference per element, sum over all elements
    diff = np.abs(original.astype(np.int16) - altered.astype(np.int16))
    total = original.size
    # Maximum possible difference per byte is 255
    return (np.sum(diff) / (total * 255)) * 100.0
