import numpy as np
from typing import Dict, Tuple

def image_histogram(image_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Return histogram (256 bins) for each channel.

    Returns a dict with keys:
        - 'gray' for 2‑D grayscale images
        - 'r', 'g', 'b' for RGB/A images (alpha ignored)
    Each value is a NumPy array of shape (256,) containing counts.
    """
    if image_arr.ndim == 2:
        # Grayscale image
        hist, _ = np.histogram(image_arr, bins=256, range=(0, 255))
        return {'gray': hist}
    elif image_arr.ndim == 3:
        # Color image – ignore alpha channel if present
        channels = image_arr.shape[2]
        result = {}
        channel_names = ['r', 'g', 'b']
        for i in range(min(3, channels)):
            hist, _ = np.histogram(image_arr[..., i], bins=256, range=(0, 255))
            result[channel_names[i]] = hist
        return result
    else:
        raise ValueError('Unsupported image dimensions for histogram')
