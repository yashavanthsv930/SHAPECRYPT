import numpy as np
from typing import Dict, Tuple

def pixel_correlation(image_arr: np.ndarray) -> Dict[str, float]:
    """Calculate horizontal, vertical and diagonal correlation coefficients.

    Returns a dict with keys ``horizontal``, ``vertical`` and ``diagonal``.
    For multi‑channel images the correlation is computed per channel and then
    averaged.
    """
    if image_arr.ndim == 2:
        # Grayscale – add a dummy channel dimension for uniform processing
        img = image_arr[:, :, np.newaxis]
    elif image_arr.ndim == 3:
        img = image_arr
    else:
        raise ValueError('Unsupported image dimensions for correlation')

    h, w, c = img.shape
    corr = {'horizontal': 0.0, 'vertical': 0.0, 'diagonal': 0.0}
    for ch in range(c):
        channel = img[..., ch]
        # Horizontal pairs
        x_h = channel[:, :-1].ravel()
        y_h = channel[:, 1:].ravel()
        # Vertical pairs
        x_v = channel[:-1, :].ravel()
        y_v = channel[1:, :].ravel()
        # Diagonal (down‑right) pairs
        x_d = channel[:-1, :-1].ravel()
        y_d = channel[1:, 1:].ravel()
        # Correlation coefficient – handle constant arrays safely
        def safe_corr(x, y):
            if x.size == 0 or np.all(x == x[0]) or np.all(y == y[0]):
                return 0.0
            return np.corrcoef(x, y)[0, 1]
        corr['horizontal'] += safe_corr(x_h, y_h)
        corr['vertical'] += safe_corr(x_v, y_v)
        corr['diagonal'] += safe_corr(x_d, y_d)
    # Average over channels
    corr = {k: v / c for k, v in corr.items()}
    return corr
