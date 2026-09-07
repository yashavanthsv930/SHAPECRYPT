"""Key-Dependent RGB Channel Permutation

Provides deterministic, reversible channel ordering for RGB(A) images based on a cryptographic seed.
Supported orders: RGB, RBG, GRB, GBR, BRG, BGR.
The alpha channel in RGBA images is left unchanged.
"""

import hashlib
from typing import Union
import numpy as np

# Re‑use the seed normalisation utility from the shapes module for consistency.
from .shapes import _normalize_seed

# Valid channel permutations for the three colour channels.
VALID_CHANNEL_ORDERS = [
    (0, 1, 2),  # RGB
    (0, 2, 1),  # RBG
    (1, 0, 2),  # GRB
    (1, 2, 0),  # GBR
    (2, 0, 1),  # BRG
    (2, 1, 0),  # BGR
]

def _select_order(seed: Union[str, bytes, int]) -> tuple[int, int, int]:
    """Derive a deterministic channel order from *seed*.

    The algorithm mirrors the deterministic PRNG approach used in the
    geometric permutation generators: the seed is normalised, a domain‑specific
    SHA‑256 hash is computed, and the first 16 bytes are interpreted as a
    128‑bit integer to seed NumPy's PCG64 RNG. The RNG is *not* used for the
    final selection – instead the hash digest directly indexes into
    ``VALID_CHANNEL_ORDERS`` to guarantee a pure, repeatable mapping.
    """
    seed_bytes = _normalize_seed(seed)
    domain = b"SHAPECRYPT_RGB_CHANNEL_TRAVERSAL_V1_"
    digest = hashlib.sha256(domain + seed_bytes).digest()
    # Use the first two bytes (16 bits) for indexing – ample space for six orders.
    idx = int.from_bytes(digest[:2], "big") % len(VALID_CHANNEL_ORDERS)
    return VALID_CHANNEL_ORDERS[idx]

def channel_permutation(image: np.ndarray, seed: Union[str, bytes, int]) -> np.ndarray:
    """Permute the colour channels of *image* according to a deterministic order.

    Parameters
    ----------
    image:
        ``numpy.ndarray`` with shape ``(H, W, C)`` where ``C`` is 1, 3, or 4.
        * Grayscale (``C == 1``) is returned unchanged.
        * RGB (``C == 3``) has its three channels reordered.
        * RGBA (``C == 4``) has its first three channels reordered while the
          alpha channel (index 3) is preserved.
    seed:
        Cryptographic seed used to select the permutation. The same ``seed``
        always yields the same channel ordering.

    Returns
    -------
    np.ndarray
        An image with channels permuted according to the selected order.
    """
    if image.ndim != 3:
        raise ValueError("Image must have three dimensions (H, W, C).")
    h, w, c = image.shape
    if c == 1:
        # Grayscale – nothing to permute.
        return image.copy()
    if c not in (3, 4):
        raise ValueError("Supported channel counts are 1 (grayscale), 3 (RGB), or 4 (RGBA).")

    order = _select_order(seed)
    if c == 3:
        return image[..., order].copy()
    # c == 4: preserve alpha channel.
    permuted = image.copy()
    permuted[..., :3] = image[..., order]
    return permuted

def inverse_channel_permutation(image: np.ndarray, seed: Union[str, bytes, int]) -> np.ndarray:
    """Restore the original channel ordering of an image that was processed with ``channel_permutation``.

    The inverse operation computes the inverse of the deterministic order derived
    from the same ``seed`` and applies it to the colour channels.
    """
    if image.ndim != 3:
        raise ValueError("Image must have three dimensions (H, W, C).")
    h, w, c = image.shape
    if c == 1:
        return image.copy()
    if c not in (3, 4):
        raise ValueError("Supported channel counts are 1 (grayscale), 3 (RGB), or 4 (RGBA).")

    order = _select_order(seed)
    # Compute inverse permutation for the three colour channels.
    inv_order = tuple(order.index(i) for i in range(3))
    if c == 3:
        return image[..., inv_order].copy()
    # c == 4 – keep alpha unchanged.
    restored = image.copy()
    restored[..., :3] = image[..., inv_order]
    return restored

__all__ = ["channel_permutation", "inverse_channel_permutation", "VALID_CHANNEL_ORDERS"]
