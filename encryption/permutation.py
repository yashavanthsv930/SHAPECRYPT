"""Deterministic Pixel Permutation Engine
Applies geometric order mappings and their exact mathematical inverses
to pixel matrices, ensuring 100% lossless image reconstruction.

Mathematical Convention:
------------------------
For a 1D pixel array `X` of length `N` and a permutation vector `P` where
`P[i]` is an index in `[0, N-1]`:
    - Forward Permutation (apply_permutation):
        Y[i] = X[P[i]]
      The output at position `i` takes the pixel originally located at `P[i]`.

    - Inverse Permutation (inverse_permutation):
      The inverse permutation vector `P_inv` satisfies:
        P_inv[P[i]] = i
      Consequently:
        Y[P_inv] = X
      Reconstruction is guaranteed to be exact and lossless:
        apply_inverse_permutation(apply_permutation(X, P), P) == X
"""

from __future__ import annotations

from typing import Union
import numpy as np

# Re-export shape generators for clean module imports
from encryption.shapes import (
    generate_circle_permutation,
    generate_triangle_permutation,
    generate_square_permutation,
    generate_star_permutation,
    generate_spiral_permutation,
)


def is_valid_permutation(
    permutation: Union[np.ndarray, list],
    expected_length: int | None = None,
) -> bool:
    """Validates that a sequence is a mathematically sound permutation of [0, N-1].

    Checks:
    1. 1-dimensional sequence
    2. Length matches expected_length (if supplied)
    3. Contains no duplicates (uniqueness)
    4. Range is strictly [0, N-1] with no missing coordinates (coverage)

    Args:
        permutation: Array or list of indices.
        expected_length: Optional expected length N.

    Returns:
        True if valid, False otherwise.
    """
    p = np.asarray(permutation, dtype=np.int64)
    if p.ndim != 1:
        return False
    n = len(p)
    if expected_length is not None and n != expected_length:
        return False
    if n == 0:
        return True
    if np.min(p) < 0 or np.max(p) >= n:
        return False
    return len(np.unique(p)) == n


def inverse_permutation(permutation: Union[np.ndarray, list]) -> np.ndarray:
    """Computes the exact mathematical inverse of a permutation vector.

    For a permutation vector `P`, the inverse `P_inv` satisfies:
        P_inv[P[i]] = i  for all i in [0, len(P)-1]

    Args:
        permutation: 1D array of indices representing the forward permutation.

    Returns:
        1D NumPy array with dtype=np.int64 representing the inverse permutation.

    Raises:
        ValueError: If the permutation is invalid (duplicates, out-of-range, etc.).
    """
    p = np.asarray(permutation, dtype=np.int64)
    if not is_valid_permutation(p):
        raise ValueError("Invalid permutation: must contain every index in [0, N-1] exactly once.")

    n = len(p)
    inv_p = np.empty(n, dtype=np.int64)
    inv_p[p] = np.arange(n, dtype=np.int64)
    return inv_p


def apply_permutation(
    pixel_array: np.ndarray,
    permutation: Union[np.ndarray, list],
) -> np.ndarray:
    """Applies a coordinate permutation to an image array.

    Supports:
    - 2D grayscale images of shape (H, W)
    - 3D color images of shape (H, W, C) (e.g. RGB, RGBA)
    - 1D flattened arrays of shape (N,) or (N, C)

    The spatial coordinates are permuted according to:
        shuffled_flat[i] = input_flat[permutation[i]]

    Args:
        pixel_array: Input NumPy array (2D, 3D, or 1D).
        permutation: 1D permutation vector of length equal to total spatial pixels.

    Returns:
        Shuffled NumPy array with the exact same shape and dtype as pixel_array.

    Raises:
        ValueError: If permutation length does not match total spatial pixels in pixel_array,
                    or if pixel_array has unsupported dimensionality.
    """
    arr = np.asarray(pixel_array)
    orig_shape = arr.shape
    orig_dtype = arr.dtype
    p = np.asarray(permutation, dtype=np.int64)

    if arr.ndim == 2:
        # Grayscale: (H, W)
        h, w = orig_shape
        total_pixels = h * w
        if len(p) != total_pixels:
            raise ValueError(
                f"Permutation length ({len(p)}) does not match image pixels ({total_pixels})."
            )
        flat = arr.ravel()
        shuffled_flat = flat[p]
        return shuffled_flat.reshape((h, w)).astype(orig_dtype)

    elif arr.ndim == 3:
        # Multi-channel image: (H, W, C)
        h, w, c = orig_shape
        total_pixels = h * w
        if len(p) != total_pixels:
            raise ValueError(
                f"Permutation length ({len(p)}) does not match image pixels ({total_pixels})."
            )
        # Reshape to (H * W, C) to permute pixel vectors atomically across all channels
        flat_channels = arr.reshape((total_pixels, c))
        shuffled_channels = flat_channels[p]
        return shuffled_channels.reshape((h, w, c)).astype(orig_dtype)

    elif arr.ndim == 1:
        # 1D flattened array
        if len(p) != len(arr):
            raise ValueError(
                f"Permutation length ({len(p)}) does not match array length ({len(arr)})."
            )
        return arr[p].astype(orig_dtype)

    else:
        raise ValueError(f"Unsupported array dimensions: {arr.ndim}. Expected 1D, 2D, or 3D.")


def apply_inverse_permutation(
    pixel_array: np.ndarray,
    permutation: Union[np.ndarray, list],
) -> np.ndarray:
    """Reverses a permutation applied to an image array, restoring the original data.

    Computes the mathematical inverse permutation and applies it to the shuffled array:
        recovered = apply_permutation(pixel_array, inverse_permutation(permutation))

    Args:
        pixel_array: Shuffled NumPy array.
        permutation: The forward permutation vector used during shuffling.

    Returns:
        Reconstructed NumPy array with the exact same shape and dtype.
    """
    inv_p = inverse_permutation(permutation)
    return apply_permutation(pixel_array, inv_p)
