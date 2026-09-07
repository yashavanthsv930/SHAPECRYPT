"""Multi‑pattern geometric chaining utilities.

Provides a clean, deterministic, reversible API for applying one or more
geometric permutations in a user‑specified order.

The functions deliberately avoid any side‑effects on the existing permutation
implementations – they only orchestrate calls to the existing generators and
apply/undo helpers.
"""

from __future__ import annotations

from typing import Sequence, List

import numpy as np

from encryption.shapes import (
    generate_circle_permutation,
    generate_triangle_permutation,
    generate_square_permutation,
    generate_star_permutation,
    generate_spiral_permutation,
)
from encryption.permutation import apply_permutation, apply_inverse_permutation

# Mapping of pattern name → generator function
_PATTERN_GENERATORS = {
    "circle": generate_circle_permutation,
    "triangle": generate_triangle_permutation,
    "square": generate_square_permutation,
    "star": generate_star_permutation,
    "spiral": generate_spiral_permutation,
}


def _validate_patterns(patterns: Sequence[str]) -> List[str]:
    """Validate and normalise a sequence of pattern identifiers.

    Parameters
    ----------
    patterns: Sequence[str]
        The user‑supplied pattern identifiers. The function accepts any
        iterable of strings – list, tuple, etc.

    Returns
    -------
    List[str]
        Normalised (lower‑case) pattern names.

    Raises
    ------
    ValueError
        If ``patterns`` is empty, not an iterable of strings, or contains an
        unsupported name.
    """
    if not patterns:
        raise ValueError("Pattern list must contain at least one pattern.")

    normalised: List[str] = []
    for p in patterns:
        if not isinstance(p, str):
            raise ValueError(f"Pattern names must be strings, got {type(p)!r}.")
        name = p.lower()
        if name not in _PATTERN_GENERATORS:
            raise ValueError(f"Unsupported pattern '{p}'. Supported: {list(_PATTERN_GENERATORS)}")
        normalised.append(name)
    return normalised


def apply_pattern_chain(image: np.ndarray, patterns: Sequence[str], seed: int | str | bytes) -> np.ndarray:
    """Apply a deterministic chain of geometric permutations.

    The function leaves the original ``image`` untouched and returns a new
    ``ndarray`` with the same dtype and shape.

    Parameters
    ----------
    image: np.ndarray
        Input image – either 2‑D (grayscale) or 3‑D (RGB/RGBA).
    patterns: Sequence[str]
        Ordered list of pattern identifiers. Valid identifiers are
        ``"circle"``, ``"triangle"``, ``"square"``, ``"star"`` and ``"spiral"``.
    seed: int | str | bytes
        Seed material that is passed unchanged to each geometric generator.
        Using the same ``seed`` guarantees reproducibility across the whole
        chain.

    Returns
    -------
    np.ndarray
        Image after all permutations have been applied.
    """
    norm_patterns = _validate_patterns(patterns)

    # Determine image dimensions
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Unsupported image dimensions for pattern chaining.")

    result = image.copy()
    for name in norm_patterns:
        gen = _PATTERN_GENERATORS[name]
        perm = gen(height, width, seed)
        result = apply_permutation(result, perm)
    return result


def reverse_pattern_chain(image: np.ndarray, patterns: Sequence[str], seed: int | str | bytes) -> np.ndarray:
    """Undo a previously applied pattern chain.

    The reversal uses the exact same ``seed`` and applies the inverse
    permutations in *reverse* order, guaranteeing byte‑for‑byte restoration.

    Parameters
    ----------
    image: np.ndarray
        Image that has already been processed by ``apply_pattern_chain``.
    patterns: Sequence[str]
        The original pattern list used during forward processing.
    seed: int | str | bytes
        The seed that was used for the forward chain.

    Returns
    -------
    np.ndarray
        Restored image.
    """
    norm_patterns = _validate_patterns(patterns)

    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Unsupported image dimensions for pattern chaining.")

    result = image.copy()
    for name in reversed(norm_patterns):
        gen = _PATTERN_GENERATORS[name]
        perm = gen(height, width, seed)
        result = apply_inverse_permutation(result, perm)
    return result

__all__ = ["apply_pattern_chain", "reverse_pattern_chain"]
