"""Tests for multi‑pattern geometric chaining integrated with AES.
"""
import pytest
import numpy as np

from encryption.pattern_chain import apply_pattern_chain, reverse_pattern_chain
from encryption.aes import encrypt_image, decrypt_image

# Helper to generate a random image of a given mode
def random_image(mode: str, height: int = 8, width: int = 8) -> np.ndarray:
    if mode == "L":
        return np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    elif mode == "RGB":
        return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    elif mode == "RGBA":
        return np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode {mode}")

# Parameterised pattern chain lists (including single, double, triple, full)
PATTERN_COMBINATIONS = [
    ["circle"],
    ["triangle"],
    ["square"],
    ["star"],
    ["spiral"],
    ["circle", "triangle"],
    ["triangle", "square"],
    ["square", "star"],
    ["star", "spiral"],
    ["circle", "triangle", "spiral"],
    ["star", "square", "triangle"],
    ["circle", "triangle", "square", "star", "spiral"],
]

# Image modes to test
IMAGE_MODES = ["L", "RGB", "RGBA"]

SEED = "testseed"
PASSWORD = "correcthorsebatterystaple"

@pytest.mark.parametrize("patterns", PATTERN_COMBINATIONS)
@pytest.mark.parametrize("mode", IMAGE_MODES)
def test_chain_encrypt_decrypt_roundtrip(patterns, mode):
    img = random_image(mode, height=8, width=8)
    # Apply chain
    chained = apply_pattern_chain(img, patterns, SEED)
    # Build metadata for chain
    pattern_info = {"type": "chain", "patterns": patterns}
    pkg = encrypt_image(chained, PASSWORD, pattern_info=pattern_info)
    recovered = decrypt_image(pkg, PASSWORD)
    # Reverse chain
    restored = reverse_pattern_chain(recovered, patterns, SEED)
    assert np.array_equal(img, restored)

def test_invalid_pattern_name():
    img = random_image("L")
    with pytest.raises(ValueError):
        apply_pattern_chain(img, ["invalid"], SEED)

"""End of tests."""
