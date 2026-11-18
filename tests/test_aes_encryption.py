"""Tests for AES‑256‑GCM encryption with metadata AAD.
These tests verify:
- Correct decryption with the right password.
- That any tampering of the metadata, salt, nonce, ciphertext or tag
  results in a decryption failure (cryptography.exceptions.InvalidTag).
- End‑to‑end byte‑for‑byte reversibility for RGB, RGBA and grayscale images.
"""

import copy
import json
import base64
import os
import pytest
import numpy as np

from encryption.aes import encrypt_image, decrypt_image

# Helper to generate random images of various modes

def random_image(mode: str, height: int = 8, width: int = 8) -> np.ndarray:
    if mode == "L":
        return np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    elif mode == "RGB":
        return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    elif mode == "RGBA":
        return np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode {mode}")

# ---------------------------------------------------------------------------
# Core correctness test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_encrypt_decrypt_roundtrip(mode):
    img = random_image(mode)
    password = "correcthorsebatterystaple"
    package = encrypt_image(img, password)
    recovered = decrypt_image(package, password)
    assert np.array_equal(img, recovered)

# ---------------------------------------------------------------------------
# Tampering helpers
# ---------------------------------------------------------------------------

def split_package(package: bytes):
    meta_len = int.from_bytes(package[:4], byteorder="big")
    meta_blob = package[4 : 4 + meta_len]
    ciphertext = package[4 + meta_len :]
    return meta_len, meta_blob, ciphertext

def rebuild_package(meta_blob: bytes, ciphertext: bytes) -> bytes:
    meta_len = len(meta_blob).to_bytes(4, byteorder="big")
    return meta_len + meta_blob + ciphertext

def tamper_meta(meta_blob: bytes, field: str, new_value) -> bytes:
    meta = json.loads(meta_blob.decode("utf-8"))
    meta[field] = new_value
    return json.dumps(meta, separators=(",", ":")).encode("utf-8")

# ---------------------------------------------------------------------------
# Metadata tampering tests
# ---------------------------------------------------------------------------
@pytest.fixture
def base_package():
    img = random_image("RGB", 4, 4)
    pwd = "s3cureP@ss"
    return encrypt_image(img, pwd), img, pwd

@pytest.mark.parametrize("field,new_value", [
    ("mode", "L"),
    ("width", 999),
    ("height", 999),
    ("pattern", {"type": "circle", "seed": 42}),
    ("rgb_order", [0, 2, 1]),
    ("kdf", {"type": "pbkdf2_hmac_sha256", "params": {"iterations": 1}}),
])
def test_metadata_tampering(field, new_value, base_package):
    package, _, pwd = base_package
    meta_len, meta_blob, ciphertext = split_package(package)
    # Tamper metadata
    tampered_meta = tamper_meta(meta_blob, field, new_value)
    tampered_pkg = rebuild_package(tampered_meta, ciphertext)
    with pytest.raises(Exception):  # cryptography raises InvalidTag on AAD mismatch
        decrypt_image(tampered_pkg, pwd)

# ---------------------------------------------------------------------------
# Salt and nonce tampering (both are base64‑encoded strings in the JSON)
# ---------------------------------------------------------------------------
def test_salt_tampering(base_package):
    package, _, pwd = base_package
    meta_len, meta_blob, ciphertext = split_package(package)
    meta = json.loads(meta_blob.decode("utf-8"))
    # Corrupt the salt by flipping a byte after decoding
    raw_salt = base64.b64decode(meta["salt"])
    corrupted = bytearray(raw_salt)
    corrupted[0] ^= 0xFF
    meta["salt"] = base64.b64encode(bytes(corrupted)).decode("ascii")
    tampered_meta = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    tampered_pkg = rebuild_package(tampered_meta, ciphertext)
    with pytest.raises(Exception):
        decrypt_image(tampered_pkg, pwd)

def test_nonce_tampering(base_package):
    package, _, pwd = base_package
    meta_len, meta_blob, ciphertext = split_package(package)
    meta = json.loads(meta_blob.decode("utf-8"))
    raw_nonce = base64.b64decode(meta["nonce"])
    corrupted = bytearray(raw_nonce)
    corrupted[0] ^= 0xAA
    meta["nonce"] = base64.b64encode(bytes(corrupted)).decode("ascii")
    tampered_meta = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    tampered_pkg = rebuild_package(tampered_meta, ciphertext)
    with pytest.raises(Exception):
        decrypt_image(tampered_pkg, pwd)

# ---------------------------------------------------------------------------
# Ciphertext and tag tampering
# ---------------------------------------------------------------------------
def test_ciphertext_tampering(base_package):
    package, _, pwd = base_package
    meta_len, meta_blob, ciphertext = split_package(package)
    # Flip a byte in the ciphertext (which includes the tag at the end)
    corrupted = bytearray(ciphertext)
    corrupted[0] ^= 0x01
    tampered_pkg = rebuild_package(meta_blob, bytes(corrupted))
    with pytest.raises(Exception):
        decrypt_image(tampered_pkg, pwd)

def test_tag_tampering(base_package):
    package, _, pwd = base_package
    meta_len, meta_blob, ciphertext = split_package(package)
    # The tag is the last 16 bytes of the ciphertext (AESGCM default)
    if len(ciphertext) < 16:
        pytest.skip("Ciphertext too short to contain a tag")
    corrupted = bytearray(ciphertext)
    corrupted[-1] ^= 0xFF
    tampered_pkg = rebuild_package(meta_blob, bytes(corrupted))
    with pytest.raises(Exception):
        decrypt_image(tampered_pkg, pwd)

