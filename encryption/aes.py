"""Layer 2: Cryptographic Security Engine (AES-256-GCM)

Implements:
- Argon2id / PBKDF2-HMAC-SHA256 key derivation from user password
- Cryptographically secure random salt & nonce generation
- AES-256-GCM authenticated encryption and authentication tag verification
- Non‑secret metadata packing and unpacking (JSON‑based)

The functions in this module operate on NumPy ``ndarray`` images that have already
been processed by the geometric and RGB permutation layers.  No plaintext image
data is ever written to disk.
"""

import os
import json
import base64
from typing import Any, Dict, Tuple
import numpy as np

# Cryptography primitives
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Argon2id support – optional fallback to PBKDF2 if unavailable
try:
    from argon2.low_level import (
        hash_secret_raw,
        Type as Argon2Type,
    )
    _ARGON2_AVAILABLE = True
except Exception:  # pragma: no cover – defensive fallback
    _ARGON2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (all sizes are in bytes)
# ---------------------------------------------------------------------------
VERSION = "SHAPECRYPTv1"          # Package format identifier
KEY_SIZE_BYTES = 32               # 256‑bit AES key
SALT_SIZE_BYTES = 16              # 128‑bit random salt for KDF
NONCE_SIZE_BYTES = 12             # 96‑bit nonce for AES‑GCM (recommended size)

# ---------------------------------------------------------------------------
# Key‑derivation helpers
# ---------------------------------------------------------------------------
def _derive_key_argon2id(password: str, salt: bytes) -> bytes:
    """Derive a 256‑bit AES key using Argon2id.

    Parameters
    ----------
    password: str
        User supplied password (UTF‑8).
    salt: bytes
        Random salt generated per‑encryption (must be stored with the ciphertext).

    Returns
    -------
    bytes
        32‑byte key suitable for AES‑256‑GCM.
    """
    # Reasonable defaults – strong enough for a college project while staying fast
    time_cost = 2               # number of passes over memory
    memory_cost = 102400        # 100 MiB
    parallelism = 4             # CPU parallelism
    hash_len = KEY_SIZE_BYTES
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=hash_len,
        type=Argon2Type.ID,
    )


def _derive_key_pbkdf2(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    """Derive a 256‑bit AES key using PBKDF2‑HMAC‑SHA256.

    This function is used when Argon2id is unavailable. The iteration count is
    chosen to provide a reasonable work factor on typical hardware.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE_BYTES,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode("utf-8"))


def _derive_key(password: str, salt: bytes, kdf_type: str = "argon2id") -> Tuple[bytes, str, Dict[str, Any]]:
    """Derive the AES‑256‑GCM key and return diagnostic information.

    Returns ``(key, actual_kdf_used, params)`` where ``params`` holds the numeric
    arguments that were used for the selected KDF – useful for reproducibility
    and for the test suite.
    """
    if kdf_type == "argon2id" and _ARGON2_AVAILABLE:
        key = _derive_key_argon2id(password, salt)
        params = {
            "time_cost": 2,
            "memory_cost": 102400,
            "parallelism": 4,
            "hash_len": KEY_SIZE_BYTES,
        }
        return key, "argon2id", params
    # Fallback to PBKDF2
    key = _derive_key_pbkdf2(password, salt)
    params = {"iterations": 200_000, "algorithm": "SHA256"}
    return key, "pbkdf2_hmac_sha256", params

# ---------------------------------------------------------------------------
# Metadata handling – JSON blob prefixed by its length (4 bytes, big‑endian)
# ---------------------------------------------------------------------------
def _build_metadata(
    mode: str,
    width: int,
    height: int,
    pattern_info: Dict[str, Any] | None,
    rgb_order: Tuple[int, int, int] | None,
    kdf_type: str,
    kdf_params: Dict[str, Any],
    salt: bytes,
    nonce: bytes,
) -> bytes:
    """Serialize non‑secret encryption metadata.

    ``salt`` and ``nonce`` are binary values; they are base64‑encoded so the JSON
    remains UTF‑8 text.
    """
    meta = {
        "version": VERSION,
        "mode": mode,
        "width": width,
        "height": height,
        "pattern": pattern_info,
        "rgb_order": rgb_order,
        "kdf": {
            "type": kdf_type,
            "params": kdf_params,
        },
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
    }
    return json.dumps(meta, separators=(",", ":")).encode("utf-8")


def _parse_metadata(blob: bytes) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Parse the JSON metadata blob and return ``(meta, salt, nonce)``.
    """
    meta = json.loads(blob.decode("utf-8"))
    salt = base64.b64decode(meta["salt"])
    nonce = base64.b64decode(meta["nonce"])
    return meta, salt, nonce

# ---------------------------------------------------------------------------
# Public API – encryption & decryption
# ---------------------------------------------------------------------------
def encrypt_image(
    image: np.ndarray,
    password: str,
    *,
    pattern_info: Dict[str, Any] | None = None,
    rgb_order: Tuple[int, int, int] | None = None,
) -> bytes:
    """Encrypt a NumPy image with AES‑256‑GCM.

    The image should already have undergone geometric permutation and RGB
    channel permutation. The function returns a binary package that can be saved
    directly to disk.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy ndarray")

    # Determine mode and dimensions
    if image.ndim == 2:
        mode = "L"
        height, width = image.shape
    elif image.ndim == 3:
        height, width, channels = image.shape
        if channels == 3:
            mode = "RGB"
        elif channels == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Unsupported image dimensionality: {image.ndim}")

    # Serialize raw pixel bytes (project uses uint8 throughout)
    plaintext = image.astype(np.uint8).tobytes()

    # Generate fresh cryptographic salt and nonce
    salt = os.urandom(SALT_SIZE_BYTES)
    nonce = os.urandom(NONCE_SIZE_BYTES)

    # Derive AES‑256 key
    key, kdf_used, kdf_params = _derive_key(password, salt)

    # Build metadata blob
    meta_blob = _build_metadata(
        mode=mode,
        width=width,
        height=height,
        pattern_info=pattern_info,
        rgb_order=rgb_order,
        kdf_type=kdf_used,
        kdf_params=kdf_params,
        salt=salt,
        nonce=nonce,
    )

    # Perform authenticated encryption – AESGCM appends a 16‑byte tag
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, meta_blob)

    # Package layout: 4‑byte big‑endian metadata length, metadata, ciphertext
    meta_len = len(meta_blob).to_bytes(4, byteorder="big")
    return meta_len + meta_blob + ciphertext


def decrypt_image(package: bytes, password: str) -> np.ndarray:
    """Decrypt a package produced by :func:`encrypt_image`.

    The function validates the authentication tag; any tampering results in an
    exception from the ``cryptography`` library, satisfying the required security
    tests.
    """
    if len(package) < 4:
        raise ValueError("Package is too short to contain a metadata length header")
    meta_len = int.from_bytes(package[:4], byteorder="big")
    meta_blob = package[4 : 4 + meta_len]
    ciphertext = package[4 + meta_len :]

    meta, salt, nonce = _parse_metadata(meta_blob)

    # Re‑derive the key using stored KDF parameters
    kdf_type = meta["kdf"]["type"]
    key, _, _ = _derive_key(password, salt, kdf_type=kdf_type)

    # Decrypt – raises an exception if authentication fails
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, meta_blob)

    # Re‑construct NumPy array from raw bytes
    mode = meta["mode"]
    height = meta["height"]
    width = meta["width"]
    if mode == "L":
        shape = (height, width)
    elif mode == "RGB":
        shape = (height, width, 3)
    elif mode == "RGBA":
        shape = (height, width, 4)
    else:
        raise ValueError(f"Unsupported image mode in metadata: {mode}")
    return np.frombuffer(plaintext, dtype=np.uint8).reshape(shape)

__all__ = [
    "encrypt_image",
    "decrypt_image",
    "KEY_SIZE_BYTES",
    "SALT_SIZE_BYTES",
    "NONCE_SIZE_BYTES",
]
