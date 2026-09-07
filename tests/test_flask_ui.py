import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

# Import the Flask app
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client

def create_test_image():
    # Create a simple 10x10 red PNG image in memory
    img = Image.new("RGB", (10, 10), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

def test_encrypt_decrypt_roundtrip(client):
    img_bytes = create_test_image()
    # Encrypt the image
    data = {
        "rgb_order": "default",
        "password": "testpass",
    }
    # Patterns field left empty (no chain)
    response = client.post(
        "/encrypt",
        data={
            "image": (img_bytes, "test.png"),
            **data,
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, f"Encrypt failed: {response.get_json()}"
    encrypted_pkg = response.data
    assert len(encrypted_pkg) > 0

    # Decrypt the package with correct password
    pkg_bytes = io.BytesIO(encrypted_pkg)
    dec_response = client.post(
        "/decrypt",
        data={
            "package": (pkg_bytes, "encrypted.shapecrypt"),
            "password": "testpass",
        },
        content_type="multipart/form-data",
    )
    assert dec_response.status_code == 200, f"Decrypt failed: {dec_response.get_json()}"
    decrypted_img_bytes = dec_response.data

    # Compare original and decrypted images byte-for-byte
    original_bytes = create_test_image().getvalue()
    assert original_bytes == decrypted_img_bytes

def test_decrypt_wrong_password(client):
    img_bytes = create_test_image()
    response = client.post(
        "/encrypt",
        data={
            "image": (img_bytes, "test.png"),
            "rgb_order": "default",
            "password": "testpass",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    encrypted_pkg = response.data

    # Attempt decryption with wrong password
    pkg_bytes = io.BytesIO(encrypted_pkg)
    dec_response = client.post(
        "/decrypt",
        data={
            "package": (pkg_bytes, "encrypted.shapecrypt"),
            "password": "wrongpass",
        },
        content_type="multipart/form-data",
    )
    assert dec_response.status_code == 400
    json_data = dec_response.get_json()
    assert json_data is not None
    assert "error" in json_data

def test_decrypt_tampered_package(client):
    img_bytes = create_test_image()
    response = client.post(
        "/encrypt",
        data={
            "image": (img_bytes, "test.png"),
            "rgb_order": "default",
            "password": "testpass",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    encrypted_pkg = bytearray(response.data)
    # Tamper with the ciphertext (flip a byte after metadata length)
    if len(encrypted_pkg) > 10:
        encrypted_pkg[10] ^= 0xFF
    tampered_pkg = io.BytesIO(encrypted_pkg)
    dec_response = client.post(
        "/decrypt",
        data={
            "package": (tampered_pkg, "encrypted.shapecrypt"),
            "password": "testpass",
        },
        content_type="multipart/form-data",
    )
    # Should fail due to authentication error
    assert dec_response.status_code == 400
    json_data = dec_response.get_json()
    assert json_data is not None
    assert "error" in json_data
