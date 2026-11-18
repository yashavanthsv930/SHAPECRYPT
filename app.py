import os
import json
import base64
import time
from pathlib import Path
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file, abort

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload limit

# Allowed file extensions
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_PACKAGE_EXT = {"shapecrypt", "enc"}

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

# Index route
@app.route("/")
def index():
    """Renders the main SHAPECRYPT web interface."""
    return render_template("index.html")

# Health endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify backend operational status."""
    return jsonify({
        "status": "online",
        "project": "SHAPECRYPT",
        "version": "0.1.0",
        "available_patterns": ["circle", "triangle", "square", "star", "spiral"],
        "security_layers": [
            "Layer 1: Geometric Pixel Permutation & RGB Channel Shuffle",
            "Layer 2: AES-256-GCM Authenticated Encryption"
        ]
    })

# Encryption route
from encryption.aes import encrypt_image
from encryption.pattern_chain import apply_pattern_chain
from encryption.rgb_shuffle import channel_permutation, inverse_channel_permutation
import numpy as np
from PIL import Image, ImageOps

RGB_ORDER_MAP = {
    "RGB": (0, 1, 2),
    "RBG": (0, 2, 1),
    "GRB": (1, 0, 2),
    "GBR": (1, 2, 0),
    "BRG": (2, 0, 1),
    "BGR": (2, 1, 0),
}


    # Validate image file
# Import metrics
from metrics.analysis import run_all

@app.route("/encrypt", methods=["POST"])
def encrypt_route():
    # Validate image file
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(img_file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error": f"Unsupported file type: {img_file.filename}"}), 400

    try:
        pil_img = ImageOps.exif_transpose(Image.open(img_file.stream))
        # The encryption engine supports L, RGB and RGBA arrays only.
        if pil_img.mode not in ("L", "RGB", "RGBA"):
            pil_img = pil_img.convert("RGBA" if "transparency" in pil_img.info else "RGB")
        original_arr = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {str(e)}"}), 400

    # Form data
    patterns = request.form.getlist("patterns")
    rgb_order_name = request.form.get("rgb_order", "default")
    password = request.form.get("password", "").strip()
    if not password:
        return jsonify({"error": "Password is required"}), 400

    # The password is the sole user secret and deterministically drives Layer 1.
    # No separate seed is accepted from the interface.
    seed = password

    total_start = time.perf_counter()
    # Total encryption timer starts here
    pattern_time = 0.0
    img_arr = original_arr.copy()
    if patterns:
        pat_start = time.perf_counter()
        try:
            img_arr = apply_pattern_chain(img_arr, patterns, seed)
        except Exception as e:
            return jsonify({"error": f"Pattern chain error: {str(e)}"}), 400
        pattern_time = time.perf_counter() - pat_start

    # Apply RGB permutation (timed)
    rgb_time = 0.0
    rgb_order_tuple = None
    if rgb_order_name != "default":
        rgb_order_tuple = RGB_ORDER_MAP.get(rgb_order_name.upper())
        if not rgb_order_tuple:
            return jsonify({"error": f"Invalid RGB order: {rgb_order_name}"}), 400
        if img_arr.ndim == 3:
            h, w, c = img_arr.shape
            if c == 3:
                rgb_start = time.perf_counter()
                img_arr = img_arr[..., rgb_order_tuple].copy()
                rgb_time = time.perf_counter() - rgb_start
            elif c == 4:
                rgb_start = time.perf_counter()
                img_arr[..., :3] = img_arr[..., rgb_order_tuple]
                rgb_time = time.perf_counter() - rgb_start

    # Store only the selected patterns; never store the password/seed in metadata.
    if patterns:
        pattern_info = {"type": "chain", "patterns": patterns}
    else:
        pattern_info = None

    # Encrypt (timed)
    aes_start = time.perf_counter()
    try:
        encrypted_pkg = encrypt_image(
            img_arr,
            password,
            pattern_info=pattern_info,
            rgb_order=rgb_order_tuple,
        )
    except Exception as e:
        return jsonify({"error": f"Encryption failed: {str(e)}"}), 500
    aes_time = time.perf_counter() - aes_start

    total_time = time.perf_counter() - total_start

    # Compute metrics (optional, not returned to UI in raw mode)
    timings = {
        "pattern_time_sec": pattern_time,
        "rgb_time_sec": rgb_time,
        "aes_encryption_time_sec": aes_time,
        "total_encryption_time_sec": total_time,
    }
    # Metrics are supplementary. A metrics failure must not invalidate a valid
    # encrypted package or turn a successful encryption into an HTTP 500.
    try:
        _ = run_all(original_arr, img_arr, encrypted_pkg, timings)
    except Exception:
        app.logger.exception("Optional metrics calculation failed after encryption")

    # Return encrypted package as raw bytes
    response = app.response_class(
        encrypted_pkg,
        mimetype="application/octet-stream",
        direct_passthrough=True,
    )
    response.headers["Content-Disposition"] = "attachment; filename=encrypted.shapecrypt"
    return response

# Decryption route
from encryption.aes import decrypt_image, _parse_metadata
from encryption.pattern_chain import reverse_pattern_chain

@app.route("/decrypt", methods=["POST"])
def decrypt_route():
    if "package" not in request.files:
        return jsonify({"error": "No encrypted package provided"}), 400
    pkg_file = request.files["package"]
    if pkg_file.filename == "":
        return jsonify({"error": "Empty package filename"}), 400
    if not allowed_file(pkg_file.filename, ALLOWED_PACKAGE_EXT):
        return jsonify({"error": f"Unsupported package type: {pkg_file.filename}"}), 400

    password = request.form.get("password", "").strip()
    if not password:
        return jsonify({"error": "Password is required for decryption"}), 400

    pkg_bytes = pkg_file.read()

    # Total decryption timer
    total_start = time.perf_counter()

    # New packages derive the deterministic permutation seed from the password.
    seed = password
    aes_start = time.perf_counter()
    try:
        plain_img_arr = decrypt_image(pkg_bytes, password)
    except Exception as e:
        return jsonify({"error": f"Decryption failed (possible wrong password or tampered data): {str(e)}"}), 400
    aes_time = time.perf_counter() - aes_start
    total_time = time.perf_counter() - total_start

    # Parse metadata for additional info
    try:
        meta_len = int.from_bytes(pkg_bytes[:4], byteorder="big")
        meta_blob = pkg_bytes[4:4 + meta_len]
        meta, _, _ = _parse_metadata(meta_blob)
    except Exception as e:
        return jsonify({"error": f"Failed to parse metadata: {str(e)}"}), 500

    # Backward compatibility: older packages may contain a custom seed. New
    # packages do not store it and always use the password as the seed.
    seed = (meta.get("pattern") or {}).get("seed", password)

    # Reverse RGB permutation
    rgb_order = meta.get("rgb_order")
    if rgb_order:
        inv_order = tuple(rgb_order.index(i) for i in range(3))
        if plain_img_arr.ndim == 3:
            h, w, c = plain_img_arr.shape
            if c == 3:
                # Permute RGB channels safely
                plain_img_arr = plain_img_arr[..., inv_order].copy()
            elif c == 4:
                # Preserve alpha, permute only RGB channels safely
                # Create a copy to ensure writeable array
                rgba_copy = plain_img_arr.copy()
                # Apply permutation to the RGB part
                rgba_copy[..., :3] = rgba_copy[..., :3][..., inv_order]
                plain_img_arr = rgba_copy

    # Reverse pattern chain
    pattern_info = meta.get("pattern")
    if pattern_info and pattern_info.get("type") == "chain":
        patterns = pattern_info.get("patterns", [])
        if patterns:
            plain_img_arr = reverse_pattern_chain(plain_img_arr, patterns, seed)

    # Reconstruct image
    try:
        recovered_img = Image.fromarray(plain_img_arr)
        img_io = BytesIO()
        recovered_img.save(img_io, format="PNG")
        img_io.seek(0)
    except Exception as e:
        return jsonify({"error": f"Failed to reconstruct image: {str(e)}"}), 500

    # Compute SHA256 of the PNG for debugging and log it
    import hashlib
    img_sha256 = hashlib.sha256(img_io.getvalue()).hexdigest()
    app.logger.info(f"Decrypted PNG SHA256: {img_sha256}")
    # Log metadata for verification
    app.logger.info(f"Metadata: width={meta.get('width')}, height={meta.get('height')}, mode={meta.get('mode')}, pattern={meta.get('pattern')}, rgb_order={meta.get('rgb_order')}")
    # Return PNG bytes with proper MIME type and debug header
    response = app.response_class(
        img_io.getvalue(),
        mimetype="image/png",
        direct_passthrough=True,
    )
    response.headers["Content-Disposition"] = "attachment; filename=decrypted.png"
    response.headers["X-Image-SHA256"] = img_sha256
    return response

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("  SHAPECRYPT Backend Server Starting")
    print("  Access the application at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=True)
