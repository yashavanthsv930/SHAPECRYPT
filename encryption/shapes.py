"""Layer 1: Geometric Shape Traversal & Coordinate Generation
Defines coordinate generation functions for geometric pixel permutations:
- Circle (implemented)
- Triangle (implemented)
- Square (implemented)
- Star (implemented)
- Spiral (implemented)

IMPORTANT SECURITY NOTE:
Geometric pixel permutations provide visual obfuscation and disrupt spatial
autocorrelation. They represent a custom transformation layer and do NOT replace
the cryptographic security provided by AES-256-GCM in Layer 2.
"""

from __future__ import annotations

import hashlib
from typing import Union
import numpy as np

# Registered pattern identifiers for Layer 1
AVAILABLE_PATTERNS = ["circle", "triangle", "square", "star", "spiral"]


def _normalize_seed(seed: Union[str, bytes, int]) -> bytes:
    """Normalizes various seed formats into a canonical byte sequence.

    Args:
        seed: Key material provided as string, bytes, bytearray, or integer.

    Returns:
        Canonical bytes representation of the seed.
    """
    if isinstance(seed, str):
        return seed.encode("utf-8")
    elif isinstance(seed, (bytes, bytearray)):
        return bytes(seed)
    elif isinstance(seed, int):
        bit_len = max(seed.bit_length(), 1)
        byte_len = (bit_len + 7) // 8
        return seed.to_bytes(max(byte_len, 8), byteorder="big", signed=True)
    else:
        return str(seed).encode("utf-8")


def generate_circle_permutation(
    height: int,
    width: int,
    seed: Union[str, bytes, int],
    delta_r: float = 1.0,
) -> np.ndarray:
    """Generates a deterministic, reversible circle-based pixel permutation.

    Mathematical Traversal Formulation:
    -----------------------------------
    1. Grid Coordinates:
       For an image of dimensions (H, W), pixel coordinates (r, c) span:
       r in [0, H-1], c in [0, W-1].
       The geometric center is calculated as:
           c_r = (H - 1) / 2.0
           c_c = (W - 1) / 2.0

    2. Polar Coordinate Conversion:
       For every pixel (r, c):
           dr = r - c_r
           dc = c - c_c
           radius = sqrt(dr^2 + dc^2)
           angle  = atan2(dr, dc) mod 2*pi  (in [0, 2*pi))

    3. Concentric Circular Shells (Equal/Similar Radius Grouping):
       Pixels are grouped into concentric circular bands of radial thickness delta_r:
           band = floor(radius / delta_r)

    4. Key-Derived Deterministic Parameterization:
       Using SHA-256 domain separation, a 128-bit integer seed is derived from
       the caller's key material to seed a NumPy PCG64 pseudo-random generator:
           - Per-band angular phase rotation: phi_k in [0, 2*pi)
           - Per-band traversal direction: dir_k in {+1, -1} (CCW vs CW)

    5. Effective Angular Traversal:
       Within each band k, the traversal angle is rotated and oriented:
           eff_angle = (dir_k * angle + phi_k) mod 2*pi

    6. Multi-Tier Deterministic Tie-Breaking:
       Using np.lexsort, pixels are ordered by:
           Primary:   band       -> Concentric progression from center outward
           Secondary: eff_angle  -> Circular sweep around the ring
           Tertiary:  radius     -> Resolves collinear points along the same ray
           Quaternary: flat_index -> Guaranteed uniqueness for all H*W coordinates

    Args:
        height: Image height in pixels (must be >= 1).
        width: Image width in pixels (must be >= 1).
        seed: Cryptographic seed or key material (string, bytes, or int).
        delta_r: Radial width of concentric shells in pixels (default 1.0).

    Returns:
        A 1D NumPy array of shape (H * W,) containing a permutation of indices [0, H*W-1].

    Raises:
        ValueError: If height or width is less than 1, or delta_r <= 0.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height ({height}) and width ({width}) must be positive integers.")
    if delta_r <= 0:
        raise ValueError(f"delta_r ({delta_r}) must be strictly positive.")

    total_pixels = height * width
    if total_pixels == 1:
        return np.array([0], dtype=np.int64)

    # 1. Cryptographically secure deterministic seed derivation
    seed_bytes = _normalize_seed(seed)
    domain_separator = b"SHAPECRYPT_CIRCLE_TRAVERSAL_V1_"
    hash_digest = hashlib.sha256(domain_separator + seed_bytes).digest()
    pcg_seed = int.from_bytes(hash_digest[:16], byteorder="big")
    rng = np.random.default_rng(pcg_seed)

    # 2. Vectorized 2D grid coordinates relative to image center
    c_r = (height - 1) / 2.0
    c_c = (width - 1) / 2.0
    r_coords, c_coords = np.indices((height, width))
    dr = r_coords - c_r
    dc = c_coords - c_c

    # 3. Vectorized Polar Coordinates (radius & angle)
    radius = np.sqrt(dr**2 + dc**2)
    # np.arctan2 returns values in [-pi, pi]; map to [0, 2*pi)
    angle = np.mod(np.arctan2(dr, dc) + (2.0 * np.pi), 2.0 * np.pi)

    # 4. Concentric circular bands
    bands = np.floor(radius / delta_r).astype(np.int64)
    num_bands = int(np.max(bands)) + 1

    # 5. Deterministic per-band phase rotation and traversal direction
    band_offsets = rng.uniform(0.0, 2.0 * np.pi, size=num_bands)
    band_directions = rng.choice([-1.0, 1.0], size=num_bands)

    # 6. Flatten arrays for 1D sorting
    flat_bands = bands.ravel()
    flat_angles = angle.ravel()
    flat_radius = radius.ravel()
    flat_indices = np.arange(total_pixels, dtype=np.int64)

    # Calculate effective traversal angle per pixel
    eff_angles = np.mod(
        (band_directions[flat_bands] * flat_angles) + band_offsets[flat_bands],
        2.0 * np.pi,
    )

    # 7. Deterministic multi-tier sort:
    # np.lexsort sorts by keys in reverse order of argument presentation:
    # 1st priority: flat_bands (concentric shells)
    # 2nd priority: eff_angles (circular traversal within shell)
    # 3rd priority: flat_radius (radial tie-breaker)
    # 4th priority: flat_indices (unique coordinate tie-breaker)
    permutation = np.lexsort((flat_indices, flat_radius, eff_angles, flat_bands))

    return permutation.astype(np.int64)


def generate_triangle_permutation(
    height: int,
    width: int,
    seed: Union[str, bytes, int],
    delta_s: float | None = None,
) -> np.ndarray:
    """Generates a deterministic, reversible triangle-based pixel permutation.

    Mathematical Traversal Formulation:
    -----------------------------------
    1. Geometric Coordinate Normalization:
       For an image of dimensions (H, W), the center is at:
           c_r = (H - 1) / 2.0,  c_c = (W - 1) / 2.0
       Coordinates are normalized to [-1, 1] relative to the image center:
           y = -(r - c_r) / (H / 2.0)   (y increases upward)
           x =  (c - c_c) / (W / 2.0)   (x increases rightward)

    2. Triangle Orientation & Rotational Frame:
       A triangle on the image plane can be oriented towards 4 canonical directions:
           Orientation 0: Apex pointing Up (North, 0 deg)
           Orientation 1: Apex pointing Right (East, 90 deg)
           Orientation 2: Apex pointing Down (South, 180 deg)
           Orientation 3: Apex pointing Left (West, 270 deg)
       The orientation is selected deterministically from the seed. The coordinates
       are rotated by angle theta_rot into the canonical upright frame:
           x' = x * cos(theta_rot) - y * sin(theta_rot)
           y' = x * sin(theta_rot) + y * cos(theta_rot)

    3. Triangular Distance Function (Level Sets of an Equilateral Triangle):
       In the upright frame, the 3 linear boundary edges of an equilateral triangle are:
           Bottom base: y' = -h
           Left edge:   y' - sqrt(3) * x' = h
           Right edge:  y' + sqrt(3) * x' = h
       The triangular distance norm measuring concentric expanding triangles is:
           d_tri(x', y') = max(-2 * y', y' + sqrt(3) * |x'|)
       Equi-distance contours of d_tri form exact concentric triangles.

    4. Concentric Triangular Shells & Progression:
       Pixels are partitioned into concentric triangular shells:
           shell = floor(d_tri / delta_s)
       The seed determines the shell progression direction:
           - Inside-out (apex/center to outer edges)
           - Outside-in (outer edges to center)

    5. Starting Corner & Perimeter Sweep:
       Along any triangular shell, vertices are located at:
           Vertex 0 (Apex):         theta = pi/2      (90 deg)
           Vertex 1 (Bottom-Right): theta = 11*pi/6   (330 deg)
           Vertex 2 (Bottom-Left):  theta = 7*pi/6    (210 deg)
       The seed chooses the starting corner (0, 1, or 2) and traversal direction
       (+1 counter-clockwise, -1 clockwise) around the triangle perimeter:
           eff_theta = (dir * (theta - start_corner_angle) + shell_offsets[shell]) mod 2*pi

    6. Multi-Tier Deterministic Tie-Breaking (np.lexsort):
       - Primary:   shell       -> Concentric progression along triangular altitude
       - Secondary: eff_theta   -> Sweep along the 3 edges of the triangular shell
       - Tertiary:  d_tri       -> Exact continuous distance along triangular rays
       - Quaternary: flat_index -> Absolute mathematical uniqueness guarantee

    Args:
        height: Image height in pixels (must be >= 1).
        width: Image width in pixels (must be >= 1).
        seed: Cryptographic seed or key material (string, bytes, or int).
        delta_s: Optional shell thickness parameter in normalized units.
                 Defaults to 2.0 / max(H, W).

    Returns:
        A 1D NumPy array of shape (H * W,) containing a permutation of indices [0, H*W-1].

    Raises:
        ValueError: If height or width is less than 1.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height ({height}) and width ({width}) must be positive integers.")

    total_pixels = height * width
    if total_pixels == 1:
        return np.array([0], dtype=np.int64)

    # 1. Cryptographically secure deterministic seed derivation
    seed_bytes = _normalize_seed(seed)
    domain_separator = b"SHAPECRYPT_TRIANGLE_TRAVERSAL_V1_"
    hash_digest = hashlib.sha256(domain_separator + seed_bytes).digest()

    pcg_seed = int.from_bytes(hash_digest[:16], byteorder="big")
    rng = np.random.default_rng(pcg_seed)

    # 2. Derive geometric parameters from hash digest
    orientation = int(hash_digest[16] % 4)         # 0: Up, 1: Right, 2: Down, 3: Left
    start_corner = int(hash_digest[17] % 3)        # 0: Apex, 1: Corner 1, 2: Corner 2
    direction = 1.0 if (hash_digest[18] & 1) else -1.0  # +1: CCW, -1: CW
    shell_progression = int(hash_digest[19] & 1)   # 0: Inside-out, 1: Outside-in

    # Angles for 3 canonical vertices of an upright triangle
    corner_angles = [np.pi / 2.0, 11.0 * np.pi / 6.0, 7.0 * np.pi / 6.0]
    start_angle = corner_angles[start_corner]

    # 3. Vectorized normalized coordinates relative to image center
    c_r = (height - 1) / 2.0
    c_c = (width - 1) / 2.0
    r_coords, c_coords = np.indices((height, width))

    y = -(r_coords - c_r)
    x = c_coords - c_c

    # Aspect-ratio normalization to [-1, 1]
    scale_y = 1.0 / (height / 2.0) if height > 1 else 1.0
    scale_x = 1.0 / (width / 2.0) if width > 1 else 1.0
    yn = y * scale_y
    xn = x * scale_x

    # 4. Rotation for triangle orientation
    rot_angles = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    rot = rot_angles[orientation]
    xr = xn * np.cos(rot) - yn * np.sin(rot)
    yr = xn * np.sin(rot) + yn * np.cos(rot)

    # 5. Triangular distance function: level sets form concentric equilateral triangles
    d_tri = np.maximum(-2.0 * yr, yr + np.sqrt(3.0) * np.abs(xr))
    # Shift so center is 0 and all distances are non-negative
    min_d = np.min(d_tri)
    d_tri = d_tri - min_d

    # Polar angle in rotated frame: [0, 2*pi)
    theta = np.mod(np.arctan2(yr, xr) + (2.0 * np.pi), 2.0 * np.pi)

    # 6. Concentric triangular shells
    if delta_s is None:
        delta_s = 2.0 / max(height, width) if max(height, width) > 1 else 1.0
    elif delta_s <= 0:
        raise ValueError(f"delta_s ({delta_s}) must be strictly positive.")

    shells = np.floor(d_tri / delta_s).astype(np.int64)
    num_shells = int(np.max(shells)) + 1

    # PRNG per-shell angle offsets for rich key sensitivity
    shell_offsets = rng.uniform(0.0, 2.0 * np.pi, size=num_shells)

    # Apply progression direction (Inside-out vs Outside-in)
    if shell_progression == 1:
        shells = np.max(shells) - shells

    # 7. Flatten arrays for multi-tier sorting
    flat_shells = shells.ravel()
    flat_angles = theta.ravel()
    flat_dtri = d_tri.ravel()
    flat_indices = np.arange(total_pixels, dtype=np.int64)

    # Effective angle around triangular perimeter with seed-derived offsets
    eff_theta = np.mod(
        (direction * (flat_angles - start_angle) + shell_offsets[flat_shells]) + (2.0 * np.pi),
        2.0 * np.pi,
    )

    # 8. Deterministic multi-tier sort:
    # 1st priority: flat_shells (concentric triangular shells)
    # 2nd priority: eff_theta (traversal around the triangle perimeter)
    # 3rd priority: flat_dtri (radial distance along triangular rays)
    # 4th priority: flat_indices (absolute coordinate uniqueness)
    permutation = np.lexsort((flat_indices, flat_dtri, eff_theta, flat_shells))

    return permutation.astype(np.int64)


def generate_square_permutation(
    height: int,
    width: int,
    seed: Union[str, bytes, int],
    delta_s: float | None = None,
) -> np.ndarray:
    """Generates a deterministic, reversible square-based pixel permutation.

    Mathematical Traversal Formulation:
    -----------------------------------
    1. Geometric Coordinate Normalization:
       For an image of dimensions (H, W), the center is at:
           c_r = (H - 1) / 2.0,  c_c = (W - 1) / 2.0
       Coordinates are normalized to [-1, 1] relative to the image center:
           y = -(r - c_r) / (H / 2.0)   (y increases upward)
           x =  (c - c_c) / (W / 2.0)   (x increases rightward)

    2. Square Geometry (Chebyshev Norm):
       Concentric squares are defined by the L_inf (Chebyshev) distance metric:
           d_sq(x, y) = max(|x|, |y|)
       Every locus where d_sq = const forms a geometric square boundary.

    3. Concentric Square Shells & Progression:
       Pixels are grouped into expanding concentric square shells (layers):
           shell = floor(d_sq / delta_s)
       The seed determines the layer progression:
           - Inside-out (center to outer square boundary)
           - Outside-in (outer square boundary to center)

    4. Square Perimeter Edge Partitioning & Traversal:
       Each concentric square shell consists of 4 distinct perimeter edges:
           - Edge 0 (Top):    y >= |x|
           - Edge 1 (Right):  x > |y|
           - Edge 2 (Bottom): -y >= |x|
           - Edge 3 (Left):   -x > |y|
       The seed derives:
           - Edge sequence order (a permutation of [0, 1, 2, 3])
           - Traversal direction along each edge (forward vs reverse)

    5. Multi-Tier Deterministic Tie-Breaking (np.lexsort):
       - Primary:   shell          -> Concentric progression along square layers
       - Secondary: edge_priority  -> Order of traversal across the 4 edges of the square
       - Tertiary:  t_effective    -> Position along the current square edge
       - Quaternary: d_sq          -> Continuous distance along square rays
       - Quinary:   flat_index     -> Absolute coordinate uniqueness guarantee

    Args:
        height: Image height in pixels (must be >= 1).
        width: Image width in pixels (must be >= 1).
        seed: Cryptographic seed or key material (string, bytes, or int).
        delta_s: Optional shell thickness parameter in normalized units.
                 Defaults to 2.0 / max(H, W).

    Returns:
        A 1D NumPy array of shape (H * W,) containing a permutation of indices [0, H*W-1].

    Raises:
        ValueError: If height or width is less than 1.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height ({height}) and width ({width}) must be positive integers.")

    total_pixels = height * width
    if total_pixels == 1:
        return np.array([0], dtype=np.int64)

    # 1. Cryptographically secure deterministic seed derivation
    seed_bytes = _normalize_seed(seed)
    domain_separator = b"SHAPECRYPT_SQUARE_TRAVERSAL_V2_"
    hash_digest = hashlib.sha256(domain_separator + seed_bytes).digest()

    pcg_seed = int.from_bytes(hash_digest[:16], byteorder="big")
    rng = np.random.default_rng(pcg_seed)

    # 2. Vectorized normalized coordinates relative to image center
    c_r = (height - 1) / 2.0
    c_c = (width - 1) / 2.0
    r_coords, c_coords = np.indices((height, width))

    y = -(r_coords - c_r)
    x = c_coords - c_c

    scale_y = 1.0 / (height / 2.0) if height > 1 else 1.0
    scale_x = 1.0 / (width / 2.0) if width > 1 else 1.0
    yn = y * scale_y
    xn = x * scale_x

    # 3. Concentric square distance via Chebyshev L_inf norm
    abs_xn = np.abs(xn)
    abs_yn = np.abs(yn)
    d_sq = np.maximum(abs_xn, abs_yn)

    if delta_s is None:
        delta_s = 2.0 / max(height, width) if max(height, width) > 1 else 1.0
    elif delta_s <= 0:
        raise ValueError(f"delta_s ({delta_s}) must be strictly positive.")

    shells = np.floor(d_sq / delta_s).astype(np.int64)

    # 4. Layer progression: Inside-out vs Outside-in
    layer_progression = int(hash_digest[16] & 1)
    if layer_progression == 1:
        shells = np.max(shells) - shells

    # 5. Partition pixels into the 4 square perimeter edges:
    # 0: Top, 1: Right, 2: Bottom, 3: Left
    edge_idx = np.zeros((height, width), dtype=np.int64)
    edge_idx[yn >= abs_xn] = 0
    edge_idx[(xn > abs_yn) & (yn < abs_xn)] = 1
    edge_idx[(-yn >= abs_xn) & (xn <= abs_yn) & (xn >= -abs_yn)] = 2
    edge_idx[(-xn > abs_yn) & (yn < abs_xn)] = 3

    # Seed-derived edge traversal order
    edge_order = rng.permutation(4)
    edge_priority = np.zeros(4, dtype=np.int64)
    for p, e in enumerate(edge_order):
        edge_priority[e] = p
    pixel_edge_prio = edge_priority[edge_idx]

    # Coordinate position along each edge
    t_pos = np.zeros((height, width), dtype=np.float64)
    t_pos[edge_idx == 0] = xn[edge_idx == 0]
    t_pos[edge_idx == 1] = -yn[edge_idx == 1]
    t_pos[edge_idx == 2] = -xn[edge_idx == 2]
    t_pos[edge_idx == 3] = yn[edge_idx == 3]

    # Traversal direction along each edge
    edge_dirs = rng.choice([-1.0, 1.0], size=4)
    t_eff = t_pos * edge_dirs[edge_idx]

    # 6. Flatten arrays for multi-tier sorting
    flat_shells = shells.ravel()
    flat_edge_prio = pixel_edge_prio.ravel()
    flat_teff = t_eff.ravel()
    flat_dsq = d_sq.ravel()
    flat_indices = np.arange(total_pixels, dtype=np.int64)

    # 7. Multi-tier sort:
    # 1st: flat_shells (concentric square layers)
    # 2nd: flat_edge_prio (square edges sequence)
    # 3rd: flat_teff (sweep along edge)
    # 4th: flat_dsq (radial distance)
    # 5th: flat_indices (coordinate uniqueness)
    permutation = np.lexsort((flat_indices, flat_dsq, flat_teff, flat_edge_prio, flat_shells))

    return permutation.astype(np.int64)


def generate_star_permutation(
    height: int,
    width: int,
    seed: Union[str, bytes, int],
    delta_s: float | None = None,
    num_points: int = 5,
    alpha: float = 0.45,
) -> np.ndarray:
    """Generates a deterministic, reversible star-based pixel permutation.

    Mathematical Traversal Formulation:
    -----------------------------------
    1. Geometric Coordinate Normalization:
       For an image of dimensions (H, W), the center is at:
           c_r = (H - 1) / 2.0,  c_c = (W - 1) / 2.0
       Coordinates are normalized to [-1, 1] relative to the image center:
           y = -(r - c_r) / (H / 2.0)   (y increases upward)
           x =  (c - c_c) / (W / 2.0)   (x increases rightward)

    2. Polar Harmonic Star Geometry:
       In polar coordinates, an n-pointed star with modulation depth alpha is
       defined by the radial boundary function:
           r_star(theta) = 1 + alpha * cos(n * (theta - orientation))
       For any point (x, y) with polar coordinates radius = sqrt(x^2 + y^2) and
       theta = atan2(y, x), the continuous star distance is:
           d_star(x, y) = radius / (1 + alpha * cos(n * (theta - orientation)))
       Equi-distance contours of d_star form exact concentric star shapes with
       alternating outer tips and inner valleys.

    3. Concentric Star Shells & Progression:
       Pixels are grouped into expanding concentric star shells:
           shell = floor(d_star / delta_s)
       The cryptographic seed determines the layer progression:
           - Inside-out (center core outward to star tips)
           - Outside-in (outer star tips inward to center)

    4. Starting Point & Perimeter Sweep:
       Along each star shell, the seed chooses:
           - Star orientation / rotation angle in [0, 2*pi)
           - Starting tip or valley (from 0 to num_points - 1)
           - Traversal direction (+1 counter-clockwise, -1 clockwise)
             sweeping along the star perimeter:
             eff_theta = (dir * (theta - start_angle) + shell_offsets[shell]) mod 2*pi

    5. Multi-Tier Deterministic Tie-Breaking (np.lexsort):
       - Primary:   shell       -> Concentric progression along star shells
       - Secondary: eff_theta   -> Perimeter sweep along star tips and valleys
       - Tertiary:  d_star      -> Continuous distance along star rays
       - Quaternary: flat_index -> Absolute mathematical uniqueness guarantee

    Args:
        height: Image height in pixels (must be >= 1).
        width: Image width in pixels (must be >= 1).
        seed: Cryptographic seed or key material (string, bytes, or int).
        delta_s: Optional shell thickness parameter in normalized units.
                 Defaults to 2.0 / max(H, W).
        num_points: Number of star vertices/tips (default 5 for classic pentagram).
        alpha: Depth of star indentations in (0, 1) (default 0.45).

    Returns:
        A 1D NumPy array of shape (H * W,) containing a permutation of indices [0, H*W-1].

    Raises:
        ValueError: If height or width is less than 1, or alpha not in (0, 1).
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height ({height}) and width ({width}) must be positive integers.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Star modulation alpha ({alpha}) must be strictly between 0 and 1.")
    if num_points < 3:
        raise ValueError(f"num_points ({num_points}) must be at least 3.")

    total_pixels = height * width
    if total_pixels == 1:
        return np.array([0], dtype=np.int64)

    # 1. Cryptographically secure deterministic seed derivation
    seed_bytes = _normalize_seed(seed)
    domain_separator = b"SHAPECRYPT_STAR_TRAVERSAL_V1_"
    hash_digest = hashlib.sha256(domain_separator + seed_bytes).digest()

    pcg_seed = int.from_bytes(hash_digest[:16], byteorder="big")
    rng = np.random.default_rng(pcg_seed)

    # 2. Derive star parameters from hash digest
    orientation = rng.uniform(0.0, 2.0 * np.pi)
    start_tip = int(hash_digest[17] % num_points)
    start_angle = orientation + (start_tip * (2.0 * np.pi / num_points))
    direction = 1.0 if (hash_digest[18] & 1) else -1.0
    layer_progression = int(hash_digest[19] & 1)

    # 3. Vectorized normalized coordinates relative to image center
    c_r = (height - 1) / 2.0
    c_c = (width - 1) / 2.0
    r_coords, c_coords = np.indices((height, width))

    y = -(r_coords - c_r)
    x = c_coords - c_c

    scale_y = 1.0 / (height / 2.0) if height > 1 else 1.0
    scale_x = 1.0 / (width / 2.0) if width > 1 else 1.0
    yn = y * scale_y
    xn = x * scale_x

    radius = np.sqrt(xn**2 + yn**2)
    theta = np.mod(np.arctan2(yn, xn) + (2.0 * np.pi), 2.0 * np.pi)

    # 4. Polar harmonic star formula
    star_modulation = 1.0 + alpha * np.cos(num_points * (theta - orientation))
    d_star = radius / star_modulation

    # 5. Concentric star shells
    if delta_s is None:
        delta_s = 2.0 / max(height, width) if max(height, width) > 1 else 1.0
    elif delta_s <= 0:
        raise ValueError(f"delta_s ({delta_s}) must be strictly positive.")

    shells = np.floor(d_star / delta_s).astype(np.int64)
    num_shells = int(np.max(shells)) + 1
    shell_offsets = rng.uniform(0.0, 2.0 * np.pi, size=num_shells)

    # Apply layer progression direction
    if layer_progression == 1:
        shells = np.max(shells) - shells

    # 6. Flatten arrays for multi-tier sorting
    flat_shells = shells.ravel()
    flat_angles = theta.ravel()
    flat_dstar = d_star.ravel()
    flat_indices = np.arange(total_pixels, dtype=np.int64)

    # Effective angle around star perimeter with seed-derived offsets
    eff_theta = np.mod(
        (direction * (flat_angles - start_angle) + shell_offsets[flat_shells]) + (2.0 * np.pi),
        2.0 * np.pi,
    )

    # 7. Deterministic multi-tier sort:
    # 1st priority: flat_shells (concentric star shells)
    # 2nd priority: eff_theta (traversal around star perimeter)
    # 3rd priority: flat_dstar (continuous star distance along rays)
    # 4th priority: flat_indices (absolute coordinate uniqueness)
    permutation = np.lexsort((flat_indices, flat_dstar, eff_theta, flat_shells))

    return permutation.astype(np.int64)


def generate_spiral_permutation(
    height: int,
    width: int,
    seed: Union[str, bytes, int],
) -> np.ndarray:
    """Generates a deterministic, reversible spiral-based pixel permutation.

    Mathematical Traversal Formulation:
    -----------------------------------
    1. Spiral Boundary Peeling Geometry:
       In discrete image processing, a spiral matrix traversal winds around the
       rectangular grid by iteratively peeling concentric perimeter layers:
           Loop 0: Outer perimeter boundary of the image
           Loop 1: Boundary of the inner submatrix
           ...
           Final Loop: Reaches the innermost core pixel(s).

    2. Geometric Parameters Controlled by Seed:
       - Starting Corner: Top-Left (0), Top-Right (1), Bottom-Right (2), or Bottom-Left (3)
       - Traversal Direction: Clockwise (CW) vs Counter-Clockwise (CCW)
       - Winding Mode:
           - Inward: From perimeter corners winding inward to center
           - Outward: From center core winding outward to perimeter corners
       - Loop-Level Phase Shift:
           Each concentric winding loop is cyclically shifted by a seed-derived
           offset along its perimeter, introducing rich key sensitivity while
           maintaining 100% discrete coverage.

    3. Discrete Invariant:
       Because coordinates are peeled layer by layer from the bounding box,
       every pixel in [0, H-1] x [0, W-1] is visited exactly once with zero
       duplicates and zero omitted coordinates.

    Args:
        height: Image height in pixels (must be >= 1).
        width: Image width in pixels (must be >= 1).
        seed: Cryptographic seed or key material (string, bytes, or int).

    Returns:
        A 1D NumPy array of shape (H * W,) containing a permutation of indices [0, H*W-1].

    Raises:
        ValueError: If height or width is less than 1.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height ({height}) and width ({width}) must be positive integers.")

    total_pixels = height * width
    if total_pixels == 1:
        return np.array([0], dtype=np.int64)

    # 1. Cryptographically secure deterministic seed derivation
    seed_bytes = _normalize_seed(seed)
    domain_separator = b"SHAPECRYPT_SPIRAL_TRAVERSAL_V1_"
    hash_digest = hashlib.sha256(domain_separator + seed_bytes).digest()

    pcg_seed = int.from_bytes(hash_digest[:16], byteorder="big")
    rng = np.random.default_rng(pcg_seed)

    # 2. Derive geometric parameters
    corner = int(hash_digest[16] % 4)          # 0: TL, 1: TR, 2: BR, 3: BL
    clockwise = bool(hash_digest[17] & 1)      # CW vs CCW
    outward = bool(hash_digest[18] & 1)        # Inward vs Outward

    # Corner reflections
    flip_h = corner in (2, 3)
    flip_w = corner in (1, 2)

    # If CCW, transpose the dimensions and swap back at the end
    if not clockwise:
        eff_h, eff_w = width, height
    else:
        eff_h, eff_w = height, width

    out_r = np.empty(total_pixels, dtype=np.int64)
    out_c = np.empty(total_pixels, dtype=np.int64)
    idx = 0

    top = 0
    bottom = eff_h - 1
    left = 0
    right = eff_w - 1

    # 3. Vectorized boundary peeling
    while top <= bottom and left <= right:
        loop_start = idx

        # Top edge: left -> right
        span = right - left + 1
        out_r[idx:idx + span] = top
        out_c[idx:idx + span] = np.arange(left, right + 1, dtype=np.int64)
        idx += span
        top += 1

        # Right edge: top -> bottom
        if top <= bottom:
            span = bottom - top + 1
            out_r[idx:idx + span] = np.arange(top, bottom + 1, dtype=np.int64)
            out_c[idx:idx + span] = right
            idx += span
            right -= 1

        # Bottom edge: right -> left
        if top <= bottom and left <= right:
            span = right - left + 1
            out_r[idx:idx + span] = bottom
            out_c[idx:idx + span] = np.arange(right, left - 1, -1, dtype=np.int64)
            idx += span
            bottom -= 1

        # Left edge: bottom -> top
        if top <= bottom and left <= right:
            span = bottom - top + 1
            out_r[idx:idx + span] = np.arange(bottom, top - 1, -1, dtype=np.int64)
            out_c[idx:idx + span] = left
            idx += span
            left += 1

        # Seed-derived phase shift along this concentric loop's perimeter
        loop_len = idx - loop_start
        if loop_len > 1:
            shift = int(rng.integers(0, loop_len))
            if shift > 0:
                out_r[loop_start:idx] = np.roll(out_r[loop_start:idx], shift)
                out_c[loop_start:idx] = np.roll(out_c[loop_start:idx], shift)

    # 4. Invert CCW transposition if applicable
    if not clockwise:
        final_r = out_c
        final_c = out_r
    else:
        final_r = out_r
        final_c = out_c

    # 5. Apply corner reflections
    if flip_h:
        final_r = (height - 1) - final_r
    if flip_w:
        final_c = (width - 1) - final_c

    # Flatten coordinates to linear indices
    permutation = final_r * width + final_c

    # 6. Apply inward vs outward winding mode
    if outward:
        permutation = permutation[::-1]

    return permutation.astype(np.int64)




