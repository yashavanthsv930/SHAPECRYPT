"""Automated Test Suite for SHAPECRYPT Circle Geometric Pixel Permutation
Validates Coverage, Uniqueness, Range, Reversibility, Determinism,
Different Seed divergence, Multi-size robustness (4x4, 5x7, 16x16, 32x24),
and RGB / RGBA multi-channel lossless reconstruction.
"""

import unittest
import numpy as np

from encryption.shapes import generate_circle_permutation
from encryption.permutation import (
    apply_permutation,
    inverse_permutation,
    apply_inverse_permutation,
    is_valid_permutation,
)


class TestCirclePermutation(unittest.TestCase):
    """Test suite for circle-based geometric pixel permutation."""

    def setUp(self):
        self.test_sizes = [
            (4, 4),    # Small square (even)
            (5, 7),    # Small rectangular (odd x odd)
            (16, 16),  # Medium square (power of 2)
            (32, 24),  # Medium rectangular (even x even)
            (1, 1),    # Single pixel edge case
            (1, 9),    # 1D row image
            (8, 1),    # 1D column image
        ]
        self.primary_seed = "shapecrypt_master_seed_alpha"
        self.secondary_seed = "shapecrypt_master_seed_beta"

    def test_requirement_a_coverage(self):
        """Test A: Permutation length must strictly equal H * W."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_circle_permutation(h, w, self.primary_seed)
                expected_len = h * w
                self.assertEqual(
                    len(perm),
                    expected_len,
                    f"Failed coverage for {h}x{w}: expected {expected_len}, got {len(perm)}"
                )

    def test_requirement_b_uniqueness(self):
        """Test B: All coordinates in permutation must be unique (no duplicates)."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_circle_permutation(h, w, self.primary_seed)
                unique_count = len(set(perm))
                expected_count = h * w
                self.assertEqual(
                    unique_count,
                    expected_count,
                    f"Duplicate coordinates detected in {h}x{w}: {expected_count - unique_count} duplicates"
                )
                self.assertTrue(is_valid_permutation(perm, expected_count))

    def test_requirement_c_range(self):
        """Test C: Every index in permutation must be between 0 and H*W - 1."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_circle_permutation(h, w, self.primary_seed)
                total_pixels = h * w
                self.assertGreaterEqual(np.min(perm), 0, f"Negative index found in {h}x{w}")
                self.assertLess(np.max(perm), total_pixels, f"Index out of upper bound in {h}x{w}")

    def test_requirement_d_reversibility_rgb_rgba(self):
        """Test D & H: Reversibility must produce the exact identical image for RGB and RGBA."""
        np.random.seed(42)
        for h, w in [(4, 4), (5, 7), (16, 16), (32, 24)]:
            with self.subTest(height=h, width=w):
                perm = generate_circle_permutation(h, w, self.primary_seed)

                # Test 1: RGB Image (H, W, 3)
                original_rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                shuffled_rgb = apply_permutation(original_rgb, perm)
                recovered_rgb = apply_inverse_permutation(shuffled_rgb, perm)

                self.assertEqual(original_rgb.shape, recovered_rgb.shape)
                self.assertEqual(original_rgb.dtype, recovered_rgb.dtype)
                self.assertTrue(
                    np.array_equal(original_rgb, recovered_rgb),
                    f"RGB Lossless reconstruction failed for {h}x{w}!"
                )

                # Test 2: RGBA Image (H, W, 4)
                original_rgba = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)
                shuffled_rgba = apply_permutation(original_rgba, perm)
                recovered_rgba = apply_inverse_permutation(shuffled_rgba, perm)

                self.assertEqual(original_rgba.shape, recovered_rgba.shape)
                self.assertEqual(original_rgba.dtype, recovered_rgba.dtype)
                self.assertTrue(
                    np.array_equal(original_rgba, recovered_rgba),
                    f"RGBA Lossless reconstruction failed for {h}x{w}!"
                )

                # Test 3: Grayscale Image (H, W)
                original_gray = np.random.randint(0, 256, (h, w), dtype=np.uint8)
                shuffled_gray = apply_permutation(original_gray, perm)
                recovered_gray = apply_inverse_permutation(shuffled_gray, perm)
                self.assertTrue(
                    np.array_equal(original_gray, recovered_gray),
                    f"Grayscale Lossless reconstruction failed for {h}x{w}!"
                )

    def test_requirement_e_determinism(self):
        """Test E: Same dimensions + same seed must yield exactly identical permutation."""
        for h, w in [(4, 4), (5, 7), (16, 16), (32, 24)]:
            with self.subTest(height=h, width=w):
                perm_run_1 = generate_circle_permutation(h, w, self.primary_seed)
                perm_run_2 = generate_circle_permutation(h, w, self.primary_seed)
                self.assertTrue(
                    np.array_equal(perm_run_1, perm_run_2),
                    f"Determinism failed for {h}x{w}: two identical runs gave different outputs."
                )

    def test_requirement_f_different_seeds(self):
        """Test F: Same dimensions + different seeds must yield different permutations."""
        seeds = [
            "password_alpha",
            "password_beta",
            "cryptographic_key_99",
            123456789,
            b"\xde\xad\xbe\xef\x01\x02\x03\x04"
        ]
        for h, w in [(4, 4), (5, 7), (16, 16), (32, 24)]:
            with self.subTest(height=h, width=w):
                perms = [generate_circle_permutation(h, w, s) for s in seeds]
                for i in range(len(perms)):
                    for j in range(i + 1, len(perms)):
                        self.assertFalse(
                            np.array_equal(perms[i], perms[j]),
                            f"Seed collision detected for {h}x{w} between seed {i} and seed {j}!"
                        )

    def test_mathematical_inverse_identity(self):
        """Validates that P[P_inv] == Identity and P_inv[P] == Identity."""
        for h, w in [(4, 4), (5, 7), (16, 16), (32, 24)]:
            perm = generate_circle_permutation(h, w, self.primary_seed)
            inv_perm = inverse_permutation(perm)
            identity = np.arange(h * w, dtype=np.int64)

            self.assertTrue(
                np.array_equal(perm[inv_perm], identity),
                f"P[P_inv] != Identity for {h}x{w}"
            )
            self.assertTrue(
                np.array_equal(inv_perm[perm], identity),
                f"P_inv[P] != Identity for {h}x{w}"
            )

    def test_invalid_arguments(self):
        """Validates error handling for illegal inputs."""
        with self.assertRaises(ValueError):
            generate_circle_permutation(0, 10, "seed")
        with self.assertRaises(ValueError):
            generate_circle_permutation(10, -2, "seed")
        with self.assertRaises(ValueError):
            generate_circle_permutation(10, 10, "seed", delta_r=0.0)
        with self.assertRaises(ValueError):
            inverse_permutation([0, 1, 1, 3])  # Duplicate


if __name__ == "__main__":
    unittest.main(verbosity=2)
